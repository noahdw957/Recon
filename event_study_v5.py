import requests
import pandas as pd
import numpy as np
import yfinance as yf
import json
import time
from datetime import date, timedelta
from pathlib import Path

# ============================================================
# RECON EVENT STUDY V5
#
# PURPOSE:
# Build a real historical event dataset for testing:
#
# AWARD -> VALLEY -> PEAK
#
# ============================================================

DAYS_BACK = 90
MIN_TRANSACTION = 1_000_000

END = date.today()
START = END - timedelta(days=DAYS_BACK)

API = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"

# ============================================================
# PUBLIC COMPANY MASTER
# ============================================================

MASTER = {
    "PLTR": ["PALANTIR"],
    "RCAT": ["RED CAT"],
    "AVAV": ["AEROVIRONMENT"],
    "WWD": ["WOODWARD"],
    "AEVA": ["AEVA"],
    "LMT": ["LOCKHEED MARTIN"],
    "RTX": ["RAYTHEON", "RTX"],
    "BAH": ["BOOZ ALLEN"],
    "SAIC": ["SAIC", "SCIENCE APPLICATIONS INTERNATIONAL"],
    "LDOS": ["LEIDOS"],
    "LHX": ["L3HARRIS", "L3 HARRIS"],
    "NOC": ["NORTHROP GRUMMAN"],
}

# ============================================================
# OUTPUT FILES
# ============================================================

RESULTS = Path("event_study_events.csv")
JSON_RESULTS = Path("event_study_results.json")
SUMMARY = Path("event_study_summary.json")

# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "RECON-Event-Study/5.0"
})

# ============================================================
# MASTER LOOKUP
# ============================================================

def master_lookup(name):

    if not name:
        return None

    u = name.upper()

    for ticker, keywords in MASTER.items():

        for keyword in keywords:

            if keyword in u:
                return ticker

    return None


# ============================================================
# STEP 1 — DOWNLOAD TRANSACTIONS
# ============================================================

print()
print("=" * 70)
print("RECON EVENT STUDY V5")
print("=" * 70)

print(
    f"Date range: {START} -> {END}"
)

print(
    f"Minimum transaction: ${MIN_TRANSACTION:,.0f}"
)

print()
print("Downloading USAspending transaction data...")
print()

payload = {

    "filters": {

        "award_amounts": [
            {
                "lower_bound": MIN_TRANSACTION
            }
        ],

        "award_type_codes": [
            "A",
            "B",
            "C",
            "D"
        ],

        "time_period": [
            {
                "start_date": str(START),
                "end_date": str(END)
            }
        ]

    },

    "fields": [

        "Award ID",
        "Recipient Name",
        "Action Date",
        "Transaction Amount",
        "Awarding Agency",
        "Awarding Sub Agency",
        "Award Type"

    ],

    "sort": "Transaction Amount",

    "order": "desc",

    "limit": 100,

    "page": 1
}

all_rows = []

page = 1

while True:

    payload["page"] = page

    try:

        r = session.post(
            API,
            json=payload,
            timeout=90
        )

        print(
            f"Page {page}: HTTP {r.status_code}"
        )

        r.raise_for_status()

        data = r.json()

    except Exception as e:

        print(
            f"ERROR on page {page}: {e}"
        )

        break

    rows = data.get(
        "results",
        []
    )

    print(
        f"   records returned: {len(rows)}"
    )

    if not rows:
        break

    all_rows.extend(rows)

    metadata = data.get(
        "page_metadata",
        {}
    )

    if not metadata.get(
        "hasNext",
        False
    ):

        break

    page += 1

    # Safety limit.
    if page > 100:

        print(
            "Stopping at 100 pages."
        )

        break

    time.sleep(0.25)


print()
print(
    f"TOTAL TRANSACTIONS: {len(all_rows):,}"
)

# ============================================================
# STEP 2 — IDENTIFY PUBLIC COMPANIES
# ============================================================

print()
print(
    "Identifying MASTER public companies..."
)

events = []

seen = set()

for row in all_rows:

    name = row.get(
        "Recipient Name"
    )

    ticker = master_lookup(
        name
    )

    if not ticker:
        continue

    award_id = row.get(
        "Award ID"
    )

    action_date = row.get(
        "Action Date"
    )

    amount = row.get(
        "Transaction Amount"
    )

    if not action_date:
        continue

    try:

        amount = float(
            amount
        )

    except Exception:

        continue

    # Avoid duplicate transaction records.
    key = (
        award_id,
        action_date,
        amount
    )

    if key in seen:
        continue

    seen.add(key)

    agency = row.get(
        "Awarding Agency"
    )

    if isinstance(
        agency,
        dict
    ):

        agency = (
            agency.get(
                "toptier_name"
            )
            or agency.get(
                "name"
            )
        )

    subagency = row.get(
        "Awarding Sub Agency"
    )

    if isinstance(
        subagency,
        dict
    ):

        subagency = (
            subagency.get(
                "name"
            )
            or subagency.get(
                "toptier_name"
            )
        )

    events.append({

        "ticker": ticker,

        "company": name,

        "award_id": award_id,

        "award_date": action_date[:10],

        "transaction_amount": amount,

        "agency": agency,

        "subagency": subagency,

        "award_type": row.get(
            "Award Type"
        )

    })


print(
    f"MASTER transactions found: {len(events):,}"
)

if not events:

    print()
    print(
        "NO PUBLIC COMPANY EVENTS FOUND."
    )
    print(
        "The problem is upstream of Yahoo."
    )
    print()
    raise SystemExit(1)

# ============================================================
# STEP 3 — LIMIT INITIAL STUDY
#
# We want enough events to examine but don't hammer Yahoo.
# ============================================================

events = sorted(
    events,
    key=lambda x: x["transaction_amount"],
    reverse=True
)

events = events[:100]

print(
    f"Events selected for market study: {len(events)}"
)

# ============================================================
# STEP 4 — STOCK EVENT ANALYSIS
# ============================================================

results = []

print()
print(
    "Downloading historical stock prices..."
)
print()

for number, event in enumerate(
    events,
    start=1
):

    ticker = event[
        "ticker"
    ]

    award_date = date.fromisoformat(
        event[
            "award_date"
        ]
    )

    start = award_date - timedelta(
        days=15
    )

    end = award_date + timedelta(
        days=45
    )

    print(
        f"[{number:03}/{len(events):03}] "
        f"{ticker} "
        f"{award_date} "
        f"${event['transaction_amount']/1e6:.1f}M"
    )

    try:

        hist = yf.download(
            ticker,
            start=str(start),
            end=str(end),
            auto_adjust=True,
            progress=False
        )

        if hist.empty:

            print(
                "   no price data"
            )

            continue

        # Handle yfinance MultiIndex.
        if isinstance(
            hist.columns,
            pd.MultiIndex
        ):

            close = hist[
                "Close"
            ]

            if isinstance(
                close,
                pd.DataFrame
            ):

                close = close.iloc[:, 0]

        else:

            close = hist[
                "Close"
            ]

        close = close.dropna()

        if len(close) < 15:

            print(
                "   insufficient price data"
            )

            continue

        # Remove timezone.
        index = pd.to_datetime(
            close.index
        )

        if getattr(
            index,
            "tz",
            None
        ) is not None:

            index = index.tz_localize(
                None
            )

        close.index = index

        # ----------------------------------------------------
        # Event trading day
        # ----------------------------------------------------

        event_timestamp = pd.Timestamp(
            award_date
        )

        future = close[
            close.index >= event_timestamp
        ]

        if future.empty:

            print(
                "   no trading day after award"
            )

            continue

        event_day = future.index[0]

        event_price = float(
            close.loc[
                event_day
            ]
        )

        # ----------------------------------------------------
        # PRE-EVENT BASELINE
        # ----------------------------------------------------

        before = close[
            close.index < event_timestamp
        ]

        if len(before) >= 5:

            pre_price = float(
                before.iloc[-1]
            )

        else:

            pre_price = event_price

        # ----------------------------------------------------
        # POST-EVENT WINDOW
        # ----------------------------------------------------

        post = close[
            close.index >= event_day
        ].iloc[:31]

        if len(post) < 5:

            continue

        # ----------------------------------------------------
        # VALLEY
        #
        # Lowest price from event through +30 trading days.
        # ----------------------------------------------------

        valley_price = float(
            post.min()
        )

        valley_date = post.idxmin()

        # ----------------------------------------------------
        # PEAK
        #
        # Highest price from event through +30 trading days.
        # ----------------------------------------------------

        peak_price = float(
            post.max()
        )

        peak_date = post.idxmax()

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        pre_to_event = (
            event_price / pre_price
            - 1
        ) * 100

        event_to_valley = (
            valley_price / event_price
            - 1
        ) * 100

        event_to_peak = (
            peak_price / event_price
            - 1
        ) * 100

        valley_to_peak = (
            peak_price / valley_price
            - 1
        ) * 100

        award_to_peak_days = (
            post.index.get_loc(
                peak_date
            )
        )

        award_to_valley_days = (
            post.index.get_loc(
                valley_date
            )
        )

        # ----------------------------------------------------
        # TIME TO TARGET
        # ----------------------------------------------------

        target_days = {}

        for target in [
            10,
            15,
            20,
            50,
            100
        ]:

            threshold = (
                event_price
                * (
                    1
                    + target / 100
                )
            )

            hit = post[
                post >= threshold
            ]

            if len(hit):

                first_hit = hit.index[0]

                target_days[
                    f"days_to_{target}pct"
                ] = post.index.get_loc(
                    first_hit
                )

            else:

                target_days[
                    f"days_to_{target}pct"
                ] = None

        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        result = dict(
            event
        )

        result.update({

            "event_trading_day":
                str(
                    event_day.date()
                ),

            "event_price":
                round(
                    event_price,
                    4
                ),

            "pre_price":
                round(
                    pre_price,
                    4
                ),

            "pre_to_event_pct":
                round(
                    pre_to_event,
                    2
                ),

            "valley_price":
                round(
                    valley_price,
                    4
                ),

            "valley_date":
                str(
                    valley_date.date()
                ),

            "award_to_valley_days":
                award_to_valley_days,

            "event_to_valley_pct":
                round(
                    event_to_valley,
                    2
                ),

            "peak_price":
                round(
                    peak_price,
                    4
                ),

            "peak_date":
                str(
                    peak_date.date()
                ),

            "award_to_peak_days":
                award_to_peak_days,

            "event_to_peak_pct":
                round(
                    event_to_peak,
                    2
                ),

            "valley_to_peak_pct":
                round(
                    valley_to_peak,
                    2
                )

        })

        result.update(
            target_days
        )

        results.append(
            result
        )

        print(
            f"   valley {event_to_valley:.1f}% "
            f"day {award_to_valley_days}; "
            f"peak {event_to_peak:.1f}% "
            f"day {award_to_peak_days}"
        )

    except Exception as e:

        print(
            f"   ERROR: {e}"
        )

    time.sleep(
        0.2
    )

# ============================================================
# STEP 5 — SAVE
# ============================================================

print()
print(
    "=" * 70
)

print(
    f"USABLE EVENTS: {len(results):,}"
)

if not results:

    print(
        "NO USABLE MARKET EVENTS."
    )

    raise SystemExit(1)

df = pd.DataFrame(
    results
)

df.to_csv(
    RESULTS,
    index=False
)

JSON_RESULTS.write_text(
    json.dumps(
        results,
        indent=2,
        default=str
    )
)

# ============================================================
# SUMMARY
# ============================================================

summary = {

    "date_range": {
        "start": str(START),
        "end": str(END)
    },

    "transactions_downloaded":
        len(all_rows),

    "master_events":
        len(events),

    "usable_events":
        len(results),

    "tickers":
        sorted(
            df["ticker"].unique().tolist()
        ),

    "average_event_to_peak_pct":
        round(
            float(
                df[
                    "event_to_peak_pct"
                ].mean()
            ),
            2
        ),

    "median_event_to_peak_pct":
        round(
            float(
                df[
                    "event_to_peak_pct"
                ].median()
            ),
            2
        ),

    "average_valley_to_peak_pct":
        round(
            float(
                df[
                    "valley_to_peak_pct"
                ].mean()
            ),
            2
        ),

    "median_valley_to_peak_pct":
        round(
            float(
                df[
                    "valley_to_peak_pct"
                ].median()
            ),
            2
        ),

    "average_award_to_peak_days":
        round(
            float(
                df[
                    "award_to_peak_days"
                ].mean()
            ),
            2
        ),

    "median_award_to_peak_days":
        round(
            float(
                df[
                    "award_to_peak_days"
                ].median()
            ),
            2
        ),

    "peak_10pct_rate":
        round(
            float(
                df[
                    "days_to_10pct"
                ].notna().mean()
                * 100
            ),
            2
        ),

    "peak_20pct_rate":
        round(
            float(
                df[
                    "days_to_20pct"
                ].notna().mean()
                * 100
            ),
            2
        ),

    "peak_50pct_rate":
        round(
            float(
                df[
                    "days_to_50pct"
                ].notna().mean()
                * 100
            ),
            2
        ),

    "peak_100pct_rate":
        round(
            float(
                df[
                    "days_to_100pct"
                ].notna().mean()
                * 100
            ),
            2
        )
}

SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2
    )
)

print()
print(
    "FILES CREATED:"
)

print(
    f"  {RESULTS}"
)

print(
    f"  {JSON_RESULTS}"
)

print(
    f"  {SUMMARY}"
)

print()
print(
    "SUMMARY"
)

print(
    json.dumps(
        summary,
        indent=2
    )
)

print()
print(
    "=" * 70
)

print(
    "EVENT STUDY COMPLETE"
)

print(
    "=" * 70
)

