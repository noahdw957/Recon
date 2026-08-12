# ============================================================
# RECON EVENT STUDY V7.2
# ============================================================
#
# PURPOSE
# -------
# Build a large historical population of qualifying government
# contract events involving publicly traded companies, then
# randomly sample 10% for the market event study.
#
# IMPORTANT
# ---------
# 1. Uses 365 calendar days of USAspending history.
# 2. Uses PER-TICKER recipient-targeted USAspending queries.
#    This is the V7.1 approach, NOT the old broad V7 query.
# 3. Saves the COMPLETE qualifying population locally.
# 4. Random sampling occurs AFTER population collection.
# 5. Sample = exactly 10% of the qualifying population.
# 6. Fixed random seed makes the sample reproducible.
# 7. Re-sampling can therefore be performed locally without
#    querying USAspending again.
# 8. Uses 90 TRADING DAYS of forward market data.
# 9. Recent events without a complete 90-day forward window
#    are excluded from the MARKET STUDY, but remain in the
#    population file.
#
# OUTPUTS
# --------
# event_study_population.csv
# event_study_population.json
# event_study_events.csv
# event_study_events.json
# event_study_scatter.csv
# event_study_summary.json
#
# ============================================================

import requests
import pandas as pd
import json
import time
import random
from datetime import date, timedelta
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

DAYS_BACK = 365

MIN_TRANSACTION = 1_000_000

SAMPLE_FRACTION = 0.10

# Fixed seed = reproducible random sample
RANDOM_SEED = 20260812

# Maximum number of population events retained per ticker.
# None means NO artificial cap.
MAX_EVENTS_PER_TICKER = None

# Forward market study
FORWARD_TRADING_DAYS = 90

# Extra calendar days downloaded to obtain 90 trading days.
MARKET_CALENDAR_DAYS = 150

# ============================================================
# DATES
# ============================================================

END = date.today()
START = END - timedelta(days=DAYS_BACK)

# ============================================================
# USAspending
# ============================================================

API = (
    "https://api.usaspending.gov/api/v2/"
    "search/spending_by_transaction/"
)

# ============================================================
# MASTER VENDOR LIST
#
# Broad enough to test whether ticker/company really matters.
# The random sample is NOT balanced by ticker.
# ============================================================

MASTER = {

    # ---- Defense / aerospace primes ----

    "LMT": ["LOCKHEED MARTIN"],
    "RTX": ["RAYTHEON", "RTX"],
    "NOC": ["NORTHROP GRUMMAN"],
    "GD": ["GENERAL DYNAMICS"],
    "BA": ["BOEING"],
    "HII": ["HUNTINGTON INGALLS"],
    "TXT": ["TEXTRON"],
    "HWM": ["HOWMET AEROSPACE"],
    "TDG": ["TRANSDIGM"],
    "SPR": ["SPIRIT AEROSYSTEMS"],

    # ---- Defense electronics / systems ----

    "LHX": ["L3HARRIS", "L3 HARRIS"],
    "LDOS": ["LEIDOS"],
    "SAIC": [
        "SAIC",
        "SCIENCE APPLICATIONS INTERNATIONAL"
    ],
    "BAH": ["BOOZ ALLEN"],
    "CACI": ["CACI"],
    "KTOS": ["KRATOS"],
    "BWXT": ["BWX TECHNOLOGIES"],
    "MRCY": ["MERCURY SYSTEMS"],
    "DRS": ["LEONARDO DRS", "DRS"],
    "PSN": ["PARSONS"],

    # ---- Defense / aerospace smaller companies ----

    "PLTR": ["PALANTIR"],
    "RCAT": ["RED CAT"],
    "AVAV": ["AEROVIRONMENT"],
    "KTOS": ["KRATOS"],
    "JOBY": ["JOBY"],
    "ACHR": ["ARCHER AVIATION"],
    "EVEX": ["EVE AIR MOBILITY"],
    "ONDS": ["ONDO"],
    "UAVS": ["AG EAGLE"],
    "VSEC": ["VSE"],

    # ---- Advanced technology / autonomy / space ----

    "RKLB": ["ROCKET LAB"],
    "ASTS": ["AST SPACE MOBILE"],
    "RDW": ["REDWIRE"],
    "LUNR": ["INTUITIVE MACHINES"],
    "BKSY": ["BLACKSKY"],
    "SPIR": ["SPHERE"],
    "SATL": ["SATELLITE"],
    "MRCY": ["MERCURY SYSTEMS"],
    "AEVA": ["AEVA"],

    # ---- Industrial / government-exposed ----

    "WWD": ["WOODWARD"],
    "HON": ["HONEYWELL"],
    "GE": ["GENERAL ELECTRIC"],
    "ETN": ["EATON"],
    "CAT": ["CATERPILLAR"],
    "DE": ["DEERE"],
    "EMR": ["EMERSON"],
    "PH": ["PARKER HANNIFIN"],
    "IR": ["INGERSOLL RAND"],
    "ROK": ["ROCKWELL AUTOMATION"],

    # ---- IT / government services ----

    "IBM": ["INTERNATIONAL BUSINESS MACHINES"],
    "ACN": ["ACCENTURE"],
    "GLOB": ["GLOBANT"],
    "CTSH": ["COGNIZANT"],
    "LDOS": ["LEIDOS"],
}

# Remove duplicate ticker keys automatically through dict behavior,
# but keep the master easy to edit.

# ============================================================
# OUTPUT FILES
# ============================================================

POPULATION_CSV = Path("event_study_population.csv")
POPULATION_JSON = Path("event_study_population.json")

EVENTS_CSV = Path("event_study_events.csv")
EVENTS_JSON = Path("event_study_events.json")

SCATTER_CSV = Path("event_study_scatter.csv")

SUMMARY_JSON = Path("event_study_summary.json")

# Recon copies
POPULATION_CSV_R = Path("Recon/event_study_population.csv")
POPULATION_JSON_R = Path("Recon/event_study_population.json")

EVENTS_CSV_R = Path("Recon/event_study_events.csv")
EVENTS_JSON_R = Path("Recon/event_study_events.json")

SCATTER_CSV_R = Path("Recon/event_study_scatter.csv")

SUMMARY_JSON_R = Path("Recon/event_study_summary.json")

# ============================================================
# SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "RECON-Event-Study/7.2"
})

# ============================================================
# MASTER LOOKUP
# ============================================================

def master_lookup(name):

    if not name:
        return None

    u = str(name).upper()

    for ticker, keywords in MASTER.items():

        for kw in keywords:

            if kw in u:
                return ticker

    return None


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("RECON EVENT STUDY V7.2")
print("=" * 72)
print(f"Date range:       {START} -> {END}")
print(f"Minimum award:    ${MIN_TRANSACTION:,}")
print(f"Master companies: {len(MASTER)}")
print(f"Sample fraction:  {SAMPLE_FRACTION:.0%}")
print(f"Random seed:      {RANDOM_SEED}")
print("=" * 72)


# ============================================================
# STEP 1
# DOWNLOAD COMPLETE POPULATION
#
# PER-TICKER QUERY
# ============================================================

population_rows = []

seen = set()

for ticker, keywords in MASTER.items():

    # Combine keywords into one search string.
    # USAspending recipient search supports recipient targeting.
    recipient_search = " ".join(keywords)

    print()
    print("-" * 72)
    print(f"QUERY: {ticker}")
    print(f"Recipient search: {recipient_search}")

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
            ],

            "recipient_search_text": [
                recipient_search
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

    ticker_rows = []

    for page in range(1, 101):

        payload["page"] = page

        try:

            r = session.post(
                API,
                json=payload,
                timeout=90
            )

            print(
                f"{ticker} | page {page} | "
                f"HTTP {r.status_code}",
                end=" "
            )

            r.raise_for_status()

            data = r.json()

            rows = data.get("results", [])

            print(f"| records {len(rows)}")

            if not rows:
                break

            ticker_rows.extend(rows)

            if not data.get(
                "page_metadata",
                {}
            ).get(
                "hasNext",
                False
            ):
                break

            time.sleep(0.25)

        except Exception as ex:

            print(
                f"\nERROR {ticker} page {page}: {ex}"
            )

            break

    print(
        f"{ticker}: downloaded "
        f"{len(ticker_rows):,} records"
    )

    # --------------------------------------------------------
    # LOCAL FILTER + DEDUPLICATION
    # --------------------------------------------------------

    local_count = 0

    for row in ticker_rows:

        recipient = row.get("Recipient Name")

        matched_ticker = master_lookup(recipient)

        if matched_ticker != ticker:
            continue

        award_id = row.get("Award ID")

        action_date = row.get("Action Date")

        amount = row.get("Transaction Amount")

        key = (
            ticker,
            award_id,
            action_date,
            amount
        )

        if key in seen:
            continue

        seen.add(key)

        try:
            amount_float = float(amount)
        except:
            continue

        agency = row.get(
            "Awarding Agency"
        )

        if isinstance(agency, dict):

            agency = (
                agency.get("toptier_name")
                or agency.get("name")
            )

        subagency = row.get(
            "Awarding Sub Agency"
        )

        if isinstance(subagency, dict):

            subagency = (
                subagency.get("name")
                or subagency.get("toptier_name")
            )

        population_rows.append({

            "ticker": ticker,

            "company": recipient,

            "award_id": award_id,

            "award_date":
                str(action_date)[:10],

            "transaction_amount":
                amount_float,

            "agency": agency,

            "subagency": subagency,

            "award_type":
                row.get("Award Type")

        })

        local_count += 1

    print(
        f"{ticker}: qualifying "
        f"{local_count:,}"
    )

    time.sleep(0.5)


# ============================================================
# STEP 2
# SAVE COMPLETE POPULATION
# ============================================================

population_df = pd.DataFrame(
    population_rows
)

if population_df.empty:

    print("\nNO QUALIFYING EVENTS FOUND.")

    population_df = pd.DataFrame(
        [{"error": "no qualifying events"}]
    )

else:

    population_df = (
        population_df
        .drop_duplicates()
        .sort_values(
            "award_date"
        )
        .reset_index(drop=True)
    )

population_count = len(
    population_df
)

print()
print("=" * 72)
print(
    f"COMPLETE POPULATION: "
    f"{population_count:,}"
)
print("=" * 72)

# Save population locally.

for path in [
    POPULATION_CSV,
    POPULATION_CSV_R
]:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    population_df.to_csv(
        path,
        index=False
    )

population_json_data = (
    population_df
    .to_dict(orient="records")
)

for path in [
    POPULATION_JSON,
    POPULATION_JSON_R
]:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            population_json_data,
            indent=2,
            default=str
        )
    )

print(
    f"Saved complete population to "
    f"{POPULATION_CSV}"
)


# ============================================================
# STEP 3
# RANDOM SAMPLE
#
# IMPORTANT:
# This happens entirely locally.
# No additional USAspending requests.
# ============================================================

if population_count == 0:

    print("Population empty. Stopping.")

    summary = {
        "date_range": {
            "start": str(START),
            "end": str(END)
        },
        "population_events": 0,
        "sample_events": 0,
        "sample_fraction": SAMPLE_FRACTION,
        "random_seed": RANDOM_SEED
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2
        )
    )

    raise SystemExit(0)


sample_size = max(
    1,
    int(round(
        population_count *
        SAMPLE_FRACTION
    ))
)

# Never sample more than population.
sample_size = min(
    sample_size,
    population_count
)

sample_df = population_df.sample(
    n=sample_size,
    random_state=RANDOM_SEED
).copy()

sample_df = sample_df.reset_index(
    drop=True
)

print()
print("=" * 72)
print(
    f"RANDOM SAMPLE: "
    f"{sample_size:,} "
    f"of {population_count:,}"
)
print(
    f"Sampling fraction: "
    f"{sample_size / population_count:.2%}"
)
print("=" * 72)

# Save sampled events before market study.

sample_df.to_csv(
    EVENTS_CSV,
    index=False
)

sample_df.to_csv(
    EVENTS_CSV_R,
    index=False
)

sample_records = (
    sample_df
    .to_dict(orient="records")
)

EVENTS_JSON.write_text(
    json.dumps(
        sample_records,
        indent=2,
        default=str
    )
)

EVENTS_JSON_R.write_text(
    json.dumps(
        sample_records,
        indent=2,
        default=str
    )
)


# ============================================================
# STEP 4
# MARKET EVENT STUDY
#
# 90 TRADING DAYS AFTER AWARD
# ============================================================

import yfinance as yf

results = []

print()
print("=" * 72)
print("MARKET EVENT STUDY")
print("=" * 72)

for i, ev in enumerate(
    sample_records,
    1
):

    ticker = ev["ticker"]

    try:

        award_date = date.fromisoformat(
            ev["award_date"]
        )

    except:

        print(
            f"[{i:03}/{sample_size}] "
            f"{ticker} invalid date"
        )

        continue

    # Download enough calendar days to get
    # approximately 90 trading days.

    start_date = (
        award_date -
        timedelta(days=20)
    )

    end_date = (
        award_date +
        timedelta(
            days=MARKET_CALENDAR_DAYS
        )
    )

    print(
        f"[{i:03}/{sample_size}] "
        f"{ticker} "
        f"{award_date} "
        f"${ev['transaction_amount']/1e6:.1f}M",
        end=" "
    )

    try:

        hist = yf.download(
            ticker,
            start=str(start_date),
            end=str(end_date),
            auto_adjust=True,
            progress=False
        )

        if hist.empty:

            print("NO DATA")

            continue

        close = hist["Close"]

        if isinstance(
            close,
            pd.DataFrame
        ):

            close = close.iloc[:, 0]

        close = close.dropna()

        if len(close) < 20:

            print("INSUFFICIENT DATA")

            continue

        idx = pd.to_datetime(
            close.index
        )

        if getattr(
            idx,
            "tz",
            None
        ) is not None:

            idx = idx.tz_localize(
                None
            )

        close.index = idx

        event_ts = pd.Timestamp(
            award_date
        )

        # First trading day ON or after award.

        future = close[
            close.index >= event_ts
        ]

        if future.empty:

            print(
                "NO TRADING DAY"
            )

            continue

        event_day = future.index[0]

        event_price = float(
            close.loc[event_day]
        )

        # Last trading day BEFORE award.

        before = close[
            close.index < event_ts
        ]

        if len(before) < 1:

            print(
                "NO PRE-EVENT PRICE"
            )

            continue

        pre_price = float(
            before.iloc[-1]
        )

        # ----------------------------------------------------
        # 90 TRADING DAY WINDOW
        # ----------------------------------------------------

        post = (
            close[
                close.index >= event_day
            ]
            .iloc[
                :FORWARD_TRADING_DAYS + 1
            ]
        )

        # Require complete 90-trading-day window.
        #
        # event day = day 0
        # therefore need 91 observations total.

        if len(post) < (
            FORWARD_TRADING_DAYS + 1
        ):

            print(
                f"ONLY {len(post)} DAYS"
            )

            continue

        # ----------------------------------------------------
        # VALLEY
        # ----------------------------------------------------

        valley_price = float(
            post.min()
        )

        valley_date = post.idxmin()

        valley_day = int(
            post.index.get_loc(
                valley_date
            )
        )

        # ----------------------------------------------------
        # PEAK
        # ----------------------------------------------------

        peak_price = float(
            post.max()
        )

        peak_date = post.idxmax()

        peak_day = int(
            post.index.get_loc(
                peak_date
            )
        )

        # ----------------------------------------------------
        # BASIC RETURNS
        # ----------------------------------------------------

        pre_to_event = (
            (
                event_price /
                pre_price
            ) - 1
        ) * 100

        event_to_valley = (
            (
                valley_price /
                event_price
            ) - 1
        ) * 100

        event_to_peak = (
            (
                peak_price /
                event_price
            ) - 1
        ) * 100

        valley_to_peak = (
            (
                peak_price /
                valley_price
            ) - 1
        ) * 100

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        res = dict(ev)

        res.update({

            "event_trading_day":
                str(event_day.date()),

            "event_price":
                round(event_price, 4),

            "pre_price":
                round(pre_price, 4),

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
                valley_day,

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
                peak_day,

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

        # ----------------------------------------------------
        # TARGET TIMES
        # ----------------------------------------------------

        for target in [
            5,
            10,
            15,
            20,
            30,
            50,
            100
        ]:

            threshold = (
                event_price *
                (
                    1 +
                    target / 100
                )
            )

            hit = post[
                post >= threshold
            ]

            if len(hit):

                first_hit = hit.index[0]

                res[
                    f"days_to_{target}pct"
                ] = int(
                    post.index.get_loc(
                        first_hit
                    )
                )

            else:

                res[
                    f"days_to_{target}pct"
                ] = None

        # ----------------------------------------------------
        # PRICE AT FIXED TIME POINTS
        # ----------------------------------------------------

        for day_n in [
            1,
            3,
            5,
            7,
            10,
            14,
            21,
            30,
            45,
            60,
            90
        ]:

            if day_n < len(post):

                px = float(
                    post.iloc[day_n]
                )

                pct = (
                    (
                        px /
                        event_price
                    ) - 1
                ) * 100

                res[
                    f"return_day_{day_n}pct"
                ] = round(
                    pct,
                    2
                )

            else:

                res[
                    f"return_day_{day_n}pct"
                ] = None

        results.append(res)

        print(
            f"-> valley "
            f"{event_to_valley:.1f}% "
            f"peak "
            f"{event_to_peak:.1f}% "
            f"day "
            f"{peak_day}"
        )

    except Exception as ex:

        print(
            f"ERROR {ex}"
        )

    time.sleep(0.20)


# ============================================================
# STEP 5
# SAVE MARKET RESULTS
# ============================================================

print()
print("=" * 72)
print(
    f"USABLE MARKET EVENTS: "
    f"{len(results):,}"
)
print("=" * 72)

if not results:

    print(
        "No usable market events."
    )

    df = pd.DataFrame(
        [{
            "note":
                "no usable market events",
            "population_events":
                population_count,
            "sample_events":
                sample_size
        }]
    )

else:

    df = pd.DataFrame(
        results
    )


# ============================================================
# SAVE CSV
# ============================================================

for path in [
    EVENTS_CSV,
    EVENTS_CSV_R
]:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )


# Scatter file is deliberately identical
# and ready for dashboard analysis.

for path in [
    SCATTER_CSV,
    SCATTER_CSV_R
]:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        path,
        index=False
    )


# ============================================================
# SAVE JSON
# ============================================================

result_records = (
    df.to_dict(
        orient="records"
    )
)

EVENTS_JSON.write_text(
    json.dumps(
        result_records,
        indent=2,
        default=str
    )
)

EVENTS_JSON_R.write_text(
    json.dumps(
        result_records,
        indent=2,
        default=str
    )
)


# ============================================================
# STEP 6
# SUMMARY
# ============================================================

summary = {

    "version":
        "RECON Event Study V7.2",

    "date_range": {

        "start":
            str(START),

        "end":
            str(END)
    },

    "minimum_transaction":
        MIN_TRANSACTION,

    "master_company_count":
        len(MASTER),

    "population_events":
        population_count,

    "sample_fraction":
        SAMPLE_FRACTION,

    "sample_events":
        sample_size,

    "random_seed":
        RANDOM_SEED,

    "usable_market_events":
        len(results),

    "forward_trading_days":
        FORWARD_TRADING_DAYS,

    "sampling_method":
        "simple random sample without replacement",

    "sampling_stage":
        "after complete population collection",

    "usa_spending_query_method":
        "per-ticker recipient-targeted",

    "population_saved":
        True
}


# ------------------------------------------------------------
# TICKER DISTRIBUTION
# ------------------------------------------------------------

if (
    population_count > 0
    and "ticker" in population_df.columns
):

    summary[
        "population_by_ticker"
    ] = (
        population_df[
            "ticker"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

if (
    len(results) > 0
    and "ticker" in df.columns
):

    summary[
        "sample_by_ticker"
    ] = (
        df[
            "ticker"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )


# ------------------------------------------------------------
# EVENT METRICS
# ------------------------------------------------------------

if (
    len(results) > 0
    and "event_to_peak_pct" in df.columns
):

    summary.update({

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
            )
    })


# ============================================================
# SAVE SUMMARY
# ============================================================

for path in [
    SUMMARY_JSON,
    SUMMARY_JSON_R
]:

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            summary,
            indent=2
        )
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("V7.2 COMPLETE")
print("=" * 72)

print(
    f"365-day population: "
    f"{population_count:,}"
)

print(
    f"Random 10% sample: "
    f"{sample_size:,}"
)

print(
    f"Usable 90-day events: "
    f"{len(results):,}"
)

print()
print("Population distribution:")

if (
    population_count > 0
    and "ticker" in population_df.columns
):

    print(
        population_df[
            "ticker"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

print()
print("Sample distribution:")

if (
    len(results) > 0
    and "ticker" in df.columns
):

    print(
        df[
            "ticker"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

print()
print("FILES CREATED:")
print(
    "  event_study_population.csv"
)
print(
    "  event_study_population.json"
)
print(
    "  event_study_events.csv"
)
print(
    "  event_study_events.json"
)
print(
    "  event_study_scatter.csv"
)
print(
    "  event_study_summary.json"
)

print()
print(
    json.dumps(
        summary,
        indent=2
    )
)

print("=" * 72)
