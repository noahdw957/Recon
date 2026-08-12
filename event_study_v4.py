# RECON EVENT STUDY V4.0
# Historical award -> stock-response dataset
#
# PURPOSE
# Collect real USAspending transaction events for known public contractors,
# pair each event with historical market prices, and produce CSV/JSON files
# for the Event Study Lab.
#
# This version deliberately DOES NOT assume the answer.
# It records:
#   award date
#   award size
#   agency / subagency / contract metadata
#   prior award activity
#   stock price path
#   pre-event move
#   event -> peak
#   event -> valley
#   valley -> peak
#   first day reaching +10/+15/+20/+50/+100%
#   market-relative returns using SPY
#
# Press-release and earnings dates are NOT fabricated here. They are a
# separate layer to be added after this clean award/price event set exists.

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# SETTINGS
# ============================================================

DAYS_BACK = 365
MIN_AWARD = 1_000_000
MAX_EVENTS = 100

# Avoid treating a cluster of modifications/awards as independent events.
EVENT_SPACING_DAYS = 14

# Market window around the award.
PRE_TRADING_DAYS = 20
POST_TRADING_DAYS = 60

# Historical award history used to build early-known control factors.
HISTORY_DAYS = 365

USA_API = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"

OUTPUT_JSON = Path("event_study_v4.json")
OUTPUT_CSV = Path("event_study_v4.csv")
SUMMARY_JSON = Path("event_study_summary.json")

MASTER = {
    "PLTR": ["PALANTIR"],
    "RCAT": ["RED CAT"],
    "AVAV": ["AEROVIRONMENT"],
    "WWD": ["WOODWARD"],
    "AEVA": ["AEVA"],
    "LMT": ["LOCKHEED MARTIN"],
    "RTX": ["RAYTHEON", "RAYTHEON TECHNOLOGIES", "RTX"],
    "BAH": ["BOOZ ALLEN"],
    "SAIC": ["SCIENCE APPLICATIONS INTERNATIONAL", "SAIC"],
    "LDOS": ["LEIDOS"],
    "LHX": ["L3HARRIS", "L3 HARRIS"],
    "NOC": ["NORTHROP GRUMMAN"],
}


# ============================================================
# HELPERS
# ============================================================

def as_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except Exception:
        return None


def master_lookup(name):
    upper = (name or "").upper()
    for ticker, keywords in MASTER.items():
        if any(keyword in upper for keyword in keywords):
            return ticker
    return None


def normalize_dict_value(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("toptier_name")
    return value


def pct(start, end):
    if start in (None, 0) or end is None:
        return None
    return (end / start - 1.0) * 100.0


def first_hit(price_by_offset, baseline, target_pct):
    if baseline in (None, 0):
        return None

    target = baseline * (1.0 + target_pct / 100.0)

    for day in sorted(price_by_offset):
        if day <= 0:
            continue
        price = price_by_offset[day]
        if price >= target:
            return int(day)

    return None


def first_drawdown(price_by_offset, baseline, target_pct):
    """First trading day price falls target_pct below event price."""
    if baseline in (None, 0):
        return None

    target = baseline * (1.0 - abs(target_pct) / 100.0)

    for day in sorted(price_by_offset):
        if day <= 0:
            continue
        if price_by_offset[day] <= target:
            return int(day)

    return None


def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return None if s.empty else float(s.mean())


def safe_median(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return None if s.empty else float(s.median())


# ============================================================
# DOWNLOAD USAspending TRANSACTIONS
# ============================================================

END = date.today()
START = END - timedelta(days=DAYS_BACK)

payload = {
    "filters": {
        "award_amounts": [{"lower_bound": MIN_AWARD}],
        "award_type_codes": ["A", "B", "C", "D"],
        "time_period": [{
            "start_date": str(START),
            "end_date": str(END)
        }]
    },
    "fields": [
        "Award ID",
        "Recipient Name",
        "Action Date",
        "Transaction Amount",
        "Award Amount",
        "Awarding Agency",
        "Awarding Sub Agency",
        "Funding Agency",
        "Funding Sub Agency",
        "Award Type",
        "Contract Award Type",
        "Action Type",
        "Mod",
        "Description"
    ],
    "sort": "Transaction Amount",
    "order": "desc",
    "limit": 100,
    "page": 1
}

session = requests.Session()
session.headers.update({
    "User-Agent": "RECON-Event-Study/4.0"
})

all_rows = []
page = 1

print(f"Downloading USAspending transactions {START} -> {END}...")

while True:
    payload["page"] = page

    response = session.post(
        USA_API,
        json=payload,
        timeout=60
    )
    response.raise_for_status()

    page_data = response.json()
    batch = page_data.get("results", [])

    if not batch:
        break

    all_rows.extend(batch)

    print(f"  page {page}: {len(batch)} transactions")

    if not page_data.get("page_metadata", {}).get("hasNext", False):
        break

    page += 1
    time.sleep(0.25)

print(f"Transactions downloaded: {len(all_rows)}")


# ============================================================
# EXTRACT PUBLIC-COMPANY INITIAL AWARD EVENTS
# ============================================================

candidates = []

for row in all_rows:
    name = row.get("Recipient Name")
    ticker = master_lookup(name)

    if not ticker:
        continue

    action_date = row.get("Action Date")
    amount = as_float(
        row.get("Transaction Amount")
        if row.get("Transaction Amount") not in (None, "")
        else row.get("Award Amount")
    )

    if not action_date or amount is None or amount < MIN_AWARD:
        continue

    # We want initial award actions, not later modifications.
    mod = str(row.get("Mod") or "").strip().upper()
    if mod not in ("", "0", "0000", "BASE"):
        continue

    agency = normalize_dict_value(row.get("Awarding Agency"))
    subagency = normalize_dict_value(row.get("Awarding Sub Agency"))
    funding_agency = normalize_dict_value(row.get("Funding Agency"))
    funding_subagency = normalize_dict_value(row.get("Funding Sub Agency"))

    candidates.append({
        "award_id": row.get("Award ID"),
        "ticker": ticker,
        "company": name,
        "award_date": str(action_date)[:10],
        "award_amount": amount,
        "agency": agency,
        "subagency": subagency,
        "funding_agency": funding_agency,
        "funding_subagency": funding_subagency,
        "award_type": row.get("Award Type"),
        "contract_award_type": row.get("Contract Award Type"),
        "action_type": row.get("Action Type"),
        "description": row.get("Description"),
    })

events = pd.DataFrame(candidates)

if events.empty:
    raise RuntimeError(
        "No matching public-company award events were found."
    )

events["award_date"] = pd.to_datetime(
    events["award_date"],
    errors="coerce"
)

events = events.dropna(subset=["award_date"])

if "award_id" in events.columns:
    events = events.drop_duplicates(
        subset=["award_id"],
        keep="first"
    )

events = events.sort_values(
    ["ticker", "award_date", "award_amount"],
    ascending=[True, True, False]
)

# ============================================================
# SELECT EVENTS
#
# We select the largest event in each 14-day neighborhood.
# This reduces double-counting of the same news/contract cluster.
# ============================================================

selected = []

for ticker, group in events.groupby("ticker"):
    chosen = []

    for _, event in group.sort_values(
        "award_amount",
        ascending=False
    ).iterrows():

        if all(
            abs((event.award_date - prior.award_date).days)
            > EVENT_SPACING_DAYS
            for prior in chosen
        ):
            chosen.append(event)

    selected.extend(chosen)

events = pd.DataFrame(selected)

events = events.sort_values(
    "award_amount",
    ascending=False
).head(MAX_EVENTS)

print(f"Clean event candidates selected: {len(events)}")


# ============================================================
# MARKET DATA
# ============================================================

# Download SPY once. This gives us a market-relative comparison
# without pretending that raw stock movement is all event-driven.

earliest = events["award_date"].min().date()
latest = events["award_date"].max().date()

spy_start = earliest - timedelta(days=180)
spy_end = latest + timedelta(days=POST_TRADING_DAYS + 10)

print("Downloading SPY benchmark...")

spy = yf.Ticker("SPY").history(
    start=str(spy_start),
    end=str(spy_end),
    interval="1d",
    auto_adjust=True,
    actions=False
)

if spy.empty:
    raise RuntimeError("Could not download SPY history.")

spy.index = pd.to_datetime(spy.index).tz_localize(None)
spy_close = spy["Close"].dropna()


# ============================================================
# EVENT RECORDS
# ============================================================

records = []

for number, (_, event) in enumerate(
    events.iterrows(),
    start=1
):
    ticker = event["ticker"]
    award_date = event["award_date"].date()

    print(
        f"[{number}/{len(events)}] "
        f"{ticker} {award_date} "
        f"${event['award_amount']/1e6:.1f}M"
    )

    try:
        stock = yf.Ticker(ticker)

        history = stock.history(
            start=str(
                award_date
                - timedelta(days=60)
            ),
            end=str(
                award_date
                + timedelta(days=POST_TRADING_DAYS + 20)
            ),
            interval="1d",
            auto_adjust=True,
            actions=False
        )

        if history.empty:
            print("  SKIP: no market history")
            continue

        history.index = (
            pd.to_datetime(history.index)
            .tz_localize(None)
        )

        closes = history["Close"].dropna()

        event_candidates = closes[
            closes.index >= pd.Timestamp(award_date)
        ]

        prior = closes[
            closes.index < pd.Timestamp(award_date)
        ]

        if event_candidates.empty or len(prior) < PRE_TRADING_DAYS:
            print("  SKIP: insufficient price history")
            continue

        # First trading session on or after the award date.
        event_trading_date = event_candidates.index[0]
        event_price = float(event_candidates.iloc[0])

        dates = list(closes.index)
        event_position = dates.index(event_trading_date)

        lo = max(
            0,
            event_position - PRE_TRADING_DAYS
        )

        hi = min(
            len(dates) - 1,
            event_position + POST_TRADING_DAYS
        )

        price_by_offset = {
            i - event_position: float(closes.iloc[i])
            for i in range(lo, hi + 1)
        }

        post = {
            day: price
            for day, price in price_by_offset.items()
            if 0 <= day <= POST_TRADING_DAYS
        }

        pre = {
            day: price
            for day, price in price_by_offset.items()
            if -10 <= day <= -1
        }

        if not post:
            continue

        # Peak is the highest adjusted close in +1..+60.
        post_after_day_zero = {
            day: price
            for day, price in post.items()
            if day > 0
        }

        if not post_after_day_zero:
            continue

        peak_day, peak_price = max(
            post_after_day_zero.items(),
            key=lambda item: item[1]
        )

        # Valley is the lowest price from event through peak.
        pre_peak = {
            day: price
            for day, price in post.items()
            if day <= peak_day
        }

        valley_day, valley_price = min(
            pre_peak.items(),
            key=lambda item: item[1]
        )

        pre10 = pre.get(-10)
        pre1 = pre.get(-1)

        pre_move = pct(pre10, pre1)
        event_to_peak = pct(event_price, peak_price)
        event_to_valley = pct(event_price, valley_price)
        valley_to_peak = pct(valley_price, peak_price)

        # ----------------------------------------------------
        # SPY / market-relative comparison
        # ----------------------------------------------------

        spy_event_candidates = spy_close[
            spy_close.index >= pd.Timestamp(award_date)
        ]

        if spy_event_candidates.empty:
            continue

        spy_event_date = spy_event_candidates.index[0]
        spy_event_price = float(
            spy_event_candidates.iloc[0]
        )

        spy_dates = list(spy_close.index)
        spy_event_position = spy_dates.index(
            spy_event_date
        )

        spy_pre10_pos = (
            spy_event_position - 10
        )

        spy_pre1_pos = (
            spy_event_position - 1
        )

        spy_peak_pos = min(
            len(spy_dates) - 1,
            spy_event_position + POST_TRADING_DAYS
        )

        if spy_pre10_pos >= 0 and spy_pre1_pos >= 0:
            spy_pre_move = pct(
                float(spy_close.iloc[spy_pre10_pos]),
                float(spy_close.iloc[spy_pre1_pos])
            )
        else:
            spy_pre_move = None

        spy_post_window = spy_close.iloc[
            spy_event_position:
            spy_peak_pos + 1
        ]

        if not spy_post_window.empty:
            spy_peak_price = float(
                spy_post_window.max()
            )
            spy_event_to_peak = pct(
                spy_event_price,
                spy_peak_price
            )
        else:
            spy_event_to_peak = None

        market_relative_pre = (
            None
            if pre_move is None or spy_pre_move is None
            else pre_move - spy_pre_move
        )

        market_relative_peak = (
            None
            if event_to_peak is None
            or spy_event_to_peak is None
            else event_to_peak - spy_event_to_peak
        )

        # ----------------------------------------------------
        # EARLY-KNOWN AWARD HISTORY
        #
        # We use only awards whose Action Date is BEFORE this
        # event date. These are potential future control factors.
        # ----------------------------------------------------

        prior_awards = events[
            (events["ticker"] == ticker)
            & (events["award_date"] < event["award_date"])
            & (
                events["award_date"]
                >= event["award_date"]
                - pd.Timedelta(days=HISTORY_DAYS)
            )
        ]

        prior_count = len(prior_awards)

        if prior_count:
            prior_total = float(
                prior_awards["award_amount"].sum()
            )
            prior_mean = float(
                prior_awards["award_amount"].mean()
            )
            days_since_prior = int(
                (
                    event["award_date"]
                    - prior_awards["award_date"].max()
                ).days
            )
        else:
            prior_total = 0.0
            prior_mean = None
            days_since_prior = None

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        record = {
            "award_id": event["award_id"],
            "ticker": ticker,
            "company": event["company"],
            "award_date": str(award_date),
            "event_trading_date": str(
                event_trading_date.date()
            ),

            "award_amount": round(
                float(event["award_amount"]),
                2
            ),
            "award_m": round(
                float(event["award_amount"]) / 1e6,
                3
            ),

            "agency": event["agency"],
            "subagency": event["subagency"],
            "funding_agency": event["funding_agency"],
            "funding_subagency": event["funding_subagency"],
            "award_type": event["award_type"],
            "contract_award_type": event["contract_award_type"],
            "action_type": event["action_type"],
            "description": event["description"],

            "event_price": round(
                event_price,
                4
            ),

            "pre_move_10_to_1_pct": (
                None
                if pre_move is None
                else round(pre_move, 4)
            ),

            "spy_pre_move_10_to_1_pct": (
                None
                if spy_pre_move is None
                else round(spy_pre_move, 4)
            ),

            "market_relative_pre_pct": (
                None
                if market_relative_pre is None
                else round(
                    market_relative_pre,
                    4
                )
            ),

            "event_to_peak_pct": round(
                event_to_peak,
                4
            ),
            "peak_day": int(peak_day),
            "peak_price": round(
                peak_price,
                4
            ),

            "event_to_valley_pct": round(
                event_to_valley,
                4
            ),
            "valley_day": int(valley_day),
            "valley_price": round(
                valley_price,
                4
            ),

            "valley_to_peak_pct": round(
                valley_to_peak,
                4
            ),
            "valley_to_peak_days": int(
                peak_day - valley_day
            ),

            "market_relative_peak_pct": (
                None
                if market_relative_peak is None
                else round(
                    market_relative_peak,
                    4
                )
            ),

            "hit_10_day": first_hit(
                post,
                event_price,
                10
            ),
            "hit_15_day": first_hit(
                post,
                event_price,
                15
            ),
            "hit_20_day": first_hit(
                post,
                event_price,
                20
            ),
            "hit_50_day": first_hit(
                post,
                event_price,
                50
            ),
            "hit_100_day": first_hit(
                post,
                event_price,
                100
            ),

            "drawdown_5_day": first_drawdown(
                post,
                event_price,
                5
            ),
            "drawdown_10_day": first_drawdown(
                post,
                event_price,
                10
            ),

            "positive_1d": (
                post.get(1, event_price)
                > event_price
            ),
            "positive_5d": (
                post.get(5, event_price)
                > event_price
            ),
            "positive_10d": (
                post.get(10, event_price)
                > event_price
            ),
            "positive_30d": (
                post.get(30, event_price)
                > event_price
            ),

            "leaked_5pct": (
                pre_move is not None
                and pre_move >= 5
            ),

            # Early-known candidate controls:
            "prior_awards_365d": prior_count,
            "prior_award_total_365d": round(
                prior_total,
                2
            ),
            "prior_award_mean_365d": (
                None
                if prior_mean is None
                else round(
                    prior_mean,
                    2
                )
            ),
            "days_since_prior_award": (
                days_since_prior
            ),
        }

        records.append(record)

        print(
            f"  peak {event_to_peak:+.1f}% "
            f"day {peak_day}; "
            f"valley {event_to_valley:+.1f}% "
            f"day {valley_day}; "
            f"V->P {valley_to_peak:+.1f}%"
        )

    except Exception as exc:
        print(
            f"  ERROR {ticker}: {exc}"
        )

    time.sleep(0.25)


# ============================================================
# SAVE
# ============================================================

if not records:
    raise RuntimeError(
        "No usable historical market events were produced."
    )

output = pd.DataFrame(records)

output = output.sort_values(
    ["award_date", "ticker"]
)

output.to_csv(
    OUTPUT_CSV,
    index=False
)

OUTPUT_JSON.write_text(
    json.dumps(
        output.to_dict("records"),
        indent=2,
        default=str
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "version": "4.0",
    "run_date": str(date.today()),
    "requested_events": MAX_EVENTS,
    "events": int(len(output)),
    "tickers": int(output["ticker"].nunique()),
    "date_min": str(output["award_date"].min()),
    "date_max": str(output["award_date"].max()),

    "event_to_peak_mean_pct": safe_mean(
        output["event_to_peak_pct"]
    ),
    "event_to_peak_median_pct": safe_median(
        output["event_to_peak_pct"]
    ),

    "event_to_valley_mean_pct": safe_mean(
        output["event_to_valley_pct"]
    ),
    "event_to_valley_median_pct": safe_median(
        output["event_to_valley_pct"]
    ),

    "valley_to_peak_mean_pct": safe_mean(
        output["valley_to_peak_pct"]
    ),
    "valley_to_peak_median_pct": safe_median(
        output["valley_to_peak_pct"]
    ),

    "peak_day_mean": safe_mean(
        output["peak_day"]
    ),
    "peak_day_median": safe_median(
        output["peak_day"]
    ),

    "valley_day_mean": safe_mean(
        output["valley_day"]
    ),
    "valley_day_median": safe_median(
        output["valley_day"]
    ),

    "leak_rate_5pct": round(
        float(
            output["leaked_5pct"].mean()
            * 100
        ),
        3
    ),

    "hit_10_count": int(
        output["hit_10_day"].notna().sum()
    ),
    "hit_15_count": int(
        output["hit_15_day"].notna().sum()
    ),
    "hit_20_count": int(
        output["hit_20_day"].notna().sum()
    ),
    "hit_50_count": int(
        output["hit_50_day"].notna().sum()
    ),
    "hit_100_count": int(
        output["hit_100_day"].notna().sum()
    ),
}

SUMMARY_JSON.write_text(
    json.dumps(
        summary,
        indent=2
    )
)

print()
print("=" * 60)
print(" RECON EVENT STUDY V4.0 COMPLETE")
print("=" * 60)
print(f"Events       : {summary['events']}")
print(f"Tickers      : {summary['tickers']}")
print(f"Date range   : {summary['date_min']} -> {summary['date_max']}")
print(
    f"Peak mean    : "
    f"{summary['event_to_peak_mean_pct']:.2f}%"
)
print(
    f"Peak median  : "
    f"{summary['event_to_peak_median_pct']:.2f}%"
)
print(
    f"Valley mean  : "
    f"{summary['event_to_valley_mean_pct']:.2f}%"
)
print(
    f"V->P median  : "
    f"{summary['valley_to_peak_median_pct']:.2f}%"
)
print(
    f"Peak day med : "
    f"{summary['peak_day_median']:.1f}"
)
print(
    f"Leak >=5%    : "
    f"{summary['leak_rate_5pct']:.1f}%"
)
print()
print(
    f"+10%  {summary['hit_10_count']}/{summary['events']}"
)
print(
    f"+15%  {summary['hit_15_count']}/{summary['events']}"
)
print(
    f"+20%  {summary['hit_20_count']}/{summary['events']}"
)
print(
    f"+50%  {summary['hit_50_count']}/{summary['events']}"
)
print(
    f"+100% {summary['hit_100_count']}/{summary['events']}"
)
print("=" * 60)
print(f"Saved: {OUTPUT_CSV}")
print(f"Saved: {OUTPUT_JSON}")
print(f"Saved: {SUMMARY_JSON}")
