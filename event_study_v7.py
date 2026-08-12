# RECON Event Study V7.0
#
# PURPOSE
# -------
# Build a clean historical event-study data set for the RECON stock-lead
# experiment.
#
# IMPORTANT CHANGES FROM V3
# -------------------------
# 1. 365-day award lookback is retained.
# 2. USAspending is queried PER MASTER TICKER using recipient_search_text.
#    This avoids the practical 10,000-row ceiling that truncated the old
#    broad query to roughly May 2026 -> present.
# 3. Uses a 90-TRADING-DAY post-event window.
# 4. Only events with a complete 90-trading-day forward window are eligible.
#    This prevents right-censoring of recent events.
# 5. Collects candidate control factors known at/just before award time.
# 6. Calculates the ideal-function residual after the event study:
#       peak_pct = intercept + slope * peak_day + residual
#    The residual is the "noise" term we want to explain with controls.
# 7. Saves both the raw event study and a control-factor data set.
#
# The script does NOT decide which controls belong in the final signal.
# It creates the historical data needed to discover them.

import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================
# SETTINGS
# ============================================================

DAYS_BACK = 365
MIN_AWARD = 1_000_000
MAX_EVENTS = 100

PRE_TRADING_DAYS = 20
POST_TRADING_DAYS = 90
EVENT_SPACING_DAYS = 14

# A conservative calendar buffer so the last event has a full
# 90-trading-day market window available.
FORWARD_CALENDAR_BUFFER = 135

USA_API = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"

OUTPUT_JSON = Path("event_study_v7.json")
OUTPUT_CSV = Path("event_study_v7.csv")
SUMMARY_JSON = Path("event_study_v7_summary.json")
CONTROLS_CSV = Path("event_study_v7_controls.csv")
RESIDUAL_CSV = Path("event_study_v7_residuals.csv")

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

# Bootstrap revenue only.  Historical revenue is fetched separately
# where possible.  Do NOT treat the bootstrap value as point-in-time
# truth in later modeling.
MASTER_REVENUE = {
    "PLTR": 6_160_000_000,
    "RCAT": 71_540_000,
    "AVAV": 1_980_000_000,
    "WWD": 4_190_000_000,
    "AEVA": 20_970_000,
    "LMT": 75_110_000_000,
    "RTX": 90_370_000_000,
    "BAH": 11_220_000_000,
    "SAIC": 7_290_000_000,
    "LDOS": 17_330_000_000,
    "LHX": 22_930_000_000,
    "NOC": 42_370_000_000,
}

# ============================================================
# HELPERS
# ============================================================

def master_lookup(name):
    u = (name or "").upper()
    for ticker, keywords in MASTER.items():
        if any(k in u for k in keywords):
            return ticker
    return None


def as_float(v):
    try:
        if v in (None, ""):
            return None
        return float(v)
    except Exception:
        return None


def mod_zero(v):
    return v is None or str(v).strip().upper() in {"0", "0000", "BASE", ""}


def pct(a, b):
    if a in (None, 0) or b is None:
        return None
    return (b / a - 1.0) * 100.0


def safe_date(v):
    try:
        return pd.Timestamp(v).date()
    except Exception:
        return None


def first_hit(series, baseline, target):
    if baseline in (None, 0):
        return None
    level = baseline * (1.0 + target / 100.0)
    for day, price in sorted(series.items()):
        if day > 0 and price >= level:
            return int(day)
    return None


def normalize_text(v):
    if v is None:
        return ""
    return " ".join(str(v).upper().split())


def extract_name(v):
    if isinstance(v, dict):
        return v.get("name") or v.get("toptier_name") or ""
    return v or ""


# ============================================================
# USAspending DOWNLOAD
#
# The old broad query hit 10,000 transactions and therefore did NOT
# actually cover the full 365-day period.  This version filters by
# recipient_search_text for each ticker, which is an official
# USAspending transaction-search filter.
# ============================================================

END = date.today()
ELIGIBLE_END = END - timedelta(days=FORWARD_CALENDAR_BUFFER)
START = END - timedelta(days=DAYS_BACK)

session = requests.Session()
session.headers.update({
    "User-Agent": "RECON-Event-Study/6.0"
})

FIELDS = [
    "Award ID",
    "Recipient Name",
    "Action Date",
    "Transaction Amount",
    "Awarding Agency",
    "Awarding Sub Agency",
    "Award Type",
    "Contract Award Type",
    "Action Type",
    "Mod",
    "Description",
    "Period of Performance Start Date",
    "Period of Performance Current End Date",
    "Period of Performance Potential End Date",
]

def fetch_recipient(keyword):
    payload = {
        "filters": {
            "recipient_search_text": [keyword],
            "award_amounts": [{"lower_bound": MIN_AWARD}],
            "award_type_codes": ["A", "B", "C", "D"],
            "time_period": [{
                "start_date": str(START),
                "end_date": str(ELIGIBLE_END),
            }],
        },
        "fields": FIELDS,
        "sort": "Transaction Amount",
        "order": "desc",
        "limit": 100,
        "page": 1,
    }

    rows = []
    page = 1

    while True:
        payload["page"] = page

        for attempt in range(4):
            try:
                r = session.post(USA_API, json=payload, timeout=90)
                r.raise_for_status()
                data = r.json()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

        batch = data.get("results", [])
        if not batch:
            break

        rows.extend(batch)

        print(
            f"  {keyword}: page {page}, "
            f"{len(batch)} rows, total {len(rows)}"
        )

        if not data.get("page_metadata", {}).get("hasNext", False):
            break

        page += 1
        time.sleep(0.25)

    return rows


print(
    f"Downloading USAspending transactions "
    f"{START} -> {ELIGIBLE_END}"
)

all_rows = []

for ticker, keywords in MASTER.items():
    for keyword in keywords:
        try:
            rows = fetch_recipient(keyword)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"ERROR downloading {ticker}/{keyword}: {exc}")

print(f"Raw recipient-filtered transactions: {len(all_rows)}")

# ============================================================
# CLEAN / DEDUPLICATE CANDIDATE EVENTS
# ============================================================

cands = []

for row in all_rows:
    name = row.get("Recipient Name")
    ticker = master_lookup(name)

    if not ticker:
        continue

    if not mod_zero(row.get("Mod")):
        continue

    dt = safe_date(row.get("Action Date"))
    amount = as_float(row.get("Transaction Amount"))

    if dt is None or amount is None or amount < MIN_AWARD:
        continue

    agency = extract_name(row.get("Awarding Agency"))
    subagency = extract_name(row.get("Awarding Sub Agency"))

    cands.append({
        "award_id": row.get("Award ID"),
        "ticker": ticker,
        "company": name,
        "award_date": str(dt),
        "award_amount": amount,
        "agency": agency,
        "subagency": subagency,
        "award_type": row.get("Award Type"),
        "contract_award_type": row.get("Contract Award Type"),
        "action_type": row.get("Action Type"),
        "description": row.get("Description") or "",
        "pop_start": row.get("Period of Performance Start Date"),
        "pop_current_end": row.get("Period of Performance Current End Date"),
        "pop_potential_end": row.get("Period of Performance Potential End Date"),
    })

events = pd.DataFrame(cands)

if events.empty:
    raise RuntimeError("No matching public-company award events found.")

events["award_date"] = pd.to_datetime(events["award_date"], errors="coerce")
events = events.dropna(subset=["award_date"])

# One transaction per award ID.
events = events.drop_duplicates(subset=["award_id"], keep="first")

# Keep only events with a complete forward window.
events = events[events["award_date"].dt.date <= ELIGIBLE_END]

print(f"Clean candidate transactions: {len(events)}")

# ============================================================
# SELECT EVENTS
#
# Do NOT select only the largest awards. That would build the model with
# a distorted control-factor distribution and make award size look
# predictive simply because we selected on it.
#
# Instead, build a reproducible stratified sample by ticker and award-size
# quartile. Events close together are retained and explicitly measured;
# multiple nearby awards are potentially informative rather than silently
# discarded.
# ============================================================

events["award_size_q"] = events.groupby("ticker")["award_amount"].transform(
    lambda s: pd.qcut(s.rank(method="first"), 4, labels=False, duplicates="drop")
)

selected_parts = []
per_ticker = max(1, MAX_EVENTS // max(1, events["ticker"].nunique()))

for ticker, group in events.groupby("ticker"):
    group = group.sort_values(["award_size_q", "award_amount"], ascending=[True, False])
    # Evenly sample the available award-size strata.
    take = min(per_ticker, len(group))
    if len(group) <= take:
        chosen = group
    else:
        idx = np.linspace(0, len(group) - 1, take).round().astype(int)
        chosen = group.iloc[idx].drop_duplicates(subset=["award_id"])
    selected_parts.append(chosen)

events = pd.concat(selected_parts, ignore_index=True)

# Fill any remaining slots from the unused population, ordered by ticker
# and award date so the sample is deterministic.
if len(events) < MAX_EVENTS:
    used_ids = set(events["award_id"].dropna())
    remaining = events_source = globals().get("_events_before_sample")
    if remaining is None:
        # Reconstruct from the current candidate set by excluding selected IDs.
        remaining = pd.DataFrame(cands)
        remaining["award_date"] = pd.to_datetime(remaining["award_date"], errors="coerce")
    remaining = remaining[~remaining["award_id"].isin(used_ids)].copy()
    if not remaining.empty:
        events = pd.concat([events, remaining.sort_values(["ticker", "award_date"]).head(MAX_EVENTS-len(events))], ignore_index=True)

events = events.drop_duplicates(subset=["award_id"]).head(MAX_EVENTS).copy()

print(f"Selected stratified event sample: {len(events)}")

# ============================================================
# HISTORICAL REVENUE PROXY
#
# We use the latest quarterly revenue information whose PERIOD END
# is at least 45 days before the award.  This is deliberately
# conservative because yfinance does not give us a reliable SEC
# filing/publication timestamp for every historical statement.
#
# If unavailable, a MASTER fallback is recorded but flagged.
# ============================================================

revenue_cache = {}

def get_revenue_proxy(ticker, award_date):
    key = (ticker, str(award_date))
    if key in revenue_cache:
        return revenue_cache[key]

    cutoff = pd.Timestamp(award_date) - pd.Timedelta(days=45)

    try:
        stock = yf.Ticker(ticker)
        q = stock.quarterly_financials

        if q is not None and not q.empty:
            revenue_row = None
            for candidate in ["Total Revenue", "TotalRevenue"]:
                if candidate in q.index:
                    revenue_row = q.loc[candidate]
                    break

            if revenue_row is not None:
                vals = []
                for col, value in revenue_row.items():
                    try:
                        period_end = pd.Timestamp(col)
                        value = float(value)
                        if period_end <= cutoff and value > 0:
                            vals.append((period_end, value))
                    except Exception:
                        pass

                vals.sort(reverse=True)

                if vals:
                    # TTM = four most recent eligible quarters.
                    ttm = sum(v for _, v in vals[:4])
                    if ttm > 0:
                        revenue_cache[key] = (ttm, "yahoo_quarterly_proxy")
                        return revenue_cache[key]
    except Exception:
        pass

    if ticker in MASTER_REVENUE:
        revenue_cache[key] = (
            float(MASTER_REVENUE[ticker]),
            "master_fallback",
        )
        return revenue_cache[key]

    revenue_cache[key] = (None, "unavailable")
    return revenue_cache[key]


# ============================================================
# PRIOR-AWARD HISTORY
# ============================================================

history_by_ticker = {
    t: events[events["ticker"] == t].sort_values("award_date").copy()
    for t in MASTER
}

# ============================================================
# PRE-EVENT CONTRACT ACTIVITY CONTROLS
# ============================================================

# These are knowable at award time and are deliberately retained as
# candidate controls. They capture whether the award is an isolated
# event or part of an active contracting run.

for idx, e in events.iterrows():
    prior_30 = events[(events["ticker"] == e["ticker"]) &
                      (events["award_date"] < e["award_date"]) &
                      (events["award_date"] >= e["award_date"] - pd.Timedelta(days=30))]
    prior_90 = events[(events["ticker"] == e["ticker"]) &
                      (events["award_date"] < e["award_date"]) &
                      (events["award_date"] >= e["award_date"] - pd.Timedelta(days=90))]
    events.loc[idx, "prior_award_count_30d"] = len(prior_30)
    events.loc[idx, "prior_award_total_30d"] = prior_30["award_amount"].sum()
    events.loc[idx, "prior_award_count_90d"] = len(prior_90)
    events.loc[idx, "prior_award_total_90d"] = prior_90["award_amount"].sum()

# ============================================================
# MARKET DATA + EVENT MEASUREMENT
# ============================================================

records = []

# Download each ticker once.
market = {}

for ticker in sorted(events["ticker"].unique()):
    try:
        first_date = events.loc[
            events["ticker"] == ticker, "award_date"
        ].min().date()

        # 90 trading days is roughly 135 calendar days. Add generous
        # padding for weekends/holidays.
        start = first_date - timedelta(days=70)
        end = END + timedelta(days=2)

        hist = yf.Ticker(ticker).history(
            start=str(start),
            end=str(end),
            interval="1d",
            auto_adjust=True,
            actions=False,
        )

        if hist.empty:
            continue

        hist.index = pd.to_datetime(hist.index).tz_localize(None)
        hist = hist.sort_index()

        market[ticker] = hist["Close"].dropna()
        print(f"Loaded market history: {ticker} ({len(market[ticker])} days)")
    except Exception as exc:
        print(f"Market history error {ticker}: {exc}")

# SPY market benchmark.
try:
    spy_hist = yf.Ticker("SPY").history(
        start=str(START - timedelta(days=80)),
        end=str(END + timedelta(days=2)),
        interval="1d",
        auto_adjust=True,
        actions=False,
    )
    spy_hist.index = pd.to_datetime(spy_hist.index).tz_localize(None)
    spy = spy_hist["Close"].dropna()
except Exception:
    spy = pd.Series(dtype=float)


def market_controls(ticker, event_date, closes):
    event_ts = pd.Timestamp(event_date)

    before = closes[closes.index < event_ts]

    if len(before) < 61:
        return {
            "stock_pre_20d_return_pct": None,
            "stock_pre_20d_vol_pct": None,
            "stock_pre_60d_vol_pct": None,
            "stock_drawdown_60d_pct": None,
            "stock_dollar_price": None,
            "spy_pre_20d_return_pct": None,
            "relative_pre_20d_return_pct": None,
            "beta_60d": None,
        }

    p0 = float(before.iloc[-1])
    p20 = float(before.iloc[-21])
    p60 = float(before.iloc[-61])

    r20 = (p0 / p20 - 1.0) * 100.0
    returns60 = before.pct_change().dropna().tail(60)
    vol20 = before.pct_change().dropna().tail(20).std() * math.sqrt(252) * 100
    vol60 = returns60.std() * math.sqrt(252) * 100

    high60 = before.tail(60).max()
    drawdown = (p0 / high60 - 1.0) * 100.0

    spy20 = None
    rel20 = None
    beta = None

    if not spy.empty:
        sb = spy[spy.index < event_ts]
        if len(sb) >= 61:
            sp0 = float(sb.iloc[-1])
            sp20 = float(sb.iloc[-21])
            spy20 = (sp0 / sp20 - 1.0) * 100.0

            sr = sb.pct_change().dropna().tail(60)
            xr = before.pct_change().dropna().tail(60)

            aligned = pd.concat(
                [xr.rename("x"), sr.rename("m")],
                axis=1,
            ).dropna()

            if len(aligned) >= 30 and aligned["m"].var() > 0:
                beta = aligned["x"].cov(aligned["m"]) / aligned["m"].var()

            rel20 = r20 - spy20

    return {
        "stock_pre_20d_return_pct": round(r20, 4),
        "stock_pre_20d_vol_pct": round(float(vol20), 4),
        "stock_pre_60d_vol_pct": round(float(vol60), 4),
        "stock_drawdown_60d_pct": round(float(drawdown), 4),
        "stock_dollar_price": round(p0, 4),
        "spy_pre_20d_return_pct": None if spy20 is None else round(spy20, 4),
        "relative_pre_20d_return_pct": None if rel20 is None else round(rel20, 4),
        "beta_60d": None if beta is None else round(float(beta), 4),
    }


for n, (_, e) in enumerate(events.iterrows(), 1):
    ticker = e["ticker"]
    award_date = e["award_date"].date()

    print(
        f"[{n}/{len(events)}] {ticker} {award_date} "
        f"${e['award_amount']/1e6:.1f}M"
    )

    closes = market.get(ticker)
    if closes is None or closes.empty:
        print("  no market history")
        continue

    try:
        # Event trading day = first trading day on/after award date.
        event_candidates = closes[closes.index >= pd.Timestamp(award_date)]

        if event_candidates.empty:
            continue

        event_date = event_candidates.index[0]
        baseline = float(event_candidates.iloc[0])

        before = closes[closes.index < event_date]

        if len(before) < PRE_TRADING_DAYS:
            continue

        all_dates = list(closes.index)
        event_pos = all_dates.index(event_date)

        lo = max(0, event_pos - PRE_TRADING_DAYS)
        hi = min(
            len(all_dates) - 1,
            event_pos + POST_TRADING_DAYS,
        )

        offsets = {
            i - event_pos: float(closes.iloc[i])
            for i in range(lo, hi + 1)
        }

        post = {
            k: v
            for k, v in offsets.items()
            if 0 <= k <= POST_TRADING_DAYS
        }

        # Require a complete 90-trading-day forward window.
        if max(post.keys()) < POST_TRADING_DAYS:
            print("  incomplete 90-day window")
            continue

        peak_day, peak_price = max(
            post.items(),
            key=lambda kv: kv[1],
        )

        prepeak = {
            k: v
            for k, v in post.items()
            if k <= peak_day
        }

        valley_day, valley_price = min(
            prepeak.items(),
            key=lambda kv: kv[1],
        )

        pre = {
            k: v
            for k, v in offsets.items()
            if -10 <= k <= -1
        }

        pre10 = pre.get(-10)
        pre1 = pre.get(-1)

        pre_move = pct(pre10, pre1)
        drawdown = pct(baseline, valley_price)
        vtp = pct(valley_price, peak_price)
        etp = pct(baseline, peak_price)

        # Prior award controls — strictly before this event.
        prior = events[
            (events["ticker"] == ticker)
            & (events["award_date"] < e["award_date"])
            & (
                events["award_date"]
                >= e["award_date"] - pd.Timedelta(days=365)
            )
        ].sort_values("award_date")

        prior_amounts = prior["award_amount"].astype(float).tolist()

        days_since_prior = None
        if not prior.empty:
            days_since_prior = (
                e["award_date"] - prior["award_date"].iloc[-1]
            ).days

        revenue, revenue_source = get_revenue_proxy(
            ticker,
            e["award_date"],
        )

        ratio = None
        if revenue and revenue > 0:
            ratio = e["award_amount"] / revenue * 100.0

        controls = market_controls(
            ticker,
            award_date,
            closes,
        )

        pop_start = safe_date(e["pop_start"])
        pop_current_end = safe_date(e["pop_current_end"])
        pop_potential_end = safe_date(e["pop_potential_end"])

        pop_days = None
        if pop_start and pop_current_end:
            pop_days = (pop_current_end - pop_start).days

        rec = {
            # Event identity
            "award_id": e["award_id"],
            "ticker": ticker,
            "company": e["company"],
            "award_date": str(award_date),
            "event_trading_date": str(event_date.date()),

            # Award information known at event
            "award_amount": round(float(e["award_amount"]), 2),
            "award_m": round(float(e["award_amount"]) / 1e6, 3),
            "agency": e["agency"],
            "subagency": e["subagency"],
            "award_type": e["award_type"],
            "contract_award_type": e["contract_award_type"],
            "action_type": e["action_type"],
            "description": e["description"],

            # Contract duration information
            "pop_start": None if pop_start is None else str(pop_start),
            "pop_current_end": None if pop_current_end is None else str(pop_current_end),
            "pop_potential_end": None if pop_potential_end is None else str(pop_potential_end),
            "pop_days": pop_days,

            # Revenue control
            "revenue_ttm_proxy": revenue,
            "revenue_source": revenue_source,
            "award_revenue_ratio_pct": None if ratio is None else round(ratio, 4),

            # Prior-award controls
            "prior_award_count_30d": int(e.get("prior_award_count_30d", 0)),
            "prior_award_total_30d": float(e.get("prior_award_total_30d", 0)),
            "prior_award_count_90d": int(e.get("prior_award_count_90d", 0)),
            "prior_award_total_90d": float(e.get("prior_award_total_90d", 0)),
            "prior_award_count_365d": len(prior_amounts),
            "prior_award_total_365d": round(sum(prior_amounts), 2),
            "prior_award_mean_365d": (
                None if not prior_amounts
                else round(float(np.mean(prior_amounts)), 2)
            ),
            "prior_award_median_365d": (
                None if not prior_amounts
                else round(float(np.median(prior_amounts)), 2)
            ),
            "days_since_prior_award": days_since_prior,
            "award_vs_prior_mean_ratio": (
                None
                if not prior_amounts or np.mean(prior_amounts) <= 0
                else round(
                    float(e["award_amount"] / np.mean(prior_amounts)),
                    4,
                )
            ),

            # Market controls known before/at award
            **controls,

            # Event response
            "event_price": round(baseline, 4),
            "pre_abnormal_10_to_1_pct": (
                None if pre_move is None else round(pre_move, 4)
            ),
            "event_to_peak_pct": (
                None if etp is None else round(etp, 4)
            ),
            "peak_day": int(peak_day),
            "peak_price": round(peak_price, 4),
            "event_to_valley_pct": (
                None if drawdown is None else round(drawdown, 4)
            ),
            "valley_day": int(valley_day),
            "valley_price": round(valley_price, 4),
            "valley_to_peak_pct": (
                None if vtp is None else round(vtp, 4)
            ),
            "valley_to_peak_days": int(peak_day - valley_day),

            # Threshold timing
            "hit_10_day": first_hit(post, baseline, 10),
            "hit_15_day": first_hit(post, baseline, 15),
            "hit_20_day": first_hit(post, baseline, 20),
            "hit_50_day": first_hit(post, baseline, 50),
            "hit_100_day": first_hit(post, baseline, 100),

            "positive_1d": post.get(1, baseline) > baseline,
            "positive_5d": post.get(5, baseline) > baseline,
            "positive_10d": post.get(10, baseline) > baseline,
            "positive_30d": post.get(30, baseline) > baseline,

            "leaked_5pct": (
                pre_move is not None and pre_move >= 5
            ),
        }

        records.append(rec)

        print(
            f"  peak {etp:+.1f}% day {peak_day}; "
            f"valley {drawdown:+.1f}% day {valley_day}; "
            f"V->P {vtp:+.1f}%"
        )

    except Exception as exc:
        print(" ERROR:", exc)

    time.sleep(0.25)

if not records:
    raise RuntimeError(
        "No usable historical events were produced."
    )

# ============================================================
# IDEAL FUNCTION + RESIDUAL
# ============================================================

out = pd.DataFrame(records)

out["event_to_peak_pct"] = pd.to_numeric(
    out["event_to_peak_pct"],
    errors="coerce",
)

out["peak_day"] = pd.to_numeric(
    out["peak_day"],
    errors="coerce",
)

fit = out.dropna(
    subset=["event_to_peak_pct", "peak_day"]
).copy()

if len(fit) >= 3:
    slope, intercept = np.polyfit(
        fit["peak_day"],
        fit["event_to_peak_pct"],
        1,
    )

    out["ideal_peak_pct"] = (
        intercept + slope * out["peak_day"]
    )

    out["peak_residual_pct"] = (
        out["event_to_peak_pct"]
        - out["ideal_peak_pct"]
    )

    out["absolute_peak_residual_pct"] = (
        out["peak_residual_pct"].abs()
    )

    # R^2
    pred = intercept + slope * fit["peak_day"]
    ss_res = float(
        ((fit["event_to_peak_pct"] - pred) ** 2).sum()
    )
    ss_tot = float(
        (
            (
                fit["event_to_peak_pct"]
                - fit["event_to_peak_pct"].mean()
            ) ** 2
        ).sum()
    )
    r2 = None if ss_tot == 0 else 1.0 - ss_res / ss_tot

else:
    slope = None
    intercept = None
    r2 = None
    out["ideal_peak_pct"] = None
    out["peak_residual_pct"] = None
    out["absolute_peak_residual_pct"] = None

out = out.sort_values(
    ["award_date", "ticker"]
).reset_index(drop=True)

# ============================================================
# SAVE OUTPUTS
# ============================================================

out.to_csv(OUTPUT_CSV, index=False)

OUTPUT_JSON.write_text(
    json.dumps(
        out.to_dict("records"),
        indent=2,
        default=str,
    )
)

control_cols = [
    "award_id",
    "ticker",
    "award_date",
    "award_m",
    "award_revenue_ratio_pct",
    "prior_award_count_30d",
    "prior_award_total_30d",
    "prior_award_count_90d",
    "prior_award_total_90d",
    "prior_award_count_365d",
    "prior_award_total_365d",
    "prior_award_mean_365d",
    "prior_award_median_365d",
    "days_since_prior_award",
    "award_vs_prior_mean_ratio",
    "pop_days",
    "agency",
    "subagency",
    "award_type",
    "contract_award_type",
    "action_type",
    "stock_pre_20d_return_pct",
    "stock_pre_20d_vol_pct",
    "stock_pre_60d_vol_pct",
    "stock_drawdown_60d_pct",
    "spy_pre_20d_return_pct",
    "relative_pre_20d_return_pct",
    "beta_60d",
    "event_to_peak_pct",
    "peak_day",
    "ideal_peak_pct",
    "peak_residual_pct",
    "absolute_peak_residual_pct",
]

control_cols = [
    c for c in control_cols
    if c in out.columns
]

out[control_cols].to_csv(
    CONTROLS_CSV,
    index=False,
)

out[
    [
        c for c in [
            "ticker",
            "award_date",
            "award_amount",
            "award_revenue_ratio_pct",
            "prior_award_count_365d",
            "days_since_prior_award",
            "peak_day",
            "event_to_peak_pct",
            "ideal_peak_pct",
            "peak_residual_pct",
        ]
        if c in out.columns
    ]
].to_csv(
    RESIDUAL_CSV,
    index=False,
)

summary = {
    "version": "7.0",
    "date_range_requested": {
        "start": str(START),
        "end": str(ELIGIBLE_END),
    },
    "forward_window_trading_days": POST_TRADING_DAYS,
    "transactions_downloaded": len(all_rows),
    "candidate_transactions": int(len(events)),
    "usable_events": int(len(out)),
    "tickers": sorted(out["ticker"].dropna().unique().tolist()),
    "ideal_function": {
        "form": "peak_pct = intercept + slope * peak_day",
        "intercept": None if intercept is None else round(float(intercept), 6),
        "slope_pct_per_trading_day": None if slope is None else round(float(slope), 6),
        "r_squared": None if r2 is None else round(float(r2), 6),
    },
    "average_event_to_peak_pct": round(
        float(out["event_to_peak_pct"].mean()), 4
    ),
    "median_event_to_peak_pct": round(
        float(out["event_to_peak_pct"].median()), 4
    ),
    "average_peak_day": round(
        float(out["peak_day"].mean()), 4
    ),
    "median_peak_day": round(
        float(out["peak_day"].median()), 4
    ),
    "average_valley_to_peak_pct": round(
        float(out["valley_to_peak_pct"].mean()), 4
    ),
    "median_valley_to_peak_pct": round(
        float(out["valley_to_peak_pct"].median()), 4
    ),
    "peak_10pct_rate": round(
        float(out["hit_10_day"].notna().mean() * 100), 4
    ),
    "peak_20pct_rate": round(
        float(out["hit_20_day"].notna().mean() * 100), 4
    ),
    "peak_50pct_rate": round(
        float(out["hit_50_day"].notna().mean() * 100), 4
    ),
    "peak_100pct_rate": round(
        float(out["hit_100_day"].notna().mean() * 100), 4
    ),
}

SUMMARY_JSON.write_text(
    json.dumps(summary, indent=2)
)

print()
print("=" * 60)
print("RECON EVENT STUDY V7.0 COMPLETE")
print("=" * 60)
print(f"Requested award range : {START} -> {ELIGIBLE_END}")
print(f"Raw transactions      : {len(all_rows)}")
print(f"Selected candidates   : {len(events)}")
print(f"Usable events         : {len(out)}")
print(f"Tickers               : {out.ticker.nunique()}")
print(f"Forward window        : {POST_TRADING_DAYS} trading days")
print()
print("IDEAL FUNCTION")
print(f"peak_pct = {intercept:.6f} + {slope:.6f} * peak_day")
print(f"R^2 = {r2:.6f}")
print()
print("Saved:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(SUMMARY_JSON)
print(CONTROLS_CSV)
print(RESIDUAL_CSV)
print("=" * 60)
