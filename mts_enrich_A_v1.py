# ============================================================
# RECON MTS AWARD-TIME ENRICHMENT A V1.0
# ============================================================
#
# PURPOSE
# -------
# Enrich the CORRECTED unique Sample-A market events with
# information that was knowable at the time of each award.
#
# IMPORTANT
# ---------
# - NO future trajectory information is used to construct predictors.
# - Uses the SAVED award population locally. It does NOT call
#   USAspending again.
# - Downloads market history once per ticker (plus SPY and ITA)
#   and caches it so reruns do not repeatedly hit Yahoo.
# - Keeps MD4 / peak outcome ONLY as target columns for later MTS.
#
# INPUTS
# ------
# trajectory_unique_ticker_day_A.csv
# master_zero_purged.csv
#
# OUTPUTS
# -------
# mts_award_time_enriched_A.csv
# mts_award_time_features_A.csv
# mts_enrichment_summary_A.json
# market_cache/*.csv
#
# ============================================================

import json
import math
import time
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# SETTINGS
# ============================================================

EVENT_CANDIDATES = [
    Path("trajectory_unique_ticker_day_A.csv"),
    Path("Recon/trajectory_unique_ticker_day_A.csv"),
]

MASTER_CANDIDATES = [
    Path("master_zero_purged.csv"),
    Path("Recon/master_zero_purged.csv"),
]

OUT_ENRICHED = Path("mts_award_time_enriched_A.csv")
OUT_FEATURES = Path("mts_award_time_features_A.csv")
OUT_SUMMARY = Path("mts_enrichment_summary_A.json")

CACHE_DIR = Path("market_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BENCHMARK_MARKET = "SPY"
BENCHMARK_SECTOR = "ITA"

# Market history needed before earliest award to calculate 52-week state.
MARKET_LOOKBACK_CALENDAR_DAYS = 430
MARKET_FORWARD_CALENDAR_DAYS = 100

# Prior-award response windows. A prior response is only used if the
# full response window was already observable BEFORE the current award.
PRIOR_RESPONSE_DAYS = [5, 20, 60]

REQUEST_PAUSE = 0.25

# ============================================================
# HELPERS
# ============================================================

def find_existing(candidates, label):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{label} not found. Expected one of: " +
        ", ".join(str(x) for x in candidates)
    )

def safe_div(a, b):
    try:
        if b is None or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def pct_rank(values, x):
    vals = pd.Series(values).dropna().astype(float)
    if len(vals) == 0 or pd.isna(x):
        return np.nan
    return float((vals <= float(x)).mean())

def business_day_distance(a, b):
    # Approximate trading-day distance for metadata only.
    try:
        return int(np.busday_count(
            np.datetime64(pd.Timestamp(a).date()),
            np.datetime64(pd.Timestamp(b).date())
        ))
    except Exception:
        return np.nan

def normalize_yf_frame(data, ticker):
    """
    Convert yfinance output into columns Date, Close, Volume.
    Handles both simple and MultiIndex responses.
    """
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["Date", "Close", "Volume"])

    d = data.copy()

    if isinstance(d.columns, pd.MultiIndex):
        # yfinance may return Price x Ticker or Ticker x Price.
        close = None
        volume = None

        # Try direct first-level price names.
        if "Close" in d.columns.get_level_values(0):
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        elif "Close" in d.columns.get_level_values(-1):
            close = d.xs("Close", axis=1, level=-1)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

        if "Volume" in d.columns.get_level_values(0):
            volume = d["Volume"]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
        elif "Volume" in d.columns.get_level_values(-1):
            volume = d.xs("Volume", axis=1, level=-1)
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]

        out = pd.DataFrame({
            "Close": close,
            "Volume": volume if volume is not None else np.nan
        })
    else:
        out = pd.DataFrame({
            "Close": d["Close"] if "Close" in d.columns else np.nan,
            "Volume": d["Volume"] if "Volume" in d.columns else np.nan
        })

    out = out.dropna(subset=["Close"]).copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])
    return out[["Date", "Close", "Volume"]]

def load_or_download_market(ticker, start_date, end_date):
    cache = CACHE_DIR / f"{ticker}.csv"

    if cache.exists():
        try:
            c = pd.read_csv(cache, parse_dates=["Date"])
            if (
                not c.empty and
                c["Date"].min() <= pd.Timestamp(start_date) and
                c["Date"].max() >= pd.Timestamp(end_date) - pd.Timedelta(days=7)
            ):
                return c.sort_values("Date").reset_index(drop=True)
        except Exception:
            pass

    print(f"Downloading market history: {ticker}")

    data = None
    last_error = None

    for attempt in range(3):
        try:
            data = yf.download(
                ticker,
                start=str(pd.Timestamp(start_date).date()),
                end=str((pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if data is not None and not data.empty:
                break
        except Exception as ex:
            last_error = ex

        time.sleep(2.0 * (attempt + 1))

    out = normalize_yf_frame(data, ticker)

    if out.empty:
        print(f"  WARNING: no market data for {ticker}: {last_error}")
        return out

    out.to_csv(cache, index=False)
    time.sleep(REQUEST_PAUSE)
    return out

def market_index_on_or_before(market, date):
    if market is None or market.empty:
        return None
    dates = market["Date"].values
    pos = np.searchsorted(dates, np.datetime64(pd.Timestamp(date)), side="right") - 1
    if pos < 0:
        return None
    return int(pos)

def return_over_sessions(market, event_date, sessions):
    idx = market_index_on_or_before(market, event_date)
    if idx is None or idx - sessions < 0:
        return np.nan
    p0 = float(market.iloc[idx]["Close"])
    p1 = float(market.iloc[idx - sessions]["Close"])
    return (p0 / p1 - 1.0) * 100.0 if p1 else np.nan

def forward_return_sessions(market, event_date, sessions):
    """
    Return from first trading session on/after event_date to +sessions.
    """
    if market is None or market.empty:
        return np.nan

    dates = market["Date"].values
    idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(event_date)), side="left")
    if idx >= len(market) or idx + sessions >= len(market):
        return np.nan

    p0 = float(market.iloc[idx]["Close"])
    p1 = float(market.iloc[idx + sessions]["Close"])

    return (p1 / p0 - 1.0) * 100.0 if p0 else np.nan

def max_forward_gain_sessions(market, event_date, sessions):
    if market is None or market.empty:
        return np.nan

    dates = market["Date"].values
    idx = np.searchsorted(dates, np.datetime64(pd.Timestamp(event_date)), side="left")
    if idx >= len(market) or idx + sessions >= len(market):
        return np.nan

    base = float(market.iloc[idx]["Close"])
    if not base:
        return np.nan

    future = market.iloc[idx:idx + sessions + 1]["Close"].astype(float)
    return float((future.max() / base - 1.0) * 100.0)

def trailing_volatility(market, event_date, sessions):
    idx = market_index_on_or_before(market, event_date)
    if idx is None or idx - sessions < 1:
        return np.nan

    close = market.iloc[idx - sessions:idx + 1]["Close"].astype(float)
    ret = close.pct_change().dropna()
    if len(ret) < max(5, sessions // 2):
        return np.nan

    # daily percent standard deviation; no annualization so units stay intuitive
    return float(ret.std(ddof=1) * 100.0)

def distance_from_52w_high(market, event_date):
    idx = market_index_on_or_before(market, event_date)
    if idx is None:
        return np.nan

    start = max(0, idx - 252)
    window = market.iloc[start:idx + 1]["Close"].astype(float)
    if len(window) < 60:
        return np.nan

    current = float(window.iloc[-1])
    high = float(window.max())
    return (current / high - 1.0) * 100.0 if high else np.nan

def volume_ratio(market, event_date):
    idx = market_index_on_or_before(market, event_date)
    if idx is None or idx < 60:
        return np.nan

    v20 = pd.to_numeric(
        market.iloc[idx - 19:idx + 1]["Volume"], errors="coerce"
    ).replace(0, np.nan).mean()

    v60 = pd.to_numeric(
        market.iloc[idx - 59:idx + 1]["Volume"], errors="coerce"
    ).replace(0, np.nan).mean()

    return safe_div(v20, v60)

# ============================================================
# LOAD INPUTS
# ============================================================

event_path = find_existing(EVENT_CANDIDATES, "Unique Sample-A trajectory file")
master_path = find_existing(MASTER_CANDIDATES, "Zero-purged master award file")

events = pd.read_csv(event_path)
master = pd.read_csv(master_path)

events["award_date"] = pd.to_datetime(events["award_date"])
master["award_date"] = pd.to_datetime(master["award_date"])

master["transaction_amount"] = pd.to_numeric(
    master["transaction_amount"], errors="coerce"
)

# Safety: zero-dollar records stay excluded.
master = master[
    master["transaction_amount"].notna() &
    (master["transaction_amount"] != 0)
].copy()

master = master.sort_values(["ticker", "award_date"]).reset_index(drop=True)

print("=" * 78)
print("RECON MTS AWARD-TIME ENRICHMENT A V1.0")
print("=" * 78)
print(f"Unique Sample-A events : {len(events):,}")
print(f"Saved award history    : {len(master):,}")
print(f"Award history coverage : {master['award_date'].min().date()} -> {master['award_date'].max().date()}")
print("USAspending calls      : NONE")
print("=" * 78)

# ============================================================
# CURRENT EVENT AGENCY / TYPE METADATA
# ============================================================

same_day_meta = (
    master.groupby(["ticker", "award_date"])
    .agg(
        current_agency=("agency", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None),
        current_subagency=("subagency", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None),
        current_award_type=("award_type", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None),
        current_agency_count=("agency", lambda s: s.dropna().nunique()),
        current_award_type_count=("award_type", lambda s: s.dropna().nunique()),
    )
    .reset_index()
)

events = events.merge(
    same_day_meta,
    on=["ticker", "award_date"],
    how="left"
)

# ============================================================
# MARKET DATA CACHE
# ============================================================

earliest = events["award_date"].min() - pd.Timedelta(days=MARKET_LOOKBACK_CALENDAR_DAYS)
latest = events["award_date"].max() + pd.Timedelta(days=MARKET_FORWARD_CALENDAR_DAYS)

tickers = sorted(events["ticker"].dropna().astype(str).unique().tolist())
market_symbols = tickers + [BENCHMARK_MARKET, BENCHMARK_SECTOR]

market_data = {}

for symbol in market_symbols:
    market_data[symbol] = load_or_download_market(
        symbol,
        earliest,
        latest,
    )

# ============================================================
# BUILD PRIOR-AWARD EVENT TABLE
# ============================================================

# Aggregate all saved transactions to one ticker/day historical award event.
history_events = (
    master.groupby(["ticker", "award_date"])
    .agg(
        award_count=("award_id", "count"),
        signed_amount=("transaction_amount", "sum"),
        abs_amount=("transaction_amount", lambda s: np.abs(s.astype(float)).sum()),
        positive_amount=("transaction_amount", lambda s: s[s > 0].sum()),
        negative_amount=("transaction_amount", lambda s: s[s < 0].sum()),
        agency=("agency", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None),
        subagency=("subagency", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None),
        award_type=("award_type", lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None),
    )
    .reset_index()
    .sort_values(["ticker", "award_date"])
)

# Attach historical market response to each historical award event.
response_rows = []

for ticker, g in history_events.groupby("ticker"):
    market = market_data.get(str(ticker))

    for _, r in g.iterrows():
        rec = {
            "ticker": ticker,
            "award_date": r["award_date"],
        }

        for n in PRIOR_RESPONSE_DAYS:
            rec[f"prior_event_return_{n}d"] = forward_return_sessions(
                market, r["award_date"], n
            )
            rec[f"prior_event_max_gain_{n}d"] = max_forward_gain_sessions(
                market, r["award_date"], n
            )

        response_rows.append(rec)

history_response = pd.DataFrame(response_rows)

history_events = history_events.merge(
    history_response,
    on=["ticker", "award_date"],
    how="left"
)

# ============================================================
# ENRICH EACH SAMPLE-A EVENT
# ============================================================

master_start = master["award_date"].min()
records = []

for i, ev in events.iterrows():
    ticker = str(ev["ticker"])
    d = pd.Timestamp(ev["award_date"])

    prior = history_events[
        (history_events["ticker"] == ticker) &
        (history_events["award_date"] < d)
    ].copy()

    current_abs = float(ev.get("transaction_amount_abs_sum", np.nan))
    current_signed = float(ev.get("transaction_amount_sum", np.nan))

    feat = ev.to_dict()

    # --------------------------------------------------------
    # HISTORY COVERAGE
    # --------------------------------------------------------

    coverage_days = max(0, int((d - master_start).days))
    feat["history_days_available"] = coverage_days
    feat["history_coverage_30"] = int(coverage_days >= 30)
    feat["history_coverage_90"] = int(coverage_days >= 90)
    feat["history_coverage_180"] = int(coverage_days >= 180)

    # --------------------------------------------------------
    # GENERAL PRIOR-AWARD HISTORY
    # --------------------------------------------------------

    feat["prior_award_days_total"] = int(len(prior))
    feat["prior_transaction_count_total"] = int(prior["award_count"].sum()) if len(prior) else 0

    if len(prior):
        feat["days_since_last_award"] = int((d - prior["award_date"].max()).days)
        feat["prior_abs_award_mean"] = float(prior["abs_amount"].mean())
        feat["prior_abs_award_median"] = float(prior["abs_amount"].median())
        feat["prior_abs_award_max"] = float(prior["abs_amount"].max())
        feat["prior_signed_award_mean"] = float(prior["signed_amount"].mean())

        feat["current_to_prior_abs_mean"] = safe_div(
            current_abs, feat["prior_abs_award_mean"]
        )
        feat["current_to_prior_abs_median"] = safe_div(
            current_abs, feat["prior_abs_award_median"]
        )
        feat["current_award_abs_percentile"] = pct_rank(
            prior["abs_amount"], current_abs
        )
    else:
        feat["days_since_last_award"] = np.nan
        feat["prior_abs_award_mean"] = np.nan
        feat["prior_abs_award_median"] = np.nan
        feat["prior_abs_award_max"] = np.nan
        feat["prior_signed_award_mean"] = np.nan
        feat["current_to_prior_abs_mean"] = np.nan
        feat["current_to_prior_abs_median"] = np.nan
        feat["current_award_abs_percentile"] = np.nan

    # --------------------------------------------------------
    # ROLLING AWARD WINDOWS
    # --------------------------------------------------------

    for window in [30, 90, 180]:
        start = d - pd.Timedelta(days=window)
        w = prior[prior["award_date"] >= start]

        feat[f"prior_award_days_{window}d"] = int(len(w))
        feat[f"prior_transactions_{window}d"] = int(w["award_count"].sum()) if len(w) else 0
        feat[f"prior_abs_dollars_{window}d"] = float(w["abs_amount"].sum()) if len(w) else 0.0
        feat[f"prior_signed_dollars_{window}d"] = float(w["signed_amount"].sum()) if len(w) else 0.0
        feat[f"prior_positive_dollars_{window}d"] = float(w["positive_amount"].sum()) if len(w) else 0.0
        feat[f"prior_negative_dollars_{window}d"] = float(w["negative_amount"].sum()) if len(w) else 0.0

    # Award cadence / acceleration.
    # Compare most recent 30 days against the preceding 30 days.
    w_recent = prior[prior["award_date"] >= d - pd.Timedelta(days=30)]
    w_prev = prior[
        (prior["award_date"] < d - pd.Timedelta(days=30)) &
        (prior["award_date"] >= d - pd.Timedelta(days=60))
    ]

    feat["award_frequency_acceleration_30d"] = (
        (len(w_recent) + 1.0) / (len(w_prev) + 1.0)
    )

    feat["award_dollar_acceleration_30d"] = safe_div(
        float(w_recent["abs_amount"].sum()) + 1.0,
        float(w_prev["abs_amount"].sum()) + 1.0
    )

    # --------------------------------------------------------
    # SAME-AGENCY HISTORY
    # --------------------------------------------------------

    agency = ev.get("current_agency")

    if pd.notna(agency) and len(prior):
        pa = prior[prior["agency"] == agency]
        feat["prior_same_agency_award_days"] = int(len(pa))
        feat["prior_same_agency_abs_dollars"] = float(pa["abs_amount"].sum()) if len(pa) else 0.0
        feat["same_agency_fraction"] = safe_div(len(pa), len(prior))
        feat["agency_novelty"] = int(len(pa) == 0)
    else:
        feat["prior_same_agency_award_days"] = 0
        feat["prior_same_agency_abs_dollars"] = 0.0
        feat["same_agency_fraction"] = np.nan
        feat["agency_novelty"] = np.nan

    # --------------------------------------------------------
    # MARKET-DARLING STATE AT AWARD TIME
    # --------------------------------------------------------

    market = market_data.get(ticker)
    spy = market_data.get(BENCHMARK_MARKET)
    ita = market_data.get(BENCHMARK_SECTOR)

    for n in [5, 20, 60, 120]:
        r = return_over_sessions(market, d, n)
        r_spy = return_over_sessions(spy, d, n)
        r_ita = return_over_sessions(ita, d, n)

        feat[f"pre_return_{n}d"] = r
        feat[f"relative_strength_spy_{n}d"] = (
            r - r_spy if pd.notna(r) and pd.notna(r_spy) else np.nan
        )
        feat[f"relative_strength_ita_{n}d"] = (
            r - r_ita if pd.notna(r) and pd.notna(r_ita) else np.nan
        )

    feat["pre_volatility_market_20d"] = trailing_volatility(market, d, 20)
    feat["pre_volatility_market_60d"] = trailing_volatility(market, d, 60)
    feat["distance_from_52w_high_pct"] = distance_from_52w_high(market, d)
    feat["volume_20d_to_60d_ratio"] = volume_ratio(market, d)

    # --------------------------------------------------------
    # PRIOR STOCK RESPONSE TO AWARDS
    #
    # No leakage:
    # for an N-day prior response, only include prior awards whose
    # N-trading-day outcome would have been fully known before d.
    # --------------------------------------------------------

    for n in PRIOR_RESPONSE_DAYS:
        # 1.6 calendar days per trading day + small buffer.
        min_calendar_age = int(math.ceil(n * 1.6)) + 3

        matured = prior[
            prior["award_date"] <= d - pd.Timedelta(days=min_calendar_age)
        ].copy()

        ret_col = f"prior_event_return_{n}d"
        gain_col = f"prior_event_max_gain_{n}d"

        matured_ret = pd.to_numeric(
            matured[ret_col], errors="coerce"
        ).dropna()

        matured_gain = pd.to_numeric(
            matured[gain_col], errors="coerce"
        ).dropna()

        feat[f"prior_response_count_{n}d"] = int(len(matured_ret))

        feat[f"prior_response_mean_{n}d"] = (
            float(matured_ret.mean()) if len(matured_ret) else np.nan
        )

        feat[f"prior_response_median_{n}d"] = (
            float(matured_ret.median()) if len(matured_ret) else np.nan
        )

        feat[f"prior_max_gain_mean_{n}d"] = (
            float(matured_gain.mean()) if len(matured_gain) else np.nan
        )

        feat[f"prior_hit_rate_10pct_within_{n}d"] = (
            float((matured_gain >= 10.0).mean())
            if len(matured_gain) else np.nan
        )

        feat[f"prior_hit_rate_20pct_within_{n}d"] = (
            float((matured_gain >= 20.0).mean())
            if len(matured_gain) else np.nan
        )

    records.append(feat)

    if (i + 1) % 100 == 0 or (i + 1) == len(events):
        print(f"Enriched {i+1:,}/{len(events):,}")

# ============================================================
# SAVE
# ============================================================

enriched = pd.DataFrame(records)

# Full file retains targets and identifiers for model development.
enriched.to_csv(OUT_ENRICHED, index=False)

# Predictor-only handoff file: keeps award-time intelligence and target
# columns, but drops the 91 post-award trajectory columns and PCs other
# than MD4 / peak outcome labels.
drop_future = (
    [f"r_p{i:02d}" for i in range(91)] +
    [f"PC{i}" for i in range(1, 9)] +
    ["MD5", "MD6", "MD7", "MD8",
     "peak_day", "valley_pct", "valley_day",
     "recovery_days", "valley_to_peak_expansion_pct",
     "area_total_pct_days"]
)

feature_df = enriched.drop(
    columns=[c for c in drop_future if c in enriched.columns],
    errors="ignore"
)

feature_df.to_csv(OUT_FEATURES, index=False)

summary = {
    "version": "RECON MTS Award-Time Enrichment A V1.0",
    "sample_A_unique_events": int(len(events)),
    "enriched_events": int(len(enriched)),
    "saved_award_history_rows": int(len(master)),
    "saved_award_history_start": str(master["award_date"].min().date()),
    "saved_award_history_end": str(master["award_date"].max().date()),
    "usa_spending_calls": 0,
    "market_tickers_downloaded_or_cached": int(len(market_symbols)),
    "benchmark_market": BENCHMARK_MARKET,
    "benchmark_sector": BENCHMARK_SECTOR,
    "predictor_columns": int(len(feature_df.columns)),
    "events_with_30d_award_history": int(enriched["history_coverage_30"].sum()),
    "events_with_90d_award_history": int(enriched["history_coverage_90"].sum()),
    "events_with_180d_award_history": int(enriched["history_coverage_180"].sum()),
    "notes": [
        "All predictor variables use only information available on or before the award date.",
        "Historical award factors use the saved master_zero_purged.csv and do not query USAspending.",
        "Early Sample-A events have less historical award coverage; coverage flags are included.",
        "Prior award stock-response factors use only sufficiently old prior awards to avoid look-ahead leakage."
    ]
}

OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

# Recon copies
recon = Path("Recon")
if recon.exists():
    enriched.to_csv(recon / OUT_ENRICHED.name, index=False)
    feature_df.to_csv(recon / OUT_FEATURES.name, index=False)
    (recon / OUT_SUMMARY.name).write_text(json.dumps(summary, indent=2))

print()
print("=" * 78)
print("MTS AWARD-TIME ENRICHMENT COMPLETE")
print("=" * 78)
print(json.dumps(summary, indent=2))
print()
print("FILES CREATED:")
print(f"  {OUT_ENRICHED}")
print(f"  {OUT_FEATURES}")
print(f"  {OUT_SUMMARY}")
print(f"  {CACHE_DIR}/")
print("=" * 78)
