# ============================================================
# RECON SAMPLE-B BLIND VALIDATION, NON-LMT V1.0
# ============================================================
#
# PURPOSE
# -------
# Blindly validate the frozen Sample-A MTS model on Sample B
# after:
#   1) removing LMT,
#   2) collapsing duplicate ticker/date transactions,
#   3) removing ticker/date events that overlap Sample A,
#   4) keeping only events with a complete 90-trading-day outcome.
#
# NO SAMPLE-B OUTCOME IS USED TO FIT OR TUNE THE MODEL.
#
# INPUTS
# ------
# sample_B_nonLMT_raw.csv
# mts_award_time_features_A.csv
# master_zero_purged.csv
# mts_frozen_model_A.json
#
# OUTPUTS
# -------
# sample_B_nonLMT_validation_events.csv
# sample_B_nonLMT_contingency.csv
# sample_B_nonLMT_validation_summary.json
#
# ============================================================

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import chi2_contingency, fisher_exact
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------
# FILES
# ------------------------------------------------------------

B_FILE = Path("sample_B_nonLMT_raw.csv")
A_FILE = Path("mts_award_time_features_A.csv")
MASTER_FILE = Path("master_zero_purged.csv")
MODEL_FILE = Path("mts_frozen_model_A.json")

OUT_EVENTS = Path("sample_B_nonLMT_validation_events.csv")
OUT_TABLE = Path("sample_B_nonLMT_contingency.csv")
OUT_SUMMARY = Path("sample_B_nonLMT_validation_summary.json")

CACHE_DIR = Path("market_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PAUSE = 0.20


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def normalize_yf(data):
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["Date", "Close", "Volume"])

    d = data.copy()

    if isinstance(d.columns, pd.MultiIndex):
        if "Close" in d.columns.get_level_values(0):
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = d.xs("Close", axis=1, level=-1)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

        if "Volume" in d.columns.get_level_values(0):
            vol = d["Volume"]
            if isinstance(vol, pd.DataFrame):
                vol = vol.iloc[:, 0]
        else:
            try:
                vol = d.xs("Volume", axis=1, level=-1)
                if isinstance(vol, pd.DataFrame):
                    vol = vol.iloc[:, 0]
            except Exception:
                vol = np.nan

        out = pd.DataFrame({"Close": close, "Volume": vol})
    else:
        out = pd.DataFrame({
            "Close": d["Close"] if "Close" in d.columns else np.nan,
            "Volume": d["Volume"] if "Volume" in d.columns else np.nan,
        })

    out = out.dropna(subset=["Close"]).copy()
    out.index = pd.to_datetime(out.index)

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])

    return out[["Date", "Close", "Volume"]]


def load_market(ticker, start_date, end_date):
    cache = CACHE_DIR / f"{ticker}.csv"

    cached = None
    if cache.exists():
        try:
            cached = pd.read_csv(cache, parse_dates=["Date"])
        except Exception:
            cached = None

    need_download = True

    if cached is not None and not cached.empty:
        if (
            cached["Date"].min() <= pd.Timestamp(start_date)
            and cached["Date"].max() >= pd.Timestamp(end_date) - pd.Timedelta(days=5)
        ):
            need_download = False

    if not need_download:
        return cached.sort_values("Date").reset_index(drop=True)

    print(f"Downloading {ticker} market history...")

    data = yf.download(
        ticker,
        start=str(pd.Timestamp(start_date).date()),
        end=str((pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    out = normalize_yf(data)

    if not out.empty:
        out.to_csv(cache, index=False)

    time.sleep(PAUSE)

    return out


def idx_on_or_before(market, date):
    if market is None or market.empty:
        return None

    dates = market["Date"].values
    i = np.searchsorted(
        dates,
        np.datetime64(pd.Timestamp(date)),
        side="right"
    ) - 1

    return int(i) if i >= 0 else None


def trailing_return(market, date, sessions):
    i = idx_on_or_before(market, date)

    if i is None or i - sessions < 0:
        return np.nan

    p1 = float(market.iloc[i]["Close"])
    p0 = float(market.iloc[i - sessions]["Close"])

    return (p1 / p0 - 1.0) * 100.0 if p0 else np.nan


def trailing_volatility(market, date, sessions):
    i = idx_on_or_before(market, date)

    if i is None or i - sessions < 1:
        return np.nan

    close = market.iloc[i - sessions:i + 1]["Close"].astype(float)
    ret = close.pct_change().dropna()

    if len(ret) < max(5, sessions // 2):
        return np.nan

    return float(ret.std(ddof=1) * 100.0)


def forward_return(market, date, sessions):
    if market is None or market.empty:
        return np.nan

    dates = market["Date"].values
    i = np.searchsorted(
        dates,
        np.datetime64(pd.Timestamp(date)),
        side="left"
    )

    if i >= len(market) or i + sessions >= len(market):
        return np.nan

    p0 = float(market.iloc[i]["Close"])
    p1 = float(market.iloc[i + sessions]["Close"])

    return (p1 / p0 - 1.0) * 100.0 if p0 else np.nan


def outcome_90d(market, date):
    """
    Return complete 0..90 trading-day trajectory outcome.
    None means right-censored / insufficient market history.
    """
    if market is None or market.empty:
        return None

    dates = market["Date"].values
    i = np.searchsorted(
        dates,
        np.datetime64(pd.Timestamp(date)),
        side="left"
    )

    if i >= len(market):
        return None

    post = market.iloc[i:i + 91].copy()

    if len(post) < 91:
        return None

    p0 = float(post.iloc[0]["Close"])

    if not p0:
        return None

    rel = (post["Close"].astype(float) / p0 - 1.0) * 100.0

    peak_pct = float(rel.max())
    peak_day = int(np.argmax(rel.values))

    return {
        "event_trading_day": str(post.iloc[0]["Date"].date()),
        "peak_pct": peak_pct,
        "peak_day": peak_day,
        "day90_pct": float(rel.iloc[90]),
    }


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

for p in [B_FILE, A_FILE, MASTER_FILE, MODEL_FILE]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

B = pd.read_csv(B_FILE)
A = pd.read_csv(A_FILE)
master = pd.read_csv(MASTER_FILE)
model = json.loads(MODEL_FILE.read_text())

B["award_date"] = pd.to_datetime(B["award_date"], errors="coerce")
A["award_date"] = pd.to_datetime(A["award_date"], errors="coerce")
master["award_date"] = pd.to_datetime(master["award_date"], errors="coerce")

B["transaction_amount"] = pd.to_numeric(
    B["transaction_amount"], errors="coerce"
)

master["transaction_amount"] = pd.to_numeric(
    master["transaction_amount"], errors="coerce"
)

# Safety: strip LMT again even though the supplied B file is already stripped.
B = B[
    B["ticker"].astype(str).str.upper() != "LMT"
].copy()


# ------------------------------------------------------------
# COLLAPSE SAMPLE B TO UNIQUE MARKET EVENTS
# ------------------------------------------------------------

Bday = (
    B.groupby(["ticker", "award_date"])
    .agg(
        company=("company", "first"),
        same_day_award_count=("award_id", "count"),
        transaction_amount_sum=("transaction_amount", "sum"),
        transaction_amount_abs_sum=(
            "transaction_amount",
            lambda s: pd.to_numeric(s, errors="coerce").abs().sum()
        ),
    )
    .reset_index()
)


# ------------------------------------------------------------
# REMOVE ANY SAMPLE-A TICKER/DATE OVERLAP
# ------------------------------------------------------------

Akeys = A[["ticker", "award_date"]].drop_duplicates().copy()
Akeys["overlap_A"] = 1

Bday = Bday.merge(
    Akeys,
    on=["ticker", "award_date"],
    how="left"
)

overlap_count = int(Bday["overlap_A"].fillna(0).sum())

Bday = Bday[
    Bday["overlap_A"].isna()
].drop(columns=["overlap_A"]).copy()


# ------------------------------------------------------------
# BUILD FULL SAVED-AWARD HISTORY
# ------------------------------------------------------------

master = master[
    master["transaction_amount"].notna()
    & (master["transaction_amount"] != 0)
].copy()

history = (
    master.groupby(["ticker", "award_date"])
    .agg(
        award_count=("award_id", "count"),
        signed_amount=("transaction_amount", "sum"),
        abs_amount=(
            "transaction_amount",
            lambda s: np.abs(pd.to_numeric(s, errors="coerce")).sum()
        ),
    )
    .reset_index()
    .sort_values(["ticker", "award_date"])
)


# ------------------------------------------------------------
# MARKET HISTORY
# ------------------------------------------------------------

earliest = min(
    Bday["award_date"].min(),
    history["award_date"].min()
) - pd.Timedelta(days=430)

latest = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)

tickers = sorted(
    set(Bday["ticker"].dropna().astype(str))
)

market = {}

for t in tickers + ["SPY"]:
    market[t] = load_market(t, earliest, latest)


# ------------------------------------------------------------
# ATTACH HISTORICAL 20D / 60D RESPONSES TO ALL AWARD DAYS
# ------------------------------------------------------------

resp_rows = []

for ticker, g in history.groupby("ticker"):
    m = market.get(str(ticker))

    if m is None or m.empty:
        continue

    for _, row in g.iterrows():
        resp_rows.append({
            "ticker": ticker,
            "award_date": row["award_date"],
            "hist_return_20d": forward_return(m, row["award_date"], 20),
            "hist_return_60d": forward_return(m, row["award_date"], 60),
        })

resp = pd.DataFrame(resp_rows)

history = history.merge(
    resp,
    on=["ticker", "award_date"],
    how="left"
)


# ------------------------------------------------------------
# BUILD EXACT 11 FROZEN FEATURES + 90D OUTCOME
# ------------------------------------------------------------

records = []

for n, ev in Bday.iterrows():
    ticker = str(ev["ticker"])
    d = pd.Timestamp(ev["award_date"])

    prior = history[
        (history["ticker"] == ticker)
        & (history["award_date"] < d)
    ].copy()

    m = market.get(ticker)
    spy = market.get("SPY")

    out90 = outcome_90d(m, d)

    # Skip right-censored events. No prediction/outcome test is made on them.
    if out90 is None:
        continue

    rec = ev.to_dict()

    # Historical award scale
    if len(prior):
        rec["prior_abs_award_max"] = float(prior["abs_amount"].max())
        rec["prior_signed_award_mean"] = float(prior["signed_amount"].mean())
        rec["prior_abs_award_median"] = float(prior["abs_amount"].median())
    else:
        rec["prior_abs_award_max"] = np.nan
        rec["prior_signed_award_mean"] = np.nan
        rec["prior_abs_award_median"] = np.nan

    # Previous 30-day award activity
    w30 = prior[
        prior["award_date"] >= d - pd.Timedelta(days=30)
    ]

    rec["prior_transactions_30d"] = (
        int(w30["award_count"].sum()) if len(w30) else 0
    )

    rec["prior_award_days_30d"] = int(len(w30))

    # Market state at award
    rec["pre_volatility_market_20d"] = trailing_volatility(
        m, d, 20
    )

    stock60 = trailing_return(m, d, 60)
    spy60 = trailing_return(spy, d, 60)

    stock120 = trailing_return(m, d, 120)
    spy120 = trailing_return(spy, d, 120)

    rec["relative_strength_spy_60d"] = (
        stock60 - spy60
        if pd.notna(stock60) and pd.notna(spy60)
        else np.nan
    )

    rec["relative_strength_spy_120d"] = (
        stock120 - spy120
        if pd.notna(stock120) and pd.notna(spy120)
        else np.nan
    )

    # Prior 20-day response.
    # Same maturity rule used in Sample A: ceil(20*1.6)+3 = 35 calendar days.
    matured20 = prior[
        prior["award_date"] <= d - pd.Timedelta(days=35)
    ]

    vals20 = pd.to_numeric(
        matured20["hist_return_20d"],
        errors="coerce"
    ).dropna()

    rec["prior_response_mean_20d"] = (
        float(vals20.mean()) if len(vals20) else np.nan
    )

    # Prior 60-day response count.
    # Same maturity rule used in Sample A: ceil(60*1.6)+3 = 99 calendar days.
    matured60 = prior[
        prior["award_date"] <= d - pd.Timedelta(days=99)
    ]

    vals60 = pd.to_numeric(
        matured60["hist_return_60d"],
        errors="coerce"
    ).dropna()

    rec["prior_response_count_60d"] = int(len(vals60))

    rec.update(out90)
    rec["actual_winner_ge20"] = int(out90["peak_pct"] >= 20)

    records.append(rec)

    if (n + 1) % 100 == 0:
        print(f"Processed {n+1}/{len(Bday)} candidate B events...")


V = pd.DataFrame(records)

if V.empty:
    raise RuntimeError("No mature Sample-B validation events were produced.")


# ------------------------------------------------------------
# APPLY FROZEN SAMPLE-A MODEL
# ------------------------------------------------------------

features = model["features"]

Xv = pd.DataFrame(index=V.index)

for c in features:
    s = pd.to_numeric(V[c], errors="coerce")

    if model["transforms"][c] == "signed_log1p":
        Xv[c] = np.sign(s) * np.log1p(np.abs(s))
    else:
        Xv[c] = s

for c in features:
    Xv[c] = Xv[c].fillna(
        model["impute_median"][c]
    )

mu = np.array([
    model["reference_mean"][c]
    for c in features
], dtype=float)

sd = np.array([
    model["reference_sd"][c]
    for c in features
], dtype=float)

loc = np.array(
    model["lw_location"],
    dtype=float
)

precision = np.array(
    model["lw_precision"],
    dtype=float
)

Z = (
    Xv[features].to_numpy(dtype=float) - mu
) / sd

D = Z - loc

md2 = np.einsum(
    "ij,jk,ik->i",
    D,
    precision,
    D
)

scores = np.sqrt(
    np.maximum(
        md2 / float(model["dimension_normalization"]),
        1e-12
    )
)

V["MTS_signal_score"] = scores
V["predicted_buy"] = (
    V["MTS_signal_score"] >= float(model["top5_threshold"])
).astype(int)


# ------------------------------------------------------------
# BLIND VALIDATION STATISTICS
# ------------------------------------------------------------

tp = int(((V["predicted_buy"] == 1) & (V["actual_winner_ge20"] == 1)).sum())
fp = int(((V["predicted_buy"] == 1) & (V["actual_winner_ge20"] == 0)).sum())
fn = int(((V["predicted_buy"] == 0) & (V["actual_winner_ge20"] == 1)).sum())
tn = int(((V["predicted_buy"] == 0) & (V["actual_winner_ge20"] == 0)).sum())

table = np.array([
    [tp, fp],
    [fn, tn],
], dtype=int)

chi2, chi_p, chi_dof, expected = chi2_contingency(
    table,
    correction=False
)

oddsratio, fisher_p = fisher_exact(
    table,
    alternative="two-sided"
)

precision_ppv = tp / (tp + fp) if (tp + fp) else np.nan
sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
specificity = tn / (tn + fp) if (tn + fp) else np.nan
npv = tn / (tn + fn) if (tn + fn) else np.nan
false_positive_rate = fp / (fp + tn) if (fp + tn) else np.nan

base_rate = (
    V["actual_winner_ge20"].mean()
)

relative_risk = (
    precision_ppv / base_rate
    if base_rate > 0 and pd.notna(precision_ppv)
    else np.nan
)

auc = roc_auc_score(
    V["actual_winner_ge20"],
    V["MTS_signal_score"]
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

V.to_csv(OUT_EVENTS, index=False)

table_df = pd.DataFrame(
    table,
    index=["Predicted BUY", "Predicted NO BUY"],
    columns=["Actual >=20%", "Actual <20%"]
)

table_df.to_csv(OUT_TABLE)

summary = {
    "version": "RECON Sample-B Blind Validation Non-LMT V1.0",
    "sample_B_raw_non_LMT_rows": int(len(B)),
    "sample_B_unique_non_LMT_ticker_days_before_overlap_removal": int(len(Bday) + overlap_count),
    "sample_A_overlap_ticker_days_removed": int(overlap_count),
    "sample_B_nonoverlap_candidate_ticker_days": int(len(Bday)),
    "mature_90d_validation_events": int(len(V)),
    "frozen_model_version": model["version"],
    "frozen_features": features,
    "frozen_buy_threshold": float(model["top5_threshold"]),
    "contingency": {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
    },
    "actual_ge20_base_rate": float(base_rate),
    "predicted_buy_count": int(tp + fp),
    "precision_ppv": float(precision_ppv) if pd.notna(precision_ppv) else None,
    "sensitivity": float(sensitivity) if pd.notna(sensitivity) else None,
    "specificity": float(specificity) if pd.notna(specificity) else None,
    "npv": float(npv) if pd.notna(npv) else None,
    "false_positive_rate": float(false_positive_rate) if pd.notna(false_positive_rate) else None,
    "relative_risk": float(relative_risk) if pd.notna(relative_risk) else None,
    "odds_ratio": float(oddsratio),
    "auc": float(auc),
    "chi_square": float(chi2),
    "chi_square_p": float(chi_p),
    "chi_square_dof": int(chi_dof),
    "chi_square_expected_counts": expected.tolist(),
    "fisher_exact_p": float(fisher_p),
    "notes": [
        "LMT excluded before validation.",
        "Ticker is not a predictor.",
        "Any B ticker/date also present in Sample A was excluded.",
        "Only B events with a complete 90-trading-day future were evaluated.",
        "The Sample-A model and threshold were fixed before reading Sample-B outcomes."
    ]
}

OUT_SUMMARY.write_text(
    json.dumps(summary, indent=2)
)

recon = Path("Recon")

if recon.exists():
    V.to_csv(recon / OUT_EVENTS.name, index=False)
    table_df.to_csv(recon / OUT_TABLE.name)
    (recon / OUT_SUMMARY.name).write_text(
        json.dumps(summary, indent=2)
    )

print()
print("=" * 78)
print("SAMPLE B NON-LMT BLIND VALIDATION COMPLETE")
print("=" * 78)
print(table_df)
print()
print(json.dumps(summary, indent=2))
print("=" * 78)
