# ============================================================
# RECON SAMPLE-C BLIND VALIDATION V2.0
# ============================================================
#
# PURPOSE
# -------
# Open the sealed Sample C exactly once and evaluate the frozen
# two-stage RECON system:
#
#   Stage 1: frozen MTS abnormality filter
#            MTS_signal_score >= 2.1954452583448045
#
#   Stage 2: frozen XGBoost V2 regression model
#            recon_stage2_v2.ubj
#
# IMPORTANT
# ---------
# - LMT is excluded.
# - No Sample-C outcome is used to fit or tune either stage.
# - All 11 MTS predictors use information available at award time.
# - Scale features use only pre-award price/volume and shares
#   outstanding known on or before the award date.
# - Only events with a complete 90-trading-day future are scored
#   for validation statistics.
#
# INPUTS
# ------
# sample_C_nonLMT_raw.csv
# master_zero_purged.csv
# mts_frozen_model_A.json
# recon_stage2_v2.ubj
#
# OUTPUTS
# -------
# sample_C_v2_validation_events.csv
# sample_C_v2_stage1_candidates.csv
# sample_C_v2_summary.json
#
# ============================================================

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor

C_FILE = Path("sample_C_nonLMT_raw.csv")
MASTER_FILE = Path("master_zero_purged.csv")
MTS_MODEL_FILE = Path("mts_frozen_model_A.json")
STAGE2_MODEL_FILE = Path("recon_stage2_v2.ubj")

OUT_EVENTS = Path("sample_C_v2_validation_events.csv")
OUT_CANDIDATES = Path("sample_C_v2_stage1_candidates.csv")
OUT_SUMMARY = Path("sample_C_v2_summary.json")

MARKET_CACHE = Path("market_cache_C")
SCALE_CACHE = Path("scale_cache_C")
MARKET_CACHE.mkdir(exist_ok=True)
SCALE_CACHE.mkdir(exist_ok=True)

PAUSE = 0.15

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def normalize_yf(data):
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["Date", "Close", "Volume"])

    d = data.copy()

    if isinstance(d.columns, pd.MultiIndex):
        close = d["Close"]
        vol = d["Volume"] if "Volume" in d.columns.get_level_values(0) else np.nan
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
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
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)

    return out[["Date", "Close", "Volume"]]


def get_market(ticker, start, end):
    p = MARKET_CACHE / f"{ticker}.csv"

    if p.exists():
        try:
            d = pd.read_csv(p, parse_dates=["Date"])
            if len(d):
                return d
        except Exception:
            pass

    raw = yf.download(
        ticker,
        start=(start - pd.Timedelta(days=10)).date().isoformat(),
        end=(end + pd.Timedelta(days=5)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    d = normalize_yf(raw)
    d.to_csv(p, index=False)
    time.sleep(PAUSE)
    return d


def trading_window(mkt, award_date, n_after=90):
    d = pd.Timestamp(award_date)
    after = mkt[mkt["Date"] >= d].sort_values("Date").head(n_after + 1)
    return after


def trailing_return(mkt, award_date, n):
    d = pd.Timestamp(award_date)
    prior = mkt[mkt["Date"] < d].sort_values("Date").tail(n + 1)
    if len(prior) < 2:
        return np.nan
    first = float(prior.iloc[0]["Close"])
    last = float(prior.iloc[-1]["Close"])
    if first == 0:
        return np.nan
    return (last / first - 1.0) * 100.0


def trailing_volatility(mkt, award_date, n):
    d = pd.Timestamp(award_date)
    prior = mkt[mkt["Date"] < d].sort_values("Date").tail(n + 1)
    if len(prior) < 3:
        return np.nan
    r = pd.to_numeric(prior["Close"], errors="coerce").pct_change().dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * 100.0)


def forward_outcome_90(mkt, award_date):
    d = pd.Timestamp(award_date)
    w = trading_window(mkt, d, 90)

    # Need award day/baseline + 90 future trading observations.
    if len(w) < 91:
        return None

    baseline = float(w.iloc[0]["Close"])
    if baseline <= 0:
        return None

    future = w.iloc[1:91].copy()
    gains = (pd.to_numeric(future["Close"], errors="coerce") / baseline - 1.0) * 100.0
    gains = gains.dropna()

    if len(gains) < 90:
        return None

    peak_pct = float(max(0.0, gains.max()))

    hits = {}
    for threshold in [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]:
        hit_idx = np.where(gains.to_numpy() >= threshold)[0]
        hits[f"hit_{str(threshold).replace('.','p')}"] = int(len(hit_idx) > 0)
        hits[f"first_day_{str(threshold).replace('.','p')}"] = (
            int(hit_idx[0] + 1) if len(hit_idx) else np.nan
        )

    return {
        "baseline_close": baseline,
        "peak_pct": peak_pct,
        **hits,
    }


def get_shares_history(ticker, start, end):
    p = SCALE_CACHE / f"{ticker}_shares.csv"

    if p.exists():
        try:
            return pd.read_csv(p, parse_dates=["Date"])
        except Exception:
            pass

    out = pd.DataFrame(columns=["Date", "Shares"])

    try:
        raw = yf.Ticker(ticker).get_shares_full(
            start=(start - pd.Timedelta(days=30)).date().isoformat(),
            end=(end + pd.Timedelta(days=3)).date().isoformat(),
        )
        if raw is not None and len(raw):
            idx = pd.to_datetime(raw.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            out = pd.DataFrame({
                "Date": idx,
                "Shares": pd.to_numeric(raw.values, errors="coerce"),
            }).dropna().sort_values("Date")
    except Exception as exc:
        print(f"Shares history unavailable for {ticker}: {exc}")

    out.to_csv(p, index=False)
    time.sleep(PAUSE)
    return out


# ------------------------------------------------------------
# LOAD / COLLAPSE SAMPLE C
# ------------------------------------------------------------

C = pd.read_csv(C_FILE)
C["ticker"] = C["ticker"].astype(str).str.upper().str.strip()
C = C[C["ticker"] != "LMT"].copy()
C["award_date"] = pd.to_datetime(C["award_date"])
C["transaction_amount"] = pd.to_numeric(C["transaction_amount"], errors="coerce").fillna(0.0)

# Collapse duplicate ticker/date transactions exactly as B validation did.
Cday = (
    C.groupby(["ticker", "award_date"], as_index=False)
     .agg(
         company=("company", "first"),
         transaction_amount_sum=("transaction_amount", "sum"),
         transaction_amount_abs_sum=("transaction_amount", lambda s: np.abs(s).sum()),
         same_day_award_count=("award_id", "count"),
         agency=("agency", "first"),
         subagency=("subagency", "first"),
         award_type=("award_type", "first"),
     )
)

# ------------------------------------------------------------
# LOAD MASTER AWARD HISTORY
# ------------------------------------------------------------

M = pd.read_csv(MASTER_FILE)
M["ticker"] = M["ticker"].astype(str).str.upper().str.strip()
M = M[M["ticker"] != "LMT"].copy()
M["award_date"] = pd.to_datetime(M["award_date"])
M["transaction_amount"] = pd.to_numeric(M["transaction_amount"], errors="coerce").fillna(0.0)

# Collapse master to ticker/day to match prior-history logic.
Mday = (
    M.groupby(["ticker", "award_date"], as_index=False)
     .agg(
         signed_amount=("transaction_amount", "sum"),
         abs_amount=("transaction_amount", lambda s: np.abs(s).sum()),
         award_count=("award_id", "count"),
     )
)

# ------------------------------------------------------------
# MARKET DATA
# ------------------------------------------------------------

start = min(Cday["award_date"].min(), Mday["award_date"].min()) - pd.Timedelta(days=450)
end = Cday["award_date"].max() + pd.Timedelta(days=150)

tickers = sorted(Cday["ticker"].unique())
market = {}

print(f"Downloading/caching market history for {len(tickers)} C tickers + SPY...")

for t in tickers:
    market[t] = get_market(t, start, end)

spy = get_market("SPY", start, end)

# Precompute historical award stock responses needed for prior-response factors.
hist_rows = []

for t in sorted(Mday["ticker"].unique()):
    if t not in market:
        market[t] = get_market(t, start, end)

    mkt = market[t]
    if mkt.empty:
        continue

    tm = Mday[Mday["ticker"] == t].sort_values("award_date")

    for _, r in tm.iterrows():
        d = r["award_date"]
        basewin = mkt[mkt["Date"] >= d].sort_values("Date")
        if len(basewin) == 0:
            r20 = np.nan
            r60 = np.nan
        else:
            base = float(basewin.iloc[0]["Close"])

            w20 = basewin.head(21)
            w60 = basewin.head(61)

            r20 = (
                (float(w20.iloc[-1]["Close"]) / base - 1.0) * 100.0
                if len(w20) >= 21 and base > 0 else np.nan
            )
            r60 = (
                (float(w60.iloc[-1]["Close"]) / base - 1.0) * 100.0
                if len(w60) >= 61 and base > 0 else np.nan
            )

        hist_rows.append({
            "ticker": t,
            "award_date": d,
            "hist_return_20d": r20,
            "hist_return_60d": r60,
        })

Hist = pd.DataFrame(hist_rows)

# ------------------------------------------------------------
# RECONSTRUCT THE FROZEN 11 + OUTCOMES
# ------------------------------------------------------------

records = []

for n, ev in Cday.iterrows():
    t = ev["ticker"]
    d = ev["award_date"]
    mkt = market.get(t, pd.DataFrame())

    if mkt.empty:
        continue

    out90 = forward_outcome_90(mkt, d)
    if out90 is None:
        continue

    prior = Mday[
        (Mday["ticker"] == t) &
        (Mday["award_date"] < d)
    ].copy()

    rec = ev.to_dict()

    if len(prior):
        rec["prior_abs_award_max"] = float(prior["abs_amount"].max())
        rec["prior_signed_award_mean"] = float(prior["signed_amount"].mean())
        rec["prior_abs_award_median"] = float(prior["abs_amount"].median())
    else:
        rec["prior_abs_award_max"] = np.nan
        rec["prior_signed_award_mean"] = np.nan
        rec["prior_abs_award_median"] = np.nan

    w30 = prior[prior["award_date"] >= d - pd.Timedelta(days=30)]
    rec["prior_transactions_30d"] = int(w30["award_count"].sum()) if len(w30) else 0
    rec["prior_award_days_30d"] = int(len(w30))

    rec["pre_volatility_market_20d"] = trailing_volatility(mkt, d, 20)

    stock60 = trailing_return(mkt, d, 60)
    spy60 = trailing_return(spy, d, 60)
    stock120 = trailing_return(mkt, d, 120)
    spy120 = trailing_return(spy, d, 120)

    rec["relative_strength_spy_60d"] = (
        stock60 - spy60 if pd.notna(stock60) and pd.notna(spy60) else np.nan
    )
    rec["relative_strength_spy_120d"] = (
        stock120 - spy120 if pd.notna(stock120) and pd.notna(spy120) else np.nan
    )

    hprior = Hist[
        (Hist["ticker"] == t) &
        (Hist["award_date"] < d)
    ].copy()

    matured20 = hprior[hprior["award_date"] <= d - pd.Timedelta(days=35)]
    vals20 = pd.to_numeric(matured20["hist_return_20d"], errors="coerce").dropna()
    rec["prior_response_mean_20d"] = float(vals20.mean()) if len(vals20) else np.nan

    matured60 = hprior[hprior["award_date"] <= d - pd.Timedelta(days=99)]
    vals60 = pd.to_numeric(matured60["hist_return_60d"], errors="coerce").dropna()
    rec["prior_response_count_60d"] = int(len(vals60))

    rec.update(out90)
    records.append(rec)

    if (n + 1) % 100 == 0:
        print(f"Processed {n+1}/{len(Cday)} C ticker-days...")

V = pd.DataFrame(records)

if V.empty:
    raise RuntimeError("No mature Sample-C validation events were produced.")

# ------------------------------------------------------------
# STAGE 1: FROZEN MTS
# ------------------------------------------------------------

mts = json.loads(MTS_MODEL_FILE.read_text())
features11 = list(mts["features"])

X = pd.DataFrame(index=V.index)

for c in features11:
    s = pd.to_numeric(V[c], errors="coerce")

    if mts["transforms"][c] == "signed_log1p":
        X[c] = np.sign(s) * np.log1p(np.abs(s))
    else:
        X[c] = s

for c in features11:
    X[c] = X[c].fillna(mts["impute_median"][c])

mu = np.array([mts["reference_mean"][c] for c in features11], dtype=float)
sd = np.array([mts["reference_sd"][c] for c in features11], dtype=float)
loc = np.array(mts["lw_location"], dtype=float)
precision = np.array(mts["lw_precision"], dtype=float)

Z = (X[features11].to_numpy(dtype=float) - mu) / sd
D = Z - loc

md2 = np.einsum("ij,jk,ik->i", D, precision, D)
V["MTS_signal_score"] = np.sqrt(
    np.maximum(md2 / float(mts["dimension_normalization"]), 1e-12)
)

threshold = float(mts["top5_threshold"])
V["stage1_candidate"] = (V["MTS_signal_score"] >= threshold).astype(int)

# ------------------------------------------------------------
# STAGE 2 V2 SCALE FEATURES
# ------------------------------------------------------------

V["log_market_cap_preaward"] = np.nan
V["award_to_market_cap"] = np.nan
V["award_to_adv60"] = np.nan

for t, idxs in V.groupby("ticker").groups.items():
    mkt = market[t].sort_values("Date")
    shares = get_shares_history(t, start, end)

    for i in idxs:
        d = pd.Timestamp(V.at[i, "award_date"])

        prior60 = mkt[mkt["Date"] < d].tail(60)
        if len(prior60) == 0:
            continue

        close = float(prior60.iloc[-1]["Close"])
        dv = (
            pd.to_numeric(prior60["Close"], errors="coerce") *
            pd.to_numeric(prior60["Volume"], errors="coerce")
        )
        adv60 = float(dv.dropna().mean()) if dv.notna().any() else np.nan

        sh = np.nan
        if not shares.empty:
            ss = shares[shares["Date"] <= d]
            if len(ss):
                sh = float(ss.iloc[-1]["Shares"])

        mcap = close * sh if pd.notna(sh) and sh > 0 else np.nan
        award = abs(float(pd.to_numeric(V.at[i, "transaction_amount_abs_sum"], errors="coerce")))

        if pd.notna(mcap) and mcap > 0:
            V.at[i, "log_market_cap_preaward"] = math.log(mcap)
            V.at[i, "award_to_market_cap"] = award / mcap

        if pd.notna(adv60) and adv60 > 0:
            V.at[i, "award_to_adv60"] = award / adv60

# ------------------------------------------------------------
# STAGE 2: FROZEN XGBOOST V2
# ------------------------------------------------------------

stage2 = XGBRegressor()
stage2.load_model(STAGE2_MODEL_FILE)

stage2_features = features11 + [
    "log_market_cap_preaward",
    "award_to_market_cap",
    "award_to_adv60",
]

cand_mask = V["stage1_candidate"] == 1
V["stage2_v2_pred_peak"] = np.nan

if cand_mask.any():
    XC = V.loc[cand_mask, stage2_features].apply(pd.to_numeric, errors="coerce")
    V.loc[cand_mask, "stage2_v2_pred_peak"] = stage2.predict(XC)

# No new cutoff is invented here.
# C reports the frozen Stage-1 candidate set and Stage-2 ranking/predictions.
CAND = V[V["stage1_candidate"] == 1].copy()
CAND["stage2_rank_high_to_low"] = (
    CAND["stage2_v2_pred_peak"]
    .rank(method="min", ascending=False)
    .astype("Int64")
)

# ------------------------------------------------------------
# BLIND C RESULTS
# ------------------------------------------------------------

thresholds = [5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0]

hit_rates = {}
for th in thresholds:
    col = f"hit_{str(th).replace('.','p')}"
    hits = int(CAND[col].sum()) if len(CAND) else 0
    hit_rates[str(th)] = {
        "hits": hits,
        "candidates": int(len(CAND)),
        "rate": float(hits / len(CAND)) if len(CAND) else None,
    }

summary = {
    "version": "RECON Sample-C Blind Validation V2.0",
    "sample_C_raw_rows": int(len(C)),
    "sample_C_unique_nonLMT_ticker_days": int(len(Cday)),
    "mature_90d_validation_events": int(len(V)),
    "stage1_threshold": threshold,
    "stage1_candidate_count": int(len(CAND)),
    "stage1_candidate_rate": float(len(CAND) / len(V)) if len(V) else None,
    "stage2_model": "recon_stage2_v2.ubj frozen before C",
    "hit_rates_within_90_trading_days": hit_rates,
    "mean_peak_pct_stage1_candidates": (
        float(CAND["peak_pct"].mean()) if len(CAND) else None
    ),
    "median_peak_pct_stage1_candidates": (
        float(CAND["peak_pct"].median()) if len(CAND) else None
    ),
    "notes": [
        "LMT excluded.",
        "No Sample-C outcome used for fitting or tuning.",
        "Stage 1 MTS model and threshold unchanged.",
        "Stage 2 V2 model loaded from frozen UBJ.",
        "No new Stage-2 cutoff was selected using Sample C.",
    ],
}

V.to_csv(OUT_EVENTS, index=False)

cols = [
    "ticker",
    "award_date",
    "company",
    "MTS_signal_score",
    "stage2_v2_pred_peak",
    "stage2_rank_high_to_low",
    "peak_pct",
    "hit_5p0",
    "hit_7p5",
    "hit_10p0",
    "hit_12p5",
    "hit_15p0",
    "hit_20p0",
    "log_market_cap_preaward",
    "award_to_market_cap",
    "award_to_adv60",
]
CAND[cols].sort_values(
    ["stage2_v2_pred_peak", "MTS_signal_score"],
    ascending=[False, False]
).to_csv(OUT_CANDIDATES, index=False)

OUT_SUMMARY.write_text(json.dumps(summary, indent=2))

print()
print("=" * 84)
print("SAMPLE C BLIND VALIDATION COMPLETE")
print("=" * 84)
print(f"Mature C events: {len(V)}")
print(f"Stage-1 candidates (MD >= {threshold:.12f}): {len(CAND)}")
print()
for th in thresholds:
    r = hit_rates[str(th)]
    if r["rate"] is None:
        print(f">= +{th:g}% : no candidates")
    else:
        print(
            f">= +{th:g}% : {r['hits']}/{r['candidates']} "
            f"= {100*r['rate']:.1f}%"
        )

print()
print("TOP STAGE-2 CANDIDATES")
if len(CAND):
    print(
        CAND[
            ["ticker","award_date","MTS_signal_score",
             "stage2_v2_pred_peak","peak_pct"]
        ]
        .sort_values("stage2_v2_pred_peak", ascending=False)
        .head(20)
        .to_string(index=False)
    )
else:
    print("No Stage-1 candidates.")

print()
print(f"Wrote {OUT_EVENTS}")
print(f"Wrote {OUT_CANDIDATES}")
print(f"Wrote {OUT_SUMMARY}")
print("=" * 84)
