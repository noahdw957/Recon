# ============================================================
# RECON 8-FACTOR MTS: FREEZE ON A, DIAGNOSTIC ON B
# ============================================================
#
# PURPOSE
# -------
# Follow the L16 result exactly.
#
# The L16 positive-S/N factor set was:
#   1. log10_market_cap_before
#   2. prior_abs_award_max
#   3. prior_response_count_60d
#   4. prior_abs_award_median
#   5. pre_volatility_market_20d
#   6. prior_transactions_30d
#   7. prior_response_mean_20d
#   8. relative_strength_spy_60d
#
# This script:
#   - fits/finalizes the 8D MTS reference space on Sample A only
#   - derives the BUY threshold from Sample A only
#   - applies the frozen 8D model to Sample B
#   - compares against the original 11-factor B predictions
#   - reports which original BUYs were rejected and which new BUYs appear
#
# NO STAGE 2 / XGBOOST.
#
# INPUTS
# ------
# recon_L16_15factor_input_A.csv
# sample_B_nonLMT_validation_events.csv
#
# OUTPUTS
# -------
# recon_mts8_frozen_A.json
# recon_mts8_A_thresholds.csv
# recon_mts8_B_diagnostic.csv
# recon_mts8_B_summary.json
#
# ============================================================

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

A_FILE = Path("recon_L16_15factor_input_A.csv")
B_FILE = Path("sample_B_nonLMT_validation_events.csv")

OUT_MODEL = Path("recon_mts8_frozen_A.json")
OUT_THRESH = Path("recon_mts8_A_thresholds.csv")
OUT_B = Path("recon_mts8_B_diagnostic.csv")
OUT_SUM = Path("recon_mts8_B_summary.json")

CACHE = Path("mts8_market_cache")
CACHE.mkdir(exist_ok=True)

FEATURES = [
    "log10_market_cap_before",
    "prior_abs_award_max",
    "prior_response_count_60d",
    "prior_abs_award_median",
    "pre_volatility_market_20d",
    "prior_transactions_30d",
    "prior_response_mean_20d",
    "relative_strength_spy_60d",
]

TRANSFORMS = {
    "log10_market_cap_before": "identity",
    "prior_abs_award_max": "signed_log1p",
    "prior_response_count_60d": "signed_log1p",
    "prior_abs_award_median": "signed_log1p",
    "pre_volatility_market_20d": "identity",
    "prior_transactions_30d": "signed_log1p",
    "prior_response_mean_20d": "identity",
    "relative_strength_spy_60d": "identity",
}


def tx(s, kind):
    s = pd.to_numeric(s, errors="coerce")
    if kind == "signed_log1p":
        return np.sign(s) * np.log1p(np.abs(s))
    return s.astype(float)


def make_X(d):
    out = pd.DataFrame(index=d.index)
    for c in FEATURES:
        out[c] = tx(d[c], TRANSFORMS[c])
    return out


def normalize_yf(raw):
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["Date", "Close"])

    d = raw.copy()

    if isinstance(d.columns, pd.MultiIndex):
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        d = pd.DataFrame({"Close": close})
    else:
        d = pd.DataFrame({"Close": d["Close"]})

    d = d.dropna()
    d.index = pd.to_datetime(d.index)

    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)

    d = d.reset_index()
    d = d.rename(columns={d.columns[0]: "Date"})
    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)

    return d[["Date", "Close"]]


def market_history(ticker, start, end):
    p = CACHE / f"{ticker}_prices.csv"

    if p.exists():
        try:
            return pd.read_csv(p, parse_dates=["Date"])
        except Exception:
            pass

    raw = yf.download(
        ticker,
        start=(start - pd.Timedelta(days=10)).date().isoformat(),
        end=(end + pd.Timedelta(days=3)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    x = normalize_yf(raw)
    x.to_csv(p, index=False)
    time.sleep(0.10)
    return x


def shares_history(ticker, start, end):
    p = CACHE / f"{ticker}_shares.csv"

    if p.exists():
        try:
            return pd.read_csv(p, parse_dates=["Date"])
        except Exception:
            pass

    out = pd.DataFrame(columns=["Date", "Shares"])

    try:
        s = yf.Ticker(ticker).get_shares_full(
            start=(start - pd.Timedelta(days=45)).date().isoformat(),
            end=(end + pd.Timedelta(days=3)).date().isoformat(),
        )

        if s is not None and len(s):
            idx = pd.to_datetime(s.index)

            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)

            out = pd.DataFrame({
                "Date": idx,
                "Shares": pd.to_numeric(s.values, errors="coerce"),
            }).dropna().sort_values("Date")

    except Exception as exc:
        print(f"[shares] {ticker}: {exc}")

    out.to_csv(p, index=False)
    time.sleep(0.10)
    return out


# ============================================================
# SAMPLE A: FIT FROZEN 8D REFERENCE SPACE
# ============================================================

A = pd.read_csv(A_FILE)

A["ticker"] = A["ticker"].astype(str).str.upper().str.strip()
A["peak_pct"] = pd.to_numeric(A["peak_pct"], errors="coerce")
A["transaction_amount_abs_sum"] = pd.to_numeric(
    A["transaction_amount_abs_sum"], errors="coerce"
)

A = A[
    (A["ticker"] != "LMT")
    & (A["transaction_amount_abs_sum"] > 0)
    & (A["peak_pct"].notna())
].copy().reset_index(drop=True)

missing = [c for c in FEATURES if c not in A.columns]
if missing:
    raise KeyError(f"Sample A missing required 8-factor columns: {missing}")

XA = make_X(A)

normal = A["peak_pct"] < 10.0
ref = XA.loc[normal].copy()

med = ref.median().fillna(0.0)
ref = ref.fillna(med)

mu = ref.mean()
sd = ref.std(ddof=1).replace(0, 1.0)

zr = (ref - mu) / sd
lw = LedoitWolf().fit(zr.values)


def score_matrix(X):
    x = X.copy().fillna(med)
    z = (x - mu) / sd
    d = z.values - lw.location_
    md2 = np.einsum("ij,jk,ik->i", d, lw.precision_, d) / len(FEATURES)
    return np.sqrt(np.maximum(md2, 1e-12))


A["MTS8_score"] = score_matrix(XA)

# Same principle as original model:
# derive thresholds on Sample A only and use top 5% as deployable BUY threshold.
fracs = [0.20, 0.10, 0.05, 0.02, 0.01]
trs = []

for f in fracs:
    th = float(A["MTS8_score"].quantile(1 - f))
    g = A[A["MTS8_score"] >= th]

    trs.append({
        "top_fraction": f,
        "threshold": th,
        "A_signals": int(len(g)),
        "A_hit_ge7p5": float((g["peak_pct"] >= 7.5).mean()),
        "A_hit_ge10": float((g["peak_pct"] >= 10).mean()),
        "A_hit_ge15": float((g["peak_pct"] >= 15).mean()),
        "A_hit_ge20": float((g["peak_pct"] >= 20).mean()),
        "A_median_peak": float(g["peak_pct"].median()),
        "A_mean_peak": float(g["peak_pct"].mean()),
    })

T = pd.DataFrame(trs)
T.to_csv(OUT_THRESH, index=False)

TH = float(T.loc[np.isclose(T["top_fraction"], 0.05), "threshold"].iloc[0])

model = {
    "version": "RECON MTS8 Frozen A v1.0",
    "source": "L16 positive-S/N factor set",
    "features": FEATURES,
    "transforms": TRANSFORMS,
    "training_rows": int(len(A)),
    "reference_rule": "Sample A events with peak_pct < 10",
    "reference_rows": int(normal.sum()),
    "impute_median": med.to_dict(),
    "reference_mean": mu.to_dict(),
    "reference_sd": sd.to_dict(),
    "ledoit_location": lw.location_.tolist(),
    "ledoit_precision": lw.precision_.tolist(),
    "dimension_normalization": 8,
    "buy_rule": "MTS8_score >= Sample-A top-5% threshold",
    "top5_threshold": TH,
    "notes": [
        "Factor set comes directly from positive S/N effects in the 15-column L16.",
        "Sample B outcomes are not used to fit the model or choose the threshold.",
        "No Stage 2 / XGBoost.",
    ],
}

OUT_MODEL.write_text(json.dumps(model, indent=2))


# ============================================================
# SAMPLE B: POINT-IN-TIME MARKET CAP + FROZEN 8D SCORING
# ============================================================

B = pd.read_csv(B_FILE)

B["ticker"] = B["ticker"].astype(str).str.upper().str.strip()
B["award_date"] = pd.to_datetime(B["award_date"])
B["transaction_amount_abs_sum"] = pd.to_numeric(
    B["transaction_amount_abs_sum"], errors="coerce"
)
B["peak_pct"] = pd.to_numeric(B["peak_pct"], errors="coerce")

B = B[
    (B["ticker"] != "LMT")
    & (B["transaction_amount_abs_sum"] > 0)
    & (B["peak_pct"].notna())
].copy().reset_index(drop=True)

start = B["award_date"].min() - pd.Timedelta(days=500)
end = B["award_date"].max() + pd.Timedelta(days=5)

mcap = np.full(len(B), np.nan)

for n, (ticker, g) in enumerate(B.groupby("ticker"), 1):
    print(f"[{n}/{B['ticker'].nunique()}] point-in-time market cap: {ticker}")

    px = market_history(ticker, start, end)
    sh = shares_history(ticker, start, end)

    for idx in g.index:
        d = B.at[idx, "award_date"]

        p = px[px["Date"] < d].sort_values("Date")
        s = sh[sh["Date"] <= d].sort_values("Date")

        if len(p) and len(s):
            close = float(p.iloc[-1]["Close"])
            shares = float(s.iloc[-1]["Shares"])

            if close > 0 and shares > 0:
                mcap[idx] = close * shares

B["market_cap_before"] = mcap
B["log10_market_cap_before"] = np.log10(B["market_cap_before"])

missing_B = [c for c in FEATURES if c not in B.columns]
if missing_B:
    raise KeyError(f"Sample B missing required 8-factor columns: {missing_B}")

XB = make_X(B)
B["MTS8_score"] = score_matrix(XB)

B["MTS11_score_old"] = pd.to_numeric(B.get("MTS_signal_score"), errors="coerce")
B["old_buy"] = pd.to_numeric(B.get("predicted_buy"), errors="coerce").fillna(0).astype(int)
B["buy8"] = (B["MTS8_score"] >= TH).astype(int)
B["delta_score_8_minus_11"] = B["MTS8_score"] - B["MTS11_score_old"]

for x in [7.5, 10, 15, 20]:
    B[f"hit_{str(x).replace('.', 'p')}"] = (B["peak_pct"] >= x).astype(int)

B.to_csv(OUT_B, index=False)


def stats(mask):
    g = B.loc[mask]

    if not len(g):
        return {"n": 0}

    return {
        "n": int(len(g)),
        "hit_ge7p5": float((g["peak_pct"] >= 7.5).mean()),
        "hit_ge10": float((g["peak_pct"] >= 10).mean()),
        "hit_ge15": float((g["peak_pct"] >= 15).mean()),
        "hit_ge20": float((g["peak_pct"] >= 20).mean()),
        "median_peak": float(g["peak_pct"].median()),
        "mean_peak": float(g["peak_pct"].mean()),
    }


old = B["old_buy"] == 1
new = B["buy8"] == 1

rejected = B[old & ~new].copy()
added = B[~old & new].copy()
kept = B[old & new].copy()

summary = {
    "version": "RECON MTS8 B Diagnostic v1.0",
    "factor_set": FEATURES,
    "A_top5_threshold_8D": TH,
    "B_rows": int(len(B)),
    "B_market_cap_coverage": float(B["market_cap_before"].notna().mean()),
    "old_11_buy": stats(old),
    "new_8_buy": stats(new),
    "old_11_buys_kept_by_8": stats(old & new),
    "old_11_buys_rejected_by_8": stats(old & ~new),
    "new_8_buys_added_vs_old": stats(~old & new),
    "rejected_old_buy_rows": [],
    "added_new_buy_rows": [],
    "known_failure_rows": [],
}

for _, r in rejected.sort_values("MTS11_score_old", ascending=False).iterrows():
    summary["rejected_old_buy_rows"].append({
        "ticker": r["ticker"],
        "award_date": str(r["award_date"].date()),
        "peak_pct": float(r["peak_pct"]),
        "old_MD11": float(r["MTS11_score_old"]),
        "new_MD8": float(r["MTS8_score"]),
    })

for _, r in added.sort_values("MTS8_score", ascending=False).iterrows():
    summary["added_new_buy_rows"].append({
        "ticker": r["ticker"],
        "award_date": str(r["award_date"].date()),
        "peak_pct": float(r["peak_pct"]),
        "old_MD11": float(r["MTS11_score_old"]),
        "new_MD8": float(r["MTS8_score"]),
    })

# Known B failures we care about.
known = [
    ("AVAV", "2026-01-26"),
    ("BA", "2025-08-12"),
    ("KTOS", "2026-01-29"),
]

for ticker, date in known:
    q = B[
        (B["ticker"] == ticker)
        & (B["award_date"] == pd.Timestamp(date))
    ]

    if len(q):
        r = q.iloc[0]

        summary["known_failure_rows"].append({
            "ticker": ticker,
            "award_date": date,
            "peak_pct": float(r["peak_pct"]),
            "old_MD11": float(r["MTS11_score_old"]),
            "new_MD8": float(r["MTS8_score"]),
            "old_buy": int(r["old_buy"]),
            "new_buy8": int(r["buy8"]),
            "log10_market_cap_before": (
                None if pd.isna(r["log10_market_cap_before"])
                else float(r["log10_market_cap_before"])
            ),
        })

OUT_SUM.write_text(json.dumps(summary, indent=2))

print()
print("=" * 92)
print("RECON MTS8 A -> B COMPLETE")
print("=" * 92)
print(f"A-derived 8D top-5% threshold: {TH:.12f}")
print()
print("OLD 11-FACTOR BUY:")
print(summary["old_11_buy"])
print()
print("NEW 8-FACTOR BUY:")
print(summary["new_8_buy"])
print()
print("OLD 11 BUYs REJECTED BY 8:")
print(summary["old_11_buys_rejected_by_8"])
print()
print("NEW 8 BUYs ADDED VS OLD 11:")
print(summary["new_8_buys_added_vs_old"])
print()
print("KNOWN FAILURES:")
for r in summary["known_failure_rows"]:
    print(r)
print()
print("FILES:")
for p in [OUT_MODEL, OUT_THRESH, OUT_B, OUT_SUM]:
    print(f"  {p}")
print("=" * 92)
