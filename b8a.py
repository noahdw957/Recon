# RECON NEW SCALE-AWARE GANG-OF-8: FREEZE A -> TEST B
# Corrected V1.2 ("b8a"): fixes sector-baseline reconstruction for B.

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

A_FILE = Path("recon_L16_scaleaware_v2_input_A.csv")
B_FILE = Path("sample_B_nonLMT_validation_events.csv")

OUT_MODEL = Path("b8a_frozen_A.json")
OUT_THRESH = Path("b8a_A_thresholds.csv")
OUT_B = Path("b8a_B_results.csv")
OUT_SUM = Path("b8a_B_summary.json")

CACHE = Path("b8a_market_cache")
CACHE.mkdir(exist_ok=True)

FEATURES = [
    "prior_response_count_60d_adj",
    "log10_market_cap_before",
    "prior_abs_award_max",
    "relative_strength_spy_120d",
    "prior_abs_award_median_adj",
    "prior_response_count_60d",
    "prior_abs_award_median",
    "relative_strength_spy_60d",
]

if not A_FILE.exists():
    raise FileNotFoundError(f"{A_FILE} missing.")
if not B_FILE.exists():
    raise FileNotFoundError(f"{B_FILE} missing.")

A = pd.read_csv(A_FILE)
B = pd.read_csv(B_FILE)

for d in (A, B):
    d["ticker"] = d["ticker"].astype(str).str.upper().str.strip()
    d["award_date"] = pd.to_datetime(d["award_date"])
    d["peak_pct"] = pd.to_numeric(d["peak_pct"], errors="coerce")

# ------------------------------------------------------------
# Point-in-time B market cap if absent
# ------------------------------------------------------------

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

if "log10_market_cap_before" not in B.columns:
    if "market_cap_before" in B.columns:
        B["log10_market_cap_before"] = np.log10(
            pd.to_numeric(B["market_cap_before"], errors="coerce")
        )
    else:
        print("Building point-in-time B market cap...")
        start = B["award_date"].min() - pd.Timedelta(days=500)
        end = B["award_date"].max() + pd.Timedelta(days=5)
        mcap = np.full(len(B), np.nan)

        for n, (ticker, g) in enumerate(B.groupby("ticker"), 1):
            print(f"[{n}/{B['ticker'].nunique()}] market cap: {ticker}")
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

coverage = float(B["log10_market_cap_before"].notna().mean())
print(f"B market-cap coverage: {coverage:.1%}")

# ------------------------------------------------------------
# Sector assignment - same as L16
# ------------------------------------------------------------

def sector_group(row):
    t = str(row["ticker"]).upper()
    c = str(row.get("company", "")).upper()

    aero = {
        "AVAV","BA","KTOS","LHX","LMT","NOC","GD","HII","RTX","TXT",
        "RKLB","SATL","RDW","BKSY","PLTR","LDOS","SAIC","BAH","VSEC","WWD"
    }
    industrial = {"GE","CAT","ETN","HON"}
    tech = {"IBM","ACN"}

    if t in aero:
        return "AERO_DEFENSE"
    if t in industrial:
        return "INDUSTRIAL"
    if t in tech:
        return "TECH_SERVICES"

    if any(w in c for w in [
        "AEROSPACE","AEROVIRONMENT","BOEING","KRATOS","DEFENSE","DEFENCE",
        "DYNAMICS","INGALLS","RAYTHEON","LOCKHEED","NORTHROP","LEIDOS",
        "BOOZ ALLEN","ROCKET LAB","SATELLITE","SPACE","VSE","WOODWARD"
    ]):
        return "AERO_DEFENSE"

    return "OTHER"

if "sector_group" not in A.columns:
    A["sector_group"] = A.apply(sector_group, axis=1)
B["sector_group"] = B.apply(sector_group, axis=1)

def log10pos(s):
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    m = s > 0
    out.loc[m] = np.log10(s.loc[m])
    return out

def signed_log1p(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.sign(s) * np.log1p(np.abs(s))

# ------------------------------------------------------------
# CORRECT A SECTOR BASELINE RECONSTRUCTION
# ------------------------------------------------------------
# In A:
#   adjusted = raw_log10 - sector_median_log10
# therefore:
#   sector_median_log10 = raw_log10 - adjusted
#
# Recover the sector baselines from the actual A data used by the L16,
# then apply those frozen A baselines to B.
# ------------------------------------------------------------

A_raw_count_log = log10pos(A["prior_response_count_60d"])
A_raw_med_log = log10pos(A["prior_abs_award_median"])

A_count_adj = pd.to_numeric(A["prior_response_count_60d_adj"], errors="coerce")
A_med_adj = pd.to_numeric(A["prior_abs_award_median_adj"], errors="coerce")

A["count_sector_baseline_recovered"] = A_raw_count_log - A_count_adj
A["median_award_sector_baseline_recovered"] = A_raw_med_log - A_med_adj

count_centers = (
    A.groupby("sector_group")["count_sector_baseline_recovered"]
    .median()
    .to_dict()
)
med_centers = (
    A.groupby("sector_group")["median_award_sector_baseline_recovered"]
    .median()
    .to_dict()
)

print("Recovered A sector count centers:", count_centers)
print("Recovered A sector median-award centers:", med_centers)

B["prior_response_count_60d_adj"] = (
    log10pos(B["prior_response_count_60d"])
    - B["sector_group"].map(count_centers)
)

B["prior_abs_award_median_adj"] = (
    log10pos(B["prior_abs_award_median"])
    - B["sector_group"].map(med_centers)
)

# ------------------------------------------------------------
# Build A/B matrices in exact L16 representation
# ------------------------------------------------------------

XA = A[FEATURES].apply(pd.to_numeric, errors="coerce").copy()

XB = pd.DataFrame(index=B.index)
XB["prior_response_count_60d_adj"] = B["prior_response_count_60d_adj"]
XB["log10_market_cap_before"] = pd.to_numeric(
    B["log10_market_cap_before"], errors="coerce"
)
XB["prior_abs_award_max"] = signed_log1p(B["prior_abs_award_max"])
XB["relative_strength_spy_120d"] = pd.to_numeric(
    B["relative_strength_spy_120d"], errors="coerce"
)
XB["prior_abs_award_median_adj"] = B["prior_abs_award_median_adj"]
XB["prior_response_count_60d"] = signed_log1p(B["prior_response_count_60d"])
XB["prior_abs_award_median"] = signed_log1p(B["prior_abs_award_median"])
XB["relative_strength_spy_60d"] = pd.to_numeric(
    B["relative_strength_spy_60d"], errors="coerce"
)

# ------------------------------------------------------------
# Freeze A reference space + top-5% threshold
# ------------------------------------------------------------

normal = A["peak_pct"] < 10.0
ref = XA.loc[normal].copy()

med = ref.median().fillna(0.0)
ref = ref.fillna(med)

mu = ref.mean()
sd = ref.std(ddof=1).replace(0, 1.0)

zr = (ref - mu) / sd
lw = LedoitWolf().fit(zr.values)

def score(X):
    x = X.copy().fillna(med)
    z = (x - mu) / sd
    d = z.values - lw.location_
    md2 = np.einsum("ij,jk,ik->i", d, lw.precision_, d) / len(FEATURES)
    return np.sqrt(np.maximum(md2, 1e-12))

A["b8a_MD"] = score(XA)
B["b8a_MD"] = score(XB)

TH = float(A["b8a_MD"].quantile(0.95))
B["b8a_buy"] = (B["b8a_MD"] >= TH).astype(int)

if "predicted_buy" in B.columns:
    B["old11_buy"] = pd.to_numeric(
        B["predicted_buy"], errors="coerce"
    ).fillna(0).astype(int)
else:
    B["old11_buy"] = 0

def stats(mask):
    g = B.loc[mask]
    if not len(g):
        return {"n": 0}
    return {
        "n": int(len(g)),
        "ge7p5": int((g["peak_pct"] >= 7.5).sum()),
        "ge10": int((g["peak_pct"] >= 10).sum()),
        "ge15": int((g["peak_pct"] >= 15).sum()),
        "ge20": int((g["peak_pct"] >= 20).sum()),
        "rate_ge7p5": float((g["peak_pct"] >= 7.5).mean()),
        "rate_ge10": float((g["peak_pct"] >= 10).mean()),
        "rate_ge15": float((g["peak_pct"] >= 15).mean()),
        "rate_ge20": float((g["peak_pct"] >= 20).mean()),
        "median_peak": float(g["peak_pct"].median()),
        "mean_peak": float(g["peak_pct"].mean()),
    }

new = B["b8a_buy"] == 1
old = B["old11_buy"] == 1

known = []
for ticker, date in [
    ("AVAV", "2026-01-26"),
    ("BA", "2025-08-12"),
    ("KTOS", "2026-01-29"),
]:
    q = B[
        (B["ticker"] == ticker)
        & (B["award_date"] == pd.Timestamp(date))
    ]
    if len(q):
        r = q.iloc[0]
        known.append({
            "ticker": ticker,
            "award_date": date,
            "peak_pct": float(r["peak_pct"]),
            "new_MD": float(r["b8a_MD"]),
            "new_buy": int(r["b8a_buy"]),
            "old11_buy": int(r["old11_buy"]),
            "sector_group": r["sector_group"],
            "prior_response_count_60d_adj": (
                None if pd.isna(r["prior_response_count_60d_adj"])
                else float(r["prior_response_count_60d_adj"])
            ),
            "prior_abs_award_median_adj": (
                None if pd.isna(r["prior_abs_award_median_adj"])
                else float(r["prior_abs_award_median_adj"])
            ),
        })

summary = {
    "version": "RECON b8a corrected scale-aware Gang-of-8 B Test v1.2",
    "features": FEATURES,
    "A_rows": int(len(A)),
    "A_top5_threshold": TH,
    "B_rows": int(len(B)),
    "B_market_cap_coverage": coverage,
    "recovered_A_sector_count_centers": count_centers,
    "recovered_A_sector_median_award_centers": med_centers,
    "new_b8a": stats(new),
    "old11_if_present": stats(old),
    "old11_kept": stats(old & new),
    "old11_rejected": stats(old & ~new),
    "new_added_vs_old11": stats(~old & new),
    "union_old11_or_b8a": stats(old | new),
    "known_failures": known,
}

OUT_MODEL.write_text(json.dumps({
    "features": FEATURES,
    "threshold": TH,
    "A_sector_count_centers": count_centers,
    "A_sector_median_award_centers": med_centers,
    "impute_median": med.to_dict(),
    "reference_mean": mu.to_dict(),
    "reference_sd": sd.to_dict(),
    "lw_location": lw.location_.tolist(),
    "lw_precision": lw.precision_.tolist(),
}, indent=2))

pd.DataFrame([{
    "top_fraction": 0.05,
    "threshold": TH,
}]).to_csv(OUT_THRESH, index=False)

B.to_csv(OUT_B, index=False)
OUT_SUM.write_text(json.dumps(summary, indent=2))

print("=" * 92)
print("B8A CORRECTED SCALE-AWARE GANG-OF-8 -> SAMPLE B")
print("=" * 92)
print("Threshold:", TH)
print("NEW B8A:", summary["new_b8a"])
print("OLD 11:", summary["old11_if_present"])
print("OLD 11 REJECTED:", summary["old11_rejected"])
print("NEW ADDED:", summary["new_added_vs_old11"])
print("UNION:", summary["union_old11_or_b8a"])
print("KNOWN FAILURES:")
for r in known:
    print(r)
print("=" * 92)
