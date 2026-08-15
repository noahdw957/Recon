# RECON Sample-C pre-award valley / run-up audit
# Identifies the exact frozen 67 MD11+MD8 Sample-C intersection events,
# downloads market history once per ticker, reconstructs T-10..T-1,
# and tests whether an early-window valley followed by a rebound is
# associated with larger post-award moves.
#
# Inputs expected in repo root (or Recon/):
#   sample_C_v2_validation_events.csv.txt (or .csv)
#   scale8_frozen_A.json
#
# Outputs:
#   C67_preaward_audit.csv
#   C67_preaward_summary.csv
#   C67_preaward_report.txt

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

MD11_TH = 2.1954452583448045

def find_file(names):
    for name in names:
        for p in [Path(name), Path("Recon") / name]:
            if p.exists():
                return p
    raise FileNotFoundError(f"Could not find any of: {names}")

C_FILE = find_file([
    "sample_C_v2_validation_events.csv.txt",
    "sample_C_v2_validation_events.csv",
])
M8_FILE = find_file(["scale8_frozen_A.json"])

OUT_ROWS = Path("C67_preaward_audit.csv")
OUT_SUM = Path("C67_preaward_summary.csv")
OUT_REPORT = Path("C67_preaward_report.txt")
CACHE = Path("c67_market_cache")
CACHE.mkdir(exist_ok=True)

def signed_log1p(x):
    x = pd.to_numeric(x, errors="coerce")
    return np.sign(x) * np.log1p(np.abs(x))

def sector_group(ticker, company=""):
    t = str(ticker).upper().strip()
    c = str(company).upper()
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

def normalize_yf(raw):
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["Date","Close"])
    d = raw.copy()
    if isinstance(d.columns, pd.MultiIndex):
        close = d["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        out = pd.DataFrame({"Close": close})
    else:
        out = pd.DataFrame({"Close": d["Close"]})
    out = out.dropna()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out[["Date","Close"]].sort_values("Date").reset_index(drop=True)

def get_market(ticker, start, end):
    p = CACHE / f"{ticker}.csv"
    if p.exists():
        try:
            d = pd.read_csv(p, parse_dates=["Date"])
            if len(d) and d["Date"].min() <= start and d["Date"].max() >= end - pd.Timedelta(days=5):
                return d
        except Exception:
            pass
    raw = yf.download(
        ticker,
        start=(start - pd.Timedelta(days=10)).date().isoformat(),
        end=(end + pd.Timedelta(days=10)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    d = normalize_yf(raw)
    d.to_csv(p, index=False)
    time.sleep(0.15)
    return d

# ------------------------------------------------------------
# Reproduce the exact frozen 67
# ------------------------------------------------------------
C = pd.read_csv(C_FILE)
C["award_date"] = pd.to_datetime(C["award_date"])
m = json.loads(M8_FILE.read_text())

C["sector_group"] = [
    sector_group(t, c) for t, c in zip(C["ticker"], C["company"])
]

raw_count = pd.to_numeric(C["prior_response_count_60d"], errors="coerce")
raw_med = pd.to_numeric(C["prior_abs_award_median"], errors="coerce")

X = pd.DataFrame(index=C.index)
X["prior_response_count_60d_adj"] = (
    np.where(raw_count > 0, np.log10(raw_count), np.nan)
    - C["sector_group"].map(m["A_sector_count_centers"])
)
# Stored C field is natural log market cap; convert to log10.
X["log10_market_cap_before"] = (
    pd.to_numeric(C["log_market_cap_preaward"], errors="coerce") / np.log(10.0)
)
X["prior_abs_award_max"] = signed_log1p(C["prior_abs_award_max"])
X["relative_strength_spy_120d"] = pd.to_numeric(
    C["relative_strength_spy_120d"], errors="coerce"
)
X["prior_abs_award_median_adj"] = (
    np.where(raw_med > 0, np.log10(raw_med), np.nan)
    - C["sector_group"].map(m["A_sector_median_award_centers"])
)
X["prior_response_count_60d"] = signed_log1p(C["prior_response_count_60d"])
X["prior_abs_award_median"] = signed_log1p(C["prior_abs_award_median"])
X["relative_strength_spy_60d"] = pd.to_numeric(
    C["relative_strength_spy_60d"], errors="coerce"
)

features = m["features"]
X = X[features]
med = pd.Series(m["impute_median"])[features]
mu = pd.Series(m["reference_mean"])[features]
sd = pd.Series(m["reference_sd"])[features]
loc = np.array(m["lw_location"], dtype=float)
prec = np.array(m["lw_precision"], dtype=float)

z = (X.fillna(med) - mu) / sd
d = z.values - loc
md2 = np.einsum("ij,jk,ik->i", d, prec, d) / 8.0
C["MD8_replay"] = np.sqrt(np.maximum(md2, 1e-12))
C["intersection"] = (
    (pd.to_numeric(C["MTS_signal_score"], errors="coerce") >= MD11_TH)
    & (C["MD8_replay"] >= float(m["threshold"]))
)

K = C[C["intersection"]].copy().reset_index(drop=True)
if len(K) != 67:
    raise RuntimeError(f"Frozen replay failed: expected 67, got {len(K)}")

print("Frozen C intersection reproduced: 67")

# ------------------------------------------------------------
# Pull one market series per ticker spanning all its C67 events
# ------------------------------------------------------------
market = {}
for i, (ticker, g) in enumerate(K.groupby("ticker"), 1):
    start = g["award_date"].min() - pd.Timedelta(days=45)
    end = g["award_date"].max() + pd.Timedelta(days=5)
    print(f"[{i}/{K['ticker'].nunique()}] {ticker}: {start.date()} -> {end.date()}")
    market[ticker] = get_market(ticker, start, end)

# ------------------------------------------------------------
# Reconstruct T-10..T-1 and classify the shape
# ------------------------------------------------------------
rows = []

for _, r in K.iterrows():
    ticker = r["ticker"]
    award = pd.Timestamp(r["award_date"])
    px = market.get(ticker, pd.DataFrame())

    prior = px[px["Date"] < award].sort_values("Date").tail(11).copy()
    if len(prior) < 11:
        continue

    # Use the last 10 pre-award intervals, indexed -11..-1 closes.
    vals = prior["Close"].astype(float).to_numpy()
    dates = prior["Date"].to_numpy()

    # Relevant T-10..T-1 path: first close is anchor, remaining 10 are path.
    anchor = vals[0]
    path = vals[1:]
    path_dates = dates[1:]

    ret10 = (path[-1] / anchor - 1.0) * 100.0
    min_i = int(np.argmin(path))
    valley_price = float(path[min_i])
    valley_date = pd.Timestamp(path_dates[min_i])
    valley_day = -10 + min_i
    rebound = (path[-1] / valley_price - 1.0) * 100.0
    draw_to_valley = (valley_price / anchor - 1.0) * 100.0

    # Linear slope of the 10 pre-event closes in % of first path close/day.
    x = np.arange(10, dtype=float)
    slope = np.polyfit(x, path, 1)[0] / path[0] * 100.0

    # A deliberately simple, pre-declared "valley -> rise" definition:
    # valley occurs in first half of T-10..T-1 and stock rebounds >=3%
    # from that valley by T-1.
    early_valley_rebound = (valley_day <= -6) and (rebound >= 3.0)

    # Broader descriptive flags; these are NOT new BUY rules.
    rising_10d = ret10 > 3.0
    flat_10d = abs(ret10) <= 3.0
    falling_10d = ret10 < -3.0

    peak = float(r["peak_pct"])

    rows.append({
        "ticker": ticker,
        "award_date": award.date().isoformat(),
        "peak_pct": peak,
        "MD11": float(r["MTS_signal_score"]),
        "MD8": float(r["MD8_replay"]),
        "pre10_return_pct": ret10,
        "pre10_slope_pct_per_day": slope,
        "pre_window_valley_day": valley_day,
        "pre_window_valley_date": valley_date.date().isoformat(),
        "draw_anchor_to_valley_pct": draw_to_valley,
        "rebound_valley_to_Tminus1_pct": rebound,
        "early_valley_rebound": int(early_valley_rebound),
        "rising_10d": int(rising_10d),
        "flat_10d": int(flat_10d),
        "falling_10d": int(falling_10d),
        "hit20": int(peak >= 20),
        "hit50": int(peak >= 50),
        "hit100": int(peak >= 100),
    })

R = pd.DataFrame(rows)
R.to_csv(OUT_ROWS, index=False)

if len(R) != 67:
    print(f"WARNING: only {len(R)}/67 had complete T-10 histories")

def summarize(name, mask):
    g = R.loc[mask].copy()
    if not len(g):
        return {
            "group": name, "n": 0,
            "median_pre10": np.nan, "median_rebound": np.nan,
            "hit20": np.nan, "hit50": np.nan, "hit100": np.nan,
            "median_peak": np.nan, "mean_peak": np.nan,
        }
    return {
        "group": name,
        "n": len(g),
        "median_pre10": g["pre10_return_pct"].median(),
        "median_rebound": g["rebound_valley_to_Tminus1_pct"].median(),
        "hit20": (g["peak_pct"] >= 20).mean(),
        "hit50": (g["peak_pct"] >= 50).mean(),
        "hit100": (g["peak_pct"] >= 100).mean(),
        "median_peak": g["peak_pct"].median(),
        "mean_peak": g["peak_pct"].mean(),
    }

summary_rows = [
    summarize("ALL_C67", pd.Series(True, index=R.index)),
    summarize("EARLY_VALLEY_REBOUND", R["early_valley_rebound"] == 1),
    summarize("NOT_EARLY_VALLEY_REBOUND", R["early_valley_rebound"] == 0),
    summarize("RISING_10D", R["rising_10d"] == 1),
    summarize("FLAT_10D", R["flat_10d"] == 1),
    summarize("FALLING_10D", R["falling_10d"] == 1),
]
S = pd.DataFrame(summary_rows)
S.to_csv(OUT_SUM, index=False)

# Quartiles of pre-event return, useful for the "monster move" question.
valid = R["pre10_return_pct"].notna()
if valid.sum() >= 8:
    R.loc[valid, "pre10_quartile"] = pd.qcut(
        R.loc[valid, "pre10_return_pct"],
        4,
        labels=["Q1 weakest","Q2","Q3","Q4 strongest"],
        duplicates="drop",
    )
    qrows = []
    for q, g in R.dropna(subset=["pre10_quartile"]).groupby("pre10_quartile", observed=True):
        qrows.append(summarize(str(q), R.index.isin(g.index)))
    Q = pd.DataFrame(qrows)
    Q.to_csv("C67_pre10_quartiles.csv", index=False)
else:
    Q = pd.DataFrame()

lines = []
lines.append("RECON C67 PRE-AWARD VALLEY / RUN-UP AUDIT")
lines.append("=" * 72)
lines.append(f"Exact frozen intersection events: 67")
lines.append(f"Complete T-10 histories: {len(R)}")
lines.append("")
for _, row in S.iterrows():
    if int(row["n"]) == 0:
        continue
    lines.append(
        f"{row['group']}: n={int(row['n'])} | "
        f">=20% {100*row['hit20']:.1f}% | "
        f">=50% {100*row['hit50']:.1f}% | "
        f">=100% {100*row['hit100']:.1f}% | "
        f"median peak {row['median_peak']:.2f}% | "
        f"median pre10 {row['median_pre10']:+.2f}% | "
        f"median valley rebound {row['median_rebound']:+.2f}%"
    )
if len(Q):
    lines.append("")
    lines.append("PRE-EVENT RETURN QUARTILES")
    for _, row in Q.iterrows():
        lines.append(
            f"{row['group']}: n={int(row['n'])} | "
            f">=20% {100*row['hit20']:.1f}% | "
            f">=50% {100*row['hit50']:.1f}% | "
            f">=100% {100*row['hit100']:.1f}% | "
            f"median peak {row['median_peak']:.2f}%"
        )

OUT_REPORT.write_text("\n".join(lines) + "\n")
print()
print(OUT_REPORT.read_text())
print("Wrote:", OUT_ROWS, OUT_SUM, OUT_REPORT)
