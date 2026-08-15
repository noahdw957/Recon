# RECON SAMPLE-C 67 PRE-EVENT TRAJECTORY STUDY
# Corrected deployment geometry: MD11 AND corrected MD8.
# Reconstructs T-10..T-1 price paths for the exact 67 historical C intersection events.

from pathlib import Path
import math
import time
import json
import numpy as np
import pandas as pd
import yfinance as yf

MD11_TH = 2.1954452583448045
MD8_TH = 1.9463204147913817

# Corrected frozen MD8 model copied from the validated production scanner.
# IMPORTANT: peer centers are in log10(RAW) space, not log10(signed_log1p(x)).
MODEL8 = {
    "features": [
        "prior_response_count_60d_adj",
        "log10_market_cap_before",
        "prior_abs_award_max",
        "relative_strength_spy_120d",
        "prior_abs_award_median_adj",
        "prior_response_count_60d",
        "prior_abs_award_median",
        "relative_strength_spy_60d",
    ],
    "threshold": 1.9463204147913817,
    "A_sector_count_centers": {
        "AERO_DEFENSE": 1.6989700043360187,
        "INDUSTRIAL": 1.4621397262132765,
        "OTHER": 1.6483333918242467,
        "TECH_SERVICES": 1.6433396112263248,
    },
    "A_sector_median_award_centers": {
        "AERO_DEFENSE": 7.276858522781777,
        "INDUSTRIAL": 6.379214783094319,
        "OTHER": 7.050908632424625,
        "TECH_SERVICES": 6.864458778006966,
    },
    "impute_median": {
        "prior_response_count_60d_adj": 0.0492180226701817,
        "log10_market_cap_before": 10.387166780894209,
        "prior_abs_award_max": 19.354947011752447,
        "relative_strength_spy_120d": 1.843991900460896,
        "prior_abs_award_median_adj": -0.0066639210311842,
        "prior_response_count_60d": 3.80666248977032,
        "prior_abs_award_median": 16.5322049303338,
        "relative_strength_spy_60d": 1.2132923469633017,
    },
    "reference_mean": {
        "prior_response_count_60d_adj": 0.0013767411374706754,
        "log10_market_cap_before": 10.625932783789732,
        "prior_abs_award_max": 19.597688495171163,
        "relative_strength_spy_120d": 1.2359990062272446,
        "prior_abs_award_median_adj": -0.04354096940216947,
        "prior_response_count_60d": 2.8273422161532253,
        "prior_abs_award_median": 16.43449652376031,
        "relative_strength_spy_60d": 2.623598721446692,
    },
    "reference_sd": {
        "prior_response_count_60d_adj": 0.2411181730720844,
        "log10_market_cap_before": 0.5075506908228757,
        "prior_abs_award_max": 1.1421831854022753,
        "relative_strength_spy_120d": 21.180105648166858,
        "prior_abs_award_median_adj": 0.314659542161833,
        "prior_response_count_60d": 1.7908841319714495,
        "prior_abs_award_median": 0.8870167161218723,
        "relative_strength_spy_60d": 14.497162681485262,
    },
    "lw_location": [
        0.0, 1.4157430449355379e-15, 1.3177985575500604e-15,
        -3.5616177231082714e-17, -7.123235446216543e-17,
        -2.49313240617579e-16, 3.9177794954190986e-16, 0.0,
    ],
    "lw_precision": [
        [1.1245235410715015,0.11352024832397757,0.1240456298401465,-0.03193144952522353,-0.3991441548956096,-0.36490679685676564,0.013769655800668537,-0.05708349556794794],
        [0.11352024832397757,1.3770061528784387,-0.3114930412901252,-0.22521682050650216,-1.1568724382658144,-0.014832737514160942,0.9161846426492161,-0.0939101841889188],
        [0.12404562984014653,-0.3114930412901252,1.9965791456133024,-0.3451494320242806,0.2943141182659422,-1.0929572591554757,-1.1431114831222657,-0.398446057654768],
        [-0.031931449525223615,-0.22521682050650216,-0.3451494320242807,2.205304783545063,0.6514570223600256,0.6839549226950875,0.07099452031580906,-1.2606775178443608],
        [-0.3991441548956096,-1.1568724382658147,0.2943141182659423,0.6514570223600256,4.367772214507019,0.531126542035879,-3.263049460696854,0.06272408648398381],
        [-0.36490679685676564,-0.014832737514160936,-1.0929572591554757,0.6839549226950875,0.5311265420358788,1.93893598117823,0.5661053548215131,-0.10750370278020249],
        [0.013769655800668522,0.9161846426492162,-1.1431114831222657,0.07099452031580904,-3.263049460696854,0.5661053548215131,4.088226221596494,-0.033279115184692745],
        [-0.05708349556794795,-0.09391018418891882,-0.3984460576547681,-1.2606775178443608,0.06272408648398387,-0.10750370278020249,-0.033279115184692704,2.0325863172680054],
    ],
}

INPUT_CANDIDATES = [
    Path("sample_C_v2_validation_events.csv"),
    Path("sample_C_v2_validation_events.csv.txt"),
    Path("Recon/sample_C_v2_validation_events.csv"),
    Path("Recon/sample_C_v2_validation_events.csv.txt"),
]
OUT_RESULTS = Path("C67_trajectory_results.csv")
OUT_GROUPS = Path("C67_group_stats.csv")
OUT_SUMMARY = Path("C67_trajectory_summary.txt")
CACHE = Path("market_cache_C67")
CACHE.mkdir(exist_ok=True)


def find_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("sample_C_v2_validation_events.csv(.txt) not found")


def signed_log1p(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.sign(s) * np.log1p(np.abs(s))


def sector_group(ticker, company=""):
    t = str(ticker).upper().strip()
    c = str(company).upper()
    aero = {"AVAV","BA","KTOS","LHX","LMT","NOC","GD","HII","RTX","TXT","RKLB","SATL","RDW","BKSY","PLTR","LDOS","SAIC","BAH","VSEC","WWD"}
    industrial = {"GE","CAT","ETN","HON"}
    tech = {"IBM","ACN"}
    if t in aero: return "AERO_DEFENSE"
    if t in industrial: return "INDUSTRIAL"
    if t in tech: return "TECH_SERVICES"
    if any(w in c for w in ["AEROSPACE","AEROVIRONMENT","BOEING","KRATOS","DEFENSE","DEFENCE","DYNAMICS","INGALLS","RAYTHEON","LOCKHEED","NORTHROP","LEIDOS","BOOZ ALLEN","ROCKET LAB","SATELLITE","SPACE","VSE","WOODWARD"]):
        return "AERO_DEFENSE"
    return "OTHER"


def score_md8(df):
    d = df.copy()
    d["sector_group"] = [sector_group(t,c) for t,c in zip(d["ticker"], d["company"])]
    raw_count = pd.to_numeric(d["prior_response_count_60d"], errors="coerce")
    raw_med = pd.to_numeric(d["prior_abs_award_median"], errors="coerce")
    d["prior_response_count_60d_adj"] = np.where(raw_count > 0, np.log10(raw_count), np.nan) - d["sector_group"].map(MODEL8["A_sector_count_centers"])
    d["prior_abs_award_median_adj"] = np.where(raw_med > 0, np.log10(raw_med), np.nan) - d["sector_group"].map(MODEL8["A_sector_median_award_centers"])

    # Sample-C stored this as natural log(market cap). Convert to log10.
    d["log10_market_cap_before"] = pd.to_numeric(d["log_market_cap_preaward"], errors="coerce") / np.log(10.0)

    X = pd.DataFrame(index=d.index)
    X["prior_response_count_60d_adj"] = d["prior_response_count_60d_adj"]
    X["log10_market_cap_before"] = d["log10_market_cap_before"]
    X["prior_abs_award_max"] = signed_log1p(d["prior_abs_award_max"])
    X["relative_strength_spy_120d"] = pd.to_numeric(d["relative_strength_spy_120d"], errors="coerce")
    X["prior_abs_award_median_adj"] = d["prior_abs_award_median_adj"]
    X["prior_response_count_60d"] = signed_log1p(d["prior_response_count_60d"])
    X["prior_abs_award_median"] = signed_log1p(d["prior_abs_award_median"])
    X["relative_strength_spy_60d"] = pd.to_numeric(d["relative_strength_spy_60d"], errors="coerce")
    X = X[MODEL8["features"]]

    med = pd.Series(MODEL8["impute_median"])[MODEL8["features"]]
    mu = pd.Series(MODEL8["reference_mean"])[MODEL8["features"]]
    sd = pd.Series(MODEL8["reference_sd"])[MODEL8["features"]]
    loc = np.array(MODEL8["lw_location"], dtype=float)
    prec = np.array(MODEL8["lw_precision"], dtype=float)
    z = (X.fillna(med) - mu) / sd
    delta = z.values - loc
    md2 = np.einsum("ij,jk,ik->i", delta, prec, delta) / 8.0
    return np.sqrt(np.maximum(md2, 1e-12))


def normalize_yf(raw):
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["Date","Close"])
    x = raw.copy()
    if isinstance(x.columns, pd.MultiIndex):
        close = x["Close"]
        if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
        out = pd.DataFrame({"Close": close})
    else:
        out = pd.DataFrame({"Close": x["Close"]})
    out = out.dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.reset_index().rename(columns={out.reset_index().columns[0]: "Date"}) if False else out.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out[["Date","Close"]]


def get_market(ticker, start, end):
    p = CACHE / f"{ticker}.csv"
    if p.exists():
        try:
            d = pd.read_csv(p, parse_dates=["Date"])
            if len(d) and d["Date"].min() <= start and d["Date"].max() >= end - pd.Timedelta(days=3):
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
    if len(d): d.to_csv(p, index=False)
    time.sleep(0.15)
    return d


def pct(a,b):
    return np.nan if pd.isna(a) or pd.isna(b) or a == 0 else (b/a - 1.0) * 100.0


# -----------------------------
# 1) Reproduce the exact C67
# -----------------------------
infile = find_input()
df = pd.read_csv(infile)
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df["award_date"] = pd.to_datetime(df["award_date"])
df["MD8_corrected"] = score_md8(df)
mask = (pd.to_numeric(df["MTS_signal_score"], errors="coerce") >= MD11_TH) & (df["MD8_corrected"] >= MD8_TH)
C67 = df.loc[mask].copy().reset_index(drop=True)

# Hard validation. Do NOT continue if geometry does not reproduce the paper.
if len(C67) != 67:
    raise RuntimeError(f"STOP: expected exact 67 intersection events, reproduced {len(C67)}")
med_peak = float(pd.to_numeric(C67["peak_pct"], errors="coerce").median())
hit10 = float((pd.to_numeric(C67["peak_pct"], errors="coerce") >= 10).mean())
hit20 = float((pd.to_numeric(C67["peak_pct"], errors="coerce") >= 20).mean())
if abs(med_peak - 48.04) > 0.10 or abs(hit10 - 64/67) > 1e-9 or abs(hit20 - 55/67) > 1e-9:
    raise RuntimeError(f"STOP: C67 performance parity failed: median={med_peak:.4f}, hit10={hit10:.6f}, hit20={hit20:.6f}")

print(f"C67 parity PASS: n=67, median peak={med_peak:.2f}%, >=10={hit10:.1%}, >=20={hit20:.1%}")

# -----------------------------
# 2) Pull T-10 .. T-1 prices
# -----------------------------
rows = []
for n, r in C67.iterrows():
    ticker = r["ticker"]
    ad = pd.Timestamp(r["award_date"])
    print(f"[{n+1:02d}/67] {ticker} {ad.date()}")
    mkt = get_market(ticker, ad - pd.Timedelta(days=40), ad + pd.Timedelta(days=2))
    prior = mkt[mkt["Date"] < ad].sort_values("Date").tail(10).copy()
    rec = {
        "ticker": ticker,
        "award_date": ad.date().isoformat(),
        "company": r.get("company", ""),
        "MD11": float(r["MTS_signal_score"]),
        "MD8": float(r["MD8_corrected"]),
        "peak_pct_90d": float(r["peak_pct"]),
        "rs60_at_event": float(r["relative_strength_spy_60d"]) if pd.notna(r["relative_strength_spy_60d"]) else np.nan,
        "rs120_at_event": float(r["relative_strength_spy_120d"]) if pd.notna(r["relative_strength_spy_120d"]) else np.nan,
        "pre_sessions_found": int(len(prior)),
    }
    if len(prior) < 10:
        rec.update({"pre10_return_pct": np.nan, "pre_valley_day": np.nan, "rebound_from_valley_pct": np.nan,
                    "valley_reversal_3pct": 0, "valley_reversal_5pct": 0, "t10_zone_reversal_3pct": 0})
        rows.append(rec)
        continue

    closes = prior["Close"].astype(float).to_numpy()
    # Label oldest session T-10, newest T-1.
    for j, val in enumerate(closes):
        rec[f"close_T{j-10:+d}"] = float(val)
    valley_i = int(np.argmin(closes))
    valley_day = valley_i - 10
    valley_px = float(closes[valley_i])
    t1 = float(closes[-1])
    pre10_ret = pct(float(closes[0]), t1)
    rebound = pct(valley_px, t1)
    # Linear slope in % of T-10 close per trading session.
    y = (closes / closes[0] - 1.0) * 100.0
    slope10 = float(np.polyfit(np.arange(10), y, 1)[0])
    y5 = y[-5:]
    slope5 = float(np.polyfit(np.arange(5), y5, 1)[0])

    rec.update({
        "pre10_return_pct": pre10_ret,
        "pre_slope10_pct_per_day": slope10,
        "pre_slope5_pct_per_day": slope5,
        "pre_valley_day": valley_day,
        "pre_valley_close": valley_px,
        "rebound_from_valley_pct": rebound,
        # Transparent, predeclared descriptive rules; all raw metrics are also saved.
        "valley_reversal_3pct": int(valley_day <= -3 and rebound >= 3.0 and slope5 > 0),
        "valley_reversal_5pct": int(valley_day <= -3 and rebound >= 5.0 and slope5 > 0),
        "t10_zone_reversal_3pct": int(-10 <= valley_day <= -7 and rebound >= 3.0 and slope5 > 0),
    })
    rows.append(rec)

R = pd.DataFrame(rows)
R.to_csv(OUT_RESULTS, index=False)

# -----------------------------
# 3) Group statistics
# -----------------------------
def group_stats(name, mask):
    g = R.loc[mask & R["peak_pct_90d"].notna()].copy()
    return {
        "group": name,
        "n": len(g),
        "median_pre10_return_pct": g["pre10_return_pct"].median(),
        "median_rebound_from_valley_pct": g["rebound_from_valley_pct"].median(),
        "hit_20pct": (g["peak_pct_90d"] >= 20).mean() if len(g) else np.nan,
        "hit_50pct": (g["peak_pct_90d"] >= 50).mean() if len(g) else np.nan,
        "hit_100pct": (g["peak_pct_90d"] >= 100).mean() if len(g) else np.nan,
        "median_peak_pct": g["peak_pct_90d"].median() if len(g) else np.nan,
        "mean_peak_pct": g["peak_pct_90d"].mean() if len(g) else np.nan,
    }

complete = R["pre_sessions_found"] >= 10
groups = [group_stats("ALL_COMPLETE", complete)]
for col in ["valley_reversal_3pct","valley_reversal_5pct","t10_zone_reversal_3pct"]:
    groups.append(group_stats(col + "_YES", complete & (R[col] == 1)))
    groups.append(group_stats(col + "_NO", complete & (R[col] == 0)))
G = pd.DataFrame(groups)
G.to_csv(OUT_GROUPS, index=False)

# -----------------------------
# 4) Human-readable summary
# -----------------------------
lines = []
lines.append("RECON C67 PRE-EVENT TRAJECTORY STUDY")
lines.append("====================================")
lines.append(f"Exact C intersection events reproduced: {len(C67)}")
lines.append(f"C67 parity: >=10% {64}/67 = {64/67:.1%}; >=20% {55}/67 = {55/67:.1%}; median peak {med_peak:.2f}%")
lines.append(f"Events with full T-10..T-1 price window: {int(complete.sum())}/67")
lines.append("")
for col, label in [
    ("valley_reversal_3pct", "Valley then >=3% rebound with positive final-5 slope"),
    ("valley_reversal_5pct", "Valley then >=5% rebound with positive final-5 slope"),
    ("t10_zone_reversal_3pct", "Valley in T-10..T-7 zone then >=3% rebound"),
]:
    n_yes = int((complete & (R[col] == 1)).sum())
    lines.append(f"{label}: {n_yes}/{int(complete.sum())} = {n_yes/max(1,int(complete.sum())):.1%}")
lines.append("")
lines.append("GROUP OUTCOMES")
lines.append(G.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
lines.append("")
lines.append("Interpretation note: these are descriptive post-hoc trajectory classifications, not a newly frozen trading rule. Raw T-10..T-1 closes and continuous metrics are saved for later Taguchi/SELL work.")
OUT_SUMMARY.write_text("\n".join(lines) + "\n")
print("\n" + OUT_SUMMARY.read_text())
print("Saved:", OUT_RESULTS, OUT_GROUPS, OUT_SUMMARY)
