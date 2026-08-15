# RECON C67 PRE-EVENT TRAJECTORY STUDY
# Rebuild the exact Sample-C MD11 AND MD8 intersection population (67 events),
# then measure the 10 trading days BEFORE each award.
#
# Outputs:
#   c67_trajectories.csv   event-level T-10..T-1 trajectory metrics
#   c67_summary.csv        grouped outcome table
#   c67_summary.txt        readable report
#
# Uses the same frozen thresholds:
#   MD11 >= 2.1954452583448045
#   MD8  >= 1.9463204147913817

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

MD11_TH = 2.1954452583448045
MD8_TH = 1.9463204147913817

C_CANDIDATES = [
    Path("sample_C_v2_validation_events.csv"),
    Path("sample_C_v2_validation_events.csv.txt"),
    Path("Recon/sample_C_v2_validation_events.csv"),
    Path("Recon/sample_C_v2_validation_events.csv.txt"),
]
MODEL8_CANDIDATES = [
    Path("scale8_frozen_A.json"),
    Path("Recon/scale8_frozen_A.json"),
    Path("b8a_frozen_A.json"),
    Path("b8a_frozen_A.json.txt"),
    Path("Recon/b8a_frozen_A.json"),
    Path("Recon/b8a_frozen_A.json.txt"),
]

CACHE = Path("market_cache_C67")
OUT_EVENTS = Path("c67_trajectories.csv")
OUT_SUMMARY = Path("c67_summary.csv")
OUT_TEXT = Path("c67_summary.txt")
CACHE.mkdir(exist_ok=True)

AERO = {
    "AVAV","BA","KTOS","LHX","LMT","NOC","GD","HII","RTX","TXT","RKLB","SATL",
    "RDW","BKSY","PLTR","LDOS","SAIC","BAH","VSEC","WWD","ONDS"
}
INDUSTRIAL = {"GE","CAT","ETN","HON"}
TECH = {"IBM","ACN"}

def find_existing(candidates, label):
    for p in candidates:
        if p.exists():
            print(f"{label}: {p}")
            return p
    raise FileNotFoundError(f"{label} not found. Tried: {candidates}")

def signed_log1p(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.sign(s) * np.log1p(np.abs(s))

def sector_group(ticker, company=""):
    t = str(ticker).upper().strip()
    c = str(company).upper()
    if t in AERO:
        return "AERO_DEFENSE"
    if t in INDUSTRIAL:
        return "INDUSTRIAL"
    if t in TECH:
        return "TECH_SERVICES"
    if any(w in c for w in [
        "AEROSPACE","AEROVIRONMENT","BOEING","KRATOS","DEFENSE","DEFENCE",
        "DYNAMICS","INGALLS","RAYTHEON","LOCKHEED","NORTHROP","LEIDOS",
        "BOOZ ALLEN","ROCKET LAB","SATELLITE","SPACE","VSE","WOODWARD",
        "ONDAS"
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
    out = out.dropna(subset=["Close"])
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out[["Date","Close"]].sort_values("Date").reset_index(drop=True)

def load_market(ticker, start, end, attempts=4):
    p = CACHE / f"{ticker}.csv"
    if p.exists():
        try:
            d = pd.read_csv(p, parse_dates=["Date"])
            if len(d):
                have_start = d["Date"].min()
                have_end = d["Date"].max()
                if have_start <= start and have_end >= end - pd.Timedelta(days=3):
                    return d.sort_values("Date").reset_index(drop=True)
        except Exception:
            pass

    last_exc = None
    for k in range(attempts):
        try:
            raw = yf.download(
                ticker,
                start=(start - pd.Timedelta(days=10)).date().isoformat(),
                end=(end + pd.Timedelta(days=5)).date().isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            d = normalize_yf(raw)
            if len(d):
                d.to_csv(p, index=False)
                return d
        except Exception as exc:
            last_exc = exc
        wait = 5 * (k + 1)
        print(f"  {ticker}: retry {k+1}/{attempts} after {wait}s")
        time.sleep(wait)

    print(f"  WARNING {ticker}: no market data ({last_exc})")
    return pd.DataFrame(columns=["Date","Close"])

def add_md8(C, model):
    F = model["features"]
    out = C.copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["sector_group"] = [
        sector_group(t, c) for t, c in zip(out["ticker"], out["company"].fillna(""))
    ]

    # C stored market cap as natural log. Convert ln(market cap) -> log10(market cap).
    ln_mcap = pd.to_numeric(out["log_market_cap_preaward"], errors="coerce")
    out["log10_market_cap_before"] = ln_mcap / math.log(10.0)

    raw_count = pd.to_numeric(out["prior_response_count_60d"], errors="coerce")
    raw_med = pd.to_numeric(out["prior_abs_award_median"], errors="coerce")

    count_cent = model["A_sector_count_centers"]
    award_cent = model["A_sector_median_award_centers"]

    out["prior_response_count_60d_adj"] = (
        np.where(raw_count > 0, np.log10(raw_count), np.nan)
        - out["sector_group"].map(count_cent)
    )
    out["prior_abs_award_median_adj"] = (
        np.where(raw_med > 0, np.log10(raw_med), np.nan)
        - out["sector_group"].map(award_cent)
    )

    X = pd.DataFrame(index=out.index)
    X["prior_response_count_60d_adj"] = out["prior_response_count_60d_adj"]
    X["log10_market_cap_before"] = out["log10_market_cap_before"]
    X["prior_abs_award_max"] = signed_log1p(out["prior_abs_award_max"])
    X["relative_strength_spy_120d"] = pd.to_numeric(out["relative_strength_spy_120d"], errors="coerce")
    X["prior_abs_award_median_adj"] = out["prior_abs_award_median_adj"]
    X["prior_response_count_60d"] = signed_log1p(out["prior_response_count_60d"])
    X["prior_abs_award_median"] = signed_log1p(out["prior_abs_award_median"])
    X["relative_strength_spy_60d"] = pd.to_numeric(out["relative_strength_spy_60d"], errors="coerce")
    X = X[F]

    med = pd.Series(model["impute_median"])[F]
    mu = pd.Series(model["reference_mean"])[F]
    sd = pd.Series(model["reference_sd"])[F]
    loc = np.asarray(model.get("lw_location", model.get("ledoit_location")), dtype=float)
    prec = np.asarray(model.get("lw_precision", model.get("ledoit_precision")), dtype=float)

    z = (X.fillna(med) - mu) / sd
    delta = z.values - loc
    md2 = np.einsum("ij,jk,ik->i", delta, prec, delta) / len(F)
    out["MD8"] = np.sqrt(np.maximum(md2, 1e-12))
    return out

def pre_window_metrics(mkt, award_date):
    d = pd.Timestamp(award_date)

    # EXACTLY historical definition: market observations strictly before award date.
    prior = mkt[mkt["Date"] < d].sort_values("Date").tail(10).copy()
    if len(prior) < 10:
        return None

    px = prior["Close"].astype(float).to_numpy()
    dates = prior["Date"].tolist()
    offsets = np.arange(-10, 0)

    trough_i = int(np.argmin(px))
    peak_i = int(np.argmax(px))
    trough_offset = int(offsets[trough_i])
    peak_offset = int(offsets[peak_i])

    t10 = float(px[0])
    t1 = float(px[-1])
    trough = float(px[trough_i])

    pre10_return = (t1 / t10 - 1.0) * 100.0
    rebound = (t1 / trough - 1.0) * 100.0

    # Regression slope expressed as % of mean price per trading session.
    x = np.arange(10, dtype=float)
    slope = float(np.polyfit(x, px, 1)[0])
    slope_pct_per_day = slope / float(np.mean(px)) * 100.0

    # "Early valley" mirrors what we observed live: trough in first half of T-10..T-1.
    early_valley = trough_offset <= -6

    # Mechanical reversal flags. We report more than one threshold rather than
    # pretending there is one magic definition.
    valley_reversal_2 = bool(early_valley and rebound >= 2.0 and t1 > trough)
    valley_reversal_5 = bool(early_valley and rebound >= 5.0 and t1 > trough)
    valley_reversal_10 = bool(early_valley and rebound >= 10.0 and t1 > trough)

    rec = {
        "tminus10_close": t10,
        "tminus1_close": t1,
        "pre10_return_pct": pre10_return,
        "pre10_slope_pct_per_day": slope_pct_per_day,
        "pre_trough_day": trough_offset,
        "pre_trough_close": trough,
        "trough_to_tminus1_pct": rebound,
        "pre_peak_day": peak_offset,
        "early_valley_T10_T6": int(early_valley),
        "valley_reversal_2pct": int(valley_reversal_2),
        "valley_reversal_5pct": int(valley_reversal_5),
        "valley_reversal_10pct": int(valley_reversal_10),
    }
    for off, val, dt in zip(offsets, px, dates):
        rec[f"close_m{abs(int(off))}"] = float(val)
        rec[f"date_m{abs(int(off))}"] = str(pd.Timestamp(dt).date())
    return rec

def group_stats(df, label, mask):
    g = df.loc[mask].copy()
    if not len(g):
        return {
            "group": label, "n": 0, "pct_of_67": 0.0,
            "hit20_pct": np.nan, "hit50_pct": np.nan, "hit100_pct": np.nan,
            "median_peak_pct": np.nan, "mean_peak_pct": np.nan,
            "median_pre10_return_pct": np.nan,
            "median_rebound_pct": np.nan,
        }
    return {
        "group": label,
        "n": int(len(g)),
        "pct_of_67": 100.0 * len(g) / len(df),
        "hit20_pct": 100.0 * (g["peak_pct"] >= 20).mean(),
        "hit50_pct": 100.0 * (g["peak_pct"] >= 50).mean(),
        "hit100_pct": 100.0 * (g["peak_pct"] >= 100).mean(),
        "median_peak_pct": float(g["peak_pct"].median()),
        "mean_peak_pct": float(g["peak_pct"].mean()),
        "median_pre10_return_pct": float(g["pre10_return_pct"].median()),
        "median_rebound_pct": float(g["trough_to_tminus1_pct"].median()),
    }

def main():
    c_path = find_existing(C_CANDIDATES, "Sample C validation")
    m_path = find_existing(MODEL8_CANDIDATES, "Frozen MD8 model")

    C = pd.read_csv(c_path)
    C["award_date"] = pd.to_datetime(C["award_date"])
    C["MTS_signal_score"] = pd.to_numeric(C["MTS_signal_score"], errors="coerce")
    C["peak_pct"] = pd.to_numeric(C["peak_pct"], errors="coerce")

    model = json.loads(m_path.read_text())
    scored = add_md8(C, model)

    inter = scored[
        (scored["MTS_signal_score"] >= MD11_TH)
        & (scored["MD8"] >= MD8_TH)
    ].copy().sort_values(["award_date","ticker"]).reset_index(drop=True)

    if len(inter) != 67:
        raise RuntimeError(
            f"PARITY FAILURE: expected 67 C intersection events, got {len(inter)}. "
            "Do not interpret trajectory results."
        )

    # Hard parity checks from the paper.
    hit10 = int((inter["peak_pct"] >= 10).sum())
    hit20 = int((inter["peak_pct"] >= 20).sum())
    med_peak = float(inter["peak_pct"].median())
    if hit10 != 64 or hit20 != 55 or abs(med_peak - 48.04) > 0.10:
        raise RuntimeError(
            f"OUTCOME PARITY FAILURE: got hit10={hit10}, hit20={hit20}, "
            f"median={med_peak:.4f}. Expected 64,55,~48.04."
        )

    print("=" * 72)
    print("C67 PARITY PASS")
    print(f"Intersection events : {len(inter)}")
    print(f">=10%               : {hit10}/67")
    print(f">=20%               : {hit20}/67")
    print(f"Median peak         : {med_peak:.2f}%")
    print("=" * 72)

    start = inter["award_date"].min() - pd.Timedelta(days=40)
    end = inter["award_date"].max() + pd.Timedelta(days=5)

    market = {}
    tickers = sorted(inter["ticker"].unique())
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {ticker}")
        market[ticker] = load_market(ticker, start, end)
        time.sleep(0.15)

    rows = []
    missing = []
    for _, r in inter.iterrows():
        rec = pre_window_metrics(market.get(r["ticker"], pd.DataFrame()), r["award_date"])
        if rec is None:
            missing.append(f"{r['ticker']} {r['award_date'].date()}")
            continue
        base = {
            "ticker": r["ticker"],
            "award_date": str(r["award_date"].date()),
            "company": r["company"],
            "transaction_amount_sum": r["transaction_amount_sum"],
            "transaction_amount_abs_sum": r["transaction_amount_abs_sum"],
            "MD11": r["MTS_signal_score"],
            "MD8": r["MD8"],
            "peak_pct": r["peak_pct"],
            "relative_strength_spy_60d": r["relative_strength_spy_60d"],
            "relative_strength_spy_120d": r["relative_strength_spy_120d"],
        }
        base.update(rec)
        rows.append(base)

    R = pd.DataFrame(rows)
    if len(R) != 67:
        raise RuntimeError(
            f"MARKET DATA INCOMPLETE: trajectories built for {len(R)}/67. "
            f"Missing: {', '.join(missing)}. Re-run later; no summary was published."
        )

    R["pre10_up"] = (R["pre10_return_pct"] > 0).astype(int)
    R["pre10_up_5pct"] = (R["pre10_return_pct"] >= 5).astype(int)
    R["pre10_up_10pct"] = (R["pre10_return_pct"] >= 10).astype(int)
    R.to_csv(OUT_EVENTS, index=False)

    specs = [
        ("ALL 67", pd.Series(True, index=R.index)),
        ("Pre T-10->T-1 UP", R["pre10_up"] == 1),
        ("Pre T-10->T-1 >= +5%", R["pre10_up_5pct"] == 1),
        ("Pre T-10->T-1 >= +10%", R["pre10_up_10pct"] == 1),
        ("Early trough T-10..T-6", R["early_valley_T10_T6"] == 1),
        ("Early valley + rebound >=2%", R["valley_reversal_2pct"] == 1),
        ("Early valley + rebound >=5%", R["valley_reversal_5pct"] == 1),
        ("Early valley + rebound >=10%", R["valley_reversal_10pct"] == 1),
        ("NOT early-valley+5%", R["valley_reversal_5pct"] == 0),
    ]
    S = pd.DataFrame([group_stats(R, label, mask) for label, mask in specs])
    S.to_csv(OUT_SUMMARY, index=False)

    # Trough-day distribution.
    dist = R["pre_trough_day"].value_counts().sort_index()
    lines = [
        "RECON SAMPLE C: PRE-EVENT TRAJECTORY OF THE 67 MD11 x MD8 SIGNALS",
        "=" * 72,
        f"Exact intersection population: 67 (parity PASS)",
        f"64/67 reached +10%; 55/67 reached +20%; median peak {R['peak_pct'].median():.2f}%",
        "",
        "KEY QUESTION: Did the winners commonly sit in a valley and start rising before T0?",
        "",
        "GROUP RESULTS",
        S.to_string(index=False, float_format=lambda x: f"{x:,.2f}"),
        "",
        "PRE-WINDOW TROUGH DAY DISTRIBUTION",
        dist.to_string(),
        "",
        "DEFINITIONS",
        "T-10..T-1 uses the 10 trading closes strictly BEFORE the award date.",
        "Early trough = minimum close occurs in T-10 through T-6.",
        "Valley reversal 5% = early trough AND rebound from that trough to T-1 >= 5%.",
        "The 2%, 5%, and 10% thresholds are all reported so the conclusion does not depend on one arbitrary cutoff.",
        "",
        "Files:",
        f"  {OUT_EVENTS}",
        f"  {OUT_SUMMARY}",
    ]
    OUT_TEXT.write_text("\n".join(lines) + "\n")

    print()
    print(OUT_TEXT.read_text())

if __name__ == "__main__":
    main()
