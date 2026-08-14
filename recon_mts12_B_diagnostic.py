# RECON 12-FACTOR MTS: FREEZE ON A, DIAGNOSTIC ON B
# Factor 12 selected by Sample-A L16: log10(market_cap_before)
#
# Inputs in repo root:
#   recon_L16_15factor_input_A.csv
#   sample_B_nonLMT_validation_events.csv
#
# Outputs:
#   recon_mts12_frozen_A.json
#   recon_mts12_A_thresholds.csv
#   recon_mts12_B_diagnostic.csv
#   recon_mts12_B_summary.json
#
# IMPORTANT:
#   - A alone determines reference space AND thresholds.
#   - B outcomes are used only after scores/predictions are frozen.
#   - LMT and zero-dollar rows are excluded.
#   - No Stage 2 / XGBoost.

import json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.covariance import LedoitWolf

A_FILE = Path("recon_L16_15factor_input_A.csv")
B_FILE = Path("sample_B_nonLMT_validation_events.csv")

OUT_MODEL = Path("recon_mts12_frozen_A.json")
OUT_THRESH = Path("recon_mts12_A_thresholds.csv")
OUT_B = Path("recon_mts12_B_diagnostic.csv")
OUT_SUM = Path("recon_mts12_B_summary.json")

CACHE = Path("mts12_market_cache")
CACHE.mkdir(exist_ok=True)

BASE11 = [
    "prior_response_count_60d",
    "pre_volatility_market_20d",
    "relative_strength_spy_120d",
    "prior_abs_award_max",
    "prior_signed_award_mean",
    "prior_abs_award_median",
    "prior_response_mean_20d",
    "transaction_amount_abs_sum",
    "prior_transactions_30d",
    "relative_strength_spy_60d",
    "prior_award_days_30d",
]
F12 = "log10_market_cap_before"
FEATURES = BASE11 + [F12]

TRANSFORMS = {
    "prior_response_count_60d": "signed_log1p",
    "pre_volatility_market_20d": "identity",
    "relative_strength_spy_120d": "identity",
    "prior_abs_award_max": "signed_log1p",
    "prior_signed_award_mean": "identity",
    "prior_abs_award_median": "signed_log1p",
    "prior_response_mean_20d": "identity",
    "transaction_amount_abs_sum": "signed_log1p",
    "prior_transactions_30d": "signed_log1p",
    "relative_strength_spy_60d": "identity",
    "prior_award_days_30d": "signed_log1p",
    "log10_market_cap_before": "identity",
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
        return pd.DataFrame(columns=["Date","Close"])
    d = raw.copy()
    if isinstance(d.columns, pd.MultiIndex):
        close = d["Close"]
        if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
        d = pd.DataFrame({"Close":close})
    else:
        d = pd.DataFrame({"Close":d["Close"]})
    d = d.dropna()
    d.index = pd.to_datetime(d.index)
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    d = d.reset_index()
    d = d.rename(columns={d.columns[0]:"Date"})
    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None)
    return d[["Date","Close"]]

def market_history(ticker, start, end):
    p = CACHE / f"{ticker}_prices.csv"
    if p.exists():
        try: return pd.read_csv(p, parse_dates=["Date"])
        except: pass
    raw = yf.download(ticker,
        start=(start-pd.Timedelta(days=10)).date().isoformat(),
        end=(end+pd.Timedelta(days=3)).date().isoformat(),
        auto_adjust=True, progress=False, threads=False)
    x = normalize_yf(raw)
    x.to_csv(p,index=False)
    time.sleep(.1)
    return x

def shares_history(ticker, start, end):
    p = CACHE / f"{ticker}_shares.csv"
    if p.exists():
        try: return pd.read_csv(p, parse_dates=["Date"])
        except: pass
    out = pd.DataFrame(columns=["Date","Shares"])
    try:
        s = yf.Ticker(ticker).get_shares_full(
            start=(start-pd.Timedelta(days=45)).date().isoformat(),
            end=(end+pd.Timedelta(days=3)).date().isoformat())
        if s is not None and len(s):
            idx = pd.to_datetime(s.index)
            if getattr(idx,"tz",None) is not None: idx=idx.tz_localize(None)
            out = pd.DataFrame({"Date":idx,"Shares":pd.to_numeric(s.values,errors="coerce")})
            out=out.dropna().sort_values("Date")
    except Exception as e:
        print("shares",ticker,e)
    out.to_csv(p,index=False)
    time.sleep(.1)
    return out

# -------------------- A: fit frozen 12D reference --------------------
A = pd.read_csv(A_FILE)
A["ticker"] = A["ticker"].astype(str).str.upper().str.strip()
A["peak_pct"] = pd.to_numeric(A["peak_pct"],errors="coerce")
A["transaction_amount_abs_sum"] = pd.to_numeric(A["transaction_amount_abs_sum"],errors="coerce")
A = A[(A.ticker!="LMT") & (A.transaction_amount_abs_sum>0) & A.peak_pct.notna()].copy()

missing=[c for c in FEATURES if c not in A.columns]
if missing: raise KeyError(f"A missing: {missing}")

XA = make_X(A)
normal = A["peak_pct"] < 10.0
ref = XA.loc[normal].copy()
med = ref.median().fillna(0.0)
ref = ref.fillna(med)
mu = ref.mean()
sd = ref.std(ddof=1).replace(0,1.0)
zr = (ref-mu)/sd
lw = LedoitWolf().fit(zr.values)

def score_matrix(X):
    x = X.copy().fillna(med)
    z=(x-mu)/sd
    d=z.values-lw.location_
    md2=np.einsum("ij,jk,ik->i",d,lw.precision_,d)/len(FEATURES)
    return np.sqrt(np.maximum(md2,1e-12))

A["MTS12_score"] = score_matrix(XA)
# Freeze several A-derived quantile thresholds, including the same top-5% principle
# used by the original deployable model. B never chooses a threshold.
fracs=[.20,.10,.05,.02,.01]
trs=[]
for f in fracs:
    th=float(A.MTS12_score.quantile(1-f))
    g=A[A.MTS12_score>=th]
    trs.append({
        "top_fraction":f,
        "threshold":th,
        "A_signals":int(len(g)),
        "A_hit_ge7p5":float((g.peak_pct>=7.5).mean()),
        "A_hit_ge10":float((g.peak_pct>=10).mean()),
        "A_hit_ge15":float((g.peak_pct>=15).mean()),
        "A_hit_ge20":float((g.peak_pct>=20).mean()),
        "A_median_peak":float(g.peak_pct.median()),
    })
T=pd.DataFrame(trs)
T.to_csv(OUT_THRESH,index=False)
TH=float(T.loc[np.isclose(T.top_fraction,.05),"threshold"].iloc[0])

model={
    "version":"RECON MTS12 Frozen A v1.0",
    "features":FEATURES,
    "factor12":F12,
    "transforms":TRANSFORMS,
    "training_rows":int(len(A)),
    "reference_rule":"Sample A peak_pct < 10",
    "reference_rows":int(normal.sum()),
    "impute_median":med.to_dict(),
    "reference_mean":mu.to_dict(),
    "reference_sd":sd.to_dict(),
    "ledoit_location":lw.location_.tolist(),
    "ledoit_precision":lw.precision_.tolist(),
    "dimension_normalization":12,
    "buy_rule":"MTS12_score >= Sample-A top-5% threshold",
    "top5_threshold":TH,
    "notes":["Factor 12 selected on A L16 only.","B outcomes not used to fit or select threshold.","No Stage 2."]
}
OUT_MODEL.write_text(json.dumps(model,indent=2))

# -------------------- B: add point-in-time market cap --------------------
B=pd.read_csv(B_FILE)
B["ticker"]=B["ticker"].astype(str).str.upper().str.strip()
B["award_date"]=pd.to_datetime(B["award_date"])
B["transaction_amount_abs_sum"]=pd.to_numeric(B["transaction_amount_abs_sum"],errors="coerce")
B["peak_pct"]=pd.to_numeric(B["peak_pct"],errors="coerce")
B=B[(B.ticker!="LMT") & (B.transaction_amount_abs_sum>0) & B.peak_pct.notna()].copy()

start=B.award_date.min()-pd.Timedelta(days=500)
end=B.award_date.max()+pd.Timedelta(days=5)
mcap=np.full(len(B),np.nan)

for n,(ticker,g) in enumerate(B.groupby("ticker"),1):
    print(f"[{n}/{B.ticker.nunique()}] {ticker}")
    px=market_history(ticker,start,end)
    sh=shares_history(ticker,start,end)
    for idx in g.index:
        d=B.at[idx,"award_date"]
        p=px[px.Date<d].sort_values("Date")
        s=sh[sh.Date<=d].sort_values("Date")
        if len(p) and len(s):
            close=float(p.iloc[-1].Close)
            shares=float(s.iloc[-1].Shares)
            if close>0 and shares>0:
                mcap[B.index.get_loc(idx)]=close*shares

B["market_cap_before"]=mcap
B[F12]=np.log10(B["market_cap_before"])
XB=make_X(B)
B["MTS12_score"]=score_matrix(XB)
B["MTS11_score_old"]=pd.to_numeric(B.get("MTS_signal_score"),errors="coerce")
B["old_buy"]=pd.to_numeric(B.get("predicted_buy"),errors="coerce").fillna(0).astype(int)
B["buy12"]=(B.MTS12_score>=TH).astype(int)
B["delta_score_12_minus_11"]=B.MTS12_score-B.MTS11_score_old

# Outcome flags are reporting only, AFTER frozen scoring.
for x in [7.5,10,15,20]:
    B[f"hit_{str(x).replace('.','p')}"]=(B.peak_pct>=x).astype(int)

B.to_csv(OUT_B,index=False)

def stats(mask):
    g=B.loc[mask]
    if not len(g): return {"n":0}
    return {
        "n":int(len(g)),
        "hit_ge7p5":float((g.peak_pct>=7.5).mean()),
        "hit_ge10":float((g.peak_pct>=10).mean()),
        "hit_ge15":float((g.peak_pct>=15).mean()),
        "hit_ge20":float((g.peak_pct>=20).mean()),
        "median_peak":float(g.peak_pct.median()),
        "mean_peak":float(g.peak_pct.mean()),
    }

old=B.old_buy==1
new=B.buy12==1
summary={
    "version":"RECON MTS12 B Diagnostic v1.0",
    "factor12":"log10(market_cap_before)",
    "A_top5_threshold_12D":TH,
    "B_rows":int(len(B)),
    "B_market_cap_coverage":float(B.market_cap_before.notna().mean()),
    "old_11_buy":stats(old),
    "new_12_buy":stats(new),
    "old_buy_rejected_by_12":stats(old & ~new),
    "new_12_added_vs_old":stats(~old & new),
    "known_failure_rows":[]
}
for ticker,date in [("AVAV","2026-01-26"),("BA","2025-08-12")]:
    q=B[(B.ticker==ticker)&(B.award_date==pd.Timestamp(date))]
    if len(q):
        r=q.iloc[0]
        summary["known_failure_rows"].append({
            "ticker":ticker,"award_date":date,
            "peak_pct":float(r.peak_pct),
            "old_MD":float(r.MTS11_score_old),
            "new_MD12":float(r.MTS12_score),
            "old_buy":int(r.old_buy),
            "new_buy12":int(r.buy12),
            "log10_market_cap_before":None if pd.isna(r[F12]) else float(r[F12]),
        })
# all KTOS old buys, since dates may matter
for _,r in B[(B.ticker=="KTOS")&(B.old_buy==1)].iterrows():
    summary["known_failure_rows"].append({
        "ticker":"KTOS","award_date":str(r.award_date.date()),
        "peak_pct":float(r.peak_pct),"old_MD":float(r.MTS11_score_old),
        "new_MD12":float(r.MTS12_score),"old_buy":1,"new_buy12":int(r.buy12),
        "log10_market_cap_before":None if pd.isna(r[F12]) else float(r[F12]),
    })

OUT_SUM.write_text(json.dumps(summary,indent=2))

print("\n"+"="*88)
print("RECON MTS12 A->B COMPLETE")
print("="*88)
print(f"A 12D top-5% threshold: {TH:.12f}")
print("OLD 11 BUY:",summary["old_11_buy"])
print("NEW 12 BUY:",summary["new_12_buy"])
print("OLD BUYS REJECTED BY 12:",summary["old_buy_rejected_by_12"])
print("\nKNOWN FAILURES:")
for r in summary["known_failure_rows"]: print(r)
print("\nFILES:")
for p in [OUT_MODEL,OUT_THRESH,OUT_B,OUT_SUM]: print(" ",p)
