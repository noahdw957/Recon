# RECON L16 SCALE-AWARE SCREEN V2.0
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

A_FILE = Path("recon_L16_15factor_input_A.csv")
OUT_INPUT = Path("recon_L16_scaleaware_v2_input_A.csv")
OUT_RUNS = Path("recon_L16_scaleaware_v2_runs_A.csv")
OUT_EFFECTS = Path("recon_L16_scaleaware_v2_effects_A.csv")
OUT_SUMMARY = Path("recon_L16_scaleaware_v2_summary.json")

if not A_FILE.exists():
    raise FileNotFoundError("recon_L16_15factor_input_A.csv not found in repo root")

df = pd.read_csv(A_FILE)
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
df["peak_pct"] = pd.to_numeric(df["peak_pct"], errors="coerce")
df["transaction_amount_abs_sum"] = pd.to_numeric(df["transaction_amount_abs_sum"], errors="coerce")
df = df[(df["ticker"]!="LMT") & (df["transaction_amount_abs_sum"]>0) & df["peak_pct"].notna()].copy().reset_index(drop=True)

gang8 = [
    "log10_market_cap_before","prior_abs_award_max","prior_response_count_60d",
    "prior_abs_award_median","pre_volatility_market_20d","prior_transactions_30d",
    "prior_response_mean_20d","relative_strength_spy_60d"
]
engineered = [
    "prior_abs_award_max_adj","prior_abs_award_median_adj",
    "prior_response_count_60d_adj","log10_mcap_x_prior_abs_award_max_adj"
]
resurrected = ["relative_strength_spy_120d","prior_signed_award_mean","transaction_amount_abs_sum"]
factors = gang8 + engineered + resurrected
assert len(factors)==15

def signed_log1p(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.sign(s)*np.log1p(np.abs(s))
def safe_log10_positive(s):
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan,index=s.index,dtype=float)
    m = s>0
    out.loc[m] = np.log10(s.loc[m])
    return out

X = pd.DataFrame(index=df.index)
X["log10_market_cap_before"] = pd.to_numeric(df["log10_market_cap_before"],errors="coerce")
X["prior_abs_award_max"] = signed_log1p(df["prior_abs_award_max"])
X["prior_response_count_60d"] = signed_log1p(df["prior_response_count_60d"])
X["prior_abs_award_median"] = signed_log1p(df["prior_abs_award_median"])
X["pre_volatility_market_20d"] = pd.to_numeric(df["pre_volatility_market_20d"],errors="coerce")
X["prior_transactions_30d"] = signed_log1p(df["prior_transactions_30d"])
X["prior_response_mean_20d"] = pd.to_numeric(df["prior_response_mean_20d"],errors="coerce")
X["relative_strength_spy_60d"] = pd.to_numeric(df["relative_strength_spy_60d"],errors="coerce")
X["relative_strength_spy_120d"] = pd.to_numeric(df["relative_strength_spy_120d"],errors="coerce")
X["prior_signed_award_mean"] = pd.to_numeric(df["prior_signed_award_mean"],errors="coerce")
X["transaction_amount_abs_sum"] = signed_log1p(df["transaction_amount_abs_sum"])

def sector_group(row):
    t = str(row["ticker"]).upper()
    c = str(row.get("company","")).upper()
    aero = {"AVAV","BA","KTOS","LHX","LMT","NOC","GD","HII","RTX","TXT","RKLB","SATL","RDW","BKSY","PLTR","LDOS","SAIC","BAH","VSEC","WWD"}
    industrial = {"GE","CAT","ETN","HON"}
    tech = {"IBM","ACN"}
    if t in aero: return "AERO_DEFENSE"
    if t in industrial: return "INDUSTRIAL"
    if t in tech: return "TECH_SERVICES"
    if any(w in c for w in ["AEROSPACE","AEROVIRONMENT","BOEING","KRATOS","DEFENSE","DEFENCE","DYNAMICS","INGALLS","RAYTHEON","LOCKHEED","NORTHROP","LEIDOS","BOOZ ALLEN","ROCKET LAB","SATELLITE","SPACE","VSE","WOODWARD"]):
        return "AERO_DEFENSE"
    return "OTHER"

df["sector_group"] = df.apply(sector_group,axis=1)
raw_log_max = safe_log10_positive(df["prior_abs_award_max"])
raw_log_med = safe_log10_positive(df["prior_abs_award_median"])
raw_log_count = safe_log10_positive(df["prior_response_count_60d"])
X["prior_abs_award_max_adj"] = raw_log_max - raw_log_max.groupby(df["sector_group"]).transform("median")
X["prior_abs_award_median_adj"] = raw_log_med - raw_log_med.groupby(df["sector_group"]).transform("median")
X["prior_response_count_60d_adj"] = raw_log_count - raw_log_count.groupby(df["sector_group"]).transform("median")
X["log10_mcap_x_prior_abs_award_max_adj"] = X["log10_market_cap_before"] * X["prior_abs_award_max_adj"]

peak = df["peak_pct"].to_numpy(float)
normal = peak < 10.0
y20 = peak >= 20.0
y50 = peak >= 50.0

def fold_md(tr,te,cols):
    ref = X.iloc[tr][cols].loc[normal[tr]].copy()
    test = X.iloc[te][cols].copy()
    med = ref.median().fillna(0.0)
    ref = ref.fillna(med); test = test.fillna(med)
    keep = ref.std(ddof=1) > 1e-12
    ref = ref.loc[:,keep]; test = test.loc[:,keep]
    if ref.shape[1]==0: return np.full(len(te),np.nan)
    mu = ref.mean(); sd = ref.std(ddof=1).replace(0,1.0)
    zr = (ref-mu)/sd; zt=(test-mu)/sd
    lw = LedoitWolf().fit(zr.values)
    d = zt.values - lw.location_
    md2 = np.einsum("ij,jk,ik->i",d,lw.precision_,d)/ref.shape[1]
    return np.sqrt(np.maximum(md2,1e-12))

def score(cols):
    oof = np.full(len(df),np.nan)
    cv = GroupKFold(n_splits=5)
    for tr,te in cv.split(X,y20,groups=df["ticker"]):
        oof[te] = fold_md(tr,te,cols)
    v = np.isfinite(oof)
    auc20 = roc_auc_score(y20[v],oof[v]); auc50 = roc_auc_score(y50[v],oof[v])
    rho = spearmanr(oof[v],peak[v]).statistic
    sig = oof[v & y20]
    sn = -10*np.log10(np.mean(1.0/np.maximum(sig**2,1e-12)))
    return {"scores":oof,"auc20":float(auc20),"auc50":float(auc50),"rho":float(rho),"sn":float(sn)}

H=hadamard(16); D=(H[:,1:16]>0).astype(int)
runs=[]
for run in range(16):
    active=[factors[j] for j in range(15) if D[run,j]==1]
    if not active:
        row={"run":run+1,"n_features":0,"sn":np.nan,"auc20":np.nan,"auc50":np.nan,"rho":np.nan}
    else:
        r=score(active)
        row={"run":run+1,"n_features":len(active),"sn":r["sn"],"auc20":r["auc20"],"auc50":r["auc50"],"rho":r["rho"]}
    for c in factors: row[c]=int(c in active)
    runs.append(row)
    print(f"L16 run {run+1:2d}/16 n={len(active):2d} SN={row['sn'] if np.isfinite(row['sn']) else float('nan'):.4f}")

rd=pd.DataFrame(runs)
effects=[]
for c in factors:
    on=rd.loc[(rd[c]==1)&rd.sn.notna(),"sn"]; off=rd.loc[(rd[c]==0)&rd.sn.notna(),"sn"]
    aon=rd.loc[(rd[c]==1)&rd.auc20.notna(),"auc20"]; aoff=rd.loc[(rd[c]==0)&rd.auc20.notna(),"auc20"]
    kind = "gang8" if c in gang8 else ("engineered" if c in engineered else "resurrected")
    effects.append({"variable":c,"kind":kind,"SN_on_dB":float(on.mean()),"SN_off_dB":float(off.mean()),"SN_effect_dB":float(on.mean()-off.mean()),"AUC20_effect":float(aon.mean()-aoff.mean())})

eff=pd.DataFrame(effects)
sq=eff["SN_effect_dB"]**2
eff["relative_SN_contribution_pct"]=100*sq/sq.sum()
eff=eff.sort_values(["SN_effect_dB","AUC20_effect"],ascending=False).reset_index(drop=True)
eff["rank_SN_effect"]=np.arange(1,len(eff)+1)
positive=eff.loc[eff.SN_effect_dB>0,"variable"].tolist()
pos_score=score(positive) if positive else None
full_score=score(factors)

out=df[["ticker","award_date","company","sector_group","peak_pct","transaction_amount_abs_sum"]].copy()
for c in factors: out[c]=X[c]
out.to_csv(OUT_INPUT,index=False); rd.to_csv(OUT_RUNS,index=False); eff.to_csv(OUT_EFFECTS,index=False)

summary={
"version":"RECON L16 Scale-Aware Screen V2.0","sample":"A only","rows_screened":int(len(df)),
"factor_roster":{"gang8":gang8,"engineered":engineered,"resurrected":resurrected},
"sector_groups":df["sector_group"].value_counts().to_dict(),
"ranked_effects":eff[["rank_SN_effect","variable","kind","SN_effect_dB","AUC20_effect","relative_SN_contribution_pct"]].to_dict("records"),
"positive_SN_effect_factors":positive,
"positive_effect_model":None if pos_score is None else {"sn_db":pos_score["sn"],"auc20":pos_score["auc20"],"auc50":pos_score["auc50"],"rho_peak":pos_score["rho"]},
"full_15":{"sn_db":full_score["sn"],"auc20":full_score["auc20"],"auc50":full_score["auc50"],"rho_peak":full_score["rho"]},
"notes":["No Sample B or C data are read.","All 15 L16 columns are occupied.","Sector-adjusted variables are centered against sector-group medians within Sample A.","Interaction factor is log10_market_cap_before * prior_abs_award_max_adj.","Resurrected factors chosen by original accepted-11 L16 S/N strengths."]
}
OUT_SUMMARY.write_text(json.dumps(summary,indent=2))
print("\nCOMPLETE")
print(eff[["rank_SN_effect","variable","kind","SN_effect_dB","AUC20_effect","relative_SN_contribution_pct"]].to_string(index=False))
