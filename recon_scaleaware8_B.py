# RECON NEW SCALE-AWARE GANG-OF-8: FREEZE A -> TEST B
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

A_FILE = Path("recon_L16_scaleaware_v2_input_A.csv")
B_FILE = Path("sample_B_nonLMT_validation_events.csv")

OUT_MODEL = Path("recon_scaleaware8_frozen_A.json")
OUT_THRESH = Path("recon_scaleaware8_A_thresholds.csv")
OUT_B = Path("recon_scaleaware8_B_results.csv")
OUT_SUM = Path("recon_scaleaware8_B_summary.json")

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
    raise FileNotFoundError(f"{A_FILE} missing. Run the scale-aware L16 first.")
if not B_FILE.exists():
    raise FileNotFoundError(f"{B_FILE} missing from repo root.")

A = pd.read_csv(A_FILE)
B = pd.read_csv(B_FILE)

for d in (A,B):
    d["ticker"] = d["ticker"].astype(str).str.upper().str.strip()
    d["award_date"] = pd.to_datetime(d["award_date"])
    d["peak_pct"] = pd.to_numeric(d["peak_pct"], errors="coerce")

# A file already contains the exact transformed/engineered columns used in the L16.
# Reconstruct the same engineered columns for B from its raw fields plus point-in-time market cap.
# Prefer existing log10_market_cap_before if present. Otherwise use market_cap_before.
if "log10_market_cap_before" not in B.columns:
    if "market_cap_before" not in B.columns:
        raise KeyError("B needs log10_market_cap_before or market_cap_before.")
    B["log10_market_cap_before"] = np.log10(pd.to_numeric(B["market_cap_before"], errors="coerce"))

# Sector assignment MUST match the L16.
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

if "sector_group" not in A.columns:
    A["sector_group"] = A.apply(sector_group, axis=1)
B["sector_group"] = B.apply(sector_group, axis=1)

def log10pos(s):
    s = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    m = s > 0
    out.loc[m] = np.log10(s.loc[m])
    return out

# Freeze sector medians from A only.
A_raw_count = log10pos(A["prior_response_count_60d"]) if "prior_response_count_60d_adj" not in A.columns else None
A_raw_med = log10pos(A["prior_abs_award_median"]) if "prior_abs_award_median_adj" not in A.columns else None

# The L16 input has adjusted columns. Derive the A sector centers from raw values directly
# so B is centered using A information only.
A_count_log = log10pos(A["prior_response_count_60d"])
A_med_log = log10pos(A["prior_abs_award_median"])
count_centers = A_count_log.groupby(A["sector_group"]).median().to_dict()
med_centers = A_med_log.groupby(A["sector_group"]).median().to_dict()

B["prior_response_count_60d_adj"] = log10pos(B["prior_response_count_60d"]) - B["sector_group"].map(count_centers)
B["prior_abs_award_median_adj"] = log10pos(B["prior_abs_award_median"]) - B["sector_group"].map(med_centers)

# Raw variables in the L16 input were already transformed:
# prior_abs_award_max, prior_response_count_60d, prior_abs_award_median use signed_log1p.
# Build B to exactly that representation.
def signed_log1p(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.sign(s) * np.log1p(np.abs(s))

XB = pd.DataFrame(index=B.index)
XB["prior_response_count_60d_adj"] = B["prior_response_count_60d_adj"]
XB["log10_market_cap_before"] = pd.to_numeric(B["log10_market_cap_before"], errors="coerce")
XB["prior_abs_award_max"] = signed_log1p(B["prior_abs_award_max"])
XB["relative_strength_spy_120d"] = pd.to_numeric(B["relative_strength_spy_120d"], errors="coerce")
XB["prior_abs_award_median_adj"] = B["prior_abs_award_median_adj"]
XB["prior_response_count_60d"] = signed_log1p(B["prior_response_count_60d"])
XB["prior_abs_award_median"] = signed_log1p(B["prior_abs_award_median"])
XB["relative_strength_spy_60d"] = pd.to_numeric(B["relative_strength_spy_60d"], errors="coerce")

XA = A[FEATURES].apply(pd.to_numeric, errors="coerce").copy()

# Same reference-space rule as L16: <10% peak is normal.
normal = A["peak_pct"] < 10.0
ref = XA.loc[normal].copy()
med = ref.median().fillna(0.0)
ref = ref.fillna(med)
mu = ref.mean()
sd = ref.std(ddof=1).replace(0,1.0)
zr = (ref-mu)/sd
lw = LedoitWolf().fit(zr.values)

def score(X):
    x = X.copy().fillna(med)
    z = (x-mu)/sd
    d = z.values-lw.location_
    md2 = np.einsum("ij,jk,ik->i", d, lw.precision_, d)/len(FEATURES)
    return np.sqrt(np.maximum(md2,1e-12))

A["scaleaware8_MD"] = score(XA)
B["scaleaware8_MD"] = score(XB)

# Freeze threshold on A: top 5%, consistent with prior diagnostics.
TH = float(A["scaleaware8_MD"].quantile(.95))
B["scaleaware8_buy"] = (B["scaleaware8_MD"] >= TH).astype(int)

# Preserve old 11-factor comparison when available.
if "predicted_buy" in B.columns:
    B["old11_buy"] = pd.to_numeric(B["predicted_buy"], errors="coerce").fillna(0).astype(int)
else:
    B["old11_buy"] = 0

for x in [7.5,10,15,20]:
    B[f"hit_ge{x}"] = (B["peak_pct"] >= x).astype(int)

def stats(mask):
    g=B.loc[mask]
    if not len(g): return {"n":0}
    return {
        "n":int(len(g)),
        "ge7p5":int((g["peak_pct"]>=7.5).sum()),
        "ge10":int((g["peak_pct"]>=10).sum()),
        "ge15":int((g["peak_pct"]>=15).sum()),
        "ge20":int((g["peak_pct"]>=20).sum()),
        "rate_ge7p5":float((g["peak_pct"]>=7.5).mean()),
        "rate_ge10":float((g["peak_pct"]>=10).mean()),
        "rate_ge15":float((g["peak_pct"]>=15).mean()),
        "rate_ge20":float((g["peak_pct"]>=20).mean()),
        "median_peak":float(g["peak_pct"].median()),
        "mean_peak":float(g["peak_pct"].mean()),
    }

new=B["scaleaware8_buy"]==1
old=B["old11_buy"]==1

known=[]
for ticker,date in [("AVAV","2026-01-26"),("BA","2025-08-12"),("KTOS","2026-01-29")]:
    q=B[(B["ticker"]==ticker)&(B["award_date"]==pd.Timestamp(date))]
    if len(q):
        r=q.iloc[0]
        known.append({
            "ticker":ticker,"award_date":date,"peak_pct":float(r["peak_pct"]),
            "new_MD":float(r["scaleaware8_MD"]),"new_buy":int(r["scaleaware8_buy"]),
            "old11_buy":int(r["old11_buy"])
        })

summary={
    "version":"RECON New Scale-Aware Gang-of-8 B Test v1.0",
    "features":FEATURES,
    "A_rows":int(len(A)),
    "A_top5_threshold":TH,
    "B_rows":int(len(B)),
    "new_scaleaware8":stats(new),
    "old11_if_present":stats(old),
    "old11_kept":stats(old & new),
    "old11_rejected":stats(old & ~new),
    "new_added_vs_old11":stats(~old & new),
    "union_old11_or_new8":stats(old | new),
    "known_failures":known,
}
OUT_MODEL.write_text(json.dumps({
    "features":FEATURES,"threshold":TH,
    "A_sector_count_centers":count_centers,
    "A_sector_median_award_centers":med_centers,
    "impute_median":med.to_dict(),"reference_mean":mu.to_dict(),
    "reference_sd":sd.to_dict(),"lw_location":lw.location_.tolist(),
    "lw_precision":lw.precision_.tolist()
},indent=2))
pd.DataFrame([{"top_fraction":.05,"threshold":TH}]).to_csv(OUT_THRESH,index=False)
B.to_csv(OUT_B,index=False)
OUT_SUM.write_text(json.dumps(summary,indent=2))

print("="*88)
print("NEW SCALE-AWARE GANG-OF-8 -> SAMPLE B")
print("="*88)
print("Threshold:",TH)
print("NEW 8:",summary["new_scaleaware8"])
print("OLD 11:",summary["old11_if_present"])
print("OLD 11 REJECTED:",summary["old11_rejected"])
print("NEW ADDED:",summary["new_added_vs_old11"])
print("UNION:",summary["union_old11_or_new8"])
print("KNOWN FAILURES:")
for r in known: print(r)
print("="*88)
