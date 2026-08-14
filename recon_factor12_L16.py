# RECON FACTOR-12 L16 SCREEN V1.0
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd, requests, yfinance as yf
from scipy.linalg import hadamard
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

A_FILE = next((p for p in [Path("mts_award_time_features_A.csv"),Path("Recon/mts_award_time_features_A.csv")] if p.exists()), None)
MODEL_FILE = next((p for p in [Path("mts_frozen_model_A.json"),Path("Recon/mts_frozen_model_A.json")] if p.exists()), None)
if A_FILE is None: raise FileNotFoundError("mts_award_time_features_A.csv not found")
if MODEL_FILE is None: raise FileNotFoundError("mts_frozen_model_A.json not found")

OUT_INPUT=Path("recon_L16_15factor_input_A.csv")
OUT_RUNS=Path("recon_L16_15factor_runs_A.csv")
OUT_EFFECTS=Path("recon_L16_15factor_effects_A.csv")
OUT_SUMMARY=Path("recon_L16_15factor_summary.json")
OUT_SCALE=Path("recon_scale_point_in_time_A.csv")
CACHE=Path("factor12_cache"); CACHE.mkdir(exist_ok=True)

SEC_HEADERS={"User-Agent":"RECON research contact recon-research@example.com","Accept-Encoding":"gzip, deflate"}
REVENUE_CONCEPT_PRIORITY=[
"RevenueFromContractWithCustomerExcludingAssessedTax",
"RevenueFromContractWithCustomerIncludingAssessedTax",
"Revenues","SalesRevenueNet","SalesRevenueGoodsNet","SalesRevenueServicesNet",
"RegulatedAndUnregulatedOperatingRevenue","OperatingRevenues"]

df=pd.read_csv(A_FILE)
model=json.loads(MODEL_FILE.read_text())
df["ticker"]=df["ticker"].astype(str).str.upper().str.strip()
df["award_date"]=pd.to_datetime(df["award_date"])
df["peak_pct"]=pd.to_numeric(df["peak_pct"],errors="coerce")
df["transaction_amount_abs_sum"]=pd.to_numeric(df["transaction_amount_abs_sum"],errors="coerce")
before_n=len(df)
df=df[(df["ticker"]!="LMT")&(df["transaction_amount_abs_sum"]>0)&df["peak_pct"].notna()].copy().reset_index(drop=True)
base11=list(model["features"])
assert len(base11)==11

def normalize_yf(data):
    if data is None or len(data)==0: return pd.DataFrame(columns=["Date","Close","Volume"])
    d=data.copy()
    if isinstance(d.columns,pd.MultiIndex):
        close=d["Close"]; vol=d["Volume"]
        if isinstance(close,pd.DataFrame): close=close.iloc[:,0]
        if isinstance(vol,pd.DataFrame): vol=vol.iloc[:,0]
        out=pd.DataFrame({"Close":close,"Volume":vol})
    else:
        out=d[["Close","Volume"]].copy()
    out=out.dropna(subset=["Close"])
    out.index=pd.to_datetime(out.index)
    if getattr(out.index,"tz",None) is not None: out.index=out.index.tz_localize(None)
    out=out.reset_index()
    out=out.rename(columns={out.columns[0]:"Date"})
    out["Date"]=pd.to_datetime(out["Date"]).dt.tz_localize(None)
    return out[["Date","Close","Volume"]]

def get_market(ticker,start,end):
    p=CACHE/f"{ticker}_market.csv"
    if p.exists():
        try:
            x=pd.read_csv(p,parse_dates=["Date"])
            if len(x): return x
        except: pass
    raw=yf.download(ticker,start=(start-pd.Timedelta(days=10)).date().isoformat(),
                    end=(end+pd.Timedelta(days=3)).date().isoformat(),
                    auto_adjust=True,progress=False,threads=False)
    x=normalize_yf(raw); x.to_csv(p,index=False); time.sleep(.12); return x

def get_shares(ticker,start,end):
    p=CACHE/f"{ticker}_shares.csv"
    if p.exists():
        try: return pd.read_csv(p,parse_dates=["Date"])
        except: pass
    out=pd.DataFrame(columns=["Date","Shares"])
    try:
        raw=yf.Ticker(ticker).get_shares_full(start=(start-pd.Timedelta(days=45)).date().isoformat(),
                                              end=(end+pd.Timedelta(days=3)).date().isoformat())
        if raw is not None and len(raw):
            idx=pd.to_datetime(raw.index)
            if getattr(idx,"tz",None) is not None: idx=idx.tz_localize(None)
            out=pd.DataFrame({"Date":idx,"Shares":pd.to_numeric(raw.values,errors="coerce")}).dropna().sort_values("Date")
    except Exception as e: print("shares",ticker,e)
    out.to_csv(p,index=False); time.sleep(.12); return out

def sec_get_json(url,cache_name):
    p=CACHE/cache_name
    if p.exists():
        try: return json.loads(p.read_text())
        except: pass
    r=requests.get(url,headers=SEC_HEADERS,timeout=45); r.raise_for_status()
    p.write_text(r.text); time.sleep(.12); return r.json()

def ticker_cik_map():
    data=sec_get_json("https://www.sec.gov/files/company_tickers.json","sec_company_tickers.json")
    return {str(v["ticker"]).upper():int(v["cik_str"]) for v in data.values()}

def companyfacts(cik):
    return sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",f"sec_companyfacts_{cik:010d}.json")

def choose_revenue_facts(cf):
    ug=cf.get("facts",{}).get("us-gaap",{})
    for c in REVENUE_CONCEPT_PRIORITY:
        if c in ug and "USD" in ug[c].get("units",{}): return c,ug[c]["units"]["USD"]
    for c,obj in ug.items():
        if "revenue" in str(obj.get("label","")).lower() and "USD" in obj.get("units",{}): return c,obj["units"]["USD"]
    return None,[]

def clean_facts(facts):
    rows=[]
    for f in facts:
        try:
            form=str(f.get("form","")).replace("/A","")
            if form not in ("10-Q","10-K") or not all(k in f for k in ("start","end","filed")): continue
            val=float(f["val"]); start=pd.Timestamp(f["start"]); end=pd.Timestamp(f["end"]); filed=pd.Timestamp(f["filed"])
            days=(end-start).days
            if val<0 or days<60 or days>410: continue
            rows.append({"start":start,"end":end,"filed":filed,"val":val,"form":form,"fp":str(f.get("fp","")),"days":days})
        except: pass
    return pd.DataFrame(rows).sort_values(["filed","end","start"]) if rows else pd.DataFrame()

def ltm_revenue_asof(x,asof):
    if x is None or x.empty: return np.nan,"no_revenue_facts",None
    known=x[x["filed"]<=asof].copy()
    annual=known[(known["form"]=="10-K")&(known["days"]>=300)].sort_values(["filed","end"])
    if annual.empty: return np.nan,"no_annual_before_award",None
    ann=annual.iloc[-1]; aval=float(ann["val"]); aend=ann["end"]
    q=known[(known["form"]=="10-Q")&(known["end"]>aend)&(known["days"]<=300)].sort_values(["filed","end"])
    if q.empty: return aval,"latest_10k",str(ann["filed"].date())
    cur=q.iloc[-1]
    target=cur["end"]-pd.DateOffset(years=1)
    prior=known[(known["form"]=="10-Q")&(np.abs((known["end"]-target).dt.days)<=20)&(np.abs(known["days"]-int(cur["days"]))<=20)].copy()
    if cur["fp"]:
        sf=prior[prior["fp"]==cur["fp"]]
        if not sf.empty: prior=sf
    if prior.empty: return aval,"latest_10k_no_comparable_ytd",str(ann["filed"].date())
    prv=prior.sort_values(["filed","end"]).iloc[-1]
    ltm=aval+float(cur["val"])-float(prv["val"])
    if not np.isfinite(ltm) or ltm<=0: return aval,"latest_10k_fallback_bad_ttm",str(ann["filed"].date())
    return ltm,"ttm_from_filed_statements",f"10K({ann['filed'].date()})+YTD({cur['filed'].date()})-PY_YTD({prv['filed'].date()})"

cikmap=ticker_cik_map()
start=df["award_date"].min()-pd.Timedelta(days=500); end=df["award_date"].max()+pd.Timedelta(days=5)
audit=[]
for k,(ticker,g) in enumerate(df.groupby("ticker"),1):
    print(f"[{k}/{df.ticker.nunique()}] {ticker}")
    m=get_market(ticker,start,end); s=get_shares(ticker,start,end)
    cik=cikmap.get(ticker); concept=None; revf=pd.DataFrame()
    if cik:
        try:
            concept,raw=choose_revenue_facts(companyfacts(cik)); revf=clean_facts(raw)
        except Exception as e: print("SEC",ticker,e)
    for idx in g.index:
        d=df.at[idx,"award_date"]; award=float(df.at[idx,"transaction_amount_abs_sum"])
        pm=m[m["Date"]<d].sort_values("Date")
        close=float(pm.iloc[-1]["Close"]) if len(pm) else np.nan
        ss=s[s["Date"]<=d].sort_values("Date") if len(s) else s
        sh=float(ss.iloc[-1]["Shares"]) if len(ss) else np.nan
        mcap=close*sh if np.isfinite(close) and np.isfinite(sh) and close>0 and sh>0 else np.nan
        rev,method,source=ltm_revenue_asof(revf,d)
        audit.append({"row_id":idx,"ticker":ticker,"award_date":d,"award_abs":award,"preaward_close":close,
                      "shares_known_at_award":sh,"market_cap_before":mcap,"sec_cik":cik,"revenue_concept":concept,
                      "ltm_revenue_before":rev,"ltm_revenue_method":method,"ltm_revenue_source":source})
scale=pd.DataFrame(audit).set_index("row_id").sort_index()
df["market_cap_before"]=scale["market_cap_before"]
df["ltm_revenue_before"]=scale["ltm_revenue_before"]
df["award_to_market_cap"]=df["transaction_amount_abs_sum"]/df["market_cap_before"]
df["log10_award_to_market_cap"]=np.log10(df["award_to_market_cap"])
df["log10_market_cap_before"]=np.log10(df["market_cap_before"])
df["award_to_ltm_revenue"]=df["transaction_amount_abs_sum"]/df["ltm_revenue_before"]

for c in ["award_to_market_cap","log10_award_to_market_cap","log10_market_cap_before","award_to_ltm_revenue"]:
    scale[c]=df[c].to_numpy()
scale.to_csv(OUT_SCALE,index=False)

challengers=["award_to_market_cap","log10_award_to_market_cap","log10_market_cap_before","award_to_ltm_revenue"]
factors=base11+challengers
X=pd.DataFrame(index=df.index)
for c in base11:
    s=pd.to_numeric(df[c],errors="coerce")
    X[c]=np.sign(s)*np.log1p(np.abs(s)) if model.get("transforms",{}).get(c)=="signed_log1p" else s
for c in challengers: X[c]=pd.to_numeric(df[c],errors="coerce")

peak=df["peak_pct"].to_numpy(float); normal=peak<10; y20=peak>=20; y50=peak>=50
def fold_md(tr,te,cols):
    ref=X.iloc[tr][cols].loc[normal[tr]].copy(); test=X.iloc[te][cols].copy()
    med=ref.median().fillna(0); ref=ref.fillna(med); test=test.fillna(med)
    keep=ref.std(ddof=1)>1e-12; ref=ref.loc[:,keep]; test=test.loc[:,keep]
    if ref.shape[1]==0: return np.full(len(te),np.nan)
    mu=ref.mean(); sd=ref.std(ddof=1).replace(0,1)
    zr=(ref-mu)/sd; zt=(test-mu)/sd
    lw=LedoitWolf().fit(zr.values); d=zt.values-lw.location_
    md2=np.einsum("ij,jk,ik->i",d,lw.precision_,d)/ref.shape[1]
    return np.sqrt(np.maximum(md2,1e-12))
def score(cols):
    o=np.full(len(df),np.nan)
    for tr,te in GroupKFold(5).split(X,y20,groups=df["ticker"]): o[te]=fold_md(tr,te,cols)
    v=np.isfinite(o)
    sig=o[v & y20]
    return {"scores":o,"auc20":float(roc_auc_score(y20[v],o[v])),"auc50":float(roc_auc_score(y50[v],o[v])),
            "rho":float(spearmanr(o[v],peak[v]).statistic),
            "sn":float(-10*np.log10(np.mean(1/np.maximum(sig**2,1e-12))))}

H=hadamard(16); D=(H[:,1:16]>0).astype(int); runs=[]
for run in range(16):
    active=[factors[j] for j in range(15) if D[run,j]==1]
    if active:
        r=score(active); row={"run":run+1,"n_features":len(active),"sn":r["sn"],"auc20":r["auc20"],"auc50":r["auc50"],"rho":r["rho"]}
    else:
        row={"run":run+1,"n_features":0,"sn":np.nan,"auc20":np.nan,"auc50":np.nan,"rho":np.nan}
    for c in factors: row[c]=int(c in active)
    runs.append(row)
rd=pd.DataFrame(runs); effects=[]
for c in factors:
    on=rd.loc[(rd[c]==1)&rd.sn.notna(),"sn"]; off=rd.loc[(rd[c]==0)&rd.sn.notna(),"sn"]
    aon=rd.loc[(rd[c]==1)&rd.auc20.notna(),"auc20"]; aoff=rd.loc[(rd[c]==0)&rd.auc20.notna(),"auc20"]
    effects.append({"variable":c,"kind":"existing_11" if c in base11 else "scale_challenger",
                    "SN_on_dB":float(on.mean()),"SN_off_dB":float(off.mean()),"SN_effect_dB":float(on.mean()-off.mean()),
                    "AUC20_effect":float(aon.mean()-aoff.mean())})
eff=pd.DataFrame(effects)
sq=eff.SN_effect_dB**2; eff["relative_SN_contribution_pct"]=100*sq/sq.sum()
eff=eff.sort_values(["SN_effect_dB","AUC20_effect"],ascending=False).reset_index(drop=True)
eff["rank_SN_effect"]=np.arange(1,len(eff)+1)

full=score(factors)
positive=eff.loc[eff.SN_effect_dB>0,"variable"].tolist()
pos=score(positive) if positive else None

df[["ticker","award_date","company","transaction_amount_abs_sum","peak_pct"]+factors+["market_cap_before","ltm_revenue_before"]].to_csv(OUT_INPUT,index=False)
rd.to_csv(OUT_RUNS,index=False); eff.to_csv(OUT_EFFECTS,index=False)

summary={"version":"RECON Factor-12 L16 Screen V1.0","sample":"A only",
"rows_before_project_exclusions":before_n,"rows_screened":len(df),
"rules":{"LMT_excluded":True,"zero_dollar_excluded":True,"B_C_not_read":True,"market_cap_point_in_time":True,"revenue_filed_on_or_before_award":True},
"frozen_11":base11,"challengers":challengers,
"coverage":{"market_cap_rows":int(df.market_cap_before.notna().sum()),"market_cap_pct":float(df.market_cap_before.notna().mean()),
            "ltm_revenue_rows":int(df.ltm_revenue_before.notna().sum()),"ltm_revenue_pct":float(df.ltm_revenue_before.notna().mean())},
"ranked_effects":eff[["rank_SN_effect","variable","kind","SN_effect_dB","AUC20_effect","relative_SN_contribution_pct"]].to_dict("records"),
"full_15":{"sn_db":full["sn"],"auc20":full["auc20"],"auc50":full["auc50"],"rho_peak":full["rho"]},
"positive_SN_effect_factors":positive,
"positive_effect_model":None if pos is None else {"sn_db":pos["sn"],"auc20":pos["auc20"],"auc50":pos["auc50"],"rho_peak":pos["rho"]},
"notes":["Existing 11 use transforms stored in frozen model.","Four challengers are tested exactly as defined.",
         "ln ratio omitted because it is linearly equivalent to log10 ratio after standardization.","Sample B and C are not loaded."]}
OUT_SUMMARY.write_text(json.dumps(summary,indent=2))
print(eff[["rank_SN_effect","variable","kind","SN_effect_dB","AUC20_effect","relative_SN_contribution_pct"]].to_string(index=False))
print(json.dumps(summary["coverage"],indent=2))
