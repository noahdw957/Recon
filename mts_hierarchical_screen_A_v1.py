import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import hadamard
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

INPUTS=[Path('mts_award_time_features_A.csv'),Path('Recon/mts_award_time_features_A.csv')]
IN=next((p for p in INPUTS if p.exists()),None)
if IN is None: raise FileNotFoundError('mts_award_time_features_A.csv not found')

df=pd.read_csv(IN)
peak=pd.to_numeric(df['peak_pct'],errors='coerce').to_numpy()
normal=peak<10; y20=peak>=20; y50=peak>=50

families={
'prior_scale':['prior_award_days_total','prior_transaction_count_total','days_since_last_award','prior_abs_award_mean','prior_abs_award_median','prior_abs_award_max','prior_signed_award_mean','current_to_prior_abs_mean','current_to_prior_abs_median','current_award_abs_percentile'],
'prior_response_60d':['prior_response_count_60d','prior_response_mean_60d','prior_response_median_60d','prior_max_gain_mean_60d','prior_hit_rate_10pct_within_60d','prior_hit_rate_20pct_within_60d'],
'prior_response_20d':['prior_response_count_20d','prior_response_mean_20d','prior_response_median_20d','prior_max_gain_mean_20d','prior_hit_rate_10pct_within_20d','prior_hit_rate_20pct_within_20d'],
'market_volatility':['pre_volatility_market_20d','pre_volatility_market_60d'],
'award_30d':['prior_award_days_30d','prior_transactions_30d','prior_abs_dollars_30d','prior_signed_dollars_30d','prior_positive_dollars_30d','prior_negative_dollars_30d','award_frequency_acceleration_30d','award_dollar_acceleration_30d'],
'market_momentum':['pre_return_5d','pre_return_20d','pre_return_60d','pre_return_120d'],
'rel_strength_ita':['relative_strength_ita_5d','relative_strength_ita_20d','relative_strength_ita_60d','relative_strength_ita_120d'],
'rel_strength_spy':['relative_strength_spy_5d','relative_strength_spy_20d','relative_strength_spy_60d','relative_strength_spy_120d'],
'award_size':['same_day_award_count','transaction_amount_sum','transaction_amount_abs_sum','current_agency_count','current_award_type_count'],
'market_position':['distance_from_52w_high_pct','volume_20d_to_60d_ratio']}
families={k:[c for c in v if c in df.columns] for k,v in families.items()}
families={k:v for k,v in families.items() if v}
features=sorted(set(sum(families.values(),[])))

X=pd.DataFrame(index=df.index)
def transform(c,s):
    lc=c.lower(); s=pd.to_numeric(s,errors='coerce')
    scale=any(t in lc for t in ['dollars','amount','count','award_days','transactions','days_since','abs_award','same_day_award'])
    exc=any(t in lc for t in ['percentile','fraction','rate','acceleration'])
    return np.sign(s)*np.log1p(np.abs(s)) if scale and not exc else s
for c in features: X[c]=transform(c,df[c])

def fold_md(tr,te,cols):
    ref=X.iloc[tr][cols].loc[normal[tr]].copy(); test=X.iloc[te][cols].copy()
    med=ref.median().fillna(0); ref=ref.fillna(med); test=test.fillna(med)
    sd=ref.std(ddof=1); keep=sd>1e-12; ref=ref.loc[:,keep]; test=test.loc[:,keep]
    if ref.shape[1]==0: return np.full(len(te),np.nan)
    mu=ref.mean(); sd=ref.std(ddof=1).replace(0,1)
    zr=(ref-mu)/sd; zt=(test-mu)/sd
    lw=LedoitWolf().fit(zr.values); d=zt.values-lw.location_
    md2=np.einsum('ij,jk,ik->i',d,lw.precision_,d)/ref.shape[1]
    return np.sqrt(np.maximum(md2,1e-12))

def score(cols):
    o=np.full(len(df),np.nan); cv=GroupKFold(5)
    for tr,te in cv.split(X,y20,groups=df['ticker']): o[te]=fold_md(tr,te,cols)
    v=np.isfinite(o)
    auc20=roc_auc_score(y20[v],o[v]); auc50=roc_auc_score(y50[v],o[v])
    rho=spearmanr(o[v],peak[v]).statistic
    sig=o[v & y20]; sn=-10*np.log10(np.mean(1/np.maximum(sig**2,1e-12)))
    return {'scores':o,'auc20':float(auc20),'auc50':float(auc50),'rho':float(rho),'sn':float(sn)}

family_runs=[]; family_effects=[]; survivors={}
for fam,cols in families.items():
    if len(cols)<=2:
        ranked=[]
        for c in cols:
            r=score([c]); ranked.append((r['auc20'],c)); family_effects.append({'family':fam,'variable':c,'SN_effect_dB':r['sn'],'AUC20_effect':r['auc20']-0.5})
        survivors[fam]=[c for _,c in sorted(ranked,reverse=True)[:4]]; continue
    chosen=[]
    for block_id,start in enumerate(range(0,len(cols),7),1):
        block=cols[start:start+7]; H=hadamard(8); D=(H[:,1:len(block)+1]>0).astype(int); runs=[]
        for run in range(8):
            active=[block[j] for j in range(len(block)) if D[run,j]==1]
            if not active: continue
            r=score(active); row={'family':fam,'block':block_id,'run':run+1,'sn':r['sn'],'auc20':r['auc20'],'auc50':r['auc50']}
            for c in block: row[c]=int(c in active)
            runs.append(row); family_runs.append(row)
        rd=pd.DataFrame(runs); eff=[]
        for c in block:
            on=rd.loc[rd[c]==1,'sn'].mean(); off=rd.loc[rd[c]==0,'sn'].mean(); ae=rd.loc[rd[c]==1,'auc20'].mean()-rd.loc[rd[c]==0,'auc20'].mean()
            eff.append((on-off,ae,c)); family_effects.append({'family':fam,'block':block_id,'variable':c,'SN_effect_dB':on-off,'AUC20_effect':ae})
        pos=[c for s,a,c in sorted(eff,reverse=True) if s>0] or [max(eff)[2]]; chosen.extend(pos)
    ranked=[]
    for c in sorted(set(chosen)):
        r=score([c]); ranked.append((r['auc20'],c))
    survivors[fam]=[c for _,c in sorted(ranked,reverse=True)[:4]]

pd.DataFrame(family_runs).to_csv('mts_family_L8_results_A.csv',index=False)
pd.DataFrame(family_effects).to_csv('mts_family_variable_effects_A.csv',index=False)

candidates=sorted(set(sum(survivors.values(),[])))
if len(candidates)>15:
    ranked=[]
    for c in candidates: ranked.append((score([c])['auc20'],c))
    candidates=[c for _,c in sorted(ranked,reverse=True)[:15]]

H=hadamard(16); D=(H[:,1:len(candidates)+1]>0).astype(int); runs=[]
for run in range(16):
    active=[candidates[j] for j in range(len(candidates)) if D[run,j]==1]
    if not active: continue
    r=score(active); row={'run':run+1,'n_features':len(active),'sn':r['sn'],'auc20':r['auc20'],'auc50':r['auc50'],'rho':r['rho']}
    for c in candidates: row[c]=int(c in active)
    runs.append(row)
rd=pd.DataFrame(runs); effects=[]
for c in candidates:
    sn_on=rd.loc[rd[c]==1,'sn'].mean(); sn_off=rd.loc[rd[c]==0,'sn'].mean(); auc_on=rd.loc[rd[c]==1,'auc20'].mean(); auc_off=rd.loc[rd[c]==0,'auc20'].mean()
    effects.append({'variable':c,'SN_effect_dB':sn_on-sn_off,'AUC20_effect':auc_on-auc_off})
eff=pd.DataFrame(effects).sort_values(['SN_effect_dB','AUC20_effect'],ascending=False)
final=eff.loc[eff.SN_effect_dB>0,'variable'].tolist()
if len(final)<3: final=eff.head(min(5,len(eff))).variable.tolist()
res=score(final); scores=res['scores']

out=df[['ticker','award_date','company','peak_pct','peak_class','MD4']].copy(); out['MTS_signal_score']=scores; out.to_csv('mts_final_scores_A.csv',index=False)
valid=out[np.isfinite(out.MTS_signal_score)].copy(); base20=(valid.peak_pct>=20).mean(); base50=(valid.peak_pct>=50).mean(); tr=[]
for frac in [0.20,0.10,0.05,0.02,0.01]:
    th=valid.MTS_signal_score.quantile(1-frac); g=valid[valid.MTS_signal_score>=th]; h20=(g.peak_pct>=20).mean(); h50=(g.peak_pct>=50).mean()
    tr.append({'top_fraction':frac,'score_threshold':float(th),'n_signals':len(g),'hit_rate_ge20':float(h20),'hit_rate_ge50':float(h50),'median_peak_pct':float(g.peak_pct.median()),'mean_peak_pct':float(g.peak_pct.mean()),'enrichment_ge20':float(h20/base20),'enrichment_ge50':float(h50/base50)})
thresholds=pd.DataFrame(tr); thresholds.to_csv('mts_signal_thresholds_A.csv',index=False)
rd.to_csv('mts_L16_combined_results_A.csv',index=False); eff.to_csv('mts_L16_variable_effects_A.csv',index=False)
summary={'version':'RECON MTS Hierarchical Screen A V1.0','events':len(df),'family_survivors':survivors,'l16_candidates':candidates,'final_variables':final,'final_auc20':res['auc20'],'final_auc50':res['auc50'],'final_rho_peak':res['rho'],'final_sn_db':res['sn'],'base_ge20_rate':float(base20),'base_ge50_rate':float(base50),'thresholds':tr,'notes':['Ticker used only as leave-group-out CV key, never as predictor.','Sample B not used.','Thresholds are Sample-A exploratory and must be frozen before validation.']}
Path('mts_hierarchical_summary_A.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
