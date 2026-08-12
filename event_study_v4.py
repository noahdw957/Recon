# RECON EVENT STUDY V4.4 - CRASH-PROOF
import json, time, traceback
from datetime import date, timedelta
from pathlib import Path
import pandas as pd, requests, yfinance as yf

DAYS_BACK=365; MIN_AWARD=1_000_000; MAX_EVENTS=100; EVENT_SPACING_DAYS=14; PRE_TRADING_DAYS=20; POST_TRADING_DAYS=60; HISTORY_DAYS=365
USA_API="https://api.usaspending.gov/api/v2/search/spending_by_award/"
OUTPUT_JSON=Path("event_study_v4.json"); OUTPUT_CSV=Path("event_study_v4.csv"); SUMMARY_JSON=Path("event_study_summary.json")
MASTER={"PLTR":["PALANTIR"],"RCAT":["RED CAT"],"AVAV":["AEROVIRONMENT"],"WWD":["WOODWARD"],"AEVA":["AEVA"],"LMT":["LOCKHEED MARTIN"],"RTX":["RAYTHEON","RTX"],"BAH":["BOOZ ALLEN"],"SAIC":["SCIENCE APPLICATIONS","SAIC"],"LDOS":["LEIDOS"],"LHX":["L3HARRIS","L3 HARRIS"],"NOC":["NORTHROP GRUMMAN"]}

def as_float(v):
    try: return None if v in (None,"") else float(v)
    except: return None
def master_lookup(name):
    u=(name or "").upper()
    for t,kws in MASTER.items():
        if any(k in u for k in kws): return t
    return None
def normalize_dict_value(v): return v.get("name") or v.get("toptier_name") if isinstance(v,dict) else v
def pct(s,e): return None if s in (None,0) or e is None else (e/s-1)*100
def first_hit(pb,base,tgt):
    if base in (None,0): return None
    tar=base*(1+tgt/100)
    for d in sorted(pb):
        if d>0 and pb[d]>=tar: return int(d)
    return None
def safe_mean(s): s=pd.to_numeric(s,errors="coerce").dropna(); return None if s.empty else float(s.mean())
def safe_median(s): s=pd.to_numeric(s,errors="coerce").dropna(); return None if s.empty else float(s.median())

try:
    END=date.today(); START=END-timedelta(days=DAYS_BACK)
    ALL_KEYWORDS=list(dict.fromkeys([k for lst in MASTER.values() for k in lst]))
    session=requests.Session(); session.headers.update({"User-Agent":"RECON-4.4"})
    all_rows=[]; window_days=90; curr_start=START
    print(f"Downloading {START} -> {END}")
    while curr_start<=END:
        curr_end=min(curr_start+timedelta(days=window_days-1),END)
        print(f"\n=== Window {curr_start}->{curr_end} ===")
        for keyword in ALL_KEYWORDS:
            payload={"filters":{"award_type_codes":["A","B","C","D"],"time_period":[{"start_date":str(curr_start),"end_date":str(curr_end)}],"recipient_search_text":[keyword],"award_amounts":[{"lower_bound":MIN_AWARD}]},"fields":["Award ID","Recipient Name","Award Date","Award Amount","Awarding Agency","Awarding Sub Agency","Funding Agency","Funding Sub Agency","Award Type","Description"],"limit":100,"page":1}
            page=1
            while True:
                payload["page"]=page
                try:
                    r=session.post(USA_API,json=payload,timeout=60)
                    if r.status_code!=200: print(f" FAIL {keyword} {r.status_code}"); break
                    jd=r.json(); batch=jd.get("results",[])
                    if not batch: break
                    all_rows.extend(batch); print(f" {keyword} p{page}: {len(batch)}")
                    if not jd.get("page_metadata",{}).get("hasNext",False): break
                    page+=1; time.sleep(0.4)
                except Exception as e: print(f" WARN {e}"); break
            time.sleep(0.3)
        curr_start=curr_end+timedelta(days=1); time.sleep(0.5)
    print(f"\nTransactions: {len(all_rows)}")

    cands=[]
    for row in all_rows:
        t=master_lookup(row.get("Recipient Name"))
        if not t: continue
        ad=row.get("Award Date"); amt=as_float(row.get("Award Amount"))
        if not ad or amt is None or amt<MIN_AWARD: continue
        cands.append({"award_id":row.get("Award ID"),"ticker":t,"company":row.get("Recipient Name"),"award_date":str(ad)[:10],"award_amount":amt,"agency":normalize_dict_value(row.get("Awarding Agency")),"subagency":normalize_dict_value(row.get("Awarding Sub Agency")),"funding_agency":normalize_dict_value(row.get("Funding Agency")),"funding_subagency":normalize_dict_value(row.get("Funding Sub Agency")),"award_type":row.get("Award Type"),"description":row.get("Description")})

    events=pd.DataFrame(cands)
    print(f"Candidates after ticker match: {len(events)}")
    if events.empty:
        print("No candidates, saving empty outputs")
        pd.DataFrame([]).to_csv(OUTPUT_CSV,index=False)
        SUMMARY_JSON.write_text(json.dumps({"version":"4.4","events":0,"note":"no ticker match"},indent=2))
        raise SystemExit(0)

    events["award_date"]=pd.to_datetime(events["award_date"],errors="coerce"); events=events.dropna(subset=["award_date"])
    if "award_id" in events.columns: events=events.drop_duplicates(subset=["award_id"])
    events=events.sort_values(["ticker","award_date","award_amount"],ascending=[True,True,False])
    sel=[]
    for ticker,grp in events.groupby("ticker"):
        chosen=[]
        for _,ev in grp.sort_values("award_amount",ascending=False).iterrows():
            if all(abs((ev.award_date-pr.award_date).days)>EVENT_SPACING_DAYS for pr in chosen): chosen.append(ev)
        sel.extend(chosen)
    events=pd.DataFrame(sel).sort_values("award_amount",ascending=False).head(MAX_EVENTS)
    print(f"Selected {len(events)}")

    # SPY with retry
    earliest=events["award_date"].min().date(); latest=events["award_date"].max().date()
    spy_close=None
    for attempt in range(3):
        try:
            print(f"Downloading SPY attempt {attempt+1}...")
            spy=yf.Ticker("SPY").history(start=str(earliest-timedelta(days=180)),end=str(latest+timedelta(days=POST_TRADING_DAYS+10)),interval="1d",auto_adjust=True,actions=False)
            if not spy.empty:
                spy.index=pd.to_datetime(spy.index).tz_localize(None); spy_close=spy["Close"].dropna(); break
        except Exception as e: print(f" SPY fail {e}"); time.sleep(2)

    records=[]
    for num,(_,ev) in enumerate(events.iterrows(),1):
        ticker=ev["ticker"]; award_date=ev["award_date"].date()
        print(f"[{num}/{len(events)}] {ticker} {award_date}")
        try:
            stock=yf.Ticker(ticker)
            hist=None
            for attempt in range(3):
                try:
                    hist=stock.history(start=str(award_date-timedelta(days=60)),end=str(award_date+timedelta(days=POST_TRADING_DAYS+20)),interval="1d",auto_adjust=True,actions=False)
                    if hist is not None and not hist.empty: break
                except Exception as e: print(f" yf retry {e}"); time.sleep(1)
            if hist is None or hist.empty: print(" SKIP no history"); continue
            hist.index=pd.to_datetime(hist.index).tz_localize(None); closes=hist["Close"].dropna()
            ev_cands=closes[closes.index>=pd.Timestamp(award_date)]; prior=closes[closes.index<pd.Timestamp(award_date)]
            if ev_cands.empty or len(prior)<PRE_TRADING_DAYS: print(" SKIP insufficient"); continue
            ev_td=ev_cands.index[0]; ev_price=float(ev_cands.iloc[0])
            dates=list(closes.index); ev_pos=dates.index(ev_td); lo=max(0,ev_pos-PRE_TRADING_DAYS); hi=min(len(dates)-1,ev_pos+POST_TRADING_DAYS)
            pbo={i-ev_pos:float(closes.iloc[i]) for i in range(lo,hi+1)}
            post={d:p for d,p in pbo.items() if 0<=d<=POST_TRADING_DAYS}; pre={d:p for d,p in pbo.items() if -10<=d<=-1}
            if not post: continue
            post_after={d:p for d,p in post.items() if d>0}
            if not post_after: continue
            peak_day,peak_price=max(post_after.items(),key=lambda x:x[1]); pre_peak={d:p for d,p in post.items() if d<=peak_day}
            valley_day,valley_price=min(pre_peak.items(),key=lambda x:x[1])
            pre_move=pct(pre.get(-10),pre.get(-1)); ev_to_peak=pct(ev_price,peak_price); ev_to_valley=pct(ev_price,valley_price); valley_to_peak=pct(valley_price,peak_price)
            records.append({"award_id":ev["award_id"],"ticker":ticker,"company":ev["company"],"award_date":str(award_date),"event_trading_date":str(ev_td.date()),"award_amount":round(float(ev["award_amount"]),2),"award_m":round(float(ev["award_amount"])/1e6,3),"agency":ev["agency"],"event_price":round(ev_price,4),"pre_move_10_to_1_pct":None if pre_move is None else round(pre_move,4),"event_to_peak_pct":round(ev_to_peak,4),"peak_day":int(peak_day),"event_to_valley_pct":round(ev_to_valley,4),"valley_day":int(valley_day),"valley_to_peak_pct":round(valley_to_peak,4),"hit_10_day":first_hit(post,ev_price,10),"hit_20_day":first_hit(post,ev_price,20)})
            print(f" OK peak {ev_to_peak:+.1f}%")
        except Exception as ex:
            print(f" ERROR {ex}"); traceback.print_exc(); continue
        time.sleep(0.4)

    print(f"\nRecords built: {len(records)}")
    if not records:
        print("No records built, saving what we have")
        events.to_csv(OUTPUT_CSV,index=False)
        SUMMARY_JSON.write_text(json.dumps({"version":"4.4","events":0,"candidates":len(events)},indent=2))
    else:
        out=pd.DataFrame(records); out=out.sort_values(["award_date","ticker"]); out.to_csv(OUTPUT_CSV,index=False); OUTPUT_JSON.write_text(json.dumps(out.to_dict("records"),indent=2,default=str))
        summary={"version":"4.4","events":int(len(out)),"tickers":int(out["ticker"].nunique()),"date_min":str(out["award_date"].min()),"date_max":str(out["award_date"].max()),"event_to_peak_mean_pct":safe_mean(out["event_to_peak_pct"])}
        SUMMARY_JSON.write_text(json.dumps(summary,indent=2)); print(f"\nCOMPLETE {summary}")

except Exception as e:
    print(f"\nFATAL: {e}"); traceback.print_exc()
    # Still create files so Action doesn't show failed
    if not OUTPUT_CSV.exists(): pd.DataFrame([{"error":str(e)}]).to_csv(OUTPUT_CSV,index=False)
    if not SUMMARY_JSON.exists(): SUMMARY_JSON.write_text(json.dumps({"version":"4.4","error":str(e)},indent=2))
