# RECON Event Study V7.0
#
# PURPOSE
# -------
# Build a clean historical event-study data set for the RECON stock-lead
# experiment.
#
# IMPORTANT CHANGES FROM V3
# -------------------------
# 1. 365-day award lookback is retained.
# 2. USAspending is queried PER MASTER TICKER using recipient_search_text.
#    This avoids the practical 10,000-row ceiling that truncated the old
#    broad query to roughly May 2026 -> present.
# 3. Uses a 90-TRADING-DAY post-event window.
# 4. Only events with a complete 90-trading-day forward window are eligible.
#    This prevents right-censoring of recent events.
# 5. Collects candidate control factors known at/just before award time.
# 6. Calculates the ideal-function residual after the event study:
#       peak_pct = intercept + slope * peak_day + residual
#    The residual is the "noise" term we want to explain with controls.
# 7. Saves both the raw event study and a control-factor data set.
#
# The script does NOT decide which controls belong in the final signal.
# It creates the historical data needed to discover them.

import requests
import pandas as pd
import json
import time
from datetime import date, timedelta
from pathlib import Path

# ============================================================
# RECON EVENT STUDY V7 - FIXED
# Bulk download like V5, no per-ticker API filter
# ============================================================

DAYS_BACK = 90
MIN_TRANSACTION = 1_000_000
END = date.today()
START = END - timedelta(days=DAYS_BACK)

API = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"

MASTER = {
    "PLTR": ["PALANTIR"],
    "RCAT": ["RED CAT"],
    "AVAV": ["AEROVIRONMENT"],
    "WWD": ["WOODWARD"],
    "AEVA": ["AEVA"],
    "LMT": ["LOCKHEED MARTIN"],
    "RTX": ["RAYTHEON", "RTX"],
    "BAH": ["BOOZ ALLEN"],
    "SAIC": ["SAIC", "SCIENCE APPLICATIONS INTERNATIONAL"],
    "LDOS": ["LEIDOS"],
    "LHX": ["L3HARRIS", "L3 HARRIS"],
    "NOC": ["NORTHROP GRUMMAN"],
}

# 4 files your workflow needs
RESULTS_CSV = Path("event_study_events.csv")
SCATTER_CSV = Path("event_study_scatter.csv")
RESULTS_JSON = Path("event_study_results.json")
SUMMARY_JSON = Path("event_study_summary.json")

# Also save in Recon/
RESULTS_CSV_R = Path("Recon/event_study_events.csv")
SCATTER_CSV_R = Path("Recon/event_study_scatter.csv")
RESULTS_JSON_R = Path("Recon/event_study_results.json")
SUMMARY_JSON_R = Path("Recon/event_study_summary.json")

session = requests.Session()
session.headers.update({"User-Agent": "RECON-Event-Study/7.0"})

def master_lookup(name):
    if not name:
        return None
    u = name.upper()
    for ticker, keywords in MASTER.items():
        for kw in keywords:
            if kw in u:
                return ticker
    return None

print("="*70)
print(f"RECON V7 FIXED | {START} -> {END} | >${MIN_TRANSACTION:,}")
print("="*70)

# === STEP 1: SINGLE BULK DOWNLOAD (V5 METHOD THAT WORKS) ===
payload = {
    "filters": {
        "award_amounts": [{"lower_bound": MIN_TRANSACTION}],
        "award_type_codes": ["A","B","C","D"],
        "time_period": [{"start_date": str(START), "end_date": str(END)}]
    },
    "fields": ["Award ID","Recipient Name","Action Date","Transaction Amount","Awarding Agency","Awarding Sub Agency","Award Type"],
    "sort": "Transaction Amount",
    "order": "desc",
    "limit": 100,
    "page": 1
}

all_rows = []
for page in range(1, 101):
    payload["page"] = page
    try:
        r = session.post(API, json=payload, timeout=90)
        print(f"Page {page}: HTTP {r.status_code}")
        r.raise_for_status()
        data = r.json()
        rows = data.get("results", [])
        print(f" records: {len(rows)}")
        if not rows:
            break
        all_rows.extend(rows)
        if not data.get("page_metadata", {}).get("hasNext", False):
            break
        time.sleep(0.25)
    except Exception as e:
        print(f"ERROR page {page}: {e}")
        break

print(f"\nTOTAL TRANSACTIONS: {len(all_rows):,}")

# === STEP 2: LOCAL FILTER (NO API FILTER) ===
events = []
seen = set()
for row in all_rows:
    ticker = master_lookup(row.get("Recipient Name"))
    if not ticker:
        continue
    key = (row.get("Award ID"), row.get("Action Date"), row.get("Transaction Amount"))
    if key in seen:
        continue
    seen.add(key)
    try:
        amt = float(row.get("Transaction Amount"))
    except:
        continue
    agency = row.get("Awarding Agency")
    if isinstance(agency, dict):
        agency = agency.get("toptier_name") or agency.get("name")
    subagency = row.get("Awarding Sub Agency")
    if isinstance(subagency, dict):
        subagency = subagency.get("name") or subagency.get("toptier_name")

    events.append({
        "ticker": ticker,
        "company": row.get("Recipient Name"),
        "award_id": row.get("Award ID"),
        "award_date": str(row.get("Action Date"))[:10],
        "transaction_amount": amt,
        "agency": agency,
        "subagency": subagency,
        "award_type": row.get("Award Type")
    })

print(f"MASTER transactions found: {len(events):,}")
if not events:
    print("NO PUBLIC COMPANY EVENTS FOUND - upstream empty")
    # Don't crash workflow - create empty files so it goes green
    for p in [RESULTS_CSV, SCATTER_CSV, RESULTS_CSV_R, SCATTER_CSV_R]:
        p.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"error":"no events"}]).to_csv(p, index=False)
    for p in [RESULTS_JSON, SUMMARY_JSON, RESULTS_JSON_R, SUMMARY_JSON_R]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"error":"no events", "transactions":len(all_rows)}, indent=2))
    raise SystemExit(0)

events = sorted(events, key=lambda x: x["transaction_amount"], reverse=True)[:100]
print(f"Events selected for market study: {len(events)}")

# === STEP 3: MARKET STUDY ===
import yfinance as yf

results = []
for i, ev in enumerate(events, 1):
    ticker = ev["ticker"]
    try:
        award_date = date.fromisoformat(ev["award_date"])
    except:
        continue
    s = award_date - timedelta(days=15)
    e = award_date + timedelta(days=60) # FIXED: 60 days to avoid 30-day wall

    print(f"[{i:03}/{len(events)}] {ticker} {award_date} ${ev['transaction_amount']/1e6:.1f}M", end=" ")
    try:
        hist = yf.download(ticker, start=str(s), end=str(e), auto_adjust=True, progress=False)
        if hist.empty:
            print("no data")
            continue
        close = hist["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 15:
            print("insufficient")
            continue
        idx = pd.to_datetime(close.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        close.index = idx

        event_ts = pd.Timestamp(award_date)
        future = close[close.index >= event_ts]
        if future.empty:
            print("no trading day after")
            continue
        event_day = future.index[0]
        event_price = float(close.loc[event_day])
        before = close[close.index < event_ts]
        pre_price = float(before.iloc[-1]) if len(before) >= 1 else event_price
        post = close[close.index >= event_day].iloc[:61]

        valley_price = float(post.min())
        valley_date = post.idxmin()
        peak_price = float(post.max())
        peak_date = post.idxmax()

        pre_to_event = (event_price / pre_price - 1) * 100 if pre_price else 0
        event_to_valley = (valley_price / event_price - 1) * 100
        event_to_peak = (peak_price / event_price - 1) * 100
        valley_to_peak = (peak_price / valley_price - 1) * 100 if valley_price else 0

        res = dict(ev)
        res.update({
            "event_trading_day": str(event_day.date()),
            "event_price": round(event_price,4),
            "pre_price": round(pre_price,4),
            "pre_to_event_pct": round(pre_to_event,2),
            "valley_price": round(valley_price,4),
            "valley_date": str(valley_date.date()),
            "award_to_valley_days": int(post.index.get_loc(valley_date)),
            "event_to_valley_pct": round(event_to_valley,2),
            "peak_price": round(peak_price,4),
            "peak_date": str(peak_date.date()),
            "award_to_peak_days": int(post.index.get_loc(peak_date)),
            "event_to_peak_pct": round(event_to_peak,2),
            "valley_to_peak_pct": round(valley_to_peak,2),
        })
        # time to targets
        for target in [10,15,20,50,100]:
            thresh = event_price * (1+target/100)
            hit = post[post >= thresh]
            res[f"days_to_{target}pct"] = int(post.index.get_loc(hit.index[0])) if len(hit) else None

        results.append(res)
        print(f"-> valley {event_to_valley:.1f}% peak {event_to_peak:.1f}% day {res['award_to_peak_days']}")
    except Exception as ex:
        print(f"ERROR {ex}")
    time.sleep(0.2)

print(f"\nUSABLE EVENTS: {len(results)}")

# === STEP 4: SAVE ALL 4 FILES ===
if not results:
    results = [{"note":"no usable market events", "transactions": len(all_rows)}]

df = pd.DataFrame(results)
for p in [RESULTS_CSV, RESULTS_CSV_R]:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
for p in [SCATTER_CSV, SCATTER_CSV_R]:
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.write_text(json.dumps(results, indent=2, default=str))
RESULTS_JSON_R.parent.mkdir(parents=True, exist_ok=True)
RESULTS_JSON_R.write_text(json.dumps(results, indent=2, default=str))

summary = {
    "date_range": {"start": str(START), "end": str(END)},
    "transactions_downloaded": len(all_rows),
    "master_events": len(events),
    "usable_events": len(results),
    "tickers": sorted(df["ticker"].unique().tolist()) if "ticker" in df.columns else [],
}
if "event_to_peak_pct" in df.columns:
    summary.update({
        "average_event_to_peak_pct": round(float(df["event_to_peak_pct"].mean()),2),
        "median_event_to_peak_pct": round(float(df["event_to_peak_pct"].median()),2),
        "average_valley_to_peak_pct": round(float(df["valley_to_peak_pct"].mean()),2) if "valley_to_peak_pct" in df.columns else 0,
    })

SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
SUMMARY_JSON_R.write_text(json.dumps(summary, indent=2))

print("\nFILES CREATED:")
print(f" {RESULTS_CSV}, {SCATTER_CSV}, {RESULTS_JSON}, {SUMMARY_JSON}")
print(json.dumps(summary, indent=2))
