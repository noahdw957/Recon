import json
import time
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

PRE_DAYS = 20
POST_DAYS = 90
PRE_CALENDAR_DAYS = 45
POST_CALENDAR_DAYS = 150
REQUEST_PAUSE = 0.12

INPUT_CANDIDATES = [
    Path("random_10pct_sample_A_zero_purged.csv"),
    Path("Recon/random_10pct_sample_A_zero_purged.csv"),
    Path("random_10pct_sample_A.csv"),
    Path("Recon/random_10pct_sample_A.csv"),
]

OUT_EVENTS = Path("trajectory_events_A.csv")
OUT_MATRIX = Path("trajectory_matrix_A.csv")
OUT_PARAMS = Path("trajectory_parameters_A.csv")
OUT_SUMMARY = Path("trajectory_summary_A.json")
RECON_DIR = Path("Recon")

def find_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Sample A not found")

def get_close_series(ticker, start_date, end_date):
    hist = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if hist is None or hist.empty:
        return None
    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.empty:
        return None
    idx = pd.to_datetime(close.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    close.index = idx
    return close.astype(float)

def slope_per_day(values):
    y = np.asarray(values, dtype=float)
    if len(y) < 2 or np.isnan(y).any():
        return None
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])

def area_trapz(values):
    y = np.asarray(values, dtype=float)
    if len(y) < 2 or np.isnan(y).any():
        return None
    return float(np.trapezoid(y))

def max_one_day_change(values):
    y = np.asarray(values, dtype=float)
    if len(y) < 2:
        return (None, None)
    d = np.diff(y)
    return float(np.max(d)), float(np.min(d))

input_path = find_input()
df = pd.read_csv(input_path)

if "ticker" not in df.columns or "award_date" not in df.columns:
    raise ValueError("Input must contain ticker and award_date")

if "transaction_amount" in df.columns:
    df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
    df = df[df["transaction_amount"] != 0].copy()

df = df.reset_index(drop=True)

print("="*72)
print("RECON TRAJECTORY ENRICHMENT V1.0")
print(f"Input events: {len(df):,}")
print(f"Window: -{PRE_DAYS} to +{POST_DAYS} trading days")
print("="*72)

event_rows = []
matrix_rows = []
parameter_rows = []

for i, row in df.iterrows():
    ticker = str(row["ticker"]).strip()
    try:
        award_date = date.fromisoformat(str(row["award_date"])[:10])
    except Exception:
        print(f"[{i+1}/{len(df)}] {ticker}: bad date")
        continue

    start_date = award_date - timedelta(days=PRE_CALENDAR_DAYS)
    end_date = award_date + timedelta(days=POST_CALENDAR_DAYS)

    print(f"[{i+1:04}/{len(df)}] {ticker:6} {award_date}", end=" ")

    close = None
    for attempt in range(2):
        try:
            close = get_close_series(ticker, start_date, end_date)
            if close is not None and not close.empty:
                break
        except Exception:
            pass
        time.sleep(1.0)

    if close is None or close.empty:
        print("NO DATA")
        continue

    t = pd.Timestamp(award_date)
    future = close[close.index >= t]
    if future.empty:
        print("NO EVENT DAY")
        continue
    event_day = future.index[0]
    event_loc = close.index.get_loc(event_day)

    if event_loc < PRE_DAYS:
        print("INSUFFICIENT PRE")
        continue
    if event_loc + POST_DAYS >= len(close):
        print("INSUFFICIENT POST")
        continue

    raw_window = close.iloc[event_loc-PRE_DAYS:event_loc+POST_DAYS+1].copy()
    if len(raw_window) != PRE_DAYS + POST_DAYS + 1:
        print("BAD WINDOW")
        continue

    event_price = float(close.loc[event_day])
    standardized = (raw_window / event_price - 1.0) * 100.0
    values = standardized.to_numpy(dtype=float)
    day_numbers = list(range(-PRE_DAYS, POST_DAYS+1))

    pre5 = standardized.iloc[PRE_DAYS-5:PRE_DAYS]
    pre10 = standardized.iloc[PRE_DAYS-10:PRE_DAYS]
    pre20 = standardized.iloc[:PRE_DAYS]

    pre_slope_5 = slope_per_day(pre5.values)
    pre_slope_10 = slope_per_day(pre10.values)
    pre_slope_20 = slope_per_day(pre20.values)
    pre_volatility = float(np.std(np.diff(pre20.values), ddof=1)) if len(pre20)>2 else None

    post = standardized.iloc[PRE_DAYS:]
    post_values = post.to_numpy(dtype=float)

    valley_pos = int(np.argmin(post_values))
    peak_pos = int(np.argmax(post_values))
    valley_pct = float(post_values[valley_pos])
    peak_pct = float(post_values[peak_pos])
    valley_date = post.index[valley_pos]
    peak_date = post.index[peak_pos]

    after_valley = post.iloc[valley_pos:]
    local_peak = int(np.argmax(after_valley.to_numpy(dtype=float)))
    peak_after_valley_pos = valley_pos + local_peak
    peak_after_valley_pct = float(post_values[peak_after_valley_pos])

    recovery_days = peak_after_valley_pos - valley_pos
    expansion_pct = peak_after_valley_pct - valley_pct
    velocity = expansion_pct / recovery_days if recovery_days > 0 else None

    total_area = area_trapz(post_values)
    positive_area = area_trapz(np.maximum(post_values, 0))
    negative_area = area_trapz(np.minimum(post_values, 0))
    max_up_1d, max_down_1d = max_one_day_change(post_values)

    target_days = {}
    for target in [5,10,15,20,30,50,100]:
        hits = np.where(post_values >= target)[0]
        target_days[f"days_to_{target}pct"] = int(hits[0]) if len(hits) else None

    fixed_returns = {}
    for d in [1,3,5,7,10,14,21,30,45,60,90]:
        fixed_returns[f"return_day_{d}pct"] = float(post_values[d])

    base = row.to_dict()
    event_record = dict(base)
    event_record.update({
        "event_trading_day": str(event_day.date()),
        "event_price": round(event_price,6),
        "pre_slope_5d": pre_slope_5,
        "pre_slope_10d": pre_slope_10,
        "pre_slope_20d": pre_slope_20,
        "pre_volatility_20d": pre_volatility,
        "valley_day": valley_pos,
        "valley_pct": valley_pct,
        "valley_date": str(valley_date.date()),
        "peak_day": peak_pos,
        "peak_pct": peak_pct,
        "peak_date": str(peak_date.date()),
        "peak_after_valley_day": peak_after_valley_pos,
        "peak_after_valley_pct": peak_after_valley_pct,
        "recovery_days": recovery_days,
        "valley_to_peak_expansion_pct": expansion_pct,
        "valley_to_peak_velocity_pct_per_day": velocity,
        "area_total_pct_days": total_area,
        "area_positive_pct_days": positive_area,
        "area_negative_pct_days": negative_area,
        "max_up_1d_pct": max_up_1d,
        "max_down_1d_pct": max_down_1d,
    })
    event_record.update(target_days)
    event_record.update(fixed_returns)
    event_rows.append(event_record)

    matrix_record = {
        "ticker": ticker,
        "company": base.get("company"),
        "award_id": base.get("award_id"),
        "award_date": str(award_date),
        "transaction_amount": base.get("transaction_amount"),
    }
    for day_num, val in zip(day_numbers, values):
        label = f"r_m{abs(day_num):02d}" if day_num < 0 else f"r_p{day_num:02d}"
        matrix_record[label] = float(val)
    matrix_rows.append(matrix_record)

    parameter_rows.append({
        "ticker": ticker,
        "company": base.get("company"),
        "award_id": base.get("award_id"),
        "award_date": str(award_date),
        "transaction_amount": base.get("transaction_amount"),
        "agency": base.get("agency"),
        "subagency": base.get("subagency"),
        "award_type": base.get("award_type"),
        "pre_slope_5d": pre_slope_5,
        "pre_slope_10d": pre_slope_10,
        "pre_slope_20d": pre_slope_20,
        "pre_volatility_20d": pre_volatility,
        "valley_day": valley_pos,
        "valley_pct": valley_pct,
        "peak_day": peak_pos,
        "peak_pct": peak_pct,
        "peak_after_valley_day": peak_after_valley_pos,
        "peak_after_valley_pct": peak_after_valley_pct,
        "recovery_days": recovery_days,
        "valley_to_peak_expansion_pct": expansion_pct,
        "valley_to_peak_velocity_pct_per_day": velocity,
        "area_total_pct_days": total_area,
        "area_positive_pct_days": positive_area,
        "area_negative_pct_days": negative_area,
        "max_up_1d_pct": max_up_1d,
        "max_down_1d_pct": max_down_1d,
        **target_days,
        **fixed_returns,
    })

    print(f"OK valley {valley_pct:+.1f}% d{valley_pos} peak {peak_pct:+.1f}% d{peak_pos}")
    time.sleep(REQUEST_PAUSE)

events_df = pd.DataFrame(event_rows)
matrix_df = pd.DataFrame(matrix_rows)
params_df = pd.DataFrame(parameter_rows)

events_df.to_csv(OUT_EVENTS,index=False)
matrix_df.to_csv(OUT_MATRIX,index=False)
params_df.to_csv(OUT_PARAMS,index=False)

if RECON_DIR.exists():
    events_df.to_csv(RECON_DIR/OUT_EVENTS.name,index=False)
    matrix_df.to_csv(RECON_DIR/OUT_MATRIX.name,index=False)
    params_df.to_csv(RECON_DIR/OUT_PARAMS.name,index=False)

summary = {
    "version":"RECON Trajectory Enrichment V1.0",
    "input_file":str(input_path),
    "input_events":int(len(df)),
    "usable_trajectory_events":int(len(events_df)),
    "pre_trading_days":PRE_DAYS,
    "post_trading_days":POST_DAYS,
    "trajectory_vector_length":PRE_DAYS+POST_DAYS+1,
    "standardization":"100 * (price_t / award_day_price - 1)"
}
if not params_df.empty:
    summary.update({
        "median_valley_day":float(params_df["valley_day"].median()),
        "median_valley_pct":float(params_df["valley_pct"].median()),
        "median_peak_day":float(params_df["peak_day"].median()),
        "median_peak_pct":float(params_df["peak_pct"].median()),
        "median_recovery_days":float(params_df["recovery_days"].median()),
        "median_expansion_pct":float(params_df["valley_to_peak_expansion_pct"].median())
    })

OUT_SUMMARY.write_text(json.dumps(summary,indent=2))
if RECON_DIR.exists():
    (RECON_DIR/OUT_SUMMARY.name).write_text(json.dumps(summary,indent=2))

print(json.dumps(summary,indent=2))
