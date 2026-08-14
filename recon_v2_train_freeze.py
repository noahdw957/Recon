# RECON V2 STAGE-2 TRAIN/FREEZE
# Trains ONLY on Sample A. Sample B is diagnostic only and is never used for fitting.
# Stage 1 MTS remains frozen and unchanged.
#
# Inputs:
#   mts_award_time_features_A.csv
#   sample_B_nonLMT_validation_events.csv
#   mts_frozen_model_A.json
#
# Outputs:
#   recon_stage2_v2.ubj
#   recon_stage2_v2_meta.json
#   recon_stage2_v2_A_cv_predictions.csv
#   recon_stage2_v2_B_diagnostic.csv
#
# Added award-time scale features:
#   log_market_cap_preaward
#   award_to_market_cap
#   award_to_adv60
#
# Leakage rule:
#   Price/ADV use trading sessions STRICTLY BEFORE award_date.
#   Shares outstanding use the latest observation dated <= award_date.
#   Sample B outcomes are never used to fit or tune the model.

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from xgboost import XGBRegressor

A_FILE = Path("mts_award_time_features_A.csv")
B_FILE = Path("sample_B_nonLMT_validation_events.csv")
MTS_MODEL = Path("mts_frozen_model_A.json")

OUT_MODEL = Path("recon_stage2_v2.ubj")
OUT_META = Path("recon_stage2_v2_meta.json")
OUT_A_CV = Path("recon_stage2_v2_A_cv_predictions.csv")
OUT_B = Path("recon_stage2_v2_B_diagnostic.csv")

CACHE_DIR = Path("scale_cache_v2")
CACHE_DIR.mkdir(exist_ok=True)

DEPTH = 4
N_TREES = 100
LEARNING_RATE = 0.05
RANDOM_STATE = 20260813

def norm_market(d):
    if d is None or len(d) == 0:
        return pd.DataFrame(columns=["Date","Close","Volume"])
    x = d.copy()
    if isinstance(x.columns, pd.MultiIndex):
        close = x["Close"]
        vol = x["Volume"]
        if isinstance(close, pd.DataFrame): close = close.iloc[:,0]
        if isinstance(vol, pd.DataFrame): vol = vol.iloc[:,0]
        x = pd.DataFrame({"Close":close, "Volume":vol})
    else:
        x = x[["Close","Volume"]].copy()
    x = x.dropna(subset=["Close"])
    x.index = pd.to_datetime(x.index)
    if getattr(x.index, "tz", None) is not None:
        x.index = x.index.tz_localize(None)
    x = x.reset_index().rename(columns={x.reset_index().columns[0]:"Date"})
    if "Date" not in x.columns:
        x = x.rename(columns={x.columns[0]:"Date"})
    x["Date"] = pd.to_datetime(x["Date"]).dt.tz_localize(None)
    return x[["Date","Close","Volume"]]

def get_market(ticker, start, end):
    p = CACHE_DIR / f"{ticker}_market.csv"
    if p.exists():
        try:
            d = pd.read_csv(p, parse_dates=["Date"])
            if len(d) and d["Date"].min() <= start and d["Date"].max() >= end - pd.Timedelta(days=5):
                return d
        except Exception:
            pass
    d = yf.download(
        ticker,
        start=(start - pd.Timedelta(days=10)).date().isoformat(),
        end=(end + pd.Timedelta(days=3)).date().isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    d = norm_market(d)
    d.to_csv(p, index=False)
    time.sleep(0.15)
    return d

def get_shares(ticker, start, end):
    p = CACHE_DIR / f"{ticker}_shares.csv"
    if p.exists():
        try:
            s = pd.read_csv(p, parse_dates=["Date"])
            return s
        except Exception:
            pass
    out = pd.DataFrame(columns=["Date","Shares"])
    try:
        raw = yf.Ticker(ticker).get_shares_full(
            start=(start - pd.Timedelta(days=30)).date().isoformat(),
            end=(end + pd.Timedelta(days=3)).date().isoformat(),
        )
        if raw is not None and len(raw):
            idx = pd.to_datetime(raw.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            out = pd.DataFrame({
                "Date": idx,
                "Shares": pd.to_numeric(raw.values, errors="coerce")
            }).dropna().sort_values("Date")
    except Exception as exc:
        print(f"Shares history unavailable for {ticker}: {exc}")
    out.to_csv(p, index=False)
    time.sleep(0.15)
    return out

def scale_features(df):
    z = df.copy()
    z["award_date"] = pd.to_datetime(z["award_date"])
    z["log_market_cap_preaward"] = np.nan
    z["award_to_market_cap"] = np.nan
    z["award_to_adv60"] = np.nan
    z["shares_asof_award"] = np.nan
    z["preaward_close"] = np.nan
    z["adv60_dollars"] = np.nan

    global_start = z["award_date"].min() - pd.Timedelta(days=140)
    global_end = z["award_date"].max() + pd.Timedelta(days=5)

    for ticker, idxs in z.groupby("ticker").groups.items():
        ticker = str(ticker).upper().strip()
        if ticker == "LMT":
            continue
        m = get_market(ticker, global_start, global_end)
        s = get_shares(ticker, global_start, global_end)
        if m.empty:
            continue

        m = m.sort_values("Date")
        if not s.empty:
            s = s.sort_values("Date")

        for i in idxs:
            d = z.at[i, "award_date"]

            # STRICTLY pre-award market data.
            prior = m[m["Date"] < d].tail(60)
            if len(prior) == 0:
                continue

            close = float(prior.iloc[-1]["Close"])
            dollar_vol = pd.to_numeric(prior["Close"], errors="coerce") * pd.to_numeric(prior["Volume"], errors="coerce")
            adv60 = float(dollar_vol.dropna().mean()) if dollar_vol.notna().any() else np.nan

            shares = np.nan
            if not s.empty:
                ss = s[s["Date"] <= d]
                if len(ss):
                    shares = float(ss.iloc[-1]["Shares"])

            mcap = close * shares if np.isfinite(shares) and shares > 0 else np.nan
            award = abs(float(pd.to_numeric(z.at[i, "transaction_amount_abs_sum"], errors="coerce")))

            z.at[i, "preaward_close"] = close
            z.at[i, "adv60_dollars"] = adv60
            z.at[i, "shares_asof_award"] = shares

            if np.isfinite(mcap) and mcap > 0:
                z.at[i, "log_market_cap_preaward"] = math.log(mcap)
                z.at[i, "award_to_market_cap"] = award / mcap

            if np.isfinite(adv60) and adv60 > 0:
                z.at[i, "award_to_adv60"] = award / adv60

    return z

def make_model():
    return XGBRegressor(
        objective="reg:squarederror",
        max_depth=DEPTH,
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        subsample=0.85,
        colsample_bytree=0.90,
        min_child_weight=5,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

mts = json.loads(MTS_MODEL.read_text())
base11 = list(mts["features"])
scale3 = [
    "log_market_cap_preaward",
    "award_to_market_cap",
    "award_to_adv60",
]
features = base11 + scale3

A = pd.read_csv(A_FILE)
A = A[A["ticker"].astype(str).str.upper() != "LMT"].copy()
A["peak_pct"] = pd.to_numeric(A["peak_pct"], errors="coerce")
A = A.dropna(subset=["peak_pct","ticker","award_date"]).reset_index(drop=True)

print(f"Training Sample A rows, non-LMT: {len(A)}")
A = scale_features(A)

# XGBoost can handle NaN natively. Keep NaN instead of importing information
# from B or from future observations.
X = A[features].apply(pd.to_numeric, errors="coerce")
y = A["peak_pct"].astype(float)
groups = A["ticker"].astype(str)

n_splits = min(5, groups.nunique())
gkf = GroupKFold(n_splits=n_splits)
oof = np.full(len(A), np.nan)

for fold, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
    model = make_model()
    model.fit(X.iloc[tr], y.iloc[tr])
    oof[va] = model.predict(X.iloc[va])
    print(f"Fold {fold}/{n_splits} complete")

mae = mean_absolute_error(y, oof)
rho = spearmanr(y, oof, nan_policy="omit").statistic

Aout = A[["ticker","award_date","company","peak_pct"] + features].copy()
Aout["stage2_v2_oof_pred_peak"] = oof
Aout.to_csv(OUT_A_CV, index=False)

# Freeze on all Sample A only.
model = make_model()
model.fit(X, y)
model.save_model(OUT_MODEL)

meta = {
    "version": "RECON Stage2 V2.0",
    "training_population": "Sample A only, LMT excluded",
    "training_rows": int(len(A)),
    "base_features": base11,
    "added_scale_features": scale3,
    "uses_MD_as_input": False,
    "target": "peak_pct / peak_gain_90d regression",
    "xgboost": {
        "max_depth": DEPTH,
        "n_estimators": N_TREES,
        "learning_rate": LEARNING_RATE,
        "random_state": RANDOM_STATE,
    },
    "grouped_ticker_cv": {
        "folds": int(n_splits),
        "mae": float(mae),
        "spearman_rho": float(rho),
    },
    "leakage_rules": [
        "Price and ADV use dates strictly before award_date.",
        "Shares outstanding use latest shares observation dated <= award_date.",
        "Sample B outcomes are diagnostic only and are never used for fit/tuning.",
        "Stage 1 MTS threshold/model remain unchanged."
    ],
}
OUT_META.write_text(json.dumps(meta, indent=2))

# Diagnostic Sample B only after freeze.
if B_FILE.exists():
    B = pd.read_csv(B_FILE)
    B = B[
        (B["ticker"].astype(str).str.upper() != "LMT") &
        (pd.to_numeric(B.get("predicted_buy",0), errors="coerce").fillna(0).astype(int) == 1)
    ].copy().reset_index(drop=True)

    print(f"Blind-B candidate rows to rescore: {len(B)}")
    B = scale_features(B)
    XB = B[features].apply(pd.to_numeric, errors="coerce")
    B["stage2_v2_pred_peak"] = model.predict(XB)
    B["stage2_v2_rank_low_to_high"] = B["stage2_v2_pred_peak"].rank(method="min", ascending=True).astype(int)
    B["stage2_v2_rank_high_to_low"] = B["stage2_v2_pred_peak"].rank(method="min", ascending=False).astype(int)

    cols = [
        "ticker","award_date","peak_pct","MTS_signal_score",
        "stage2_v2_pred_peak","stage2_v2_rank_low_to_high",
        "log_market_cap_preaward","award_to_market_cap","award_to_adv60"
    ]
    B[cols].sort_values("stage2_v2_pred_peak").to_csv(OUT_B, index=False)

    print()
    print("LOWEST 8 B CANDIDATES AFTER V2")
    print(B[cols].sort_values("stage2_v2_pred_peak").head(8).to_string(index=False))

print()
print("="*72)
print("RECON STAGE 2 V2 FROZEN")
print(f"A grouped-CV MAE: {mae:.3f}")
print(f"A grouped-CV Spearman rho: {rho:.3f}")
print(f"Model: {OUT_MODEL}")
print(f"Metadata: {OUT_META}")
print("="*72)
