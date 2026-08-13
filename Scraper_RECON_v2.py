import json
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================
# RECON LIVE BUY SCANNER V2.0
#
# USER-FACING OUTPUT:
#     buy_tickers.txt
#
# Everything else is stored under ReconData/ for analysis.
#
# BUY MODEL:
#     Frozen 11-factor Sample-A MTS model
#     Threshold comes from mts_frozen_model_A.json
#
# IMPORTANT:
#     This is a sandbox scanner. It generates experimental BUY
#     signals; it does not contain a validated SELL model yet.
# ============================================================

LOOKBACK_DAYS = 7
MAX_PAGES = 100
PAGE_SIZE = 100
MIN_POSITIVE_TRANSACTION = 1.0

AUTO_DISCOVER_MIN = 1_000_000
MAX_AUTO_LOOKUPS = 30

TODAY = date.today()
START = TODAY - timedelta(days=LOOKBACK_DAYS)

USA_API = (
    "https://api.usaspending.gov/api/v2/"
    "search/spending_by_transaction/"
)

YAHOO_SEARCH_API = (
    "https://query2.finance.yahoo.com/v1/finance/search"
)

# ============================================================
# REQUIRED EXISTING FILES
# ============================================================

MODEL_FILE = Path("mts_frozen_model_A.json")
MASTER_HISTORY_FILE = Path("master_zero_purged.csv")

# ============================================================
# CLEAN OUTPUT / STATE AREA
# ============================================================

DATA_DIR = Path("ReconData")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BUY_FILE = Path("buy_tickers.txt")

SCAN_LOG = DATA_DIR / "recon_scan_log.csv"
LIVE_HISTORY = DATA_DIR / "live_history.csv"
SEEN_FILE = DATA_DIR / "seen_events.json"
TICKER_CACHE_FILE = DATA_DIR / "ticker_cache.json"
UNKNOWN_FILE = DATA_DIR / "unknown_companies.json"

# ============================================================
# FALLBACK PUBLIC-COMPANY MAP
#
# The large master_zero_purged.csv is also used to bootstrap
# recipient -> ticker mappings automatically, so this list is
# only a fallback for familiar names.
# ============================================================

MASTER = {
    "PLTR": ["PALANTIR"],
    "RCAT": ["RED CAT"],
    "AVAV": ["AEROVIRONMENT"],
    "WWD": ["WOODWARD"],
    "AEVA": ["AEVA TECHNOLOGIES", "AEVA"],
    "LMT": ["LOCKHEED MARTIN"],
    "RTX": ["RAYTHEON", "RTX CORPORATION"],
    "BAH": ["BOOZ ALLEN"],
    "SAIC": ["SCIENCE APPLICATIONS INTERNATIONAL", "SAIC"],
    "LDOS": ["LEIDOS"],
    "LHX": ["L3HARRIS", "L3 HARRIS"],
    "NOC": ["NORTHROP GRUMMAN"],
    "HII": ["HUNTINGTON INGALLS"],
    "CACI": ["CACI"],
    "BA": ["BOEING"],
    "GD": ["GENERAL DYNAMICS"],
    "ACN": ["ACCENTURE FEDERAL"],
    "HON": ["HONEYWELL"],
    "TXT": ["TEXTRON", "BELL TEXTRON"],
    "KTOS": ["KRATOS"],
    "MRCY": ["MERCURY SYSTEMS"],
    "PSN": ["PARSONS"],
    "ETN": ["EATON"],
    "CAT": ["CATERPILLAR"],
    "IBM": ["INTERNATIONAL BUSINESS MACHINES"],
}

# ============================================================
# BASIC HELPERS
# ============================================================

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
    return default


def normalize_name(name):
    return " ".join(str(name).upper().replace(",", " ").split())


session = requests.Session()
session.headers.update({"User-Agent": "RECON-Live/2.0"})

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Missing {MODEL_FILE}. Upload the frozen Sample-A model."
    )

if not MASTER_HISTORY_FILE.exists():
    raise FileNotFoundError(
        f"Missing {MASTER_HISTORY_FILE}. Upload the saved nonzero award master."
    )

model = json.loads(MODEL_FILE.read_text())
seen = load_json(SEEN_FILE, {})
ticker_cache = load_json(TICKER_CACHE_FILE, {})
unknown_previous = load_json(UNKNOWN_FILE, {})

# ============================================================
# LOAD / CLEAN SAVED HISTORICAL AWARD DATABASE
# ============================================================

master_history = pd.read_csv(MASTER_HISTORY_FILE)

required_history = {
    "ticker",
    "company",
    "award_id",
    "award_date",
    "transaction_amount",
}

missing = required_history - set(master_history.columns)
if missing:
    raise ValueError(
        f"{MASTER_HISTORY_FILE} missing required columns: {sorted(missing)}"
    )

master_history["award_date"] = pd.to_datetime(
    master_history["award_date"], errors="coerce"
)
master_history["transaction_amount"] = pd.to_numeric(
    master_history["transaction_amount"], errors="coerce"
)

master_history = master_history[
    master_history["ticker"].notna()
    & master_history["award_date"].notna()
    & master_history["transaction_amount"].notna()
    & (master_history["transaction_amount"] != 0)
].copy()

# ============================================================
# BOOTSTRAP RECIPIENT -> TICKER CACHE FROM SAVED MASTER
# ============================================================

for _, row in master_history[["company", "ticker"]].dropna().drop_duplicates().iterrows():
    ticker_cache[normalize_name(row["company"])] = str(row["ticker"]).upper()

for ticker, keywords in MASTER.items():
    for keyword in keywords:
        ticker_cache[normalize_name(keyword)] = ticker

# ============================================================
# TICKER MATCHING
# ============================================================

lookups_used = 0


def master_lookup(name):
    upper = normalize_name(name)

    # Exact historic recipient mapping first.
    if upper in ticker_cache:
        return ticker_cache[upper]

    # Then fallback keywords.
    for ticker, keywords in MASTER.items():
        for keyword in keywords:
            if normalize_name(keyword) in upper:
                return ticker

    return None


def find_ticker(name, transaction_amount):
    global lookups_used

    cached = master_lookup(name)
    if cached:
        return cached

    # Avoid wasting Yahoo calls on tiny unknown transactions.
    if abs(float(transaction_amount or 0)) < AUTO_DISCOVER_MIN:
        return None

    if lookups_used >= MAX_AUTO_LOOKUPS:
        return None

    lookups_used += 1

    try:
        response = session.get(
            YAHOO_SEARCH_API,
            params={
                "q": name,
                "quotesCount": 8,
                "newsCount": 0,
            },
            timeout=15,
        )

        if response.status_code == 429:
            print("Yahoo search rate limited; skipping auto-discovery.")
            time.sleep(5)
            return None

        response.raise_for_status()
        data = response.json()

        company_upper = normalize_name(name)
        company_tokens = [
            token
            for token in company_upper.split()
            if len(token) >= 4
            and token not in {
                "CORPORATION", "CORP", "INC", "INCORPORATED",
                "COMPANY", "LLC", "LIMITED", "FEDERAL", "SERVICES"
            }
        ]

        for quote in data.get("quotes", []):
            if quote.get("quoteType") != "EQUITY":
                continue

            symbol = quote.get("symbol")
            if not symbol or "." in symbol:
                continue

            yahoo_name = normalize_name(
                quote.get("longname")
                or quote.get("shortname")
                or ""
            )

            overlap = sum(
                token in yahoo_name
                for token in company_tokens
            )

            # Require at least one meaningful overlapping company token.
            if company_tokens and overlap == 0:
                continue

            ticker_cache[company_upper] = symbol.upper()
            print(f"Discovered {name} -> {symbol.upper()}")
            return symbol.upper()

    except Exception as exc:
        print(f"Yahoo ticker search error for {name}: {exc}")

    return None

# ============================================================
# DOWNLOAD RECENT POSITIVE TRANSACTIONS
#
# The BUY scanner is looking for newly awarded money. Negative
# modifications remain represented in the historical database
# and therefore still influence historical factors.
# ============================================================

payload = {
    "filters": {
        "award_amounts": [
            {"lower_bound": MIN_POSITIVE_TRANSACTION}
        ],
        "award_type_codes": ["A", "B", "C", "D"],
        "time_period": [
            {
                "start_date": str(START),
                "end_date": str(TODAY),
            }
        ],
    },
    "fields": [
        "Award ID",
        "Recipient Name",
        "Action Date",
        "Transaction Amount",
        "Awarding Agency",
        "Awarding Sub Agency",
        "Award Type",
    ],
    "sort": "Transaction Amount",
    "order": "desc",
    "limit": PAGE_SIZE,
    "page": 1,
}

all_rows = []

for page in range(1, MAX_PAGES + 1):
    payload["page"] = page

    response = session.post(
        USA_API,
        json=payload,
        timeout=90,
    )
    response.raise_for_status()

    page_data = response.json()
    rows = page_data.get("results", [])

    if not rows:
        break

    all_rows.extend(rows)

    if not page_data.get("page_metadata", {}).get("hasNext", False):
        break

    time.sleep(0.20)

print(
    f"Downloaded {len(all_rows):,} recent transaction records "
    f"({START} through {TODAY})."
)

# ============================================================
# IDENTIFY PUBLIC-COMPANY TRANSACTIONS
# ============================================================

recent_rows = []
unknown = {}

for row in all_rows:
    name = row.get("Recipient Name")
    if not name:
        continue

    try:
        amount = float(row.get("Transaction Amount") or 0)
    except Exception:
        continue

    ticker = find_ticker(name, amount)

    if not ticker:
        norm = normalize_name(name)
        old = unknown.get(norm, {
            "company": name,
            "largest_transaction": 0.0,
            "seen_count": 0,
        })
        old["largest_transaction"] = max(
            old["largest_transaction"],
            abs(amount),
        )
        old["seen_count"] += 1
        unknown[norm] = old
        continue

    agency = row.get("Awarding Agency")
    if isinstance(agency, dict):
        agency = agency.get("toptier_name") or agency.get("name")

    subagency = row.get("Awarding Sub Agency")
    if isinstance(subagency, dict):
        subagency = subagency.get("name") or subagency.get("toptier_name")

    action_date = str(row.get("Action Date") or "")[:10]
    if not action_date:
        continue

    recent_rows.append({
        "ticker": ticker,
        "company": name,
        "award_id": row.get("Award ID"),
        "award_date": action_date,
        "transaction_amount": amount,
        "agency": agency,
        "subagency": subagency,
        "award_type": row.get("Award Type"),
    })

recent = pd.DataFrame(recent_rows)

if recent.empty:
    BUY_FILE.write_text("")
    print("No matched public-company transactions. NO BUYS.")
    TICKER_CACHE_FILE.write_text(json.dumps(ticker_cache, indent=2, sort_keys=True))
    UNKNOWN_FILE.write_text(json.dumps(unknown, indent=2))
    raise SystemExit(0)

recent["award_date"] = pd.to_datetime(recent["award_date"], errors="coerce")
recent = recent.dropna(subset=["award_date"]).copy()

# Deduplicate exact transaction records returned across API pages/runs.
recent["_key"] = (
    recent["ticker"].astype(str)
    + "|"
    + recent["award_date"].dt.strftime("%Y-%m-%d")
    + "|"
    + recent["award_id"].astype(str)
    + "|"
    + recent["transaction_amount"].round(2).astype(str)
)

recent = recent.drop_duplicates("_key").drop(columns="_key")

# ============================================================
# COMBINE SAVED MASTER + PREVIOUS LIVE HISTORY + CURRENT RECENT
# ============================================================

history_parts = [
    master_history[
        [
            "ticker",
            "company",
            "award_id",
            "award_date",
            "transaction_amount",
        ]
    ].copy()
]

if LIVE_HISTORY.exists():
    try:
        live_old = pd.read_csv(LIVE_HISTORY)
        live_old["award_date"] = pd.to_datetime(
            live_old["award_date"], errors="coerce"
        )
        live_old["transaction_amount"] = pd.to_numeric(
            live_old["transaction_amount"], errors="coerce"
        )
        history_parts.append(
            live_old[
                [
                    "ticker",
                    "company",
                    "award_id",
                    "award_date",
                    "transaction_amount",
                ]
            ].copy()
        )
    except Exception as exc:
        print(f"Could not load prior live history: {exc}")

history_parts.append(
    recent[
        [
            "ticker",
            "company",
            "award_id",
            "award_date",
            "transaction_amount",
        ]
    ].copy()
)

history_tx = pd.concat(history_parts, ignore_index=True)

history_tx = history_tx.dropna(
    subset=["ticker", "award_date", "transaction_amount"]
)

history_tx["_dedupe"] = (
    history_tx["ticker"].astype(str)
    + "|"
    + history_tx["award_date"].dt.strftime("%Y-%m-%d")
    + "|"
    + history_tx["award_id"].astype(str)
    + "|"
    + history_tx["transaction_amount"].round(2).astype(str)
)

history_tx = (
    history_tx
    .drop_duplicates("_dedupe")
    .drop(columns="_dedupe")
    .sort_values(["ticker", "award_date"])
    .reset_index(drop=True)
)

# Persist only the post-master live additions, keeping root clean.
recent[
    [
        "ticker",
        "company",
        "award_id",
        "award_date",
        "transaction_amount",
    ]
].to_csv(LIVE_HISTORY, index=False)

# ============================================================
# CURRENT EVENTS = ONE TICKER / AWARD DATE
# ============================================================

events = (
    recent.groupby(["ticker", "award_date"], as_index=False)
    .agg(
        company=("company", "first"),
        same_day_award_count=("award_id", "count"),
        transaction_amount_sum=("transaction_amount", "sum"),
        transaction_amount_abs_sum=(
            "transaction_amount",
            lambda s: pd.to_numeric(s, errors="coerce").abs().sum()
        ),
    )
)

# ============================================================
# MARKET DATA HELPERS
# ============================================================

def normalize_yf(data):
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["Date", "Close", "Volume"])

    d = data.copy()

    if isinstance(d.columns, pd.MultiIndex):
        if "Close" in d.columns.get_level_values(0):
            close = d["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = d.xs("Close", axis=1, level=-1)
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

        if "Volume" in d.columns.get_level_values(0):
            volume = d["Volume"]
            if isinstance(volume, pd.DataFrame):
                volume = volume.iloc[:, 0]
        else:
            try:
                volume = d.xs("Volume", axis=1, level=-1)
                if isinstance(volume, pd.DataFrame):
                    volume = volume.iloc[:, 0]
            except Exception:
                volume = np.nan

        out = pd.DataFrame({"Close": close, "Volume": volume})

    else:
        out = pd.DataFrame({
            "Close": d["Close"] if "Close" in d.columns else np.nan,
            "Volume": d["Volume"] if "Volume" in d.columns else np.nan,
        })

    out = out.dropna(subset=["Close"]).copy()
    out.index = pd.to_datetime(out.index)

    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "Date"})
    out["Date"] = pd.to_datetime(out["Date"])

    return out[["Date", "Close", "Volume"]]


def download_market(ticker, start_date, end_date):
    try:
        data = yf.download(
            ticker,
            start=str(pd.Timestamp(start_date).date()),
            end=str((pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return normalize_yf(data)
    except Exception as exc:
        print(f"Market data error {ticker}: {exc}")
        return pd.DataFrame(columns=["Date", "Close", "Volume"])


def idx_on_or_before(market, event_date):
    if market.empty:
        return None

    dates = market["Date"].values
    i = np.searchsorted(
        dates,
        np.datetime64(pd.Timestamp(event_date)),
        side="right",
    ) - 1

    return int(i) if i >= 0 else None


def trailing_return(market, event_date, sessions):
    i = idx_on_or_before(market, event_date)

    if i is None or i - sessions < 0:
        return np.nan

    p1 = float(market.iloc[i]["Close"])
    p0 = float(market.iloc[i - sessions]["Close"])

    return (p1 / p0 - 1.0) * 100.0 if p0 else np.nan


def trailing_volatility(market, event_date, sessions):
    i = idx_on_or_before(market, event_date)

    if i is None or i - sessions < 1:
        return np.nan

    close = market.iloc[i - sessions:i + 1]["Close"].astype(float)
    ret = close.pct_change().dropna()

    if len(ret) < max(5, sessions // 2):
        return np.nan

    return float(ret.std(ddof=1) * 100.0)


def forward_return(market, event_date, sessions):
    if market.empty:
        return np.nan

    dates = market["Date"].values

    i = np.searchsorted(
        dates,
        np.datetime64(pd.Timestamp(event_date)),
        side="left",
    )

    if i >= len(market) or i + sessions >= len(market):
        return np.nan

    p0 = float(market.iloc[i]["Close"])
    p1 = float(market.iloc[i + sessions]["Close"])

    return (p1 / p0 - 1.0) * 100.0 if p0 else np.nan

# ============================================================
# HISTORICAL AWARD EVENTS
# ============================================================

history_events = (
    history_tx.groupby(["ticker", "award_date"], as_index=False)
    .agg(
        award_count=("award_id", "count"),
        signed_amount=("transaction_amount", "sum"),
        abs_amount=(
            "transaction_amount",
            lambda s: pd.to_numeric(s, errors="coerce").abs().sum()
        ),
    )
    .sort_values(["ticker", "award_date"])
)

# ============================================================
# DOWNLOAD MARKET HISTORY ONLY FOR CURRENT CANDIDATE TICKERS
# ============================================================

candidate_tickers = sorted(events["ticker"].astype(str).unique())

market_start = min(
    history_events["award_date"].min(),
    events["award_date"].min() - pd.Timedelta(days=430),
)

market_end = pd.Timestamp(TODAY) + pd.Timedelta(days=2)

market = {
    "SPY": download_market("SPY", market_start, market_end)
}

for ticker in candidate_tickers:
    market[ticker] = download_market(
        ticker,
        market_start,
        market_end,
    )
    time.sleep(0.15)

# ============================================================
# CALCULATE PRIOR STOCK RESPONSES FOR EACH CURRENT TICKER
# ============================================================

history_response = {}

for ticker in candidate_tickers:
    hist = history_events[
        history_events["ticker"] == ticker
    ].copy()

    mkt = market.get(ticker, pd.DataFrame())

    rows = []

    for _, row in hist.iterrows():
        rows.append({
            "award_date": row["award_date"],
            "return_20d": forward_return(
                mkt, row["award_date"], 20
            ),
            "return_60d": forward_return(
                mkt, row["award_date"], 60
            ),
        })

    history_response[ticker] = pd.DataFrame(rows)

# ============================================================
# FROZEN MTS SCORE
# ============================================================

FEATURES = model["features"]
THRESHOLD = float(model["top5_threshold"])

mu = np.array(
    [model["reference_mean"][c] for c in FEATURES],
    dtype=float,
)

sd = np.array(
    [model["reference_sd"][c] for c in FEATURES],
    dtype=float,
)

location = np.array(model["lw_location"], dtype=float)
precision = np.array(model["lw_precision"], dtype=float)
dimension = float(model["dimension_normalization"])


def frozen_score(feature_values):
    transformed = []

    for c in FEATURES:
        value = feature_values.get(c, np.nan)

        try:
            value = float(value)
        except Exception:
            value = np.nan

        if not np.isfinite(value):
            value = float(model["impute_median"][c])

        if model["transforms"][c] == "signed_log1p":
            value = float(np.sign(value) * np.log1p(abs(value)))

        transformed.append(value)

    z = (np.array(transformed, dtype=float) - mu) / sd
    diff = z - location

    md2 = float(diff @ precision @ diff)

    return float(
        np.sqrt(max(md2 / dimension, 1e-12))
    )

# ============================================================
# SCORE CURRENT EVENTS
# ============================================================

scan_rows = []
new_buys = []

for _, ev in events.sort_values("award_date").iterrows():
    ticker = str(ev["ticker"])
    d = pd.Timestamp(ev["award_date"])

    prior = history_events[
        (history_events["ticker"] == ticker)
        & (history_events["award_date"] < d)
    ].copy()

    if len(prior):
        prior_abs_award_max = float(prior["abs_amount"].max())
        prior_signed_award_mean = float(prior["signed_amount"].mean())
        prior_abs_award_median = float(prior["abs_amount"].median())
    else:
        prior_abs_award_max = np.nan
        prior_signed_award_mean = np.nan
        prior_abs_award_median = np.nan

    w30 = prior[
        prior["award_date"] >= d - pd.Timedelta(days=30)
    ]

    prior_transactions_30d = (
        int(w30["award_count"].sum())
        if len(w30)
        else 0
    )

    prior_award_days_30d = int(len(w30))

    stock = market.get(ticker, pd.DataFrame())
    spy = market.get("SPY", pd.DataFrame())

    pre_volatility_market_20d = trailing_volatility(
        stock, d, 20
    )

    stock60 = trailing_return(stock, d, 60)
    spy60 = trailing_return(spy, d, 60)

    stock120 = trailing_return(stock, d, 120)
    spy120 = trailing_return(spy, d, 120)

    relative_strength_spy_60d = (
        stock60 - spy60
        if pd.notna(stock60) and pd.notna(spy60)
        else np.nan
    )

    relative_strength_spy_120d = (
        stock120 - spy120
        if pd.notna(stock120) and pd.notna(spy120)
        else np.nan
    )

    resp = history_response.get(
        ticker,
        pd.DataFrame(columns=["award_date", "return_20d", "return_60d"])
    )

    # Same no-look-ahead maturity rules as Sample A/B.
    matured20 = resp[
        resp["award_date"] <= d - pd.Timedelta(days=35)
    ]

    vals20 = pd.to_numeric(
        matured20["return_20d"],
        errors="coerce",
    ).dropna()

    prior_response_mean_20d = (
        float(vals20.mean())
        if len(vals20)
        else np.nan
    )

    matured60 = resp[
        resp["award_date"] <= d - pd.Timedelta(days=99)
    ]

    vals60 = pd.to_numeric(
        matured60["return_60d"],
        errors="coerce",
    ).dropna()

    prior_response_count_60d = int(len(vals60))

    feature_values = {
        "prior_response_count_60d": prior_response_count_60d,
        "pre_volatility_market_20d": pre_volatility_market_20d,
        "relative_strength_spy_120d": relative_strength_spy_120d,
        "prior_abs_award_max": prior_abs_award_max,
        "prior_signed_award_mean": prior_signed_award_mean,
        "prior_abs_award_median": prior_abs_award_median,
        "prior_response_mean_20d": prior_response_mean_20d,
        "transaction_amount_abs_sum": float(
            ev["transaction_amount_abs_sum"]
        ),
        "prior_transactions_30d": prior_transactions_30d,
        "relative_strength_spy_60d": relative_strength_spy_60d,
        "prior_award_days_30d": prior_award_days_30d,
    }

    score = frozen_score(feature_values)
    buy = score >= THRESHOLD

    # Fingerprint changes if the same ticker/day receives additional
    # transactions later, allowing a materially changed event to rescore.
    fingerprint = (
        f"{ticker}|{d.date()}|"
        f"{float(ev['transaction_amount_abs_sum']):.2f}|"
        f"{int(ev['same_day_award_count'])}"
    )

    previously_seen = fingerprint in seen

    # Current market price is logged for future sandbox analysis only.
    current_price = np.nan
    if not stock.empty:
        i = idx_on_or_before(stock, pd.Timestamp(TODAY))
        if i is not None:
            current_price = float(stock.iloc[i]["Close"])

    row = {
        "scan_utc": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "company": ev["company"],
        "award_date": str(d.date()),
        "transaction_amount_abs_sum": float(
            ev["transaction_amount_abs_sum"]
        ),
        "same_day_award_count": int(ev["same_day_award_count"]),
        "mts_score": score,
        "threshold": THRESHOLD,
        "signal": "BUY" if buy else "PASS",
        "previously_seen": int(previously_seen),
        "current_price_for_log": current_price,
        **feature_values,
    }

    scan_rows.append(row)

    if buy and not previously_seen:
        new_buys.append(ticker)

    seen[fingerprint] = {
        "first_seen_utc": seen.get(
            fingerprint,
            {}
        ).get(
            "first_seen_utc",
            datetime.now(timezone.utc).isoformat(),
        ),
        "last_score": score,
        "signal": "BUY" if buy else "PASS",
    }

# ============================================================
# USER-FACING OUTPUT: TICKERS ONLY
# ============================================================

new_buys = sorted(set(new_buys))

BUY_FILE.write_text(
    "".join(f"{ticker}\n" for ticker in new_buys)
)

# ============================================================
# INTERNAL AUDIT LOG
# ============================================================

scan_df = pd.DataFrame(scan_rows)

if SCAN_LOG.exists():
    try:
        old_log = pd.read_csv(SCAN_LOG)
        scan_df = pd.concat(
            [old_log, scan_df],
            ignore_index=True,
        )
    except Exception as exc:
        print(f"Could not append prior scan log: {exc}")

scan_df.to_csv(SCAN_LOG, index=False)

SEEN_FILE.write_text(
    json.dumps(seen, indent=2, sort_keys=True)
)

TICKER_CACHE_FILE.write_text(
    json.dumps(ticker_cache, indent=2, sort_keys=True)
)

UNKNOWN_FILE.write_text(
    json.dumps(unknown, indent=2)
)

# ============================================================
# SIMPLE CONSOLE OUTPUT
# ============================================================

print()
print("=" * 60)
print("RECON LIVE BUY SCANNER")
print("=" * 60)

if new_buys:
    print("BUY:")
    for ticker in new_buys:
        print(ticker)
else:
    print("NO BUYS")

print("=" * 60)
print(f"Frozen threshold: {THRESHOLD:.6f}")
print(f"Events scored   : {len(events)}")
print(f"New BUY tickers : {len(new_buys)}")
print(f"Output          : {BUY_FILE}")
print("=" * 60)
