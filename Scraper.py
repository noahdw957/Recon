import json
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================
# RECON LIVE BUY SCANNER V4.3 - MD14 + MD11 + AWARD/IDV WATCH
#
# USER-FACING OUTPUT:
#     buy_tickers.txt
#
# Everything else is stored under ReconData/ for analysis.
#
# FROZEN BUY ARCHITECTURE:
#     PRIMARY   : MD14 >= 1.8729299828283774
#     SECONDARY : MD11 >= threshold from mts_frozen_model_A.json
#     BUY only when BOTH frozen spaces pass.
#
# The former MD8/Gang-of-8 veto is RETIRED from production BUY logic.
# Same-day BUY persistence is preserved across reruns.
#
# The legacy 90-day peak regression is retained in the audit log only.
# It does NOT participate in BUY; it is shown only as informational Pred Gain in buy_tickers.txt.
#
# SELL is intentionally not implemented here yet.  The project state machine is:
# BUY = new MD14+MD11 opportunity; HOLD = normal post-award development;
# SELL = failure bailout or maturity exit once those trajectory rules are frozen.
# ============================================================

LOOKBACK_DAYS = 7
MAX_PAGES = 100
PAGE_SIZE = 100
MIN_NONZERO_TRANSACTION = 1.0

AUTO_DISCOVER_MIN = 1_000_000
MAX_AUTO_LOOKUPS = 30

# USAspending resilience policy.
# On HTTP 503, wait ten minutes and retry. After three delayed retries
# (four total attempts including the initial request), fail CLOSED but exit
# cleanly so GitHub can commit the system-failure consumer message and
# diagnostics.
HTTP_503_WAIT_SECONDS = 600
HTTP_503_DELAYED_RETRIES = 3

TODAY = date.today()


USA_API = (
    "https://api.usaspending.gov/api/v2/"
    "search/spending_by_transaction/"
)

# Award-level shadow watch.  This is deliberately separate from the frozen
# BUY transaction pipeline.  It catches large contract ceilings/frameworks
# and IDVs that may have zero or delayed obligations and therefore never
# appear in the non-zero transaction feed used by MD14/MD11.
USA_AWARD_API = (
    "https://api.usaspending.gov/api/v2/"
    "search/spending_by_award/"
)
PROCUREMENT_AWARD_TYPE_CODES = [
    "A", "B", "C", "D",
    "IDV_A", "IDV_B", "IDV_B_A", "IDV_B_B", "IDV_B_C",
    "IDV_C", "IDV_D", "IDV_E",
]
AWARD_WATCH_MIN_DISPLAY_VALUE = 50_000_000.0
# V4.3: treat HTTP 422 validation failures like 400 and step down award-watch fields.

YAHOO_SEARCH_API = (
    "https://query2.finance.yahoo.com/v1/finance/search"
)

# ============================================================
# REQUIRED EXISTING FILES
# ============================================================

MODEL_FILE = Path("mts_frozen_model_A.json")
MD14_MODEL_FILE = Path("mts_frozen_model_MD14_A.json")
MASTER_HISTORY_FILE = Path("master_zero_purged.csv")

# ============================================================
# CLEAN OUTPUT / STATE AREA
# ============================================================

DATA_DIR = Path("ReconData")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BUY_FILE = Path("buy_tickers.txt")

SCAN_LOG = DATA_DIR / "recon_scan_log.csv"
DAILY_DIAG = DATA_DIR / "daily_diagnostics.csv"
DIAG_LATEST = DATA_DIR / "diagnostic_latest.json"
LIVE_HISTORY = DATA_DIR / "live_history.csv"
SEEN_FILE = DATA_DIR / "seen_events.json"
TICKER_CACHE_FILE = DATA_DIR / "ticker_cache.json"
UNKNOWN_FILE = DATA_DIR / "unknown_companies.json"
ACQ_STATE_FILE = DATA_DIR / "acquisition_state.json"
DAILY_BUY_STATE_FILE = DATA_DIR / "daily_buy_state.json"
AWARD_WATCH_LOG = DATA_DIR / "award_watch.csv"
AWARD_WATCH_SEEN_FILE = DATA_DIR / "seen_award_watch.json"
DAILY_AWARD_WATCH_STATE_FILE = DATA_DIR / "daily_award_watch_state.json"
SHARES_CACHE_DIR = DATA_DIR / "shares_cache"
SHARES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

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


# Incremental acquisition watermark.
# Normal operation rescans only one overlap day plus today. If a prior run
# failed, catch up only the missing dates. A seven-day cap is retained as a
# recovery guardrail, not as the normal daily lookback.
try:
    if ACQ_STATE_FILE.exists():
        _acq_state = json.loads(ACQ_STATE_FILE.read_text())
    else:
        _acq_state = {}
except Exception:
    _acq_state = {}
_last_success = _acq_state.get("last_successful_scan_date")

if _last_success:
    try:
        _last_success_date = date.fromisoformat(str(_last_success))
    except Exception:
        _last_success_date = None
else:
    _last_success_date = None

# If no explicit acquisition watermark exists yet, use the newest locally
# persisted live award date as a bootstrap hint. This prevents the first V3.4
# run from needlessly re-querying a full week when the prior successful
# scanner already captured recent transactions.
if _last_success_date is None and LIVE_HISTORY.exists():
    try:
        _lh = pd.read_csv(LIVE_HISTORY, usecols=["award_date"])
        _lh["award_date"] = pd.to_datetime(_lh["award_date"], errors="coerce")
        if _lh["award_date"].notna().any():
            _last_success_date = _lh["award_date"].max().date()
    except Exception:
        _last_success_date = None

if _last_success_date is None:
    START = TODAY - timedelta(days=1)
    WATERMARK_SOURCE = "bootstrap_yesterday"
else:
    # Re-fetch one overlap day to catch late postings/revisions.
    START = _last_success_date - timedelta(days=1)
    WATERMARK_SOURCE = "acquisition_state_or_live_history"

# Never ask for more than the recovery cap automatically.
_recovery_floor = TODAY - timedelta(days=LOOKBACK_DAYS)
if START < _recovery_floor:
    START = _recovery_floor
    WATERMARK_SOURCE += "_capped_7d"

if START > TODAY:
    START = TODAY


def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        print(f"Could not read {path}: {exc}")
    return default


def normalize_name(name):
    return " ".join(str(name).upper().replace(",", " ").split())


def sector_group(ticker, company=""):
    t = str(ticker).upper().strip()
    c = str(company).upper()

    aero = {
        "AVAV", "BA", "KTOS", "LHX", "LMT", "NOC", "GD", "HII",
        "RTX", "TXT", "RKLB", "SATL", "RDW", "BKSY", "PLTR",
        "LDOS", "SAIC", "BAH", "VSEC", "WWD",
    }
    industrial = {"GE", "CAT", "ETN", "HON"}
    tech = {"IBM", "ACN"}

    if t in aero:
        return "AERO_DEFENSE"
    if t in industrial:
        return "INDUSTRIAL"
    if t in tech:
        return "TECH_SERVICES"

    defense_words = [
        "AEROSPACE", "AEROVIRONMENT", "BOEING", "KRATOS", "DEFENSE",
        "DEFENCE", "DYNAMICS", "INGALLS", "RAYTHEON", "LOCKHEED",
        "NORTHROP", "LEIDOS", "BOOZ ALLEN", "ROCKET LAB", "SATELLITE",
        "SPACE", "VSE", "WOODWARD",
    ]
    if any(word in c for word in defense_words):
        return "AERO_DEFENSE"

    return "OTHER"


def signed_log1p_scalar(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(value):
        return np.nan
    return float(np.sign(value) * np.log1p(abs(value)))


session = requests.Session()
session.headers.update({"User-Agent": "RECON-Live/2.0"})

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Missing {MODEL_FILE}. Upload the frozen MD11 Sample-A model."
    )

if not MD14_MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Missing {MD14_MODEL_FILE}. Upload the frozen MD14 Sample-A model."
    )

if not MASTER_HISTORY_FILE.exists():
    raise FileNotFoundError(
        f"Missing {MASTER_HISTORY_FILE}. Upload the saved nonzero award master."
    )

model = json.loads(MODEL_FILE.read_text())
model14 = json.loads(MD14_MODEL_FILE.read_text())

MODEL_VERSION = "RECON_MD14_PRIMARY_MD11_SECONDARY_V1"
SCANNER_VERSION = "RECON_LIVE_V4.3_MD14_MD11_AWARD_WATCH_422FIX"
MD14_THRESHOLD = float(model14["threshold"])

# Informational 90-trading-day peak projection.
# Fit previously on the 616-award Sample-C population (R^2 ~= 0.38).
# IMPORTANT: this projection does NOT participate in the BUY decision.
# Expected Peak % = 111.0 + 8.8*MD11 - 4.8*ln(MarketCap) + 7.1*preVol_20d
PEAK_PROJ_INTERCEPT = 111.0
PEAK_PROJ_MD11_COEF = 8.8
PEAK_PROJ_LN_MCAP_COEF = -4.8
PEAK_PROJ_PREVOL_COEF = 7.1
PEAK_PROJ_NAME = "C616_OLS_MD11_LNMCAP_PREVOL_R2_0.38"


# Same-day consumer persistence.  A later rerun may legitimately download no
# new USAspending rows, but that must not erase BUYs already discovered today.
def _projection_from_values(md11_score, market_cap_before, pre_volatility_market_20d):
    try:
        md = float(md11_score)
        mcap = float(market_cap_before)
        prevol = float(pre_volatility_market_20d)
    except Exception:
        return None
    if not (np.isfinite(md) and np.isfinite(mcap) and np.isfinite(prevol)) or mcap <= 0:
        return None
    return float(
        PEAK_PROJ_INTERCEPT
        + PEAK_PROJ_MD11_COEF * md
        + PEAK_PROJ_LN_MCAP_COEF * np.log(mcap)
        + PEAK_PROJ_PREVOL_COEF * prevol
    )


def load_daily_buy_state():
    """Load today's retained BUY events; bootstrap from today's scan log if needed."""
    state = load_json(DAILY_BUY_STATE_FILE, {})
    if str(state.get("date")) == str(TODAY) and isinstance(state.get("events"), dict):
        return state

    events_state = {}
    if SCAN_LOG.exists():
        try:
            log = pd.read_csv(SCAN_LOG)
            if len(log) and {"ticker", "award_date", "signal", "scan_utc"}.issubset(log.columns):
                utc_day = pd.to_datetime(log["scan_utc"], errors="coerce", utc=True).dt.date
                today_log = log[utc_day == TODAY].copy()
                if len(today_log):
                    today_log["_scan_dt"] = pd.to_datetime(today_log["scan_utc"], errors="coerce", utc=True)
                    today_log = today_log.sort_values("_scan_dt")
                    # Latest observation of each ticker/award-date event governs that event.
                    latest = today_log.groupby(["ticker", "award_date"], as_index=False).tail(1)
                    latest = latest[latest["signal"].astype(str).str.upper() == "BUY"]
                    for _, r in latest.iterrows():
                        ticker = str(r["ticker"]).upper().strip()
                        award_date = str(r["award_date"])[:10]
                        key = f"{ticker}|{award_date}"
                        proj = r.get("projected_peak_pct_90d", np.nan)
                        if pd.isna(proj):
                            proj = _projection_from_values(
                                r.get("md11_score", np.nan),
                                r.get("market_cap_before", np.nan),
                                r.get("pre_volatility_market_20d", np.nan),
                            )
                        events_state[key] = {
                            "ticker": ticker,
                            "award_date": award_date,
                            "projected_peak_pct_90d": None if proj is None or pd.isna(proj) else float(proj),
                            "last_seen_utc": str(r.get("scan_utc", "")),
                        }
        except Exception as exc:
            print(f"Could not bootstrap today's BUY state from scan log: {exc}")

    state = {"date": str(TODAY), "events": events_state}
    DAILY_BUY_STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))
    return state


def save_daily_buy_state(state):
    state["date"] = str(TODAY)
    DAILY_BUY_STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def current_buy_rows(state):
    """Return retained BUY events for today as one ledger row per ticker/award date."""
    rows = []
    for ev in state.get("events", {}).values():
        ticker = str(ev.get("ticker", "")).upper().strip()
        award_date = str(ev.get("award_date", ""))[:10]
        if not ticker:
            continue
        rows.append({
            "ticker": ticker,
            "award_date": award_date,
            "projected_peak_pct_90d": ev.get("projected_peak_pct_90d"),
        })
    rows.sort(key=lambda r: (r["award_date"], r["ticker"]))
    return rows


def compact_date(value):
    try:
        return pd.Timestamp(value).strftime("%d%b%y").upper()
    except Exception:
        return str(value)


def format_consumer_buy_line(ticker, award_date, projection=None):
    pred = (
        f"{float(projection):+.0f}%"
        if projection is not None and pd.notna(projection)
        else ""
    )
    return (
        f"{ticker}\t"
        f"{compact_date(award_date)}\t"
        f"{pred}\t"
        f"\t"      # Sell Price
        f"\t"      # Sell Date
        f"\n"      # %Gain blank until SELL
    )


def write_current_buy_file(state):
    rows = current_buy_rows(state)
    header_date = pd.Timestamp(TODAY).strftime("%d%b%y").upper()

    if not rows:
        BUY_FILE.write_text(
            f"## RECON BUY TICKERS {header_date}\n\n"
            "NO BUYS\n"
        )
        return [], {}

    lines = [
        f"## RECON BUY TICKERS {header_date}\n\n",
        "Ticker\tAward Date\tPred Gain\tSell Price\tSell Date\t%Gain\n",
    ]
    lines.extend(
        format_consumer_buy_line(
            r["ticker"],
            r["award_date"],
            r.get("projected_peak_pct_90d"),
        )
        for r in rows
    )
    BUY_FILE.write_text("".join(lines))

    tickers = sorted({r["ticker"] for r in rows})
    projection_map = {}
    for r in rows:
        t = r["ticker"]
        p = r.get("projected_peak_pct_90d")
        if t not in projection_map:
            projection_map[t] = p
        elif p is not None and (projection_map[t] is None or float(p) > float(projection_map[t])):
            projection_map[t] = p
    return tickers, projection_map


def load_daily_award_watch_state():
    state = load_json(DAILY_AWARD_WATCH_STATE_FILE, {})
    if str(state.get("date")) == str(TODAY) and isinstance(state.get("events"), dict):
        return state
    return {"date": str(TODAY), "events": {}}


def save_daily_award_watch_state(state):
    state["date"] = str(TODAY)
    DAILY_AWARD_WATCH_STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True))


def _money(value):
    try:
        value = float(value)
    except Exception:
        return ""
    if not np.isfinite(value):
        return ""
    a = abs(value)
    if a >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def current_award_watch_rows(state):
    rows = list(state.get("events", {}).values())
    rows.sort(key=lambda r: (str(r.get("ticker", "")), str(r.get("award_id", ""))))
    return rows


def append_award_watch_to_buy_file(state):
    rows = current_award_watch_rows(state)
    if not rows:
        return
    with BUY_FILE.open("a") as f:
        f.write("\n## LARGE / FRAMEWORK AWARD WATCH - NOT BUY-SCORED\n")
        f.write("Ticker\tAward ID\tObligated\tPotential/Ceiling\tType\tDetected\n")
        for r in rows:
            f.write(
                f"{r.get('ticker','')}\t"
                f"{r.get('award_id','')}\t"
                f"{_money(r.get('award_amount'))}\t"
                f"{_money(r.get('potential_amount'))}\t"
                f"{r.get('award_type','')}\t"
                f"{compact_date(r.get('detected_date', TODAY))}\n"
            )
        f.write("NOTE: watch-only. MD14/MD11 remain transaction-trained and unchanged.\n")


DAILY_BUY_STATE = load_daily_buy_state()
DAILY_AWARD_WATCH_STATE = load_daily_award_watch_state()

seen = load_json(SEEN_FILE, {})
award_watch_seen = load_json(AWARD_WATCH_SEEN_FILE, {})
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
# DOWNLOAD RECENT NONZERO TRANSACTIONS -- RECIPIENT SCOPED
#
# IMPORTANT:
# The historical event-study abandoned the broad federal firehose because
# the USAspending transaction search can exceed a practical 10,000-row
# result ceiling. Live V3.2 uses the same solution: query by recipient.
#
# Recipient aliases are learned from master_zero_purged.csv and grouped by
# ticker. Each ticker is queried twice over the live lookback: positive and
# negative non-zero transactions. LMT is omitted because the production
# universe is non-LMT.
# ============================================================

BASE_FIELDS = [
    'Award ID',
    'Recipient Name',
    'Action Date',
    'Transaction Amount',
    'Awarding Agency',
    'Awarding Sub Agency',
    'Award Type',
]

recipient_aliases = {}
for ticker, g in master_history.groupby("ticker"):
    ticker = str(ticker).upper()
    if ticker == "LMT":
        continue

    aliases = []

    # Prefer the compact curated keywords used successfully in historical
    # acquisition. Broad curated terms capture subsidiaries while minimizing
    # API calls.
    if ticker in MASTER and MASTER[ticker]:
        aliases.extend(MASTER[ticker])
    else:
        # For tickers without a curated keyword, use up to three most-common
        # exact historical recipient names.
        vc = (
            g["company"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .dropna()
            .value_counts()
        )
        aliases.extend(vc.head(3).index.tolist())

    seen_norm = set()
    cleaned = []
    for alias in aliases:
        n = normalize_name(alias)
        if not n or n in seen_norm:
            continue
        seen_norm.add(n)
        cleaned.append(alias)

    if cleaned:
        recipient_aliases[ticker] = cleaned


def fetch_alias_sign(ticker, alias, sign):
    if sign == 'positive':
        amount_filter = {'lower_bound': MIN_NONZERO_TRANSACTION}
        order = 'desc'
    else:
        amount_filter = {'upper_bound': -MIN_NONZERO_TRANSACTION}
        order = 'asc'

    payload = {
        'filters': {
            'recipient_search_text': [alias],
            'award_amounts': [amount_filter],
            'award_type_codes': ['A', 'B', 'C', 'D'],
            'time_period': [{
                'start_date': str(START),
                'end_date': str(TODAY),
            }],
        },
        'fields': BASE_FIELDS,
        'sort': 'Transaction Amount',
        'order': order,
        'limit': PAGE_SIZE,
        'page': 1,
    }

    rows_out = []

    for page in range(1, MAX_PAGES + 1):
        payload['page'] = page
        page_data = None

        # Explicit 503 recovery policy. A 503 means the upstream service
        # is unavailable; do not hammer it. Wait ten minutes, then retry,
        # up to three delayed retries. Other transient HTTP failures retain
        # a short backoff. Persistent failure is raised to the top-level
        # acquisition guard, which writes SYSTEM FAILURE and exits cleanly.
        page_data = None
        last_exc = None
        delayed_503_retries = 0
        short_retry = 0

        while True:
            try:
                response = session.post(
                    USA_API,
                    json=payload,
                    timeout=90,
                )

                if response.status_code == 503:
                    if delayed_503_retries >= HTTP_503_DELAYED_RETRIES:
                        raise RuntimeError(
                            f'USAspending HTTP 503 persisted after '
                            f'{HTTP_503_DELAYED_RETRIES} delayed retries'
                        )

                    delayed_503_retries += 1
                    print(
                        f'USAspending 503 for {ticker} / {alias!r} / '
                        f'{sign} / page {page}. '
                        f'Waiting {HTTP_503_WAIT_SECONDS // 60} minutes '
                        f'before retry {delayed_503_retries}/'
                        f'{HTTP_503_DELAYED_RETRIES}...'
                    )
                    time.sleep(HTTP_503_WAIT_SECONDS)
                    continue

                response.raise_for_status()
                page_data = response.json()
                last_exc = None
                break

            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                short_retry += 1
                if short_retry > 3:
                    break
                wait_s = [5, 15, 30][short_retry - 1]
                print(
                    f'USAspending request error for {ticker} / {alias!r} / '
                    f'{sign} / page {page}: {exc}. '
                    f'Retrying in {wait_s}s...'
                )
                time.sleep(wait_s)

        if page_data is None:
            raise RuntimeError(
                f'USAspending failed after retries for {ticker} / {alias!r} '
                f'/ {sign} / page {page}: {last_exc}'
            )

        rows = page_data.get('results', [])
        if not rows:
            break

        for row in rows:
            row = dict(row)
            row['_query_ticker'] = ticker
            row['_query_alias'] = alias
            rows_out.append(row)

        has_next = bool(
            page_data.get('page_metadata', {}).get('hasNext', False)
        )
        if not has_next:
            break

        if page == MAX_PAGES:
            raise RuntimeError(
                f'SCAN INCOMPLETE: recipient query {ticker}/{alias!r} '
                f'({sign}) exceeded {MAX_PAGES * PAGE_SIZE:,} rows.'
            )

        time.sleep(0.20)

    return rows_out


all_rows = []
downloaded_positive = 0
downloaded_negative = 0
recipient_queries = 0
alias_queries_failed = 0
SYSTEM_FAILURE = None

try:
    for n, ticker in enumerate(sorted(recipient_aliases), 1):
        pos_n = 0
        neg_n = 0

        for alias in recipient_aliases[ticker]:
            pos = fetch_alias_sign(ticker, alias, 'positive')
            recipient_queries += 1
            neg = fetch_alias_sign(ticker, alias, 'negative')
            recipient_queries += 1

            pos_n += len(pos)
            neg_n += len(neg)
            downloaded_positive += len(pos)
            downloaded_negative += len(neg)
            all_rows.extend(pos)
            all_rows.extend(neg)

        print(
            f'[{n}/{len(recipient_aliases)}] {ticker}: '
            f'+{pos_n} / -{neg_n} across {len(recipient_aliases[ticker])} aliases'
        )

    # Alias overlap can return the same transaction more than once.
    raw_unique = {}
    for row in all_rows:
        key = (
            str(row.get('Award ID')),
            str(row.get('Action Date'))[:10],
            str(row.get('Recipient Name')),
            str(row.get('Transaction Amount')),
        )
        raw_unique[key] = row
    all_rows = list(raw_unique.values())

    print(
        f'Recipient-scoped download: {len(all_rows):,} unique nonzero rows '
        f'({downloaded_positive:,} positive responses, '
        f'{downloaded_negative:,} negative responses before alias dedupe) '
        f'from {len(recipient_aliases):,} tickers / {recipient_queries:,} queries '
        f'({START} through {TODAY}).'
    )

except Exception as exc:
    SYSTEM_FAILURE = str(exc)
    now_utc = datetime.now(timezone.utc).isoformat()

    # Consumer output: unmistakably not a trading signal.
    BUY_FILE.write_text(
        "SYSTEM FAILURE - NO SIGNAL GENERATED\\n"
        "USAspending acquisition incomplete\\n"
    )

    diag = {
        "scan_utc": now_utc,
        "scan_start": str(START),
        "scan_end": str(TODAY),
        "watermark_source": WATERMARK_SOURCE,
        "downloaded_rows_partial": int(len(all_rows)),
        "downloaded_positive_partial": int(downloaded_positive),
        "downloaded_negative_partial": int(downloaded_negative),
        "recipient_universe_tickers": int(len(recipient_aliases)),
        "recipient_queries_completed": int(recipient_queries),
        "status": "SYSTEM_FAILURE_USASPENDING",
        "error": SYSTEM_FAILURE,
        "watermark_advanced": 0,
    }

    DIAG_LATEST.write_text(json.dumps(diag, indent=2))
    pd.DataFrame([diag]).to_csv(
        DAILY_DIAG,
        mode="a",
        header=not DAILY_DIAG.exists(),
        index=False,
    )

    print("SYSTEM FAILURE - NO SIGNAL GENERATED")
    print(SYSTEM_FAILURE)
    print("Acquisition watermark NOT advanced; next run will catch up.")

    # Preserve caches/unknowns if already available, but do not modify seen
    # signal state or acquisition watermark.
    TICKER_CACHE_FILE.write_text(
        json.dumps(ticker_cache, indent=2, sort_keys=True)
    )
    UNKNOWN_FILE.write_text(json.dumps({}, indent=2))

    raise SystemExit(0)

# ============================================================
# AWARD-LEVEL SHADOW WATCH -- LARGE CONTRACTS + IDVs / FRAMEWORKS
# ============================================================
# IMPORTANT: These rows NEVER enter history_tx, events, MD14, or MD11.  The
# frozen BUY models were developed on non-zero transaction events.  Award-level
# totals and potential ceilings are different quantities and mixing them into
# the model would be leakage/semantic drift.  This path is acquisition coverage
# and diagnostics only.

AWARD_WATCH_FIELDS_RICH = [
    "Award ID", "Recipient Name", "Start Date", "End Date",
    "Award Amount", "Potential Award Amount", "Awarding Agency",
    "Awarding Sub Agency", "Contract Award Type", "Description", "Signed Date",
]
# Preserve ceiling/type information on the first fallback.  Only fall all the
# way back to the minimal set if the current USAspending deployment rejects
# one of these core procurement fields.
AWARD_WATCH_FIELDS_CEILING = [
    "Award ID", "Recipient Name", "Start Date", "End Date",
    "Award Amount", "Potential Award Amount", "Awarding Agency",
    "Awarding Sub Agency", "Contract Award Type",
]
AWARD_WATCH_FIELDS_MIN = [
    "Award ID", "Recipient Name", "Start Date", "End Date",
    "Award Amount", "Awarding Agency", "Awarding Sub Agency",
]


def fetch_award_watch(ticker, alias, date_type):
    fields = list(AWARD_WATCH_FIELDS_RICH)
    payload = {
        "filters": {
            "recipient_search_text": [alias],
            "award_type_codes": PROCUREMENT_AWARD_TYPE_CODES,
            "time_period": [{
                "start_date": str(START),
                "end_date": str(TODAY),
                "date_type": date_type,
            }],
        },
        "fields": fields,
        "sort": "Award Amount",
        "order": "desc",
        "limit": PAGE_SIZE,
        "page": 1,
        "subawards": False,
    }

    rows_out = []
    field_mode = 0  # 0=rich, 1=ceiling-focused, 2=minimal

    for page in range(1, MAX_PAGES + 1):
        payload["page"] = page
        page_data = None
        last_exc = None
        delayed_503_retries = 0
        short_retry = 0

        while True:
            try:
                response = session.post(USA_AWARD_API, json=payload, timeout=90)

                # USAspending field support has changed over time.  If a rich
                # optional field is rejected, retry the request with the small
                # stable field set rather than losing the whole watch layer.
                if response.status_code in (400, 422) and field_mode < 2:
                    # USAspending currently uses 422 as well as 400 for
                    # request-schema / unsupported-field validation failures.
                    # Step down the field set before declaring the shadow watch dead.
                    print(
                        f"USAspending award-watch validation {response.status_code} for "
                        f"{ticker} / {alias!r} / {date_type} / page {page}; "
                        f"falling back from field mode {field_mode}. "
                        f"Body: {response.text[:500]}"
                    )
                    field_mode += 1
                    payload["fields"] = list(
                        AWARD_WATCH_FIELDS_CEILING if field_mode == 1
                        else AWARD_WATCH_FIELDS_MIN
                    )
                    continue

                if response.status_code in (400, 422):
                    raise RuntimeError(
                        f"USAspending award-watch validation {response.status_code} after "
                        f"minimal-field fallback for {ticker} / {alias!r} / "
                        f"{date_type} / page {page}: {response.text[:1500]}"
                    )

                if response.status_code == 503:
                    if delayed_503_retries >= HTTP_503_DELAYED_RETRIES:
                        raise RuntimeError(
                            f"USAspending award-watch HTTP 503 persisted after "
                            f"{HTTP_503_DELAYED_RETRIES} delayed retries"
                        )
                    delayed_503_retries += 1
                    print(
                        f"USAspending award-watch 503 for {ticker} / {alias!r} / "
                        f"{date_type} / page {page}. Waiting "
                        f"{HTTP_503_WAIT_SECONDS // 60} minutes before retry "
                        f"{delayed_503_retries}/{HTTP_503_DELAYED_RETRIES}..."
                    )
                    time.sleep(HTTP_503_WAIT_SECONDS)
                    continue

                response.raise_for_status()
                page_data = response.json()
                last_exc = None
                break

            except RuntimeError:
                raise
            except Exception as exc:
                last_exc = exc
                short_retry += 1
                if short_retry > 3:
                    break
                wait_s = [5, 15, 30][short_retry - 1]
                print(
                    f"USAspending award-watch request error for {ticker} / {alias!r} / "
                    f"{date_type} / page {page}: {exc}. Retrying in {wait_s}s..."
                )
                time.sleep(wait_s)

        if page_data is None:
            raise RuntimeError(
                f"USAspending award-watch failed after retries for {ticker} / "
                f"{alias!r} / {date_type} / page {page}: {last_exc}"
            )

        rows = page_data.get("results", [])
        if not rows:
            break

        for row in rows:
            row = dict(row)
            row["_query_ticker"] = ticker
            row["_query_alias"] = alias
            row["_date_type"] = date_type
            rows_out.append(row)

        meta = page_data.get("page_metadata", {}) or {}
        has_next = bool(meta.get("hasNext", meta.get("has_next", False)))
        if not has_next:
            break
        if page == MAX_PAGES:
            raise RuntimeError(
                f"SCAN INCOMPLETE: award-watch query {ticker}/{alias!r} "
                f"({date_type}) exceeded {MAX_PAGES * PAGE_SIZE:,} rows."
            )
        time.sleep(0.20)

    return rows_out


def _finite_number(value):
    try:
        value = float(value)
    except Exception:
        return np.nan
    return value if np.isfinite(value) else np.nan


def _looks_like_framework(award_type):
    s = normalize_name(award_type)
    words = (
        "INDEFINITE", "IDIQ", "IDC", "GWAC", "BASIC ORDERING",
        "BLANKET PURCHASE", "FEDERAL SUPPLY SCHEDULE", "FSS",
    )
    return any(w in s for w in words)


award_watch_rows_raw = []
award_watch_queries = 0
award_watch_status = "OK"
award_watch_error = None

try:
    # Two date lenses close an important gap: action_date catches a fresh
    # transaction/change, while last_modified_date catches an old framework
    # whose ceiling/terms were revised or formally posted later.
    for ticker in sorted(recipient_aliases):
        for alias in recipient_aliases[ticker]:
            for date_type in ("action_date", "last_modified_date"):
                award_watch_rows_raw.extend(fetch_award_watch(ticker, alias, date_type))
                award_watch_queries += 1
except Exception as exc:
    # Do not destroy a valid BUY scan because the supplemental watch failed,
    # but make the loss of coverage impossible to miss.
    award_watch_status = "FAILED"
    award_watch_error = str(exc)
    print(f"WARNING: award/framework watch failed: {award_watch_error}")

# Alias/date-lens overlap can return the same award repeatedly.
award_watch_unique = {}
for row in award_watch_rows_raw:
    key = (
        str(row.get("Award ID")),
        str(row.get("Recipient Name")),
        str(row.get("Award Amount")),
        str(row.get("Potential Award Amount")),
        str(row.get("Contract Award Type") or row.get("Award Type")),
    )
    existing = award_watch_unique.get(key)
    if existing is None:
        row["_date_types_seen"] = {str(row.get("_date_type"))}
        award_watch_unique[key] = row
    else:
        existing.setdefault("_date_types_seen", set()).add(str(row.get("_date_type")))

award_watch_records = []
new_award_watch_alerts = 0
now_utc_watch = datetime.now(timezone.utc).isoformat()

for row in award_watch_unique.values():
    name = row.get("Recipient Name")
    if not name:
        continue
    query_ticker = str(row.get("_query_ticker") or "").upper().strip()
    mapped_ticker = master_lookup(name)
    if mapped_ticker and query_ticker and mapped_ticker != query_ticker:
        continue
    ticker = mapped_ticker or query_ticker
    if not ticker or ticker == "LMT":
        continue

    award_amount = _finite_number(row.get("Award Amount"))
    potential_amount = _finite_number(row.get("Potential Award Amount"))
    award_type = str(row.get("Contract Award Type") or row.get("Award Type") or "")
    display_value = max(
        abs(award_amount) if np.isfinite(award_amount) else 0.0,
        abs(potential_amount) if np.isfinite(potential_amount) else 0.0,
    )
    framework_like = _looks_like_framework(award_type)

    # We log every award-level result internally.  Consumer-facing alerts are
    # limited to framework-like awards or very large values to avoid drowning
    # the actual BUY ledger in ordinary award-level duplicates.
    is_alert = framework_like or display_value >= AWARD_WATCH_MIN_DISPLAY_VALUE

    rec = {
        "scan_utc": now_utc_watch,
        "ticker": ticker,
        "recipient": name,
        "award_id": str(row.get("Award ID") or ""),
        "detected_date": str(TODAY),
        "date_match_modes": ",".join(sorted(row.get("_date_types_seen", {str(row.get('_date_type',''))}))),
        "signed_date": str(row.get("Signed Date") or "")[:10],
        "start_date": str(row.get("Start Date") or "")[:10],
        "end_date": str(row.get("End Date") or "")[:10],
        "award_amount": None if not np.isfinite(award_amount) else float(award_amount),
        "potential_amount": None if not np.isfinite(potential_amount) else float(potential_amount),
        "award_type": award_type,
        "agency": row.get("Awarding Agency"),
        "subagency": row.get("Awarding Sub Agency"),
        "description": row.get("Description") or row.get("Contract Description"),
        "framework_like": int(framework_like),
        "consumer_alert": int(is_alert),
        "buy_scored": 0,
        "reason": "AWARD_LEVEL_SHADOW_WATCH_NOT_FROZEN_TRANSACTION_MODEL",
        "scanner_version": SCANNER_VERSION,
    }
    award_watch_records.append(rec)

    fingerprint = (
        f"{ticker}|{rec['award_id']}|{rec['award_amount']}|{rec['potential_amount']}|"
        f"{rec['award_type']}|{rec['signed_date']}|{rec['start_date']}|{rec['end_date']}"
    )

    if is_alert and fingerprint not in award_watch_seen:
        state_key = f"{ticker}|{rec['award_id']}"
        DAILY_AWARD_WATCH_STATE.setdefault("events", {})[state_key] = {
            "ticker": ticker,
            "award_id": rec["award_id"],
            "award_amount": rec["award_amount"],
            "potential_amount": rec["potential_amount"],
            "award_type": rec["award_type"],
            "detected_date": rec["detected_date"],
            "description": rec["description"],
            "fingerprint": fingerprint,
        }
        new_award_watch_alerts += 1

    award_watch_seen[fingerprint] = {
        "first_seen_utc": award_watch_seen.get(fingerprint, {}).get("first_seen_utc", now_utc_watch),
        "last_seen_utc": now_utc_watch,
        "scanner_version": SCANNER_VERSION,
    }

if award_watch_records:
    watch_df = pd.DataFrame(award_watch_records)
    if AWARD_WATCH_LOG.exists():
        try:
            old_watch = pd.read_csv(AWARD_WATCH_LOG)
            watch_df = pd.concat([old_watch, watch_df], ignore_index=True)
        except Exception as exc:
            print(f"Could not append prior award watch log: {exc}")
    watch_df.to_csv(AWARD_WATCH_LOG, index=False)

save_daily_award_watch_state(DAILY_AWARD_WATCH_STATE)
AWARD_WATCH_SEEN_FILE.write_text(json.dumps(award_watch_seen, indent=2, sort_keys=True))


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

    query_ticker = str(row.get('_query_ticker') or '').upper().strip()
    mapped_ticker = master_lookup(name)

    if mapped_ticker and query_ticker and mapped_ticker != query_ticker:
        norm = normalize_name(name)
        old = unknown.get(norm, {
            'company': name,
            'largest_transaction': 0.0,
            'seen_count': 0,
            'reason': 'recipient_query_cross_match',
        })
        old['largest_transaction'] = max(
            old['largest_transaction'],
            abs(amount),
        )
        old['seen_count'] += 1
        unknown[norm] = old
        continue

    ticker = mapped_ticker or query_ticker

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
    # Preserve BUYs already discovered earlier today.  An empty incremental
    # fetch means "no new rows", not "today has no BUYs".
    active_buys, active_buy_projections = write_current_buy_file(DAILY_BUY_STATE)
    append_award_watch_to_buy_file(DAILY_AWARD_WATCH_STATE)
    now_utc = datetime.now(timezone.utc).isoformat()
    diag = {
        "scan_utc": now_utc,
        "scan_start": str(START),
        "scan_end": str(TODAY),
        "watermark_source": WATERMARK_SOURCE,
        "downloaded_rows_total": int(len(all_rows)),
        "downloaded_positive": int(downloaded_positive),
        "downloaded_negative": int(downloaded_negative),
        "recipient_universe_tickers": int(len(recipient_aliases)),
        "recipient_queries": int(recipient_queries),
        "recipient_aliases": int(sum(len(v) for v in recipient_aliases.values())),
        "matched_public_company_transactions": 0,
        "ticker_day_events": 0,
        "md14_passes": 0,
        "md11_passes": 0,
        "intersection_passes": 0,
        "new_buy_tickers": 0,
        "active_buy_tickers": int(len(active_buys)),
        "market_cap_missing": 0,
        "award_watch_status": award_watch_status,
        "award_watch_error": award_watch_error,
        "award_watch_queries": int(award_watch_queries),
        "award_watch_rows": int(len(award_watch_records)),
        "new_award_watch_alerts": int(new_award_watch_alerts),
        "active_award_watch_alerts": int(len(current_award_watch_rows(DAILY_AWARD_WATCH_STATE))),
        "status": "NO_MATCHED_PUBLIC_COMPANY_TRANSACTIONS",
    }
    DIAG_LATEST.write_text(json.dumps(diag, indent=2))
    pd.DataFrame([diag]).to_csv(
        DAILY_DIAG,
        mode="a",
        header=not DAILY_DIAG.exists(),
        index=False,
    )
    if active_buys:
        print(f"No new matched transactions; carrying forward {len(active_buys)} BUY(s) found earlier today.")
        for ticker in active_buys:
            matching = [r for r in current_buy_rows(DAILY_BUY_STATE) if r["ticker"] == ticker]
            award_date = matching[-1]["award_date"] if matching else ""
            projection = matching[-1].get("projected_peak_pct_90d") if matching else None
            print(format_consumer_buy_line(ticker, award_date, projection).strip())
    else:
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

history_cols = [
    "ticker", "company", "award_id", "award_date", "transaction_amount",
    "agency", "subagency", "award_type",
]

# master_zero_purged.csv contains these award-time metadata columns.  If an
# older local file is missing any, create them as null rather than breaking.
for c in history_cols:
    if c not in master_history.columns:
        master_history[c] = np.nan

history_parts = [master_history[history_cols].copy()]

if LIVE_HISTORY.exists():
    try:
        live_old = pd.read_csv(LIVE_HISTORY)
        live_old["award_date"] = pd.to_datetime(
            live_old["award_date"], errors="coerce"
        )
        live_old["transaction_amount"] = pd.to_numeric(
            live_old["transaction_amount"], errors="coerce"
        )
        for c in history_cols:
            if c not in live_old.columns:
                live_old[c] = np.nan
        history_parts.append(live_old[history_cols].copy())
    except Exception as exc:
        print(f"Could not load prior live history: {exc}")

for c in history_cols:
    if c not in recent.columns:
        recent[c] = np.nan
history_parts.append(recent[history_cols].copy())

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

# Persist accumulated post-master live additions, not just this 7-day scan.
live_cols = history_cols.copy()
live_parts = []
if LIVE_HISTORY.exists():
    try:
        old_live = pd.read_csv(LIVE_HISTORY)
        old_live["award_date"] = pd.to_datetime(old_live["award_date"], errors="coerce")
        old_live["transaction_amount"] = pd.to_numeric(old_live["transaction_amount"], errors="coerce")
        for c in live_cols:
            if c not in old_live.columns:
                old_live[c] = np.nan
        live_parts.append(old_live[live_cols])
    except Exception as exc:
        print(f"Could not preserve prior live history: {exc}")
live_parts.append(recent[live_cols])
live_persist = pd.concat(live_parts, ignore_index=True).dropna(
    subset=["ticker", "award_date", "transaction_amount"]
)
live_persist["_dedupe"] = (
    live_persist["ticker"].astype(str) + "|"
    + live_persist["award_date"].dt.strftime("%Y-%m-%d") + "|"
    + live_persist["award_id"].astype(str) + "|"
    + live_persist["transaction_amount"].round(2).astype(str)
)
live_persist = live_persist.drop_duplicates("_dedupe").drop(columns="_dedupe")
live_persist.to_csv(LIVE_HISTORY, index=False)

# ============================================================
# CURRENT EVENTS = ONE TICKER / AWARD DATE
# ============================================================

# Production rule follows the validated non-LMT universe.
recent = recent[recent["ticker"].astype(str).str.upper() != "LMT"].copy()

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
        current_agency=(
            "agency",
            lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None
        ),
        current_agency_count=("agency", lambda s: s.dropna().nunique()),
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


def download_shares(ticker, start_date, end_date):
    cache_file = SHARES_CACHE_DIR / f"{ticker}.csv"

    # A recent cache is enough for a daily scanner, but always include the
    # requested range when a refresh is needed.
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file, parse_dates=["Date"])
            if len(cached):
                lo = cached["Date"].min()
                hi = cached["Date"].max()
                if lo <= pd.Timestamp(start_date) and hi >= pd.Timestamp(end_date) - pd.Timedelta(days=7):
                    return cached
        except Exception:
            pass

    try:
        raw = yf.Ticker(ticker).get_shares_full(
            start=str((pd.Timestamp(start_date) - pd.Timedelta(days=60)).date()),
            end=str((pd.Timestamp(end_date) + pd.Timedelta(days=2)).date()),
        )
        if raw is None or not len(raw):
            return pd.DataFrame(columns=["Date", "Shares"])

        idx = pd.to_datetime(raw.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)

        out = pd.DataFrame({
            "Date": idx,
            "Shares": pd.to_numeric(raw.values, errors="coerce"),
        }).dropna().sort_values("Date")

        out.to_csv(cache_file, index=False)
        return out
    except Exception as exc:
        print(f"Shares data error {ticker}: {exc}")
        return pd.DataFrame(columns=["Date", "Shares"])


def close_strictly_before(market, event_date):
    if market.empty:
        return np.nan
    d = pd.Timestamp(event_date)
    prior = market[market["Date"] < d]
    if not len(prior):
        return np.nan
    return float(prior.iloc[-1]["Close"])


def shares_on_or_before(shares, event_date):
    if shares.empty:
        return np.nan
    d = pd.Timestamp(event_date)
    prior = shares[shares["Date"] <= d]
    if not len(prior):
        return np.nan
    value = pd.to_numeric(prior.iloc[-1]["Shares"], errors="coerce")
    return float(value) if pd.notna(value) and value > 0 else np.nan


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


def idx_strictly_before(market, event_date):
    if market.empty:
        return None
    dates = market["Date"].values
    i = np.searchsorted(
        dates,
        np.datetime64(pd.Timestamp(event_date)),
        side="left",
    ) - 1
    return int(i) if i >= 0 else None


def trailing_return(market, event_date, sessions):
    i = idx_strictly_before(market, event_date)

    if i is None or i - sessions < 0:
        return np.nan

    p1 = float(market.iloc[i]["Close"])
    p0 = float(market.iloc[i - sessions]["Close"])

    return (p1 / p0 - 1.0) * 100.0 if p0 else np.nan


def trailing_volatility(market, event_date, sessions):
    i = idx_strictly_before(market, event_date)

    if i is None or i - sessions < 1:
        return np.nan

    close = market.iloc[i - sessions:i + 1]["Close"].astype(float)
    ret = close.pct_change().dropna()

    if len(ret) < max(5, sessions // 2):
        return np.nan

    return float(ret.std(ddof=1) * 100.0)


def distance_from_52w_high(market, event_date):
    i = idx_strictly_before(market, event_date)
    if i is None:
        return np.nan
    start = max(0, i - 251)
    window = pd.to_numeric(market.iloc[start:i + 1]["Close"], errors="coerce").dropna()
    if len(window) < 20:
        return np.nan
    current = float(window.iloc[-1])
    high = float(window.max())
    return (current / high - 1.0) * 100.0 if high > 0 else np.nan


def volume_ratio(market, event_date):
    i = idx_strictly_before(market, event_date)
    if i is None or i - 59 < 0:
        return np.nan
    vol20 = pd.to_numeric(market.iloc[i - 19:i + 1]["Volume"], errors="coerce").dropna()
    vol60 = pd.to_numeric(market.iloc[i - 59:i + 1]["Volume"], errors="coerce").dropna()
    if len(vol20) < 10 or len(vol60) < 30:
        return np.nan
    den = float(vol60.mean())
    return float(vol20.mean() / den) if den > 0 else np.nan


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
        positive_amount=(
            "transaction_amount",
            lambda s: pd.to_numeric(s, errors="coerce").clip(lower=0).sum()
        ),
        negative_amount=(
            "transaction_amount",
            lambda s: pd.to_numeric(s, errors="coerce").clip(upper=0).sum()
        ),
        agency=(
            "agency",
            lambda s: s.dropna().mode().iloc[0] if len(s.dropna()) else None
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

# Historical shares outstanding are required by MD14 market-cap factor.
shares_history = {}
for ticker in candidate_tickers:
    shares_history[ticker] = download_shares(ticker, market_start, market_end)
    time.sleep(0.10)

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
            "return_5d": forward_return(
                mkt, row["award_date"], 5
            ),
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
    # Frozen 11-factor detector score.
    transformed = []

    for c in FEATURES:
        value = feature_values.get(c, np.nan)

        try:
            value = float(value)
        except Exception:
            value = np.nan

        # Transform observed raw values first. The frozen imputation medians
        # are already stored in transformed space, so missing values must NOT
        # be transformed a second time.
        if np.isfinite(value):
            if model["transforms"][c] == "signed_log1p":
                value = float(np.sign(value) * np.log1p(abs(value)))
        else:
            value = float(model["impute_median"][c])

        transformed.append(value)

    z = (np.array(transformed, dtype=float) - mu) / sd
    diff = z - location

    md2 = float(diff @ precision @ diff)

    return float(
        np.sqrt(max(md2 / dimension, 1e-12))
    )

# ============================================================
# FROZEN MD14 PRIMARY DETECTOR
# ============================================================

FEATURES14 = model14["features"]
MU14 = np.array([model14["reference_mean"][c] for c in FEATURES14], dtype=float)
SD14 = np.array([model14["reference_sd"][c] for c in FEATURES14], dtype=float)
LOC14 = np.array(model14["lw_location"], dtype=float)
PREC14 = np.array(model14["lw_precision"], dtype=float)
MED14 = model14["impute_median"]
COUNT_CENTERS14 = model14["A_sector_count_centers"]
AWARD_CENTERS14 = model14["A_sector_median_award_centers"]


def frozen_md14_score(feature_values):
    transformed = []
    for c in FEATURES14:
        value = feature_values.get(c, np.nan)
        try:
            value = float(value)
        except Exception:
            value = np.nan

        if np.isfinite(value):
            if model14["transforms"].get(c) == "signed_log1p":
                value = float(np.sign(value) * np.log1p(abs(value)))
        else:
            value = float(MED14[c])

        transformed.append(value)

    z = (np.array(transformed, dtype=float) - MU14) / SD14
    diff = z - LOC14
    md2 = float(diff @ PREC14 @ diff)
    return float(np.sqrt(max(md2 / float(model14["dimension_normalization"]), 1e-12)))


def projected_peak_90d(md11_score, market_cap_before, pre_volatility_market_20d):
    """Informational expected 90-trading-day maximum gain in percent.

    Returns NaN unless all three regression inputs are finite and market cap is positive.
    This is deliberately not imputed because the projection is an audit/ranking layer,
    not part of the frozen BUY architecture.
    """
    try:
        md = float(md11_score)
        mcap = float(market_cap_before)
        prevol = float(pre_volatility_market_20d)
    except Exception:
        return np.nan

    if not (np.isfinite(md) and np.isfinite(mcap) and np.isfinite(prevol)) or mcap <= 0:
        return np.nan

    return float(
        PEAK_PROJ_INTERCEPT
        + PEAK_PROJ_MD11_COEF * md
        + PEAK_PROJ_LN_MCAP_COEF * np.log(mcap)
        + PEAK_PROJ_PREVOL_COEF * prevol
    )


# ============================================================
# SCORE CURRENT EVENTS
# ============================================================

scan_rows = []
new_buys = []
new_buy_projections = {}

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
        prior_abs_award_mean = float(prior["abs_amount"].mean())
    else:
        prior_abs_award_max = np.nan
        prior_signed_award_mean = np.nan
        prior_abs_award_median = np.nan
        prior_abs_award_mean = np.nan

    w30 = prior[
        prior["award_date"] >= d - pd.Timedelta(days=30)
    ]

    prior_transactions_30d = (
        int(w30["award_count"].sum())
        if len(w30)
        else 0
    )

    prior_award_days_30d = int(len(w30))

    positive_30d = float(pd.to_numeric(w30.get("positive_amount"), errors="coerce").fillna(0).sum()) if len(w30) else 0.0
    negative_30d = float(pd.to_numeric(w30.get("negative_amount"), errors="coerce").fillna(0).sum()) if len(w30) else 0.0
    gross_30d = positive_30d + abs(negative_30d)
    positive_negative_balance_30d = (
        (positive_30d + negative_30d) / gross_30d if gross_30d > 0 else np.nan
    )

    current_agency = ev.get("current_agency")
    if pd.notna(current_agency) and len(prior):
        same_agency_fraction = float((prior["agency"] == current_agency).mean())
    else:
        same_agency_fraction = np.nan

    stock = market.get(ticker, pd.DataFrame())
    spy = market.get("SPY", pd.DataFrame())

    pre_volatility_market_20d = trailing_volatility(stock, d, 20)
    pre_volatility_market_60d = trailing_volatility(stock, d, 60)
    volatility_regime_ratio_20d_60d = (
        pre_volatility_market_20d / pre_volatility_market_60d
        if pd.notna(pre_volatility_market_20d)
        and pd.notna(pre_volatility_market_60d)
        and pre_volatility_market_60d > 0
        else np.nan
    )
    volatility_expansion_flag = (
        float(volatility_regime_ratio_20d_60d > 1.0)
        if pd.notna(volatility_regime_ratio_20d_60d) else np.nan
    )
    distance_from_52w_high_pct = distance_from_52w_high(stock, d)
    volume_20d_to_60d_ratio = volume_ratio(stock, d)
    volume_high_regime_flag = (
        float(volume_20d_to_60d_ratio > 1.0)
        if pd.notna(volume_20d_to_60d_ratio) else np.nan
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
        pd.DataFrame(columns=["award_date", "return_5d", "return_20d", "return_60d"])
    )

    # Same no-look-ahead maturity rules as Sample A.  A 5-trading-day
    # response used an 11-calendar-day maturity buffer in the original feature store.
    matured5 = resp[resp["award_date"] <= d - pd.Timedelta(days=11)]
    vals5 = pd.to_numeric(matured5.get("return_5d"), errors="coerce").dropna()
    prior_response_median_5d = float(vals5.median()) if len(vals5) else np.nan

    # Existing MD11 response features.
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
        "transaction_amount_abs_sum": float(ev["transaction_amount_abs_sum"]),
        "prior_transactions_30d": prior_transactions_30d,
        "relative_strength_spy_60d": relative_strength_spy_60d,
        "prior_award_days_30d": prior_award_days_30d,
    }

    # Frozen MD11 is now the SECONDARY confirmation space.
    md11_score = frozen_score(feature_values)
    md11_pass = md11_score >= THRESHOLD

    # Point-in-time scale: prior trading-day close x shares outstanding known
    # on or before the award date.  This matches MD14 development/validation.
    prior_close = close_strictly_before(stock, d)
    shares = shares_on_or_before(shares_history.get(ticker, pd.DataFrame()), d)
    market_cap_before = (
        prior_close * shares
        if pd.notna(prior_close) and pd.notna(shares)
        and prior_close > 0 and shares > 0
        else np.nan
    )
    log_mcap = (
        np.log10(float(market_cap_before))
        if pd.notna(market_cap_before) and market_cap_before > 0 else np.nan
    )

    sector = sector_group(ticker, ev["company"])
    count_adj = (
        np.log10(float(prior_response_count_60d)) - COUNT_CENTERS14[sector]
        if prior_response_count_60d > 0 else np.nan
    )
    median_adj = (
        np.log10(float(prior_abs_award_median)) - AWARD_CENTERS14[sector]
        if pd.notna(prior_abs_award_median) and prior_abs_award_median > 0 else np.nan
    )

    feature_values14 = {
        "log10_market_cap_before": log_mcap,
        "prior_abs_award_median_adj": median_adj,
        "distance_from_52w_high_pct": distance_from_52w_high_pct,
        "volatility_regime_ratio_20d_60d": volatility_regime_ratio_20d_60d,
        "prior_signed_award_mean": prior_signed_award_mean,
        "prior_abs_award_median": prior_abs_award_median,
        "current_agency_count": int(ev.get("current_agency_count", 0)),
        "volatility_expansion_flag": volatility_expansion_flag,
        "prior_abs_award_mean": prior_abs_award_mean,
        "prior_response_count_60d_adj": count_adj,
        "positive_negative_balance_30d": positive_negative_balance_30d,
        "volume_high_regime_flag": volume_high_regime_flag,
        "same_agency_fraction": same_agency_fraction,
        "prior_response_median_5d": prior_response_median_5d,
    }

    # Frozen MD14 is the PRIMARY detector.
    md14_score = frozen_md14_score(feature_values14)
    md14_pass = md14_score >= MD14_THRESHOLD
    market_cap_available = int(pd.notna(market_cap_before) and market_cap_before > 0)

    # Informational magnitude estimate only. It does NOT alter MD14/MD11 BUY logic.
    projected_peak_pct_90d = projected_peak_90d(
        md11_score,
        market_cap_before,
        pre_volatility_market_20d,
    )

    # FINAL FROZEN BUY RULE: MD14 primary AND MD11 secondary.
    buy = md14_pass and md11_pass

    # Fingerprint changes if the same ticker/day receives additional
    # transactions later, allowing a materially changed event to rescore.
    # Model version is included so the first MD14+MD11 production run is not
    # suppressed by fingerprints created by the previous production scanner.
    fingerprint = (
        f"{ticker}|{d.date()}|"
        f"{float(ev['transaction_amount_abs_sum']):.2f}|"
        f"{int(ev['same_day_award_count'])}|{MODEL_VERSION}"
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
        "md11_score": md11_score,
        "md11_threshold": THRESHOLD,
        "md14_score": md14_score,
        "md14_threshold": MD14_THRESHOLD,
        "md14_primary_pass": int(md14_pass),
        "md11_secondary_pass": int(md11_pass),
        "market_cap_before": market_cap_before,
        "log10_market_cap_before": log_mcap,
        "sector_group": sector,
        "prior_response_count_60d_adj": count_adj,
        "prior_abs_award_median_adj": median_adj,
        "market_cap_available": market_cap_available,
        "model_version": MODEL_VERSION,
        "scanner_version": SCANNER_VERSION,
        "projected_peak_pct_90d": projected_peak_pct_90d,
        "projection_model": PEAK_PROJ_NAME,
        "signal": "BUY" if buy else "PASS",
        "previously_seen": int(previously_seen),
        "current_price_for_log": current_price,
        **feature_values,
        **feature_values14,
    }

    scan_rows.append(row)

    event_state_key = f"{ticker}|{d.date()}"
    if buy:
        # Retain every qualifying event for the rest of TODAY, regardless of
        # whether its fingerprint was already seen on an earlier rerun.
        DAILY_BUY_STATE.setdefault("events", {})[event_state_key] = {
            "ticker": ticker,
            "award_date": str(d.date()),
            "projected_peak_pct_90d": (
                float(projected_peak_pct_90d) if pd.notna(projected_peak_pct_90d) else None
            ),
            "last_seen_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
        }
        if not previously_seen:
            new_buys.append(ticker)
            new_buy_projections[ticker] = projected_peak_pct_90d
    else:
        # If this exact ticker/date event is materially revised and now fails,
        # remove that event from today's retained BUY state.
        DAILY_BUY_STATE.setdefault("events", {}).pop(event_state_key, None)

    seen[fingerprint] = {
        "first_seen_utc": seen.get(
            fingerprint,
            {}
        ).get(
            "first_seen_utc",
            datetime.now(timezone.utc).isoformat(),
        ),
        "last_md14_score": md14_score,
        "last_md11_score": md11_score,
        "last_projected_peak_pct_90d": projected_peak_pct_90d if pd.notna(projected_peak_pct_90d) else None,
        "model_version": MODEL_VERSION,
        "scanner_version": SCANNER_VERSION,
        "signal": "BUY" if buy else "PASS",
    }

# ============================================================
# USER-FACING OUTPUT: ALL BUYs FOUND TODAY + PROJECTED PEAK
# ============================================================

new_buys = sorted(set(new_buys))
save_daily_buy_state(DAILY_BUY_STATE)
active_buys, active_buy_projections = write_current_buy_file(DAILY_BUY_STATE)
append_award_watch_to_buy_file(DAILY_AWARD_WATCH_STATE)

def format_buy_line(ticker):
    matching = [r for r in current_buy_rows(DAILY_BUY_STATE) if r["ticker"] == ticker]
    award_date = matching[-1]["award_date"] if matching else ""
    projection = matching[-1].get("projected_peak_pct_90d") if matching else None
    return format_consumer_buy_line(ticker, award_date, projection)


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

# ============================================================
# DAILY DIAGNOSTIC FUNNEL - INTERNAL ONLY
# ============================================================

today_scan = pd.DataFrame(scan_rows)

md14_passes = int(
    pd.to_numeric(
        today_scan.get("md14_primary_pass", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).sum()
)

md11_passes = int(
    pd.to_numeric(
        today_scan.get("md11_secondary_pass", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0).sum()
)

intersection_passes = int(
    (today_scan.get("signal", pd.Series(dtype=str)) == "BUY").sum()
)

market_cap_missing = int(
    (pd.to_numeric(
        today_scan.get("market_cap_available", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0) == 0).sum()
)

diag = {
    "scan_utc": datetime.now(timezone.utc).isoformat(),
    "scan_start": str(START),
    "scan_end": str(TODAY),
    "watermark_source": WATERMARK_SOURCE,
    "downloaded_rows_total": int(len(all_rows)),
    "downloaded_positive": int(downloaded_positive),
    "downloaded_negative": int(downloaded_negative),
    "recipient_universe_tickers": int(len(recipient_aliases)),
    "recipient_queries": int(recipient_queries),
    "recipient_aliases": int(sum(len(v) for v in recipient_aliases.values())),
    "matched_public_company_transactions": int(len(recent)),
    "ticker_day_events": int(len(events)),
    "events_scored": int(len(today_scan)),
    "md14_passes": md14_passes,
    "md11_passes": md11_passes,
    "intersection_passes": intersection_passes,
    "new_buy_tickers": int(len(new_buys)),
    "active_buy_tickers": int(len(active_buys)),
    "market_cap_missing": market_cap_missing,
    "projected_peak_available": int(pd.to_numeric(
        today_scan.get("projected_peak_pct_90d", pd.Series(dtype=float)),
        errors="coerce",
    ).notna().sum()),
    "unknown_recipient_names": int(len(unknown)),
    "award_watch_status": award_watch_status,
    "award_watch_error": award_watch_error,
    "award_watch_queries": int(award_watch_queries),
    "award_watch_rows": int(len(award_watch_records)),
    "new_award_watch_alerts": int(new_award_watch_alerts),
    "active_award_watch_alerts": int(len(current_award_watch_rows(DAILY_AWARD_WATCH_STATE))),
    "status": "OK" if award_watch_status == "OK" else "OK_BUY_FRAMEWORK_WATCH_FAILED",
}

DIAG_LATEST.write_text(json.dumps(diag, indent=2))

pd.DataFrame([diag]).to_csv(
    DAILY_DIAG,
    mode="a",
    header=not DAILY_DIAG.exists(),
    index=False,
)

print(
    "DIAGNOSTIC FUNNEL: "
    f"downloaded={diag['downloaded_rows_total']:,} "
    f"matched={diag['matched_public_company_transactions']:,} "
    f"events={diag['ticker_day_events']:,} "
    f"MD14={diag['md14_passes']:,} "
    f"MD11={diag['md11_passes']:,} "
    f"AND={diag['intersection_passes']:,} "
    f"new_BUYs={diag['new_buy_tickers']:,}"
)

# Advance acquisition watermark only after the complete scan, scoring and
# diagnostics have all succeeded. Failed runs therefore do not move the
# watermark and will automatically catch up next time.
ACQ_STATE_FILE.write_text(json.dumps({
    "last_successful_scan_date": str(TODAY),
    "last_fetch_start": str(START),
    "last_success_utc": datetime.now(timezone.utc).isoformat(),
    "watermark_source": WATERMARK_SOURCE,
    "model_version": MODEL_VERSION,
    "scanner_version": SCANNER_VERSION,
}, indent=2))

SEEN_FILE.write_text(
    json.dumps(seen, indent=2, sort_keys=True)
)

AWARD_WATCH_SEEN_FILE.write_text(
    json.dumps(award_watch_seen, indent=2, sort_keys=True)
)
save_daily_award_watch_state(DAILY_AWARD_WATCH_STATE)

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
print("RECON LIVE BUY SCANNER V4.2 - MD14 + MD11 + AWARD/IDV WATCH")
print("=" * 60)

if active_buys:
    print("BUY:")
    for ticker in active_buys:
        print(format_buy_line(ticker).strip())
else:
    print("NO BUYS")

watch_rows_console = current_award_watch_rows(DAILY_AWARD_WATCH_STATE)
if watch_rows_console:
    print("-")
    print("LARGE / FRAMEWORK AWARD WATCH (NOT BUY-SCORED):")
    for r in watch_rows_console:
        amount_bits = [x for x in (_money(r.get("award_amount")), _money(r.get("potential_amount"))) if x]
        amount_text = " / ".join(amount_bits) if amount_bits else "amount unavailable"
        print(f"{r.get('ticker','')}  {r.get('award_id','')}  {amount_text}  {r.get('award_type','')}")
if award_watch_status != "OK":
    print(f"AWARD WATCH WARNING: {award_watch_error}")

print("=" * 60)
print(f"MD14 threshold  : {MD14_THRESHOLD:.6f}")
print(f"MD11 threshold  : {THRESHOLD:.6f}")
print(f"Events scored   : {len(events)}")
print(f"New BUY tickers : {len(new_buys)}")
print(f"Active BUYs today: {len(active_buys)}")
print(f"Award-watch alerts today: {len(current_award_watch_rows(DAILY_AWARD_WATCH_STATE))}")
print(f"Award-watch status: {award_watch_status}")
print(f"Output          : {BUY_FILE}")
print("=" * 60)