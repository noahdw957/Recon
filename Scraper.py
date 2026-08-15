import json
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================================================
# RECON LIVE BUY SCANNER V3.8 - MD11 + MD8 + PROJECTED PEAK
#
# USER-FACING OUTPUT:
#     buy_tickers.txt
#
# Everything else is stored under ReconData/ for analysis.
#
# BUY MODEL:
#     BUY only when BOTH frozen spaces pass:
#       MD11 detector >= threshold from mts_frozen_model_A.json
#       MD8 scale veto >= 1.9463204147913817
#     MD8 frozen parameters are embedded in this file.
#
# IMPORTANT:
#     This is a sandbox scanner. It generates experimental BUY
#     signals; it does not contain a validated SELL model yet.
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
DAILY_DIAG = DATA_DIR / "daily_diagnostics.csv"
DIAG_LATEST = DATA_DIR / "diagnostic_latest.json"
LIVE_HISTORY = DATA_DIR / "live_history.csv"
SEEN_FILE = DATA_DIR / "seen_events.json"
TICKER_CACHE_FILE = DATA_DIR / "ticker_cache.json"
UNKNOWN_FILE = DATA_DIR / "unknown_companies.json"
ACQ_STATE_FILE = DATA_DIR / "acquisition_state.json"
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
        f"Missing {MODEL_FILE}. Upload the frozen Sample-A model."
    )

if not MASTER_HISTORY_FILE.exists():
    raise FileNotFoundError(
        f"Missing {MASTER_HISTORY_FILE}. Upload the saved nonzero award master."
    )

model = json.loads(MODEL_FILE.read_text())

# Frozen corrected scale-aware MTS8 deployment model. Embedded here so the
# daily scanner remains a single Python file. The only external model file
# required is the original frozen 11-factor model above.
MODEL8 = {'features': ['prior_response_count_60d_adj', 'log10_market_cap_before', 'prior_abs_award_max', 'relative_strength_spy_120d', 'prior_abs_award_median_adj', 'prior_response_count_60d', 'prior_abs_award_median', 'relative_strength_spy_60d'], 'threshold': 1.9463204147913817, 'A_sector_count_centers': {'AERO_DEFENSE': 1.6989700043360187, 'INDUSTRIAL': 1.4621397262132765, 'OTHER': 1.6483333918242467, 'TECH_SERVICES': 1.6433396112263248}, 'A_sector_median_award_centers': {'AERO_DEFENSE': 7.276858522781777, 'INDUSTRIAL': 6.379214783094319, 'OTHER': 7.050908632424625, 'TECH_SERVICES': 6.864458778006966}, 'impute_median': {'prior_response_count_60d_adj': 0.0492180226701817, 'log10_market_cap_before': 10.387166780894209, 'prior_abs_award_max': 19.354947011752447, 'relative_strength_spy_120d': 1.843991900460896, 'prior_abs_award_median_adj': -0.0066639210311842, 'prior_response_count_60d': 3.80666248977032, 'prior_abs_award_median': 16.5322049303338, 'relative_strength_spy_60d': 1.2132923469633017}, 'reference_mean': {'prior_response_count_60d_adj': 0.0013767411374706754, 'log10_market_cap_before': 10.625932783789732, 'prior_abs_award_max': 19.597688495171163, 'relative_strength_spy_120d': 1.2359990062272446, 'prior_abs_award_median_adj': -0.04354096940216947, 'prior_response_count_60d': 2.8273422161532253, 'prior_abs_award_median': 16.43449652376031, 'relative_strength_spy_60d': 2.623598721446692}, 'reference_sd': {'prior_response_count_60d_adj': 0.2411181730720844, 'log10_market_cap_before': 0.5075506908228757, 'prior_abs_award_max': 1.1421831854022753, 'relative_strength_spy_120d': 21.180105648166858, 'prior_abs_award_median_adj': 0.314659542161833, 'prior_response_count_60d': 1.7908841319714495, 'prior_abs_award_median': 0.8870167161218723, 'relative_strength_spy_60d': 14.497162681485262}, 'lw_location': [0.0, 1.4157430449355379e-15, 1.3177985575500604e-15, -3.5616177231082714e-17, -7.123235446216543e-17, -2.49313240617579e-16, 3.9177794954190986e-16, 0.0], 'lw_precision': [[1.1245235410715015, 0.11352024832397757, 0.1240456298401465, -0.03193144952522353, -0.3991441548956096, -0.36490679685676564, 0.013769655800668537, -0.05708349556794794], [0.11352024832397757, 1.3770061528784387, -0.3114930412901252, -0.22521682050650216, -1.1568724382658144, -0.014832737514160942, 0.9161846426492161, -0.0939101841889188], [0.12404562984014653, -0.3114930412901252, 1.9965791456133024, -0.3451494320242806, 0.2943141182659422, -1.0929572591554757, -1.1431114831222657, -0.398446057654768], [-0.031931449525223615, -0.22521682050650216, -0.3451494320242807, 2.205304783545063, 0.6514570223600256, 0.6839549226950875, 0.07099452031580906, -1.2606775178443608], [-0.3991441548956096, -1.1568724382658147, 0.2943141182659423, 0.6514570223600256, 4.367772214507019, 0.531126542035879, -3.263049460696854, 0.06272408648398381], [-0.36490679685676564, -0.014832737514160936, -1.0929572591554757, 0.6839549226950875, 0.5311265420358788, 1.93893598117823, 0.5661053548215131, -0.10750370278020249], [0.013769655800668522, 0.9161846426492162, -1.1431114831222657, 0.07099452031580904, -3.263049460696854, 0.5661053548215131, 4.088226221596494, -0.033279115184692745], [-0.05708349556794795, -0.09391018418891882, -0.3984460576547681, -1.2606775178443608, 0.06272408648398387, -0.10750370278020249, -0.033279115184692704, 2.0325863172680054]], 'version': 'RECON Scale-Aware MTS8 Frozen A - Corrected Deployment v1.0', 'notes': ['Sector centers are frozen Sample-A medians in log10(raw) space.', 'For stored signed_log1p A fields, reconstruction used raw=expm1(stored), then log10(raw).', 'Deployment rule is intersection with frozen MD11 threshold 2.1954452583448045.']}
MODEL_VERSION = "RECON_11x8_INTERSECTION_INCREMENTAL_V5"
SCANNER_VERSION = "RECON_LIVE_V3.8_PROJECTED_PEAK"
MD8_THRESHOLD = float(MODEL8["threshold"])

# Informational 90-trading-day peak projection.
# Fit previously on the 616-award Sample-C population (R^2 ~= 0.38).
# IMPORTANT: this projection does NOT participate in the BUY decision.
# Expected Peak % = 111.0 + 8.8*MD11 - 4.8*ln(MarketCap) + 7.1*preVol_20d
PEAK_PROJ_INTERCEPT = 111.0
PEAK_PROJ_MD11_COEF = 8.8
PEAK_PROJ_LN_MCAP_COEF = -4.8
PEAK_PROJ_PREVOL_COEF = 7.1
PEAK_PROJ_NAME = "C616_OLS_MD11_LNMCAP_PREVOL_R2_0.38"

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
    BUY_FILE.write_text("NO BUYS\n")
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
        "md11_passes": 0,
        "md8_passes_all_events": 0,
        "intersection_passes": 0,
        "new_buy_tickers": 0,
        "scale_data_missing": 0,
        "status": "NO_MATCHED_PUBLIC_COMPANY_TRANSACTIONS",
    }
    DIAG_LATEST.write_text(json.dumps(diag, indent=2))
    pd.DataFrame([diag]).to_csv(
        DAILY_DIAG,
        mode="a",
        header=not DAILY_DIAG.exists(),
        index=False,
    )
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

# Persist accumulated post-master live additions, not just this 7-day scan.
live_cols = [
    "ticker", "company", "award_id", "award_date", "transaction_amount"
]
live_parts = []
if LIVE_HISTORY.exists():
    try:
        old_live = pd.read_csv(LIVE_HISTORY)
        old_live["award_date"] = pd.to_datetime(old_live["award_date"], errors="coerce")
        old_live["transaction_amount"] = pd.to_numeric(old_live["transaction_amount"], errors="coerce")
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

# Historical shares outstanding are needed only by the MD8 scale veto.
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
# FROZEN SCALE-AWARE MTS8 VETO
# ============================================================

FEATURES8 = MODEL8["features"]
MU8 = np.array([MODEL8["reference_mean"][c] for c in FEATURES8], dtype=float)
SD8 = np.array([MODEL8["reference_sd"][c] for c in FEATURES8], dtype=float)
LOC8 = np.array(MODEL8["lw_location"], dtype=float)
PREC8 = np.array(MODEL8["lw_precision"], dtype=float)
MED8 = MODEL8["impute_median"]
COUNT_CENTERS8 = MODEL8["A_sector_count_centers"]
AWARD_CENTERS8 = MODEL8["A_sector_median_award_centers"]


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


def frozen_scale8_score(feature_values, ticker, company, market_cap_before):
    sector = sector_group(ticker, company)

    raw_count = feature_values.get("prior_response_count_60d", np.nan)
    raw_median = feature_values.get("prior_abs_award_median", np.nan)

    try:
        raw_count = float(raw_count)
    except Exception:
        raw_count = np.nan
    try:
        raw_median = float(raw_median)
    except Exception:
        raw_median = np.nan

    count_adj = (
        np.log10(raw_count) - COUNT_CENTERS8[sector]
        if np.isfinite(raw_count) and raw_count > 0
        else np.nan
    )
    median_adj = (
        np.log10(raw_median) - AWARD_CENTERS8[sector]
        if np.isfinite(raw_median) and raw_median > 0
        else np.nan
    )
    log_mcap = (
        np.log10(float(market_cap_before))
        if pd.notna(market_cap_before) and float(market_cap_before) > 0
        else np.nan
    )

    vals = {
        "prior_response_count_60d_adj": count_adj,
        "log10_market_cap_before": log_mcap,
        "prior_abs_award_max": signed_log1p_scalar(feature_values.get("prior_abs_award_max")),
        "relative_strength_spy_120d": feature_values.get("relative_strength_spy_120d", np.nan),
        "prior_abs_award_median_adj": median_adj,
        "prior_response_count_60d": signed_log1p_scalar(raw_count),
        "prior_abs_award_median": signed_log1p_scalar(raw_median),
        "relative_strength_spy_60d": feature_values.get("relative_strength_spy_60d", np.nan),
    }

    transformed = []
    for c in FEATURES8:
        try:
            value = float(vals.get(c, np.nan))
        except Exception:
            value = np.nan
        if not np.isfinite(value):
            value = float(MED8[c])
        transformed.append(value)

    z = (np.array(transformed, dtype=float) - MU8) / SD8
    diff = z - LOC8
    md2 = float(diff @ PREC8 @ diff)
    score = float(np.sqrt(max(md2 / 8.0, 1e-12)))

    return score, sector, count_adj, median_adj, log_mcap


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

    md11_score = frozen_score(feature_values)
    detector_pass = md11_score >= THRESHOLD

    # Point-in-time scale: prior trading-day close x shares outstanding known
    # on or before the award date. This matches the validation definition.
    prior_close = close_strictly_before(stock, d)
    shares = shares_on_or_before(
        shares_history.get(ticker, pd.DataFrame()), d
    )
    market_cap_before = (
        prior_close * shares
        if pd.notna(prior_close) and pd.notna(shares)
        and prior_close > 0 and shares > 0
        else np.nan
    )

    md8_score, sector, count_adj, median_adj, log_mcap = frozen_scale8_score(
        feature_values, ticker, ev["company"], market_cap_before
    )
    scale_data_complete = int(pd.notna(market_cap_before) and market_cap_before > 0)
    scale_veto_pass = (md8_score >= MD8_THRESHOLD) and bool(scale_data_complete)

    # Informational magnitude estimate only. It does NOT alter detector/veto logic.
    projected_peak_pct_90d = projected_peak_90d(
        md11_score,
        market_cap_before,
        pre_volatility_market_20d,
    )

    # FINAL FROZEN BUY RULE: detector AND scale-aware veto.
    buy = detector_pass and scale_veto_pass

    # Fingerprint changes if the same ticker/day receives additional
    # transactions later, allowing a materially changed event to rescore.
    # Model version is included so the first 11x8 production run is not
    # suppressed by fingerprints created by the previous MD11-only scanner.
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
        "md11_detector_pass": int(detector_pass),
        "md8_score": md8_score,
        "md8_threshold": MD8_THRESHOLD,
        "md8_scale_veto_pass": int(scale_veto_pass),
        "market_cap_before": market_cap_before,
        "log10_market_cap_before": log_mcap,
        "sector_group": sector,
        "prior_response_count_60d_adj": count_adj,
        "prior_abs_award_median_adj": median_adj,
        "scale_data_complete": scale_data_complete,
        "model_version": MODEL_VERSION,
        "scanner_version": SCANNER_VERSION,
        "projected_peak_pct_90d": projected_peak_pct_90d,
        "projection_model": PEAK_PROJ_NAME,
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
        "last_md11_score": md11_score,
        "last_md8_score": md8_score,
        "last_projected_peak_pct_90d": projected_peak_pct_90d if pd.notna(projected_peak_pct_90d) else None,
        "model_version": MODEL_VERSION,
        "scanner_version": SCANNER_VERSION,
        "signal": "BUY" if buy else "PASS",
    }

# ============================================================
# USER-FACING OUTPUT: TICKERS ONLY
# ============================================================

new_buys = sorted(set(new_buys))

BUY_FILE.write_text(
    "".join(f"{ticker}\n" for ticker in new_buys)
    if new_buys
    else "NO BUYS\n"
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

# ============================================================
# DAILY DIAGNOSTIC FUNNEL - INTERNAL ONLY
# ============================================================

today_scan = pd.DataFrame(scan_rows)

md11_passes = int(
    pd.to_numeric(
        today_scan.get("md11_detector_pass", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()
)

md8_passes_all = int(
    pd.to_numeric(
        today_scan.get("md8_scale_veto_pass", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0).sum()
)

intersection_passes = int(
    (today_scan.get("signal", pd.Series(dtype=str)) == "BUY").sum()
)

scale_missing = int(
    (pd.to_numeric(
        today_scan.get("scale_data_complete", pd.Series(dtype=float)),
        errors="coerce",
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
    "md11_passes": md11_passes,
    "md8_passes_all_events": md8_passes_all,
    "intersection_passes": intersection_passes,
    "new_buy_tickers": int(len(new_buys)),
    "scale_data_missing": scale_missing,
    "projected_peak_available": int(pd.to_numeric(
        today_scan.get("projected_peak_pct_90d", pd.Series(dtype=float)),
        errors="coerce",
    ).notna().sum()),
    "unknown_recipient_names": int(len(unknown)),
    "status": "OK",
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
    f"MD11={diag['md11_passes']:,} "
    f"MD8={diag['md8_passes_all_events']:,} "
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
print("RECON LIVE BUY SCANNER - MD11 DETECTOR + MD8 SCALE VETO + PEAK PROJECTION")
print("=" * 60)

if new_buys:
    print("BUY:")
    for ticker in new_buys:
        print(ticker)
else:
    print("NO BUYS")

print("=" * 60)
print(f"MD11 threshold  : {THRESHOLD:.6f}")
print(f"MD8 threshold   : {MD8_THRESHOLD:.6f}")
print(f"Events scored   : {len(events)}")
print(f"New BUY tickers : {len(new_buys)}")
print(f"Output          : {BUY_FILE}")
print("=" * 60)
