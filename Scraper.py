
import requests
import json
import time
from datetime import date, timedelta
from pathlib import Path
import yfinance as yf

# ============================================================
# RECON V1.0
# Federal Contract -> Public Company -> Revenue -> Signal
# ============================================================

MIN_AWARD = 1_000_000
AUTO_DISCOVER_MIN = 10_000_000
MAX_AUTO_LOOKUPS = 25
DAYS = 7

END = date.today()
START = END - timedelta(days=DAYS)

USA_API = (
    "https://api.usaspending.gov/api/v2/"
    "search/spending_by_award/"
)

YAHOO_SEARCH_API = (
    "https://query2.finance.yahoo.com/v1/finance/search"
)

# ============================================================
# MASTER
#
# Known / manually verified public companies.
#
# Revenue is a bootstrap/fallback value.
# Yahoo revenue is allowed to replace it.
# ============================================================

MASTER = {
    "PLTR": {
        "keywords": ["PALANTIR"],
        "revenue": 6_160_000_000
    },
    "RCAT": {
        "keywords": ["RED CAT"],
        "revenue": 71_540_000
    },
    "AVAV": {
        "keywords": ["AEROVIRONMENT"],
        "revenue": 1_980_000_000
    },
    "WWD": {
        "keywords": ["WOODWARD"],
        "revenue": 4_190_000_000
    },
    "AEVA": {
        "keywords": ["AEVA TECHNOLOGIES", "AEVA"],
        "revenue": 20_970_000
    },
    "LMT": {
        "keywords": ["LOCKHEED MARTIN"],
        "revenue": 75_110_000_000
    },
    "RTX": {
        "keywords": ["RAYTHEON", "RAYTHEON TECHNOLOGIES"],
        "revenue": 90_370_000_000
    },
    "BAH": {
        "keywords": ["BOOZ ALLEN"],
        "revenue": 11_220_000_000
    },
    "SAIC": {
        "keywords": ["SCIENCE APPLICATIONS INTERNATIONAL"],
        "revenue": 7_290_000_000
    },
    "LDOS": {
        "keywords": ["LEIDOS"],
        "revenue": 17_330_000_000
    },
    "LHX": {
        "keywords": ["L3HARRIS", "L3 HARRIS"],
        "revenue": 22_930_000_000
    },
    "NOC": {
        "keywords": ["NORTHROP GRUMMAN"],
        "revenue": 42_370_000_000
    },
}

# ============================================================
# FILES
# ============================================================

CACHE_TICKER = Path("ticker_cache.json")
CACHE_REVENUE = Path("revenue_cache.json")
UNKNOWN_FILE = Path("unknown_companies.json")
RECON_FILE = Path("recon.json")

# ============================================================
# JSON LOADER
# ============================================================

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as e:
        print(f"Cache read error {path}: {e}")

    return default


ticker_cache = load_json(CACHE_TICKER, {})
revenue_cache = load_json(CACHE_REVENUE, {})

# ============================================================
# MASTER BOOTSTRAP
# ============================================================

for ticker, info in MASTER.items():

    for keyword in info["keywords"]:
        ticker_cache[keyword] = ticker

# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent": "RECON/1.0"
})

lookups_used = 0

# ============================================================
# MASTER LOOKUP
# ============================================================

def master_lookup(name):

    upper = name.upper()

    for ticker, info in MASTER.items():

        for keyword in info["keywords"]:

            if keyword in upper:
                return ticker

    return None


# ============================================================
# PRIVATE ENTITY FILTER
#
# Do NOT reject LLCs automatically.
#
# Public companies frequently receive awards through
# subsidiaries and contracting entities.
# ============================================================

def is_obvious_private_entity(name):

    upper = name.upper().strip()

    # Known public companies always pass.
    if master_lookup(name):
        return False

    padded = f" {upper} "

    if " JOINT VENTURE " in padded:
        return True

    if " CONSORTIUM " in padded:
        return True

    return False


# ============================================================
# TICKER DISCOVERY
# ============================================================

def find_ticker(name):

    global lookups_used

    # --------------------------------------------------------
    # Cached successful result
    # --------------------------------------------------------

    cached = ticker_cache.get(name)

    if cached:
        return cached

    # --------------------------------------------------------
    # Master
    # --------------------------------------------------------

    ticker = master_lookup(name)

    if ticker:

        ticker_cache[name] = ticker

        return ticker

    # --------------------------------------------------------
    # Yahoo lookup limit
    # --------------------------------------------------------

    if lookups_used >= MAX_AUTO_LOOKUPS:
        return None

    lookups_used += 1

    print(
        f" Yahoo {lookups_used}/"
        f"{MAX_AUTO_LOOKUPS}: {name}"
    )

    try:

        response = session.get(
            YAHOO_SEARCH_API,
            params={
                "q": name,
                "quotesCount": 5,
                "newsCount": 0
            },
            timeout=10
        )

        if response.status_code == 429:

            print(" Yahoo 429 - backing off")

            lookups_used -= 1

            time.sleep(10)

            return None

        response.raise_for_status()

        data = response.json()

        company_upper = name.upper()

        # ----------------------------------------------------
        # Examine Yahoo quotes
        # ----------------------------------------------------

        for quote in data.get("quotes", []):

            if quote.get("quoteType") != "EQUITY":
                continue

            symbol = quote.get("symbol")

            if not symbol:
                continue

            # Skip foreign exchange suffixes.
            if "." in symbol:
                continue

            exchange = quote.get("exchange", "")

            allowed = {
                "",
                "NMS",
                "NYQ",
                "ASE",
                "NGM",
                "NCM",
                "NasdaqGS",
                "NasdaqCM",
                "NasdaqGM",
                "NYSE",
                "NYSEArca",
                "TXSE"
            }

            if exchange not in allowed:
                continue

            # ------------------------------------------------
            # Confidence check.
            #
            # Yahoo can return unrelated companies for vague
            # searches. Require some name overlap unless the
            # Yahoo result is an unusually strong exact match.
            # ------------------------------------------------

            yahoo_name = (
                quote.get("longname")
                or quote.get("shortname")
                or ""
            ).upper()

            tokens = [
                token
                for token in company_upper.replace(",", " ").split()
                if len(token) >= 4
            ]

            if tokens:

                overlap = sum(
                    1
                    for token in tokens
                    if token in yahoo_name
                )

                if overlap == 0:
                    continue

            ticker_cache[name] = symbol

            print(
                f" FOUND: {name} -> {symbol}"
            )

            return symbol

    except Exception as e:

        print(
            f" Yahoo error {name}: {e}"
        )

    # Do NOT cache failures.
    # Temporary Yahoo failures should be retried later.

    return None


# ============================================================
# REVENUE LOOKUP
#
# Priority:
#
# 1. Yahoo financial statement
# 2. Yahoo info
# 3. Existing cache
# 4. MASTER fallback
#
# This intentionally allows monthly Yahoo refreshes.
# ============================================================

def get_revenue(ticker):

    print(
        f" Revenue fetch {ticker}..."
    )

    # --------------------------------------------------------
    # Yahoo financial data
    # --------------------------------------------------------

    try:

        stock = yf.Ticker(ticker)

        # ----------------------------------------------------
        # Financial statements
        # ----------------------------------------------------

        try:

            financials = stock.financials

            if (
                financials is not None
                and not financials.empty
                and "Total Revenue" in financials.index
            ):

                values = []

                for value in financials.loc[
                    "Total Revenue"
                ]:

                    try:

                        value = float(value)

                        if value > 0:
                            values.append(value)

                    except Exception:
                        continue

                if values:

                    # Yahoo normally places the most recent
                    # annual period first.
                    revenue = values[0]

                    revenue_cache[ticker] = revenue

                    return revenue

        except Exception as e:

            print(
                f"  Financials unavailable "
                f"for {ticker}: {e}"
            )

        # ----------------------------------------------------
        # Yahoo info fallback
        # ----------------------------------------------------

        try:

            info = stock.info

            revenue = info.get(
                "totalRevenue"
            )

            if revenue:

                revenue = float(revenue)

                if revenue > 0:

                    revenue_cache[ticker] = revenue

                    return revenue

        except Exception as e:

            print(
                f"  Yahoo info unavailable "
                f"for {ticker}: {e}"
            )

    except Exception as e:

        print(
            f"  Revenue error "
            f"{ticker}: {e}"
        )

    # --------------------------------------------------------
    # Existing cache
    #
    # Used if Yahoo is temporarily unavailable.
    # --------------------------------------------------------

    cached = revenue_cache.get(ticker)

    if cached:

        try:

            cached = float(cached)

            if cached > 0:

                print(
                    f"  Using cached revenue "
                    f"for {ticker}"
                )

                return cached

        except Exception:
            pass

    # --------------------------------------------------------
    # MASTER fallback
    # --------------------------------------------------------

    if ticker in MASTER:

        revenue = float(
            MASTER[ticker]["revenue"]
        )

        revenue_cache[ticker] = revenue

        print(
            f"  Using MASTER fallback "
            f"for {ticker}"
        )

        return revenue

    return None


# ============================================================
# USAspending DOWNLOAD
# ============================================================

payload = {

    "filters": {

        "award_amounts": [
            {
                "lower_bound": MIN_AWARD
            }
        ],

        "award_type_codes": [
            "A",
            "B",
            "C",
            "D"
        ],

        "time_period": [
            {
                "start_date": str(START),
                "end_date": str(END)
            }
        ]
    },

    "fields": [
        "Recipient Name",
        "Award Amount",
        "Awarding Agency",
        "Award ID",
        "Last Modified Date",
        "Start Date",
        "End Date"
    ],

    "sort": "Award Amount",

    "order": "desc",

    "limit": 100,

    "page": 1
}


# ============================================================
# DOWNLOAD ALL AVAILABLE PAGES
# ============================================================

print(
    f"Downloading USAspending "
    f"{START} -> {END}..."
)

all_rows = []

page = 1

while True:

    payload["page"] = page

    try:

        response = session.post(
            USA_API,
            json=payload,
            timeout=60
        )

        response.raise_for_status()

        page_data = response.json()

    except Exception as e:

        print(
            f"USAspending error on "
            f"page {page}: {e}"
        )

        raise

    rows = page_data.get(
        "results",
        []
    )

    if not rows:
        break

    all_rows.extend(rows)

    print(
        f"  Page {page}: "
        f"{len(rows)} awards"
    )

    page_metadata = page_data.get(
        "page_metadata",
        {}
    )

    has_next = page_metadata.get(
        "hasNext",
        False
    )

    if not has_next:
        break

    page += 1

    time.sleep(0.25)


print(
    f"Total award records downloaded: "
    f"{len(all_rows)}"
)


# ============================================================
# AGGREGATE BY RECIPIENT
# ============================================================

companies = {}

seen_awards = set()

for row in all_rows:

    name = row.get(
        "Recipient Name"
    )

    if not name:
        continue

    if is_obvious_private_entity(name):
        continue

    award = row.get(
        "Award Amount",
        0
    ) or 0

    try:
        award = float(award)
    except Exception:
        continue

    award_id = row.get(
        "Award ID"
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    if award_id:

        if award_id in seen_awards:
            continue

        seen_awards.add(
            award_id
        )

    # --------------------------------------------------------
    # Initialize
    # --------------------------------------------------------

    if name not in companies:

        companies[name] = {

            "total": 0,

            "count": 0,

            "largest": 0,

            "agencies": set(),

            "award_ids": [],

            "dates": [],

            "start_dates": [],

            "end_dates": []

        }

    company = companies[name]

    company["total"] += award

    company["count"] += 1

    company["largest"] = max(
        company["largest"],
        award
    )

    # --------------------------------------------------------
    # Agency
    # --------------------------------------------------------

    agency = row.get(
        "Awarding Agency"
    )

    if isinstance(
        agency,
        dict
    ):

        agency = (
            agency.get(
                "toptier_name"
            )
            or agency.get(
                "name"
            )
        )

    if agency:

        company[
            "agencies"
        ].add(
            agency
        )

    # --------------------------------------------------------
    # Award ID
    # --------------------------------------------------------

    if award_id:

        company[
            "award_ids"
        ].append(
            award_id
        )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    modified = row.get(
        "Last Modified Date"
    )

    if modified:

        company[
            "dates"
        ].append(
            modified
        )

    start_date = row.get(
        "Start Date"
    )

    if start_date:

        company[
            "start_dates"
        ].append(
            start_date
        )

    end_date = row.get(
        "End Date"
    )

    if end_date:

        company[
            "end_dates"
        ].append(
            end_date
        )


# ============================================================
# SORT COMPANIES BY TOTAL AWARDS
# ============================================================

sorted_companies = sorted(
    companies.items(),
    key=lambda x: x[1]["total"],
    reverse=True
)


# ============================================================
# PROCESS COMPANIES
# ============================================================

output = []

unknown = []

for company, stats in sorted_companies:

    ticker = master_lookup(
        company
    )

    # --------------------------------------------------------
    # Master match
    # --------------------------------------------------------

    if ticker:

        ticker_cache[
            company
        ] = ticker

    # --------------------------------------------------------
    # Cached match
    # --------------------------------------------------------

    else:

        ticker = ticker_cache.get(
            company
        )

    # --------------------------------------------------------
    # Automatic Yahoo discovery
    #
    # Only meaningful awards consume Yahoo requests.
    # --------------------------------------------------------

    if (
        not ticker
        and stats["total"]
        >= AUTO_DISCOVER_MIN
        and lookups_used
        < MAX_AUTO_LOOKUPS
    ):

        ticker = find_ticker(
            company
        )

        if ticker:

            time.sleep(
                1.2
            )

    # --------------------------------------------------------
    # No ticker
    # --------------------------------------------------------

    if not ticker:

        unknown.append({

            "company": company,

            "award_raw": stats[
                "total"
            ],

            "award": (
                f"${stats['total']/1e6:,.1f}M"
            ),

            "award_count": stats[
                "count"
            ],

            "largest_award": (
                f"${stats['largest']/1e6:,.1f}M"
            ),

            "agencies": sorted(
                list(
                    stats["agencies"]
                )
            )[:3]

        })

        continue

    # --------------------------------------------------------
    # Revenue
    # --------------------------------------------------------

    revenue = get_revenue(
        ticker
    )

    if not revenue:

        unknown.append({

            "company": company,

            "ticker": ticker,

            "reason": "No revenue available",

            "award_raw": stats[
                "total"
            ],

            "award": (
                f"${stats['total']/1e6:,.1f}M"
            )

        })

        continue

    # --------------------------------------------------------
    # CONTRACT / REVENUE RATIO
    # --------------------------------------------------------

    ratio = (
        stats["total"]
        / revenue
        * 100
    )

    # --------------------------------------------------------
    # SIGNAL
    # --------------------------------------------------------

    if ratio >= 50:

        signal = "BUY"

    elif ratio >= 10:

        signal = "WATCH"

    else:

        signal = "SELL"

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output.append({

        "ticker": ticker,

        "company": company,

        "award": (
            f"${stats['total']/1e6:,.1f}M"
        ),

        "award_raw": stats[
            "total"
        ],

        "revenue_m": round(
            revenue / 1e6,
            1
        ),

        "ratio": round(
            ratio,
            2
        ),

        "signal": signal,

        "award_count": stats[
            "count"
        ],

        "largest_award": (
            f"${stats['largest']/1e6:,.1f}M"
        ),

        "agencies": sorted(
            list(
                stats["agencies"]
            )
        )[:3],

        "award_ids": stats[
            "award_ids"
        ][:10],

        "last_modified": sorted(
            stats["dates"],
            reverse=True
        )[:5],

        "start_dates": sorted(
            stats["start_dates"],
            reverse=True
        )[:5],

        "end_dates": sorted(
            stats["end_dates"],
            reverse=True
        )[:5]

    })

    time.sleep(
        0.3
    )


# ============================================================
# SORT
# ============================================================

output.sort(
    key=lambda x: x["ratio"],
    reverse=True
)

unknown.sort(
    key=lambda x: x.get(
        "award_raw",
        0
    ),
    reverse=True
)


# ============================================================
# SAVE
# ============================================================

CACHE_TICKER.write_text(
    json.dumps(
        ticker_cache,
        indent=2,
        sort_keys=True
    )
)

CACHE_REVENUE.write_text(
    json.dumps(
        revenue_cache,
        indent=2,
        sort_keys=True
    )
)

RECON_FILE.write_text(
    json.dumps(
        output,
        indent=2
    )
)

UNKNOWN_FILE.write_text(
    json.dumps(
        unknown,
        indent=2
    )
)


# ============================================================
# SUMMARY
# ============================================================

print()

print(
    "=========================================="
)

print(
    " RECON V1.0 COMPLETE"
)

print(
    "=========================================="
)

print(
    f"Date range       : "
    f"{START} -> {END}"
)

print(
    f"Award records    : "
    f"{len(all_rows)}"
)

print(
    f"Recipients       : "
    f"{len(companies)}"
)

print(
    f"Recommendations  : "
    f"{len(output)}"
)

print(
    f"Unknown companies: "
    f"{len(unknown)}"
)

print(
    f"Yahoo lookups    : "
    f"{lookups_used}"
)

print(
    f"Ticker cache     : "
    f"{len(ticker_cache)}"
)

print(
    f"Revenue cache    : "
    f"{len(revenue_cache)}"
)

print(
    "=========================================="
)

print()

print(
    "TOP RECOMMENDATIONS"
)

print(
    "------------------------------------------"
)

for item in output[:20]:

    print(
        f"{item['signal']:6} "
        f"{item['ticker']:6} "
        f"{item['ratio']:8.2f}% "
        f"{item['award']:>12}  "
        f"{item['company'][:40]}"
    )
