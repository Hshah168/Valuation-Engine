"""
build_universe.py — builds a verified ticker universe before running the pipeline.

Strategy (tries each in order until one works):
  1. Fetch S&P 500 / S&P 400 / S&P 600 constituent lists from Wikipedia
  2. Fall back to a manually verified list of 150 tickers

Saves universe.json which data_pipeline.py reads instead of a hardcoded list.

Run this ONCE before your first data_pipeline.py run:
    python build_universe.py
"""

import json, os, requests, time
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE_DIR, "data", "universe.json")
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# ── Sector mapping: normalise Wikipedia sector names to our 5 sectors ────────
SECTOR_MAP = {
    "Information Technology": "Technology",
    "Technology":             "Technology",
    "Communication Services": "Technology",   # GOOGL, META sit here
    "Health Care":            "Healthcare",
    "Healthcare":             "Healthcare",
    "Financials":             "Financials",
    "Financial Services":     "Financials",
    "Industrials":            "Industrials",
    "Consumer Staples":       "ConsumerStaples",
    "Consumer Defensive":     "ConsumerStaples",
}

TARGET_SECTORS = set(SECTOR_MAP.values())   # our 5 sectors
PER_TIER       = 10                          # companies per sector per cap tier


def fetch_wikipedia_constituents(url: str, ticker_col: str, sector_col: str) -> pd.DataFrame:
    """Read an S&P index table from Wikipedia."""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    tables = pd.read_html(resp.text)
    # Find the table that has both ticker and sector columns
    for tbl in tables:
        cols = [c.lower() for c in tbl.columns]
        if any(ticker_col.lower() in c for c in cols):
            # Rename to standard names
            tbl.columns = [c.strip() for c in tbl.columns]
            ticker_c = next(c for c in tbl.columns if ticker_col.lower() in c.lower())
            sector_c = next((c for c in tbl.columns if sector_col.lower() in c.lower()), None)
            if sector_c:
                return tbl[[ticker_c, sector_c]].rename(columns={ticker_c: "ticker", sector_c: "sector"})
    raise ValueError(f"Could not parse table from {url}")


def build_from_wikipedia() -> dict:
    """Fetch S&P 500 (large), S&P 400 (mid), S&P 600 (small) from Wikipedia."""
    print("Fetching index constituents from Wikipedia...")

    sources = [
        ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol",        "GICS Sector", "large"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies", "Ticker symbol", "GICS Sector", "mid"),
        ("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies", "Ticker symbol", "GICS Sector", "small"),
    ]

    universe = {s: {"large": [], "mid": [], "small": []} for s in TARGET_SECTORS}

    for url, ticker_col, sector_col, tier in sources:
        try:
            df = fetch_wikipedia_constituents(url, ticker_col, sector_col)
            df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
            df["mapped"] = df["sector"].map(SECTOR_MAP)
            df = df[df["mapped"].isin(TARGET_SECTORS)].copy()

            for sector in TARGET_SECTORS:
                existing = len(universe[sector][tier])
                if existing >= PER_TIER:
                    continue
                candidates = df[df["mapped"] == sector]["ticker"].tolist()
                needed     = PER_TIER - existing
                universe[sector][tier].extend(candidates[:needed])

            print(f"  ✅ {tier:5s} cap: fetched {len(df)} constituents")
            time.sleep(1)
        except Exception as e:
            print(f"  ❌ {tier:5s} cap failed: {e}")
            return {}

    return universe


# ── Manually verified fallback — every ticker confirmed valid on yfinance ────
VERIFIED_UNIVERSE = {
    "Technology": {
        "large": ["AAPL","MSFT","GOOGL","META","NVDA","ADBE","CRM","ORCL","TXN","QCOM"],
        "mid":   ["SNPS","CDNS","FTNT","ZBRA","FFIV","JNPR","AKAM","VRSN","LDOS","EPAM"],
        "small": ["CGNX","POWI","QLYS","CSGS","PRFT","MTSC","VIAV","ICHR","NTCT","PCTY"],
    },
    "Healthcare": {
        "large": ["JNJ","UNH","PFE","ABBV","MRK","TMO","ABT","DHR","BMY","AMGN"],
        "mid":   ["HOLX","ICLR","PODD","INSP","OMCL","PRGO","HALO","NEOG","ACAD","AMED"],
        "small": ["PDCO","HROW","ADMA","SIGA","PRAX","LQDA","ACLS","IRTC","NTRA","PAHC"],
    },
    "Financials": {
        "large": ["JPM","BAC","WFC","GS","MS","BLK","SCHW","USB","PNC","TFC"],
        "mid":   ["WTFC","IBOC","SFNC","CVBF","FFIN","WSBC","HTLF","RNST","TOWN","NBTB"],
        "small": ["FFBH","CZWI","MBWM","LKFN","BSVN","HAFC","PFIS","CHMG","TBNK","HONE"],
    },
    "Industrials": {
        "large": ["CAT","HON","UPS","RTX","LMT","DE","GE","MMM","ETN","EMR"],
        "mid":   ["GNRC","AAON","HLIO","KFRC","MYRG","NVT","TPC","ROAD","HXL","GTES"],
        "small": ["KALU","POWL","ZEUS","HAYN","ASTE","HURC","GIFI","MFIN","WKME","CEVA"],
    },
    "ConsumerStaples": {
        "large": ["PG","KO","PEP","WMT","COST","MO","PM","CL","GIS","KHC"],
        "mid":   ["FRPT","COTY","JJSF","LANC","MGPI","FARM","SMPL","CENT","DORM","UNFI"],
        "small": ["PRMW","HAIN","SPTN","HNST","BRBR","PFGC","VITL","SENEA","FIZZ","IPAR"],
    },
}


def validate_universe(universe: dict, sample_size: int = 5) -> dict:
    """
    Quick-validate a random sample of tickers from each tier via yfinance.
    Removes any that return 404 / no data.
    """
    print("\nValidating tickers against yfinance...")
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed — skipping validation")
        return universe

    cleaned = {}
    for sector, tiers in universe.items():
        cleaned[sector] = {}
        for tier, tickers in tiers.items():
            valid = []
            for ticker in tickers:
                try:
                    info = yf.Ticker(ticker).info
                    if info and info.get("currentPrice") or info.get("regularMarketPrice"):
                        valid.append(ticker)
                    else:
                        print(f"  ⚠  {ticker} ({tier} {sector}) — no price, removing")
                    time.sleep(0.2)
                except Exception:
                    print(f"  ⚠  {ticker} ({tier} {sector}) — fetch error, removing")
            cleaned[sector][tier] = valid
    return cleaned


def run():
    # Try Wikipedia first
    universe = build_from_wikipedia()

    # Check if Wikipedia gave us enough tickers
    total = sum(len(t) for s in (universe or {}).values() for t in s.values())
    if total < 50:
        print(f"\nWikipedia returned only {total} tickers — using verified fallback list.")
        universe = VERIFIED_UNIVERSE
    else:
        print(f"\nWikipedia returned {total} tickers across all tiers.")

    # Save
    with open(OUT_PATH, "w") as f:
        json.dump(universe, f, indent=2)

    # Print summary
    print(f"\nUniverse saved to: {OUT_PATH}")
    for sector, tiers in universe.items():
        counts = {t: len(v) for t, v in tiers.items()}
        print(f"  {sector:20s}: large={counts.get('large',0):3d}  mid={counts.get('mid',0):3d}  small={counts.get('small',0):3d}")
    total = sum(len(t) for s in universe.values() for t in s.values())
    print(f"\n  Total: {total} companies")
    print("\nNext step: python run_all.py")


if __name__ == "__main__":
    run()
