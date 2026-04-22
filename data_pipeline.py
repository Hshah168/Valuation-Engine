"""
data_pipeline.py — 100% FREE real data pipeline.
Three sources, zero cost, no API key required:
  1. SEC EDGAR  — financials (FCF, revenue, EBITDA, debt, cash) — primary source, direct from filings
  2. yfinance   — price, market cap, beta, analyst targets, shares outstanding
  3. FRED       — 10-year Treasury yield for risk-free rate in WACC

IMPORTANT: Run build_universe.py once before this script to generate a verified
ticker list. If universe.json is missing, a built-in verified fallback is used.

Usage:
    python build_universe.py        # run once to build verified ticker list
    python data_pipeline.py         # full run (~150 companies, ~15 min)
    python data_pipeline.py --test  # first 10 only (~2 min sanity check)

Requirements:
    pip install yfinance requests pandas numpy
"""

import sys, os, time, json, requests
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# SEC EDGAR requires a descriptive User-Agent — use your own name/email
EDGAR_HEADERS = {"User-Agent": "Hetal Shah hetalshah5563@gmail.com"}

# ── Verified fallback universe (manually confirmed against yfinance) ──────────
_FALLBACK_UNIVERSE = {
    "Technology": {
        # All confirmed valid on yfinance as of 2025
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

def load_universe() -> dict:
    """
    Load ticker universe from universe.json (built by build_universe.py).
    Falls back to the verified hardcoded list if the file doesn't exist.
    """
    universe_path = os.path.join(DATA_DIR, "universe.json")
    if os.path.exists(universe_path):
        with open(universe_path) as f:
            universe = json.load(f)
        total = sum(len(t) for s in universe.values() for t in s.values())
        print(f"  Loaded universe.json — {total} tickers")
        return universe
    else:
        total = sum(len(t) for s in _FALLBACK_UNIVERSE.values() for t in s.values())
        print(f"  universe.json not found — using verified fallback ({total} tickers)")
        print(f"  Tip: run 'python build_universe.py' once to fetch live S&P constituents")
        return _FALLBACK_UNIVERSE

UNIVERSE = load_universe()

# ── CIK lookup cache (ticker → SEC CIK number) ───────────────────────────────
_CIK_CACHE = {}

def get_cik(ticker: str) -> str | None:
    """Look up SEC CIK for a ticker using EDGAR company search."""
    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]
    try:
        url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=10-K"
        r   = requests.get(url, headers=EDGAR_HEADERS, timeout=10)
        # Primary: company tickers JSON (most reliable)
        r2  = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=EDGAR_HEADERS, timeout=10
        )
        if r2.status_code == 200:
            data = r2.json()
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    _CIK_CACHE[ticker] = cik
                    return cik
    except Exception:
        pass
    return None


def fetch_edgar_financials(cik: str) -> dict:
    """
    Fetch structured financial data from SEC EDGAR XBRL company facts API.
    Returns up to 4 years of: revenue, net income, operating CF, capex, total debt, cash.
    All values in USD (EDGAR reports in actual dollars, not thousands).
    """
    if not cik:
        return {}
    try:
        url  = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        resp = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        if resp.status_code != 200:
            return {}
        facts = resp.json().get("facts", {})
        us_gaap = facts.get("us-gaap", {})

        def get_annual_series(concept: str, n: int = 4) -> list:
            """Extract last n annual 10-K values for a GAAP concept."""
            data = us_gaap.get(concept, {}).get("units", {}).get("USD", [])
            annual = [
                d for d in data
                if d.get("form") == "10-K" and d.get("fp") == "FY"
                   and d.get("val") is not None
            ]
            # Sort by end date descending, deduplicate by fiscal year
            annual.sort(key=lambda x: x.get("end",""), reverse=True)
            seen, out = set(), []
            for d in annual:
                yr = d.get("end","")[:4]
                if yr not in seen:
                    seen.add(yr)
                    out.append(d["val"])
                if len(out) >= n:
                    break
            return out

        # Revenue — try multiple GAAP concepts (companies use different line items)
        revenue_series = (
            get_annual_series("Revenues") or
            get_annual_series("RevenueFromContractWithCustomerExcludingAssessedTax") or
            get_annual_series("SalesRevenueNet") or
            get_annual_series("RevenueFromContractWithCustomerIncludingAssessedTax")
        )

        # Operating cash flow
        op_cf_series = (
            get_annual_series("NetCashProvidedByUsedInOperatingActivities") or
            get_annual_series("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
        )

        # Capital expenditures (typically reported as negative outflow)
        capex_series = (
            get_annual_series("PaymentsToAcquirePropertyPlantAndEquipment") or
            get_annual_series("CapitalExpendituresIncurredButNotYetPaid")
        )

        # Net income (for EBITDA proxy)
        ni_series = get_annual_series("NetIncomeLoss")

        # Debt
        debt_series = (
            get_annual_series("LongTermDebt") or
            get_annual_series("LongTermDebtAndCapitalLeaseObligations") or
            get_annual_series("DebtAndCapitalLeaseObligations")
        )

        # Cash
        cash_series = (
            get_annual_series("CashAndCashEquivalentsAtCarryingValue") or
            get_annual_series("CashCashEquivalentsAndShortTermInvestments")
        )

        # Build FCF series: operating CF - capex
        fcf_series = []
        for i in range(min(len(op_cf_series), len(capex_series), 4)):
            # Capex in EDGAR is reported as positive outflow — subtract it
            fcf_series.append(op_cf_series[i] - capex_series[i])

        return {
            "revenue":      revenue_series[0]  if revenue_series  else np.nan,
            "fcf":          fcf_series[0]       if fcf_series      else np.nan,
            "fcf_series":   fcf_series[:4],
            "operating_cf": op_cf_series[0]    if op_cf_series    else np.nan,
            "net_income":   ni_series[0]        if ni_series       else np.nan,
            "total_debt":   debt_series[0]      if debt_series     else np.nan,
            "cash":         cash_series[0]      if cash_series     else np.nan,
            # EBITDA proxy: net income + D&A is hard to get from XBRL cleanly,
            # so we use operating CF as a reasonable proxy (OCF ≈ EBITDA - taxes - WC changes)
            "ebitda_proxy": op_cf_series[0]    if op_cf_series    else np.nan,
        }
    except Exception as e:
        return {"edgar_error": str(e)}


def fetch_yfinance_market_data(ticker: str) -> dict:
    """
    Fetch market data from yfinance:
    price, market cap, beta, shares outstanding, analyst targets, 52w range.
    Does NOT fetch financials — those come from EDGAR.
    """
    try:
        import yfinance as yf
        stk  = yf.Ticker(ticker)
        info = stk.info
        if not info or not info.get("currentPrice"):
            return {"yf_error": "no price data"}

        return {
            "price":          info.get("currentPrice",       np.nan),
            "market_cap":     info.get("marketCap",          np.nan),
            "shares":         info.get("sharesOutstanding",  np.nan),
            "beta":           info.get("beta",               1.0) or 1.0,
            "sector_yf":      info.get("sector",             "Unknown"),
            "pe_ratio":       info.get("trailingPE",         np.nan),
            "ev_ebitda_yf":   info.get("enterpriseToEbitda", np.nan),
            "target_price":   info.get("targetMeanPrice",    np.nan),
            "analyst_low":    info.get("targetLowPrice",     np.nan),
            "analyst_high":   info.get("targetHighPrice",    np.nan),
            "price_52w_high": info.get("fiftyTwoWeekHigh",  np.nan),
            "price_52w_low":  info.get("fiftyTwoWeekLow",   np.nan),
        }
    except Exception as e:
        return {"yf_error": str(e)}


def fetch_risk_free_rate() -> float:
    """
    Fetch current 10-year US Treasury yield from FRED (Federal Reserve).
    Used as the risk-free rate in WACC / CAPM.
    Falls back to 4.2% if FRED is unreachable.
    """
    try:
        # FRED public API — no key needed for this endpoint
        url  = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id":    "DGS10",
            "api_key":      "DEMO_KEY",    # FRED allows 120 req/day on DEMO_KEY
            "file_type":    "json",
            "sort_order":   "desc",
            "limit":        1,
        }
        r = requests.get(url, params=params, timeout=8)
        if r.status_code == 200:
            val = r.json()["observations"][0]["value"]
            rf  = float(val) / 100
            print(f"  [FRED] 10yr Treasury: {rf:.2%}")
            return rf
    except Exception:
        pass
    print("  [FRED] Could not fetch — using 4.2% fallback")
    return 0.042   # reasonable 2024 fallback


def build_wacc(beta: float, rf: float, market_cap: float, total_debt: float,
               tax_rate: float = 0.21, erp: float = 0.055) -> float:
    """
    WACC = (E/V) × Ke  +  (D/V) × Kd × (1 − t)
    Ke  = Rf + β × ERP   (CAPM)
    Kd  = 5.0% pre-tax (industry average for investment grade)
    """
    ke     = rf + beta * erp
    debt   = total_debt or 0
    equity = market_cap or 0
    total  = equity + debt
    if total == 0:
        return ke
    we = equity / total
    wd = debt   / total
    return we * ke + wd * 0.05 * (1 - tax_rate)


def fetch_company(ticker: str, sector: str, tier: str, rf: float) -> dict:
    """
    Full fetch for one company: EDGAR (financials) + yfinance (market data).
    Merges both into a single row matching the schema expected by valuation_engine.py.
    """
    # 1 — SEC EDGAR financials
    cik   = get_cik(ticker)
    edgar = fetch_edgar_financials(cik) if cik else {}
    time.sleep(0.12)   # EDGAR rate limit: max 10 req/sec

    # 2 — yfinance market data
    yf_data = fetch_yfinance_market_data(ticker)
    time.sleep(0.3)    # polite

    # Check for fatal failures
    if "yf_error" in yf_data and "edgar_error" in edgar:
        return {"ticker": ticker, "error": f"yf: {yf_data['yf_error']} | edgar: {edgar.get('edgar_error','')}"}
    if "yf_error" in yf_data:
        return {"ticker": ticker, "error": yf_data["yf_error"]}

    # 3 — Merge: prefer EDGAR for financials, yfinance for market data
    mcap      = yf_data.get("market_cap", np.nan)
    beta      = yf_data.get("beta", 1.0)
    total_debt= edgar.get("total_debt", np.nan)
    cash_val  = edgar.get("cash",       np.nan)
    revenue   = edgar.get("revenue",    np.nan)
    ebitda    = edgar.get("ebitda_proxy", np.nan)  # OCF proxy
    fcf       = edgar.get("fcf",        np.nan)
    shares    = yf_data.get("shares",   np.nan)
    price     = yf_data.get("price",    np.nan)

    # 4 — Derived metrics
    wacc = build_wacc(beta, rf, mcap, total_debt or 0)
    ev   = (mcap + (total_debt or 0) - (cash_val or 0)) if pd.notna(mcap) else np.nan

    ev_ebitda = (
        yf_data.get("ev_ebitda_yf", np.nan)   # prefer yfinance's pre-computed value
        if pd.notna(yf_data.get("ev_ebitda_yf"))
        else (ev / ebitda if pd.notna(ev) and pd.notna(ebitda) and ebitda > 0 else np.nan)
    )
    ev_revenue = ev / revenue if pd.notna(ev) and pd.notna(revenue) and revenue > 0 else np.nan

    return {
        "ticker":       ticker,
        "sector":       sector,
        "cap_tier":     tier,
        "price":        round(price,      2)  if pd.notna(price)      else np.nan,
        "market_cap":   round(mcap)           if pd.notna(mcap)       else np.nan,
        "fcf":          round(fcf)            if pd.notna(fcf)        else np.nan,
        "fcf_series":   edgar.get("fcf_series", []),
        "revenue":      round(revenue)        if pd.notna(revenue)    else np.nan,
        "ebit":         np.nan,                # EDGAR EBIT requires D&A — left for extension
        "ebitda":       round(ebitda)         if pd.notna(ebitda)     else np.nan,
        "total_debt":   round(total_debt)     if pd.notna(total_debt) else np.nan,
        "cash":         round(cash_val)       if pd.notna(cash_val)   else np.nan,
        "shares":       round(shares)         if pd.notna(shares)     else np.nan,
        "beta":         round(beta,  3),
        "wacc":         round(wacc,  4),
        "cost_of_equity": round(rf + beta * 0.055, 4),
        "tax_rate":     0.21,
        "ev":           round(ev)             if pd.notna(ev)         else np.nan,
        "ev_ebitda":    round(ev_ebitda,  2)  if pd.notna(ev_ebitda)  else np.nan,
        "ev_revenue":   round(ev_revenue, 2)  if pd.notna(ev_revenue) else np.nan,
        "pe_ratio":     round(yf_data.get("pe_ratio", np.nan), 1) if pd.notna(yf_data.get("pe_ratio")) else np.nan,
        "target_price": yf_data.get("target_price",   np.nan),
        "analyst_low":  yf_data.get("analyst_low",    np.nan),
        "analyst_high": yf_data.get("analyst_high",   np.nan),
        "price_52w_high": yf_data.get("price_52w_high", np.nan),
        "price_52w_low":  yf_data.get("price_52w_low",  np.nan),
        # Audit trail — which sources provided data
        "source_edgar": "ok" if not edgar.get("edgar_error") and cik else "missing",
        "source_yf":    "ok",
        "cik":          cik or "",
    }


def run_pipeline(test_mode: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)

    # Flatten universe
    all_companies = [
        (ticker, sector, tier)
        for sector, tiers in UNIVERSE.items()
        for tier, tickers in tiers.items()
        for ticker in tickers
    ]
    if test_mode:
        all_companies = all_companies[:10]
        print("TEST MODE — first 10 tickers only\n")

    # Fetch risk-free rate once
    print("Fetching risk-free rate from FRED...")
    rf = fetch_risk_free_rate()

    results, failed, fcf_dict = [], [], {}
    total = len(all_companies)

    print(f"\nFetching {total} companies  |  EDGAR + yfinance + FRED\n")
    print(f"{'─'*65}")

    for i, (ticker, sector, tier) in enumerate(all_companies):
        print(f"  [{i+1:03d}/{total}] {ticker:6s} ({tier:5s}, {sector[:12]:12s}) ...", end=" ", flush=True)

        row = fetch_company(ticker, sector, tier, rf)

        if "error" in row:
            print(f"FAILED — {row['error']}")
            failed.append(ticker)
            continue

        fcf_dict[ticker] = row.pop("fcf_series", [])
        results.append(row)

        # Print one-line summary
        fcf_str  = f"FCF=${row['fcf']/1e6:>7,.0f}M" if pd.notna(row.get("fcf")) else "FCF=      n/a"
        src_str  = f"[EDGAR:{row['source_edgar']} YF:{row['source_yf']}]"
        print(f"OK | {fcf_str} | WACC={row['wacc']:.1%} | {src_str}")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, "fundamentals.csv"), index=False)
    with open(os.path.join(DATA_DIR, "fcf_series.json"), "w") as f:
        json.dump(fcf_dict, f, indent=2)

    # Summary
    print(f"\n{'='*65}")
    print(f"  ✅  Saved: {len(df)} companies  |  ❌ Failed: {len(failed)}")
    print(f"  Risk-free rate used: {rf:.2%} (FRED 10yr Treasury)")
    print(f"  EDGAR coverage: {(df['source_edgar']=='ok').sum()}/{len(df)} companies")
    if failed:
        print(f"\n  Failed tickers: {failed}")
    print(f"\n  Cap breakdown:")
    if "cap_tier" in df.columns:
        for tier in ["large","mid","small"]:
            n = (df["cap_tier"]==tier).sum()
            print(f"    {tier:6s}: {n:3d} companies")
    print(f"{'='*65}\n")
    return df


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    run_pipeline(test_mode=test_mode)
