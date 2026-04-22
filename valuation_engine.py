"""
Valuation Engine — DCF + CCA for all 50 companies.
All paths relative to this file's location.
"""

import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def dcf_valuation(fcf, wacc, fcf_series=None, growth_stage1=0.08, growth_stage2=0.03,
                  years_stage1=5, total_debt=0, cash=0, shares=1):
    if fcf is None or np.isnan(fcf) or fcf <= 0 or wacc <= growth_stage2:
        return np.nan
    if fcf_series and len(fcf_series) >= 2:
        valid = [v for v in fcf_series if v and v > 0]
        if len(valid) >= 2:
            growth_stage1 = max(0.0, min((valid[0]/valid[-1])**(1/(len(valid)-1))-1, 0.20))
    pv_stage1 = 0
    cf = fcf
    for yr in range(1, years_stage1 + 1):
        cf *= (1 + growth_stage1)
        pv_stage1 += cf / (1 + wacc) ** yr
    terminal_value = cf * (1 + growth_stage2) / (wacc - growth_stage2)
    pv_terminal    = terminal_value / (1 + wacc) ** years_stage1
    enterprise_value = pv_stage1 + pv_terminal
    equity_value = enterprise_value - (total_debt or 0) + (cash or 0)
    price = equity_value / shares if shares and shares > 0 else np.nan
    tv_pct = pv_terminal / enterprise_value if enterprise_value > 0 else np.nan
    return {
        "dcf_price": max(price, 0),
        "enterprise_value": enterprise_value,
        "pv_stage1": pv_stage1,
        "pv_terminal": pv_terminal,
        "terminal_value_pct": tv_pct,
        "growth_stage1_used": growth_stage1,
        "growth_stage2_used": growth_stage2,
        "wacc_used": wacc,
    }


def cca_valuation(row, sector_medians):
    sector = row["sector"]
    if sector not in sector_medians:
        return {"cca_ebitda_price": np.nan, "cca_revenue_price": np.nan}
    med    = sector_medians[sector]
    shares = row.get("shares", 1)
    debt   = row.get("total_debt", 0) or 0
    cash   = row.get("cash", 0) or 0

    ebitda_price = np.nan
    ebitda = row.get("ebitda")
    if ebitda and ebitda > 0 and med.get("ev_ebitda"):
        equity_val   = ebitda * med["ev_ebitda"] - debt + cash
        ebitda_price = equity_val / shares if shares > 0 else np.nan

    revenue_price = np.nan
    revenue = row.get("revenue")
    if revenue and revenue > 0 and med.get("ev_revenue"):
        equity_val    = revenue * med["ev_revenue"] - debt + cash
        revenue_price = equity_val / shares if shares > 0 else np.nan

    return {
        "cca_ebitda_price":        max(ebitda_price,  0) if not np.isnan(ebitda_price)  else np.nan,
        "cca_revenue_price":       max(revenue_price, 0) if not np.isnan(revenue_price) else np.nan,
        "sector_median_ev_ebitda": med.get("ev_ebitda"),
        "sector_median_ev_revenue":med.get("ev_revenue"),
    }


def compute_errors(row):
    def pct_err(model_p, actual_p):
        if pd.isna(model_p) or pd.isna(actual_p) or actual_p == 0:
            return np.nan
        return (model_p - actual_p) / actual_p

    cca_e  = row.get("cca_ebitda_price",  np.nan)
    cca_r  = row.get("cca_revenue_price", np.nan)
    cca_blend = np.nanmean([v for v in [cca_e, cca_r] if not np.isnan(v)]) \
                if not (np.isnan(cca_e) and np.isnan(cca_r)) else np.nan

    return {
        "dcf_vs_market_pct":    pct_err(row.get("dcf_price"),  row.get("price")),
        "dcf_vs_analyst_pct":   pct_err(row.get("dcf_price"),  row.get("target_price")),
        "cca_vs_market_pct":    pct_err(cca_blend,             row.get("price")),
        "cca_vs_analyst_pct":   pct_err(cca_blend,             row.get("target_price")),
        "dcf_cca_divergence_pct":pct_err(row.get("dcf_price"), cca_blend),
        "cca_blended_price":    round(cca_blend, 2) if not np.isnan(cca_blend) else np.nan,
    }


def run_valuations():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
    with open(os.path.join(DATA_DIR, "fcf_series.json")) as f:
        fcf_series_map = json.load(f)

    sector_medians = df.groupby("sector").agg(
        ev_ebitda=("ev_ebitda",  "median"),
        ev_revenue=("ev_revenue","median"),
    ).to_dict("index")

    results = []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        dcf    = dcf_valuation(
            fcf=row.get("fcf"), wacc=row.get("wacc"),
            fcf_series=fcf_series_map.get(ticker, []),
            total_debt=row.get("total_debt", 0),
            cash=row.get("cash", 0), shares=row.get("shares", 1),
        )
        cca = cca_valuation(row.to_dict(), sector_medians)
        combined = {**row.to_dict()}
        if isinstance(dcf, dict):
            combined.update(dcf)
        else:
            combined["dcf_price"] = np.nan
        combined.update(cca)
        combined.update(compute_errors(combined))
        results.append(combined)

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(RESULTS_DIR, "valuations.csv"), index=False)
    print(f"Valuations complete for {len(out)} companies")
    cols = ["ticker","sector","price","dcf_price","cca_blended_price",
            "dcf_vs_market_pct","cca_vs_market_pct","terminal_value_pct","dcf_cca_divergence_pct"]
    s = out[cols].copy()
    for c in ["dcf_vs_market_pct","cca_vs_market_pct","terminal_value_pct"]:
        s[c] = s[c].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    print(s.head(10).to_string())
    return out

if __name__ == "__main__":
    run_valuations()
