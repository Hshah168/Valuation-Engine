"""
Monte Carlo Valuation Uncertainty Engine
All paths relative to this file's location.
"""

import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings('ignore')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

N_SIMULATIONS = 10_000


def monte_carlo_dcf(fcf, base_wacc, fcf_series=None, total_debt=0, cash=0,
                    shares=1, n_sims=N_SIMULATIONS, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if fcf is None or np.isnan(fcf) or fcf <= 0:
        return None

    hist_growth = 0.07
    if fcf_series and len(fcf_series) >= 2:
        valid = [v for v in fcf_series if v and v > 0]
        if len(valid) >= 2:
            hist_growth = max(0.0, min((valid[0]/valid[-1])**(1/(len(valid)-1))-1, 0.20))

    wacc_draws = np.clip(np.random.normal(base_wacc, 0.015, n_sims), 0.04, 0.25)
    g1_draws   = np.clip(np.random.normal(hist_growth, 0.04,  n_sims), -0.05, 0.25)
    g2_draws   = np.random.uniform(0.01, 0.04, n_sims)

    prices = []
    for wacc, g1, g2 in zip(wacc_draws, g1_draws, g2_draws):
        if wacc <= g2:
            continue
        cf = fcf
        pv1 = sum(cf * (1+g1)**yr / (1+wacc)**yr for yr in range(1, 6))
        cf  = fcf * (1+g1)**5
        tv  = cf * (1+g2) / (wacc-g2)
        ev  = pv1 + tv / (1+wacc)**5
        p   = (ev - (total_debt or 0) + (cash or 0)) / shares
        if p > 0:
            prices.append(p)

    prices = np.array(prices)
    if len(prices) < 100:
        return None

    return {
        "prices": prices,
        "mean":   prices.mean(),
        "median": np.median(prices),
        "std":    prices.std(),
        "p5":     np.percentile(prices,  5),
        "p25":    np.percentile(prices, 25),
        "p75":    np.percentile(prices, 75),
        "p95":    np.percentile(prices, 95),
        "cv":     prices.std() / prices.mean(),
        "n_valid":len(prices),
    }


def run_monte_carlo_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_DIR, "fundamentals.csv"))
    with open(os.path.join(DATA_DIR, "fcf_series.json")) as f:
        fcf_map = json.load(f)

    mc_results, mc_distributions = [], {}
    print(f"Running {N_SIMULATIONS:,} simulations per company...\n")

    for _, row in df.iterrows():
        ticker = row["ticker"]
        res    = monte_carlo_dcf(
            fcf=row.get("fcf"), base_wacc=row.get("wacc"),
            fcf_series=fcf_map.get(ticker, []),
            total_debt=row.get("total_debt", 0),
            cash=row.get("cash", 0), shares=row.get("shares", 1), seed=42,
        )
        if res is None:
            continue
        mkt = row.get("price", np.nan)
        prob = (res["prices"] > mkt).mean() if not np.isnan(mkt) else np.nan
        mc_distributions[ticker] = res["prices"].tolist()
        mc_results.append({
            "ticker": ticker, "sector": row["sector"], "market_price": mkt,
            "mc_mean":   round(res["mean"],   2),
            "mc_median": round(res["median"], 2),
            "mc_std":    round(res["std"],    2),
            "mc_p5":     round(res["p5"],  2),
            "mc_p25":    round(res["p25"], 2),
            "mc_p75":    round(res["p75"], 2),
            "mc_p95":    round(res["p95"], 2),
            "mc_cv":     round(res["cv"],  3),
            "prob_above_market": round(prob, 3) if not np.isnan(prob) else np.nan,
        })
        print(f"  {ticker}: mean={res['mean']:.0f}  P5={res['p5']:.0f}  P95={res['p95']:.0f}  CV={res['cv']:.2f}")

    out_df = pd.DataFrame(mc_results)
    out_df.to_csv(os.path.join(RESULTS_DIR, "mc_results.csv"), index=False)

    sample = {k: v[:500] for k, v in list(mc_distributions.items())[:10]}
    with open(os.path.join(RESULTS_DIR, "mc_distributions_sample.json"), "w") as f:
        json.dump(sample, f)

    print(f"\nMonte Carlo complete. {len(out_df)} companies.")
    print("\nTop 5 highest model risk (CV):")
    print(out_df.nlargest(5, "mc_cv")[["ticker","sector","market_price","mc_mean","mc_cv","mc_p5","mc_p95"]].to_string())
    print("\nTop 5 lowest model risk (CV):")
    print(out_df.nsmallest(5, "mc_cv")[["ticker","sector","market_price","mc_mean","mc_cv","mc_p5","mc_p95"]].to_string())
    return out_df

if __name__ == "__main__":
    run_monte_carlo_all()
