"""
run_all.py — full Model Risk pipeline, 100% free data.
Sources: SEC EDGAR (financials) + yfinance (market data) + FRED (risk-free rate)

Usage:
    python run_all.py              # full run (~150 companies, ~15 min)
    python run_all.py --test       # first 10 tickers only (~2 min sanity check)
    python run_all.py --dashboard  # run pipeline then launch Streamlit
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
TEST = "--test" in sys.argv
DASH = "--dashboard" in sys.argv

def run(script, label, extra_args=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE, script)] + (extra_args or [])
    )
    if result.returncode != 0:
        print(f"\n❌  {script} failed. Stopping.")
        sys.exit(1)
    print(f"✅  {script} complete.")

print("\n🚀  Model Risk in Valuation Engine")
print("    Data: SEC EDGAR + yfinance + FRED  (100% free)")
if TEST:
    print("    Mode: TEST — first 10 tickers only")

# Step 0 — build verified universe (only if universe.json doesn't exist yet)
universe_path = os.path.join(BASE, "data", "universe.json")
if not os.path.exists(universe_path):
    run("build_universe.py", "Step 0/4 — Building verified ticker universe")
else:
    print(f"\n  universe.json already exists — skipping universe build")
    print(f"  (delete data/universe.json and re-run to refresh it)")

pipeline_args = ["--test"] if TEST else []

run("data_pipeline.py",    "Step 1/4 — Fetching real data (EDGAR + yfinance + FRED)", pipeline_args)
run("valuation_engine.py", "Step 2/4 — Running DCF + CCA valuations")
run("monte_carlo.py",      "Step 3/4 — Monte Carlo simulations (10,000 per company)")
run("analysis_engine.py",  "Step 4/4 — Generating analysis charts")

print(f"\n{'='*60}")
print("  ✅  Pipeline complete!")
print("  📁  Outputs saved in: results/")
print("       — valuations.csv")
print("       — mc_results.csv")
print("       — model_risk_analysis.png")
print(f"{'='*60}")

if DASH:
    print("\n🌐  Launching Streamlit dashboard...")
    subprocess.run(["streamlit", "run", os.path.join(BASE, "dashboard.py")])
else:
    print("\n  Launch dashboard:   streamlit run dashboard.py")
    print("  Full run + dashboard: python run_all.py --dashboard\n")
