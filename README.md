# Model Risk in Valuation Engine

**Auditing DCF and CCA across 50 S&P 500 companies — quantifying how wrong valuation models actually are.**

---

## Project Structure

```
model_risk/
├── data_pipeline.py        # Live data fetch via yfinance (run locally)
├── generate_data.py        # Calibrated synthetic data (no API key needed)
├── valuation_engine.py     # 2-stage DCF + CCA models + error computation
├── monte_carlo.py          # 10,000 simulations per company (WACC + growth uncertainty)
├── analysis_engine.py      # 5-panel analytical chart
├── dashboard.py            # Streamlit interactive dashboard (5 pages)
├── data/
│   ├── fundamentals.csv
│   └── fcf_series.json
└── results/
    ├── valuations.csv
    ├── mc_results.csv
    ├── mc_distributions_sample.json
    └── model_risk_analysis.png
```

---

## Quick Start (Real Data — Local)

```bash
# 1. Install dependencies
pip install yfinance pandas numpy scipy matplotlib seaborn plotly streamlit requests

# 2. Fetch real data (requires internet + Yahoo Finance access)
python data_pipeline.py

# 3. Run valuation models
python valuation_engine.py

# 4. Run Monte Carlo (10,000 sims per company — ~2 min)
python monte_carlo.py

# 5. Generate static analysis chart
python analysis_engine.py

# 6. Launch interactive dashboard
streamlit run dashboard.py
```

## Quick Start (Synthetic Data — Any Environment)

```bash
python generate_data.py     # generates calibrated synthetic fundamentals
python valuation_engine.py
python monte_carlo.py
python analysis_engine.py
streamlit run dashboard.py
```

---

## Data Sources (Live Run)

| Data | Source | Cost |
|---|---|---|
| Price, FCF, EBITDA, Beta | yfinance | Free |
| Analyst targets | yfinance | Free |
| Historical financials (cleaner) | Financial Modeling Prep | $15/mo |
| Risk-free rate | FRED API | Free |

---

## Key Findings

| Finding | Result |
|---|---|
| DCF MAE vs market | ~43% |
| CCA MAE vs market | ~27% |
| Companies with TV > 70% of DCF | ~96% |
| Average terminal value % | ~78% |
| Median Monte Carlo CV | ~0.4 |

**The terminal value problem:** In nearly all companies, >70% of the DCF enterprise value sits in the terminal value — meaning the 5-year FCF forecast is nearly irrelevant. The model is essentially two assumptions: WACC and terminal growth rate.

---

## Dashboard Pages

1. **Executive Summary** — KPI cards, 4 key findings, full universe table
2. **Error Analysis** — Error distributions, sector heatmap, DCF vs CCA scatter
3. **Monte Carlo** — CV rankings, P5-P95 ranges, single-company distributions
4. **Ticker Deep Dive** — Company-level price comparison, inputs, MC summary
5. **Sensitivity Surface** — Interactive WACC × growth contour plot per company

---

## Tech Stack

- Python: pandas, numpy, scipy, matplotlib, plotly, streamlit
- Data: yfinance / FMP API / FRED
- Models: Pure Python (no external valuation libraries)
- Document: docx-js

---

## Author

Hetal Shah · hetalshah5563@gmail.com · [LinkedIn](https://linkedin.com/in/shah-hetal/) · [GitHub](https://github.com/Hshah168)
