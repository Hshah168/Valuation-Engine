"""
Model Risk in Valuation — Interactive Dashboard
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json, os, sys

# ── Resolve paths relative to THIS file — works on Windows, Mac, Linux ──────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
sys.path.insert(0, BASE_DIR)

st.set_page_config(page_title="Model Risk Engine", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card { background:white; border-radius:10px; padding:16px 20px;
  border-left:4px solid #2563EB; box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:8px; }
.metric-value { font-size:26px; font-weight:700; }
.metric-label { font-size:12px; color:#6B7280; margin-top:2px; }
.finding-box { background:#EFF6FF; border-left:4px solid #2563EB; border-radius:6px;
  padding:14px 18px; margin:8px 0; font-size:14px; }
.section-header { font-size:18px; font-weight:700; color:#111827;
  margin:20px 0 10px 0; padding-bottom:6px; border-bottom:2px solid #E5E7EB; }
div[data-testid="stSidebarContent"] { background:#1E293B; }
div[data-testid="stSidebarContent"] label,
div[data-testid="stSidebarContent"] p { color:#CBD5E1 !important; }
</style>
""", unsafe_allow_html=True)

BLUE="#2563EB"; PURPLE="#7C3AED"; GREEN="#059669"; RED="#DC2626"; AMBER="#D97706"; GRAY="#6B7280"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    paths = {
        "valuations":  os.path.join(RESULTS_DIR, "valuations.csv"),
        "mc_results":  os.path.join(RESULTS_DIR, "mc_results.csv"),
        "mc_dist":     os.path.join(RESULTS_DIR, "mc_distributions_sample.json"),
    }
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing data files:\n" + "\n".join(missing) +
            "\n\nRun in order from your project folder:\n"
            "  python generate_data.py\n"
            "  python valuation_engine.py\n"
            "  python monte_carlo.py"
        )
    val = pd.read_csv(paths["valuations"])
    mc  = pd.read_csv(paths["mc_results"])
    with open(paths["mc_dist"]) as f:
        mc_dist = json.load(f)
    return val, mc, mc_dist

try:
    val_df, mc_df, mc_dist = load_data()
    merged = val_df.merge(mc_df, on=["ticker","sector"], suffixes=("","_mc"), how="left")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Model Risk Engine")
    st.markdown("*DCF & CCA Audit of Large/Medium/Small Companies*")
    st.markdown("---")
    page = st.radio("Navigate", [" Executive Summary"," Error Analysis",
                                  " Monte Carlo"," Ticker Deep Dive"," Sensitivity Surface"])
    st.markdown("---")
    sel_sector = st.selectbox("Filter by Sector", ["All"] + sorted(merged["sector"].unique().tolist()))
    st.markdown("---")
    st.caption("Hetal V Shah · Model Risk Project · 2026")

try:
    page = page.split("  ", 1)[1].strip()
except IndexError:
    page = page.strip()
view_df = merged[merged["sector"] == sel_sector].copy() if sel_sector != "All" else merged.copy()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════
if page == "Executive Summary":
    st.title("Model Risk in Valuation Engine")
    st.markdown("**Auditing DCF & CCA across S&P 500, 400 & 600 companies**")
    st.markdown("---")

    dcf_mae  = view_df["dcf_vs_market_pct"].abs().mean() * 100
    cca_mae  = view_df["cca_vs_market_pct"].abs().mean() * 100
    tv_mean  = view_df["terminal_value_pct"].mean() * 100
    tv_above = (view_df["terminal_value_pct"] > 0.70).mean() * 100
    mc_cv    = mc_df.loc[mc_df["ticker"].isin(view_df["ticker"]), "mc_cv"].median()
    div_mean = view_df["dcf_cca_divergence_pct"].abs().mean() * 100

    kpis = [
        (f"{dcf_mae:.0f}%",  "DCF MAE vs Market",      BLUE),
        (f"{cca_mae:.0f}%",  "CCA MAE vs Market",      PURPLE),
        (f"{tv_mean:.0f}%",  "Avg Terminal Value %",   AMBER),
        (f"{tv_above:.0f}%", "Companies TV > 70%",     RED),
        (f"{mc_cv:.2f}",     "Median MC Risk (CV)",    GREEN),
        (f"{div_mean:.0f}%", "Avg DCF–CCA Divergence", GRAY),
    ]
    for col, (val, label, color) in zip(st.columns(6), kpis):
        col.markdown(f'<div class="metric-card"><div class="metric-value" style="color:{color}">{val}</div>'
                     f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="finding-box"><b>Finding 1 — CCA outperforms DCF on raw accuracy</b><br>'
                    f'DCF MAE ({dcf_mae:.0f}%) is {dcf_mae-cca_mae:.0f}pp worse than CCA ({cca_mae:.0f}%). '
                    f'Simpler methodology, more market-aligned outputs.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="finding-box"><b>Finding 2 — Terminal value dominates DCF</b><br>'
                    f'{tv_above:.0f}% of companies have terminal value >70% of enterprise value. '
                    f'The 5-year FCF forecast is nearly irrelevant to the final price.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="finding-box"><b>Finding 3 — Monte Carlo reveals extreme uncertainty</b><br>'
                    f'Median CV = {mc_cv:.2f}. High-CV companies have P5–P95 ranges spanning 4–5x, '
                    f'making single point estimates unreliable.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="finding-box"><b>Finding 4 — Divergence signals uncertainty</b><br>'
                    f'Average DCF–CCA divergence: {div_mean:.0f}%. When methods disagree by >25%, '
                    f'widen the implied price range rather than averaging outputs.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Universe Overview</div>', unsafe_allow_html=True)
    tbl = view_df[["ticker","sector","price","dcf_price","cca_blended_price",
                   "dcf_vs_market_pct","cca_vs_market_pct","terminal_value_pct"]].copy()
    tbl.columns = ["Ticker","Sector","Market Price","DCF Price","CCA Price","DCF Error","CCA Error","TV %"]
    tbl["DCF Error"]    = tbl["DCF Error"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    tbl["CCA Error"]    = tbl["CCA Error"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
    tbl["TV %"]         = tbl["TV %"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "—")
    tbl["Market Price"] = tbl["Market Price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    tbl["DCF Price"]    = tbl["DCF Price"].map(lambda x: f"${x:,.0f}" if pd.notna(x) else "—")
    tbl["CCA Price"]    = tbl["CCA Price"].map(lambda x: f"${x:,.2f}" if pd.notna(x) else "—")
    st.dataframe(tbl, use_container_width=True, height=440)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Error Analysis":
    st.title("Error Analysis — DCF vs CCA")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Error Distribution</div>', unsafe_allow_html=True)
        dcf_e = view_df["dcf_vs_market_pct"].dropna() * 100
        cca_e = view_df["cca_vs_market_pct"].dropna() * 100
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=dcf_e, name=f"DCF  MAE={dcf_e.abs().mean():.0f}%",
                                   nbinsx=25, opacity=0.65, marker_color=BLUE))
        fig.add_trace(go.Histogram(x=cca_e, name=f"CCA  MAE={cca_e.abs().mean():.0f}%",
                                   nbinsx=25, opacity=0.55, marker_color=PURPLE))
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1.5)
        fig.add_vline(x=dcf_e.mean(), line_dash="dot", line_color=BLUE, line_width=1.5,
                      annotation_text=f"DCF bias {dcf_e.mean():+.0f}%")
        fig.add_vline(x=cca_e.mean(), line_dash="dot", line_color=PURPLE, line_width=1.5,
                      annotation_text=f"CCA bias {cca_e.mean():+.0f}%")
        fig.update_layout(barmode="overlay", template="plotly_white", height=360,
                          xaxis_title="Error vs Market Price (%)", yaxis_title="# Companies",
                          legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Error by Sector</div>', unsafe_allow_html=True)
        sec = view_df.groupby("sector").agg(
            dcf_mae=("dcf_vs_market_pct",  lambda x: x.abs().mean()*100),
            cca_mae=("cca_vs_market_pct",  lambda x: x.abs().mean()*100),
            dcf_bias=("dcf_vs_market_pct", lambda x: x.mean()*100),
            cca_bias=("cca_vs_market_pct", lambda x: x.mean()*100),
        ).reset_index()
        z = sec[["dcf_mae","cca_mae","dcf_bias","cca_bias"]].values
        fig2 = go.Figure(go.Heatmap(z=z.T, x=sec["sector"].tolist(),
                                    y=["DCF MAE","CCA MAE","DCF Bias","CCA Bias"],
                                    colorscale="RdYlGn_r",
                                    text=np.round(z.T,1), texttemplate="%{text}%"))
        fig2.update_layout(template="plotly_white", height=360, xaxis_tickangle=-20)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">DCF vs CCA — Agreement & Divergence</div>', unsafe_allow_html=True)
    sc = view_df.dropna(subset=["dcf_price","cca_blended_price"]).copy()
    fig3 = px.scatter(sc, x="cca_blended_price", y="dcf_price", color="sector",
                      size="market_cap", hover_name="ticker",
                      hover_data={"dcf_vs_market_pct":":.1%","cca_vs_market_pct":":.1%"},
                      labels={"cca_blended_price":"CCA Price ($)","dcf_price":"DCF Price ($)"},
                      template="plotly_white", height=420,
                      title="DCF vs CCA — bubble size = market cap | dashed = perfect agreement")
    lim = max(sc["dcf_price"].max(), sc["cca_blended_price"].max()) * 1.05
    fig3.add_trace(go.Scatter(x=[0,lim], y=[0,lim], mode="lines",
                              line=dict(color="gray", dash="dash", width=1), name="DCF = CCA"))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Terminal Value % vs WACC</div>', unsafe_allow_html=True)
    tv_df = view_df.dropna(subset=["terminal_value_pct","wacc"]).copy()
    fig4  = px.scatter(tv_df, x=tv_df["wacc"]*100, y=tv_df["terminal_value_pct"]*100,
                       color="sector", hover_name="ticker",
                       labels={"x":"WACC (%)","y":"Terminal Value % of EV"},
                       template="plotly_white", height=360,
                       title="Terminal Value Dominance — nearly all companies above 70% threshold")
    fig4.add_hline(y=70, line_dash="dash", line_color=RED,
                   annotation_text="70% threshold", annotation_position="right")
    st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONTE CARLO
# ════════════════════════════════════════════════════════════════════════════
elif page == "Monte Carlo":
    st.title("Monte Carlo Valuation Uncertainty")
    st.markdown("**10,000 simulations per company. WACC ~ N(base, 1.5%). Terminal growth ~ Uniform(1%, 4%).**")
    st.markdown("---")

    mc_view = mc_df[mc_df["ticker"].isin(view_df["ticker"])].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Model Risk by Company (CV)</div>', unsafe_allow_html=True)
        mc_s = mc_view[mc_view["mc_cv"] <= 5].sort_values("mc_cv")
        fig = px.bar(mc_s, x="ticker", y="mc_cv", color="sector",
                     labels={"mc_cv":"Coefficient of Variation","ticker":""},
                     template="plotly_white", height=380,
                     title="Lower CV = more certain | Higher CV = wider price range")
        fig.add_hline(y=0.5, line_dash="dash", line_color=RED, annotation_text="High risk (CV=0.5)")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">P5–P95 Range vs Market Price</div>', unsafe_allow_html=True)
        mc_plot = mc_view.merge(view_df[["ticker","price"]], on="ticker")
        mc_plot = mc_plot.dropna(subset=["mc_p5","mc_p95","price"])
        mc_plot = mc_plot[mc_plot["mc_cv"] <= 5].head(25)
        fig2 = go.Figure()
        for _, r in mc_plot.iterrows():
            fig2.add_trace(go.Scatter(x=[r["mc_p5"],r["mc_p95"]], y=[r["ticker"],r["ticker"]],
                mode="lines", line=dict(color=BLUE,width=3), showlegend=False, opacity=0.5))
            fig2.add_trace(go.Scatter(x=[r["mc_median"]], y=[r["ticker"]],
                mode="markers", marker=dict(color=BLUE,size=8), showlegend=False))
            fig2.add_trace(go.Scatter(x=[r["price"]], y=[r["ticker"]],
                mode="markers", marker=dict(color=RED,size=9,symbol="diamond"), showlegend=False))
        fig2.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
            marker=dict(color=BLUE,size=8), name="MC Median"))
        fig2.add_trace(go.Scatter(x=[None],y=[None],mode="markers",
            marker=dict(color=RED,size=9,symbol="diamond"), name="Market Price"))
        fig2.update_layout(template="plotly_white", height=380,
                           xaxis_title="Price ($)", legend=dict(yanchor="bottom",y=0.01))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Price Distribution: Single Company</div>', unsafe_allow_html=True)
    avail = [t for t in mc_dist.keys() if t in view_df["ticker"].values]
    if avail:
        sel    = st.selectbox("Select company", avail)
        prices = np.array(mc_dist[sel])
        mkt    = view_df.loc[view_df["ticker"]==sel, "price"].values[0]
        prob_above = (prices > mkt).mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=prices, nbinsx=60, marker_color=BLUE, opacity=0.7, name="Simulated"))
        fig3.add_vline(x=mkt,                       line_color=RED,   line_width=2, line_dash="dash",
                       annotation_text=f"Market ${mkt:.0f}")
        fig3.add_vline(x=np.percentile(prices,  5), line_color=AMBER, line_width=1.5, line_dash="dot",
                       annotation_text="P5")
        fig3.add_vline(x=np.percentile(prices, 95), line_color=AMBER, line_width=1.5, line_dash="dot",
                       annotation_text="P95")
        fig3.update_layout(template="plotly_white", height=360,
                           title=f"{sel}  |  P(DCF > Market) = {prob_above:.0%}  |  "
                                 f"P5=${np.percentile(prices,5):.0f}  "
                                 f"Median=${np.median(prices):.0f}  "
                                 f"P95=${np.percentile(prices,95):.0f}",
                           xaxis_title="Simulated DCF Price ($)", yaxis_title="Frequency")
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TICKER DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Ticker Deep Dive":
    st.title("Ticker Deep Dive")
    st.markdown("---")

    ticker = st.selectbox("Select ticker", sorted(view_df["ticker"].tolist()))
    row    = view_df[view_df["ticker"] == ticker].iloc[0]
    mc_row = mc_df[mc_df["ticker"] == ticker]

    # ── Company Overview (introduction) — fetched live from yfinance ────────────
    def safe_num(val):
        try:
            v = float(val)
            return np.nan if np.isnan(v) else v
        except (TypeError, ValueError):
            return np.nan

    @st.cache_data(ttl=3600)
    def get_company_profile(tkr: str) -> dict:
        """Fetch company introduction data from yfinance. Cached 1hr per ticker."""
        try:
            import yfinance as yf
            info = yf.Ticker(tkr).info
            return {
                "name":          info.get("longName")           or info.get("shortName") or tkr,
                "description":   info.get("longBusinessSummary") or "",
                "website":       info.get("website")             or "",
                "employees":     info.get("fullTimeEmployees"),
                "headquarters":  ", ".join(filter(None, [
                                     info.get("city"),
                                     info.get("state"),
                                     info.get("country"),
                                 ])),
                "founded":       info.get("founded"),
                "exchange":      info.get("exchange")            or "",
                "industry":      info.get("industry")            or "",
                "sector":        info.get("sector")              or "",
                "ceo":           info.get("companyOfficers", [{}])[0].get("name", "") if info.get("companyOfficers") else "",
            }
        except Exception:
            return {}

    profile = get_company_profile(ticker)

    # Build the overview card
    company_name = profile.get("name") or ticker
    description  = profile.get("description") or "Company description not available."
    # Trim description to ~400 chars for readability
    if len(description) > 420:
        description = description[:420].rsplit(" ", 1)[0] + "…"

    industry      = profile.get("industry")     or str(row.get("sector") or "—")
    hq            = profile.get("headquarters") or "—"
    exchange      = profile.get("exchange")     or "—"
    website       = profile.get("website")      or ""
    employees     = profile.get("employees")
    emp_str       = f"{employees:,}" if employees else "—"
    cap_tier      = str(row.get("cap_tier") or "—").title()
    mcap          = safe_num(row.get("market_cap"))
    mcap_str      = f"${mcap/1e9:.1f}B" if not np.isnan(mcap) else "—"
    price         = safe_num(row.get("price"))
    price_str     = f"${price:,.2f}" if not np.isnan(price) else "—"
    website_html  = f'<a href="{website}" target="_blank" style="color:#60A5FA;text-decoration:none;">{website.replace("https://","").replace("http://","").rstrip("/")}</a>' if website else "—"

    st.markdown(f"""
<div style="background:#1E293B;border-radius:12px;padding:22px 26px;margin-bottom:20px;border:1px solid #334155;">

  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
    <div>
      <div style="font-size:24px;font-weight:700;color:#F8FAFC;line-height:1.2;">{company_name}</div>
      <div style="font-size:13px;color:#94A3B8;margin-top:3px;">{ticker} &nbsp;·&nbsp; {exchange} &nbsp;·&nbsp; {industry}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:22px;font-weight:700;color:#60A5FA;">{price_str}</div>
      <div style="font-size:12px;color:#94A3B8;">Current Price &nbsp;·&nbsp; Mkt Cap {mcap_str}</div>
    </div>
  </div>

  <div style="font-size:13.5px;color:#CBD5E1;line-height:1.7;margin-bottom:16px;border-top:1px solid #334155;padding-top:14px;">
    {description}
  </div>

  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
    <div style="background:#0F172A;border-radius:8px;padding:10px 14px;">
      <div style="font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Headquarters</div>
      <div style="font-size:13px;color:#E2E8F0;font-weight:500;">{hq if hq != "—" else "—"}</div>
    </div>
    <div style="background:#0F172A;border-radius:8px;padding:10px 14px;">
      <div style="font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Employees</div>
      <div style="font-size:13px;color:#E2E8F0;font-weight:500;">{emp_str}</div>
    </div>
    <div style="background:#0F172A;border-radius:8px;padding:10px 14px;">
      <div style="font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Cap Tier</div>
      <div style="font-size:13px;color:#E2E8F0;font-weight:500;">{cap_tier} Cap</div>
    </div>
    <div style="background:#0F172A;border-radius:8px;padding:10px 14px;">
      <div style="font-size:10px;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;">Website</div>
      <div style="font-size:13px;font-weight:500;">{website_html}</div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)
    # ── End company overview ──────────────────────────────────────────────────

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Market Price",  f"${row['price']:,.2f}")
    c2.metric("DCF Price",
              f"${row['dcf_price']:,.2f}" if pd.notna(row.get("dcf_price")) else "—",
              delta=f"{row['dcf_vs_market_pct']:.1%} vs market" if pd.notna(row.get("dcf_vs_market_pct")) else None)
    c3.metric("CCA Price",
              f"${row['cca_blended_price']:,.2f}" if pd.notna(row.get("cca_blended_price")) else "—",
              delta=f"{row['cca_vs_market_pct']:.1%} vs market" if pd.notna(row.get("cca_vs_market_pct")) else None)
    c4.metric("Terminal Value %",
              f"{row['terminal_value_pct']:.0%}" if pd.notna(row.get("terminal_value_pct")) else "—")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Price Comparison</div>', unsafe_allow_html=True)
        labels = ["Market","DCF","CCA","Analyst Target"]
        values = [row.get("price"), row.get("dcf_price"),
                  row.get("cca_blended_price"), row.get("target_price")]
        colors = [RED, BLUE, PURPLE, GREEN]
        valid  = [(l,v,c) for l,v,c in zip(labels,values,colors) if pd.notna(v)]
        fig = go.Figure(go.Bar(
            x=[v[0] for v in valid], y=[v[1] for v in valid],
            marker_color=[v[2] for v in valid],
            text=[f"${v[1]:,.0f}" for v in valid], textposition="outside"))
        fig.update_layout(template="plotly_white", height=320, yaxis_title="Price ($)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Model Inputs</div>', unsafe_allow_html=True)
        inputs = {
            "Sector":         row.get("sector","—"),
            "Beta":           f"{row['beta']:.2f}"          if pd.notna(row.get("beta"))       else "—",
            "WACC":           f"{row['wacc']:.2%}"          if pd.notna(row.get("wacc"))       else "—",
            "FCF":            f"${row['fcf']/1e6:,.0f}M"    if pd.notna(row.get("fcf"))        else "—",
            "EBITDA":         f"${row['ebitda']/1e6:,.0f}M" if pd.notna(row.get("ebitda"))     else "—",
            "EV/EBITDA":      f"{row['ev_ebitda']:.1f}x"    if pd.notna(row.get("ev_ebitda"))  else "—",
            "Debt":           f"${row['total_debt']/1e9:,.1f}B" if pd.notna(row.get("total_debt")) else "—",
            "Cash":           f"${row['cash']/1e9:,.1f}B"   if pd.notna(row.get("cash"))       else "—",
            "Analyst Target": f"${row['target_price']:,.2f}" if pd.notna(row.get("target_price")) else "—",
        }
        st.dataframe(pd.DataFrame(inputs.items(), columns=["Input","Value"]),
                     use_container_width=True, hide_index=True, height=320)

    if not mc_row.empty:
        mc = mc_row.iloc[0]
        st.markdown('<div class="section-header">Monte Carlo Summary</div>', unsafe_allow_html=True)
        m1,m2,m3,m4,m5 = st.columns(5)
        m1.metric("MC Mean",     f"${mc['mc_mean']:,.0f}")
        m2.metric("MC P5",       f"${mc['mc_p5']:,.0f}")
        m3.metric("MC P95",      f"${mc['mc_p95']:,.0f}")
        m4.metric("CV (Risk)",   f"{mc['mc_cv']:.3f}")
        m5.metric("P(>Market)",  f"{mc['prob_above_market']:.0%}" if pd.notna(mc.get("prob_above_market")) else "—")

    # ── Company-level error analysis (from the Error Analysis page, focused on this ticker) ──
    st.markdown("---")
    st.markdown('<div class="section-header">How This Company Compares to Its Peers</div>', unsafe_allow_html=True)

    sector      = row.get("sector")
    cap_tier    = row.get("cap_tier", "unknown")
    sector_df   = view_df[view_df["sector"] == sector].copy()
    tier_df     = view_df[view_df["cap_tier"] == cap_tier].copy() if "cap_tier" in view_df.columns else pd.DataFrame()

    pa1, pa2, pa3 = st.columns(3)

    # DCF error vs sector peers
    dcf_err_this   = row.get("dcf_vs_market_pct", np.nan)
    dcf_err_sector = sector_df["dcf_vs_market_pct"].abs().mean()
    dcf_err_tier   = tier_df["dcf_vs_market_pct"].abs().mean() if not tier_df.empty else np.nan

    pa1.markdown(f"""
**DCF Error**

This company: **{dcf_err_this:.1%}** {"🔴" if abs(dcf_err_this) > dcf_err_sector else "🟢"}

Sector avg ({sector}): {dcf_err_sector:.1%}

Cap tier avg ({cap_tier}): {dcf_err_tier:.1%}
    """)

    # CCA error vs sector peers
    cca_err_this   = row.get("cca_vs_market_pct", np.nan)
    cca_err_sector = sector_df["cca_vs_market_pct"].abs().mean()

    pa2.markdown(f"""
**CCA Error**

This company: **{cca_err_this:.1%}** {"🔴" if abs(cca_err_this) > cca_err_sector else "🟢"}

Sector avg ({sector}): {cca_err_sector:.1%}

🟢 = below sector avg error | 🔴 = above
    """)

    # Terminal value vs peers
    tv_this   = row.get("terminal_value_pct", np.nan)
    tv_sector = sector_df["terminal_value_pct"].mean()
    tv_tier   = tier_df["terminal_value_pct"].mean() if not tier_df.empty else np.nan

    pa3.markdown(f"""
**Terminal Value %**

This company: **{tv_this:.0%}**

Sector avg: {tv_sector:.0%}

Cap tier avg: {tv_tier:.0%}
    """)

    st.markdown("<br>", unsafe_allow_html=True)

    # Peer ranking chart — where does this company sit vs all sector peers on DCF error?
    ca1, ca2 = st.columns(2)

    with ca1:
        st.markdown('<div class="section-header">DCF Error: Ranked vs Sector Peers</div>', unsafe_allow_html=True)
        peer_dcf = sector_df[["ticker","dcf_vs_market_pct"]].dropna().copy()
        peer_dcf["abs_err"] = peer_dcf["dcf_vs_market_pct"].abs() * 100
        peer_dcf = peer_dcf.sort_values("abs_err")
        peer_dcf["color"] = peer_dcf["ticker"].apply(lambda t: RED if t == ticker else BLUE)
        fig_peer = go.Figure(go.Bar(
            x=peer_dcf["ticker"], y=peer_dcf["abs_err"],
            marker_color=peer_dcf["color"].tolist(),
            text=peer_dcf["abs_err"].map(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig_peer.update_layout(template="plotly_white", height=320,
                               xaxis_title="", yaxis_title="DCF Absolute Error (%)",
                               title=f"{ticker} highlighted in red vs {sector} peers",
                               showlegend=False)
        st.plotly_chart(fig_peer, use_container_width=True)

    with ca2:
        st.markdown('<div class="section-header">Price Spread: All Four Estimates</div>', unsafe_allow_html=True)
        # Waterfall-style: show all prices on a single axis to visualise the spread
        price_points = {
            "52w Low":        row.get("price_52w_low"),
            "CCA Price":      row.get("cca_blended_price"),
            "Analyst Low":    row.get("analyst_low"),
            "DCF Price":      row.get("dcf_price"),
            "Market Price":   row.get("price"),
            "Analyst Target": row.get("target_price"),
            "Analyst High":   row.get("analyst_high"),
            "52w High":       row.get("price_52w_high"),
        }
        valid_pts = {k: v for k, v in price_points.items() if pd.notna(v)}
        pt_colors = {
            "52w Low": GRAY, "52w High": GRAY,
            "Market Price": RED, "DCF Price": BLUE,
            "CCA Price": PURPLE, "Analyst Target": GREEN,
            "Analyst Low": "#8393AE", "Analyst High": "#597FC0",
        }
        fig_spread = go.Figure()
        sorted_pts = sorted(valid_pts.items(), key=lambda x: x[1])
        for label, val in sorted_pts:
            fig_spread.add_trace(go.Scatter(
                x=[val], y=[label],
                mode="markers+text",
                marker=dict(color=pt_colors.get(label, GRAY), size=12, symbol="diamond"),
                text=[f"${val:,.0f}"], textposition="middle right",
                showlegend=False,
            ))
        # Reference line at market price
        mkt = row.get("price")
        if mkt:
            fig_spread.add_vline(x=mkt, line_color=RED, line_width=1.5, line_dash="dash")
        fig_spread.update_layout(template="plotly_white", height=320,
                                 xaxis_title="Price ($)",
                                 title="Full price spectrum: models vs market vs analyst range",
                                 margin=dict(r=80))
        st.plotly_chart(fig_spread, use_container_width=True)

    # DCF vs CCA divergence interpretation
    div_pct = row.get("dcf_cca_divergence_pct", np.nan)
    if pd.notna(div_pct):
        abs_div = abs(div_pct) * 100
        if abs_div < 15:
            signal = " Low divergence — DCF and CCA are broadly aligned. Higher confidence in the implied price range."
            color  = "#D1FAE5"
        elif abs_div < 40:
            signal = " Moderate divergence — the two models disagree materially. Present both outputs to stakeholders rather than blending."
            color  = "#FEF9C3"
        else:
            signal = " High divergence — DCF and CCA are telling very different stories. This is a signal of genuine valuation uncertainty. Investigate the driver before using either number."
            color  = "#FEE2E2"

        st.markdown(
            f'<div style="background:{color};border-radius:8px;padding:14px 18px;margin-top:8px;font-size:14px;">'
            f'<b>DCF vs CCA Divergence: {div_pct:.1%}</b><br>{signal}</div>',
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SENSITIVITY SURFACE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Sensitivity Surface":
    st.title("DCF Sensitivity Surface")
    st.markdown("**How does DCF price change across WACC and terminal growth combinations?**")
    st.markdown("---")

    ticker = st.selectbox("Select company", sorted(view_df["ticker"].tolist()))
    row    = view_df[view_df["ticker"] == ticker].iloc[0]

    try:
        from valuation_engine import dcf_valuation
    except ImportError:
        st.error("valuation_engine.py must be in the same folder as dashboard.py.")
        st.stop()

    wacc_range   = np.linspace(0.04, 0.16, 30)
    growth_range = np.linspace(0.005, 0.05, 30)
    W, G = np.meshgrid(wacc_range, growth_range)
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            res = dcf_valuation(row["fcf"], W[i,j],
                                total_debt=row.get("total_debt",0),
                                cash=row.get("cash",0), shares=row.get("shares",1),
                                growth_stage2=G[i,j])
            val = res["dcf_price"] if isinstance(res, dict) else res
            Z[i,j] = val if val and not np.isnan(val) else 0

    market_p = row["price"]
    col1, col2 = st.columns([2,1])

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Contour(z=Z, x=wacc_range*100, y=growth_range*100,
                                 colorscale="RdYlGn",
                                 contours=dict(showlabels=True, labelfont=dict(size=10,color="white")),
                                 colorbar=dict(title="DCF Price ($)"), name="DCF Price"))
        fig.add_trace(go.Contour(z=Z-market_p, x=wacc_range*100, y=growth_range*100,
                                 contours=dict(type="constraint", operation="=", value=0,
                                               showlabels=True, labelfont=dict(size=10)),
                                 line=dict(color="black", width=2.5, dash="dash"),
                                 name=f"DCF = Market (${market_p:.0f})", showscale=False))
        fig.update_layout(template="plotly_white", height=460,
                          title=f"{ticker}  |  Black dashed = where DCF = market price (${market_p:.0f})",
                          xaxis_title="WACC (%)", yaxis_title="Terminal Growth Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        base_wacc = row.get("wacc", 0.09)
        r1 = dcf_valuation(row["fcf"], base_wacc+0.01,
                           total_debt=row.get("total_debt",0),
                           cash=row.get("cash",0), shares=row.get("shares",1))
        r2 = dcf_valuation(row["fcf"], base_wacc-0.01,
                           total_debt=row.get("total_debt",0),
                           cash=row.get("cash",0), shares=row.get("shares",1))
        p_plus  = r1["dcf_price"] if isinstance(r1, dict) else 0
        p_minus = r2["dcf_price"] if isinstance(r2, dict) else 0
        base_p  = row.get("dcf_price", 1) or 1

        st.markdown('<div class="section-header">Interpretation</div>', unsafe_allow_html=True)
        st.markdown(f"""
**Base WACC:** {base_wacc:.2%}

**+1% WACC impact:** {(p_plus-base_p)/base_p:+.0%}

**−1% WACC impact:** {(p_minus-base_p)/base_p:+.0%}

**Terminal Value %:** {row.get('terminal_value_pct',0):.0%}

---
 **Green** = DCF > market (undervalued)

 **Red** = DCF < market (overvalued)

 **Dashed** = DCF equals market price

Steep gradients = high model risk. Small WACC changes produce large price swings.
        """)

        st.markdown('<div class="section-header">Price Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
| | |
|---|---|
| Market Price | ${market_p:,.2f} |
| DCF Price | ${row.get('dcf_price',0):,.2f} |
| CCA Price | ${row.get('cca_blended_price',0):,.2f} |
| Analyst Target | ${row.get('target_price',0):,.2f} |
| DCF Error | {row.get('dcf_vs_market_pct',0):.1%} |
| CCA Error | {row.get('cca_vs_market_pct',0):.1%} |
        """)
