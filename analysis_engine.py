"""
Model Risk Analysis Engine — 6-panel chart including cap tier breakdown.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import warnings, os
warnings.filterwarnings('ignore')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

C = {"dcf":"#2563EB","cca":"#7C3AED","red":"#DC2626","green":"#059669","gray":"#6B7280",
     "large":"#1D4ED8","mid":"#7C3AED","small":"#DC2626"}

def load():
    return pd.read_csv(os.path.join(RESULTS_DIR, "valuations.csv"))

def plot_error_distribution(df, ax_main, ax_box):
    dcf_e = df["dcf_vs_market_pct"].dropna()*100
    cca_e = df["cca_vs_market_pct"].dropna()*100
    bins  = np.linspace(-200, 300, 32)
    ax_main.hist(dcf_e, bins=bins, alpha=0.65, color=C["dcf"],
                 label=f"DCF  MAE={dcf_e.abs().mean():.0f}%", edgecolor="white", lw=0.4)
    ax_main.hist(cca_e, bins=bins, alpha=0.55, color=C["cca"],
                 label=f"CCA  MAE={cca_e.abs().mean():.0f}%", edgecolor="white", lw=0.4)
    ax_main.axvline(0, color="black", lw=1.2, ls="--", alpha=0.6)
    ax_main.axvline(dcf_e.mean(), color=C["dcf"], lw=1.5, ls=":")
    ax_main.axvline(cca_e.mean(), color=C["cca"], lw=1.5, ls=":")
    ax_main.set_xlabel("Error vs Market (%)"); ax_main.set_ylabel("# Companies")
    ax_main.set_title("Error Distribution: DCF vs CCA", fontsize=12, fontweight="bold")
    ax_main.legend(fontsize=9); ax_main.spines[["top","right"]].set_visible(False)

    ax_box.boxplot([dcf_e, cca_e], labels=["DCF","CCA"], patch_artist=True,
                   boxprops=dict(facecolor="#EFF6FF", lw=1.2),
                   medianprops=dict(color="black", lw=2))
    ax_box.axhline(0, color="black", lw=1, ls="--", alpha=0.5)
    ax_box.set_ylabel("Error (%)"); ax_box.set_title("Error Spread", fontsize=11, fontweight="bold")
    ax_box.spines[["top","right"]].set_visible(False)
    return {"dcf_mae":dcf_e.abs().mean(),"dcf_bias":dcf_e.mean(),
            "cca_mae":cca_e.abs().mean(),"cca_bias":cca_e.mean()}

def plot_cap_tier_error(df, ax):
    """NEW: Error by cap tier — the key chart showing large vs mid vs small model risk."""
    tiers = ["large","mid","small"]
    tier_col = "cap_tier" if "cap_tier" in df.columns else None
    if tier_col is None:
        ax.text(0.5, 0.5, "cap_tier not in data", transform=ax.transAxes, ha="center")
        return

    x = np.arange(len(tiers)); w = 0.35
    dcf_mae = [df.loc[df[tier_col]==t,"dcf_vs_market_pct"].abs().mean()*100 for t in tiers]
    cca_mae = [df.loc[df[tier_col]==t,"cca_vs_market_pct"].abs().mean()*100 for t in tiers]

    bars1 = ax.bar(x-w/2, dcf_mae, w, label="DCF MAE", color=C["dcf"], alpha=0.85)
    bars2 = ax.bar(x+w/2, cca_mae, w, label="CCA MAE", color=C["cca"], alpha=0.85)
    ax.bar_label(bars1, fmt="%.0f%%", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.0f%%", fontsize=8, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(["Large Cap","Mid Cap","Small Cap"])
    ax.set_ylabel("Mean Absolute Error (%)"); ax.set_title("Model Error by Cap Tier", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.spines[["top","right"]].set_visible(False)

def plot_tv_dominance(df, ax):
    tv   = df["terminal_value_pct"].dropna()*100
    tier = df.loc[df["terminal_value_pct"].notna(), "cap_tier"] if "cap_tier" in df.columns else None
    colors = [C.get(t, C["gray"]) for t in (tier if tier is not None else ["gray"]*len(tv))]
    ax.scatter(df.loc[tv.index,"wacc"]*100, tv, c=colors, alpha=0.6, s=35,
               edgecolors="white", lw=0.4)
    ax.axhline(70, color=C["red"], lw=1.2, ls="--", alpha=0.7)
    ax.text(15.2, 71.5, "70%", fontsize=8, color=C["red"])
    ax.set_xlabel("WACC (%)"); ax.set_ylabel("Terminal Value %")
    ax.set_title("Terminal Value Dominance", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    for t, c in [("large","Large Cap"),("mid","Mid Cap"),("small","Small Cap")]:
        ax.scatter([],[], c=C[t], s=35, label=c)
    ax.legend(fontsize=8)
    return {"tv_mean":tv.mean(),"tv_above_70pct":(tv>70).mean()*100}

def plot_sector_heatmap(df, ax):
    sec = df.groupby("sector").agg(
        dcf_mae=("dcf_vs_market_pct",  lambda x: x.abs().mean()*100),
        cca_mae=("cca_vs_market_pct",  lambda x: x.abs().mean()*100),
        dcf_bias=("dcf_vs_market_pct", lambda x: x.mean()*100),
        cca_bias=("cca_vs_market_pct", lambda x: x.mean()*100),
    ).reset_index()
    matrix  = sec[["dcf_mae","cca_mae","dcf_bias","cca_bias"]].values
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(4)); ax.set_xticklabels(["DCF MAE","CCA MAE","DCF Bias","CCA Bias"], fontsize=9)
    ax.set_yticks(range(len(sec))); ax.set_yticklabels(sec["sector"].tolist(), fontsize=9)
    ax.set_title("Error by Sector (%)", fontsize=12, fontweight="bold")
    for i in range(len(sec)):
        for j in range(4):
            ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white" if matrix[i,j]>70 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

def plot_divergence(df, ax):
    div   = df["dcf_cca_divergence_pct"].dropna()*100
    dcf_e = df.loc[div.index,"dcf_vs_market_pct"]*100
    cca_e = df.loc[div.index,"cca_vs_market_pct"]*100
    min_e = pd.concat([dcf_e.abs(), cca_e.abs()], axis=1).min(axis=1)
    sc = ax.scatter(div.abs(), min_e, c=div, cmap="coolwarm", alpha=0.7,
                    s=30, edgecolors="white", lw=0.4, vmin=-150, vmax=150)
    ax.axvline(25, color=C["gray"], lw=1, ls="--", alpha=0.6)
    ax.set_xlabel("|DCF–CCA Divergence| (%)"); ax.set_ylabel("Best Model Error (%)")
    ax.set_title("Divergence as Uncertainty Signal", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="DCF > CCA (+)")
    corr, pval = stats.pearsonr(div.abs(), min_e)
    ax.text(0.05, 0.93, f"r={corr:.2f} (p={pval:.3f})", transform=ax.transAxes,
            fontsize=9, color=C["gray"])
    return {"divergence_corr":corr,"divergence_pval":pval}

def plot_sensitivity(df, ax):
    from valuation_engine import dcf_valuation
    sample = df[df["fcf"].notna() & (df["fcf"]>0)].iloc[0]
    W, G   = np.meshgrid(np.linspace(0.05,0.15,20), np.linspace(0.01,0.05,20))
    Z = np.zeros_like(W)
    bp = sample["price"]
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            res = dcf_valuation(fcf=sample["fcf"], wacc=W[i,j],
                                total_debt=sample.get("total_debt",0),
                                cash=sample.get("cash",0),
                                shares=sample.get("shares",1),
                                growth_stage2=G[i,j])
            val = res["dcf_price"] if isinstance(res, dict) else res
            Z[i,j] = ((val-bp)/bp*100) if val and not np.isnan(val) else 0
    cf = ax.contourf(W*100, G*100, Z, levels=20, cmap="RdYlGn")
    ax.contour(W*100, G*100, Z, levels=[0], colors="black", linewidths=1.5)
    plt.colorbar(cf, ax=ax, label="DCF vs Market (%)")
    ax.set_xlabel("WACC (%)"); ax.set_ylabel("Terminal Growth (%)")
    ax.set_title(f"Sensitivity: {sample['ticker']}", fontsize=12, fontweight="bold")

def run_analysis():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df  = load()
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax1a = fig.add_subplot(gs[0,:2]); ax1b = fig.add_subplot(gs[0,2])
    ax2  = fig.add_subplot(gs[1,0]);  ax3  = fig.add_subplot(gs[1,1])
    ax4  = fig.add_subplot(gs[1,2]);  ax5  = fig.add_subplot(gs[2,:2])
    ax6  = fig.add_subplot(gs[2,2])

    err = plot_error_distribution(df, ax1a, ax1b)
    plot_cap_tier_error(df, ax2)
    plot_tv_dominance(df, ax3)
    plot_sector_heatmap(df, ax4)
    div = plot_divergence(df, ax5)
    plot_sensitivity(df, ax6)

    n = len(df)
    fig.suptitle(f"Model Risk in Valuation — DCF vs CCA Audit Across {n} Companies",
                 fontsize=15, fontweight="bold", y=0.98)

    out = os.path.join(RESULTS_DIR, "model_risk_analysis.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Chart saved: {out}")
    print(f"\n── Key Findings ({n} companies) ─────────────────────")
    print(f"  DCF MAE:             {err['dcf_mae']:.1f}%  (bias: {err['dcf_bias']:+.1f}%)")
    print(f"  CCA MAE:             {err['cca_mae']:.1f}%  (bias: {err['cca_bias']:+.1f}%)")
    if "cap_tier" in df.columns:
        for t in ["large","mid","small"]:
            sub = df[df["cap_tier"]==t]
            mae = sub["dcf_vs_market_pct"].abs().mean()*100
            print(f"  DCF MAE ({t:6s} cap): {mae:.1f}%")
    tv_above = (df["terminal_value_pct"]>0.70).mean()*100
    tv_mean  = df["terminal_value_pct"].mean()*100
    print(f"  Terminal value >70%: {tv_above:.0f}% of companies")
    print(f"  Terminal value mean: {tv_mean:.1f}%")
    print(f"  Divergence corr:     r={div['divergence_corr']:.2f}")

if __name__ == "__main__":
    run_analysis()
