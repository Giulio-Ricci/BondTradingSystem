"""New equity line comparison: multiple configs side by side."""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from config import SAVE_DIR, BACKTEST_START, DEFAULT_PARAMS
from data_loader import load_all
from signals import compute_all_signals
from regime import compute_regime
from backtest import backtest

SAVE = os.path.join(SAVE_DIR, "equity_new")
os.makedirs(SAVE, exist_ok=True)


def run_config(D, label, z_key, params_override):
    """Run a single backtest config and return results."""
    params = {**DEFAULT_PARAMS, **params_override}
    res = backtest(D[z_key], D["regime"], D["etf_ret"],
                   params=params, start=BACKTEST_START)
    m = res["metrics"]
    print(f"  {label:<45s}  Sharpe={m['sharpe']:+.3f}  "
          f"Excess={m['excess']:+.4f}  MDD={m['mdd']:.3f}  "
          f"Events={m['n_events']:>3d}  AnnRet={m['ann_ret']:.4f}")
    return res


def main():
    print("Loading data...")
    D = load_all()
    D = compute_all_signals(D)
    D = compute_regime(D)

    # =====================================================================
    # DEFINE CONFIGS
    # =====================================================================
    configs = {}

    # --- 1. Two main candidates ---
    configs["ACTIVE (tl=-3.0, ts=2.0)"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -3.0, "ts_calm": 2.0, "tl_stress": -4.0},
        "color": "#1f77b4", "ls": "-", "lw": 2.5,
    }
    configs["CONSERVATIVE (tl=-3.5, ts=3.0)"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0},
        "color": "#d62728", "ls": "-", "lw": 2.5,
    }

    # --- 2. Blend / Middle ground ---
    configs["BALANCED (tl=-3.0, ts=2.5)"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -3.0, "ts_calm": 2.5, "tl_stress": -4.0},
        "color": "#2ca02c", "ls": "-", "lw": 2.0,
    }
    configs["MODERATE (tl=-2.5, ts=2.0)"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -2.5, "ts_calm": 2.0, "tl_stress": -4.0},
        "color": "#ff7f0e", "ls": "-", "lw": 2.0,
    }

    # --- 3. Wider stress threshold ---
    configs["ACTIVE + tls=-3.5"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -3.0, "ts_calm": 2.0, "tl_stress": -3.5},
        "color": "#17becf", "ls": "--", "lw": 1.8,
    }
    configs["ACTIVE + tls=-5.0"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -3.0, "ts_calm": 2.0, "tl_stress": -5.0},
        "color": "#9467bd", "ls": "--", "lw": 1.8,
    }

    # --- 4. z_std variants ---
    configs["z_std ACTIVE (tl=-3.0, ts=2.0)"] = {
        "z_key": "z_std",
        "params": {"tl_calm": -3.0, "ts_calm": 2.0, "tl_stress": -4.0},
        "color": "#8c564b", "ls": "-.", "lw": 1.5,
    }
    configs["z_std CONSERV (tl=-3.5, ts=3.0)"] = {
        "z_key": "z_std",
        "params": {"tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0},
        "color": "#e377c2", "ls": "-.", "lw": 1.5,
    }

    # --- 5. Asymmetric configs ---
    configs["TIGHT SHORT (tl=-2.0, ts=2.5)"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -2.0, "ts_calm": 2.5, "tl_stress": -3.0},
        "color": "#bcbd22", "ls": ":", "lw": 1.5,
    }
    configs["WIDE SHORT (tl=-4.0, ts=2.0)"] = {
        "z_key": "z_move",
        "params": {"tl_calm": -4.0, "ts_calm": 2.0, "tl_stress": -4.5},
        "color": "#7f7f7f", "ls": ":", "lw": 1.5,
    }

    # =====================================================================
    # RUN ALL BACKTESTS
    # =====================================================================
    print(f"\n{'='*80}")
    print("RUNNING BACKTESTS")
    print(f"{'='*80}")

    results = {}
    for label, cfg in configs.items():
        results[label] = run_config(D, label, cfg["z_key"], cfg["params"])

    # =====================================================================
    # PLOT 1: ALL EQUITY CURVES (LOG SCALE)
    # =====================================================================
    print("\nGenerating Plot 1: All equity curves...")
    fig, ax = plt.subplots(figsize=(18, 10))

    for label, cfg in configs.items():
        res = results[label]
        eq = (1 + res["net_ret"]).cumprod() * 1e7
        ax.plot(eq.index, eq.values, color=cfg["color"], ls=cfg["ls"],
                lw=cfg["lw"], label=f"{label} (Sh={res['metrics']['sharpe']:.2f})",
                alpha=0.9)

    # B&H TLT
    idx = results[list(configs.keys())[0]]["net_ret"].index
    tlt = (1 + D["etf_ret"]["TLT"].reindex(idx).fillna(0)).cumprod() * 1e7
    ax.plot(tlt.index, tlt.values, color="black", ls="--", lw=1.5,
            alpha=0.5, label="B&H TLT")

    # Regime shading
    regime = D["regime"].reindex(idx).fillna("CALM")
    stress = regime == "STRESS"
    ylim = ax.get_ylim()
    ax.fill_between(idx, ylim[0], ylim[1], where=stress,
                    alpha=0.06, color="red", label="STRESS regime")
    ax.set_ylim(ylim)

    ax.set_yscale("log")
    ax.set_ylabel("NAV ($, log scale)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_title("Equity Curves - All Configurations (from $10M)", fontsize=14)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    path = os.path.join(SAVE, "01_all_equity_log.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =====================================================================
    # PLOT 2: DRAWDOWN COMPARISON (TOP 4)
    # =====================================================================
    print("Generating Plot 2: Drawdown comparison...")
    top4 = list(configs.keys())[:4]  # ACTIVE, CONSERVATIVE, BALANCED, MODERATE

    fig, axes = plt.subplots(len(top4), 1, figsize=(18, 12), sharex=True)
    for ax, label in zip(axes, top4):
        res = results[label]
        eq = (1 + res["net_ret"]).cumprod()
        dd = eq / eq.cummax() - 1
        ax.fill_between(dd.index, dd.values, 0, alpha=0.5,
                        color=configs[label]["color"])
        ax.set_ylabel("Drawdown")
        ax.set_title(f"{label}  |  MDD={res['metrics']['mdd']:.3f}  "
                     f"Sharpe={res['metrics']['sharpe']:.3f}", fontsize=10)
        ax.set_ylim(-0.35, 0.02)
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, alpha=0.2)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle("Drawdown Comparison - Top 4 Configs", fontsize=14, y=1.01)
    fig.tight_layout()

    path = os.path.join(SAVE, "02_drawdown_top4.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =====================================================================
    # PLOT 3: ACTIVE vs CONSERVATIVE vs BALANCED - Detailed
    # =====================================================================
    print("Generating Plot 3: Three-way comparison...")
    three = ["ACTIVE (tl=-3.0, ts=2.0)", "CONSERVATIVE (tl=-3.5, ts=3.0)",
             "BALANCED (tl=-3.0, ts=2.5)"]

    fig, axes = plt.subplots(3, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    # Panel A: Equity curves
    ax = axes[0]
    for label in three:
        res = results[label]
        eq = (1 + res["net_ret"]).cumprod() * 1e7
        c = configs[label]["color"]
        ax.plot(eq.index, eq.values, color=c, lw=2.5,
                label=f"{label} (Sh={res['metrics']['sharpe']:.2f})", alpha=0.9)

        # Mark trade events
        ev = res["events"]
        if len(ev) > 0:
            for _, e in ev.iterrows():
                dt = e["date"]
                if dt in eq.index:
                    marker = "^" if e["signal"] == "LONG" else "v"
                    mc = "green" if e["signal"] == "LONG" else "red"
                    ax.scatter(dt, eq.loc[dt], marker=marker, color=mc, s=40,
                              zorder=5, edgecolors="black", linewidths=0.3, alpha=0.7)

    tlt = (1 + D["etf_ret"]["TLT"].reindex(idx).fillna(0)).cumprod() * 1e7
    ax.plot(tlt.index, tlt.values, color="black", ls="--", lw=1.2, alpha=0.4,
            label="B&H TLT")

    regime = D["regime"].reindex(idx).fillna("CALM")
    stress = regime == "STRESS"
    ylim = ax.get_ylim()
    ax.fill_between(idx, ylim[0], ylim[1], where=stress,
                    alpha=0.06, color="red")
    ax.set_ylim(ylim)

    ax.set_yscale("log")
    ax.set_ylabel("NAV ($, log scale)", fontsize=11)
    ax.set_title("Three-Way Comparison: Active vs Conservative vs Balanced", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel B: Rolling 1Y excess vs TLT
    ax = axes[1]
    for label in three:
        res = results[label]
        strat_ret = res["net_ret"]
        tlt_ret = D["etf_ret"]["TLT"].reindex(strat_ret.index).fillna(0)
        excess_daily = strat_ret - tlt_ret
        rolling_excess = excess_daily.rolling(252, min_periods=126).sum()
        c = configs[label]["color"]
        ax.plot(rolling_excess.index, rolling_excess.values, color=c, lw=1.5, alpha=0.8)

    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_ylabel("Rolling 1Y Excess vs TLT", fontsize=10)
    ax.grid(True, alpha=0.2)

    # Panel C: Weight / Allocation
    ax = axes[2]
    for label in three:
        res = results[label]
        w = res["weight"]
        c = configs[label]["color"]
        ax.plot(w.index, w.values, color=c, lw=1.0, alpha=0.7, label=label)

    ax.axhline(0.5, color="black", lw=0.5, ls=":")
    ax.set_ylabel("Weight in TLT", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

    for a in axes:
        a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        a.xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].tick_params(axis="x", rotation=45)
    fig.tight_layout()

    path = os.path.join(SAVE, "03_three_way_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =====================================================================
    # PLOT 4: STRESS THRESHOLD SENSITIVITY
    # =====================================================================
    print("Generating Plot 4: Stress threshold sensitivity...")
    stress_labels = ["ACTIVE + tls=-3.5", "ACTIVE (tl=-3.0, ts=2.0)",
                     "ACTIVE + tls=-5.0"]

    fig, ax = plt.subplots(figsize=(18, 9))

    for label in stress_labels:
        res = results[label]
        eq = (1 + res["net_ret"]).cumprod() * 1e7
        c = configs[label]["color"]
        ls = configs[label]["ls"]
        ax.plot(eq.index, eq.values, color=c, ls=ls, lw=2.0,
                label=f"{label} (Sh={res['metrics']['sharpe']:.2f}, Ev={res['metrics']['n_events']})",
                alpha=0.9)

    tlt = (1 + D["etf_ret"]["TLT"].reindex(idx).fillna(0)).cumprod() * 1e7
    ax.plot(tlt.index, tlt.values, color="black", ls="--", lw=1.2, alpha=0.4,
            label="B&H TLT")

    ax.set_yscale("log")
    ax.set_ylabel("NAV ($, log scale)", fontsize=12)
    ax.set_title("Stress Threshold Sensitivity (tl_stress = -3.5, -4.0, -5.0)", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    path = os.path.join(SAVE, "04_stress_threshold.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =====================================================================
    # PLOT 5: z_move vs z_std
    # =====================================================================
    print("Generating Plot 5: z_move vs z_std...")
    zscore_labels = [
        "ACTIVE (tl=-3.0, ts=2.0)", "z_std ACTIVE (tl=-3.0, ts=2.0)",
        "CONSERVATIVE (tl=-3.5, ts=3.0)", "z_std CONSERV (tl=-3.5, ts=3.0)",
    ]

    fig, ax = plt.subplots(figsize=(18, 9))
    for label in zscore_labels:
        res = results[label]
        eq = (1 + res["net_ret"]).cumprod() * 1e7
        c = configs[label]["color"]
        ls = configs[label]["ls"]
        ax.plot(eq.index, eq.values, color=c, ls=ls, lw=configs[label]["lw"],
                label=f"{label} (Sh={res['metrics']['sharpe']:.2f})",
                alpha=0.9)

    tlt = (1 + D["etf_ret"]["TLT"].reindex(idx).fillna(0)).cumprod() * 1e7
    ax.plot(tlt.index, tlt.values, color="black", ls="--", lw=1.2, alpha=0.4,
            label="B&H TLT")

    ax.set_yscale("log")
    ax.set_ylabel("NAV ($, log scale)", fontsize=12)
    ax.set_title("z_move vs z_std: Same Thresholds, Different Z-Score Construction", fontsize=13)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    path = os.path.join(SAVE, "05_zmove_vs_zstd.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =====================================================================
    # PLOT 6: SUMMARY TABLE AS FIGURE
    # =====================================================================
    print("Generating Plot 6: Summary table...")
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis("off")

    col_labels = ["Config", "Z-Score", "tl_calm", "ts_calm", "tl_stress",
                  "Sharpe", "Ann Ret", "Excess", "MDD", "Vol", "Events"]
    table_data = []
    for label, cfg in configs.items():
        m = results[label]["metrics"]
        p = cfg["params"]
        short_label = label.split("(")[0].strip()
        table_data.append([
            short_label, cfg["z_key"],
            f"{p.get('tl_calm', DEFAULT_PARAMS['tl_calm']):.1f}",
            f"{p.get('ts_calm', DEFAULT_PARAMS['ts_calm']):.1f}",
            f"{p.get('tl_stress', DEFAULT_PARAMS['tl_stress']):.1f}",
            f"{m['sharpe']:.3f}",
            f"{m['ann_ret']:.3f}",
            f"{m['excess']:+.4f}",
            f"{m['mdd']:.3f}",
            f"{m['vol']:.3f}",
            f"{m['n_events']}",
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)

    # Color header
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight best Sharpe row
    sharpes = [results[l]["metrics"]["sharpe"] for l in configs]
    best_idx = np.argmax(sharpes)
    for j in range(len(col_labels)):
        table[(best_idx + 1, j)].set_facecolor("#E2EFDA")

    ax.set_title("Configuration Comparison Summary", fontsize=14, pad=20)

    path = os.path.join(SAVE, "06_summary_table.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =====================================================================
    # PLOT 7: YEARLY EXCESS HEATMAP
    # =====================================================================
    print("Generating Plot 7: Yearly excess heatmap...")

    yearly_excess = {}
    for label in configs:
        res = results[label]
        yr = res["yearly"]
        if len(yr) > 0:
            short_label = label.split("(")[0].strip()
            yearly_excess[short_label] = yr.set_index("year")["excess"]

    df_heatmap = pd.DataFrame(yearly_excess)

    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(df_heatmap.T.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.15, vmax=0.15)

    ax.set_xticks(range(len(df_heatmap.index)))
    ax.set_xticklabels(df_heatmap.index.astype(int), rotation=45, fontsize=9)
    ax.set_yticks(range(len(df_heatmap.columns)))
    ax.set_yticklabels(df_heatmap.columns, fontsize=9)

    # Add values
    for i in range(len(df_heatmap.columns)):
        for j in range(len(df_heatmap.index)):
            val = df_heatmap.T.iloc[i, j]
            if pd.notna(val):
                color = "white" if abs(val) > 0.10 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Yearly Excess Return")
    ax.set_title("Yearly Excess Return vs Benchmark - All Configs", fontsize=14)

    path = os.path.join(SAVE, "07_yearly_excess_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    print(f"\nAll plots saved to: {SAVE}")
    print("Done.")


if __name__ == "__main__":
    main()
