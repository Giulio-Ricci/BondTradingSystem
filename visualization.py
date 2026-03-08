"""Plotting utilities for the Bond Trading System."""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import SAVE_DIR

# ── Style defaults ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (14, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

COLORS = {
    "strategy": "#1f77b4",
    "benchmark": "#7f7f7f",
    "calm": "#2ca02c",
    "stress": "#d62728",
    "long": "#2ca02c",
    "short": "#d62728",
}


def _save(fig, name: str):
    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def setup_ax(ax, title: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=45)


# ═════════════════════════════════════════════════════════════════════════════
# Equity Curve
# ═════════════════════════════════════════════════════════════════════════════

def plot_equity(res: dict, D: dict, title: str = "Strategy Equity Curve"):
    """Plot strategy equity vs benchmark with regime shading."""
    fig, ax = plt.subplots()
    eq = res["equity"] * 1e7
    ax.plot(eq.index, eq.values, color=COLORS["strategy"], label="Strategy", linewidth=1.2)

    # Benchmark
    idx = eq.index
    etf_ret = D["etf_ret"]
    bm = etf_ret["TLT"].reindex(idx).fillna(0)
    bm_eq = (1 + bm).cumprod() * 1e7
    ax.plot(bm_eq.index, bm_eq.values, color=COLORS["benchmark"], label="B&H TLT",
            linewidth=0.8, alpha=0.7)

    # Regime shading
    regime = D.get("regime")
    if regime is not None:
        _shade_regime(ax, regime, idx)

    setup_ax(ax, title, "NAV ($)")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    _save(fig, "equity_curve")


def _shade_regime(ax, regime: pd.Series, idx):
    """Shade STRESS periods in light red."""
    reg = regime.reindex(idx).fillna("CALM")
    stress_mask = reg == "STRESS"
    ylim = ax.get_ylim()
    ax.fill_between(idx, ylim[0], ylim[1], where=stress_mask,
                    alpha=0.10, color="red", label="STRESS regime")
    ax.set_ylim(ylim)


# ═════════════════════════════════════════════════════════════════════════════
# Signal & Z-Score Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_zscore(D: dict, z_key: str = "z_move", params: dict = None):
    """Plot z-score time series with threshold lines and trade markers."""
    fig, ax = plt.subplots()
    z = D[z_key].dropna()
    ax.plot(z.index, z.values, color=COLORS["strategy"], linewidth=0.5, alpha=0.8)

    if params:
        ax.axhline(params.get("tl_calm", -3), color="red", ls="--", alpha=0.5, label="tl_calm")
        ax.axhline(params.get("ts_calm", 2.5), color="green", ls="--", alpha=0.5, label="ts_calm")
        ax.axhline(params.get("tl_stress", -4), color="darkred", ls=":", alpha=0.5, label="tl_stress")
    ax.axhline(0, color="black", ls="-", alpha=0.3)

    setup_ax(ax, f"Z-Score ({z_key})", "Z")
    ax.legend(loc="lower left", fontsize=8)
    _save(fig, f"zscore_{z_key}")


# ═════════════════════════════════════════════════════════════════════════════
# Sensitivity Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(sens_results: dict):
    """Plot 1D sensitivity sweeps for each parameter."""
    fig, axes = plt.subplots(1, len(sens_results), figsize=(5 * len(sens_results), 5))
    if len(sens_results) == 1:
        axes = [axes]

    for ax, (param_name, data) in zip(axes, sens_results.items()):
        ax.plot(data["values"], data["sharpes"], "o-", markersize=3, color=COLORS["strategy"])
        ax.set_xlabel(param_name)
        ax.set_ylabel("Sharpe")
        ax.set_title(f"Sensitivity: {param_name}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, "sensitivity")


# ═════════════════════════════════════════════════════════════════════════════
# Walk-Forward Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_walk_forward(wf_df: pd.DataFrame):
    """Bar chart of OOS excess return by year."""
    if len(wf_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [COLORS["long"] if v > 0 else COLORS["short"] for v in wf_df["oos_excess"]]
    ax.bar(wf_df["year"], wf_df["oos_excess"], color=colors, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("OOS Excess Return")
    ax.set_title("Walk-Forward: OOS Excess by Year")
    _save(fig, "walk_forward")


# ═════════════════════════════════════════════════════════════════════════════
# Bootstrap Distribution
# ═════════════════════════════════════════════════════════════════════════════

def plot_bootstrap(boot: dict):
    """Histogram of bootstrap Sharpe distribution."""
    # We only have CI summary - create a simple visual
    fig, ax = plt.subplots(figsize=(8, 5))
    ci = boot["sharpe_ci"]
    ax.axvline(ci[0], color="red", ls="--", label=f"2.5% = {ci[0]:.3f}")
    ax.axvline(ci[1], color="blue", ls="-", linewidth=2, label=f"Median = {ci[1]:.3f}")
    ax.axvline(ci[2], color="red", ls="--", label=f"97.5% = {ci[2]:.3f}")
    ax.axvline(0, color="black", ls=":", alpha=0.5)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title(f"Bootstrap Sharpe 95% CI  |  P(Sharpe>0)={boot['p_sharpe_pos']:.1%}")
    ax.legend()
    _save(fig, "bootstrap_sharpe")


# ═════════════════════════════════════════════════════════════════════════════
# TC Sensitivity Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_tc_sensitivity(tc_df: pd.DataFrame):
    """Sharpe vs transaction costs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tc_df["tc_bps"], tc_df["sharpe"], "o-", color=COLORS["strategy"], markersize=5)
    ax.axhline(0, color="black", ls=":", alpha=0.5)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Transaction Cost Sensitivity")
    be = tc_df.attrs.get("breakeven_tc", None)
    if be and np.isfinite(be):
        ax.axvline(be, color="red", ls="--", alpha=0.7, label=f"Breakeven: {be:.0f} bps")
        ax.legend()
    _save(fig, "tc_sensitivity")


# ═════════════════════════════════════════════════════════════════════════════
# Benchmark Comparison
# ═════════════════════════════════════════════════════════════════════════════

def plot_benchmarks(bm_df: pd.DataFrame, res: dict, D: dict):
    """Equity curves for all benchmarks."""
    fig, ax = plt.subplots()
    idx = res["equity"].index

    # Strategy
    eq = res["equity"] * 1e7
    ax.plot(eq.index, eq.values, color=COLORS["strategy"], label="Strategy", linewidth=1.5)

    # Other benchmarks
    etf_ret = D["etf_ret"]
    palette = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    i = 0
    for _, row in bm_df.iterrows():
        if row["name"] == "Strategy":
            continue
        name = row["name"]
        if "TLT" in name and "50" not in name:
            r = etf_ret["TLT"].reindex(idx).fillna(0)
        elif "TLH" in name:
            r = etf_ret["TLH"].reindex(idx).fillna(0)
        elif "IEF" in name:
            r = etf_ret["IEF"].reindex(idx).fillna(0)
        else:
            r = 0.5 * etf_ret["TLT"].reindex(idx).fillna(0) + 0.5 * etf_ret["SHV"].reindex(idx).fillna(0)
        bm_eq = (1 + r).cumprod() * 1e7
        ax.plot(bm_eq.index, bm_eq.values, color=palette[i % len(palette)],
                label=name, linewidth=0.8, alpha=0.7)
        i += 1

    setup_ax(ax, "Strategy vs Benchmarks", "NAV ($)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_yscale("log")
    _save(fig, "benchmarks")


# ═════════════════════════════════════════════════════════════════════════════
# Regime Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_regime(D: dict):
    """MOVE index with CALM/STRESS shading."""
    fig, ax = plt.subplots()
    move = D["move"].dropna()
    ax.plot(move.index, move.values, color="navy", linewidth=0.6)

    regime = D.get("regime")
    if regime is not None:
        _shade_regime(ax, regime, move.index)

    setup_ax(ax, "MOVE Index with Regime", "MOVE")
    _save(fig, "regime_move")


# ═════════════════════════════════════════════════════════════════════════════
# Drawdown Plot
# ═════════════════════════════════════════════════════════════════════════════

def plot_drawdown(res: dict):
    """Drawdown chart."""
    fig, ax = plt.subplots()
    eq = res["equity"]
    dd = eq / eq.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.4)
    setup_ax(ax, "Strategy Drawdown", "Drawdown")
    _save(fig, "drawdown")


def generate_all_plots(res: dict, D: dict, params: dict, z_key: str,
                       sens=None, wf_df=None, boot=None, tc_df=None, bm_df=None):
    """Generate all plots in one call."""
    print("\n  Generating plots...")
    plot_equity(res, D)
    plot_zscore(D, z_key, params)
    plot_regime(D)
    plot_drawdown(res)

    if sens:
        plot_sensitivity(sens)
    if wf_df is not None and len(wf_df) > 0:
        plot_walk_forward(wf_df)
    if boot:
        plot_bootstrap(boot)
    if tc_df is not None:
        plot_tc_sensitivity(tc_df)
    if bm_df is not None:
        plot_benchmarks(bm_df, res, D)
