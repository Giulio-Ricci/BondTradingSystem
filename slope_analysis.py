"""
===============================================================================
  YIELD CURVE SLOPE ANALYSIS
  Scatterplots, Regime study, Rising/Falling performance, Historical periods

  Generates 15 PNG charts + 1 TXT documentation in output/slope_analysis/
===============================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from data_loader import load_all
from regime import fit_hmm_regime, fallback_regime

# ─── Output directory ─────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(__file__), "output", "slope_analysis")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
SPREADS = {
    "2s30s":  ("US_2Y",  "US_30Y"),
    "2s10s":  ("US_2Y",  "US_10Y"),
    "3m2y":   ("US_3M",  "US_2Y"),
    "2s5s":   ("US_2Y",  "US_5Y"),
    "5s10s":  ("US_5Y",  "US_10Y"),
    "10s30s": ("US_10Y", "US_30Y"),
}
SP_NAMES = list(SPREADS.keys())

HORIZONS = {"1d": 1, "5d": 5, "21d": 21, "63d": 63, "126d": 126, "252d": 252}

PERIODS = {
    "Pre-Crisis":   ("2007-01", "2008-12"),
    "Recovery":     ("2009-01", "2012-12"),
    "Taper Tantrum":("2013-01", "2013-12"),
    "Low Vol":      ("2014-01", "2019-12"),
    "COVID":        ("2020-01", "2020-12"),
    "Rate Hikes":   ("2021-01", "2023-12"),
    "Current":      ("2024-01", "2026-12"),
}

SCORE_MAP = {"BULL_FLAT": +2, "BULL_STEEP": +1, "UNDEFINED": 0,
             "BEAR_FLAT": -1, "BEAR_STEEP": -2}

ANN = 252
RISK_FREE = 0.02


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, name):
    path = os.path.join(SAVE_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {path}")


def compute_yc_regime(yld, short_col, long_col, window=21):
    """Classify each day into bull/bear steep/flat regime."""
    level  = (yld[short_col] + yld[long_col]) / 2
    spread = yld[long_col] - yld[short_col]
    dl = level.diff(window)
    ds = spread.diff(window)
    r = pd.Series("UNDEFINED", index=yld.index)
    r[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    r[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    r[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    r[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    return r


def _sharpe(returns):
    """Annualized Sharpe ratio from daily returns array."""
    r = np.asarray(returns)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return np.nan
    excess = r - RISK_FREE / ANN
    return np.mean(excess) / (np.std(excess) + 1e-10) * np.sqrt(ANN)


def _ann_ret(returns):
    r = np.asarray(returns)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return np.nan
    eq = np.prod(1 + r)
    return eq ** (ANN / len(r)) - 1


def _hit_rate(returns):
    r = np.asarray(returns)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    return np.mean(r > 0)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_data(D):
    """Build all derived series needed for the 15 charts."""
    yld     = D["yields"]
    etf_ret = D["etf_ret"]

    # Common index
    common = yld.index.intersection(etf_ret.index).sort_values()
    yld = yld.loc[common]
    tlt_ret = etf_ret.loc[common, "TLT"]

    out = {"yld": yld, "tlt_ret": tlt_ret, "dates": common}

    # 1) Spread levels
    spread_lvl = pd.DataFrame(index=common)
    for sp, (sc, lc) in SPREADS.items():
        spread_lvl[sp] = (yld[lc] - yld[sc]) * 100  # in bps
    out["spread_lvl"] = spread_lvl

    # 2) Spread changes (daily)
    spread_chg = spread_lvl.diff()
    out["spread_chg"] = spread_chg

    # 3) Level changes (average yield change) for 2s10s
    out["level_chg_21"] = ((yld["US_2Y"] + yld["US_10Y"]) / 2).diff(21)

    # 4) Forward TLT returns at all horizons
    fwd_tlt = pd.DataFrame(index=common)
    for hname, h in HORIZONS.items():
        fwd_tlt[hname] = tlt_ret.rolling(h).sum().shift(-h)
    out["fwd_tlt"] = fwd_tlt

    # 5) HMM regime
    try:
        hmm_regime = fit_hmm_regime(D["move"], D["vix"])
    except (ImportError, Exception):
        hmm_regime = fallback_regime(D["move"])
    out["hmm_regime"] = hmm_regime.reindex(common).fillna("CALM")

    # 6) YC regimes for 2s10s at window 21
    out["yc_regime_2s10s"] = compute_yc_regime(yld, "US_2Y", "US_10Y", 21)

    # 7) YC regimes for all spreads at window 21
    yc_regimes = {}
    for sp, (sc, lc) in SPREADS.items():
        yc_regimes[sp] = compute_yc_regime(yld, sc, lc, 21)
    out["yc_regimes"] = yc_regimes

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# A) SCATTERPLOT SECTION (01-04)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_01_slope_vs_fwd_tlt(data):
    """01: 2x3 grid — slope change 2s10s vs forward TLT at 6 horizons."""
    spread_chg = data["spread_chg"]["2s10s"]
    fwd_tlt = data["fwd_tlt"]
    regime = data["hmm_regime"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()

    for i, (hname, h) in enumerate(HORIZONS.items()):
        ax = axes[i]
        x = spread_chg
        y = fwd_tlt[hname]
        df = pd.DataFrame({"x": x, "y": y, "regime": regime}).dropna()
        if len(df) < 30:
            ax.set_title(f"{hname}: insufficient data")
            continue

        # Color by regime
        colors = df["regime"].map({"CALM": "#2ca02c", "STRESS": "#d62728"})
        ax.scatter(df["x"], df["y"], c=colors, alpha=0.25, s=8, edgecolors="none")

        # Regression
        slope, intercept, r_val, p_val, _ = stats.linregress(df["x"], df["y"])
        xr = np.linspace(df["x"].min(), df["x"].max(), 100)
        ax.plot(xr, intercept + slope * xr, "k-", lw=1.5, alpha=0.8)

        ax.set_title(f"2s10s chg vs Fwd TLT {hname}\n"
                     f"R²={r_val**2:.3f}  β={slope:.4f}  p={p_val:.1e}",
                     fontsize=10)
        ax.set_xlabel("2s10s spread change (bps)")
        ax.set_ylabel(f"Forward TLT return ({hname})")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        ax.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        Patch(facecolor="#2ca02c", label="CALM"),
        Patch(facecolor="#d62728", label="STRESS"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Slope Change (2s10s) vs Forward TLT Returns — by HMM Regime",
                 y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "01_slope_vs_fwd_tlt")
    return True


def plot_02_level_vs_spread_scatter(data):
    """02: Level change vs spread change, colored by forward TLT 63d."""
    yld = data["yld"]
    level_chg = data["level_chg_21"]
    spread_chg_21 = (data["spread_lvl"]["2s10s"]).diff(21)
    fwd63 = data["fwd_tlt"]["63d"]

    df = pd.DataFrame({
        "level_chg": level_chg,
        "spread_chg": spread_chg_21,
        "fwd_tlt_63d": fwd63,
    }).dropna()

    fig, ax = plt.subplots(figsize=(12, 9))
    sc = ax.scatter(df["level_chg"], df["spread_chg"],
                    c=df["fwd_tlt_63d"], cmap="RdYlGn", alpha=0.5, s=12,
                    edgecolors="none", vmin=-0.15, vmax=0.15)
    plt.colorbar(sc, ax=ax, label="Forward TLT 63d return")

    # Quadrant labels
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.text(0.02, 0.98, "BULL STEEP\n(yields↓, curve steepens)",
            transform=ax.transAxes, va="top", fontsize=9, color="#2ca02c",
            fontweight="bold")
    ax.text(0.98, 0.98, "BEAR STEEP\n(yields↑, curve steepens)",
            transform=ax.transAxes, va="top", ha="right", fontsize=9,
            color="#e67e22", fontweight="bold")
    ax.text(0.02, 0.02, "BULL FLAT\n(yields↓, curve flattens)",
            transform=ax.transAxes, fontsize=9, color="#3498db",
            fontweight="bold")
    ax.text(0.98, 0.02, "BEAR FLAT\n(yields↑, curve flattens)",
            transform=ax.transAxes, ha="right", fontsize=9, color="#e74c3c",
            fontweight="bold")

    ax.set_xlabel("21d Change in Yield Level (avg 2Y+10Y)", fontsize=11)
    ax.set_ylabel("21d Change in 2s10s Spread (bps)", fontsize=11)
    ax.set_title("Yield Level Change vs Spread Change — Color = Forward TLT 63d",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    _save(fig, "02_level_vs_spread_scatter")
    return True


def plot_03_spread_correlation(data):
    """03: Heatmap correlation tra i 6 spread (daily changes)."""
    corr = data["spread_chg"].dropna().corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    n = len(corr)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            color = "white" if abs(v) > 0.7 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, color=color)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Daily Spread Change Correlation Matrix",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_spread_correlation")
    return corr


def plot_04_spread_vs_tlt_scatters(data):
    """04: 2x3 grid — each spread change vs TLT return (daily)."""
    spread_chg = data["spread_chg"]
    tlt_ret = data["tlt_ret"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.ravel()

    results = {}
    for i, sp in enumerate(SP_NAMES):
        ax = axes[i]
        df = pd.DataFrame({"x": spread_chg[sp], "y": tlt_ret}).dropna()
        if len(df) < 30:
            continue

        ax.scatter(df["x"], df["y"], alpha=0.15, s=6, color="#1f77b4",
                   edgecolors="none")

        slope, intercept, r_val, p_val, _ = stats.linregress(df["x"], df["y"])
        xr = np.linspace(df["x"].quantile(0.01), df["x"].quantile(0.99), 100)
        ax.plot(xr, intercept + slope * xr, "r-", lw=1.5)

        results[sp] = {"beta": slope, "r2": r_val**2, "p": p_val}

        ax.set_title(f"{sp} chg vs TLT daily\n"
                     f"β={slope:.5f}  R²={r_val**2:.4f}",
                     fontsize=10)
        ax.set_xlabel(f"{sp} change (bps)")
        ax.set_ylabel("TLT daily return")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Spread Changes vs TLT Daily Returns",
                 y=1.01, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "04_spread_vs_tlt_scatters")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B) REGIME ANALYSIS (05-09)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_05_tlt_by_regime_bar(data):
    """05: Mean annualized TLT return per YC regime (bar + error bar)."""
    regime = data["yc_regime_2s10s"]
    tlt_ret = data["tlt_ret"]

    regimes_order = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT", "UNDEFINED"]
    colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b", "#95a5a6"]

    df = pd.DataFrame({"ret": tlt_ret, "regime": regime}).dropna()
    df = df[df["regime"] != "UNDEFINED"]

    stats_list = []
    for reg in regimes_order:
        sub = df[df["regime"] == reg]["ret"]
        if len(sub) < 10:
            stats_list.append({"regime": reg, "mean": 0, "std": 0, "n": 0})
            continue
        ann = sub.mean() * ANN
        vol = sub.std() * np.sqrt(ANN)
        stats_list.append({"regime": reg, "mean": ann, "std": vol, "n": len(sub)})
    sdf = pd.DataFrame(stats_list)
    sdf = sdf[sdf["n"] > 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sdf))
    c = [colors[regimes_order.index(r)] for r in sdf["regime"]]
    bars = ax.bar(x, sdf["mean"] * 100, color=c, edgecolor="black", lw=0.5)
    ax.errorbar(x, sdf["mean"] * 100, yerr=sdf["std"] * 100 / np.sqrt(sdf["n"]) * 1.96,
                fmt="none", color="black", capsize=5, lw=1.5)

    ax.set_xticks(x)
    labels = [f"{r}\n(n={n})" for r, n in zip(sdf["regime"], sdf["n"])]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Annualized TLT Return (%)")
    ax.set_title("Mean Annualized TLT Return by YC Regime (2s10s, w=21)\n"
                 "Error bars = 95% CI",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, "05_tlt_by_regime_bar")
    return sdf


def plot_06_regime_frequency(data):
    """06: Stacked area chart — rolling 126d proportion of each regime."""
    regime = data["yc_regime_2s10s"]
    regimes_order = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]
    colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]

    # Create dummy columns
    df = pd.DataFrame(index=data["dates"])
    for reg in regimes_order:
        df[reg] = (regime == reg).astype(float)

    # Rolling proportion
    rolling = df.rolling(126, min_periods=21).mean().dropna()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.stackplot(rolling.index, *[rolling[r].values for r in regimes_order],
                 labels=regimes_order, colors=colors, alpha=0.8)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1)
    ax.set_title("Rolling 126d Regime Frequency (2s10s, w=21)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    _save(fig, "06_regime_frequency")
    return True


def plot_07_hit_rate_by_regime(data):
    """07: Hit rate (% TLT positive) per regime at 1d, 5d, 21d."""
    regime = data["yc_regime_2s10s"]
    fwd_tlt = data["fwd_tlt"]

    regimes_order = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]
    horizons_sel = ["1d", "5d", "21d"]
    colors_h = ["#3498db", "#2ecc71", "#e67e22"]

    results = {}
    for reg in regimes_order:
        mask = regime == reg
        results[reg] = {}
        for hname in horizons_sel:
            sub = fwd_tlt.loc[mask, hname].dropna()
            results[reg][hname] = np.mean(sub > 0) * 100 if len(sub) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regimes_order))
    w = 0.25
    for i, hname in enumerate(horizons_sel):
        vals = [results[reg][hname] for reg in regimes_order]
        ax.bar(x + i * w, vals, w, label=hname, color=colors_h[i],
               edgecolor="black", lw=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels(regimes_order, fontsize=10)
    ax.set_ylabel("Hit Rate (% TLT > 0)")
    ax.axhline(50, color="gray", ls="--", lw=1, label="50%")
    ax.set_title("TLT Positive Hit Rate by YC Regime (2s10s, w=21)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, "07_hit_rate_by_regime")
    return results


def plot_08_tlt_boxplot_regime(data):
    """08: Boxplot TLT returns 21d per regime."""
    regime = data["yc_regime_2s10s"]
    fwd_tlt = data["fwd_tlt"]["21d"]

    regimes_order = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]
    colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]

    box_data = []
    labels = []
    for reg in regimes_order:
        mask = regime == reg
        sub = fwd_tlt.loc[mask].dropna()
        if len(sub) > 5:
            box_data.append(sub.values * 100)
            labels.append(f"{reg}\n(n={len(sub)})")

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=2, alpha=0.3))
    for patch, color in zip(bp["boxes"], colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("TLT Forward 21d Return (%)")
    ax.set_title("Distribution of TLT 21d Forward Returns by YC Regime",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, "08_tlt_boxplot_regime")
    return True


def plot_09_transition_persistence(data):
    """09: Transition matrix + average duration per regime."""
    regime = data["yc_regime_2s10s"]
    regimes_order = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]

    # Filter out UNDEFINED
    reg_clean = regime[regime.isin(regimes_order)]

    # Transition matrix
    n = len(regimes_order)
    trans = np.zeros((n, n))
    reg_vals = reg_clean.values
    for i in range(len(reg_vals) - 1):
        fr = regimes_order.index(reg_vals[i])
        to = regimes_order.index(reg_vals[i + 1])
        trans[fr, to] += 1

    # Normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_pct = trans / row_sums

    # Average duration (consecutive days in same regime)
    durations = {r: [] for r in regimes_order}
    current = reg_vals[0] if len(reg_vals) > 0 else None
    count = 1
    for i in range(1, len(reg_vals)):
        if reg_vals[i] == current:
            count += 1
        else:
            if current in durations:
                durations[current].append(count)
            current = reg_vals[i]
            count = 1
    if current in durations:
        durations[current].append(count)

    avg_dur = {r: np.mean(d) if len(d) > 0 else 0 for r, d in durations.items()}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6),
                                    gridspec_kw={"width_ratios": [2, 1]})

    # Transition matrix heatmap
    im = ax1.imshow(trans_pct, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(regimes_order, rotation=45, ha="right", fontsize=9)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(regimes_order, fontsize=9)
    ax1.set_xlabel("To")
    ax1.set_ylabel("From")
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f"{trans_pct[i,j]:.1%}", ha="center", va="center",
                     fontsize=10, color="white" if trans_pct[i,j] > 0.5 else "black")
    plt.colorbar(im, ax=ax1, label="Probability", shrink=0.8)
    ax1.set_title("Regime Transition Matrix", fontsize=12, fontweight="bold")

    # Average duration bar chart
    colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]
    x = np.arange(n)
    ax2.barh(x, [avg_dur[r] for r in regimes_order], color=colors,
             edgecolor="black", lw=0.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(regimes_order, fontsize=9)
    ax2.set_xlabel("Average Duration (days)")
    ax2.set_title("Mean Regime Persistence", fontsize=12, fontweight="bold")
    for i, r in enumerate(regimes_order):
        ax2.text(avg_dur[r] + 0.5, i, f"{avg_dur[r]:.1f}d", va="center", fontsize=10)
    ax2.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    _save(fig, "09_transition_persistence")
    return {"trans_pct": trans_pct, "avg_dur": avg_dur}


# ═══════════════════════════════════════════════════════════════════════════════
# C) RISING vs FALLING YIELDS (10-11)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_10_rising_falling_performance(data):
    """10: 2x2 — returns and hit rate for Rising/Falling and Steep/Flat."""
    yld = data["yld"]
    tlt_ret = data["tlt_ret"]
    fwd_tlt = data["fwd_tlt"]

    windows = [21, 63, 126]
    window_labels = ["21d", "63d", "126d"]

    # Classify by yield direction and curve shape
    results = {"Rising": {}, "Falling": {}, "Steepening": {}, "Flattening": {}}

    for wi, win in enumerate(windows):
        level_chg = ((yld["US_2Y"] + yld["US_10Y"]) / 2).diff(win)
        spread_chg = (yld["US_10Y"] - yld["US_2Y"]).diff(win)

        rising = level_chg > 0
        falling = level_chg < 0
        steepening = spread_chg > 0
        flattening = spread_chg < 0

        wlbl = window_labels[wi]
        for label, mask in [("Rising", rising), ("Falling", falling),
                            ("Steepening", steepening), ("Flattening", flattening)]:
            sub = tlt_ret[mask].dropna()
            results[label][wlbl] = {
                "ann_ret": sub.mean() * ANN,
                "hit_rate": (sub > 0).mean() * 100 if len(sub) > 0 else np.nan,
                "n": len(sub),
            }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Annualized return by yield direction
    ax = axes[0, 0]
    x = np.arange(len(windows))
    w = 0.35
    r_vals = [results["Rising"][wl]["ann_ret"] * 100 for wl in window_labels]
    f_vals = [results["Falling"][wl]["ann_ret"] * 100 for wl in window_labels]
    ax.bar(x - w/2, r_vals, w, label="Rising Yields", color="#e74c3c", edgecolor="black", lw=0.5)
    ax.bar(x + w/2, f_vals, w, label="Falling Yields", color="#2ecc71", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(window_labels)
    ax.set_ylabel("Ann. TLT Return (%)")
    ax.set_title("TLT Return: Rising vs Falling Yields")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.2, axis="y")

    # Top-right: Hit rate by yield direction
    ax = axes[0, 1]
    r_hr = [results["Rising"][wl]["hit_rate"] for wl in window_labels]
    f_hr = [results["Falling"][wl]["hit_rate"] for wl in window_labels]
    ax.bar(x - w/2, r_hr, w, label="Rising Yields", color="#e74c3c", edgecolor="black", lw=0.5)
    ax.bar(x + w/2, f_hr, w, label="Falling Yields", color="#2ecc71", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(window_labels)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("TLT Hit Rate: Rising vs Falling")
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    # Bottom-left: Annualized return by curve shape
    ax = axes[1, 0]
    s_vals = [results["Steepening"][wl]["ann_ret"] * 100 for wl in window_labels]
    fl_vals = [results["Flattening"][wl]["ann_ret"] * 100 for wl in window_labels]
    ax.bar(x - w/2, s_vals, w, label="Steepening", color="#3498db", edgecolor="black", lw=0.5)
    ax.bar(x + w/2, fl_vals, w, label="Flattening", color="#e67e22", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(window_labels)
    ax.set_ylabel("Ann. TLT Return (%)")
    ax.set_title("TLT Return: Steepening vs Flattening")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.2, axis="y")

    # Bottom-right: Hit rate by curve shape
    ax = axes[1, 1]
    s_hr = [results["Steepening"][wl]["hit_rate"] for wl in window_labels]
    fl_hr = [results["Flattening"][wl]["hit_rate"] for wl in window_labels]
    ax.bar(x - w/2, s_hr, w, label="Steepening", color="#3498db", edgecolor="black", lw=0.5)
    ax.bar(x + w/2, fl_hr, w, label="Flattening", color="#e67e22", edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(window_labels)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("TLT Hit Rate: Steepening vs Flattening")
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Rising/Falling Yields & Steepening/Flattening — TLT Performance",
                 y=1.01, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_rising_falling_performance")
    return results


def plot_11_four_quadrant(data):
    """11: 4 combined scenarios — return, Sharpe, hit rate."""
    yld = data["yld"]
    tlt_ret = data["tlt_ret"]

    scenarios = {
        "Rising+Steep":  {"dl": ">0", "ds": ">0"},
        "Rising+Flat":   {"dl": ">0", "ds": "<0"},
        "Falling+Steep": {"dl": "<0", "ds": ">0"},
        "Falling+Flat":  {"dl": "<0", "ds": "<0"},
    }
    colors = ["#e67e22", "#e74c3c", "#2ecc71", "#3498db"]

    level_chg = ((yld["US_2Y"] + yld["US_10Y"]) / 2).diff(21)
    spread_chg = (yld["US_10Y"] - yld["US_2Y"]).diff(21)

    masks = {
        "Rising+Steep":  (level_chg > 0) & (spread_chg > 0),
        "Rising+Flat":   (level_chg > 0) & (spread_chg < 0),
        "Falling+Steep": (level_chg < 0) & (spread_chg > 0),
        "Falling+Flat":  (level_chg < 0) & (spread_chg < 0),
    }

    results = {}
    for scen, mask in masks.items():
        sub = tlt_ret[mask].dropna()
        results[scen] = {
            "ann_ret": sub.mean() * ANN if len(sub) > 0 else np.nan,
            "sharpe": _sharpe(sub.values) if len(sub) > 10 else np.nan,
            "hit_rate": (sub > 0).mean() * 100 if len(sub) > 0 else np.nan,
            "n": len(sub),
        }

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    scen_names = list(scenarios.keys())
    x = np.arange(len(scen_names))

    # Return
    ax = axes[0]
    vals = [results[s]["ann_ret"] * 100 for s in scen_names]
    ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scen_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Ann. Return (%)")
    ax.set_title("Annualized TLT Return")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.2, axis="y")

    # Sharpe
    ax = axes[1]
    vals = [results[s]["sharpe"] for s in scen_names]
    ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scen_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Ratio")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.2, axis="y")

    # Hit Rate
    ax = axes[2]
    vals = [results[s]["hit_rate"] for s in scen_names]
    n_vals = [results[s]["n"] for s in scen_names]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(scen_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Hit Rate (% TLT > 0)")
    ax.axhline(50, color="gray", ls="--", lw=1)
    ax.grid(True, alpha=0.2, axis="y")
    for i, (v, n) in enumerate(zip(vals, n_vals)):
        if not np.isnan(v):
            ax.text(i, v + 1, f"n={n}", ha="center", fontsize=8)

    fig.suptitle("Four-Quadrant Analysis: Level Direction x Curve Shape (21d window)",
                 y=1.02, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "11_four_quadrant")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# D) HISTORICAL PERIODS (12-13)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_12_period_regime_returns(data):
    """12: TLT return per period and per regime (grouped bar)."""
    regime = data["yc_regime_2s10s"]
    tlt_ret = data["tlt_ret"]

    regimes_order = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]
    regime_colors = ["#2ecc71", "#27ae60", "#e74c3c", "#c0392b"]

    results = {}
    for pname, (start, end) in PERIODS.items():
        mask = (tlt_ret.index >= start) & (tlt_ret.index <= end)
        sub_ret = tlt_ret[mask]
        sub_reg = regime.reindex(sub_ret.index)
        results[pname] = {}
        for reg in regimes_order:
            r = sub_ret[sub_reg == reg]
            results[pname][reg] = r.mean() * ANN if len(r) > 5 else np.nan

    period_names = list(PERIODS.keys())
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(period_names))
    w = 0.2

    for i, reg in enumerate(regimes_order):
        vals = [results[p].get(reg, np.nan) * 100 for p in period_names]
        offset = (i - 1.5) * w
        ax.bar(x + offset, vals, w, label=reg, color=regime_colors[i],
               edgecolor="black", lw=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(period_names, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Annualized TLT Return (%)")
    ax.set_title("TLT Returns by Period and YC Regime",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    _save(fig, "12_period_regime_returns")
    return results


def plot_13_period_spread_analysis(data):
    """13: Spread level per period + heatmap correlation spread-TLT per period."""
    spread_lvl = data["spread_lvl"]
    tlt_ret = data["tlt_ret"]
    spread_chg = data["spread_chg"]

    period_names = list(PERIODS.keys())

    # Top: Average spread level per period
    avg_spread = {}
    for pname, (start, end) in PERIODS.items():
        mask = (spread_lvl.index >= start) & (spread_lvl.index <= end)
        avg_spread[pname] = spread_lvl[mask].mean()

    avg_df = pd.DataFrame(avg_spread).T

    # Bottom: Correlation spread change vs TLT per period
    corr_matrix = {}
    for pname, (start, end) in PERIODS.items():
        mask = (spread_chg.index >= start) & (spread_chg.index <= end)
        sc = spread_chg[mask]
        tr = tlt_ret[mask]
        corr_row = {}
        for sp in SP_NAMES:
            df_tmp = pd.DataFrame({"sp": sc[sp], "tlt": tr}).dropna()
            corr_row[sp] = df_tmp["sp"].corr(df_tmp["tlt"]) if len(df_tmp) > 20 else np.nan
        corr_matrix[pname] = corr_row

    corr_df = pd.DataFrame(corr_matrix).T

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Top: Grouped bar for spread levels
    x = np.arange(len(period_names))
    n_sp = len(SP_NAMES)
    w = 0.8 / n_sp
    sp_colors = plt.cm.Set2(np.linspace(0, 1, n_sp))
    for i, sp in enumerate(SP_NAMES):
        vals = [avg_df.loc[p, sp] if p in avg_df.index else np.nan for p in period_names]
        ax1.bar(x + (i - n_sp/2 + 0.5) * w, vals, w, label=sp,
                color=sp_colors[i], edgecolor="black", lw=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(period_names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Spread Level (bps)")
    ax1.set_title("Average Spread Levels by Historical Period",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.2, axis="y")

    # Bottom: Heatmap
    corr_vals = corr_df.reindex(period_names)[SP_NAMES].values
    im = ax2.imshow(corr_vals, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(SP_NAMES)))
    ax2.set_xticklabels(SP_NAMES, fontsize=9)
    ax2.set_yticks(range(len(period_names)))
    ax2.set_yticklabels(period_names, fontsize=9)
    for i in range(len(period_names)):
        for j in range(len(SP_NAMES)):
            v = corr_vals[i, j]
            if np.isfinite(v):
                color = "white" if abs(v) > 0.5 else "black"
                ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=9, color=color)
    plt.colorbar(im, ax=ax2, label="Correlation", shrink=0.8)
    ax2.set_title("Spread Change vs TLT Return Correlation by Period",
                  fontsize=12, fontweight="bold")

    plt.tight_layout()
    _save(fig, "13_period_spread_analysis")
    return {"avg_spread": avg_df, "corr": corr_df}


# ═══════════════════════════════════════════════════════════════════════════════
# E) ADDITIONAL ANALYTICS (14-15)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_14_rolling_correlation(data):
    """14: Rolling 126d correlation spread change vs TLT + term structure."""
    import matplotlib.dates as mdates

    spread_chg = data["spread_chg"]
    tlt_ret = data["tlt_ret"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    sp_colors = plt.cm.tab10(np.linspace(0, 1, len(SP_NAMES)))

    # Top: Rolling correlation for each spread
    for i, sp in enumerate(SP_NAMES):
        df_tmp = pd.DataFrame({"sp": spread_chg[sp], "tlt": tlt_ret}).dropna()
        if len(df_tmp) < 126:
            continue
        rolling_corr = df_tmp["sp"].rolling(126, min_periods=63).corr(df_tmp["tlt"])
        ax1.plot(rolling_corr.index, rolling_corr.values, lw=1, alpha=0.7,
                 label=sp, color=sp_colors[i])
    ax1.axhline(0, color="black", ls="--", lw=0.5)
    ax1.set_ylabel("Correlation")
    ax1.set_title("Rolling 126d Correlation: Spread Change vs TLT Return",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(-1, 1)

    # Bottom: Term structure of correlation (average by maturity buckets)
    # Compute static correlation for each spread
    static_corr = {}
    for sp in SP_NAMES:
        df_tmp = pd.DataFrame({"sp": spread_chg[sp], "tlt": tlt_ret}).dropna()
        static_corr[sp] = df_tmp["sp"].corr(df_tmp["tlt"])

    x = np.arange(len(SP_NAMES))
    colors_bar = [sp_colors[i] for i in range(len(SP_NAMES))]
    ax2.bar(x, [static_corr[sp] for sp in SP_NAMES], color=colors_bar,
            edgecolor="black", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(SP_NAMES, fontsize=10)
    ax2.set_ylabel("Correlation")
    ax2.set_title("Full-Sample Correlation: Spread Change vs TLT",
                  fontsize=12, fontweight="bold")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.grid(True, alpha=0.2, axis="y")
    for i, sp in enumerate(SP_NAMES):
        ax2.text(i, static_corr[sp], f"{static_corr[sp]:.3f}", ha="center",
                 va="bottom" if static_corr[sp] >= 0 else "top", fontsize=9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.tight_layout()
    _save(fig, "14_rolling_correlation")
    return static_corr


def plot_15_slope_momentum(data):
    """15: Forward TLT return based on past steepening/flattening momentum."""
    yld = data["yld"]
    fwd_tlt = data["fwd_tlt"]

    # Past spread change over various lookback windows
    spread_2s10s = (yld["US_10Y"] - yld["US_2Y"]) * 100  # bps
    lookbacks = [21, 63, 126]
    fwd_horizons = ["21d", "63d"]

    fig, axes = plt.subplots(len(lookbacks), len(fwd_horizons),
                              figsize=(14, 12))

    results = {}
    for i, lb in enumerate(lookbacks):
        past_chg = spread_2s10s.diff(lb)
        steepening = past_chg > 0
        flattening = past_chg < 0

        for j, fh in enumerate(fwd_horizons):
            ax = axes[i, j]
            fwd = fwd_tlt[fh]

            steep_ret = fwd[steepening].dropna()
            flat_ret = fwd[flattening].dropna()

            steep_ann = steep_ret.mean() * (ANN / HORIZONS[fh])
            flat_ann = flat_ret.mean() * (ANN / HORIZONS[fh])
            steep_hr = (steep_ret > 0).mean() * 100
            flat_hr = (flat_ret > 0).mean() * 100

            key = f"lb{lb}_fh{fh}"
            results[key] = {
                "steep_ann": steep_ann, "flat_ann": flat_ann,
                "steep_hr": steep_hr, "flat_hr": flat_hr,
                "steep_n": len(steep_ret), "flat_n": len(flat_ret),
            }

            x = [0, 1]
            vals = [steep_ann * 100, flat_ann * 100]
            colors = ["#3498db", "#e67e22"]
            ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Past Steep\n(n={len(steep_ret)})",
                                f"Past Flat\n(n={len(flat_ret)})"], fontsize=9)
            ax.set_ylabel("Ann. Fwd Return (%)")
            ax.set_title(f"Lookback={lb}d → Forward {fh}\n"
                         f"HR: {steep_hr:.1f}% / {flat_hr:.1f}%",
                         fontsize=10)
            ax.axhline(0, color="black", lw=0.8)
            ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Slope Momentum: Past Steepening/Flattening → Forward TLT Return",
                 y=1.01, fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "15_slope_momentum")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

def write_documentation(data, all_results):
    """Write slope_analysis_documentation.txt with all numerical results."""
    path = os.path.join(SAVE_DIR, "slope_analysis_documentation.txt")
    lines = []
    L = lines.append

    L("=" * 80)
    L("  YIELD CURVE SLOPE ANALYSIS - FULL DOCUMENTATION")
    L(f"  Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    L("=" * 80)

    dates = data["dates"]
    L(f"\n  Data range: {dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d} ({len(dates)} days)")

    # ── A) Scatterplot results ──
    L("\n" + "=" * 80)
    L("  SECTION A: SCATTERPLOT ANALYSIS")
    L("=" * 80)

    if "scatter_04" in all_results:
        L("\n  Spread Change vs TLT Daily Return (Regression):")
        L(f"  {'Spread':>8} {'Beta':>10} {'R2':>8} {'p-value':>12}")
        L("  " + "-" * 42)
        for sp, r in all_results["scatter_04"].items():
            L(f"  {sp:>8} {r['beta']:>10.6f} {r['r2']:>8.4f} {r['p']:>12.2e}")

    if "corr_03" in all_results:
        corr = all_results["corr_03"]
        L("\n  Spread Change Correlation Matrix:")
        L(f"\n{corr.round(3).to_string()}")

    # ── B) Regime Analysis ──
    L("\n" + "=" * 80)
    L("  SECTION B: REGIME ANALYSIS")
    L("=" * 80)

    if "regime_05" in all_results:
        L("\n  Annualized TLT Return by YC Regime (2s10s, w=21):")
        sdf = all_results["regime_05"]
        L(f"  {'Regime':>12} {'Ann.Ret%':>10} {'Vol%':>8} {'N':>6}")
        L("  " + "-" * 40)
        for _, r in sdf.iterrows():
            L(f"  {r['regime']:>12} {r['mean']*100:>+10.2f} {r['std']*100:>8.2f} {r['n']:>6.0f}")

    if "hit_rate_07" in all_results:
        L("\n  Hit Rate by Regime:")
        hr = all_results["hit_rate_07"]
        L(f"  {'Regime':>12} {'1d':>8} {'5d':>8} {'21d':>8}")
        L("  " + "-" * 40)
        for reg in ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]:
            if reg in hr:
                L(f"  {reg:>12} {hr[reg].get('1d', 0):>7.1f}% {hr[reg].get('5d', 0):>7.1f}% "
                  f"{hr[reg].get('21d', 0):>7.1f}%")

    if "transition_09" in all_results:
        t = all_results["transition_09"]
        L("\n  Regime Persistence (average duration):")
        for reg in ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]:
            L(f"    {reg}: {t['avg_dur'].get(reg, 0):.1f} days")

    # ── C) Rising/Falling ──
    L("\n" + "=" * 80)
    L("  SECTION C: RISING vs FALLING YIELDS")
    L("=" * 80)

    if "four_quad_11" in all_results:
        L("\n  Four-Quadrant Analysis (21d window):")
        fq = all_results["four_quad_11"]
        L(f"  {'Scenario':>18} {'Ann.Ret%':>10} {'Sharpe':>8} {'HitRate%':>10} {'N':>6}")
        L("  " + "-" * 56)
        for scen, r in fq.items():
            ann = r["ann_ret"] * 100 if not np.isnan(r["ann_ret"]) else 0
            sh = r["sharpe"] if not np.isnan(r["sharpe"]) else 0
            hr = r["hit_rate"] if not np.isnan(r["hit_rate"]) else 0
            L(f"  {scen:>18} {ann:>+10.2f} {sh:>8.3f} {hr:>9.1f}% {r['n']:>6}")

    # ── D) Periods ──
    L("\n" + "=" * 80)
    L("  SECTION D: HISTORICAL PERIOD ANALYSIS")
    L("=" * 80)

    if "period_13" in all_results:
        p13 = all_results["period_13"]
        if "avg_spread" in p13:
            L("\n  Average Spread Levels by Period (bps):")
            L(f"\n{p13['avg_spread'].round(1).to_string()}")
        if "corr" in p13:
            L("\n  Spread-TLT Correlation by Period:")
            L(f"\n{p13['corr'].round(3).to_string()}")

    # ── E) Rolling correlation ──
    L("\n" + "=" * 80)
    L("  SECTION E: ADDITIONAL ANALYTICS")
    L("=" * 80)

    if "rolling_14" in all_results:
        L("\n  Full-Sample Correlation: Spread Change vs TLT:")
        for sp, c in all_results["rolling_14"].items():
            L(f"    {sp:>7}: {c:>+.4f}")

    if "momentum_15" in all_results:
        L("\n  Slope Momentum Results:")
        mom = all_results["momentum_15"]
        L(f"  {'Config':>20} {'Steep Ann%':>12} {'Flat Ann%':>11} "
          f"{'Steep HR%':>10} {'Flat HR%':>10}")
        L("  " + "-" * 67)
        for key, r in mom.items():
            L(f"  {key:>20} {r['steep_ann']*100:>+12.2f} {r['flat_ann']*100:>+11.2f} "
              f"{r['steep_hr']:>9.1f}% {r['flat_hr']:>9.1f}%")

    # ── Key Findings ──
    L("\n" + "=" * 80)
    L("  KEY FINDINGS")
    L("=" * 80)

    # Auto-generate findings
    if "four_quad_11" in all_results:
        fq = all_results["four_quad_11"]
        best_scen = max(fq.items(), key=lambda x: x[1]["ann_ret"] if not np.isnan(x[1]["ann_ret"]) else -999)
        worst_scen = min(fq.items(), key=lambda x: x[1]["ann_ret"] if not np.isnan(x[1]["ann_ret"]) else 999)
        L(f"\n  1. Best scenario for TLT: {best_scen[0]} "
          f"(ann. ret = {best_scen[1]['ann_ret']*100:+.1f}%)")
        L(f"  2. Worst scenario for TLT: {worst_scen[0]} "
          f"(ann. ret = {worst_scen[1]['ann_ret']*100:+.1f}%)")

    if "regime_05" in all_results:
        sdf = all_results["regime_05"]
        best_reg = sdf.loc[sdf["mean"].idxmax()]
        L(f"  3. Best regime for TLT: {best_reg['regime']} "
          f"(ann. ret = {best_reg['mean']*100:+.1f}%)")

    if "scatter_04" in all_results:
        best_sp = max(all_results["scatter_04"].items(),
                      key=lambda x: abs(x[1]["r2"]))
        L(f"  4. Strongest spread-TLT relationship: {best_sp[0]} "
          f"(R2 = {best_sp[1]['r2']:.4f})")

    L("\n" + "=" * 80)
    L("  END OF DOCUMENTATION")
    L("=" * 80)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  YIELD CURVE SLOPE ANALYSIS")
    print("  Scatterplots, Regimes, Rising/Falling, Historical Periods")
    print("=" * 70)

    # ── Load data ──
    print("\n[1/7] Loading data...")
    D = load_all()

    # ── Prepare derived data ──
    print("\n[2/7] Preparing derived data...")
    data = prepare_data(D)
    print(f"  Period: {data['dates'][0]:%Y-%m-%d} -> {data['dates'][-1]:%Y-%m-%d}")
    print(f"  {len(data['dates'])} trading days")

    all_results = {}

    # ══════════════════════════════════════════════════════════════════
    # A) SCATTERPLOTS (01-04)
    # ══════════════════════════════════════════════════════════════════
    print("\n[3/7] Section A: Scatterplots...")

    print("  01: Slope vs Forward TLT")
    plot_01_slope_vs_fwd_tlt(data)

    print("  02: Level vs Spread scatter")
    plot_02_level_vs_spread_scatter(data)

    print("  03: Spread correlation")
    all_results["corr_03"] = plot_03_spread_correlation(data)

    print("  04: Spread vs TLT scatters")
    all_results["scatter_04"] = plot_04_spread_vs_tlt_scatters(data)

    # ══════════════════════════════════════════════════════════════════
    # B) REGIME ANALYSIS (05-09)
    # ══════════════════════════════════════════════════════════════════
    print("\n[4/7] Section B: Regime Analysis...")

    print("  05: TLT by regime bar")
    all_results["regime_05"] = plot_05_tlt_by_regime_bar(data)

    print("  06: Regime frequency")
    plot_06_regime_frequency(data)

    print("  07: Hit rate by regime")
    all_results["hit_rate_07"] = plot_07_hit_rate_by_regime(data)

    print("  08: TLT boxplot by regime")
    plot_08_tlt_boxplot_regime(data)

    print("  09: Transition persistence")
    all_results["transition_09"] = plot_09_transition_persistence(data)

    # ══════════════════════════════════════════════════════════════════
    # C) RISING vs FALLING (10-11)
    # ══════════════════════════════════════════════════════════════════
    print("\n[5/7] Section C: Rising vs Falling Yields...")

    print("  10: Rising/Falling performance")
    all_results["rising_10"] = plot_10_rising_falling_performance(data)

    print("  11: Four-quadrant analysis")
    all_results["four_quad_11"] = plot_11_four_quadrant(data)

    # ══════════════════════════════════════════════════════════════════
    # D) HISTORICAL PERIODS (12-13)
    # ══════════════════════════════════════════════════════════════════
    print("\n[6/7] Section D: Historical Periods...")

    print("  12: Period regime returns")
    all_results["period_12"] = plot_12_period_regime_returns(data)

    print("  13: Period spread analysis")
    all_results["period_13"] = plot_13_period_spread_analysis(data)

    # ══════════════════════════════════════════════════════════════════
    # E) ADDITIONAL ANALYTICS (14-15)
    # ══════════════════════════════════════════════════════════════════
    print("\n[7/7] Section E: Additional Analytics...")

    print("  14: Rolling correlation")
    all_results["rolling_14"] = plot_14_rolling_correlation(data)

    print("  15: Slope momentum")
    all_results["momentum_15"] = plot_15_slope_momentum(data)

    # ── Documentation ──
    print("\n  Writing documentation...")
    write_documentation(data, all_results)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  DONE - {elapsed:.1f}s")
    print(f"  Output: {SAVE_DIR}")
    print(f"  15 PNG + 1 TXT generated")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
