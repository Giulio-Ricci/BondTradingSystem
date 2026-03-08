"""
filter_optimization.py - Test all combinations of 4 additional filters
on top of the RR Duration Strategy.

Filters:
  F1: Yield Curve Slope  - block LONG when 10Y-2Y slope < threshold
  F2: ETF Momentum       - block LONG when TLT price < SMA(N)
  F3: Absolute MOVE Level - block LONG when MOVE > high, block SHORT when MOVE < low
  F4: Cross-Tenor Confirm - require N tenors to agree in same direction

Tests all 16 ON/OFF combinations (2^4), with parameter grid per filter.
Generates equity curves and comparison tables.
"""

import os
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE,
    ZSCORE_WINDOW, ZSCORE_MIN_PERIODS,
    MOVE_MEDIAN_WINDOW, MOVE_RATIO_CLIP,
)
from data_loader import load_all
from regime import fit_hmm_regime
from signals import compute_zscore_move_adj

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

FILT_DIR = os.path.join(SAVE_DIR, "filters")
os.makedirs(FILT_DIR, exist_ok=True)

# Base strategy config (baseline V1)
BASE_PARAMS = {
    "tl_calm": -3.5,
    "ts_calm": 3.0,
    "tl_stress": -4.0,
    "cooldown": 5,
    "delay": 1,
    "tcost_bps": 5,
    "w_init": 0.5,
}

# Also test the optimized config
OPT_PARAMS = {
    "tl_calm": -3.25,
    "ts_calm": 2.75,
    "tl_stress": -4.0,
    "cooldown": 5,
    "delay": 1,
    "tcost_bps": 5,
    "w_init": 0.5,
}

BASE_CONFIGS = {
    "baseline":  BASE_PARAMS,
    "optimized": OPT_PARAMS,
}

# -- Filter parameter grids --
# F1: Yield Curve Slope (10Y - 2Y)
#   Block LONG when slope < threshold
F1_SLOPE_THRESHOLDS = [0.0, -0.25, -0.50, -1.0]

# F2: ETF Momentum
#   Block LONG when TLT < SMA(window)
F2_SMA_WINDOWS = [50, 100, 200]

# F3: Absolute MOVE level
#   Block LONG when MOVE > move_high
#   Block SHORT when MOVE < move_low
F3_MOVE_BOUNDS = [
    (80, 120),
    (90, 130),
    (80, 140),
    (100, 150),
]

# F4: Cross-Tenor Confirmation
#   Require >= min_tenors of (1W, 1M, 3M, 6M) to agree
F4_MIN_TENORS = [2, 3]


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL & Z-SCORE
# ═══════════════════════════════════════════════════════════════════════════

def compute_zscore_std(rr, window=ZSCORE_WINDOW, min_per=ZSCORE_MIN_PERIODS):
    mu = rr.rolling(window, min_periods=min_per).mean()
    sig = rr.rolling(window, min_periods=min_per).std()
    return (rr - mu) / sig.replace(0, np.nan)


def compute_all_tenor_zscores(D):
    """Compute z-scores for all 4 tenors. Returns dict[tenor] -> z_series."""
    zscores = {}
    for tenor in ["1W", "1M", "3M", "6M"]:
        if tenor in D["rr"].columns:
            zscores[tenor] = compute_zscore_std(D["rr"][tenor])
    return zscores


# ═══════════════════════════════════════════════════════════════════════════
# FILTER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def make_slope_filter(yields_df, threshold):
    """F1: Returns Series of bool — True = LONG allowed."""
    slope = yields_df["US_10Y"] - yields_df["US_2Y"]
    return slope >= threshold


def make_momentum_filter(etf_px, window):
    """F2: Returns Series of bool — True = LONG allowed."""
    sma = etf_px["TLT"].rolling(window, min_periods=window).mean()
    return etf_px["TLT"] >= sma


def make_move_filter(move, low, high):
    """F3: Returns (long_ok, short_ok) Series of bool."""
    long_ok = move <= high      # block LONG when MOVE too high
    short_ok = move >= low      # block SHORT when MOVE too low
    return long_ok, short_ok


def make_cross_tenor_filter(tenor_zscores, min_tenors, tl, ts):
    """F4: Returns (long_ok, short_ok) based on cross-tenor confirmation.

    LONG ok if >= min_tenors have z >= ts
    SHORT ok if >= min_tenors have z <= tl
    """
    tenor_names = list(tenor_zscores.keys())
    idx = tenor_zscores[tenor_names[0]].index
    for t in tenor_names[1:]:
        idx = idx.intersection(tenor_zscores[t].index)

    n_long = pd.Series(0, index=idx)
    n_short = pd.Series(0, index=idx)

    for t in tenor_names:
        z = tenor_zscores[t].reindex(idx)
        n_long += (z >= ts).astype(int)
        n_short += (z <= tl).astype(int)

    long_ok = n_long >= min_tenors
    short_ok = n_short >= min_tenors
    return long_ok, short_ok


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST WITH FILTERS
# ═══════════════════════════════════════════════════════════════════════════

def filtered_backtest(z_arr, reg_arr, r_long, r_short, dates,
                      tl_c, ts_c, tl_s, cooldown, delay, tcost_bps, w_init,
                      f_long_ok=None, f_short_ok=None):
    """
    Fast backtest with filter arrays.
    f_long_ok[i] = True means LONG signal allowed at index i.
    f_short_ok[i] = True means SHORT signal allowed at index i.
    """
    n = len(z_arr)
    if n < 50:
        return None

    if f_long_ok is None:
        f_long_ok = np.ones(n, dtype=bool)
    if f_short_ok is None:
        f_short_ok = np.ones(n, dtype=bool)

    weight = np.empty(n)
    weight[0] = w_init
    events = []
    last_ev = -cooldown - 1

    for i in range(n):
        zv = z_arr[i]
        w_prev = weight[i - 1] if i > 0 else w_init

        if np.isnan(zv):
            weight[i] = w_prev
            continue

        reg = reg_arr[i]  # 0=CALM, 1=STRESS
        triggered = False

        if reg == 0:  # CALM
            if zv <= tl_c and (i - last_ev) >= cooldown and f_short_ok[i]:
                weight[i] = 0.0
                triggered = True
                events.append((i, "SHORT"))
            elif zv >= ts_c and (i - last_ev) >= cooldown and f_long_ok[i]:
                weight[i] = 1.0
                triggered = True
                events.append((i, "LONG"))
            else:
                weight[i] = w_prev
        else:  # STRESS
            if zv <= tl_s and (i - last_ev) >= cooldown and f_short_ok[i]:
                weight[i] = 0.0
                triggered = True
                events.append((i, "SHORT"))
            else:
                weight[i] = w_prev

        if triggered:
            last_ev = i

    # delay
    shift = 1 + delay
    w_del = np.empty(n)
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    # portfolio return
    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost_bps / 10_000)

    # equity
    equity = np.cumprod(1.0 + net)

    # metrics
    total = equity[-1] - 1.0
    years = n / 252.0
    ann_ret = (1.0 + total) ** (1.0 / max(years, 0.01)) - 1.0
    vol = np.std(net) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

    down = net[net < 0]
    down_vol = np.std(down) * np.sqrt(252) if len(down) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol

    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0

    # benchmark (B&H TLT)
    bm_eq = np.cumprod(1.0 + r_long)
    bm_total = bm_eq[-1] - 1.0
    bm_ann = (1.0 + bm_total) ** (1.0 / max(years, 0.01)) - 1.0
    excess = ann_ret - bm_ann

    # yearly breakdown
    yearly_excess = {}
    for i_ev, (idx, sig) in enumerate(events):
        if idx < len(dates):
            yr = dates[idx].year
            yearly_excess[yr] = yearly_excess.get(yr, 0)

    return {
        "equity": equity,
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_ret": ann_ret,
        "vol": vol,
        "mdd": mdd,
        "calmar": calmar,
        "excess": excess,
        "n_events": len(events),
        "equity_final": equity[-1],
        "dates": dates,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()

    print("=" * 70)
    print("  FILTER OPTIMIZATION — All 16 combinations x parameter grids")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    D = load_all()

    # ── Regime ─────────────────────────────────────────────────────────────
    print("\n[2/6] Fitting HMM regime ...")
    regime = fit_hmm_regime(D["move"], D["vix"])
    regime_num = (regime == "STRESS").astype(np.int8)

    # ── Pre-compute signals ────────────────────────────────────────────────
    print("\n[3/6] Computing signals & filter inputs ...")

    # z-score signals: both standard and MOVE-adjusted
    z_std_1m = compute_zscore_std(D["rr"]["1M"])
    z_move_1m = compute_zscore_move_adj(D["rr"]["1M"], D["move"])

    SIGNAL_VARIANTS = {
        "z_std":  z_std_1m,
        "z_move": z_move_1m,
    }
    print(f"       Signal variants: {list(SIGNAL_VARIANTS.keys())}")

    # All tenor z-scores for cross-tenor filter
    tenor_zscores = compute_all_tenor_zscores(D)
    print(f"       Tenor z-scores: {list(tenor_zscores.keys())}")

    # Yield slope
    slope = D["yields"]["US_10Y"] - D["yields"]["US_2Y"]
    print(f"       Yield slope: {slope.dropna().shape[0]} values, "
          f"current = {slope.dropna().iloc[-1]:.2f}")

    # MOVE
    move = D["move"]

    # ETF prices
    etf_px = D["etf_px"]
    etf_ret = D["etf_ret"]

    start = pd.Timestamp(BACKTEST_START)

    # ── Build common index ─────────────────────────────────────────────────
    common = (
        z_std_1m.dropna().index
        .intersection(z_move_1m.dropna().index)
        .intersection(regime.index)
        .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
        .intersection(slope.dropna().index)
        .intersection(move.dropna().index)
        .intersection(etf_px.dropna(subset=["TLT"]).index)
    )
    # Cross-tenor intersection
    for t, zs in tenor_zscores.items():
        common = common.intersection(zs.dropna().index)

    common = common[common >= start].sort_values()
    N = len(common)
    print(f"       Common trading days: {N}")

    # Align all arrays to common index
    z_arrays = {k: v.reindex(common).values for k, v in SIGNAL_VARIANTS.items()}
    reg_arr = regime_num.reindex(common).values
    rl = etf_ret["TLT"].reindex(common).fillna(0).values
    rs = etf_ret["SHV"].reindex(common).fillna(0).values
    dates = common.values

    slope_arr = slope.reindex(common).values
    move_arr = move.reindex(common).values
    tlt_px_arr = etf_px["TLT"].reindex(common).values

    # Tenor z-scores aligned
    tenor_z_aligned = {}
    for t, zs in tenor_zscores.items():
        tenor_z_aligned[t] = zs.reindex(common).values

    # ── Pre-compute filter masks ───────────────────────────────────────────
    print("\n[4/6] Pre-computing filter masks ...")

    # F1: Slope filters
    f1_masks = {}
    for thresh in F1_SLOPE_THRESHOLDS:
        f1_masks[thresh] = slope_arr >= thresh   # True = LONG ok
    print(f"       F1 (slope): {len(f1_masks)} variants")

    # F2: Momentum filters
    f2_masks = {}
    for win in F2_SMA_WINDOWS:
        sma = pd.Series(tlt_px_arr).rolling(win, min_periods=win).mean().values
        f2_masks[win] = tlt_px_arr >= sma       # True = LONG ok
        # Handle NaN at start
        f2_masks[win] = np.where(np.isnan(sma), True, f2_masks[win])
    print(f"       F2 (momentum): {len(f2_masks)} variants")

    # F3: MOVE level filters
    f3_masks = {}
    for low, high in F3_MOVE_BOUNDS:
        long_ok = move_arr <= high
        short_ok = move_arr >= low
        f3_masks[(low, high)] = (long_ok, short_ok)
    print(f"       F3 (MOVE level): {len(f3_masks)} variants")

    # F4: Cross-tenor filters
    f4_masks = {}
    for min_t in F4_MIN_TENORS:
        n_long = np.zeros(N)
        n_short = np.zeros(N)
        for t in tenor_z_aligned:
            z_t = tenor_z_aligned[t]
            # Use a moderate threshold for tenor confirmation
            # We'll test with the base strategy thresholds
            n_long += (z_t >= 2.0).astype(int)    # any tenor above 2.0
            n_short += (z_t <= -2.5).astype(int)   # any tenor below -2.5
        long_ok = n_long >= min_t
        short_ok = n_short >= min_t
        f4_masks[min_t] = (long_ok, short_ok)
    print(f"       F4 (cross-tenor): {len(f4_masks)} variants")

    # ── Run all filter combinations ────────────────────────────────────────
    print("\n[5/6] Running all filter combinations ...")

    # Define filter ON/OFF combinations (2^4 = 16)
    filter_names = ["F1_slope", "F2_momentum", "F3_move", "F4_xtenor"]
    all_combos = list(itertools.product([False, True], repeat=4))

    all_results = []
    equity_curves = {}  # store for plotting

    total_runs = 0

    for combo in all_combos:
        f1_on, f2_on, f3_on, f4_on = combo
        combo_name = ""
        if not any(combo):
            combo_name = "NO_FILTER"
        else:
            parts = []
            if f1_on: parts.append("F1")
            if f2_on: parts.append("F2")
            if f3_on: parts.append("F3")
            if f4_on: parts.append("F4")
            combo_name = "+".join(parts)

        # Determine parameter grid for active filters
        f1_grid = F1_SLOPE_THRESHOLDS if f1_on else [None]
        f2_grid = F2_SMA_WINDOWS if f2_on else [None]
        f3_grid = F3_MOVE_BOUNDS if f3_on else [None]
        f4_grid = F4_MIN_TENORS if f4_on else [None]

        combo_count = 0

        for f1_p in f1_grid:
            for f2_p in f2_grid:
                for f3_p in f3_grid:
                    for f4_p in f4_grid:
                        for base_name, base_p in BASE_CONFIGS.items():
                            for sig_name, z_arr in z_arrays.items():

                                # Build composite filter arrays
                                long_ok = np.ones(N, dtype=bool)
                                short_ok = np.ones(N, dtype=bool)

                                if f1_on and f1_p is not None:
                                    long_ok &= f1_masks[f1_p]

                                if f2_on and f2_p is not None:
                                    long_ok &= f2_masks[f2_p]

                                if f3_on and f3_p is not None:
                                    lo, so = f3_masks[f3_p]
                                    long_ok &= lo
                                    short_ok &= so

                                if f4_on and f4_p is not None:
                                    lo, so = f4_masks[f4_p]
                                    long_ok &= lo
                                    short_ok &= so

                                # Run backtest
                                res = filtered_backtest(
                                    z_arr, reg_arr, rl, rs, common,
                                    tl_c=base_p["tl_calm"],
                                    ts_c=base_p["ts_calm"],
                                    tl_s=base_p["tl_stress"],
                                    cooldown=base_p["cooldown"],
                                    delay=base_p["delay"],
                                    tcost_bps=base_p["tcost_bps"],
                                    w_init=base_p["w_init"],
                                    f_long_ok=long_ok,
                                    f_short_ok=short_ok,
                                )

                                if res is None:
                                    continue

                                row = {
                                    "combo": combo_name,
                                    "signal": sig_name,
                                    "base": base_name,
                                    "f1_slope": f1_p,
                                    "f2_sma": f2_p,
                                    "f3_move": str(f3_p) if f3_p else None,
                                    "f4_xtenor": f4_p,
                                    "sharpe": res["sharpe"],
                                    "sortino": res["sortino"],
                                    "ann_ret": res["ann_ret"],
                                    "vol": res["vol"],
                                    "mdd": res["mdd"],
                                    "calmar": res["calmar"],
                                    "excess": res["excess"],
                                    "n_events": res["n_events"],
                                    "eq_final": res["equity_final"],
                                }
                                all_results.append(row)

                                # Store equity for plotting
                                combo_full = (f"{combo_name}|{sig_name}|"
                                              f"{base_name}|"
                                              f"f1={f1_p}|f2={f2_p}|"
                                              f"f3={f3_p}|f4={f4_p}")
                                equity_curves[combo_full] = (
                                    common, res["equity"], res["sharpe"]
                                )

                                combo_count += 1
                                total_runs += 1

        print(f"    {combo_name:25s}  {combo_count:>5} runs")

    df = pd.DataFrame(all_results)
    df.sort_values("sharpe", ascending=False, inplace=True)

    print(f"\n    Total runs: {total_runs:,}")

    # ── Analysis ───────────────────────────────────────────────────────────
    print(f"\n[6/6] Analysis & Visualization ...")

    # Best per filter combination
    print(f"\n  {'='*110}")
    print(f"  BEST CONFIG PER FILTER COMBINATION")
    print(f"  {'='*110}")
    print(f"  {'Combo':<25s} {'Sig':<7s} {'Base':<10s} {'F1':>5s} {'F2':>5s} "
          f"{'F3':>12s} {'F4':>4s}  "
          f"{'Sharpe':>7s} {'AnnRet':>7s} {'Vol':>6s} {'MDD':>7s} "
          f"{'Excess':>7s} {'Ev':>4s} {'EqFin':>7s}")
    print(f"  {'-'*120}")

    best_per_combo = {}
    for combo_name in df["combo"].unique():
        sub = df[df["combo"] == combo_name]
        best = sub.iloc[0]
        best_per_combo[combo_name] = best
        print(f"  {best['combo']:<25s} {best['signal']:<7s} {best['base']:<10s} "
              f"{str(best['f1_slope']):>5s} {str(best['f2_sma']):>5s} "
              f"{str(best['f3_move']):>12s} {str(best['f4_xtenor']):>4s}  "
              f"{best['sharpe']:+7.3f} {best['ann_ret']:+7.4f} "
              f"{best['vol']:6.4f} {best['mdd']:+7.4f} "
              f"{best['excess']:+7.4f} {best['n_events']:4.0f} "
              f"{best['eq_final']:7.3f}")

    print(f"  {'='*110}")

    # Ranking of filter combos
    combo_ranking = (df.groupby("combo")["sharpe"]
                     .max().sort_values(ascending=False))
    print(f"\n  FILTER COMBO RANKING (by best Sharpe):")
    for rank, (combo, sh) in enumerate(combo_ranking.items(), 1):
        marker = " <-- BASELINE" if combo == "NO_FILTER" else ""
        print(f"    #{rank:2d}  {combo:<25s}  Sharpe={sh:+.4f}{marker}")

    # Save CSV
    csv_path = os.path.join(FILT_DIR, "filter_optimization_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Results saved -> {csv_path}")

    # ── Visualizations ─────────────────────────────────────────────────────

    # --- Plot 1: Bar chart - Best Sharpe per filter combo ---
    fig, ax = plt.subplots(figsize=(14, 7))
    combos_sorted = combo_ranking.index.tolist()
    sharpes = [combo_ranking[c] for c in combos_sorted]
    colors = ["gold" if c == "NO_FILTER" else "steelblue" for c in combos_sorted]
    bars = ax.barh(range(len(combos_sorted)), sharpes, color=colors,
                   edgecolor="white")
    ax.set_yticks(range(len(combos_sorted)))
    ax.set_yticklabels(combos_sorted, fontsize=8)
    ax.axvline(0.218, color="green", ls="--", lw=1.5, label="Baseline V1 (0.218)")
    ax.set_xlabel("Best Sharpe Ratio")
    ax.set_title("Best Sharpe by Filter Combination")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "01_sharpe_by_combo.png"), dpi=150)
    plt.close()

    # --- Plot 2: Top 15 equity curves ---
    top15_keys = df.head(15).apply(
        lambda r: (f"{r['combo']}|{r['signal']}|{r['base']}|"
                   f"f1={r['f1_slope']}|f2={r['f2_sma']}|"
                   f"f3={r['f3_move']}|f4={r['f4_xtenor']}"),
        axis=1,
    ).tolist()

    fig, ax = plt.subplots(figsize=(16, 7))
    plotted = set()
    for rank, key in enumerate(top15_keys):
        if key in equity_curves and key not in plotted:
            dates_eq, eq, sh = equity_curves[key]
            # short label
            parts = key.split("|")
            label = f"#{rank+1} Sh={sh:.3f} {parts[0]}|{parts[1]}"
            ax.plot(dates_eq, eq, linewidth=1.0, alpha=0.8, label=label)
            plotted.add(key)
        if len(plotted) >= 12:
            break

    # No-filter baseline (z_move)
    nf_key = None
    for k in equity_curves:
        if "NO_FILTER" in k and "z_move" in k and "baseline" in k:
            nf_key = k
            break
    if nf_key and nf_key not in plotted:
        d, eq, sh = equity_curves[nf_key]
        ax.plot(d, eq, color="gold", linewidth=2, ls="-",
                label=f"NO_FILTER|baseline Sh={sh:.3f}")

    # B&H TLT
    bm = np.cumprod(1.0 + rl)
    ax.plot(common, bm, color="grey", ls="--", linewidth=1.5,
            alpha=0.5, label="B&H TLT")

    ax.set_title("Top Filtered Configurations — Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "02_top15_equity_curves.png"), dpi=150)
    plt.close()

    # --- Plot 3: Best equity per filter combo (one curve each) ---
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = plt.cm.tab20(np.linspace(0, 1, len(combos_sorted)))

    for ci, combo_name in enumerate(combos_sorted):
        best = best_per_combo[combo_name]
        key = (f"{best['combo']}|{best['signal']}|{best['base']}|"
               f"f1={best['f1_slope']}|f2={best['f2_sma']}|"
               f"f3={best['f3_move']}|f4={best['f4_xtenor']}")
        if key in equity_curves:
            d, eq, sh = equity_curves[key]
            lw = 2.5 if combo_name == "NO_FILTER" else 1.2
            ls = "--" if combo_name == "NO_FILTER" else "-"
            ax.plot(d, eq, color=cmap[ci], linewidth=lw, ls=ls,
                    alpha=0.8,
                    label=f"{combo_name} ({sh:+.3f})")

    ax.plot(common, bm, color="grey", ls=":", linewidth=1.5,
            alpha=0.4, label="B&H TLT")
    ax.set_title("Best Equity Curve per Filter Combination (all 16)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=6, loc="upper left", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "03_all_combos_equity.png"), dpi=150)
    plt.close()

    # --- Plot 4: Filter contribution analysis ---
    # For each filter, compare ON vs OFF (marginal effect)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax_i, (fname, fidx) in enumerate(zip(
        filter_names, [0, 1, 2, 3]
    )):
        ax = axes.flat[ax_i]

        on_sharpes = []
        off_sharpes = []
        for combo in all_combos:
            combo_name_local = ""
            if not any(combo):
                combo_name_local = "NO_FILTER"
            else:
                parts_local = []
                if combo[0]: parts_local.append("F1")
                if combo[1]: parts_local.append("F2")
                if combo[2]: parts_local.append("F3")
                if combo[3]: parts_local.append("F4")
                combo_name_local = "+".join(parts_local)

            sub = df[df["combo"] == combo_name_local]
            if len(sub) == 0:
                continue

            best_sh = sub["sharpe"].max()
            if combo[fidx]:
                on_sharpes.append(best_sh)
            else:
                off_sharpes.append(best_sh)

        ax.boxplot([off_sharpes, on_sharpes],
                   tick_labels=["OFF", "ON"],
                   patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
        ax.set_title(fname)
        ax.set_ylabel("Best Sharpe")
        ax.grid(True, alpha=0.3)

        # Add mean lines
        if off_sharpes:
            ax.axhline(np.mean(off_sharpes), color="red", ls="--",
                       alpha=0.5, label=f"OFF mean={np.mean(off_sharpes):.3f}")
        if on_sharpes:
            ax.axhline(np.mean(on_sharpes), color="blue", ls="--",
                       alpha=0.5, label=f"ON mean={np.mean(on_sharpes):.3f}")
        ax.legend(fontsize=7)

    fig.suptitle("Filter Marginal Effect — ON vs OFF", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "04_filter_marginal_effect.png"), dpi=150)
    plt.close()

    # --- Plot 5: Heatmap - pairwise filter interactions ---
    pair_names = []
    pair_sharpes = []
    for i in range(4):
        for j in range(i + 1, 4):
            fname_i = filter_names[i]
            fname_j = filter_names[j]
            pair = f"{fname_i}+{fname_j}"

            # Find combos where both i and j are ON (and possibly others)
            both_on = []
            only_i = []
            only_j = []
            neither = []

            for combo in all_combos:
                combo_name_local = ""
                if not any(combo):
                    combo_name_local = "NO_FILTER"
                else:
                    parts_local = []
                    if combo[0]: parts_local.append("F1")
                    if combo[1]: parts_local.append("F2")
                    if combo[2]: parts_local.append("F3")
                    if combo[3]: parts_local.append("F4")
                    combo_name_local = "+".join(parts_local)

                sub = df[df["combo"] == combo_name_local]
                if len(sub) == 0:
                    continue
                best_sh = sub["sharpe"].max()

                if combo[i] and combo[j]:
                    both_on.append(best_sh)
                elif combo[i] and not combo[j]:
                    only_i.append(best_sh)
                elif not combo[i] and combo[j]:
                    only_j.append(best_sh)
                else:
                    neither.append(best_sh)

            pair_names.append(pair)
            pair_sharpes.append({
                "both": np.mean(both_on) if both_on else 0,
                "only_i": np.mean(only_i) if only_i else 0,
                "only_j": np.mean(only_j) if only_j else 0,
                "neither": np.mean(neither) if neither else 0,
            })

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pair_names))
    w = 0.2
    ax.bar(x - 1.5*w, [p["neither"] for p in pair_sharpes], w,
           label="Neither", color="lightgrey")
    ax.bar(x - 0.5*w, [p["only_i"] for p in pair_sharpes], w,
           label="Only first", color="steelblue")
    ax.bar(x + 0.5*w, [p["only_j"] for p in pair_sharpes], w,
           label="Only second", color="coral")
    ax.bar(x + 1.5*w, [p["both"] for p in pair_sharpes], w,
           label="Both ON", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(pair_names, rotation=30, fontsize=8)
    ax.set_ylabel("Mean best Sharpe")
    ax.set_title("Pairwise Filter Interaction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "05_pairwise_interaction.png"), dpi=150)
    plt.close()

    # --- Plot 6: Sharpe vs Events scatter (filtered vs unfiltered) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    nf = df[df["combo"] == "NO_FILTER"]
    filt = df[df["combo"] != "NO_FILTER"]
    ax.scatter(nf["n_events"], nf["sharpe"], c="gold", s=30,
               alpha=0.8, label="No filter", edgecolors="black", linewidth=0.3)
    sc = ax.scatter(filt["n_events"], filt["sharpe"],
                    c=filt["mdd"], cmap="RdYlGn_r",
                    s=12, alpha=0.4, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Max Drawdown")
    ax.set_xlabel("Number of Events")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Filtered vs Unfiltered: Events vs Sharpe")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "06_events_vs_sharpe_filtered.png"),
                dpi=150)
    plt.close()

    # --- Plot 7: Parameter sensitivity per filter ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # F1: slope threshold
    ax = axes[0, 0]
    for thresh in F1_SLOPE_THRESHOLDS:
        sub = df[(df["f1_slope"] == thresh)]
        if len(sub) > 0:
            ax.scatter([thresh] * len(sub), sub["sharpe"], alpha=0.3, s=10)
    # Best per threshold
    f1_best = []
    for thresh in F1_SLOPE_THRESHOLDS:
        sub = df[df["f1_slope"] == thresh]
        if len(sub) > 0:
            f1_best.append((thresh, sub["sharpe"].max()))
    if f1_best:
        ax.plot([x[0] for x in f1_best], [x[1] for x in f1_best],
                "ro-", linewidth=2, markersize=8, label="Best")
    ax.set_xlabel("Slope threshold")
    ax.set_ylabel("Sharpe")
    ax.set_title("F1: Yield Curve Slope")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F2: SMA window
    ax = axes[0, 1]
    f2_best = []
    for win in F2_SMA_WINDOWS:
        sub = df[df["f2_sma"] == win]
        if len(sub) > 0:
            ax.scatter([win] * len(sub), sub["sharpe"], alpha=0.3, s=10)
            f2_best.append((win, sub["sharpe"].max()))
    if f2_best:
        ax.plot([x[0] for x in f2_best], [x[1] for x in f2_best],
                "ro-", linewidth=2, markersize=8, label="Best")
    ax.set_xlabel("SMA window")
    ax.set_ylabel("Sharpe")
    ax.set_title("F2: ETF Momentum")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F3: MOVE bounds
    ax = axes[1, 0]
    f3_best = []
    for low, high in F3_MOVE_BOUNDS:
        label = f"({low},{high})"
        sub = df[df["f3_move"] == str((low, high))]
        if len(sub) > 0:
            x_pos = low + high
            ax.scatter([x_pos] * len(sub), sub["sharpe"], alpha=0.3, s=10)
            f3_best.append((label, x_pos, sub["sharpe"].max()))
    if f3_best:
        ax.plot([x[1] for x in f3_best], [x[2] for x in f3_best],
                "ro-", linewidth=2, markersize=8, label="Best")
        ax.set_xticks([x[1] for x in f3_best])
        ax.set_xticklabels([x[0] for x in f3_best], fontsize=8)
    ax.set_xlabel("MOVE bounds (low, high)")
    ax.set_ylabel("Sharpe")
    ax.set_title("F3: Absolute MOVE Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F4: Min tenors
    ax = axes[1, 1]
    f4_best = []
    for mt in F4_MIN_TENORS:
        sub = df[df["f4_xtenor"] == mt]
        if len(sub) > 0:
            ax.scatter([mt] * len(sub), sub["sharpe"], alpha=0.3, s=10)
            f4_best.append((mt, sub["sharpe"].max()))
    if f4_best:
        ax.plot([x[0] for x in f4_best], [x[1] for x in f4_best],
                "ro-", linewidth=2, markersize=8, label="Best")
    ax.set_xlabel("Min tenors agreeing")
    ax.set_ylabel("Sharpe")
    ax.set_title("F4: Cross-Tenor Confirmation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Filter Parameter Sensitivity", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "07_filter_param_sensitivity.png"),
                dpi=150)
    plt.close()

    # --- Plot 8: Summary table as figure ---
    top20 = df.head(20)
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis("off")

    col_labels = ["Rank", "Filters", "Signal", "Base", "F1", "F2", "F3", "F4",
                  "Sharpe", "AnnRet", "MDD", "Events"]
    table_data = []
    for rank, (_, r) in enumerate(top20.iterrows(), 1):
        table_data.append([
            f"#{rank}",
            r["combo"],
            r["signal"],
            r["base"],
            str(r["f1_slope"]) if r["f1_slope"] is not None else "-",
            str(r["f2_sma"]) if r["f2_sma"] is not None else "-",
            str(r["f3_move"]) if r["f3_move"] is not None else "-",
            str(r["f4_xtenor"]) if r["f4_xtenor"] is not None else "-",
            f"{r['sharpe']:+.3f}",
            f"{r['ann_ret']:+.4f}",
            f"{r['mdd']:+.4f}",
            f"{r['n_events']:.0f}",
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(list(range(len(col_labels))))
    table.scale(1, 1.4)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("steelblue")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Top 20 Filtered Configurations", fontsize=14, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FILT_DIR, "08_top20_table.png"), dpi=150)
    plt.close()

    n_plots = 8
    print(f"\n  {n_plots} plots saved to {FILT_DIR}/")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - T0
    best_row = df.iloc[0]
    nf_best = df[df["combo"] == "NO_FILTER"]["sharpe"].max()

    print(f"\n{'='*70}")
    print(f"  FILTER OPTIMIZATION COMPLETE")
    print(f"  Total configurations: {len(df):,}")
    print(f"  No-filter best Sharpe: {nf_best:+.4f}")
    print(f"  Best filtered Sharpe:  {best_row['sharpe']:+.4f}  "
          f"({best_row['combo']})")
    print(f"  Improvement: {best_row['sharpe'] - nf_best:+.4f}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
