"""
composite_signal.py - Composite Signal: RR z-score + Yield Curve components.

Instead of using yield curve as a filter (ON/OFF), we combine it directly
into the z-score signal. The yield curve slope/curvature becomes part of
the signal strength, not a gate.

Yield curve components tested:
  S1: 10Y - 2Y   (classic slope)
  S2: 30Y - 10Y   (long end steepness)
  S3: 10Y - 5Y    (belly-to-long)
  S4: 5Y - 2Y     (belly slope)
  S5: 30Y - 2Y    (full curve slope)
  S6: 10Y - 3M    (term premium proxy)
  S7: (2Y+10Y)/2 - 5Y  (butterfly / curvature)

Combination methods:
  A) z_composite = w_rr * z_rr + w_slope * z_slope   (linear blend)
  B) z_composite = z_rr * (1 + k * z_slope)           (multiplicative)
  C) z_composite = z_rr + k * delta_slope              (slope momentum)

Tests all yield components x combination methods x weight grids,
on top of both z_std and z_move base signals.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE,
    ZSCORE_WINDOW, ZSCORE_MIN_PERIODS, MOVE_MEDIAN_WINDOW, MOVE_RATIO_CLIP,
)
from data_loader import load_all
from regime import fit_hmm_regime

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

COMP_DIR = os.path.join(SAVE_DIR, "composite")
os.makedirs(COMP_DIR, exist_ok=True)

# Base strategy thresholds
THRESHOLD_SETS = {
    "baseline":  (-3.5, 3.0, -4.0),
    "optimized": (-3.25, 2.75, -4.0),
}

COOLDOWN = 5
DELAY = 1
TCOST_BPS = 5
W_INIT = 0.5

# Yield curve components: name -> (col_long, col_short)
YIELD_COMPONENTS = {
    "10Y-2Y":   ("US_10Y", "US_2Y"),
    "30Y-10Y":  ("US_30Y", "US_10Y"),
    "10Y-5Y":   ("US_10Y", "US_5Y"),
    "5Y-2Y":    ("US_5Y",  "US_2Y"),
    "30Y-2Y":   ("US_30Y", "US_2Y"),
    "10Y-3M":   ("US_10Y", "US_3M"),
}

# Butterfly: (2Y + 10Y)/2 - 5Y  (computed separately)
YIELD_BUTTERFLY = "butterfly"

# Combination method weights
# Method A: z = w_rr * z_rr + (1-w_rr) * z_slope
BLEND_WEIGHTS_RR = [0.5, 0.6, 0.7, 0.8, 0.9]

# Method B: z = z_rr * (1 + k * z_slope_norm)
MULT_K = [0.1, 0.2, 0.3, 0.5, 0.7]

# Method C: z = z_rr + k * delta_slope_z
MOMENTUM_K = [0.2, 0.3, 0.5, 0.7, 1.0]
SLOPE_MOMENTUM_WINDOWS = [21, 63, 126]  # days for slope change


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def rolling_zscore(s, window=ZSCORE_WINDOW, min_per=ZSCORE_MIN_PERIODS):
    """Standard rolling z-score."""
    mu = s.rolling(window, min_periods=min_per).mean()
    sig = s.rolling(window, min_periods=min_per).std()
    return (s - mu) / sig.replace(0, np.nan)


def compute_rr_zscore_std(rr_1m, window=ZSCORE_WINDOW):
    return rolling_zscore(rr_1m, window)


def compute_rr_zscore_move(rr_1m, move, window=ZSCORE_WINDOW):
    min_per = ZSCORE_MIN_PERIODS
    mu = rr_1m.rolling(window, min_periods=min_per).mean()
    sig = rr_1m.rolling(window, min_periods=min_per).std()
    move_med = move.rolling(MOVE_MEDIAN_WINDOW, min_periods=min_per).median()
    move_ratio = (move / move_med).clip(*MOVE_RATIO_CLIP)
    sig_adj = sig * np.sqrt(move_ratio)
    return (rr_1m - mu) / sig_adj.replace(0, np.nan)


def compute_yield_components(yields_df):
    """Compute all yield curve components and their z-scores."""
    components = {}
    z_components = {}

    for name, (col_long, col_short) in YIELD_COMPONENTS.items():
        raw = yields_df[col_long] - yields_df[col_short]
        components[name] = raw
        z_components[name] = rolling_zscore(raw)

    # Butterfly
    raw_bf = (yields_df["US_2Y"] + yields_df["US_10Y"]) / 2 - yields_df["US_5Y"]
    components[YIELD_BUTTERFLY] = raw_bf
    z_components[YIELD_BUTTERFLY] = rolling_zscore(raw_bf)

    return components, z_components


def compute_slope_momentum(slope_raw, windows):
    """Compute slope change (delta) over different windows, z-scored."""
    momentum = {}
    for win in windows:
        delta = slope_raw.diff(win)
        momentum[win] = rolling_zscore(delta)
    return momentum


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE SIGNAL BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def method_A_blend(z_rr, z_slope, w_rr):
    """Linear blend: z = w_rr * z_rr + (1 - w_rr) * z_slope"""
    return w_rr * z_rr + (1 - w_rr) * z_slope


def method_B_mult(z_rr, z_slope, k):
    """Multiplicative: z = z_rr * (1 + k * z_slope)
    When slope z is positive (steepening), amplifies the RR signal.
    When slope z is negative (flattening/inversion), dampens it.
    """
    return z_rr * (1 + k * z_slope)


def method_C_momentum(z_rr, z_slope_mom, k):
    """Slope momentum: z = z_rr + k * z_delta_slope
    Adds yield curve momentum to the RR signal.
    """
    return z_rr + k * z_slope_mom


# ═══════════════════════════════════════════════════════════════════════════
# FAST BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def fast_bt(z_arr, reg_arr, r_long, r_short,
            tl_c, ts_c, tl_s, cooldown=COOLDOWN,
            delay=DELAY, tcost_bps=TCOST_BPS, w_init=W_INIT):
    """Fast backtest returning metrics + equity array."""
    n = len(z_arr)
    if n < 50:
        return None

    weight = np.empty(n)
    weight[0] = w_init
    n_events = 0
    last_ev = -cooldown - 1

    for i in range(n):
        zv = z_arr[i]
        w_prev = weight[i - 1] if i > 0 else w_init
        if np.isnan(zv):
            weight[i] = w_prev
            continue
        reg = reg_arr[i]
        triggered = False
        if reg == 0:  # CALM
            if zv <= tl_c and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True
            elif zv >= ts_c and (i - last_ev) >= cooldown:
                weight[i] = 1.0; triggered = True
            else:
                weight[i] = w_prev
        else:  # STRESS
            if zv <= tl_s and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True
            else:
                weight[i] = w_prev
        if triggered:
            last_ev = i
            n_events += 1

    shift = 1 + delay
    w_del = np.empty(n)
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost_bps / 10_000)

    equity = np.cumprod(1.0 + net)
    total = equity[-1] - 1.0
    years = n / 252.0
    ann_ret = (1.0 + total) ** (1.0 / max(years, 0.01)) - 1.0
    vol = np.std(net) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

    down = net[net < 0]
    down_vol = np.std(down) * np.sqrt(252) if len(down) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol

    running_max = np.maximum.accumulate(equity)
    mdd = (equity / running_max - 1.0).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0

    bm_eq = np.cumprod(1.0 + r_long)
    bm_ann = (1.0 + bm_eq[-1] - 1.0) ** (1.0 / max(years, 0.01)) - 1.0
    excess = ann_ret - bm_ann

    return {
        "sharpe": sharpe, "sortino": sortino,
        "ann_ret": ann_ret, "vol": vol,
        "mdd": mdd, "calmar": calmar,
        "excess": excess, "n_events": n_events,
        "equity": equity, "eq_final": equity[-1],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()

    print("=" * 70)
    print("  COMPOSITE SIGNAL — RR + Yield Curve Components")
    print("=" * 70)

    # ── Load ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    D = load_all()

    # ── Regime ─────────────────────────────────────────────────────────────
    print("\n[2/7] Fitting HMM regime ...")
    regime = fit_hmm_regime(D["move"], D["vix"])
    regime_num = (regime == "STRESS").astype(np.int8)

    etf_ret = D["etf_ret"]
    start = pd.Timestamp(BACKTEST_START)

    # ── Compute base RR signals ────────────────────────────────────────────
    print("\n[3/7] Computing base RR z-scores ...")
    rr_1m = D["rr"]["1M"]
    z_rr_std = compute_rr_zscore_std(rr_1m)
    z_rr_move = compute_rr_zscore_move(rr_1m, D["move"])
    base_signals = {"z_std": z_rr_std, "z_move": z_rr_move}

    # ── Compute yield curve components ─────────────────────────────────────
    print("\n[4/7] Computing yield curve components ...")
    yc_raw, yc_z = compute_yield_components(D["yields"])

    # Slope momentum for each component
    yc_momentum = {}
    for name, raw in yc_raw.items():
        yc_momentum[name] = compute_slope_momentum(raw, SLOPE_MOMENTUM_WINDOWS)

    all_yc_names = list(yc_z.keys())
    print(f"       Components: {all_yc_names}")
    print(f"       Momentum windows: {SLOPE_MOMENTUM_WINDOWS}")

    # Also test multi-component blends
    MULTI_BLENDS = {
        "10Y2Y+30Y10Y":   ["10Y-2Y", "30Y-10Y"],
        "10Y2Y+5Y2Y":     ["10Y-2Y", "5Y-2Y"],
        "10Y2Y+10Y3M":    ["10Y-2Y", "10Y-3M"],
        "5Y2Y+30Y10Y":    ["5Y-2Y",  "30Y-10Y"],
        "full_3seg":       ["5Y-2Y",  "10Y-5Y", "30Y-10Y"],
    }

    # Pre-compute multi-blend z-scores (average of components)
    for blend_name, comp_list in MULTI_BLENDS.items():
        z_parts = [yc_z[c] for c in comp_list]
        yc_z[blend_name] = pd.concat(z_parts, axis=1).mean(axis=1)
        # Raw for momentum
        raw_parts = [yc_raw[c] for c in comp_list]
        yc_raw[blend_name] = pd.concat(raw_parts, axis=1).mean(axis=1)
        yc_momentum[blend_name] = compute_slope_momentum(
            yc_raw[blend_name], SLOPE_MOMENTUM_WINDOWS
        )

    all_yc_names = list(yc_z.keys())
    print(f"       Total (with blends): {len(all_yc_names)} components")

    # ── Build common index ─────────────────────────────────────────────────
    common = (
        z_rr_std.dropna().index
        .intersection(z_rr_move.dropna().index)
        .intersection(regime.index)
        .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
    )
    for name in all_yc_names:
        common = common.intersection(yc_z[name].dropna().index)

    common = common[common >= start].sort_values()
    N = len(common)
    print(f"\n       Common trading days: {N}")

    # Align arrays
    reg_arr = regime_num.reindex(common).values
    rl = etf_ret["TLT"].reindex(common).fillna(0).values
    rs = etf_ret["SHV"].reindex(common).fillna(0).values

    base_z_aligned = {k: v.reindex(common).values for k, v in base_signals.items()}
    yc_z_aligned = {k: v.reindex(common).values for k, v in yc_z.items()}
    yc_mom_aligned = {}
    for name in all_yc_names:
        yc_mom_aligned[name] = {}
        for win in SLOPE_MOMENTUM_WINDOWS:
            if name in yc_momentum and win in yc_momentum[name]:
                yc_mom_aligned[name][win] = (
                    yc_momentum[name][win].reindex(common).fillna(0).values
                )

    # ── Run all combinations ───────────────────────────────────────────────
    print("\n[5/7] Running composite signal optimization ...")

    all_results = []
    equity_store = {}
    total_runs = 0

    # First: baseline (no composite) for reference
    for base_name, z_base in base_z_aligned.items():
        for thresh_name, (tl_c, ts_c, tl_s) in THRESHOLD_SETS.items():
            res = fast_bt(z_base, reg_arr, rl, rs, tl_c, ts_c, tl_s)
            if res:
                key = f"PURE_RR|{base_name}|{thresh_name}"
                all_results.append({
                    "method": "PURE_RR",
                    "base_signal": base_name,
                    "yc_component": "-",
                    "param": "-",
                    "thresholds": thresh_name,
                    **{k: v for k, v in res.items() if k != "equity"},
                })
                equity_store[key] = (common, res["equity"], res["sharpe"])
                total_runs += 1

    # Method A: Linear blend
    print("    Method A (linear blend) ...")
    for base_name, z_base in base_z_aligned.items():
        for yc_name in all_yc_names:
            z_yc = yc_z_aligned[yc_name]
            for w_rr in BLEND_WEIGHTS_RR:
                z_comp = w_rr * z_base + (1 - w_rr) * z_yc
                for thresh_name, (tl_c, ts_c, tl_s) in THRESHOLD_SETS.items():
                    res = fast_bt(z_comp, reg_arr, rl, rs, tl_c, ts_c, tl_s)
                    if res:
                        key = f"A|{base_name}|{yc_name}|w={w_rr}|{thresh_name}"
                        all_results.append({
                            "method": "A_blend",
                            "base_signal": base_name,
                            "yc_component": yc_name,
                            "param": f"w_rr={w_rr}",
                            "thresholds": thresh_name,
                            **{k: v for k, v in res.items() if k != "equity"},
                        })
                        equity_store[key] = (common, res["equity"], res["sharpe"])
                        total_runs += 1

    print(f"      {total_runs} runs so far")

    # Method B: Multiplicative
    print("    Method B (multiplicative) ...")
    for base_name, z_base in base_z_aligned.items():
        for yc_name in all_yc_names:
            z_yc = yc_z_aligned[yc_name]
            for k in MULT_K:
                z_comp = z_base * (1 + k * np.nan_to_num(z_yc, 0))
                for thresh_name, (tl_c, ts_c, tl_s) in THRESHOLD_SETS.items():
                    res = fast_bt(z_comp, reg_arr, rl, rs, tl_c, ts_c, tl_s)
                    if res:
                        key = f"B|{base_name}|{yc_name}|k={k}|{thresh_name}"
                        all_results.append({
                            "method": "B_mult",
                            "base_signal": base_name,
                            "yc_component": yc_name,
                            "param": f"k={k}",
                            "thresholds": thresh_name,
                            **{k2: v for k2, v in res.items() if k2 != "equity"},
                        })
                        equity_store[key] = (common, res["equity"], res["sharpe"])
                        total_runs += 1

    print(f"      {total_runs} runs so far")

    # Method C: Slope momentum
    print("    Method C (slope momentum) ...")
    for base_name, z_base in base_z_aligned.items():
        for yc_name in all_yc_names:
            for mom_win in SLOPE_MOMENTUM_WINDOWS:
                if yc_name not in yc_mom_aligned:
                    continue
                if mom_win not in yc_mom_aligned[yc_name]:
                    continue
                z_mom = yc_mom_aligned[yc_name][mom_win]
                for k in MOMENTUM_K:
                    z_comp = z_base + k * np.nan_to_num(z_mom, 0)
                    for thresh_name, (tl_c, ts_c, tl_s) in THRESHOLD_SETS.items():
                        res = fast_bt(z_comp, reg_arr, rl, rs, tl_c, ts_c, tl_s)
                        if res:
                            key = (f"C|{base_name}|{yc_name}|"
                                   f"w={mom_win}|k={k}|{thresh_name}")
                            all_results.append({
                                "method": "C_momentum",
                                "base_signal": base_name,
                                "yc_component": yc_name,
                                "param": f"win={mom_win},k={k}",
                                "thresholds": thresh_name,
                                **{k2: v for k2, v in res.items()
                                   if k2 != "equity"},
                            })
                            equity_store[key] = (
                                common, res["equity"], res["sharpe"]
                            )
                            total_runs += 1

    print(f"      {total_runs} total runs")

    # ── Analysis ───────────────────────────────────────────────────────────
    print(f"\n[6/7] Analysis ({total_runs:,} configurations) ...")

    df = pd.DataFrame(all_results)
    df.sort_values("sharpe", ascending=False, inplace=True)

    # Baseline reference
    baseline_sh = df[df["method"] == "PURE_RR"]["sharpe"].max()

    # Top 40
    print(f"\n  {'='*130}")
    print(f"  {'Method':<12s} {'Base':<7s} {'YC Component':<18s} "
          f"{'Param':<18s} {'Thresh':<10s}  "
          f"{'Sharpe':>7s} {'AnnRet':>7s} {'Vol':>6s} {'MDD':>7s} "
          f"{'Excess':>7s} {'Ev':>4s} {'EqFin':>7s}")
    print(f"  {'-'*130}")
    for _, r in df.head(40).iterrows():
        marker = " ***" if r["sharpe"] > baseline_sh else ""
        print(f"  {r['method']:<12s} {r['base_signal']:<7s} "
              f"{r['yc_component']:<18s} {r['param']:<18s} "
              f"{r['thresholds']:<10s}  "
              f"{r['sharpe']:+7.3f} {r['ann_ret']:+7.4f} "
              f"{r['vol']:6.4f} {r['mdd']:+7.4f} "
              f"{r['excess']:+7.4f} {r['n_events']:4.0f} "
              f"{r['eq_final']:7.3f}{marker}")
    print(f"  {'='*130}")
    print(f"\n  Baseline (PURE_RR) best Sharpe: {baseline_sh:+.4f}")

    # Save CSV
    csv_path = os.path.join(COMP_DIR, "composite_results.csv")
    df.to_csv(csv_path, index=False)

    # ── Summary tables ─────────────────────────────────────────────────────

    # Best by method
    print(f"\n  BEST BY METHOD:")
    for method in ["PURE_RR", "A_blend", "B_mult", "C_momentum"]:
        sub = df[df["method"] == method]
        if len(sub) > 0:
            best = sub.iloc[0]
            print(f"    {method:<12s}  Sharpe={best['sharpe']:+.4f}  "
                  f"{best['base_signal']}|{best['yc_component']}|"
                  f"{best['param']}|{best['thresholds']}")

    # Best by YC component
    print(f"\n  BEST BY YIELD CURVE COMPONENT:")
    comp_best = (df[df["method"] != "PURE_RR"]
                 .groupby("yc_component")["sharpe"]
                 .max().sort_values(ascending=False))
    for comp, sh in comp_best.items():
        marker = " <<<" if sh > baseline_sh else ""
        print(f"    {comp:<18s}  Sharpe={sh:+.4f}{marker}")

    # Best by base signal
    print(f"\n  BEST BY BASE SIGNAL:")
    for bs in ["z_std", "z_move"]:
        sub = df[df["base_signal"] == bs]
        if len(sub):
            print(f"    {bs:<7s}  Sharpe={sub['sharpe'].max():+.4f}")

    # ── Visualizations ─────────────────────────────────────────────────────
    print(f"\n[7/7] Generating plots ...")

    # --- Plot 1: Sharpe distribution by method ---
    fig, ax = plt.subplots(figsize=(12, 6))
    methods = ["PURE_RR", "A_blend", "B_mult", "C_momentum"]
    method_data = [df[df["method"] == m]["sharpe"].values for m in methods]
    bp = ax.boxplot(method_data, tick_labels=methods, patch_artist=True,
                    boxprops=dict(facecolor="lightblue"))
    ax.axhline(baseline_sh, color="green", ls="--", lw=2,
               label=f"Baseline ({baseline_sh:.3f})")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Distribution by Combination Method")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "01_sharpe_by_method.png"), dpi=150)
    plt.close()

    # --- Plot 2: Best Sharpe by YC component ---
    fig, ax = plt.subplots(figsize=(14, 7))
    comp_sorted = comp_best.sort_values(ascending=True)
    colors = ["seagreen" if sh > baseline_sh else "steelblue"
              for sh in comp_sorted.values]
    ax.barh(range(len(comp_sorted)), comp_sorted.values, color=colors,
            edgecolor="white")
    ax.set_yticks(range(len(comp_sorted)))
    ax.set_yticklabels(comp_sorted.index, fontsize=8)
    ax.axvline(baseline_sh, color="green", ls="--", lw=2,
               label=f"Baseline ({baseline_sh:.3f})")
    ax.set_xlabel("Best Sharpe")
    ax.set_title("Best Sharpe by Yield Curve Component")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "02_sharpe_by_yc_component.png"), dpi=150)
    plt.close()

    # --- Plot 3: Top 15 equity curves ---
    fig, ax = plt.subplots(figsize=(16, 7))
    plotted = set()
    for rank, (_, r) in enumerate(df.head(20).iterrows()):
        # find matching key
        for key in equity_store:
            if (r["method"].replace("_blend", "").replace("_mult", "")
                    .replace("_momentum", "").replace("PURE_RR", "PURE_RR")
                    in key
                    and r["base_signal"] in key
                    and r["thresholds"] in key
                    and key not in plotted):
                d, eq, sh = equity_store[key]
                short_label = (f"#{rank+1} Sh={sh:.3f} "
                               f"{r['method']}|{r['yc_component']}")
                ax.plot(d, eq, linewidth=1.0, alpha=0.8, label=short_label)
                plotted.add(key)
                break
        if len(plotted) >= 12:
            break

    # B&H TLT
    bm = np.cumprod(1.0 + rl)
    ax.plot(common, bm, color="grey", ls="--", linewidth=1.5,
            alpha=0.4, label="B&H TLT")
    ax.set_title("Top Composite Signal Configurations — Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "03_top_equity_curves.png"), dpi=150)
    plt.close()

    # --- Plot 4: Heatmap - Method x YC component ---
    pivot = (df[df["method"] != "PURE_RR"]
             .groupby(["method", "yc_component"])["sharpe"]
             .max().unstack())
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    fig.colorbar(im, ax=ax, label="Best Sharpe")
    ax.set_title("Best Sharpe: Method x YC Component")
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "04_heatmap_method_x_component.png"),
                dpi=150)
    plt.close()

    # --- Plot 5: Method A weight sensitivity ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_i, base_name in enumerate(["z_std", "z_move"]):
        ax = axes[ax_i]
        sub_a = df[(df["method"] == "A_blend") & (df["base_signal"] == base_name)]
        # Extract w_rr from param
        sub_a = sub_a.copy()
        sub_a["w_rr"] = sub_a["param"].str.extract(r"w_rr=([\d.]+)").astype(float)

        for yc in sub_a["yc_component"].unique():
            yc_sub = sub_a[sub_a["yc_component"] == yc]
            best_per_w = yc_sub.groupby("w_rr")["sharpe"].max()
            if best_per_w.max() > baseline_sh * 0.8:
                ax.plot(best_per_w.index, best_per_w.values,
                        marker="o", markersize=4, alpha=0.6, label=yc)

        ax.axhline(baseline_sh, color="green", ls="--", lw=1.5)
        ax.set_xlabel("w_rr (RR weight)")
        ax.set_ylabel("Best Sharpe")
        ax.set_title(f"Method A — {base_name}")
        ax.legend(fontsize=6, loc="best", ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Method A: Blend Weight Sensitivity", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "05_method_A_weight_sensitivity.png"),
                dpi=150)
    plt.close()

    # --- Plot 6: Method C momentum window comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sub_c = df[df["method"] == "C_momentum"].copy()
    sub_c["mom_win"] = sub_c["param"].str.extract(r"win=(\d+)").astype(float)
    sub_c["mom_k"] = sub_c["param"].str.extract(r"k=([\d.]+)").astype(float)

    for win in SLOPE_MOMENTUM_WINDOWS:
        w_sub = sub_c[sub_c["mom_win"] == win]
        best_per_k = w_sub.groupby("mom_k")["sharpe"].max()
        ax.plot(best_per_k.index, best_per_k.values,
                marker="o", linewidth=2, label=f"window={win}d")

    ax.axhline(baseline_sh, color="green", ls="--", lw=1.5,
               label=f"Baseline ({baseline_sh:.3f})")
    ax.set_xlabel("k (momentum weight)")
    ax.set_ylabel("Best Sharpe")
    ax.set_title("Method C: Slope Momentum — Window x Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "06_method_C_momentum.png"), dpi=150)
    plt.close()

    # --- Plot 7: Best composite vs baseline equity comparison ---
    fig, ax = plt.subplots(figsize=(16, 7))

    # Best composite overall
    best_row = df.iloc[0]
    best_composite_key = None
    for key in equity_store:
        d, eq, sh = equity_store[key]
        if abs(sh - best_row["sharpe"]) < 1e-6:
            best_composite_key = key
            break

    if best_composite_key:
        d, eq, sh = equity_store[best_composite_key]
        ax.plot(d, eq, color="steelblue", linewidth=2,
                label=f"Best composite: Sh={sh:.3f}")

    # Best per method
    colors_m = {"A_blend": "coral", "B_mult": "seagreen", "C_momentum": "orchid"}
    for method, color in colors_m.items():
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        best = sub.iloc[0]
        for key in equity_store:
            d, eq, sh_k = equity_store[key]
            if abs(sh_k - best["sharpe"]) < 1e-6 and method[0] in key:
                ax.plot(d, eq, color=color, linewidth=1.2, alpha=0.7,
                        label=f"Best {method}: Sh={sh_k:.3f}")
                break

    # Baseline PURE_RR
    for key in equity_store:
        if "PURE_RR" in key and "z_move" in key and "baseline" in key:
            d, eq, sh = equity_store[key]
            ax.plot(d, eq, color="gold", linewidth=2.5, ls="--",
                    label=f"PURE_RR z_move: Sh={sh:.3f}")
            break

    # B&H TLT
    ax.plot(common, bm, color="grey", ls=":", linewidth=1.5,
            alpha=0.4, label="B&H TLT")

    ax.set_title("Best Composite vs Baseline — Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(COMP_DIR, "07_best_vs_baseline.png"), dpi=150)
    plt.close()

    n_plots = 7
    print(f"\n  {n_plots} plots saved to {COMP_DIR}/")

    # ── Final summary ──────────────────────────────────────────────────────
    elapsed = time.time() - T0
    best = df.iloc[0]
    improvement = best["sharpe"] - baseline_sh

    print(f"\n{'='*70}")
    print(f"  COMPOSITE SIGNAL OPTIMIZATION COMPLETE")
    print(f"  Total configurations: {total_runs:,}")
    print(f"  Baseline Sharpe:      {baseline_sh:+.4f}")
    print(f"  Best composite Sharpe: {best['sharpe']:+.4f}")
    print(f"  Improvement:          {improvement:+.4f}")
    print(f"  Best config: {best['method']}|{best['base_signal']}|"
          f"{best['yc_component']}|{best['param']}|{best['thresholds']}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
