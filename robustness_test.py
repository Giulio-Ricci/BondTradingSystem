"""
Robustness Testing Suite for Versione Presentazione Strategy.

Tests:
1. Walk-Forward Analysis (train/test splits)
2. Parameter Sensitivity (perturbation around optimal)
3. Bootstrap Stability (block bootstrap confidence intervals)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

from config import SAVE_DIR, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime
from versione_presentazione import (
    build_rr_blend, build_slope_zscore, build_rr_multitf,
    build_yc_signal, build_combined, fast_bt, composite_score, ANN
)

OUT_DIR = os.path.join(SAVE_DIR, "robustness")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# BEST CONFIG (from latest optimization)
# ======================================================================

def get_best_config():
    """Load best config from metrics.csv, fallback to hardcoded."""
    metrics_path = os.path.join(SAVE_DIR, "versione_presentazione", "metrics.csv")
    if os.path.exists(metrics_path):
        df = pd.read_csv(metrics_path)
        r = df.iloc[0]
        return {
            "z_windows": eval(r["z_windows"]) if isinstance(r["z_windows"], str) else r["z_windows"],
            "k_slope": float(r["k_slope"]),
            "yc_cw": int(r["yc_cw"]),
            "k_yc": float(r["k_yc"]),
            "tl_c": float(r["tl_c"]),
            "ts_c": float(r["ts_c"]),
            "tl_s": float(r["tl_s"]),
            "adapt_days": int(r["adapt_days"]),
            "adapt_raise": float(r["adapt_raise"]),
            "max_hold_days": int(r["max_hold_days"]),
        }
    # Fallback
    return {
        "z_windows": [63, 252],
        "k_slope": 0.75,
        "yc_cw": 126,
        "k_yc": 0.20,
        "tl_c": -3.50,
        "ts_c": 3.00,
        "tl_s": -2.875,
        "adapt_days": 63,
        "adapt_raise": 1.50,
        "max_hold_days": 252,
    }


def prepare_data():
    """Load and prepare data for backtesting."""
    D = load_all()
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])
    return D, regime_hmm


def build_signal(D, cfg):
    """Build z_final signal for a given config."""
    z_rr, z_base, z_slope = build_rr_multitf(
        D, cfg["z_windows"], k_slope=cfg["k_slope"])
    z_yc, _, _ = build_yc_signal(D, change_window=cfg["yc_cw"])
    z_final = build_combined(z_rr, z_yc, cfg["k_yc"])
    return z_final


def align_data(D, regime_hmm, z_final, start_date=None):
    """Align all data to common dates."""
    etf_ret = D["etf_ret"]
    bt_start = pd.Timestamp(start_date or BACKTEST_START)

    common = (z_final.dropna().index
              .intersection(regime_hmm.dropna().index)
              .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index))
    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])
    z_arr = z_final.reindex(common).values

    return common, z_arr, reg_arr, r_long, r_short, year_arr


def run_bt(z_arr, reg_arr, r_long, r_short, year_arr, cfg):
    """Run backtest with given config."""
    return fast_bt(z_arr, reg_arr, r_long, r_short, year_arr,
                   cfg["tl_c"], cfg["ts_c"], cfg["tl_s"],
                   adapt_days=cfg["adapt_days"],
                   adapt_raise=cfg["adapt_raise"],
                   max_hold_days=cfg["max_hold_days"])


# ======================================================================
# TEST 1: WALK-FORWARD ANALYSIS
# ======================================================================

def run_walk_forward(D, regime_hmm, cfg, verbose=True):
    """Walk-forward: optimize on train period, test on test period.

    Splits:
    - 2009-2015 train / 2016-2026 test
    - 2009-2017 train / 2018-2026 test
    - 2009-2019 train / 2020-2026 test
    - Rolling 5-year train / 3-year test
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  TEST 1: WALK-FORWARD ANALYSIS")
        print("=" * 70)

    z_final = build_signal(D, cfg)
    common, z_arr, reg_arr, r_long, r_short, year_arr = align_data(
        D, regime_hmm, z_final)

    # Full-period baseline
    m_full = run_bt(z_arr, reg_arr, r_long, r_short, year_arr, cfg)

    if verbose:
        print(f"\n  Full period ({common[0]:%Y}-{common[-1]:%Y}): "
              f"Sharpe={m_full['sharpe']:.3f}  Ann={m_full['ann_ret']*100:.1f}%")

    # Fixed splits
    splits = [
        ("2009-2015 / 2016-2026", 2009, 2015, 2016, 2026),
        ("2009-2017 / 2018-2026", 2009, 2017, 2018, 2026),
        ("2009-2019 / 2020-2026", 2009, 2019, 2020, 2026),
        ("2009-2012 / 2013-2018", 2009, 2012, 2013, 2018),
        ("2013-2018 / 2019-2026", 2013, 2018, 2019, 2026),
    ]

    results = []
    if verbose:
        print(f"\n  {'Split':<28} {'Train Sh':>9} {'Test Sh':>8} "
              f"{'Train Ann':>10} {'Test Ann':>9} {'Delta Sh':>9}")
        print("  " + "-" * 78)

    for label, tr_s, tr_e, te_s, te_e in splits:
        # Train period
        tr_mask = (year_arr >= tr_s) & (year_arr <= tr_e)
        te_mask = (year_arr >= te_s) & (year_arr <= te_e)

        if tr_mask.sum() < 252 or te_mask.sum() < 126:
            continue

        # Run backtest on each sub-period with SAME parameters
        # (we're testing if the params generalize, not re-optimizing)
        m_tr = fast_bt(z_arr[tr_mask], reg_arr[tr_mask],
                       r_long[tr_mask], r_short[tr_mask],
                       year_arr[tr_mask],
                       cfg["tl_c"], cfg["ts_c"], cfg["tl_s"],
                       adapt_days=cfg["adapt_days"],
                       adapt_raise=cfg["adapt_raise"],
                       max_hold_days=cfg["max_hold_days"])

        m_te = fast_bt(z_arr[te_mask], reg_arr[te_mask],
                       r_long[te_mask], r_short[te_mask],
                       year_arr[te_mask],
                       cfg["tl_c"], cfg["ts_c"], cfg["tl_s"],
                       adapt_days=cfg["adapt_days"],
                       adapt_raise=cfg["adapt_raise"],
                       max_hold_days=cfg["max_hold_days"])

        delta = m_te["sharpe"] - m_tr["sharpe"]
        results.append({
            "split": label,
            "train_sharpe": m_tr["sharpe"],
            "test_sharpe": m_te["sharpe"],
            "train_ann": m_tr["ann_ret"],
            "test_ann": m_te["ann_ret"],
            "delta_sharpe": delta,
            "train_dead": m_tr["n_dead"],
            "test_dead": m_te["n_dead"],
        })

        if verbose:
            print(f"  {label:<28} {m_tr['sharpe']:>9.3f} {m_te['sharpe']:>8.3f} "
                  f"{m_tr['ann_ret']*100:>9.1f}% {m_te['ann_ret']*100:>8.1f}% "
                  f"{delta:>+9.3f}")

    # Walk-forward re-optimization test
    if verbose:
        print(f"\n  Walk-forward with re-optimization:")
        print(f"  (optimize on train, apply best to test)")
        print(f"  {'Split':<28} {'Train Sh':>9} {'Test Sh':>8} "
              f"{'Test cfg':>40}")
        print("  " + "-" * 90)

    rr_blend = build_rr_blend(D)
    z_slope = build_slope_zscore(D)

    wf_reopt_splits = [
        ("2009-2017 / 2018-2026", 2009, 2017, 2018, 2026),
        ("2009-2019 / 2020-2026", 2009, 2019, 2020, 2026),
    ]

    for label, tr_s, tr_e, te_s, te_e in wf_reopt_splits:
        tr_mask = (year_arr >= tr_s) & (year_arr <= tr_e)
        te_mask = (year_arr >= te_s) & (year_arr <= te_e)

        if tr_mask.sum() < 252 or te_mask.sum() < 126:
            continue

        # Mini grid search on train period
        best_train_sh = -999
        best_cfg = None
        for k_sl in [0.50, 0.75, 1.00]:
            # Rebuild signal with different k_slope
            z_parts = []
            for w in cfg["z_windows"]:
                mu = rr_blend.rolling(w, min_periods=20).mean()
                sd = rr_blend.rolling(w, min_periods=20).std().replace(0, np.nan)
                z_parts.append((rr_blend - mu) / sd)
            z_base = pd.concat(z_parts, axis=1).mean(axis=1)
            z_rr_test = z_base + k_sl * z_slope
            z_yc_test, _, _ = build_yc_signal(D, change_window=cfg["yc_cw"])

            for k_yc in [0.20, 0.30, 0.40]:
                z_f_test = (z_rr_test + k_yc * z_yc_test).reindex(common).values
                for tl_c in [-3.75, -3.50, -3.25, -3.00]:
                    for ts_c in [2.50, 2.75, 3.00, 3.25]:
                        for tl_s in [-3.50, -3.00, -2.875]:
                            m_tr = fast_bt(
                                z_f_test[tr_mask], reg_arr[tr_mask],
                                r_long[tr_mask], r_short[tr_mask],
                                year_arr[tr_mask],
                                tl_c, ts_c, tl_s,
                                adapt_days=cfg["adapt_days"],
                                adapt_raise=cfg["adapt_raise"],
                                max_hold_days=cfg["max_hold_days"])
                            sc = composite_score(m_tr)
                            if sc > best_train_sh:
                                best_train_sh = sc
                                best_cfg = {
                                    **cfg,
                                    "k_slope": k_sl, "k_yc": k_yc,
                                    "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s,
                                }
                                best_train_sharpe = m_tr["sharpe"]

        # Test with best train config
        z_f_best = build_signal(D, best_cfg).reindex(common).values
        m_te = fast_bt(z_f_best[te_mask], reg_arr[te_mask],
                       r_long[te_mask], r_short[te_mask],
                       year_arr[te_mask],
                       best_cfg["tl_c"], best_cfg["ts_c"], best_cfg["tl_s"],
                       adapt_days=best_cfg["adapt_days"],
                       adapt_raise=best_cfg["adapt_raise"],
                       max_hold_days=best_cfg["max_hold_days"])

        cfg_str = (f"k_sl={best_cfg['k_slope']:.2f} k_yc={best_cfg['k_yc']:.2f} "
                   f"[{best_cfg['tl_c']:.2f},{best_cfg['ts_c']:.2f},{best_cfg['tl_s']:.3f}]")

        if verbose:
            print(f"  {label:<28} {best_train_sharpe:>9.3f} {m_te['sharpe']:>8.3f} "
                  f"  {cfg_str}")

        results.append({
            "split": label + " (reopt)",
            "train_sharpe": best_train_sharpe,
            "test_sharpe": m_te["sharpe"],
            "train_ann": 0,
            "test_ann": m_te["ann_ret"],
            "delta_sharpe": m_te["sharpe"] - best_train_sharpe,
            "train_dead": 0,
            "test_dead": m_te["n_dead"],
        })

    return results, m_full


# ======================================================================
# TEST 2: PARAMETER SENSITIVITY
# ======================================================================

def run_sensitivity(D, regime_hmm, cfg, verbose=True):
    """Perturb each parameter and measure Sharpe impact."""
    if verbose:
        print("\n" + "=" * 70)
        print("  TEST 2: PARAMETER SENSITIVITY")
        print("=" * 70)

    z_final = build_signal(D, cfg)
    common, z_arr, reg_arr, r_long, r_short, year_arr = align_data(
        D, regime_hmm, z_final)
    m_base = run_bt(z_arr, reg_arr, r_long, r_short, year_arr, cfg)

    if verbose:
        print(f"\n  Baseline Sharpe: {m_base['sharpe']:.4f}")

    # Parameters to perturb (name, base_value, perturbations)
    perturbations = [
        ("tl_c", cfg["tl_c"], [-0.50, -0.25, 0, +0.25, +0.50]),
        ("ts_c", cfg["ts_c"], [-0.50, -0.25, 0, +0.25, +0.50]),
        ("tl_s", cfg["tl_s"], [-0.50, -0.25, 0, +0.25, +0.50]),
        ("adapt_days", cfg["adapt_days"], [-21, -10, 0, +10, +21]),
        ("adapt_raise", cfg["adapt_raise"], [-0.50, -0.25, 0, +0.25, +0.50]),
        ("max_hold_days", cfg["max_hold_days"], [-63, -21, 0, +21, +63]),
    ]

    # For signal-level params, rebuild signal
    rr_blend = build_rr_blend(D)
    z_slope = build_slope_zscore(D)

    signal_perturbations = [
        ("k_slope", cfg["k_slope"], [-0.25, -0.10, 0, +0.10, +0.25]),
        ("k_yc", cfg["k_yc"], [-0.10, -0.05, 0, +0.05, +0.10]),
    ]

    all_results = []

    if verbose:
        print(f"\n  {'Param':<16} {'Base':>8} {'Perturbed':>10} {'Sharpe':>8} "
              f"{'Delta':>8} {'Dead':>5} {'WR':>5}")
        print("  " + "-" * 65)

    # Threshold perturbations (same signal, different thresholds)
    for param_name, base_val, deltas in perturbations:
        for d in deltas:
            test_cfg = cfg.copy()
            new_val = base_val + d
            # Clamp
            if param_name == "adapt_days":
                new_val = max(0, int(new_val))
            elif param_name == "max_hold_days":
                new_val = max(0, int(new_val))
            elif param_name == "adapt_raise":
                new_val = max(0.0, new_val)
            test_cfg[param_name] = new_val

            m = run_bt(z_arr, reg_arr, r_long, r_short, year_arr, test_cfg)
            delta_sh = m["sharpe"] - m_base["sharpe"]
            all_results.append({
                "param": param_name, "base": base_val,
                "perturbed": new_val, "sharpe": m["sharpe"],
                "delta": delta_sh, "n_dead": m["n_dead"],
                "win_rate": m["win_rate"],
            })

            if verbose and d != 0:
                print(f"  {param_name:<16} {base_val:>8.3f} {new_val:>10.3f} "
                      f"{m['sharpe']:>8.4f} {delta_sh:>+8.4f} "
                      f"{m['n_dead']:>5.0f} {m['win_rate']:>5.0%}")

    # Signal perturbations (need to rebuild signal)
    for param_name, base_val, deltas in signal_perturbations:
        for d in deltas:
            new_val = base_val + d
            if new_val < 0:
                continue

            test_cfg = cfg.copy()
            test_cfg[param_name] = new_val
            z_f_test = build_signal(D, test_cfg)
            z_f_arr = z_f_test.reindex(common).values

            m = run_bt(z_f_arr, reg_arr, r_long, r_short, year_arr, test_cfg)
            delta_sh = m["sharpe"] - m_base["sharpe"]
            all_results.append({
                "param": param_name, "base": base_val,
                "perturbed": new_val, "sharpe": m["sharpe"],
                "delta": delta_sh, "n_dead": m["n_dead"],
                "win_rate": m["win_rate"],
            })

            if verbose and d != 0:
                print(f"  {param_name:<16} {base_val:>8.3f} {new_val:>10.3f} "
                      f"{m['sharpe']:>8.4f} {delta_sh:>+8.4f} "
                      f"{m['n_dead']:>5.0f} {m['win_rate']:>5.0%}")

    return all_results, m_base


# ======================================================================
# TEST 3: BOOTSTRAP STABILITY
# ======================================================================

def run_bootstrap(D, regime_hmm, cfg, n_boot=1000, block_size=21,
                  verbose=True):
    """Block bootstrap to estimate Sharpe confidence interval.

    Uses non-overlapping blocks of block_size days, resampled with replacement.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("  TEST 3: BOOTSTRAP STABILITY")
        print(f"  ({n_boot} iterations, block_size={block_size} days)")
        print("=" * 70)

    z_final = build_signal(D, cfg)
    common, z_arr, reg_arr, r_long, r_short, year_arr = align_data(
        D, regime_hmm, z_final)

    # Run strategy once to get daily returns
    m_base = run_bt(z_arr, reg_arr, r_long, r_short, year_arr, cfg)
    strat_ret = m_base["net_ret"]
    bh_ret = r_long  # benchmark

    n = len(strat_ret)
    n_blocks = n // block_size

    # Compute excess daily returns (strategy - benchmark)
    excess_daily = strat_ret - bh_ret

    # Block bootstrap of excess returns
    rng = np.random.RandomState(42)
    boot_sharpes = []
    boot_ann_rets = []
    boot_excess = []

    for b in range(n_boot):
        # Resample blocks
        block_indices = rng.randint(0, n_blocks, size=n_blocks)
        boot_idx = []
        for bi in block_indices:
            start = bi * block_size
            boot_idx.extend(range(start, min(start + block_size, n)))
        boot_idx = np.array(boot_idx[:n])

        # Bootstrap strategy returns
        boot_strat = strat_ret[boot_idx]
        boot_bh = bh_ret[boot_idx]

        # Compute metrics
        eq = np.cumprod(1.0 + boot_strat)
        total_ret = eq[-1] - 1.0
        years = len(boot_strat) / ANN
        ann_ret = (1.0 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0
        vol = np.std(boot_strat) * np.sqrt(ANN)
        sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0

        bh_eq = np.cumprod(1.0 + boot_bh)
        bh_total = bh_eq[-1] - 1.0
        bh_ann = (1.0 + bh_total) ** (1.0 / max(years, 0.01)) - 1.0

        boot_sharpes.append(sharpe)
        boot_ann_rets.append(ann_ret)
        boot_excess.append(ann_ret - bh_ann)

    boot_sharpes = np.array(boot_sharpes)
    boot_ann_rets = np.array(boot_ann_rets)
    boot_excess = np.array(boot_excess)

    # Confidence intervals
    ci_levels = [5, 10, 25, 50, 75, 90, 95]
    percentiles = np.percentile(boot_sharpes, ci_levels)

    if verbose:
        print(f"\n  Original Sharpe: {m_base['sharpe']:.4f}")
        print(f"\n  Bootstrap Sharpe distribution ({n_boot} samples):")
        print(f"    Mean:   {np.mean(boot_sharpes):.4f}")
        print(f"    Std:    {np.std(boot_sharpes):.4f}")
        print(f"    Median: {np.median(boot_sharpes):.4f}")
        for ci, p in zip(ci_levels, percentiles):
            print(f"    {ci:>3}th:  {p:.4f}")

        print(f"\n  P(Sharpe > 0):     {np.mean(boot_sharpes > 0)*100:.1f}%")
        print(f"  P(Sharpe > 0.20):  {np.mean(boot_sharpes > 0.20)*100:.1f}%")
        print(f"  P(Sharpe > 0.30):  {np.mean(boot_sharpes > 0.30)*100:.1f}%")

        print(f"\n  Bootstrap Excess Return distribution:")
        print(f"    Mean:   {np.mean(boot_excess)*100:+.2f}%")
        print(f"    5th:    {np.percentile(boot_excess, 5)*100:+.2f}%")
        print(f"    95th:   {np.percentile(boot_excess, 95)*100:+.2f}%")
        print(f"    P(excess > 0): {np.mean(boot_excess > 0)*100:.1f}%")

    return boot_sharpes, boot_ann_rets, boot_excess, m_base


# ======================================================================
# PLOTTING
# ======================================================================

def plot_robustness(wf_results, sensitivity_results, boot_sharpes,
                    m_base, cfg, out_dir):
    """Generate comprehensive robustness charts."""

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle("ROBUSTNESS ANALYSIS — Versione Presentazione",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30,
                          left=0.07, right=0.95, top=0.93, bottom=0.05)

    # 1. Walk-forward bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    wf_df = pd.DataFrame(wf_results)
    x = range(len(wf_df))
    w = 0.35
    bars1 = ax1.bar([i - w/2 for i in x], wf_df["train_sharpe"],
                     w, label="Train", color="#2874a6", alpha=0.8)
    bars2 = ax1.bar([i + w/2 for i in x], wf_df["test_sharpe"],
                     w, label="Test", color="#e67e22", alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[:20] for s in wf_df["split"]], rotation=45,
                         ha="right", fontsize=8)
    ax1.axhline(m_base["sharpe"], color="red", linestyle="--",
                alpha=0.5, label=f"Full Sharpe={m_base['sharpe']:.3f}")
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.set_ylabel("Sharpe Ratio")
    ax1.set_title("Walk-Forward: Train vs Test Sharpe", fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # 2. Bootstrap Sharpe distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(boot_sharpes, bins=50, color="#2874a6", alpha=0.7,
             edgecolor="white", density=True)
    ax2.axvline(m_base["sharpe"], color="red", linewidth=2,
                label=f"Original={m_base['sharpe']:.3f}")
    ax2.axvline(np.mean(boot_sharpes), color="#e67e22", linewidth=2,
                linestyle="--", label=f"Mean={np.mean(boot_sharpes):.3f}")
    ci5 = np.percentile(boot_sharpes, 5)
    ci95 = np.percentile(boot_sharpes, 95)
    ax2.axvline(ci5, color="gray", linestyle=":", alpha=0.7,
                label=f"90% CI: [{ci5:.3f}, {ci95:.3f}]")
    ax2.axvline(ci95, color="gray", linestyle=":", alpha=0.7)
    ax2.axvspan(ci5, ci95, alpha=0.08, color="#27ae60")
    ax2.axvline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Sharpe Ratio")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Bootstrap Distribution (n={len(boot_sharpes)})",
                   fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    # 3-4. Sensitivity tornado charts
    sens_df = pd.DataFrame(sensitivity_results)
    # Only non-zero deltas
    sens_nz = sens_df[sens_df["delta"].abs() > 0.0001].copy()

    # Split into two groups
    thresh_params = ["tl_c", "ts_c", "tl_s", "adapt_days",
                     "adapt_raise", "max_hold_days"]
    signal_params = ["k_slope", "k_yc"]

    for idx, (params, title) in enumerate([
        (thresh_params, "Threshold Parameter Sensitivity"),
        (signal_params, "Signal Parameter Sensitivity"),
    ]):
        ax = fig.add_subplot(gs[1, idx])
        sub = sens_nz[sens_nz["param"].isin(params)]
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontweight="bold")
            continue

        # Group by param, show min/max delta
        param_ranges = []
        for p in params:
            p_data = sub[sub["param"] == p]
            if len(p_data) == 0:
                continue
            min_d = p_data["delta"].min()
            max_d = p_data["delta"].max()
            param_ranges.append({"param": p, "min_delta": min_d,
                                  "max_delta": max_d,
                                  "range": max_d - min_d})

        if not param_ranges:
            ax.set_title(title, fontweight="bold")
            continue

        pr_df = pd.DataFrame(param_ranges).sort_values("range", ascending=True)
        y = range(len(pr_df))
        for i, (_, row) in enumerate(pr_df.iterrows()):
            color_l = "#e74c3c" if row["min_delta"] < 0 else "#27ae60"
            color_r = "#27ae60" if row["max_delta"] > 0 else "#e74c3c"
            ax.barh(i, row["min_delta"], color=color_l, alpha=0.7, height=0.6)
            ax.barh(i, row["max_delta"], color=color_r, alpha=0.7, height=0.6)

        ax.set_yticks(y)
        ax.set_yticklabels(pr_df["param"], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta Sharpe")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")

    # 5. Summary statistics table
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis("off")

    wf_test_sharpes = [r["test_sharpe"] for r in wf_results
                       if "reopt" not in r["split"]]

    summary = [
        ["Full-Period Sharpe", f"{m_base['sharpe']:.4f}"],
        ["Bootstrap Mean Sharpe", f"{np.mean(boot_sharpes):.4f}"],
        ["Bootstrap 90% CI",
         f"[{np.percentile(boot_sharpes, 5):.3f}, "
         f"{np.percentile(boot_sharpes, 95):.3f}]"],
        ["P(Sharpe > 0)", f"{np.mean(boot_sharpes > 0)*100:.1f}%"],
        ["P(Sharpe > 0.20)", f"{np.mean(boot_sharpes > 0.20)*100:.1f}%"],
        ["Walk-Forward Avg Test Sharpe",
         f"{np.mean(wf_test_sharpes):.4f}" if wf_test_sharpes else "N/A"],
        ["WF Train-Test Gap",
         f"{np.mean([r['delta_sharpe'] for r in wf_results if 'reopt' not in r['split']]):+.4f}"
         if wf_results else "N/A"],
        ["Max Param Sensitivity",
         f"{sens_nz['delta'].abs().max():.4f}" if len(sens_nz) > 0 else "N/A"],
        ["Dead Years", f"{m_base['n_dead']}"],
        ["Win Rate", f"{m_base['win_rate']:.0%}"],
    ]

    table = ax5.table(cellText=summary, colLabels=["Metric", "Value"],
                       loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2874a6")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eaf2f8")
    ax5.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=15)

    # 6. Robustness verdict
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    # Compute verdict
    boot_mean = np.mean(boot_sharpes)
    boot_std = np.std(boot_sharpes)
    p_positive = np.mean(boot_sharpes > 0) * 100
    wf_gap = abs(np.mean([r["delta_sharpe"] for r in wf_results
                           if "reopt" not in r["split"]])) if wf_results else 0
    max_sens = sens_nz["delta"].abs().max() if len(sens_nz) > 0 else 0

    checks = []
    checks.append(("Bootstrap P(Sh>0) > 80%",
                    p_positive > 80, f"{p_positive:.0f}%"))
    checks.append(("Bootstrap mean within 20% of original",
                    abs(boot_mean - m_base["sharpe"]) < 0.2 * abs(m_base["sharpe"]),
                    f"{boot_mean:.3f} vs {m_base['sharpe']:.3f}"))
    checks.append(("WF train-test gap < 0.15",
                    wf_gap < 0.15, f"{wf_gap:.3f}"))
    checks.append(("Max param sensitivity < 0.10",
                    max_sens < 0.10, f"{max_sens:.3f}"))

    passed = sum(1 for _, ok, _ in checks if ok)
    total = len(checks)
    verdict = "ROBUST" if passed >= 3 else ("MODERATE" if passed >= 2 else "FRAGILE")
    verdict_color = ("#27ae60" if verdict == "ROBUST"
                     else "#f39c12" if verdict == "MODERATE" else "#e74c3c")

    y_pos = 0.90
    ax6.text(0.5, y_pos, f"VERDICT: {verdict}",
             fontsize=18, fontweight="bold", color=verdict_color,
             ha="center", va="top", transform=ax6.transAxes)

    ax6.text(0.5, y_pos - 0.08, f"({passed}/{total} checks passed)",
             fontsize=11, color="gray", ha="center", va="top",
             transform=ax6.transAxes)

    for i, (check_name, ok, val) in enumerate(checks):
        symbol = "PASS" if ok else "FAIL"
        color = "#27ae60" if ok else "#e74c3c"
        ax6.text(0.10, y_pos - 0.20 - i * 0.12, f"  [{symbol}] {check_name}",
                 fontsize=10, color=color, va="top", transform=ax6.transAxes,
                 fontfamily="monospace")
        ax6.text(0.85, y_pos - 0.20 - i * 0.12, val,
                 fontsize=10, color="gray", va="top", transform=ax6.transAxes,
                 ha="right", fontfamily="monospace")

    ax6.set_title("Robustness Verdict", fontsize=12, fontweight="bold", pad=15)

    fig.savefig(os.path.join(out_dir, "robustness_report.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
# MAIN
# ======================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("  ROBUSTNESS TESTING SUITE")
    print("  Versione Presentazione Strategy")
    print("=" * 70)

    D, regime_hmm = prepare_data()
    cfg = get_best_config()

    print(f"\n  Config under test:")
    for k, v in cfg.items():
        print(f"    {k:<20} = {v}")

    # Test 1: Walk-Forward
    wf_results, m_full = run_walk_forward(D, regime_hmm, cfg)

    # Test 2: Sensitivity
    sens_results, m_base = run_sensitivity(D, regime_hmm, cfg)

    # Test 3: Bootstrap
    boot_sharpes, boot_anns, boot_excess, _ = run_bootstrap(
        D, regime_hmm, cfg, n_boot=2000, block_size=21)

    # Plot
    print("\n  Generating robustness report chart...")
    plot_robustness(wf_results, sens_results, boot_sharpes,
                    m_base, cfg, OUT_DIR)
    print(f"  Saved: {OUT_DIR}/robustness_report.png")

    # Save results
    pd.DataFrame(wf_results).to_csv(
        os.path.join(OUT_DIR, "walk_forward.csv"), index=False)
    pd.DataFrame(sens_results).to_csv(
        os.path.join(OUT_DIR, "sensitivity.csv"), index=False)
    pd.DataFrame({"boot_sharpe": boot_sharpes}).to_csv(
        os.path.join(OUT_DIR, "bootstrap_sharpes.csv"), index=False)
    print(f"  Saved: walk_forward.csv, sensitivity.csv, bootstrap_sharpes.csv")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
