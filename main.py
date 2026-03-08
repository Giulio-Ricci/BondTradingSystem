"""Main orchestrator for the Bond Trading System - RR Duration Strategy.

Usage:
    python main.py
"""

import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from config import DEFAULT_PARAMS, BACKTEST_START


def run_validation():
    """Execute all validation phases sequentially."""
    t0 = time.time()

    # ── Step 1: Load Data ───────────────────────────────────────────────────
    print("\n" + "#" * 60)
    print("# BOND TRADING SYSTEM - RR Duration Strategy")
    print("#" * 60)

    print("\n[1/8] Loading data...")
    from data_loader import load_all
    D = load_all()

    # ── Step 2: Compute Signals ─────────────────────────────────────────────
    print("\n[2/8] Computing signals...")
    from signals import compute_all_signals
    D = compute_all_signals(D)

    # ── Step 3: Fit Regime ──────────────────────────────────────────────────
    print("\n[3/8] Fitting HMM regime...")
    from regime import compute_regime
    D = compute_regime(D)

    # ── Step 4: Full-Sample Backtest (baseline) ─────────────────────────────
    print("\n[4/8] Running baseline backtest...")
    from backtest import backtest

    # Pick the best z-key from a quick comparison
    z_key = "z_move"  # default choice (MOVE-adjusted = "best compromise")
    params = dict(DEFAULT_PARAMS)

    res = backtest(D[z_key], D["regime"], D["etf_ret"], params=params, start=BACKTEST_START)
    m = res["metrics"]
    print(f"\n  Baseline ({z_key}):")
    print(f"    Ann Return : {m['ann_ret']:.3f}")
    print(f"    Volatility : {m['vol']:.3f}")
    print(f"    Sharpe     : {m['sharpe']:.3f}")
    print(f"    Sortino    : {m['sortino']:.3f}")
    print(f"    Max DD     : {m['mdd']:.3f}")
    print(f"    Calmar     : {m['calmar']:.3f}")
    print(f"    Excess     : {m['excess']:.4f}")
    print(f"    Events     : {m['n_events']}")

    if res["yearly"] is not None and len(res["yearly"]) > 0:
        print("\n  Yearly Breakdown:")
        print(res["yearly"].to_string(index=False))

    # ── Step 5: Grid Search ─────────────────────────────────────────────────
    print("\n[5/8] Grid search...")
    from validation import (
        phase1_grid, phase2_sensitivity, phase3_walk_forward,
        phase4_bootstrap, phase5_regime_stability, phase6_tc_sensitivity,
        phase7_benchmarks, phase8_summary,
    )

    grid_df = phase1_grid(D)

    # Use top configuration
    if len(grid_df) > 0:
        top = grid_df.iloc[0]
        z_key = top["z_key"]
        params["tl_calm"] = top["tl_calm"]
        params["ts_calm"] = top["ts_calm"]
        params["tl_stress"] = top["tl_stress"]
        print(f"\n  Selected: z={z_key}  tl_calm={params['tl_calm']}  "
              f"ts_calm={params['ts_calm']}  tl_stress={params['tl_stress']}")

        # Re-run backtest with top params
        res = backtest(D[z_key], D["regime"], D["etf_ret"], params=params, start=BACKTEST_START)

    # ── Step 6: Sensitivity ─────────────────────────────────────────────────
    print("\n[6/8] Sensitivity, walk-forward, bootstrap, regime, TC, benchmarks...")
    sens = phase2_sensitivity(D, params, z_key)

    # ── Step 7: Walk-Forward ────────────────────────────────────────────────
    wf_df = phase3_walk_forward(D, params, z_key)

    # ── Bootstrap ───────────────────────────────────────────────────────────
    boot = phase4_bootstrap(D, params, z_key)

    # ── Regime Stability ────────────────────────────────────────────────────
    regime_df = phase5_regime_stability(D)

    # ── TC Sensitivity ──────────────────────────────────────────────────────
    tc_df = phase6_tc_sensitivity(D, params, z_key)

    # ── Benchmarks ──────────────────────────────────────────────────────────
    bm_df = phase7_benchmarks(D, params, z_key)

    # ── Summary Scorecard ───────────────────────────────────────────────────
    summary = phase8_summary(D, params, z_key, grid_df, wf_df, boot, regime_df, tc_df)

    # ── Generate Plots ──────────────────────────────────────────────────────
    print("\n[8/8] Generating plots...")
    from visualization import generate_all_plots
    generate_all_plots(res, D, params, z_key, sens, wf_df, boot, tc_df, bm_df)

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done. Total time: {elapsed:.1f}s")
    print(f"Outputs saved to: {os.path.abspath(os.path.join(os.path.dirname(__file__), 'output'))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_validation()
