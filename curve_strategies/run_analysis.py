"""
run_analysis.py - Orchestrator for the Yield Curve Relative Value Trading System.

Usage:
    python -m curve_strategies.run_analysis
    python curve_strategies/run_analysis.py

Steps:
    1. Load all data
    2. Compute HMM + YC regimes -> combined regime
    3. Backtest all 25 strategies
    4. Build regime-strategy performance matrix
    5. Run meta-system (LOOKUP + ENSEMBLE)
    6. Walk-forward validation
    7. Generate dashboard and CSV output
"""

import os
import sys
import time
import numpy as np
import pandas as pd

# Ensure parent dir is on path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BACKTEST_START, SAVE_DIR
from data_loader import load_all
from regime import fit_hmm_regime

from curve_strategies.strategy_definitions import STRATEGY_POOL
from curve_strategies.strategy_backtest import (
    backtest_all, compute_yearly_breakdown,
)
from curve_strategies.regime_analysis import (
    compute_yc_regime, build_combined_regime,
    build_regime_performance_matrix, compute_transition_matrix,
    compute_slope_signals,
)
from curve_strategies.meta_system import MetaSystem
from curve_strategies.interactive_dashboard import build_full_dashboard


# ── Output directory ─────────────────────────────────────────────────────
OUT_DIR = os.path.join(SAVE_DIR, "curve_strategies")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    t0 = time.time()
    print("=" * 72)
    print("  YIELD CURVE RELATIVE VALUE TRADING SYSTEM")
    print("  25 strategies | Regime analysis | Meta-system | Walk-forward")
    print("=" * 72)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/7] Loading data ...")
    D = load_all()
    etf_ret = D["etf_ret"]
    yields = D["yields"]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: COMPUTE REGIMES
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/7] Computing regimes ...")

    # HMM regime (CALM / STRESS)
    hmm_regime = fit_hmm_regime(D["move"], D["vix"])
    calm_pct = (hmm_regime == "CALM").mean() * 100
    print(f"  HMM: CALM {calm_pct:.1f}% | STRESS {100-calm_pct:.1f}%")

    # YC regime (BULL_STEEP, BULL_FLAT, BEAR_STEEP, BEAR_FLAT)
    yc_regime = compute_yc_regime(yields, "US_2Y", "US_10Y", window=21)
    for r in ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT", "UNDEFINED"]:
        pct = (yc_regime == r).mean() * 100
        if pct > 0:
            print(f"  YC: {r} {pct:.1f}%")

    # Combined regime
    combined_regime = build_combined_regime(yc_regime, hmm_regime)
    print(f"  Combined: {combined_regime.nunique()} unique states")
    for r in sorted(combined_regime.unique()):
        pct = (combined_regime == r).mean() * 100
        print(f"    {r}: {pct:.1f}%")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: BACKTEST ALL 25 STRATEGIES
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[3/7] Backtesting {len(STRATEGY_POOL)} strategies ...")
    summary_df, results = backtest_all(STRATEGY_POOL, etf_ret, BACKTEST_START, tcost_bps=5)

    print(f"\n  {'#':>3} {'Name':<22} {'Type':<12} {'Sharpe':>7} "
          f"{'AnnRet':>8} {'Vol':>7} {'MDD':>8} {'Calmar':>7}")
    print("  " + "-" * 80)
    for i, (_, r) in enumerate(summary_df.head(25).iterrows(), 1):
        print(f"  {i:>3} {r['display']:<22} {r['type']:<12} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+8.2%} "
              f"{r['vol']:>7.2%} {r['mdd']:>8.2%} {r['calmar']:>7.3f}")

    # Save summary CSV
    summary_df.to_csv(os.path.join(OUT_DIR, "strategy_summary.csv"), index=False)
    print(f"\n  -> strategy_summary.csv ({len(summary_df)} strategies)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: REGIME-STRATEGY PERFORMANCE MATRIX
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/7] Building regime-strategy performance matrix ...")

    strategy_returns = {name: results[name]["daily_ret"] for name in results}
    regime_matrix = build_regime_performance_matrix(strategy_returns, combined_regime)

    # Save matrix
    regime_matrix.to_csv(os.path.join(OUT_DIR, "regime_performance_matrix.csv"))
    print(f"  Matrix shape: {regime_matrix.shape}")
    print(f"  -> regime_performance_matrix.csv")

    # Transition matrix
    trans_matrix = compute_transition_matrix(combined_regime)
    trans_matrix.to_csv(os.path.join(OUT_DIR, "regime_transitions.csv"))
    print(f"  -> regime_transitions.csv")

    # Slope signals (for reference)
    slope_signals = compute_slope_signals(yields)
    print(f"  Slope signals: {slope_signals.shape[1]} columns computed")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: META-SYSTEM
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5/7] Running meta-system ...")

    # Run both modes
    for mode in ["LOOKUP", "ENSEMBLE"]:
        meta = MetaSystem(
            strategy_returns=strategy_returns,
            regime_series=combined_regime,
            mode=mode,
            top_n=3,
            cooldown=5,
            tcost_bps=5,
        )
        meta.fit()
        meta_result = meta.backtest(start=BACKTEST_START)
        m = meta_result["metrics"]

        print(f"\n  {mode}:")
        print(f"    Sharpe  = {m['sharpe']:.4f}")
        print(f"    AnnRet  = {m['ann_ret']:+.2%}")
        print(f"    Vol     = {m['vol']:.2%}")
        print(f"    MDD     = {m['mdd']:.2%}")
        print(f"    Calmar  = {m['calmar']:.4f}")
        print(f"    HitRate = {m['hit_rate']:.2%}")

        if mode == "LOOKUP":
            # Show regime allocations
            allocs = meta.get_regime_allocations()
            print(f"\n    Regime -> Best Strategy:")
            for regime_name, alloc in sorted(allocs.items()):
                best = max(alloc, key=alloc.get)
                print(f"      {regime_name}: {best} ({alloc[best]:.0%})")

    # Use ENSEMBLE as the primary meta-system for dashboard
    meta_ensemble = MetaSystem(
        strategy_returns=strategy_returns,
        regime_series=combined_regime,
        mode="ENSEMBLE",
        top_n=3,
        cooldown=5,
        tcost_bps=5,
    )
    meta_ensemble.fit()
    meta_result_final = meta_ensemble.backtest(start=BACKTEST_START)
    meta_metrics_final = meta_result_final["metrics"]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n[6/7] Walk-forward validation ...")

    wf_results = []
    for mode in ["LOOKUP", "ENSEMBLE"]:
        meta_wf = MetaSystem(
            strategy_returns=strategy_returns,
            regime_series=combined_regime,
            mode=mode,
            top_n=3,
            cooldown=5,
            tcost_bps=5,
        )
        wf_df = meta_wf.walk_forward(
            train_years=3, test_years=1,
            start_year=2012, end_year=2025,
        )
        wf_df["mode"] = mode
        wf_results.append(wf_df)

        avg_sh = wf_df["sharpe"].mean()
        pos = (wf_df["sharpe"] > 0).sum()
        print(f"\n  {mode} Walk-Forward:")
        print(f"    Avg OOS Sharpe: {avg_sh:.3f}")
        print(f"    Positive years: {pos}/{len(wf_df)}")

        for _, r in wf_df.iterrows():
            marker = "+" if r["sharpe"] > 0 else " "
            print(f"    {marker} {r['year']:.0f}: "
                  f"Sharpe={r['sharpe']:>+.3f}  "
                  f"Ret={r['ann_ret']:>+.2%}  "
                  f"MDD={r['mdd']:.2%}")

    wf_combined = pd.concat(wf_results, ignore_index=True)
    wf_combined.to_csv(os.path.join(OUT_DIR, "walk_forward.csv"), index=False)
    print(f"\n  -> walk_forward.csv")

    # Use ENSEMBLE walk-forward for the dashboard
    wf_ensemble = wf_combined[wf_combined["mode"] == "ENSEMBLE"].copy()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: GENERATE DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    print("\n[7/7] Generating interactive dashboard ...")

    dashboard_path = build_full_dashboard(
        results=results,
        strategies=STRATEGY_POOL,
        meta_result=meta_result_final,
        meta_metrics=meta_metrics_final,
        regime_matrix=regime_matrix,
        wf_df=wf_ensemble,
        out_dir=OUT_DIR,
    )

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  COMPLETE  ({elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Files:")
    print(f"    - strategy_summary.csv          ({len(summary_df)} strategies)")
    print(f"    - regime_performance_matrix.csv  ({regime_matrix.shape})")
    print(f"    - regime_transitions.csv         ({trans_matrix.shape})")
    print(f"    - walk_forward.csv               ({len(wf_combined)} rows)")
    print(f"    - curve_strategies_dashboard.html (interactive)")
    print(f"    - tab1_equity_curves.html")
    print(f"    - tab2_meta_system.html")
    print(f"    - tab3_regime_heatmap.html")
    print(f"    - tab4_walk_forward.html")

    # Top 5 strategies
    print(f"\n  TOP 5 STRATEGIES:")
    for i, (_, r) in enumerate(summary_df.head(5).iterrows(), 1):
        print(f"    {i}. {r['display']}: Sharpe={r['sharpe']:.3f} "
              f"Ret={r['ann_ret']:+.2%} MDD={r['mdd']:.2%}")

    # Meta-system summary
    print(f"\n  META-SYSTEM (ENSEMBLE):")
    print(f"    Sharpe={meta_metrics_final['sharpe']:.3f}  "
          f"Ret={meta_metrics_final['ann_ret']:+.2%}  "
          f"MDD={meta_metrics_final['mdd']:.2%}")

    # WF summary
    print(f"\n  WALK-FORWARD (ENSEMBLE):")
    print(f"    Avg OOS Sharpe: {wf_ensemble['sharpe'].mean():.3f}")
    print(f"    Positive years: {(wf_ensemble['sharpe'] > 0).sum()}/{len(wf_ensemble)}")

    print(f"\n{'='*72}")
    return results, meta_result_final, regime_matrix, wf_combined


if __name__ == "__main__":
    main()
