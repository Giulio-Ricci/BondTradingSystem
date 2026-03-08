"""
mean_reversion_analysis.py - Analyze z_level mean-reversion properties.

Goal: understand whether z_level predicts FUTURE spread changes (contrarian signal)
and compare its predictive power vs z_mom.

Analysis:
  1. Correlation of z_level(t) with forward spread change (63d, 126d)
  2. Same for z_mom(t) vs forward spread change
  3. Sharpe of simple contrarian strategy: z_level > +1 -> flattener, z_level < -1 -> steepener
  4. Regime conditioning: does z_level work better in CALM vs STRESS?
  5. Integration with existing regime_analysis.py
"""

import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_all
from curve_strategies.regime_analysis import (
    compute_slope_signals, compute_yc_regime, SLOPE_SEGMENTS
)
from curve_strategies.strategy_backtest import backtest_all, compute_metrics
from curve_strategies.strategy_definitions import STRATEGY_POOL
from curve_strategies.portfolio_combinations import build_returns_matrix
from config import BACKTEST_START, RISK_FREE_RATE

ANN = 252

# Extended segments (same as in curve_signal_system.py)
EXTENDED_SLOPE_SEGMENTS = {
    **SLOPE_SEGMENTS,
    "5s30s":  ("US_5Y",  "US_30Y"),
    "10s20s": ("US_10Y", "US_20Y"),
}

# Segment -> (steepener_strategy, flattener_strategy) mapping for PnL
SEGMENT_STRAT_MAP = {
    "2s5s":   ("steep_2s5s",   None),
    "5s10s":  ("steep_5s10s",  None),
    "10s30s": ("steep_20s30s", "flat_30s10s"),
    "2s10s":  ("steep_2s10s",  "flat_10s2s"),
    "2s30s":  ("steep_2s30s",  "flat_30s2s"),
    "5s30s":  ("steep_5s30s",  "flat_30s5s"),
}


def main():
    print("=" * 80)
    print("  MEAN-REVERSION ANALYSIS OF z_level SIGNALS")
    print("  Contrarian yield curve spread trading")
    print("=" * 80)

    # ══════════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1] Loading data ...")
    D = load_all()
    yields = D["yields"]
    etf_ret = D["etf_ret"]
    vix = D["vix"]
    move = D["move"]

    # Compute slope signals
    print("\n[2] Computing slope signals ...")
    seg_signals = compute_slope_signals(yields, segments=EXTENDED_SLOPE_SEGMENTS,
                                         z_window=252, mom_window=63)
    print(f"  Signal shape: {seg_signals.shape}")
    print(f"  Date range: {seg_signals.index[0]} to {seg_signals.index[-1]}")

    # Backtest strategies to get daily returns
    print("\n[3] Backtesting strategies for PnL mapping ...")
    summary_df, results = backtest_all(STRATEGY_POOL, etf_ret,
                                        start=BACKTEST_START, tcost_bps=0)
    strategy_returns = {name: res["daily_ret"] for name, res in results.items()}
    returns_matrix = build_returns_matrix(strategy_returns)

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 1: PREDICTIVE POWER OF z_level vs z_mom
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 1: PREDICTIVE POWER - z_level(t) vs FORWARD spread change")
    print("=" * 80)
    print("\n  If z_level is MEAN-REVERTING, we expect NEGATIVE correlation:")
    print("  z_level high -> spread DECLINES in future (reverts to mean)")
    print("  z_level low  -> spread RISES in future")
    print()

    horizons = [21, 63, 126, 252]
    segments = ["2s5s", "2s10s", "5s10s", "10s30s", "2s30s", "5s30s"]

    # Header
    print(f"  {'Segment':<10s}", end="")
    for h in horizons:
        print(f"  {'corr_zlev_'+str(h)+'d':>18s}", end="")
    for h in horizons:
        print(f"  {'corr_zmom_'+str(h)+'d':>18s}", end="")
    print()
    print("  " + "-" * (10 + len(horizons) * 2 * 20))

    corr_results = []

    for seg in segments:
        spread_col = f"{seg}_spread"
        zlevel_col = f"{seg}_z_level"
        zmom_col = f"{seg}_z_mom"

        if spread_col not in seg_signals.columns:
            continue

        spread = seg_signals[spread_col].dropna()
        z_level = seg_signals[zlevel_col].dropna()
        z_mom = seg_signals[zmom_col].dropna()

        row = {"segment": seg}
        print(f"  {seg:<10s}", end="")

        for h in horizons:
            # Forward spread change
            fwd_change = spread.diff(h).shift(-h)

            # Align
            common = z_level.index.intersection(fwd_change.dropna().index)
            if len(common) < 100:
                print(f"  {'N/A':>18s}", end="")
                row[f"zlev_corr_{h}d"] = np.nan
                continue

            corr_zlev = z_level.reindex(common).corr(fwd_change.reindex(common))
            row[f"zlev_corr_{h}d"] = corr_zlev
            # Color coding: negative = good for mean-reversion
            marker = "***" if corr_zlev < -0.10 else "  *" if corr_zlev < -0.05 else "   "
            print(f"  {corr_zlev:>+12.4f} {marker}", end="")

        for h in horizons:
            fwd_change = spread.diff(h).shift(-h)
            common = z_mom.index.intersection(fwd_change.dropna().index)
            if len(common) < 100:
                print(f"  {'N/A':>18s}", end="")
                row[f"zmom_corr_{h}d"] = np.nan
                continue

            corr_zmom = z_mom.reindex(common).corr(fwd_change.reindex(common))
            row[f"zmom_corr_{h}d"] = corr_zmom
            marker = "***" if corr_zmom < -0.10 else "  *" if corr_zmom < -0.05 else "   "
            print(f"  {corr_zmom:>+12.4f} {marker}", end="")

        print()
        corr_results.append(row)

    print("\n  Legend: *** = strong mean-reversion (corr < -0.10)")
    print("           * = moderate mean-reversion (corr < -0.05)")
    print("  NEGATIVE correlation = signal is MEAN-REVERTING (good for contrarian)")
    print("  POSITIVE correlation = signal is TREND-FOLLOWING")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 1b: RANK CORRELATION (Spearman) - more robust
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("  ANALYSIS 1b: RANK CORRELATION (Spearman) - more robust to outliers")
    print("-" * 80)

    print(f"\n  {'Segment':<10s}", end="")
    for h in horizons:
        print(f"  {'spearman_zlev_'+str(h)+'d':>20s}", end="")
    print()
    print("  " + "-" * (10 + len(horizons) * 22))

    for seg in segments:
        spread_col = f"{seg}_spread"
        zlevel_col = f"{seg}_z_level"

        if spread_col not in seg_signals.columns:
            continue

        spread = seg_signals[spread_col].dropna()
        z_level = seg_signals[zlevel_col].dropna()

        print(f"  {seg:<10s}", end="")

        for h in horizons:
            fwd_change = spread.diff(h).shift(-h)
            common = z_level.index.intersection(fwd_change.dropna().index)
            if len(common) < 100:
                print(f"  {'N/A':>20s}", end="")
                continue

            spearman_corr = z_level.reindex(common).rank().corr(
                fwd_change.reindex(common).rank())
            marker = "***" if spearman_corr < -0.10 else "  *" if spearman_corr < -0.05 else "   "
            print(f"  {spearman_corr:>+14.4f} {marker}", end="")

        print()

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 2: CONDITIONAL FORWARD RETURNS BY z_level QUINTILE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 2: FORWARD SPREAD CHANGE BY z_level QUINTILE")
    print("=" * 80)
    print("  Mean forward spread change (bps) by z_level quintile")
    print("  If mean-reverting: Q5 (high z) should show negative fwd change")
    print()

    for seg in segments:
        spread_col = f"{seg}_spread"
        zlevel_col = f"{seg}_z_level"

        if spread_col not in seg_signals.columns:
            continue

        spread = seg_signals[spread_col]
        z_level = seg_signals[zlevel_col]

        print(f"\n  --- {seg} ---")
        for h in [63, 126]:
            fwd_change = (spread.diff(h).shift(-h)) * 100  # in bps
            df = pd.DataFrame({
                "z_level": z_level,
                "fwd_change": fwd_change
            }).dropna()

            if len(df) < 200:
                print(f"    {h}d: insufficient data")
                continue

            df["quintile"] = pd.qcut(df["z_level"], 5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"])
            qstats = df.groupby("quintile")["fwd_change"].agg(["mean", "std", "count"])

            print(f"    Forward {h}d spread change (bps):")
            for q in qstats.index:
                m = qstats.loc[q, "mean"]
                s = qstats.loc[q, "std"]
                n = qstats.loc[q, "count"]
                t_stat = m / (s / np.sqrt(n)) if s > 0 else 0
                print(f"      {q:>10s}: mean={m:>+7.1f} bps, std={s:>6.1f}, n={n:>5.0f}, t={t_stat:>+5.2f}")

            # Q5-Q1 spread (long-short quintile)
            q5_mean = qstats.loc["Q5(high)", "mean"]
            q1_mean = qstats.loc["Q1(low)", "mean"]
            print(f"      Q5-Q1 spread: {q5_mean - q1_mean:>+7.1f} bps "
                  f"{'(MEAN-REVERTING)' if q5_mean < q1_mean else '(TREND-FOLLOWING)'}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 3: CONTRARIAN STRATEGY SHARPE RATIO
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 3: CONTRARIAN STRATEGY SHARPE RATIOS")
    print("=" * 80)
    print("  Strategy: z_level > +1 -> FLATTENER, z_level < -1 -> STEEPENER")
    print("  (This is the OPPOSITE of the current momentum system)")
    print()

    thresholds_to_test = [0.5, 1.0, 1.5, 2.0]

    print(f"  {'Segment':<10s}", end="")
    for t in thresholds_to_test:
        print(f"  {'thresh='+str(t):>14s}", end="")
    print(f"  {'MOM(follow)':>14s}")
    print("  " + "-" * (10 + (len(thresholds_to_test) + 1) * 16))

    for seg in segments:
        zlevel_col = f"{seg}_z_level"
        zmom_col = f"{seg}_z_mom"

        if zlevel_col not in seg_signals.columns:
            continue

        strat_map = SEGMENT_STRAT_MAP.get(seg)
        if strat_map is None:
            continue

        steep_name, flat_name = strat_map

        # We need both steepener AND flattener to trade contrarian
        if steep_name is None or steep_name not in returns_matrix.columns:
            continue
        if flat_name is None or flat_name not in returns_matrix.columns:
            # If no flattener, skip (we can only go one direction)
            continue

        steep_ret = returns_matrix[steep_name]
        flat_ret = returns_matrix[flat_name]
        z_level = seg_signals[zlevel_col].reindex(returns_matrix.index)
        z_mom = seg_signals[zmom_col].reindex(returns_matrix.index)

        print(f"  {seg:<10s}", end="")

        for thresh in thresholds_to_test:
            # CONTRARIAN: z_level high -> flattener, z_level low -> steepener
            signal_flat = (z_level.shift(1) > thresh).astype(float)
            signal_steep = (z_level.shift(1) < -thresh).astype(float)

            daily_ret = signal_flat * flat_ret + signal_steep * steep_ret
            daily_ret = daily_ret.dropna()

            if len(daily_ret) < 252:
                print(f"  {'N/A':>14s}", end="")
                continue

            # Compute Sharpe
            invested_days = ((signal_flat + signal_steep) > 0).sum()
            if invested_days < 50:
                print(f"  {'few_trades':>14s}", end="")
                continue

            metrics = compute_metrics(daily_ret)
            sharpe = metrics["sharpe"]
            ann_ret = metrics["ann_ret"]
            pct_invested = invested_days / len(daily_ret) * 100
            print(f"  {sharpe:>+8.3f}({pct_invested:>3.0f}%)", end="")

        # Also compute MOMENTUM-FOLLOWING strategy for comparison
        thresh_comp = 1.0
        signal_steep_mom = (z_mom.shift(1) > thresh_comp).astype(float)
        signal_flat_mom = (z_mom.shift(1) < -thresh_comp).astype(float)
        daily_ret_mom = signal_steep_mom * steep_ret + signal_flat_mom * flat_ret
        daily_ret_mom = daily_ret_mom.dropna()
        if len(daily_ret_mom) >= 252:
            invested_mom = ((signal_steep_mom + signal_flat_mom) > 0).sum()
            if invested_mom >= 50:
                metrics_mom = compute_metrics(daily_ret_mom)
                pct_inv_mom = invested_mom / len(daily_ret_mom) * 100
                print(f"  {metrics_mom['sharpe']:>+8.3f}({pct_inv_mom:>3.0f}%)", end="")

        print()

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 3b: DETAILED CONTRARIAN STATS (threshold=1.0)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("  ANALYSIS 3b: DETAILED CONTRARIAN STATS (threshold=1.0)")
    print("-" * 80)

    for seg in segments:
        zlevel_col = f"{seg}_z_level"
        strat_map = SEGMENT_STRAT_MAP.get(seg)
        if strat_map is None:
            continue

        steep_name, flat_name = strat_map
        if steep_name is None or steep_name not in returns_matrix.columns:
            continue
        if flat_name is None or flat_name not in returns_matrix.columns:
            continue

        steep_ret = returns_matrix[steep_name]
        flat_ret = returns_matrix[flat_name]
        z_level = seg_signals[zlevel_col].reindex(returns_matrix.index)

        thresh = 1.0
        signal_flat = (z_level.shift(1) > thresh).astype(float)
        signal_steep = (z_level.shift(1) < -thresh).astype(float)
        daily_ret = signal_flat * flat_ret + signal_steep * steep_ret
        daily_ret = daily_ret.dropna()

        if len(daily_ret) < 252:
            continue

        invested = ((signal_flat + signal_steep) > 0)
        invested_days = invested.sum()
        if invested_days < 50:
            continue

        # Only compute on invested days
        active_ret = daily_ret[invested.reindex(daily_ret.index, fill_value=False)]

        metrics = compute_metrics(daily_ret)
        print(f"\n  {seg}:")
        print(f"    Sharpe:        {metrics['sharpe']:>+.3f}")
        print(f"    Ann Return:    {metrics['ann_ret']:>+.2%}")
        print(f"    Volatility:    {metrics['vol']:>.2%}")
        print(f"    Max DD:        {metrics['mdd']:>.2%}")
        print(f"    Hit Rate:      {metrics['hit_rate']:>.1%}")
        print(f"    Invested:      {invested_days} / {len(daily_ret)} days ({invested_days/len(daily_ret)*100:.1f}%)")

        # Breakdown: flattener vs steepener contribution
        flat_days = (signal_flat > 0).sum()
        steep_days = (signal_steep > 0).sum()
        flat_contrib = (signal_flat * flat_ret).dropna()
        steep_contrib = (signal_steep * steep_ret).dropna()
        print(f"    Flattener days:  {flat_days} (avg daily ret: {flat_contrib[flat_contrib != 0].mean()*10000:+.2f} bps)")
        print(f"    Steepener days:  {steep_days} (avg daily ret: {steep_contrib[steep_contrib != 0].mean()*10000:+.2f} bps)")

        # Yearly breakdown
        if invested_days > 100:
            print(f"    Year-by-year:")
            years = sorted(daily_ret.index.year.unique())
            for y in years:
                yr_mask = daily_ret.index.year == y
                yr_ret = daily_ret[yr_mask]
                if len(yr_ret) < 20:
                    continue
                eq = np.cumprod(1.0 + yr_ret.values)
                yr_total = eq[-1] - 1
                yr_invested = invested.reindex(yr_ret.index, fill_value=False).sum()
                print(f"      {y}: return={yr_total:>+6.2%}, invested={yr_invested:>3d} days")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 4: REGIME CONDITIONING (CALM vs STRESS)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 4: REGIME CONDITIONING - CALM vs STRESS")
    print("=" * 80)
    print("  Does z_level mean-reversion work better in CALM or STRESS regimes?")
    print("  CALM = VIX < median AND MOVE < median")
    print("  STRESS = VIX > 75th pctl OR MOVE > 75th pctl")
    print()

    # Define regime
    vix_aligned = vix.reindex(returns_matrix.index).ffill()
    move_aligned = move.reindex(returns_matrix.index).ffill()

    vix_median = vix_aligned.quantile(0.50)
    vix_75 = vix_aligned.quantile(0.75)
    move_median = move_aligned.quantile(0.50)
    move_75 = move_aligned.quantile(0.75)

    calm_mask = (vix_aligned < vix_median) & (move_aligned < move_median)
    stress_mask = (vix_aligned > vix_75) | (move_aligned > move_75)
    neutral_mask = ~calm_mask & ~stress_mask

    print(f"  VIX median={vix_median:.1f}, 75th={vix_75:.1f}")
    print(f"  MOVE median={move_median:.1f}, 75th={move_75:.1f}")
    print(f"  CALM days:    {calm_mask.sum():>5d} ({calm_mask.mean()*100:.1f}%)")
    print(f"  STRESS days:  {stress_mask.sum():>5d} ({stress_mask.mean()*100:.1f}%)")
    print(f"  NEUTRAL days: {neutral_mask.sum():>5d} ({neutral_mask.mean()*100:.1f}%)")

    print(f"\n  {'Segment':<10s}  {'CALM Sharpe':>12s}  {'STRESS Sharpe':>14s}  {'NEUTRAL Sharpe':>15s}  {'ALL Sharpe':>11s}")
    print("  " + "-" * 70)

    for seg in segments:
        zlevel_col = f"{seg}_z_level"
        strat_map = SEGMENT_STRAT_MAP.get(seg)
        if strat_map is None:
            continue

        steep_name, flat_name = strat_map
        if steep_name is None or steep_name not in returns_matrix.columns:
            continue
        if flat_name is None or flat_name not in returns_matrix.columns:
            continue

        steep_ret = returns_matrix[steep_name]
        flat_ret = returns_matrix[flat_name]
        z_level = seg_signals[zlevel_col].reindex(returns_matrix.index)

        thresh = 1.0
        signal_flat = (z_level.shift(1) > thresh).astype(float)
        signal_steep = (z_level.shift(1) < -thresh).astype(float)
        daily_ret = signal_flat * flat_ret + signal_steep * steep_ret
        daily_ret = daily_ret.dropna()

        # Overall
        metrics_all = compute_metrics(daily_ret)

        # CALM
        calm_ret = daily_ret[calm_mask.reindex(daily_ret.index, fill_value=False)]
        metrics_calm = compute_metrics(calm_ret) if len(calm_ret) > 50 else {"sharpe": np.nan}

        # STRESS
        stress_ret = daily_ret[stress_mask.reindex(daily_ret.index, fill_value=False)]
        metrics_stress = compute_metrics(stress_ret) if len(stress_ret) > 50 else {"sharpe": np.nan}

        # NEUTRAL
        neutral_ret = daily_ret[neutral_mask.reindex(daily_ret.index, fill_value=False)]
        metrics_neutral = compute_metrics(neutral_ret) if len(neutral_ret) > 50 else {"sharpe": np.nan}

        print(f"  {seg:<10s}  {metrics_calm['sharpe']:>+12.3f}  {metrics_stress['sharpe']:>+14.3f}  "
              f"{metrics_neutral['sharpe']:>+15.3f}  {metrics_all['sharpe']:>+11.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 4b: FORWARD CORRELATION BY REGIME
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "-" * 80)
    print("  ANALYSIS 4b: PREDICTIVE CORRELATION BY REGIME (63d forward)")
    print("-" * 80)
    print(f"\n  {'Segment':<10s}  {'CALM corr':>10s}  {'STRESS corr':>12s}  {'ALL corr':>10s}")
    print("  " + "-" * 50)

    for seg in segments:
        spread_col = f"{seg}_spread"
        zlevel_col = f"{seg}_z_level"

        if spread_col not in seg_signals.columns:
            continue

        spread = seg_signals[spread_col]
        z_level = seg_signals[zlevel_col]
        fwd_change = spread.diff(63).shift(-63)

        # Align to returns_matrix index
        df = pd.DataFrame({
            "z_level": z_level,
            "fwd": fwd_change,
            "calm": calm_mask.reindex(z_level.index, fill_value=False),
            "stress": stress_mask.reindex(z_level.index, fill_value=False),
        }).dropna(subset=["z_level", "fwd"])

        corr_all = df["z_level"].corr(df["fwd"])

        calm_df = df[df["calm"]]
        corr_calm = calm_df["z_level"].corr(calm_df["fwd"]) if len(calm_df) > 100 else np.nan

        stress_df = df[df["stress"]]
        corr_stress = stress_df["z_level"].corr(stress_df["fwd"]) if len(stress_df) > 100 else np.nan

        print(f"  {seg:<10s}  {corr_calm:>+10.4f}  {corr_stress:>+12.4f}  {corr_all:>+10.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 5: YC REGIME INTEGRATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 5: YC REGIME INTEGRATION")
    print("=" * 80)
    print("  How does contrarian z_level perform in each YC regime?")
    print("  (BULL_STEEP, BULL_FLAT, BEAR_STEEP, BEAR_FLAT)")
    print()

    yc_regime = compute_yc_regime(yields, short_col="US_2Y", long_col="US_10Y", window=21)
    yc_regime_aligned = yc_regime.reindex(returns_matrix.index).ffill()

    regime_labels = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]

    print(f"  Regime distribution:")
    for r in regime_labels:
        n = (yc_regime_aligned == r).sum()
        print(f"    {r:<15s}: {n:>5d} days ({n/len(yc_regime_aligned)*100:.1f}%)")

    print(f"\n  {'Segment':<10s}", end="")
    for r in regime_labels:
        print(f"  {r:>14s}", end="")
    print()
    print("  " + "-" * (10 + len(regime_labels) * 16))

    for seg in segments:
        zlevel_col = f"{seg}_z_level"
        strat_map = SEGMENT_STRAT_MAP.get(seg)
        if strat_map is None:
            continue

        steep_name, flat_name = strat_map
        if steep_name is None or steep_name not in returns_matrix.columns:
            continue
        if flat_name is None or flat_name not in returns_matrix.columns:
            continue

        steep_ret = returns_matrix[steep_name]
        flat_ret = returns_matrix[flat_name]
        z_level = seg_signals[zlevel_col].reindex(returns_matrix.index)

        thresh = 1.0
        signal_flat = (z_level.shift(1) > thresh).astype(float)
        signal_steep = (z_level.shift(1) < -thresh).astype(float)
        daily_ret = signal_flat * flat_ret + signal_steep * steep_ret
        daily_ret = daily_ret.dropna()

        print(f"  {seg:<10s}", end="")

        for r in regime_labels:
            regime_mask = (yc_regime_aligned == r).reindex(daily_ret.index, fill_value=False)
            regime_ret = daily_ret[regime_mask]
            if len(regime_ret) > 30:
                m = compute_metrics(regime_ret)
                print(f"  {m['sharpe']:>+14.3f}", end="")
            else:
                print(f"  {'N/A':>14s}", end="")

        print()

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 6: COMBINED SIGNAL (z_level contrarian + regime filter)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 6: COMBINED CONTRARIAN + REGIME FILTER")
    print("=" * 80)
    print("  Test: Only trade z_level contrarian when in CALM regime")
    print()

    print(f"  {'Segment':<10s}  {'Unfiltered':>11s}  {'CALM only':>11s}  {'Improvement':>12s}")
    print("  " + "-" * 52)

    for seg in segments:
        zlevel_col = f"{seg}_z_level"
        strat_map = SEGMENT_STRAT_MAP.get(seg)
        if strat_map is None:
            continue

        steep_name, flat_name = strat_map
        if steep_name is None or steep_name not in returns_matrix.columns:
            continue
        if flat_name is None or flat_name not in returns_matrix.columns:
            continue

        steep_ret = returns_matrix[steep_name]
        flat_ret = returns_matrix[flat_name]
        z_level = seg_signals[zlevel_col].reindex(returns_matrix.index)

        thresh = 1.0
        signal_flat = (z_level.shift(1) > thresh).astype(float)
        signal_steep = (z_level.shift(1) < -thresh).astype(float)

        # Unfiltered
        daily_ret_unf = (signal_flat * flat_ret + signal_steep * steep_ret).dropna()
        metrics_unf = compute_metrics(daily_ret_unf)

        # CALM filtered
        calm_filter = calm_mask.reindex(returns_matrix.index, fill_value=False).astype(float)
        daily_ret_calm = (signal_flat * flat_ret * calm_filter +
                          signal_steep * steep_ret * calm_filter).dropna()
        metrics_calm = compute_metrics(daily_ret_calm)

        improvement = metrics_calm["sharpe"] - metrics_unf["sharpe"]
        print(f"  {seg:<10s}  {metrics_unf['sharpe']:>+11.3f}  {metrics_calm['sharpe']:>+11.3f}  "
              f"{improvement:>+12.3f}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 7: MEAN-REVERSION HALF-LIFE ESTIMATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 7: MEAN-REVERSION SPEED (HALF-LIFE)")
    print("=" * 80)
    print("  AR(1) regression: z_level(t+1) = alpha + beta * z_level(t) + eps")
    print("  Half-life = -ln(2) / ln(beta)")
    print()

    print(f"  {'Segment':<10s}  {'beta':>8s}  {'half_life(days)':>16s}  {'R2':>8s}  {'autocorr':>10s}")
    print("  " + "-" * 60)

    for seg in segments:
        zlevel_col = f"{seg}_z_level"
        if zlevel_col not in seg_signals.columns:
            continue

        z = seg_signals[zlevel_col].dropna()
        if len(z) < 100:
            continue

        z_t = z.iloc[:-1].values
        z_t1 = z.iloc[1:].values

        # AR(1) regression
        n = len(z_t)
        X = np.column_stack([np.ones(n), z_t])
        beta_hat = np.linalg.lstsq(X, z_t1, rcond=None)[0]
        beta = beta_hat[1]

        # Half-life
        if 0 < beta < 1:
            half_life = -np.log(2) / np.log(beta)
        else:
            half_life = np.nan

        # R-squared
        z_pred = X @ beta_hat
        ss_res = np.sum((z_t1 - z_pred) ** 2)
        ss_tot = np.sum((z_t1 - z_t1.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Autocorrelation at lag 1
        autocorr = np.corrcoef(z_t, z_t1)[0, 1]

        print(f"  {seg:<10s}  {beta:>8.4f}  {half_life:>16.1f}  {r2:>8.4f}  {autocorr:>10.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # ANALYSIS 8: EXTREME Z-LEVEL EVENTS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  ANALYSIS 8: EXTREME z_level EVENTS -> FORWARD PERFORMANCE")
    print("=" * 80)
    print("  When z_level crosses +/- 2, what happens next?")
    print()

    for seg in segments:
        spread_col = f"{seg}_spread"
        zlevel_col = f"{seg}_z_level"

        if spread_col not in seg_signals.columns:
            continue

        spread = seg_signals[spread_col]
        z_level = seg_signals[zlevel_col]

        # Find extreme events (z > 2 or z < -2)
        extreme_high = z_level > 2.0
        extreme_low = z_level < -2.0

        # Compute forward spread change at various horizons
        print(f"\n  --- {seg} ---")
        for label, mask in [("z_level > +2 (too steep)", extreme_high),
                             ("z_level < -2 (too flat)", extreme_low)]:
            n_events = mask.sum()
            if n_events < 10:
                print(f"    {label}: only {n_events} events, skipping")
                continue

            print(f"    {label}: {n_events} days")
            for h in [21, 63, 126]:
                fwd = spread.diff(h).shift(-h)
                df = pd.DataFrame({"z": z_level, "fwd": fwd}).dropna()
                df_filtered = df[mask.reindex(df.index, fill_value=False)]

                if len(df_filtered) < 5:
                    continue

                avg_fwd = df_filtered["fwd"].mean() * 100  # bps
                std_fwd = df_filtered["fwd"].std() * 100
                t_stat = avg_fwd / (std_fwd / np.sqrt(len(df_filtered))) if std_fwd > 0 else 0
                hit = (df_filtered["fwd"] < 0).mean() if "high" in label else (df_filtered["fwd"] > 0).mean()

                direction = "FLATTENED" if "high" in label and avg_fwd < 0 else \
                            "STEEPENED" if "low" in label and avg_fwd > 0 else "NO REVERSION"

                print(f"      {h:>3d}d fwd: avg={avg_fwd:>+6.1f} bps, t={t_stat:>+5.2f}, "
                      f"hit={hit:.0%}, -> {direction}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("  SUMMARY & CONCLUSIONS")
    print("=" * 80)
    print("""
  Key findings to look for:

  1. PREDICTIVE POWER:
     - Negative correlation of z_level(t) with forward spread change
       confirms mean-reversion property (contrarian signal)
     - Compare magnitude vs z_mom correlation

  2. CONTRARIAN STRATEGY SHARPE:
     - Positive Sharpe for contrarian rule validates the approach
     - Compare across thresholds (0.5, 1.0, 1.5, 2.0)

  3. REGIME CONDITIONING:
     - If CALM Sharpe > STRESS Sharpe, mean-reversion works better
       when volatility is low (markets are orderly)
     - STRESS may cause trends/dislocations that persist

  4. YC REGIME:
     - Contrarian may work better in BULL_FLAT or BEAR_STEEP
       (these regimes have natural mean-reversion tendencies)

  5. HALF-LIFE:
     - Short half-life (< 60 days) = fast mean reversion, trade shorter horizon
     - Long half-life (> 120 days) = slow mean reversion, need patience

  6. DESIGN IMPLICATIONS:
     - z_level contrarian + CALM regime filter could be a strong system
     - Can layer on top of existing momentum system as diversifier
     - Optimal threshold likely around |z| > 1.0-1.5
    """)


if __name__ == "__main__":
    main()
