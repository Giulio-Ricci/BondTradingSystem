"""
Diagnostic: Can we build a signal that correctly goes SHORT during the 2013
taper tantrum while maintaining good overall performance?

Tests:
1. Higher k_yc (1.0, 2.0, 3.0) - amplify the YC bearish signal
2. TLT momentum component - trend-following to balance contrarian RR
3. Yield trend signal - 10Y yield above/below moving average
4. Inverted z_rr during rate rising regimes
5. Different z-score windows (very short: 21d)
6. Signal combinations

For each: show z_final during 2013, check if threshold crossings happen,
and compute full-period Sharpe.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime
from versione_presentazione import (
    build_rr_blend, build_slope_zscore, build_rr_multitf,
    build_yc_signal, fast_bt, ANN
)

def quick_bt(z_arr, reg_arr, r_long, r_short, year_arr,
             tl_c=-3.0, ts_c=2.5, tl_s=-3.5):
    """Quick backtest, return sharpe + yearly excess."""
    m = fast_bt(z_arr, reg_arr, r_long, r_short, year_arr,
                tl_c, ts_c, tl_s)
    return m


def analyze_signal(name, z_series, dates, common, reg_arr, r_long, r_short,
                   year_arr, tl_c=-3.0, ts_c=2.5, tl_s=-3.5):
    """Analyze a signal: 2013 behavior + full-period metrics."""
    z_arr = z_series.reindex(common).values

    # 2013 stats
    mask_2013 = np.array([d.year == 2013 for d in common])
    z_2013 = z_arr[mask_2013]
    z_2013_valid = z_2013[~np.isnan(z_2013)]

    # Full period backtest with multiple thresholds
    results = {}
    for tl in [-3.0, -2.0, -1.5, -1.0]:
        for ts in [2.5, 2.0, 1.5, 1.0]:
            m = fast_bt(z_arr, reg_arr, r_long, r_short, year_arr,
                        tl, ts, tl_s)
            results[(tl, ts)] = m

    # Best config
    best_key = max(results.keys(), key=lambda k: results[k]["sharpe"])
    best = results[best_key]

    # 2013 yearly excess for best config
    exc_2013 = 0.0
    for i, y in enumerate(best["yearly_years"]):
        if y == 2013:
            exc_2013 = best["yearly_excess"][i]
            break

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  2013 z-score: mean={np.nanmean(z_2013):+.3f}  "
          f"min={np.nanmin(z_2013):+.3f}  max={np.nanmax(z_2013):+.3f}")
    print(f"  2013 days z<0: {np.sum(z_2013_valid < 0)}/{len(z_2013_valid)}  "
          f"z<-1: {np.sum(z_2013_valid < -1)}  "
          f"z<-2: {np.sum(z_2013_valid < -2)}  "
          f"z<-3: {np.sum(z_2013_valid < -3)}")

    print(f"\n  Best config: tl={best_key[0]:.1f}, ts={best_key[1]:.1f}")
    print(f"  Full-period Sharpe: {best['sharpe']:.4f}")
    print(f"  Dead years: {best['n_dead']}  Win rate: {best['win_rate']:.0%}")
    print(f"  2013 excess: {exc_2013:+.4f} ({exc_2013*100:+.2f}%)")

    # Monthly 2013 breakdown
    print(f"\n  Monthly 2013 z-score:")
    for month in range(1, 13):
        mask_m = np.array([(d.year == 2013 and d.month == month) for d in common])
        zm = z_arr[mask_m]
        zm_v = zm[~np.isnan(zm)]
        if len(zm_v) > 0:
            print(f"    {month:02d}/2013: mean={np.mean(zm_v):+.3f}  "
                  f"min={np.min(zm_v):+.3f}  max={np.max(zm_v):+.3f}")

    return best, z_arr


def main():
    print("=" * 70)
    print("  DIAGNOSTIC: Can we fix the 2013 taper tantrum problem?")
    print("=" * 70)

    # Load data
    D = load_all()
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])

    # Common dates
    bt_start = pd.Timestamp(BACKTEST_START)
    etf_ret = D["etf_ret"]
    yields = D["yields"]

    # Base signals
    rr_blend = build_rr_blend(D)
    z_slope = build_slope_zscore(D)
    z_rr_63, _, _ = build_rr_multitf(D, [63])
    z_rr_mtf, _, _ = build_rr_multitf(D, [63, 252])
    z_yc_42, _, _ = build_yc_signal(D, change_window=42)
    z_yc_84, _, _ = build_yc_signal(D, change_window=84)

    # TLT price and returns
    tlt_px = D["etf_px"]["TLT"]
    tlt_ret = etf_ret["TLT"]

    # Align
    all_idx = [z_rr_63.dropna().index, z_yc_42.dropna().index,
               regime_hmm.dropna().index,
               etf_ret.dropna(subset=["TLT", "SHV"]).index]
    common = all_idx[0]
    for idx in all_idx[1:]:
        common = common.intersection(idx)
    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])

    # ================================================================
    # TEST 0: Current V2 baseline
    # ================================================================
    z_v2 = z_rr_63 + 0.40 * z_yc_42
    analyze_signal("TEST 0: V2 baseline (z_rr_63 + 0.40*z_yc_42)",
                   z_v2, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 1: V_Presentazione current best
    # ================================================================
    z_vpres = z_rr_mtf + 0.20 * z_yc_84
    analyze_signal("TEST 1: V_Pres (z_rr_mtf[63,252] + 0.20*z_yc_84)",
                   z_vpres, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 2: Much higher k_yc (amplify YC)
    # ================================================================
    for k in [1.0, 2.0, 3.0]:
        z_test = z_rr_mtf + k * z_yc_84
        analyze_signal(f"TEST 2: High k_yc={k:.1f} (z_rr_mtf + {k}*z_yc_84)",
                       z_test, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 3: TLT Momentum z-score (trend-following component)
    # ================================================================
    for mom_w in [63, 126, 252]:
        tlt_mom = tlt_ret.rolling(mom_w).sum()  # cumulative return over window
        mu = tlt_mom.rolling(252, min_periods=63).mean()
        sd = tlt_mom.rolling(252, min_periods=63).std().replace(0, np.nan)
        z_tlt_mom = (tlt_mom - mu) / sd

        # Add to base signal
        for k_mom in [0.5, 1.0, 1.5]:
            z_test = z_rr_mtf + 0.20 * z_yc_84 + k_mom * z_tlt_mom
            analyze_signal(
                f"TEST 3: +TLT momentum (mom_w={mom_w}, k={k_mom})",
                z_test, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 4: Yield trend signal (10Y above/below SMA)
    # ================================================================
    y10 = yields["US_10Y"]
    for sma_w in [126, 200, 252]:
        y10_sma = y10.rolling(sma_w, min_periods=63).mean()
        y10_dev = (y10 - y10_sma)  # positive = yield above SMA (bearish for bonds)
        # Z-score the deviation
        mu = y10_dev.rolling(252, min_periods=63).mean()
        sd = y10_dev.rolling(252, min_periods=63).std().replace(0, np.nan)
        z_yield_trend = (y10_dev - mu) / sd

        # Subtract from signal (bearish yield trend reduces z_final)
        for k_yt in [0.5, 1.0, 1.5]:
            z_test = z_rr_mtf + 0.20 * z_yc_84 - k_yt * z_yield_trend
            analyze_signal(
                f"TEST 4: -Yield trend (sma={sma_w}, k={k_yt})",
                z_test, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 5: Pure YC signal (no RR at all)
    # ================================================================
    for cw in [42, 63, 84, 126]:
        z_yc_test, _, _ = build_yc_signal(D, change_window=cw)
        analyze_signal(f"TEST 5: Pure z_yc (cw={cw})",
                       z_yc_test, common, common, reg_arr, r_long, r_short,
                       year_arr, tl_c=-1.5, ts_c=1.0, tl_s=-2.0)

    # ================================================================
    # TEST 6: Very short z-score window (reactive)
    # ================================================================
    for w in [21, 42]:
        mu = rr_blend.rolling(w, min_periods=10).mean()
        sd = rr_blend.rolling(w, min_periods=10).std().replace(0, np.nan)
        z_short = (rr_blend - mu) / sd + 0.50 * z_slope
        z_test = z_short + 0.40 * z_yc_42
        analyze_signal(f"TEST 6: Short z-window (w={w})",
                       z_test, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 7: Negative z_rr (invert contrarian → trend-following)
    # ================================================================
    z_test = -z_rr_63 + 0.40 * z_yc_42
    analyze_signal("TEST 7: INVERTED z_rr (-z_rr_63 + 0.40*z_yc_42)",
                   z_test, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 8: RR blend rate of change (not z-score, raw momentum)
    # ================================================================
    for mom_w in [21, 42, 63]:
        rr_roc = rr_blend.diff(mom_w)
        mu = rr_roc.rolling(252, min_periods=63).mean()
        sd = rr_roc.rolling(252, min_periods=63).std().replace(0, np.nan)
        z_rr_roc = (rr_roc - mu) / sd
        z_test = z_rr_roc + 0.40 * z_yc_84
        analyze_signal(f"TEST 8: RR rate-of-change (mom={mom_w}d)",
                       z_test, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 9: TLT SMA filter (binary: above/below 200d SMA)
    # ================================================================
    for sma_w in [126, 200]:
        tlt_sma = tlt_px.rolling(sma_w, min_periods=63).mean()
        tlt_above = (tlt_px > tlt_sma).astype(float)  # 1 when above, 0 below
        # Use as multiplicative filter on z_final
        z_base = z_rr_mtf + 0.20 * z_yc_84
        # When TLT below SMA: z_final is capped at 0 (can't go LONG)
        z_filtered = z_base.copy()
        below_mask = tlt_above.reindex(z_filtered.index) < 0.5
        z_filtered[below_mask] = z_filtered[below_mask].clip(upper=0)
        analyze_signal(f"TEST 9: TLT SMA filter (sma={sma_w}d, cap z at 0 when below)",
                       z_filtered, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # TEST 10: MOVE z-score as additional signal
    # ================================================================
    move = D["move"]
    for w in [63, 126]:
        mu = move.rolling(w, min_periods=20).mean()
        sd = move.rolling(w, min_periods=20).std().replace(0, np.nan)
        z_move = (move - mu) / sd
        # High MOVE = bearish for bonds (risk-off but also rate vol)
        for k_mv in [0.3, 0.5, 1.0]:
            z_test = z_rr_mtf + 0.20 * z_yc_84 - k_mv * z_move
            analyze_signal(
                f"TEST 10: -MOVE zscore (w={w}, k={k_mv})",
                z_test, common, common, reg_arr, r_long, r_short, year_arr)

    # ================================================================
    # SUMMARY: Print top 5 approaches by Sharpe with 2013 excess != 0
    # ================================================================
    print("\n\n" + "=" * 70)
    print("  DONE - Review results above to find signals that fix 2013")
    print("=" * 70)


if __name__ == "__main__":
    main()
