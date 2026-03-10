"""
Diagnostic: RR Term Structure Spreads as Signals.

Tests whether the spread between different RR tenors (1W, 1M, 3M, 6M)
contains useful information beyond the simple blend average.

Idea: The "term structure slope" of risk reversals may capture
changes in market expectations at different horizons.

Tests all 6 pairwise spreads:
  1W-1M, 1W-3M, 1W-6M, 1M-3M, 1M-6M, 3M-6M

For each:
  A) Z-score the spread → test standalone
  B) Add to current best signal → test improvement
  C) Try multiple z-score windows
"""

import numpy as np
import pandas as pd
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RR_TENORS, BACKTEST_START, RISK_FREE_RATE, SAVE_DIR
from data_loader import load_all
from regime import fit_hmm_regime
from versione_presentazione import (
    build_rr_blend, build_slope_zscore, build_rr_multitf,
    build_yc_signal, build_combined, fast_bt, composite_score, ANN
)

import os
OUT_DIR = os.path.join(SAVE_DIR, "rr_term_structure")
os.makedirs(OUT_DIR, exist_ok=True)


def build_rr_series(D):
    """Build individual RR series for each tenor."""
    options = D["options"]
    rr_dict = {}
    for tenor in ["1W", "1M", "3M", "6M"]:
        put_col, call_col = RR_TENORS[tenor]
        rr = (pd.to_numeric(options[put_col], errors="coerce")
              - pd.to_numeric(options[call_col], errors="coerce"))
        rr_dict[tenor] = rr
    return rr_dict


def build_spread_zscore(rr_a, rr_b, z_window=126, z_min=30):
    """Z-score of (rr_a - rr_b) spread."""
    spread = rr_a - rr_b
    mu = spread.rolling(z_window, min_periods=z_min).mean()
    sd = spread.rolling(z_window, min_periods=z_min).std().replace(0, np.nan)
    return (spread - mu) / sd, spread


def main():
    t0 = time.time()
    print("=" * 70)
    print("  RR TERM STRUCTURE DIAGNOSTIC")
    print("  Testing pairwise RR tenor spreads as signals")
    print("=" * 70)

    D = load_all()
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])
    etf_ret = D["etf_ret"]

    # Build individual RR series
    rr_dict = build_rr_series(D)
    tenors = ["1W", "1M", "3M", "6M"]

    # Print basic stats
    print("\n  RR tenor correlations:")
    rr_df = pd.DataFrame(rr_dict).dropna()
    print(rr_df.corr().round(3).to_string())

    # Build all pairwise spreads
    pairs = []
    for i in range(len(tenors)):
        for j in range(i + 1, len(tenors)):
            pairs.append((tenors[i], tenors[j]))

    print(f"\n  Testing {len(pairs)} pairs: {pairs}")

    # Current best config
    best_cfg = {
        "z_windows": [63, 252],
        "k_slope": 0.75,
        "yc_cw": 126,
        "k_yc": 0.20,
        "tl_c": -3.375,
        "ts_c": 3.00,
        "tl_s": -2.875,
        "adapt_days": 63,
        "adapt_raise": 1.0,
        "max_hold_days": 252,
    }

    # Build current best signal
    z_rr_best, _, _ = build_rr_multitf(D, best_cfg["z_windows"],
                                        k_slope=best_cfg["k_slope"])
    z_yc_best, _, _ = build_yc_signal(D, change_window=best_cfg["yc_cw"])
    z_base_signal = build_combined(z_rr_best, z_yc_best, best_cfg["k_yc"])

    # Align
    bt_start = pd.Timestamp(BACKTEST_START)
    common = (z_base_signal.dropna().index
              .intersection(regime_hmm.dropna().index)
              .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index))

    # Also need all RR tenors available
    for tenor in tenors:
        common = common.intersection(rr_dict[tenor].dropna().index)

    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])

    print(f"  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d} ({len(common)} days)")

    # Baseline (current best, no term structure)
    z_base_arr = z_base_signal.reindex(common).values
    m_baseline = fast_bt(z_base_arr, reg_arr, r_long, r_short, year_arr,
                         best_cfg["tl_c"], best_cfg["ts_c"], best_cfg["tl_s"],
                         adapt_days=best_cfg["adapt_days"],
                         adapt_raise=best_cfg["adapt_raise"],
                         max_hold_days=best_cfg["max_hold_days"])

    print(f"\n  BASELINE (current best): Sharpe={m_baseline['sharpe']:.4f}  "
          f"dead={m_baseline['n_dead']}  wr={m_baseline['win_rate']:.0%}")

    # ================================================================
    # TEST A: Each spread z-score as STANDALONE signal
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST A: Standalone spread signals")
    print("=" * 70)

    z_windows_test = [42, 63, 126, 252]
    standalone_results = []

    print(f"\n  {'Pair':<10} {'z_w':>4} {'Sharpe':>7} {'Dead':>5} {'WR':>5} "
          f"{'Ann%':>6} {'Events':>7}")
    print("  " + "-" * 50)

    for ta, tb in pairs:
        for z_w in z_windows_test:
            z_spread, raw_spread = build_spread_zscore(
                rr_dict[ta], rr_dict[tb], z_window=z_w)
            z_sp_arr = z_spread.reindex(common).values

            # Test with multiple thresholds
            best_sh = -999
            best_m = None
            for tl in [-3.0, -2.5, -2.0, -1.5]:
                for ts in [1.5, 2.0, 2.5, 3.0]:
                    m = fast_bt(z_sp_arr, reg_arr, r_long, r_short,
                                year_arr, tl, ts, tl - 0.5)
                    if m["sharpe"] > best_sh and m["n_events"] >= 6:
                        best_sh = m["sharpe"]
                        best_m = m

            if best_m is not None:
                standalone_results.append({
                    "pair": f"{ta}-{tb}",
                    "z_window": z_w,
                    "sharpe": best_m["sharpe"],
                    "n_dead": best_m["n_dead"],
                    "win_rate": best_m["win_rate"],
                    "ann_ret": best_m["ann_ret"],
                    "n_events": best_m["n_events"],
                })
                print(f"  {ta}-{tb:<7} {z_w:>4} {best_m['sharpe']:>7.3f} "
                      f"{best_m['n_dead']:>5} {best_m['win_rate']:>5.0%} "
                      f"{best_m['ann_ret']*100:>5.1f}% {best_m['n_events']:>7}")

    # Best standalone
    if standalone_results:
        df_sa = pd.DataFrame(standalone_results)
        top_sa = df_sa.nlargest(5, "sharpe")
        print(f"\n  Top 5 standalone:")
        for _, r in top_sa.iterrows():
            print(f"    {r['pair']:<10} z_w={r['z_window']:>3}  "
                  f"Sh={r['sharpe']:.3f}  dead={r['n_dead']:.0f}  "
                  f"wr={r['win_rate']:.0%}")

    # ================================================================
    # TEST B: Add spread z-score to current best signal
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST B: Spread added to current best signal")
    print(f"  z_final = z_current + k_spread * z_spread")
    print("=" * 70)

    additive_results = []

    k_spread_list = [0.10, 0.20, 0.30, 0.50, 0.75, 1.00]

    print(f"\n  {'Pair':<10} {'z_w':>4} {'k':>5} {'Sharpe':>7} {'dSh':>6} "
          f"{'Dead':>5} {'WR':>5} {'Events':>7}")
    print("  " + "-" * 58)

    for ta, tb in pairs:
        for z_w in z_windows_test:
            z_spread, _ = build_spread_zscore(
                rr_dict[ta], rr_dict[tb], z_window=z_w)
            z_sp_arr = z_spread.reindex(common).values

            for k_sp in k_spread_list:
                # Try both signs: + and -
                for sign in [+1, -1]:
                    z_test = z_base_arr + sign * k_sp * z_sp_arr

                    m = fast_bt(z_test, reg_arr, r_long, r_short, year_arr,
                                best_cfg["tl_c"], best_cfg["ts_c"],
                                best_cfg["tl_s"],
                                adapt_days=best_cfg["adapt_days"],
                                adapt_raise=best_cfg["adapt_raise"],
                                max_hold_days=best_cfg["max_hold_days"])

                    delta_sh = m["sharpe"] - m_baseline["sharpe"]

                    additive_results.append({
                        "pair": f"{ta}-{tb}",
                        "z_window": z_w,
                        "k_spread": sign * k_sp,
                        "sharpe": m["sharpe"],
                        "delta_sharpe": delta_sh,
                        "n_dead": m["n_dead"],
                        "win_rate": m["win_rate"],
                        "n_events": m["n_events"],
                    })

    df_add = pd.DataFrame(additive_results)
    # Show only improvements
    df_better = df_add[df_add["delta_sharpe"] > 0.005].sort_values(
        "sharpe", ascending=False)

    if len(df_better) > 0:
        print(f"\n  Configs that IMPROVE on baseline ({len(df_better)} found):")
        print(f"\n  {'Pair':<10} {'z_w':>4} {'k':>6} {'Sharpe':>7} {'dSh':>7} "
              f"{'Dead':>5} {'WR':>5}")
        print("  " + "-" * 50)
        for _, r in df_better.head(25).iterrows():
            print(f"  {r['pair']:<10} {r['z_window']:>4} {r['k_spread']:>+6.2f} "
                  f"{r['sharpe']:>7.3f} {r['delta_sharpe']:>+7.3f} "
                  f"{r['n_dead']:>5.0f} {r['win_rate']:>5.0%}")
    else:
        print("\n  No config improves on baseline!")

    # Show worst degradations too
    df_worst = df_add.nsmallest(10, "delta_sharpe")
    print(f"\n  Worst degradations:")
    for _, r in df_worst.iterrows():
        print(f"    {r['pair']:<10} z_w={r['z_window']:>3} k={r['k_spread']:>+5.2f}  "
              f"Sh={r['sharpe']:.3f}  dSh={r['delta_sharpe']:+.3f}")

    # ================================================================
    # TEST C: Full grid — spread + re-optimize thresholds
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST C: Spread + threshold re-optimization")
    print("  (top 5 spreads from Test B, full threshold grid)")
    print("=" * 70)

    # Take top 5 additive configs
    top5_add = df_add.nlargest(5, "sharpe")

    tl_c_grid = [-3.75, -3.50, -3.375, -3.25, -3.00, -2.75]
    ts_c_grid = [2.50, 2.75, 3.00, 3.25, 3.50]
    tl_s_grid = [-3.50, -3.00, -2.875, -2.75]

    reopt_results = []
    for _, top_r in top5_add.iterrows():
        pair = top_r["pair"]
        ta, tb = pair.split("-")
        z_w = int(top_r["z_window"])
        k_sp = top_r["k_spread"]

        z_spread, _ = build_spread_zscore(
            rr_dict[ta], rr_dict[tb], z_window=z_w)
        z_sp_arr = z_spread.reindex(common).values
        z_test = z_base_arr + k_sp * z_sp_arr

        best_sh = -999
        best_m = None
        best_thresholds = None

        for tl_c in tl_c_grid:
            for ts_c in ts_c_grid:
                for tl_s in tl_s_grid:
                    for ad in [0, 63]:
                        for mh in [0, 252]:
                            ar = 1.0 if ad > 0 else 0.0
                            m = fast_bt(z_test, reg_arr, r_long, r_short,
                                        year_arr, tl_c, ts_c, tl_s,
                                        adapt_days=ad, adapt_raise=ar,
                                        max_hold_days=mh)
                            sc = composite_score(m)
                            if sc > best_sh and m["n_events"] >= 6:
                                best_sh = sc
                                best_m = m
                                best_thresholds = (tl_c, ts_c, tl_s, ad, mh)

        if best_m:
            tl_c, ts_c, tl_s, ad, mh = best_thresholds
            delta = best_m["sharpe"] - m_baseline["sharpe"]
            reopt_results.append({
                "pair": pair, "z_window": z_w, "k_spread": k_sp,
                "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s,
                "adapt_days": ad, "max_hold": mh,
                "sharpe": best_m["sharpe"],
                "delta": delta,
                "n_dead": best_m["n_dead"],
                "win_rate": best_m["win_rate"],
                "n_events": best_m["n_events"],
            })
            print(f"  {pair:<10} z_w={z_w:>3} k={k_sp:>+5.2f}  "
                  f"Sh={best_m['sharpe']:.3f} (dSh={delta:+.3f})  "
                  f"dead={best_m['n_dead']}  wr={best_m['win_rate']:.0%}  "
                  f"[{tl_c},{ts_c},{tl_s}] ad={ad} mh={mh}")

    # ================================================================
    # TEST D: Spread as regime filter (not additive)
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST D: Spread as regime FILTER")
    print("  (block LONG when short-term RR >> long-term RR)")
    print("=" * 70)

    filter_results = []
    for ta, tb in pairs:
        for z_w in [42, 63, 126]:
            z_spread, raw_spread = build_spread_zscore(
                rr_dict[ta], rr_dict[tb], z_window=z_w)
            z_sp_arr = z_spread.reindex(common).values

            for thresh in [-1.5, -1.0, -0.5, 0.5, 1.0, 1.5]:
                # Filter: when spread z-score exceeds threshold,
                # cap z_final to prevent LONG
                z_filtered = z_base_arr.copy()
                if thresh > 0:
                    # Block LONG when spread is high
                    mask = z_sp_arr > thresh
                    z_filtered[mask & ~np.isnan(z_sp_arr)] = np.minimum(
                        z_filtered[mask & ~np.isnan(z_sp_arr)], 0)
                else:
                    # Block LONG when spread is low
                    mask = z_sp_arr < thresh
                    z_filtered[mask & ~np.isnan(z_sp_arr)] = np.minimum(
                        z_filtered[mask & ~np.isnan(z_sp_arr)], 0)

                m = fast_bt(z_filtered, reg_arr, r_long, r_short, year_arr,
                            best_cfg["tl_c"], best_cfg["ts_c"],
                            best_cfg["tl_s"],
                            adapt_days=best_cfg["adapt_days"],
                            adapt_raise=best_cfg["adapt_raise"],
                            max_hold_days=best_cfg["max_hold_days"])

                delta = m["sharpe"] - m_baseline["sharpe"]
                filter_results.append({
                    "pair": f"{ta}-{tb}", "z_window": z_w,
                    "filter_thresh": thresh,
                    "sharpe": m["sharpe"], "delta": delta,
                    "n_dead": m["n_dead"], "win_rate": m["win_rate"],
                })

    df_filt = pd.DataFrame(filter_results)
    df_filt_better = df_filt[df_filt["delta"] > 0.005].sort_values(
        "sharpe", ascending=False)

    if len(df_filt_better) > 0:
        print(f"\n  Filter configs that improve ({len(df_filt_better)}):")
        print(f"  {'Pair':<10} {'z_w':>4} {'thresh':>7} {'Sharpe':>7} "
              f"{'dSh':>7} {'Dead':>5} {'WR':>5}")
        print("  " + "-" * 50)
        for _, r in df_filt_better.head(15).iterrows():
            print(f"  {r['pair']:<10} {r['z_window']:>4} {r['filter_thresh']:>+7.1f} "
                  f"{r['sharpe']:>7.3f} {r['delta']:>+7.3f} "
                  f"{r['n_dead']:>5.0f} {r['win_rate']:>5.0%}")
    else:
        print("\n  No filter improves on baseline!")

    # ================================================================
    # TEST E: Raw spread level (not z-scored) — trend signal
    # ================================================================
    print("\n" + "=" * 70)
    print("  TEST E: Raw spread momentum (change over N days)")
    print("=" * 70)

    momentum_results = []
    for ta, tb in pairs:
        spread = rr_dict[ta] - rr_dict[tb]
        for mom_w in [21, 42, 63]:
            spread_mom = spread.diff(mom_w)
            mu = spread_mom.rolling(252, min_periods=63).mean()
            sd = spread_mom.rolling(252, min_periods=63).std().replace(0, np.nan)
            z_mom = (spread_mom - mu) / sd

            z_mom_arr = z_mom.reindex(common).values

            for k_sp in [0.20, 0.50, 1.00]:
                for sign in [+1, -1]:
                    z_test = z_base_arr + sign * k_sp * z_mom_arr

                    m = fast_bt(z_test, reg_arr, r_long, r_short, year_arr,
                                best_cfg["tl_c"], best_cfg["ts_c"],
                                best_cfg["tl_s"],
                                adapt_days=best_cfg["adapt_days"],
                                adapt_raise=best_cfg["adapt_raise"],
                                max_hold_days=best_cfg["max_hold_days"])

                    delta = m["sharpe"] - m_baseline["sharpe"]
                    momentum_results.append({
                        "pair": f"{ta}-{tb}", "mom_window": mom_w,
                        "k": sign * k_sp,
                        "sharpe": m["sharpe"], "delta": delta,
                        "n_dead": m["n_dead"], "win_rate": m["win_rate"],
                    })

    df_mom = pd.DataFrame(momentum_results)
    df_mom_better = df_mom[df_mom["delta"] > 0.005].sort_values(
        "sharpe", ascending=False)

    if len(df_mom_better) > 0:
        print(f"\n  Momentum configs that improve ({len(df_mom_better)}):")
        print(f"  {'Pair':<10} {'mom_w':>6} {'k':>6} {'Sharpe':>7} "
              f"{'dSh':>7} {'Dead':>5} {'WR':>5}")
        print("  " + "-" * 50)
        for _, r in df_mom_better.head(15).iterrows():
            print(f"  {r['pair']:<10} {r['mom_window']:>6} {r['k']:>+6.2f} "
                  f"{r['sharpe']:>7.3f} {r['delta']:>+7.3f} "
                  f"{r['n_dead']:>5.0f} {r['win_rate']:>5.0%}")
    else:
        print("\n  No momentum spread improves on baseline!")

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Baseline Sharpe: {m_baseline['sharpe']:.4f}")

    # Collect all improvements
    all_improvements = []

    if len(df_better) > 0:
        best_add = df_better.iloc[0]
        all_improvements.append(
            (f"Additive: {best_add['pair']} z_w={best_add['z_window']:.0f} "
             f"k={best_add['k_spread']:+.2f}",
             best_add["sharpe"], best_add["delta_sharpe"]))

    if reopt_results:
        best_reopt = max(reopt_results, key=lambda x: x["sharpe"])
        all_improvements.append(
            (f"Re-optimized: {best_reopt['pair']} z_w={best_reopt['z_window']} "
             f"k={best_reopt['k_spread']:+.2f}",
             best_reopt["sharpe"], best_reopt["delta"]))

    if len(df_filt_better) > 0:
        best_filt = df_filt_better.iloc[0]
        all_improvements.append(
            (f"Filter: {best_filt['pair']} z_w={best_filt['z_window']:.0f} "
             f"thresh={best_filt['filter_thresh']:+.1f}",
             best_filt["sharpe"], best_filt["delta"]))

    if len(df_mom_better) > 0:
        best_mom = df_mom_better.iloc[0]
        all_improvements.append(
            (f"Momentum: {best_mom['pair']} mom_w={best_mom['mom_window']} "
             f"k={best_mom['k']:+.2f}",
             best_mom["sharpe"], best_mom["delta"]))

    if all_improvements:
        print(f"\n  Best improvements found:")
        for desc, sh, delta in sorted(all_improvements, key=lambda x: -x[1]):
            print(f"    {desc}")
            print(f"      Sharpe={sh:.4f}  delta={delta:+.4f}")
    else:
        print("\n  No RR term structure signal improves on baseline.")

    # Save
    df_add.to_csv(os.path.join(OUT_DIR, "additive_results.csv"), index=False)
    df_filt.to_csv(os.path.join(OUT_DIR, "filter_results.csv"), index=False)
    df_mom.to_csv(os.path.join(OUT_DIR, "momentum_results.csv"), index=False)

    # ================================================================
    # PLOT: Spread analysis
    # ================================================================
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle("RR Term Structure Spread Analysis",
                 fontsize=14, fontweight="bold")

    for idx, (ta, tb) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        spread = (rr_dict[ta] - rr_dict[tb]).reindex(common)
        v = ~spread.isna()
        ax.plot(common[v], spread.values[v], linewidth=0.5, alpha=0.7,
                color="#2874a6")
        # Rolling mean
        rm = spread.rolling(63, min_periods=20).mean()
        v2 = ~rm.isna()
        ax.plot(common[v2], rm.values[v2], linewidth=1.5, color="#e74c3c",
                alpha=0.8, label="63d MA")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_title(f"RR Spread: {ta} - {tb}", fontweight="bold")
        ax.set_ylabel("Spread (vol pts)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rr_spreads.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved: {OUT_DIR}/rr_spreads.png")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
