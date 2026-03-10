"""
Integration test: RR Term Structure spread into main strategy.

Tests the two best spreads (1W-1M, 1M-3M) as:
1. REVERSAL signal (contrarian: spread deviation → mean reversion)
2. CONFIRMATION signal (trend: spread confirms z_final direction)
3. FILTER (block positions when spread disagrees)
4. COMBINED (reversal + filter together)

Full grid search with threshold re-optimization for each mode.
"""

import numpy as np
import pandas as pd
import time
import os

from config import RR_TENORS, BACKTEST_START, RISK_FREE_RATE, SAVE_DIR
from data_loader import load_all
from regime import fit_hmm_regime
from versione_presentazione import (
    build_rr_blend, build_slope_zscore, build_rr_multitf,
    build_yc_signal, build_combined, fast_bt, composite_score, ANN
)

OUT_DIR = os.path.join(SAVE_DIR, "rr_term_integrate")
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


def fast_bt_filtered(z_arr, reg_arr, r_long, r_short, year_arr,
                     tl_c, ts_c, tl_s,
                     adapt_days=0, adapt_raise=0.0, max_hold_days=0,
                     filter_arr=None, filter_mode=None, filter_thresh=0.0):
    """Backtest with optional spread filter applied to z_arr before trading.

    filter_mode:
      'block_long':  when filter_arr > filter_thresh, cap z at 0 (no LONG)
      'block_short': when filter_arr < -filter_thresh, floor z at 0 (no SHORT)
      'block_both':  block_long + block_short
      'dampen':      multiply z by (1 - |filter_arr|/scale) when filter disagrees
    """
    z_eff = z_arr.copy()

    if filter_arr is not None and filter_mode is not None:
        valid = ~np.isnan(filter_arr)

        if filter_mode == "block_long":
            mask = valid & (filter_arr > filter_thresh)
            z_eff[mask] = np.minimum(z_eff[mask], 0)

        elif filter_mode == "block_short":
            mask = valid & (filter_arr < -filter_thresh)
            z_eff[mask] = np.maximum(z_eff[mask], 0)

        elif filter_mode == "block_both":
            mask_l = valid & (filter_arr > filter_thresh)
            z_eff[mask_l] = np.minimum(z_eff[mask_l], 0)
            mask_s = valid & (filter_arr < -filter_thresh)
            z_eff[mask_s] = np.maximum(z_eff[mask_s], 0)

        elif filter_mode == "dampen":
            # Dampen z when filter disagrees with sign
            # If z > 0 (wants LONG) but filter > thresh (bearish) → reduce z
            # If z < 0 (wants SHORT) but filter < -thresh (bullish) → reduce z
            scale = 2.0
            dampen_long = valid & (z_eff > 0) & (filter_arr > filter_thresh)
            dampen_short = valid & (z_eff < 0) & (filter_arr < -filter_thresh)
            factor = np.clip(1.0 - np.abs(filter_arr) / scale, 0.0, 1.0)
            z_eff[dampen_long] *= factor[dampen_long]
            z_eff[dampen_short] *= factor[dampen_short]

    return fast_bt(z_eff, reg_arr, r_long, r_short, year_arr,
                   tl_c, ts_c, tl_s,
                   adapt_days=adapt_days, adapt_raise=adapt_raise,
                   max_hold_days=max_hold_days)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  RR TERM STRUCTURE INTEGRATION TEST")
    print("  Reversal vs Confirmation vs Filter vs Combined")
    print("=" * 70)

    D = load_all()
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])
    etf_ret = D["etf_ret"]
    rr_dict = build_rr_series(D)

    # Current best config
    best_cfg = {
        "z_windows": [63, 252], "k_slope": 0.75,
        "yc_cw": 126, "k_yc": 0.20,
        "tl_c": -3.375, "ts_c": 3.00, "tl_s": -2.875,
        "adapt_days": 63, "adapt_raise": 1.0, "max_hold_days": 252,
    }

    # Build base signal
    z_rr_best, _, _ = build_rr_multitf(D, best_cfg["z_windows"],
                                        k_slope=best_cfg["k_slope"])
    z_yc_best, _, _ = build_yc_signal(D, change_window=best_cfg["yc_cw"])
    z_base_signal = build_combined(z_rr_best, z_yc_best, best_cfg["k_yc"])

    # Align
    bt_start = pd.Timestamp(BACKTEST_START)
    common = (z_base_signal.dropna().index
              .intersection(regime_hmm.dropna().index)
              .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index))
    for t in ["1W", "1M", "3M", "6M"]:
        common = common.intersection(rr_dict[t].dropna().index)
    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])
    z_base_arr = z_base_signal.reindex(common).values

    # Baseline
    m_bl = fast_bt(z_base_arr, reg_arr, r_long, r_short, year_arr,
                   best_cfg["tl_c"], best_cfg["ts_c"], best_cfg["tl_s"],
                   adapt_days=best_cfg["adapt_days"],
                   adapt_raise=best_cfg["adapt_raise"],
                   max_hold_days=best_cfg["max_hold_days"])

    print(f"\n  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d} ({len(common)} days)")
    print(f"  BASELINE: Sharpe={m_bl['sharpe']:.4f}  dead={m_bl['n_dead']}  "
          f"wr={m_bl['win_rate']:.0%}  events={m_bl['n_events']}")

    # Pre-compute all spread z-scores
    spread_pairs = [
        ("1W", "1M"), ("1W", "3M"), ("1W", "6M"),
        ("1M", "3M"), ("1M", "6M"), ("3M", "6M"),
    ]
    z_win_list = [42, 63, 126, 252]

    spread_cache = {}
    for ta, tb in spread_pairs:
        for zw in z_win_list:
            z_sp, raw = build_spread_zscore(rr_dict[ta], rr_dict[tb], z_window=zw)
            spread_cache[(ta, tb, zw)] = z_sp.reindex(common).values

    # Threshold grid
    tl_c_grid = [-3.75, -3.50, -3.375, -3.25, -3.00]
    ts_c_grid = [2.50, 2.75, 3.00, 3.25, 3.50]
    tl_s_grid = [-3.50, -3.00, -2.875, -2.75]
    enhance_grid = [
        (0, 0.0, 0),
        (63, 1.0, 252),
    ]

    all_results = []
    count = 0

    # ================================================================
    # MODE 1: REVERSAL (additive, positive k → spread up = bullish)
    # ================================================================
    print("\n" + "=" * 70)
    print("  MODE 1: REVERSAL (additive, spread as contrarian signal)")
    print("  z_final = z_base + k * z_spread")
    print("=" * 70)

    k_add_list = [0.10, 0.20, 0.30, 0.50]

    for ta, tb in spread_pairs:
        for zw in z_win_list:
            z_sp = spread_cache[(ta, tb, zw)]
            for k in k_add_list:
                # Positive k: high spread = bullish (reversal)
                z_test = z_base_arr + k * z_sp
                for tl_c in tl_c_grid:
                    for ts_c in ts_c_grid:
                        for tl_s in tl_s_grid:
                            for ad, ar, mh in enhance_grid:
                                m = fast_bt(z_test, reg_arr, r_long, r_short,
                                            year_arr, tl_c, ts_c, tl_s,
                                            adapt_days=ad, adapt_raise=ar,
                                            max_hold_days=mh)
                                if m["n_events"] >= 6:
                                    count += 1
                                    all_results.append({
                                        "mode": "reversal",
                                        "pair": f"{ta}-{tb}", "z_window": zw,
                                        "k_spread": k,
                                        "filter_mode": "none",
                                        "filter_thresh": 0,
                                        "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s,
                                        "adapt_days": ad, "max_hold": mh,
                                        "sharpe": m["sharpe"],
                                        "n_dead": m["n_dead"],
                                        "win_rate": m["win_rate"],
                                        "n_events": m["n_events"],
                                        "ann_ret": m["ann_ret"],
                                        "score": composite_score(m),
                                    })

    print(f"  Tested reversal(+): {count} configs")

    # ================================================================
    # MODE 2: CONFIRMATION (additive, negative k → spread down = bullish)
    # ================================================================
    print("\n" + "=" * 70)
    print("  MODE 2: CONFIRMATION (additive, spread as trend signal)")
    print("  z_final = z_base - k * z_spread")
    print("=" * 70)

    count2 = 0
    for ta, tb in spread_pairs:
        for zw in z_win_list:
            z_sp = spread_cache[(ta, tb, zw)]
            for k in k_add_list:
                # Negative k: low spread = bullish (confirmation/trend)
                z_test = z_base_arr - k * z_sp
                for tl_c in tl_c_grid:
                    for ts_c in ts_c_grid:
                        for tl_s in tl_s_grid:
                            for ad, ar, mh in enhance_grid:
                                m = fast_bt(z_test, reg_arr, r_long, r_short,
                                            year_arr, tl_c, ts_c, tl_s,
                                            adapt_days=ad, adapt_raise=ar,
                                            max_hold_days=mh)
                                if m["n_events"] >= 6:
                                    count2 += 1
                                    all_results.append({
                                        "mode": "confirmation",
                                        "pair": f"{ta}-{tb}", "z_window": zw,
                                        "k_spread": -k,
                                        "filter_mode": "none",
                                        "filter_thresh": 0,
                                        "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s,
                                        "adapt_days": ad, "max_hold": mh,
                                        "sharpe": m["sharpe"],
                                        "n_dead": m["n_dead"],
                                        "win_rate": m["win_rate"],
                                        "n_events": m["n_events"],
                                        "ann_ret": m["ann_ret"],
                                        "score": composite_score(m),
                                    })

    print(f"  Tested confirmation(-): {count2} configs")

    # ================================================================
    # MODE 3: FILTER (block positions when spread disagrees)
    # ================================================================
    print("\n" + "=" * 70)
    print("  MODE 3: FILTER (block LONG/SHORT when spread disagrees)")
    print("=" * 70)

    filter_modes = ["block_long", "block_short", "block_both", "dampen"]
    filter_thresholds = [0.0, 0.5, 1.0, 1.5]

    count3 = 0
    for ta, tb in spread_pairs:
        for zw in z_win_list:
            z_sp = spread_cache[(ta, tb, zw)]
            for fm in filter_modes:
                for ft in filter_thresholds:
                    for tl_c in tl_c_grid:
                        for ts_c in ts_c_grid:
                            for tl_s in tl_s_grid:
                                for ad, ar, mh in enhance_grid:
                                    m = fast_bt_filtered(
                                        z_base_arr, reg_arr, r_long, r_short,
                                        year_arr, tl_c, ts_c, tl_s,
                                        adapt_days=ad, adapt_raise=ar,
                                        max_hold_days=mh,
                                        filter_arr=z_sp,
                                        filter_mode=fm,
                                        filter_thresh=ft)
                                    if m["n_events"] >= 6:
                                        count3 += 1
                                        all_results.append({
                                            "mode": "filter",
                                            "pair": f"{ta}-{tb}",
                                            "z_window": zw,
                                            "k_spread": 0,
                                            "filter_mode": fm,
                                            "filter_thresh": ft,
                                            "tl_c": tl_c, "ts_c": ts_c,
                                            "tl_s": tl_s,
                                            "adapt_days": ad, "max_hold": mh,
                                            "sharpe": m["sharpe"],
                                            "n_dead": m["n_dead"],
                                            "win_rate": m["win_rate"],
                                            "n_events": m["n_events"],
                                            "ann_ret": m["ann_ret"],
                                            "score": composite_score(m),
                                        })

    print(f"  Tested filter: {count3} configs")

    # ================================================================
    # MODE 4: COMBINED (reversal additive + filter)
    # ================================================================
    print("\n" + "=" * 70)
    print("  MODE 4: COMBINED (reversal + filter)")
    print("  z_final = z_base + k*z_spread_A, filtered by z_spread_B")
    print("=" * 70)

    # Use best additive pair (1W-1M) + best filter pair (1M-3M)
    combo_pairs = [
        (("1W", "1M", 126), ("1M", "3M", 126)),
        (("1W", "1M", 126), ("1M", "3M", 42)),
        (("1W", "3M", 42), ("1M", "3M", 126)),
    ]

    count4 = 0
    for (add_ta, add_tb, add_zw), (filt_ta, filt_tb, filt_zw) in combo_pairs:
        z_sp_add = spread_cache[(add_ta, add_tb, add_zw)]
        z_sp_filt = spread_cache[(filt_ta, filt_tb, filt_zw)]

        for k in [0.20, 0.30, 0.50]:
            z_test = z_base_arr + k * z_sp_add

            for fm in ["block_long", "block_both"]:
                for ft in [0.0, 0.5, 1.0]:
                    for tl_c in tl_c_grid:
                        for ts_c in ts_c_grid:
                            for tl_s in tl_s_grid:
                                for ad, ar, mh in enhance_grid:
                                    m = fast_bt_filtered(
                                        z_test, reg_arr, r_long, r_short,
                                        year_arr, tl_c, ts_c, tl_s,
                                        adapt_days=ad, adapt_raise=ar,
                                        max_hold_days=mh,
                                        filter_arr=z_sp_filt,
                                        filter_mode=fm,
                                        filter_thresh=ft)
                                    if m["n_events"] >= 6:
                                        count4 += 1
                                        all_results.append({
                                            "mode": "combined",
                                            "pair": f"{add_ta}-{add_tb}+F:{filt_ta}-{filt_tb}",
                                            "z_window": add_zw,
                                            "k_spread": k,
                                            "filter_mode": fm,
                                            "filter_thresh": ft,
                                            "tl_c": tl_c, "ts_c": ts_c,
                                            "tl_s": tl_s,
                                            "adapt_days": ad, "max_hold": mh,
                                            "sharpe": m["sharpe"],
                                            "n_dead": m["n_dead"],
                                            "win_rate": m["win_rate"],
                                            "n_events": m["n_events"],
                                            "ann_ret": m["ann_ret"],
                                            "score": composite_score(m),
                                        })

    print(f"  Tested combined: {count4} configs")

    # ================================================================
    # ANALYSIS
    # ================================================================
    df = pd.DataFrame(all_results)
    total = len(df)
    print(f"\n  Total configs tested: {total}")

    print("\n" + "=" * 70)
    print("  RESULTS BY MODE")
    print("=" * 70)

    bl_sharpe = m_bl["sharpe"]
    bl_score = composite_score(m_bl)

    for mode in ["reversal", "confirmation", "filter", "combined"]:
        sub = df[df["mode"] == mode]
        if len(sub) == 0:
            continue

        top = sub.nlargest(10, "score")
        top_sh = sub.nlargest(10, "sharpe")

        print(f"\n  --- {mode.upper()} ---")
        print(f"  Top 10 by composite score:")
        print(f"  {'#':>3} {'Pair':<28} {'z_w':>4} {'k':>6} {'filt':>11} {'ft':>4} "
              f"{'tl':>6} {'ts':>5} {'tls':>6} {'ad':>3} {'mh':>4} "
              f"{'Sh':>6} {'dSh':>6} {'dead':>4} {'wr':>4} {'score':>6}")
        print("  " + "-" * 120)
        for i, (_, r) in enumerate(top.iterrows(), 1):
            delta = r["sharpe"] - bl_sharpe
            filt_str = r["filter_mode"][:11] if r["filter_mode"] != "none" else "-"
            print(f"  {i:>3} {r['pair']:<28} {r['z_window']:>4} "
                  f"{r['k_spread']:>+6.2f} {filt_str:>11} {r['filter_thresh']:>4.1f} "
                  f"{r['tl_c']:>6.3f} {r['ts_c']:>5.2f} {r['tl_s']:>6.3f} "
                  f"{r['adapt_days']:>3.0f} {r['max_hold']:>4.0f} "
                  f"{r['sharpe']:>6.3f} {delta:>+6.3f} "
                  f"{r['n_dead']:>4.0f} {r['win_rate']:>4.0%} "
                  f"{r['score']:>6.3f}")

    # ================================================================
    # OVERALL BEST vs BASELINE
    # ================================================================
    print("\n" + "=" * 70)
    print("  OVERALL TOP 20 (all modes)")
    print("=" * 70)

    top_all = df.nlargest(20, "score")
    print(f"\n  Baseline: Sharpe={bl_sharpe:.4f}  score={bl_score:.4f}  "
          f"dead={m_bl['n_dead']}  wr={m_bl['win_rate']:.0%}")
    print(f"\n  {'#':>3} {'Mode':<13} {'Pair':<28} {'z_w':>4} {'k':>6} "
          f"{'filt':>11} {'ft':>4} "
          f"{'Sh':>6} {'dSh':>6} {'dead':>4} {'wr':>4} {'score':>6}")
    print("  " + "-" * 110)
    for i, (_, r) in enumerate(top_all.iterrows(), 1):
        delta = r["sharpe"] - bl_sharpe
        filt_str = r["filter_mode"][:11] if r["filter_mode"] != "none" else "-"
        print(f"  {i:>3} {r['mode']:<13} {r['pair']:<28} {r['z_window']:>4} "
              f"{r['k_spread']:>+6.2f} {filt_str:>11} {r['filter_thresh']:>4.1f} "
              f"{r['sharpe']:>6.3f} {delta:>+6.3f} "
              f"{r['n_dead']:>4.0f} {r['win_rate']:>4.0%} "
              f"{r['score']:>6.3f}")

    # ================================================================
    # YEARLY BREAKDOWN of top 3
    # ================================================================
    print("\n" + "=" * 70)
    print("  YEARLY BREAKDOWN: Top 3 vs Baseline")
    print("=" * 70)

    top3 = top_all.head(3)
    yearly_data = {}

    # Baseline yearly
    yearly_data["Baseline"] = {
        int(m_bl["yearly_years"][i]): m_bl["yearly_excess"][i]
        for i in range(len(m_bl["yearly_years"]))
    }

    for idx, (_, r) in enumerate(top3.iterrows(), 1):
        ta_tb = r["pair"].split("+")[0] if "+" in r["pair"] else r["pair"]
        parts = ta_tb.split("-")
        ta, tb = parts[0], parts[1]
        zw = int(r["z_window"])
        z_sp = spread_cache[(ta, tb, zw)]

        if r["mode"] in ("reversal", "confirmation"):
            z_test = z_base_arr + r["k_spread"] * z_sp
            m_test = fast_bt(z_test, reg_arr, r_long, r_short, year_arr,
                             r["tl_c"], r["ts_c"], r["tl_s"],
                             adapt_days=int(r["adapt_days"]),
                             adapt_raise=1.0 if r["adapt_days"] > 0 else 0.0,
                             max_hold_days=int(r["max_hold"]))
        elif r["mode"] == "filter":
            m_test = fast_bt_filtered(
                z_base_arr, reg_arr, r_long, r_short, year_arr,
                r["tl_c"], r["ts_c"], r["tl_s"],
                adapt_days=int(r["adapt_days"]),
                adapt_raise=1.0 if r["adapt_days"] > 0 else 0.0,
                max_hold_days=int(r["max_hold"]),
                filter_arr=z_sp,
                filter_mode=r["filter_mode"],
                filter_thresh=r["filter_thresh"])
        else:  # combined
            add_pair = r["pair"].split("+F:")[0]
            filt_pair = r["pair"].split("+F:")[1] if "+F:" in r["pair"] else ta_tb
            add_ta, add_tb = add_pair.split("-")
            filt_ta, filt_tb = filt_pair.split("-")
            z_sp_add = spread_cache[(add_ta, add_tb, zw)]
            # Try to find filter z_window — use same as additive
            filt_zw = zw
            for fw in [42, 63, 126, 252]:
                if (filt_ta, filt_tb, fw) in spread_cache:
                    filt_zw = fw
                    break
            z_sp_filt = spread_cache.get((filt_ta, filt_tb, filt_zw), z_sp)
            z_test = z_base_arr + r["k_spread"] * z_sp_add
            m_test = fast_bt_filtered(
                z_test, reg_arr, r_long, r_short, year_arr,
                r["tl_c"], r["ts_c"], r["tl_s"],
                adapt_days=int(r["adapt_days"]),
                adapt_raise=1.0 if r["adapt_days"] > 0 else 0.0,
                max_hold_days=int(r["max_hold"]),
                filter_arr=z_sp_filt,
                filter_mode=r["filter_mode"],
                filter_thresh=r["filter_thresh"])

        label = f"#{idx} {r['mode'][:4]}"
        yearly_data[label] = {
            int(m_test["yearly_years"][i]): m_test["yearly_excess"][i]
            for i in range(len(m_test["yearly_years"]))
        }

    # Print yearly comparison
    all_years = sorted(set().union(*[set(v.keys()) for v in yearly_data.values()]))
    labels = list(yearly_data.keys())

    print(f"\n  {'Year':>6}", end="")
    for l in labels:
        print(f"  {l:>12}", end="")
    print()
    print("  " + "-" * (6 + 14 * len(labels)))

    for y in all_years:
        print(f"  {y:>6}", end="")
        for l in labels:
            exc = yearly_data[l].get(y, 0)
            dead = " D" if abs(exc) < 0.01 else "  "
            print(f"  {exc:>+10.4f}{dead}", end="")
        print()

    # Save
    df.to_csv(os.path.join(OUT_DIR, "all_results.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
