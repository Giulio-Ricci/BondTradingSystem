"""
Diagnostic: Does each component actually HELP the strategy?

Phase 1: Coarse grid over component combinations (~12K backtests)
Phase 2: Fine-tune top combos (~3K backtests)

For each: Sharpe, 2013 excess, dead years, win rate
"""

import numpy as np
import pandas as pd
import time

from config import RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime
from versione_presentazione import (
    build_rr_blend, build_slope_zscore, build_yc_signal, fast_bt, ANN
)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  COMPONENT ABLATION STUDY")
    print("  What helps? What hurts? What's optimal?")
    print("=" * 70)

    D = load_all()
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])
    etf_ret = D["etf_ret"]

    # Base signals
    rr_blend = build_rr_blend(D)
    z_slope = build_slope_zscore(D)

    # Pre-compute z_base for different windows
    z_base_cache = {}
    for w in [21, 42, 63, 126, 252]:
        mu = rr_blend.rolling(w, min_periods=min(20, w)).mean()
        sd = rr_blend.rolling(w, min_periods=min(20, w)).std().replace(0, np.nan)
        z_base_cache[w] = (rr_blend - mu) / sd

    # Pre-compute z_yc for different change windows
    z_yc_cache = {}
    for cw in [42, 63, 84, 126]:
        z_yc, _, _ = build_yc_signal(D, change_window=cw)
        z_yc_cache[cw] = z_yc

    # Align dates
    bt_start = pd.Timestamp(BACKTEST_START)
    common = (rr_blend.dropna().index
              .intersection(z_slope.dropna().index)
              .intersection(regime_hmm.dropna().index)
              .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index))
    for cw in z_yc_cache:
        common = common.intersection(z_yc_cache[cw].dropna().index)
    for w in [63, 126, 252]:
        common = common.intersection(z_base_cache[w].dropna().index)
    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])

    print(f"  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d} ({len(common)} days)")

    # ================================================================
    # Pre-compute z_rr arrays for all window + slope combos
    # ================================================================
    z_window_options = [
        ([63], "w63"),
        ([126], "w126"),
        ([252], "w252"),
        ([63, 126], "w63+126"),
        ([63, 252], "w63+252"),
        ([126, 252], "w126+252"),
        ([63, 126, 252], "w63+126+252"),
    ]

    k_slope_options = [0.0, 0.25, 0.50, 0.75, 1.0]

    z_rr_cache = {}
    z_slope_arr = z_slope.reindex(common).values
    for z_wins, z_label in z_window_options:
        z_parts = [z_base_cache[w].reindex(common).values for w in z_wins]
        z_base_arr = np.nanmean(z_parts, axis=0)
        for k_slope in k_slope_options:
            z_rr_cache[(z_label, k_slope)] = z_base_arr + k_slope * z_slope_arr

    # Pre-compute z_yc arrays
    z_yc_arr_cache = {}
    for cw in z_yc_cache:
        z_yc_arr_cache[cw] = z_yc_cache[cw].reindex(common).values

    yc_options = [
        (0.0, 0, "no_yc"),
        (0.20, 42, "yc42_k020"),
        (0.40, 42, "yc42_k040"),
        (0.20, 84, "yc84_k020"),
        (0.40, 84, "yc84_k040"),
        (0.60, 84, "yc84_k060"),
        (0.20, 126, "yc126_k020"),
        (0.40, 126, "yc126_k040"),
    ]

    # ================================================================
    # PHASE 1: Coarse grid — component combos × coarse thresholds
    # ================================================================
    tl_c_coarse = [-3.50, -3.00, -2.50]
    ts_c_coarse = [2.00, 2.50, 3.00, 3.50]
    tl_s_coarse = [-4.00, -3.50, -3.00]
    enhance_coarse = [
        (0, 0.0, 0, "no_enhance"),
        (42, 0.75, 252, "adapt42+mh252"),
        (63, 1.0, 252, "adapt63+mh252"),
    ]

    n_thresh = len(tl_c_coarse) * len(ts_c_coarse) * len(tl_s_coarse) * len(enhance_coarse)
    n_signal = len(z_window_options) * len(k_slope_options) * len(yc_options)
    total = n_signal * n_thresh
    print(f"\n  Phase 1: {n_signal} signal combos x {n_thresh} thresh/enh = {total} backtests")

    all_rows = []
    count = 0

    for z_wins, z_label in z_window_options:
        for k_slope in k_slope_options:
            z_rr_arr = z_rr_cache[(z_label, k_slope)]

            for k_yc, yc_cw, yc_label in yc_options:
                if k_yc > 0:
                    z_final = z_rr_arr + k_yc * z_yc_arr_cache[yc_cw]
                else:
                    z_final = z_rr_arr

                for tl_c in tl_c_coarse:
                    for ts_c in ts_c_coarse:
                        for tl_s in tl_s_coarse:
                            for ad, ar, mh, enh_label in enhance_coarse:
                                m = fast_bt(z_final, reg_arr, r_long, r_short,
                                            year_arr, tl_c, ts_c, tl_s,
                                            adapt_days=ad, adapt_raise=ar,
                                            max_hold_days=mh)

                                exc_2013 = 0.0
                                for i, y in enumerate(m["yearly_years"]):
                                    if y == 2013:
                                        exc_2013 = m["yearly_excess"][i]
                                        break

                                count += 1
                                all_rows.append({
                                    "z_windows": z_label,
                                    "k_slope": k_slope,
                                    "yc_config": yc_label,
                                    "tl_c": tl_c,
                                    "ts_c": ts_c,
                                    "tl_s": tl_s,
                                    "enhancement": enh_label,
                                    "sharpe": m["sharpe"],
                                    "sortino": m["sortino"],
                                    "calmar": m["calmar"],
                                    "ann_ret": m["ann_ret"],
                                    "mdd": m["mdd"],
                                    "n_events": m["n_events"],
                                    "n_dead": m["n_dead"],
                                    "win_rate": m["win_rate"],
                                    "exc_2013": exc_2013,
                                })

        elapsed_now = time.time() - t0
        print(f"    z_windows={z_label} done — {count}/{total} ({elapsed_now:.0f}s)")

    df = pd.DataFrame(all_rows)
    df_f = df[df["n_events"] >= 6].copy()
    elapsed = time.time() - t0
    print(f"\n  Phase 1 done: {count} configs in {elapsed:.0f}s ({len(df_f)} valid)")

    # ================================================================
    # PHASE 2: Fine-tune top 30 signal combos with full threshold grid
    # ================================================================
    print("\n" + "=" * 70)
    print("  PHASE 2: Fine-tuning top signal combos")
    print("=" * 70)

    # Find top 30 unique signal combos by max Sharpe
    sig_cols = ["z_windows", "k_slope", "yc_config"]
    sig_best = (df_f.groupby(sig_cols)["sharpe"].max()
                .reset_index().nlargest(30, "sharpe"))

    tl_c_fine = [-3.75, -3.50, -3.25, -3.00, -2.75, -2.50, -2.25]
    ts_c_fine = [2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50]
    tl_s_fine = [-4.50, -4.00, -3.75, -3.50, -3.25, -3.00]
    enhance_fine = [
        (0, 0.0, 0, "no_enhance"),
        (42, 0.50, 0, "adapt42_r050"),
        (42, 0.75, 0, "adapt42_r075"),
        (42, 0.75, 252, "adapt42+mh252"),
        (63, 0.75, 0, "adapt63_r075"),
        (63, 1.0, 0, "adapt63_r100"),
        (63, 1.0, 252, "adapt63+mh252"),
        (0, 0.0, 189, "mh189_only"),
        (0, 0.0, 252, "mh252_only"),
    ]

    n_thresh2 = len(tl_c_fine) * len(ts_c_fine) * len(tl_s_fine) * len(enhance_fine)
    total2 = len(sig_best) * n_thresh2
    print(f"  {len(sig_best)} signal combos x {n_thresh2} thresh/enh = {total2} backtests")

    rows2 = []
    count2 = 0
    for _, row in sig_best.iterrows():
        z_label = row["z_windows"]
        k_slope = row["k_slope"]
        yc_label = row["yc_config"]

        # Find matching yc params
        k_yc_val = 0.0
        yc_cw_val = 0
        for k_yc, yc_cw, yl in yc_options:
            if yl == yc_label:
                k_yc_val = k_yc
                yc_cw_val = yc_cw
                break

        z_rr_arr = z_rr_cache[(z_label, k_slope)]
        if k_yc_val > 0:
            z_final = z_rr_arr + k_yc_val * z_yc_arr_cache[yc_cw_val]
        else:
            z_final = z_rr_arr

        for tl_c in tl_c_fine:
            for ts_c in ts_c_fine:
                for tl_s in tl_s_fine:
                    for ad, ar, mh, enh_label in enhance_fine:
                        m = fast_bt(z_final, reg_arr, r_long, r_short,
                                    year_arr, tl_c, ts_c, tl_s,
                                    adapt_days=ad, adapt_raise=ar,
                                    max_hold_days=mh)

                        exc_2013 = 0.0
                        for i, y in enumerate(m["yearly_years"]):
                            if y == 2013:
                                exc_2013 = m["yearly_excess"][i]
                                break

                        count2 += 1
                        rows2.append({
                            "z_windows": z_label,
                            "k_slope": k_slope,
                            "yc_config": yc_label,
                            "tl_c": tl_c,
                            "ts_c": ts_c,
                            "tl_s": tl_s,
                            "enhancement": enh_label,
                            "sharpe": m["sharpe"],
                            "sortino": m["sortino"],
                            "calmar": m["calmar"],
                            "ann_ret": m["ann_ret"],
                            "mdd": m["mdd"],
                            "n_events": m["n_events"],
                            "n_dead": m["n_dead"],
                            "win_rate": m["win_rate"],
                            "exc_2013": exc_2013,
                        })

    df2 = pd.DataFrame(rows2)
    df2_f = df2[df2["n_events"] >= 6].copy()
    elapsed2 = time.time() - t0
    print(f"  Phase 2 done: {count2} configs in {elapsed2 - elapsed:.0f}s ({len(df2_f)} valid)")

    # Merge both phases
    df_all = pd.concat([df_f, df2_f], ignore_index=True)
    # Keep best per unique combo
    idx_cols = ["z_windows", "k_slope", "yc_config", "tl_c", "ts_c", "tl_s", "enhancement"]
    df_all = df_all.sort_values("sharpe", ascending=False).drop_duplicates(subset=idx_cols, keep="first")
    print(f"\n  Total unique configs: {len(df_all)}")

    # ================================================================
    # ANALYSIS 1: Top 20 by Sharpe
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Top 20 by Sharpe (no constraint)")
    print("=" * 70)

    top = df_all.nlargest(20, "sharpe")
    print(f"  {'#':>3} {'z_win':>12} {'k_sl':>5} {'yc':>14} {'enh':>16} "
          f"{'tl_c':>5} {'ts_c':>5} {'tl_s':>5} {'Sh':>6} {'dead':>4} "
          f"{'wr':>4} {'e13':>6}")
    print("  " + "-" * 112)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"  {i:>3} {r['z_windows']:>12} {r['k_slope']:>5.2f} "
              f"{r['yc_config']:>14} {r['enhancement']:>16} "
              f"{r['tl_c']:>5.2f} {r['ts_c']:>5.2f} {r['tl_s']:>5.2f} "
              f"{r['sharpe']:>6.3f} {r['n_dead']:>4.0f} "
              f"{r['win_rate']:>4.0%} {r['exc_2013']:>+6.1%}")

    # ================================================================
    # ANALYSIS 2: Top by Sharpe WHERE 2013 fixed
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Top 15 by Sharpe WHERE 2013 NOT dead (|exc|>1%)")
    print("=" * 70)

    df_fix13 = df_all[df_all["exc_2013"].abs() > 0.01]
    if len(df_fix13) > 0:
        top13 = df_fix13.nlargest(15, "sharpe")
        print(f"  {'#':>3} {'z_win':>12} {'k_sl':>5} {'yc':>14} {'enh':>16} "
              f"{'tl_c':>5} {'ts_c':>5} {'tl_s':>5} {'Sh':>6} {'dead':>4} "
              f"{'wr':>4} {'e13':>6}")
        print("  " + "-" * 112)
        for i, (_, r) in enumerate(top13.iterrows(), 1):
            print(f"  {i:>3} {r['z_windows']:>12} {r['k_slope']:>5.2f} "
                  f"{r['yc_config']:>14} {r['enhancement']:>16} "
                  f"{r['tl_c']:>5.2f} {r['ts_c']:>5.2f} {r['tl_s']:>5.2f} "
                  f"{r['sharpe']:>6.3f} {r['n_dead']:>4.0f} "
                  f"{r['win_rate']:>4.0%} {r['exc_2013']:>+6.1%}")
    else:
        print("  No configs fix 2013!")

    # ================================================================
    # ANALYSIS 3: Component contribution — avg Sharpe by component
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 3: Component contribution (avg Sharpe of top 50 per group)")
    print("=" * 70)

    def top_avg(group_col, df=df_all, n=50):
        groups = {}
        for val in sorted(df[group_col].unique(), key=str):
            sub = df[df[group_col] == val]
            topN = sub.nlargest(min(n, len(sub)), "sharpe")
            groups[val] = {
                "avg_sharpe": topN["sharpe"].mean(),
                "max_sharpe": topN["sharpe"].max(),
                "avg_dead": topN["n_dead"].mean(),
                "avg_wr": topN["win_rate"].mean(),
                "avg_e13": topN["exc_2013"].mean(),
                "n": len(sub),
            }
        return groups

    # By z_windows
    print("\n  --- z_windows ---")
    for k, v in sorted(top_avg("z_windows").items(),
                        key=lambda x: -x[1]["avg_sharpe"]):
        print(f"    {k:>14}: avg_sh={v['avg_sharpe']:.3f}  "
              f"max_sh={v['max_sharpe']:.3f}  "
              f"dead={v['avg_dead']:.1f}  wr={v['avg_wr']:.0%}  "
              f"e13={v['avg_e13']:+.2%}")

    # By k_slope
    print("\n  --- k_slope ---")
    for k, v in sorted(top_avg("k_slope").items(),
                        key=lambda x: -x[1]["avg_sharpe"]):
        print(f"    {k:>5.2f}: avg_sh={v['avg_sharpe']:.3f}  "
              f"max_sh={v['max_sharpe']:.3f}  "
              f"dead={v['avg_dead']:.1f}  wr={v['avg_wr']:.0%}  "
              f"e13={v['avg_e13']:+.2%}")

    # By yc_config
    print("\n  --- yc_config ---")
    for k, v in sorted(top_avg("yc_config").items(),
                        key=lambda x: -x[1]["avg_sharpe"]):
        print(f"    {k:>14}: avg_sh={v['avg_sharpe']:.3f}  "
              f"max_sh={v['max_sharpe']:.3f}  "
              f"dead={v['avg_dead']:.1f}  wr={v['avg_wr']:.0%}  "
              f"e13={v['avg_e13']:+.2%}")

    # By enhancement
    print("\n  --- enhancement ---")
    for k, v in sorted(top_avg("enhancement").items(),
                        key=lambda x: -x[1]["avg_sharpe"]):
        print(f"    {k:>16}: avg_sh={v['avg_sharpe']:.3f}  "
              f"max_sh={v['max_sharpe']:.3f}  "
              f"dead={v['avg_dead']:.1f}  wr={v['avg_wr']:.0%}  "
              f"e13={v['avg_e13']:+.2%}")

    # ================================================================
    # ANALYSIS 4: Best balanced configs
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 4: Best balanced configs")
    print("=" * 70)

    # Composite score like versione_presentazione
    df_scored = df_all.copy()
    sh_max = max(df_scored["sharpe"].max(), 0.01)
    so_max = max(df_scored["sortino"].max(), 0.01)
    ca_max = max(df_scored["calmar"].max(), 0.01)
    df_scored["score"] = (
        0.50 * df_scored["sharpe"] / sh_max
        + 0.15 * df_scored["sortino"] / so_max
        + 0.10 * df_scored["calmar"] / ca_max
        + 0.05 * df_scored["win_rate"]
        - 0.03 * df_scored["n_dead"]
    )

    top_balanced = df_scored.nlargest(20, "score")
    print(f"\n  Top 20 by composite score:")
    print(f"  {'#':>3} {'z_win':>12} {'k_sl':>5} {'yc':>14} {'enh':>16} "
          f"{'tl_c':>5} {'ts_c':>5} {'tl_s':>5} {'Sh':>6} {'dead':>4} "
          f"{'wr':>4} {'e13':>6} {'score':>6}")
    print("  " + "-" * 118)
    for i, (_, r) in enumerate(top_balanced.iterrows(), 1):
        print(f"  {i:>3} {r['z_windows']:>12} {r['k_slope']:>5.2f} "
              f"{r['yc_config']:>14} {r['enhancement']:>16} "
              f"{r['tl_c']:>5.2f} {r['ts_c']:>5.2f} {r['tl_s']:>5.2f} "
              f"{r['sharpe']:>6.3f} {r['n_dead']:>4.0f} "
              f"{r['win_rate']:>4.0%} {r['exc_2013']:>+6.1%} "
              f"{r['score']:>6.3f}")

    # ================================================================
    # ANALYSIS 5: Slope impact — controlled comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 5: Slope impact (best Sharpe per k_slope, same windows)")
    print("=" * 70)

    for z_label in ["w63", "w63+126", "w63+252", "w63+126+252"]:
        sub = df_all[df_all["z_windows"] == z_label]
        if len(sub) == 0:
            continue
        print(f"\n  {z_label}:")
        for k_sl in k_slope_options:
            sub_k = sub[sub["k_slope"] == k_sl]
            if len(sub_k) == 0:
                continue
            best = sub_k.loc[sub_k["sharpe"].idxmax()]
            print(f"    k_slope={k_sl:.2f}: Sh={best['sharpe']:.3f}  "
                  f"dead={best['n_dead']:.0f}  wr={best['win_rate']:.0%}  "
                  f"e13={best['exc_2013']:+.2%}  "
                  f"yc={best['yc_config']}  "
                  f"enh={best['enhancement']}  "
                  f"[{best['tl_c']:.2f},{best['ts_c']:.2f},{best['tl_s']:.2f}]")

    # ================================================================
    # ANALYSIS 6: Direct A/B — "with slope" vs "without slope"
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 6: Direct A/B — slope ON vs OFF (matched combos)")
    print("=" * 70)

    # For each (z_windows, yc_config, enhancement), compare best k_slope=0 vs best k_slope>0
    group_cols = ["z_windows", "yc_config", "enhancement"]
    for name, grp in df_all.groupby(group_cols):
        no_slope = grp[grp["k_slope"] == 0.0]
        with_slope = grp[grp["k_slope"] > 0.0]
        if len(no_slope) == 0 or len(with_slope) == 0:
            continue
        best_no = no_slope.loc[no_slope["sharpe"].idxmax()]
        best_with = with_slope.loc[with_slope["sharpe"].idxmax()]
        delta = best_with["sharpe"] - best_no["sharpe"]
        if abs(delta) < 0.001:
            continue  # skip ties
        sign = "+" if delta > 0 else ""
        z_l, yc_l, enh_l = name
        print(f"  {z_l:>12} {yc_l:>14} {enh_l:>16}: "
              f"no_slope={best_no['sharpe']:.3f}  "
              f"best_slope(k={best_with['k_slope']:.2f})={best_with['sharpe']:.3f}  "
              f"delta={sign}{delta:.3f}")

    # ================================================================
    # ANALYSIS 7: YC impact — "with YC" vs "without YC"
    # ================================================================
    print("\n" + "=" * 70)
    print("  ANALYSIS 7: Direct A/B — YC ON vs OFF (matched combos)")
    print("=" * 70)

    group_cols2 = ["z_windows", "k_slope", "enhancement"]
    for name, grp in df_all.groupby(group_cols2):
        no_yc = grp[grp["yc_config"] == "no_yc"]
        with_yc = grp[grp["yc_config"] != "no_yc"]
        if len(no_yc) == 0 or len(with_yc) == 0:
            continue
        best_no = no_yc.loc[no_yc["sharpe"].idxmax()]
        best_with = with_yc.loc[with_yc["sharpe"].idxmax()]
        delta = best_with["sharpe"] - best_no["sharpe"]
        if abs(delta) < 0.001:
            continue
        sign = "+" if delta > 0 else ""
        z_l, k_sl, enh_l = name
        print(f"  {z_l:>12} k_sl={k_sl:.2f} {enh_l:>16}: "
              f"no_yc={best_no['sharpe']:.3f}  "
              f"best_yc({best_with['yc_config']})={best_with['sharpe']:.3f}  "
              f"delta={sign}{delta:.3f}")

    # Save
    df_all.to_csv("output/versione_presentazione/component_ablation.csv", index=False)
    elapsed_final = time.time() - t0
    print(f"\n  Total time: {elapsed_final:.0f}s")
    print(f"  Saved {len(df_all)} configs to component_ablation.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
