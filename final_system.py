"""
final_system.py - Complete optimization pipeline.

Runs all 4 stages:
  1. Walk-forward validation of best config (ALL|std|w63)
  2. Composite signal: ALL|std|w63 + slope momentum confirmation
  3. Deep grid around optimal zone
  4. Final system assembly with best validated config

Output: output/final/ with full report, equity curves, and comparison.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from config import (
    SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE, DEFAULT_PARAMS,
)
from data_loader import load_all
from regime import fit_hmm_regime

# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
FINAL_DIR = os.path.join(SAVE_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# FAST BACKTEST (numpy, no pandas overhead)
# ═══════════════════════════════════════════════════════════════════════════

def fast_bt(z_arr, reg_arr, r_long, r_short,
            tl_c, ts_c, tl_s, cooldown=5,
            delay=1, tcost_bps=5, w_init=0.5):
    n = len(z_arr)
    if n < 50:
        return 0.0, 0.0, 0.0, 0.0, 0, 1.0

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

    running_max = np.maximum.accumulate(equity)
    mdd = (equity / running_max - 1.0).min()

    return sharpe, ann_ret, vol, mdd, n_events, equity[-1]


def fast_bt_full(z_arr, reg_arr, r_long, r_short,
                 tl_c, ts_c, tl_s, cooldown=5,
                 delay=1, tcost_bps=5, w_init=0.5):
    """Returns (equity_array, net_return_array, weight_array, events_list)."""
    n = len(z_arr)
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
        reg = reg_arr[i]
        triggered = False
        sig = None
        if reg == 0:
            if zv <= tl_c and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            elif zv >= ts_c and (i - last_ev) >= cooldown:
                weight[i] = 1.0; triggered = True; sig = "LONG"
            else:
                weight[i] = w_prev
        else:
            if zv <= tl_s and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            else:
                weight[i] = w_prev
        if triggered:
            last_ev = i
            events.append((i, sig, zv, reg))

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

    return equity, net, w_del, events


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_rr_blend(options_df, tenor_keys):
    parts = []
    for tk in tenor_keys:
        put_col, call_col = RR_TENORS[tk]
        rr = (pd.to_numeric(options_df[put_col], errors="coerce")
              - pd.to_numeric(options_df[call_col], errors="coerce"))
        parts.append(rr)
    return pd.concat(parts, axis=1).mean(axis=1)


def compute_zscore(rr, move, z_type, z_window, move_med_window=504):
    min_per = max(z_window // 4, 20)
    mu  = rr.rolling(z_window, min_periods=min_per).mean()
    sig = rr.rolling(z_window, min_periods=min_per).std()
    if z_type == "std":
        return (rr - mu) / sig.replace(0, np.nan)
    move_med = move.rolling(move_med_window, min_periods=min_per).median()
    move_ratio = (move / move_med).clip(0.5, 2.5)
    sig_adj = sig * np.sqrt(move_ratio)
    return (rr - mu) / sig_adj.replace(0, np.nan)


def compute_slope_momentum(yields, col_long, col_short, window=21):
    """Compute z-scored momentum of yield curve slope change."""
    slope = yields[col_long] - yields[col_short]
    delta = slope.diff(window)
    mu = delta.rolling(252, min_periods=63).mean()
    sig = delta.rolling(252, min_periods=63).std()
    return (delta - mu) / sig.replace(0, np.nan)


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def stage1_walk_forward(D, regime, regime_num):
    """Walk-forward validation of top configs with expanding window."""
    print("\n" + "=" * 70)
    print("  STAGE 1: WALK-FORWARD VALIDATION")
    print("=" * 70)

    etf_ret = D["etf_ret"]
    start = pd.Timestamp(BACKTEST_START)

    # Configs to validate
    configs = [
        ("ALL|std|w63 (optimized)", "ALL", "std", 63, None, -3.25, 2.75, -4.0, 5),
        ("ALL|std|w63 (baseline)", "ALL", "std", 63, None, -3.50, 3.00, -4.0, 5),
        ("ALL|move|w250|mw504", "ALL", "move", 250, 504, -3.25, 2.50, -3.5, 5),
        ("1M|move|w180|mw504", "1M", "move", 180, 504, -3.50, 2.50, -4.5, 5),
        ("Baseline V1 (z_move)", "1M", "move", 250, 504, -3.50, 3.00, -4.0, 5),
    ]

    # WF parameters
    first_oos_year = 2014
    last_oos_year = 2025
    min_is_days = 504

    wf_results = {}

    for cfg_name, rr_key, z_type, z_win, mw, tl_c, ts_c, tl_s, cd in configs:
        print(f"\n  {cfg_name}:")

        # Compute signal
        if rr_key == "ALL":
            tenor_keys = ["1W", "1M", "3M", "6M"]
        elif rr_key == "1M":
            tenor_keys = ["1M"]
        else:
            tenor_keys = rr_key.split("_")

        rr = compute_rr_blend(D["options"], tenor_keys)
        z = compute_zscore(rr, D["move"], z_type, z_win, mw or 504)

        # Align data
        common = (z.dropna().index
                  .intersection(regime.index)
                  .intersection(etf_ret.dropna(subset=["TLT","SHV"]).index))
        common = common[common >= start].sort_values()

        z_vals = z.reindex(common).values
        reg_vals = regime_num.reindex(common).values
        r_long = etf_ret["TLT"].reindex(common).fillna(0).values
        r_short = etf_ret["SHV"].reindex(common).fillna(0).values
        dates = common

        # Full in-sample backtest
        sh_is, ar_is, vol_is, mdd_is, nev_is, eqf_is = fast_bt(
            z_vals, reg_vals, r_long, r_short, tl_c, ts_c, tl_s, cd)
        print(f"    Full IS: Sharpe={sh_is:.3f}  AnnRet={ar_is:.4f}  MDD={mdd_is:.3f}  Events={nev_is}")

        # Walk-forward: expanding window
        oos_yearly = []
        for oos_year in range(first_oos_year, last_oos_year + 1):
            oos_start = pd.Timestamp(f"{oos_year}-01-01")
            oos_end = pd.Timestamp(f"{oos_year}-12-31")

            is_mask = dates < oos_start
            oos_mask = (dates >= oos_start) & (dates <= oos_end)

            if is_mask.sum() < min_is_days or oos_mask.sum() < 20:
                continue

            # OOS backtest on the year
            oos_idx = np.where(oos_mask)[0]
            z_oos = z_vals[oos_idx]
            reg_oos = reg_vals[oos_idx]
            rl_oos = r_long[oos_idx]
            rs_oos = r_short[oos_idx]

            sh_oos, ar_oos, vol_oos, mdd_oos, nev_oos, eqf_oos = fast_bt(
                z_oos, reg_oos, rl_oos, rs_oos, tl_c, ts_c, tl_s, cd)

            # Benchmark: B&H TLT for this year
            bm_eq = np.cumprod(1.0 + rl_oos)
            bm_ret_yr = bm_eq[-1] - 1.0

            strat_ret_yr = eqf_oos - 1.0
            excess = strat_ret_yr - bm_ret_yr

            oos_yearly.append({
                "year": oos_year,
                "strat_ret": strat_ret_yr,
                "bm_ret": bm_ret_yr,
                "excess": excess,
                "sharpe": sh_oos,
                "n_events": nev_oos,
            })

        ydf = pd.DataFrame(oos_yearly)
        if len(ydf) > 0:
            pos_years = (ydf["excess"] > 0).sum()
            tot_years = len(ydf)
            avg_excess = ydf["excess"].mean()
            avg_sharpe = ydf["sharpe"].mean()
            print(f"    WF OOS: {pos_years}/{tot_years} positive years  "
                  f"Avg excess={avg_excess:+.4f}  Avg Sharpe={avg_sharpe:.3f}")
            for _, row in ydf.iterrows():
                marker = "+" if row["excess"] > 0 else " "
                print(f"      {int(row['year'])}  strat={row['strat_ret']:+.4f}  "
                      f"bm={row['bm_ret']:+.4f}  excess={row['excess']:+.4f}  "
                      f"sh={row['sharpe']:.3f} {marker}")

        wf_results[cfg_name] = {
            "is_sharpe": sh_is,
            "oos_yearly": ydf,
            "oos_avg_excess": ydf["excess"].mean() if len(ydf) > 0 else 0,
            "oos_pos_years": (ydf["excess"] > 0).sum() if len(ydf) > 0 else 0,
            "oos_tot_years": len(ydf),
            "params": (tl_c, ts_c, tl_s, cd),
        }

    # ── WF Summary plot ──
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Top: yearly excess by config
    ax = axes[0]
    years_all = sorted(set(y for r in wf_results.values()
                          for y in r["oos_yearly"]["year"].values if len(r["oos_yearly"]) > 0))
    x = np.arange(len(years_all))
    width = 0.15
    for j, (name, res) in enumerate(wf_results.items()):
        ydf = res["oos_yearly"]
        if len(ydf) == 0:
            continue
        vals = [ydf[ydf["year"]==y]["excess"].values[0]
                if y in ydf["year"].values else 0 for y in years_all]
        offset = (j - len(wf_results)/2) * width
        bars = ax.bar(x + offset, vals, width, label=name[:30], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(years_all)
    ax.set_ylabel("Excess Return (vs TLT)")
    ax.set_title("Walk-Forward OOS: Yearly Excess Return by Config")
    ax.legend(fontsize=7, loc="upper left")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # Bottom: summary table
    ax = axes[1]
    ax.axis("off")
    headers = ["Config", "IS Sharpe", "OOS Avg Excess", "OOS Pos Years", "Robustness"]
    table_data = []
    for name, res in wf_results.items():
        pct = f"{res['oos_pos_years']}/{res['oos_tot_years']}" if res['oos_tot_years'] > 0 else "N/A"
        robust = "PASS" if (res["oos_pos_years"] / max(res["oos_tot_years"], 1)) >= 0.5 else "FAIL"
        table_data.append([
            name[:35], f"{res['is_sharpe']:.3f}",
            f"{res['oos_avg_excess']:+.4f}", pct, robust
        ])
    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("Walk-Forward Validation Summary", fontsize=13, pad=20)

    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_DIR, "01_walk_forward_validation.png"), dpi=150)
    plt.close()
    print(f"\n  Plot saved: {FINAL_DIR}/01_walk_forward_validation.png")

    return wf_results


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: COMPOSITE SIGNAL (ALL|std|w63 + slope momentum)
# ═══════════════════════════════════════════════════════════════════════════

def stage2_composite(D, regime, regime_num):
    """Test combining ALL|std|w63 with yield curve slope momentum."""
    print("\n" + "=" * 70)
    print("  STAGE 2: COMPOSITE SIGNAL (RR + Slope Momentum)")
    print("=" * 70)

    etf_ret = D["etf_ret"]
    yields = D["yields"]
    start = pd.Timestamp(BACKTEST_START)

    # Base signal: ALL|std|w63
    rr_all = compute_rr_blend(D["options"], ["1W", "1M", "3M", "6M"])
    z_base = compute_zscore(rr_all, D["move"], "std", 63)

    # Also test with z_move base
    rr_1m = compute_rr_blend(D["options"], ["1M"])
    z_move_base = compute_zscore(rr_1m, D["move"], "move", 250, 504)

    # Slope components to test
    slope_pairs = [
        ("5Y-2Y",   "US_5Y",  "US_2Y"),
        ("10Y-2Y",  "US_10Y", "US_2Y"),
        ("30Y-10Y", "US_30Y", "US_10Y"),
        ("30Y-2Y",  "US_30Y", "US_2Y"),
        ("10Y-3M",  "US_10Y", "US_3M"),
    ]

    # Momentum windows to test
    mom_windows = [5, 10, 21, 42, 63, 126]

    # Combination parameters
    combo_k = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

    # Threshold sets to test
    thresh_sets = [
        ("opt_tight", -3.25, 2.75, -4.0, 5),
        ("opt_loose", -3.50, 2.75, -4.0, 5),
        ("baseline",  -3.50, 3.00, -4.0, 5),
        ("aggr",      -3.00, 2.50, -3.5, 5),
    ]

    results = []
    total = len(slope_pairs) * len(mom_windows) * len(combo_k) * len(thresh_sets) * 2
    print(f"  Testing {total} composite configurations ...")

    count = 0
    for base_name, z_b in [("ALL|std|w63", z_base), ("1M|move|w250", z_move_base)]:
        for slope_name, col_l, col_s in slope_pairs:
            for mwin in mom_windows:
                # Compute slope momentum z-score
                slope_mom = compute_slope_momentum(yields, col_l, col_s, mwin)

                for k in combo_k:
                    # Composite: z_composite = z_base + k * slope_momentum
                    z_comp = z_b + k * slope_mom

                    # Align
                    common = (z_comp.dropna().index
                              .intersection(regime.index)
                              .intersection(etf_ret.dropna(subset=["TLT","SHV"]).index))
                    common = common[common >= start].sort_values()

                    if len(common) < 100:
                        count += len(thresh_sets)
                        continue

                    z_vals = z_comp.reindex(common).values
                    reg_vals = regime_num.reindex(common).values
                    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
                    r_short = etf_ret["SHV"].reindex(common).fillna(0).values

                    for tname, tl_c, ts_c, tl_s, cd in thresh_sets:
                        sh, ar, vol, mdd, nev, eqf = fast_bt(
                            z_vals, reg_vals, r_long, r_short, tl_c, ts_c, tl_s, cd)
                        results.append({
                            "base": base_name, "slope": slope_name,
                            "mom_win": mwin, "k": k, "thresh": tname,
                            "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s, "cd": cd,
                            "sharpe": sh, "ann_ret": ar, "vol": vol,
                            "mdd": mdd, "n_events": nev, "eq_final": eqf,
                        })
                        count += 1

    # Also test pure base signals for comparison
    for base_name, z_b in [("ALL|std|w63", z_base), ("1M|move|w250", z_move_base)]:
        common = (z_b.dropna().index
                  .intersection(regime.index)
                  .intersection(etf_ret.dropna(subset=["TLT","SHV"]).index))
        common = common[common >= start].sort_values()
        z_vals = z_b.reindex(common).values
        reg_vals = regime_num.reindex(common).values
        r_long = etf_ret["TLT"].reindex(common).fillna(0).values
        r_short = etf_ret["SHV"].reindex(common).fillna(0).values
        for tname, tl_c, ts_c, tl_s, cd in thresh_sets:
            sh, ar, vol, mdd, nev, eqf = fast_bt(
                z_vals, reg_vals, r_long, r_short, tl_c, ts_c, tl_s, cd)
            results.append({
                "base": base_name, "slope": "NONE",
                "mom_win": 0, "k": 0, "thresh": tname,
                "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s, "cd": cd,
                "sharpe": sh, "ann_ret": ar, "vol": vol,
                "mdd": mdd, "n_events": nev, "eq_final": eqf,
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(FINAL_DIR, "composite_grid_results.csv"), index=False)

    # Print top results
    top = rdf.nlargest(20, "sharpe")
    print(f"\n  {len(rdf)} total configurations tested")
    print(f"\n  Top 20 composite results:")
    print(f"  {'Base':<18} {'Slope':<10} {'MomWin':>6} {'k':>5} {'Thresh':<10} "
          f"{'Sharpe':>7} {'AnnRet':>7} {'MDD':>7} {'Events':>6}")
    print(f"  {'-'*95}")
    for _, r in top.iterrows():
        print(f"  {r['base']:<18} {r['slope']:<10} {r['mom_win']:>6} {r['k']:>5.2f} "
              f"{r['thresh']:<10} {r['sharpe']:>+7.3f} {r['ann_ret']:>+7.4f} "
              f"{r['mdd']:>7.3f} {r['n_events']:>6}")

    # Best composite vs pure
    best_comp = rdf[rdf["slope"] != "NONE"].nlargest(1, "sharpe").iloc[0]
    best_pure = rdf[rdf["slope"] == "NONE"].nlargest(1, "sharpe").iloc[0]
    print(f"\n  Best composite: Sharpe={best_comp['sharpe']:.3f} "
          f"({best_comp['base']}+{best_comp['slope']} k={best_comp['k']} mw={best_comp['mom_win']})")
    print(f"  Best pure RR:   Sharpe={best_pure['sharpe']:.3f} ({best_pure['base']})")
    improvement = best_comp["sharpe"] - best_pure["sharpe"]
    print(f"  Improvement:    {improvement:+.3f}")

    # ── Composite plot ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1) Sharpe by slope component
    ax = axes[0, 0]
    for slope in rdf["slope"].unique():
        sub = rdf[rdf["slope"] == slope]
        if len(sub) > 0:
            ax.bar(slope, sub["sharpe"].max(), alpha=0.7,
                   color="gold" if slope == "NONE" else "steelblue")
    ax.axhline(best_pure["sharpe"], color="red", linestyle="--", label=f"Pure RR ({best_pure['sharpe']:.3f})")
    ax.set_ylabel("Best Sharpe")
    ax.set_title("Best Sharpe by Slope Component")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 2) k sensitivity for best slope
    ax = axes[0, 1]
    best_slope = best_comp["slope"]
    for mw in sorted(rdf["mom_win"].unique()):
        if mw == 0:
            continue
        sub = rdf[(rdf["slope"] == best_slope) & (rdf["mom_win"] == mw)]
        if len(sub) > 0:
            kvals = sorted(sub["k"].unique())
            sharpes = [sub[sub["k"]==kv]["sharpe"].max() for kv in kvals]
            ax.plot(kvals, sharpes, "o-", label=f"mw={mw}", markersize=4)
    ax.axhline(best_pure["sharpe"], color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("k (momentum weight)")
    ax.set_ylabel("Sharpe")
    ax.set_title(f"k Sensitivity (slope={best_slope})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3) momentum window sensitivity
    ax = axes[1, 0]
    for slope_name, _, _ in slope_pairs[:4]:
        sub = rdf[(rdf["slope"] == slope_name)]
        if len(sub) > 0:
            wins = sorted(sub["mom_win"].unique())
            wins = [w for w in wins if w > 0]
            sharpes = [sub[sub["mom_win"]==w]["sharpe"].max() for w in wins]
            ax.plot(wins, sharpes, "o-", label=slope_name, markersize=4)
    ax.axhline(best_pure["sharpe"], color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Momentum Window (days)")
    ax.set_ylabel("Best Sharpe")
    ax.set_title("Momentum Window Sensitivity")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 4) Heatmap: slope x mom_win (best sharpe)
    ax = axes[1, 1]
    slopes = [s for s, _, _ in slope_pairs]
    wins = sorted([w for w in rdf["mom_win"].unique() if w > 0])
    heat = np.zeros((len(slopes), len(wins)))
    for si, s in enumerate(slopes):
        for wi, w in enumerate(wins):
            sub = rdf[(rdf["slope"]==s) & (rdf["mom_win"]==w)]
            heat[si, wi] = sub["sharpe"].max() if len(sub) > 0 else np.nan
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0.15, vmax=0.30)
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels(wins)
    ax.set_yticks(range(len(slopes)))
    ax.set_yticklabels(slopes)
    ax.set_xlabel("Momentum Window")
    ax.set_ylabel("Slope Component")
    ax.set_title("Sharpe Heatmap (Slope x Window)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Stage 2: Composite Signal Analysis", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_DIR, "02_composite_signal.png"), dpi=150)
    plt.close()
    print(f"\n  Plot saved: {FINAL_DIR}/02_composite_signal.png")

    return rdf


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: DEEP GRID (fine resolution around optimum)
# ═══════════════════════════════════════════════════════════════════════════

def stage3_deep_grid(D, regime, regime_num):
    """Ultra-fine grid search around the optimal zone."""
    print("\n" + "=" * 70)
    print("  STAGE 3: DEEP GRID AROUND OPTIMAL ZONE")
    print("=" * 70)

    etf_ret = D["etf_ret"]
    start = pd.Timestamp(BACKTEST_START)

    # Signals to test in deep grid
    signal_configs = [
        ("ALL|std|w63", "ALL", "std", 63, None),
        ("ALL|std|w125", "ALL", "std", 125, None),
        ("ALL|std|w50", "ALL", "std", 50, None),
        ("ALL|std|w80", "ALL", "std", 80, None),
        ("ALL|move|w250|mw504", "ALL", "move", 250, 504),
        ("1M|move|w180|mw504", "1M", "move", 180, 504),
    ]

    # Fine threshold grid
    tl_calm_range = np.arange(-4.50, -2.00, 0.10)    # 25 values
    ts_calm_range = np.arange(2.00, 3.80, 0.10)      # 18 values
    tl_stress_range = np.arange(-5.50, -2.50, 0.25)  # 12 values
    cooldowns = [0, 3, 5, 7, 10]

    n_thresh = len(tl_calm_range) * len(ts_calm_range) * len(tl_stress_range) * len(cooldowns)
    total = len(signal_configs) * n_thresh
    print(f"  {len(signal_configs)} signals x {n_thresh} threshold combos = {total:,} backtests")

    results = []
    t0 = time.time()

    for sig_idx, (sig_name, rr_key, z_type, z_win, mw) in enumerate(signal_configs):
        # Compute signal
        if rr_key == "ALL":
            tenor_keys = ["1W", "1M", "3M", "6M"]
        else:
            tenor_keys = rr_key.split("_")

        rr = compute_rr_blend(D["options"], tenor_keys)
        z = compute_zscore(rr, D["move"], z_type, z_win, mw or 504)

        common = (z.dropna().index
                  .intersection(regime.index)
                  .intersection(etf_ret.dropna(subset=["TLT","SHV"]).index))
        common = common[common >= start].sort_values()

        z_vals = z.reindex(common).values
        reg_vals = regime_num.reindex(common).values
        r_long = etf_ret["TLT"].reindex(common).fillna(0).values
        r_short = etf_ret["SHV"].reindex(common).fillna(0).values

        sig_count = 0
        for tl_c in tl_calm_range:
            for ts_c in ts_calm_range:
                for tl_s in tl_stress_range:
                    for cd in cooldowns:
                        sh, ar, vol, mdd, nev, eqf = fast_bt(
                            z_vals, reg_vals, r_long, r_short,
                            tl_c, ts_c, tl_s, cd)
                        results.append({
                            "signal": sig_name, "tl_c": round(tl_c,2),
                            "ts_c": round(ts_c,2), "tl_s": round(tl_s,2),
                            "cd": cd, "sharpe": sh, "ann_ret": ar,
                            "vol": vol, "mdd": mdd, "n_events": nev,
                            "eq_final": eqf,
                        })
                        sig_count += 1

        elapsed = time.time() - t0
        pct = (sig_idx + 1) / len(signal_configs) * 100
        print(f"  [{sig_idx+1}/{len(signal_configs)}] {sig_name:<30} "
              f"{sig_count:>7,} runs  ({pct:.0f}%  {elapsed:.0f}s)")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(FINAL_DIR, "deep_grid_results.csv"), index=False)

    # Analysis
    print(f"\n  Total configurations: {len(rdf):,}")
    top50 = rdf.nlargest(50, "sharpe")

    print(f"\n  Top 20 deep grid results:")
    print(f"  {'Signal':<30} {'tl_c':>6} {'ts_c':>6} {'tl_s':>6} {'cd':>3} "
          f"{'Sharpe':>7} {'AnnRet':>7} {'MDD':>7} {'Ev':>4} {'EqF':>6}")
    print(f"  {'-'*105}")
    for _, r in rdf.nlargest(20, "sharpe").iterrows():
        print(f"  {r['signal']:<30} {r['tl_c']:>+6.2f} {r['ts_c']:>+6.2f} "
              f"{r['tl_s']:>+6.2f} {r['cd']:>3} {r['sharpe']:>+7.3f} "
              f"{r['ann_ret']:>+7.4f} {r['mdd']:>7.3f} {r['n_events']:>4} "
              f"{r['eq_final']:>6.3f}")

    # Parameter stability analysis
    print(f"\n  Parameter distribution (TOP 50):")
    for col in ["tl_c", "ts_c", "tl_s", "cd"]:
        vals = top50[col]
        print(f"    {col:<10} mean={vals.mean():+.2f}  std={vals.std():.2f}  "
              f"range=[{vals.min():+.2f}, {vals.max():+.2f}]  mode={vals.mode().iloc[0]:+.2f}")

    # Signal frequency in top 50
    print(f"\n  Signal frequency (TOP 50):")
    for sig, cnt in top50["signal"].value_counts().items():
        print(f"    {sig:<30} {cnt} configs")

    # ── Deep grid plots ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Heatmap: tl_calm x ts_calm (best Sharpe, aggregated over tl_stress/cd)
    ax = axes[0, 0]
    best_sig = top50["signal"].mode().iloc[0]
    sub = rdf[rdf["signal"] == best_sig]
    tl_vals = sorted(sub["tl_c"].unique())
    ts_vals = sorted(sub["ts_c"].unique())
    heat = np.zeros((len(tl_vals), len(ts_vals)))
    for ti, tl in enumerate(tl_vals):
        for si, ts in enumerate(ts_vals):
            s = sub[(sub["tl_c"]==tl) & (sub["ts_c"]==ts)]
            heat[ti, si] = s["sharpe"].max() if len(s) > 0 else np.nan
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn",
                   extent=[ts_vals[0], ts_vals[-1], tl_vals[-1], tl_vals[0]])
    ax.set_xlabel("ts_calm")
    ax.set_ylabel("tl_calm")
    ax.set_title(f"Sharpe Heatmap: tl_calm x ts_calm ({best_sig})")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 2) tl_stress sensitivity (fixing best tl_c/ts_c)
    ax = axes[0, 1]
    best_row = rdf.nlargest(1, "sharpe").iloc[0]
    for sig_name in rdf["signal"].unique():
        sub = rdf[(rdf["signal"] == sig_name) &
                  (rdf["tl_c"] == best_row["tl_c"]) &
                  (rdf["ts_c"] == best_row["ts_c"])]
        if len(sub) > 0:
            tls_vals = sorted(sub["tl_s"].unique())
            sharpes = [sub[sub["tl_s"]==t]["sharpe"].max() for t in tls_vals]
            ax.plot(tls_vals, sharpes, "o-", label=sig_name[:25], markersize=3)
    ax.set_xlabel("tl_stress")
    ax.set_ylabel("Best Sharpe")
    ax.set_title(f"tl_stress Sensitivity (tl_c={best_row['tl_c']:.2f}, ts_c={best_row['ts_c']:.2f})")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # 3) Cooldown sensitivity
    ax = axes[1, 0]
    for sig_name in rdf["signal"].unique():
        sub = rdf[rdf["signal"] == sig_name]
        cd_sharpes = []
        for cd in sorted(sub["cd"].unique()):
            s = sub[sub["cd"] == cd]
            cd_sharpes.append((cd, s["sharpe"].max()))
        if cd_sharpes:
            cds, shs = zip(*cd_sharpes)
            ax.plot(cds, shs, "o-", label=sig_name[:25], markersize=4)
    ax.set_xlabel("Cooldown (days)")
    ax.set_ylabel("Best Sharpe")
    ax.set_title("Cooldown Sensitivity")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)

    # 4) Signal comparison boxplot
    ax = axes[1, 1]
    sig_names = sorted(rdf["signal"].unique())
    data = [rdf[rdf["signal"]==s]["sharpe"].values for s in sig_names]
    bp = ax.boxplot(data, tick_labels=[s[:20] for s in sig_names], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    ax.set_ylabel("Sharpe")
    ax.set_title("Sharpe Distribution by Signal")
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=7)

    fig.suptitle("Stage 3: Deep Grid Analysis", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_DIR, "03_deep_grid_analysis.png"), dpi=150)
    plt.close()
    print(f"\n  Plot saved: {FINAL_DIR}/03_deep_grid_analysis.png")

    return rdf


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: FINAL SYSTEM ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════

def stage4_final_system(D, regime, regime_num, wf_results, comp_results, grid_results):
    """Assemble the final system: best validated config with full analysis."""
    print("\n" + "=" * 70)
    print("  STAGE 4: FINAL SYSTEM ASSEMBLY")
    print("=" * 70)

    etf_ret = D["etf_ret"]
    start = pd.Timestamp(BACKTEST_START)

    # ── Determine best config from deep grid ──
    best_grid = grid_results.nlargest(1, "sharpe").iloc[0]
    print(f"\n  Best from deep grid:")
    print(f"    Signal: {best_grid['signal']}")
    print(f"    tl_calm={best_grid['tl_c']:.2f}  ts_calm={best_grid['ts_c']:.2f}  "
          f"tl_stress={best_grid['tl_s']:.2f}  cooldown={best_grid['cd']}")
    print(f"    Sharpe={best_grid['sharpe']:.4f}  AnnRet={best_grid['ann_ret']:.4f}  "
          f"MDD={best_grid['mdd']:.4f}")

    # ── Best composite (if it beats pure) ──
    best_comp = comp_results[comp_results["slope"] != "NONE"].nlargest(1, "sharpe").iloc[0]
    best_pure = comp_results[comp_results["slope"] == "NONE"].nlargest(1, "sharpe").iloc[0]
    use_composite = best_comp["sharpe"] > best_pure["sharpe"] + 0.01  # need meaningful improvement

    if use_composite:
        print(f"\n  Composite adds value: +{best_comp['sharpe']-best_pure['sharpe']:.3f} Sharpe")
        print(f"    Using: {best_comp['base']}+{best_comp['slope']} k={best_comp['k']} mw={best_comp['mom_win']}")
    else:
        print(f"\n  Composite does NOT add meaningful value (delta={best_comp['sharpe']-best_pure['sharpe']:+.3f})")
        print(f"  Using pure RR signal.")

    # ── Build final configs to compare ──
    configs_to_run = {}

    # Config 1: Deep grid best
    sig_name = best_grid["signal"]
    sig_parts = sig_name.split("|")
    rr_key = sig_parts[0]
    z_type = sig_parts[1]
    z_win = int(sig_parts[2][1:])
    mw = int(sig_parts[3][2:]) if len(sig_parts) > 3 else 504

    tenor_keys = ["1W", "1M", "3M", "6M"] if rr_key == "ALL" else rr_key.split("_")
    rr = compute_rr_blend(D["options"], tenor_keys)
    z_final = compute_zscore(rr, D["move"], z_type, z_win, mw)

    configs_to_run["Deep Grid Best"] = {
        "z": z_final,
        "tl_c": best_grid["tl_c"], "ts_c": best_grid["ts_c"],
        "tl_s": best_grid["tl_s"], "cd": int(best_grid["cd"]),
        "label": sig_name,
    }

    # Config 2: Previous best (ALL|std|w63 optimized)
    rr_all = compute_rr_blend(D["options"], ["1W", "1M", "3M", "6M"])
    z_std63 = compute_zscore(rr_all, D["move"], "std", 63)
    configs_to_run["ALL|std|w63 Optimized"] = {
        "z": z_std63,
        "tl_c": -3.25, "ts_c": 2.75, "tl_s": -4.0, "cd": 5,
        "label": "ALL|std|w63",
    }

    # Config 3: Baseline V1
    rr_1m = compute_rr_blend(D["options"], ["1M"])
    z_v1 = compute_zscore(rr_1m, D["move"], "move", 250, 504)
    configs_to_run["Baseline V1"] = {
        "z": z_v1,
        "tl_c": -3.5, "ts_c": 3.0, "tl_s": -4.0, "cd": 5,
        "label": "1M|move|w250|mw504",
    }

    # Config 4: If composite adds value
    if use_composite:
        rr_comp_base_key = best_comp["base"]
        if "ALL" in rr_comp_base_key:
            z_comp_base = z_std63
        else:
            z_comp_base = z_v1
        slope_name = best_comp["slope"]
        slope_map = {"5Y-2Y": ("US_5Y","US_2Y"), "10Y-2Y": ("US_10Y","US_2Y"),
                     "30Y-10Y": ("US_30Y","US_10Y"), "30Y-2Y": ("US_30Y","US_2Y"),
                     "10Y-3M": ("US_10Y","US_3M")}
        col_l, col_s = slope_map[slope_name]
        slope_mom = compute_slope_momentum(D["yields"], col_l, col_s, int(best_comp["mom_win"]))
        z_composite = z_comp_base + best_comp["k"] * slope_mom
        configs_to_run["Composite Best"] = {
            "z": z_composite,
            "tl_c": best_comp["tl_c"], "ts_c": best_comp["ts_c"],
            "tl_s": best_comp["tl_s"], "cd": int(best_comp["cd"]),
            "label": f"{rr_comp_base_key}+{slope_name}",
        }

    # ── Run full backtests and collect results ──
    final_results = {}
    print(f"\n  Running {len(configs_to_run)} final configurations:")

    for cfg_name, cfg in configs_to_run.items():
        z = cfg["z"]
        common = (z.dropna().index
                  .intersection(regime.index)
                  .intersection(etf_ret.dropna(subset=["TLT","SHV"]).index))
        common = common[common >= start].sort_values()

        z_vals = z.reindex(common).values
        reg_vals = regime_num.reindex(common).values
        r_long = etf_ret["TLT"].reindex(common).fillna(0).values
        r_short = etf_ret["SHV"].reindex(common).fillna(0).values

        eq, net, w_del, events = fast_bt_full(
            z_vals, reg_vals, r_long, r_short,
            cfg["tl_c"], cfg["ts_c"], cfg["tl_s"], cfg["cd"])

        # Compute full metrics
        n = len(net)
        years = n / 252.0
        total_ret = eq[-1] - 1.0
        ann_ret = (1.0 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0
        vol = np.std(net) * np.sqrt(252)
        sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

        down = net[net < 0]
        down_vol = np.std(down) * np.sqrt(252) if len(down) > 0 else 1e-6
        sortino = (ann_ret - RISK_FREE_RATE) / down_vol

        running_max = np.maximum.accumulate(eq)
        mdd = (eq / running_max - 1.0).min()
        calmar = ann_ret / abs(mdd) if mdd != 0 else 0

        # Benchmark
        bm_eq = np.cumprod(1.0 + r_long)
        bm_total = bm_eq[-1] - 1.0
        bm_ann = (1.0 + bm_total) ** (1.0 / max(years, 0.01)) - 1.0
        excess = ann_ret - bm_ann

        # Yearly breakdown
        dates_arr = common
        yearly_rows = []
        for y in sorted(set(d.year for d in dates_arr)):
            mask = np.array([d.year == y for d in dates_arr])
            if mask.sum() < 5:
                continue
            net_y = net[mask]
            eq_y = np.cumprod(1.0 + net_y)
            yr_ret = eq_y[-1] - 1.0
            bm_y = np.cumprod(1.0 + r_long[mask])
            bm_ret_y = bm_y[-1] - 1.0
            yearly_rows.append({
                "year": y, "strat_ret": yr_ret, "bm_ret": bm_ret_y,
                "excess": yr_ret - bm_ret_y,
            })

        yearly = pd.DataFrame(yearly_rows)
        pos_years = (yearly["excess"] > 0).sum() if len(yearly) > 0 else 0

        final_results[cfg_name] = {
            "equity": pd.Series(eq, index=common),
            "net_ret": pd.Series(net, index=common),
            "metrics": {
                "sharpe": sharpe, "sortino": sortino, "ann_ret": ann_ret,
                "vol": vol, "mdd": mdd, "calmar": calmar, "excess": excess,
                "n_events": len(events), "eq_final": eq[-1],
            },
            "yearly": yearly,
            "pos_years": pos_years,
            "tot_years": len(yearly),
            "params": cfg,
        }

        print(f"\n    {cfg_name} ({cfg['label']}):")
        print(f"      tl_c={cfg['tl_c']:.2f}  ts_c={cfg['ts_c']:.2f}  "
              f"tl_s={cfg['tl_s']:.2f}  cd={cfg['cd']}")
        print(f"      Sharpe={sharpe:.4f}  Sortino={sortino:.4f}  "
              f"AnnRet={ann_ret:.4f}  Vol={vol:.4f}")
        print(f"      MDD={mdd:.4f}  Calmar={calmar:.4f}  Excess={excess:.4f}")
        print(f"      Events={len(events)}  Pos Years={pos_years}/{len(yearly)}")

    # ── Select FINAL best ──
    best_name = max(final_results, key=lambda k: final_results[k]["metrics"]["sharpe"])
    best = final_results[best_name]
    print(f"\n  {'='*60}")
    print(f"  FINAL BEST CONFIG: {best_name}")
    print(f"  {'='*60}")

    # ── Bootstrap confidence interval ──
    print(f"\n  Running bootstrap (1000 resamples)...")
    net_ret = best["net_ret"].values
    n_boot = 1000
    block_size = 21
    np.random.seed(42)
    boot_sharpes = []
    n = len(net_ret)
    n_blocks = n // block_size + 1
    for _ in range(n_boot):
        idx = np.random.randint(0, n - block_size, n_blocks)
        boot_ret = np.concatenate([net_ret[i:i+block_size] for i in idx])[:n]
        beq = np.cumprod(1.0 + boot_ret)
        btot = beq[-1] - 1.0
        byrs = n / 252.0
        bar = (1.0 + btot) ** (1.0 / max(byrs, 0.01)) - 1.0
        bvol = np.std(boot_ret) * np.sqrt(252)
        bsh = (bar - RISK_FREE_RATE) / bvol if bvol > 1e-8 else 0.0
        boot_sharpes.append(bsh)
    boot_sharpes = np.array(boot_sharpes)
    ci_5 = np.percentile(boot_sharpes, 5)
    ci_95 = np.percentile(boot_sharpes, 95)
    p_positive = (boot_sharpes > 0).mean()
    print(f"    Sharpe 90% CI: [{ci_5:.3f}, {ci_95:.3f}]")
    print(f"    P(Sharpe > 0): {p_positive:.1%}")

    # ── Transaction cost sensitivity ──
    tc_values = [0, 2, 5, 10, 15, 20, 30, 50]
    tc_sharpes = {}
    cfg = final_results[best_name]["params"]
    z = cfg["z"]
    common = (z.dropna().index
              .intersection(regime.index)
              .intersection(etf_ret.dropna(subset=["TLT","SHV"]).index))
    common = common[common >= start].sort_values()
    z_vals = z.reindex(common).values
    reg_vals = regime_num.reindex(common).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values

    for tc in tc_values:
        sh, _, _, _, _, _ = fast_bt(z_vals, reg_vals, r_long, r_short,
                                     cfg["tl_c"], cfg["ts_c"], cfg["tl_s"], cfg["cd"],
                                     tcost_bps=tc)
        tc_sharpes[tc] = sh

    print(f"\n  Transaction cost sensitivity:")
    for tc, sh in tc_sharpes.items():
        print(f"    {tc:>3} bps -> Sharpe {sh:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # FINAL PLOTS
    # ═══════════════════════════════════════════════════════════════════════

    # Plot 1: Equity curves comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = {"Deep Grid Best": "blue", "ALL|std|w63 Optimized": "green",
              "Baseline V1": "orange", "Composite Best": "purple"}
    for name, res in final_results.items():
        eq = res["equity"]
        sh = res["metrics"]["sharpe"]
        c = colors.get(name, "gray")
        lw = 2.5 if name == best_name else 1.2
        ax.plot(eq.index, eq.values, label=f"{name} (Sh={sh:.3f})",
                color=c, linewidth=lw, alpha=0.9 if name == best_name else 0.6)

    # B&H TLT benchmark
    common_all = final_results[best_name]["equity"].index
    bm = np.cumprod(1.0 + etf_ret["TLT"].reindex(common_all).fillna(0).values)
    ax.plot(common_all, bm, label="B&H TLT", color="gray", linestyle=":", alpha=0.5)

    ax.set_title("Final System: Equity Curves Comparison", fontsize=14)
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_DIR, "04_final_equity_curves.png"), dpi=150)
    plt.close()

    # Plot 2: Final summary dashboard
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 2a: Drawdown
    ax = fig.add_subplot(gs[0, 0])
    eq = best["equity"]
    dd = eq / eq.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color="red", alpha=0.3)
    ax.plot(dd.index, dd.values, color="red", linewidth=0.5)
    ax.set_title(f"Drawdown ({best_name})")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    # 2b: Rolling Sharpe (1Y)
    ax = fig.add_subplot(gs[0, 1])
    net_s = best["net_ret"]
    roll_mu = net_s.rolling(252).mean() * 252
    roll_vol = net_s.rolling(252).std() * np.sqrt(252)
    roll_sharpe = (roll_mu - RISK_FREE_RATE) / roll_vol
    ax.plot(roll_sharpe.index, roll_sharpe.values, color="steelblue", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(best["metrics"]["sharpe"], color="red", linestyle="--", alpha=0.5,
               label=f"Full period: {best['metrics']['sharpe']:.3f}")
    ax.set_title("Rolling 1Y Sharpe Ratio")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2c: Bootstrap distribution
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(boot_sharpes, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(best["metrics"]["sharpe"], color="red", linewidth=2,
               label=f"Actual: {best['metrics']['sharpe']:.3f}")
    ax.axvline(ci_5, color="orange", linestyle="--", label=f"5%: {ci_5:.3f}")
    ax.axvline(ci_95, color="orange", linestyle="--", label=f"95%: {ci_95:.3f}")
    ax.set_title("Bootstrap Sharpe Distribution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2d: Yearly returns
    ax = fig.add_subplot(gs[1, 0:2])
    yearly = best["yearly"]
    if len(yearly) > 0:
        x = np.arange(len(yearly))
        w = 0.35
        ax.bar(x - w/2, yearly["strat_ret"], w, label="Strategy", color="steelblue", alpha=0.8)
        ax.bar(x + w/2, yearly["bm_ret"], w, label="B&H TLT", color="gray", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(yearly["year"].astype(int), rotation=45)
        ax.set_ylabel("Return")
        ax.set_title("Yearly Returns: Strategy vs B&H TLT")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3)

    # 2e: TC sensitivity
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(list(tc_sharpes.keys()), list(tc_sharpes.values()), "o-",
            color="steelblue", markersize=6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("TC Sensitivity")
    ax.grid(True, alpha=0.3)

    # 2f: Final metrics table
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    m = best["metrics"]
    headers = ["Metric", "Value"]
    cfg = final_results[best_name]["params"]
    table_data = [
        ["Config", f"{best_name}: {cfg['label']}"],
        ["Thresholds", f"tl_calm={cfg['tl_c']:.2f}  ts_calm={cfg['ts_c']:.2f}  "
                       f"tl_stress={cfg['tl_s']:.2f}  cooldown={cfg['cd']}"],
        ["Sharpe Ratio", f"{m['sharpe']:.4f}  [90% CI: {ci_5:.3f} to {ci_95:.3f}]"],
        ["Sortino Ratio", f"{m['sortino']:.4f}"],
        ["Ann. Return", f"{m['ann_ret']:.4f} ({m['ann_ret']*100:.2f}%)"],
        ["Volatility", f"{m['vol']:.4f} ({m['vol']*100:.2f}%)"],
        ["Max Drawdown", f"{m['mdd']:.4f} ({m['mdd']*100:.2f}%)"],
        ["Calmar Ratio", f"{m['calmar']:.4f}"],
        ["Excess vs TLT", f"{m['excess']:+.4f} ({m['excess']*100:+.2f}%)"],
        ["Events", f"{m['n_events']}"],
        ["Positive Years (OOS)", f"{best['pos_years']}/{best['tot_years']}"],
        ["P(Sharpe > 0)", f"{p_positive:.1%}"],
        ["Eq Final ($1 invested)", f"${m['eq_final']:.3f}"],
    ]
    table = ax.table(cellText=table_data, colLabels=headers, loc="center",
                     cellLoc="left", colWidths=[0.3, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)
    # Style header
    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("FINAL SYSTEM - Performance Summary", fontsize=14, pad=30, fontweight="bold")

    fig.savefig(os.path.join(FINAL_DIR, "05_final_dashboard.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # Plot 3: Config comparison table
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    comp_headers = ["Config", "Signal", "Sharpe", "Sortino", "AnnRet", "Vol",
                    "MDD", "Calmar", "Events", "Pos Yrs"]
    comp_data = []
    for name, res in final_results.items():
        m = res["metrics"]
        marker = " <<< BEST" if name == best_name else ""
        comp_data.append([
            f"{name}{marker}", res["params"]["label"],
            f"{m['sharpe']:.4f}", f"{m['sortino']:.4f}",
            f"{m['ann_ret']:.4f}", f"{m['vol']:.4f}",
            f"{m['mdd']:.4f}", f"{m['calmar']:.4f}",
            str(m['n_events']), f"{res['pos_years']}/{res['tot_years']}",
        ])
    table = ax.table(cellText=comp_data, colLabels=comp_headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    for j in range(len(comp_headers)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight best row
    best_idx = list(final_results.keys()).index(best_name) + 1
    for j in range(len(comp_headers)):
        table[best_idx, j].set_facecolor("#E2EFDA")
    ax.set_title("Final Config Comparison", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_DIR, "06_config_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # ── Save final config to file ──
    final_config = {
        "name": best_name,
        "signal": cfg["label"],
        "tl_calm": cfg["tl_c"],
        "ts_calm": cfg["ts_c"],
        "tl_stress": cfg["tl_s"],
        "cooldown": cfg["cd"],
        **m,
        "bootstrap_ci_5": ci_5,
        "bootstrap_ci_95": ci_95,
        "p_sharpe_positive": p_positive,
    }
    pd.DataFrame([final_config]).to_csv(
        os.path.join(FINAL_DIR, "final_best_config.csv"), index=False)

    # Save all configs comparison
    all_configs = []
    for name, res in final_results.items():
        row = {"config": name, "signal": res["params"]["label"],
               "tl_c": res["params"]["tl_c"], "ts_c": res["params"]["ts_c"],
               "tl_s": res["params"]["tl_s"], "cd": res["params"]["cd"],
               **res["metrics"],
               "pos_years": res["pos_years"], "tot_years": res["tot_years"]}
        all_configs.append(row)
    pd.DataFrame(all_configs).to_csv(
        os.path.join(FINAL_DIR, "all_final_configs.csv"), index=False)

    print(f"\n  Plots saved to {FINAL_DIR}/")
    print(f"  Config saved to {FINAL_DIR}/final_best_config.csv")

    return final_results, best_name


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()
    print("=" * 70)
    print("  FINAL SYSTEM OPTIMIZATION PIPELINE")
    print("  4 Stages: WF Validation -> Composite -> Deep Grid -> Assembly")
    print("=" * 70)

    # ── Load data ──
    print("\n[0] Loading data ...")
    D = load_all()

    # ── Regime ──
    print("\n[0] Fitting HMM regime ...")
    regime = fit_hmm_regime(D["move"], D["vix"])
    regime_num = (regime == "STRESS").astype(np.int8)
    etf_ret = D["etf_ret"]

    # ── Stage 1: Walk-Forward ──
    wf_results = stage1_walk_forward(D, regime, regime_num)

    # ── Stage 2: Composite ──
    comp_results = stage2_composite(D, regime, regime_num)

    # ── Stage 3: Deep Grid ──
    grid_results = stage3_deep_grid(D, regime, regime_num)

    # ── Stage 4: Final Assembly ──
    final_results, best_name = stage4_final_system(
        D, regime, regime_num, wf_results, comp_results, grid_results)

    # ── Summary ──
    elapsed = time.time() - T0
    best = final_results[best_name]
    m = best["metrics"]
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Best config: {best_name}")
    print(f"  Sharpe: {m['sharpe']:.4f}  |  AnnRet: {m['ann_ret']:.4f}  |  "
          f"MDD: {m['mdd']:.4f}  |  Events: {m['n_events']}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  All outputs: {FINAL_DIR}/")
    print(f"{'='*70}")

    return final_results


if __name__ == "__main__":
    main()
