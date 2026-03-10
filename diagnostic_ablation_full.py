"""
diagnostic_ablation_full.py - Ablazione completa: tutte le combinazioni di componenti.

z_final = z_base + k_slope*z_slope + k_yc*z_yc + k_term*z_term

Testa tutte le 8 combinazioni on/off di (slope, YC, term),
con grid search indipendente per ciascuna.

Output in: output/ablation_full/
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

ANN = 252
OUT_DIR = os.path.join(SAVE_DIR, "ablation_full")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Signal builders (same as versione_rr_term) ──

def build_rr_blend(D):
    options = D["options"]
    parts = []
    for tenor in ["1W", "1M", "3M", "6M"]:
        p, c = RR_TENORS[tenor]
        rr = pd.to_numeric(options[p], errors="coerce") - pd.to_numeric(options[c], errors="coerce")
        parts.append(rr)
    return pd.concat(parts, axis=1).mean(axis=1)

def build_rr_individual(D):
    options = D["options"]
    out = {}
    for tenor in ["1W", "1M", "3M", "6M"]:
        p, c = RR_TENORS[tenor]
        out[tenor] = pd.to_numeric(options[p], errors="coerce") - pd.to_numeric(options[c], errors="coerce")
    return out

def build_slope_zscore(D, mom_window=126, z_window=252, z_min=63):
    yld = D["yields"]
    slope = yld["US_10Y"] - yld["US_2Y"]
    ds = slope.diff(mom_window)
    mu = ds.rolling(z_window, min_periods=z_min).mean()
    sd = ds.rolling(z_window, min_periods=z_min).std().replace(0, np.nan)
    return (ds - mu) / sd

def build_yc_signal(D, change_window=42, z_window=252, z_min_per=20):
    yld = D["yields"]
    level = (yld["US_10Y"] + yld["US_30Y"]) / 2
    spread = yld["US_30Y"] - yld["US_10Y"]
    dl = level.diff(change_window)
    ds = spread.diff(change_window)
    regime = pd.Series("UNDEFINED", index=yld.index)
    regime[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    regime[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    regime[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    regime[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    score_map = {"BULL_FLAT": 2, "BULL_STEEP": 1, "UNDEFINED": 0, "BEAR_FLAT": -1, "BEAR_STEEP": -2}
    sc = regime.map(score_map).astype(float)
    mu = sc.rolling(z_window, min_periods=z_min_per).mean()
    sd = sc.rolling(z_window, min_periods=z_min_per).std().replace(0, np.nan)
    return (sc - mu) / sd

def build_term_zscore(rr_a, rr_b, z_window=126, z_min=30):
    spread = rr_a - rr_b
    mu = spread.rolling(z_window, min_periods=z_min).mean()
    sd = spread.rolling(z_window, min_periods=z_min).std().replace(0, np.nan)
    return (spread - mu) / sd


# ── Backtest ──

def fast_bt(z_arr, reg_arr, r_long, r_short, year_arr,
            tl_c, ts_c, tl_s, adapt_days=0, adapt_raise=0.0,
            max_hold_days=0, cooldown=5, delay=1, w_init=0.5):
    n = len(z_arr)
    weight = np.full(n, w_init)
    cur_w = w_init
    last_ev = -cooldown - 1
    n_events = 0
    hold_days = 0

    for i in range(n):
        zv = z_arr[i]
        if np.isnan(zv):
            weight[i] = cur_w
            hold_days += 1
            continue

        ts_c_eff = ts_c
        if adapt_days > 0 and cur_w > 0.7 and hold_days >= adapt_days:
            ts_c_eff = ts_c + adapt_raise

        if max_hold_days > 0 and hold_days >= max_hold_days and abs(cur_w - 0.5) > 0.01:
            cur_w = 0.5; hold_days = 0; n_events += 1; last_ev = i
            weight[i] = cur_w; continue

        triggered = False
        if reg_arr[i] == 0:
            if zv <= tl_c and (i - last_ev) >= cooldown:
                cur_w = 0.0; triggered = True
            elif zv >= ts_c_eff and (i - last_ev) >= cooldown:
                cur_w = 1.0; triggered = True
        else:
            if zv <= tl_s and (i - last_ev) >= cooldown:
                cur_w = 0.0; triggered = True

        if triggered:
            last_ev = i; n_events += 1; hold_days = 0
        else:
            hold_days += 1
        weight[i] = cur_w

    shift = 1 + delay
    w_del = np.empty(n); w_del[:shift] = w_init; w_del[shift:] = weight[:n - shift]
    net = w_del * r_long + (1.0 - w_del) * r_short
    equity = np.cumprod(1.0 + net)

    years_total = n / ANN
    total_ret = equity[-1] / equity[0] - 1.0
    ann_ret = (1.0 + total_ret) ** (1.0 / max(years_total, 0.01)) - 1.0
    vol = np.std(net) * np.sqrt(ANN)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0

    down = net[net < 0]
    down_vol = np.std(down) * np.sqrt(ANN) if len(down) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol

    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0

    bh_eq = np.cumprod(1.0 + r_long)
    bh_ann = (1.0 + bh_eq[-1] - 1.0) ** (1.0 / max(years_total, 0.01)) - 1.0

    unique_years = np.unique(year_arr)
    y_exc, y_strat, y_bm, y_yrs = [], [], [], []
    for y in unique_years:
        mask = year_arr == y
        if mask.sum() < 5: continue
        sr = np.cumprod(1.0 + net[mask])[-1] - 1.0
        br = np.cumprod(1.0 + r_long[mask])[-1] - 1.0
        y_strat.append(sr); y_bm.append(br); y_exc.append(sr - br); y_yrs.append(y)

    y_exc = np.array(y_exc)
    n_dead = int(np.sum(np.abs(y_exc) < 0.01))
    win_rate = float(np.sum(y_exc > 0)) / max(len(y_exc), 1)
    min_excess = float(np.min(y_exc)) if len(y_exc) > 0 else -1.0

    return {
        "sharpe": sharpe, "sortino": sortino, "ann_ret": ann_ret, "vol": vol,
        "mdd": mdd, "calmar": calmar, "n_events": n_events,
        "n_dead": n_dead, "win_rate": win_rate, "min_excess": min_excess,
        "equity": equity, "weight": w_del, "net_ret": net, "dd": dd,
        "yearly_years": np.array(y_yrs), "yearly_strat": np.array(y_strat),
        "yearly_bm": np.array(y_bm), "yearly_excess": y_exc,
    }

def composite_score(m):
    sh = m["sharpe"]
    so = m["sortino"] * 0.5
    ca = np.clip(m["calmar"], 0, 1)
    me = np.clip(m["min_excess"] + 0.15, -0.1, 0.3)
    wr = m["win_rate"]
    nd = m["n_dead"]
    return 0.50*sh + 0.15*so + 0.10*ca + 0.05*wr + 0.05*me - 0.03*nd


# ── All 8 combinations ──

COMBOS = [
    # (name, use_slope, use_yc, use_term)
    ("RR only",              False, False, False),
    ("RR + Slope",           True,  False, False),
    ("RR + YC",              False, True,  False),
    ("RR + Term",            False, False, True),
    ("RR + Slope + YC",      True,  True,  False),
    ("RR + Slope + Term",    True,  False, True),
    ("RR + YC + Term",       False, True,  True),
    ("RR + Slope + YC + Term", True, True,  True),
]


def run_combo_grid(combo_name, use_slope, use_yc, use_term,
                   z_base_arr, z_slope_arr, z_yc_cache, z_term_cache,
                   reg_arr, r_long, r_short, year_arr, common):
    """Grid search for one combo. Returns best result + full grid df."""

    # Build k lists based on what's used
    k_slope_list = [0.50, 0.75, 1.00] if use_slope else [0.0]
    yc_configs = [(cw, k) for cw in [42, 84, 126] for k in [0.20, 0.30, 0.40]] if use_yc else [(42, 0.0)]
    term_configs = [(tzw, kt) for tzw in [63, 126, 252] for kt in [0.20, 0.30, 0.40, 0.50]] if use_term else [(126, 0.0)]

    # Thresholds
    tl_c_list = np.arange(-4.00, -2.50, 0.25)
    ts_c_list = np.arange(2.50, 4.00, 0.25)
    tl_s_list = [-4.00, -3.50, -3.00, -2.75]

    # Enhancement
    enhance_list = [(0, 0.0, 0), (63, 0.75, 0), (63, 1.0, 0), (63, 0.75, 252)]

    best_score = -999
    best_row = None
    all_rows = []
    count = 0

    for k_slope in k_slope_list:
        # Build z_rr = z_base + k_slope * z_slope
        z_rr = z_base_arr + k_slope * z_slope_arr if use_slope else z_base_arr.copy()

        for yc_cw, k_yc in yc_configs:
            z_yc_arr = z_yc_cache[yc_cw].reindex(common).values if use_yc else np.zeros(len(common))
            z_rr_yc = z_rr + k_yc * z_yc_arr

            for term_zw, k_term in term_configs:
                z_t_arr = z_term_cache[term_zw].reindex(common).values if use_term else np.zeros(len(common))
                z_final = z_rr_yc + k_term * z_t_arr

                for tl_c in tl_c_list:
                    for ts_c in ts_c_list:
                        for tl_s in tl_s_list:
                            m = fast_bt(z_final, reg_arr, r_long, r_short,
                                        year_arr, tl_c, ts_c, tl_s)
                            if m["n_events"] < 6:
                                continue
                            sc = composite_score(m)
                            count += 1

                            if sc > best_score:
                                best_score = sc
                                best_row = {
                                    "combo": combo_name,
                                    "k_slope": k_slope, "yc_cw": yc_cw, "k_yc": k_yc,
                                    "term_zw": term_zw, "k_term": k_term,
                                    "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s,
                                    "adapt_days": 0, "adapt_raise": 0.0, "max_hold_days": 0,
                                    "sharpe": m["sharpe"], "sortino": m["sortino"],
                                    "calmar": m["calmar"], "mdd": m["mdd"],
                                    "n_events": m["n_events"], "n_dead": m["n_dead"],
                                    "win_rate": m["win_rate"], "min_excess": m["min_excess"],
                                    "ann_ret": m["ann_ret"],
                                    "score": sc,
                                }

    # Phase 2: enhance top config
    if best_row is not None:
        z_rr = z_base_arr + best_row["k_slope"] * z_slope_arr if use_slope else z_base_arr.copy()
        z_yc_arr = z_yc_cache[best_row["yc_cw"]].reindex(common).values if use_yc else np.zeros(len(common))
        z_t_arr = z_term_cache[best_row["term_zw"]].reindex(common).values if use_term else np.zeros(len(common))
        z_final = z_rr + best_row["k_yc"] * z_yc_arr + best_row["k_term"] * z_t_arr

        for ad, ar, mh in enhance_list:
            if ad == 0 and mh == 0:
                continue
            m = fast_bt(z_final, reg_arr, r_long, r_short, year_arr,
                        best_row["tl_c"], best_row["ts_c"], best_row["tl_s"],
                        adapt_days=ad, adapt_raise=ar, max_hold_days=mh)
            if m["n_events"] < 6:
                continue
            sc = composite_score(m)
            if sc > best_score:
                best_score = sc
                best_row.update({
                    "adapt_days": ad, "adapt_raise": ar, "max_hold_days": mh,
                    "sharpe": m["sharpe"], "sortino": m["sortino"],
                    "calmar": m["calmar"], "mdd": m["mdd"],
                    "n_events": m["n_events"], "n_dead": m["n_dead"],
                    "win_rate": m["win_rate"], "min_excess": m["min_excess"],
                    "ann_ret": m["ann_ret"], "score": sc,
                })

        # Phase 3: fine-tune thresholds
        tl_c_fine = [best_row["tl_c"] + d for d in [-0.125, 0, 0.125]]
        ts_c_fine = [best_row["ts_c"] + d for d in [-0.125, 0, 0.125]]
        tl_s_fine = [best_row["tl_s"] + d for d in [-0.125, 0, 0.125]]

        for tl_c in tl_c_fine:
            for ts_c in ts_c_fine:
                for tl_s in tl_s_fine:
                    m = fast_bt(z_final, reg_arr, r_long, r_short, year_arr,
                                tl_c, ts_c, tl_s,
                                adapt_days=best_row["adapt_days"],
                                adapt_raise=best_row["adapt_raise"],
                                max_hold_days=best_row["max_hold_days"])
                    if m["n_events"] < 6:
                        continue
                    sc = composite_score(m)
                    if sc > best_score:
                        best_score = sc
                        best_row.update({
                            "tl_c": tl_c, "ts_c": ts_c, "tl_s": tl_s,
                            "sharpe": m["sharpe"], "sortino": m["sortino"],
                            "calmar": m["calmar"], "mdd": m["mdd"],
                            "n_events": m["n_events"], "n_dead": m["n_dead"],
                            "win_rate": m["win_rate"], "min_excess": m["min_excess"],
                            "ann_ret": m["ann_ret"], "score": sc,
                        })

    # Get detailed result for best config
    if best_row is not None:
        z_rr = z_base_arr + best_row["k_slope"] * z_slope_arr if use_slope else z_base_arr.copy()
        z_yc_arr = z_yc_cache[best_row["yc_cw"]].reindex(common).values if use_yc else np.zeros(len(common))
        z_t_arr = z_term_cache[best_row["term_zw"]].reindex(common).values if use_term else np.zeros(len(common))
        z_final = z_rr + best_row["k_yc"] * z_yc_arr + best_row["k_term"] * z_t_arr
        m_best = fast_bt(z_final, reg_arr, r_long, r_short, year_arr,
                         best_row["tl_c"], best_row["ts_c"], best_row["tl_s"],
                         adapt_days=best_row["adapt_days"],
                         adapt_raise=best_row["adapt_raise"],
                         max_hold_days=best_row["max_hold_days"])
    else:
        m_best = None

    return best_row, m_best, count


def main():
    t0 = time.time()
    print("=" * 72)
    print("  ABLAZIONE COMPLETA: tutte le combinazioni di componenti")
    print("  z = z_base [+ k_slope*z_slope] [+ k_yc*z_yc] [+ k_term*z_term]")
    print("=" * 72)

    D = load_all()
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])
    rr_dict = build_rr_individual(D)
    etf_ret = D["etf_ret"]
    bt_start = pd.Timestamp(BACKTEST_START)

    # Pre-compute all signals
    print("\n  Pre-computing signals...")
    rr_blend = build_rr_blend(D)

    # z_base: always [63, 252] (best windows from earlier ablation)
    z_wins = [63, 252]
    z_parts = []
    for w in z_wins:
        mu = rr_blend.rolling(w, min_periods=20).mean()
        sd = rr_blend.rolling(w, min_periods=20).std().replace(0, np.nan)
        z_parts.append((rr_blend - mu) / sd)
    z_base = pd.concat(z_parts, axis=1).mean(axis=1)

    z_slope = build_slope_zscore(D)

    z_yc_cache = {}
    for cw in [42, 84, 126]:
        z_yc_cache[cw] = build_yc_signal(D, change_window=cw)

    z_term_cache = {}
    for tzw in [63, 126, 252]:
        z_term_cache[tzw] = build_term_zscore(rr_dict["1W"], rr_dict["3M"], z_window=tzw)

    # Align dates
    all_idx = [z_base.dropna().index, z_slope.dropna().index,
               regime_hmm.dropna().index,
               etf_ret.dropna(subset=["TLT", "SHV"]).index]
    for zyc in z_yc_cache.values():
        all_idx.append(zyc.dropna().index)
    for zt in z_term_cache.values():
        all_idx.append(zt.dropna().index)

    common = all_idx[0]
    for idx in all_idx[1:]:
        common = common.intersection(idx)
    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])

    z_base_arr = z_base.reindex(common).values
    z_slope_arr = z_slope.reindex(common).values

    print(f"  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d} ({len(common)} days)")

    # Run all 8 combos
    results = []
    detailed_results = {}

    for combo_name, use_slope, use_yc, use_term in COMBOS:
        t1 = time.time()
        components = []
        if use_slope: components.append("Slope")
        if use_yc: components.append("YC")
        if use_term: components.append("Term")
        comp_str = " + ".join(["RR"] + components) if components else "RR only"

        print(f"\n  {'-'*60}")
        print(f"  COMBO: {comp_str}")
        print(f"  {'-'*60}")

        best_row, m_best, n_tested = run_combo_grid(
            combo_name, use_slope, use_yc, use_term,
            z_base_arr, z_slope_arr, z_yc_cache, z_term_cache,
            reg_arr, r_long, r_short, year_arr, common)

        dt = time.time() - t1
        if best_row is not None:
            results.append(best_row)
            detailed_results[combo_name] = m_best
            print(f"    Tested: {n_tested} configs in {dt:.0f}s")
            print(f"    Sharpe: {best_row['sharpe']:.4f}  Score: {best_row['score']:.4f}  "
                  f"Dead: {best_row['n_dead']}  WR: {best_row['win_rate']:.0%}  "
                  f"MDD: {best_row['mdd']*100:.1f}%  Events: {best_row['n_events']}")
            print(f"    k_slope={best_row['k_slope']:.2f}  k_yc={best_row['k_yc']:.2f}  "
                  f"k_term={best_row['k_term']:.2f}  "
                  f"[{best_row['tl_c']:.2f}, {best_row['ts_c']:.2f}, {best_row['tl_s']:.2f}]  "
                  f"ad={best_row['adapt_days']}  mh={best_row['max_hold_days']}")
        else:
            print(f"    FAILED - no valid configs")

    # ── Summary table ──
    print(f"\n{'='*90}")
    print(f"  SUMMARY: All 8 Combinations")
    print(f"{'='*90}")
    print(f"  {'#':>2} {'Combo':<25} {'Sharpe':>7} {'Score':>7} {'Dead':>5} "
          f"{'WR':>5} {'MDD':>7} {'Ev':>4} {'AnnRet':>8} {'k_sl':>5} {'k_yc':>5} {'k_tm':>5}")
    print(f"  {'-'*95}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("score", ascending=False).reset_index(drop=True)

    for i, r in df_results.iterrows():
        print(f"  {i+1:>2} {r['combo']:<25} {r['sharpe']:>7.4f} {r['score']:>7.4f} "
              f"{r['n_dead']:>5.0f} {r['win_rate']:>4.0%} {r['mdd']*100:>6.1f}% "
              f"{r['n_events']:>4.0f} {r['ann_ret']*100:>7.2f}% "
              f"{r['k_slope']:>5.2f} {r['k_yc']:>5.2f} {r['k_term']:>5.2f}")

    # ── Yearly breakdown for all combos ──
    print(f"\n{'='*90}")
    print(f"  YEARLY EXCESS BREAKDOWN")
    print(f"{'='*90}")

    # Build header
    combo_names = df_results["combo"].tolist()
    short_names = []
    for cn in combo_names:
        sn = cn.replace("RR + ", "+").replace("RR only", "RR")
        sn = sn.replace("Slope", "Sl").replace("Term", "Tm")
        short_names.append(sn[:12])

    header = f"  {'Year':>6}"
    for sn in short_names:
        header += f"  {sn:>10}"
    print(header)
    print(f"  {'-'*(8 + 12*len(short_names))}")

    # Get all years
    all_years = sorted(set(
        int(y) for cn in combo_names if cn in detailed_results
        for y in detailed_results[cn]["yearly_years"]
    ))

    for y in all_years:
        line = f"  {y:>6}"
        for cn in combo_names:
            if cn in detailed_results:
                m = detailed_results[cn]
                idx = np.where(m["yearly_years"] == y)[0]
                if len(idx) > 0:
                    exc = m["yearly_excess"][idx[0]]
                    dead = " D" if abs(exc) < 0.01 else "  "
                    line += f"  {exc:>+8.4f}{dead}"
                else:
                    line += f"  {'N/A':>10}"
            else:
                line += f"  {'N/A':>10}"
        print(line)

    # ── Component contribution analysis ──
    print(f"\n{'='*90}")
    print(f"  COMPONENT CONTRIBUTION ANALYSIS")
    print(f"{'='*90}")

    # Find base (RR only)
    rr_only = df_results[df_results["combo"] == "RR only"].iloc[0] if len(df_results[df_results["combo"] == "RR only"]) > 0 else None
    if rr_only is not None:
        base_sh = rr_only["sharpe"]
        base_sc = rr_only["score"]
        print(f"\n  Base (RR only):  Sharpe={base_sh:.4f}  Score={base_sc:.4f}")
        print(f"\n  {'Component Added':<25} {'dSharpe':>8} {'dScore':>8} {'New Sharpe':>11} {'Dead':>5} {'WR':>5}")
        print(f"  {'-'*65}")

        for _, r in df_results.iterrows():
            if r["combo"] == "RR only":
                continue
            dsh = r["sharpe"] - base_sh
            dsc = r["score"] - base_sc
            print(f"  {r['combo']:<25} {dsh:>+8.4f} {dsc:>+8.4f} {r['sharpe']:>11.4f} "
                  f"{r['n_dead']:>5.0f} {r['win_rate']:>4.0%}")

    # ── Marginal contribution ──
    print(f"\n  MARGINAL CONTRIBUTION (adding one component at a time)")
    print(f"  {'-'*65}")

    # Slope contribution: (RR+Slope) - (RR only)
    def _get_sharpe(combo_name):
        row = df_results[df_results["combo"] == combo_name]
        return row.iloc[0]["sharpe"] if len(row) > 0 else None

    pairs = [
        ("Slope alone",   "RR + Slope",           "RR only"),
        ("YC alone",      "RR + YC",              "RR only"),
        ("Term alone",    "RR + Term",            "RR only"),
        ("Slope+YC",      "RR + Slope + YC",      "RR only"),
        ("Slope+Term",    "RR + Slope + Term",    "RR only"),
        ("YC+Term",       "RR + YC + Term",       "RR only"),
        ("All three",     "RR + Slope + YC + Term", "RR only"),
        ("", "", ""),
        ("Term to Sl+YC", "RR + Slope + YC + Term", "RR + Slope + YC"),
        ("YC to Sl+Tm",   "RR + Slope + YC + Term", "RR + Slope + Term"),
        ("Slope to YC+Tm","RR + Slope + YC + Term", "RR + YC + Term"),
    ]

    for label, combo_with, combo_without in pairs:
        if not label:
            print()
            continue
        sh_with = _get_sharpe(combo_with)
        sh_without = _get_sharpe(combo_without)
        if sh_with is not None and sh_without is not None:
            delta = sh_with - sh_without
            print(f"  {label:<20} {combo_with:<25} vs {combo_without:<20} = {delta:>+.4f} Sharpe")

    # ── Charts ──
    print(f"\n{'='*90}")
    print("  Generating charts...")

    # Chart 1: Sharpe comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1a: Sharpe bars
    ax = axes[0, 0]
    x = np.arange(len(df_results))
    colors = ['#c0392b' if 'Term' in c else '#2874a6' if 'Slope' in c else '#27ae60' if 'YC' in c else '#7f8c8d'
              for c in df_results["combo"]]
    ax.barh(x, df_results["sharpe"], color=colors, edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels(df_results["combo"], fontsize=9)
    ax.set_xlabel("Sharpe Ratio", fontsize=11)
    ax.set_title("Sharpe per Combinazione", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()
    for i, v in enumerate(df_results["sharpe"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    # 1b: Composite score
    ax = axes[0, 1]
    ax.barh(x, df_results["score"], color=colors, edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels(df_results["combo"], fontsize=9)
    ax.set_xlabel("Composite Score", fontsize=11)
    ax.set_title("Composite Score per Combinazione", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()
    for i, v in enumerate(df_results["score"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)

    # 1c: Dead years
    ax = axes[1, 0]
    ax.barh(x, df_results["n_dead"], color=colors, edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels(df_results["combo"], fontsize=9)
    ax.set_xlabel("Dead Years", fontsize=11)
    ax.set_title("Dead Years per Combinazione", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()

    # 1d: Win Rate
    ax = axes[1, 1]
    ax.barh(x, df_results["win_rate"] * 100, color=colors, edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels(df_results["combo"], fontsize=9)
    ax.set_xlabel("Win Rate (%)", fontsize=11)
    ax.set_title("Win Rate per Combinazione", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()

    fig.suptitle("Ablazione Completa: 8 Combinazioni di Segnale\n"
                 "z = z_base(RR) [+ Slope] [+ YC] [+ Term(1W-3M)]",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT_DIR, "01_ablation_summary.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Salvato: 01_ablation_summary.png")

    # Chart 2: Equity curves overlay
    fig, ax = plt.subplots(figsize=(20, 10))
    color_cycle = ["#7f8c8d", "#2874a6", "#27ae60", "#c0392b",
                   "#6c3483", "#d35400", "#16a085", "#1a5276"]
    bh_eq = np.cumprod(1.0 + r_long)
    ax.plot(common, bh_eq, color="#aab7b8", linewidth=1.5, linestyle="--",
            alpha=0.6, label="B&H TLT", zorder=2)

    for i, cn in enumerate(df_results["combo"]):
        if cn in detailed_results:
            m = detailed_results[cn]
            sh = df_results[df_results["combo"] == cn].iloc[0]["sharpe"]
            ax.plot(common, m["equity"],
                    color=color_cycle[i % len(color_cycle)],
                    linewidth=1.5 if i < 3 else 1.0,
                    alpha=0.9 if i < 3 else 0.6,
                    label=f"{cn} (Sh={sh:.3f})",
                    zorder=3 + i)

    ax.set_ylabel("Growth of $1", fontsize=12, fontweight="bold")
    ax.set_title("Equity Curves: Tutte le Combinazioni di Segnale",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(os.path.join(OUT_DIR, "02_equity_overlay.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Salvato: 02_equity_overlay.png")

    # Chart 3: Yearly excess heatmap
    n_combos = len(combo_names)
    n_years = len(all_years)
    heat = np.full((n_combos, n_years), np.nan)
    for i, cn in enumerate(combo_names):
        if cn in detailed_results:
            m = detailed_results[cn]
            for j, y in enumerate(all_years):
                idx = np.where(m["yearly_years"] == y)[0]
                if len(idx) > 0:
                    heat[i, j] = m["yearly_excess"][idx[0]] * 100

    fig, ax = plt.subplots(figsize=(20, 8))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=-20, vmax=35)
    ax.set_xticks(range(n_years))
    ax.set_xticklabels(all_years, fontsize=9, rotation=45)
    ax.set_yticks(range(n_combos))
    ax.set_yticklabels(combo_names, fontsize=9)
    plt.colorbar(im, ax=ax, label="Excess Return (%)")

    # Annotate
    for i in range(n_combos):
        for j in range(n_years):
            v = heat[i, j]
            if not np.isnan(v):
                color = "white" if abs(v) > 15 else "black"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold" if abs(v) > 10 else "normal")

    ax.set_title("Yearly Excess Return (%) per Combinazione — Heatmap",
                 fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(OUT_DIR, "03_yearly_heatmap.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Salvato: 03_yearly_heatmap.png")

    # Chart 4: Marginal contribution chart
    fig, ax = plt.subplots(figsize=(14, 6))
    if rr_only is not None:
        combos_sorted = df_results.sort_values("sharpe", ascending=True)
        x = np.arange(len(combos_sorted))
        deltas = combos_sorted["sharpe"].values - base_sh
        colors_d = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
        ax.barh(x, deltas, color=colors_d, edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels(combos_sorted["combo"], fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Delta Sharpe vs RR Only", fontsize=11)
        ax.set_title("Contributo Marginale di Ogni Componente (vs RR Only)",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")
        for i, (v, sh) in enumerate(zip(deltas, combos_sorted["sharpe"])):
            ax.text(v + 0.005 * (1 if v >= 0 else -1), i,
                    f"{v:+.3f} (Sh={sh:.3f})", va="center", fontsize=9)

    fig.savefig(os.path.join(OUT_DIR, "04_marginal_contribution.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Salvato: 04_marginal_contribution.png")

    # Save CSV
    df_results.to_csv(os.path.join(OUT_DIR, "ablation_results.csv"), index=False)
    print("  Salvato: ablation_results.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"  ABLAZIONE COMPLETATA in {elapsed:.0f}s")
    print(f"  Output: {OUT_DIR}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
