"""
versione_presentazione.py - VERSIONE PRESENTAZIONE della strategia RR+YC Duration.

Evoluzione della V2 (Sharpe ~0.40) con 3 innovazioni chiave:

1. Multi-Timeframe Z-Score:
   z_base = mean(zscore(rr_blend, w) for w in [63, 126, 252])
   Riduce il bias LONG persistente nei bull market stabili (2011-2014).

2. Soglia LONG adattiva + Limite durata posizione:
   - Se z_final > 0 per > adapt_days consecutivi, alza ts_calm
   - Dopo max_hold_days nella stessa posizione, torna a neutrale (50/50)

3. Scoring Multi-Obiettivo:
   score = 0.25*sharpe + 0.20*consistency + 0.15*min_yearly_excess
         + 0.10*calmar + 0.15*win_rate_anni + 0.10*sortino
         - 0.10 * n_anni_morti

Risolve il problema dei 5 anni "morti" (2011-2013, 2017, 2019) dove V2
resta LONG tutto l'anno come B&H TLT senza aggiungere valore.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from itertools import product

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

ANN = 252

# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================

OUT_DIR = os.path.join(SAVE_DIR, "versione_presentazione")
os.makedirs(OUT_DIR, exist_ok=True)

# ======================================================================
# SCORE MAP per YC regime
# ======================================================================

SCORE_MAP = {
    "BULL_FLAT": +2, "BULL_STEEP": +1,
    "UNDEFINED": 0,
    "BEAR_FLAT": -1, "BEAR_STEEP": -2,
}

# ======================================================================
# V2 PARAMETRI (frozen, per confronto)
# ======================================================================

V2_PARAMS = {
    "z_windows": [63],
    "z_min_periods": 20,
    "k_slope": 0.50,
    "yc_change_window": 42,
    "yc_z_window": 252,
    "yc_z_min_periods": 20,
    "k_yc": 0.40,
    "tl_calm": -3.25,
    "ts_calm": 2.50,
    "tl_stress": -3.50,
    "ts_stress": 99,
    "cooldown": 5,
    "delay": 1,
    "tcost_bps": 0,
    "adapt_days": 0,
    "adapt_raise": 0.0,
    "max_hold_days": 0,
}


# ======================================================================
# SIGNAL CONSTRUCTION
# ======================================================================

def build_rr_blend(D):
    """Build RR blend (mean of all tenors)."""
    options = D["options"]
    parts = []
    for tenor in ["1W", "1M", "3M", "6M"]:
        put_col, call_col = RR_TENORS[tenor]
        rr = (pd.to_numeric(options[put_col], errors="coerce")
              - pd.to_numeric(options[call_col], errors="coerce"))
        parts.append(rr)
    return pd.concat(parts, axis=1).mean(axis=1)


def build_slope_zscore(D, mom_window=126, z_window=252, z_min=63):
    """Build slope momentum z-score (10Y-2Y)."""
    yields = D["yields"]
    slope = yields["US_10Y"] - yields["US_2Y"]
    delta_slope = slope.diff(mom_window)
    mu = delta_slope.rolling(z_window, min_periods=z_min).mean()
    sd = delta_slope.rolling(z_window, min_periods=z_min).std().replace(0, np.nan)
    return (delta_slope - mu) / sd


def build_rr_multitf(D, z_windows, z_min_per=20, k_slope=0.50):
    """Build multi-timeframe RR z-score.

    z_base = mean(zscore(rr_blend, w) for w in z_windows)
    z_rr = z_base + k_slope * z_slope

    Multiple windows capture different horizons:
    - 63d:  short-term mean reversion
    - 126d: medium-term regime shift
    - 252d: long-term context (reduces persistent LONG bias in bull markets)
    """
    rr_blend = build_rr_blend(D)
    z_parts = []
    for w in z_windows:
        mu = rr_blend.rolling(w, min_periods=z_min_per).mean()
        sd = rr_blend.rolling(w, min_periods=z_min_per).std().replace(0, np.nan)
        z_w = (rr_blend - mu) / sd
        z_parts.append(z_w)

    z_base = pd.concat(z_parts, axis=1).mean(axis=1)

    # Slope component
    z_slope = build_slope_zscore(D)
    z_rr = z_base + k_slope * z_slope

    return z_rr, z_base, z_slope


def build_yc_signal(D, change_window=42, z_window=252, z_min_per=20):
    """Build YC regime_z signal with configurable change window."""
    yld = D["yields"]
    level = (yld["US_10Y"] + yld["US_30Y"]) / 2
    spread = yld["US_30Y"] - yld["US_10Y"]

    d_level = level.diff(change_window)
    d_spread = spread.diff(change_window)

    regime_yc = pd.Series("UNDEFINED", index=yld.index)
    regime_yc[(d_level < 0) & (d_spread > 0)] = "BULL_STEEP"
    regime_yc[(d_level > 0) & (d_spread > 0)] = "BEAR_STEEP"
    regime_yc[(d_level < 0) & (d_spread < 0)] = "BULL_FLAT"
    regime_yc[(d_level > 0) & (d_spread < 0)] = "BEAR_FLAT"

    score = regime_yc.map(SCORE_MAP).astype(float)
    mu = score.rolling(z_window, min_periods=z_min_per).mean()
    sd = score.rolling(z_window, min_periods=z_min_per).std().replace(0, np.nan)
    z_yc = (score - mu) / sd

    return z_yc, regime_yc, score


def build_combined(z_rr, z_yc, k_yc):
    """Combine z_rr and z_yc additively: z_final = z_rr + k_yc * z_yc."""
    return z_rr + k_yc * z_yc


# ======================================================================
# FAST NUMPY BACKTEST (per grid search)
# ======================================================================

def fast_bt(z_arr, reg_arr, r_long, r_short, year_arr,
            tl_c, ts_c, tl_s, ts_s=99.0,
            adapt_days=0, adapt_raise=0.0, max_hold_days=0,
            cooldown=5, delay=1, tcost_bps=0, w_init=0.5):
    """Ultra-fast numpy-only backtest with adaptive threshold + max hold.

    Returns dict with metrics + yearly excess array.
    """
    n = len(z_arr)
    weight = np.full(n, w_init)
    cur_w = w_init
    last_ev = -cooldown - 1
    n_events = 0
    consec_positive = 0  # consecutive days z > 0
    hold_days = 0        # days in current position

    for i in range(n):
        zv = z_arr[i]
        if np.isnan(zv):
            weight[i] = cur_w
            hold_days += 1
            continue

        # Adaptive threshold: if LONG for > adapt_days, raise ts_calm
        # Makes it harder to STAY long, forcing re-evaluation in bull markets
        ts_c_eff = ts_c
        if adapt_days > 0 and cur_w > 0.7 and hold_days >= adapt_days:
            ts_c_eff = ts_c + adapt_raise

        # Max hold: revert to neutral if held too long in any direction
        if max_hold_days > 0 and hold_days >= max_hold_days and abs(cur_w - 0.5) > 0.01:
            cur_w = 0.5
            hold_days = 0
            n_events += 1
            last_ev = i
            weight[i] = cur_w
            continue

        triggered = False
        reg = reg_arr[i]

        if reg == 0:  # CALM
            if zv <= tl_c and (i - last_ev) >= cooldown:
                cur_w = 0.0
                triggered = True
            elif zv >= ts_c_eff and (i - last_ev) >= cooldown:
                cur_w = 1.0
                triggered = True
        else:  # STRESS
            if zv <= tl_s and (i - last_ev) >= cooldown:
                cur_w = 0.0
                triggered = True

        if triggered:
            last_ev = i
            n_events += 1
            hold_days = 0
        else:
            hold_days += 1

        weight[i] = cur_w

    # Execution delay
    shift = 1 + delay
    w_del = np.empty(n)
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    # Portfolio returns
    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost_bps / 10_000)

    # Equity
    equity = np.cumprod(1.0 + net)

    # Metrics
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

    # B&H benchmark
    bh_eq = np.cumprod(1.0 + r_long)
    bh_total = bh_eq[-1] - 1.0
    bh_ann = (1.0 + bh_total) ** (1.0 / max(years_total, 0.01)) - 1.0
    excess = ann_ret - bh_ann

    # Yearly breakdown
    unique_years = np.unique(year_arr)
    yearly_strat = []
    yearly_bm = []
    yearly_excess = []
    yearly_years = []
    for y in unique_years:
        mask = year_arr == y
        if mask.sum() < 5:
            continue
        eq_y = np.cumprod(1.0 + net[mask])
        bm_y = np.cumprod(1.0 + r_long[mask])
        yr_ret = eq_y[-1] - 1.0
        bm_ret = bm_y[-1] - 1.0
        yearly_strat.append(yr_ret)
        yearly_bm.append(bm_ret)
        yearly_excess.append(yr_ret - bm_ret)
        yearly_years.append(y)

    yearly_excess_arr = np.array(yearly_excess)
    yearly_strat_arr = np.array(yearly_strat)
    yearly_bm_arr = np.array(yearly_bm)
    yearly_years_arr = np.array(yearly_years)
    n_years = len(yearly_excess_arr)

    # Dead years: |excess| < 1%
    n_dead = int(np.sum(np.abs(yearly_excess_arr) < 0.01))

    # Win rate (years with excess > 0)
    win_rate = float(np.sum(yearly_excess_arr > 0)) / max(n_years, 1)

    # Consistency: std of yearly excess (lower = more consistent)
    consistency = 1.0 / (np.std(yearly_excess_arr) + 0.01) if n_years > 1 else 0

    # Min yearly excess
    min_excess = float(np.min(yearly_excess_arr)) if n_years > 0 else -1.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_ret": ann_ret,
        "vol": vol,
        "mdd": mdd,
        "calmar": calmar,
        "excess": excess,
        "total_ret": total_ret,
        "n_events": n_events,
        "n_dead": n_dead,
        "win_rate": win_rate,
        "consistency": consistency,
        "min_excess": min_excess,
        "equity": equity,
        "net_ret": net,
        "weight": w_del,
        "dd": dd,
        "yearly_years": yearly_years_arr,
        "yearly_strat": yearly_strat_arr,
        "yearly_bm": yearly_bm_arr,
        "yearly_excess": yearly_excess_arr,
    }


# ======================================================================
# MULTI-OBJECTIVE SCORING
# ======================================================================

def composite_score(m):
    """Compute multi-objective composite score.

    Sharpe-dominated scoring with linear dead-year penalty calibrated so:
    - Sh=0.40, dead=5 scores ~equal to Sh=0.30, dead=2
    - The sweet spot is Sharpe 0.30+ with 2-3 dead years

    score = 0.50*sharpe + 0.15*(sortino*0.5) + 0.10*calmar
          + 0.05*win_rate + 0.05*min_excess_adj - 0.03*n_dead
    """
    sharpe = m["sharpe"]
    sortino_scaled = m["sortino"] * 0.5       # ~same scale as Sharpe
    calmar = np.clip(m["calmar"], 0, 1)
    min_exc = np.clip(m["min_excess"] + 0.15, -0.1, 0.3)
    win_rate = m["win_rate"]
    n_dead = m["n_dead"]

    score = (0.50 * sharpe
             + 0.15 * sortino_scaled
             + 0.10 * calmar
             + 0.05 * win_rate
             + 0.05 * min_exc
             - 0.03 * n_dead)

    return score


# ======================================================================
# GRID SEARCH (3 phases)
# ======================================================================

def run_grid_search(D, regime_hmm, verbose=True):
    """Run 3-phase grid search with multi-objective scoring.

    Phase 1: z_windows x yc_cw x k_yc x thresholds (base configs)
    Phase 2: top configs x adapt_days x adapt_raise x max_hold_days
    Phase 3: fine-tuning thresholds on top configs
    """
    t0 = time.time()

    # Pre-compute signals for all z_window combos
    z_windows_list = [
        [63],
        [63, 126],
        [63, 126, 252],
        [63, 252],
        [126, 252],
    ]
    k_slope_list = [0.50, 0.75, 1.00]
    yc_cw_list = [42, 63, 84, 126]
    k_yc_list = [0.20, 0.30, 0.40, 0.50, 0.60]

    # Threshold grids (phase 1)
    tl_c_list = np.arange(-4.00, -2.50, 0.25)  # [-4.00, -3.75, ..., -2.75]
    ts_c_list = np.arange(2.50, 4.00, 0.25)    # [2.50, 2.75, 3.00, 3.25, 3.50, 3.75]
    tl_s_list = [-4.00, -3.50, -3.00]

    # Pre-compute common arrays
    etf_ret = D["etf_ret"]
    bt_start = pd.Timestamp(BACKTEST_START)

    # Pre-compute slope z-score (shared)
    z_slope = build_slope_zscore(D)

    # Pre-compute all RR multi-tf z-scores (per z_windows and k_slope)
    if verbose:
        print("  Pre-computing signals...")
    rr_blend = build_rr_blend(D)
    z_base_only_cache = {}  # (z_windows,) -> z_base (without slope)
    z_base_cache = {}       # (z_windows, k_slope) -> z_rr (with slope)
    for z_wins in z_windows_list:
        z_parts = []
        for w in z_wins:
            mu = rr_blend.rolling(w, min_periods=20).mean()
            sd = rr_blend.rolling(w, min_periods=20).std().replace(0, np.nan)
            z_parts.append((rr_blend - mu) / sd)
        z_base = pd.concat(z_parts, axis=1).mean(axis=1)
        z_base_only_cache[tuple(z_wins)] = z_base
        for k_slope in k_slope_list:
            z_rr = z_base + k_slope * z_slope
            z_base_cache[(tuple(z_wins), k_slope)] = z_rr

    # Pre-compute all YC signals
    z_yc_cache = {}
    regime_yc_cache = {}
    for cw in yc_cw_list:
        z_yc, regime_yc, _ = build_yc_signal(D, change_window=cw)
        z_yc_cache[cw] = z_yc
        regime_yc_cache[cw] = regime_yc

    # Align common dates (use broadest intersection)
    all_indices = [z_rr.dropna().index for z_rr in z_base_cache.values()]
    all_indices += [z_yc.dropna().index for z_yc in z_yc_cache.values()]
    all_indices.append(regime_hmm.dropna().index)
    all_indices.append(etf_ret.dropna(subset=["TLT", "SHV"]).index)

    common = all_indices[0]
    for idx in all_indices[1:]:
        common = common.intersection(idx)
    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])

    n_days = len(common)
    if verbose:
        print(f"  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d} ({n_days} days)")

    # ── PHASE 1: Base grid search ──
    if verbose:
        print("\n  Phase 1: Base grid search...")

    phase1_rows = []
    total_p1 = (len(z_windows_list) * len(k_slope_list) * len(yc_cw_list)
                * len(k_yc_list) * len(tl_c_list) * len(ts_c_list) * len(tl_s_list))
    count = 0

    for z_wins in z_windows_list:
        for k_slope in k_slope_list:
            z_rr_arr = z_base_cache[(tuple(z_wins), k_slope)].reindex(common).values
            for yc_cw in yc_cw_list:
                z_yc_arr = z_yc_cache[yc_cw].reindex(common).values
                for k_yc in k_yc_list:
                    z_final = z_rr_arr + k_yc * z_yc_arr
                    for tl_c in tl_c_list:
                        for ts_c in ts_c_list:
                            for tl_s in tl_s_list:
                                m = fast_bt(z_final, reg_arr, r_long, r_short,
                                            year_arr, tl_c, ts_c, tl_s)
                                score = composite_score(m)
                                count += 1
                                phase1_rows.append({
                                    "z_windows": str(z_wins),
                                    "k_slope": k_slope,
                                    "yc_cw": yc_cw,
                                    "k_yc": k_yc,
                                    "tl_c": tl_c,
                                    "ts_c": ts_c,
                                    "tl_s": tl_s,
                                    "adapt_days": 0,
                                    "adapt_raise": 0.0,
                                    "max_hold_days": 0,
                                    "sharpe": m["sharpe"],
                                    "sortino": m["sortino"],
                                    "calmar": m["calmar"],
                                    "ann_ret": m["ann_ret"],
                                    "mdd": m["mdd"],
                                    "excess": m["excess"],
                                    "n_events": m["n_events"],
                                    "n_dead": m["n_dead"],
                                    "win_rate": m["win_rate"],
                                    "consistency": m["consistency"],
                                    "min_excess": m["min_excess"],
                                    "composite_score": score,
                                })

    df_p1 = pd.DataFrame(phase1_rows)
    # Filter: minimum events
    df_p1f = df_p1[df_p1["n_events"] >= 6].copy()

    if verbose:
        print(f"    Tested {count} configs ({len(df_p1f)} with >=6 events)")
        top5 = df_p1f.nlargest(5, "composite_score")
        print(f"    Top 5 composite scores:")
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            print(f"      {i}. score={r['composite_score']:.4f}  "
                  f"Sh={r['sharpe']:.3f}  dead={r['n_dead']:.0f}  "
                  f"wr={r['win_rate']:.0%}  z_w={r['z_windows']}  "
                  f"k_sl={r['k_slope']:.2f}  "
                  f"yc_cw={r['yc_cw']:.0f}  k_yc={r['k_yc']:.2f}")

    # ── PHASE 2: Enhancement grid (top 100 configs) ──
    if verbose:
        print("\n  Phase 2: Enhancement grid (adaptive + max hold)...")

    top100 = df_p1f.nlargest(100, "composite_score")
    adapt_days_list = [0, 42, 63, 126, 189]
    adapt_raise_list = [0.50, 0.75, 1.0, 1.50]
    max_hold_list = [0, 126, 189, 252]

    phase2_rows = []
    for _, cfg in top100.iterrows():
        z_wins = eval(cfg["z_windows"])  # Convert string back to list
        cfg_k_slope = float(cfg["k_slope"])
        z_rr_arr = z_base_cache[(tuple(z_wins), cfg_k_slope)].reindex(common).values
        z_yc_arr = z_yc_cache[int(cfg["yc_cw"])].reindex(common).values
        z_final = z_rr_arr + cfg["k_yc"] * z_yc_arr

        for ad in adapt_days_list:
            for ar in adapt_raise_list:
                for mh in max_hold_list:
                    # Skip if no enhancement (already tested in phase 1)
                    if ad == 0 and mh == 0:
                        # Copy from phase 1
                        phase2_rows.append({
                            "z_windows": cfg["z_windows"],
                            "k_slope": cfg_k_slope,
                            "yc_cw": cfg["yc_cw"],
                            "k_yc": cfg["k_yc"],
                            "tl_c": cfg["tl_c"],
                            "ts_c": cfg["ts_c"],
                            "tl_s": cfg["tl_s"],
                            "adapt_days": 0,
                            "adapt_raise": 0.0,
                            "max_hold_days": 0,
                            "sharpe": cfg["sharpe"],
                            "sortino": cfg["sortino"],
                            "calmar": cfg["calmar"],
                            "ann_ret": cfg["ann_ret"],
                            "mdd": cfg["mdd"],
                            "excess": cfg["excess"],
                            "n_events": cfg["n_events"],
                            "n_dead": cfg["n_dead"],
                            "win_rate": cfg["win_rate"],
                            "consistency": cfg["consistency"],
                            "min_excess": cfg["min_excess"],
                            "composite_score": cfg["composite_score"],
                        })
                        continue

                    m = fast_bt(z_final, reg_arr, r_long, r_short,
                                year_arr, cfg["tl_c"], cfg["ts_c"],
                                cfg["tl_s"], adapt_days=ad,
                                adapt_raise=ar, max_hold_days=mh)
                    score = composite_score(m)
                    phase2_rows.append({
                        "z_windows": cfg["z_windows"],
                        "k_slope": cfg_k_slope,
                        "yc_cw": cfg["yc_cw"],
                        "k_yc": cfg["k_yc"],
                        "tl_c": cfg["tl_c"],
                        "ts_c": cfg["ts_c"],
                        "tl_s": cfg["tl_s"],
                        "adapt_days": ad,
                        "adapt_raise": ar,
                        "max_hold_days": mh,
                        "sharpe": m["sharpe"],
                        "sortino": m["sortino"],
                        "calmar": m["calmar"],
                        "ann_ret": m["ann_ret"],
                        "mdd": m["mdd"],
                        "excess": m["excess"],
                        "n_events": m["n_events"],
                        "n_dead": m["n_dead"],
                        "win_rate": m["win_rate"],
                        "consistency": m["consistency"],
                        "min_excess": m["min_excess"],
                        "composite_score": score,
                    })

    df_p2 = pd.DataFrame(phase2_rows)
    df_p2f = df_p2[df_p2["n_events"] >= 6].copy()

    if verbose:
        print(f"    Tested {len(df_p2)} configs ({len(df_p2f)} with >=6 events)")
        top5 = df_p2f.nlargest(5, "composite_score")
        print(f"    Top 5 composite scores:")
        for i, (_, r) in enumerate(top5.iterrows(), 1):
            print(f"      {i}. score={r['composite_score']:.4f}  "
                  f"Sh={r['sharpe']:.3f}  dead={r['n_dead']:.0f}  "
                  f"ad={r['adapt_days']:.0f}  ar={r['adapt_raise']:.2f}  "
                  f"mh={r['max_hold_days']:.0f}")

    # ── PHASE 3: Fine-tuning thresholds on top 20 ──
    if verbose:
        print("\n  Phase 3: Fine-tuning thresholds...")

    top20 = df_p2f.nlargest(20, "composite_score")
    # Fine-tune thresholds with 0.125 step around best values
    phase3_rows = []
    for _, cfg in top20.iterrows():
        z_wins = eval(cfg["z_windows"])
        cfg_k_slope = float(cfg["k_slope"])
        z_rr_arr = z_base_cache[(tuple(z_wins), cfg_k_slope)].reindex(common).values
        z_yc_arr = z_yc_cache[int(cfg["yc_cw"])].reindex(common).values
        z_final = z_rr_arr + cfg["k_yc"] * z_yc_arr

        # Fine grid around current best
        tl_c_fine = [cfg["tl_c"] - 0.125, cfg["tl_c"], cfg["tl_c"] + 0.125]
        ts_c_fine = [cfg["ts_c"] - 0.125, cfg["ts_c"], cfg["ts_c"] + 0.125]
        tl_s_fine = [cfg["tl_s"] - 0.125, cfg["tl_s"], cfg["tl_s"] + 0.125]

        for tl_c in tl_c_fine:
            for ts_c in ts_c_fine:
                for tl_s in tl_s_fine:
                    m = fast_bt(z_final, reg_arr, r_long, r_short,
                                year_arr, tl_c, ts_c, tl_s,
                                adapt_days=int(cfg["adapt_days"]),
                                adapt_raise=float(cfg["adapt_raise"]),
                                max_hold_days=int(cfg["max_hold_days"]))
                    score = composite_score(m)
                    phase3_rows.append({
                        "z_windows": cfg["z_windows"],
                        "k_slope": cfg_k_slope,
                        "yc_cw": cfg["yc_cw"],
                        "k_yc": cfg["k_yc"],
                        "tl_c": tl_c,
                        "ts_c": ts_c,
                        "tl_s": tl_s,
                        "adapt_days": cfg["adapt_days"],
                        "adapt_raise": cfg["adapt_raise"],
                        "max_hold_days": cfg["max_hold_days"],
                        "sharpe": m["sharpe"],
                        "sortino": m["sortino"],
                        "calmar": m["calmar"],
                        "ann_ret": m["ann_ret"],
                        "mdd": m["mdd"],
                        "excess": m["excess"],
                        "n_events": m["n_events"],
                        "n_dead": m["n_dead"],
                        "win_rate": m["win_rate"],
                        "consistency": m["consistency"],
                        "min_excess": m["min_excess"],
                        "composite_score": score,
                    })

    df_p3 = pd.DataFrame(phase3_rows)
    df_p3f = df_p3[df_p3["n_events"] >= 6].copy()

    if verbose:
        print(f"    Tested {len(df_p3)} fine-tuning configs ({len(df_p3f)} valid)")

    # ── Combine all phases ──
    df_all = pd.concat([df_p1f, df_p2f, df_p3f], ignore_index=True)
    df_all = df_all.drop_duplicates(
        subset=["z_windows", "k_slope", "yc_cw", "k_yc", "tl_c", "ts_c", "tl_s",
                "adapt_days", "adapt_raise", "max_hold_days"]
    ).copy()
    df_all = df_all.sort_values("composite_score", ascending=False).reset_index(drop=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Grid search complete: {len(df_all)} unique configs in {elapsed:.1f}s")

    # Return everything needed
    return {
        "df_all": df_all,
        "common": common,
        "reg_arr": reg_arr,
        "r_long": r_long,
        "r_short": r_short,
        "year_arr": year_arr,
        "z_base_cache": z_base_cache,
        "z_base_only_cache": z_base_only_cache,
        "z_yc_cache": z_yc_cache,
        "regime_yc_cache": regime_yc_cache,
    }


# ======================================================================
# RUN V2 BASELINE FOR COMPARISON
# ======================================================================

def run_v2_baseline(D, regime_hmm, common, reg_arr, r_long, r_short, year_arr):
    """Run V2 strategy for comparison."""
    z_rr_v2, z_base_v2, z_slope_v2 = build_rr_multitf(D, [63])
    z_yc_v2, regime_yc_v2, _ = build_yc_signal(D, change_window=42)
    z_final_v2 = build_combined(z_rr_v2, z_yc_v2, 0.40)

    z_v2_arr = z_final_v2.reindex(common).values
    m_v2 = fast_bt(z_v2_arr, reg_arr, r_long, r_short, year_arr,
                   tl_c=-3.25, ts_c=2.50, tl_s=-3.50)
    return m_v2, z_final_v2, z_rr_v2, z_yc_v2, regime_yc_v2


# ======================================================================
# CHARTS
# ======================================================================

def plot_01_equity_zscore(dates, result, z_final, z_rr, z_yc, regime_hmm,
                          best_params, out_dir):
    """01: Equity + z-scores + trade markers."""
    equity = result["equity"]
    bh_eq = np.cumprod(1.0 + result["r_long"])
    w_vals = result["weight"]

    z_f = z_final.reindex(dates).values
    z_r = z_rr.reindex(dates).values
    z_y = z_yc.reindex(dates).values
    reg_s = regime_hmm.reindex(dates)

    fig, axes = plt.subplots(
        4, 1, figsize=(22, 16),
        gridspec_kw={"height_ratios": [3, 1.2, 1.2, 0.8], "hspace": 0.06},
        sharex=True)
    ax1, ax2, ax3, ax4 = axes

    # Regime shading
    stress_mask = (reg_s.values == "STRESS")
    i = 0
    while i < len(dates):
        if stress_mask[i]:
            j = i
            while j < len(dates) and stress_mask[j]:
                j += 1
            for ax in axes:
                ax.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                           color="#FFE0E0", alpha=0.30, zorder=0)
            i = j
        else:
            i += 1

    # TOP: Equity
    ax1.plot(dates, equity, color="#1a5276", linewidth=1.8, zorder=4,
             label="V_Presentazione")
    ax1.plot(dates, bh_eq, color="#aab7b8", linewidth=1.2, linestyle="--",
             alpha=0.7, zorder=3, label="B&H TLT")
    ax1.fill_between(dates, equity, bh_eq,
                     where=(equity >= bh_eq), color="#27ae60", alpha=0.06)
    ax1.fill_between(dates, equity, bh_eq,
                     where=(equity < bh_eq), color="#e74c3c", alpha=0.06)

    # Trade markers
    dw = np.abs(np.diff(w_vals, prepend=w_vals[0]))
    trade_idx = np.where(dw > 0.01)[0]
    for ti in trade_idx:
        marker = "^" if w_vals[ti] > 0.7 else ("v" if w_vals[ti] < 0.3 else "o")
        color = "#27ae60" if w_vals[ti] > 0.7 else (
            "#e74c3c" if w_vals[ti] < 0.3 else "#f39c12")
        ax1.scatter(dates[ti], equity[ti], marker=marker, s=80, c=color,
                    edgecolors="black", linewidths=0.5, zorder=6)

    m = result
    ax1.set_ylabel("Growth of $1", fontsize=12, fontweight="bold")
    z_w_str = best_params["z_windows"]
    ax1.set_title(
        f"VERSIONE PRESENTAZIONE -- RR+Slope+YC Multi-TF Duration Strategy\n"
        f"z_base=mean(zscore(rr, w={z_w_str})) + {best_params['k_slope']:.2f}*Slope "
        f"+ {best_params['k_yc']:.2f}*z_yc(cw={best_params['yc_cw']})",
        fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.set_ylim(bottom=0.55)

    legend_handles = [
        Line2D([0], [0], color="#1a5276", linewidth=2,
               label=f"Strategy (Sh={m['sharpe']:.3f})"),
        Line2D([0], [0], color="#aab7b8", linewidth=1.2, linestyle="--",
               label="B&H TLT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="LONG"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="SHORT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f39c12",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="NEUTRAL"),
        plt.Rectangle((0, 0), 1, 1, fc="#FFE0E0", alpha=0.5,
                       label="STRESS regime"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=9,
               framealpha=0.9)

    ann_text = (
        f"Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}  |  "
        f"Ann.Ret: {m['ann_ret']*100:.2f}%\n"
        f"Vol: {m['vol']*100:.1f}%  |  MDD: {m['mdd']*100:.1f}%  |  "
        f"Calmar: {m['calmar']:.3f}  |  Events: {m['n_events']}  |  "
        f"Dead yrs: {m['n_dead']}"
    )
    ax1.text(0.98, 0.03, ann_text, transform=ax1.transAxes,
             fontsize=8, fontfamily="monospace", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.85, edgecolor="#cccccc"))

    # PANEL 2: Z-score RR
    tl_c = best_params["tl_c"]
    ts_c = best_params["ts_c"]
    tl_s = best_params["tl_s"]

    v_rr = ~np.isnan(z_r)
    ax2.plot(dates[v_rr], z_r[v_rr], color="#2c3e50", linewidth=0.6, alpha=0.8)
    ax2.fill_between(dates, np.nan_to_num(z_r), 0,
                     where=(z_r > 0), color="#27ae60", alpha=0.06)
    ax2.fill_between(dates, np.nan_to_num(z_r), 0,
                     where=(z_r < 0), color="#e74c3c", alpha=0.06)
    ax2.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax2.set_ylabel("z RR Multi-TF", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.15)
    ax2.set_ylim(-8, 8)

    # PANEL 3: Z-score YC
    v_yc = ~np.isnan(z_y)
    ax3.plot(dates[v_yc], z_y[v_yc], color="#8e44ad", linewidth=0.6, alpha=0.8)
    ax3.fill_between(dates, np.nan_to_num(z_y), 0,
                     where=(z_y > 0), color="#27ae60", alpha=0.06)
    ax3.fill_between(dates, np.nan_to_num(z_y), 0,
                     where=(z_y < 0), color="#e74c3c", alpha=0.06)
    ax3.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax3.set_ylabel("z YC Regime", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.15)
    ax3.set_ylim(-4, 4)

    # PANEL 4: Combined z + thresholds
    v_f = ~np.isnan(z_f)
    ax4.plot(dates[v_f], z_f[v_f], color="#2c3e50", linewidth=0.7, alpha=0.85)
    ax4.fill_between(dates, np.nan_to_num(z_f), 0,
                     where=(z_f > 0), color="#27ae60", alpha=0.06)
    ax4.fill_between(dates, np.nan_to_num(z_f), 0,
                     where=(z_f < 0), color="#e74c3c", alpha=0.06)
    ax4.axhline(tl_c, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7)
    ax4.axhline(ts_c, color="#27ae60", linewidth=1.2, linestyle="--", alpha=0.7)
    ax4.axhline(tl_s, color="#c0392b", linewidth=1.0, linestyle=":", alpha=0.6)
    ax4.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Trade markers on z-score panel
    for ti in trade_idx:
        zv = z_f[ti] if not np.isnan(z_f[ti]) else 0
        marker = "^" if w_vals[ti] > 0.7 else ("v" if w_vals[ti] < 0.3 else "o")
        color = "#27ae60" if w_vals[ti] > 0.7 else (
            "#e74c3c" if w_vals[ti] < 0.3 else "#f39c12")
        ax4.scatter(dates[ti], zv, marker=marker, s=90, c=color,
                    edgecolors="black", linewidths=0.6, zorder=6)

    ax4.set_ylabel("z Combined", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Date", fontsize=11)
    ax4.grid(True, alpha=0.15)
    ax4.set_ylim(-9, 9)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    z_legend = [
        Line2D([0], [0], color="#e74c3c", linewidth=1.2, linestyle="--",
               label=f"SHORT calm ({tl_c:.2f})"),
        Line2D([0], [0], color="#27ae60", linewidth=1.2, linestyle="--",
               label=f"LONG calm ({ts_c:.2f})"),
        Line2D([0], [0], color="#c0392b", linewidth=1.0, linestyle=":",
               label=f"SHORT stress ({tl_s:.2f})"),
    ]
    ax4.legend(handles=z_legend, loc="lower left", fontsize=8,
               framealpha=0.9, ncol=3)

    fig.savefig(os.path.join(out_dir, "01_equity_zscore.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_02_yearly_comparison(dates, result_pres, result_v2, r_long, year_arr,
                               out_dir):
    """02: V_pres vs V2 vs B&H anno per anno + delta excess."""
    # Compute yearly for B&H
    unique_years = np.unique(year_arr)
    yearly_data = []
    for y in unique_years:
        mask = year_arr == y
        if mask.sum() < 5:
            continue
        bh_y = np.cumprod(1.0 + r_long[mask])[-1] - 1.0

        # Find matching year in results
        pres_exc = 0.0
        v2_exc = 0.0
        pres_ret = bh_y
        v2_ret = bh_y
        for i, yy in enumerate(result_pres["yearly_years"]):
            if yy == y:
                pres_ret = result_pres["yearly_strat"][i]
                pres_exc = result_pres["yearly_excess"][i]
                break
        for i, yy in enumerate(result_v2["yearly_years"]):
            if yy == y:
                v2_ret = result_v2["yearly_strat"][i]
                v2_exc = result_v2["yearly_excess"][i]
                break

        yearly_data.append({
            "year": y,
            "pres_ret": pres_ret, "v2_ret": v2_ret, "bh_ret": bh_y,
            "pres_excess": pres_exc, "v2_excess": v2_exc,
        })

    df_yr = pd.DataFrame(yearly_data)

    fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                              gridspec_kw={"height_ratios": [2, 1.2, 1],
                                           "hspace": 0.18})
    ax1, ax2, ax3 = axes

    x = np.arange(len(df_yr))
    w = 0.25

    # Bar chart: returns
    ax1.bar(x - w, df_yr["pres_ret"]*100, w, label="V_Presentazione",
            color="#2874a6", edgecolor="white", linewidth=0.5)
    ax1.bar(x, df_yr["v2_ret"]*100, w, label="V2",
            color="#6c3483", edgecolor="white", linewidth=0.5, alpha=0.7)
    ax1.bar(x + w, df_yr["bh_ret"]*100, w, label="B&H TLT",
            color="#aab7b8", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_yr["year"].astype(int), fontsize=10)
    ax1.set_ylabel("Return (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Confronto Annuale: V_Presentazione vs V2 vs B&H TLT",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.axhline(0, color="black", linewidth=0.8)

    # Excess returns
    w2 = 0.35
    ax2.bar(x - w2/2, df_yr["pres_excess"]*100, w2,
            label="V_Pres excess", color="#2874a6", edgecolor="white")
    ax2.bar(x + w2/2, df_yr["v2_excess"]*100, w2,
            label="V2 excess", color="#6c3483", edgecolor="white", alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_yr["year"].astype(int), fontsize=10)
    ax2.set_ylabel("Excess Return (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Excess Return vs B&H TLT", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.axhline(0, color="black", linewidth=0.8)

    # Delta: V_pres excess - V2 excess
    delta = df_yr["pres_excess"] - df_yr["v2_excess"]
    colors_delta = ["#27ae60" if d > 0 else "#e74c3c" for d in delta]
    ax3.bar(x, delta*100, 0.6, color=colors_delta, edgecolor="white")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df_yr["year"].astype(int), fontsize=10)
    ax3.set_ylabel("Delta Excess (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Delta: V_Presentazione excess - V2 excess", fontsize=11)
    ax3.grid(True, alpha=0.2, axis="y")
    ax3.axhline(0, color="black", linewidth=0.8)

    # Annotation
    pres_dead = int(np.sum(np.abs(df_yr["pres_excess"]) < 0.01))
    v2_dead = int(np.sum(np.abs(df_yr["v2_excess"]) < 0.01))
    ax3.text(0.98, 0.95,
             f"V_Pres dead years: {pres_dead}  |  V2 dead years: {v2_dead}",
             transform=ax3.transAxes, fontsize=9, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="#cccccc"))

    fig.savefig(os.path.join(out_dir, "02_yearly_comparison.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_03_drawdown(dates, result_pres, r_long, out_dir):
    """03: Drawdown strategy vs B&H."""
    dd_s = result_pres["dd"]
    bh_eq = np.cumprod(1.0 + r_long)
    bh_peak = np.maximum.accumulate(bh_eq)
    dd_bm = bh_eq / bh_peak - 1

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.fill_between(dates, dd_s * 100, 0, color="#e74c3c", alpha=0.3,
                    label="V_Presentazione")
    ax.plot(dates, dd_s * 100, color="#c0392b", linewidth=0.8)
    ax.plot(dates, dd_bm * 100, color="#7f8c8d", linewidth=0.8,
            linestyle="--", alpha=0.7, label="B&H TLT")
    ax.set_ylabel("Drawdown (%)", fontsize=12, fontweight="bold")
    ax.set_title("Drawdown: V_Presentazione vs B&H TLT", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(top=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    mdd_idx = np.argmin(dd_s)
    mdd_val = dd_s[mdd_idx] * 100
    ax.annotate(f"MDD: {mdd_val:.1f}%\n{dates[mdd_idx]:%Y-%m-%d}",
                xy=(dates[mdd_idx], mdd_val),
                xytext=(dates[mdd_idx], mdd_val - 5),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#c0392b"))

    fig.savefig(os.path.join(out_dir, "03_drawdown.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_04_regime_timeline(dates, result_pres, regime_hmm, regime_yc, out_dir):
    """04: HMM + YC regime timeline."""
    equity = result_pres["equity"]
    hmm_vals = regime_hmm.reindex(dates)
    yc_vals = regime_yc.reindex(dates)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(18, 8),
        gridspec_kw={"height_ratios": [2, 0.6, 0.6], "hspace": 0.12},
        sharex=True)

    ax1.plot(dates, equity, color="#1a5276", linewidth=1.5)
    stress_mask = (hmm_vals.values == "STRESS")
    i = 0
    while i < len(dates):
        if stress_mask[i]:
            j = i
            while j < len(dates) and stress_mask[j]:
                j += 1
            ax1.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                        color="#FFE0E0", alpha=0.35, zorder=0)
            i = j
        else:
            i += 1
    ax1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    ax1.set_title("Regime Timeline: HMM (CALM/STRESS) + YC (Bull/Bear Steep/Flat)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))

    hmm_numeric = np.where(hmm_vals.values == "STRESS", 1, 0)
    ax2.fill_between(dates, hmm_numeric, 0, color="#e74c3c", alpha=0.5, step="mid")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["CALM", "STRESS"], fontsize=9)
    ax2.set_ylabel("HMM", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.15)

    yc_score_map = {"BULL_FLAT": 2, "BULL_STEEP": 1, "UNDEFINED": 0,
                    "BEAR_FLAT": -1, "BEAR_STEEP": -2}
    yc_numeric = np.array([yc_score_map.get(v, 0) for v in yc_vals.values])
    ax3.fill_between(dates, yc_numeric, 0, where=(yc_numeric > 0),
                     color="#27ae60", alpha=0.4, step="mid")
    ax3.fill_between(dates, yc_numeric, 0, where=(yc_numeric < 0),
                     color="#e74c3c", alpha=0.4, step="mid")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylim(-2.5, 2.5)
    ax3.set_yticks([-2, -1, 0, 1, 2])
    ax3.set_yticklabels(["BEAR_ST", "BEAR_FL", "", "BULL_ST", "BULL_FL"], fontsize=8)
    ax3.set_ylabel("YC 10s30s", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.15)
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(os.path.join(out_dir, "04_regime_timeline.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_05_dashboard(result_pres, result_v2, best_params, out_dir):
    """05: Performance dashboard with V_pres vs V2 comparison."""
    m = result_pres

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PERFORMANCE DASHBOARD -- V_Presentazione (Multi-TF + Adaptive)",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.92, bottom=0.06)

    # 1. Key Metrics
    ax_tab = fig.add_subplot(gs[0, 0])
    ax_tab.axis("off")
    metrics_data = [
        ["Sharpe Ratio", f"{m['sharpe']:.4f}"],
        ["Sortino Ratio", f"{m['sortino']:.4f}"],
        ["Ann. Return", f"{m['ann_ret']*100:.2f}%"],
        ["Volatility", f"{m['vol']*100:.2f}%"],
        ["Max Drawdown", f"{m['mdd']*100:.2f}%"],
        ["Calmar Ratio", f"{m['calmar']:.4f}"],
        ["Excess vs TLT", f"{m['excess']*100:+.2f}%"],
        ["Total Return", f"{m['total_ret']*100:.1f}%"],
        ["Equity Finale", f"${m['equity'][-1]:.3f}"],
        ["N. Operazioni", f"{m['n_events']}"],
        ["Dead Years", f"{m['n_dead']}"],
        ["Win Rate", f"{m['win_rate']:.0%}"],
    ]
    table = ax_tab.table(cellText=metrics_data, colLabels=["Metrica", "Valore"],
                         loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2874a6")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eaf2f8")
    ax_tab.set_title("Metriche Chiave", fontsize=11, fontweight="bold", pad=10)

    # 2. Signal Parameters
    ax_params = fig.add_subplot(gs[0, 1])
    ax_params.axis("off")
    params_data = [
        ["z_windows", str(best_params["z_windows"])],
        ["z_min_periods", "20"],
        ["k_slope", f"{best_params['k_slope']:.2f}"],
        ["yc_change_window", str(best_params["yc_cw"])],
        ["k_yc", f"{best_params['k_yc']:.2f}"],
        ["tl_calm", f"{best_params['tl_c']:.3f}"],
        ["ts_calm", f"{best_params['ts_c']:.3f}"],
        ["tl_stress", f"{best_params['tl_s']:.3f}"],
        ["adapt_days", str(best_params["adapt_days"])],
        ["adapt_raise", f"{best_params['adapt_raise']:.2f}"],
        ["max_hold_days", str(best_params["max_hold_days"])],
        ["Cooldown", "5"],
        ["TC", "0 bps"],
    ]
    table2 = ax_params.table(cellText=params_data,
                              colLabels=["Parametro", "Valore"],
                              loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.0, 1.25)
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#e8f8f5")
    ax_params.set_title("Parametri Segnale", fontsize=11, fontweight="bold", pad=10)

    # 3. Trade Distribution
    ax_trades = fig.add_subplot(gs[0, 2])
    w_arr = m["weight"]
    dw = np.abs(np.diff(w_arr, prepend=w_arr[0]))
    trade_w = w_arr[dw > 0.01]
    n_long = int(np.sum(trade_w > 0.7))
    n_short = int(np.sum(trade_w < 0.3))
    n_neutral = int(np.sum((trade_w >= 0.3) & (trade_w <= 0.7)))
    labels = ["LONG", "SHORT"]
    sizes = [max(n_long, 1), max(n_short, 1)]
    if n_neutral > 0:
        labels.append("NEUTRAL")
        sizes.append(n_neutral)
    colors_pie = ["#27ae60", "#e74c3c", "#f39c12"][:len(labels)]
    ax_trades.pie(sizes, labels=labels, colors=colors_pie,
                  autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
    ax_trades.set_title(
        f"Distribuzione Operazioni ({m['n_events']} totali)",
        fontsize=11, fontweight="bold", pad=10)

    # 4. Rolling Sharpe
    ax_roll = fig.add_subplot(gs[1, 0:2])
    net_ret = m["net_ret"]
    exc = net_ret - RISK_FREE_RATE / ANN
    # Rolling sharpe (252d)
    rolling_mean = pd.Series(exc).rolling(252, min_periods=126).mean()
    rolling_std = pd.Series(exc).rolling(252, min_periods=126).std()
    rolling_sh = (rolling_mean / rolling_std * np.sqrt(ANN)).values
    n_pts = len(rolling_sh)
    ax_roll.plot(range(n_pts), rolling_sh, color="#2874a6", linewidth=1.2)
    ax_roll.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_roll.axhline(m["sharpe"], color="#27ae60", linewidth=1, linestyle="--",
                    alpha=0.7, label=f"Full-period: {m['sharpe']:.3f}")
    ax_roll.fill_between(range(n_pts), rolling_sh, 0,
                         where=(rolling_sh > 0), color="#27ae60", alpha=0.06)
    ax_roll.fill_between(range(n_pts), rolling_sh, 0,
                         where=(rolling_sh < 0), color="#e74c3c", alpha=0.06)
    ax_roll.set_ylabel("Rolling 1Y Sharpe", fontsize=10, fontweight="bold")
    ax_roll.set_title("Rolling Sharpe Ratio (252gg)", fontsize=11, fontweight="bold")
    ax_roll.legend(fontsize=9)
    ax_roll.grid(True, alpha=0.2)

    # 5. V_pres vs V2 comparison
    ax_comp = fig.add_subplot(gs[1, 2])
    ax_comp.axis("off")
    m2 = result_v2
    comp_data = [
        ["Sharpe", f"{m2['sharpe']:.3f}", f"{m['sharpe']:.3f}"],
        ["Sortino", f"{m2['sortino']:.3f}", f"{m['sortino']:.3f}"],
        ["Ann. Ret", f"{m2['ann_ret']*100:.2f}%", f"{m['ann_ret']*100:.2f}%"],
        ["MDD", f"{m2['mdd']*100:.1f}%", f"{m['mdd']*100:.1f}%"],
        ["Calmar", f"{m2['calmar']:.3f}", f"{m['calmar']:.3f}"],
        ["Events", f"{m2['n_events']}", f"{m['n_events']}"],
        ["Dead yrs", f"{m2['n_dead']}", f"{m['n_dead']}"],
        ["Win rate", f"{m2['win_rate']:.0%}", f"{m['win_rate']:.0%}"],
    ]
    table3 = ax_comp.table(cellText=comp_data,
                            colLabels=["", "V2", "V_Pres"],
                            loc="center", cellLoc="center")
    table3.auto_set_font_size(False)
    table3.set_fontsize(9)
    table3.scale(1.0, 1.5)
    for (row, col), cell in table3.get_celld().items():
        if row == 0:
            cell.set_facecolor("#6c3483")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2 and row > 0:
            cell.set_facecolor("#f4ecf7")
    ax_comp.set_title("V2 vs V_Presentazione", fontsize=11, fontweight="bold", pad=10)

    fig.savefig(os.path.join(out_dir, "05_dashboard.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_06_allocation(dates, result_pres, out_dir):
    """06: TLT weight over time."""
    equity = result_pres["equity"]
    w_vals = result_pres["weight"]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        sharex=True)

    ax1.plot(dates, equity, color="#1a5276", linewidth=1.5)
    ax1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    ax1.set_title("Allocazione TLT nel Tempo -- V_Presentazione",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))

    ax2.fill_between(dates, w_vals, 0.5,
                     where=(w_vals > 0.7), color="#27ae60", alpha=0.4,
                     step="mid", label="LONG (100% TLT)")
    ax2.fill_between(dates, w_vals, 0.5,
                     where=(w_vals < 0.3), color="#e74c3c", alpha=0.4,
                     step="mid", label="SHORT (100% SHV)")
    ax2.fill_between(dates, w_vals, 0.5,
                     where=((w_vals >= 0.3) & (w_vals <= 0.7)),
                     color="#f39c12", alpha=0.2, step="mid",
                     label="NEUTRAL (50/50)")
    ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("TLT Weight", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(os.path.join(out_dir, "06_allocation.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_07_signal_decomposition(dates, D, best_params, out_dir):
    """07: Each z-score component separately (multi-tf overlay)."""
    z_wins = eval(best_params["z_windows"]) if isinstance(best_params["z_windows"], str) \
        else best_params["z_windows"]
    rr_blend = build_rr_blend(D)

    n_panels = len(z_wins) + 2  # individual z-scores + slope + combined
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 3 * n_panels),
                              sharex=True,
                              gridspec_kw={"hspace": 0.10})

    colors = ["#2874a6", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]

    # Individual z-scores per window
    for idx, w in enumerate(z_wins):
        ax = axes[idx]
        mu = rr_blend.rolling(w, min_periods=20).mean()
        sd = rr_blend.rolling(w, min_periods=20).std().replace(0, np.nan)
        z_w = ((rr_blend - mu) / sd).reindex(dates)
        v = ~z_w.isna()
        ax.plot(dates[v], z_w.values[v], color=colors[idx % len(colors)],
                linewidth=0.7, alpha=0.8)
        ax.fill_between(dates, z_w.fillna(0).values, 0,
                        where=(z_w.fillna(0).values > 0),
                        color="#27ae60", alpha=0.06)
        ax.fill_between(dates, z_w.fillna(0).values, 0,
                        where=(z_w.fillna(0).values < 0),
                        color="#e74c3c", alpha=0.06)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        ax.set_ylabel(f"z RR (w={w})", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.15)
        ax.set_ylim(-6, 6)

    # Slope z-score
    ax_slope = axes[len(z_wins)]
    z_slope = build_slope_zscore(D).reindex(dates)
    v = ~z_slope.isna()
    ax_slope.plot(dates[v], z_slope.values[v], color="#e67e22",
                  linewidth=0.7, alpha=0.8)
    ax_slope.fill_between(dates, z_slope.fillna(0).values, 0,
                          where=(z_slope.fillna(0).values > 0),
                          color="#27ae60", alpha=0.06)
    ax_slope.fill_between(dates, z_slope.fillna(0).values, 0,
                          where=(z_slope.fillna(0).values < 0),
                          color="#e74c3c", alpha=0.06)
    ax_slope.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax_slope.set_ylabel("z Slope (10Y-2Y)", fontsize=10, fontweight="bold")
    ax_slope.grid(True, alpha=0.15)
    ax_slope.set_ylim(-4, 4)

    # Combined multi-tf z_base (mean of individual)
    ax_comb = axes[-1]
    z_parts = []
    for w in z_wins:
        mu = rr_blend.rolling(w, min_periods=20).mean()
        sd = rr_blend.rolling(w, min_periods=20).std().replace(0, np.nan)
        z_parts.append((rr_blend - mu) / sd)
    z_base_mtf = pd.concat(z_parts, axis=1).mean(axis=1).reindex(dates)
    v = ~z_base_mtf.isna()
    ax_comb.plot(dates[v], z_base_mtf.values[v], color="#1a5276",
                 linewidth=0.8, alpha=0.9)
    ax_comb.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax_comb.set_ylabel("z_base Multi-TF", fontsize=10, fontweight="bold")
    ax_comb.set_xlabel("Date", fontsize=11)
    ax_comb.grid(True, alpha=0.15)
    ax_comb.set_ylim(-6, 6)
    ax_comb.xaxis.set_major_locator(mdates.YearLocator())
    ax_comb.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    axes[0].set_title("Signal Decomposition: Multi-Timeframe Z-Scores",
                       fontsize=13, fontweight="bold", pad=12)

    fig.savefig(os.path.join(out_dir, "07_signal_decomposition.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_08_multi_objective_breakdown(df_all, out_dir):
    """08: How top configs score on each criterion."""
    top20 = df_all.nlargest(20, "composite_score")

    criteria = ["sharpe", "sortino", "calmar", "win_rate", "n_dead",
                "min_excess", "consistency"]
    labels = ["Sharpe", "Sortino", "Calmar", "Win Rate", "Dead Yrs",
              "Min Excess", "Consistency"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (crit, label) in enumerate(zip(criteria, labels)):
        if idx >= len(axes):
            break
        ax = axes[idx]
        vals = top20[crit].values
        x = range(len(vals))
        colors = ["#2874a6" if i == 0 else "#aab7b8" for i in x]
        ax.barh(x, vals, color=colors, edgecolor="white")
        ax.set_yticks(x)
        ax.set_yticklabels([f"#{i+1}" for i in x], fontsize=7)
        ax.set_xlabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()

    # Last panel: composite score
    ax = axes[7]
    vals = top20["composite_score"].values
    x = range(len(vals))
    colors = ["#e74c3c" if i == 0 else "#f5b7b1" for i in x]
    ax.barh(x, vals, color=colors, edgecolor="white")
    ax.set_yticks(x)
    ax.set_yticklabels([f"#{i+1}" for i in x], fontsize=7)
    ax.set_xlabel("Composite Score", fontsize=10)
    ax.set_title("Composite Score", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    ax.invert_yaxis()

    fig.suptitle("Multi-Objective Breakdown: Top 20 Configurations",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(out_dir, "08_multi_objective_breakdown.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_09_grid_landscape(df_all, out_dir):
    """09: Heatmap of composite score vs key parameters."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Prepare pivot data for heatmaps
    param_pairs = [
        ("ts_c", "tl_c", "ts_calm", "tl_calm"),
        ("k_yc", "yc_cw", "k_yc", "yc_change_window"),
        ("ts_c", "k_yc", "ts_calm", "k_yc"),
        ("adapt_days", "max_hold_days", "adapt_days", "max_hold_days"),
        ("tl_s", "tl_c", "tl_stress", "tl_calm"),
        ("ts_c", "tl_s", "ts_calm", "tl_stress"),
    ]

    for idx, (p1, p2, label1, label2) in enumerate(param_pairs):
        ax = axes.flatten()[idx]
        try:
            pivot = df_all.pivot_table(values="composite_score",
                                        index=p2, columns=p1,
                                        aggfunc="max")
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                          origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns],
                              fontsize=7, rotation=45)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=7)
            ax.set_xlabel(label1, fontsize=9)
            ax.set_ylabel(label2, fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data",
                   transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{label1} vs {label2}", fontsize=10, fontweight="bold")

    fig.suptitle("Grid Search Landscape: Composite Score vs Parameters",
                 fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(out_dir, "09_grid_landscape.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


# ======================================================================
# DOCUMENTATION
# ======================================================================

def save_documentation(result_pres, result_v2, best_params, out_dir):
    """Save strategy documentation."""
    m = result_pres
    m2 = result_v2

    lines = []
    lines.append("=" * 72)
    lines.append("  STRATEGY DOCUMENTATION -- VERSIONE PRESENTAZIONE")
    lines.append("  RR + Slope + YC Multi-Timeframe Duration Strategy")
    lines.append("=" * 72)
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 72)
    lines.append("Strategy: RR + Slope + YC Combined Duration Timing")
    lines.append("Instruments: TLT (long duration) / SHV (short duration)")
    lines.append("Universe: US Treasury ETFs")
    lines.append("")

    lines.append("KEY INNOVATIONS vs V2")
    lines.append("-" * 72)
    lines.append("1. Multi-Timeframe Z-Score:")
    lines.append(f"   z_base = mean(zscore(rr_blend, w) for w in {best_params['z_windows']})")
    lines.append("   Captures short/medium/long-term context simultaneously")
    lines.append("   Reduces persistent LONG bias in stable bull markets (2011-2014)")
    lines.append("")
    lines.append("2. Adaptive Threshold (position-duration based):")
    ad = best_params["adapt_days"]
    ar = best_params["adapt_raise"]
    if ad > 0:
        lines.append(f"   After {ad} days LONG, ts_calm raises by {ar:.2f}")
        lines.append("   Prevents 'always LONG' in stable bull markets")
    else:
        lines.append("   Disabled (adapt_days=0)")
    lines.append("")
    mh = best_params["max_hold_days"]
    lines.append("3. Max Hold Days:")
    if mh > 0:
        lines.append(f"   After {mh} days in same position, reverts to neutral (50/50)")
        lines.append("   Forces periodic re-evaluation of conviction")
    else:
        lines.append("   Disabled (max_hold_days=0)")
    lines.append("")
    lines.append("4. Multi-Objective Scoring:")
    lines.append("   score = 0.50*sharpe + 0.15*sortino_scaled + 0.10*calmar")
    lines.append("         + 0.05*win_rate + 0.05*min_excess - 0.03*n_dead")
    lines.append("   Sharpe-dominated with linear dead-year penalty")
    lines.append("")

    lines.append("SIGNAL CONSTRUCTION")
    lines.append("-" * 72)
    lines.append(f"z_final = z_rr_multitf + {best_params['k_yc']:.2f} * z_yc_regime_z")
    lines.append("")
    lines.append("Component 1: RR Multi-TF Composite (z_rr)")
    lines.append(f"  z_base = mean(zscore(rr_blend_ALL, w) for w in {best_params['z_windows']})")
    lines.append(f"  z_rr = z_base + {best_params['k_slope']:.2f} * z_slope_momentum(10Y-2Y)")
    lines.append("")
    lines.append("Component 2: YC Regime Z-Score (z_yc)")
    lines.append(f"  Change window: {best_params['yc_cw']} days")
    lines.append("  Score: BULL_FLAT=+2, BULL_STEEP=+1, BEAR_FLAT=-1, BEAR_STEEP=-2")
    lines.append("  Z-score window: 252 days")
    lines.append("")

    lines.append("TRADING RULES")
    lines.append("-" * 72)
    lines.append(f"  CALM:   z <= {best_params['tl_c']:.3f} -> SHORT")
    lines.append(f"          z >= {best_params['ts_c']:.3f} -> LONG")
    lines.append(f"  STRESS: z <= {best_params['tl_s']:.3f} -> SHORT")
    if ad > 0:
        lines.append(f"  ADAPTIVE: after {ad}d LONG, ts_calm += {ar:.2f}")
    if mh > 0:
        lines.append(f"  MAX HOLD: after {mh}d same pos -> NEUTRAL (50/50)")
    lines.append(f"  Cooldown: 5 days | Delay: T+2 | TC: 0 bps")
    lines.append("")

    lines.append("PERFORMANCE COMPARISON")
    lines.append("-" * 72)
    lines.append(f"  {'Metric':<18} {'V2':>12} {'V_Pres':>12} {'Change':>10}")
    lines.append(f"  {'-'*54}")
    lines.append(f"  {'Sharpe':<18} {m2['sharpe']:>12.4f} {m['sharpe']:>12.4f} "
                 f"{(m['sharpe']-m2['sharpe']):>+10.4f}")
    lines.append(f"  {'Sortino':<18} {m2['sortino']:>12.4f} {m['sortino']:>12.4f} "
                 f"{(m['sortino']-m2['sortino']):>+10.4f}")
    lines.append(f"  {'Ann. Return':<18} {m2['ann_ret']*100:>11.2f}% {m['ann_ret']*100:>11.2f}% "
                 f"{(m['ann_ret']-m2['ann_ret'])*100:>+9.2f}%")
    lines.append(f"  {'MDD':<18} {m2['mdd']*100:>11.2f}% {m['mdd']*100:>11.2f}%")
    lines.append(f"  {'Calmar':<18} {m2['calmar']:>12.4f} {m['calmar']:>12.4f}")
    lines.append(f"  {'Events':<18} {m2['n_events']:>12} {m['n_events']:>12}")
    lines.append(f"  {'Dead Years':<18} {m2['n_dead']:>12} {m['n_dead']:>12} "
                 f"{(m['n_dead']-m2['n_dead']):>+10}")
    lines.append(f"  {'Win Rate':<18} {m2['win_rate']:>11.0%} {m['win_rate']:>11.0%}")
    lines.append("")

    lines.append("YEARLY BREAKDOWN")
    lines.append("-" * 72)
    lines.append(f"  {'Year':>6}  {'V_Pres':>10}  {'V2':>10}  {'B&H':>10}  "
                 f"{'Pres Exc':>10}  {'V2 Exc':>10}")
    lines.append(f"  {'-'*64}")
    for i in range(len(m["yearly_years"])):
        y = int(m["yearly_years"][i])
        pres_ret = m["yearly_strat"][i]
        pres_exc = m["yearly_excess"][i]
        bm_ret = m["yearly_bm"][i]

        # Find V2 year
        v2_ret = bm_ret
        v2_exc = 0.0
        for j in range(len(m2["yearly_years"])):
            if m2["yearly_years"][j] == y:
                v2_ret = m2["yearly_strat"][j]
                v2_exc = m2["yearly_excess"][j]
                break

        dead_p = " D" if abs(pres_exc) < 0.01 else "  "
        dead_v2 = " D" if abs(v2_exc) < 0.01 else "  "
        lines.append(
            f"  {y:>6}  {pres_ret:>+10.4f}  {v2_ret:>+10.4f}  "
            f"{bm_ret:>+10.4f}  {pres_exc:>+10.4f}{dead_p}  "
            f"{v2_exc:>+10.4f}{dead_v2}")

    lines.append("")
    lines.append("=" * 72)

    path = os.path.join(out_dir, "strategy_documentation.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "strategy_documentation.txt"


# ======================================================================
# MAIN
# ======================================================================

def main():
    T0 = time.time()

    print("=" * 72)
    print("  VERSIONE PRESENTAZIONE")
    print("  RR + Slope + YC Multi-Timeframe Duration Strategy")
    print("  Multi-Objective Optimization (Sharpe + Consistency + Dead Years)")
    print("=" * 72)

    # ── 1. Load data ──
    print("\n[1/8] Caricamento dati ...")
    D = load_all()

    # ── 2. Regime HMM ──
    print("\n[2/8] Regime HMM ...")
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])

    # ── 3. Grid search ──
    print("\n[3/8] Grid search multi-obiettivo (3 fasi) ...")
    gs_result = run_grid_search(D, regime_hmm, verbose=True)

    df_all = gs_result["df_all"]
    common = gs_result["common"]
    reg_arr = gs_result["reg_arr"]
    r_long = gs_result["r_long"]
    r_short = gs_result["r_short"]
    year_arr = gs_result["year_arr"]

    # Best config
    best_row = df_all.iloc[0]
    best_params = {
        "z_windows": best_row["z_windows"],
        "k_slope": float(best_row["k_slope"]),
        "yc_cw": int(best_row["yc_cw"]),
        "k_yc": best_row["k_yc"],
        "tl_c": best_row["tl_c"],
        "ts_c": best_row["ts_c"],
        "tl_s": best_row["tl_s"],
        "adapt_days": int(best_row["adapt_days"]),
        "adapt_raise": float(best_row["adapt_raise"]),
        "max_hold_days": int(best_row["max_hold_days"]),
    }

    print(f"\n  {'='*62}")
    print(f"  BEST CONFIG (composite score = {best_row['composite_score']:.4f})")
    print(f"  {'='*62}")
    for k, v in best_params.items():
        print(f"    {k:<20} = {v}")

    # ── 4. Run best config for detailed results ──
    print("\n[4/8] Running best config ...")
    z_wins = eval(best_params["z_windows"]) if isinstance(best_params["z_windows"], str) \
        else best_params["z_windows"]
    z_rr_best, z_base_best, z_slope_best = build_rr_multitf(
        D, z_wins, k_slope=best_params["k_slope"])
    z_yc_best, regime_yc_best, yc_score_best = build_yc_signal(
        D, change_window=best_params["yc_cw"])
    z_final_best = build_combined(z_rr_best, z_yc_best, best_params["k_yc"])

    z_final_arr = z_final_best.reindex(common).values
    result_pres = fast_bt(z_final_arr, reg_arr, r_long, r_short, year_arr,
                          best_params["tl_c"], best_params["ts_c"],
                          best_params["tl_s"],
                          adapt_days=best_params["adapt_days"],
                          adapt_raise=best_params["adapt_raise"],
                          max_hold_days=best_params["max_hold_days"])
    # Add r_long for B&H comparison in charts
    result_pres["r_long"] = r_long

    m = result_pres
    print(f"\n  Sharpe:       {m['sharpe']:.4f}")
    print(f"  Sortino:      {m['sortino']:.4f}")
    print(f"  Ann. Return:  {m['ann_ret']*100:.2f}%")
    print(f"  Vol:          {m['vol']*100:.2f}%")
    print(f"  MDD:          {m['mdd']*100:.2f}%")
    print(f"  Calmar:       {m['calmar']:.4f}")
    print(f"  Excess:       {m['excess']*100:+.2f}%")
    print(f"  Events:       {m['n_events']}")
    print(f"  Dead years:   {m['n_dead']}")
    print(f"  Win rate:     {m['win_rate']:.0%}")

    # ── 5. V2 baseline ──
    print("\n[5/8] V2 baseline per confronto ...")
    result_v2, z_final_v2, z_rr_v2, z_yc_v2, regime_yc_v2 = \
        run_v2_baseline(D, regime_hmm, common, reg_arr, r_long, r_short, year_arr)

    print(f"  V2 Sharpe:    {result_v2['sharpe']:.4f}")
    print(f"  V2 Dead yrs:  {result_v2['n_dead']}")
    print(f"  V2 Win rate:  {result_v2['win_rate']:.0%}")

    # Yearly breakdown
    print(f"\n  {'Year':>6}  {'V_Pres':>10}  {'V2':>10}  {'B&H':>10}  "
          f"{'P_exc':>10}  {'V2_exc':>10}")
    print(f"  {'-'*64}")
    for i in range(len(m["yearly_years"])):
        y = int(m["yearly_years"][i])
        pres_ret = m["yearly_strat"][i]
        pres_exc = m["yearly_excess"][i]
        bm_ret = m["yearly_bm"][i]

        v2_exc = 0.0
        v2_ret = bm_ret
        for j in range(len(result_v2["yearly_years"])):
            if result_v2["yearly_years"][j] == y:
                v2_ret = result_v2["yearly_strat"][j]
                v2_exc = result_v2["yearly_excess"][j]
                break

        dead_p = " D" if abs(pres_exc) < 0.01 else "  "
        dead_v = " D" if abs(v2_exc) < 0.01 else "  "
        print(f"  {y:>6}  {pres_ret:>+10.4f}  {v2_ret:>+10.4f}  "
              f"{bm_ret:>+10.4f}  {pres_exc:>+10.4f}{dead_p}  "
              f"{v2_exc:>+10.4f}{dead_v}")

    pos_p = int(np.sum(m["yearly_excess"] > 0))
    pos_v2 = int(np.sum(result_v2["yearly_excess"] > 0))
    tot = len(m["yearly_years"])
    print(f"\n  V_Pres: {pos_p}/{tot} positive excess years  "
          f"({m['n_dead']} dead)")
    print(f"  V2:     {pos_v2}/{tot} positive excess years  "
          f"({result_v2['n_dead']} dead)")

    # ── 6. Grafici ──
    print("\n[6/8] Grafici ...")
    dates = common

    plot_01_equity_zscore(dates, result_pres, z_final_best, z_rr_best,
                          z_yc_best, regime_hmm, best_params, OUT_DIR)
    print("  Salvato: 01_equity_zscore.png")

    plot_02_yearly_comparison(dates, result_pres, result_v2, r_long,
                               year_arr, OUT_DIR)
    print("  Salvato: 02_yearly_comparison.png")

    plot_03_drawdown(dates, result_pres, r_long, OUT_DIR)
    print("  Salvato: 03_drawdown.png")

    plot_04_regime_timeline(dates, result_pres, regime_hmm,
                            regime_yc_best, OUT_DIR)
    print("  Salvato: 04_regime_timeline.png")

    plot_05_dashboard(result_pres, result_v2, best_params, OUT_DIR)
    print("  Salvato: 05_dashboard.png")

    plot_06_allocation(dates, result_pres, OUT_DIR)
    print("  Salvato: 06_allocation.png")

    plot_07_signal_decomposition(dates, D, best_params, OUT_DIR)
    print("  Salvato: 07_signal_decomposition.png")

    plot_08_multi_objective_breakdown(df_all, OUT_DIR)
    print("  Salvato: 08_multi_objective_breakdown.png")

    plot_09_grid_landscape(df_all, OUT_DIR)
    print("  Salvato: 09_grid_landscape.png")

    # ── 7. Salvataggio dati ──
    print("\n[7/8] Salvataggio dati ...")

    # Daily equity CSV
    eq_export = pd.DataFrame({
        "date": dates,
        "strategy_pres": result_pres["equity"],
        "benchmark_tlt": np.cumprod(1.0 + r_long),
        "weight_tlt": result_pres["weight"],
        "net_return": result_pres["net_ret"],
        "z_final": z_final_best.reindex(dates).values,
        "z_rr": z_rr_best.reindex(dates).values,
        "z_yc": z_yc_best.reindex(dates).values,
        "regime_hmm": regime_hmm.reindex(dates).values,
        "regime_yc": regime_yc_best.reindex(dates).values,
    })
    eq_export.to_csv(os.path.join(OUT_DIR, "daily_equity.csv"), index=False)
    print("  Salvato: daily_equity.csv")

    # Trades CSV
    w_arr = result_pres["weight"]
    dw = np.abs(np.diff(w_arr, prepend=w_arr[0]))
    trade_idx = np.where(dw > 0.01)[0]
    trades_list = []
    for ti in trade_idx:
        signal = "LONG" if w_arr[ti] > 0.7 else ("SHORT" if w_arr[ti] < 0.3 else "NEUTRAL")
        trades_list.append({
            "date": dates[ti],
            "signal": signal,
            "z_final": z_final_best.reindex(dates).values[ti]
                if ti < len(dates) else np.nan,
            "weight": w_arr[ti],
            "regime": regime_hmm.reindex(dates).values[ti]
                if ti < len(dates) else "UNKNOWN",
        })
    trades_df = pd.DataFrame(trades_list)
    trades_df.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
    print(f"  Salvato: trades.csv ({len(trades_df)} trades)")

    # Yearly CSV
    yearly_export = pd.DataFrame({
        "year": m["yearly_years"].astype(int),
        "strat_ret": m["yearly_strat"],
        "bm_ret": m["yearly_bm"],
        "excess": m["yearly_excess"],
    })
    yearly_export.to_csv(os.path.join(OUT_DIR, "yearly.csv"), index=False)
    print("  Salvato: yearly.csv")

    # Metrics CSV
    metrics_export = {
        **best_params,
        "sharpe": m["sharpe"],
        "sortino": m["sortino"],
        "ann_ret": m["ann_ret"],
        "vol": m["vol"],
        "mdd": m["mdd"],
        "calmar": m["calmar"],
        "excess": m["excess"],
        "total_ret": m["total_ret"],
        "n_events": m["n_events"],
        "n_dead": m["n_dead"],
        "win_rate": m["win_rate"],
        "composite_score": best_row["composite_score"],
    }
    pd.DataFrame([metrics_export]).to_csv(
        os.path.join(OUT_DIR, "metrics.csv"), index=False)
    print("  Salvato: metrics.csv")

    # Grid results CSV (top 200)
    df_all.head(200).to_csv(
        os.path.join(OUT_DIR, "grid_results.csv"), index=False)
    print("  Salvato: grid_results.csv (top 200)")

    # Top configs CSV
    df_all.head(20).to_csv(
        os.path.join(OUT_DIR, "top_configs.csv"), index=False)
    print("  Salvato: top_configs.csv (top 20)")

    # ── 8. Documentazione ──
    print("\n[8/8] Documentazione ...")
    doc = save_documentation(result_pres, result_v2, best_params, OUT_DIR)
    print(f"  Salvato: {doc}")

    elapsed = time.time() - T0
    print(f"\n{'='*72}")
    print(f"  VERSIONE PRESENTAZIONE COMPLETATA")
    print(f"  Best Composite Score: {best_row['composite_score']:.4f}")
    print(f"  Sharpe: {m['sharpe']:.4f}  |  Dead years: {m['n_dead']}  |  "
          f"Win rate: {m['win_rate']:.0%}")
    print(f"  vs V2:  Sharpe {result_v2['sharpe']:.4f}  |  "
          f"Dead years: {result_v2['n_dead']}  |  "
          f"Win rate: {result_v2['win_rate']:.0%}")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Files:  9 grafici + 6 CSV + 1 documentazione")
    print(f"  Tempo:  {elapsed:.1f}s")
    print(f"{'='*72}")

    return result_pres, result_v2, best_params, df_all


if __name__ == "__main__":
    main()
