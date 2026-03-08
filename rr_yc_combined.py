"""
===============================================================================
  RR + YC COMBINED STRATEGY
  Combines RR composite z-score with YC regime_z signal.

  Four combination methods:
    A) Additive:   z_final = z_rr + k_yc * z_yc  (with HMM regime thresholds)
    B) Weighted:   z_final = alpha * z_rr + (1-alpha) * z_yc  (unified thresholds)
    C) Filter:     RR signals only when YC confirms direction
    D) Agreement:  trade only when both signals agree on direction
===============================================================================
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

SAVE_DIR = os.path.join(os.path.dirname(__file__), "output", "rr_yc_combined")
os.makedirs(SAVE_DIR, exist_ok=True)

ANN = 252

# ─── RR parameters (frozen from versione_finale) ───────────────────────────
RR_PARAMS = dict(
    rr_tenors=["1W", "1M", "3M", "6M"],
    z_window=63, z_min_periods=20,
    slope_long="US_10Y", slope_short="US_2Y",
    slope_mom_window=126, slope_z_window=252, slope_z_min=63,
    k_slope=0.50,
    tl_calm=-3.25, ts_calm=+2.75, tl_stress=-4.00,
)

# ─── YC regime_z parameters (best from V2) ─────────────────────────────────
YC_PARAMS = dict(
    spread_short="US_10Y", spread_long="US_30Y",
    change_window=42, z_window=252, z_min_periods=20,
)

COOLDOWN  = 5
DELAY     = 1
TCOST_BPS = 5
W_INIT    = 0.5

SCORE_MAP = {"BULL_FLAT": +2, "BULL_STEEP": +1, "UNDEFINED": 0,
             "BEAR_FLAT": -1, "BEAR_STEEP": -2}


# ─── Signal construction ───────────────────────────────────────────────────

def build_z_rr(D):
    """Reconstruct the RR composite z-score."""
    options = D["options"]
    yields  = D["yields"]
    p = RR_PARAMS

    # RR blend (all tenors)
    parts = []
    for tenor in p["rr_tenors"]:
        pc, cc = RR_TENORS[tenor]
        rr = (pd.to_numeric(options[pc], errors="coerce")
              - pd.to_numeric(options[cc], errors="coerce"))
        parts.append(rr)
    rr_blend = pd.concat(parts, axis=1).mean(axis=1)

    # Z-score
    w, mp = p["z_window"], p["z_min_periods"]
    mu = rr_blend.rolling(w, min_periods=mp).mean()
    sd = rr_blend.rolling(w, min_periods=mp).std().replace(0, np.nan)
    z_base = (rr_blend - mu) / sd

    # Slope momentum
    slope = yields[p["slope_long"]] - yields[p["slope_short"]]
    ds = slope.diff(p["slope_mom_window"])
    sw, sm = p["slope_z_window"], p["slope_z_min"]
    mu_s = ds.rolling(sw, min_periods=sm).mean()
    sd_s = ds.rolling(sw, min_periods=sm).std().replace(0, np.nan)
    z_slope = (ds - mu_s) / sd_s

    z_composite = z_base + p["k_slope"] * z_slope
    return z_composite, z_base, z_slope


def build_z_yc(D):
    """Build YC regime_z signal."""
    yld = D["yields"]
    p = YC_PARAMS
    sc, lc = p["spread_short"], p["spread_long"]
    cw = p["change_window"]

    lev = (yld[sc] + yld[lc]) / 2
    spr = yld[lc] - yld[sc]
    dl, ds = lev.diff(cw), spr.diff(cw)

    r = pd.Series("UNDEFINED", index=yld.index)
    r[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    r[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    r[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    r[(dl > 0) & (ds < 0)] = "BEAR_FLAT"

    score = r.map(SCORE_MAP).astype(float)
    zw, zm = p["z_window"], p["z_min_periods"]
    mu = score.rolling(zw, min_periods=zm).mean()
    sd = score.rolling(zw, min_periods=zm).std().replace(0, np.nan)
    z_yc = (score - mu) / sd
    return z_yc


# ─── Backtest engines ──────────────────────────────────────────────────────

def bt_with_regime(z_arr, reg_arr, ret_tlt, ret_shv,
                   tl_c, ts_c, tl_s, ts_s=99.0):
    """Backtest using HMM regime (CALM/STRESS) thresholds."""
    n = len(z_arr)
    aw = np.full(n, W_INIT)
    cur = W_INIT
    last_sig = -COOLDOWN - 1
    n_trades = 0

    for i in range(n):
        zi = z_arr[i]
        if zi != zi:
            continue
        nw = None
        if reg_arr[i] == 0:   # CALM
            if zi <= tl_c and cur != 0.0:
                nw = 0.0
            elif zi >= ts_c and cur != 1.0:
                nw = 1.0
        else:                  # STRESS
            if zi <= tl_s and cur != 0.0:
                nw = 0.0
            elif ts_s < 90 and zi >= ts_s and cur != 1.0:
                nw = 1.0

        if nw is not None and abs(nw - cur) > 0.01 and (i - last_sig) > COOLDOWN:
            ed = i + DELAY
            if ed < n:
                cur = nw
                last_sig = i
                n_trades += 1
                aw[ed:] = nw

    return _compute_metrics(aw, ret_tlt, ret_shv, n_trades)


def bt_sticky(z_arr, ret_tlt, ret_shv, tl, ts):
    """Sticky threshold backtest (no regime)."""
    n = len(z_arr)
    aw = np.full(n, W_INIT)
    cur = W_INIT
    last_sig = -COOLDOWN - 1
    n_trades = 0

    for i in range(n):
        zi = z_arr[i]
        if zi != zi:
            continue
        nw = None
        if zi >= ts and cur != 1.0:
            nw = 1.0
        elif zi <= tl and cur != 0.0:
            nw = 0.0
        if nw is not None and abs(nw - cur) > 0.01 and (i - last_sig) > COOLDOWN:
            ed = i + DELAY
            if ed < n:
                cur = nw
                last_sig = i
                n_trades += 1
                aw[ed:] = nw

    return _compute_metrics(aw, ret_tlt, ret_shv, n_trades)


def bt_filter(z_rr, z_yc, reg_arr, ret_tlt, ret_shv,
              tl_c, ts_c, tl_s, yc_confirm):
    """RR signals filtered by YC direction.
    yc_confirm: require z_yc > yc_confirm for LONG, z_yc < -yc_confirm for SHORT.
    """
    n = len(z_rr)
    aw = np.full(n, W_INIT)
    cur = W_INIT
    last_sig = -COOLDOWN - 1
    n_trades = 0

    for i in range(n):
        zr = z_rr[i]
        zy = z_yc[i]
        if zr != zr or zy != zy:
            continue
        nw = None
        if reg_arr[i] == 0:   # CALM
            if zr <= tl_c and zy < -yc_confirm and cur != 0.0:
                nw = 0.0
            elif zr >= ts_c and zy > yc_confirm and cur != 1.0:
                nw = 1.0
        else:                  # STRESS
            if zr <= tl_s and zy < -yc_confirm and cur != 0.0:
                nw = 0.0

        if nw is not None and abs(nw - cur) > 0.01 and (i - last_sig) > COOLDOWN:
            ed = i + DELAY
            if ed < n:
                cur = nw
                last_sig = i
                n_trades += 1
                aw[ed:] = nw

    return _compute_metrics(aw, ret_tlt, ret_shv, n_trades)


def bt_agreement(z_rr, z_yc, ret_tlt, ret_shv,
                 rr_long_th, rr_short_th, yc_long_th, yc_short_th):
    """Trade only when both RR and YC signals agree on direction."""
    n = len(z_rr)
    aw = np.full(n, W_INIT)
    cur = W_INIT
    last_sig = -COOLDOWN - 1
    n_trades = 0

    for i in range(n):
        zr = z_rr[i]
        zy = z_yc[i]
        if zr != zr or zy != zy:
            continue
        nw = None
        both_long  = zr >= rr_long_th and zy >= yc_long_th
        both_short = zr <= rr_short_th and zy <= yc_short_th
        if both_long and cur != 1.0:
            nw = 1.0
        elif both_short and cur != 0.0:
            nw = 0.0
        if nw is not None and abs(nw - cur) > 0.01 and (i - last_sig) > COOLDOWN:
            ed = i + DELAY
            if ed < n:
                cur = nw
                last_sig = i
                n_trades += 1
                aw[ed:] = nw

    return _compute_metrics(aw, ret_tlt, ret_shv, n_trades)


def _compute_metrics(aw, ret_tlt, ret_shv, n_trades):
    dr = aw * ret_tlt + (1 - aw) * ret_shv
    wd = np.abs(np.diff(aw, prepend=aw[0]))
    dr -= wd * (TCOST_BPS / 10000.0)
    eq = np.cumprod(1 + dr)
    n = len(dr)
    ann_ret = eq[-1] ** (ANN / n) - 1
    vol = np.std(dr) * np.sqrt(ANN)
    exc = dr - RISK_FREE_RATE / ANN
    sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)
    neg = exc[exc < 0]
    ds = np.sqrt(np.mean(neg**2)) * np.sqrt(ANN) if len(neg) > 0 else 1e-10
    sortino = np.mean(exc) * ANN / (ds + 1e-10)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0.0
    return dict(sharpe=sharpe, sortino=sortino, ann_ret=ann_ret, vol=vol,
                mdd=mdd, calmar=calmar, total_ret=eq[-1]-1,
                n_trades=n_trades, equity=eq, daily_ret=dr, actual_w=aw)


# ─── Charts ────────────────────────────────────────────────────────────────

def plot_equity(dates, results_dict, eq_bh, path, title=""):
    """Plot multiple equity curves."""
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = ["blue", "red", "green", "purple", "orange", "brown"]
    for i, (label, res) in enumerate(results_dict.items()):
        ax.plot(dates, res["equity"], color=colors[i % len(colors)],
                lw=1.5, label=f"{label} (Sh={res['sharpe']:.3f})")
    ax.plot(dates, eq_bh, color="gray", lw=1, alpha=0.5, label="B&H TLT")
    ax.set_ylabel("Equity ($)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_best_equity(dates, eq_s, eq_bh, aw, z_rr, z_yc, path, title=""):
    fig, axes = plt.subplots(4, 1, figsize=(15, 12),
        height_ratios=[3, 1, 1, 0.8], sharex=True,
        gridspec_kw={"hspace": 0.08})
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(dates, eq_s, "b-", lw=1.5, label="Combined RR+YC")
    ax1.plot(dates, eq_bh, color="gray", lw=1, alpha=0.5, label="B&H TLT")
    wd = np.abs(np.diff(aw, prepend=aw[0]))
    for d in np.where(wd > 0.01)[0]:
        c = "green" if aw[d] > 0.7 else "red"
        mk = "^" if aw[d] > 0.7 else "v"
        ax1.plot(dates[d], eq_s[d], mk, color=c, ms=7, zorder=5)
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    v = ~np.isnan(z_rr)
    ax2.plot(dates[v], z_rr[v], "b-", lw=0.6, alpha=0.7)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("z RR")
    ax2.grid(True, alpha=0.3)

    v2 = ~np.isnan(z_yc)
    ax3.plot(dates[v2], z_yc[v2], "r-", lw=0.6, alpha=0.7)
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.set_ylabel("z YC")
    ax3.grid(True, alpha=0.3)

    ax4.fill_between(dates, aw, 0.5, where=aw > 0.7,
                     color="green", alpha=0.3, label="LONG")
    ax4.fill_between(dates, aw, 0.5, where=aw < 0.3,
                     color="red", alpha=0.3, label="SHORT")
    ax4.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax4.set_ylabel("TLT Wt")
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(loc="lower right", fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_yearly(yearly, path, title=""):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(yearly))
    w = 0.35
    ax.bar(x - w/2, yearly["strategy"]*100, w, label="Strategy", color="steelblue")
    ax.bar(x + w/2, yearly["benchmark"]*100, w, label="B&H TLT", color="gray", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly["year"].astype(int), rotation=45)
    ax.set_ylabel("Return (%)")
    ax.set_title(title or "Yearly Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─── Helpers ────────────────────────────────────────────────────────────────

def yearly_table(daily_ret, actual_w, dates, ret_tlt):
    years = sorted(set(d.year for d in dates))
    rows = []
    for y in years:
        mask = np.array([d.year == y for d in dates])
        if mask.sum() == 0: continue
        eq_y = np.cumprod(1 + daily_ret[mask])
        bh_y = np.cumprod(1 + ret_tlt[mask])
        wd = np.abs(np.diff(actual_w[mask], prepend=actual_w[mask][0]))
        rows.append(dict(year=y, strategy=eq_y[-1]-1, benchmark=bh_y[-1]-1,
                         excess=eq_y[-1]-1-(bh_y[-1]-1),
                         n_trades=int(np.sum(wd > 0.01))))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  RR + YC COMBINED STRATEGY")
    print("  Combining RR composite z-score with YC regime_z signal")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    D = load_all()
    yld     = D["yields"]
    etf_ret = D["etf_ret"]

    # HMM regime
    print("\n  Fitting HMM regime...")
    regime = fit_hmm_regime(D["move"], D["vix"])

    # Build signals
    print("  Building RR composite signal...")
    z_rr, z_rr_base, z_slope = build_z_rr(D)

    print("  Building YC regime_z signal...")
    z_yc = build_z_yc(D)

    # Align dates
    bt_start = pd.Timestamp(BACKTEST_START)
    common = (z_rr.dropna().index
              .intersection(z_yc.dropna().index)
              .intersection(regime.dropna().index)
              .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index))
    common = common[common >= bt_start].sort_values()

    z_rr_a  = z_rr.reindex(common).values
    z_yc_a  = z_yc.reindex(common).values
    reg_a   = (regime.reindex(common) == "STRESS").astype(int).values
    ret_tlt = etf_ret["TLT"].reindex(common).fillna(0).values
    ret_shv = etf_ret["SHV"].reindex(common).fillna(0).values
    dates   = common
    n_days  = len(common)

    eq_bh = np.cumprod(1 + ret_tlt)
    bh_exc = ret_tlt - RISK_FREE_RATE / ANN
    bh_sharpe = np.mean(bh_exc) / (np.std(bh_exc)+1e-10) * np.sqrt(ANN)

    print(f"\n  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d}  "
          f"({n_days} days)")
    print(f"  B&H TLT: Sharpe={bh_sharpe:.3f}  Return={eq_bh[-1]-1:+.1%}")

    # ── Baseline: RR-only (versione_finale config) ────────────────────
    print("\n  Running baseline (RR only)...")
    p = RR_PARAMS
    res_rr = bt_with_regime(z_rr_a, reg_a, ret_tlt, ret_shv,
                            p["tl_calm"], p["ts_calm"], p["tl_stress"])
    print(f"    RR-only: Sharpe={res_rr['sharpe']:.4f}  "
          f"Ret={res_rr['ann_ret']:+.2%}  MDD={res_rr['mdd']:.2%}  "
          f"Trades={res_rr['n_trades']}")

    # ── Baseline: YC-only ─────────────────────────────────────────────
    res_yc = bt_sticky(z_yc_a, ret_tlt, ret_shv, -1.0, 0.5)
    print(f"    YC-only: Sharpe={res_yc['sharpe']:.4f}  "
          f"Ret={res_yc['ann_ret']:+.2%}  MDD={res_yc['mdd']:.2%}  "
          f"Trades={res_yc['n_trades']}")

    # ══════════════════════════════════════════════════════════════════
    # METHOD A: ADDITIVE  z = z_rr + k_yc * z_yc (with HMM thresholds)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  METHOD A: ADDITIVE  z = z_rr + k * z_yc  (HMM thresholds)")
    print("=" * 70)

    k_grid = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
              0.75, 1.00]
    # Also test with adjusted thresholds
    tl_c_grid = [-3.75, -3.50, -3.25, -3.00, -2.75]
    ts_c_grid = [2.25, 2.50, 2.75, 3.00, 3.25]
    tl_s_grid = [-4.50, -4.25, -4.00, -3.75, -3.50]

    rows_a = []
    for k_yc in k_grid:
        z_comb = z_rr_a + k_yc * z_yc_a
        for tl_c in tl_c_grid:
            for ts_c in ts_c_grid:
                for tl_s in tl_s_grid:
                    m = bt_with_regime(z_comb, reg_a, ret_tlt, ret_shv,
                                      tl_c, ts_c, tl_s)
                    rows_a.append(dict(
                        method="A", k_yc=k_yc,
                        tl_c=tl_c, ts_c=ts_c, tl_s=tl_s,
                        sharpe=m["sharpe"], sortino=m["sortino"],
                        ann_ret=m["ann_ret"], vol=m["vol"],
                        mdd=m["mdd"], calmar=m["calmar"],
                        total_ret=m["total_ret"], n_trades=m["n_trades"]))

    df_a = pd.DataFrame(rows_a)
    df_af = df_a[df_a["n_trades"] >= 8]
    print(f"  Tested {len(df_a)} configs ({len(df_af)} with >=8 trades)")

    top10_a = df_af.nlargest(10, "sharpe")
    print(f"\n  {'#':>3} {'k_yc':>5} {'tl_c':>6} {'ts_c':>6} {'tl_s':>6}"
          f" {'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Tr':>4}")
    print("  " + "-" * 62)
    for i, (_, r) in enumerate(top10_a.iterrows(), 1):
        print(f"  {i:>3} {r['k_yc']:>5.2f} {r['tl_c']:>6.2f} "
              f"{r['ts_c']:>6.2f} {r['tl_s']:>6.2f} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>4.0f}")

    # Best per k_yc
    print("\n  Best per k_yc:")
    for k in k_grid:
        sub = df_af[df_af["k_yc"] == k]
        if len(sub) == 0: continue
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"    k={k:.2f}: Sharpe={b['sharpe']:.3f}  "
              f"[{b['tl_c']:.2f},{b['ts_c']:.2f},{b['tl_s']:.2f}]  "
              f"tr={b['n_trades']:.0f}")

    # ══════════════════════════════════════════════════════════════════
    # METHOD B: WEIGHTED  z = alpha*z_rr + (1-alpha)*z_yc (no HMM)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  METHOD B: WEIGHTED  z = a*z_rr + (1-a)*z_yc  (unified thresholds)")
    print("=" * 70)

    alpha_grid = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    tl_grid_b  = [-3.0, -2.5, -2.0, -1.5, -1.0]
    ts_grid_b  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    rows_b = []
    for alpha in alpha_grid:
        z_w = alpha * z_rr_a + (1 - alpha) * z_yc_a
        for tl in tl_grid_b:
            for ts in ts_grid_b:
                m = bt_sticky(z_w, ret_tlt, ret_shv, tl, ts)
                rows_b.append(dict(
                    method="B", alpha=alpha, tl=tl, ts=ts,
                    sharpe=m["sharpe"], sortino=m["sortino"],
                    ann_ret=m["ann_ret"], vol=m["vol"],
                    mdd=m["mdd"], calmar=m["calmar"],
                    total_ret=m["total_ret"], n_trades=m["n_trades"]))

    df_b = pd.DataFrame(rows_b)
    df_bf = df_b[df_b["n_trades"] >= 8]
    print(f"  Tested {len(df_b)} configs ({len(df_bf)} with >=8 trades)")

    top10_b = df_bf.nlargest(10, "sharpe")
    print(f"\n  {'#':>3} {'alpha':>6} {'TL':>5} {'TS':>5}"
          f" {'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Tr':>4}")
    print("  " + "-" * 52)
    for i, (_, r) in enumerate(top10_b.iterrows(), 1):
        print(f"  {i:>3} {r['alpha']:>6.2f} {r['tl']:>5.1f} "
              f"{r['ts']:>5.1f} {r['sharpe']:>7.3f} "
              f"{r['ann_ret']:>+7.1%} {r['mdd']:>8.1%} "
              f"{r['n_trades']:>4.0f}")

    print("\n  Best per alpha:")
    for a in alpha_grid:
        sub = df_bf[df_bf["alpha"] == a]
        if len(sub) == 0: continue
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"    a={a:.1f}: Sharpe={b['sharpe']:.3f}  "
              f"tl={b['tl']:.1f} ts={b['ts']:.1f}  tr={b['n_trades']:.0f}")

    # ══════════════════════════════════════════════════════════════════
    # METHOD C: FILTER (RR signals confirmed by YC direction)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  METHOD C: FILTER  RR signals + YC confirmation")
    print("=" * 70)

    confirm_grid = [-0.5, 0.0, 0.25, 0.5, 1.0]

    rows_c = []
    for yc_cf in confirm_grid:
        for tl_c in tl_c_grid:
            for ts_c in ts_c_grid:
                for tl_s in tl_s_grid:
                    m = bt_filter(z_rr_a, z_yc_a, reg_a, ret_tlt, ret_shv,
                                  tl_c, ts_c, tl_s, yc_cf)
                    rows_c.append(dict(
                        method="C", yc_confirm=yc_cf,
                        tl_c=tl_c, ts_c=ts_c, tl_s=tl_s,
                        sharpe=m["sharpe"], sortino=m["sortino"],
                        ann_ret=m["ann_ret"], vol=m["vol"],
                        mdd=m["mdd"], calmar=m["calmar"],
                        total_ret=m["total_ret"], n_trades=m["n_trades"]))

    df_c = pd.DataFrame(rows_c)
    df_cf = df_c[df_c["n_trades"] >= 8]
    print(f"  Tested {len(df_c)} configs ({len(df_cf)} with >=8 trades)")

    top10_c = df_cf.nlargest(10, "sharpe")
    print(f"\n  {'#':>3} {'YC_cf':>6} {'tl_c':>6} {'ts_c':>6} {'tl_s':>6}"
          f" {'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Tr':>4}")
    print("  " + "-" * 64)
    for i, (_, r) in enumerate(top10_c.iterrows(), 1):
        print(f"  {i:>3} {r['yc_confirm']:>6.2f} {r['tl_c']:>6.2f} "
              f"{r['ts_c']:>6.2f} {r['tl_s']:>6.2f} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>4.0f}")

    print("\n  Best per YC confirm threshold:")
    for cf in confirm_grid:
        sub = df_cf[df_cf["yc_confirm"] == cf]
        if len(sub) == 0: continue
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"    confirm={cf:>5.2f}: Sharpe={b['sharpe']:.3f}  "
              f"tr={b['n_trades']:.0f}")

    # ══════════════════════════════════════════════════════════════════
    # METHOD D: AGREEMENT (both signals must agree)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  METHOD D: AGREEMENT  both signals must agree")
    print("=" * 70)

    rr_l_grid = [2.0, 2.5, 2.75, 3.0]
    rr_s_grid = [-3.75, -3.5, -3.25, -3.0, -2.5]
    yc_l_grid = [-0.5, 0.0, 0.25, 0.5, 1.0]
    yc_s_grid = [-1.0, -0.5, -0.25, 0.0, 0.5]

    rows_d = []
    for rr_l in rr_l_grid:
        for rr_s in rr_s_grid:
            for yc_l in yc_l_grid:
                for yc_s in yc_s_grid:
                    m = bt_agreement(z_rr_a, z_yc_a, ret_tlt, ret_shv,
                                     rr_l, rr_s, yc_l, yc_s)
                    rows_d.append(dict(
                        method="D", rr_long=rr_l, rr_short=rr_s,
                        yc_long=yc_l, yc_short=yc_s,
                        sharpe=m["sharpe"], sortino=m["sortino"],
                        ann_ret=m["ann_ret"], vol=m["vol"],
                        mdd=m["mdd"], calmar=m["calmar"],
                        total_ret=m["total_ret"], n_trades=m["n_trades"]))

    df_d = pd.DataFrame(rows_d)
    df_df = df_d[df_d["n_trades"] >= 8]
    print(f"  Tested {len(df_d)} configs ({len(df_df)} with >=8 trades)")

    top10_d = df_df.nlargest(10, "sharpe")
    print(f"\n  {'#':>3} {'RR_L':>5} {'RR_S':>6} {'YC_L':>5} {'YC_S':>5}"
          f" {'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Tr':>4}")
    print("  " + "-" * 58)
    for i, (_, r) in enumerate(top10_d.iterrows(), 1):
        print(f"  {i:>3} {r['rr_long']:>5.1f} {r['rr_short']:>6.2f} "
              f"{r['yc_long']:>5.2f} {r['yc_short']:>5.2f} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>4.0f}")

    # ══════════════════════════════════════════════════════════════════
    # OVERALL COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  OVERALL COMPARISON")
    print("=" * 70)

    best_a = df_af.loc[df_af["sharpe"].idxmax()] if len(df_af) > 0 else None
    best_b = df_bf.loc[df_bf["sharpe"].idxmax()] if len(df_bf) > 0 else None
    best_c = df_cf.loc[df_cf["sharpe"].idxmax()] if len(df_cf) > 0 else None
    best_d = df_df.loc[df_df["sharpe"].idxmax()] if len(df_df) > 0 else None

    results_summary = {
        "B&H TLT":  {"sharpe": bh_sharpe, "ann_ret": eq_bh[-1]**(ANN/n_days)-1,
                      "mdd": ((eq_bh-np.maximum.accumulate(eq_bh))/np.maximum.accumulate(eq_bh)).min(),
                      "n_trades": 0},
        "RR-only":  {"sharpe": res_rr["sharpe"], "ann_ret": res_rr["ann_ret"],
                      "mdd": res_rr["mdd"], "n_trades": res_rr["n_trades"]},
        "YC-only":  {"sharpe": res_yc["sharpe"], "ann_ret": res_yc["ann_ret"],
                      "mdd": res_yc["mdd"], "n_trades": res_yc["n_trades"]},
    }

    for lbl, best, name_map in [
        ("A-Additive", best_a, lambda r: f"k={r['k_yc']:.2f}"),
        ("B-Weighted", best_b, lambda r: f"a={r['alpha']:.1f}"),
        ("C-Filter",   best_c, lambda r: f"cf={r['yc_confirm']:.2f}"),
        ("D-Agreement",best_d, lambda r: ""),
    ]:
        if best is not None:
            results_summary[lbl] = {
                "sharpe": best["sharpe"], "ann_ret": best["ann_ret"],
                "mdd": best["mdd"], "n_trades": best["n_trades"]}

    print(f"\n  {'Method':>14} {'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Tr':>5}")
    print("  " + "-" * 48)
    for name, vals in results_summary.items():
        print(f"  {name:>14} {vals['sharpe']:>7.3f} "
              f"{vals['ann_ret']:>+7.1%} {vals['mdd']:>8.1%} "
              f"{vals['n_trades']:>5.0f}")

    # ── Find overall best combined method ─────────────────────────────
    all_bests = []
    if best_a is not None: all_bests.append(("A", best_a))
    if best_b is not None: all_bests.append(("B", best_b))
    if best_c is not None: all_bests.append(("C", best_c))
    if best_d is not None: all_bests.append(("D", best_d))

    overall_best = max(all_bests, key=lambda x: x[1]["sharpe"])
    ob_method, ob = overall_best

    print(f"\n  >>> OVERALL BEST: Method {ob_method}  "
          f"Sharpe={ob['sharpe']:.4f}")

    # ── Re-run overall best for deep dive ─────────────────────────────
    if ob_method == "A":
        z_best = z_rr_a + ob["k_yc"] * z_yc_a
        res_best = bt_with_regime(z_best, reg_a, ret_tlt, ret_shv,
                                  ob["tl_c"], ob["ts_c"], ob["tl_s"])
        label = (f"Additive k={ob['k_yc']:.2f} "
                 f"[{ob['tl_c']:.2f},{ob['ts_c']:.2f},{ob['tl_s']:.2f}]")
    elif ob_method == "B":
        z_best = ob["alpha"] * z_rr_a + (1 - ob["alpha"]) * z_yc_a
        res_best = bt_sticky(z_best, ret_tlt, ret_shv, ob["tl"], ob["ts"])
        label = f"Weighted a={ob['alpha']:.1f} [{ob['tl']:.1f},{ob['ts']:.1f}]"
    elif ob_method == "C":
        res_best = bt_filter(z_rr_a, z_yc_a, reg_a, ret_tlt, ret_shv,
                             ob["tl_c"], ob["ts_c"], ob["tl_s"],
                             ob["yc_confirm"])
        label = (f"Filter cf={ob['yc_confirm']:.2f} "
                 f"[{ob['tl_c']:.2f},{ob['ts_c']:.2f},{ob['tl_s']:.2f}]")
    else:
        res_best = bt_agreement(z_rr_a, z_yc_a, ret_tlt, ret_shv,
                                ob["rr_long"], ob["rr_short"],
                                ob["yc_long"], ob["yc_short"])
        label = (f"Agreement RR[{ob['rr_short']:.1f},{ob['rr_long']:.1f}] "
                 f"YC[{ob['yc_short']:.2f},{ob['yc_long']:.2f}]")

    print(f"\n  Config: {label}")
    print(f"    Sharpe  = {res_best['sharpe']:.4f}")
    print(f"    Sortino = {res_best['sortino']:.4f}")
    print(f"    AnnRet  = {res_best['ann_ret']:+.2%}")
    print(f"    Vol     = {res_best['vol']:.2%}")
    print(f"    MDD     = {res_best['mdd']:.2%}")
    print(f"    Calmar  = {res_best['calmar']:.4f}")
    print(f"    TotRet  = {res_best['total_ret']:+.2%}")
    print(f"    Trades  = {res_best['n_trades']}")

    # Yearly
    yearly = yearly_table(res_best["daily_ret"], res_best["actual_w"],
                          dates, ret_tlt)
    print(f"\n    {'Year':>6} {'Strat':>9} {'B&H':>9} {'Excess':>9} {'Tr':>4}")
    print("    " + "-" * 40)
    for _, r in yearly.iterrows():
        mk = " *" if abs(r["excess"]) > 0.01 else ""
        print(f"    {r['year']:>6.0f} {r['strategy']:>+9.1%} "
              f"{r['benchmark']:>+9.1%} {r['excess']:>+9.1%} "
              f"{r['n_trades']:>4.0f}{mk}")
    pos_ex = (yearly["excess"] > 0).sum()
    print(f"\n    Positive excess: {pos_ex}/{len(yearly)} "
          f"({pos_ex/len(yearly):.0%})")
    print(f"    Avg excess: {yearly['excess'].mean():+.2%}")

    # ── Charts ────────────────────────────────────────────────────────
    # Equity comparison
    eq_curves = {"RR-only": res_rr, "YC-only": res_yc}
    eq_curves[f"Best Combined ({ob_method})"] = res_best
    plot_equity(dates, eq_curves, eq_bh,
                os.path.join(SAVE_DIR, "01_equity_comparison.png"),
                "RR + YC Combined: Equity Comparison")
    print(f"\n  -> 01_equity_comparison.png")

    # Best config detailed
    plot_best_equity(dates, res_best["equity"], eq_bh,
                     res_best["actual_w"], z_rr_a, z_yc_a,
                     os.path.join(SAVE_DIR, "02_best_detailed.png"),
                     f"Best Combined — {label}\n"
                     f"Sharpe={res_best['sharpe']:.3f}  "
                     f"Return={res_best['total_ret']:+.1%}  "
                     f"MDD={res_best['mdd']:.1%}")
    print(f"  -> 02_best_detailed.png")

    # Yearly
    plot_yearly(yearly, os.path.join(SAVE_DIR, "03_yearly.png"),
                f"RR+YC Combined: Yearly ({label})")
    print(f"  -> 03_yearly.png")

    # ── Save ──────────────────────────────────────────────────────────
    df_a.to_csv(os.path.join(SAVE_DIR, "method_a_results.csv"), index=False)
    df_b.to_csv(os.path.join(SAVE_DIR, "method_b_results.csv"), index=False)
    df_c.to_csv(os.path.join(SAVE_DIR, "method_c_results.csv"), index=False)
    df_d.to_csv(os.path.join(SAVE_DIR, "method_d_results.csv"), index=False)
    yearly.to_csv(os.path.join(SAVE_DIR, "yearly_best.csv"), index=False)
    print(f"\n  CSVs saved to {SAVE_DIR}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
