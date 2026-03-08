"""
===============================================================================
  YC CONTINUOUS SIGNAL STRATEGY
  Uses yield curve level/spread changes as continuous z-scored signals.
  Trades TLT vs SHV based on z-score threshold crossings (sticky positions).

  Signal construction:
    For each spread pair (short, long):
      level  = (short + long) / 2
      spread = long - short
      raw    = -(level.diff(cw) + k * spread.diff(cw))

    k=0.00 → pure level direction (avg yield change)
    k=0.50 → pure long yield change (most direct for TLT)
    k=1.00 → heavy spread weight

    Raw signals are averaged across spreads, then rolling z-scored.
    When z >= ts → LONG (100% TLT), held until z <= tl → SHORT (100% SHV).
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time
from data_loader import load_all

# ─── Configuration ──────────────────────────────────────────────────────────
SAVE_DIR = os.path.join(os.path.dirname(__file__), "output", "yc_continuous")
os.makedirs(SAVE_DIR, exist_ok=True)

BACKTEST_START = "2009-01-01"
RISK_FREE = 0.02
ANN = 252

SPREADS = {
    "2s30s":  ("US_2Y",  "US_30Y"),
    "2s10s":  ("US_2Y",  "US_10Y"),
    "3m2y":   ("US_3M",  "US_2Y"),
    "2s5s":   ("US_2Y",  "US_5Y"),
    "5s10s":  ("US_5Y",  "US_10Y"),
    "10s30s": ("US_10Y", "US_30Y"),
}
SP_NAMES = list(SPREADS.keys())

# Grid parameters
CHANGE_WINS = [10, 15, 21, 42, 63]
Z_WINS      = [63, 126, 252]
K_VALUES    = [0.0, 0.25, 0.50, 0.75, 1.00]
TL_GRID     = np.arange(-3.0, -0.4, 0.5)   # SHORT thresholds
TS_GRID     = np.arange(0.5, 3.1, 0.5)      # LONG thresholds

COOLDOWN  = 5
DELAY     = 1
TCOST_BPS = 5
W_INIT    = 0.5

# Regime scoring for the "regime_z" signal type
SCORE_MAP = {"BULL_FLAT": +2, "BULL_STEEP": +1, "UNDEFINED": 0,
             "BEAR_FLAT": -1, "BEAR_STEEP": -2}


# ─── Core functions ─────────────────────────────────────────────────────────

def rolling_z(arr, window, min_p=20):
    """Rolling z-score via pandas."""
    s = pd.Series(arr)
    mu = s.rolling(window, min_periods=min_p).mean()
    sd = s.rolling(window, min_periods=min_p).std()
    return ((s - mu) / sd).values


def compute_regime(yld, sc, lc, window):
    level  = (yld[sc] + yld[lc]) / 2
    spread = yld[lc] - yld[sc]
    dl, ds = level.diff(window), spread.diff(window)
    r = pd.Series("UNDEFINED", index=yld.index)
    r[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    r[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    r[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    r[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    return r


def sticky_bt(z_arr, tl, ts, ret_tlt, ret_shv):
    """Sticky threshold backtest: hold until opposite signal."""
    n = len(z_arr)
    aw = np.full(n, W_INIT)
    cur = W_INIT
    last_sig = -COOLDOWN - 1
    sigs = []

    for i in range(n):
        zi = z_arr[i]
        if np.isnan(zi):
            continue
        nw = None
        if zi >= ts and cur != 1.0:
            nw = 1.0
        elif zi <= tl and cur != 0.0:
            nw = 0.0
        if nw is not None and (i - last_sig) > COOLDOWN:
            ed = i + DELAY
            if ed < n:
                sigs.append((i, ed, nw))
                cur = nw
                last_sig = i
                aw[ed:] = nw

    # Returns with transaction costs
    tc = TCOST_BPS / 10000.0
    dr = aw * ret_tlt + (1 - aw) * ret_shv
    wd = np.abs(np.diff(aw, prepend=aw[0]))
    dr -= wd * tc
    eq = np.cumprod(1 + dr)

    # Metrics
    ann_ret = eq[-1] ** (ANN / n) - 1
    vol = np.std(dr) * np.sqrt(ANN)
    rf_d = RISK_FREE / ANN
    exc = dr - rf_d
    sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)
    neg = exc[exc < 0]
    ds_vol = np.sqrt(np.mean(neg**2)) * np.sqrt(ANN) if len(neg) > 0 else 1e-10
    sortino = np.mean(exc) * ANN / (ds_vol + 1e-10)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0.0

    return dict(sharpe=sharpe, sortino=sortino, ann_ret=ann_ret, vol=vol,
                mdd=mdd, calmar=calmar, total_ret=eq[-1]-1,
                n_trades=len(sigs), equity=eq, daily_ret=dr,
                actual_w=aw, signals=sigs)


# ─── Charts ─────────────────────────────────────────────────────────────────

def plot_equity(dates, eq_s, eq_bh, sigs, aw, z_arr, path, title=""):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10),
        height_ratios=[3, 1.2, 1], sharex=True,
        gridspec_kw={"hspace": 0.08})

    ax1, ax2, ax3 = axes

    # Equity
    ax1.plot(dates, eq_s, "b-", lw=1.5, label="YC Continuous Strategy")
    ax1.plot(dates, eq_bh, color="gray", lw=1, alpha=0.6, label="B&H TLT")
    for sd, ed, w in sigs:
        if ed < len(dates):
            c = "green" if w > 0.7 else "red"
            mk = "^" if w > 0.7 else "v"
            ax1.plot(dates[ed], eq_s[ed], mk, color=c, ms=8, zorder=5)
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Z-score
    valid = ~np.isnan(z_arr)
    ax2.plot(dates[valid], z_arr[valid], "k-", lw=0.7, alpha=0.7)
    ax2.axhline(0, color="gray", ls="-", lw=0.5)
    ax2.set_ylabel("Z-score")
    ax2.grid(True, alpha=0.3)

    # Position
    ax3.fill_between(dates, aw, 0.5, where=aw > 0.5,
                     color="green", alpha=0.3, label="LONG TLT")
    ax3.fill_between(dates, aw, 0.5, where=aw < 0.5,
                     color="red", alpha=0.3, label="SHORT (SHV)")
    ax3.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax3.set_ylabel("TLT Weight")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_yearly(yearly, path):
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(yearly))
    w = 0.35
    ax.bar(x - w/2, yearly["strategy"] * 100, w,
           label="Strategy", color="steelblue")
    ax.bar(x + w/2, yearly["benchmark"] * 100, w,
           label="B&H TLT", color="gray", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(yearly["year"].astype(int), rotation=45)
    ax.set_ylabel("Return (%)")
    ax.set_title("Yearly: YC Continuous Strategy vs B&H TLT")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_heatmap_signal_vs_cw(df, path):
    """Heatmap: signal type vs change window (best Sharpe per cell)."""
    sig_types = sorted(df["signal"].unique())
    cw_vals = sorted(df["change_win"].unique())

    best = df.loc[df.groupby(["signal", "change_win"])["sharpe"].idxmax()]
    pivot = best.pivot(index="signal", columns="change_win", values="sharpe")
    pivot = pivot.reindex(index=sig_types, columns=cw_vals)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.1, vmax=0.4)
    ax.set_xticks(range(len(cw_vals)))
    ax.set_xticklabels(cw_vals)
    ax.set_yticks(range(len(sig_types)))
    ax.set_yticklabels(sig_types)
    for i in range(len(sig_types)):
        for j in range(len(cw_vals)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold" if v > 0.15 else "normal")
    plt.colorbar(im, ax=ax, label="Sharpe")
    ax.set_xlabel("Change Window (days)")
    ax.set_ylabel("Signal Type")
    ax.set_title("Best Sharpe: Signal Type × Change Window")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─── Yearly helper ──────────────────────────────────────────────────────────

def yearly_table(equity, daily_ret, actual_w, dates, ret_tlt):
    years = sorted(set(d.year for d in dates))
    rows = []
    for y in years:
        mask = np.array([d.year == y for d in dates])
        if mask.sum() == 0:
            continue
        eq_y = np.cumprod(1 + daily_ret[mask])
        bh_y = np.cumprod(1 + ret_tlt[mask])
        s_r = eq_y[-1] - 1
        b_r = bh_y[-1] - 1
        wd = np.abs(np.diff(actual_w[mask], prepend=actual_w[mask][0]))
        rows.append(dict(year=y, strategy=s_r, benchmark=b_r,
                         excess=s_r - b_r,
                         n_trades=int(np.sum(wd > 0.01))))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  YC CONTINUOUS SIGNAL STRATEGY")
    print("  Z-scored yield curve changes -> sticky threshold trading")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────
    D = load_all()
    yld = D["yields"]
    etf_ret = D["etf_ret"]

    bt_start = pd.Timestamp(BACKTEST_START)
    common = yld.index.intersection(etf_ret.index)
    common = common[common >= bt_start]

    yld_bt  = yld.loc[common]
    ret_tlt = etf_ret.loc[common, "TLT"].values
    ret_shv = etf_ret.loc[common, "SHV"].values
    dates   = common
    n_days  = len(common)

    eq_bh = np.cumprod(1 + ret_tlt)
    rf_d = RISK_FREE / ANN
    bh_exc = ret_tlt - rf_d
    bh_sharpe = np.mean(bh_exc) / (np.std(bh_exc) + 1e-10) * np.sqrt(ANN)
    bh_mdd = ((eq_bh - np.maximum.accumulate(eq_bh)) /
              np.maximum.accumulate(eq_bh)).min()

    print(f"\n  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d}  "
          f"({n_days} days)")
    print(f"  B&H TLT: Sharpe={bh_sharpe:.3f}  "
          f"Return={eq_bh[-1]-1:+.1%}  MDD={bh_mdd:.1%}")

    # ── Precompute dl, ds caches ──────────────────────────────────────
    print("\n  Precomputing level/spread changes...")
    dl_c, ds_c = {}, {}
    for sp in SP_NAMES:
        sc, lc = SPREADS[sp]
        lev = (yld_bt[sc] + yld_bt[lc]) / 2
        spr = yld_bt[lc] - yld_bt[sc]
        for cw in CHANGE_WINS:
            dl_c[(sp, cw)] = lev.diff(cw).values
            ds_c[(sp, cw)] = spr.diff(cw).values

    # ── Spread groups ─────────────────────────────────────────────────
    groups = {
        "all_6":   SP_NAMES,
        "no_3m2y": [s for s in SP_NAMES if s != "3m2y"],
        "broad":   ["2s10s", "2s30s"],
        "10s30s":  ["10s30s"],
        "2s30s":   ["2s30s"],
    }

    # ── Precompute z-score series ─────────────────────────────────────
    print("  Building z-score series...")
    zseries = {}

    for cw in CHANGE_WINS:
        for gn, gsp in groups.items():
            # --- K-based continuous signals ---
            for k in K_VALUES:
                raws = [-(dl_c[(sp, cw)] + k * ds_c[(sp, cw)]) for sp in gsp]
                raw_avg = np.nanmean(raws, axis=0)
                for zw in Z_WINS:
                    zseries[(f"k{k:.2f}", cw, zw, gn)] = rolling_z(raw_avg, zw)

            # --- Pure spread signal ---
            raws = [-ds_c[(sp, cw)] for sp in gsp]
            raw_avg = np.nanmean(raws, axis=0)
            for zw in Z_WINS:
                zseries[("spread", cw, zw, gn)] = rolling_z(raw_avg, zw)

            # --- Regime score z-scored ---
            score_sum = np.zeros(n_days)
            for sp in gsp:
                sc, lc = SPREADS[sp]
                reg = compute_regime(yld_bt, sc, lc, cw)
                score_sum += reg.map(SCORE_MAP).values.astype(float)
            for zw in Z_WINS:
                zseries[("regime_z", cw, zw, gn)] = rolling_z(score_sum, zw)

    n_z = len(zseries)
    n_th = len(TL_GRID) * len(TS_GRID)
    total = n_z * n_th
    print(f"  {n_z} z-series × {n_th} threshold pairs = {total} backtests")

    # ── Grid search ───────────────────────────────────────────────────
    print("  Running grid search...")
    t1 = time.time()
    rows = []
    for key, z_arr in zseries.items():
        sig, cw, zw, gn = key
        for tl in TL_GRID:
            for ts in TS_GRID:
                m = sticky_bt(z_arr, tl, ts, ret_tlt, ret_shv)
                rows.append(dict(
                    signal=sig, change_win=cw, z_win=zw, group=gn,
                    tl=tl, ts=ts, sharpe=m["sharpe"], sortino=m["sortino"],
                    ann_ret=m["ann_ret"], vol=m["vol"], mdd=m["mdd"],
                    calmar=m["calmar"], total_ret=m["total_ret"],
                    n_trades=m["n_trades"]))

    df = pd.DataFrame(rows)
    print(f"  Done in {time.time()-t1:.1f}s  ({len(df)} configs)")

    # ── Filter: require at least 10 trades ────────────────────────────
    df_f = df[df["n_trades"] >= 10].copy()
    print(f"  After filter (>=10 trades): {len(df_f)} configs")

    # ══════════════════════════════════════════════════════════════════
    # RESULTS ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TOP 30 CONFIGURATIONS")
    print("=" * 70)

    top30 = df_f.nlargest(30, "sharpe")
    print(f"\n  {'#':>3} {'Signal':>8} {'CW':>3} {'ZW':>4} {'Group':>8} "
          f"{'TL':>5} {'TS':>5} {'Sharpe':>7} {'Return':>7} "
          f"{'MDD':>8} {'Trades':>6}")
    print("  " + "-" * 78)
    for i, (_, r) in enumerate(top30.iterrows(), 1):
        print(f"  {i:>3} {r['signal']:>8} {r['change_win']:>3.0f} "
              f"{r['z_win']:>4.0f} {r['group']:>8} "
              f"{r['tl']:>5.1f} {r['ts']:>5.1f} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>6.0f}")

    # ── Best per signal type ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BEST PER SIGNAL TYPE")
    print("=" * 70)
    for sig in sorted(df_f["signal"].unique()):
        sub = df_f[df_f["signal"] == sig]
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"  {sig:>8}:  Sharpe={b['sharpe']:.3f}  "
              f"cw={b['change_win']:.0f} zw={b['z_win']:.0f} "
              f"grp={b['group']}  tl={b['tl']:.1f} ts={b['ts']:.1f}  "
              f"ret={b['ann_ret']:+.1%}  trades={b['n_trades']:.0f}")

    # ── Best per group ────────────────────────────────────────────────
    print(f"\n  BEST PER SPREAD GROUP:")
    for gn in groups:
        sub = df_f[df_f["group"] == gn]
        if len(sub) == 0:
            continue
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"  {gn:>8}:  Sharpe={b['sharpe']:.3f}  "
              f"sig={b['signal']}  cw={b['change_win']:.0f} "
              f"zw={b['z_win']:.0f}")

    # ── Best per change window ────────────────────────────────────────
    print(f"\n  BEST PER CHANGE WINDOW:")
    for cw in CHANGE_WINS:
        sub = df_f[df_f["change_win"] == cw]
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"  cw={cw:>3}:  Sharpe={b['sharpe']:.3f}  "
              f"sig={b['signal']}  grp={b['group']}  "
              f"zw={b['z_win']:.0f}")

    # ── Best per z-score window ───────────────────────────────────────
    print(f"\n  BEST PER Z-SCORE WINDOW:")
    for zw in Z_WINS:
        sub = df_f[df_f["z_win"] == zw]
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"  zw={zw:>3}:  Sharpe={b['sharpe']:.3f}  "
              f"sig={b['signal']}  cw={b['change_win']:.0f} "
              f"grp={b['group']}")

    # ── Heatmap ───────────────────────────────────────────────────────
    plot_heatmap_signal_vs_cw(df_f,
        os.path.join(SAVE_DIR, "01_heatmap_sig_cw.png"))
    print(f"\n  -> 01_heatmap_sig_cw.png")

    # ══════════════════════════════════════════════════════════════════
    # BEST CONFIG DEEP DIVE
    # ══════════════════════════════════════════════════════════════════
    best_row = df_f.loc[df_f["sharpe"].idxmax()]
    sig_best = best_row["signal"]
    cw_best  = int(best_row["change_win"])
    zw_best  = int(best_row["z_win"])
    gn_best  = best_row["group"]
    tl_best  = best_row["tl"]
    ts_best  = best_row["ts"]

    print("\n" + "=" * 70)
    print("  BEST CONFIG — DEEP DIVE")
    print("=" * 70)
    label = (f"signal={sig_best}  cw={cw_best}  zw={zw_best}  "
             f"grp={gn_best}  tl={tl_best:.1f}  ts={ts_best:.1f}")
    print(f"  {label}")

    # Re-run with full output
    z_best = zseries[(sig_best, cw_best, zw_best, gn_best)]
    res = sticky_bt(z_best, tl_best, ts_best, ret_tlt, ret_shv)

    print(f"\n    Sharpe  = {res['sharpe']:.4f}")
    print(f"    Sortino = {res['sortino']:.4f}")
    print(f"    AnnRet  = {res['ann_ret']:+.2%}")
    print(f"    Vol     = {res['vol']:.2%}")
    print(f"    MDD     = {res['mdd']:.2%}")
    print(f"    Calmar  = {res['calmar']:.4f}")
    print(f"    TotRet  = {res['total_ret']:+.2%}")
    print(f"    Trades  = {res['n_trades']}")

    # Time in each position
    aw = res["actual_w"]
    pct_long = np.mean(aw > 0.7) * 100
    pct_short = np.mean(aw < 0.3) * 100
    pct_neutral = np.mean((aw >= 0.3) & (aw <= 0.7)) * 100
    print(f"\n    Time LONG:    {pct_long:.1f}%")
    print(f"    Time SHORT:   {pct_short:.1f}%")
    print(f"    Time NEUTRAL: {pct_neutral:.1f}%")

    # Yearly
    yearly = yearly_table(res["equity"], res["daily_ret"], aw, dates, ret_tlt)
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
    plot_equity(dates, res["equity"], eq_bh, res["signals"],
                aw, z_best,
                os.path.join(SAVE_DIR, "02_best_equity.png"),
                f"YC Continuous — {label}\n"
                f"Sharpe={res['sharpe']:.3f}  "
                f"Return={res['total_ret']:+.1%}  MDD={res['mdd']:.1%}")
    print(f"\n  -> 02_best_equity.png")

    plot_yearly(yearly, os.path.join(SAVE_DIR, "03_yearly.png"))
    print(f"  -> 03_yearly.png")

    # ══════════════════════════════════════════════════════════════════
    # ROBUSTNESS: NEARBY CONFIGS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ROBUSTNESS: TOP 5 CONFIGS WITH SAME SIGNAL TYPE")
    print("=" * 70)
    same_sig = df_f[df_f["signal"] == sig_best].nlargest(10, "sharpe")
    print(f"\n  {'#':>3} {'CW':>3} {'ZW':>4} {'Group':>8} "
          f"{'TL':>5} {'TS':>5} {'Sharpe':>7} {'Return':>7} "
          f"{'MDD':>8} {'Trades':>6}")
    print("  " + "-" * 62)
    for i, (_, r) in enumerate(same_sig.iterrows(), 1):
        print(f"  {i:>3} {r['change_win']:>3.0f} {r['z_win']:>4.0f} "
              f"{r['group']:>8} {r['tl']:>5.1f} {r['ts']:>5.1f} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>6.0f}")

    # ══════════════════════════════════════════════════════════════════
    # SENSITIVITY: THRESHOLD HEATMAP FOR BEST SIGNAL
    # ══════════════════════════════════════════════════════════════════
    print("\n  Generating threshold sensitivity heatmap...")
    sub_th = df_f[(df_f["signal"] == sig_best) &
                  (df_f["change_win"] == cw_best) &
                  (df_f["z_win"] == zw_best) &
                  (df_f["group"] == gn_best)]

    if len(sub_th) > 0:
        pivot_th = sub_th.pivot_table(index="tl", columns="ts",
                                       values="sharpe", aggfunc="first")
        fig, ax = plt.subplots(figsize=(8, 6))
        tl_vals = sorted(pivot_th.index)
        ts_vals = sorted(pivot_th.columns)
        data = pivot_th.reindex(index=tl_vals, columns=ts_vals).values
        im = ax.imshow(data, cmap="RdYlGn", aspect="auto",
                       vmin=-0.1, vmax=max(0.3, np.nanmax(data)))
        ax.set_xticks(range(len(ts_vals)))
        ax.set_xticklabels([f"{v:.1f}" for v in ts_vals])
        ax.set_yticks(range(len(tl_vals)))
        ax.set_yticklabels([f"{v:.1f}" for v in tl_vals])
        for i in range(len(tl_vals)):
            for j in range(len(ts_vals)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8)
        plt.colorbar(im, ax=ax, label="Sharpe")
        ax.set_xlabel("LONG threshold (ts)")
        ax.set_ylabel("SHORT threshold (tl)")
        ax.set_title(f"Threshold Sensitivity: {sig_best} cw{cw_best} "
                     f"zw{zw_best} {gn_best}")
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "04_threshold_heatmap.png"),
                    dpi=150)
        plt.close()
        print(f"  -> 04_threshold_heatmap.png")

    # ── Save CSVs ─────────────────────────────────────────────────────
    df.to_csv(os.path.join(SAVE_DIR, "grid_results.csv"), index=False)
    top30.to_csv(os.path.join(SAVE_DIR, "top30.csv"), index=False)
    yearly.to_csv(os.path.join(SAVE_DIR, "yearly_best.csv"), index=False)
    print(f"\n  CSVs saved to {SAVE_DIR}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
