"""
===============================================================================
  YC CONTINUOUS V2 — Extended Optimization
  Three improvements over V1:
    1. Wider change/z-score windows (cw up to 126, zw up to 378)
    2. Full asymmetric entry threshold grid
    3. Exit-to-neutral mechanism (take profits when z returns to middle zone)
===============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, time
from data_loader import load_all

SAVE_DIR = os.path.join(os.path.dirname(__file__), "output", "yc_cont_v2")
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

# V2 grid — extended
CW_GRID  = [21, 42, 63, 80, 126]
ZW_GRID  = [126, 252, 378]
K_VALS   = [0.50, 0.75, 1.00]       # Best k-values from V1
TL_GRID  = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
TS_GRID  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
EXIT_GRID = [None, -1.0, -0.5, 0.0, 0.5]   # None = no exit (sticky)

COOLDOWN  = 5
DELAY     = 1
TCOST_BPS = 5
W_INIT    = 0.5

SCORE_MAP = {"BULL_FLAT": +2, "BULL_STEEP": +1, "UNDEFINED": 0,
             "BEAR_FLAT": -1, "BEAR_STEEP": -2}


# ─── Core ───────────────────────────────────────────────────────────────────

def rolling_z(arr, window, min_p=20):
    s = pd.Series(arr)
    mu = s.rolling(window, min_periods=min_p).mean()
    sd = s.rolling(window, min_periods=min_p).std()
    return ((s - mu) / sd).values


def compute_regime(yld, sc, lc, window):
    lev = (yld[sc] + yld[lc]) / 2
    spr = yld[lc] - yld[sc]
    dl, ds = lev.diff(window), spr.diff(window)
    r = pd.Series("UNDEFINED", index=yld.index)
    r[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    r[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    r[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    r[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    return r


def generate_weights(z_arr, tl, ts, exit_z=None):
    """Sticky threshold trading with optional exit-to-neutral.

    Entry:  z >= ts -> LONG (1.0),  z <= tl -> SHORT (0.0)
    Exit:   after LONG,  z <= exit_z   -> NEUTRAL (0.5)
            after SHORT, z >= -exit_z  -> NEUTRAL (0.5)
    Entries take priority over exits (direct reversal allowed).
    """
    n = len(z_arr)
    aw = np.full(n, W_INIT)
    cur = W_INIT
    last_sig = -COOLDOWN - 1
    n_trades = 0
    has_exit = exit_z is not None

    for i in range(n):
        zi = z_arr[i]
        if zi != zi:          # fast NaN check
            continue

        nw = -1.0             # sentinel = no change
        # Entry (priority)
        if zi >= ts and cur != 1.0:
            nw = 1.0
        elif zi <= tl and cur != 0.0:
            nw = 0.0
        # Exit (only if no entry)
        if nw < 0.0 and has_exit:
            if cur == 1.0 and zi <= exit_z:
                nw = 0.5
            elif cur == 0.0 and zi >= -exit_z:
                nw = 0.5

        if nw >= 0.0 and abs(nw - cur) > 0.01 and (i - last_sig) > COOLDOWN:
            ed = i + DELAY
            if ed < n:
                cur = nw
                last_sig = i
                n_trades += 1
                aw[ed:] = nw

    return aw, n_trades


def fast_metrics(aw, ret_tlt, ret_shv):
    """Fast backtest returning (sharpe, sortino, ann_ret, vol, mdd, total_ret)."""
    dr = aw * ret_tlt + (1 - aw) * ret_shv
    wd = np.abs(np.diff(aw, prepend=aw[0]))
    dr -= wd * (TCOST_BPS / 10000.0)
    eq = np.cumprod(1 + dr)
    n = len(dr)
    ann_ret = eq[-1] ** (ANN / n) - 1
    vol = np.std(dr) * np.sqrt(ANN)
    exc = dr - RISK_FREE / ANN
    sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)
    neg = exc[exc < 0]
    ds = np.sqrt(np.mean(neg**2)) * np.sqrt(ANN) if len(neg) > 0 else 1e-10
    sortino = np.mean(exc) * ANN / (ds + 1e-10)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    return sharpe, sortino, ann_ret, vol, mdd, eq[-1] - 1, eq, dr


# ─── Charts ─────────────────────────────────────────────────────────────────

def plot_equity(dates, eq_s, eq_bh, aw, z_arr, tl, ts, exit_z, path, title):
    fig, axes = plt.subplots(3, 1, figsize=(15, 10),
        height_ratios=[3, 1.2, 1], sharex=True,
        gridspec_kw={"hspace": 0.08})
    ax1, ax2, ax3 = axes

    ax1.plot(dates, eq_s, "b-", lw=1.5, label="YC Continuous V2")
    ax1.plot(dates, eq_bh, color="gray", lw=1, alpha=0.6, label="B&H TLT")
    # Mark trades
    wd = np.abs(np.diff(aw, prepend=aw[0]))
    trade_days = np.where(wd > 0.01)[0]
    for d in trade_days:
        c = "green" if aw[d] > 0.7 else ("red" if aw[d] < 0.3 else "orange")
        mk = "^" if aw[d] > 0.7 else ("v" if aw[d] < 0.3 else "s")
        ax1.plot(dates[d], eq_s[d], mk, color=c, ms=7, zorder=5)
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # Z-score with thresholds
    valid = ~np.isnan(z_arr)
    ax2.plot(dates[valid], z_arr[valid], "k-", lw=0.6, alpha=0.7)
    ax2.axhline(ts, color="green", ls="--", lw=1, alpha=0.7, label=f"ts={ts:.1f}")
    ax2.axhline(tl, color="red", ls="--", lw=1, alpha=0.7, label=f"tl={tl:.1f}")
    if exit_z is not None:
        ax2.axhline(exit_z, color="orange", ls=":", lw=1, alpha=0.7,
                    label=f"exit_long={exit_z:.1f}")
        ax2.axhline(-exit_z, color="orange", ls=":", lw=1, alpha=0.7,
                    label=f"exit_short={-exit_z:.1f}")
    ax2.axhline(0, color="gray", ls="-", lw=0.5)
    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Position
    ax3.fill_between(dates, aw, 0.5, where=aw > 0.7,
                     color="green", alpha=0.3, label="LONG")
    ax3.fill_between(dates, aw, 0.5, where=aw < 0.3,
                     color="red", alpha=0.3, label="SHORT")
    ax3.fill_between(dates, aw, 0.5, where=(aw >= 0.3) & (aw <= 0.7) & (aw != 0.5),
                     color="orange", alpha=0.3, label="NEUTRAL")
    ax3.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax3.set_ylabel("TLT Weight")
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="lower right", fontsize=8)
    ax3.grid(True, alpha=0.3)

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
    ax.set_title(title or "Yearly: Strategy vs B&H TLT")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_exit_comparison(dates, eq_no, eq_ex, eq_bh, path, label_no, label_ex):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, eq_no, "b-", lw=1.5, label=label_no)
    ax.plot(dates, eq_ex, "r-", lw=1.5, label=label_ex)
    ax.plot(dates, eq_bh, color="gray", lw=1, alpha=0.5, label="B&H TLT")
    ax.set_ylabel("Equity ($)")
    ax.set_title("Impact of Exit-to-Neutral")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_threshold_heatmap(sub, tl_vals, ts_vals, path, title=""):
    pivot = sub.pivot_table(index="tl", columns="ts", values="sharpe", aggfunc="first")
    pivot = pivot.reindex(index=sorted(tl_vals), columns=sorted(ts_vals))
    fig, ax = plt.subplots(figsize=(9, 7))
    vmax = max(0.25, np.nanmax(pivot.values))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.1, vmax=vmax)
    ax.set_xticks(range(len(ts_vals)))
    ax.set_xticklabels([f"{v:.1f}" for v in sorted(ts_vals)])
    ax.set_yticks(range(len(tl_vals)))
    ax.set_yticklabels([f"{v:.1f}" for v in sorted(tl_vals)])
    for i in range(len(tl_vals)):
        for j in range(len(ts_vals)):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold" if v > 0.15 else "normal")
    plt.colorbar(im, ax=ax, label="Sharpe")
    ax.set_xlabel("LONG threshold (ts)")
    ax.set_ylabel("SHORT threshold (tl)")
    ax.set_title(title or "Threshold Sensitivity")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ─── Yearly helper ──────────────────────────────────────────────────────────

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
    print("  YC CONTINUOUS V2 — Extended Optimization")
    print("  Wider windows + Asymmetric thresholds + Exit-to-neutral")
    print("=" * 70)

    # ── Load ──────────────────────────────────────────────────────────
    D = load_all()
    yld     = D["yields"]
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
    exc_bh = ret_tlt - RISK_FREE / ANN
    bh_sharpe = np.mean(exc_bh) / (np.std(exc_bh)+1e-10) * np.sqrt(ANN)
    bh_mdd = ((eq_bh - np.maximum.accumulate(eq_bh)) /
              np.maximum.accumulate(eq_bh)).min()

    print(f"\n  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d}  "
          f"({n_days} days)")
    print(f"  B&H TLT: Sharpe={bh_sharpe:.3f}  "
          f"Return={eq_bh[-1]-1:+.1%}  MDD={bh_mdd:.1%}")

    # ── Precompute caches ─────────────────────────────────────────────
    print("\n  Precomputing caches...")
    dl_c, ds_c = {}, {}
    for sp in SP_NAMES:
        sc, lc = SPREADS[sp]
        lev = (yld_bt[sc] + yld_bt[lc]) / 2
        spr = yld_bt[lc] - yld_bt[sc]
        for cw in CW_GRID:
            dl_c[(sp, cw)] = lev.diff(cw).values
            ds_c[(sp, cw)] = spr.diff(cw).values

    groups = {
        "10s30s":  ["10s30s"],
        "no_3m2y": [s for s in SP_NAMES if s != "3m2y"],
        "broad":   ["2s10s", "2s30s"],
        "all_6":   SP_NAMES,
    }

    # ── Build z-score series ──────────────────────────────────────────
    print("  Building z-score series...")
    zseries = {}

    for cw in CW_GRID:
        for gn, gsp in groups.items():
            # K-based signals
            for k in K_VALS:
                raws = [-(dl_c[(sp, cw)] + k * ds_c[(sp, cw)]) for sp in gsp]
                raw_avg = np.nanmean(raws, axis=0)
                for zw in ZW_GRID:
                    zseries[(f"k{k:.2f}", cw, zw, gn)] = rolling_z(raw_avg, zw)

            # Pure spread
            raws = [-ds_c[(sp, cw)] for sp in gsp]
            raw_avg = np.nanmean(raws, axis=0)
            for zw in ZW_GRID:
                zseries[("spread", cw, zw, gn)] = rolling_z(raw_avg, zw)

            # Regime z-score
            score_sum = np.zeros(n_days)
            for sp in gsp:
                sc, lc = SPREADS[sp]
                reg = compute_regime(yld_bt, sc, lc, cw)
                score_sum += reg.map(SCORE_MAP).values.astype(float)
            for zw in ZW_GRID:
                zseries[("regime_z", cw, zw, gn)] = rolling_z(score_sum, zw)

    n_z = len(zseries)
    n_entry = len(TL_GRID) * len(TS_GRID)
    n_exit = len(EXIT_GRID)
    total = n_z * n_entry * n_exit
    print(f"  {n_z} z-series x {n_entry} entry pairs x {n_exit} exit opts "
          f"= {total:,} configs")

    # ── GRID SEARCH ───────────────────────────────────────────────────
    print("  Running grid search...")
    t1 = time.time()
    rows = []
    count = 0
    report_every = total // 10

    for key, z_arr in zseries.items():
        sig, cw, zw, gn = key
        for tl in TL_GRID:
            for ts in TS_GRID:
                for ex in EXIT_GRID:
                    aw, nt = generate_weights(z_arr, tl, ts, ex)
                    sh, so, ar, vo, md, tr, _, _ = fast_metrics(
                        aw, ret_tlt, ret_shv)
                    rows.append((sig, cw, zw, gn, tl, ts,
                                 ex if ex is not None else 99.0,
                                 sh, so, ar, vo, md, tr, nt))
                    count += 1
                    if count % report_every == 0:
                        pct = count / total * 100
                        print(f"    {pct:.0f}% ({count:,}/{total:,})  "
                              f"elapsed {time.time()-t1:.0f}s")

    cols = ["signal", "change_win", "z_win", "group", "tl", "ts", "exit_z",
            "sharpe", "sortino", "ann_ret", "vol", "mdd", "total_ret",
            "n_trades"]
    df = pd.DataFrame(rows, columns=cols)
    # Mark exit=None as 99.0 for filtering convenience
    df["has_exit"] = df["exit_z"] < 90

    elapsed_grid = time.time() - t1
    print(f"  Done: {len(df):,} configs in {elapsed_grid:.1f}s")

    # ── Filter ────────────────────────────────────────────────────────
    df_f = df[df["n_trades"] >= 8].copy()
    print(f"  After filter (>=8 trades): {len(df_f):,} configs")

    # ══════════════════════════════════════════════════════════════════
    # TOP 30 OVERALL
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TOP 30 CONFIGURATIONS")
    print("=" * 70)

    top30 = df_f.nlargest(30, "sharpe")
    print(f"\n  {'#':>3} {'Sig':>8} {'CW':>3} {'ZW':>4} {'Grp':>8} "
          f"{'TL':>5} {'TS':>5} {'Exit':>5} "
          f"{'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Tr':>4}")
    print("  " + "-" * 82)
    for i, (_, r) in enumerate(top30.iterrows(), 1):
        ex_str = f"{r['exit_z']:.1f}" if r['has_exit'] else " None"
        print(f"  {i:>3} {r['signal']:>8} {r['change_win']:>3.0f} "
              f"{r['z_win']:>4.0f} {r['group']:>8} "
              f"{r['tl']:>5.1f} {r['ts']:>5.1f} {ex_str:>5} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>4.0f}")

    # ══════════════════════════════════════════════════════════════════
    # EXIT vs NO-EXIT COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  EXIT-TO-NEUTRAL IMPACT ANALYSIS")
    print("=" * 70)

    df_no = df_f[~df_f["has_exit"]]
    df_ex = df_f[df_f["has_exit"]]

    best_no = df_no.loc[df_no["sharpe"].idxmax()] if len(df_no) > 0 else None
    best_ex = df_ex.loc[df_ex["sharpe"].idxmax()] if len(df_ex) > 0 else None

    if best_no is not None:
        print(f"\n  Best WITHOUT exit:")
        print(f"    sig={best_no['signal']}  cw={best_no['change_win']:.0f}  "
              f"zw={best_no['z_win']:.0f}  grp={best_no['group']}")
        print(f"    tl={best_no['tl']:.1f}  ts={best_no['ts']:.1f}")
        print(f"    Sharpe={best_no['sharpe']:.4f}  Ret={best_no['ann_ret']:+.2%}  "
              f"MDD={best_no['mdd']:.2%}  Trades={best_no['n_trades']:.0f}")

    if best_ex is not None:
        print(f"\n  Best WITH exit:")
        print(f"    sig={best_ex['signal']}  cw={best_ex['change_win']:.0f}  "
              f"zw={best_ex['z_win']:.0f}  grp={best_ex['group']}")
        print(f"    tl={best_ex['tl']:.1f}  ts={best_ex['ts']:.1f}  "
              f"exit_z={best_ex['exit_z']:.1f}")
        print(f"    Sharpe={best_ex['sharpe']:.4f}  Ret={best_ex['ann_ret']:+.2%}  "
              f"MDD={best_ex['mdd']:.2%}  Trades={best_ex['n_trades']:.0f}")

    # Average improvement from exit across same configs
    print("\n  Avg Sharpe by exit_z value:")
    for ex_val in EXIT_GRID:
        if ex_val is None:
            sub = df_f[~df_f["has_exit"]]
            lbl = "  None"
        else:
            sub = df_f[df_f["exit_z"] == ex_val]
            lbl = f"{ex_val:>5.1f}"
        if len(sub) > 0:
            print(f"    exit={lbl}: avg_sharpe={sub['sharpe'].mean():.4f}  "
                  f"median={sub['sharpe'].median():.4f}  "
                  f"best={sub['sharpe'].max():.4f}  "
                  f"avg_trades={sub['n_trades'].mean():.0f}")

    # ══════════════════════════════════════════════════════════════════
    # BEST PER DIMENSION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  BEST PER DIMENSION")
    print("=" * 70)

    print("\n  By signal type:")
    for sig in sorted(df_f["signal"].unique()):
        b = df_f[df_f["signal"] == sig]
        bx = b.loc[b["sharpe"].idxmax()]
        ex_s = f"{bx['exit_z']:.1f}" if bx['has_exit'] else "None"
        print(f"    {sig:>8}: Sharpe={bx['sharpe']:.3f}  "
              f"cw={bx['change_win']:.0f} zw={bx['z_win']:.0f} "
              f"grp={bx['group']}  [{bx['tl']:.1f},{bx['ts']:.1f}] "
              f"exit={ex_s}  tr={bx['n_trades']:.0f}")

    print("\n  By change window:")
    for cw in CW_GRID:
        b = df_f[df_f["change_win"] == cw]
        bx = b.loc[b["sharpe"].idxmax()]
        print(f"    cw={cw:>3}: Sharpe={bx['sharpe']:.3f}  sig={bx['signal']}")

    print("\n  By z-score window:")
    for zw in ZW_GRID:
        b = df_f[df_f["z_win"] == zw]
        bx = b.loc[b["sharpe"].idxmax()]
        print(f"    zw={zw:>3}: Sharpe={bx['sharpe']:.3f}  sig={bx['signal']}")

    print("\n  By spread group:")
    for gn in groups:
        b = df_f[df_f["group"] == gn]
        bx = b.loc[b["sharpe"].idxmax()]
        print(f"    {gn:>8}: Sharpe={bx['sharpe']:.3f}  sig={bx['signal']}")

    # ══════════════════════════════════════════════════════════════════
    # OVERALL BEST — DEEP DIVE
    # ══════════════════════════════════════════════════════════════════
    best = df_f.loc[df_f["sharpe"].idxmax()]
    sig_b = best["signal"]
    cw_b  = int(best["change_win"])
    zw_b  = int(best["z_win"])
    gn_b  = best["group"]
    tl_b  = best["tl"]
    ts_b  = best["ts"]
    ex_b  = best["exit_z"] if best["has_exit"] else None

    print("\n" + "=" * 70)
    print("  OVERALL BEST — DEEP DIVE")
    print("=" * 70)

    ex_str = f"{ex_b:.1f}" if ex_b is not None else "None"
    label = (f"{sig_b} cw={cw_b} zw={zw_b} {gn_b} "
             f"[{tl_b:.1f},{ts_b:.1f}] exit={ex_str}")
    print(f"  {label}")

    z_best = zseries[(sig_b, cw_b, zw_b, gn_b)]
    aw_best, nt_best = generate_weights(z_best, tl_b, ts_b, ex_b)
    sh, so, ar, vo, md, tr, eq_best, dr_best = fast_metrics(
        aw_best, ret_tlt, ret_shv)

    print(f"\n    Sharpe  = {sh:.4f}")
    print(f"    Sortino = {so:.4f}")
    print(f"    AnnRet  = {ar:+.2%}")
    print(f"    Vol     = {vo:.2%}")
    print(f"    MDD     = {md:.2%}")
    print(f"    TotRet  = {tr:+.2%}")
    print(f"    Trades  = {nt_best}")

    pct_l = np.mean(aw_best > 0.7) * 100
    pct_s = np.mean(aw_best < 0.3) * 100
    pct_n = np.mean((aw_best >= 0.3) & (aw_best <= 0.7)) * 100
    print(f"\n    Time LONG:    {pct_l:.1f}%")
    print(f"    Time SHORT:   {pct_s:.1f}%")
    print(f"    Time NEUTRAL: {pct_n:.1f}%")

    yearly = yearly_table(dr_best, aw_best, dates, ret_tlt)
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
    plot_equity(dates, eq_best, eq_bh, aw_best, z_best,
                tl_b, ts_b, ex_b,
                os.path.join(SAVE_DIR, "01_best_equity.png"),
                f"YC Continuous V2 — {label}\n"
                f"Sharpe={sh:.3f}  Return={tr:+.1%}  MDD={md:.1%}")
    print(f"\n  -> 01_best_equity.png")

    plot_yearly(yearly, os.path.join(SAVE_DIR, "02_yearly.png"),
                f"YC Continuous V2 — Yearly: {label}")
    print(f"  -> 02_yearly.png")

    # Threshold heatmap for best signal/cw/zw/group/exit
    sub_th = df_f[(df_f["signal"] == sig_b) &
                  (df_f["change_win"] == cw_b) &
                  (df_f["z_win"] == zw_b) &
                  (df_f["group"] == gn_b) &
                  ((df_f["exit_z"] == (ex_b if ex_b is not None else 99.0)) |
                   (~df_f["has_exit"] & (ex_b is None)))]
    plot_threshold_heatmap(sub_th, TL_GRID, TS_GRID,
        os.path.join(SAVE_DIR, "03_threshold_heatmap.png"),
        f"Threshold Sensitivity: {sig_b} cw{cw_b} zw{zw_b} {gn_b} exit={ex_str}")
    print(f"  -> 03_threshold_heatmap.png")

    # Exit comparison: best no-exit config run with and without exit
    if best_no is not None and best_ex is not None:
        # Same signal config, compare with/without exit
        z_cmp = zseries[(best_no["signal"], int(best_no["change_win"]),
                         int(best_no["z_win"]), best_no["group"])]
        aw_no, _ = generate_weights(z_cmp, best_no["tl"], best_no["ts"], None)
        _, _, _, _, _, _, eq_no, _ = fast_metrics(aw_no, ret_tlt, ret_shv)
        # Same config but with best exit
        best_exit_for_config = df_f[
            (df_f["signal"] == best_no["signal"]) &
            (df_f["change_win"] == best_no["change_win"]) &
            (df_f["z_win"] == best_no["z_win"]) &
            (df_f["group"] == best_no["group"]) &
            (df_f["tl"] == best_no["tl"]) &
            (df_f["ts"] == best_no["ts"]) &
            (df_f["has_exit"])
        ]
        if len(best_exit_for_config) > 0:
            bec = best_exit_for_config.loc[
                best_exit_for_config["sharpe"].idxmax()]
            aw_ex2, _ = generate_weights(z_cmp, best_no["tl"], best_no["ts"],
                                         bec["exit_z"])
            _, _, _, _, _, _, eq_ex2, _ = fast_metrics(aw_ex2, ret_tlt, ret_shv)
            plot_exit_comparison(dates, eq_no, eq_ex2, eq_bh,
                os.path.join(SAVE_DIR, "04_exit_comparison.png"),
                f"No exit (Sh={best_no['sharpe']:.3f})",
                f"Exit={bec['exit_z']:.1f} (Sh={bec['sharpe']:.3f})")
            print(f"  -> 04_exit_comparison.png")

    # ══════════════════════════════════════════════════════════════════
    # ROBUSTNESS: TOP 10 NEARBY
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  ROBUSTNESS: TOP 10 SAME SIGNAL TYPE")
    print("=" * 70)

    same = df_f[df_f["signal"] == sig_b].nlargest(10, "sharpe")
    print(f"\n  {'#':>3} {'CW':>3} {'ZW':>4} {'Grp':>8} "
          f"{'TL':>5} {'TS':>5} {'Exit':>5} "
          f"{'Sharpe':>7} {'Ret':>7} {'MDD':>8} {'Tr':>4}")
    print("  " + "-" * 70)
    for i, (_, r) in enumerate(same.iterrows(), 1):
        ex_s = f"{r['exit_z']:.1f}" if r['has_exit'] else " None"
        print(f"  {i:>3} {r['change_win']:>3.0f} {r['z_win']:>4.0f} "
              f"{r['group']:>8} {r['tl']:>5.1f} {r['ts']:>5.1f} "
              f"{ex_s:>5} {r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>4.0f}")

    # ── Save ──────────────────────────────────────────────────────────
    df.to_csv(os.path.join(SAVE_DIR, "grid_results.csv"), index=False)
    top30.to_csv(os.path.join(SAVE_DIR, "top30.csv"), index=False)
    yearly.to_csv(os.path.join(SAVE_DIR, "yearly_best.csv"), index=False)
    print(f"\n  CSVs saved to {SAVE_DIR}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
