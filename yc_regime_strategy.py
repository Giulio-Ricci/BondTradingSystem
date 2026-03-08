"""
===============================================================================
  YC REGIME STRATEGY
  Yield Curve Bull/Bear Steepening/Flattening Trading System

  A completely new strategy based purely on yield curve regime classification.
  No RR signals used. Trades TLT vs SHV based on yield curve regime state.
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
SAVE_DIR = os.path.join(os.path.dirname(__file__), "output", "yc_regime")
os.makedirs(SAVE_DIR, exist_ok=True)

BACKTEST_START = "2009-01-01"
RISK_FREE      = 0.02
ANN            = 252

SPREADS = {
    "2s30s":  ("US_2Y",  "US_30Y"),
    "2s10s":  ("US_2Y",  "US_10Y"),
    "3m2y":   ("US_3M",  "US_2Y"),
    "2s5s":   ("US_2Y",  "US_5Y"),
    "5s10s":  ("US_5Y",  "US_10Y"),
    "10s30s": ("US_10Y", "US_30Y"),
}
SP_NAMES = list(SPREADS.keys())

WINDOWS = [10, 15, 21, 42, 63, 126]

# Weight mappings: regime → TLT weight
W_MAPS = {
    "bull_bear": {                          # Bull→LONG, Bear→SHORT
        "BULL_STEEP": 1.0, "BULL_FLAT": 1.0,
        "BEAR_STEEP": 0.0, "BEAR_FLAT": 0.0, "UNDEFINED": 0.5,
    },
    "steep_flat": {                         # Flat→LONG, Steep→SHORT
        "BULL_STEEP": 0.0, "BULL_FLAT": 1.0,
        "BEAR_STEEP": 0.0, "BEAR_FLAT": 1.0, "UNDEFINED": 0.5,
    },
    "full_4": {                             # BullFlat→LONG, BearSteep→SHORT
        "BULL_STEEP": 0.5, "BULL_FLAT": 1.0,
        "BEAR_STEEP": 0.0, "BEAR_FLAT": 0.5, "UNDEFINED": 0.5,
    },
    "directional": {                        # Bull→LONG, BearSteep→SHORT
        "BULL_STEEP": 1.0, "BULL_FLAT": 1.0,
        "BEAR_STEEP": 0.0, "BEAR_FLAT": 0.5, "UNDEFINED": 0.5,
    },
}

# Scoring for multi-spread aggregation
SCORE = {"BULL_FLAT": +2, "BULL_STEEP": +1, "UNDEFINED": 0,
         "BEAR_FLAT": -1, "BEAR_STEEP": -2}

COOLDOWN  = 5
DELAY     = 1
TCOST_BPS = 5
W_INIT    = 0.5


# ─── Regime computation ────────────────────────────────────────────────────

def compute_regime(yld, short_col, long_col, window=21):
    """Classify each day into bull/bear steep/flat regime."""
    level  = (yld[short_col] + yld[long_col]) / 2
    spread = yld[long_col] - yld[short_col]
    dl = level.diff(window)
    ds = spread.diff(window)
    r = pd.Series("UNDEFINED", index=yld.index)
    r[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    r[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    r[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    r[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    return r


# ─── Backtest engine ───────────────────────────────────────────────────────

def apply_signals(target_w, cooldown=5, delay=1, w_init=0.5):
    """Target weights → actual weights with cooldown + delay."""
    n = len(target_w)
    signals = []
    cur = w_init
    last_sig = -cooldown - 1
    for i in range(n):
        tw = target_w[i]
        if np.isnan(tw):
            continue
        if abs(tw - cur) > 0.01 and (i - last_sig) > cooldown:
            signals.append((i, i + delay, tw))
            cur = tw
            last_sig = i
    actual = np.full(n, w_init)
    for _, ex, w in signals:
        if ex < n:
            actual[ex:] = w
    return actual, signals


def calc_metrics(daily_ret, equity):
    """Standard performance metrics."""
    n = len(daily_ret)
    ann_ret = equity[-1] ** (ANN / n) - 1
    vol     = np.std(daily_ret) * np.sqrt(ANN)
    rf_d    = RISK_FREE / ANN
    exc     = daily_ret - rf_d
    sharpe  = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)
    neg     = exc[exc < 0]
    ds      = np.sqrt(np.mean(neg**2)) * np.sqrt(ANN) if len(neg) > 0 else 1e-10
    sortino = np.mean(exc) * ANN / (ds + 1e-10)
    peak    = np.maximum.accumulate(equity)
    mdd     = ((equity - peak) / peak).min()
    calmar  = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0.0
    return dict(sharpe=sharpe, sortino=sortino, ann_ret=ann_ret,
                vol=vol, mdd=mdd, calmar=calmar, total_ret=equity[-1] - 1)


def run_bt(actual_w, ret_tlt, ret_shv, tcost_bps=5):
    """Backtest with given actual weights. Returns metrics + equity."""
    dr = actual_w * ret_tlt + (1 - actual_w) * ret_shv
    wd = np.abs(np.diff(actual_w, prepend=actual_w[0]))
    dr -= wd * (tcost_bps / 10000.0)
    eq = np.cumprod(1 + dr)
    m  = calc_metrics(dr, eq)
    m["n_trades"]  = int(np.sum(wd > 0.01))
    m["equity"]    = eq
    m["daily_ret"] = dr
    return m


# ─── Phase 1: single spread grid ──────────────────────────────────────────

def phase1(yld, ret_tlt, ret_shv):
    rows = []
    for sp in SP_NAMES:
        sc, lc = SPREADS[sp]
        for win in WINDOWS:
            regime = compute_regime(yld, sc, lc, win)
            for mp_name, mp in W_MAPS.items():
                tw = regime.map(mp).values.astype(float)
                tw = np.nan_to_num(tw, nan=0.5)
                aw, _ = apply_signals(tw, COOLDOWN, DELAY, W_INIT)
                m = run_bt(aw, ret_tlt, ret_shv, TCOST_BPS)
                rows.append(dict(spread=sp, window=win, mapping=mp_name,
                    sharpe=m["sharpe"], sortino=m["sortino"],
                    ann_ret=m["ann_ret"], vol=m["vol"], mdd=m["mdd"],
                    calmar=m["calmar"], total_ret=m["total_ret"],
                    n_trades=m["n_trades"]))
    return pd.DataFrame(rows)


# ─── Phase 2: multi-spread scoring ────────────────────────────────────────

def score_array(yld, spread_list, window):
    total = np.zeros(len(yld))
    for sp in spread_list:
        sc, lc = SPREADS[sp]
        regime = compute_regime(yld, sc, lc, window)
        s = regime.map(SCORE).values.astype(float)
        total += np.nan_to_num(s, nan=0.0)
    return total


def phase2(yld, ret_tlt, ret_shv, groups):
    rows = []
    for gname, gspreads in groups.items():
        n_sp = len(gspreads)
        max_sc = n_sp * 2
        for win in WINDOWS:
            sc_arr = score_array(yld, gspreads, win)
            for lt in range(1, max_sc + 1):
                for st in range(1, max_sc + 1):
                    tw = np.full(len(sc_arr), 0.5)
                    tw[sc_arr >= lt]  = 1.0
                    tw[sc_arr <= -st] = 0.0
                    aw, _ = apply_signals(tw, COOLDOWN, DELAY, W_INIT)
                    m = run_bt(aw, ret_tlt, ret_shv, TCOST_BPS)
                    rows.append(dict(group=gname, n_spreads=n_sp,
                        window=win, long_th=lt, short_th=-st,
                        sharpe=m["sharpe"], sortino=m["sortino"],
                        ann_ret=m["ann_ret"], vol=m["vol"], mdd=m["mdd"],
                        calmar=m["calmar"], total_ret=m["total_ret"],
                        n_trades=m["n_trades"]))
    return pd.DataFrame(rows)


# ─── Phase 3: alignment counting ──────────────────────────────────────────

def phase3(yld, ret_tlt, ret_shv):
    rows = []
    for win in WINDOWS:
        bull_c = np.zeros(len(yld))
        bear_c = np.zeros(len(yld))
        for sp in SP_NAMES:
            sc, lc = SPREADS[sp]
            regime = compute_regime(yld, sc, lc, win)
            bull_c += ((regime == "BULL_STEEP") | (regime == "BULL_FLAT")).values.astype(float)
            bear_c += ((regime == "BEAR_STEEP") | (regime == "BEAR_FLAT")).values.astype(float)
        for ath in [3, 4, 5, 6]:
            tw = np.full(len(yld), 0.5)
            tw[bull_c >= ath] = 1.0
            tw[bear_c >= ath] = 0.0
            conflict = (bull_c >= ath) & (bear_c >= ath)
            if conflict.any():
                tw[conflict & (bull_c > bear_c)] = 1.0
                tw[conflict & (bear_c > bull_c)] = 0.0
                tw[conflict & (bull_c == bear_c)] = 0.5
            aw, _ = apply_signals(tw, COOLDOWN, DELAY, W_INIT)
            m = run_bt(aw, ret_tlt, ret_shv, TCOST_BPS)
            rows.append(dict(window=win, align_th=ath,
                sharpe=m["sharpe"], sortino=m["sortino"],
                ann_ret=m["ann_ret"], vol=m["vol"], mdd=m["mdd"],
                calmar=m["calmar"], total_ret=m["total_ret"],
                n_trades=m["n_trades"]))
    return pd.DataFrame(rows)


# ─── Charts ────────────────────────────────────────────────────────────────

def plot_heatmap(df, path):
    best = df.loc[df.groupby(["spread", "window"])["sharpe"].idxmax()]
    pivot = best.pivot(index="spread", columns="window", values="sharpe")
    pivot = pivot.reindex(SP_NAMES)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-0.15, vmax=0.35)
    ax.set_xticks(range(len(WINDOWS)))
    ax.set_xticklabels(WINDOWS)
    ax.set_yticks(range(len(SP_NAMES)))
    ax.set_yticklabels(SP_NAMES)
    for i in range(len(SP_NAMES)):
        for j in range(len(WINDOWS)):
            v = pivot.values[i, j]
            mp = best[(best["spread"] == SP_NAMES[i]) &
                      (best["window"] == WINDOWS[j])]["mapping"].values
            mp_str = mp[0] if len(mp) > 0 else "?"
            ax.text(j, i, f"{v:.3f}\n{mp_str}", ha="center", va="center",
                    fontsize=7)
    plt.colorbar(im, ax=ax, label="Sharpe")
    ax.set_xlabel("Window (days)")
    ax.set_ylabel("Spread")
    ax.set_title("Phase 1: Single Spread — Best Sharpe per cell")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_equity(dates, eq_s, eq_bh, sigs, aw, path, title=""):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
        height_ratios=[3, 1], sharex=True,
        gridspec_kw={"hspace": 0.08})

    ax1.plot(dates, eq_s, "b-", lw=1.5, label="YC Regime Strategy")
    ax1.plot(dates, eq_bh, color="gray", lw=1, alpha=0.7, label="B&H TLT")
    for sd, ed, w in sigs:
        if ed < len(dates):
            c = "green" if w > 0.7 else ("red" if w < 0.3 else "orange")
            mk = "^" if w > 0.7 else ("v" if w < 0.3 else "s")
            ax1.plot(dates[ed], eq_s[ed], mk, color=c, markersize=7, zorder=5)
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(dates, aw, 0.5, where=aw > 0.5,
                     color="green", alpha=0.3, label="LONG TLT")
    ax2.fill_between(dates, aw, 0.5, where=aw < 0.5,
                     color="red", alpha=0.3, label="SHORT (SHV)")
    ax2.axhline(0.5, color="gray", ls="--", lw=0.5)
    ax2.set_ylabel("TLT Weight")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

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
    ax.set_title("Yearly Performance: YC Regime Strategy vs B&H TLT")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_corr(corr, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    n = len(corr)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i,j]:.2f}",
                    ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Cross-Spread Regime Score Correlation (window=21)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_regime_timeline(yld, dates_idx, best_win, path):
    """Show regime state over time for all spreads at given window."""
    color_map = {"BULL_FLAT": 0, "BULL_STEEP": 1,
                 "UNDEFINED": 2, "BEAR_FLAT": 3, "BEAR_STEEP": 4}
    cmap = plt.cm.colors.ListedColormap(
        ["#2ecc71", "#82e0aa", "#d5d8dc", "#f1948a", "#e74c3c"])

    fig, axes = plt.subplots(len(SP_NAMES), 1, figsize=(16, 10),
                             sharex=True)
    for ax, sp in zip(axes, SP_NAMES):
        sc, lc = SPREADS[sp]
        regime = compute_regime(yld.loc[dates_idx], sc, lc, best_win)
        vals = regime.map(color_map).values.astype(float)
        vals = np.nan_to_num(vals, nan=2)
        ax.imshow(vals.reshape(1, -1), aspect="auto", cmap=cmap,
                  vmin=0, vmax=4,
                  extent=[0, len(dates_idx), 0, 1])
        ax.set_yticks([0.5])
        ax.set_yticklabels([sp], fontsize=9)
        ax.set_xlim(0, len(dates_idx))

    # X-axis labels
    tick_positions = np.linspace(0, len(dates_idx)-1, 12).astype(int)
    axes[-1].set_xticks(tick_positions)
    axes[-1].set_xticklabels(
        [dates_idx[i].strftime("%Y-%m") for i in tick_positions],
        rotation=45, fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Bull Flat"),
        Patch(facecolor="#82e0aa", label="Bull Steep"),
        Patch(facecolor="#d5d8dc", label="Undefined"),
        Patch(facecolor="#f1948a", label="Bear Flat"),
        Patch(facecolor="#e74c3c", label="Bear Steep"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle(f"Regime Timeline — All Spreads (window={best_win})",
                 y=1.01, fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ─── Yearly analysis helper ────────────────────────────────────────────────

def yearly_table(equity, daily_ret, actual_w, dates, ret_tlt):
    years = sorted(set(d.year for d in dates))
    rows = []
    for y in years:
        mask = np.array([d.year == y for d in dates])
        if mask.sum() == 0:
            continue
        dr = daily_ret[mask]
        bh = ret_tlt[mask]
        eq_y = np.cumprod(1 + dr)
        bh_y = np.cumprod(1 + bh)
        s_ret = eq_y[-1] - 1
        b_ret = bh_y[-1] - 1
        wd = np.abs(np.diff(actual_w[mask], prepend=actual_w[mask][0]))
        rows.append(dict(year=y, strategy=s_ret, benchmark=b_ret,
                         excess=s_ret - b_ret, vol=np.std(dr)*np.sqrt(252),
                         n_trades=int(np.sum(wd > 0.01))))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("  YC REGIME STRATEGY")
    print("  Yield Curve Bull/Bear Steep/Flat Trading System")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
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

    eq_bh = np.cumprod(1 + ret_tlt)
    bh = calc_metrics(ret_tlt, eq_bh)

    print(f"\n  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d}"
          f"  ({len(common)} days)")
    print(f"  B&H TLT: Sharpe={bh['sharpe']:.3f}  "
          f"Return={bh['total_ret']:.1%}  MDD={bh['mdd']:.1%}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: SINGLE SPREAD
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 1: Single Spread Analysis")
    print("  (6 spreads x 6 windows x 4 mappings = 144 configs)")
    print("=" * 70)

    df1 = phase1(yld_bt, ret_tlt, ret_shv)
    print(f"  Tested {len(df1)} configurations\n")

    top20 = df1.nlargest(20, "sharpe")
    print(f"  {'#':>3} {'Spread':>7} {'Win':>4} {'Mapping':>12} "
          f"{'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Trades':>6}")
    print("  " + "-" * 60)
    for i, (_, r) in enumerate(top20.iterrows(), 1):
        print(f"  {i:>3} {r['spread']:>7} {r['window']:>4.0f} "
              f"{r['mapping']:>12} {r['sharpe']:>7.3f} "
              f"{r['ann_ret']:>+7.1%} {r['mdd']:>8.1%} "
              f"{r['n_trades']:>6.0f}")

    print("\n  Best per spread:")
    for sp in SP_NAMES:
        sub = df1[df1["spread"] == sp]
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"    {sp:>7}: Sharpe={b['sharpe']:.3f}  w={b['window']:.0f}"
              f"  map={b['mapping']}  ret={b['ann_ret']:+.1%}")

    print("\n  Best per mapping:")
    for mp in W_MAPS:
        sub = df1[df1["mapping"] == mp]
        b = sub.loc[sub["sharpe"].idxmax()]
        print(f"    {mp:>12}: Sharpe={b['sharpe']:.3f}  "
              f"sp={b['spread']}  w={b['window']:.0f}")

    plot_heatmap(df1, os.path.join(SAVE_DIR, "01_phase1_heatmap.png"))
    print("\n  -> 01_phase1_heatmap.png")

    # ══════════════════════════════════════════════════════════════════
    # CORRELATION ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  CROSS-SPREAD CORRELATION (window=21)")
    print("=" * 70)

    scores_df = pd.DataFrame(index=yld_bt.index)
    for sp in SP_NAMES:
        sc, lc = SPREADS[sp]
        regime = compute_regime(yld_bt, sc, lc, 21)
        scores_df[sp] = regime.map(SCORE).astype(float)
    corr = scores_df.corr()
    print(f"\n{corr.round(2).to_string()}")

    plot_corr(corr, os.path.join(SAVE_DIR, "02_correlation.png"))
    print("\n  -> 02_correlation.png")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: MULTI-SPREAD SCORING
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 2: Multi-Spread Scoring Strategy")
    print("=" * 70)

    # Select spread groups: top3/top4 from Phase 1
    bps = df1.loc[df1.groupby("spread")["sharpe"].idxmax()]
    top_sp = bps.nlargest(4, "sharpe")["spread"].tolist()
    top3, top4 = top_sp[:3], top_sp[:4]

    groups = {
        "all_6":   SP_NAMES,
        "top3":    top3,
        "top4":    top4,
        "broad":   ["2s10s", "2s30s"],
        "belly":   ["2s5s", "5s10s"],
        "ends":    ["3m2y", "10s30s"],
        "no_3m2y": [s for s in SP_NAMES if s != "3m2y"],
    }
    print(f"  Groups: {list(groups.keys())}")
    print(f"  Top3: {top3}  |  Top4: {top4}")

    df2 = phase2(yld_bt, ret_tlt, ret_shv, groups)
    print(f"\n  Tested {len(df2)} configurations")

    top20_2 = df2.nlargest(20, "sharpe")
    print(f"\n  {'#':>3} {'Group':>8} {'Win':>4} {'L_th':>5} {'S_th':>5}"
          f" {'Sharpe':>7} {'Return':>7} {'MDD':>8} {'Trades':>6}")
    print("  " + "-" * 62)
    for i, (_, r) in enumerate(top20_2.iterrows(), 1):
        print(f"  {i:>3} {r['group']:>8} {r['window']:>4.0f} "
              f"{r['long_th']:>5.0f} {r['short_th']:>5.0f} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>6.0f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: ALIGNMENT COUNTING
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PHASE 3: Alignment Counting (N/6 spreads agree)")
    print("=" * 70)

    df3 = phase3(yld_bt, ret_tlt, ret_shv)
    print(f"  Tested {len(df3)} configurations\n")

    print(f"  {'Win':>4} {'Align':>6} {'Sharpe':>7} {'Return':>7} "
          f"{'MDD':>8} {'Trades':>6}")
    print("  " + "-" * 46)
    for _, r in df3.sort_values("sharpe", ascending=False).iterrows():
        print(f"  {r['window']:>4.0f} {r['align_th']:>6.0f} "
              f"{r['sharpe']:>7.3f} {r['ann_ret']:>+7.1%} "
              f"{r['mdd']:>8.1%} {r['n_trades']:>6.0f}")

    # ══════════════════════════════════════════════════════════════════
    # OVERALL BEST
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  OVERALL BEST CONFIGURATIONS")
    print("=" * 70)

    b1 = df1.loc[df1["sharpe"].idxmax()]
    b2 = df2.loc[df2["sharpe"].idxmax()]
    b3 = df3.loc[df3["sharpe"].idxmax()]

    configs = [
        ("Phase1-Single",  b1["sharpe"], b1),
        ("Phase2-Score",   b2["sharpe"], b2),
        ("Phase3-Align",   b3["sharpe"], b3),
    ]
    for label, sh, b in configs:
        print(f"\n  {label}:  Sharpe={sh:.4f}  "
              f"Return={b['ann_ret']:.2%}  MDD={b['mdd']:.2%}  "
              f"Trades={b['n_trades']:.0f}")

    best_idx = np.argmax([c[1] for c in configs])
    best_phase = best_idx + 1
    best_sharpe = configs[best_idx][1]
    print(f"\n  >>> WINNER: {configs[best_idx][0]}  "
          f"(Sharpe = {best_sharpe:.4f})")

    # ── Re-run winner with full output ────────────────────────────────
    if best_phase == 1:
        sp = b1["spread"]; win = int(b1["window"]); mp = b1["mapping"]
        sc, lc = SPREADS[sp]
        regime = compute_regime(yld_bt, sc, lc, win)
        tw = regime.map(W_MAPS[mp]).values.astype(float)
        tw = np.nan_to_num(tw, nan=0.5)
        label = f"{sp} w{win} {mp}"
    elif best_phase == 2:
        grp = b2["group"]; win = int(b2["window"])
        lt = int(b2["long_th"]); st = int(b2["short_th"])
        sc_arr = score_array(yld_bt, groups[grp], win)
        tw = np.full(len(sc_arr), 0.5)
        tw[sc_arr >= lt]  = 1.0
        tw[sc_arr <= st]  = 0.0
        label = f"{grp} w{win} th[{lt},{st}]"
    else:
        win = int(b3["window"]); ath = int(b3["align_th"])
        bull_c = np.zeros(len(yld_bt))
        bear_c = np.zeros(len(yld_bt))
        for sp in SP_NAMES:
            sc, lc = SPREADS[sp]
            regime = compute_regime(yld_bt, sc, lc, win)
            bull_c += ((regime == "BULL_STEEP") |
                       (regime == "BULL_FLAT")).values.astype(float)
            bear_c += ((regime == "BEAR_STEEP") |
                       (regime == "BEAR_FLAT")).values.astype(float)
        tw = np.full(len(yld_bt), 0.5)
        tw[bull_c >= ath] = 1.0
        tw[bear_c >= ath] = 0.0
        label = f"align w{win} th{ath}"

    aw, sigs = apply_signals(tw, COOLDOWN, DELAY, W_INIT)
    res = run_bt(aw, ret_tlt, ret_shv, TCOST_BPS)

    print(f"\n  Best config: {label}")
    print(f"    Sharpe  = {res['sharpe']:.4f}")
    print(f"    Sortino = {res['sortino']:.4f}")
    print(f"    AnnRet  = {res['ann_ret']:+.2%}")
    print(f"    Vol     = {res['vol']:.2%}")
    print(f"    MDD     = {res['mdd']:.2%}")
    print(f"    Calmar  = {res['calmar']:.4f}")
    print(f"    TotRet  = {res['total_ret']:+.2%}")
    print(f"    Trades  = {res['n_trades']}")

    # Yearly
    yearly = yearly_table(res["equity"], res["daily_ret"], aw, dates, ret_tlt)
    print(f"\n  {'Year':>6} {'Strat':>9} {'B&H':>9} {'Excess':>9} {'Tr':>4}")
    print("  " + "-" * 40)
    for _, r in yearly.iterrows():
        mk = " *" if abs(r["excess"]) > 0.01 else ""
        print(f"  {r['year']:>6.0f} {r['strategy']:>+9.1%} "
              f"{r['benchmark']:>+9.1%} {r['excess']:>+9.1%} "
              f"{r['n_trades']:>4.0f}{mk}")
    pos_ex = (yearly["excess"] > 0).sum()
    print(f"\n  Positive excess: {pos_ex}/{len(yearly)} "
          f"({pos_ex/len(yearly):.0%})")
    print(f"  Avg excess: {yearly['excess'].mean():+.2%}")

    # ── Charts ────────────────────────────────────────────────────────
    plot_equity(dates, res["equity"], eq_bh, sigs, aw,
                os.path.join(SAVE_DIR, "03_best_equity.png"),
                f"YC Regime Strategy — {label}\n"
                f"Sharpe={res['sharpe']:.3f}  "
                f"Return={res['total_ret']:+.1%}  "
                f"MDD={res['mdd']:.1%}")
    print(f"\n  -> 03_best_equity.png")

    plot_yearly(yearly, os.path.join(SAVE_DIR, "04_yearly.png"))
    print(f"  -> 04_yearly.png")

    # Regime timeline for best window
    best_win_overall = win
    plot_regime_timeline(yld, dates, best_win_overall,
                         os.path.join(SAVE_DIR, "05_regime_timeline.png"))
    print(f"  -> 05_regime_timeline.png")

    # ── Save CSVs ─────────────────────────────────────────────────────
    df1.to_csv(os.path.join(SAVE_DIR, "phase1_results.csv"), index=False)
    df2.to_csv(os.path.join(SAVE_DIR, "phase2_results.csv"), index=False)
    df3.to_csv(os.path.join(SAVE_DIR, "phase3_results.csv"), index=False)
    yearly.to_csv(os.path.join(SAVE_DIR, "yearly_best.csv"), index=False)
    print(f"\n  CSVs saved to {SAVE_DIR}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
