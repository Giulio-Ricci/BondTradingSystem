"""
full_optimization.py - Exhaustive grid search for the RR Duration Strategy.

Explores ALL combinations of signal structure and thresholds to map
the full parameter landscape.  Intentionally overfitting-style to
understand WHERE the strategy works best and how the parameters interact.

Stages:
  1. Pre-compute all signal variants (tenor x z_type x z_window x move_window)
  2. Quick screen with default thresholds -> rank signal structures
  3. Fine threshold grid on top signals -> exhaustive search
  4. Analysis & visualisation
"""

import os
import time
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
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# -- Signal structure dimensions --
RR_CONFIGS = {
    "1W":        ["1W"],
    "1M":        ["1M"],
    "3M":        ["3M"],
    "6M":        ["6M"],
    "1M_3M":     ["1M", "3M"],
    "1M_3M_6M":  ["1M", "3M", "6M"],
    "ALL":       ["1W", "1M", "3M", "6M"],
}

ZSCORE_TYPES   = ["std", "move"]
ZSCORE_WINDOWS = [63, 125, 180, 250, 375, 504]
MOVE_MED_WINS  = [252, 504, 756]

# -- Threshold grid (fine) --
TL_CALM_RANGE   = np.arange(-5.0, -0.75, 0.25)    # 17 values
TS_CALM_RANGE   = np.arange(1.0, 5.25, 0.25)      # 17 values
TL_STRESS_RANGE = np.arange(-6.0, -1.75, 0.50)    #  9 values
COOLDOWNS       = [0, 3, 5, 10]

# -- Quick screen threshold sets (Stage 2) --
SCREEN_THRESHOLDS = [
    (-3.5, 3.0, -4.0),
    (-3.0, 2.5, -3.5),
    (-3.0, 2.0, -4.0),
    (-2.5, 2.0, -3.5),
    (-4.0, 3.5, -4.5),
    (-2.0, 1.5, -3.0),
    (-3.5, 2.5, -4.5),
    (-4.0, 3.0, -5.0),
]

TOP_N_SIGNALS = 20   # signals advanced to Stage 3

# -- Output --
OPT_DIR = os.path.join(SAVE_DIR, "optimization")
os.makedirs(OPT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_rr_blend(options_df, rr_tenors, tenor_keys):
    """Compute RR for given tenor keys; average if multiple tenors."""
    parts = []
    for tk in tenor_keys:
        put_col, call_col = rr_tenors[tk]
        rr = (
            pd.to_numeric(options_df[put_col], errors="coerce")
            - pd.to_numeric(options_df[call_col], errors="coerce")
        )
        parts.append(rr)
    return pd.concat(parts, axis=1).mean(axis=1)


def compute_zscore(rr, move, z_type, z_window, move_med_window=504):
    """Compute rolling z-score (standard or MOVE-adjusted)."""
    min_per = max(z_window // 4, 20)
    mu  = rr.rolling(z_window, min_periods=min_per).mean()
    sig = rr.rolling(z_window, min_periods=min_per).std()

    if z_type == "std":
        return (rr - mu) / sig.replace(0, np.nan)

    # MOVE-adjusted
    move_med = move.rolling(move_med_window, min_periods=min_per).median()
    move_ratio = (move / move_med).clip(0.5, 2.5)
    sig_adj = sig * np.sqrt(move_ratio)
    return (rr - mu) / sig_adj.replace(0, np.nan)


def build_all_signals(D):
    """Pre-compute every signal variant.  Returns dict[key] -> pd.Series."""
    signals = {}
    options = D["options"]
    move = D["move"]

    for rr_name, tenor_keys in RR_CONFIGS.items():
        rr = compute_rr_blend(options, RR_TENORS, tenor_keys)

        for z_type in ZSCORE_TYPES:
            for z_win in ZSCORE_WINDOWS:
                if z_type == "std":
                    z = compute_zscore(rr, move, "std", z_win)
                    key = f"{rr_name}|std|w{z_win}"
                    signals[key] = z
                else:
                    for mw in MOVE_MED_WINS:
                        z = compute_zscore(rr, move, "move", z_win, mw)
                        key = f"{rr_name}|move|w{z_win}|mw{mw}"
                        signals[key] = z

    return signals


# ═══════════════════════════════════════════════════════════════════════════
# FAST BACKTEST  (numpy arrays, no pandas overhead)
# ═══════════════════════════════════════════════════════════════════════════

def fast_bt(z_arr, reg_arr, r_long, r_short,
            tl_c, ts_c, tl_s, cooldown=5,
            delay=1, tcost_bps=5, w_init=0.5):
    """
    Vectorised-ish backtest.  reg_arr: 0=CALM, 1=STRESS.
    Returns (sharpe, ann_ret, vol, mdd, n_events, equity_final).
    """
    n = len(z_arr)
    if n < 50:
        return 0.0, 0.0, 0.0, 0.0, 0, 1.0

    weight = np.empty(n)
    weight[0] = w_init
    n_events = 0
    last_ev = -cooldown - 1

    for i in range(n):
        zv = z_arr[i]

        # propagate previous weight by default
        w_prev = weight[i - 1] if i > 0 else w_init

        if np.isnan(zv):
            weight[i] = w_prev
            continue

        reg = reg_arr[i]
        triggered = False

        if reg == 0:  # CALM
            if zv <= tl_c and (i - last_ev) >= cooldown:
                weight[i] = 0.0
                triggered = True
            elif zv >= ts_c and (i - last_ev) >= cooldown:
                weight[i] = 1.0
                triggered = True
            else:
                weight[i] = w_prev
        else:  # STRESS
            if zv <= tl_s and (i - last_ev) >= cooldown:
                weight[i] = 0.0
                triggered = True
            else:
                weight[i] = w_prev

        if triggered:
            last_ev = i
            n_events += 1

    # execution delay
    w_del = np.empty(n)
    shift = 1 + delay
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    # portfolio return
    gross = w_del * r_long + (1.0 - w_del) * r_short

    # transaction costs
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost_bps / 10_000)

    # equity
    equity = np.cumprod(1.0 + net)
    total = equity[-1] - 1.0
    years = n / 252.0
    ann_ret = (1.0 + total) ** (1.0 / max(years, 0.01)) - 1.0
    vol = np.std(net) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0.0

    running_max = np.maximum.accumulate(equity)
    mdd = (equity / running_max - 1.0).min()

    return sharpe, ann_ret, vol, mdd, n_events, equity[-1]


def fast_bt_equity(z_arr, reg_arr, r_long, r_short,
                   tl_c, ts_c, tl_s, cooldown=5,
                   delay=1, tcost_bps=5, w_init=0.5):
    """Same as fast_bt but returns the full equity curve array."""
    n = len(z_arr)
    weight = np.empty(n)
    weight[0] = w_init
    last_ev = -cooldown - 1

    for i in range(n):
        zv = z_arr[i]
        w_prev = weight[i - 1] if i > 0 else w_init
        if np.isnan(zv):
            weight[i] = w_prev
            continue
        reg = reg_arr[i]
        triggered = False
        if reg == 0:
            if zv <= tl_c and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True
            elif zv >= ts_c and (i - last_ev) >= cooldown:
                weight[i] = 1.0; triggered = True
            else:
                weight[i] = w_prev
        else:
            if zv <= tl_s and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True
            else:
                weight[i] = w_prev
        if triggered:
            last_ev = i

    shift = 1 + delay
    w_del = np.empty(n)
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost_bps / 10_000)
    return np.cumprod(1.0 + net)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()

    print("=" * 70)
    print("  EXHAUSTIVE OPTIMIZATION  —  RR Duration Strategy")
    print("=" * 70)

    # ── 1. Load data ───────────────────────────────────────────────────────
    print("\n[1/7] Loading data ...")
    D = load_all()

    # ── 2. Regime (fitted once, shared) ────────────────────────────────────
    print("\n[2/7] Fitting HMM regime ...")
    regime = fit_hmm_regime(D["move"], D["vix"])
    regime_num = (regime == "STRESS").astype(np.int8)

    etf_ret = D["etf_ret"]
    start = pd.Timestamp(BACKTEST_START)

    # ── 3. Pre-compute all signals ─────────────────────────────────────────
    print("\n[3/7] Building signal variants ...")
    signals = build_all_signals(D)
    print(f"       {len(signals)} variants generated")

    # ── 4. Quick screen (Stage 1) ──────────────────────────────────────────
    print(f"\n[4/7] Stage 1 — Quick screen ({len(signals)} signals x "
          f"{len(SCREEN_THRESHOLDS)} threshold sets) ...")

    screen_rows = []

    for sig_key, z_series in signals.items():
        common = (
            z_series.dropna().index
            .intersection(regime.index)
            .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
        )
        common = common[common >= start].sort_values()
        if len(common) < 100:
            continue

        z_arr  = z_series.reindex(common).values
        rg_arr = regime_num.reindex(common).values
        rl     = etf_ret["TLT"].reindex(common).fillna(0).values
        rs     = etf_ret["SHV"].reindex(common).fillna(0).values

        for tl_c, ts_c, tl_s in SCREEN_THRESHOLDS:
            sh, ar, vol, mdd, ne, _ = fast_bt(z_arr, rg_arr, rl, rs,
                                               tl_c, ts_c, tl_s)
            screen_rows.append({
                "signal": sig_key,
                "tl_calm": tl_c, "ts_calm": ts_c, "tl_stress": tl_s,
                "sharpe": sh, "ann_ret": ar, "vol": vol,
                "mdd": mdd, "n_events": ne,
            })

    screen_df = pd.DataFrame(screen_rows)

    # rank signals by best sharpe across threshold sets
    sig_rank = (screen_df.groupby("signal")["sharpe"]
                .max().sort_values(ascending=False))

    print(f"\n       Top {TOP_N_SIGNALS} signal structures:")
    print(f"       {'Signal':<42s}  Best Sharpe")
    print("       " + "-" * 56)
    for sig, sh in sig_rank.head(TOP_N_SIGNALS).items():
        print(f"       {sig:<42s}  {sh:+.3f}")

    top_keys = sig_rank.head(TOP_N_SIGNALS).index.tolist()

    # ── 5. Fine grid (Stage 2) ─────────────────────────────────────────────
    n_thresh = (len(TL_CALM_RANGE) * len(TS_CALM_RANGE)
                * len(TL_STRESS_RANGE) * len(COOLDOWNS))
    total_runs = len(top_keys) * n_thresh
    print(f"\n[5/7] Stage 2 — Fine grid on top {len(top_keys)} signals")
    print(f"       {n_thresh:,} threshold combos per signal  =  "
          f"{total_runs:,} total backtests")

    fine_rows = []
    t1 = time.time()

    for si, sig_key in enumerate(top_keys):
        z_series = signals[sig_key]
        common = (
            z_series.dropna().index
            .intersection(regime.index)
            .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
        )
        common = common[common >= start].sort_values()
        if len(common) < 100:
            continue

        z_arr  = z_series.reindex(common).values
        rg_arr = regime_num.reindex(common).values
        rl     = etf_ret["TLT"].reindex(common).fillna(0).values
        rs     = etf_ret["SHV"].reindex(common).fillna(0).values

        count = 0
        for tl_c in TL_CALM_RANGE:
            for ts_c in TS_CALM_RANGE:
                for tl_s in TL_STRESS_RANGE:
                    for cd in COOLDOWNS:
                        sh, ar, vol, mdd, ne, eqf = fast_bt(
                            z_arr, rg_arr, rl, rs,
                            tl_c, ts_c, tl_s, cooldown=cd,
                        )
                        fine_rows.append({
                            "signal": sig_key,
                            "tl_calm": tl_c, "ts_calm": ts_c,
                            "tl_stress": tl_s, "cooldown": cd,
                            "sharpe": sh, "ann_ret": ar, "vol": vol,
                            "mdd": mdd, "n_events": ne,
                            "equity_final": eqf,
                        })
                        count += 1

        elapsed_s = time.time() - t1
        pct = (si + 1) / len(top_keys) * 100
        print(f"       [{si+1:2d}/{len(top_keys)}]  {sig_key:<42s}  "
              f"{count:>6,} runs   ({pct:.0f}%  {elapsed_s:.0f}s)")

    df = pd.DataFrame(fine_rows)
    df.sort_values("sharpe", ascending=False, inplace=True)

    # ── 6. Analysis ────────────────────────────────────────────────────────
    print(f"\n[6/7] Analysis  ({len(df):,} total configurations)")
    print()

    # Top 30
    print("  " + "=" * 115)
    print(f"  {'Signal':<42s} tl_c   ts_c  tl_s   cd  "
          f"Sharpe  AnnRet    Vol     MDD  Events  EqFinal")
    print("  " + "-" * 115)
    for _, r in df.head(40).iterrows():
        print(f"  {r['signal']:<42s} {r['tl_calm']:5.2f}  "
              f"{r['ts_calm']:5.2f} {r['tl_stress']:5.1f}  "
              f"{r['cooldown']:3.0f}  {r['sharpe']:+6.3f}  "
              f"{r['ann_ret']:+6.3f}  {r['vol']:5.3f}  "
              f"{r['mdd']:7.3f}  {r['n_events']:5.0f}  "
              f"{r['equity_final']:7.3f}")
    print("  " + "=" * 115)

    # Save full results
    csv_path = os.path.join(OPT_DIR, "full_optimization_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Full results saved -> {csv_path}")

    # ── Summary statistics on top 50 ───────────────────────────────────────
    top50 = df.head(50)
    print(f"\n  Parameter distribution of TOP 50 configs:")
    for col in ["tl_calm", "ts_calm", "tl_stress", "cooldown"]:
        vals = top50[col]
        print(f"    {col:12s}  mean={vals.mean():+.2f}  "
              f"std={vals.std():.2f}  "
              f"range=[{vals.min():+.2f}, {vals.max():+.2f}]  "
              f"mode={vals.mode().iloc[0]:+.2f}")

    # Signal structure breakdown
    print(f"\n  Signal structure in TOP 50:")
    sig_counts = top50["signal"].value_counts()
    for sig, cnt in sig_counts.items():
        print(f"    {sig:<42s}  {cnt} configs")

    # Tenor breakdown
    print(f"\n  Tenor breakdown in TOP 50:")
    tenor_counts = {}
    for sig in top50["signal"]:
        tenor = sig.split("|")[0]
        tenor_counts[tenor] = tenor_counts.get(tenor, 0) + 1
    for t, c in sorted(tenor_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<15s}  {c} configs")

    # Z-type breakdown
    print(f"\n  Z-score type in TOP 50:")
    ztype_counts = {}
    for sig in top50["signal"]:
        zt = sig.split("|")[1]
        ztype_counts[zt] = ztype_counts.get(zt, 0) + 1
    for zt, c in sorted(ztype_counts.items(), key=lambda x: -x[1]):
        print(f"    {zt:<10s}  {c} configs")

    # ── 7. Visualisations ──────────────────────────────────────────────────
    print(f"\n[7/7] Generating plots ...")

    # --- Plot 1: Sharpe distribution histogram ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["sharpe"], bins=80, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(df["sharpe"].quantile(0.95), color="red", ls="--",
               label=f"95th pct = {df['sharpe'].quantile(0.95):.3f}")
    ax.axvline(df["sharpe"].quantile(0.99), color="darkred", ls="--",
               label=f"99th pct = {df['sharpe'].quantile(0.99):.3f}")
    ax.axvline(0.218, color="green", ls="-", lw=2,
               label="Baseline V1 (0.218)")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Sharpe Ratios — All Configurations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "01_sharpe_distribution.png"), dpi=150)
    plt.close()

    # --- Plot 2: Heatmap tl_calm vs ts_calm (best signal, best tl_stress) ---
    best_sig = df.iloc[0]["signal"]
    sub = df[df["signal"] == best_sig].copy()
    heat_data = sub.groupby(["tl_calm", "ts_calm"])["sharpe"].max().unstack()

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heat_data.values, aspect="auto", cmap="RdYlGn",
                   origin="lower")
    ax.set_xticks(range(len(heat_data.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in heat_data.columns], rotation=45,
                       fontsize=7)
    ax.set_yticks(range(len(heat_data.index)))
    ax.set_yticklabels([f"{x:.2f}" for x in heat_data.index], fontsize=7)
    ax.set_xlabel("ts_calm (LONG threshold)")
    ax.set_ylabel("tl_calm (SHORT threshold)")
    ax.set_title(f"Sharpe Heatmap — {best_sig}\n(best across tl_stress & cooldown)")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "02_heatmap_tl_ts_calm.png"), dpi=150)
    plt.close()

    # --- Plot 3: Best Sharpe by tenor ---
    tenor_best = {}
    for _, r in df.iterrows():
        tenor = r["signal"].split("|")[0]
        if tenor not in tenor_best or r["sharpe"] > tenor_best[tenor]:
            tenor_best[tenor] = r["sharpe"]

    fig, ax = plt.subplots(figsize=(10, 5))
    tenors_sorted = sorted(tenor_best.items(), key=lambda x: -x[1])
    ax.barh([t[0] for t in tenors_sorted],
            [t[1] for t in tenors_sorted],
            color="steelblue", edgecolor="white")
    ax.axvline(0.218, color="green", ls="--", label="Baseline V1")
    ax.set_xlabel("Best Sharpe")
    ax.set_title("Best Sharpe by RR Tenor")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "03_sharpe_by_tenor.png"), dpi=150)
    plt.close()

    # --- Plot 4: Best Sharpe by z-score window ---
    win_best = {}
    for _, r in df.iterrows():
        parts = r["signal"].split("|")
        win = parts[2]  # e.g. "w250"
        if win not in win_best or r["sharpe"] > win_best[win]:
            win_best[win] = r["sharpe"]

    fig, ax = plt.subplots(figsize=(10, 5))
    wins_sorted = sorted(win_best.items(),
                         key=lambda x: int(x[0].replace("w", "")))
    ax.bar([w[0] for w in wins_sorted],
           [w[1] for w in wins_sorted],
           color="coral", edgecolor="white")
    ax.axhline(0.218, color="green", ls="--", label="Baseline V1")
    ax.set_xlabel("Z-Score Window")
    ax.set_ylabel("Best Sharpe")
    ax.set_title("Best Sharpe by Z-Score Rolling Window")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "04_sharpe_by_zscore_window.png"), dpi=150)
    plt.close()

    # --- Plot 5: z_std vs z_move comparison ---
    ztype_sharpe = {"std": [], "move": []}
    for _, r in df.iterrows():
        zt = r["signal"].split("|")[1]
        ztype_sharpe[zt].append(r["sharpe"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([ztype_sharpe["std"], ztype_sharpe["move"]],
               labels=["z_std", "z_move"],
               patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Sharpe Distribution: z_std vs z_move")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "05_zstd_vs_zmove.png"), dpi=150)
    plt.close()

    # --- Plot 6: Top 10 equity curves ---
    top10 = df.head(10)

    # need to rebuild equity for top configs
    fig, ax = plt.subplots(figsize=(14, 6))

    for rank, (_, r) in enumerate(top10.iterrows()):
        sig_key = r["signal"]
        z_series = signals[sig_key]
        common = (
            z_series.dropna().index
            .intersection(regime.index)
            .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
        )
        common = common[common >= start].sort_values()

        z_arr  = z_series.reindex(common).values
        rg_arr = regime_num.reindex(common).values
        rl     = etf_ret["TLT"].reindex(common).fillna(0).values
        rs     = etf_ret["SHV"].reindex(common).fillna(0).values

        eq = fast_bt_equity(z_arr, rg_arr, rl, rs,
                            r["tl_calm"], r["ts_calm"], r["tl_stress"],
                            cooldown=int(r["cooldown"]))

        label = (f"#{rank+1} Sh={r['sharpe']:.3f} "
                 f"{sig_key.split('|')[0]}|"
                 f"tl={r['tl_calm']:.1f}|ts={r['ts_calm']:.1f}")
        ax.plot(common, eq, linewidth=1.0, alpha=0.8, label=label)

    # B&H TLT benchmark
    common_all = etf_ret.dropna(subset=["TLT"]).index
    common_all = common_all[common_all >= start]
    bm = (1 + etf_ret["TLT"].reindex(common_all).fillna(0)).cumprod()
    ax.plot(common_all, bm.values, color="grey", linewidth=1.5,
            alpha=0.5, ls="--", label="B&H TLT")

    ax.set_title("Top 10 Configurations — Equity Curves (overfitted)")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "06_top10_equity_curves.png"), dpi=150)
    plt.close()

    # --- Plot 7: Parameter distribution of top 50 ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for ax, col, color in zip(
        axes.flat,
        ["tl_calm", "ts_calm", "tl_stress", "cooldown"],
        ["steelblue", "coral", "seagreen", "orchid"],
    ):
        ax.hist(top50[col], bins=20, color=color, alpha=0.7, edgecolor="white")
        ax.axvline(top50[col].mean(), color="red", ls="--",
                   label=f"mean={top50[col].mean():.2f}")
        ax.set_title(f"{col}  (top 50)")
        ax.set_xlabel(col)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Parameter Distribution — Top 50 Configs", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "07_top50_param_distribution.png"), dpi=150)
    plt.close()

    # --- Plot 8: Heatmap tl_stress vs tl_calm (best ts_calm) ---
    heat2 = sub.groupby(["tl_calm", "tl_stress"])["sharpe"].max().unstack()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(heat2.values, aspect="auto", cmap="RdYlGn", origin="lower")
    ax.set_xticks(range(len(heat2.columns)))
    ax.set_xticklabels([f"{x:.1f}" for x in heat2.columns], rotation=45,
                       fontsize=8)
    ax.set_yticks(range(len(heat2.index)))
    ax.set_yticklabels([f"{x:.2f}" for x in heat2.index], fontsize=8)
    ax.set_xlabel("tl_stress (STRESS SHORT threshold)")
    ax.set_ylabel("tl_calm (CALM SHORT threshold)")
    ax.set_title(f"Sharpe Heatmap — {best_sig}\ntl_calm vs tl_stress")
    fig.colorbar(im, ax=ax, label="Sharpe")
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "08_heatmap_tl_calm_vs_stress.png"), dpi=150)
    plt.close()

    # --- Plot 9: Sharpe by MOVE median window (z_move only) ---
    mw_best = {}
    for _, r in df.iterrows():
        parts = r["signal"].split("|")
        if len(parts) >= 4 and parts[1] == "move":
            mw = parts[3]
            if mw not in mw_best or r["sharpe"] > mw_best[mw]:
                mw_best[mw] = r["sharpe"]

    if mw_best:
        fig, ax = plt.subplots(figsize=(8, 5))
        mw_sorted = sorted(mw_best.items(),
                           key=lambda x: int(x[0].replace("mw", "")))
        ax.bar([m[0] for m in mw_sorted],
               [m[1] for m in mw_sorted],
               color="teal", edgecolor="white")
        ax.axhline(0.218, color="green", ls="--", label="Baseline V1")
        ax.set_xlabel("MOVE Median Window")
        ax.set_ylabel("Best Sharpe")
        ax.set_title("Best Sharpe by MOVE Median Window (z_move only)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(os.path.join(OPT_DIR, "09_sharpe_by_move_window.png"),
                    dpi=150)
        plt.close()

    # --- Plot 10: Events vs Sharpe scatter ---
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(df["n_events"], df["sharpe"],
                    c=df["mdd"], cmap="RdYlGn_r",
                    alpha=0.3, s=8, edgecolors="none")
    ax.set_xlabel("Number of Events (trades)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Events vs Sharpe  (color = Max Drawdown)")
    fig.colorbar(sc, ax=ax, label="Max Drawdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OPT_DIR, "10_events_vs_sharpe.png"), dpi=150)
    plt.close()

    # ── Wrap up ────────────────────────────────────────────────────────────
    elapsed = time.time() - T0
    n_plots = 10
    print(f"\n  {n_plots} plots saved to {OPT_DIR}/")

    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Total configurations tested: {len(df):,}")
    print(f"  Best Sharpe: {df.iloc[0]['sharpe']:.4f}  "
          f"({df.iloc[0]['signal']})")
    print(f"  Baseline V1 Sharpe: 0.218")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
