"""Deep grid search with extended ranges + filter analysis."""

import sys, os, warnings
import numpy as np
import pandas as pd
from itertools import product
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from config import SAVE_DIR, BACKTEST_START, RISK_FREE_RATE, DEFAULT_PARAMS
from data_loader import load_all
from signals import compute_all_signals, compute_zscore_standard, compute_zscore_move_adj
from regime import compute_regime
from backtest import backtest

SAVE = os.path.join(SAVE_DIR, "robust")
os.makedirs(SAVE, exist_ok=True)


def build_equity(net_ret):
    return (1 + net_ret).cumprod() * 1e7


def main():
    print("Loading data...")
    D = load_all()
    D = compute_all_signals(D)
    D = compute_regime(D)

    regime = D["regime"]
    etf_ret = D["etf_ret"]
    rr_1m = D["rr_1m"]

    # =================================================================
    # 1. EXTENDED GRID: tl from -4.5 to +4.5 step 0.5 for both sides
    # =================================================================
    print("\n" + "=" * 70)
    print("1. EXTENDED GRID SEARCH (tl/ts from -4.5 to 4.5)")
    print("=" * 70)

    # For SHORT threshold (tl): test from -4.5 to -0.5
    # For LONG threshold (ts): test from 0.5 to 4.5
    # For STRESS SHORT (tl_stress): test from -5.0 to -1.0
    tl_range = np.arange(-4.5, 0.0, 0.5)   # -4.5, -4.0, ..., -0.5
    ts_range = np.arange(0.5, 5.0, 0.5)    # 0.5, 1.0, ..., 4.5
    tl_stress_range = np.arange(-5.0, -0.5, 0.5)

    z_keys = ["z_std", "z_move"]

    rows = []
    combos = list(product(z_keys, tl_range, ts_range, tl_stress_range))
    total = len(combos)
    print(f"  Testing {total} configurations...")

    for i, (zk, tlc, tsc, tls) in enumerate(combos):
        if (i + 1) % 500 == 0:
            print(f"  ... {i + 1}/{total}")
        # Skip invalid: SHORT threshold must be < LONG threshold
        if tlc >= tsc:
            continue

        p = {**DEFAULT_PARAMS, "tl_calm": float(tlc), "ts_calm": float(tsc),
             "tl_stress": float(tls)}
        res = backtest(D[zk], regime, etf_ret, params=p, start=BACKTEST_START)
        m = res["metrics"]

        rows.append({
            "z_key": zk, "tl_calm": float(tlc), "ts_calm": float(tsc),
            "tl_stress": float(tls),
            "sharpe": m["sharpe"], "excess": m["excess"], "mdd": m["mdd"],
            "ann_ret": m["ann_ret"], "vol": m["vol"],
            "n_events": m["n_events"], "sortino": m["sortino"],
            "calmar": m["calmar"],
        })

    grid = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    print(f"\n  Total valid configs: {len(grid)}")
    print(f"\n  Top 20 by Sharpe:")
    cols = ["z_key", "tl_calm", "ts_calm", "tl_stress", "sharpe", "excess",
            "mdd", "ann_ret", "n_events"]
    print(grid[cols].head(20).to_string(index=False))

    # =================================================================
    # 2. FILTER: Only configs with >= 10 events (active trading)
    # =================================================================
    print("\n" + "=" * 70)
    print("2. TOP CONFIGS WITH >= 10 EVENTS (active trading)")
    print("=" * 70)

    active = grid[grid["n_events"] >= 10].sort_values("sharpe", ascending=False)
    print(f"  {len(active)} configs with >= 10 events")
    print(f"\n  Top 20:")
    print(active[cols].head(20).to_string(index=False))

    # =================================================================
    # 3. FILTER: >= 20 events (frequent trading)
    # =================================================================
    print("\n" + "=" * 70)
    print("3. TOP CONFIGS WITH >= 20 EVENTS (frequent trading)")
    print("=" * 70)

    frequent = grid[grid["n_events"] >= 20].sort_values("sharpe", ascending=False)
    print(f"  {len(frequent)} configs with >= 20 events")
    print(f"\n  Top 20:")
    print(frequent[cols].head(20).to_string(index=False))

    # =================================================================
    # 4. FILTER: >= 30 events AND Sharpe > 0
    # =================================================================
    print("\n" + "=" * 70)
    print("4. TOP CONFIGS WITH >= 30 EVENTS AND Sharpe > 0")
    print("=" * 70)

    very_active = grid[(grid["n_events"] >= 30) & (grid["sharpe"] > 0)].sort_values(
        "sharpe", ascending=False)
    print(f"  {len(very_active)} configs")
    if len(very_active) > 0:
        print(f"\n  Top 20:")
        print(very_active[cols].head(20).to_string(index=False))
    else:
        print("  None found!")

    # =================================================================
    # 5. BEST SHARPE vs BEST EXCESS vs BEST CALMAR vs MOST EVENTS
    # =================================================================
    print("\n" + "=" * 70)
    print("5. BEST BY DIFFERENT CRITERIA")
    print("=" * 70)

    for criterion, ascending in [("sharpe", False), ("excess", False),
                                  ("calmar", False), ("n_events", False),
                                  ("mdd", True)]:
        top = grid.sort_values(criterion, ascending=ascending).head(5)
        print(f"\n  Best by {criterion}:")
        extra = cols + ["calmar", "sortino"]
        print(top[[c for c in extra if c in top.columns]].to_string(index=False))

    # =================================================================
    # 6. HEATMAP: Sharpe as function of tl_calm x ts_calm (z_move, best tl_stress)
    # =================================================================
    print("\n" + "=" * 70)
    print("6. SHARPE HEATMAPS")
    print("=" * 70)

    for zk in ["z_std", "z_move"]:
        sub = grid[grid["z_key"] == zk]
        # For each (tl, ts) pair, take best tl_stress
        best_stress = sub.loc[sub.groupby(["tl_calm", "ts_calm"])["sharpe"].idxmax()]
        pivot = best_stress.pivot_table(index="tl_calm", columns="ts_calm",
                                         values="sharpe", aggfunc="max")

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       origin="lower", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index])
        ax.set_xlabel("ts_calm (LONG threshold)")
        ax.set_ylabel("tl_calm (SHORT threshold)")
        ax.set_title(f"Sharpe Heatmap - {zk} (best tl_stress per cell)")
        plt.colorbar(im, ax=ax, label="Sharpe")

        # Annotate
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > 0.3 else "black")

        path = os.path.join(SAVE, f"heatmap_sharpe_{zk}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {path}")

        # Events heatmap
        pivot_ev = best_stress.pivot_table(index="tl_calm", columns="ts_calm",
                                            values="n_events", aggfunc="max")
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(pivot_ev.values, cmap="YlOrRd", aspect="auto",
                       origin="lower", vmin=0, vmax=100)
        ax.set_xticks(range(len(pivot_ev.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot_ev.columns])
        ax.set_yticks(range(len(pivot_ev.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot_ev.index])
        ax.set_xlabel("ts_calm (LONG threshold)")
        ax.set_ylabel("tl_calm (SHORT threshold)")
        ax.set_title(f"N Events Heatmap - {zk}")
        plt.colorbar(im, ax=ax, label="N Events")

        for i in range(len(pivot_ev.index)):
            for j in range(len(pivot_ev.columns)):
                v = pivot_ev.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7)

        path = os.path.join(SAVE, f"heatmap_events_{zk}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {path}")

    # =================================================================
    # 7. EQUITY CURVES: Top 5 by Sharpe vs Top 5 by Events (>= 20)
    # =================================================================
    print("\n" + "=" * 70)
    print("7. EQUITY CURVES - TOP CONFIGS")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(18, 10))

    # Benchmarks
    idx_ref = None
    palette_top = ["#d62728", "#e74c3c", "#c0392b", "#ff6b6b", "#ff9999"]
    palette_active = ["#1f77b4", "#2980b9", "#3498db", "#5dade2", "#85c1e9"]

    top5_sharpe = grid.head(5)
    top5_active = grid[grid["n_events"] >= 20].sort_values("sharpe", ascending=False).head(5)

    # Plot top5 by Sharpe
    for i, (_, row) in enumerate(top5_sharpe.iterrows()):
        p = {**DEFAULT_PARAMS, "tl_calm": row["tl_calm"], "ts_calm": row["ts_calm"],
             "tl_stress": row["tl_stress"]}
        res = backtest(D[row["z_key"]], regime, etf_ret, params=p, start=BACKTEST_START)
        eq = build_equity(res["net_ret"])
        if idx_ref is None:
            idx_ref = eq.index
        label = (f"Top Sharpe #{i+1}: {row['z_key']} "
                 f"tl={row['tl_calm']:.1f}/ts={row['ts_calm']:.1f}/tls={row['tl_stress']:.1f} "
                 f"(Sh={row['sharpe']:.3f}, ev={int(row['n_events'])})")
        ax.plot(eq.index, eq.values, color=palette_top[i], lw=2.0 if i == 0 else 1.2,
                alpha=1.0 if i == 0 else 0.7, label=label)

    # Plot top5 by Sharpe with >= 20 events
    for i, (_, row) in enumerate(top5_active.iterrows()):
        p = {**DEFAULT_PARAMS, "tl_calm": row["tl_calm"], "ts_calm": row["ts_calm"],
             "tl_stress": row["tl_stress"]}
        res = backtest(D[row["z_key"]], regime, etf_ret, params=p, start=BACKTEST_START)
        eq = build_equity(res["net_ret"])
        label = (f"Top Active #{i+1}: {row['z_key']} "
                 f"tl={row['tl_calm']:.1f}/ts={row['ts_calm']:.1f}/tls={row['tl_stress']:.1f} "
                 f"(Sh={row['sharpe']:.3f}, ev={int(row['n_events'])})")
        ax.plot(eq.index, eq.values, color=palette_active[i], lw=2.0 if i == 0 else 1.2,
                alpha=1.0 if i == 0 else 0.7, label=label, ls="--")

    # B&H TLT
    bm = build_equity(etf_ret["TLT"].reindex(idx_ref).fillna(0))
    ax.plot(bm.index, bm.values, color="#555555", lw=1.5, ls=":", alpha=0.6, label="B&H TLT")

    ax.set_yscale("log")
    ax.set_ylabel("NAV ($)")
    ax.set_title("Top 5 by Sharpe (red) vs Top 5 Active >=20 events (blue)", fontsize=13,
                 fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    path = os.path.join(SAVE, "equity_top_vs_active.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =================================================================
    # 8. SHARPE vs N_EVENTS scatter
    # =================================================================
    print("\n" + "=" * 70)
    print("8. SHARPE vs N_EVENTS TRADE-OFF")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, zk in zip(axes, ["z_std", "z_move"]):
        sub = grid[grid["z_key"] == zk]
        scatter = ax.scatter(sub["n_events"], sub["sharpe"], c=sub["excess"],
                            cmap="RdYlGn", s=8, alpha=0.5, vmin=-0.05, vmax=0.05)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("N Events")
        ax.set_ylabel("Sharpe")
        ax.set_title(f"{zk}: Sharpe vs N Events")
        ax.grid(True, alpha=0.2)
        plt.colorbar(scatter, ax=ax, label="Excess Return")

    fig.tight_layout()
    path = os.path.join(SAVE, "sharpe_vs_events.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =================================================================
    # 9. TRADE ANALYSIS for top config
    # =================================================================
    print("\n" + "=" * 70)
    print("9. TRADE ANALYSIS - TOP CONFIG")
    print("=" * 70)

    top = grid.iloc[0]
    p = {**DEFAULT_PARAMS, "tl_calm": top["tl_calm"], "ts_calm": top["ts_calm"],
         "tl_stress": top["tl_stress"]}
    res = backtest(D[top["z_key"]], regime, etf_ret, params=p, start=BACKTEST_START)

    events = res["events"]
    if len(events) > 0:
        print(f"\n  Config: {top['z_key']} tl={top['tl_calm']} ts={top['ts_calm']} "
              f"tls={top['tl_stress']}")
        print(f"  Total events: {len(events)}")
        print(f"\n  All trades:")
        print(f"  {'Date':<12} {'Signal':<8} {'Z-Score':>8} {'Regime':<8}")
        print("  " + "-" * 40)
        for _, ev in events.iterrows():
            print(f"  {ev['date'].strftime('%Y-%m-%d'):<12} {ev['signal']:<8} "
                  f"{ev['z']:>8.2f} {ev['regime']:<8}")

        # How long between trades
        if len(events) > 1:
            gaps = events["date"].diff().dt.days.dropna()
            print(f"\n  Gap between trades: mean={gaps.mean():.0f}d  "
                  f"median={gaps.median():.0f}d  min={gaps.min():.0f}d  max={gaps.max():.0f}d")

        # Equity periods: how long in each state
        w = res["weight"]
        in_tlt = (w > 0.5).sum()
        in_shv = (w <= 0.5).sum()
        total = len(w)
        print(f"\n  Time in TLT: {in_tlt/total:.1%} ({in_tlt} days)")
        print(f"  Time in SHV: {in_shv/total:.1%} ({in_shv} days)")

    # =================================================================
    # 10. SAME FOR BEST ACTIVE CONFIG (>= 20 events)
    # =================================================================
    print("\n" + "=" * 70)
    print("10. TRADE ANALYSIS - BEST ACTIVE CONFIG (>= 20 events)")
    print("=" * 70)

    if len(top5_active) > 0:
        top_a = top5_active.iloc[0]
        p = {**DEFAULT_PARAMS, "tl_calm": top_a["tl_calm"], "ts_calm": top_a["ts_calm"],
             "tl_stress": top_a["tl_stress"]}
        res_a = backtest(D[top_a["z_key"]], regime, etf_ret, params=p, start=BACKTEST_START)

        events_a = res_a["events"]
        if len(events_a) > 0:
            print(f"\n  Config: {top_a['z_key']} tl={top_a['tl_calm']} ts={top_a['ts_calm']} "
                  f"tls={top_a['tl_stress']}")
            print(f"  Sharpe={top_a['sharpe']:.3f}  Excess={top_a['excess']:.4f}  "
                  f"Events={int(top_a['n_events'])}")
            print(f"\n  All trades:")
            print(f"  {'Date':<12} {'Signal':<8} {'Z-Score':>8} {'Regime':<8}")
            print("  " + "-" * 40)
            for _, ev in events_a.iterrows():
                print(f"  {ev['date'].strftime('%Y-%m-%d'):<12} {ev['signal']:<8} "
                      f"{ev['z']:>8.2f} {ev['regime']:<8}")

            if len(events_a) > 1:
                gaps = events_a["date"].diff().dt.days.dropna()
                print(f"\n  Gap between trades: mean={gaps.mean():.0f}d  "
                      f"median={gaps.median():.0f}d  min={gaps.min():.0f}d  max={gaps.max():.0f}d")

            w = res_a["weight"]
            in_tlt = (w > 0.5).sum()
            in_shv = (w <= 0.5).sum()
            total_d = len(w)
            print(f"\n  Time in TLT: {in_tlt/total_d:.1%} ({in_tlt} days)")
            print(f"  Time in SHV: {in_shv/total_d:.1%} ({in_shv} days)")

    print("\nDone.")


if __name__ == "__main__":
    main()
