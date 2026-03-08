"""Walk-forward head-to-head: tl=-3.0/ts=2.0 vs tl=-3.5/ts=3.0."""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from config import SAVE_DIR, BACKTEST_START, RISK_FREE_RATE, DEFAULT_PARAMS
from data_loader import load_all
from signals import compute_all_signals
from regime import compute_regime
from backtest import backtest

SAVE = os.path.join(SAVE_DIR, "robust")
os.makedirs(SAVE, exist_ok=True)


def wf_single_config(D, z_key, params, label):
    """Expanding-window walk-forward for a single fixed config.

    IS: all data up to Dec 31 of year-1
    OOS: year Y (Jan 1 to Dec 31)
    No re-optimization -- same params every year.
    """
    z = D[z_key]
    regime = D["regime"]
    etf_ret = D["etf_ret"]

    rows = []
    for year in range(2012, 2026):
        oos_start = f"{year}-01-01"
        oos_end = f"{year}-12-31"

        # OOS backtest
        z_oos = z[oos_start:oos_end].dropna()
        if len(z_oos) < 20:
            continue

        res = backtest(z[oos_start:oos_end], regime, etf_ret,
                       params=params, start=oos_start)
        m = res["metrics"]

        # OOS TLT benchmark
        tlt_oos = etf_ret["TLT"][oos_start:oos_end].dropna()
        tlt_ret_yr = (1 + tlt_oos).prod() - 1

        # Count events in this year
        ev = res["events"]
        n_ev = len(ev) if len(ev) > 0 else 0
        ev_types = ""
        if n_ev > 0:
            longs = (ev["signal"] == "LONG").sum()
            shorts = (ev["signal"] == "SHORT").sum()
            ev_types = f"{longs}L/{shorts}S"

        rows.append({
            "year": year,
            "strat_ret": m["ann_ret"] if m["n_days"] > 200 else m["total_ret"],
            "tlt_ret": float(tlt_ret_yr),
            "excess": m["excess"],
            "sharpe": m["sharpe"],
            "mdd": m["mdd"],
            "n_events": n_ev,
            "ev_types": ev_types,
            "vol": m["vol"],
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    D = load_all()
    D = compute_all_signals(D)
    D = compute_regime(D)

    # Two configs to compare
    configs = {
        "ACTIVE (tl=-3.0, ts=2.0)": {
            "z_key": "z_move",
            "params": {**DEFAULT_PARAMS, "tl_calm": -3.0, "ts_calm": 2.0, "tl_stress": -4.0},
        },
        "CONSERVATIVE (tl=-3.5, ts=3.0)": {
            "z_key": "z_move",
            "params": {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0},
        },
    }

    wf_results = {}
    for label, cfg in configs.items():
        print(f"\n{'=' * 70}")
        print(f"WALK-FORWARD: {label}")
        print(f"  z_key={cfg['z_key']}  tl_calm={cfg['params']['tl_calm']}  "
              f"ts_calm={cfg['params']['ts_calm']}  tl_stress={cfg['params']['tl_stress']}")
        print("=" * 70)

        wf = wf_single_config(D, cfg["z_key"], cfg["params"], label)
        wf_results[label] = wf

        print(f"\n  {'Year':<6} {'Strat':>8} {'TLT':>8} {'Excess':>8} {'Sharpe':>8} "
              f"{'MDD':>8} {'Vol':>8} {'Events':>8} {'Types':>8}")
        print("  " + "-" * 75)
        for _, r in wf.iterrows():
            print(f"  {r['year']:<6} {r['strat_ret']:>8.3f} {r['tlt_ret']:>8.3f} "
                  f"{r['excess']:>+8.4f} {r['sharpe']:>8.3f} {r['mdd']:>8.3f} "
                  f"{r['vol']:>8.3f} {r['n_events']:>8} {r['ev_types']:>8}")

        # Summary stats
        print(f"\n  --- SUMMARY ---")
        print(f"  Avg annual excess:    {wf['excess'].mean():+.4f}")
        print(f"  Median annual excess: {wf['excess'].median():+.4f}")
        print(f"  % years positive:     {(wf['excess'] > 0).mean():.0%} ({(wf['excess'] > 0).sum()}/{len(wf)})")
        print(f"  Worst year excess:    {wf['excess'].min():+.4f} ({int(wf.loc[wf['excess'].idxmin(), 'year'])})")
        print(f"  Best year excess:     {wf['excess'].max():+.4f} ({int(wf.loc[wf['excess'].idxmax(), 'year'])})")
        print(f"  Avg events/year:      {wf['n_events'].mean():.1f}")
        print(f"  Total events:         {wf['n_events'].sum()}")
        cum_excess = (1 + wf["strat_ret"]).prod() / (1 + wf["tlt_ret"]).prod() - 1
        print(f"  Cumulative excess:    {cum_excess:+.4f}")

    # =========================================================================
    # HEAD-TO-HEAD COMPARISON TABLE
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    labels = list(wf_results.keys())
    wf_a = wf_results[labels[0]]
    wf_b = wf_results[labels[1]]

    merged = wf_a[["year", "excess", "strat_ret", "n_events"]].merge(
        wf_b[["year", "excess", "strat_ret", "n_events"]],
        on="year", suffixes=("_active", "_conserv"))

    print(f"\n  {'Year':<6} {'Active Exc':>12} {'Conserv Exc':>12} {'Diff':>10} {'Winner':>12}")
    print("  " + "-" * 55)
    for _, r in merged.iterrows():
        diff = r["excess_active"] - r["excess_conserv"]
        winner = "ACTIVE" if diff > 0.001 else "CONSERV" if diff < -0.001 else "TIE"
        print(f"  {int(r['year']):<6} {r['excess_active']:>+12.4f} {r['excess_conserv']:>+12.4f} "
              f"{diff:>+10.4f} {winner:>12}")

    active_wins = (merged["excess_active"] > merged["excess_conserv"] + 0.001).sum()
    conserv_wins = (merged["excess_conserv"] > merged["excess_active"] + 0.001).sum()
    ties = len(merged) - active_wins - conserv_wins
    print(f"\n  ACTIVE wins: {active_wins}  |  CONSERVATIVE wins: {conserv_wins}  |  Ties: {ties}")
    print(f"  Active avg excess:  {wf_a['excess'].mean():+.4f}")
    print(f"  Conserv avg excess: {wf_b['excess'].mean():+.4f}")

    # =========================================================================
    # FULL-SAMPLE BACKTEST COMPARISON
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("FULL-SAMPLE BACKTEST COMPARISON")
    print("=" * 70)

    full_results = {}
    for label, cfg in configs.items():
        res = backtest(D[cfg["z_key"]], D["regime"], D["etf_ret"],
                       params=cfg["params"], start=BACKTEST_START)
        full_results[label] = res
        m = res["metrics"]
        print(f"\n  {label}:")
        print(f"    Ann Return:  {m['ann_ret']:.4f}")
        print(f"    Volatility:  {m['vol']:.4f}")
        print(f"    Sharpe:      {m['sharpe']:.4f}")
        print(f"    Sortino:     {m['sortino']:.4f}")
        print(f"    Max DD:      {m['mdd']:.4f}")
        print(f"    Calmar:      {m['calmar']:.4f}")
        print(f"    Excess:      {m['excess']:.4f}")
        print(f"    N Events:    {m['n_events']}")

        # Trade list
        ev = res["events"]
        if len(ev) > 0:
            print(f"\n    Trades ({len(ev)}):")
            for _, e in ev.iterrows():
                print(f"      {e['date'].strftime('%Y-%m-%d')} {e['signal']:<6} "
                      f"z={e['z']:>6.2f}  {e['regime']}")

            # Time allocation
            w = res["weight"]
            in_tlt = (w > 0.5).mean()
            print(f"\n    Time in TLT: {in_tlt:.1%}  |  Time in SHV: {1-in_tlt:.1%}")

    # =========================================================================
    # PLOTS
    # =========================================================================
    print(f"\nGenerating plots...")

    # --- Plot 1: OOS Excess by Year (side-by-side bars) ---
    fig, ax = plt.subplots(figsize=(14, 7))
    years = merged["year"].values
    x = np.arange(len(years))
    w = 0.35
    bars1 = ax.bar(x - w/2, merged["excess_active"], w, label=labels[0],
                   color="#1f77b4", alpha=0.8)
    bars2 = ax.bar(x + w/2, merged["excess_conserv"], w, label=labels[1],
                   color="#d62728", alpha=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(years.astype(int))
    ax.set_xlabel("Year")
    ax.set_ylabel("OOS Excess Return")
    ax.set_title("Walk-Forward OOS Excess: Active vs Conservative")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        if abs(h) > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{h:+.3f}", ha="center", va="bottom" if h > 0 else "top", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if abs(h) > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2, h,
                    f"{h:+.3f}", ha="center", va="bottom" if h > 0 else "top", fontsize=7)

    path = os.path.join(SAVE, "wf_excess_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # --- Plot 2: Cumulative OOS equity ---
    fig, ax = plt.subplots(figsize=(14, 7))
    for label, wf in wf_results.items():
        cum = (1 + wf["strat_ret"]).cumprod()
        color = "#1f77b4" if "ACTIVE" in label else "#d62728"
        ax.plot(wf["year"], cum, "o-", color=color, lw=2, markersize=6, label=label)

    # TLT cumulative
    tlt_cum = (1 + wf_a["tlt_ret"]).cumprod()
    ax.plot(wf_a["year"], tlt_cum, "o--", color="#555555", lw=1.5, markersize=4,
            label="B&H TLT", alpha=0.7)

    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Return (1 = start)")
    ax.set_title("Walk-Forward Cumulative OOS Performance")
    ax.legend()
    ax.grid(True, alpha=0.2)

    path = os.path.join(SAVE, "wf_cumulative_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # --- Plot 3: Full-sample equity curves ---
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = {"ACTIVE": "#1f77b4", "CONSERVATIVE": "#d62728"}

    for label, res in full_results.items():
        eq = (1 + res["net_ret"]).cumprod() * 1e7
        c = colors["ACTIVE"] if "ACTIVE" in label else colors["CONSERVATIVE"]
        ax.plot(eq.index, eq.values, color=c, lw=2.5, label=label, alpha=0.9)

        # Mark trades
        ev = res["events"]
        if len(ev) > 0:
            for _, e in ev.iterrows():
                dt = e["date"]
                if dt in eq.index:
                    marker = "^" if e["signal"] == "LONG" else "v"
                    mc = "green" if e["signal"] == "LONG" else "red"
                    ax.scatter(dt, eq.loc[dt], marker=marker, color=mc, s=60,
                              zorder=5, edgecolors="black", linewidths=0.5)

    # B&H TLT
    idx = full_results[labels[0]]["net_ret"].index
    bm = (1 + D["etf_ret"]["TLT"].reindex(idx).fillna(0)).cumprod() * 1e7
    ax.plot(bm.index, bm.values, color="#555555", lw=1.5, ls="--", alpha=0.6,
            label="B&H TLT")

    # Regime shading
    regime = D["regime"].reindex(idx).fillna("CALM")
    stress = regime == "STRESS"
    ylim = ax.get_ylim()
    ax.fill_between(idx, ylim[0], ylim[1], where=stress,
                    alpha=0.08, color="red", label="STRESS")
    ax.set_ylim(ylim)

    ax.set_yscale("log")
    ax.set_ylabel("NAV ($)")
    ax.set_title("Full-Sample Equity: Active vs Conservative (triangles = trades)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    path = os.path.join(SAVE, "wf_equity_fullsample.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # --- Plot 4: Events timeline comparison ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    for ax, (label, res) in zip(axes, full_results.items()):
        ev = res["events"]
        w = res["weight"]
        color = "#1f77b4" if "ACTIVE" in label else "#d62728"

        ax.fill_between(w.index, w.values, 0.5, where=w > 0.5,
                       color="green", alpha=0.3, label="In TLT")
        ax.fill_between(w.index, w.values, 0.5, where=w <= 0.5,
                       color="red", alpha=0.3, label="In SHV")
        ax.axhline(0.5, color="black", lw=0.5, ls=":")
        ax.set_ylabel("Weight (TLT)")
        ax.set_title(f"{label} - Allocation Over Time")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_ylim(-0.1, 1.1)

        if len(ev) > 0:
            for _, e in ev.iterrows():
                mc = "green" if e["signal"] == "LONG" else "red"
                y = 1.05 if e["signal"] == "LONG" else -0.05
                ax.scatter(e["date"], y, marker="|", color=mc, s=100, zorder=5)

    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path = os.path.join(SAVE, "wf_allocation_timeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
