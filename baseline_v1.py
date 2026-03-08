"""
baseline_v1.py - Snapshot of the current best strategy configuration.

This script preserves the baseline RR Duration Strategy as of Feb 2024.
Run it anytime to reproduce the reference results.

Best config: z_move, tl_calm=-3.5, ts_calm=3.0, tl_stress=-4.0
Expected: Sharpe ~0.22, 19 events, 9/11 scorecard
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_all
from signals import compute_all_signals
from regime import compute_regime
from backtest import backtest
from config import SAVE_DIR
import os

BASELINE_DIR = os.path.join(SAVE_DIR, "baseline_v1")
os.makedirs(BASELINE_DIR, exist_ok=True)

# ── Frozen best parameters ─────────────────────────────────────────────────
BEST_PARAMS = {
    "tl_calm": -3.5,
    "ts_calm": 3.0,
    "tl_stress": -4.0,
    "ts_stress": 99,
    "allow_long_stress": False,
    "long_etf": "TLT",
    "short_etf": "SHV",
    "cooldown": 5,
    "delay": 1,
    "tcost_bps": 5,
    "w_long": 1.0,
    "w_short": 0.0,
    "w_initial": 0.5,
}


def main():
    t0 = time.time()
    print("=" * 60)
    print("BASELINE V1 - RR Duration Strategy (frozen config)")
    print("=" * 60)

    # Load
    print("\n[1] Loading data...")
    D = load_all()

    # Signals
    print("\n[2] Computing signals...")
    D = compute_all_signals(D)

    # Regime
    print("\n[3] Fitting regime...")
    D = compute_regime(D)

    # Backtest
    print("\n[4] Running baseline backtest...")
    result = backtest(
        z=D["z_move"],
        regime=D["regime"],
        etf_ret=D["etf_ret"],
        params=BEST_PARAMS,
    )

    # Print results
    m = result["metrics"]
    print(f"\n{'='*60}")
    print(f"BASELINE V1 RESULTS")
    print(f"{'='*60}")
    print(f"  Config: z_move | tl_calm={BEST_PARAMS['tl_calm']} | "
          f"ts_calm={BEST_PARAMS['ts_calm']} | tl_stress={BEST_PARAMS['tl_stress']}")
    print(f"  Ann Return : {m['ann_ret']:.4f}")
    print(f"  Volatility : {m['vol']:.4f}")
    print(f"  Sharpe     : {m['sharpe']:.4f}")
    print(f"  Sortino    : {m['sortino']:.4f}")
    print(f"  Max DD     : {m['mdd']:.4f}")
    print(f"  Calmar     : {m['calmar']:.4f}")
    print(f"  Excess     : {m['excess']:.4f}")
    print(f"  Events     : {m['n_events']}")

    # Yearly
    print(f"\n  Yearly Breakdown:")
    yearly = result["yearly"]
    if len(yearly) > 0:
        for _, row in yearly.iterrows():
            print(f"    {int(row['year'])}  strat={row['strat_ret']:+.4f}  "
                  f"bm={row['bm_ret']:+.4f}  excess={row['excess']:+.4f}")

    # Events
    if len(result["events"]) > 0:
        print(f"\n  Trade Events:")
        for _, ev in result["events"].iterrows():
            print(f"    {ev['date'].strftime('%Y-%m-%d')}  {ev['signal']:5s}  "
                  f"z={ev['z']:.2f}  regime={ev['regime']}")

    # Plot equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    result["equity"].plot(ax=ax, label="Strategy", color="steelblue", linewidth=1.5)

    # Benchmark (B&H TLT)
    common = result["equity"].index
    bm = (1 + D["etf_ret"]["TLT"].reindex(common).fillna(0)).cumprod()
    bm.plot(ax=ax, label="B&H TLT", color="grey", alpha=0.6)

    ax.set_title("Baseline V1 - Equity Curve", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylabel("Growth of $1")
    fig.tight_layout()
    fig.savefig(os.path.join(BASELINE_DIR, "baseline_v1_equity.png"), dpi=150)
    plt.close()
    print(f"\n  Plot saved: {BASELINE_DIR}/baseline_v1_equity.png")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([{**BEST_PARAMS, **m}])
    metrics_df.to_csv(os.path.join(BASELINE_DIR, "baseline_v1_metrics.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Baseline V1 completed in {elapsed:.1f}s")
    print(f"{'='*60}")

    return result


if __name__ == "__main__":
    main()
