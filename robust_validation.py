"""Robust Validation Framework - Phases A, B, C.

Phase A: Signal validation (before strategy)
Phase B: Z-score construction optimization
Phase C: Strategy & threshold variants
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from config import SAVE_DIR, BACKTEST_START, RISK_FREE_RATE, DEFAULT_PARAMS
from data_loader import load_all
from signals import compute_zscore_standard, compute_zscore_move_adj, compute_all_signals
from regime import compute_regime
from backtest import backtest

SAVE = os.path.join(SAVE_DIR, "robust")
os.makedirs(SAVE, exist_ok=True)

def _save_fig(fig, name):
    path = os.path.join(SAVE, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {path}")


# =============================================================================
# PHASE A: SIGNAL VALIDATION
# =============================================================================

def phase_a1_quintile_by_regime(D):
    """Quintile analysis of forward TLT returns, split by CALM / STRESS."""
    print("\n" + "=" * 70)
    print("PHASE A1 - QUINTILE ANALYSIS BY REGIME")
    print("=" * 70)

    z = D["z_std"]
    regime = D["regime"]
    tlt_ret = D["etf_ret"]["TLT"]

    horizons = [5, 21, 63]
    fwd = {}
    for h in horizons:
        fwd[h] = tlt_ret.rolling(h).sum().shift(-h)

    common = z.dropna().index.intersection(regime.dropna().index)
    df = pd.DataFrame({"z": z, "regime": regime}).reindex(common).dropna()
    for h in horizons:
        df[f"fwd_{h}d"] = fwd[h].reindex(common)

    results = {}
    for reg_name in ["CALM", "STRESS"]:
        sub = df[df["regime"] == reg_name].dropna()
        if len(sub) < 100:
            print(f"  {reg_name}: too few observations ({len(sub)})")
            continue

        sub["q"] = pd.qcut(sub["z"], 5, labels=[1, 2, 3, 4, 5])
        table = sub.groupby("q")[[f"fwd_{h}d" for h in horizons]].mean() * (252 / np.array(horizons))
        table.columns = [f"{h}d (ann.)" for h in horizons]

        print(f"\n  {reg_name} regime ({len(sub)} obs):")
        print(table.to_string())

        # Monotonicity: Spearman rho of quintile rank vs mean return
        for h in horizons:
            col = f"fwd_{h}d"
            means = sub.groupby("q")[col].mean()
            rho, p = stats.spearmanr(range(5), means.values)
            mono = "MONOTONIC" if abs(rho) >= 0.9 else "PARTIAL" if abs(rho) >= 0.7 else "WEAK"
            print(f"    {h}d: rho={rho:.3f} p={p:.3f} -> {mono}")

        results[reg_name] = table

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, reg_name in zip(axes, ["CALM", "STRESS"]):
        if reg_name not in results:
            continue
        tbl = results[reg_name]
        tbl.plot(kind="bar", ax=ax, width=0.7)
        ax.set_title(f"Forward Returns by Z-Quintile ({reg_name})")
        ax.set_xlabel("Z-Score Quintile (1=low, 5=high)")
        ax.set_ylabel("Annualized Return")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    _save_fig(fig, "A1_quintile_by_regime")

    return results


def phase_a2_temporal_stability(D):
    """Rolling quintile spread (Q1-Q5) over time to check stability."""
    print("\n" + "=" * 70)
    print("PHASE A2 - TEMPORAL STABILITY OF SIGNAL")
    print("=" * 70)

    z = D["z_std"]
    tlt_ret = D["etf_ret"]["TLT"]
    fwd_21d = tlt_ret.rolling(21).sum().shift(-21)

    common = z.dropna().index.intersection(fwd_21d.dropna().index)
    df = pd.DataFrame({"z": z, "fwd": fwd_21d}).reindex(common).dropna()

    # Rolling 504-day (2yr) correlation
    roll_corr = df["z"].rolling(504, min_periods=252).corr(df["fwd"])

    # Rolling 504-day quintile spread (Q1 mean - Q5 mean)
    def rolling_spread(window=504):
        spreads = []
        dates = []
        for i in range(window, len(df), 63):  # step every quarter
            chunk = df.iloc[i - window:i]
            try:
                chunk_q = pd.qcut(chunk["z"], 5, labels=[1, 2, 3, 4, 5])
                q1_mean = chunk.loc[chunk_q == 1, "fwd"].mean()
                q5_mean = chunk.loc[chunk_q == 5, "fwd"].mean()
                spreads.append(q1_mean - q5_mean)
                dates.append(chunk.index[-1])
            except Exception:
                continue
        return pd.Series(spreads, index=dates)

    spread_504 = rolling_spread(504)

    # Year-by-year correlation
    years = sorted(df.index.year.unique())
    print(f"\n  {'Year':<6} {'Corr(z, fwd21d)':<18} {'N obs':<8} {'Verdict'}")
    print("  " + "-" * 50)
    yr_results = []
    for y in years:
        sub = df[df.index.year == y]
        if len(sub) < 50:
            continue
        c = sub["z"].corr(sub["fwd"])
        verdict = "WORKS" if c < -0.05 else "FLAT" if abs(c) < 0.05 else "INVERTED"
        print(f"  {y:<6} {c:<18.4f} {len(sub):<8} {verdict}")
        yr_results.append({"year": y, "corr": c, "n": len(sub), "verdict": verdict})

    yr_df = pd.DataFrame(yr_results)
    n_works = (yr_df["verdict"] == "WORKS").sum()
    n_total = len(yr_df)
    print(f"\n  Signal works in {n_works}/{n_total} years ({n_works/n_total:.0%})")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.plot(roll_corr.index, roll_corr.values, color="navy", lw=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(-0.05, color="red", ls="--", alpha=0.5)
    ax.set_ylabel("Rolling 504d Corr(z, fwd_21d)")
    ax.set_title("Temporal Stability of RR Signal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(spread_504.index, spread_504.values,
           width=50, color=["green" if v > 0 else "red" for v in spread_504.values], alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Q1-Q5 Spread (21d fwd)")
    ax.set_title("Rolling 2Y Quintile Spread (Q1 minus Q5)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, "A2_temporal_stability")

    return yr_df


def phase_a3_decay_curve(D):
    """Signal decay: correlation of z_std with forward TLT returns at various horizons."""
    print("\n" + "=" * 70)
    print("PHASE A3 - SIGNAL DECAY CURVE")
    print("=" * 70)

    z = D["z_std"]
    tlt_ret = D["etf_ret"]["TLT"]

    horizons = [1, 2, 3, 5, 10, 15, 21, 42, 63, 126, 252]
    correlations = []
    t_stats = []

    print(f"\n  {'Horizon':<10} {'Corr':<10} {'t-stat':<10} {'p-value':<10} {'Strength'}")
    print("  " + "-" * 55)

    for h in horizons:
        fwd = tlt_ret.rolling(h).sum().shift(-h)
        common = z.dropna().index.intersection(fwd.dropna().index)
        zz = z.reindex(common).dropna()
        ff = fwd.reindex(common).dropna()
        idx = zz.index.intersection(ff.index)

        c, p = stats.pearsonr(zz.loc[idx], ff.loc[idx])
        n = len(idx)
        t = c * np.sqrt(n - 2) / np.sqrt(1 - c**2) if abs(c) < 1 else 0

        strength = "STRONG" if abs(c) > 0.10 else "MODERATE" if abs(c) > 0.05 else "WEAK"
        print(f"  {h:<10} {c:<10.4f} {t:<10.2f} {p:<10.4f} {strength}")
        correlations.append(c)
        t_stats.append(t)

    # Peak signal horizon
    peak_idx = np.argmin(correlations)
    peak_h = horizons[peak_idx]
    print(f"\n  Peak signal at {peak_h}d horizon (corr={correlations[peak_idx]:.4f})")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(horizons, correlations, "o-", color="navy", markersize=6)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(-0.05, color="red", ls="--", alpha=0.5, label="-0.05 threshold")
    ax.axvline(peak_h, color="green", ls="--", alpha=0.5, label=f"Peak: {peak_h}d")
    ax.set_xlabel("Forward Return Horizon (days)")
    ax.set_ylabel("Correlation with z_std")
    ax.set_title("Signal Decay Curve: Corr(z_std, TLT forward return)")
    ax.set_xscale("log")
    ax.set_xticks(horizons)
    ax.set_xticklabels(horizons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "A3_decay_curve")

    return pd.DataFrame({"horizon": horizons, "corr": correlations, "t_stat": t_stats})


# =============================================================================
# PHASE B: Z-SCORE CONSTRUCTION OPTIMIZATION
# =============================================================================

def phase_b4_wf_window_search(D):
    """Walk-forward window search: for each OOS year, find best window in IS."""
    print("\n" + "=" * 70)
    print("PHASE B4 - WALK-FORWARD WINDOW SEARCH")
    print("=" * 70)

    windows = [63, 126, 189, 250, 378, 504]
    rr_1m = D["rr_1m"]
    regime = D["regime"]
    etf_ret = D["etf_ret"]
    params = {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}

    rows = []
    for year in range(2014, 2026):
        is_end = f"{year - 1}-12-31"
        oos_start = f"{year}-01-01"
        oos_end = f"{year}-12-31"

        # IS: find best window
        best_w, best_sh = None, -np.inf
        for w in windows:
            z = compute_zscore_standard(rr_1m, window=w)
            z_is = z[:is_end].dropna()
            if len(z_is) < 504:
                continue
            res = backtest(z[:is_end], regime, etf_ret, params=params, start=BACKTEST_START)
            if res["metrics"]["sharpe"] > best_sh:
                best_sh = res["metrics"]["sharpe"]
                best_w = w

        if best_w is None:
            continue

        # OOS: evaluate with best window
        z_best = compute_zscore_standard(rr_1m, window=best_w)
        res_oos = backtest(z_best[oos_start:oos_end], regime, etf_ret,
                           params=params, start=oos_start)

        # Also evaluate fixed 250d for comparison
        z_250 = compute_zscore_standard(rr_1m, window=250)
        res_250 = backtest(z_250[oos_start:oos_end], regime, etf_ret,
                           params=params, start=oos_start)

        rows.append({
            "year": year,
            "best_window": best_w,
            "is_sharpe": best_sh,
            "oos_sharpe_adaptive": res_oos["metrics"]["sharpe"],
            "oos_excess_adaptive": res_oos["metrics"]["excess"],
            "oos_sharpe_fixed250": res_250["metrics"]["sharpe"],
            "oos_excess_fixed250": res_250["metrics"]["excess"],
        })
        print(f"  {year}: best_w={best_w:>3}d  "
              f"OOS adaptive={res_oos['metrics']['excess']:+.4f}  "
              f"OOS fixed250={res_250['metrics']['excess']:+.4f}")

    wf_df = pd.DataFrame(rows)
    if len(wf_df) > 0:
        print(f"\n  Window selection frequency:")
        print(f"  {wf_df['best_window'].value_counts().to_dict()}")
        print(f"  Avg OOS excess (adaptive): {wf_df['oos_excess_adaptive'].mean():.4f}")
        print(f"  Avg OOS excess (fixed250): {wf_df['oos_excess_fixed250'].mean():.4f}")
        adaptive_better = (wf_df['oos_excess_adaptive'] > wf_df['oos_excess_fixed250']).mean()
        print(f"  Adaptive wins: {adaptive_better:.0%} of years")

    return wf_df


def phase_b5_expanding_vs_rolling(D):
    """Compare expanding window z-score vs rolling window."""
    print("\n" + "=" * 70)
    print("PHASE B5 - EXPANDING VS ROLLING WINDOW")
    print("=" * 70)

    rr_1m = D["rr_1m"]
    regime = D["regime"]
    etf_ret = D["etf_ret"]
    params = {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}

    # Rolling 250
    z_roll = compute_zscore_standard(rr_1m, window=250)

    # Expanding (min 250 obs)
    mu_exp = rr_1m.expanding(min_periods=250).mean()
    sig_exp = rr_1m.expanding(min_periods=250).std()
    z_exp = (rr_1m - mu_exp) / sig_exp.replace(0, np.nan)
    z_exp.name = "z_expanding"

    # Compare
    variants = {"Rolling 250d": z_roll, "Expanding (min 250d)": z_exp}
    print(f"\n  {'Variant':<25} {'Sharpe':>8} {'Excess':>8} {'Events':>8} {'MDD':>8}")
    print("  " + "-" * 60)

    for name, z in variants.items():
        res = backtest(z, regime, etf_ret, params=params, start=BACKTEST_START)
        m = res["metrics"]
        print(f"  {name:<25} {m['sharpe']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")

    # Year-by-year comparison
    rows = []
    for year in range(2010, 2026):
        oos = f"{year}-01-01"
        oos_e = f"{year}-12-31"
        r_roll = backtest(z_roll[oos:oos_e], regime, etf_ret, params=params, start=oos)
        r_exp = backtest(z_exp[oos:oos_e], regime, etf_ret, params=params, start=oos)
        rows.append({
            "year": year,
            "roll_excess": r_roll["metrics"]["excess"],
            "exp_excess": r_exp["metrics"]["excess"],
        })

    cmp_df = pd.DataFrame(rows)
    exp_wins = (cmp_df["exp_excess"] > cmp_df["roll_excess"]).mean()
    print(f"\n  Expanding wins {exp_wins:.0%} of years")
    print(f"  Avg rolling excess:    {cmp_df['roll_excess'].mean():.4f}")
    print(f"  Avg expanding excess:  {cmp_df['exp_excess'].mean():.4f}")

    # Plot: z_roll vs z_exp scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    common = z_roll.dropna().index.intersection(z_exp.dropna().index)
    ax.scatter(z_roll.loc[common], z_exp.loc[common], s=1, alpha=0.3)
    ax.plot([-6, 6], [-6, 6], "r--", alpha=0.5)
    ax.set_xlabel("Z (Rolling 250d)")
    ax.set_ylabel("Z (Expanding)")
    ax.set_title(f"Rolling vs Expanding Z-Score (corr={z_roll.loc[common].corr(z_exp.loc[common]):.3f})")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "B5_expanding_vs_rolling")

    return cmp_df


def phase_b6_rank_transform(D):
    """Rolling rank (percentile) transform as robust alternative."""
    print("\n" + "=" * 70)
    print("PHASE B6 - RANK-BASED NORMALIZATION")
    print("=" * 70)

    rr_1m = D["rr_1m"]
    regime = D["regime"]
    etf_ret = D["etf_ret"]

    # Rolling percentile rank (0 to 1)
    windows = [126, 250, 504]

    print(f"\n  {'Variant':<35} {'Sharpe':>8} {'Excess':>8} {'Events':>8} {'MDD':>8}")
    print("  " + "-" * 70)

    for w in windows:
        z_rank = rr_1m.rolling(w, min_periods=63).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
        )
        # Map percentile to z-like scale: 0->-3, 0.5->0, 1->+3
        z_rank_scaled = pd.Series(
            stats.norm.ppf(z_rank.clip(0.001, 0.999).values),
            index=z_rank.index, name=f"z_rank_{w}",
        )

        params_rank = {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}
        res = backtest(z_rank_scaled, regime, etf_ret, params=params_rank, start=BACKTEST_START)
        m = res["metrics"]
        print(f"  Rank {w}d (probit-mapped)      {m['sharpe']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")

    # Compare with standard z_std
    z_std = D["z_std"]
    res_std = backtest(z_std, regime, etf_ret,
                       params={**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0},
                       start=BACKTEST_START)
    m = res_std["metrics"]
    print(f"  Z-score standard 250d            {m['sharpe']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")

    # Also test raw percentile thresholds (10th/90th)
    for w in [250]:
        z_rank = rr_1m.rolling(w, min_periods=63).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
        )
        # Use percentile thresholds directly
        params_pct = {**DEFAULT_PARAMS, "tl_calm": -1.28, "ts_calm": 1.28,  # ~10th/90th
                      "tl_stress": -1.65}  # ~5th percentile
        z_pct = pd.Series(stats.norm.ppf(z_rank.clip(0.001, 0.999).values),
                          index=z_rank.index)
        res = backtest(z_pct, regime, etf_ret, params=params_pct, start=BACKTEST_START)
        m = res["metrics"]
        print(f"  Rank {w}d (10/90 thresholds)    {m['sharpe']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")


# =============================================================================
# PHASE C: STRATEGY & THRESHOLD VARIANTS
# =============================================================================

def phase_c7_adaptive_thresholds(D):
    """Rolling percentile-based thresholds instead of fixed z-score levels."""
    print("\n" + "=" * 70)
    print("PHASE C7 - ADAPTIVE THRESHOLDS")
    print("=" * 70)

    z = D["z_std"]
    regime = D["regime"]
    etf_ret = D["etf_ret"]

    # Adaptive: use rolling quantiles of z itself as thresholds
    # Instead of fixed tl=-3.5, use rolling 5th percentile
    # Instead of fixed ts=3.0, use rolling 95th percentile

    quantile_pairs = [
        (0.05, 0.95, "5/95"),
        (0.10, 0.90, "10/90"),
        (0.15, 0.85, "15/85"),
    ]
    window = 504  # 2 years of context

    print(f"\n  {'Variant':<35} {'Sharpe':>8} {'Excess':>8} {'Events':>8} {'MDD':>8}")
    print("  " + "-" * 70)

    # Fixed baseline
    params_fixed = {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}
    res_fixed = backtest(z, regime, etf_ret, params=params_fixed, start=BACKTEST_START)
    m = res_fixed["metrics"]
    print(f"  Fixed (tl=-3.5, ts=3.0)          {m['sharpe']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")

    for q_lo, q_hi, label in quantile_pairs:
        roll_lo = z.rolling(window, min_periods=250).quantile(q_lo)
        roll_hi = z.rolling(window, min_periods=250).quantile(q_hi)

        # Build adaptive signal: compare z against rolling quantiles
        common = z.dropna().index.intersection(regime.dropna().index) \
                   .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index) \
                   .intersection(roll_lo.dropna().index)
        common = common[common >= BACKTEST_START].sort_values()

        weight = pd.Series(np.nan, index=common)
        events = []
        last_event = pd.Timestamp("1900-01-01")
        cooldown = 5

        for dt in common:
            zv = z.loc[dt]
            reg = regime.loc[dt]
            lo = roll_lo.loc[dt]
            hi = roll_hi.loc[dt]
            days_since = (dt - last_event).days

            signal = None
            if reg == "CALM":
                if zv <= lo:
                    signal = "SHORT"
                elif zv >= hi:
                    signal = "LONG"
            else:  # STRESS
                stress_lo = roll_lo.loc[dt] * 1.3  # more extreme in stress
                if zv <= stress_lo:
                    signal = "SHORT"

            if signal and days_since < cooldown:
                signal = None

            if signal == "LONG":
                weight.loc[dt] = 1.0
                events.append(dt)
                last_event = dt
            elif signal == "SHORT":
                weight.loc[dt] = 0.0
                events.append(dt)
                last_event = dt

        weight = weight.ffill().fillna(0.5)
        w_del = weight.shift(2).fillna(0.5)

        r_long = etf_ret["TLT"].reindex(common).fillna(0)
        r_short = etf_ret["SHV"].reindex(common).fillna(0)
        net = w_del * r_long + (1 - w_del) * r_short
        rebal = w_del.diff().abs().fillna(0)
        net = net - rebal * (5 / 10_000)

        eq = (1 + net).cumprod()
        yrs = len(net) / 252
        total = eq.iloc[-1] - 1
        ann = (1 + total) ** (1 / max(yrs, 0.01)) - 1
        vol = net.std() * np.sqrt(252)
        sh = (ann - RISK_FREE_RATE) / vol if vol > 0 else 0
        dd = eq / eq.cummax() - 1
        mdd = dd.min()

        # Benchmark excess
        bm = etf_ret["TLT"].reindex(common).fillna(0)
        bm_eq = (1 + bm).cumprod()
        bm_ann = (1 + bm_eq.iloc[-1] - 1) ** (1 / max(yrs, 0.01)) - 1
        excess = ann - bm_ann

        print(f"  Adaptive {label} ({window}d window)     {sh:>8.3f} {excess:>8.4f} {len(events):>8} {mdd:>8.4f}")


def phase_c8_short_only(D):
    """Test SHORT-only strategy: only protect from downside, never go LONG on signal."""
    print("\n" + "=" * 70)
    print("PHASE C8 - SHORT-ONLY STRATEGY")
    print("=" * 70)

    z_std = D["z_std"]
    regime = D["regime"]
    etf_ret = D["etf_ret"]

    print(f"\n  {'Variant':<40} {'Sharpe':>8} {'Excess':>8} {'Events':>8} {'MDD':>8}")
    print("  " + "-" * 75)

    # Bidirectional baseline
    params_bi = {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}
    res_bi = backtest(z_std, regime, etf_ret, params=params_bi, start=BACKTEST_START)
    m = res_bi["metrics"]
    print(f"  Bidirectional (tl=-3.5, ts=3.0)       {m['sharpe']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")

    # SHORT-only: ts_calm=99 disables LONG signals
    # Start at w=1 (100% TLT), only switch to SHV on SHORT signal
    # Re-enter TLT after cooldown period expires (auto-return)
    for holding in [21, 42, 63, 126]:
        common = z_std.dropna().index.intersection(regime.dropna().index) \
                      .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
        common = common[common >= BACKTEST_START].sort_values()

        weight = pd.Series(np.nan, index=common)
        events = []
        last_short_date = pd.Timestamp("1900-01-01")

        for dt in common:
            zv = z_std.loc[dt]
            reg = regime.loc[dt]
            days_since_short = (dt - last_short_date).days

            # Auto-return to LONG after holding period
            if days_since_short == holding:
                weight.loc[dt] = 1.0

            # SHORT signal
            tl = -3.5 if reg == "CALM" else -4.0
            if zv <= tl and days_since_short >= 5:
                weight.loc[dt] = 0.0
                events.append(dt)
                last_short_date = dt

        weight = weight.ffill().fillna(1.0)  # default: 100% TLT
        w_del = weight.shift(2).fillna(1.0)

        r_long = etf_ret["TLT"].reindex(common).fillna(0)
        r_short = etf_ret["SHV"].reindex(common).fillna(0)
        net = w_del * r_long + (1 - w_del) * r_short
        rebal = w_del.diff().abs().fillna(0)
        net = net - rebal * (5 / 10_000)

        eq = (1 + net).cumprod()
        yrs = len(net) / 252
        total = eq.iloc[-1] - 1
        ann = (1 + total) ** (1 / max(yrs, 0.01)) - 1
        vol = net.std() * np.sqrt(252)
        sh = (ann - RISK_FREE_RATE) / vol if vol > 0 else 0
        dd = eq / eq.cummax() - 1
        mdd = dd.min()
        bm = etf_ret["TLT"].reindex(common).fillna(0)
        bm_eq = (1 + bm).cumprod()
        bm_ann = (1 + bm_eq.iloc[-1] - 1) ** (1 / max(yrs, 0.01)) - 1
        excess = ann - bm_ann

        print(f"  SHORT-only (hold={holding:>3}d, then TLT)    {sh:>8.3f} {excess:>8.4f} {len(events):>8} {mdd:>8.4f}")


def phase_c9_fixed_holding(D):
    """Fixed holding period strategy: enter on signal, exit after N days."""
    print("\n" + "=" * 70)
    print("PHASE C9 - FIXED HOLDING PERIOD (BIDIRECTIONAL)")
    print("=" * 70)

    z_std = D["z_std"]
    regime = D["regime"]
    etf_ret = D["etf_ret"]

    print(f"\n  {'Variant':<45} {'Sharpe':>8} {'Excess':>8} {'Events':>8} {'MDD':>8}")
    print("  " + "-" * 80)

    # Standard (signal-to-signal)
    params_base = {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}
    res_base = backtest(z_std, regime, etf_ret, params=params_base, start=BACKTEST_START)
    m = res_base["metrics"]
    print(f"  Signal-to-signal (original)                {m['sharpe']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")

    for holding in [5, 10, 21, 42, 63]:
        common = z_std.dropna().index.intersection(regime.dropna().index) \
                      .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
        common = common[common >= BACKTEST_START].sort_values()

        weight = pd.Series(np.nan, index=common)
        events = []
        last_event = pd.Timestamp("1900-01-01")
        last_signal_type = None

        for dt in common:
            zv = z_std.loc[dt]
            reg = regime.loc[dt]
            days_since = (dt - last_event).days

            # Auto-return to neutral after holding period
            if last_signal_type is not None and days_since == holding:
                weight.loc[dt] = 0.5  # neutral

            # New signals (only if not in holding)
            if days_since >= holding or last_signal_type is None:
                tl = -3.5 if reg == "CALM" else -4.0
                ts = 3.0 if reg == "CALM" else 99

                if zv <= tl:
                    weight.loc[dt] = 0.0
                    events.append({"date": dt, "signal": "SHORT"})
                    last_event = dt
                    last_signal_type = "SHORT"
                elif zv >= ts:
                    weight.loc[dt] = 1.0
                    events.append({"date": dt, "signal": "LONG"})
                    last_event = dt
                    last_signal_type = "LONG"

        weight = weight.ffill().fillna(0.5)
        w_del = weight.shift(2).fillna(0.5)

        r_long = etf_ret["TLT"].reindex(common).fillna(0)
        r_short = etf_ret["SHV"].reindex(common).fillna(0)
        net = w_del * r_long + (1 - w_del) * r_short
        rebal = w_del.diff().abs().fillna(0)
        net = net - rebal * (5 / 10_000)

        eq = (1 + net).cumprod()
        yrs = len(net) / 252
        total = eq.iloc[-1] - 1
        ann = (1 + total) ** (1 / max(yrs, 0.01)) - 1
        vol = net.std() * np.sqrt(252)
        sh = (ann - RISK_FREE_RATE) / vol if vol > 0 else 0
        dd = eq / eq.cummax() - 1
        mdd = dd.min()
        bm = etf_ret["TLT"].reindex(common).fillna(0)
        bm_eq = (1 + bm).cumprod()
        bm_ann = (1 + bm_eq.iloc[-1] - 1) ** (1 / max(yrs, 0.01)) - 1
        excess = ann - bm_ann

        print(f"  Fixed hold={holding:>3}d, then neutral (50/50)    {sh:>8.3f} {excess:>8.4f} {len(events):>8} {mdd:>8.4f}")


# =============================================================================
# SUMMARY COMPARISON
# =============================================================================

def final_summary(D):
    """Run the best variants from each phase and compare."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON - BEST VARIANTS")
    print("=" * 70)

    z_std = D["z_std"]
    z_move = D["z_move"]
    regime = D["regime"]
    etf_ret = D["etf_ret"]
    rr_1m = D["rr_1m"]

    # Expanding z-score
    mu_exp = rr_1m.expanding(min_periods=250).mean()
    sig_exp = rr_1m.expanding(min_periods=250).std()
    z_exp = (rr_1m - mu_exp) / sig_exp.replace(0, np.nan)

    # Rank-based z-score (250d)
    z_rank = rr_1m.rolling(250, min_periods=63).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False
    )
    z_rank_probit = pd.Series(stats.norm.ppf(z_rank.clip(0.001, 0.999).values),
                              index=z_rank.index)

    variants = {
        "z_std 250d (original)": (z_std, {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}),
        "z_move 250d (MOVE-adj)": (z_move, {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}),
        "z_expanding (min 250d)": (z_exp, {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}),
        "z_rank 250d (probit)": (z_rank_probit, {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}),
        "z_rank 250d (10/90 thr)": (z_rank_probit, {**DEFAULT_PARAMS, "tl_calm": -1.28, "ts_calm": 1.28, "tl_stress": -1.65}),
    }

    print(f"\n  {'Variant':<30} {'Sharpe':>8} {'Sortino':>8} {'Excess':>8} {'Events':>8} {'MDD':>8}")
    print("  " + "-" * 80)

    rows = []
    for name, (z, params) in variants.items():
        res = backtest(z, regime, etf_ret, params=params, start=BACKTEST_START)
        m = res["metrics"]
        print(f"  {name:<30} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['excess']:>8.4f} {m['n_events']:>8} {m['mdd']:>8.4f}")
        rows.append({"name": name, **m})

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def run_robust_validation():
    """Execute all phases A, B, C."""
    import time
    t0 = time.time()

    print("#" * 70)
    print("# ROBUST VALIDATION FRAMEWORK - Phases A, B, C")
    print("#" * 70)

    print("\nLoading data...")
    D = load_all()
    D = compute_all_signals(D)
    D = compute_regime(D)

    # === PHASE A ===
    print("\n" + "+" * 70)
    print("+ PHASE A: SIGNAL VALIDATION")
    print("+" * 70)
    a1 = phase_a1_quintile_by_regime(D)
    a2 = phase_a2_temporal_stability(D)
    a3 = phase_a3_decay_curve(D)

    # === PHASE B ===
    print("\n" + "+" * 70)
    print("+ PHASE B: Z-SCORE CONSTRUCTION")
    print("+" * 70)
    b4 = phase_b4_wf_window_search(D)
    b5 = phase_b5_expanding_vs_rolling(D)
    b6 = phase_b6_rank_transform(D)

    # === PHASE C ===
    print("\n" + "+" * 70)
    print("+ PHASE C: STRATEGY VARIANTS")
    print("+" * 70)
    c7 = phase_c7_adaptive_thresholds(D)
    c8 = phase_c8_short_only(D)
    c9 = phase_c9_fixed_holding(D)

    # === FINAL ===
    summary = final_summary(D)

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"Done. Total time: {elapsed:.1f}s")
    print(f"Plots saved to: {SAVE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_robust_validation()
