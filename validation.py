"""Validation framework: grid search, walk-forward, bootstrap, benchmarks."""

import warnings
import numpy as np
import pandas as pd
from itertools import product

from config import (
    DEFAULT_PARAMS, BACKTEST_START, RISK_FREE_RATE,
    GRID_TL_CALM, GRID_TS_CALM, GRID_TL_STRESS,
    WF_FIRST_OOS_YEAR, WF_LAST_OOS_YEAR, WF_MIN_IS_OBS, WF_MIN_OOS_OBS,
    BOOTSTRAP_N, BOOTSTRAP_BLOCK, BOOTSTRAP_SEED,
    TC_TEST_VALUES, HMM_N_ITER,
    SC_IS_SHARPE_MIN, SC_IS_EXCESS_MIN, SC_WF_EXCESS_MIN,
    SC_WF_POS_YEARS_MIN, SC_DECAY_MAX, SC_PARAM_STD_MAX,
    SC_BOOT_P_MIN, SC_REGIME_CONCORDANCE_MIN, SC_MIN_EVENTS,
)
from backtest import backtest


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1: Grid Search
# ═════════════════════════════════════════════════════════════════════════════

def phase1_grid(D: dict) -> pd.DataFrame:
    """Exhaustive grid search over threshold parameters for both z variants."""
    print("\n" + "=" * 60)
    print("PHASE 1 - GRID SEARCH")
    print("=" * 60)

    tl_c = np.arange(*GRID_TL_CALM)
    ts_c = np.arange(*GRID_TS_CALM)
    tl_s = np.arange(*GRID_TL_STRESS)
    z_keys = ["z_std", "z_move"]

    rows = []
    combos = list(product(z_keys, tl_c, ts_c, tl_s))
    total = len(combos)
    print(f"  Testing {total} configurations...")

    for i, (zk, tlc, tsc, tls) in enumerate(combos):
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{total}")

        p = {
            **DEFAULT_PARAMS,
            "tl_calm": float(tlc),
            "ts_calm": float(tsc),
            "tl_stress": float(tls),
        }
        res = backtest(D[zk], D["regime"], D["etf_ret"], params=p, start=BACKTEST_START)
        m = res["metrics"]
        rows.append({
            "z_key": zk,
            "tl_calm": float(tlc),
            "ts_calm": float(tsc),
            "tl_stress": float(tls),
            "sharpe": m["sharpe"],
            "excess": m["excess"],
            "mdd": m["mdd"],
            "calmar": m["calmar"],
            "sortino": m["sortino"],
            "n_events": m["n_events"],
            "ann_ret": m["ann_ret"],
            "vol": m["vol"],
        })

    grid_df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    print(f"\n  Top 10 configurations by Sharpe:")
    cols = ["z_key", "tl_calm", "ts_calm", "tl_stress", "sharpe", "excess", "mdd", "n_events"]
    print(grid_df[cols].head(10).to_string(index=False))

    return grid_df


# ═════════════════════════════════════════════════════════════════════════════
# Phase 2: Sensitivity Analysis
# ═════════════════════════════════════════════════════════════════════════════

def phase2_sensitivity(D: dict, params: dict, z_key: str = "z_move") -> dict:
    """1D sensitivity sweep for each threshold parameter."""
    print("\n" + "=" * 60)
    print("PHASE 2 - SENSITIVITY ANALYSIS")
    print("=" * 60)

    dims = {
        "tl_calm":   np.arange(-5.0, -1.0, 0.25),
        "ts_calm":   np.arange(1.0, 5.0, 0.25),
        "tl_stress": np.arange(-6.0, -2.0, 0.25),
    }

    results = {}
    for param_name, values in dims.items():
        sharpes = []
        for v in values:
            p = {**params, param_name: float(v)}
            res = backtest(D[z_key], D["regime"], D["etf_ret"], params=p, start=BACKTEST_START)
            sharpes.append(res["metrics"]["sharpe"])
        sharpes = np.array(sharpes)
        local_std = np.std(sharpes)
        verdict = "FLAT (robust)" if local_std < 0.05 else "SENSITIVE"
        print(f"  {param_name:12s}: Sharpe range [{sharpes.min():.3f}, {sharpes.max():.3f}]  "
              f"std={local_std:.4f}  -> {verdict}")
        results[param_name] = {"values": values, "sharpes": sharpes, "std": local_std}

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3: Walk-Forward Validation
# ═════════════════════════════════════════════════════════════════════════════

def phase3_walk_forward(D: dict, params: dict, z_key: str = "z_move") -> pd.DataFrame:
    """Expanding-window IS/OOS walk-forward validation."""
    print("\n" + "=" * 60)
    print("PHASE 3 - WALK-FORWARD VALIDATION")
    print("=" * 60)

    z = D[z_key]
    regime = D["regime"]
    etf_ret = D["etf_ret"]

    tl_c = np.arange(*GRID_TL_CALM)
    ts_c = np.arange(*GRID_TS_CALM)
    tl_s = np.arange(*GRID_TL_STRESS)

    rows = []
    for year in range(WF_FIRST_OOS_YEAR, WF_LAST_OOS_YEAR + 1):
        is_end = f"{year - 1}-12-31"
        oos_start = f"{year}-01-01"
        oos_end = f"{year}-12-31"

        # IS data check
        z_is = z[:is_end].dropna()
        if len(z_is) < WF_MIN_IS_OBS:
            print(f"  {year}: skipped (IS too short: {len(z_is)} obs)")
            continue

        # OOS data check
        z_oos = z[oos_start:oos_end].dropna()
        if len(z_oos) < WF_MIN_OOS_OBS:
            print(f"  {year}: skipped (OOS too short: {len(z_oos)} obs)")
            continue

        # Re-optimize on IS period
        best_sharpe, best_p = -np.inf, None
        for tlc, tsc, tls in product(tl_c, ts_c, tl_s):
            p = {**params, "tl_calm": float(tlc), "ts_calm": float(tsc), "tl_stress": float(tls)}
            res_is = backtest(z[:is_end], regime, etf_ret, params=p, start=BACKTEST_START)
            if res_is["metrics"]["n_events"] >= 2 and res_is["metrics"]["sharpe"] > best_sharpe:
                best_sharpe = res_is["metrics"]["sharpe"]
                best_p = p.copy()

        if best_p is None:
            print(f"  {year}: no valid IS configuration found")
            continue

        # OOS evaluation
        res_oos = backtest(z[oos_start:oos_end], regime, etf_ret,
                           params=best_p, start=oos_start)

        rows.append({
            "year": year,
            "is_sharpe": best_sharpe,
            "oos_sharpe": res_oos["metrics"]["sharpe"],
            "oos_excess": res_oos["metrics"]["excess"],
            "oos_mdd": res_oos["metrics"]["mdd"],
            "tl_calm": best_p["tl_calm"],
            "ts_calm": best_p["ts_calm"],
            "tl_stress": best_p["tl_stress"],
            "n_events_oos": res_oos["metrics"]["n_events"],
        })
        print(f"  {year}: IS Sharpe={best_sharpe:.3f}  "
              f"OOS Sharpe={res_oos['metrics']['sharpe']:.3f}  "
              f"OOS Excess={res_oos['metrics']['excess']:.3f}")

    wf_df = pd.DataFrame(rows)
    if len(wf_df) > 0:
        avg_oos = wf_df["oos_excess"].mean()
        pct_pos = (wf_df["oos_excess"] > 0).mean()
        print(f"\n  Avg OOS Excess: {avg_oos:.4f}")
        print(f"  % Positive OOS Years: {pct_pos:.1%}")

    return wf_df


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: Block Bootstrap
# ═════════════════════════════════════════════════════════════════════════════

def phase4_bootstrap(
    D: dict,
    params: dict,
    z_key: str = "z_move",
    n_boot: int = BOOTSTRAP_N,
    block_size: int = BOOTSTRAP_BLOCK,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Block bootstrap confidence intervals for key metrics."""
    print("\n" + "=" * 60)
    print("PHASE 4 - BLOCK BOOTSTRAP")
    print("=" * 60)

    # Run base backtest to get the net returns
    res = backtest(D[z_key], D["regime"], D["etf_ret"], params=params, start=BACKTEST_START)
    net = res["net_ret"].values
    n = len(net)

    rng = np.random.RandomState(seed)
    boot_sharpe, boot_excess, boot_mdd = [], [], []

    for b in range(n_boot):
        # Block bootstrap: sample random starting points
        n_blocks = n // block_size + 1
        starts = rng.randint(0, n - block_size, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
        boot_ret = net[idx]

        eq = np.cumprod(1 + boot_ret)
        total = eq[-1] - 1
        years = n / 252
        ann = (1 + total) ** (1 / max(years, 0.01)) - 1
        vol = np.std(boot_ret) * np.sqrt(252)
        sh = (ann - RISK_FREE_RATE) / vol if vol > 0 else 0

        dd = eq / np.maximum.accumulate(eq) - 1
        mdd = dd.min()

        # Excess (approx: use base benchmark return)
        excess = ann - res["metrics"]["bm_ann"]

        boot_sharpe.append(sh)
        boot_excess.append(excess)
        boot_mdd.append(mdd)

    boot_sharpe = np.array(boot_sharpe)
    boot_excess = np.array(boot_excess)
    boot_mdd = np.array(boot_mdd)

    ci = lambda arr: (np.percentile(arr, 2.5), np.percentile(arr, 50), np.percentile(arr, 97.5))

    results = {
        "sharpe_ci": ci(boot_sharpe),
        "excess_ci": ci(boot_excess),
        "mdd_ci": ci(boot_mdd),
        "p_sharpe_pos": (boot_sharpe > 0).mean(),
        "p_excess_pos": (boot_excess > 0).mean(),
    }

    print(f"  Sharpe  95% CI: [{results['sharpe_ci'][0]:.3f}, {results['sharpe_ci'][2]:.3f}]  "
          f"median={results['sharpe_ci'][1]:.3f}")
    print(f"  Excess  95% CI: [{results['excess_ci'][0]:.4f}, {results['excess_ci'][2]:.4f}]  "
          f"median={results['excess_ci'][1]:.4f}")
    print(f"  MDD     95% CI: [{results['mdd_ci'][0]:.4f}, {results['mdd_ci'][2]:.4f}]")
    print(f"  P(Sharpe > 0): {results['p_sharpe_pos']:.1%}")
    print(f"  P(Excess > 0): {results['p_excess_pos']:.1%}")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5: Regime Stability
# ═════════════════════════════════════════════════════════════════════════════

def phase5_regime_stability(D: dict) -> pd.DataFrame:
    """Expanding HMM re-estimation to check regime stability."""
    print("\n" + "=" * 60)
    print("PHASE 5 - REGIME STABILITY")
    print("=" * 60)

    from hmmlearn.hmm import GaussianHMM
    from sklearn.preprocessing import StandardScaler

    move = D["move"]
    vix = D["vix"]
    full_regime = D["regime"]

    rows = []
    for year in range(2012, 2026):
        end = f"{year}-12-31"
        mv_sub = move[:end].dropna()
        vx_sub = vix[:end].dropna()

        if len(mv_sub) < WF_MIN_IS_OBS:
            continue

        # Build feature matrix for training period
        train_df = pd.DataFrame({
            "log_move": np.log(mv_sub.clip(lower=1)),
            "log_vix": np.log(vx_sub.reindex(mv_sub.index).clip(lower=1)),
        }).dropna()
        if len(train_df) < WF_MIN_IS_OBS:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df.values)

        # Fit HMM (best of 5 seeds)
        best_model, best_ll = None, -np.inf
        for seed in range(5):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = GaussianHMM(n_components=2, covariance_type="full",
                                    n_iter=200, random_state=seed, tol=1e-5)
                    m.fit(X_train)
                    ll = m.score(X_train)
                    if ll > best_ll:
                        best_ll, best_model = ll, m
            except Exception:
                continue

        if best_model is None:
            continue

        # Predict on training data
        train_states = best_model.predict(X_train)
        mv0 = move.reindex(train_df.index)[train_states == 0].mean()
        mv1 = move.reindex(train_df.index)[train_states == 1].mean()
        lmap = {0: "STRESS", 1: "CALM"} if mv0 > mv1 else {0: "CALM", 1: "STRESS"}

        train_regime = pd.Series([lmap[s] for s in train_states], index=train_df.index)

        # IS concordance with full-sample regime
        overlap = train_regime.index.intersection(full_regime.index)
        concordance = (train_regime.loc[overlap] == full_regime.loc[overlap]).mean() if len(overlap) > 0 else np.nan

        # OOS: predict on next year using the same fitted model + scaler
        oos_start = f"{year + 1}-01-01"
        oos_end = f"{year + 1}-12-31"
        mv_oos = move[oos_start:oos_end].dropna()
        vx_oos = vix[oos_start:oos_end].reindex(mv_oos.index).dropna()
        oos_df = pd.DataFrame({
            "log_move": np.log(mv_oos.reindex(vx_oos.index).clip(lower=1)),
            "log_vix": np.log(vx_oos.clip(lower=1)),
        }).dropna()

        oos_conc = np.nan
        if len(oos_df) > 0:
            X_oos = scaler.transform(oos_df.values)
            oos_states = best_model.predict(X_oos)
            oos_regime = pd.Series([lmap[s] for s in oos_states], index=oos_df.index)
            oos_overlap = oos_regime.index.intersection(full_regime.index)
            if len(oos_overlap) > 0:
                oos_conc = (oos_regime.loc[oos_overlap] == full_regime.loc[oos_overlap]).mean()

        rows.append({
            "train_end": year,
            "is_concordance": concordance,
            "oos_concordance": oos_conc,
            "n_obs": len(mv_sub),
        })
        print(f"  Train->{year}: IS conc={concordance:.3f}  OOS conc={oos_conc:.3f}")

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 6: Transaction Cost Sensitivity
# ═════════════════════════════════════════════════════════════════════════════

def phase6_tc_sensitivity(
    D: dict,
    params: dict,
    z_key: str = "z_move",
    tc_values: list = TC_TEST_VALUES,
) -> pd.DataFrame:
    """Evaluate strategy performance across a range of transaction costs."""
    print("\n" + "=" * 60)
    print("PHASE 6 - TRANSACTION COST SENSITIVITY")
    print("=" * 60)

    rows = []
    for tc in tc_values:
        p = {**params, "tcost_bps": tc}
        res = backtest(D[z_key], D["regime"], D["etf_ret"], params=p, start=BACKTEST_START)
        m = res["metrics"]
        rows.append({
            "tc_bps": tc,
            "sharpe": m["sharpe"],
            "excess": m["excess"],
            "ann_ret": m["ann_ret"],
            "mdd": m["mdd"],
            "n_events": m["n_events"],
        })
        print(f"  TC={tc:3d} bps  Sharpe={m['sharpe']:.3f}  Excess={m['excess']:.4f}")

    tc_df = pd.DataFrame(rows)

    # Breakeven TC (interpolate where Sharpe crosses zero)
    sharpes = tc_df["sharpe"].values
    tcs = tc_df["tc_bps"].values
    be_tc = np.nan
    for i in range(len(sharpes) - 1):
        if sharpes[i] > 0 >= sharpes[i + 1]:
            be_tc = tcs[i] + (0 - sharpes[i]) / (sharpes[i + 1] - sharpes[i]) * (tcs[i + 1] - tcs[i])
            break
    if np.isnan(be_tc) and sharpes[-1] > 0:
        be_tc = float("inf")
    print(f"\n  Breakeven TC: {be_tc:.1f} bps")

    tc_df.attrs["breakeven_tc"] = be_tc
    return tc_df


# ═════════════════════════════════════════════════════════════════════════════
# Phase 7: Benchmark Comparison
# ═════════════════════════════════════════════════════════════════════════════

def phase7_benchmarks(
    D: dict,
    params: dict,
    z_key: str = "z_move",
) -> pd.DataFrame:
    """Compare strategy vs buy-and-hold benchmarks."""
    print("\n" + "=" * 60)
    print("PHASE 7 - BENCHMARK COMPARISON")
    print("=" * 60)

    etf_ret = D["etf_ret"]
    res_strat = backtest(D[z_key], D["regime"], etf_ret, params=params, start=BACKTEST_START)
    strat_ret = res_strat["net_ret"]

    start_dt = strat_ret.index[0]
    end_dt = strat_ret.index[-1]
    idx = strat_ret.index

    benchmarks = {
        "Strategy": strat_ret,
        "B&H TLT": etf_ret["TLT"].reindex(idx).fillna(0),
        "50/50 SHV/TLT": (0.5 * etf_ret["TLT"].reindex(idx).fillna(0)
                           + 0.5 * etf_ret["SHV"].reindex(idx).fillna(0)),
    }
    for etf in ["TLH", "IEF"]:
        if etf in etf_ret.columns:
            s = etf_ret[etf].reindex(idx).fillna(0)
            if s.abs().sum() > 0:
                benchmarks[f"B&H {etf}"] = s

    rows = []
    for name, ret in benchmarks.items():
        eq = (1 + ret).cumprod()
        n = len(ret)
        years = n / 252
        total = eq.iloc[-1] - 1
        ann = (1 + total) ** (1 / max(years, 0.01)) - 1
        vol = ret.std() * np.sqrt(252)
        sh = (ann - RISK_FREE_RATE) / vol if vol > 0 else 0
        dd = eq / eq.cummax() - 1
        mdd = dd.min()
        rows.append({
            "name": name,
            "ann_ret": ann,
            "vol": vol,
            "sharpe": sh,
            "mdd": mdd,
            "total_ret": total,
        })
        print(f"  {name:20s}: Ann={ann:.3f}  Vol={vol:.3f}  Sharpe={sh:.3f}  MDD={mdd:.3f}")

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 8: Summary Scorecard
# ═════════════════════════════════════════════════════════════════════════════

def phase8_summary(
    D: dict,
    params: dict,
    z_key: str,
    grid_df: pd.DataFrame,
    wf_df: pd.DataFrame,
    boot: dict,
    regime_df: pd.DataFrame,
    tc_df: pd.DataFrame,
) -> dict:
    """Produce a validation scorecard and print final report."""
    print("\n" + "=" * 60)
    print("PHASE 8 - VALIDATION SCORECARD")
    print("=" * 60)

    res = backtest(D[z_key], D["regime"], D["etf_ret"], params=params, start=BACKTEST_START)
    m = res["metrics"]

    checks = {}

    # 1. IS performance
    checks["IS_sharpe"] = m["sharpe"] > SC_IS_SHARPE_MIN
    checks["IS_excess"] = m["excess"] > SC_IS_EXCESS_MIN
    print(f"  IS Sharpe > 0    : {'PASS' if checks['IS_sharpe'] else 'FAIL'}  ({m['sharpe']:.3f})")
    print(f"  IS Excess > 1%   : {'PASS' if checks['IS_excess'] else 'FAIL'}  ({m['excess']:.4f})")

    # 2. Walk-forward
    if len(wf_df) > 0:
        avg_oos = wf_df["oos_excess"].mean()
        pct_pos = (wf_df["oos_excess"] > 0).mean()
        checks["WF_excess"] = avg_oos > SC_WF_EXCESS_MIN
        checks["WF_pct_pos"] = pct_pos > SC_WF_POS_YEARS_MIN

        # Decay
        if len(wf_df) > 0 and "is_sharpe" in wf_df.columns:
            avg_is = wf_df["is_sharpe"].mean()
            avg_oos_sh = wf_df["oos_sharpe"].mean()
            decay = avg_is - avg_oos_sh
            checks["WF_decay"] = abs(decay) < SC_DECAY_MAX
            print(f"  WF IS->OOS decay  : {'PASS' if checks['WF_decay'] else 'FAIL'}  ({decay:.3f})")

        # Parameter stability
        param_stds = [wf_df[c].std() for c in ["tl_calm", "ts_calm", "tl_stress"] if c in wf_df.columns]
        max_std = max(param_stds) if param_stds else 0
        checks["WF_param_stab"] = max_std < SC_PARAM_STD_MAX
        print(f"  WF avg OOS exc   : {'PASS' if checks['WF_excess'] else 'FAIL'}  ({avg_oos:.4f})")
        print(f"  WF % pos years   : {'PASS' if checks['WF_pct_pos'] else 'FAIL'}  ({pct_pos:.1%})")
        print(f"  WF param stab    : {'PASS' if checks['WF_param_stab'] else 'FAIL'}  (max std={max_std:.3f})")
    else:
        checks["WF_excess"] = False
        checks["WF_pct_pos"] = False

    # 3. Bootstrap
    checks["Boot_sharpe"] = boot["p_sharpe_pos"] > SC_BOOT_P_MIN
    checks["Boot_excess"] = boot["p_excess_pos"] > SC_BOOT_P_MIN
    print(f"  Boot P(Sh>0)     : {'PASS' if checks['Boot_sharpe'] else 'FAIL'}  ({boot['p_sharpe_pos']:.1%})")
    print(f"  Boot P(Ex>0)     : {'PASS' if checks['Boot_excess'] else 'FAIL'}  ({boot['p_excess_pos']:.1%})")

    # 4. Regime stability
    if len(regime_df) > 0 and "oos_concordance" in regime_df.columns:
        min_oos_conc = regime_df["oos_concordance"].dropna().min()
        checks["Regime_stable"] = min_oos_conc > SC_REGIME_CONCORDANCE_MIN
        print(f"  Regime min OOS   : {'PASS' if checks['Regime_stable'] else 'FAIL'}  ({min_oos_conc:.3f})")
    else:
        checks["Regime_stable"] = False

    # 5. TC robustness
    tc_at_10 = tc_df[tc_df["tc_bps"] == 10]["sharpe"].values
    checks["TC_robust"] = tc_at_10[0] > 0 if len(tc_at_10) > 0 else False
    print(f"  TC robust @10bps : {'PASS' if checks['TC_robust'] else 'FAIL'}")

    # 6. Events
    checks["Events"] = m["n_events"] >= SC_MIN_EVENTS
    print(f"  Events >= 4      : {'PASS' if checks['Events'] else 'FAIL'}  ({m['n_events']})")

    # Overall verdict
    n_pass = sum(checks.values())
    n_total = len(checks)
    if n_pass >= n_total - 1:
        verdict = "VALIDATED"
    elif n_pass >= n_total / 2:
        verdict = "PARTIALLY VALIDATED"
    else:
        verdict = "NOT VALIDATED"

    print(f"\n  Score: {n_pass}/{n_total} checks passed")
    print(f"  Verdict: {verdict}")
    print("=" * 60)

    return {"checks": checks, "verdict": verdict, "n_pass": n_pass, "n_total": n_total}
