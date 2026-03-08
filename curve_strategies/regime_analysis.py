"""
regime_analysis.py - Combined YC+HMM regime analysis and performance attribution.

Combines yield curve regime (BULL_STEEP, BULL_FLAT, BEAR_STEEP, BEAR_FLAT)
with HMM regime (CALM, STRESS) into 10 combined states. Builds the
regime-strategy performance matrix that feeds the meta-system.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE

ANN = 252


# ══════════════════════════════════════════════════════════════════════════
# YC REGIME CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════

def compute_yc_regime(yields: pd.DataFrame, short_col: str = "US_2Y",
                      long_col: str = "US_10Y", window: int = 21) -> pd.Series:
    """Classify each day into bull/bear steep/flat yield curve regime.

    Reuses pattern from yc_regime_strategy.py:71.

    Args:
        yields: yield curve DataFrame with columns like US_2Y, US_10Y, etc.
        short_col: short-end yield column
        long_col: long-end yield column
        window: lookback for change computation

    Returns:
        pd.Series of regime labels: BULL_STEEP, BULL_FLAT, BEAR_STEEP, BEAR_FLAT, UNDEFINED
    """
    level = (yields[short_col] + yields[long_col]) / 2
    spread = yields[long_col] - yields[short_col]

    dl = level.diff(window)
    ds = spread.diff(window)

    r = pd.Series("UNDEFINED", index=yields.index)
    r[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    r[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    r[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    r[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    return r


# ══════════════════════════════════════════════════════════════════════════
# COMBINED REGIME
# ══════════════════════════════════════════════════════════════════════════

def build_combined_regime(yc_regime: pd.Series, hmm_regime: pd.Series) -> pd.Series:
    """Combine YC regime and HMM regime into a single label.

    Result format: "BULL_STEEP/CALM", "BEAR_FLAT/STRESS", etc.
    Total possible states: 5 YC x 2 HMM = 10.

    Args:
        yc_regime: Series of YC regime labels
        hmm_regime: Series of HMM regime labels (CALM/STRESS)

    Returns:
        pd.Series of combined regime labels
    """
    common = yc_regime.index.intersection(hmm_regime.index)
    yc = yc_regime.reindex(common).fillna("UNDEFINED")
    hmm = hmm_regime.reindex(common).fillna("CALM")
    combined = yc.astype(str) + "/" + hmm.astype(str)
    combined.name = "combined_regime"
    return combined


# ══════════════════════════════════════════════════════════════════════════
# REGIME-STRATEGY PERFORMANCE METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_regime_metrics(daily_ret: pd.Series, regime: pd.Series,
                           min_obs: int = 20) -> pd.DataFrame:
    """Compute strategy metrics for each regime state.

    Args:
        daily_ret: daily strategy returns
        regime: regime labels (same index as daily_ret)
        min_obs: minimum observations per regime to compute metrics

    Returns:
        DataFrame with one row per regime: sharpe, mean_ret, hit_rate, n_obs
    """
    common = daily_ret.index.intersection(regime.index)
    dr = daily_ret.reindex(common)
    reg = regime.reindex(common)

    rows = []
    for state in sorted(reg.unique()):
        mask = reg == state
        r = dr[mask]
        n = len(r)
        if n < min_obs:
            rows.append({"regime": state, "sharpe": np.nan,
                         "mean_ret_ann": np.nan, "hit_rate": np.nan, "n_obs": n})
            continue

        rf_d = RISK_FREE_RATE / ANN
        exc = r.values - rf_d
        sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)
        mean_ret = np.mean(r.values) * ANN
        hit_rate = np.mean(r.values > 0)

        rows.append({
            "regime": state,
            "sharpe": sharpe,
            "mean_ret_ann": mean_ret,
            "hit_rate": hit_rate,
            "n_obs": n,
        })

    return pd.DataFrame(rows)


def build_regime_performance_matrix(strategy_returns: dict,
                                    combined_regime: pd.Series,
                                    min_obs: int = 20) -> pd.DataFrame:
    """Build the regime-strategy performance matrix.

    Rows = strategies, Columns = regimes. Values = Sharpe ratios.
    This is the main input for the meta-system.

    Args:
        strategy_returns: dict of {strategy_name: pd.Series of daily returns}
        combined_regime: pd.Series of combined regime labels
        min_obs: minimum observations per cell

    Returns:
        DataFrame with Sharpe for each (strategy, regime) pair
    """
    regimes = sorted(combined_regime.unique())
    rows = []
    for name, daily_ret in strategy_returns.items():
        metrics = compute_regime_metrics(daily_ret, combined_regime, min_obs)
        row = {"strategy": name}
        for _, m in metrics.iterrows():
            row[m["regime"]] = m["sharpe"]
        rows.append(row)

    matrix = pd.DataFrame(rows).set_index("strategy")
    # Reorder columns to a consistent order
    ordered_cols = [c for c in regimes if c in matrix.columns]
    matrix = matrix[ordered_cols]
    return matrix


# ══════════════════════════════════════════════════════════════════════════
# REGIME TRANSITIONS
# ══════════════════════════════════════════════════════════════════════════

def compute_transition_matrix(regime: pd.Series) -> pd.DataFrame:
    """Compute regime transition probability matrix.

    Args:
        regime: pd.Series of regime labels

    Returns:
        DataFrame transition matrix (rows = from, cols = to), values = probability
    """
    states = sorted(regime.unique())
    r = regime.values
    n = len(r)

    counts = pd.DataFrame(0, index=states, columns=states, dtype=float)
    for i in range(n - 1):
        counts.loc[r[i], r[i + 1]] += 1

    # Normalize rows to probabilities
    row_sums = counts.sum(axis=1)
    transition = counts.div(row_sums, axis=0).fillna(0)
    return transition


def compute_post_transition_performance(daily_ret: pd.Series,
                                        regime: pd.Series,
                                        horizons: list = None) -> pd.DataFrame:
    """Compute average strategy return after regime transitions.

    Args:
        daily_ret: daily returns
        regime: regime labels
        horizons: list of forward-looking horizons in days (default [5, 10, 21])

    Returns:
        DataFrame with columns: from_regime, to_regime, horizon, mean_ret, n_transitions
    """
    if horizons is None:
        horizons = [5, 10, 21]

    common = daily_ret.index.intersection(regime.index)
    dr = daily_ret.reindex(common)
    reg = regime.reindex(common)

    r_vals = reg.values
    dr_vals = dr.values
    n = len(r_vals)

    # Find transition points
    transitions = []
    for i in range(1, n):
        if r_vals[i] != r_vals[i - 1]:
            transitions.append((i, r_vals[i - 1], r_vals[i]))

    rows = []
    for horizon in horizons:
        for idx, from_r, to_r in transitions:
            end = min(idx + horizon, n)
            if end - idx < horizon * 0.5:
                continue
            fwd_ret = np.sum(dr_vals[idx:end])
            rows.append({
                "from_regime": from_r,
                "to_regime": to_r,
                "horizon": horizon,
                "fwd_ret": fwd_ret,
            })

    if not rows:
        return pd.DataFrame(columns=["from_regime", "to_regime", "horizon",
                                     "mean_ret", "n_transitions"])

    df = pd.DataFrame(rows)
    agg = df.groupby(["from_regime", "to_regime", "horizon"]).agg(
        mean_ret=("fwd_ret", "mean"),
        n_transitions=("fwd_ret", "count"),
    ).reset_index()
    return agg


# ══════════════════════════════════════════════════════════════════════════
# SLOPE SIGNALS
# ══════════════════════════════════════════════════════════════════════════

# Yield segments for slope analysis
SLOPE_SEGMENTS = {
    "2s5s":  ("US_2Y",  "US_5Y"),
    "2s10s": ("US_2Y",  "US_10Y"),
    "5s10s": ("US_5Y",  "US_10Y"),
    "10s30s": ("US_10Y", "US_30Y"),
    "2s30s": ("US_2Y",  "US_30Y"),
}


def compute_slope_signals(yields: pd.DataFrame,
                          segments: dict = None,
                          z_window: int = 252,
                          mom_window: int = 63) -> pd.DataFrame:
    """Compute z-score of level and momentum for multiple yield curve segments.

    Args:
        yields: yield curve DataFrame
        segments: dict of {name: (short_col, long_col)}
        z_window: rolling window for z-score
        mom_window: momentum lookback

    Returns:
        DataFrame with columns: {seg}_spread, {seg}_z_level, {seg}_z_mom
    """
    if segments is None:
        segments = SLOPE_SEGMENTS

    result = pd.DataFrame(index=yields.index)

    for seg_name, (short_col, long_col) in segments.items():
        if short_col not in yields.columns or long_col not in yields.columns:
            continue

        spread = yields[long_col] - yields[short_col]
        result[f"{seg_name}_spread"] = spread

        # Z-score of level
        mu = spread.rolling(z_window, min_periods=63).mean()
        sigma = spread.rolling(z_window, min_periods=63).std()
        result[f"{seg_name}_z_level"] = (spread - mu) / (sigma + 1e-10)

        # Momentum: change over mom_window
        mom = spread.diff(mom_window)
        mu_m = mom.rolling(z_window, min_periods=63).mean()
        sigma_m = mom.rolling(z_window, min_periods=63).std()
        result[f"{seg_name}_z_mom"] = (mom - mu_m) / (sigma_m + 1e-10)

    return result
