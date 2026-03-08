"""
trading_indicators.py - Trading indicators for regime detection and strategy selection.

5 families of indicators computed day-by-day for each strategy:
    1. Rolling momentum (cumulative return over various windows)
    2. Rolling Sharpe ratio
    3. Volatility mean-reversion signals
    4. Cross-strategy consensus
    5. Composite score (weighted blend of all indicators)
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE

from curve_strategies.strategy_definitions import STRATEGY_POOL

ANN = 252

# Strategy type mapping for consensus indicator
STRATEGY_TYPES = {}
for s in STRATEGY_POOL:
    stype = s["type"]
    direction = s.get("direction", "")
    if stype == "spread" and direction == "steepener":
        group = "steepener"
    elif stype == "spread" and direction == "flattener":
        group = "flattener"
    elif stype == "butterfly":
        group = "butterfly"
    elif stype == "directional":
        group = "directional"
    elif stype == "barbell":
        group = "barbell"
    else:
        group = "other"
    STRATEGY_TYPES[s["name"]] = group


# ══════════════════════════════════════════════════════════════════════════
# INDICATOR 1: ROLLING MOMENTUM
# ══════════════════════════════════════════════════════════════════════════

def compute_rolling_momentum(returns_matrix: pd.DataFrame,
                             windows: list = None) -> dict:
    """Compute rolling cumulative return (momentum) for each strategy.

    mom_w(t) = prod(1 + r[t-w:t]) - 1
    z_mom_w(t) = z-score of mom_w over 252 days

    Args:
        returns_matrix: (T x N) DataFrame of daily returns
        windows: list of lookback windows (default: [21, 63, 126, 252])

    Returns:
        dict mapping indicator name -> DataFrame(T, N):
            'mom_21', 'mom_63', 'mom_126', 'mom_252',
            'z_mom_21', 'z_mom_63', 'z_mom_126', 'z_mom_252'
    """
    if windows is None:
        windows = [21, 63, 126, 252]

    result = {}

    for w in windows:
        # Rolling cumulative return
        mom = (1 + returns_matrix).rolling(window=w, min_periods=w).apply(
            np.prod, raw=True
        ) - 1

        result[f"mom_{w}"] = mom

        # Z-score over 252 days
        z_window = 252
        mom_mean = mom.rolling(window=z_window, min_periods=z_window // 2).mean()
        mom_std = mom.rolling(window=z_window, min_periods=z_window // 2).std()
        z_mom = (mom - mom_mean) / (mom_std + 1e-10)

        result[f"z_mom_{w}"] = z_mom

    return result


# ══════════════════════════════════════════════════════════════════════════
# INDICATOR 2: ROLLING SHARPE
# ══════════════════════════════════════════════════════════════════════════

def compute_rolling_sharpe(returns_matrix: pd.DataFrame,
                           windows: list = None,
                           risk_free: float = RISK_FREE_RATE) -> dict:
    """Compute rolling Sharpe ratio for each strategy.

    sharpe_w(t) = (mean(r[t-w:t]) - rf/252) / std(r[t-w:t]) * sqrt(252)

    Args:
        returns_matrix: (T x N) DataFrame of daily returns
        windows: lookback windows (default: [63, 126, 252])
        risk_free: annual risk-free rate

    Returns:
        dict mapping indicator name -> DataFrame(T, N):
            'sharpe_63', 'sharpe_126', 'sharpe_252'
    """
    if windows is None:
        windows = [63, 126, 252]

    rf_d = risk_free / ANN
    result = {}

    for w in windows:
        roll_mean = returns_matrix.rolling(window=w, min_periods=w).mean()
        roll_std = returns_matrix.rolling(window=w, min_periods=w).std()

        sharpe = (roll_mean - rf_d) / (roll_std + 1e-10) * np.sqrt(ANN)
        result[f"sharpe_{w}"] = sharpe

    return result


# ══════════════════════════════════════════════════════════════════════════
# INDICATOR 3: VOLATILITY MEAN-REVERSION
# ══════════════════════════════════════════════════════════════════════════

def compute_vol_mean_reversion(returns_matrix: pd.DataFrame,
                               short_w: int = 21,
                               long_w: int = 252) -> dict:
    """Compute volatility mean-reversion indicators.

    vol_ratio(t) = vol_short / vol_long
    vol_z(t) = z-score of vol_ratio over 504 days
    dd_percentile(t) = percentile rank of current drawdown over 504 days

    Logic:
        vol_z > 2   -> extreme volatility, possible reversal
        vol_z < -1.5 -> unusually low vol, good entry

    Args:
        returns_matrix: (T x N) DataFrame of daily returns
        short_w: short volatility window
        long_w: long volatility window

    Returns:
        dict mapping indicator name -> DataFrame(T, N):
            'vol_ratio', 'vol_z', 'dd_percentile'
    """
    vol_short = returns_matrix.rolling(window=short_w, min_periods=short_w).std()
    vol_long = returns_matrix.rolling(window=long_w, min_periods=long_w).std()

    vol_ratio = vol_short / (vol_long + 1e-10)

    # Z-score over 504 days
    z_window = 504
    vr_mean = vol_ratio.rolling(window=z_window, min_periods=z_window // 2).mean()
    vr_std = vol_ratio.rolling(window=z_window, min_periods=z_window // 2).std()
    vol_z = (vol_ratio - vr_mean) / (vr_std + 1e-10)

    # Drawdown percentile
    equity = (1 + returns_matrix).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak  # negative values

    # Rolling percentile rank of current drawdown over 504 days
    dd_pct = drawdown.rolling(window=z_window, min_periods=z_window // 2).rank(pct=True)

    return {
        "vol_ratio": vol_ratio,
        "vol_z": vol_z,
        "dd_percentile": dd_pct,
    }


# ══════════════════════════════════════════════════════════════════════════
# INDICATOR 4: CROSS-STRATEGY CONSENSUS
# ══════════════════════════════════════════════════════════════════════════

def compute_cross_strategy_consensus(returns_matrix: pd.DataFrame,
                                     strategy_pool: list = None,
                                     window: int = 63) -> pd.DataFrame:
    """Compute cross-strategy consensus by strategy type.

    For each TYPE (steepener, flattener, butterfly, directional, barbell):
        consensus = % of variants with positive rolling return
        avg_mom = average momentum of variants

    Consensus > 0.75 AND avg_mom > 0 -> type "strong"
    Consensus < 0.25 -> type "weak"

    Args:
        returns_matrix: (T x N) DataFrame of daily returns
        strategy_pool: list of strategy dicts (default: STRATEGY_POOL)
        window: lookback window for rolling return

    Returns:
        pd.DataFrame (T x N) consensus score for each strategy based on
        the consensus of its type group. Values in [0, 1].
    """
    if strategy_pool is None:
        strategy_pool = STRATEGY_POOL

    # Rolling cumulative return
    rolling_ret = (1 + returns_matrix).rolling(window=window, min_periods=window).apply(
        np.prod, raw=True
    ) - 1

    # Group strategies by type
    groups = {}
    for col in returns_matrix.columns:
        group = STRATEGY_TYPES.get(col, "other")
        if group not in groups:
            groups[group] = []
        groups[group].append(col)

    # Compute consensus for each strategy based on its group
    consensus_df = pd.DataFrame(np.nan, index=returns_matrix.index,
                                columns=returns_matrix.columns)

    for group, members in groups.items():
        if len(members) < 2:
            # Single member: consensus = 1 if positive, 0 otherwise
            for m in members:
                consensus_df[m] = (rolling_ret[m] > 0).astype(float)
            continue

        # For each member, consensus = fraction of OTHER members with positive return
        group_positive = (rolling_ret[members] > 0).astype(float)
        group_consensus = group_positive.mean(axis=1)  # fraction positive

        for m in members:
            consensus_df[m] = group_consensus

    return consensus_df


# ══════════════════════════════════════════════════════════════════════════
# INDICATOR 5: COMPOSITE SCORE
# ══════════════════════════════════════════════════════════════════════════

def compute_composite_score(mom: dict, sharpe: dict, vol: dict,
                            consensus: pd.DataFrame,
                            weights: dict = None) -> tuple:
    """Compute composite indicator score and cross-sectional rank.

    Default weights:
        z_mom_126:   0.30
        sharpe_126:  0.25
        vol_z:       0.15
        consensus:   0.15
        z_mom_252:   0.15

    composite_i(t) = sum(w_k * indicator_k_i(t))
    rank_i(t) = cross-sectional rank (1=best, N=worst)

    Args:
        mom: dict from compute_rolling_momentum()
        sharpe: dict from compute_rolling_sharpe()
        vol: dict from compute_vol_mean_reversion()
        consensus: DataFrame from compute_cross_strategy_consensus()
        weights: dict of indicator name -> weight (optional)

    Returns:
        (scores_df, ranks_df)
        scores_df: DataFrame (T x N) composite scores
        ranks_df: DataFrame (T x N) cross-sectional ranks (1=best)
    """
    if weights is None:
        weights = {
            "z_mom_126": 0.30,
            "sharpe_126": 0.25,
            "vol_z": 0.15,
            "consensus": 0.15,
            "z_mom_252": 0.15,
        }

    # Collect indicator DataFrames
    indicators = {}
    if "z_mom_126" in weights and "z_mom_126" in mom:
        indicators["z_mom_126"] = mom["z_mom_126"]
    if "z_mom_252" in weights and "z_mom_252" in mom:
        indicators["z_mom_252"] = mom["z_mom_252"]
    if "sharpe_126" in weights and "sharpe_126" in sharpe:
        indicators["sharpe_126"] = sharpe["sharpe_126"]
    if "vol_z" in weights and "vol_z" in vol:
        # Invert vol_z: high vol_z is bad (extreme vol), low vol_z is good
        indicators["vol_z"] = -vol["vol_z"]
    if "consensus" in weights:
        indicators["consensus"] = consensus

    # Compute weighted composite
    # Align all indicators to the same index/columns
    if not indicators:
        return pd.DataFrame(), pd.DataFrame()

    ref = list(indicators.values())[0]
    scores = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)

    for name, df in indicators.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        # Align to scores
        aligned = df.reindex(index=scores.index, columns=scores.columns)
        scores += w * aligned.fillna(0.0)

    # Cross-sectional rank (1 = best = highest score)
    ranks = scores.rank(axis=1, ascending=False, method="min")

    return scores, ranks


# ══════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION: COMPUTE ALL INDICATORS
# ══════════════════════════════════════════════════════════════════════════

def compute_raw_indicators(returns_matrix: pd.DataFrame,
                           strategy_pool: list = None,
                           verbose: bool = True) -> dict:
    """Compute raw indicators (momentum, sharpe, vol, consensus) without composite.

    These are independent of indicator weights and can be cached.

    Args:
        returns_matrix: (T x N) DataFrame of daily returns
        strategy_pool: list of strategy dicts
        verbose: print progress messages

    Returns:
        dict with keys: 'momentum', 'sharpe', 'vol', 'consensus'
    """
    if verbose:
        print("    Computing rolling momentum ...", flush=True)
    mom = compute_rolling_momentum(returns_matrix)

    if verbose:
        print("    Computing rolling Sharpe ...", flush=True)
    sharpe = compute_rolling_sharpe(returns_matrix)

    if verbose:
        print("    Computing vol mean-reversion ...", flush=True)
    vol = compute_vol_mean_reversion(returns_matrix)

    if verbose:
        print("    Computing cross-strategy consensus ...", flush=True)
    consensus = compute_cross_strategy_consensus(returns_matrix, strategy_pool)

    return {
        "momentum": mom,
        "sharpe": sharpe,
        "vol": vol,
        "consensus": consensus,
    }


def compute_all_indicators(returns_matrix: pd.DataFrame,
                           strategy_pool: list = None,
                           indicator_weights: dict = None,
                           raw_cache: dict = None,
                           verbose: bool = True) -> dict:
    """Compute all 5 indicator families and return consolidated dict.

    Args:
        returns_matrix: (T x N) DataFrame of daily returns
        strategy_pool: list of strategy dicts
        indicator_weights: weights for composite score
        raw_cache: pre-computed raw indicators (skip recomputation)
        verbose: print progress messages

    Returns:
        dict with keys:
            'momentum': dict of momentum DataFrames
            'sharpe': dict of rolling Sharpe DataFrames
            'vol': dict of vol mean-reversion DataFrames
            'consensus': DataFrame of consensus scores
            'composite_scores': DataFrame of composite scores
            'composite_ranks': DataFrame of cross-sectional ranks
    """
    if raw_cache is not None:
        raw = raw_cache
    else:
        raw = compute_raw_indicators(returns_matrix, strategy_pool, verbose=verbose)

    if verbose:
        print("    Computing composite score ...", flush=True)
    scores, ranks = compute_composite_score(
        raw["momentum"], raw["sharpe"], raw["vol"], raw["consensus"],
        weights=indicator_weights,
    )

    return {
        **raw,
        "composite_scores": scores,
        "composite_ranks": ranks,
    }
