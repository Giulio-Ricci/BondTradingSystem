"""
portfolio_combinations.py - Enumerate all portfolio combinations and measure performance.

Enumerates C(25,k) combinations for k=2..5 (~68K total), computes equal-weight
portfolio performance for each, and filters top performers.
"""

import itertools
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE, BACKTEST_START

from curve_strategies.strategy_definitions import STRATEGY_POOL
from curve_strategies.strategy_backtest import (
    compute_strategy_returns, compute_metrics, backtest_all,
)

ANN = 252


def build_returns_matrix(strategy_returns: dict) -> pd.DataFrame:
    """Build aligned (T x N) daily returns matrix from strategy returns dict.

    Args:
        strategy_returns: {strategy_name: pd.Series of daily returns}

    Returns:
        pd.DataFrame with shape (T, N), columns = strategy names,
        aligned to common dates (inner join), sorted by date.
    """
    # Build from dict of Series
    df = pd.DataFrame(strategy_returns)
    df = df.dropna()
    df = df.sort_index()
    return df


def compute_correlation_matrix(returns_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute 25x25 correlation matrix with hierarchical clustering reorder.

    Args:
        returns_matrix: (T x N) DataFrame of daily returns

    Returns:
        pd.DataFrame (N x N) correlation matrix, reordered by
        hierarchical clustering for visual clarity.
    """
    corr = returns_matrix.corr()

    # Hierarchical clustering for reordering
    # Convert correlation to distance: d = 1 - |corr|
    dist = 1.0 - np.abs(corr.values)
    np.fill_diagonal(dist, 0.0)

    # Ensure symmetry and non-negative
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, None)

    # Condensed distance for linkage
    n = len(corr)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist[i, j])
    condensed = np.array(condensed)

    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    # Reorder
    names_ordered = [corr.columns[i] for i in order]
    corr_ordered = corr.loc[names_ordered, names_ordered]

    return corr_ordered


def _compute_metrics_fast(daily_ret_np: np.ndarray,
                          risk_free: float = RISK_FREE_RATE) -> dict:
    """Fast metrics computation on numpy array (no pandas overhead).

    Args:
        daily_ret_np: 1D numpy array of daily returns
        risk_free: annual risk-free rate

    Returns:
        dict with sharpe, sortino, ann_ret, vol, mdd, calmar, total_ret, hit_rate, n_days
    """
    n = len(daily_ret_np)
    if n < 10:
        return {k: np.nan for k in
                ["sharpe", "sortino", "ann_ret", "vol", "mdd", "calmar",
                 "total_ret", "hit_rate", "n_days"]}

    equity = np.cumprod(1.0 + daily_ret_np)
    ann_ret = equity[-1] ** (ANN / n) - 1
    vol = np.std(daily_ret_np) * np.sqrt(ANN)

    rf_d = risk_free / ANN
    exc = daily_ret_np - rf_d
    sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)

    neg = exc[exc < 0]
    ds = np.sqrt(np.mean(neg ** 2)) * np.sqrt(ANN) if len(neg) > 0 else 1e-10
    sortino = np.mean(exc) * ANN / (ds + 1e-10)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0.0

    hit_rate = np.mean(daily_ret_np > 0)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_ret": ann_ret,
        "vol": vol,
        "mdd": mdd,
        "calmar": calmar,
        "total_ret": equity[-1] - 1,
        "hit_rate": hit_rate,
        "n_days": n,
    }


def _chunks(lst, size):
    """Yield successive chunks from list."""
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def _batch_metrics(port_returns_2d: np.ndarray,
                   risk_free: float = RISK_FREE_RATE) -> dict:
    """Compute metrics for a batch of portfolios simultaneously.

    Args:
        port_returns_2d: (T, B) array where B = batch size
        risk_free: annual risk-free rate

    Returns:
        dict of arrays, each of length B
    """
    T, B = port_returns_2d.shape
    rf_d = risk_free / ANN

    # Excess returns
    exc = port_returns_2d - rf_d  # (T, B)

    # Means and stds (along time axis)
    exc_mean = exc.mean(axis=0)        # (B,)
    exc_std = exc.std(axis=0) + 1e-10  # (B,)
    ret_std = port_returns_2d.std(axis=0)  # (B,)

    sharpe = exc_mean / exc_std * np.sqrt(ANN)

    # Downside std for sortino
    neg_exc = np.where(exc < 0, exc, 0.0)  # (T, B)
    ds = np.sqrt((neg_exc ** 2).mean(axis=0)) * np.sqrt(ANN) + 1e-10  # (B,)
    sortino = exc_mean * ANN / ds

    # Equity curves
    equity = np.cumprod(1.0 + port_returns_2d, axis=0)  # (T, B)
    total_ret = equity[-1] - 1
    ann_ret = np.sign(equity[-1]) * (np.abs(equity[-1]) ** (ANN / T)) - 1
    vol = ret_std * np.sqrt(ANN)

    # MDD (vectorized over columns)
    peak = np.maximum.accumulate(equity, axis=0)  # (T, B)
    dd = (equity - peak) / (peak + 1e-10)  # (T, B)
    mdd = dd.min(axis=0)  # (B,)

    calmar = np.where(np.abs(mdd) > 1e-10, ann_ret / np.abs(mdd), 0.0)
    hit_rate = (port_returns_2d > 0).mean(axis=0)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_ret": ann_ret,
        "vol": vol,
        "mdd": mdd,
        "calmar": calmar,
        "total_ret": total_ret,
        "hit_rate": hit_rate,
    }


def enumerate_combinations(returns_matrix: pd.DataFrame,
                           max_k: int = 5, min_k: int = 2,
                           batch_size: int = 2000) -> pd.DataFrame:
    """Enumerate all C(N, k) equal-weight portfolio combinations.

    For each combination, computes EW portfolio returns and performance metrics.
    Uses fully vectorized numpy operations: processes batch_size combinations
    simultaneously via 2D array operations.

    Args:
        returns_matrix: (T x N) DataFrame of daily strategy returns
        max_k: maximum number of strategies per portfolio
        min_k: minimum number of strategies per portfolio
        batch_size: number of combinations to process at once

    Returns:
        pd.DataFrame with columns:
            combo_id, k, strategies, sharpe, sortino, ann_ret, vol,
            mdd, calmar, total_ret, hit_rate, n_days
    """
    ret_np = returns_matrix.values  # (T, N)
    col_names = list(returns_matrix.columns)
    T, N = ret_np.shape

    all_sharpe = []
    all_sortino = []
    all_ann_ret = []
    all_vol = []
    all_mdd = []
    all_calmar = []
    all_total_ret = []
    all_hit_rate = []
    all_k = []
    all_strategies = []

    for k in range(min_k, max_k + 1):
        combos = list(itertools.combinations(range(N), k))
        n_combos = len(combos)
        print(f"    k={k}: {n_combos:,} combinations ...", end="", flush=True)

        for batch in _chunks(combos, batch_size):
            # Build index array for the batch
            idx_array = np.array(batch)  # (B, k)
            B = len(batch)

            # Compute EW portfolio returns for entire batch at once
            # port_returns[:, b] = mean of ret_np[:, batch[b]] for each b
            # Use advanced indexing: gather all columns, reshape, mean
            # For each combo, select k columns and average
            selected = ret_np[:, idx_array.ravel()].reshape(T, B, k)  # (T, B, k)
            port_returns = selected.mean(axis=2)  # (T, B)

            # Batch metrics computation (fully vectorized)
            metrics = _batch_metrics(port_returns)

            all_sharpe.append(metrics["sharpe"])
            all_sortino.append(metrics["sortino"])
            all_ann_ret.append(metrics["ann_ret"])
            all_vol.append(metrics["vol"])
            all_mdd.append(metrics["mdd"])
            all_calmar.append(metrics["calmar"])
            all_total_ret.append(metrics["total_ret"])
            all_hit_rate.append(metrics["hit_rate"])

            for combo in batch:
                all_k.append(k)
                all_strategies.append("+".join(col_names[i] for i in combo))

        print(f" done ({n_combos:,} processed)")

    df = pd.DataFrame({
        "combo_id": np.arange(len(all_k)),
        "k": all_k,
        "strategies": all_strategies,
        "sharpe": np.concatenate(all_sharpe),
        "sortino": np.concatenate(all_sortino),
        "ann_ret": np.concatenate(all_ann_ret),
        "vol": np.concatenate(all_vol),
        "mdd": np.concatenate(all_mdd),
        "calmar": np.concatenate(all_calmar),
        "total_ret": np.concatenate(all_total_ret),
        "hit_rate": np.concatenate(all_hit_rate),
        "n_days": T,
    })
    print(f"    Total: {len(df):,} combinations enumerated")
    return df


def filter_top_combinations(combo_df: pd.DataFrame,
                            top_n: int = 200,
                            metric: str = "sharpe") -> pd.DataFrame:
    """Filter top N combinations by selected metric.

    Args:
        combo_df: DataFrame from enumerate_combinations()
        top_n: number of top combinations to keep
        metric: column to sort by (descending)

    Returns:
        pd.DataFrame with top_n rows sorted by metric (descending)
    """
    df = combo_df.dropna(subset=[metric]).copy()
    df = df.sort_values(metric, ascending=False).head(top_n)
    df = df.reset_index(drop=True)
    return df
