"""
strategy_backtest.py - Multi-leg backtest engine for curve strategies.

Computes daily returns for multi-leg strategies, performance metrics,
and provides batch backtesting with ranking by Sharpe.
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE, BACKTEST_START

ANN = 252


def compute_strategy_returns(strategy: dict, etf_ret: pd.DataFrame,
                             start: str = None) -> pd.Series:
    """Compute daily returns for a multi-leg strategy.

    daily_ret = sum(weight_i * etf_ret_i) for all legs.

    Args:
        strategy: dict with 'weights' mapping ETF -> weight
        etf_ret: DataFrame of daily ETF returns
        start: optional start date filter

    Returns:
        pd.Series of daily strategy returns
    """
    weights = strategy["weights"]
    etfs = list(weights.keys())

    # Align to common dates
    common = etf_ret[etfs].dropna()
    if start is not None:
        common = common.loc[common.index >= pd.Timestamp(start)]

    daily_ret = pd.Series(0.0, index=common.index)
    for etf, w in weights.items():
        daily_ret += w * common[etf]

    daily_ret.name = strategy["name"]
    return daily_ret


def compute_metrics(daily_ret: pd.Series, risk_free: float = RISK_FREE_RATE) -> dict:
    """Compute performance metrics from daily returns.

    Returns:
        dict with sharpe, sortino, ann_ret, vol, mdd, calmar, total_ret, hit_rate
    """
    dr = daily_ret.values
    n = len(dr)
    if n < 10:
        return {k: np.nan for k in
                ["sharpe", "sortino", "ann_ret", "vol", "mdd", "calmar",
                 "total_ret", "hit_rate", "n_days"]}

    equity = np.cumprod(1.0 + dr)
    ann_ret = equity[-1] ** (ANN / n) - 1
    vol = np.std(dr) * np.sqrt(ANN)

    rf_d = risk_free / ANN
    exc = dr - rf_d
    sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)

    neg = exc[exc < 0]
    ds = np.sqrt(np.mean(neg ** 2)) * np.sqrt(ANN) if len(neg) > 0 else 1e-10
    sortino = np.mean(exc) * ANN / (ds + 1e-10)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0.0

    hit_rate = np.mean(dr > 0)

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


def backtest_strategy(strategy: dict, etf_ret: pd.DataFrame,
                      start: str = BACKTEST_START,
                      tcost_bps: float = 0) -> dict:
    """Backtest a single strategy.

    For buy-and-hold strategies tcost is only initial (negligible).

    Args:
        strategy: strategy dict from strategy_definitions
        etf_ret: DataFrame of daily ETF returns
        start: backtest start date
        tcost_bps: transaction cost in bps (applied at inception)

    Returns:
        dict with 'daily_ret', 'equity', 'metrics', 'strategy'
    """
    daily_ret = compute_strategy_returns(strategy, etf_ret, start)

    # Apply initial transaction cost
    if tcost_bps > 0 and len(daily_ret) > 0:
        total_turnover = sum(abs(w) for w in strategy["weights"].values())
        initial_cost = total_turnover * tcost_bps / 10000.0
        dr = daily_ret.copy()
        dr.iloc[0] -= initial_cost
        daily_ret = dr

    equity = (1.0 + daily_ret).cumprod()
    metrics = compute_metrics(daily_ret)

    return {
        "daily_ret": daily_ret,
        "equity": equity,
        "metrics": metrics,
        "strategy": strategy,
    }


def backtest_all(strategies: list, etf_ret: pd.DataFrame,
                 start: str = BACKTEST_START,
                 tcost_bps: float = 0) -> tuple:
    """Backtest all strategies and return summary + detailed results.

    Args:
        strategies: list of strategy dicts
        etf_ret: DataFrame of daily ETF returns
        start: backtest start date
        tcost_bps: transaction cost in bps

    Returns:
        (summary_df, results_dict)
        summary_df: DataFrame ranked by Sharpe (descending)
        results_dict: {strategy_name: backtest_result}
    """
    results = {}
    rows = []

    for strat in strategies:
        res = backtest_strategy(strat, etf_ret, start, tcost_bps)
        results[strat["name"]] = res
        m = res["metrics"]
        rows.append({
            "name": strat["name"],
            "display": strat["display"],
            "type": strat["type"],
            "direction": strat["direction"],
            **m,
        })

    summary = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    summary = summary.reset_index(drop=True)
    return summary, results


def compute_yearly_breakdown(daily_ret: pd.Series) -> pd.DataFrame:
    """Compute annual return breakdown from daily returns.

    Returns:
        DataFrame with columns: year, return, vol, sharpe, mdd, hit_rate
    """
    if len(daily_ret) == 0:
        return pd.DataFrame()

    years = sorted(daily_ret.index.year.unique())
    rows = []
    for y in years:
        mask = daily_ret.index.year == y
        dr = daily_ret[mask]
        if len(dr) < 5:
            continue
        eq = np.cumprod(1.0 + dr.values)
        ret = eq[-1] - 1
        vol = np.std(dr.values) * np.sqrt(ANN)
        rf_d = RISK_FREE_RATE / ANN
        exc = dr.values - rf_d
        sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)
        peak = np.maximum.accumulate(eq)
        mdd = ((eq - peak) / peak).min()
        hit = np.mean(dr.values > 0)
        rows.append({
            "year": y,
            "return": ret,
            "vol": vol,
            "sharpe": sharpe,
            "mdd": mdd,
            "hit_rate": hit,
        })

    return pd.DataFrame(rows)
