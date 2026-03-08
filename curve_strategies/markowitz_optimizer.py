"""
markowitz_optimizer.py - Mean-variance portfolio optimization.

Implements Markowitz optimization with scipy.optimize.minimize (SLSQP):
    - Ledoit-Wolf shrinkage covariance (no sklearn dependency)
    - Minimum variance portfolio
    - Maximum Sharpe ratio portfolio
    - Risk parity portfolio
    - Efficient frontier computation
    - Rolling/expanding Markowitz with periodic rebalancing
    - Batch optimization of top EW combinations
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE

ANN = 252


# ══════════════════════════════════════════════════════════════════════════
# COVARIANCE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════

def estimate_covariance(returns_matrix: pd.DataFrame,
                        method: str = "ledoit_wolf") -> np.ndarray:
    """Estimate covariance matrix with optional Ledoit-Wolf shrinkage.

    Ledoit-Wolf Oracle Approximating Shrinkage (OAS):
        Sigma_shrunk = (1 - alpha) * S + alpha * mu * I
    where S is the sample covariance, mu = trace(S)/N, and alpha is the
    optimal shrinkage intensity.

    Args:
        returns_matrix: (T x N) DataFrame or ndarray of returns
        method: "sample" or "ledoit_wolf"

    Returns:
        np.ndarray (N x N) covariance matrix
    """
    if isinstance(returns_matrix, pd.DataFrame):
        X = returns_matrix.values
    else:
        X = np.asarray(returns_matrix)

    T, N = X.shape

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Sample covariance
    S = X_centered.T @ X_centered / T

    if method == "sample":
        return S

    # Ledoit-Wolf shrinkage
    mu = np.trace(S) / N  # target: scaled identity

    # Compute optimal alpha (Oracle Approximating Shrinkage)
    # delta = ||S - mu*I||^2 / N^2  (squared Frobenius norm of deviation)
    delta = np.sum((S - mu * np.eye(N)) ** 2) / N

    # sum of squared norms of rows of X_centered^2 minus S^2
    # beta = 1/(T^2 * N) * sum_t || x_t x_t' - S ||^2
    X2 = X_centered ** 2
    beta_sum = 0.0
    for t in range(T):
        xt = X_centered[t:t+1, :]  # (1, N)
        M_t = xt.T @ xt  # (N, N) outer product
        beta_sum += np.sum((M_t - S) ** 2)
    beta = beta_sum / (T * T * N)

    # Optimal shrinkage intensity
    alpha = min(beta / (delta + 1e-10), 1.0)
    alpha = max(alpha, 0.0)

    Sigma = (1 - alpha) * S + alpha * mu * np.eye(N)

    return Sigma


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION: MINIMUM VARIANCE
# ══════════════════════════════════════════════════════════════════════════

def optimize_min_variance(mu: np.ndarray, cov: np.ndarray,
                          allow_short: bool = False) -> np.ndarray:
    """Minimum variance portfolio: min w'Cov*w s.t. sum(w)=1.

    Args:
        mu: (N,) expected returns (not used, kept for API consistency)
        cov: (N, N) covariance matrix
        allow_short: if True, allow negative weights

    Returns:
        np.ndarray (N,) optimal weights
    """
    N = len(cov)
    w0 = np.ones(N) / N

    def objective(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if allow_short:
        bounds = [(-1.0, 1.0)] * N
    else:
        bounds = [(0.0, 1.0)] * N

    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    w = result.x
    w = np.maximum(w, 0.0) if not allow_short else w
    w /= w.sum() + 1e-10
    return w


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION: MAXIMUM SHARPE
# ══════════════════════════════════════════════════════════════════════════

def optimize_max_sharpe(mu: np.ndarray, cov: np.ndarray,
                        risk_free: float = RISK_FREE_RATE / ANN) -> np.ndarray:
    """Maximum Sharpe ratio portfolio.

    Max (w'mu - rf) / sqrt(w'Cov*w)  s.t.  sum(w)=1, w>=0.

    Args:
        mu: (N,) expected daily returns
        cov: (N, N) daily covariance matrix
        risk_free: daily risk-free rate

    Returns:
        np.ndarray (N,) optimal weights
    """
    N = len(mu)
    w0 = np.ones(N) / N

    def neg_sharpe(w):
        port_ret = w @ mu - risk_free
        port_vol = np.sqrt(w @ cov @ w + 1e-12)
        return -port_ret / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * N

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-12})

    w = result.x
    w = np.maximum(w, 0.0)
    w /= w.sum() + 1e-10
    return w


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZATION: RISK PARITY
# ══════════════════════════════════════════════════════════════════════════

def optimize_risk_parity(cov: np.ndarray) -> np.ndarray:
    """Risk parity portfolio: equalize risk contributions.

    Minimizes sum((RC_i - 1/N)^2) where RC_i = w_i*(Cov*w)_i / w'Cov*w.

    Args:
        cov: (N, N) covariance matrix

    Returns:
        np.ndarray (N,) optimal weights
    """
    N = len(cov)
    w0 = np.ones(N) / N

    def risk_parity_objective(w):
        w = np.maximum(w, 1e-8)
        port_var = w @ cov @ w
        if port_var < 1e-12:
            return 1e6
        marginal_risk = cov @ w
        risk_contrib = w * marginal_risk / port_var
        target = 1.0 / N
        return np.sum((risk_contrib - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * N

    result = minimize(risk_parity_objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-14})

    w = result.x
    w = np.maximum(w, 0.0)
    w /= w.sum() + 1e-10
    return w


# ══════════════════════════════════════════════════════════════════════════
# EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════════════════

def compute_efficient_frontier(mu: np.ndarray, cov: np.ndarray,
                               n_points: int = 50) -> np.ndarray:
    """Compute the efficient frontier by sweeping target returns.

    For each target return, solve:
        min w'Cov*w  s.t.  w'mu = target,  sum(w)=1,  w>=0

    Args:
        mu: (N,) expected daily returns
        cov: (N, N) daily covariance matrix
        n_points: number of points on the frontier

    Returns:
        np.ndarray of shape (n_points, 2) with columns [vol, ret]
        (annualized)
    """
    N = len(mu)
    min_ret = mu.min()
    max_ret = mu.max()

    # Ensure we have a range
    if abs(max_ret - min_ret) < 1e-12:
        max_ret = min_ret + 1e-6

    targets = np.linspace(min_ret, max_ret, n_points)
    frontier = []

    for target in targets:
        w0 = np.ones(N) / N

        def objective(w):
            return w @ cov @ w

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        bounds = [(0.0, 1.0)] * N

        result = minimize(objective, w0, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 500, "ftol": 1e-12})

        if result.success:
            w = result.x
            port_vol = np.sqrt(w @ cov @ w) * np.sqrt(ANN)
            port_ret = (w @ mu) * ANN
            frontier.append([port_vol, port_ret])

    return np.array(frontier) if frontier else np.empty((0, 2))


# ══════════════════════════════════════════════════════════════════════════
# ROLLING / EXPANDING MARKOWITZ
# ══════════════════════════════════════════════════════════════════════════

def rolling_markowitz(returns_matrix: pd.DataFrame,
                      method: str = "max_sharpe",
                      min_window: int = 504,
                      rebalance_freq: int = 21,
                      expanding: bool = True) -> tuple:
    """Rolling/expanding Markowitz optimization with periodic rebalancing.

    Every rebalance_freq days, re-estimates mu and cov from available data
    and re-optimizes portfolio weights.

    Args:
        returns_matrix: (T x N) DataFrame of daily strategy returns
        method: "min_variance", "max_sharpe", or "risk_parity"
        min_window: minimum estimation window (days)
        rebalance_freq: rebalance every N trading days
        expanding: if True, use expanding window; if False, use rolling

    Returns:
        (weights_df, portfolio_returns)
        weights_df: DataFrame (T x N) of daily weights
        portfolio_returns: Series of daily portfolio returns
    """
    T, N = returns_matrix.shape
    dates = returns_matrix.index
    col_names = returns_matrix.columns.tolist()
    ret_np = returns_matrix.values

    weights_history = np.zeros((T, N))
    port_returns = np.zeros(T)

    current_w = np.ones(N) / N  # start EW

    for t in range(T):
        if t < min_window:
            # Not enough data yet, use EW
            current_w_used = np.ones(N) / N
        elif (t - min_window) % rebalance_freq == 0:
            # Rebalance day
            if expanding:
                window_data = ret_np[:t, :]
            else:
                start_idx = max(0, t - min_window)
                window_data = ret_np[start_idx:t, :]

            mu = window_data.mean(axis=0)
            cov = estimate_covariance(pd.DataFrame(window_data), method="ledoit_wolf")

            if method == "min_variance":
                current_w = optimize_min_variance(mu, cov)
            elif method == "max_sharpe":
                current_w = optimize_max_sharpe(mu, cov)
            elif method == "risk_parity":
                current_w = optimize_risk_parity(cov)
            else:
                current_w = np.ones(N) / N

            current_w_used = current_w
        else:
            current_w_used = current_w

        weights_history[t] = current_w_used
        port_returns[t] = current_w_used @ ret_np[t]

    weights_df = pd.DataFrame(weights_history, index=dates, columns=col_names)
    port_ret_series = pd.Series(port_returns, index=dates, name="markowitz_portfolio")

    return weights_df, port_ret_series


# ══════════════════════════════════════════════════════════════════════════
# BATCH OPTIMIZE TOP COMBINATIONS
# ══════════════════════════════════════════════════════════════════════════

def batch_optimize_top_combinations(returns_matrix: pd.DataFrame,
                                    combo_df: pd.DataFrame,
                                    top_n: int = 100,
                                    methods: list = None) -> pd.DataFrame:
    """For top EW combinations, apply all Markowitz methods and compare.

    For each top combo from EW enumeration, re-optimizes using
    min_variance, max_sharpe, and risk_parity. Compares optimized vs EW.

    Args:
        returns_matrix: (T x N) full returns matrix
        combo_df: DataFrame from enumerate_combinations (already filtered)
        top_n: number of top combinations to optimize
        methods: list of optimization methods

    Returns:
        pd.DataFrame with columns:
            combo_id, k, strategies, method, sharpe, sortino, ann_ret,
            vol, mdd, calmar, total_ret
    """
    if methods is None:
        methods = ["min_variance", "max_sharpe", "risk_parity"]

    col_names = list(returns_matrix.columns)
    ret_np = returns_matrix.values

    top_combos = combo_df.head(top_n)
    rows = []

    for _, combo_row in top_combos.iterrows():
        strat_names = combo_row["strategies"].split("+")
        indices = [col_names.index(s) for s in strat_names if s in col_names]
        if len(indices) < 2:
            continue

        sub_ret = ret_np[:, indices]
        mu = sub_ret.mean(axis=0)
        cov = estimate_covariance(pd.DataFrame(sub_ret), method="ledoit_wolf")

        # EW baseline
        ew_ret = sub_ret.mean(axis=1)
        ew_metrics = _compute_combo_metrics(ew_ret)
        rows.append({
            "combo_id": combo_row["combo_id"],
            "k": combo_row["k"],
            "strategies": combo_row["strategies"],
            "method": "equal_weight",
            **ew_metrics,
        })

        for method_name in methods:
            if method_name == "min_variance":
                w = optimize_min_variance(mu, cov)
            elif method_name == "max_sharpe":
                w = optimize_max_sharpe(mu, cov)
            elif method_name == "risk_parity":
                w = optimize_risk_parity(cov)
            else:
                continue

            opt_ret = sub_ret @ w
            opt_metrics = _compute_combo_metrics(opt_ret)
            rows.append({
                "combo_id": combo_row["combo_id"],
                "k": combo_row["k"],
                "strategies": combo_row["strategies"],
                "method": method_name,
                **opt_metrics,
            })

    return pd.DataFrame(rows)


def _compute_combo_metrics(daily_ret_np: np.ndarray) -> dict:
    """Compute performance metrics from numpy array of daily returns."""
    n = len(daily_ret_np)
    if n < 10:
        return {k: np.nan for k in
                ["sharpe", "sortino", "ann_ret", "vol", "mdd", "calmar", "total_ret"]}

    equity = np.cumprod(1.0 + daily_ret_np)
    ann_ret = equity[-1] ** (ANN / n) - 1
    vol = np.std(daily_ret_np) * np.sqrt(ANN)

    rf_d = RISK_FREE_RATE / ANN
    exc = daily_ret_np - rf_d
    sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)

    neg = exc[exc < 0]
    ds = np.sqrt(np.mean(neg ** 2)) * np.sqrt(ANN) if len(neg) > 0 else 1e-10
    sortino = np.mean(exc) * ANN / (ds + 1e-10)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0.0

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "ann_ret": ann_ret,
        "vol": vol,
        "mdd": mdd,
        "calmar": calmar,
        "total_ret": equity[-1] - 1,
    }
