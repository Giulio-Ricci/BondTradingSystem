"""
curve_signal_system.py - Signal-driven yield curve strategy selection.

Replaces Markowitz/ML optimization with a signal-based approach:
    1. Compute slope momentum/level z-scores per curve segment
    2. Activate steepener when segment is steepening, flattener when flattening
    3. RR z-score + MOVE/VIX as global risk overlay
    4. Position sizing proportional to signal strength

Usage:
    from curve_strategies.curve_signal_system import CurveSignalSystem
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RISK_FREE_RATE
from signals import compute_zscore_standard
from curve_strategies.regime_analysis import compute_slope_signals, SLOPE_SEGMENTS
from curve_strategies.strategy_backtest import compute_metrics

ANN = 252


# ══════════════════════════════════════════════════════════════════════════
# SEGMENT -> STRATEGY MAPPING
# ══════════════════════════════════════════════════════════════════════════

# Each yield curve segment maps to steepener/flattener ETF strategies
SEGMENT_STRATEGIES = {
    "2s5s":   {"steepener": "steep_2s5s",   "flattener": None},
    "5s10s":  {"steepener": "steep_5s10s",  "flattener": None},
    "10s30s": {"steepener": "steep_20s30s", "flattener": "flat_30s10s"},
    "2s10s":  {"steepener": "steep_2s10s",  "flattener": "flat_10s2s"},
    "2s30s":  {"steepener": "steep_2s30s",  "flattener": "flat_30s2s"},
    "5s30s":  {"steepener": "steep_5s30s",  "flattener": "flat_30s5s"},
    "10s20s": {"steepener": "steep_10s20s", "flattener": None},
}

# Butterfly: curvature = 2*mid - short - long
BUTTERFLY_SEGMENTS = {
    "2s5s10s":  {"positive": "bfly_2s5s10s_pos",  "negative": "bfly_2s5s10s_neg"},
    "2s10s30s": {"positive": "bfly_2s10s30s_pos", "negative": "bfly_2s10s30s_neg"},
    "5s10s30s": {"positive": "bfly_5s10s30s_pos", "negative": "bfly_5s10s30s_neg"},
}

# Extended slope segments (adds 5s30s and 10s20s to base SLOPE_SEGMENTS)
EXTENDED_SLOPE_SEGMENTS = {
    **SLOPE_SEGMENTS,
    "5s30s":  ("US_5Y",  "US_30Y"),
    "10s20s": ("US_10Y", "US_20Y"),  # may not exist -> graceful skip
}

# Butterfly yield mappings: (short_yield, mid_yield, long_yield)
BUTTERFLY_YIELD_MAP = {
    "2s5s10s":  ("US_2Y", "US_5Y",  "US_10Y"),
    "2s10s30s": ("US_2Y", "US_10Y", "US_30Y"),
    "5s10s30s": ("US_5Y", "US_10Y", "US_30Y"),
}


# ══════════════════════════════════════════════════════════════════════════
# COVARIANCE ESTIMATION (Ledoit-Wolf, standalone, vectorized)
# ══════════════════════════════════════════════════════════════════════════

def estimate_covariance(returns_data, method="ledoit_wolf"):
    """Ledoit-Wolf Oracle Approximating Shrinkage covariance.

    Vectorized implementation: O(T*N + N^2) instead of O(T*N^2).

    Args:
        returns_data: (T x N) DataFrame or ndarray of returns
        method: "sample" or "ledoit_wolf"

    Returns:
        np.ndarray (N x N) covariance matrix
    """
    if isinstance(returns_data, pd.DataFrame):
        X = returns_data.values
    else:
        X = np.asarray(returns_data)

    T, N = X.shape
    X_c = X - X.mean(axis=0)
    S = X_c.T @ X_c / T

    if method == "sample":
        return S

    mu = np.trace(S) / N
    delta = np.sum((S - mu * np.eye(N)) ** 2) / N

    if delta < 1e-14:
        return S

    # Vectorized beta: ||x_t x_t' - S||_F^2 = (x_t'x_t)^2 - 2(x_t'Sx_t) + ||S||_F^2
    norms_sq = np.sum(X_c ** 2, axis=1)           # (T,)
    SX = X_c @ S                                   # (T, N)
    quad_forms = np.sum(X_c * SX, axis=1)          # (T,)
    S_frob_sq = np.sum(S ** 2)

    beta_sum = np.sum(norms_sq ** 2) - 2.0 * np.sum(quad_forms) + T * S_frob_sq
    beta = beta_sum / (T * T * N)

    alpha = np.clip(beta / (delta + 1e-10), 0.0, 1.0)

    return (1 - alpha) * S + alpha * mu * np.eye(N)


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════════════════

def compute_segment_signals(yields, segments=None, z_window=252, mom_window=63):
    """Compute z-score signals for each yield curve segment.

    Reuses regime_analysis.compute_slope_signals() with extended segments
    covering all pairs in SEGMENT_STRATEGIES.

    Args:
        yields: DataFrame with US_2Y, US_5Y, US_10Y, US_30Y columns
        segments: dict of {name: (short_col, long_col)}, defaults to extended set
        z_window: rolling window for z-score (default 252)
        mom_window: momentum lookback (default 63)

    Returns:
        DataFrame with columns: {seg}_spread, {seg}_z_level, {seg}_z_mom
    """
    if segments is None:
        segments = EXTENDED_SLOPE_SEGMENTS

    return compute_slope_signals(yields, segments=segments,
                                 z_window=z_window, mom_window=mom_window)


def compute_curvature_signals(yields, z_window=252, mom_window=63):
    """Compute curvature z-scores for butterfly segments.

    curvature = 2 * yield_mid - yield_short - yield_long
    z_curv = rolling_zscore(curvature, z_window)
    z_curv_mom = rolling_zscore(diff(curvature, mom_window), z_window)

    Args:
        yields: DataFrame with yield columns
        z_window: rolling window for z-score
        mom_window: momentum lookback

    Returns:
        DataFrame with columns: {seg}_curv, {seg}_z_curv, {seg}_z_curv_mom
    """
    result = pd.DataFrame(index=yields.index)

    for seg_name, (short_col, mid_col, long_col) in BUTTERFLY_YIELD_MAP.items():
        if not all(c in yields.columns for c in [short_col, mid_col, long_col]):
            continue

        curv = 2 * yields[mid_col] - yields[short_col] - yields[long_col]
        result[f"{seg_name}_curv"] = curv

        # Z-score of curvature level
        mu = curv.rolling(z_window, min_periods=63).mean()
        sigma = curv.rolling(z_window, min_periods=63).std()
        result[f"{seg_name}_z_curv"] = (curv - mu) / (sigma + 1e-10)

        # Momentum of curvature
        curv_mom = curv.diff(mom_window)
        mu_m = curv_mom.rolling(z_window, min_periods=63).mean()
        sigma_m = curv_mom.rolling(z_window, min_periods=63).std()
        result[f"{seg_name}_z_curv_mom"] = (curv_mom - mu_m) / (sigma_m + 1e-10)

    return result


def compute_global_overlays(rr_1m, move, vix, z_window=252):
    """Compute global risk overlay signals.

    Args:
        rr_1m: Series of 1M risk reversal
        move: Series of MOVE index
        vix: Series of VIX

    Returns:
        DataFrame with columns: rr_z, move_z, vix_z
    """
    # Determine index from first available series
    idx = None
    for s in [rr_1m, move, vix]:
        if s is not None and len(s) > 0:
            idx = s.index
            break
    if idx is None:
        return pd.DataFrame()

    result = pd.DataFrame(index=idx)

    if rr_1m is not None:
        result["rr_z"] = compute_zscore_standard(rr_1m, window=z_window)

    if move is not None:
        move_aligned = move.reindex(idx)
        mu = move_aligned.rolling(z_window, min_periods=63).mean()
        sigma = move_aligned.rolling(z_window, min_periods=63).std()
        result["move_z"] = (move_aligned - mu) / (sigma + 1e-10)

    if vix is not None:
        vix_aligned = vix.reindex(idx)
        mu = vix_aligned.rolling(z_window, min_periods=63).mean()
        sigma = vix_aligned.rolling(z_window, min_periods=63).std()
        result["vix_z"] = (vix_aligned - mu) / (sigma + 1e-10)

    return result


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL -> WEIGHTS CONVERSION
# ══════════════════════════════════════════════════════════════════════════

def signal_to_weights(segment_signals, curvature_signals, global_overlays,
                      strategy_names, signal_threshold=0.5,
                      w_slope_mom=0.6, w_slope_level=0.2, w_rr=0.2,
                      max_strategies=8,
                      level_sign=+1, mom_sign=+1):
    """Convert signals to portfolio weights for a single time step.

    For each yield curve segment:
      1. composite = mom_sign*w_mom*z_mom + level_sign*w_level*z_level + w_rr*rr_z
      2. If composite > threshold -> activate steepener
      3. If composite < -threshold -> activate flattener
      4. weight_raw = |composite| - threshold (signal strength)

    The sign parameters control interpretation:
      level_sign=+1: z_level high -> steepener (trend-following)
      level_sign=-1: z_level high -> flattener (contrarian/mean-reversion)
      mom_sign=+1:   z_mom  high -> steepener (trend-following)
      mom_sign=-1:   z_mom  high -> flattener (contrarian)

    For butterfly segments:
      composite = mom_sign*w_mom*z_curv_mom + level_sign*w_level*z_curv
      Same threshold logic for positive/negative butterfly

    Global overlay modulates total exposure:
      - RR very negative (< -2): risk-off, reduce exposure
      - MOVE very high (> 2): reduce exposure
      - VIX very high (> 2): reduce exposure

    Selects top max_strategies by signal strength, normalizes to sum=1.

    Args:
        segment_signals: Series/dict with {seg}_z_level, {seg}_z_mom values
        curvature_signals: Series/dict with {seg}_z_curv, {seg}_z_curv_mom values
        global_overlays: Series/dict with rr_z, move_z, vix_z values
        strategy_names: list of strategy column names in returns_matrix
        signal_threshold: minimum |composite| to activate
        w_slope_mom: weight on momentum z-score
        w_slope_level: weight on level z-score
        w_rr: weight on RR z-score
        max_strategies: maximum number of active strategies
        level_sign: +1 for trend-following, -1 for mean-reversion on z_level
        mom_sign: +1 for trend-following, -1 for contrarian on z_momentum

    Returns:
        np.ndarray of weights (len = len(strategy_names))
    """
    N = len(strategy_names)
    strat_idx = {name: i for i, name in enumerate(strategy_names)}
    raw_weights = {}

    # Get global RR z-score for composite signal
    rr_z = _safe_get(global_overlays, "rr_z")

    # --- Spread segments ---
    for seg_name, strat_map in SEGMENT_STRATEGIES.items():
        z_mom = _safe_get(segment_signals, f"{seg_name}_z_mom")
        z_level = _safe_get(segment_signals, f"{seg_name}_z_level")

        if np.isnan(z_mom) or np.isnan(z_level):
            continue

        # Composite signal: weighted sum of momentum + level + RR
        composite = (mom_sign * w_slope_mom * z_mom +
                     level_sign * w_slope_level * z_level +
                     w_rr * rr_z)

        if composite > signal_threshold:
            # Steepening -> activate steepener
            strat_name = strat_map["steepener"]
            if strat_name and strat_name in strat_idx:
                raw_weights[strat_name] = composite - signal_threshold
        elif composite < -signal_threshold:
            # Flattening -> activate flattener
            strat_name = strat_map["flattener"]
            if strat_name and strat_name in strat_idx:
                raw_weights[strat_name] = abs(composite) - signal_threshold

    # --- Butterfly segments ---
    for seg_name, strat_map in BUTTERFLY_SEGMENTS.items():
        z_curv_mom = _safe_get(curvature_signals, f"{seg_name}_z_curv_mom")
        z_curv = _safe_get(curvature_signals, f"{seg_name}_z_curv")

        if np.isnan(z_curv_mom):
            continue

        # Butterfly composite: momentum + level
        bfly_composite = (mom_sign * w_slope_mom * z_curv_mom +
                          level_sign * w_slope_level * (z_curv if not np.isnan(z_curv) else 0.0))

        if bfly_composite > signal_threshold:
            strat_name = strat_map["positive"]
            if strat_name in strat_idx:
                raw_weights[strat_name] = bfly_composite - signal_threshold
        elif bfly_composite < -signal_threshold:
            strat_name = strat_map["negative"]
            if strat_name in strat_idx:
                raw_weights[strat_name] = abs(bfly_composite) - signal_threshold

    # --- Global overlay factor ---
    move_z = _safe_get(global_overlays, "move_z")
    vix_z = _safe_get(global_overlays, "vix_z")

    overlay_factor = 1.0
    if rr_z < -2.0:
        overlay_factor *= max(0.3, 1.0 + (rr_z + 2.0) * 0.2)
    if move_z > 2.0:
        overlay_factor *= max(0.5, 1.0 - (move_z - 2.0) * 0.15)
    if vix_z > 2.0:
        overlay_factor *= max(0.5, 1.0 - (vix_z - 2.0) * 0.15)
    overlay_factor = max(overlay_factor, 0.1)

    for k in raw_weights:
        raw_weights[k] *= overlay_factor

    # --- Select top max_strategies by strength ---
    if len(raw_weights) == 0:
        return np.zeros(N)

    sorted_strats = sorted(raw_weights.items(), key=lambda x: x[1], reverse=True)
    selected = sorted_strats[:max_strategies]

    # Build weight vector, normalize to sum = 1
    weights = np.zeros(N)
    for strat_name, strength in selected:
        weights[strat_idx[strat_name]] = strength

    total = weights.sum()
    if total > 1e-10:
        weights /= total

    return weights


def _safe_get(data, key, default=0.0):
    """Safely get a value from Series/dict, returning default if NaN/missing."""
    if data is None:
        return default
    try:
        val = data.get(key, default) if isinstance(data, dict) else data.get(key, default)
    except (KeyError, TypeError, AttributeError):
        try:
            val = data[key]
        except (KeyError, IndexError, TypeError):
            return default
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        val = float(val)
        if np.isnan(val):
            return default
    except (ValueError, TypeError):
        return default
    return val


# ══════════════════════════════════════════════════════════════════════════
# CURVE SIGNAL SYSTEM CLASS
# ══════════════════════════════════════════════════════════════════════════

class CurveSignalSystem:
    """Signal-driven yield curve strategy selection system.

    Instead of Markowitz/ML, uses slope momentum + z-score signals to
    select which segment of the curve to trade and in which direction.
    """

    def __init__(self, returns_matrix, yields, rr_1m, move, vix,
                 strategy_pool=None,
                 # Signal params
                 z_window=252, mom_window=63,
                 signal_threshold=0.5,
                 w_slope_mom=0.6, w_slope_level=0.2, w_rr=0.2,
                 # Portfolio params
                 max_strategies=8,
                 rebalance_freq=21,
                 # Risk management
                 target_vol=0.15, max_leverage=2.0,
                 dd_threshold=-0.10, dd_reduction=0.5,
                 tcost_bps=5):
        self.returns_matrix = returns_matrix
        self.yields = yields
        self.rr_1m = rr_1m
        self.move = move
        self.vix = vix
        self.strategy_pool = strategy_pool

        self.z_window = z_window
        self.mom_window = mom_window
        self.signal_threshold = signal_threshold
        self.w_slope_mom = w_slope_mom
        self.w_slope_level = w_slope_level
        self.w_rr = w_rr

        self.max_strategies = max_strategies
        self.rebalance_freq = rebalance_freq
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.dd_threshold = dd_threshold
        self.dd_reduction = dd_reduction
        self.tcost_bps = tcost_bps

        self._signals = None
        self._backtest_result = None

    def _compute_all_signals(self):
        """Compute and cache all signals (segment + curvature + overlay).

        Aligns everything to returns_matrix.index.
        """
        if self._signals is not None:
            return self._signals

        seg_signals = compute_segment_signals(
            self.yields, z_window=self.z_window, mom_window=self.mom_window)
        curv_signals = compute_curvature_signals(
            self.yields, z_window=self.z_window, mom_window=self.mom_window)
        overlay = compute_global_overlays(
            self.rr_1m, self.move, self.vix, z_window=self.z_window)

        # Align all to returns_matrix index
        common_idx = self.returns_matrix.index
        seg_signals = seg_signals.reindex(common_idx)
        curv_signals = curv_signals.reindex(common_idx)
        overlay = overlay.reindex(common_idx)

        self._signals = {
            "segment": seg_signals,
            "curvature": curv_signals,
            "overlay": overlay,
        }
        return self._signals

    def _vol_scale(self, weights, cov):
        """Compute scaling factor to achieve target volatility."""
        port_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(ANN)
        if port_vol < 1e-8:
            return 1.0
        scale = self.target_vol / port_vol
        return min(scale, self.max_leverage)

    def backtest(self, start_date=None, end_date=None):
        """Run signal-driven backtest with vol targeting and drawdown control.

        Args:
            start_date: backtest start (default: first date in returns_matrix)
            end_date: backtest end (default: last date)

        Returns:
            dict with daily_ret, equity, weights, metrics,
            active_strategies, exposure, turnover
        """
        signals = self._compute_all_signals()
        seg_df = signals["segment"]
        curv_df = signals["curvature"]
        overlay_df = signals["overlay"]

        rm = self.returns_matrix
        dates = rm.index
        T, N = rm.shape
        ret_np = rm.values
        col_names = rm.columns.tolist()

        # Determine backtest range
        start_idx = (dates.searchsorted(pd.Timestamp(start_date))
                     if start_date else 0)
        end_idx = (dates.searchsorted(pd.Timestamp(end_date), side="right")
                   if end_date else T)

        bt_len = end_idx - start_idx
        bt_dates = dates[start_idx:end_idx]

        # Storage
        weights_history = np.zeros((bt_len, N))
        port_returns = np.zeros(bt_len)
        turnover_arr = np.zeros(bt_len)
        exposure_arr = np.zeros(bt_len)

        current_w = np.zeros(N)
        prev_w = np.zeros(N)
        running_peak = 1.0
        equity_val = 1.0

        for bt_t in range(bt_len):
            t = start_idx + bt_t  # index into full data

            # === Rebalance ===
            should_rebalance = (bt_t % self.rebalance_freq == 0)

            if should_rebalance and t > 0:
                sig_t = t - 1  # no look-ahead

                seg_row = seg_df.iloc[sig_t]
                curv_row = curv_df.iloc[sig_t]
                overlay_row = overlay_df.iloc[sig_t]

                has_signals = not seg_row.isna().all()

                if has_signals:
                    w = signal_to_weights(
                        seg_row, curv_row, overlay_row,
                        col_names,
                        signal_threshold=self.signal_threshold,
                        w_slope_mom=self.w_slope_mom,
                        w_slope_level=self.w_slope_level,
                        w_rr=self.w_rr,
                        max_strategies=self.max_strategies,
                    )

                    # Vol targeting with Ledoit-Wolf
                    cov_window = min(t, 504)
                    if cov_window >= 63 and w.sum() > 1e-10:
                        cov = estimate_covariance(ret_np[t - cov_window:t])
                        vol_scale = self._vol_scale(w, cov)
                    else:
                        vol_scale = 1.0

                    # Drawdown control
                    dd_scale = 1.0
                    if bt_t > 0:
                        current_dd = equity_val / running_peak - 1
                        if current_dd < self.dd_threshold:
                            dd_severity = min(1.0,
                                              abs(current_dd / self.dd_threshold) - 1.0)
                            dd_scale = (1.0
                                        - dd_severity * (1.0 - self.dd_reduction))

                    current_w = w * vol_scale * dd_scale
                else:
                    current_w = np.zeros(N)

            # Record exposure
            exposure_arr[bt_t] = np.sum(np.abs(current_w))

            # Transaction costs
            turn = np.sum(np.abs(current_w - prev_w))
            turnover_arr[bt_t] = turn
            tcost = turn * self.tcost_bps / 10000.0

            # Portfolio return
            port_returns[bt_t] = current_w @ ret_np[t] - tcost
            equity_val *= (1 + port_returns[bt_t])
            running_peak = max(running_peak, equity_val)

            weights_history[bt_t] = current_w
            prev_w = current_w.copy()

        # Build output
        equity = pd.Series(np.cumprod(1.0 + port_returns), index=bt_dates,
                           name="signal_portfolio")
        daily_ret = pd.Series(port_returns, index=bt_dates,
                              name="signal_portfolio")
        weights_df = pd.DataFrame(weights_history, index=bt_dates,
                                  columns=col_names)
        turnover_s = pd.Series(turnover_arr, index=bt_dates, name="turnover")
        exposure_s = pd.Series(exposure_arr, index=bt_dates, name="exposure")

        metrics = compute_metrics(daily_ret)

        active_counts = (np.abs(weights_history) > 1e-6).sum(axis=1)
        active_s = pd.Series(active_counts, index=bt_dates, name="n_active")

        self._backtest_result = {
            "daily_ret": daily_ret,
            "equity": equity,
            "weights": weights_df,
            "metrics": metrics,
            "turnover": turnover_s,
            "exposure": exposure_s,
            "active_strategies": active_s,
        }
        return self._backtest_result

    def _fast_is_sharpe(self, end_date=None):
        """Fast IS Sharpe computation without vol targeting.

        Skips Ledoit-Wolf covariance for speed during grid search.
        Returns scalar Sharpe ratio.
        """
        signals = self._signals
        seg_df = signals["segment"]
        curv_df = signals["curvature"]
        overlay_df = signals["overlay"]

        rm = self.returns_matrix
        dates = rm.index
        T, N = rm.shape
        ret_np = rm.values
        col_names = rm.columns.tolist()

        end_idx = T
        if end_date:
            end_idx = dates.searchsorted(pd.Timestamp(end_date), side="right")

        port_returns = np.zeros(end_idx)
        current_w = np.zeros(N)
        prev_w = np.zeros(N)

        for t in range(end_idx):
            if t % self.rebalance_freq == 0 and t > 0:
                sig_t = t - 1
                seg_row = seg_df.iloc[sig_t]
                curv_row = curv_df.iloc[sig_t]
                overlay_row = overlay_df.iloc[sig_t]

                if not seg_row.isna().all():
                    current_w = signal_to_weights(
                        seg_row, curv_row, overlay_row,
                        col_names,
                        signal_threshold=self.signal_threshold,
                        w_slope_mom=self.w_slope_mom,
                        w_slope_level=self.w_slope_level,
                        w_rr=self.w_rr,
                        max_strategies=self.max_strategies,
                    )

            # Transaction costs
            turn = np.sum(np.abs(current_w - prev_w))
            tcost = turn * self.tcost_bps / 10000.0

            port_returns[t] = current_w @ ret_np[t] - tcost
            prev_w = current_w.copy()

        if end_idx < 63:
            return -999.0

        rf_d = RISK_FREE_RATE / ANN
        exc = port_returns[:end_idx] - rf_d
        return np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)

    def walk_forward(self, start_year=2012, end_year=2025):
        """Walk-forward validation with parameter grid search.

        For each OOS year:
          1. IS: run parameter grid on [data_start, Dec(year-1)]
          2. Select best IS params (highest Sharpe)
          3. OOS: full backtest on year with best params

        Grid:
          signal_threshold: [0.3, 0.5, 0.7, 1.0]
          rebalance_freq:   [5, 10, 21]
          max_strategies:   [4, 6, 8, 10]
          w_slope_mom:      [0.4, 0.5, 0.6, 0.7]
          Total: 4 x 3 x 4 x 4 = 192 configs per year

        Args:
            start_year: first OOS year
            end_year: last OOS year

        Returns:
            DataFrame with year, sharpe, ann_ret, vol, mdd, hit_rate, n_days,
            is_sharpe, best_* params
        """
        self._compute_all_signals()

        # Parameter grid
        thresholds = [0.3, 0.5, 0.7, 1.0]
        rebal_freqs = [5, 10, 21]
        max_strats = [4, 6, 8, 10]
        w_moms = [0.4, 0.5, 0.6, 0.7]

        n_configs = len(thresholds) * len(rebal_freqs) * len(max_strats) * len(w_moms)

        results = []

        for test_year in range(start_year, end_year + 1):
            train_end = f"{test_year - 1}-12-31"
            test_start = f"{test_year}-01-01"
            test_end = f"{test_year}-12-31"

            # Check OOS data exists
            test_mask = (
                (self.returns_matrix.index >= pd.Timestamp(test_start)) &
                (self.returns_matrix.index <= pd.Timestamp(test_end))
            )
            if test_mask.sum() < 20:
                continue

            print(f"  WF year {test_year}: grid search ({n_configs} configs) ...",
                  end="", flush=True)

            # === Grid search on IS (fast, no vol targeting) ===
            best_sharpe = -999
            best_params = {
                "signal_threshold": 0.5,
                "rebalance_freq": 21,
                "max_strategies": 8,
                "w_slope_mom": 0.6,
                "w_slope_level": 0.2,
                "w_rr": 0.2,
            }

            for thresh in thresholds:
                for rebal in rebal_freqs:
                    for max_s in max_strats:
                        for w_m in w_moms:
                            w_l = (1.0 - w_m) / 2  # split remainder
                            w_r = (1.0 - w_m) / 2

                            sys_is = CurveSignalSystem(
                                returns_matrix=self.returns_matrix,
                                yields=self.yields,
                                rr_1m=self.rr_1m,
                                move=self.move,
                                vix=self.vix,
                                z_window=self.z_window,
                                mom_window=self.mom_window,
                                signal_threshold=thresh,
                                w_slope_mom=w_m,
                                w_slope_level=w_l,
                                w_rr=w_r,
                                max_strategies=max_s,
                                rebalance_freq=rebal,
                                target_vol=self.target_vol,
                                max_leverage=self.max_leverage,
                                dd_threshold=self.dd_threshold,
                                dd_reduction=self.dd_reduction,
                                tcost_bps=self.tcost_bps,
                            )
                            # Share pre-computed signals
                            sys_is._signals = self._signals

                            is_sharpe = sys_is._fast_is_sharpe(end_date=train_end)

                            if is_sharpe > best_sharpe:
                                best_sharpe = is_sharpe
                                best_params = {
                                    "signal_threshold": thresh,
                                    "rebalance_freq": rebal,
                                    "max_strategies": max_s,
                                    "w_slope_mom": w_m,
                                    "w_slope_level": w_l,
                                    "w_rr": w_r,
                                }

            print(f" IS={best_sharpe:.3f}", end="")

            # === OOS test with best params (full backtest) ===
            sys_oos = CurveSignalSystem(
                returns_matrix=self.returns_matrix,
                yields=self.yields,
                rr_1m=self.rr_1m,
                move=self.move,
                vix=self.vix,
                z_window=self.z_window,
                mom_window=self.mom_window,
                signal_threshold=best_params["signal_threshold"],
                w_slope_mom=best_params["w_slope_mom"],
                w_slope_level=best_params["w_slope_level"],
                w_rr=best_params["w_rr"],
                max_strategies=best_params["max_strategies"],
                rebalance_freq=best_params["rebalance_freq"],
                target_vol=self.target_vol,
                max_leverage=self.max_leverage,
                dd_threshold=self.dd_threshold,
                dd_reduction=self.dd_reduction,
                tcost_bps=self.tcost_bps,
            )
            sys_oos._signals = self._signals

            oos_result = sys_oos.backtest(start_date=test_start,
                                          end_date=test_end)
            m = oos_result["metrics"]

            print(f" -> OOS Sharpe={m.get('sharpe', np.nan):.3f}, "
                  f"Ret={m.get('ann_ret', np.nan):+.2%}")

            results.append({
                "year": test_year,
                "sharpe": m.get("sharpe", np.nan),
                "ann_ret": m.get("ann_ret", np.nan),
                "vol": m.get("vol", np.nan),
                "mdd": m.get("mdd", np.nan),
                "hit_rate": m.get("hit_rate", np.nan),
                "n_days": m.get("n_days", 0),
                "is_sharpe": best_sharpe,
                "best_threshold": best_params["signal_threshold"],
                "best_rebal": best_params["rebalance_freq"],
                "best_max_strats": best_params["max_strategies"],
                "best_w_mom": best_params["w_slope_mom"],
            })

        return pd.DataFrame(results)

    def explain_signals(self):
        """Show which segments are active and their signal strength per day.

        Returns:
            DataFrame with date, n_active, active_strategies, weights,
            rr_z, move_z, exposure
        """
        if self._backtest_result is None:
            return pd.DataFrame()

        signals = self._compute_all_signals()
        overlay_df = signals["overlay"]
        weights_df = self._backtest_result["weights"]
        col_names = weights_df.columns.tolist()

        rows = []
        for date in weights_df.index:
            w = weights_df.loc[date].values
            active_mask = np.abs(w) > 1e-6
            if not active_mask.any():
                continue

            active_strats = [col_names[i] for i in range(len(col_names))
                             if active_mask[i]]
            active_weights = [w[i] for i in range(len(col_names))
                              if active_mask[i]]

            overlay_row = (overlay_df.loc[date]
                           if date in overlay_df.index
                           else pd.Series(dtype=float))

            rows.append({
                "date": date,
                "n_active": len(active_strats),
                "active_strategies": ", ".join(active_strats),
                "weights": ", ".join(f"{x:.3f}" for x in active_weights),
                "rr_z": _safe_get(overlay_row, "rr_z", np.nan),
                "move_z": _safe_get(overlay_row, "move_z", np.nan),
                "exposure": np.sum(np.abs(w)),
            })

        return pd.DataFrame(rows)
