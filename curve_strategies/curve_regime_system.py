"""
curve_regime_system.py - Multi-mode regime-adaptive strategy selection.

Runs 6 signal modes (MR_level, TF_level, MOM_TF, MOM_MR, BLEND_TF, BLEND_MR)
in parallel, selects the best mode based on rolling performance + YC regime
filter, and goes flat when no mode is profitable.

Key insight: no single mode works in all regimes. MOM_MR works in BEAR regimes,
while BULL regimes are best avoided. The adaptive system exploits this by
selecting the best-performing mode dynamically.

Usage:
    from curve_strategies.curve_regime_system import CurveRegimeSystem
"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RISK_FREE_RATE
from curve_strategies.curve_signal_system import (
    CurveSignalSystem,
    compute_segment_signals,
    compute_curvature_signals,
    compute_global_overlays,
    signal_to_weights,
    estimate_covariance,
    _safe_get,
)
from curve_strategies.regime_analysis import compute_yc_regime
from curve_strategies.strategy_backtest import compute_metrics

ANN = 252


# ══════════════════════════════════════════════════════════════════════════
# SIGNAL MODE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════

SIGNAL_MODES = {
    "MR_level": {
        "description": "Contrarian z_level",
        "level_sign": -1, "mom_sign": 0,
        "w_level": 0.8, "w_mom": 0.0, "w_rr": 0.2,
    },
    "TF_level": {
        "description": "Trend z_level",
        "level_sign": +1, "mom_sign": 0,
        "w_level": 0.8, "w_mom": 0.0, "w_rr": 0.2,
    },
    "MOM_TF": {
        "description": "Trend z_momentum",
        "level_sign": 0, "mom_sign": +1,
        "w_level": 0.0, "w_mom": 0.8, "w_rr": 0.2,
    },
    "MOM_MR": {
        "description": "Contrarian z_momentum",
        "level_sign": 0, "mom_sign": -1,
        "w_level": 0.0, "w_mom": 0.8, "w_rr": 0.2,
    },
    "BLEND_TF": {
        "description": "Combined trend",
        "level_sign": +1, "mom_sign": +1,
        "w_level": 0.4, "w_mom": 0.4, "w_rr": 0.2,
    },
    "BLEND_MR": {
        "description": "Combined contrarian",
        "level_sign": -1, "mom_sign": -1,
        "w_level": 0.4, "w_mom": 0.4, "w_rr": 0.2,
    },
}


# ══════════════════════════════════════════════════════════════════════════
# CURVE REGIME SYSTEM CLASS
# ══════════════════════════════════════════════════════════════════════════

class CurveRegimeSystem:
    """Multi-mode regime-adaptive yield curve strategy selection.

    Runs all 6 signal modes in parallel, selects the best based on
    rolling performance and YC regime filter, and goes flat when no
    mode is profitable.
    """

    def __init__(self, returns_matrix, yields, rr_1m, move, vix,
                 strategy_pool=None,
                 # Signal params
                 z_window=252, mom_window=63, signal_threshold=0.5,
                 max_strategies=8, rebalance_freq=21,
                 # Risk management
                 target_vol=0.15, max_leverage=2.0,
                 dd_threshold=-0.10, dd_reduction=0.5, tcost_bps=5,
                 # Adaptive params
                 lookback_window=126, min_mode_sharpe=0.0,
                 regime_filter=True):

        self.returns_matrix = returns_matrix
        self.yields = yields
        self.rr_1m = rr_1m
        self.move = move
        self.vix = vix
        self.strategy_pool = strategy_pool

        self.z_window = z_window
        self.mom_window = mom_window
        self.signal_threshold = signal_threshold
        self.max_strategies = max_strategies
        self.rebalance_freq = rebalance_freq

        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.dd_threshold = dd_threshold
        self.dd_reduction = dd_reduction
        self.tcost_bps = tcost_bps

        self.lookback_window = lookback_window
        self.min_mode_sharpe = min_mode_sharpe
        self.regime_filter = regime_filter

        # Caches
        self._signals = None
        self._mode_returns = None
        self._yc_regime = None

    # ──────────────────────────────────────────────────────────────────────
    # SIGNAL COMPUTATION
    # ──────────────────────────────────────────────────────────────────────

    def _compute_all_signals(self):
        """Compute and cache segment + curvature + overlay signals."""
        if self._signals is not None:
            return self._signals

        seg_signals = compute_segment_signals(
            self.yields, z_window=self.z_window, mom_window=self.mom_window)
        curv_signals = compute_curvature_signals(
            self.yields, z_window=self.z_window, mom_window=self.mom_window)
        overlay = compute_global_overlays(
            self.rr_1m, self.move, self.vix, z_window=self.z_window)

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

    def _compute_yc_regime(self):
        """Compute and cache YC regime classification."""
        if self._yc_regime is not None:
            return self._yc_regime
        self._yc_regime = compute_yc_regime(self.yields)
        self._yc_regime = self._yc_regime.reindex(self.returns_matrix.index,
                                                   method="ffill")
        return self._yc_regime

    # ──────────────────────────────────────────────────────────────────────
    # MODE RETURNS COMPUTATION
    # ──────────────────────────────────────────────────────────────────────

    def compute_all_mode_returns(self):
        """Compute daily returns for each of the 6 signal modes.

        Fast computation (no vol targeting) - uses signal_to_weights with
        mode-specific level_sign, mom_sign, and weight parameters.

        Returns:
            dict of {mode_name: pd.Series of daily returns}
        """
        if self._mode_returns is not None:
            return self._mode_returns

        signals = self._compute_all_signals()
        seg_df = signals["segment"]
        curv_df = signals["curvature"]
        overlay_df = signals["overlay"]

        rm = self.returns_matrix
        dates = rm.index
        T, N = rm.shape
        ret_np = rm.values
        col_names = rm.columns.tolist()

        mode_returns = {}

        for mode_name, mode_cfg in SIGNAL_MODES.items():
            port_returns = np.zeros(T)
            current_w = np.zeros(N)
            prev_w = np.zeros(N)

            for t in range(T):
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
                            w_slope_mom=mode_cfg["w_mom"],
                            w_slope_level=mode_cfg["w_level"],
                            w_rr=mode_cfg["w_rr"],
                            max_strategies=self.max_strategies,
                            level_sign=mode_cfg["level_sign"],
                            mom_sign=mode_cfg["mom_sign"],
                        )

                # Transaction costs
                turn = np.sum(np.abs(current_w - prev_w))
                tcost = turn * self.tcost_bps / 10000.0

                port_returns[t] = current_w @ ret_np[t] - tcost
                prev_w = current_w.copy()

            mode_returns[mode_name] = pd.Series(
                port_returns, index=dates, name=mode_name)

        self._mode_returns = mode_returns
        return mode_returns

    # ──────────────────────────────────────────────────────────────────────
    # MODE-REGIME PERFORMANCE MATRIX
    # ──────────────────────────────────────────────────────────────────────

    def compute_mode_regime_matrix(self, yc_regime=None):
        """Build Sharpe matrix: rows = 6 modes, columns = YC regimes.

        Args:
            yc_regime: pd.Series of YC regime labels. If None, computes it.

        Returns:
            pd.DataFrame with Sharpe for each (mode, regime) pair
        """
        mode_returns = self.compute_all_mode_returns()

        if yc_regime is None:
            yc_regime = self._compute_yc_regime()

        regimes = sorted(yc_regime.unique())
        rf_d = RISK_FREE_RATE / ANN

        rows = []
        for mode_name, mode_ret in mode_returns.items():
            common = mode_ret.index.intersection(yc_regime.index)
            mr = mode_ret.reindex(common)
            yr = yc_regime.reindex(common)

            row = {"mode": mode_name}
            for reg in regimes:
                mask = yr == reg
                r = mr[mask].values
                if len(r) < 20:
                    row[reg] = np.nan
                    continue
                exc = r - rf_d
                sharpe = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)
                row[reg] = sharpe

            # Also compute full-sample Sharpe
            exc_full = mr.values - rf_d
            row["FULL"] = (np.mean(exc_full) / (np.std(exc_full) + 1e-10)
                           * np.sqrt(ANN))
            rows.append(row)

        matrix = pd.DataFrame(rows).set_index("mode")
        return matrix

    # ──────────────────────────────────────────────────────────────────────
    # ADAPTIVE BACKTEST
    # ──────────────────────────────────────────────────────────────────────

    def adaptive_backtest(self, start_date=None, end_date=None):
        """Run adaptive multi-mode backtest.

        At each rebalance:
          1. Compute rolling Sharpe for each mode (lookback_window)
          2. If regime_filter: check YC regime, reduce exposure in BULL
          3. Select mode with best rolling Sharpe (if > min_mode_sharpe)
          4. If no mode > threshold -> go flat
          5. Apply vol targeting + drawdown control

        Args:
            start_date: backtest start (default: first date)
            end_date: backtest end (default: last date)

        Returns:
            dict with daily_ret, equity, weights, metrics,
            mode_timeline, exposure, turnover
        """
        mode_returns = self.compute_all_mode_returns()
        signals = self._compute_all_signals()
        seg_df = signals["segment"]
        curv_df = signals["curvature"]
        overlay_df = signals["overlay"]
        yc_regime = self._compute_yc_regime()

        rm = self.returns_matrix
        dates = rm.index
        T, N = rm.shape
        ret_np = rm.values
        col_names = rm.columns.tolist()

        # Convert mode returns to numpy for fast rolling computation
        mode_names = list(SIGNAL_MODES.keys())
        n_modes = len(mode_names)
        mode_ret_np = np.column_stack(
            [mode_returns[m].values for m in mode_names])

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
        mode_timeline = ["FLAT"] * bt_len

        current_w = np.zeros(N)
        prev_w = np.zeros(N)
        running_peak = 1.0
        equity_val = 1.0
        active_mode = "FLAT"

        rf_d = RISK_FREE_RATE / ANN

        for bt_t in range(bt_len):
            t = start_idx + bt_t

            # === Rebalance ===
            should_rebalance = (bt_t % self.rebalance_freq == 0)

            if should_rebalance and t > 0:
                sig_t = t - 1

                # 1. Compute rolling Sharpe for each mode
                lb_start = max(0, t - self.lookback_window)
                rolling_sharpes = np.full(n_modes, -999.0)

                if t - lb_start >= 21:
                    for mi in range(n_modes):
                        window_ret = mode_ret_np[lb_start:t, mi]
                        exc = window_ret - rf_d
                        std = np.std(exc)
                        if std > 1e-10:
                            rolling_sharpes[mi] = (np.mean(exc) / std
                                                   * np.sqrt(ANN))

                # 2. Regime filter
                exposure_scale = 1.0
                if self.regime_filter:
                    if t < len(yc_regime):
                        current_regime = yc_regime.iloc[t]
                        if (isinstance(current_regime, str)
                                and current_regime.startswith("BULL")):
                            exposure_scale = 0.3

                # 3. Select best mode
                best_idx = np.argmax(rolling_sharpes)
                best_sharpe = rolling_sharpes[best_idx]

                if best_sharpe < self.min_mode_sharpe:
                    # No mode is good -> go flat
                    current_w = np.zeros(N)
                    active_mode = "FLAT"
                else:
                    active_mode = mode_names[best_idx]
                    mode_cfg = SIGNAL_MODES[active_mode]

                    seg_row = seg_df.iloc[sig_t]
                    curv_row = curv_df.iloc[sig_t]
                    overlay_row = overlay_df.iloc[sig_t]

                    has_signals = not seg_row.isna().all()

                    if has_signals:
                        w = signal_to_weights(
                            seg_row, curv_row, overlay_row,
                            col_names,
                            signal_threshold=self.signal_threshold,
                            w_slope_mom=mode_cfg["w_mom"],
                            w_slope_level=mode_cfg["w_level"],
                            w_rr=mode_cfg["w_rr"],
                            max_strategies=self.max_strategies,
                            level_sign=mode_cfg["level_sign"],
                            mom_sign=mode_cfg["mom_sign"],
                        )

                        # Apply regime exposure scale
                        w *= exposure_scale

                        # Vol targeting with Ledoit-Wolf
                        cov_window = min(t, 504)
                        if cov_window >= 63 and w.sum() > 1e-10:
                            cov = estimate_covariance(ret_np[t - cov_window:t])
                            port_vol = np.sqrt(w @ cov @ w) * np.sqrt(ANN)
                            if port_vol > 1e-8:
                                vol_scale = min(self.target_vol / port_vol,
                                                self.max_leverage)
                            else:
                                vol_scale = 1.0
                        else:
                            vol_scale = 1.0

                        # Drawdown control
                        dd_scale = 1.0
                        if bt_t > 0:
                            current_dd = equity_val / running_peak - 1
                            if current_dd < self.dd_threshold:
                                dd_severity = min(
                                    1.0,
                                    abs(current_dd / self.dd_threshold) - 1.0)
                                dd_scale = (1.0 - dd_severity
                                            * (1.0 - self.dd_reduction))

                        current_w = w * vol_scale * dd_scale
                    else:
                        current_w = np.zeros(N)
                        active_mode = "FLAT"

            mode_timeline[bt_t] = active_mode

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
                           name="adaptive_regime")
        daily_ret = pd.Series(port_returns, index=bt_dates,
                              name="adaptive_regime")
        weights_df = pd.DataFrame(weights_history, index=bt_dates,
                                  columns=col_names)
        turnover_s = pd.Series(turnover_arr, index=bt_dates, name="turnover")
        exposure_s = pd.Series(exposure_arr, index=bt_dates, name="exposure")
        mode_s = pd.Series(mode_timeline, index=bt_dates, name="mode")

        metrics = compute_metrics(daily_ret)

        return {
            "daily_ret": daily_ret,
            "equity": equity,
            "weights": weights_df,
            "metrics": metrics,
            "mode_timeline": mode_s,
            "exposure": exposure_s,
            "turnover": turnover_s,
        }

    # ──────────────────────────────────────────────────────────────────────
    # FAST IS SHARPE (for grid search)
    # ──────────────────────────────────────────────────────────────────────

    def _fast_adaptive_sharpe(self, end_date=None):
        """Fast adaptive Sharpe without vol targeting (for grid search).

        Returns scalar Sharpe ratio.
        """
        mode_returns = self._mode_returns
        if mode_returns is None:
            mode_returns = self.compute_all_mode_returns()

        signals = self._signals
        yc_regime = self._yc_regime
        if yc_regime is None:
            yc_regime = self._compute_yc_regime()

        rm = self.returns_matrix
        dates = rm.index
        T, N = rm.shape
        ret_np = rm.values
        col_names = rm.columns.tolist()

        mode_names = list(SIGNAL_MODES.keys())
        n_modes = len(mode_names)
        mode_ret_np = np.column_stack(
            [mode_returns[m].values for m in mode_names])

        end_idx = T
        if end_date:
            end_idx = dates.searchsorted(pd.Timestamp(end_date), side="right")

        seg_df = signals["segment"]
        curv_df = signals["curvature"]
        overlay_df = signals["overlay"]

        rf_d = RISK_FREE_RATE / ANN
        port_returns = np.zeros(end_idx)
        current_w = np.zeros(N)
        prev_w = np.zeros(N)

        for t in range(end_idx):
            if t % self.rebalance_freq == 0 and t > 0:
                sig_t = t - 1

                # Rolling Sharpe for each mode
                lb_start = max(0, t - self.lookback_window)
                rolling_sharpes = np.full(n_modes, -999.0)

                if t - lb_start >= 21:
                    for mi in range(n_modes):
                        window_ret = mode_ret_np[lb_start:t, mi]
                        exc = window_ret - rf_d
                        std = np.std(exc)
                        if std > 1e-10:
                            rolling_sharpes[mi] = (np.mean(exc) / std
                                                   * np.sqrt(ANN))

                # Regime filter
                exposure_scale = 1.0
                if self.regime_filter:
                    if t < len(yc_regime):
                        cr = yc_regime.iloc[t]
                        if isinstance(cr, str) and cr.startswith("BULL"):
                            exposure_scale = 0.3

                best_idx = np.argmax(rolling_sharpes)
                best_sharpe = rolling_sharpes[best_idx]

                if best_sharpe < self.min_mode_sharpe:
                    current_w = np.zeros(N)
                else:
                    mode_cfg = SIGNAL_MODES[mode_names[best_idx]]
                    seg_row = seg_df.iloc[sig_t]
                    curv_row = curv_df.iloc[sig_t]
                    overlay_row = overlay_df.iloc[sig_t]

                    if not seg_row.isna().all():
                        current_w = signal_to_weights(
                            seg_row, curv_row, overlay_row,
                            col_names,
                            signal_threshold=self.signal_threshold,
                            w_slope_mom=mode_cfg["w_mom"],
                            w_slope_level=mode_cfg["w_level"],
                            w_rr=mode_cfg["w_rr"],
                            max_strategies=self.max_strategies,
                            level_sign=mode_cfg["level_sign"],
                            mom_sign=mode_cfg["mom_sign"],
                        )
                        current_w *= exposure_scale
                    else:
                        current_w = np.zeros(N)

            # Transaction costs
            turn = np.sum(np.abs(current_w - prev_w))
            tcost = turn * self.tcost_bps / 10000.0

            port_returns[t] = current_w @ ret_np[t] - tcost
            prev_w = current_w.copy()

        if end_idx < 63:
            return -999.0

        exc = port_returns[:end_idx] - rf_d
        return np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)

    # ──────────────────────────────────────────────────────────────────────
    # WALK-FORWARD VALIDATION
    # ──────────────────────────────────────────────────────────────────────

    def walk_forward(self, start_year=2012, end_year=2025):
        """Walk-forward validation with parameter grid search.

        For each OOS year:
          1. IS: compute mode_regime_matrix on [start, Dec(year-1)]
          2. IS: calibrate min_mode_sharpe, lookback_window on reduced grid
          3. OOS: apply adaptive_backtest on test year

        Grid (54 configs):
          lookback_window: [63, 126, 252]
          min_mode_sharpe: [-0.5, 0.0, 0.3]
          signal_threshold: [0.3, 0.5, 0.7]
          regime_filter: [True, False]

        Args:
            start_year: first OOS year
            end_year: last OOS year

        Returns:
            DataFrame with year, sharpe, ann_ret, vol, mdd, hit_rate,
            is_sharpe, best_params, dominant_mode
        """
        self._compute_all_signals()
        self.compute_all_mode_returns()
        self._compute_yc_regime()

        # Parameter grid
        lookback_windows = [63, 126, 252]
        min_sharpes = [-0.5, 0.0, 0.3]
        thresholds = [0.3, 0.5, 0.7]
        regime_filters = [True, False]

        n_configs = (len(lookback_windows) * len(min_sharpes)
                     * len(thresholds) * len(regime_filters))

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

            print(f"  WF year {test_year}: grid search ({n_configs} configs)"
                  f" ...", end="", flush=True)

            # === Grid search on IS ===
            best_sharpe = -999
            best_params = {
                "lookback_window": 126,
                "min_mode_sharpe": 0.0,
                "signal_threshold": 0.5,
                "regime_filter": True,
            }

            for lb in lookback_windows:
                for ms in min_sharpes:
                    for thresh in thresholds:
                        for rf in regime_filters:
                            sys_is = CurveRegimeSystem(
                                returns_matrix=self.returns_matrix,
                                yields=self.yields,
                                rr_1m=self.rr_1m,
                                move=self.move,
                                vix=self.vix,
                                z_window=self.z_window,
                                mom_window=self.mom_window,
                                signal_threshold=thresh,
                                max_strategies=self.max_strategies,
                                rebalance_freq=self.rebalance_freq,
                                target_vol=self.target_vol,
                                max_leverage=self.max_leverage,
                                dd_threshold=self.dd_threshold,
                                dd_reduction=self.dd_reduction,
                                tcost_bps=self.tcost_bps,
                                lookback_window=lb,
                                min_mode_sharpe=ms,
                                regime_filter=rf,
                            )
                            # Share pre-computed caches
                            sys_is._signals = self._signals
                            sys_is._mode_returns = self._mode_returns
                            sys_is._yc_regime = self._yc_regime

                            is_sharpe = sys_is._fast_adaptive_sharpe(
                                end_date=train_end)

                            if is_sharpe > best_sharpe:
                                best_sharpe = is_sharpe
                                best_params = {
                                    "lookback_window": lb,
                                    "min_mode_sharpe": ms,
                                    "signal_threshold": thresh,
                                    "regime_filter": rf,
                                }

            print(f" IS={best_sharpe:.3f}", end="")

            # === OOS test with best params ===
            sys_oos = CurveRegimeSystem(
                returns_matrix=self.returns_matrix,
                yields=self.yields,
                rr_1m=self.rr_1m,
                move=self.move,
                vix=self.vix,
                z_window=self.z_window,
                mom_window=self.mom_window,
                signal_threshold=best_params["signal_threshold"],
                max_strategies=self.max_strategies,
                rebalance_freq=self.rebalance_freq,
                target_vol=self.target_vol,
                max_leverage=self.max_leverage,
                dd_threshold=self.dd_threshold,
                dd_reduction=self.dd_reduction,
                tcost_bps=self.tcost_bps,
                lookback_window=best_params["lookback_window"],
                min_mode_sharpe=best_params["min_mode_sharpe"],
                regime_filter=best_params["regime_filter"],
            )
            sys_oos._signals = self._signals
            sys_oos._mode_returns = self._mode_returns
            sys_oos._yc_regime = self._yc_regime

            oos_result = sys_oos.adaptive_backtest(
                start_date=test_start, end_date=test_end)
            m = oos_result["metrics"]

            # Dominant mode in OOS
            mode_tl = oos_result["mode_timeline"]
            mode_counts = mode_tl.value_counts()
            dominant = (mode_counts.index[0]
                        if len(mode_counts) > 0 else "FLAT")

            print(f" -> OOS Sharpe={m.get('sharpe', np.nan):.3f}, "
                  f"Ret={m.get('ann_ret', np.nan):+.2%}, "
                  f"Mode={dominant}")

            results.append({
                "year": test_year,
                "sharpe": m.get("sharpe", np.nan),
                "ann_ret": m.get("ann_ret", np.nan),
                "vol": m.get("vol", np.nan),
                "mdd": m.get("mdd", np.nan),
                "hit_rate": m.get("hit_rate", np.nan),
                "n_days": m.get("n_days", 0),
                "is_sharpe": best_sharpe,
                "dominant_mode": dominant,
                "best_lookback": best_params["lookback_window"],
                "best_min_sharpe": best_params["min_mode_sharpe"],
                "best_threshold": best_params["signal_threshold"],
                "best_regime_filter": best_params["regime_filter"],
            })

        return pd.DataFrame(results)

    # ──────────────────────────────────────────────────────────────────────
    # EXPLAIN
    # ──────────────────────────────────────────────────────────────────────

    def explain(self, backtest_result=None):
        """Explain daily decisions: active mode, rolling Sharpe, regime, weights.

        Args:
            backtest_result: output from adaptive_backtest(). If None,
                runs adaptive_backtest() first.

        Returns:
            DataFrame with date, active_mode, rolling_sharpe per mode,
            yc_regime, n_active_strategies, exposure
        """
        if backtest_result is None:
            backtest_result = self.adaptive_backtest()

        mode_returns = self.compute_all_mode_returns()
        yc_regime = self._compute_yc_regime()

        mode_timeline = backtest_result["mode_timeline"]
        weights_df = backtest_result["weights"]
        exposure = backtest_result["exposure"]

        mode_names = list(SIGNAL_MODES.keys())
        rf_d = RISK_FREE_RATE / ANN

        rows = []
        for date in mode_timeline.index:
            row = {
                "date": date,
                "active_mode": mode_timeline[date],
            }

            # YC regime
            if date in yc_regime.index:
                row["yc_regime"] = yc_regime[date]
            else:
                row["yc_regime"] = "UNKNOWN"

            # Rolling Sharpe for each mode (lookback_window)
            for mn in mode_names:
                mr = mode_returns[mn]
                loc = mr.index.get_loc(date)
                lb_start = max(0, loc - self.lookback_window)
                if loc - lb_start >= 21:
                    window_ret = mr.values[lb_start:loc]
                    exc = window_ret - rf_d
                    std = np.std(exc)
                    if std > 1e-10:
                        row[f"sharpe_{mn}"] = (np.mean(exc) / std
                                               * np.sqrt(ANN))
                    else:
                        row[f"sharpe_{mn}"] = 0.0
                else:
                    row[f"sharpe_{mn}"] = np.nan

            # Active strategies
            w = weights_df.loc[date].values
            active_mask = np.abs(w) > 1e-6
            row["n_active"] = int(active_mask.sum())
            row["exposure"] = exposure[date]

            if active_mask.any():
                col_names = weights_df.columns.tolist()
                active_strats = [col_names[i] for i in range(len(col_names))
                                 if active_mask[i]]
                row["active_strategies"] = ", ".join(active_strats)
            else:
                row["active_strategies"] = ""

            rows.append(row)

        return pd.DataFrame(rows)
