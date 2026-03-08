"""
ml_optimizer.py - ML-enhanced portfolio optimization.

Uses GradientBoosting to predict forward strategy returns, combined with:
    - Regime-conditional features (YC + HMM regime)
    - Macro features (VIX, MOVE, yield curve shape)
    - Strategy-specific features (momentum, Sharpe, vol)
    - Volatility targeting and drawdown control
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE

from curve_strategies.strategy_backtest import compute_metrics
from curve_strategies.markowitz_optimizer import (
    estimate_covariance, optimize_min_variance,
    optimize_max_sharpe, optimize_risk_parity,
)

ANN = 252
FORWARD_WINDOW = 63  # predict 63-day forward return (quarterly horizon)


# ══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

def build_ml_features(returns_matrix: pd.DataFrame,
                      yields: pd.DataFrame = None,
                      vix: pd.Series = None,
                      move: pd.Series = None) -> pd.DataFrame:
    """Build feature matrix for ML prediction.

    Features per strategy (cross-sectional, at each time t):
        - mom_5, mom_21, mom_63, mom_126: rolling cumulative return
        - vol_21, vol_63: rolling volatility
        - sharpe_63: rolling Sharpe
        - vol_ratio: short/long vol ratio
        - dd_current: current drawdown from peak

    Global features (same for all strategies at time t):
        - vix_level, vix_z: VIX and its z-score
        - move_level, move_z: MOVE and its z-score
        - slope_2s10s, slope_z: yield curve slope and z-score
        - curve_mom: 21d change in slope
        - level_change: 21d change in yield level

    Args:
        returns_matrix: (T x N) daily strategy returns
        yields: yield curve DataFrame (optional)
        vix: VIX Series (optional)
        move: MOVE Series (optional)

    Returns:
        dict with:
            'strategy_features': dict of {name: DataFrame(T, n_features)}
            'global_features': DataFrame(T, n_global_features)
            'feature_names': list of all feature names
    """
    T, N = returns_matrix.shape
    col_names = returns_matrix.columns.tolist()

    # === Strategy-specific features (reduced set to prevent overfitting) ===
    strategy_features = {}
    for col in col_names:
        ret = returns_matrix[col]
        feats = pd.DataFrame(index=returns_matrix.index)

        # Momentum (only 2 horizons: medium and long)
        feats["mom_63"] = ret.rolling(63, min_periods=63).sum()
        feats["mom_252"] = ret.rolling(252, min_periods=126).sum()

        # Volatility (one window)
        feats["vol_63"] = ret.rolling(63, min_periods=63).std() * np.sqrt(ANN)

        # Current drawdown
        equity = (1 + ret).cumprod()
        peak = equity.cummax()
        feats["dd_current"] = (equity - peak) / peak

        strategy_features[col] = feats

    # === Global/macro features (reduced set for robustness) ===
    global_feats = pd.DataFrame(index=returns_matrix.index)

    # Market momentum (average across strategies)
    avg_ret = returns_matrix.mean(axis=1)
    global_feats["mkt_mom_63"] = avg_ret.rolling(63, min_periods=63).sum()

    if vix is not None:
        vix_aligned = vix.reindex(returns_matrix.index)
        vix_mean = vix_aligned.rolling(252, min_periods=63).mean()
        vix_std = vix_aligned.rolling(252, min_periods=63).std()
        global_feats["vix_z"] = (vix_aligned - vix_mean) / (vix_std + 1e-10)

    if yields is not None:
        yld = yields.reindex(returns_matrix.index)
        if "US_2Y" in yld.columns and "US_10Y" in yld.columns:
            slope = yld["US_10Y"] - yld["US_2Y"]
            global_feats["slope_2s10s"] = slope

            level = (yld["US_2Y"] + yld["US_10Y"]) / 2
            global_feats["level_change_63"] = level.diff(63)

        if "US_5Y" in yld.columns and "US_2Y" in yld.columns and "US_10Y" in yld.columns:
            global_feats["curvature"] = 2 * yld["US_5Y"] - yld["US_2Y"] - yld["US_10Y"]

    strat_feat_names = list(strategy_features[col_names[0]].columns)
    global_feat_names = list(global_feats.columns)

    return {
        "strategy_features": strategy_features,
        "global_features": global_feats,
        "feature_names": strat_feat_names + global_feat_names,
        "strat_feat_names": strat_feat_names,
        "global_feat_names": global_feat_names,
    }


def build_target(returns_matrix: pd.DataFrame,
                 forward_window: int = FORWARD_WINDOW) -> pd.DataFrame:
    """Build forward return target for ML training.

    target_i(t) = sum(r_i[t+1:t+forward_window+1])

    Uses vectorized rolling sum with shift for speed.

    Args:
        returns_matrix: (T x N) daily returns
        forward_window: forward horizon in days

    Returns:
        DataFrame (T x N) of forward cumulative returns (NaN at end)
    """
    # Forward rolling sum: shift returns backwards, compute rolling sum
    # cumsum approach for O(1) per element
    cumsum = returns_matrix.cumsum()
    # target[t] = cumsum[t+fw] - cumsum[t]  (sum of r[t+1:t+fw+1])
    target = cumsum.shift(-forward_window) - cumsum
    return target


# ══════════════════════════════════════════════════════════════════════════
# ML RETURN PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

class MLReturnPredictor:
    """Ridge regression predictor for forward strategy returns.

    Uses Ridge (L2-regularized linear model) to prevent overfitting.
    Trains one model per strategy. Features = strategy-specific + global macro.
    Target = 63-day forward cumulative return.
    """

    def __init__(self, alpha: float = 10.0, **kwargs):
        self.alpha = alpha
        self.models_ = {}
        self.scalers_ = {}
        self.feature_names_ = []

    def fit(self, features: dict, target: pd.DataFrame,
            train_end: str = None):
        """Train one Ridge model per strategy.

        Args:
            features: output from build_ml_features()
            target: output from build_target()
            train_end: end date for training data
        """
        strat_feats = features["strategy_features"]
        global_feats = features["global_features"]
        self.feature_names_ = features["feature_names"]

        for col in target.columns:
            if col not in strat_feats:
                continue

            # Combine strategy-specific and global features
            X = pd.concat([strat_feats[col], global_feats], axis=1)
            y = target[col]

            # Filter to training period
            if train_end is not None:
                mask = X.index <= pd.Timestamp(train_end)
                X = X[mask]
                y = y[mask]

            # Drop NaN rows
            valid = X.notna().all(axis=1) & y.notna()
            X = X[valid]
            y = y[valid]

            if len(X) < 200:
                continue

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)

            # Train Ridge regression (strong regularization)
            model = Ridge(alpha=self.alpha, fit_intercept=True)
            model.fit(X_scaled, y.values)

            self.models_[col] = model
            self.scalers_[col] = scaler

    def predict(self, features: dict, date_idx: int = None,
                dates: pd.DatetimeIndex = None) -> pd.Series:
        """Predict forward returns for all strategies at a given date.

        Args:
            features: output from build_ml_features()
            date_idx: integer index into the feature DataFrames
            dates: if provided, predict for these dates (returns DataFrame)

        Returns:
            pd.Series of predicted forward returns (one per strategy)
        """
        strat_feats = features["strategy_features"]
        global_feats = features["global_features"]
        cols = list(strat_feats.keys())

        if dates is not None:
            # Predict for multiple dates
            preds = pd.DataFrame(np.nan, index=dates, columns=cols)
            for col in cols:
                if col not in self.models_:
                    continue
                X = pd.concat([strat_feats[col], global_feats], axis=1)
                X = X.reindex(dates)
                valid = X.notna().all(axis=1)
                if valid.sum() == 0:
                    continue
                X_valid = X[valid]
                X_scaled = self.scalers_[col].transform(X_valid.values)
                preds.loc[valid, col] = self.models_[col].predict(X_scaled)
            return preds

        # Single date prediction
        predictions = {}
        for col in cols:
            if col not in self.models_:
                predictions[col] = 0.0
                continue

            X = pd.concat([strat_feats[col].iloc[[date_idx]],
                           global_feats.iloc[[date_idx]]], axis=1)

            if X.isna().any(axis=1).iloc[0]:
                predictions[col] = 0.0
                continue

            X_scaled = self.scalers_[col].transform(X.values)
            predictions[col] = self.models_[col].predict(X_scaled)[0]

        return pd.Series(predictions)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get average feature importance across all strategy models.

        For Ridge, uses absolute coefficient values as importance.
        """
        if not self.models_:
            return pd.DataFrame()

        importances = {}
        for col, model in self.models_.items():
            importances[col] = np.abs(model.coef_)

        imp_df = pd.DataFrame(importances, index=self.feature_names_)
        # Normalize to sum to 1 per strategy
        col_sums = imp_df.sum(axis=0)
        imp_df = imp_df.div(col_sums + 1e-10, axis=1)
        imp_df["mean"] = imp_df.mean(axis=1)
        return imp_df.sort_values("mean", ascending=False)


# ══════════════════════════════════════════════════════════════════════════
# ML-ENHANCED PORTFOLIO SYSTEM
# ══════════════════════════════════════════════════════════════════════════

class MLPortfolioSystem:
    """Portfolio system using ML predictions + Markowitz + risk management.

    Improvements over basic system:
        1. ML-predicted returns replace historical mean as mu input
        2. Prediction-weighted allocation (softmax on positive predictions)
        3. Volatility targeting (scales exposure to target vol)
        4. Drawdown control (reduces exposure during drawdowns)
        5. Adaptive cash allocation based on prediction confidence
        6. Markowitz covariance for diversification
    """

    def __init__(self, returns_matrix: pd.DataFrame,
                 strategy_pool: list = None,
                 yields: pd.DataFrame = None,
                 vix: pd.Series = None,
                 move: pd.Series = None,
                 markowitz_method: str = "max_sharpe",
                 min_train_window: int = 504,
                 retrain_freq: int = 63,
                 rebalance_freq: int = 21,
                 target_vol: float = 0.15,
                 max_leverage: float = 2.0,
                 dd_threshold: float = -0.10,
                 dd_reduction: float = 0.5,
                 top_n_strategies: int = 10,
                 allow_short: bool = False,
                 tcost_bps: float = 5,
                 ml_weight: float = 0.5,
                 n_estimators: int = 100):
        """
        Args:
            returns_matrix: (T x N) daily strategy returns
            strategy_pool: list of strategy dicts
            yields: yield data for features
            vix: VIX series for features
            move: MOVE series for features
            markowitz_method: optimization method
            min_train_window: minimum days before ML starts
            retrain_freq: retrain ML every N days
            rebalance_freq: rebalance portfolio every N days
            target_vol: target annualized portfolio volatility
            max_leverage: maximum total exposure
            dd_threshold: drawdown level to trigger risk reduction
            dd_reduction: exposure multiplier when in drawdown
            top_n_strategies: keep only top N strategies
            allow_short: allow negative weights
            tcost_bps: transaction cost in basis points
            ml_weight: blend weight for ML predictions vs Markowitz
            n_estimators: number of trees in GBM
        """
        self.returns_matrix = returns_matrix
        self.strategy_pool = strategy_pool
        self.yields = yields
        self.vix = vix
        self.move = move
        self.markowitz_method = markowitz_method
        self.min_train_window = min_train_window
        self.retrain_freq = retrain_freq
        self.rebalance_freq = rebalance_freq
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.dd_threshold = dd_threshold
        self.dd_reduction = dd_reduction
        self.top_n_strategies = top_n_strategies
        self.allow_short = allow_short
        self.tcost_bps = tcost_bps
        self.ml_weight = ml_weight
        self.n_estimators = n_estimators

        self._features = None
        self._target = None
        self._backtest_result = None
        self._injected_predictor = None  # For walk-forward: pre-trained predictor

    def _build_features_and_target(self):
        """Build features and target matrices once."""
        if self._features is None:
            self._features = build_ml_features(
                self.returns_matrix, self.yields, self.vix, self.move
            )
        if self._target is None:
            self._target = build_target(self.returns_matrix)

    def _vol_scale(self, weights: np.ndarray, cov: np.ndarray) -> float:
        """Compute scaling factor to achieve target volatility."""
        port_vol = np.sqrt(weights @ cov @ weights) * np.sqrt(ANN)
        if port_vol < 1e-8:
            return 1.0
        scale = self.target_vol / port_vol
        return min(scale, self.max_leverage)

    def _prediction_weights(self, pred_values: np.ndarray,
                             cov: np.ndarray) -> np.ndarray:
        """Convert ML predictions to portfolio weights.

        Strategy:
        1. Select top_n strategies with highest predictions
        2. Allocate proportionally to prediction magnitude (softmax)
        3. Apply Markowitz diversification adjustment
        4. Scale to target volatility

        Args:
            pred_values: (N,) array of predicted forward returns
            cov: (N, N) covariance matrix

        Returns:
            (N,) weight array (before vol scaling)
        """
        N = len(pred_values)

        # Select top_n strategies
        top_n = min(self.top_n_strategies, N)
        sorted_idx = np.argsort(pred_values)[::-1]  # descending by prediction
        top_idx = sorted_idx[:top_n]

        # Softmax on top predictions for smooth allocation
        top_preds = pred_values[top_idx]
        # Temperature controls concentration: lower = more concentrated
        temperature = max(np.std(top_preds) + 1e-8, 0.001)
        exp_preds = np.exp(top_preds / temperature)
        softmax_w = exp_preds / (exp_preds.sum() + 1e-10)

        w = np.zeros(N)
        w[top_idx] = softmax_w

        # Blend with Markowitz for diversification
        if self.ml_weight < 1.0:
            try:
                mu_hat = pred_values.copy()
                w_mkw = optimize_max_sharpe(mu_hat, cov,
                                            risk_free=RISK_FREE_RATE / ANN)
                w = self.ml_weight * w + (1 - self.ml_weight) * w_mkw
                # Re-normalize
                total_abs = np.sum(np.abs(w))
                if total_abs > 1e-10:
                    w /= total_abs
            except Exception:
                pass

        return w

    def backtest(self, start_date: str = None, end_date: str = None) -> dict:
        """Run ML-enhanced backtest.

        Args:
            start_date: backtest start
            end_date: backtest end

        Returns:
            dict with daily_ret, equity, weights, metrics, ml_predictions,
            exposure, turnover
        """
        self._build_features_and_target()

        rm = self.returns_matrix
        dates = rm.index
        T, N = rm.shape
        ret_np = rm.values
        col_names = rm.columns.tolist()

        # Determine backtest range within the full data
        if start_date is not None:
            start_idx = dates.searchsorted(pd.Timestamp(start_date))
        else:
            start_idx = 0
        if end_date is not None:
            end_idx = dates.searchsorted(pd.Timestamp(end_date), side="right")
        else:
            end_idx = T

        bt_len = end_idx - start_idx
        bt_dates = dates[start_idx:end_idx]

        # Storage
        weights_history = np.zeros((bt_len, N))
        pred_history = np.zeros((bt_len, N))
        port_returns = np.zeros(bt_len)
        turnover = np.zeros(bt_len)
        exposure_hist = np.zeros(bt_len)

        current_w = np.zeros(N)
        prev_w = np.zeros(N)
        # Use injected predictor if available (for walk-forward)
        current_predictor = self._injected_predictor
        running_peak = 1.0
        equity_val = 1.0

        for bt_t in range(bt_len):
            t = start_idx + bt_t  # index into full data

            # === ML Training (retrain periodically) ===
            if (self._injected_predictor is None and
                    t >= self.min_train_window and
                    (t - self.min_train_window) % self.retrain_freq == 0):
                predictor = MLReturnPredictor(alpha=10.0)
                train_end = str(dates[t - 1].date())
                print(f"    ML retrain @ {train_end} (t={t}/{T}) ...",
                      end="", flush=True)
                predictor.fit(self._features, self._target, train_end=train_end)
                current_predictor = predictor
                print(" done", flush=True)

            # === Rebalance ===
            should_rebalance = (
                (t >= self.min_train_window and
                 (t - self.min_train_window) % self.rebalance_freq == 0)
                or (self._injected_predictor is not None and
                    bt_t % self.rebalance_freq == 0)
            )

            if should_rebalance:
                # Get ML predictions (no look-ahead: use features at t-1)
                if current_predictor is not None and t > 0:
                    ml_pred = current_predictor.predict(self._features, date_idx=t - 1)
                    pred_values = ml_pred.reindex(col_names).fillna(0.0).values
                else:
                    pred_values = np.zeros(N)

                pred_history[bt_t] = pred_values

                # Covariance (recent window for reactivity)
                cov_window = min(t, 504)
                cov_data = ret_np[t - cov_window:t]
                cov = estimate_covariance(pd.DataFrame(cov_data), method="ledoit_wolf")

                # Get weights from predictions
                if current_predictor is not None:
                    w = self._prediction_weights(pred_values, cov)
                else:
                    # Before ML: equal weight
                    w = np.ones(N) / N

                # === Volatility targeting ===
                vol_scale = self._vol_scale(w, cov)

                # === Drawdown control ===
                dd_scale = 1.0
                if bt_t > 0:
                    current_dd = equity_val / running_peak - 1
                    if current_dd < self.dd_threshold:
                        # Proportional reduction: deeper dd = more reduction
                        dd_severity = min(1.0, abs(current_dd / self.dd_threshold) - 1.0)
                        dd_scale = 1.0 - dd_severity * (1.0 - self.dd_reduction)

                # === Confidence-based cash allocation ===
                pred_scale = 1.0
                if current_predictor is not None:
                    # Use prediction magnitude as confidence signal
                    avg_pred = pred_values.mean()
                    max_pred = pred_values.max()
                    if max_pred <= 0:
                        # All predictions negative: scale by how negative
                        pred_scale = max(0.2, 1.0 + max_pred * 20)  # ~0.2-1.0
                    elif avg_pred < 0:
                        # Some positive, reduce slightly
                        pred_scale = max(0.5, 0.5 + (pred_values > 0).sum() / N)

                total_scale = vol_scale * dd_scale * pred_scale
                current_w = w * total_scale

            elif t < self.min_train_window and self._injected_predictor is None:
                # Before ML starts: equal weight with vol targeting
                current_w = np.ones(N) / N
                if t >= 63:
                    cov_data = ret_np[max(0, t - 252):t]
                    cov = estimate_covariance(pd.DataFrame(cov_data),
                                              method="ledoit_wolf")
                    ew_w = np.ones(N) / N
                    vol_scale = self._vol_scale(ew_w, cov)
                    current_w = ew_w * min(vol_scale, 1.0)

            # Record exposure
            exposure_hist[bt_t] = np.sum(np.abs(current_w))

            # Transaction costs
            turn = np.sum(np.abs(current_w - prev_w))
            turnover[bt_t] = turn
            tcost = turn * self.tcost_bps / 10000.0

            # Portfolio return
            port_returns[bt_t] = current_w @ ret_np[t] - tcost
            equity_val *= (1 + port_returns[bt_t])
            running_peak = max(running_peak, equity_val)

            weights_history[bt_t] = current_w
            prev_w = current_w.copy()

        # Build output
        equity = pd.Series(np.cumprod(1.0 + port_returns), index=bt_dates,
                           name="ml_portfolio")
        daily_ret = pd.Series(port_returns, index=bt_dates, name="ml_portfolio")
        weights_df = pd.DataFrame(weights_history, index=bt_dates, columns=col_names)
        pred_df = pd.DataFrame(pred_history, index=bt_dates, columns=col_names)
        turnover_s = pd.Series(turnover, index=bt_dates, name="turnover")
        exposure_s = pd.Series(exposure_hist, index=bt_dates, name="exposure")

        metrics = compute_metrics(daily_ret)

        self._backtest_result = {
            "daily_ret": daily_ret,
            "equity": equity,
            "weights": weights_df,
            "ml_predictions": pred_df,
            "metrics": metrics,
            "turnover": turnover_s,
            "exposure": exposure_s,
        }
        return self._backtest_result

    def walk_forward(self, start_year: int = 2012,
                     end_year: int = 2025) -> pd.DataFrame:
        """Walk-forward validation with ML.

        For each test year:
            1. Train ML on all data up to Dec(year-1)
            2. Inject pre-trained predictor into OOS system
            3. Test on year, collect OOS metrics

        Args:
            start_year: first OOS year
            end_year: last OOS year

        Returns:
            DataFrame with year, sharpe, ann_ret, vol, mdd, hit_rate, n_days
        """
        self._build_features_and_target()

        results = []

        for test_year in range(start_year, end_year + 1):
            train_end = f"{test_year - 1}-12-31"
            test_start = f"{test_year}-01-01"
            test_end = f"{test_year}-12-31"

            test_data = self.returns_matrix.loc[
                (self.returns_matrix.index >= pd.Timestamp(test_start)) &
                (self.returns_matrix.index <= pd.Timestamp(test_end))
            ]
            if len(test_data) < 20:
                continue

            print(f"    ML-WF year {test_year} ...", end="", flush=True)

            # Train ML predictor on all data up to train_end
            predictor = MLReturnPredictor(alpha=10.0)
            predictor.fit(self._features, self._target, train_end=train_end)

            # Build a fresh system for OOS test
            sys_test = MLPortfolioSystem(
                returns_matrix=self.returns_matrix,
                strategy_pool=self.strategy_pool,
                yields=self.yields,
                vix=self.vix,
                move=self.move,
                markowitz_method=self.markowitz_method,
                min_train_window=self.min_train_window,
                retrain_freq=999999,  # don't retrain in OOS
                rebalance_freq=self.rebalance_freq,
                target_vol=self.target_vol,
                max_leverage=self.max_leverage,
                dd_threshold=self.dd_threshold,
                dd_reduction=self.dd_reduction,
                top_n_strategies=self.top_n_strategies,
                allow_short=self.allow_short,
                tcost_bps=self.tcost_bps,
                ml_weight=self.ml_weight,
                n_estimators=self.n_estimators,
            )
            sys_test._features = self._features
            sys_test._target = self._target
            # CRITICAL: inject pre-trained predictor so it's used from day 1
            sys_test._injected_predictor = predictor

            # Run backtest for OOS period
            oos_result = sys_test.backtest(start_date=test_start, end_date=test_end)
            m = oos_result["metrics"]

            print(f" Sharpe={m.get('sharpe', np.nan):.3f}, "
                  f"Ret={m.get('ann_ret', np.nan):+.2%}")

            results.append({
                "year": test_year,
                "sharpe": m.get("sharpe", np.nan),
                "ann_ret": m.get("ann_ret", np.nan),
                "vol": m.get("vol", np.nan),
                "mdd": m.get("mdd", np.nan),
                "hit_rate": m.get("hit_rate", np.nan),
                "n_days": m.get("n_days", 0),
            })

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the last trained ML model."""
        if self._backtest_result is None:
            return pd.DataFrame()

        # Retrain on all data
        self._build_features_and_target()
        predictor = MLReturnPredictor(alpha=10.0)
        predictor.fit(self._features, self._target)
        return predictor.get_feature_importance()
