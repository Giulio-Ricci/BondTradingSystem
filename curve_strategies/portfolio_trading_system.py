"""
portfolio_trading_system.py - Combined portfolio trading system.

Operates on 2 time scales:
    - Slow (21 days): re-estimate Markowitz weights with expanding window
    - Fast (daily): indicator-based tilt adjustments

Walk-forward validation optimizes only meta-parameters (tilt_strength,
top_n, indicator weights), while Markowitz weights are re-estimated
within each backtest via expanding window.
"""

import numpy as np
import pandas as pd
from scipy.special import softmax

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE

from curve_strategies.markowitz_optimizer import (
    estimate_covariance, optimize_min_variance,
    optimize_max_sharpe, optimize_risk_parity,
)
from curve_strategies.trading_indicators import (
    compute_all_indicators, compute_raw_indicators,
    compute_composite_score,
)
from curve_strategies.strategy_backtest import compute_metrics

ANN = 252


# Preset indicator weight sets for walk-forward grid search
INDICATOR_WEIGHT_PRESETS = [
    # Preset 0: default (balanced)
    {"z_mom_126": 0.30, "sharpe_126": 0.25, "vol_z": 0.15,
     "consensus": 0.15, "z_mom_252": 0.15},
    # Preset 1: momentum-heavy
    {"z_mom_126": 0.40, "sharpe_126": 0.15, "vol_z": 0.10,
     "consensus": 0.10, "z_mom_252": 0.25},
    # Preset 2: sharpe-heavy
    {"z_mom_126": 0.20, "sharpe_126": 0.40, "vol_z": 0.10,
     "consensus": 0.15, "z_mom_252": 0.15},
    # Preset 3: vol-focused
    {"z_mom_126": 0.20, "sharpe_126": 0.20, "vol_z": 0.35,
     "consensus": 0.10, "z_mom_252": 0.15},
    # Preset 4: consensus-focused
    {"z_mom_126": 0.20, "sharpe_126": 0.20, "vol_z": 0.10,
     "consensus": 0.35, "z_mom_252": 0.15},
    # Preset 5: short-term momentum
    {"z_mom_126": 0.45, "sharpe_126": 0.20, "vol_z": 0.10,
     "consensus": 0.10, "z_mom_252": 0.15},
    # Preset 6: long-term momentum
    {"z_mom_126": 0.15, "sharpe_126": 0.20, "vol_z": 0.10,
     "consensus": 0.15, "z_mom_252": 0.40},
    # Preset 7: no vol signal
    {"z_mom_126": 0.30, "sharpe_126": 0.30, "vol_z": 0.00,
     "consensus": 0.20, "z_mom_252": 0.20},
]


class PortfolioTradingSystem:
    """Combined Markowitz + indicator-tilt trading system.

    Blends Markowitz optimization (slow, 21-day rebalance) with
    indicator-based tilt (fast, daily) for strategy selection.
    """

    def __init__(self, returns_matrix: pd.DataFrame,
                 strategy_pool: list = None,
                 markowitz_method: str = "max_sharpe",
                 indicator_weights: dict = None,
                 min_estimation_window: int = 504,
                 rebalance_freq: int = 21,
                 tilt_strength: float = 0.3,
                 top_n_strategies: int = 10,
                 tcost_bps: float = 5):
        """
        Args:
            returns_matrix: (T x N) daily strategy returns
            strategy_pool: list of strategy dicts (for indicator computation)
            markowitz_method: "min_variance", "max_sharpe", or "risk_parity"
            indicator_weights: weights for composite score
            min_estimation_window: min days before Markowitz estimation begins
            rebalance_freq: rebalance Markowitz every N days
            tilt_strength: blend ratio (0=pure Markowitz, 1=pure indicators)
            top_n_strategies: keep only top N strategies by weight
            tcost_bps: transaction cost in basis points
        """
        self.returns_matrix = returns_matrix
        self.strategy_pool = strategy_pool
        self.markowitz_method = markowitz_method
        self.indicator_weights = indicator_weights
        self.min_estimation_window = min_estimation_window
        self.rebalance_freq = rebalance_freq
        self.tilt_strength = tilt_strength
        self.top_n_strategies = top_n_strategies
        self.tcost_bps = tcost_bps

        self._raw_indicators = None  # cached raw (independent of weights)
        self._indicators = None
        self._backtest_result = None
        self._markowitz_cache = None  # precomputed Markowitz weights at each rebalance

    def _get_markowitz_weights(self, ret_sub: np.ndarray) -> np.ndarray:
        """Compute Markowitz weights from a sub-array of returns."""
        mu = ret_sub.mean(axis=0)
        cov = estimate_covariance(pd.DataFrame(ret_sub), method="ledoit_wolf")
        N = len(mu)

        if self.markowitz_method == "min_variance":
            return optimize_min_variance(mu, cov)
        elif self.markowitz_method == "max_sharpe":
            return optimize_max_sharpe(mu, cov)
        elif self.markowitz_method == "risk_parity":
            return optimize_risk_parity(cov)
        else:
            return np.ones(N) / N

    def _apply_top_n_filter(self, w: np.ndarray) -> np.ndarray:
        """Keep only top_n strategies by weight, renormalize."""
        n = self.top_n_strategies
        if n >= len(w):
            return w

        # Zero out all but top n weights
        threshold_idx = np.argsort(w)[:-n]
        w_filtered = w.copy()
        w_filtered[threshold_idx] = 0.0

        total = w_filtered.sum()
        if total > 1e-10:
            w_filtered /= total
        else:
            w_filtered = np.ones(len(w)) / len(w)

        return w_filtered

    def backtest(self, start_date: str = None, end_date: str = None) -> dict:
        """Run full backtest with Markowitz + indicator tilt.

        Daily loop:
            1. Every rebalance_freq days: re-estimate Markowitz weights
            2. Every day: compute tilt from indicators (info at t-1 only)
            3. final_w = (1-tilt)*markowitz_w + tilt*softmax(scores)
            4. Apply concentration limit (top_n)
            5. Transaction costs on turnover
            6. port_ret = final_w @ strategy_returns[t]

        Args:
            start_date: backtest start (default: beginning of data)
            end_date: backtest end (default: end of data)

        Returns:
            dict with keys:
                daily_ret, equity, weights, markowitz_weights,
                indicator_scores, metrics, turnover
        """
        rm = self.returns_matrix
        if start_date is not None:
            rm = rm.loc[rm.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            rm = rm.loc[rm.index <= pd.Timestamp(end_date)]

        T, N = rm.shape
        dates = rm.index
        ret_np = rm.values

        # Precompute indicators on the full available data
        if self._indicators is None:
            if self._raw_indicators is None:
                self._raw_indicators = compute_raw_indicators(
                    self.returns_matrix, self.strategy_pool, verbose=True
                )
            self._indicators = compute_all_indicators(
                self.returns_matrix, self.strategy_pool, self.indicator_weights,
                raw_cache=self._raw_indicators, verbose=False,
            )

        composite_scores = self._indicators["composite_scores"]

        # Align scores to backtest dates
        scores_aligned = composite_scores.reindex(index=dates, columns=rm.columns)

        # Storage
        weights_history = np.zeros((T, N))
        markowitz_weights_hist = np.zeros((T, N))
        scores_history = np.zeros((T, N))
        port_returns = np.zeros(T)
        turnover = np.zeros(T)

        current_markowitz_w = np.ones(N) / N
        prev_w = np.ones(N) / N

        min_win = self.min_estimation_window

        # Use full returns_matrix for Markowitz (expanding window from data start)
        full_ret_np = self.returns_matrix.values
        full_dates = self.returns_matrix.index

        # Map backtest dates to positions in full returns matrix
        date_to_full_idx = {d: i for i, d in enumerate(full_dates)}

        for t in range(T):
            # 1. Markowitz rebalance (every rebalance_freq days)
            full_idx = date_to_full_idx.get(dates[t], t)
            if full_idx >= min_win and (full_idx - min_win) % self.rebalance_freq == 0:
                # Check cache first
                if self._markowitz_cache is not None and full_idx in self._markowitz_cache:
                    current_markowitz_w = self._markowitz_cache[full_idx].copy()
                else:
                    current_markowitz_w = self._get_markowitz_weights(
                        full_ret_np[:full_idx, :]
                    )
                    if self._markowitz_cache is not None:
                        self._markowitz_cache[full_idx] = current_markowitz_w.copy()

            markowitz_w = current_markowitz_w.copy()
            markowitz_weights_hist[t] = markowitz_w

            # 2. Indicator tilt (using info at t-1, no look-ahead)
            if t > 0 and not scores_aligned.iloc[t - 1].isna().all():
                raw_scores = scores_aligned.iloc[t - 1].values
                raw_scores = np.nan_to_num(raw_scores, nan=0.0)
                scores_history[t] = raw_scores

                # Softmax tilt weights (temperature=1.0)
                tilt_w = softmax(raw_scores)
            else:
                tilt_w = np.ones(N) / N
                scores_history[t] = 0.0

            # 3. Blend
            tilt_s = self.tilt_strength
            final_w = (1.0 - tilt_s) * markowitz_w + tilt_s * tilt_w

            # 4. Concentration filter
            final_w = self._apply_top_n_filter(final_w)

            # 5. Transaction costs
            turn = np.sum(np.abs(final_w - prev_w))
            turnover[t] = turn
            tcost = turn * self.tcost_bps / 10000.0

            # 6. Portfolio return
            port_returns[t] = final_w @ ret_np[t] - tcost

            weights_history[t] = final_w
            prev_w = final_w.copy()

        # Build output
        col_names = rm.columns.tolist()
        equity = pd.Series(np.cumprod(1.0 + port_returns), index=dates,
                           name="portfolio_system")
        daily_ret = pd.Series(port_returns, index=dates, name="portfolio_system")
        weights_df = pd.DataFrame(weights_history, index=dates, columns=col_names)
        markowitz_df = pd.DataFrame(markowitz_weights_hist, index=dates, columns=col_names)
        scores_df = pd.DataFrame(scores_history, index=dates, columns=col_names)
        turnover_s = pd.Series(turnover, index=dates, name="turnover")

        metrics = compute_metrics(daily_ret)

        self._backtest_result = {
            "daily_ret": daily_ret,
            "equity": equity,
            "weights": weights_df,
            "markowitz_weights": markowitz_df,
            "indicator_scores": scores_df,
            "metrics": metrics,
            "turnover": turnover_s,
        }
        return self._backtest_result

    def walk_forward(self, start_year: int = 2012,
                     end_year: int = 2025) -> pd.DataFrame:
        """Walk-forward validation of meta-parameters.

        For each test year:
            1. Train: all data up to Dec(year-1)
            2. Grid search on train set:
                - tilt_strength: [0.0, 0.1, 0.2, 0.3, 0.5]
                - top_n: [5, 8, 10, 15, 25]
                - indicator weight presets: 8 presets
            3. Best params -> test on year OOS
            4. Collect OOS metrics

        Note: Markowitz weights are NOT optimized in WF (they use
        expanding window within each backtest). Only meta-parameters
        are WF-validated.

        Args:
            start_year: first OOS year
            end_year: last OOS year

        Returns:
            pd.DataFrame with columns:
                year, sharpe, ann_ret, vol, mdd, hit_rate, n_days,
                best_tilt, best_top_n, best_preset
        """
        tilt_grid = [0.0, 0.1, 0.2, 0.3, 0.5]
        topn_grid = [5, 8, 10, 15, 25]

        # Compute raw indicators ONCE (they don't depend on weights)
        print("    Computing raw indicators (once) ...", flush=True)
        if self._raw_indicators is None:
            self._raw_indicators = compute_raw_indicators(
                self.returns_matrix, self.strategy_pool, verbose=True
            )
        raw = self._raw_indicators

        # Pre-compute composite scores for each preset (only the
        # composite depends on weights, raw indicators are shared)
        print("    Pre-computing composite scores for all presets ...", flush=True)
        preset_indicators = {}
        for preset_idx, preset in enumerate(INDICATOR_WEIGHT_PRESETS):
            scores, ranks = compute_composite_score(
                raw["momentum"], raw["sharpe"], raw["vol"], raw["consensus"],
                weights=preset,
            )
            preset_indicators[preset_idx] = {
                **raw,
                "composite_scores": scores,
                "composite_ranks": ranks,
            }

        # Pre-compute Markowitz weights (they depend only on data and method,
        # NOT on meta-parameters like tilt_strength or top_n)
        print("    Pre-computing Markowitz weights (once) ...", flush=True)
        markowitz_cache = {}
        full_ret_np = self.returns_matrix.values
        T_full = len(full_ret_np)
        min_win = self.min_estimation_window
        for t_idx in range(min_win, T_full):
            if (t_idx - min_win) % self.rebalance_freq == 0:
                mu = full_ret_np[:t_idx].mean(axis=0)
                cov = estimate_covariance(
                    pd.DataFrame(full_ret_np[:t_idx]), method="ledoit_wolf"
                )
                if self.markowitz_method == "min_variance":
                    w = optimize_min_variance(mu, cov)
                elif self.markowitz_method == "max_sharpe":
                    w = optimize_max_sharpe(mu, cov)
                elif self.markowitz_method == "risk_parity":
                    w = optimize_risk_parity(cov)
                else:
                    w = np.ones(len(mu)) / len(mu)
                markowitz_cache[t_idx] = w
        print(f"    Pre-computed {len(markowitz_cache)} Markowitz rebalance points")

        results = []

        for test_year in range(start_year, end_year + 1):
            train_end = f"{test_year - 1}-12-31"
            test_start = f"{test_year}-01-01"
            test_end = f"{test_year}-12-31"

            # Check we have test data
            test_data = self.returns_matrix.loc[
                (self.returns_matrix.index >= pd.Timestamp(test_start)) &
                (self.returns_matrix.index <= pd.Timestamp(test_end))
            ]
            if len(test_data) < 20:
                continue

            print(f"    WF year {test_year}: grid search ...", end="", flush=True)

            # Grid search on train period
            best_sharpe = -np.inf
            best_params = (0.3, 10, 0)

            for tilt_s in tilt_grid:
                for top_n in topn_grid:
                    for preset_idx in range(len(INDICATOR_WEIGHT_PRESETS)):
                        sys_train = PortfolioTradingSystem(
                            returns_matrix=self.returns_matrix,
                            strategy_pool=self.strategy_pool,
                            markowitz_method=self.markowitz_method,
                            indicator_weights=INDICATOR_WEIGHT_PRESETS[preset_idx],
                            min_estimation_window=self.min_estimation_window,
                            rebalance_freq=self.rebalance_freq,
                            tilt_strength=tilt_s,
                            top_n_strategies=top_n,
                            tcost_bps=self.tcost_bps,
                        )
                        # Share all precomputed caches (no recomputation!)
                        sys_train._raw_indicators = raw
                        sys_train._indicators = preset_indicators[preset_idx]
                        sys_train._markowitz_cache = markowitz_cache

                        try:
                            res = sys_train.backtest(end_date=train_end)
                            train_sharpe = res["metrics"].get("sharpe", -np.inf)
                            if np.isnan(train_sharpe):
                                train_sharpe = -np.inf
                        except Exception:
                            train_sharpe = -np.inf

                        if train_sharpe > best_sharpe:
                            best_sharpe = train_sharpe
                            best_params = (tilt_s, top_n, preset_idx)

            # OOS test with best params
            best_tilt, best_topn, best_preset_idx = best_params

            sys_test = PortfolioTradingSystem(
                returns_matrix=self.returns_matrix,
                strategy_pool=self.strategy_pool,
                markowitz_method=self.markowitz_method,
                indicator_weights=INDICATOR_WEIGHT_PRESETS[best_preset_idx],
                min_estimation_window=self.min_estimation_window,
                rebalance_freq=self.rebalance_freq,
                tilt_strength=best_tilt,
                top_n_strategies=best_topn,
                tcost_bps=self.tcost_bps,
            )
            sys_test._raw_indicators = raw
            sys_test._indicators = preset_indicators[best_preset_idx]
            sys_test._markowitz_cache = markowitz_cache

            oos_result = sys_test.backtest(start_date=test_start, end_date=test_end)
            m = oos_result["metrics"]

            print(f" best=(tilt={best_tilt}, top_n={best_topn}, "
                  f"preset={best_preset_idx}) -> OOS Sharpe={m.get('sharpe', np.nan):.3f}")

            results.append({
                "year": test_year,
                "sharpe": m.get("sharpe", np.nan),
                "ann_ret": m.get("ann_ret", np.nan),
                "vol": m.get("vol", np.nan),
                "mdd": m.get("mdd", np.nan),
                "hit_rate": m.get("hit_rate", np.nan),
                "n_days": m.get("n_days", 0),
                "best_tilt": best_tilt,
                "best_top_n": best_topn,
                "best_preset": best_preset_idx,
            })

        return pd.DataFrame(results)

    def explain_performance(self) -> pd.DataFrame:
        """Decompose portfolio returns into component contributions.

        Decomposes returns into:
            1. Markowitz contribution (pure Markowitz weights)
            2. Indicator tilt contribution (tilt effect)
            3. Transaction cost drag
            4. Selection value (system vs equal-weight)

        Returns:
            pd.DataFrame with annual breakdown, columns:
                year, total_ret, markowitz_contrib, tilt_contrib,
                tcost_drag, ew_ret, selection_value, best_indicator
        """
        if self._backtest_result is None:
            raise RuntimeError("Must run backtest() before explain_performance()")

        res = self._backtest_result
        daily_ret = res["daily_ret"]
        weights = res["weights"]
        markowitz_w = res["markowitz_weights"]
        scores = res["indicator_scores"]
        turnover = res["turnover"]

        rm = self.returns_matrix.reindex(index=daily_ret.index,
                                         columns=weights.columns)
        ret_np = rm.values

        # Daily Markowitz-only return
        markowitz_daily = np.sum(markowitz_w.values * ret_np, axis=1)
        # Daily EW return
        ew_daily = ret_np.mean(axis=1)
        # Daily tcost
        tcost_daily = turnover.values * self.tcost_bps / 10000.0

        years = sorted(daily_ret.index.year.unique())
        rows = []

        for y in years:
            mask = daily_ret.index.year == y
            if mask.sum() < 5:
                continue

            dr = daily_ret.values[mask]
            mk_dr = markowitz_daily[mask]
            ew_dr = ew_daily[mask]
            tc_dr = tcost_daily[mask]

            total_ret = np.prod(1 + dr) - 1
            markowitz_ret = np.prod(1 + mk_dr) - 1
            ew_ret = np.prod(1 + ew_dr) - 1
            tcost_drag = -np.sum(tc_dr)

            tilt_contrib = total_ret - markowitz_ret - tcost_drag
            selection_value = total_ret - ew_ret

            # Determine which indicator was most predictive
            yr_scores = scores.values[mask]
            yr_ret = ret_np[mask]
            # Correlation between lagged scores and next-day returns
            best_ind = "N/A"
            if len(yr_scores) > 10 and self._indicators is not None:
                ind_names = ["z_mom_126", "sharpe_126", "vol_z", "consensus", "z_mom_252"]
                best_corr = -np.inf
                for ind_name in ind_names:
                    if ind_name in self._indicators.get("momentum", {}):
                        ind_df = self._indicators["momentum"][ind_name]
                    elif ind_name in self._indicators.get("sharpe", {}):
                        ind_df = self._indicators["sharpe"][ind_name]
                    elif ind_name in self._indicators.get("vol", {}):
                        ind_df = self._indicators["vol"][ind_name]
                    elif ind_name == "consensus":
                        ind_df = self._indicators.get("consensus", pd.DataFrame())
                    else:
                        continue

                    if ind_df.empty:
                        continue

                    yr_ind = ind_df.reindex(index=daily_ret.index[mask],
                                           columns=weights.columns)
                    # Cross-sectional IC: correlation between rank and next-day return
                    ic_vals = []
                    for t in range(len(yr_ind) - 1):
                        ind_row = yr_ind.iloc[t].values
                        ret_row = yr_ret[t + 1]
                        valid = ~(np.isnan(ind_row) | np.isnan(ret_row))
                        if valid.sum() > 5:
                            ic = np.corrcoef(ind_row[valid], ret_row[valid])[0, 1]
                            if not np.isnan(ic):
                                ic_vals.append(ic)
                    avg_ic = np.mean(ic_vals) if ic_vals else -np.inf
                    if avg_ic > best_corr:
                        best_corr = avg_ic
                        best_ind = ind_name

            rows.append({
                "year": y,
                "total_ret": total_ret,
                "markowitz_contrib": markowitz_ret,
                "tilt_contrib": tilt_contrib,
                "tcost_drag": tcost_drag,
                "ew_ret": ew_ret,
                "selection_value": selection_value,
                "best_indicator": best_ind,
            })

        return pd.DataFrame(rows)
