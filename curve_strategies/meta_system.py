"""
meta_system.py - Adaptive regime-based strategy selection system.

The MetaSystem selects or blends yield curve strategies based on the
current regime state, using historical performance as the guide.

Two modes:
    LOOKUP   - deterministic mapping: regime -> best Sharpe strategy
    ENSEMBLE - blend top-N strategies with Sharpe-proportional weights
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import RISK_FREE_RATE

ANN = 252


class MetaSystem:
    """Adaptive regime-based strategy selection.

    Args:
        strategy_returns: dict of {strategy_name: pd.Series of daily returns}
        regime_series: pd.Series of combined regime labels
        mode: 'LOOKUP' or 'ENSEMBLE'
        top_n: number of top strategies to blend in ENSEMBLE mode
        cooldown: min days between allocation changes
        tcost_bps: transaction cost per change (bps)
        min_obs: minimum regime observations for reliable Sharpe
    """

    def __init__(self, strategy_returns: dict, regime_series: pd.Series,
                 mode: str = "LOOKUP", top_n: int = 3, cooldown: int = 5,
                 tcost_bps: float = 5, min_obs: int = 20):
        self.strategy_returns = strategy_returns
        self.regime_series = regime_series
        self.mode = mode.upper()
        self.top_n = top_n
        self.cooldown = cooldown
        self.tcost_bps = tcost_bps
        self.min_obs = min_obs

        self.strategy_names = sorted(strategy_returns.keys())
        self.regime_allocations_ = {}
        self.fitted_ = False

    def fit(self, train_end=None):
        """Build the regime -> strategy mapping from training data.

        Args:
            train_end: if set, only use data up to this date for fitting
        """
        regime = self.regime_series
        if train_end is not None:
            regime = regime.loc[regime.index <= pd.Timestamp(train_end)]

        regimes = sorted(regime.unique())
        self.regime_allocations_ = {}

        for state in regimes:
            mask = regime == state
            dates_in_regime = regime.index[mask]
            n_obs = len(dates_in_regime)

            if n_obs < self.min_obs:
                # Default: equal weight across all strategies
                alloc = {s: 1.0 / len(self.strategy_names)
                         for s in self.strategy_names}
                self.regime_allocations_[state] = alloc
                continue

            # Compute Sharpe for each strategy in this regime
            sharpes = {}
            for name in self.strategy_names:
                dr = self.strategy_returns[name]
                r = dr.reindex(dates_in_regime).dropna()
                if len(r) < self.min_obs:
                    sharpes[name] = np.nan
                    continue
                rf_d = RISK_FREE_RATE / ANN
                exc = r.values - rf_d
                sharpes[name] = np.mean(exc) / (np.std(exc) + 1e-10) * np.sqrt(ANN)

            if self.mode == "LOOKUP":
                # Pick the single best strategy
                best = max(sharpes, key=lambda k: sharpes[k] if not np.isnan(sharpes[k]) else -999)
                alloc = {s: (1.0 if s == best else 0.0) for s in self.strategy_names}
            else:
                # ENSEMBLE: blend top_n with Sharpe-proportional weights
                valid = {k: v for k, v in sharpes.items() if not np.isnan(v)}
                if not valid:
                    alloc = {s: 1.0 / len(self.strategy_names)
                             for s in self.strategy_names}
                else:
                    sorted_strats = sorted(valid, key=lambda k: valid[k], reverse=True)
                    top = sorted_strats[:self.top_n]
                    # Use max(0, sharpe) to avoid negative weights
                    raw_w = {s: max(0, valid[s]) for s in top}
                    total_w = sum(raw_w.values())
                    if total_w < 1e-10:
                        alloc = {s: (1.0 / len(top) if s in top else 0.0)
                                 for s in self.strategy_names}
                    else:
                        alloc = {s: (raw_w.get(s, 0.0) / total_w)
                                 for s in self.strategy_names}

            self.regime_allocations_[state] = alloc

        self.fitted_ = True
        return self

    def get_regime_allocations(self) -> dict:
        """Return the regime -> allocation mapping for inspection."""
        return self.regime_allocations_

    def backtest(self, start=None, end=None) -> dict:
        """Run the meta-system backtest.

        Daily: check regime, if changed (with cooldown), update allocation.
        Returns portfolio daily returns, equity, and allocation history.
        """
        if not self.fitted_:
            raise RuntimeError("Must call .fit() before .backtest()")

        regime = self.regime_series
        if start is not None:
            regime = regime.loc[regime.index >= pd.Timestamp(start)]
        if end is not None:
            regime = regime.loc[regime.index <= pd.Timestamp(end)]

        dates = regime.index
        n = len(dates)
        if n == 0:
            return {"daily_ret": pd.Series(dtype=float),
                    "equity": pd.Series(dtype=float),
                    "allocations": pd.DataFrame(),
                    "metrics": {}}

        # Build aligned returns matrix
        ret_matrix = pd.DataFrame(index=dates)
        for name in self.strategy_names:
            ret_matrix[name] = self.strategy_returns[name].reindex(dates).fillna(0)

        # Simulate allocation changes
        portfolio_ret = np.zeros(n)
        alloc_history = np.zeros((n, len(self.strategy_names)))
        current_regime = None
        last_change = -self.cooldown - 1
        current_alloc = np.array([1.0 / len(self.strategy_names)] * len(self.strategy_names))

        for i in range(n):
            new_regime = regime.iloc[i]

            # Check if regime changed and cooldown allows
            if (new_regime != current_regime and
                    (i - last_change) >= self.cooldown and
                    new_regime in self.regime_allocations_):
                old_alloc = current_alloc.copy()
                alloc_map = self.regime_allocations_[new_regime]
                current_alloc = np.array([alloc_map.get(s, 0.0) for s in self.strategy_names])
                current_regime = new_regime
                last_change = i

                # Transaction cost
                if i > 0:
                    turnover = np.sum(np.abs(current_alloc - old_alloc))
                    portfolio_ret[i] -= turnover * self.tcost_bps / 10000.0

            # Portfolio return for day i
            day_rets = ret_matrix.iloc[i].values
            portfolio_ret[i] += np.dot(current_alloc, day_rets)
            alloc_history[i] = current_alloc

        daily_ret = pd.Series(portfolio_ret, index=dates, name="meta_system")
        equity = (1.0 + daily_ret).cumprod()
        equity.name = "meta_system"

        alloc_df = pd.DataFrame(alloc_history, index=dates, columns=self.strategy_names)

        # Metrics
        from curve_strategies.strategy_backtest import compute_metrics
        metrics = compute_metrics(daily_ret)

        return {
            "daily_ret": daily_ret,
            "equity": equity,
            "allocations": alloc_df,
            "metrics": metrics,
            "regime_timeline": regime,
        }

    def walk_forward(self, train_years: int = 3, test_years: int = 1,
                     start_year: int = 2012, end_year: int = 2025) -> pd.DataFrame:
        """Walk-forward validation: fit on train window, test on next year.

        Args:
            train_years: number of years in training window
            test_years: number of years in test window (usually 1)
            start_year: first OOS test year
            end_year: last OOS test year

        Returns:
            DataFrame with one row per OOS year: year, sharpe, return, mdd, mode
        """
        from curve_strategies.strategy_backtest import compute_metrics

        rows = []
        for test_year in range(start_year, end_year + 1):
            train_end = pd.Timestamp(f"{test_year - 1}-12-31")
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_year + test_years - 1}-12-31")

            # Check we have enough training data
            train_regime = self.regime_series.loc[self.regime_series.index <= train_end]
            if len(train_regime) < 252:
                continue

            # Fit on training data
            self.fit(train_end=train_end)

            # Backtest on test period
            result = self.backtest(start=test_start, end=test_end)
            if len(result["daily_ret"]) < 20:
                continue

            m = result["metrics"]
            rows.append({
                "year": test_year,
                "sharpe": m["sharpe"],
                "ann_ret": m["ann_ret"],
                "vol": m["vol"],
                "mdd": m["mdd"],
                "hit_rate": m["hit_rate"],
                "n_days": m["n_days"],
                "mode": self.mode,
            })

        return pd.DataFrame(rows)
