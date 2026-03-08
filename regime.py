"""HMM-based regime detection (CALM / STRESS)."""

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import HMM_N_SEEDS, HMM_N_ITER, MOVE_MEDIAN_WINDOW, ZSCORE_MIN_PERIODS


def fit_hmm_regime(
    move: pd.Series,
    vix: pd.Series,
    n_seeds: int = HMM_N_SEEDS,
    n_iter: int = HMM_N_ITER,
) -> pd.Series:
    """Fit a 2-state Gaussian HMM on log(MOVE) and log(VIX).

    Returns a Series of 'CALM' / 'STRESS' labels indexed like the inputs.
    Higher average MOVE -> STRESS state.
    """
    from hmmlearn.hmm import GaussianHMM

    reg_df = pd.DataFrame({
        "log_move": np.log(move.clip(lower=1)),
        "log_vix": np.log(vix.clip(lower=1)),
    }).dropna()

    if len(reg_df) < 100:
        print("  [WARN] Not enough data for HMM, using fallback regime.")
        return fallback_regime(move)

    scaler = StandardScaler()
    X = scaler.fit_transform(reg_df.values)

    best_model, best_ll = None, -np.inf
    for seed in range(n_seeds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m = GaussianHMM(
                    n_components=2,
                    covariance_type="full",
                    n_iter=n_iter,
                    random_state=seed,
                    tol=1e-5,
                )
                m.fit(X)
                ll = m.score(X)
                if ll > best_ll:
                    best_ll = ll
                    best_model = m
            except Exception:
                continue

    if best_model is None:
        print("  [WARN] HMM fitting failed on all seeds, using fallback.")
        return fallback_regime(move)

    states = best_model.predict(X)

    # Label: higher MOVE mean -> STRESS
    mv0 = move.reindex(reg_df.index)[states == 0].mean()
    mv1 = move.reindex(reg_df.index)[states == 1].mean()
    if mv0 > mv1:
        lmap = {0: "STRESS", 1: "CALM"}
    else:
        lmap = {0: "CALM", 1: "STRESS"}

    regime = pd.Series([lmap[s] for s in states], index=reg_df.index, name="regime")

    calm_pct = (regime == "CALM").mean() * 100
    print(f"  HMM regime: CALM {calm_pct:.1f}% | STRESS {100 - calm_pct:.1f}%")

    return regime


def fallback_regime(
    move: pd.Series,
    window: int = MOVE_MEDIAN_WINDOW,
    min_periods: int = ZSCORE_MIN_PERIODS,
) -> pd.Series:
    """Quantile-based fallback when HMM is unavailable.

    MOVE >= rolling median -> STRESS, else CALM.
    """
    q50 = move.rolling(window, min_periods=min_periods).median()
    regime = pd.Series("CALM", index=move.index, name="regime")
    regime[move >= q50] = "STRESS"
    return regime


def compute_regime(D: dict) -> dict:
    """Add regime to data dict D. Tries HMM first, falls back to quantile."""
    try:
        D["regime"] = fit_hmm_regime(D["move"], D["vix"])
    except ImportError:
        print("  [WARN] hmmlearn not installed, using fallback regime.")
        D["regime"] = fallback_regime(D["move"])
    return D
