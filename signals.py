"""Risk Reversal signal computation and z-score variants."""

import numpy as np
import pandas as pd
from config import ZSCORE_WINDOW, ZSCORE_MIN_PERIODS, MOVE_MEDIAN_WINDOW, MOVE_RATIO_CLIP


def compute_rr(options: pd.DataFrame, rr_tenors: dict) -> pd.DataFrame:
    """Compute Risk Reversal (Put IV - Call IV) for each tenor.

    Parameters
    ----------
    options : DataFrame with raw options columns
    rr_tenors : dict  tenor -> (put_col, call_col)

    Returns
    -------
    DataFrame with columns per tenor (1W, 1M, 3M, 6M)
    """
    rr = pd.DataFrame(index=options.index)
    for tenor, (put_col, call_col) in rr_tenors.items():
        rr[tenor] = (
            pd.to_numeric(options[put_col], errors="coerce")
            - pd.to_numeric(options[call_col], errors="coerce")
        ).round(4)
    return rr


def compute_zscore_standard(
    rr_1m: pd.Series,
    window: int = ZSCORE_WINDOW,
    min_periods: int = ZSCORE_MIN_PERIODS,
) -> pd.Series:
    """Standard rolling z-score of the 1M RR.

    z = (rr - mu_250) / sigma_250
    """
    mu = rr_1m.rolling(window, min_periods=min_periods).mean()
    sig = rr_1m.rolling(window, min_periods=min_periods).std()
    z = (rr_1m - mu) / sig.replace(0, np.nan)
    z.name = "z_std"
    return z


def compute_zscore_move_adj(
    rr_1m: pd.Series,
    move: pd.Series,
    window: int = ZSCORE_WINDOW,
    min_periods: int = ZSCORE_MIN_PERIODS,
    move_med_window: int = MOVE_MEDIAN_WINDOW,
    move_clip: tuple = MOVE_RATIO_CLIP,
) -> pd.Series:
    """MOVE-adjusted rolling z-score.

    sigma_adj = sigma_250 * sqrt(MOVE / MOVE_median_504)
    z = (rr - mu_250) / sigma_adj
    """
    mu = rr_1m.rolling(window, min_periods=min_periods).mean()
    sig = rr_1m.rolling(window, min_periods=min_periods).std()

    move_med = move.rolling(move_med_window, min_periods=min_periods).median()
    move_ratio = (move / move_med).clip(move_clip[0], move_clip[1])
    sig_adj = sig * np.sqrt(move_ratio)

    z = (rr_1m - mu) / sig_adj.replace(0, np.nan)
    z.name = "z_move"
    return z


def compute_all_signals(D: dict) -> dict:
    """Compute both z-score variants and attach them to data dict D.

    Adds keys: z_std, z_move
    """
    rr_1m = D["rr_1m"]
    move = D["move"]

    D["z_std"] = compute_zscore_standard(rr_1m)
    D["z_move"] = compute_zscore_move_adj(rr_1m, move)

    for key in ("z_std", "z_move"):
        n_valid = D[key].dropna().shape[0]
        print(f"  {key:10s}: {n_valid:>5} non-NaN values")

    return D
