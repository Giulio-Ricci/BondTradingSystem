"""Data loading and cleaning for the Bond Trading System."""

import io
import os
import numpy as np
import pandas as pd
from config import DATA_DIR, RR_TENORS, ETF_TICKERS


def _read_csv_in_xlsx(path: str) -> pd.DataFrame:
    """Read an xlsx file where all data is stored as CSV text in a single column."""
    raw = pd.read_excel(path, header=None, engine="openpyxl")
    text = raw.iloc[:, 0].astype(str).str.cat(sep="\n")
    return pd.read_csv(io.StringIO(text), sep=",")


def load_options_data() -> pd.DataFrame:
    """Load dataset1.xlsx (options, VIX, MOVE, yields, SPX).

    Returns a DataFrame indexed by datetime, sorted ascending.
    """
    path = os.path.join(DATA_DIR, "dataset1.xlsx")
    df = _read_csv_in_xlsx(path)

    # Parse dates (MM/DD/YYYY)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()

    # Coerce numerics
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_etf_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load etf.xlsx.

    Returns (prices, returns) DataFrames indexed by datetime.
    """
    path = os.path.join(DATA_DIR, "etf.xlsx")
    df = pd.read_excel(path, engine="openpyxl")

    # Normalise column names
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    prices = df[ETF_TICKERS].copy()

    # Clean corrupted prices: some entries have the decimal point removed
    # (e.g., 93045 instead of 93.045, or 59341375 instead of ~59.34).
    # Two-pass approach for robustness against dense corruption clusters.
    for col in prices.columns:
        s = prices[col]
        valid = s.dropna()
        if len(valid) < 10:
            continue
        # Pass 1: remove extreme outliers (> 10x global median)
        global_med = valid.median()
        extreme = s > global_med * 10
        if extreme.any():
            prices.loc[extreme, col] = np.nan
        # Pass 2: remove remaining outliers using local rolling median
        s2 = prices[col]
        med_local = s2.rolling(63, min_periods=3, center=True).median()
        outlier = s2 > med_local * 3
        outlier = outlier.fillna(False)
        if outlier.any():
            prices.loc[outlier, col] = np.nan
        n_removed = s.notna().sum() - prices[col].notna().sum()
        if n_removed > 0:
            print(f"  [CLEAN] {col}: removed {n_removed} corrupted prices")
    # Interpolate removed outliers
    prices = prices.interpolate(method="index", limit_direction="forward")

    returns = prices.pct_change(fill_method=None)

    # Cap daily returns: bond ETFs should not move >15% in a day.
    # Extreme returns indicate data errors (level changes, splits, etc.)
    MAX_DAILY_RET = 0.15
    for col in returns.columns:
        bad = returns[col].abs() > MAX_DAILY_RET
        if bad.any():
            n_bad = bad.sum()
            returns.loc[bad, col] = 0.0
            print(f"  [CLEAN] {col}: zeroed {n_bad} extreme daily returns (>{MAX_DAILY_RET:.0%})")

    return prices, returns


def load_yields() -> pd.DataFrame:
    """Load yields.xlsx (yield curve)."""
    path = os.path.join(DATA_DIR, "yields.xlsx")
    df = _read_csv_in_xlsx(path)

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.set_index("DATE").sort_index()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_all() -> dict:
    """Master loader returning a unified data dict D.

    D keys:
        options  - raw options / macro DataFrame
        etf_px   - ETF prices
        etf_ret  - ETF daily returns
        yields   - yield curve
        rr       - computed risk reversals per tenor
        vix      - VIX series
        move     - MOVE series
    """
    options = load_options_data()
    etf_px, etf_ret = load_etf_data()
    yields = load_yields()

    # Compute risk reversals
    rr = pd.DataFrame(index=options.index)
    for tenor, (put_col, call_col) in RR_TENORS.items():
        rr[tenor] = (
            pd.to_numeric(options[put_col], errors="coerce")
            - pd.to_numeric(options[call_col], errors="coerce")
        ).round(4)

    D = {
        "options": options,
        "etf_px": etf_px,
        "etf_ret": etf_ret,
        "yields": yields,
        "rr": rr,
        "rr_1m": rr["1M"].copy(),
        "vix": options["VIX"].copy(),
        "move": options["MOVE"].copy(),
    }

    # Print summary
    print("=" * 60)
    print("DATA LOAD SUMMARY")
    print("=" * 60)
    print(f"  Options : {options.shape[0]:>5} rows  "
          f"[{options.index.min():%Y-%m-%d} -> {options.index.max():%Y-%m-%d}]")
    print(f"  ETF px  : {etf_px.shape[0]:>5} rows  "
          f"[{etf_px.index.min():%Y-%m-%d} -> {etf_px.index.max():%Y-%m-%d}]")
    print(f"  Yields  : {yields.shape[0]:>5} rows  "
          f"[{yields.index.min():%Y-%m-%d} -> {yields.index.max():%Y-%m-%d}]")
    print(f"  RR 1M   : {rr['1M'].dropna().shape[0]:>5} non-NaN")
    print(f"  VIX     : {D['vix'].dropna().shape[0]:>5} non-NaN")
    print(f"  MOVE    : {D['move'].dropna().shape[0]:>5} non-NaN")
    print("=" * 60)

    return D
