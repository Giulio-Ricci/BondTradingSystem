"""Backtest engine for the RR Duration Strategy."""

import numpy as np
import pandas as pd
from config import DEFAULT_PARAMS, BACKTEST_START, RISK_FREE_RATE


def backtest(
    z: pd.Series,
    regime: pd.Series,
    etf_ret: pd.DataFrame,
    params: dict | None = None,
    start: str = BACKTEST_START,
) -> dict:
    """Run a single backtest.

    Parameters
    ----------
    z      : z-score signal series (z_std or z_move)
    regime : Series of 'CALM' / 'STRESS'
    etf_ret: DataFrame of daily ETF returns
    params : strategy parameters (uses DEFAULT_PARAMS if None)
    start  : backtest start date

    Returns
    -------
    dict with keys: equity, net_ret, metrics, events, yearly, weight
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    tl_calm = p["tl_calm"]
    ts_calm = p["ts_calm"]
    tl_stress = p["tl_stress"]
    ts_stress = p["ts_stress"]
    allow_long_stress = p["allow_long_stress"]
    long_etf = p["long_etf"]
    short_etf = p["short_etf"]
    cooldown = p["cooldown"]
    delay = p["delay"]
    tcost_bps = p["tcost_bps"]
    w_long = p["w_long"]
    w_short = p["w_short"]
    w_initial = p["w_initial"]

    # Align all series on common dates
    common = (
        z.dropna().index
        .intersection(regime.dropna().index)
        .intersection(etf_ret.dropna(subset=[long_etf, short_etf]).index)
    )
    common = common[common >= start]
    common = common.sort_values()

    if len(common) < 20:
        return _empty_result()

    # ── Generate signals & weights ──────────────────────────────────────────
    weight = pd.Series(np.nan, index=common)
    events = []
    last_event_date = pd.Timestamp("1900-01-01")

    for dt in common:
        zv = z.loc[dt]
        reg = regime.loc[dt]
        days_since = (dt - last_event_date).days

        signal = None
        if reg == "CALM":
            if zv <= tl_calm:
                signal = "SHORT"
            elif zv >= ts_calm:
                signal = "LONG"
        else:  # STRESS
            if zv <= tl_stress:
                signal = "SHORT"
            if allow_long_stress and zv >= ts_stress:
                signal = "LONG"

        # Apply cooldown
        if signal and days_since < cooldown:
            signal = None

        if signal == "LONG":
            weight.loc[dt] = w_long
            events.append({"date": dt, "signal": "LONG", "z": zv, "regime": reg})
            last_event_date = dt
        elif signal == "SHORT":
            weight.loc[dt] = w_short
            events.append({"date": dt, "signal": "SHORT", "z": zv, "regime": reg})
            last_event_date = dt

    # Forward-fill weights
    weight = weight.ffill().fillna(w_initial)

    # Apply execution delay
    w_del = weight.shift(1 + delay).fillna(w_initial)

    # ── Portfolio returns ───────────────────────────────────────────────────
    r_long = etf_ret[long_etf].reindex(common).fillna(0)
    r_short = etf_ret[short_etf].reindex(common).fillna(0)

    gross_ret = w_del * r_long + (1 - w_del) * r_short

    # Transaction costs
    rebal = w_del.diff().abs().fillna(0)
    costs = rebal * (tcost_bps / 10_000)
    net_ret = gross_ret - costs

    # Equity curve
    equity = (1 + net_ret).cumprod()

    # ── Metrics ─────────────────────────────────────────────────────────────
    metrics = _compute_metrics(net_ret, equity, etf_ret, common, long_etf, short_etf)
    metrics["n_events"] = len(events)

    # ── Yearly breakdown ────────────────────────────────────────────────────
    yearly = _yearly_breakdown(net_ret, etf_ret, common, long_etf, short_etf)

    return {
        "equity": equity,
        "net_ret": net_ret,
        "weight": w_del,
        "metrics": metrics,
        "events": pd.DataFrame(events) if events else pd.DataFrame(),
        "yearly": yearly,
        "params": p,
    }


def _compute_metrics(
    net_ret: pd.Series,
    equity: pd.Series,
    etf_ret: pd.DataFrame,
    common: pd.DatetimeIndex,
    long_etf: str,
    short_etf: str,
) -> dict:
    """Compute performance metrics."""
    n_days = len(net_ret)
    years = n_days / 252

    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    ann_ret = (1 + total_ret) ** (1 / max(years, 0.01)) - 1
    vol = net_ret.std() * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 0 else 0.0

    down = net_ret[net_ret < 0]
    down_vol = down.std() * np.sqrt(252) if len(down) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol

    dd = equity / equity.cummax() - 1
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0.0

    # Benchmark: TLH if available, else 50/50 TLT/SHV
    if "TLH" in etf_ret.columns and etf_ret["TLH"].reindex(common).dropna().shape[0] > 100:
        bm_ret = etf_ret["TLH"].reindex(common).fillna(0)
    else:
        bm_ret = (
            0.5 * etf_ret[long_etf].reindex(common).fillna(0)
            + 0.5 * etf_ret[short_etf].reindex(common).fillna(0)
        )
    bm_eq = (1 + bm_ret).cumprod()
    bm_total = bm_eq.iloc[-1] / bm_eq.iloc[0] - 1
    bm_ann = (1 + bm_total) ** (1 / max(years, 0.01)) - 1

    excess = ann_ret - bm_ann

    return {
        "ann_ret": ann_ret,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "mdd": mdd,
        "calmar": calmar,
        "excess": excess,
        "bm_ann": bm_ann,
        "total_ret": total_ret,
        "n_days": n_days,
    }


def _yearly_breakdown(
    net_ret: pd.Series,
    etf_ret: pd.DataFrame,
    common: pd.DatetimeIndex,
    long_etf: str,
    short_etf: str,
) -> pd.DataFrame:
    """Per-year performance summary."""
    years = sorted(net_ret.index.year.unique())
    rows = []
    for y in years:
        mask = net_ret.index.year == y
        nr = net_ret[mask]
        if len(nr) < 5:
            continue
        eq = (1 + nr).cumprod()
        yr_ret = eq.iloc[-1] - 1
        yr_vol = nr.std() * np.sqrt(252)

        # Benchmark
        bm = etf_ret[long_etf].reindex(common)[mask].fillna(0)
        bm_ret = (1 + bm).cumprod().iloc[-1] - 1

        rows.append({
            "year": y,
            "strat_ret": yr_ret,
            "bm_ret": bm_ret,
            "excess": yr_ret - bm_ret,
            "vol": yr_vol,
            "n_days": len(nr),
        })
    return pd.DataFrame(rows)


def _empty_result() -> dict:
    """Return an empty result dict when backtest cannot run."""
    return {
        "equity": pd.Series(dtype=float),
        "net_ret": pd.Series(dtype=float),
        "weight": pd.Series(dtype=float),
        "metrics": {
            "ann_ret": 0, "vol": 0, "sharpe": 0, "sortino": 0,
            "mdd": 0, "calmar": 0, "excess": 0, "bm_ann": 0,
            "total_ret": 0, "n_days": 0, "n_events": 0,
        },
        "events": pd.DataFrame(),
        "yearly": pd.DataFrame(),
        "params": {},
    }
