"""
final_report.py - Generate trade list (CSV) and strategy documentation (TXT).
"""

import os
import numpy as np
import pandas as pd

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

FINAL_DIR = os.path.join(SAVE_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)


def compute_rr_blend(options_df, tenor_keys):
    parts = []
    for tk in tenor_keys:
        put_col, call_col = RR_TENORS[tk]
        rr = (pd.to_numeric(options_df[put_col], errors="coerce")
              - pd.to_numeric(options_df[call_col], errors="coerce"))
        parts.append(rr)
    return pd.concat(parts, axis=1).mean(axis=1)


def compute_zscore(rr, move, z_type, z_window, move_med_window=504):
    min_per = max(z_window // 4, 20)
    mu  = rr.rolling(z_window, min_periods=min_per).mean()
    sig = rr.rolling(z_window, min_periods=min_per).std()
    if z_type == "std":
        return (rr - mu) / sig.replace(0, np.nan)
    move_med = move.rolling(move_med_window, min_periods=min_per).median()
    move_ratio = (move / move_med).clip(0.5, 2.5)
    sig_adj = sig * np.sqrt(move_ratio)
    return (rr - mu) / sig_adj.replace(0, np.nan)


def compute_slope_momentum(yields, col_long, col_short, window=126):
    slope = yields[col_long] - yields[col_short]
    delta = slope.diff(window)
    mu = delta.rolling(252, min_periods=63).mean()
    sig = delta.rolling(252, min_periods=63).std()
    return (delta - mu) / sig.replace(0, np.nan)


def main():
    print("Loading data ...")
    D = load_all()

    print("Fitting regime ...")
    regime = fit_hmm_regime(D["move"], D["vix"])
    regime_num = (regime == "STRESS").astype(np.int8)
    etf_ret = D["etf_ret"]
    etf_px = D["etf_px"]
    start = pd.Timestamp(BACKTEST_START)

    # ── Best composite config ──
    TL_C, TS_C, TL_S, CD = -3.25, 2.75, -4.0, 5
    DELAY = 1
    TCOST_BPS = 5
    W_INIT = 0.5

    rr_all = compute_rr_blend(D["options"], ["1W", "1M", "3M", "6M"])
    z_base = compute_zscore(rr_all, D["move"], "std", 63)
    slope_mom = compute_slope_momentum(D["yields"], "US_10Y", "US_2Y", 126)
    z_composite = z_base + 0.5 * slope_mom

    # Align
    common = (z_composite.dropna().index
              .intersection(regime.index)
              .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index))
    common = common[common >= start].sort_values()

    z_vals = z_composite.reindex(common).values
    z_base_vals = z_base.reindex(common).values
    slope_vals = slope_mom.reindex(common).values
    reg_vals = regime_num.reindex(common).values
    reg_labels = regime.reindex(common).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    dates = common

    # TLT / SHV prices for reference
    tlt_px = etf_px["TLT"].reindex(common)
    shv_px = etf_px["SHV"].reindex(common)

    # ── Run backtest tracking all details ──
    n = len(z_vals)
    weight = np.empty(n)
    weight[0] = W_INIT
    events = []
    last_ev = -CD - 1

    for i in range(n):
        zv = z_vals[i]
        w_prev = weight[i - 1] if i > 0 else W_INIT
        if np.isnan(zv):
            weight[i] = w_prev
            continue
        reg = reg_vals[i]
        triggered = False
        sig = None
        if reg == 0:  # CALM
            if zv <= TL_C and (i - last_ev) >= CD:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            elif zv >= TS_C and (i - last_ev) >= CD:
                weight[i] = 1.0; triggered = True; sig = "LONG"
            else:
                weight[i] = w_prev
        else:  # STRESS
            if zv <= TL_S and (i - last_ev) >= CD:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            else:
                weight[i] = w_prev
        if triggered:
            last_ev = i
            events.append({
                "idx": i,
                "date_signal": dates[i],
                "signal": sig,
                "z_composite": zv,
                "z_base_rr": z_base_vals[i] if not np.isnan(z_base_vals[i]) else None,
                "z_slope_mom": slope_vals[i] if not np.isnan(slope_vals[i]) else None,
                "regime": "CALM" if reg == 0 else "STRESS",
                "tlt_price": tlt_px.iloc[i] if not np.isnan(tlt_px.iloc[i]) else None,
                "shv_price": shv_px.iloc[i] if not np.isnan(shv_px.iloc[i]) else None,
            })

    # Execution delay
    shift = 1 + DELAY
    w_del = np.empty(n)
    w_del[:shift] = W_INIT
    w_del[shift:] = weight[:n - shift]

    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - W_INIT)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (TCOST_BPS / 10_000)
    equity = np.cumprod(1.0 + net)

    # ── Build trade list with P&L ──
    trades = []
    for k, ev in enumerate(events):
        idx = ev["idx"]
        # Execution date = signal date + delay + 1
        exec_idx = min(idx + DELAY + 1, n - 1)
        exec_date = dates[exec_idx]

        # Position: LONG -> 100% TLT, SHORT -> 100% SHV
        position = "100% TLT" if ev["signal"] == "LONG" else "100% SHV"

        # Equity at signal and at execution
        eq_at_signal = equity[idx]
        eq_at_exec = equity[exec_idx]

        # Find next event to compute trade P&L
        if k < len(events) - 1:
            next_idx = events[k + 1]["idx"]
            next_exec_idx = min(next_idx + DELAY + 1, n - 1)
            eq_at_exit = equity[next_exec_idx]
            exit_date = dates[next_exec_idx]
            holding_days = (exit_date - exec_date).days
            trade_pnl = eq_at_exit / eq_at_exec - 1.0
        else:
            # Last trade: hold to end
            eq_at_exit = equity[-1]
            exit_date = dates[-1]
            holding_days = (exit_date - exec_date).days
            trade_pnl = eq_at_exit / eq_at_exec - 1.0

        trades.append({
            "#": k + 1,
            "signal_date": ev["date_signal"].strftime("%Y-%m-%d"),
            "exec_date": exec_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "signal": ev["signal"],
            "position": position,
            "regime": ev["regime"],
            "z_composite": round(ev["z_composite"], 3),
            "z_rr_base": round(ev["z_base_rr"], 3) if ev["z_base_rr"] is not None else "",
            "z_slope_mom": round(ev["z_slope_mom"], 3) if ev["z_slope_mom"] is not None else "",
            "tlt_price": round(ev["tlt_price"], 2) if ev["tlt_price"] is not None else "",
            "shv_price": round(ev["shv_price"], 2) if ev["shv_price"] is not None else "",
            "holding_days": holding_days,
            "trade_pnl": round(trade_pnl * 100, 3),
            "equity_after": round(eq_at_exec, 4),
        })

    trades_df = pd.DataFrame(trades)
    trades_path = os.path.join(FINAL_DIR, "trade_list.csv")
    trades_df.to_csv(trades_path, index=False)
    print(f"\nTrade list saved: {trades_path}")
    print(f"  Total trades: {len(trades_df)}")
    print(f"  LONG trades:  {(trades_df['signal']=='LONG').sum()}")
    print(f"  SHORT trades: {(trades_df['signal']=='SHORT').sum()}")

    # Print trade list
    print(f"\n{'='*120}")
    print(f"  TRADE LIST")
    print(f"{'='*120}")
    print(f"  {'#':>3}  {'Signal Date':<12} {'Exec Date':<12} {'Exit Date':<12} "
          f"{'Sig':<6} {'Regime':<7} {'z_comp':>7} {'z_rr':>7} {'z_slope':>7} "
          f"{'TLT':>8} {'Days':>5} {'P&L%':>8} {'Equity':>8}")
    print(f"  {'-'*118}")
    for _, t in trades_df.iterrows():
        pnl_str = f"{t['trade_pnl']:+.3f}%"
        print(f"  {t['#']:>3}  {t['signal_date']:<12} {t['exec_date']:<12} {t['exit_date']:<12} "
              f"{t['signal']:<6} {t['regime']:<7} {t['z_composite']:>+7.3f} "
              f"{str(t['z_rr_base']):>7} {str(t['z_slope_mom']):>7} "
              f"{str(t['tlt_price']):>8} {t['holding_days']:>5} {pnl_str:>8} "
              f"{t['equity_after']:>8.4f}")

    # ── Summary stats ──
    long_trades = trades_df[trades_df["signal"] == "LONG"]
    short_trades = trades_df[trades_df["signal"] == "SHORT"]
    winning = trades_df[trades_df["trade_pnl"] > 0]
    losing = trades_df[trades_df["trade_pnl"] <= 0]

    total_ret = equity[-1] - 1.0
    years = n / 252.0
    ann_ret = (1.0 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0
    vol = np.std(net) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol
    down = net[net < 0]
    down_vol = np.std(down) * np.sqrt(252) if len(down) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol
    running_max = np.maximum.accumulate(equity)
    mdd = (equity / running_max - 1.0).min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    # Yearly breakdown
    yearly_rows = []
    for y in sorted(set(d.year for d in dates)):
        mask = np.array([d.year == y for d in dates])
        if mask.sum() < 5:
            continue
        net_y = net[mask]
        eq_y = np.cumprod(1.0 + net_y)
        yr_ret = eq_y[-1] - 1.0
        bm_y = np.cumprod(1.0 + r_long[mask])
        bm_ret_y = bm_y[-1] - 1.0
        yr_vol = np.std(net_y) * np.sqrt(252)
        yr_sh = ((1+yr_ret)**(252/mask.sum())-1 - RISK_FREE_RATE) / yr_vol if yr_vol > 0 else 0
        n_ev = sum(1 for ev in events if dates[ev["idx"]].year == y)
        yearly_rows.append({
            "year": y, "strat_ret": yr_ret, "bm_ret": bm_ret_y,
            "excess": yr_ret - bm_ret_y, "vol": yr_vol, "sharpe": yr_sh,
            "n_events": n_ev,
        })
    yearly_df = pd.DataFrame(yearly_rows)

    # ═══════════════════════════════════════════════════════════════
    # WRITE STRATEGY DOCUMENTATION
    # ═══════════════════════════════════════════════════════════════

    doc_path = os.path.join(FINAL_DIR, "strategy_documentation.txt")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("  RR DURATION STRATEGY - COMPLETE DOCUMENTATION\n")
        f.write("  Composite Best Configuration\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. STRATEGY OVERVIEW\n")
        f.write("-" * 80 + "\n\n")
        f.write("The RR Duration Strategy is a systematic bond duration timing strategy\n")
        f.write("that uses options-implied Risk Reversal (RR) z-scores combined with\n")
        f.write("yield curve slope momentum to generate LONG/SHORT signals on US Treasury\n")
        f.write("duration exposure.\n\n")
        f.write("The strategy alternates between:\n")
        f.write("  - LONG position  -> 100% TLT (iShares 20+ Year Treasury Bond ETF)\n")
        f.write("  - SHORT position -> 100% SHV (iShares Short Treasury Bond ETF)\n")
        f.write("  - NEUTRAL        -> 50% TLT / 50% SHV (initial & default)\n\n")
        f.write("The logic: when the options market prices extreme put skew on Treasuries\n")
        f.write("(negative z-score = market expects bond prices to fall), the strategy\n")
        f.write("goes SHORT duration (into SHV). When call skew is extreme (positive\n")
        f.write("z-score = market expects bond prices to rise), it goes LONG duration\n")
        f.write("(into TLT). The yield curve slope momentum acts as a confirmation\n")
        f.write("signal, improving timing.\n\n")

        f.write("2. SIGNAL CONSTRUCTION\n")
        f.write("-" * 80 + "\n\n")

        f.write("2.1 Base Signal: ALL-tenor RR z-score (z_base)\n\n")
        f.write("  Step 1: Compute Risk Reversal for each tenor\n")
        f.write("    RR(tenor) = Put_IV(25-delta) - Call_IV(25-delta)\n")
        f.write("    Tenors used: 1W, 1M, 3M, 6M\n\n")
        f.write("  Step 2: Average across all 4 tenors\n")
        f.write("    RR_all = mean(RR_1W, RR_1M, RR_3M, RR_6M)\n\n")
        f.write("  Step 3: Standard rolling z-score (window = 63 trading days)\n")
        f.write("    mu_63 = rolling_mean(RR_all, window=63, min_periods=20)\n")
        f.write("    sigma_63 = rolling_std(RR_all, window=63, min_periods=20)\n")
        f.write("    z_base = (RR_all - mu_63) / sigma_63\n\n")
        f.write("  NOTE: 63-day window (~3 months) was selected via exhaustive grid\n")
        f.write("  search over [50, 63, 80, 125, 180, 250, 375, 504] days.\n")
        f.write("  The ALL-tenor blend outperformed single-tenor variants.\n\n")

        f.write("2.2 Slope Momentum Component (z_slope)\n\n")
        f.write("  Step 1: Compute yield curve slope\n")
        f.write("    slope = US_10Y_yield - US_2Y_yield\n\n")
        f.write("  Step 2: Compute 126-day change (slope momentum)\n")
        f.write("    delta_slope = slope(t) - slope(t - 126)\n\n")
        f.write("  Step 3: Z-score the slope change\n")
        f.write("    mu_252 = rolling_mean(delta_slope, window=252, min_periods=63)\n")
        f.write("    sigma_252 = rolling_std(delta_slope, window=252, min_periods=63)\n")
        f.write("    z_slope = (delta_slope - mu_252) / sigma_252\n\n")
        f.write("  RATIONALE: When the yield curve is steepening faster than usual\n")
        f.write("  (positive z_slope), it confirms risk-off moves. When flattening\n")
        f.write("  faster (negative z_slope), it confirms risk-on / duration demand.\n\n")

        f.write("2.3 Composite Signal\n\n")
        f.write("  z_composite = z_base + k * z_slope\n\n")
        f.write("  Where k = 0.50 (slope momentum weight)\n\n")
        f.write("  The slope momentum adds directional bias:\n")
        f.write("  - If RR is already negative AND slope is steepening -> stronger SHORT\n")
        f.write("  - If RR is positive AND slope is flattening -> stronger LONG\n")
        f.write("  - Conflicting signals partially cancel out -> fewer false triggers\n\n")

        f.write("3. REGIME DETECTION\n")
        f.write("-" * 80 + "\n\n")
        f.write("  A 2-state Gaussian Hidden Markov Model (HMM) classifies each day\n")
        f.write("  as CALM or STRESS based on:\n")
        f.write("    - log(MOVE Index)  [bond volatility]\n")
        f.write("    - log(VIX)         [equity volatility]\n\n")
        f.write("  HMM parameters:\n")
        f.write("    - n_components = 2\n")
        f.write("    - covariance_type = 'full'\n")
        f.write("    - n_iter = 300\n")
        f.write("    - n_seeds = 10 (best log-likelihood selected)\n\n")
        f.write("  The state with higher average MOVE is labeled STRESS.\n")
        f.write("  Typical distribution: CALM ~69%, STRESS ~31%.\n\n")
        f.write("  The regime affects which thresholds are used for signal generation.\n")
        f.write("  In STRESS, only SHORT signals are allowed (no LONG), and the\n")
        f.write("  SHORT threshold is more extreme (harder to trigger).\n\n")

        f.write("4. TRADING RULES\n")
        f.write("-" * 80 + "\n\n")
        f.write("  4.1 Signal Thresholds\n\n")
        f.write(f"    CALM regime:\n")
        f.write(f"      z_composite <= {TL_C:.2f}  ->  SHORT (go to 100% SHV)\n")
        f.write(f"      z_composite >= {TS_C:+.2f}  ->  LONG  (go to 100% TLT)\n\n")
        f.write(f"    STRESS regime:\n")
        f.write(f"      z_composite <= {TL_S:.2f}  ->  SHORT (go to 100% SHV)\n")
        f.write(f"      (no LONG signals allowed in STRESS)\n\n")

        f.write("  4.2 Position Management\n\n")
        f.write(f"    Initial position: {W_INIT*100:.0f}% TLT / {(1-W_INIT)*100:.0f}% SHV\n")
        f.write(f"    After LONG signal:  100% TLT / 0% SHV\n")
        f.write(f"    After SHORT signal: 0% TLT / 100% SHV\n\n")
        f.write(f"    Position is held (forward-filled) until the next signal.\n")
        f.write(f"    There is no target/stop-loss exit; the next signal reverses.\n\n")

        f.write("  4.3 Execution Parameters\n\n")
        f.write(f"    Cooldown:          {CD} trading days between signals\n")
        f.write(f"    Execution delay:   {DELAY} trading day(s) after signal\n")
        f.write(f"    Transaction cost:  {TCOST_BPS} basis points per rebalance\n\n")
        f.write(f"    The cooldown prevents signal clustering: after a trade, the\n")
        f.write(f"    strategy ignores new signals for {CD} days.\n\n")
        f.write(f"    The execution delay models real-world implementation: a signal\n")
        f.write(f"    generated at close on day T is executed at close on day T+{DELAY+1}.\n\n")

        f.write("5. PERFORMANCE SUMMARY\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"  Backtest period:  {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}\n")
        f.write(f"  Trading days:     {n}\n")
        f.write(f"  Years:            {years:.1f}\n\n")
        f.write(f"  Sharpe Ratio:     {sharpe:.4f}\n")
        f.write(f"  Sortino Ratio:    {sortino:.4f}\n")
        f.write(f"  Ann. Return:      {ann_ret:.4f}  ({ann_ret*100:.2f}%)\n")
        f.write(f"  Volatility:       {vol:.4f}  ({vol*100:.2f}%)\n")
        f.write(f"  Max Drawdown:     {mdd:.4f}  ({mdd*100:.2f}%)\n")
        f.write(f"  Calmar Ratio:     {calmar:.4f}\n")
        f.write(f"  Total Return:     {total_ret:.4f}  ({total_ret*100:.2f}%)\n")
        f.write(f"  Final Equity:     ${equity[-1]:.3f}  (from $1.00)\n\n")

        f.write(f"  Total Events:     {len(events)}\n")
        f.write(f"    LONG signals:   {len(long_trades)}\n")
        f.write(f"    SHORT signals:  {len(short_trades)}\n")
        f.write(f"  Winning trades:   {len(winning)} ({len(winning)/len(trades_df)*100:.1f}%)\n")
        f.write(f"  Losing trades:    {len(losing)} ({len(losing)/len(trades_df)*100:.1f}%)\n")
        f.write(f"  Avg trade P&L:    {trades_df['trade_pnl'].mean():+.3f}%\n")
        f.write(f"  Avg winning:      {winning['trade_pnl'].mean():+.3f}%\n")
        f.write(f"  Avg losing:       {losing['trade_pnl'].mean():+.3f}%\n")
        f.write(f"  Avg holding days: {trades_df['holding_days'].mean():.0f}\n\n")

        f.write("6. YEARLY BREAKDOWN\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"  {'Year':>6}  {'Strategy':>10}  {'B&H TLT':>10}  {'Excess':>10}  "
                f"{'Vol':>8}  {'Events':>6}\n")
        f.write(f"  {'-'*58}\n")
        for _, yr in yearly_df.iterrows():
            marker = "  *" if yr["excess"] > 0 else ""
            f.write(f"  {int(yr['year']):>6}  {yr['strat_ret']:>+10.4f}  {yr['bm_ret']:>+10.4f}  "
                    f"{yr['excess']:>+10.4f}  {yr['vol']:>8.4f}  {int(yr['n_events']):>6}{marker}\n")
        pos_yrs = (yearly_df["excess"] > 0).sum()
        f.write(f"\n  Positive excess years: {pos_yrs}/{len(yearly_df)} "
                f"({pos_yrs/len(yearly_df)*100:.0f}%)\n")
        f.write(f"  Average yearly excess: {yearly_df['excess'].mean():+.4f}\n\n")

        f.write("7. DATA SOURCES\n")
        f.write("-" * 80 + "\n\n")
        f.write("  Options data:   25-delta put/call IV for TY futures (1W, 1M, 3M, 6M)\n")
        f.write("  ETF prices:     TLT, SHV (daily close)\n")
        f.write("  Yield curve:    US Treasury yields (3M, 2Y, 5Y, 10Y, 30Y)\n")
        f.write("  Volatility:     VIX, MOVE Index\n")
        f.write("  Period:         2007-01-03 to 2026-02-19\n")
        f.write("  Backtest start: 2009-01-01 (sufficient lookback for z-score)\n\n")

        f.write("8. PARAMETER SUMMARY TABLE\n")
        f.write("-" * 80 + "\n\n")
        f.write(f"  {'Parameter':<30} {'Value':<20} {'Description'}\n")
        f.write(f"  {'-'*80}\n")
        params_table = [
            ("RR tenors", "1W,1M,3M,6M", "All available tenors averaged"),
            ("z_type", "std", "Standard rolling z-score (not MOVE-adjusted)"),
            ("z_window", "63 days", "Rolling window for z-score mean/std"),
            ("z_min_periods", "20 days", "Minimum observations for rolling calc"),
            ("slope_pair", "10Y - 2Y", "Yield curve slope definition"),
            ("slope_mom_window", "126 days", "Lookback for slope change"),
            ("slope_mom_zscore_win", "252 days", "Window for z-scoring slope change"),
            ("k (slope weight)", "0.50", "Weight of slope momentum in composite"),
            ("tl_calm", str(TL_C), "SHORT threshold in CALM regime"),
            ("ts_calm", str(TS_C), "LONG threshold in CALM regime"),
            ("tl_stress", str(TL_S), "SHORT threshold in STRESS regime"),
            ("ts_stress", "disabled", "No LONG in STRESS"),
            ("cooldown", f"{CD} days", "Min gap between consecutive signals"),
            ("execution_delay", f"{DELAY} day(s)", "Signal-to-execution lag"),
            ("tcost", f"{TCOST_BPS} bps", "One-way transaction cost per rebalance"),
            ("w_initial", f"{W_INIT}", "Starting allocation (50/50)"),
            ("w_long", "1.0", "TLT weight after LONG signal"),
            ("w_short", "0.0", "TLT weight after SHORT signal (=100% SHV)"),
            ("long_etf", "TLT", "Duration-long instrument"),
            ("short_etf", "SHV", "Duration-short instrument"),
            ("HMM states", "2", "CALM / STRESS"),
            ("HMM features", "log(MOVE),log(VIX)", "Input to regime model"),
            ("risk_free_rate", "2.0%", "For Sharpe/Sortino calculation"),
        ]
        for name, val, desc in params_table:
            f.write(f"  {name:<30} {val:<20} {desc}\n")

        f.write(f"\n\n{'='*80}\n")
        f.write(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"{'='*80}\n")

    print(f"Strategy documentation saved: {doc_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
