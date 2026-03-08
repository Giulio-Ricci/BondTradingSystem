"""
final_chart.py - Publication-quality chart: Equity + Z-score with trade signals.

Top panel:  Equity curve (strategy vs B&H TLT) with regime background shading
Bottom panel: Composite z-score with threshold lines and signal triangles
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

FINAL_DIR = os.path.join(SAVE_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)


# ── Signal computation (same as final_system.py) ──

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


def fast_bt_full(z_arr, reg_arr, r_long, r_short,
                 tl_c, ts_c, tl_s, cooldown=5,
                 delay=1, tcost_bps=5, w_init=0.5):
    n = len(z_arr)
    weight = np.empty(n)
    weight[0] = w_init
    events = []
    last_ev = -cooldown - 1

    for i in range(n):
        zv = z_arr[i]
        w_prev = weight[i - 1] if i > 0 else w_init
        if np.isnan(zv):
            weight[i] = w_prev
            continue
        reg = reg_arr[i]
        triggered = False
        sig = None
        if reg == 0:
            if zv <= tl_c and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            elif zv >= ts_c and (i - last_ev) >= cooldown:
                weight[i] = 1.0; triggered = True; sig = "LONG"
            else:
                weight[i] = w_prev
        else:
            if zv <= tl_s and (i - last_ev) >= cooldown:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            else:
                weight[i] = w_prev
        if triggered:
            last_ev = i
            events.append((i, sig, zv, reg))

    shift = 1 + delay
    w_del = np.empty(n)
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost_bps / 10_000)
    equity = np.cumprod(1.0 + net)

    return equity, net, w_del, events


def main():
    print("Loading data ...")
    D = load_all()

    print("Fitting regime ...")
    regime = fit_hmm_regime(D["move"], D["vix"])
    regime_num = (regime == "STRESS").astype(np.int8)
    etf_ret = D["etf_ret"]
    start = pd.Timestamp(BACKTEST_START)

    # ── Best composite config ──
    # ALL|std|w63 + 10Y-2Y slope momentum (k=0.5, window=126)
    TL_C, TS_C, TL_S, CD = -3.25, 2.75, -4.0, 5

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
    reg_vals = regime_num.reindex(common).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    dates = common

    # Run backtest
    eq, net, w_del, events = fast_bt_full(
        z_vals, reg_vals, r_long, r_short, TL_C, TS_C, TL_S, CD)

    # Benchmark
    bm_eq = np.cumprod(1.0 + r_long)

    # Regime series for shading
    reg_series = regime.reindex(common)

    # ═══════════════════════════════════════════════════════════════════
    # BUILD THE CHART
    # ═══════════════════════════════════════════════════════════════════

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 11),
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
        sharex=True,
    )

    # ── Regime background shading (both panels) ──
    for ax in (ax1, ax2):
        stress_mask = (reg_series.values == "STRESS")
        i = 0
        while i < len(dates):
            if stress_mask[i]:
                j = i
                while j < len(dates) and stress_mask[j]:
                    j += 1
                ax.axvspan(dates[i], dates[min(j, len(dates)-1)],
                           color="#FFE0E0", alpha=0.35, zorder=0)
                i = j
            else:
                i += 1

    # ══════════════════════════════════════════════════════════════
    # TOP PANEL: EQUITY CURVE
    # ══════════════════════════════════════════════════════════════

    ax1.plot(dates, eq, color="#1a5276", linewidth=1.8, label="Composite Best (Sh=0.325)", zorder=3)
    ax1.plot(dates, bm_eq, color="#aab7b8", linewidth=1.2, linestyle="--",
             label="B&H TLT", alpha=0.7, zorder=2)

    # Fill between strategy and benchmark
    ax1.fill_between(dates, eq, bm_eq,
                      where=(eq >= bm_eq),
                      color="#27ae60", alpha=0.08, zorder=1)
    ax1.fill_between(dates, eq, bm_eq,
                      where=(eq < bm_eq),
                      color="#e74c3c", alpha=0.08, zorder=1)

    # Signal markers on equity curve
    for idx, sig, zv, reg in events:
        dt = dates[idx]
        eq_val = eq[idx]
        if sig == "LONG":
            ax1.scatter(dt, eq_val, marker="^", s=70, c="#27ae60",
                        edgecolors="black", linewidths=0.5, zorder=5)
        else:
            ax1.scatter(dt, eq_val, marker="v", s=70, c="#e74c3c",
                        edgecolors="black", linewidths=0.5, zorder=5)

    ax1.set_ylabel("Growth of $1", fontsize=12, fontweight="bold")
    ax1.set_title("RR Duration Strategy — Composite Best (ALL|std|w63 + 10Y-2Y Slope Momentum)",
                  fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.2, linewidth=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.set_ylim(bottom=0.60)

    # Legend with custom handles
    legend_handles = [
        Line2D([0], [0], color="#1a5276", linewidth=2, label="Strategy (Sh=0.325)"),
        Line2D([0], [0], color="#aab7b8", linewidth=1.2, linestyle="--", label="B&H TLT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5, label="LONG signal"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5, label="SHORT signal"),
        plt.Rectangle((0, 0), 1, 1, fc="#FFE0E0", alpha=0.5, label="STRESS regime"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=9,
               framealpha=0.9, fancybox=True)

    # Annotations: key metrics
    ann_text = (f"Sharpe: 0.325  |  Sortino: 0.399  |  Ann.Ret: 5.39%\n"
                f"Vol: 10.4%  |  MDD: -23.6%  |  Calmar: 0.228  |  Events: {len(events)}")
    ax1.text(0.98, 0.03, ann_text, transform=ax1.transAxes,
             fontsize=8, fontfamily="monospace",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#cccccc"))

    # ══════════════════════════════════════════════════════════════
    # BOTTOM PANEL: Z-SCORE WITH THRESHOLDS AND SIGNALS
    # ══════════════════════════════════════════════════════════════

    ax2.plot(dates, z_vals, color="#2c3e50", linewidth=0.7, alpha=0.85, zorder=3)

    # Fill z-score regions
    ax2.fill_between(dates, z_vals, 0,
                      where=(z_vals > 0), color="#27ae60", alpha=0.08, zorder=1)
    ax2.fill_between(dates, z_vals, 0,
                      where=(z_vals < 0), color="#e74c3c", alpha=0.08, zorder=1)

    # Threshold lines
    ax2.axhline(TL_C, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2,
                label=f"tl_calm = {TL_C}")
    ax2.axhline(TS_C, color="#27ae60", linewidth=1.2, linestyle="--", alpha=0.7, zorder=2,
                label=f"ts_calm = {TS_C}")
    ax2.axhline(TL_S, color="#c0392b", linewidth=1.0, linestyle=":", alpha=0.6, zorder=2,
                label=f"tl_stress = {TL_S}")
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3, zorder=1)

    # Signal triangles on z-score panel
    for idx, sig, zv, reg in events:
        dt = dates[idx]
        z_at = z_vals[idx]
        if sig == "LONG":
            ax2.scatter(dt, z_at, marker="^", s=90, c="#27ae60",
                        edgecolors="black", linewidths=0.6, zorder=6)
        else:
            ax2.scatter(dt, z_at, marker="v", s=90, c="#e74c3c",
                        edgecolors="black", linewidths=0.6, zorder=6)

    ax2.set_ylabel("Composite Z-Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    ax2.set_ylim(-8, 8)

    # Z-score legend
    z_legend_handles = [
        Line2D([0], [0], color="#e74c3c", linewidth=1.2, linestyle="--",
               label=f"SHORT calm ({TL_C})"),
        Line2D([0], [0], color="#27ae60", linewidth=1.2, linestyle="--",
               label=f"LONG calm ({TS_C})"),
        Line2D([0], [0], color="#c0392b", linewidth=1.0, linestyle=":",
               label=f"SHORT stress ({TL_S})"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="LONG trigger"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="SHORT trigger"),
    ]
    ax2.legend(handles=z_legend_handles, loc="lower left", fontsize=8,
               framealpha=0.9, fancybox=True, ncol=3)

    # Date formatting
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax2.get_xticklabels(), rotation=0, ha="center")

    # ── Save ──
    fig.savefig(os.path.join(FINAL_DIR, "07_final_equity_zscore.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nChart saved: {FINAL_DIR}/07_final_equity_zscore.png")


if __name__ == "__main__":
    main()
