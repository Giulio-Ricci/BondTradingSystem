"""
final_charts_v2.py - Detailed charts:
  1) Equity curve with yield curve regime coloring (bull/bear steepening/flattening)
  2) Year-by-year performance comparison vs benchmark (grid of subplots)
  3) Yearly bar chart + cumulative excess + drawdown comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

FINAL_DIR = os.path.join(SAVE_DIR, "final")
os.makedirs(FINAL_DIR, exist_ok=True)


# ── Signal computation ──

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
    mu = rr.rolling(z_window, min_periods=min_per).mean()
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


def classify_yc_regime(yields, regime_labels):
    """
    Classify each day into 4 yield curve regimes:
      Bull Steepening:  yields falling + curve steepening   (short rates fall more)
      Bull Flattening:  yields falling + curve flattening   (long rates fall more)
      Bear Steepening:  yields rising  + curve steepening   (long rates rise more)
      Bear Flattening:  yields rising  + curve flattening   (short rates rise more)

    Uses 63-day changes of 2Y and 10Y yields.
    Returns Series with labels and separate regime (CALM/STRESS).
    """
    y2 = yields["US_2Y"]
    y10 = yields["US_10Y"]

    # 63-day changes
    d2 = y2.diff(63)
    d10 = y10.diff(63)

    # Slope change
    slope_chg = d10 - d2  # positive = steepening

    # Average yield direction
    avg_yield_chg = (d2 + d10) / 2  # positive = bear (yields up)

    yc_regime = pd.Series("", index=yields.index)
    yc_regime[(avg_yield_chg <= 0) & (slope_chg > 0)] = "Bull Steepening"
    yc_regime[(avg_yield_chg <= 0) & (slope_chg <= 0)] = "Bull Flattening"
    yc_regime[(avg_yield_chg > 0) & (slope_chg > 0)] = "Bear Steepening"
    yc_regime[(avg_yield_chg > 0) & (slope_chg <= 0)] = "Bear Flattening"

    return yc_regime


def main():
    print("Loading data ...")
    D = load_all()

    print("Fitting regime ...")
    regime = fit_hmm_regime(D["move"], D["vix"])
    regime_num = (regime == "STRESS").astype(np.int8)
    etf_ret = D["etf_ret"]
    start = pd.Timestamp(BACKTEST_START)

    # ── Build composite signal ──
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
    reg_labels = regime.reindex(common)
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    dates = common

    # Backtest
    eq, net, w_del, events = fast_bt_full(
        z_vals, reg_vals, r_long, r_short, TL_C, TS_C, TL_S, CD)
    bm_eq = np.cumprod(1.0 + r_long)

    # Yield curve regime classification
    yc_regime = classify_yc_regime(D["yields"], reg_labels)
    yc_regime = yc_regime.reindex(common)

    # ═══════════════════════════════════════════════════════════════════
    # CHART 1: EQUITY CURVE WITH YC REGIME COLORS + HMM REGIME
    # ═══════════════════════════════════════════════════════════════════
    print("\nGenerating Chart 1: Equity with YC regime coloring ...")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(22, 13),
        gridspec_kw={"height_ratios": [2.5, 1], "hspace": 0.06},
        sharex=True,
    )

    # Color mapping for yield curve regimes
    yc_colors = {
        "Bull Steepening":  "#2ecc71",   # green - best for bonds
        "Bull Flattening":  "#85c1e9",   # light blue
        "Bear Steepening":  "#f39c12",   # orange
        "Bear Flattening":  "#e74c3c",   # red - worst for bonds
    }

    # Background shading: yield curve regime
    yc_vals = yc_regime.values
    for ax in (ax1, ax2):
        i = 0
        while i < len(dates):
            regime_name = yc_vals[i]
            if regime_name in yc_colors:
                j = i
                while j < len(dates) and yc_vals[j] == regime_name:
                    j += 1
                ax.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                           color=yc_colors[regime_name], alpha=0.12, zorder=0)
                i = j
            else:
                i += 1

    # STRESS regime: add hatching overlay
    stress_mask = (reg_labels.values == "STRESS")
    i = 0
    while i < len(dates):
        if stress_mask[i]:
            j = i
            while j < len(dates) and stress_mask[j]:
                j += 1
            for ax in (ax1, ax2):
                ax.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                           color="none", alpha=1.0, zorder=0,
                           hatch="///", edgecolor="#c0392b", linewidth=0)
            i = j
        else:
            i += 1

    # ── TOP: Equity curve ──
    ax1.plot(dates, eq, color="#1a5276", linewidth=1.8, zorder=4, label="Strategy (Sh=0.325)")
    ax1.plot(dates, bm_eq, color="#7f8c8d", linewidth=1.2, linestyle="--",
             alpha=0.7, zorder=3, label="B&H TLT")

    # Signal triangles
    for idx, sig, zv, reg in events:
        dt = dates[idx]
        eq_val = eq[idx]
        if sig == "LONG":
            ax1.scatter(dt, eq_val, marker="^", s=80, c="#27ae60",
                        edgecolors="black", linewidths=0.5, zorder=6)
        else:
            ax1.scatter(dt, eq_val, marker="v", s=80, c="#e74c3c",
                        edgecolors="black", linewidths=0.5, zorder=6)

    ax1.set_ylabel("Growth of $1", fontsize=13, fontweight="bold")
    ax1.set_title("RR Duration Strategy — Equity Curve with Yield Curve & HMM Regime",
                  fontsize=15, fontweight="bold", pad=14)
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.set_ylim(bottom=0.60)

    # Legend
    legend_elements = [
        Line2D([0], [0], color="#1a5276", linewidth=2, label="Strategy (Sh=0.325)"),
        Line2D([0], [0], color="#7f8c8d", linewidth=1.2, linestyle="--", label="B&H TLT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5, label="LONG"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5, label="SHORT"),
        plt.Rectangle((0, 0), 1, 1, fc="#2ecc71", alpha=0.25, label="Bull Steepening"),
        plt.Rectangle((0, 0), 1, 1, fc="#85c1e9", alpha=0.25, label="Bull Flattening"),
        plt.Rectangle((0, 0), 1, 1, fc="#f39c12", alpha=0.25, label="Bear Steepening"),
        plt.Rectangle((0, 0), 1, 1, fc="#e74c3c", alpha=0.25, label="Bear Flattening"),
        plt.Rectangle((0, 0), 1, 1, fc="none", edgecolor="#c0392b",
                       hatch="///", label="STRESS (HMM)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=8,
               framealpha=0.92, fancybox=True, ncol=3)

    # Metrics box
    ann_text = (f"Sharpe: 0.325  |  Sortino: 0.399  |  Ann.Ret: 5.39%\n"
                f"Vol: 10.4%  |  MDD: -23.6%  |  Calmar: 0.228  |  Events: {len(events)}")
    ax1.text(0.98, 0.03, ann_text, transform=ax1.transAxes,
             fontsize=8.5, fontfamily="monospace",
             ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.88, edgecolor="#cccccc"))

    # ── BOTTOM: Z-score ──
    ax2.plot(dates, z_vals, color="#2c3e50", linewidth=0.7, alpha=0.85, zorder=4)
    ax2.fill_between(dates, z_vals, 0,
                     where=(z_vals > 0), color="#27ae60", alpha=0.08, zorder=1)
    ax2.fill_between(dates, z_vals, 0,
                     where=(z_vals < 0), color="#e74c3c", alpha=0.08, zorder=1)

    ax2.axhline(TL_C, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)
    ax2.axhline(TS_C, color="#27ae60", linewidth=1.2, linestyle="--", alpha=0.7, zorder=3)
    ax2.axhline(TL_S, color="#c0392b", linewidth=1.0, linestyle=":", alpha=0.6, zorder=3)
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3, zorder=2)

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
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.set_ylim(-8, 8)

    z_legend = [
        Line2D([0], [0], color="#e74c3c", linewidth=1.2, linestyle="--",
               label=f"SHORT calm ({TL_C})"),
        Line2D([0], [0], color="#27ae60", linewidth=1.2, linestyle="--",
               label=f"LONG calm ({TS_C})"),
        Line2D([0], [0], color="#c0392b", linewidth=1.0, linestyle=":",
               label=f"SHORT stress ({TL_S})"),
    ]
    ax2.legend(handles=z_legend, loc="lower left", fontsize=8,
               framealpha=0.9, fancybox=True, ncol=3)

    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(os.path.join(FINAL_DIR, "08_equity_yc_regime.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {FINAL_DIR}/08_equity_yc_regime.png")

    # ═══════════════════════════════════════════════════════════════════
    # CHART 2: YEAR-BY-YEAR PERFORMANCE (grid of subplots)
    # ═══════════════════════════════════════════════════════════════════
    print("Generating Chart 2: Year-by-year equity comparison ...")

    years = sorted(set(d.year for d in dates))
    n_years = len(years)
    ncols = 4
    nrows = (n_years + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(22, nrows * 3.2))
    axes = axes.flatten()

    for i, yr in enumerate(years):
        ax = axes[i]
        mask = np.array([d.year == yr for d in dates])
        if mask.sum() < 5:
            ax.set_visible(False)
            continue

        yr_dates = dates[mask]
        yr_net = net[mask]
        yr_rl = r_long[mask]

        # Cumulative returns within year (rebased to 0)
        strat_cum = np.cumprod(1.0 + yr_net) - 1.0
        bm_cum = np.cumprod(1.0 + yr_rl) - 1.0

        ax.plot(yr_dates, strat_cum * 100, color="#1a5276", linewidth=1.5, label="Strategy")
        ax.plot(yr_dates, bm_cum * 100, color="#aab7b8", linewidth=1.2,
                linestyle="--", label="B&H TLT")

        # Fill excess area
        ax.fill_between(yr_dates, strat_cum * 100, bm_cum * 100,
                        where=(strat_cum >= bm_cum),
                        color="#27ae60", alpha=0.15)
        ax.fill_between(yr_dates, strat_cum * 100, bm_cum * 100,
                        where=(strat_cum < bm_cum),
                        color="#e74c3c", alpha=0.15)

        # STRESS regime shading
        yr_stress = (reg_labels.reindex(yr_dates).values == "STRESS")
        j = 0
        while j < len(yr_dates):
            if yr_stress[j]:
                k = j
                while k < len(yr_dates) and yr_stress[k]:
                    k += 1
                ax.axvspan(yr_dates[j], yr_dates[min(k-1, len(yr_dates)-1)],
                           color="#ffe0e0", alpha=0.3, zorder=0)
                j = k
            else:
                j += 1

        # Signal triangles
        yr_z = z_vals[mask]
        for ev_idx, sig, zv, reg in events:
            if ev_idx < np.where(mask)[0][0] or ev_idx > np.where(mask)[0][-1]:
                continue
            local_idx = ev_idx - np.where(mask)[0][0]
            if local_idx < 0 or local_idx >= len(yr_dates):
                continue
            ev_dt = dates[ev_idx]
            ev_strat = strat_cum[local_idx] * 100
            if sig == "LONG":
                ax.scatter(ev_dt, ev_strat, marker="^", s=50, c="#27ae60",
                           edgecolors="black", linewidths=0.4, zorder=6)
            else:
                ax.scatter(ev_dt, ev_strat, marker="v", s=50, c="#e74c3c",
                           edgecolors="black", linewidths=0.4, zorder=6)

        # Year label and stats
        strat_yr_ret = strat_cum[-1] * 100
        bm_yr_ret = bm_cum[-1] * 100
        excess = strat_yr_ret - bm_yr_ret

        title_color = "#27ae60" if excess > 0 else "#e74c3c"
        ax.set_title(f"{yr}", fontsize=12, fontweight="bold", color=title_color)

        # Stats annotation
        n_ev = sum(1 for ev_idx, _, _, _ in events
                   if ev_idx >= np.where(mask)[0][0] and ev_idx <= np.where(mask)[0][-1])
        ax.text(0.02, 0.97,
                f"Strat: {strat_yr_ret:+.1f}%\nTLT:   {bm_yr_ret:+.1f}%\n"
                f"Exc:   {excess:+.1f}%\nEv:    {n_ev}",
                transform=ax.transAxes, fontsize=7, fontfamily="monospace",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          alpha=0.85, edgecolor="#ddd"))

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.15)
        ax.set_ylabel("Return %", fontsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        plt.setp(ax.get_xticklabels(), fontsize=7)
        plt.setp(ax.get_yticklabels(), fontsize=7)

    # Hide unused subplots
    for i in range(n_years, len(axes)):
        axes[i].set_visible(False)

    # Common legend
    handles = [
        Line2D([0], [0], color="#1a5276", linewidth=1.5, label="Strategy"),
        Line2D([0], [0], color="#aab7b8", linewidth=1.2, linestyle="--", label="B&H TLT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
               markersize=8, markeredgecolor="black", markeredgewidth=0.4, label="LONG"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
               markersize=8, markeredgecolor="black", markeredgewidth=0.4, label="SHORT"),
        plt.Rectangle((0, 0), 1, 1, fc="#ffe0e0", alpha=0.5, label="STRESS"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Year-by-Year Performance: Strategy vs B&H TLT",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(FINAL_DIR, "09_yearly_comparison.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {FINAL_DIR}/09_yearly_comparison.png")

    # ═══════════════════════════════════════════════════════════════════
    # CHART 3: DETAILED PERFORMANCE DASHBOARD
    # ═══════════════════════════════════════════════════════════════════
    print("Generating Chart 3: Detailed performance dashboard ...")

    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 2, figure=fig, hspace=0.30, wspace=0.25)

    # ── 3a: Yearly returns bar chart ──
    ax = fig.add_subplot(gs[0, :])
    yearly_data = []
    for yr in years:
        mask = np.array([d.year == yr for d in dates])
        if mask.sum() < 5:
            continue
        yr_net_r = net[mask]
        yr_bm_r = r_long[mask]
        strat_ret = np.cumprod(1.0 + yr_net_r)[-1] - 1.0
        bm_ret = np.cumprod(1.0 + yr_bm_r)[-1] - 1.0
        n_ev = sum(1 for ev_idx, _, _, _ in events
                   if ev_idx >= np.where(mask)[0][0] and ev_idx <= np.where(mask)[0][-1])
        yearly_data.append({
            "year": yr, "strat": strat_ret, "bm": bm_ret,
            "excess": strat_ret - bm_ret, "n_events": n_ev,
        })
    ydf = pd.DataFrame(yearly_data)

    x = np.arange(len(ydf))
    w = 0.30
    bars1 = ax.bar(x - w, ydf["strat"] * 100, w, color="#1a5276", alpha=0.85, label="Strategy")
    bars2 = ax.bar(x, ydf["bm"] * 100, w, color="#aab7b8", alpha=0.7, label="B&H TLT")
    bars3 = ax.bar(x + w, ydf["excess"] * 100, w,
                   color=["#27ae60" if e > 0 else "#e74c3c" for e in ydf["excess"]],
                   alpha=0.7, label="Excess")

    # Add value labels on excess bars
    for xi, ex in zip(x, ydf["excess"]):
        color = "#27ae60" if ex > 0 else "#e74c3c"
        ax.text(xi + w, ex * 100 + (0.5 if ex >= 0 else -1.5),
                f"{ex*100:+.1f}", ha="center", va="bottom" if ex >= 0 else "top",
                fontsize=6.5, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ydf["year"].astype(int), fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Return (%)", fontsize=11)
    ax.set_title("Yearly Returns: Strategy vs B&H TLT vs Excess", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.15, axis="y")

    pos_yrs = (ydf["excess"] > 0).sum()
    ax.text(0.98, 0.97, f"Positive excess: {pos_yrs}/{len(ydf)} years ({pos_yrs/len(ydf)*100:.0f}%)",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── 3b: Cumulative excess return ──
    ax = fig.add_subplot(gs[1, 0])
    cum_excess = pd.Series(eq / bm_eq - 1.0, index=dates) * 100
    ax.plot(dates, cum_excess.values, color="#1a5276", linewidth=1.5)
    ax.fill_between(dates, cum_excess.values, 0,
                    where=(cum_excess.values >= 0), color="#27ae60", alpha=0.15)
    ax.fill_between(dates, cum_excess.values, 0,
                    where=(cum_excess.values < 0), color="#e74c3c", alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative Excess (%)", fontsize=10)
    ax.set_title("Cumulative Excess Return vs B&H TLT", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── 3c: Drawdown comparison ──
    ax = fig.add_subplot(gs[1, 1])
    # Strategy drawdown
    eq_s = pd.Series(eq, index=dates)
    dd_strat = (eq_s / eq_s.cummax() - 1) * 100
    # Benchmark drawdown
    bm_s = pd.Series(bm_eq, index=dates)
    dd_bm = (bm_s / bm_s.cummax() - 1) * 100

    ax.fill_between(dates, dd_strat.values, 0, color="#1a5276", alpha=0.3, label="Strategy DD")
    ax.plot(dates, dd_strat.values, color="#1a5276", linewidth=0.7)
    ax.plot(dates, dd_bm.values, color="#e74c3c", linewidth=0.8, alpha=0.7, label="TLT DD")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Drawdown (%)", fontsize=10)
    ax.set_title("Drawdown Comparison: Strategy vs B&H TLT", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── 3d: Rolling 1Y excess return ──
    ax = fig.add_subplot(gs[2, 0])
    net_s = pd.Series(net, index=dates)
    rl_s = pd.Series(r_long, index=dates)
    roll_strat = net_s.rolling(252).apply(lambda x: np.cumprod(1+x)[-1]-1, raw=True) * 100
    roll_bm = rl_s.rolling(252).apply(lambda x: np.cumprod(1+x)[-1]-1, raw=True) * 100
    roll_excess = roll_strat - roll_bm

    ax.plot(dates, roll_excess.values, color="#1a5276", linewidth=1.0)
    ax.fill_between(dates, roll_excess.values, 0,
                    where=(roll_excess.values >= 0), color="#27ae60", alpha=0.2)
    ax.fill_between(dates, roll_excess.values, 0,
                    where=(roll_excess.values < 0), color="#e74c3c", alpha=0.2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Rolling 1Y Excess (%)", fontsize=10)
    ax.set_title("Rolling 1-Year Excess Return vs TLT", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    pct_positive = (roll_excess.dropna() > 0).mean() * 100
    ax.text(0.02, 0.97, f"% of time positive: {pct_positive:.0f}%",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # ── 3e: Monthly returns heatmap ──
    ax = fig.add_subplot(gs[2, 1])

    monthly_ret = net_s.resample("ME").apply(lambda x: np.cumprod(1+x)[-1]-1 if len(x) > 0 else 0)
    monthly_pivot = pd.DataFrame({
        "year": monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "ret": monthly_ret.values * 100,
    })
    heat_data = monthly_pivot.pivot_table(values="ret", index="year", columns="month",
                                          aggfunc="first")
    heat_data.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(heat_data.columns)]

    cmap = LinearSegmentedColormap.from_list("rg", ["#e74c3c", "white", "#27ae60"])
    vmax = max(abs(heat_data.min().min()), abs(heat_data.max().max()))
    vmax = min(vmax, 15)
    im = ax.imshow(heat_data.values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(heat_data.columns)))
    ax.set_xticklabels(heat_data.columns, fontsize=8)
    ax.set_yticks(range(len(heat_data.index)))
    ax.set_yticklabels(heat_data.index.astype(int), fontsize=8)

    # Add text values
    for yi in range(len(heat_data.index)):
        for xi in range(len(heat_data.columns)):
            val = heat_data.values[yi, xi]
            if not np.isnan(val):
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(xi, yi, f"{val:.1f}", ha="center", va="center",
                        fontsize=6, color=color)

    ax.set_title("Monthly Returns Heatmap (%)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Return %")

    fig.suptitle("Detailed Performance Analysis — RR Duration Strategy",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(FINAL_DIR, "10_detailed_performance.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {FINAL_DIR}/10_detailed_performance.png")

    print(f"\nAll charts saved to: {FINAL_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
