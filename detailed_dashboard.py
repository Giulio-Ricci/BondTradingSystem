"""
detailed_dashboard.py — Dashboard dettagliato della strategia RR + Slope + YC.

Mostra TUTTI i segnali in pannelli sincronizzati:
  1. Equity curve + trade markers + drawdown
  2. z_final (segnale combinato) con soglie regime
  3. z_rr_base (Risk-Reversal puro, blend 4 tenors)
  4. z_slope_momentum (slope 10Y-2Y momentum, 126d diff, z-scored)
  5. z_yc_regime (yield curve regime score, 10Y-30Y)
  6. Yield curve slope 10Y-2Y (livello raw + media mobile)
  7. HMM regime (CALM / STRESS) + MOVE index
  8. TLT weight (posizione effettiva)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from config import BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime
from rr_yc_combined import (
    RR_PARAMS, YC_PARAMS, SCORE_MAP,
    build_z_rr, build_z_yc,
    bt_with_regime,
    COOLDOWN, DELAY, TCOST_BPS, W_INIT, ANN,
)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "output", "rr_yc_combined")
os.makedirs(SAVE_DIR, exist_ok=True)

# Best config from Method A
K_YC = 0.40
TL_C, TS_C, TL_S = -3.25, 2.50, -3.50


def build_all_signals(D):
    """Reconstruct every intermediate signal for the dashboard."""
    options = D["options"]
    yields = D["yields"]
    p = RR_PARAMS

    # ── 1. Individual RR tenors ──────────────────────────────────────
    from config import RR_TENORS
    rr_tenors = {}
    for tenor in p["rr_tenors"]:
        pc, cc = RR_TENORS[tenor]
        rr = (pd.to_numeric(options[pc], errors="coerce")
              - pd.to_numeric(options[cc], errors="coerce"))
        rr_tenors[tenor] = rr

    rr_blend = pd.concat(list(rr_tenors.values()), axis=1).mean(axis=1)

    # ── 2. z_rr_base ────────────────────────────────────────────────
    w, mp = p["z_window"], p["z_min_periods"]
    mu = rr_blend.rolling(w, min_periods=mp).mean()
    sd = rr_blend.rolling(w, min_periods=mp).std().replace(0, np.nan)
    z_base = (rr_blend - mu) / sd

    # ── 3. Slope raw + momentum ──────────────────────────────────────
    slope_raw = yields[p["slope_long"]] - yields[p["slope_short"]]
    ds = slope_raw.diff(p["slope_mom_window"])
    sw, sm = p["slope_z_window"], p["slope_z_min"]
    mu_s = ds.rolling(sw, min_periods=sm).mean()
    sd_s = ds.rolling(sw, min_periods=sm).std().replace(0, np.nan)
    z_slope = (ds - mu_s) / sd_s

    # ── 4. z_rr composite ────────────────────────────────────────────
    z_rr = z_base + p["k_slope"] * z_slope

    # ── 5. YC regime score ───────────────────────────────────────────
    yp = YC_PARAMS
    sc, lc = yp["spread_short"], yp["spread_long"]
    cw = yp["change_window"]
    lev = (yields[sc] + yields[lc]) / 2
    spr = yields[lc] - yields[sc]
    dl, ds_yc = lev.diff(cw), spr.diff(cw)

    regime_label = pd.Series("UNDEFINED", index=yields.index)
    regime_label[(dl < 0) & (ds_yc > 0)] = "BULL_STEEP"
    regime_label[(dl > 0) & (ds_yc > 0)] = "BEAR_STEEP"
    regime_label[(dl < 0) & (ds_yc < 0)] = "BULL_FLAT"
    regime_label[(dl > 0) & (ds_yc < 0)] = "BEAR_FLAT"

    score = regime_label.map(SCORE_MAP).astype(float)
    zw, zm = yp["z_window"], yp["z_min_periods"]
    mu_yc = score.rolling(zw, min_periods=zm).mean()
    sd_yc = score.rolling(zw, min_periods=zm).std().replace(0, np.nan)
    z_yc = (score - mu_yc) / sd_yc

    # ── 6. z_final ───────────────────────────────────────────────────
    z_final = z_rr + K_YC * z_yc

    return dict(
        rr_blend=rr_blend,
        rr_tenors=rr_tenors,
        z_base=z_base,
        slope_raw=slope_raw,
        z_slope=z_slope,
        z_rr=z_rr,
        yc_regime_label=regime_label,
        yc_score=score,
        z_yc=z_yc,
        z_final=z_final,
        spread_10s30s=spr,
    )


def main():
    print("Loading data...")
    D = load_all()

    print("Fitting HMM regime...")
    regime = fit_hmm_regime(D["move"], D["vix"])

    print("Building all signals...")
    S = build_all_signals(D)

    etf_ret = D["etf_ret"]
    bt_start = pd.Timestamp(BACKTEST_START)

    # Align
    common = (S["z_final"].dropna().index
              .intersection(S["z_yc"].dropna().index)
              .intersection(regime.dropna().index)
              .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index))
    common = common[common >= bt_start].sort_values()

    z_final_a = S["z_final"].reindex(common).values
    z_rr_a = S["z_rr"].reindex(common).values
    z_base_a = S["z_base"].reindex(common).values
    z_slope_a = S["z_slope"].reindex(common).values
    z_yc_a = S["z_yc"].reindex(common).values
    reg_a = (regime.reindex(common) == "STRESS").astype(int).values
    ret_tlt = etf_ret["TLT"].reindex(common).fillna(0).values
    ret_shv = etf_ret["SHV"].reindex(common).fillna(0).values
    slope_raw = S["slope_raw"].reindex(common).values
    move_a = D["move"].reindex(common).values
    vix_a = D["vix"].reindex(common).values
    dates = common.to_numpy()

    # Run best backtest
    print("Running backtest...")
    res = bt_with_regime(z_final_a, reg_a, ret_tlt, ret_shv, TL_C, TS_C, TL_S)
    equity = res["equity"]
    aw = res["actual_w"]

    # B&H
    eq_bh = np.cumprod(1 + ret_tlt)

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak - 1) * 100

    peak_bh = np.maximum.accumulate(eq_bh)
    dd_bh = (eq_bh / peak_bh - 1) * 100

    # Trade markers
    dw = np.abs(np.diff(aw, prepend=aw[0]))
    trade_idx = np.where(dw > 0.01)[0]

    # Thresholds per regime
    tl_arr = np.where(reg_a == 0, TL_C, TL_S)
    ts_arr = np.where(reg_a == 0, TS_C, 99.0)

    n_days = len(common)
    print(f"Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d} ({n_days} days)")
    print(f"Sharpe={res['sharpe']:.3f}, Return={res['ann_ret']:+.2%}, "
          f"MDD={res['mdd']:.2%}, Trades={res['n_trades']}")

    # ══════════════════════════════════════════════════════════════════
    # PLOT
    # ══════════════════════════════════════════════════════════════════
    print("\nGenerating detailed dashboard...")

    fig, axes = plt.subplots(
        8, 1, figsize=(24, 32),
        height_ratios=[3.5, 2, 1.8, 1.8, 1.8, 1.5, 1.5, 1.0],
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    # Colors
    C_STRAT = "#1f77b4"
    C_BH = "#999999"
    C_LONG = "#2ca02c"
    C_SHORT = "#d62728"
    C_CALM = "#e8f5e9"
    C_STRESS = "#ffebee"
    C_SLOPE = "#ff7f0e"
    C_YC = "#9467bd"
    C_MOVE = "#8c564b"

    # Helper: shade STRESS regime on all panels
    def shade_stress(ax):
        stress_mask = reg_a == 1
        i = 0
        while i < n_days:
            if stress_mask[i]:
                j = i
                while j < n_days and stress_mask[j]:
                    j += 1
                ax.axvspan(dates[i], dates[min(j, n_days-1)],
                           color=C_STRESS, alpha=0.25, zorder=0)
                i = j
            else:
                i += 1

    # ── Panel 1: Equity + Drawdown ────────────────────────────────────
    ax1 = axes[0]
    shade_stress(ax1)
    ax1.plot(dates, equity, color=C_STRAT, lw=1.8, label="RR+Slope+YC Strategy", zorder=3)
    ax1.plot(dates, eq_bh, color=C_BH, lw=1.2, alpha=0.6, label="B&H TLT", zorder=2)

    # Trade markers on equity
    for ti in trade_idx:
        if aw[ti] > 0.7:
            ax1.plot(dates[ti], equity[ti], "^", color=C_LONG, ms=9, zorder=5,
                     markeredgecolor="black", markeredgewidth=0.5)
        else:
            ax1.plot(dates[ti], equity[ti], "v", color=C_SHORT, ms=9, zorder=5,
                     markeredgecolor="black", markeredgewidth=0.5)

    ax1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    ax1.set_yscale("log")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.1f}"))
    ax1.set_title(
        f"RR + Slope Curve Strategy — Detailed Dashboard\n"
        f"Sharpe={res['sharpe']:.3f}  |  Ann.Return={res['ann_ret']:+.2%}  |  "
        f"MDD={res['mdd']:.2%}  |  Sortino={res['sortino']:.3f}  |  "
        f"Trades={res['n_trades']}  |  Period: {common[0]:%Y-%m-%d} to {common[-1]:%Y-%m-%d}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax1.grid(True, alpha=0.3)

    # Drawdown as filled area on twin axis
    ax1b = ax1.twinx()
    ax1b.fill_between(dates, dd, 0, color=C_STRAT, alpha=0.12, zorder=1)
    ax1b.fill_between(dates, dd_bh, 0, color=C_BH, alpha=0.08, zorder=0)
    ax1b.set_ylabel("Drawdown (%)", fontsize=9, color="gray")
    ax1b.set_ylim(-60, 5)
    ax1b.tick_params(axis="y", labelcolor="gray", labelsize=8)

    legend_handles = [
        Line2D([0], [0], color=C_STRAT, lw=2, label="Strategy"),
        Line2D([0], [0], color=C_BH, lw=1.2, alpha=0.6, label="B&H TLT"),
        Line2D([0], [0], marker="^", color=C_LONG, lw=0, ms=9, label="LONG signal"),
        Line2D([0], [0], marker="v", color=C_SHORT, lw=0, ms=9, label="SHORT signal"),
        Patch(facecolor=C_STRESS, alpha=0.3, label="STRESS regime"),
        Patch(facecolor=C_STRAT, alpha=0.15, label="Drawdown"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=8, ncol=3)

    # ── Panel 2: z_final (combined signal) with thresholds ────────────
    ax2 = axes[1]
    shade_stress(ax2)
    ax2.plot(dates, z_final_a, color=C_STRAT, lw=0.8, alpha=0.85, label="z_final")
    ax2.plot(dates, tl_arr, color=C_SHORT, lw=1.0, ls="--", alpha=0.7, label="SHORT threshold")
    ax2.plot(dates, ts_arr, color=C_LONG, lw=1.0, ls="--", alpha=0.7, label="LONG threshold")
    ax2.axhline(0, color="gray", lw=0.5)

    # Mark triggered signals
    for ti in trade_idx:
        c = C_LONG if aw[ti] > 0.7 else C_SHORT
        mk = "^" if aw[ti] > 0.7 else "v"
        ax2.plot(dates[ti], z_final_a[ti], mk, color=c, ms=10,
                 markeredgecolor="black", markeredgewidth=0.5, zorder=5)

    ax2.set_ylabel("z_final\n(RR + 0.4*YC)", fontsize=10, fontweight="bold")
    ax2.set_ylim(-8, 8)
    ax2.legend(loc="upper right", fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: z_rr_base (pure RR from options) ─────────────────────
    ax3 = axes[2]
    shade_stress(ax3)
    ax3.plot(dates, z_base_a, color="#1a5276", lw=0.7, alpha=0.8, label="z_rr_base (4-tenor blend)")
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.fill_between(dates, z_base_a, 0, where=z_base_a > 0,
                     color=C_LONG, alpha=0.15, interpolate=True)
    ax3.fill_between(dates, z_base_a, 0, where=z_base_a < 0,
                     color=C_SHORT, alpha=0.15, interpolate=True)
    ax3.set_ylabel("z_rr_base\n(Options RR)", fontsize=10, fontweight="bold")
    ax3.set_ylim(-6, 6)
    ax3.legend(loc="upper right", fontsize=7)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: z_slope momentum ─────────────────────────────────────
    ax4 = axes[3]
    shade_stress(ax4)
    ax4.plot(dates, z_slope_a, color=C_SLOPE, lw=0.7, alpha=0.8,
             label="z_slope_momentum (10Y-2Y, 126d)")
    ax4.axhline(0, color="gray", lw=0.5)
    ax4.fill_between(dates, z_slope_a, 0, where=z_slope_a > 0,
                     color=C_SLOPE, alpha=0.12, interpolate=True,
                     label="Steepening momentum")
    ax4.fill_between(dates, z_slope_a, 0, where=z_slope_a < 0,
                     color="#7b241c", alpha=0.12, interpolate=True,
                     label="Flattening momentum")
    ax4.set_ylabel("z_slope_mom\n(Curve momentum)", fontsize=10, fontweight="bold")
    ax4.set_ylim(-5, 5)
    ax4.legend(loc="upper right", fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: z_yc regime ──────────────────────────────────────────
    ax5 = axes[4]
    shade_stress(ax5)
    ax5.plot(dates, z_yc_a, color=C_YC, lw=0.7, alpha=0.8,
             label="z_yc_regime (10Y-30Y regime score)")
    ax5.axhline(0, color="gray", lw=0.5)
    ax5.fill_between(dates, z_yc_a, 0, where=z_yc_a > 0,
                     color="#7d3c98", alpha=0.12, interpolate=True,
                     label="Bullish YC regime")
    ax5.fill_between(dates, z_yc_a, 0, where=z_yc_a < 0,
                     color="#c0392b", alpha=0.12, interpolate=True,
                     label="Bearish YC regime")
    ax5.set_ylabel("z_yc_regime\n(YC regime z)", fontsize=10, fontweight="bold")
    ax5.set_ylim(-4, 4)
    ax5.legend(loc="upper right", fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)

    # ── Panel 6: Yield Curve Slope (raw 10Y-2Y) ──────────────────────
    ax6 = axes[5]
    shade_stress(ax6)
    ax6.plot(dates, slope_raw, color="#117864", lw=0.9, alpha=0.8, label="10Y-2Y spread")
    # Rolling mean
    slope_s = pd.Series(slope_raw, index=common)
    slope_ma = slope_s.rolling(126, min_periods=20).mean().values
    ax6.plot(dates, slope_ma, color="#117864", lw=1.5, ls="--", alpha=0.5,
             label="126d MA")
    ax6.axhline(0, color="black", lw=0.8, ls="-")
    ax6.fill_between(dates, slope_raw, 0, where=slope_raw > 0,
                     color="#27ae60", alpha=0.1, interpolate=True)
    ax6.fill_between(dates, slope_raw, 0, where=slope_raw < 0,
                     color="#e74c3c", alpha=0.15, interpolate=True, label="Inverted curve")
    ax6.set_ylabel("10Y-2Y Spread\n(%)", fontsize=10, fontweight="bold")
    ax6.legend(loc="upper right", fontsize=7, ncol=3)
    ax6.grid(True, alpha=0.3)

    # ── Panel 7: MOVE + VIX + HMM Regime ─────────────────────────────
    ax7 = axes[6]
    shade_stress(ax7)
    ax7.plot(dates, move_a, color=C_MOVE, lw=0.8, alpha=0.8, label="MOVE Index")
    ax7b = ax7.twinx()
    ax7b.plot(dates, vix_a, color="#e67e22", lw=0.6, alpha=0.5, label="VIX")
    ax7b.set_ylabel("VIX", fontsize=9, color="#e67e22")
    ax7b.tick_params(axis="y", labelcolor="#e67e22", labelsize=8)
    ax7.set_ylabel("MOVE Index\n+ HMM Regime", fontsize=10, fontweight="bold")
    h1 = [Line2D([0], [0], color=C_MOVE, lw=1.2, label="MOVE"),
           Line2D([0], [0], color="#e67e22", lw=0.8, alpha=0.6, label="VIX"),
           Patch(facecolor=C_STRESS, alpha=0.3, label="STRESS")]
    ax7.legend(handles=h1, loc="upper right", fontsize=7, ncol=3)
    ax7.grid(True, alpha=0.3)

    # ── Panel 8: Position (TLT weight) ───────────────────────────────
    ax8 = axes[7]
    shade_stress(ax8)
    ax8.fill_between(dates, aw, 0.5, where=aw > 0.7,
                     color=C_LONG, alpha=0.5, step="post", label="LONG TLT (100%)")
    ax8.fill_between(dates, aw, 0.5, where=aw < 0.3,
                     color=C_SHORT, alpha=0.5, step="post", label="SHORT TLT (0%)")
    ax8.fill_between(dates, aw, 0.5, where=(aw >= 0.3) & (aw <= 0.7),
                     color="#f0b27a", alpha=0.4, step="post", label="NEUTRAL (50%)")
    ax8.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax8.axhline(1.0, color=C_LONG, ls=":", lw=0.5, alpha=0.5)
    ax8.axhline(0.0, color=C_SHORT, ls=":", lw=0.5, alpha=0.5)
    ax8.set_ylabel("TLT Weight", fontsize=10, fontweight="bold")
    ax8.set_ylim(-0.05, 1.05)
    ax8.set_yticks([0, 0.5, 1.0])
    ax8.set_yticklabels(["0% (SHV)", "50%", "100% (TLT)"])
    ax8.legend(loc="upper right", fontsize=7, ncol=3)
    ax8.grid(True, alpha=0.3)

    # X-axis formatting
    ax8.xaxis.set_major_locator(mdates.YearLocator())
    ax8.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Global
    for ax in axes:
        ax.tick_params(axis="both", labelsize=8)

    path = os.path.join(SAVE_DIR, "04_detailed_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {path}")

    # ══════════════════════════════════════════════════════════════════
    # ZOOMED PANELS (last 3 years)
    # ══════════════════════════════════════════════════════════════════
    print("Generating zoomed dashboard (last 3 years)...")

    zoom_start = pd.Timestamp("2023-01-01")
    zoom_mask = common >= zoom_start
    zi = np.where(zoom_mask)[0]
    if len(zi) < 50:
        print("Not enough data for zoom. Skipping.")
        return

    zd = dates[zoom_mask]
    n_zoom = len(zd)

    fig2, axes2 = plt.subplots(
        8, 1, figsize=(24, 32),
        height_ratios=[3.5, 2, 1.8, 1.8, 1.8, 1.5, 1.5, 1.0],
        sharex=True,
        gridspec_kw={"hspace": 0.06},
    )

    def shade_stress_zoom(ax):
        stress_z = reg_a[zoom_mask] == 1
        idx = np.arange(n_zoom)
        i = 0
        while i < n_zoom:
            if stress_z[i]:
                j = i
                while j < n_zoom and stress_z[j]:
                    j += 1
                ax.axvspan(zd[i], zd[min(j, n_zoom-1)],
                           color=C_STRESS, alpha=0.25, zorder=0)
                i = j
            else:
                i += 1

    # Panel 1 zoom: Equity
    ax1z = axes2[0]
    shade_stress_zoom(ax1z)
    eq_z = equity[zoom_mask] / equity[zi[0]]
    bh_z = eq_bh[zoom_mask] / eq_bh[zi[0]]
    ax1z.plot(zd, eq_z, color=C_STRAT, lw=2, label="Strategy")
    ax1z.plot(zd, bh_z, color=C_BH, lw=1.2, alpha=0.6, label="B&H TLT")
    for ti in trade_idx:
        if ti in zi:
            c = C_LONG if aw[ti] > 0.7 else C_SHORT
            mk = "^" if aw[ti] > 0.7 else "v"
            ax1z.plot(dates[ti], equity[ti]/equity[zi[0]], mk, color=c, ms=11,
                      markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax1z.set_ylabel("Equity (rebased)", fontsize=11, fontweight="bold")
    ax1z.set_title(
        f"ZOOMED: {zoom_start:%Y-%m-%d} to {common[-1]:%Y-%m-%d}",
        fontsize=13, fontweight="bold", pad=10)
    ax1z.legend(loc="upper left", fontsize=8)
    ax1z.grid(True, alpha=0.3)

    # Panel 2 zoom: z_final
    ax2z = axes2[1]
    shade_stress_zoom(ax2z)
    ax2z.plot(zd, z_final_a[zoom_mask], color=C_STRAT, lw=1.0)
    ax2z.plot(zd, tl_arr[zoom_mask], color=C_SHORT, lw=1.2, ls="--", alpha=0.7)
    ax2z.plot(zd, ts_arr[zoom_mask], color=C_LONG, lw=1.2, ls="--", alpha=0.7)
    ax2z.axhline(0, color="gray", lw=0.5)
    for ti in trade_idx:
        if ti in zi:
            c = C_LONG if aw[ti] > 0.7 else C_SHORT
            mk = "^" if aw[ti] > 0.7 else "v"
            ax2z.plot(dates[ti], z_final_a[ti], mk, color=c, ms=12,
                      markeredgecolor="black", markeredgewidth=0.5, zorder=5)
    ax2z.set_ylabel("z_final", fontsize=10, fontweight="bold")
    ax2z.grid(True, alpha=0.3)

    # Panel 3 zoom: z_rr_base
    ax3z = axes2[2]
    shade_stress_zoom(ax3z)
    ax3z.plot(zd, z_base_a[zoom_mask], color="#1a5276", lw=0.9)
    ax3z.axhline(0, color="gray", lw=0.5)
    ax3z.fill_between(zd, z_base_a[zoom_mask], 0,
                      where=z_base_a[zoom_mask] > 0, color=C_LONG, alpha=0.15,
                      interpolate=True)
    ax3z.fill_between(zd, z_base_a[zoom_mask], 0,
                      where=z_base_a[zoom_mask] < 0, color=C_SHORT, alpha=0.15,
                      interpolate=True)
    ax3z.set_ylabel("z_rr_base\n(Options RR)", fontsize=10, fontweight="bold")
    ax3z.grid(True, alpha=0.3)

    # Panel 4 zoom: z_slope
    ax4z = axes2[3]
    shade_stress_zoom(ax4z)
    ax4z.plot(zd, z_slope_a[zoom_mask], color=C_SLOPE, lw=0.9)
    ax4z.axhline(0, color="gray", lw=0.5)
    ax4z.fill_between(zd, z_slope_a[zoom_mask], 0,
                      where=z_slope_a[zoom_mask] > 0, color=C_SLOPE, alpha=0.12,
                      interpolate=True)
    ax4z.fill_between(zd, z_slope_a[zoom_mask], 0,
                      where=z_slope_a[zoom_mask] < 0, color="#7b241c", alpha=0.12,
                      interpolate=True)
    ax4z.set_ylabel("z_slope_mom\n(10Y-2Y)", fontsize=10, fontweight="bold")
    ax4z.grid(True, alpha=0.3)

    # Panel 5 zoom: z_yc
    ax5z = axes2[4]
    shade_stress_zoom(ax5z)
    ax5z.plot(zd, z_yc_a[zoom_mask], color=C_YC, lw=0.9)
    ax5z.axhline(0, color="gray", lw=0.5)
    ax5z.fill_between(zd, z_yc_a[zoom_mask], 0,
                      where=z_yc_a[zoom_mask] > 0, color="#7d3c98", alpha=0.12,
                      interpolate=True)
    ax5z.fill_between(zd, z_yc_a[zoom_mask], 0,
                      where=z_yc_a[zoom_mask] < 0, color="#c0392b", alpha=0.12,
                      interpolate=True)
    ax5z.set_ylabel("z_yc_regime\n(10Y-30Y)", fontsize=10, fontweight="bold")
    ax5z.grid(True, alpha=0.3)

    # Panel 6 zoom: Slope raw
    ax6z = axes2[5]
    shade_stress_zoom(ax6z)
    ax6z.plot(zd, slope_raw[zoom_mask], color="#117864", lw=1.0)
    ax6z.plot(zd, slope_ma[zoom_mask], color="#117864", lw=1.5, ls="--", alpha=0.5)
    ax6z.axhline(0, color="black", lw=0.8)
    ax6z.fill_between(zd, slope_raw[zoom_mask], 0,
                      where=slope_raw[zoom_mask] > 0, color="#27ae60", alpha=0.1,
                      interpolate=True)
    ax6z.fill_between(zd, slope_raw[zoom_mask], 0,
                      where=slope_raw[zoom_mask] < 0, color="#e74c3c", alpha=0.15,
                      interpolate=True)
    ax6z.set_ylabel("10Y-2Y (%)", fontsize=10, fontweight="bold")
    ax6z.grid(True, alpha=0.3)

    # Panel 7 zoom: MOVE + VIX
    ax7z = axes2[6]
    shade_stress_zoom(ax7z)
    ax7z.plot(zd, move_a[zoom_mask], color=C_MOVE, lw=1.0)
    ax7zb = ax7z.twinx()
    ax7zb.plot(zd, vix_a[zoom_mask], color="#e67e22", lw=0.7, alpha=0.5)
    ax7zb.set_ylabel("VIX", fontsize=9, color="#e67e22")
    ax7zb.tick_params(axis="y", labelcolor="#e67e22", labelsize=8)
    ax7z.set_ylabel("MOVE + Regime", fontsize=10, fontweight="bold")
    ax7z.grid(True, alpha=0.3)

    # Panel 8 zoom: Position
    ax8z = axes2[7]
    shade_stress_zoom(ax8z)
    aw_z = aw[zoom_mask]
    ax8z.fill_between(zd, aw_z, 0.5, where=aw_z > 0.7,
                      color=C_LONG, alpha=0.5, step="post", label="LONG")
    ax8z.fill_between(zd, aw_z, 0.5, where=aw_z < 0.3,
                      color=C_SHORT, alpha=0.5, step="post", label="SHORT")
    ax8z.fill_between(zd, aw_z, 0.5, where=(aw_z >= 0.3) & (aw_z <= 0.7),
                      color="#f0b27a", alpha=0.4, step="post", label="NEUTRAL")
    ax8z.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax8z.set_ylabel("TLT Weight", fontsize=10, fontweight="bold")
    ax8z.set_ylim(-0.05, 1.05)
    ax8z.set_yticks([0, 0.5, 1.0])
    ax8z.set_yticklabels(["0% (SHV)", "50%", "100% (TLT)"])
    ax8z.legend(loc="upper right", fontsize=7, ncol=3)
    ax8z.grid(True, alpha=0.3)

    ax8z.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax8z.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax8z.xaxis.get_majorticklabels(), rotation=45, ha="right")

    for ax in axes2:
        ax.tick_params(axis="both", labelsize=8)

    path2 = os.path.join(SAVE_DIR, "05_detailed_dashboard_zoom.png")
    fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path2}")

    print("\nDone!")


if __name__ == "__main__":
    main()
