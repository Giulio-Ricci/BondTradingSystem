"""
versione_finale_v2.py - VERSIONE FINALE V2 della strategia RR+YC Duration.

Evoluzione della versione originale (RR-only, Sharpe 0.325) con aggiunta
del segnale Yield Curve regime_z basato sulla classificazione
Bull/Bear Steepening/Flattening del segmento 10s30s.

Segnale composito:
    z_final = z_rr_composite + 0.40 * z_yc_regime_z

Dove:
    z_rr_composite = z_base_rr(ALL|std|w63) + 0.50 * z_slope_momentum(10Y-2Y)
    z_yc_regime_z  = z-score(regime_score(10s30s), cw=42, zw=252)

Thresholds: tl_calm=-3.25, ts_calm=2.50, tl_stress=-3.50, cooldown=5

Risultati attesi:
    Sharpe:     0.396  (+22% vs V1)
    Sortino:    0.493
    Ann.Return: 5.76%
    Vol:        ~10%
    Max DD:     -23.6%
    Calmar:     0.244
    Events:     19
    Eq Final:   ~$2.60 (da $1.00)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

# ======================================================================
# PARAMETRI FINALI V2 (FROZEN)
# ======================================================================

PARAMS = {
    # --- Segnale base: RR z-score ---
    "rr_tenors":         ["1W", "1M", "3M", "6M"],
    "z_type":            "std",
    "z_window":          63,
    "z_min_periods":     20,

    # --- Componente slope momentum ---
    "slope_long_yield":  "US_10Y",
    "slope_short_yield": "US_2Y",
    "slope_mom_window":  126,
    "slope_zscore_window": 252,
    "slope_zscore_min_periods": 63,
    "k_slope":           0.50,

    # --- Componente YC regime_z (NUOVA in V2) ---
    "yc_spread_short":   "US_10Y",
    "yc_spread_long":    "US_30Y",
    "yc_change_window":  42,
    "yc_z_window":       252,
    "yc_z_min_periods":  20,
    "k_yc":              0.40,

    # --- Soglie di trading (aggiornate in V2) ---
    "tl_calm":           -3.25,
    "ts_calm":           +2.50,       # era +2.75 in V1
    "tl_stress":         -3.50,       # era -4.00 in V1
    "ts_stress":         99,
    "allow_long_stress": False,

    # --- Esecuzione ---
    "cooldown":          5,
    "delay":             1,
    "tcost_bps":         5,

    # --- Allocazione ---
    "w_initial":         0.50,
    "w_long":            1.00,
    "w_short":           0.00,

    # --- Strumenti ---
    "long_etf":          "TLT",
    "short_etf":         "SHV",

    # --- Regime HMM ---
    "hmm_n_seeds":       10,
    "hmm_n_iter":        300,
}

# Regime scoring per YC
SCORE_MAP = {
    "BULL_FLAT": +2, "BULL_STEEP": +1,
    "UNDEFINED": 0,
    "BEAR_FLAT": -1, "BEAR_STEEP": -2,
}

# Output
OUT_DIR = os.path.join(SAVE_DIR, "versione_finale_v2")
os.makedirs(OUT_DIR, exist_ok=True)


# ======================================================================
# COSTRUZIONE DEI SEGNALI
# ======================================================================

def build_rr_signal(D, params):
    """
    Costruisce il segnale RR composito (identico a V1):
        z_rr = z_base_rr + 0.50 * z_slope_momentum
    """
    options = D["options"]
    yields = D["yields"]

    # RR blend (media di tutti i tenors)
    rr_parts = []
    for tenor in params["rr_tenors"]:
        put_col, call_col = RR_TENORS[tenor]
        rr = (pd.to_numeric(options[put_col], errors="coerce")
              - pd.to_numeric(options[call_col], errors="coerce"))
        rr_parts.append(rr)
    rr_blend = pd.concat(rr_parts, axis=1).mean(axis=1)

    # Z-score standard del RR blend
    w = params["z_window"]
    mp = params["z_min_periods"]
    mu = rr_blend.rolling(w, min_periods=mp).mean()
    sig = rr_blend.rolling(w, min_periods=mp).std()
    z_base = (rr_blend - mu) / sig.replace(0, np.nan)

    # Slope momentum z-score
    col_l = params["slope_long_yield"]
    col_s = params["slope_short_yield"]
    slope = yields[col_l] - yields[col_s]
    delta_slope = slope.diff(params["slope_mom_window"])

    sz_w = params["slope_zscore_window"]
    sz_mp = params["slope_zscore_min_periods"]
    mu_slope = delta_slope.rolling(sz_w, min_periods=sz_mp).mean()
    sig_slope = delta_slope.rolling(sz_w, min_periods=sz_mp).std()
    z_slope = (delta_slope - mu_slope) / sig_slope.replace(0, np.nan)

    # Composito RR
    z_rr = z_base + params["k_slope"] * z_slope

    return z_rr, z_base, z_slope


def build_yc_signal(D, params):
    """
    Costruisce il segnale YC regime_z (NUOVO in V2):
        1. Classifica regime YC: Bull/Bear Steepening/Flattening
        2. Assegna score: BULL_FLAT=+2, BULL_STEEP=+1, BEAR_FLAT=-1, BEAR_STEEP=-2
        3. Z-score rolling dello score
    """
    yld = D["yields"]
    sc = params["yc_spread_short"]
    lc = params["yc_spread_long"]
    cw = params["yc_change_window"]

    # Livello e spread
    level = (yld[sc] + yld[lc]) / 2
    spread = yld[lc] - yld[sc]

    # Variazione su finestra cw
    d_level = level.diff(cw)
    d_spread = spread.diff(cw)

    # Classificazione regime
    regime_yc = pd.Series("UNDEFINED", index=yld.index)
    regime_yc[(d_level < 0) & (d_spread > 0)] = "BULL_STEEP"
    regime_yc[(d_level > 0) & (d_spread > 0)] = "BEAR_STEEP"
    regime_yc[(d_level < 0) & (d_spread < 0)] = "BULL_FLAT"
    regime_yc[(d_level > 0) & (d_spread < 0)] = "BEAR_FLAT"

    # Score numerico
    score = regime_yc.map(SCORE_MAP).astype(float)

    # Z-score rolling
    zw = params["yc_z_window"]
    zm = params["yc_z_min_periods"]
    mu = score.rolling(zw, min_periods=zm).mean()
    sd = score.rolling(zw, min_periods=zm).std().replace(0, np.nan)
    z_yc = (score - mu) / sd

    return z_yc, regime_yc, score


def build_combined_signal(D, params):
    """
    Costruisce il segnale finale combinato:
        z_final = z_rr + k_yc * z_yc
    """
    z_rr, z_base, z_slope = build_rr_signal(D, params)
    z_yc, regime_yc, yc_score = build_yc_signal(D, params)

    z_final = z_rr + params["k_yc"] * z_yc

    return z_final, z_rr, z_yc, z_base, z_slope, regime_yc, yc_score


# ======================================================================
# BACKTEST
# ======================================================================

def run_backtest(z, regime, etf_ret, params, start):
    """Esegue il backtest completo. Identico a V1 nella logica."""
    p = params
    tl_c = p["tl_calm"]
    ts_c = p["ts_calm"]
    tl_s = p["tl_stress"]
    cd = p["cooldown"]
    delay = p["delay"]
    tcost = p["tcost_bps"]
    w_init = p["w_initial"]
    long_etf = p["long_etf"]
    short_etf = p["short_etf"]

    # Allineamento date
    common = (z.dropna().index
              .intersection(regime.dropna().index)
              .intersection(etf_ret.dropna(subset=[long_etf, short_etf]).index))
    common = common[common >= start].sort_values()

    n = len(common)
    z_vals = z.reindex(common).values
    reg_vals = (regime.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret[long_etf].reindex(common).fillna(0).values
    r_short = etf_ret[short_etf].reindex(common).fillna(0).values

    # Generazione segnali e pesi
    weight = np.empty(n)
    weight[0] = w_init
    events = []
    last_ev = -cd - 1

    for i in range(n):
        zv = z_vals[i]
        w_prev = weight[i - 1] if i > 0 else w_init
        if np.isnan(zv):
            weight[i] = w_prev
            continue
        reg = reg_vals[i]
        triggered = False
        sig = None

        if reg == 0:  # CALM
            if zv <= tl_c and (i - last_ev) >= cd:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            elif zv >= ts_c and (i - last_ev) >= cd:
                weight[i] = 1.0; triggered = True; sig = "LONG"
            else:
                weight[i] = w_prev
        else:  # STRESS
            if zv <= tl_s and (i - last_ev) >= cd:
                weight[i] = 0.0; triggered = True; sig = "SHORT"
            else:
                weight[i] = w_prev

        if triggered:
            last_ev = i
            events.append({
                "date": common[i],
                "signal": sig,
                "z": zv,
                "regime": "CALM" if reg == 0 else "STRESS",
            })

    # Execution delay
    shift = 1 + delay
    w_del = np.empty(n)
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    # Rendimenti portafoglio
    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost / 10_000)

    # Equity curve
    equity = pd.Series(np.cumprod(1.0 + net), index=common)

    # Metriche
    years = n / 252.0
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    ann_ret = (1.0 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0
    vol = np.std(net) * np.sqrt(252)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0

    down = net[net < 0]
    down_vol = np.std(down) * np.sqrt(252) if len(down) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol

    running_max = equity.cummax()
    dd = equity / running_max - 1
    mdd = dd.min()
    calmar = ann_ret / abs(mdd) if mdd != 0 else 0

    bm_eq = pd.Series(np.cumprod(1.0 + r_long), index=common)
    bm_total = bm_eq.iloc[-1] - 1.0
    bm_ann = (1.0 + bm_total) ** (1.0 / max(years, 0.01)) - 1.0
    excess = ann_ret - bm_ann

    # B&H SHV benchmark
    shv_eq = pd.Series(np.cumprod(1.0 + r_short), index=common)
    shv_total = shv_eq.iloc[-1] - 1.0
    shv_ann = (1.0 + shv_total) ** (1.0 / max(years, 0.01)) - 1.0

    metrics = {
        "sharpe": sharpe, "sortino": sortino, "ann_ret": ann_ret,
        "vol": vol, "mdd": mdd, "calmar": calmar, "excess": excess,
        "total_ret": total_ret, "eq_final": equity.iloc[-1],
        "bm_ann": bm_ann, "shv_ann": shv_ann,
        "n_events": len(events), "n_days": n, "years": years,
    }

    # Yearly breakdown
    yearly_rows = []
    for y in sorted(set(d.year for d in common)):
        mask = np.array([d.year == y for d in common])
        if mask.sum() < 5:
            continue
        nr = net[mask]
        rl = r_long[mask]
        eq_y = np.cumprod(1.0 + nr)
        bm_y = np.cumprod(1.0 + rl)
        yr_ret = eq_y[-1] - 1.0
        bm_ret = bm_y[-1] - 1.0
        yr_vol = np.std(nr) * np.sqrt(252)
        n_ev = sum(1 for e in events if e["date"].year == y)
        yearly_rows.append({
            "year": y, "strat_ret": yr_ret, "bm_ret": bm_ret,
            "excess": yr_ret - bm_ret, "vol": yr_vol, "n_events": n_ev,
        })

    return {
        "equity": equity,
        "benchmark": bm_eq,
        "shv_eq": shv_eq,
        "drawdown": dd,
        "net_ret": pd.Series(net, index=common),
        "weight": pd.Series(w_del, index=common),
        "events": pd.DataFrame(events),
        "metrics": metrics,
        "yearly": pd.DataFrame(yearly_rows),
    }


# ======================================================================
# GRAFICI
# ======================================================================

def plot_main_chart(result, z_final, z_rr, z_yc, regime_hmm, params, out_dir):
    """Grafico principale: Equity + Z-score RR + Z-score YC + Peso."""
    equity = result["equity"]
    benchmark = result["benchmark"]
    ev_df = result["events"]
    m = result["metrics"]
    dates = equity.index

    z_f_vals = z_final.reindex(dates).values
    z_rr_vals = z_rr.reindex(dates).values
    z_yc_vals = z_yc.reindex(dates).values
    reg_series = regime_hmm.reindex(dates)
    w_vals = result["weight"].values

    fig, axes = plt.subplots(
        4, 1, figsize=(22, 16),
        gridspec_kw={"height_ratios": [3, 1.2, 1.2, 0.8], "hspace": 0.06},
        sharex=True,
    )
    ax1, ax2, ax3, ax4 = axes

    # Regime shading su tutti i pannelli
    stress_mask = (reg_series.values == "STRESS")
    i = 0
    while i < len(dates):
        if stress_mask[i]:
            j = i
            while j < len(dates) and stress_mask[j]:
                j += 1
            for ax in axes:
                ax.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                           color="#FFE0E0", alpha=0.30, zorder=0)
            i = j
        else:
            i += 1

    # ── TOP: Equity curve ──
    ax1.plot(dates, equity.values, color="#1a5276", linewidth=1.8,
             zorder=4, label="Strategy")
    ax1.plot(dates, benchmark.values, color="#aab7b8", linewidth=1.2,
             linestyle="--", alpha=0.7, zorder=3, label="B&H TLT")
    ax1.fill_between(dates, equity.values, benchmark.values,
                     where=(equity.values >= benchmark.values),
                     color="#27ae60", alpha=0.06, zorder=1)
    ax1.fill_between(dates, equity.values, benchmark.values,
                     where=(equity.values < benchmark.values),
                     color="#e74c3c", alpha=0.06, zorder=1)

    for _, ev in ev_df.iterrows():
        dt = ev["date"]
        if dt in equity.index:
            eq_val = equity.loc[dt]
            marker = "^" if ev["signal"] == "LONG" else "v"
            color = "#27ae60" if ev["signal"] == "LONG" else "#e74c3c"
            ax1.scatter(dt, eq_val, marker=marker, s=80, c=color,
                        edgecolors="black", linewidths=0.5, zorder=6)

    ax1.set_ylabel("Growth of $1", fontsize=12, fontweight="bold")
    ax1.set_title(
        "VERSIONE FINALE V2 -- RR + YC Combined Duration Strategy\n"
        "z_final = z_rr(ALL|std|w63 + 0.50*Slope_10Y2Y) + 0.40 * z_yc(regime_z 10s30s)",
        fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.set_ylim(bottom=0.55)

    legend_handles = [
        Line2D([0], [0], color="#1a5276", linewidth=2,
               label=f"Strategy (Sh={m['sharpe']:.3f})"),
        Line2D([0], [0], color="#aab7b8", linewidth=1.2, linestyle="--",
               label="B&H TLT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="LONG"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5,
               label="SHORT"),
        plt.Rectangle((0, 0), 1, 1, fc="#FFE0E0", alpha=0.5,
                       label="STRESS regime"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=9,
               framealpha=0.9)

    ann_text = (
        f"Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}  |  "
        f"Ann.Ret: {m['ann_ret']*100:.2f}%\n"
        f"Vol: {m['vol']*100:.1f}%  |  MDD: {m['mdd']*100:.1f}%  |  "
        f"Calmar: {m['calmar']:.3f}  |  Events: {m['n_events']}  |  "
        f"Excess: {m['excess']*100:+.1f}%"
    )
    ax1.text(0.98, 0.03, ann_text, transform=ax1.transAxes,
             fontsize=8, fontfamily="monospace", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.85, edgecolor="#cccccc"))

    # ── PANEL 2: Z-score RR ──
    tl_c = params["tl_calm"]
    ts_c = params["ts_calm"]
    tl_s = params["tl_stress"]

    v_rr = ~np.isnan(z_rr_vals)
    ax2.plot(dates[v_rr], z_rr_vals[v_rr], color="#2c3e50",
             linewidth=0.6, alpha=0.8, zorder=4)
    ax2.fill_between(dates, np.where(np.isnan(z_rr_vals), 0, z_rr_vals), 0,
                     where=(z_rr_vals > 0), color="#27ae60", alpha=0.06)
    ax2.fill_between(dates, np.where(np.isnan(z_rr_vals), 0, z_rr_vals), 0,
                     where=(z_rr_vals < 0), color="#e74c3c", alpha=0.06)
    ax2.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax2.set_ylabel("z RR Composite", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.15)
    ax2.set_ylim(-8, 8)

    # ── PANEL 3: Z-score YC ──
    v_yc = ~np.isnan(z_yc_vals)
    ax3.plot(dates[v_yc], z_yc_vals[v_yc], color="#8e44ad",
             linewidth=0.6, alpha=0.8, zorder=4)
    ax3.fill_between(dates, np.where(np.isnan(z_yc_vals), 0, z_yc_vals), 0,
                     where=(z_yc_vals > 0), color="#27ae60", alpha=0.06)
    ax3.fill_between(dates, np.where(np.isnan(z_yc_vals), 0, z_yc_vals), 0,
                     where=(z_yc_vals < 0), color="#e74c3c", alpha=0.06)
    ax3.axhline(0, color="black", linewidth=0.4, alpha=0.3)
    ax3.set_ylabel("z YC Regime", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.15)
    ax3.set_ylim(-4, 4)

    # ── PANEL 4: Combined z + thresholds + trade markers ──
    v_f = ~np.isnan(z_f_vals)
    ax4.plot(dates[v_f], z_f_vals[v_f], color="#2c3e50",
             linewidth=0.7, alpha=0.85, zorder=4)
    ax4.fill_between(dates, np.where(np.isnan(z_f_vals), 0, z_f_vals), 0,
                     where=(z_f_vals > 0), color="#27ae60", alpha=0.06)
    ax4.fill_between(dates, np.where(np.isnan(z_f_vals), 0, z_f_vals), 0,
                     where=(z_f_vals < 0), color="#e74c3c", alpha=0.06)

    ax4.axhline(tl_c, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7)
    ax4.axhline(ts_c, color="#27ae60", linewidth=1.2, linestyle="--", alpha=0.7)
    ax4.axhline(tl_s, color="#c0392b", linewidth=1.0, linestyle=":", alpha=0.6)
    ax4.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    for _, ev in ev_df.iterrows():
        dt = ev["date"]
        zv = ev["z"]
        marker = "^" if ev["signal"] == "LONG" else "v"
        color = "#27ae60" if ev["signal"] == "LONG" else "#e74c3c"
        ax4.scatter(dt, zv, marker=marker, s=90, c=color,
                    edgecolors="black", linewidths=0.6, zorder=6)

    ax4.set_ylabel("z Combined", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Date", fontsize=11)
    ax4.grid(True, alpha=0.15)
    ax4.set_ylim(-9, 9)
    ax4.xaxis.set_major_locator(mdates.YearLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    z_legend = [
        Line2D([0], [0], color="#e74c3c", linewidth=1.2, linestyle="--",
               label=f"SHORT calm ({tl_c})"),
        Line2D([0], [0], color="#27ae60", linewidth=1.2, linestyle="--",
               label=f"LONG calm ({ts_c})"),
        Line2D([0], [0], color="#c0392b", linewidth=1.0, linestyle=":",
               label=f"SHORT stress ({tl_s})"),
    ]
    ax4.legend(handles=z_legend, loc="lower left", fontsize=8,
               framealpha=0.9, ncol=3)

    fig.savefig(os.path.join(out_dir, "01_equity_zscore.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    return "01_equity_zscore.png"


def plot_yearly_comparison(result, params, out_dir):
    """Grafico confronto annuale Strategy vs B&H TLT."""
    yearly = result["yearly"]
    if len(yearly) == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                    gridspec_kw={"height_ratios": [2, 1],
                                                 "hspace": 0.15})

    x = np.arange(len(yearly))
    w = 0.35

    # Bar chart dei rendimenti
    bars_s = ax1.bar(x - w/2, yearly["strat_ret"]*100, w,
                     label="Strategy V2", color="#2874a6", edgecolor="white",
                     linewidth=0.5)
    bars_b = ax1.bar(x + w/2, yearly["bm_ret"]*100, w,
                     label="B&H TLT", color="#aab7b8", edgecolor="white",
                     linewidth=0.5)

    # Evidenzia anni con excess positivo
    for i, (_, yr) in enumerate(yearly.iterrows()):
        if yr["excess"] > 0:
            bars_s[i].set_color("#1a6b3c")

    ax1.set_xticks(x)
    ax1.set_xticklabels(yearly["year"].astype(int), fontsize=10)
    ax1.set_ylabel("Return (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Confronto Annuale: Strategy V2 vs B&H TLT",
                  fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.2, axis="y")
    ax1.axhline(0, color="black", linewidth=0.8)

    # Excess return
    colors = ["#27ae60" if e > 0 else "#e74c3c"
              for e in yearly["excess"].values]
    ax2.bar(x, yearly["excess"]*100, 0.6, color=colors,
            edgecolor="white", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(yearly["year"].astype(int), fontsize=10)
    ax2.set_ylabel("Excess Return (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Excess Return vs B&H TLT (per anno)", fontsize=11)
    ax2.grid(True, alpha=0.2, axis="y")
    ax2.axhline(0, color="black", linewidth=0.8)

    # Annotazioni
    pos = (yearly["excess"] > 0).sum()
    tot = len(yearly)
    avg_exc = yearly["excess"].mean() * 100
    ax2.text(0.98, 0.95,
             f"Anni positivi: {pos}/{tot} ({pos/tot*100:.0f}%) | "
             f"Avg excess: {avg_exc:+.1f}%",
             transform=ax2.transAxes, fontsize=9, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                       edgecolor="#cccccc"))

    fig.savefig(os.path.join(out_dir, "02_yearly_comparison.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return "02_yearly_comparison.png"


def plot_drawdown(result, out_dir):
    """Grafico drawdown con evidenziazione crisi."""
    equity = result["equity"]
    benchmark = result["benchmark"]
    dates = equity.index

    dd_s = result["drawdown"]
    peak_bm = benchmark.cummax()
    dd_bm = benchmark / peak_bm - 1

    fig, ax = plt.subplots(figsize=(18, 6))

    ax.fill_between(dates, dd_s.values * 100, 0,
                    color="#e74c3c", alpha=0.3, label="Strategy V2")
    ax.plot(dates, dd_s.values * 100, color="#c0392b", linewidth=0.8)
    ax.plot(dates, dd_bm.values * 100, color="#7f8c8d", linewidth=0.8,
            linestyle="--", alpha=0.7, label="B&H TLT")

    ax.set_ylabel("Drawdown (%)", fontsize=12, fontweight="bold")
    ax.set_title("Drawdown: Strategy V2 vs B&H TLT", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(top=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Annota il MDD
    mdd_idx = dd_s.idxmin()
    mdd_val = dd_s.min() * 100
    ax.annotate(f"MDD: {mdd_val:.1f}%\n{mdd_idx:%Y-%m-%d}",
                xy=(mdd_idx, mdd_val), xytext=(mdd_idx, mdd_val - 5),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b"),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#c0392b"))

    fig.savefig(os.path.join(out_dir, "03_drawdown.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return "03_drawdown.png"


def plot_regime_timeline(regime_hmm, regime_yc, result, out_dir):
    """Grafico timeline regime HMM + YC regime classification."""
    equity = result["equity"]
    dates = equity.index

    hmm_vals = regime_hmm.reindex(dates)
    yc_vals = regime_yc.reindex(dates)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(18, 8),
        gridspec_kw={"height_ratios": [2, 0.6, 0.6], "hspace": 0.12},
        sharex=True)

    # Equity con regime shading
    ax1.plot(dates, equity.values, color="#1a5276", linewidth=1.5)
    stress_mask = (hmm_vals.values == "STRESS")
    i = 0
    while i < len(dates):
        if stress_mask[i]:
            j = i
            while j < len(dates) and stress_mask[j]:
                j += 1
            ax1.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                        color="#FFE0E0", alpha=0.35, zorder=0)
            i = j
        else:
            i += 1
    ax1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    ax1.set_title("Regime Timeline: HMM (CALM/STRESS) + YC (Bull/Bear Steep/Flat)",
                  fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))

    # HMM regime bar
    hmm_numeric = np.where(hmm_vals.values == "STRESS", 1, 0)
    ax2.fill_between(dates, hmm_numeric, 0, color="#e74c3c", alpha=0.5,
                     step="mid")
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["CALM", "STRESS"], fontsize=9)
    ax2.set_ylabel("HMM", fontsize=10, fontweight="bold")
    ax2.grid(True, alpha=0.15)

    # YC regime color bar
    yc_color_map = {
        "BULL_FLAT": "#27ae60", "BULL_STEEP": "#82e0aa",
        "BEAR_FLAT": "#e74c3c", "BEAR_STEEP": "#f1948a",
        "UNDEFINED": "#d5dbdb",
    }
    yc_score_map = {
        "BULL_FLAT": 2, "BULL_STEEP": 1,
        "UNDEFINED": 0,
        "BEAR_FLAT": -1, "BEAR_STEEP": -2,
    }
    yc_numeric = np.array([yc_score_map.get(v, 0) for v in yc_vals.values])
    ax3.fill_between(dates, yc_numeric, 0,
                     where=(yc_numeric > 0), color="#27ae60", alpha=0.4,
                     step="mid")
    ax3.fill_between(dates, yc_numeric, 0,
                     where=(yc_numeric < 0), color="#e74c3c", alpha=0.4,
                     step="mid")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_ylim(-2.5, 2.5)
    ax3.set_yticks([-2, -1, 0, 1, 2])
    ax3.set_yticklabels(["BEAR_ST", "BEAR_FL", "", "BULL_ST", "BULL_FL"],
                        fontsize=8)
    ax3.set_ylabel("YC 10s30s", fontsize=10, fontweight="bold")
    ax3.grid(True, alpha=0.15)
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(os.path.join(out_dir, "04_regime_timeline.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return "04_regime_timeline.png"


def plot_performance_dashboard(result, params, out_dir):
    """Dashboard sintetico delle performance."""
    m = result["metrics"]
    yearly = result["yearly"]
    ev_df = result["events"]

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PERFORMANCE DASHBOARD -- RR+YC Combined Strategy V2",
                 fontsize=15, fontweight="bold", y=0.98)

    # Layout: 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.92, bottom=0.06)

    # ── 1. Key Metrics Table ──
    ax_tab = fig.add_subplot(gs[0, 0])
    ax_tab.axis("off")
    metrics_data = [
        ["Sharpe Ratio", f"{m['sharpe']:.4f}"],
        ["Sortino Ratio", f"{m['sortino']:.4f}"],
        ["Ann. Return", f"{m['ann_ret']*100:.2f}%"],
        ["Volatility", f"{m['vol']*100:.2f}%"],
        ["Max Drawdown", f"{m['mdd']*100:.2f}%"],
        ["Calmar Ratio", f"{m['calmar']:.4f}"],
        ["Excess vs TLT", f"{m['excess']*100:+.2f}%"],
        ["Total Return", f"{m['total_ret']*100:.1f}%"],
        ["Equity Finale", f"${m['eq_final']:.3f}"],
        ["N. Operazioni", f"{m['n_events']}"],
        ["Periodo", f"{m['years']:.1f} anni"],
    ]
    table = ax_tab.table(cellText=metrics_data, colLabels=["Metrica", "Valore"],
                         loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2874a6")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eaf2f8")
    ax_tab.set_title("Metriche Chiave", fontsize=11, fontweight="bold", pad=10)

    # ── 2. Signal Parameters ──
    ax_params = fig.add_subplot(gs[0, 1])
    ax_params.axis("off")
    params_data = [
        ["RR Tenors", "ALL (1W,1M,3M,6M)"],
        ["RR z-window", str(params["z_window"])],
        ["Slope", f"{params['slope_long_yield']}-{params['slope_short_yield']}"],
        ["k_slope", str(params["k_slope"])],
        ["YC Spread", f"{params['yc_spread_short']}-{params['yc_spread_long']}"],
        ["YC cw / zw", f"{params['yc_change_window']} / {params['yc_z_window']}"],
        ["k_yc", str(params["k_yc"])],
        ["tl_calm", str(params["tl_calm"])],
        ["ts_calm", str(params["ts_calm"])],
        ["tl_stress", str(params["tl_stress"])],
        ["Cooldown", str(params["cooldown"])],
        ["Delay", f"T+{params['delay']+1}"],
        ["TC", f"{params['tcost_bps']} bps"],
    ]
    table2 = ax_params.table(cellText=params_data,
                              colLabels=["Parametro", "Valore"],
                              loc="center", cellLoc="center")
    table2.auto_set_font_size(False)
    table2.set_fontsize(9)
    table2.scale(1.0, 1.3)
    for (row, col), cell in table2.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#e8f8f5")
    ax_params.set_title("Parametri Segnale", fontsize=11,
                        fontweight="bold", pad=10)

    # ── 3. Trade Distribution ──
    ax_trades = fig.add_subplot(gs[0, 2])
    if len(ev_df) > 0:
        n_long = (ev_df["signal"] == "LONG").sum()
        n_short = (ev_df["signal"] == "SHORT").sum()
        n_calm = (ev_df["regime"] == "CALM").sum()
        n_stress = (ev_df["regime"] == "STRESS").sum()

        labels = ["LONG", "SHORT"]
        sizes = [n_long, n_short]
        colors_pie = ["#27ae60", "#e74c3c"]
        ax_trades.pie(sizes, labels=labels, colors=colors_pie,
                      autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
        ax_trades.set_title(
            f"Distribuzione Operazioni ({len(ev_df)} totali)\n"
            f"CALM: {n_calm} | STRESS: {n_stress}",
            fontsize=11, fontweight="bold", pad=10)

    # ── 4. Rolling Sharpe ──
    ax_roll = fig.add_subplot(gs[1, 0:2])
    net_ret = result["net_ret"]
    exc = net_ret - RISK_FREE_RATE / 252
    rolling_sh = (exc.rolling(252, min_periods=126).mean()
                  / exc.rolling(252, min_periods=126).std()
                  * np.sqrt(252))
    ax_roll.plot(rolling_sh.index, rolling_sh.values, color="#2874a6",
                 linewidth=1.2)
    ax_roll.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_roll.axhline(m["sharpe"], color="#27ae60", linewidth=1, linestyle="--",
                    alpha=0.7, label=f"Full-period: {m['sharpe']:.3f}")
    ax_roll.fill_between(rolling_sh.index, rolling_sh.values, 0,
                         where=(rolling_sh.values > 0),
                         color="#27ae60", alpha=0.06)
    ax_roll.fill_between(rolling_sh.index, rolling_sh.values, 0,
                         where=(rolling_sh.values < 0),
                         color="#e74c3c", alpha=0.06)
    ax_roll.set_ylabel("Rolling 1Y Sharpe", fontsize=10, fontweight="bold")
    ax_roll.set_title("Rolling Sharpe Ratio (252gg)", fontsize=11,
                      fontweight="bold")
    ax_roll.legend(fontsize=9)
    ax_roll.grid(True, alpha=0.2)

    # ── 5. V1 vs V2 comparison ──
    ax_comp = fig.add_subplot(gs[1, 2])
    ax_comp.axis("off")
    comp_data = [
        ["", "V1 (RR-only)", "V2 (RR+YC)"],
        ["Sharpe", "0.325", f"{m['sharpe']:.3f}"],
        ["Sortino", "0.399", f"{m['sortino']:.3f}"],
        ["Ann. Ret", "5.39%", f"{m['ann_ret']*100:.2f}%"],
        ["MDD", "-23.6%", f"{m['mdd']*100:.1f}%"],
        ["Events", "52", f"{m['n_events']}"],
        ["ts_calm", "+2.75", f"+{params['ts_calm']:.2f}"],
        ["tl_stress", "-4.00", f"{params['tl_stress']:.2f}"],
    ]
    table3 = ax_comp.table(cellText=comp_data[1:],
                            colLabels=comp_data[0],
                            loc="center", cellLoc="center")
    table3.auto_set_font_size(False)
    table3.set_fontsize(9)
    table3.scale(1.0, 1.5)
    for (row, col), cell in table3.get_celld().items():
        if row == 0:
            cell.set_facecolor("#6c3483")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 2 and row > 0:
            cell.set_facecolor("#f4ecf7")
    ax_comp.set_title("V1 vs V2", fontsize=11, fontweight="bold", pad=10)

    fig.savefig(os.path.join(out_dir, "05_dashboard.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return "05_dashboard.png"


def plot_weight_allocation(result, out_dir):
    """Grafico dell'allocazione nel tempo."""
    equity = result["equity"]
    dates = equity.index
    w_vals = result["weight"].values

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(18, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        sharex=True)

    ax1.plot(dates, equity.values, color="#1a5276", linewidth=1.5)
    ax1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    ax1.set_title("Allocazione TLT nel Tempo", fontsize=13,
                  fontweight="bold")
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))

    ax2.fill_between(dates, w_vals, 0.5,
                     where=(w_vals > 0.7), color="#27ae60", alpha=0.4,
                     step="mid", label="LONG (100% TLT)")
    ax2.fill_between(dates, w_vals, 0.5,
                     where=(w_vals < 0.3), color="#e74c3c", alpha=0.4,
                     step="mid", label="SHORT (100% SHV)")
    ax2.fill_between(dates, w_vals, 0.5,
                     where=((w_vals >= 0.3) & (w_vals <= 0.7)),
                     color="#f39c12", alpha=0.2, step="mid",
                     label="NEUTRAL (50/50)")
    ax2.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("TLT Weight", fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.savefig(os.path.join(out_dir, "06_allocation.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return "06_allocation.png"


# ======================================================================
# DOCUMENTAZIONE
# ======================================================================

def save_documentation(result, params, out_dir):
    """Salva la documentazione della strategia."""
    m = result["metrics"]
    yearly = result["yearly"]
    ev_df = result["events"]

    lines = []
    lines.append("=" * 72)
    lines.append("  STRATEGY DOCUMENTATION -- RR+YC Combined V2")
    lines.append("=" * 72)
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 72)
    lines.append("Strategy: RR + Yield Curve Combined Duration Timing")
    lines.append("Instruments: TLT (long duration) / SHV (short duration)")
    lines.append("Universe: US Treasury ETFs")
    lines.append(f"Period: {m['years']:.1f} years ({m['n_days']} trading days)")
    lines.append("")

    lines.append("SIGNAL CONSTRUCTION")
    lines.append("-" * 72)
    lines.append("z_final = z_rr_composite + 0.40 * z_yc_regime_z")
    lines.append("")
    lines.append("Component 1: RR Composite (z_rr)")
    lines.append("  z_rr = z_base_rr + 0.50 * z_slope_momentum")
    lines.append("  z_base_rr: rolling z-score of mean Risk Reversal (1W,1M,3M,6M), w=63")
    lines.append("  z_slope_momentum: z-score of 126-day change in 10Y-2Y slope, zw=252")
    lines.append("")
    lines.append("Component 2: YC Regime Z-Score (z_yc)")
    lines.append("  1. Classify 10s30s yield curve regime every day:")
    lines.append("     - BULL_FLAT  (+2): level falls, spread falls")
    lines.append("     - BULL_STEEP (+1): level falls, spread rises")
    lines.append("     - BEAR_FLAT  (-1): level rises, spread falls")
    lines.append("     - BEAR_STEEP (-2): level rises, spread rises")
    lines.append("  2. Changes measured over rolling 42-day window")
    lines.append("  3. Z-score the regime score with 252-day rolling window")
    lines.append("")
    lines.append("Combination: Additive method (z_final = z_rr + 0.40 * z_yc)")
    lines.append("  k_yc = 0.40 provides optimal diversification without diluting RR signal")
    lines.append("")

    lines.append("TRADING RULES")
    lines.append("-" * 72)
    lines.append("HMM Regime Detection: 2-state Gaussian HMM on log(MOVE) + log(VIX)")
    lines.append("")
    lines.append("  CALM regime:")
    lines.append(f"    z_final <= {params['tl_calm']:.2f}  ->  SHORT (100% SHV)")
    lines.append(f"    z_final >= {params['ts_calm']:+.2f}  ->  LONG  (100% TLT)")
    lines.append("  STRESS regime:")
    lines.append(f"    z_final <= {params['tl_stress']:.2f}  ->  SHORT (100% SHV)")
    lines.append("    No LONG signals in STRESS")
    lines.append("")
    lines.append(f"  Cooldown: {params['cooldown']} days between signals")
    lines.append(f"  Execution delay: T+{params['delay']+1}")
    lines.append(f"  Transaction cost: {params['tcost_bps']} bps per rebalance")
    lines.append(f"  Initial allocation: {params['w_initial']*100:.0f}% TLT / "
                 f"{(1-params['w_initial'])*100:.0f}% SHV")
    lines.append("")

    lines.append("PERFORMANCE RESULTS")
    lines.append("-" * 72)
    lines.append(f"  Sharpe Ratio:     {m['sharpe']:.4f}")
    lines.append(f"  Sortino Ratio:    {m['sortino']:.4f}")
    lines.append(f"  Ann. Return:      {m['ann_ret']*100:.2f}%")
    lines.append(f"  Volatility:       {m['vol']*100:.2f}%")
    lines.append(f"  Max Drawdown:     {m['mdd']*100:.2f}%")
    lines.append(f"  Calmar Ratio:     {m['calmar']:.4f}")
    lines.append(f"  Excess vs TLT:    {m['excess']*100:+.2f}%")
    lines.append(f"  Total Return:     {m['total_ret']*100:.1f}%")
    lines.append(f"  Equity Finale:    ${m['eq_final']:.3f}")
    lines.append(f"  N. Operazioni:    {m['n_events']}")
    lines.append("")

    lines.append("COMPARISON WITH V1 (RR-ONLY)")
    lines.append("-" * 72)
    lines.append("  Metric           V1 (RR)    V2 (RR+YC)    Change")
    lines.append("  " + "-" * 56)
    lines.append(f"  Sharpe           0.325      {m['sharpe']:.3f}         "
                 f"{(m['sharpe']/0.325-1)*100:+.0f}%")
    lines.append(f"  Sortino          0.399      {m['sortino']:.3f}         "
                 f"{(m['sortino']/0.399-1)*100:+.0f}%")
    lines.append(f"  Ann. Return      5.39%      {m['ann_ret']*100:.2f}%")
    lines.append(f"  Max Drawdown     -23.6%     {m['mdd']*100:.1f}%")
    lines.append(f"  Events           52         {m['n_events']}")
    lines.append("")

    lines.append("YEARLY BREAKDOWN")
    lines.append("-" * 72)
    lines.append(f"  {'Anno':>6}  {'Strategia':>10}  {'B&H TLT':>10}  "
                 f"{'Excess':>10}  {'Vol':>8}  {'Ev':>4}")
    lines.append(f"  {'-'*56}")
    for _, yr in yearly.iterrows():
        marker = " *" if yr["excess"] > 0 else ""
        lines.append(
            f"  {int(yr['year']):>6}  {yr['strat_ret']:>+10.4f}  "
            f"{yr['bm_ret']:>+10.4f}  {yr['excess']:>+10.4f}  "
            f"{yr['vol']:>8.4f}  {int(yr['n_events']):>4}{marker}")
    pos = (yearly["excess"] > 0).sum()
    lines.append(f"\n  Anni positivi: {pos}/{len(yearly)} "
                 f"({pos/len(yearly)*100:.0f}%)")
    lines.append(f"  Avg excess: {yearly['excess'].mean()*100:+.2f}%")
    lines.append("")

    if len(ev_df) > 0:
        lines.append("TRADE LIST")
        lines.append("-" * 72)
        lines.append(f"  {'#':>3}  {'Data':<12}  {'Segnale':<7}  "
                     f"{'z':>8}  {'Regime':<7}")
        lines.append(f"  {'-'*46}")
        for i, (_, ev) in enumerate(ev_df.iterrows()):
            lines.append(
                f"  {i+1:>3}  {ev['date'].strftime('%Y-%m-%d'):<12}  "
                f"{ev['signal']:<7}  {ev['z']:>+8.3f}  {ev['regime']:<7}")
    lines.append("")

    lines.append("KEY FINDINGS")
    lines.append("-" * 72)
    lines.append("1. Adding YC regime_z signal improves Sharpe by ~22% (0.325 -> 0.396)")
    lines.append("2. 10s30s spread is the best curve segment for TLT timing")
    lines.append("3. Regime z-score (discrete->continuous) outperforms pure continuous")
    lines.append("4. Additive combination with k=0.40 is optimal; filter/agreement fail")
    lines.append("5. Fewer trades (19 vs 52) suggests more confident, higher-quality signals")
    lines.append("6. Sticky positions (trend-following) work better than exit-to-neutral")
    lines.append("7. HMM regime thresholds remain crucial for crisis protection")
    lines.append("")
    lines.append("=" * 72)

    path = os.path.join(out_dir, "strategy_documentation.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "strategy_documentation.txt"


# ======================================================================
# MAIN
# ======================================================================

def main():
    T0 = time.time()

    print("=" * 72)
    print("  VERSIONE FINALE V2 -- RR + YC Combined Duration Strategy")
    print("  z_final = z_rr(ALL|std|w63 + 0.50*Slope) + 0.40*z_yc(regime_z)")
    print("=" * 72)

    # ── 1. Caricamento dati ──
    print("\n[1/7] Caricamento dati ...")
    D = load_all()

    # ── 2. Regime HMM ──
    print("\n[2/7] Regime HMM ...")
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])

    # ── 3. Costruzione segnale combinato ──
    print("\n[3/7] Costruzione segnale combinato ...")
    z_final, z_rr, z_yc, z_base, z_slope, regime_yc, yc_score = \
        build_combined_signal(D, PARAMS)

    n_valid = z_final.dropna().shape[0]
    print(f"  z_base (RR ALL|std|w63):        {z_base.dropna().shape[0]} valori")
    print(f"  z_slope (10Y-2Y momentum):      {z_slope.dropna().shape[0]} valori")
    print(f"  z_rr (composite):               {z_rr.dropna().shape[0]} valori")
    print(f"  z_yc (regime_z 10s30s):         {z_yc.dropna().shape[0]} valori")
    print(f"  z_final (RR + 0.40*YC):         {n_valid} valori")

    # ── 4. Backtest ──
    print("\n[4/7] Backtest ...")
    start = pd.Timestamp(BACKTEST_START)
    result = run_backtest(z_final, regime_hmm, D["etf_ret"], PARAMS, start)

    m = result["metrics"]
    print(f"\n  {'='*62}")
    print(f"  RISULTATI V2")
    print(f"  {'='*62}")
    print(f"  Sharpe Ratio:     {m['sharpe']:.4f}")
    print(f"  Sortino Ratio:    {m['sortino']:.4f}")
    print(f"  Ann. Return:      {m['ann_ret']:.4f}  ({m['ann_ret']*100:.2f}%)")
    print(f"  Volatility:       {m['vol']:.4f}  ({m['vol']*100:.2f}%)")
    print(f"  Max Drawdown:     {m['mdd']:.4f}  ({m['mdd']*100:.2f}%)")
    print(f"  Calmar Ratio:     {m['calmar']:.4f}")
    print(f"  Excess vs TLT:    {m['excess']:+.4f}  ({m['excess']*100:+.2f}%)")
    print(f"  Total Return:     {m['total_ret']:.4f}  ({m['total_ret']*100:.1f}%)")
    print(f"  Equity Finale:    ${m['eq_final']:.3f}")
    print(f"  Benchmark Ann:    {m['bm_ann']:.4f}  ({m['bm_ann']*100:.2f}%)")
    print(f"  Eventi:           {m['n_events']}")
    print(f"  Giorni:           {m['n_days']}")
    print(f"  Anni:             {m['years']:.1f}")

    # Yearly
    print(f"\n  Breakdown annuale:")
    print(f"  {'Anno':>6}  {'Strategia':>10}  {'B&H TLT':>10}  "
          f"{'Excess':>10}  {'Vol':>8}  {'Ev':>4}")
    print(f"  {'-'*56}")
    yearly = result["yearly"]
    for _, yr in yearly.iterrows():
        marker = " *" if yr["excess"] > 0 else ""
        print(f"  {int(yr['year']):>6}  {yr['strat_ret']:>+10.4f}  "
              f"{yr['bm_ret']:>+10.4f}  {yr['excess']:>+10.4f}  "
              f"{yr['vol']:>8.4f}  {int(yr['n_events']):>4}{marker}")
    pos = (yearly["excess"] > 0).sum()
    print(f"\n  Anni positivi: {pos}/{len(yearly)} ({pos/len(yearly)*100:.0f}%)")

    # Events
    ev_df = result["events"]
    if len(ev_df) > 0:
        print(f"\n  Lista operazioni ({len(ev_df)} eventi):")
        print(f"  {'#':>3}  {'Data':<12}  {'Segnale':<7}  {'z':>8}  {'Regime':<7}")
        print(f"  {'-'*46}")
        for i, (_, ev) in enumerate(ev_df.iterrows()):
            print(f"  {i+1:>3}  {ev['date'].strftime('%Y-%m-%d'):<12}  "
                  f"{ev['signal']:<7}  {ev['z']:>+8.3f}  {ev['regime']:<7}")

    # ── 5. Grafici ──
    print("\n[5/7] Grafici ...")
    charts = []

    c = plot_main_chart(result, z_final, z_rr, z_yc, regime_hmm, PARAMS, OUT_DIR)
    charts.append(c)
    print(f"  Salvato: {c}")

    c = plot_yearly_comparison(result, PARAMS, OUT_DIR)
    if c:
        charts.append(c)
        print(f"  Salvato: {c}")

    c = plot_drawdown(result, OUT_DIR)
    charts.append(c)
    print(f"  Salvato: {c}")

    c = plot_regime_timeline(regime_hmm, regime_yc, result, OUT_DIR)
    charts.append(c)
    print(f"  Salvato: {c}")

    c = plot_performance_dashboard(result, PARAMS, OUT_DIR)
    charts.append(c)
    print(f"  Salvato: {c}")

    c = plot_weight_allocation(result, OUT_DIR)
    charts.append(c)
    print(f"  Salvato: {c}")

    # ── 6. Salvataggio dati ──
    print("\n[6/7] Salvataggio dati ...")

    # Metriche
    pd.DataFrame([{**PARAMS, **m}]).to_csv(
        os.path.join(OUT_DIR, "metrics.csv"), index=False)
    print("  Salvato: metrics.csv")

    # Equity curve giornaliera
    dates = result["equity"].index
    eq_export = pd.DataFrame({
        "date": dates,
        "strategy": result["equity"].values,
        "benchmark_tlt": result["benchmark"].values,
        "weight_tlt": result["weight"].values,
        "net_return": result["net_ret"].values,
        "z_final": z_final.reindex(dates).values,
        "z_rr": z_rr.reindex(dates).values,
        "z_yc": z_yc.reindex(dates).values,
        "regime_hmm": regime_hmm.reindex(dates).values,
        "regime_yc": regime_yc.reindex(dates).values,
    })
    eq_export.to_csv(os.path.join(OUT_DIR, "daily_equity.csv"), index=False)
    print("  Salvato: daily_equity.csv")

    # Trade list
    ev_df.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
    print("  Salvato: trades.csv")

    # Yearly
    yearly.to_csv(os.path.join(OUT_DIR, "yearly.csv"), index=False)
    print("  Salvato: yearly.csv")

    # ── 7. Documentazione ──
    print("\n[7/7] Documentazione ...")
    doc = save_documentation(result, PARAMS, OUT_DIR)
    print(f"  Salvato: {doc}")

    elapsed = time.time() - T0
    print(f"\n{'='*72}")
    print(f"  VERSIONE FINALE V2 COMPLETATA")
    print(f"  Sharpe: {m['sharpe']:.4f}  |  Equity: ${m['eq_final']:.3f}  |  "
          f"Events: {m['n_events']}")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Files:  {len(charts)} grafici + 4 CSV + 1 documentazione")
    print(f"  Tempo:  {elapsed:.1f}s")
    print(f"{'='*72}")

    return result


if __name__ == "__main__":
    main()
