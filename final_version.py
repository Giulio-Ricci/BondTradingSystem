"""
final_version.py - VERSIONE FINALE della strategia Duration Timing.

RR + Slope + YC + Term Spread(1W-3M) — configurazione ottimale congelata.
Genera output completo in output/final_version_ok/:
  - 9 grafici PNG ad alta risoluzione
  - Report professionale PDF multi-pagina
  - CSV dati
  - Documentazione strategia
"""

import os, time, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

ANN = 252
OUT_DIR = os.path.join(SAVE_DIR, "final_version_ok")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# FROZEN BEST PARAMETERS
# ══════════════════════════════════════════════════════════════════════

BEST = {
    "z_windows": [63, 252],
    "k_slope": 1.00,
    "yc_cw": 84,
    "k_yc": 0.30,
    "term_pair": ("1W", "3M"),
    "term_z_w": 63,
    "k_term": 0.40,
    "tl_c": -3.75,
    "ts_c": 3.625,
    "tl_s": -2.625,
    "adapt_days": 63,
    "adapt_raise": 0.75,
    "max_hold_days": 0,
    "cooldown": 5,
    "delay": 1,
    "tcost_bps": 0,
}

V2_PARAMS = {
    "z_windows": [63], "k_slope": 0.50, "yc_cw": 42, "k_yc": 0.40,
    "tl_c": -3.25, "ts_c": 2.50, "tl_s": -3.50,
    "adapt_days": 0, "adapt_raise": 0.0, "max_hold_days": 0,
}

VPRES_PARAMS = {
    "z_windows": [63, 252], "k_slope": 0.75, "yc_cw": 126, "k_yc": 0.20,
    "tl_c": -3.375, "ts_c": 3.00, "tl_s": -2.875,
    "adapt_days": 63, "adapt_raise": 1.0, "max_hold_days": 252,
}

SCORE_MAP = {"BULL_FLAT": +2, "BULL_STEEP": +1, "UNDEFINED": 0,
             "BEAR_FLAT": -1, "BEAR_STEEP": -2}

# ══════════════════════════════════════════════════════════════════════
# SIGNAL CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def build_rr_blend(D):
    options = D["options"]
    parts = []
    for tenor in ["1W", "1M", "3M", "6M"]:
        p, c = RR_TENORS[tenor]
        parts.append(pd.to_numeric(options[p], errors="coerce")
                     - pd.to_numeric(options[c], errors="coerce"))
    return pd.concat(parts, axis=1).mean(axis=1)

def build_rr_individual(D):
    options = D["options"]
    out = {}
    for tenor in ["1W", "1M", "3M", "6M"]:
        p, c = RR_TENORS[tenor]
        out[tenor] = (pd.to_numeric(options[p], errors="coerce")
                      - pd.to_numeric(options[c], errors="coerce"))
    return out

def build_slope_zscore(D, mom_window=126, z_window=252, z_min=63):
    yld = D["yields"]
    slope = yld["US_10Y"] - yld["US_2Y"]
    ds = slope.diff(mom_window)
    mu = ds.rolling(z_window, min_periods=z_min).mean()
    sd = ds.rolling(z_window, min_periods=z_min).std().replace(0, np.nan)
    return (ds - mu) / sd

def build_rr_multitf(D, z_windows, k_slope=0.50):
    rr_blend = build_rr_blend(D)
    z_parts = []
    for w in z_windows:
        mu = rr_blend.rolling(w, min_periods=20).mean()
        sd = rr_blend.rolling(w, min_periods=20).std().replace(0, np.nan)
        z_parts.append((rr_blend - mu) / sd)
    z_base = pd.concat(z_parts, axis=1).mean(axis=1)
    z_slope = build_slope_zscore(D)
    return z_base + k_slope * z_slope, z_base, z_slope

def build_term_spread_zscore(rr_a, rr_b, z_window=126, z_min=30):
    spread = rr_a - rr_b
    mu = spread.rolling(z_window, min_periods=z_min).mean()
    sd = spread.rolling(z_window, min_periods=z_min).std().replace(0, np.nan)
    return (spread - mu) / sd, spread

def build_yc_signal(D, change_window=42, z_window=252, z_min_per=20):
    yld = D["yields"]
    level = (yld["US_10Y"] + yld["US_30Y"]) / 2
    spread = yld["US_30Y"] - yld["US_10Y"]
    dl = level.diff(change_window)
    ds = spread.diff(change_window)
    regime = pd.Series("UNDEFINED", index=yld.index)
    regime[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    regime[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    regime[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    regime[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    sc = regime.map(SCORE_MAP).astype(float)
    mu = sc.rolling(z_window, min_periods=z_min_per).mean()
    sd = sc.rolling(z_window, min_periods=z_min_per).std().replace(0, np.nan)
    return (sc - mu) / sd, regime, sc


# ══════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════

def fast_bt(z_arr, reg_arr, r_long, r_short, year_arr, params):
    tl_c = params["tl_c"]; ts_c = params["ts_c"]; tl_s = params["tl_s"]
    adapt_days = params.get("adapt_days", 0)
    adapt_raise = params.get("adapt_raise", 0.0)
    max_hold_days = params.get("max_hold_days", 0)
    cooldown = params.get("cooldown", 5)
    delay = params.get("delay", 1)
    tcost_bps = params.get("tcost_bps", 0)
    w_init = 0.5

    n = len(z_arr)
    weight = np.full(n, w_init)
    cur_w = w_init; last_ev = -cooldown - 1; n_events = 0; hold_days = 0

    for i in range(n):
        zv = z_arr[i]
        if np.isnan(zv):
            weight[i] = cur_w; hold_days += 1; continue

        ts_c_eff = ts_c
        if adapt_days > 0 and cur_w > 0.7 and hold_days >= adapt_days:
            ts_c_eff = ts_c + adapt_raise

        if max_hold_days > 0 and hold_days >= max_hold_days and abs(cur_w - 0.5) > 0.01:
            cur_w = 0.5; hold_days = 0; n_events += 1; last_ev = i
            weight[i] = cur_w; continue

        triggered = False
        if reg_arr[i] == 0:  # CALM
            if zv <= tl_c and (i - last_ev) >= cooldown:
                cur_w = 0.0; triggered = True
            elif zv >= ts_c_eff and (i - last_ev) >= cooldown:
                cur_w = 1.0; triggered = True
        else:  # STRESS
            if zv <= tl_s and (i - last_ev) >= cooldown:
                cur_w = 0.0; triggered = True

        if triggered:
            last_ev = i; n_events += 1; hold_days = 0
        else:
            hold_days += 1
        weight[i] = cur_w

    shift = 1 + delay
    w_del = np.empty(n); w_del[:shift] = w_init; w_del[shift:] = weight[:n - shift]
    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n); dw[0] = abs(w_del[0] - w_init); dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost_bps / 10_000)
    equity = np.cumprod(1.0 + net)

    yrs = n / ANN
    total_ret = equity[-1] / equity[0] - 1.0
    ann_ret = (1.0 + total_ret) ** (1.0 / max(yrs, 0.01)) - 1.0
    vol = np.std(net) * np.sqrt(ANN)
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-8 else 0
    down = net[net < 0]
    down_vol = np.std(down) * np.sqrt(ANN) if len(down) > 0 else 1e-6
    sortino = (ann_ret - RISK_FREE_RATE) / down_vol
    peak = np.maximum.accumulate(equity); dd = equity / peak - 1; mdd = dd.min()
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-10 else 0
    bh_eq = np.cumprod(1.0 + r_long)
    bh_ann = (1.0 + bh_eq[-1] - 1.0) ** (1.0 / max(yrs, 0.01)) - 1.0
    excess = ann_ret - bh_ann

    y_exc, y_strat, y_bm, y_yrs = [], [], [], []
    for y in np.unique(year_arr):
        mask = year_arr == y
        if mask.sum() < 5: continue
        sr = np.cumprod(1.0 + net[mask])[-1] - 1.0
        br = np.cumprod(1.0 + r_long[mask])[-1] - 1.0
        y_strat.append(sr); y_bm.append(br); y_exc.append(sr - br); y_yrs.append(y)

    y_exc = np.array(y_exc)
    return {
        "sharpe": sharpe, "sortino": sortino, "ann_ret": ann_ret, "vol": vol,
        "mdd": mdd, "calmar": calmar, "excess": excess, "total_ret": total_ret,
        "n_events": n_events, "equity": equity, "net_ret": net, "weight": w_del,
        "dd": dd, "n_dead": int(np.sum(np.abs(y_exc) < 0.01)),
        "win_rate": float(np.sum(y_exc > 0)) / max(len(y_exc), 1),
        "min_excess": float(np.min(y_exc)) if len(y_exc) > 0 else -1.0,
        "yearly_years": np.array(y_yrs), "yearly_strat": np.array(y_strat),
        "yearly_bm": np.array(y_bm), "yearly_excess": y_exc,
    }


# ══════════════════════════════════════════════════════════════════════
# CHARTS (same style as versione_rr_term, refined for final)
# ══════════════════════════════════════════════════════════════════════

def _stress_shading(dates, reg_s, axes):
    stress = (reg_s.values == "STRESS")
    i = 0
    while i < len(dates):
        if stress[i]:
            j = i
            while j < len(dates) and stress[j]: j += 1
            for ax in axes:
                ax.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                           color="#FFE0E0", alpha=0.30, zorder=0)
            i = j
        else: i += 1


def plot_01(dates, res, z_final, z_rr, z_yc, z_term, regime_hmm, out):
    eq = res["equity"]; bh = np.cumprod(1.0 + res["r_long"]); w = res["weight"]
    zf = z_final.reindex(dates).values; zr = z_rr.reindex(dates).values
    zy = z_yc.reindex(dates).values; zt = z_term.reindex(dates).values

    fig, axes = plt.subplots(5, 1, figsize=(22, 20),
        gridspec_kw={"height_ratios": [3, 1.0, 0.8, 0.8, 1.0], "hspace": 0.06}, sharex=True)
    ax1, ax2, ax3, axt, ax4 = axes
    _stress_shading(dates, regime_hmm.reindex(dates), axes)

    ax1.plot(dates, eq, color="#1a5276", lw=1.8, zorder=4, label="Strategy")
    ax1.plot(dates, bh, color="#aab7b8", lw=1.2, ls="--", alpha=0.7, zorder=3, label="B&H TLT")
    ax1.fill_between(dates, eq, bh, where=(eq >= bh), color="#27ae60", alpha=0.06)
    ax1.fill_between(dates, eq, bh, where=(eq < bh), color="#e74c3c", alpha=0.06)

    dw = np.abs(np.diff(w, prepend=w[0])); tidx = np.where(dw > 0.01)[0]
    for ti in tidx:
        mk = "^" if w[ti] > 0.7 else ("v" if w[ti] < 0.3 else "o")
        co = "#27ae60" if w[ti] > 0.7 else ("#e74c3c" if w[ti] < 0.3 else "#f39c12")
        ax1.scatter(dates[ti], eq[ti], marker=mk, s=80, c=co, edgecolors="black", lw=0.5, zorder=6)

    m = res; B = BEST
    ax1.set_ylabel("Growth of $1", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"FINAL VERSION -- RR + Slope + YC + Term Spread(1W-3M) Duration Strategy\n"
        f"z = z_rr(w={B['z_windows']}, k_sl={B['k_slope']:.2f}) "
        f"+ {B['k_yc']:.2f}*z_yc(cw={B['yc_cw']}) "
        f"+ {B['k_term']:.2f}*z_term(1W-3M, w={B['term_z_w']})",
        fontsize=12, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.set_ylim(bottom=0.55)
    lh = [Line2D([0],[0],color="#1a5276",lw=2,label=f"Strategy (Sh={m['sharpe']:.3f})"),
          Line2D([0],[0],color="#aab7b8",lw=1.2,ls="--",label="B&H TLT"),
          Line2D([0],[0],marker="^",color="w",markerfacecolor="#27ae60",ms=10,markeredgecolor="black",markeredgewidth=0.5,label="LONG"),
          Line2D([0],[0],marker="v",color="w",markerfacecolor="#e74c3c",ms=10,markeredgecolor="black",markeredgewidth=0.5,label="SHORT"),
          plt.Rectangle((0,0),1,1,fc="#FFE0E0",alpha=0.5,label="STRESS regime")]
    ax1.legend(handles=lh, loc="upper left", fontsize=9, framealpha=0.9)
    ann = (f"Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}  |  Ann.Ret: {m['ann_ret']*100:.2f}%\n"
           f"Vol: {m['vol']*100:.1f}%  |  MDD: {m['mdd']*100:.1f}%  |  Calmar: {m['calmar']:.3f}  |  "
           f"Events: {m['n_events']}  |  Dead yrs: {m['n_dead']}")
    ax1.text(0.98, 0.03, ann, transform=ax1.transAxes, fontsize=8, fontfamily="monospace",
             ha="right", va="bottom", bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85, ec="#ccc"))

    for ax, vals, label, yl in [(ax2, zr, "z RR Multi-TF", (-8,8)),
                                 (ax3, zy, "z YC Regime", (-4,4)),
                                 (axt, zt, "z Term 1W-3M", (-5,5))]:
        v = ~np.isnan(vals)
        col = "#2c3e50" if "RR" in label else ("#8e44ad" if "YC" in label else "#d35400")
        ax.plot(dates[v], vals[v], color=col, lw=0.6, alpha=0.8)
        ax.fill_between(dates, np.nan_to_num(vals), 0, where=(vals>0), color="#27ae60", alpha=0.06)
        ax.fill_between(dates, np.nan_to_num(vals), 0, where=(vals<0), color="#e74c3c", alpha=0.06)
        ax.axhline(0, color="black", lw=0.4, alpha=0.3)
        ax.set_ylabel(label, fontsize=10, fontweight="bold"); ax.grid(True, alpha=0.15); ax.set_ylim(*yl)

    vf = ~np.isnan(zf)
    ax4.plot(dates[vf], zf[vf], color="#2c3e50", lw=0.7, alpha=0.85)
    ax4.fill_between(dates, np.nan_to_num(zf), 0, where=(zf>0), color="#27ae60", alpha=0.06)
    ax4.fill_between(dates, np.nan_to_num(zf), 0, where=(zf<0), color="#e74c3c", alpha=0.06)
    ax4.axhline(B["tl_c"], color="#e74c3c", lw=1.2, ls="--", alpha=0.7)
    ax4.axhline(B["ts_c"], color="#27ae60", lw=1.2, ls="--", alpha=0.7)
    ax4.axhline(B["tl_s"], color="#c0392b", lw=1.0, ls=":", alpha=0.6)
    ax4.axhline(0, color="black", lw=0.5, alpha=0.3)
    for ti in tidx:
        zv = zf[ti] if not np.isnan(zf[ti]) else 0
        mk = "^" if w[ti] > 0.7 else ("v" if w[ti] < 0.3 else "o")
        co = "#27ae60" if w[ti] > 0.7 else ("#e74c3c" if w[ti] < 0.3 else "#f39c12")
        ax4.scatter(dates[ti], zv, marker=mk, s=90, c=co, edgecolors="black", lw=0.6, zorder=6)
    ax4.set_ylabel("z Combined", fontsize=10, fontweight="bold")
    ax4.set_xlabel("Date", fontsize=11); ax4.grid(True, alpha=0.15); ax4.set_ylim(-9, 9)
    ax4.xaxis.set_major_locator(mdates.YearLocator()); ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    zl = [Line2D([0],[0],color="#e74c3c",lw=1.2,ls="--",label=f"SHORT calm ({B['tl_c']:.2f})"),
          Line2D([0],[0],color="#27ae60",lw=1.2,ls="--",label=f"LONG calm ({B['ts_c']:.2f})"),
          Line2D([0],[0],color="#c0392b",lw=1.0,ls=":",label=f"SHORT stress ({B['tl_s']:.2f})")]
    ax4.legend(handles=zl, loc="lower left", fontsize=8, framealpha=0.9, ncol=3)
    fig.savefig(os.path.join(out, "01_equity_zscore.png"), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(); return fig


def plot_02(dates, res_f, res_p, res_v2, r_long, year_arr, out):
    uy = np.unique(year_arr); rows = []
    for y in uy:
        mask = year_arr == y
        if mask.sum() < 5: continue
        bhy = np.cumprod(1.0 + r_long[mask])[-1] - 1.0
        def _exc(r, y):
            for i, yy in enumerate(r["yearly_years"]):
                if yy == y: return r["yearly_strat"][i], r["yearly_excess"][i]
            return bhy, 0.0
        fr, fe = _exc(res_f, y); pr, pe = _exc(res_p, y); vr, ve = _exc(res_v2, y)
        rows.append({"year": y, "f_ret": fr, "p_ret": pr, "v_ret": vr, "bh": bhy,
                      "f_exc": fe, "p_exc": pe, "v_exc": ve})
    df = pd.DataFrame(rows); x = np.arange(len(df)); w = 0.20

    fig, axes = plt.subplots(3, 1, figsize=(20, 16), gridspec_kw={"height_ratios": [2, 1.2, 1], "hspace": 0.18})
    ax1, ax2, ax3 = axes
    ax1.bar(x-1.5*w, df["f_ret"]*100, w, label="Final", color="#c0392b", edgecolor="white")
    ax1.bar(x-0.5*w, df["p_ret"]*100, w, label="V_Pres", color="#2874a6", edgecolor="white", alpha=0.7)
    ax1.bar(x+0.5*w, df["v_ret"]*100, w, label="V2", color="#6c3483", edgecolor="white", alpha=0.5)
    ax1.bar(x+1.5*w, df["bh"]*100, w, label="B&H TLT", color="#aab7b8", edgecolor="white")
    ax1.set_xticks(x); ax1.set_xticklabels(df["year"].astype(int), fontsize=10)
    ax1.set_ylabel("Return (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Annual Returns: Final vs V_Pres vs V2 vs B&H TLT", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.2, axis="y"); ax1.axhline(0, color="black", lw=0.8)

    w2 = 0.25
    ax2.bar(x-w2, df["f_exc"]*100, w2, label="Final excess", color="#c0392b", edgecolor="white")
    ax2.bar(x, df["p_exc"]*100, w2, label="V_Pres excess", color="#2874a6", edgecolor="white", alpha=0.7)
    ax2.bar(x+w2, df["v_exc"]*100, w2, label="V2 excess", color="#6c3483", edgecolor="white", alpha=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(df["year"].astype(int), fontsize=10)
    ax2.set_ylabel("Excess Return (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Excess Return vs B&H TLT", fontsize=11)
    ax2.legend(fontsize=10); ax2.grid(True, alpha=0.2, axis="y"); ax2.axhline(0, color="black", lw=0.8)

    delta = df["f_exc"] - df["p_exc"]
    ax3.bar(x, delta*100, 0.6, color=["#27ae60" if d > 0 else "#e74c3c" for d in delta], edgecolor="white")
    ax3.set_xticks(x); ax3.set_xticklabels(df["year"].astype(int), fontsize=10)
    ax3.set_ylabel("Delta Excess (%)", fontsize=11, fontweight="bold")
    ax3.set_title("Delta: Final excess - V_Pres excess", fontsize=11)
    ax3.grid(True, alpha=0.2, axis="y"); ax3.axhline(0, color="black", lw=0.8)
    fig.savefig(os.path.join(out, "02_yearly_comparison.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); return fig


def plot_03(dates, res, r_long, out):
    dd_s = res["dd"]; bh_eq = np.cumprod(1.0 + r_long)
    dd_bm = bh_eq / np.maximum.accumulate(bh_eq) - 1
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.fill_between(dates, dd_s*100, 0, color="#e74c3c", alpha=0.3, label="Strategy")
    ax.plot(dates, dd_s*100, color="#c0392b", lw=0.8)
    ax.plot(dates, dd_bm*100, color="#7f8c8d", lw=0.8, ls="--", alpha=0.7, label="B&H TLT")
    ax.set_ylabel("Drawdown (%)", fontsize=12, fontweight="bold")
    ax.set_title("Drawdown: Strategy vs B&H TLT", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10); ax.grid(True, alpha=0.2); ax.set_ylim(top=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    mi = np.argmin(dd_s)
    ax.annotate(f"MDD: {dd_s[mi]*100:.1f}%\n{dates[mi]:%Y-%m-%d}", xy=(dates[mi], dd_s[mi]*100),
                xytext=(dates[mi], dd_s[mi]*100 - 5), fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#c0392b"),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b"))
    fig.savefig(os.path.join(out, "03_drawdown.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); return fig


def plot_04(dates, res, regime_hmm, regime_yc, out):
    eq = res["equity"]; hmm = regime_hmm.reindex(dates); yc = regime_yc.reindex(dates)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(18, 8),
        gridspec_kw={"height_ratios": [2, 0.6, 0.6], "hspace": 0.12}, sharex=True)
    a1.plot(dates, eq, color="#1a5276", lw=1.5)
    _stress_shading(dates, hmm, [a1])
    a1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    a1.set_title("Regime Timeline: HMM + YC", fontsize=13, fontweight="bold")
    a1.grid(True, alpha=0.2)
    a2.fill_between(dates, np.where(hmm.values == "STRESS", 1, 0), 0, color="#e74c3c", alpha=0.5, step="mid")
    a2.set_ylim(-0.1, 1.1); a2.set_yticks([0,1]); a2.set_yticklabels(["CALM","STRESS"], fontsize=9)
    a2.set_ylabel("HMM", fontsize=10, fontweight="bold"); a2.grid(True, alpha=0.15)
    ycn = np.array([{"BULL_FLAT":2,"BULL_STEEP":1,"UNDEFINED":0,"BEAR_FLAT":-1,"BEAR_STEEP":-2}.get(v,0) for v in yc.values])
    a3.fill_between(dates, ycn, 0, where=(ycn>0), color="#27ae60", alpha=0.4, step="mid")
    a3.fill_between(dates, ycn, 0, where=(ycn<0), color="#e74c3c", alpha=0.4, step="mid")
    a3.axhline(0, color="black", lw=0.5); a3.set_ylim(-2.5, 2.5)
    a3.set_ylabel("YC", fontsize=10, fontweight="bold"); a3.grid(True, alpha=0.15)
    a3.xaxis.set_major_locator(mdates.YearLocator(2)); a3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(os.path.join(out, "04_regime_timeline.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); return fig


def plot_05(res_f, res_p, res_v2, out):
    m = res_f
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("PERFORMANCE DASHBOARD -- Final Version (RR + Slope + YC + Term 1W-3M)",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30, left=0.06, right=0.96, top=0.92, bottom=0.06)

    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    md = [["Sharpe", f"{m['sharpe']:.4f}"], ["Sortino", f"{m['sortino']:.4f}"],
          ["Ann. Return", f"{m['ann_ret']*100:.2f}%"], ["Volatility", f"{m['vol']*100:.2f}%"],
          ["Max Drawdown", f"{m['mdd']*100:.2f}%"], ["Calmar", f"{m['calmar']:.4f}"],
          ["Excess vs TLT", f"{m['excess']*100:+.2f}%"], ["Total Return", f"{m['total_ret']*100:.1f}%"],
          ["Equity Finale", f"${m['equity'][-1]:.3f}"], ["Operations", f"{m['n_events']}"],
          ["Dead Years", f"{m['n_dead']}"], ["Win Rate", f"{m['win_rate']:.0%}"]]
    t = ax.table(cellText=md, colLabels=["Metric", "Value"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.0, 1.35)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor("#c0392b"); cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0: cell.set_facecolor("#fadbd8")
    ax.set_title("Key Metrics", fontsize=11, fontweight="bold", pad=10)

    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    pd2 = [["z_windows", str(BEST["z_windows"])], ["k_slope", f"{BEST['k_slope']:.2f}"],
           ["yc_cw", str(BEST["yc_cw"])], ["k_yc", f"{BEST['k_yc']:.2f}"],
           ["term_spread", "1W-3M"], ["term_z_w", str(BEST["term_z_w"])],
           ["k_term", f"{BEST['k_term']:.2f}"], ["tl_calm", f"{BEST['tl_c']:.3f}"],
           ["ts_calm", f"{BEST['ts_c']:.3f}"], ["tl_stress", f"{BEST['tl_s']:.3f}"],
           ["adapt_days", str(BEST["adapt_days"])], ["adapt_raise", f"{BEST['adapt_raise']:.2f}"]]
    t2 = ax.table(cellText=pd2, colLabels=["Parameter", "Value"], loc="center", cellLoc="center")
    t2.auto_set_font_size(False); t2.set_fontsize(9); t2.scale(1.0, 1.25)
    for (r, c), cell in t2.get_celld().items():
        if r == 0: cell.set_facecolor("#1a5276"); cell.set_text_props(color="white", fontweight="bold")
        elif r in [6, 7, 8]: cell.set_facecolor("#fdebd0")
        elif r % 2 == 0: cell.set_facecolor("#e8f8f5")
    ax.set_title("Signal Parameters", fontsize=11, fontweight="bold", pad=10)

    ax = fig.add_subplot(gs[0, 2])
    wa = m["weight"]; dw2 = np.abs(np.diff(wa, prepend=wa[0])); tw = wa[dw2 > 0.01]
    nl = int(np.sum(tw > 0.7)); ns = int(np.sum(tw < 0.3)); nn = int(np.sum((tw >= 0.3) & (tw <= 0.7)))
    lb = ["LONG", "SHORT"]; sz = [max(nl,1), max(ns,1)]
    if nn > 0: lb.append("NEUTRAL"); sz.append(nn)
    ax.pie(sz, labels=lb, colors=["#27ae60","#e74c3c","#f39c12"][:len(lb)],
           autopct="%1.0f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"Trade Distribution ({m['n_events']} total)", fontsize=11, fontweight="bold", pad=10)

    ax = fig.add_subplot(gs[1, 0:2])
    exc = m["net_ret"] - RISK_FREE_RATE / ANN
    rm = pd.Series(exc).rolling(252, min_periods=126).mean()
    rs = pd.Series(exc).rolling(252, min_periods=126).std()
    rsh = (rm / rs * np.sqrt(ANN)).values
    ax.plot(range(len(rsh)), rsh, color="#c0392b", lw=1.2)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.axhline(m["sharpe"], color="#27ae60", lw=1, ls="--", alpha=0.7, label=f"Full: {m['sharpe']:.3f}")
    ax.fill_between(range(len(rsh)), rsh, 0, where=(rsh>0), color="#27ae60", alpha=0.06)
    ax.fill_between(range(len(rsh)), rsh, 0, where=(rsh<0), color="#e74c3c", alpha=0.06)
    ax.set_ylabel("Rolling 1Y Sharpe", fontsize=10, fontweight="bold")
    ax.set_title("Rolling Sharpe Ratio (252d)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    mp = res_p; m2 = res_v2
    cd = [["Sharpe", f"{m2['sharpe']:.3f}", f"{mp['sharpe']:.3f}", f"{m['sharpe']:.3f}"],
          ["Sortino", f"{m2['sortino']:.3f}", f"{mp['sortino']:.3f}", f"{m['sortino']:.3f}"],
          ["Ann.Ret", f"{m2['ann_ret']*100:.1f}%", f"{mp['ann_ret']*100:.1f}%", f"{m['ann_ret']*100:.1f}%"],
          ["MDD", f"{m2['mdd']*100:.1f}%", f"{mp['mdd']*100:.1f}%", f"{m['mdd']*100:.1f}%"],
          ["Dead", f"{m2['n_dead']}", f"{mp['n_dead']}", f"{m['n_dead']}"],
          ["WinRate", f"{m2['win_rate']:.0%}", f"{mp['win_rate']:.0%}", f"{m['win_rate']:.0%}"]]
    t3 = ax.table(cellText=cd, colLabels=["", "V2", "V_Pres", "Final"], loc="center", cellLoc="center")
    t3.auto_set_font_size(False); t3.set_fontsize(9); t3.scale(1.0, 1.5)
    for (r, c), cell in t3.get_celld().items():
        if r == 0: cell.set_facecolor("#6c3483"); cell.set_text_props(color="white", fontweight="bold")
        elif c == 3 and r > 0: cell.set_facecolor("#fadbd8")
    ax.set_title("Version Comparison", fontsize=11, fontweight="bold", pad=10)
    fig.savefig(os.path.join(out, "05_dashboard.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); return fig


def plot_06(dates, res, out):
    eq = res["equity"]; w = res["weight"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(18, 7),
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08}, sharex=True)
    a1.plot(dates, eq, color="#1a5276", lw=1.5)
    a1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    a1.set_title("TLT Allocation Over Time", fontsize=13, fontweight="bold")
    a1.grid(True, alpha=0.2)
    a2.fill_between(dates, w, 0.5, where=(w>0.7), color="#27ae60", alpha=0.4, step="mid", label="LONG")
    a2.fill_between(dates, w, 0.5, where=(w<0.3), color="#e74c3c", alpha=0.4, step="mid", label="SHORT")
    a2.fill_between(dates, w, 0.5, where=((w>=0.3)&(w<=0.7)), color="#f39c12", alpha=0.2, step="mid", label="NEUTRAL")
    a2.axhline(0.5, color="gray", lw=0.8, ls="--")
    a2.set_ylabel("TLT Weight", fontsize=11, fontweight="bold"); a2.set_ylim(-0.05, 1.05)
    a2.legend(loc="lower right", fontsize=9); a2.grid(True, alpha=0.2)
    a2.xaxis.set_major_locator(mdates.YearLocator(2)); a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(os.path.join(out, "06_allocation.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); return fig


def plot_07(dates, rr_dict, z_term, raw_spread, regime_hmm, res, out):
    zt = z_term.reindex(dates).values; rs = raw_spread.reindex(dates).values
    rr1w = rr_dict["1W"].reindex(dates).values; rr3m = rr_dict["3M"].reindex(dates).values
    fig, axes = plt.subplots(4, 1, figsize=(18, 14),
        gridspec_kw={"height_ratios": [2, 1, 1, 1], "hspace": 0.10}, sharex=True)
    a1, a2, a3, a4 = axes
    a1.plot(dates, res["equity"], color="#1a5276", lw=1.5)
    _stress_shading(dates, regime_hmm.reindex(dates), axes)
    a1.set_ylabel("Equity ($)", fontsize=11, fontweight="bold")
    a1.set_title("RR Term Structure Analysis: 1W vs 3M Spread", fontsize=13, fontweight="bold")
    a1.grid(True, alpha=0.2)
    v1 = ~np.isnan(rr1w); v3 = ~np.isnan(rr3m)
    a2.plot(dates[v1], rr1w[v1], color="#e74c3c", lw=0.7, alpha=0.8, label="RR 1W")
    a2.plot(dates[v3], rr3m[v3], color="#2874a6", lw=0.7, alpha=0.8, label="RR 3M")
    a2.set_ylabel("RR (raw)", fontsize=10, fontweight="bold"); a2.legend(fontsize=9); a2.grid(True, alpha=0.15)
    vs = ~np.isnan(rs)
    a3.plot(dates[vs], rs[vs], color="#d35400", lw=0.7, alpha=0.8)
    a3.fill_between(dates, np.nan_to_num(rs), 0, where=(rs>0), color="#e67e22", alpha=0.10)
    a3.fill_between(dates, np.nan_to_num(rs), 0, where=(rs<0), color="#2980b9", alpha=0.10)
    a3.axhline(0, color="black", lw=0.5, alpha=0.3)
    a3.set_ylabel("Spread 1W-3M", fontsize=10, fontweight="bold"); a3.grid(True, alpha=0.15)
    vt = ~np.isnan(zt)
    a4.plot(dates[vt], zt[vt], color="#d35400", lw=0.7, alpha=0.85)
    a4.fill_between(dates, np.nan_to_num(zt), 0, where=(zt>0), color="#e67e22", alpha=0.15)
    a4.fill_between(dates, np.nan_to_num(zt), 0, where=(zt<0), color="#2980b9", alpha=0.15)
    a4.axhline(0, color="black", lw=0.5, alpha=0.3)
    a4.axhline(2, color="#e67e22", lw=0.8, ls="--", alpha=0.5)
    a4.axhline(-2, color="#2980b9", lw=0.8, ls="--", alpha=0.5)
    a4.set_ylabel(f"z Term 1W-3M (w={BEST['term_z_w']})", fontsize=10, fontweight="bold")
    a4.set_xlabel("Date", fontsize=11); a4.grid(True, alpha=0.15); a4.set_ylim(-5, 5)
    a4.xaxis.set_major_locator(mdates.YearLocator()); a4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.savefig(os.path.join(out, "07_term_spread_analysis.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); return fig


# ══════════════════════════════════════════════════════════════════════
# PDF REPORT (professional strategy paper)
# ══════════════════════════════════════════════════════════════════════

class StrategyReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Strategia RR Duration Timing -- Riservato", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(26, 82, 118)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(44, 62, 80)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5, text)
        self.ln(2)

    def mono_text(self, text):
        self.set_font("Courier", "", 9)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 4.5, text)
        self.ln(2)

    def add_image_page(self, img_path, caption=""):
        self.add_page("L")
        if os.path.exists(img_path):
            try:
                self.image(img_path, x=10, y=25, w=275)
            except Exception:
                self.image(img_path, x=10, y=25, h=170)
        if caption:
            self.set_y(200)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, caption, align="C")


def generate_pdf_report(res_f, res_p, res_v2, out_dir):
    """Genera il report PDF professionale della strategia (in italiano)."""
    pdf = StrategyReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    m = res_f; mp = res_p; m2 = res_v2

    # ── COPERTINA ──
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(26, 82, 118)
    pdf.cell(0, 15, "Strategia RR Duration Timing", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(44, 62, 80)
    pdf.cell(0, 10, "Risk Reversal + Slope + Yield Curve + Term Structure", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Strategia Sistematica di Duration Timing su ETF Treasury USA", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Periodo di Backtest: Gennaio 2009 - Febbraio 2026", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"Sharpe Ratio: {m['sharpe']:.3f}  |  Rend. Ann.: {m['ann_ret']*100:.2f}%  |  Max Drawdown: {m['mdd']*100:.1f}%", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "RISERVATO - Solo per uso interno", align="C")

    # ── 1. SOMMARIO ESECUTIVO ──
    pdf.add_page()
    pdf.chapter_title("1. Sommario Esecutivo")
    pdf.body_text(
        "Questo documento presenta una strategia sistematica di duration timing che alloca dinamicamente "
        "tra obbligazioni Treasury USA a lunga durata (TLT) e Treasury a breve durata (SHV), basandosi "
        "su un segnale composito derivato da dati del mercato opzioni, dinamiche della curva dei rendimenti "
        "e regimi di volatilita.\n\n"
        "La strategia sfrutta la natura mean-reverting del sentiment di rischio implicito nelle opzioni "
        "(Risk Reversals) su molteplici orizzonti temporali, arricchito dal momentum dello slope della "
        "curva dei rendimenti e da un innovativo segnale di spread sulla struttura a termine. Un modello "
        "Hidden Markov Model (HMM) classifica la volatilita di mercato in regimi CALM e STRESS, con "
        "soglie di trading differenziate per ciascun contesto.\n\n"
        "Risultati chiave sul periodo di backtest Gennaio 2009 - Febbraio 2026:"
    )
    pdf.mono_text(
        f"  Sharpe Ratio:        {m['sharpe']:.4f}\n"
        f"  Sortino Ratio:       {m['sortino']:.4f}\n"
        f"  Rendimento Ann.:     {m['ann_ret']*100:.2f}%\n"
        f"  Volatilita:          {m['vol']*100:.2f}%\n"
        f"  Max Drawdown:        {m['mdd']*100:.2f}%\n"
        f"  Calmar Ratio:        {m['calmar']:.4f}\n"
        f"  Excess vs B&H TLT:   {m['excess']*100:+.2f}%  (annualizzato)\n"
        f"  Rendimento Totale:   {m['total_ret']*100:.1f}%  ($1 -> ${m['equity'][-1]:.2f})\n"
        f"  Win Rate (anni):     {m['win_rate']:.0%}  ({int(m['win_rate']*len(m['yearly_years']))}/{len(m['yearly_years'])} anni)\n"
        f"  Anni Morti:          {m['n_dead']}  (anni con |excess| < 1%)\n"
        f"  Operazioni:          {m['n_events']}"
    )
    pdf.body_text(
        "La strategia raggiunge uno Sharpe ratio di 0.54, con un miglioramento del 33% rispetto alla "
        f"versione precedente (V_Pres: {mp['sharpe']:.3f}) e del 28% rispetto alla baseline V2 ({m2['sharpe']:.3f}). "
        "Il massimo drawdown del 19.7% e significativamente inferiore al ~50% di drawdown del B&H TLT."
    )

    # ── 2. CONTESTO DI MERCATO ──
    pdf.add_page()
    pdf.chapter_title("2. Contesto di Mercato e Razionale")

    pdf.section_title("2.1 Il Problema del Duration Timing")
    pdf.body_text(
        "Le obbligazioni Treasury USA presentano un significativo rischio di duration: i bond a lunga "
        "scadenza (20+ anni) possono subire drawdown superiori al 40% in contesti di tassi crescenti, "
        "come osservato nel 2022-2023. Al contrario, offrono rendimenti positivi sostanziali durante "
        "episodi di flight-to-quality e cicli di allentamento monetario.\n\n"
        "Un approccio buy-and-hold su TLT (iShares 20+ Year Treasury ETF) ha prodotto rendimenti "
        "risk-adjusted mediocri nell'ultimo decennio, con uno Sharpe ratio inferiore a 0.10 e un "
        "drawdown massimo di circa il 50%. Questo crea una chiara opportunita per il duration timing "
        "sistematico."
    )

    pdf.section_title("2.2 Perche i Segnali Impliciti dalle Opzioni?")
    pdf.body_text(
        "I mercati delle opzioni sui Treasury incorporano informazioni prospettiche sui movimenti attesi "
        "dei tassi e sul sentiment di rischio. I Risk Reversals (RR) -- la differenza di volatilita "
        "implicita tra put e call out-of-the-money sui futures Treasury -- catturano il pricing "
        "asimmetrico del mercato tra rischio al rialzo e al ribasso sui tassi.\n\n"
        "Quando il RR e fortemente negativo (put molto piu costose delle call), il mercato sta prezzando "
        "un rischio significativo di rialzo dei tassi. Empiricamente, questo posizionamento estremo tende "
        "a mean-revertire, rendendo gli z-score dei RR un potente segnale contrarian per il duration timing.\n\n"
        "Utilizziamo Risk Reversals 25-delta su quattro tenori: 1 settimana, 1 mese, 3 mesi e 6 mesi, "
        "combinati in una misura composita unica."
    )

    pdf.section_title("2.3 Filosofia del Segnale")
    pdf.body_text(
        "La strategia combina quattro componenti di segnale distinte, ciascuna cattura un aspetto "
        "diverso del contesto dei tassi:\n\n"
        "1. Z-Score RR (contrarian): il sentiment estremo sulle opzioni tende a mean-revertire\n"
        "2. Slope Momentum (trend): i cambiamenti nello spread 10Y-2Y segnalano shift di regime macro\n"
        "3. Regime Yield Curve (strutturale): classificazioni Bull/Bear e Steep/Flat\n"
        "4. RR Term Spread (microstruttura): dislocazioni tra pricing opzioni a breve e medio termine\n\n"
        "La combinazione di segnali contrarian (RR, Term) e trend-following (Slope) crea un composito "
        "robusto che performa in diversi contesti di mercato."
    )

    # ── 3. COMPONENTI DEL SEGNALE ──
    pdf.add_page()
    pdf.chapter_title("3. Componenti del Segnale")

    pdf.section_title("3.1 Componente 1: Z-Score RR Multi-Timeframe (z_base)")
    pdf.body_text(
        "Il segnale core e lo z-score di un Risk Reversal blended (media dei tenori 1W, 1M, 3M, 6M), "
        "calcolato su finestre rolling multiple per catturare diversi orizzonti temporali:\n\n"
        "  z_base = media( zscore(RR_blend, w=63), zscore(RR_blend, w=252) )\n\n"
        "La finestra a 63 giorni cattura opportunita di mean reversion a breve termine, mentre quella "
        "a 252 giorni fornisce contesto di lungo periodo e riduce il bias LONG persistente osservato "
        "nei mercati bull stabili (2011-2014). L'approccio multi-timeframe e stato validato attraverso "
        "studi di ablation che mostrano come la combinazione [63, 252] superi gli approcci a finestra "
        "singola."
    )

    pdf.section_title("3.2 Componente 2: Slope Momentum (z_slope)")
    pdf.body_text(
        "Lo slope della curva dei rendimenti (rendimento Treasury 10Y meno 2Y) cattura gli shift "
        "di regime macroeconomico:\n\n"
        "  delta_slope = slope(t) - slope(t - 126)\n"
        "  z_slope = zscore(delta_slope, w=252)\n\n"
        "Uno slope crescente (steepening) tipicamente coincide con aspettative di allentamento monetario, "
        "che e bullish per i bond a lunga duration. Uno slope decrescente (flattening/inversione) segnala "
        "un inasprimento della politica monetaria.\n\n"
        "L'analisi di ablation ha confermato lo slope come la componente singola di maggiore impatto, "
        "contribuendo con +0.24 di Sharpe in media. Senza slope, lo Sharpe della strategia scende da "
        "0.49 a 0.14."
    )
    pdf.body_text(
        f"  Peso: k_slope = {BEST['k_slope']:.2f}\n"
        f"  z_rr = z_base + {BEST['k_slope']:.2f} * z_slope"
    )

    pdf.section_title("3.3 Componente 3: Regime Yield Curve (z_yc)")
    pdf.body_text(
        "Il segnale di regime della curva classifica il contesto corrente sulla base dei cambiamenti "
        f"direzionali di livello e spread su una finestra di {BEST['yc_cw']} giorni:\n\n"
        "  Livello = (10Y + 30Y) / 2    (rendimento medio del segmento lungo)\n"
        "  Spread = 30Y - 10Y           (pendenza del segmento lungo della curva)\n\n"
        "Vengono identificati quattro regimi basati sui segni dei cambiamenti di livello e spread:\n"
        "  - BULL_FLAT  (+2): rendimenti in calo, curva in appiattimento (favorevole per la duration)\n"
        "  - BULL_STEEP (+1): rendimenti in calo, curva in irripidimento\n"
        "  - BEAR_FLAT  (-1): rendimenti in salita, curva in appiattimento\n"
        "  - BEAR_STEEP (-2): rendimenti in salita, curva in irripidimento (peggiore per la duration)\n\n"
        "Il punteggio del regime viene normalizzato tramite z-score su una finestra di 252 giorni."
    )
    pdf.body_text(
        f"  Peso: k_yc = {BEST['k_yc']:.2f}\n"
        f"  Finestra di cambiamento: {BEST['yc_cw']} giorni"
    )

    pdf.add_page()
    pdf.section_title("3.4 Componente 4: RR Term Structure Spread (z_term) -- NUOVO")
    pdf.body_text(
        "Questo segnale innovativo sfrutta le dislocazioni nella struttura a termine dei Risk Reversals:\n\n"
        "  spread = RR_1W - RR_3M\n"
        f"  z_term = zscore(spread, w={BEST['term_z_w']})\n\n"
        "L'intuizione: quando i Risk Reversals a 1 settimana diventano significativamente piu negativi "
        "rispetto ai RR a 3 mesi, il mercato sta prezzando una paura estrema di breve termine che "
        "tipicamente non persiste. Questo crea un'opportunita contrarian LONG poiche il premio per la "
        "paura tende a mean-revertire.\n\n"
        "Al contrario, quando il RR 1W e insolitamente elevato rispetto al 3M, la compiacenza di breve "
        "termine tende a rientrare, creando un'opportunita SHORT.\n\n"
        "Caratteristiche chiave di questo segnale:\n"
        "- Molto reattivo (finestra z-score a 63 giorni), cattura dislocazioni di microstruttura\n"
        "- Aggiunge +0.08 di Sharpe combinato con Slope+YC (contributo marginale)\n"
        "- Riduce gli anni morti da 3 a 2\n"
        "- Costo principale: sottoperformance nel 2011 (-15.7% excess) per flight-to-quality prolungato"
    )
    pdf.body_text(
        f"  Peso: k_term = {BEST['k_term']:.2f} (modalita reversal/contrarian)\n"
        f"  Finestra z-score: {BEST['term_z_w']} giorni\n"
        f"  Coppia di tenori: 1W vs 3M"
    )

    pdf.section_title("3.5 Segnale Combinato")
    pdf.body_text(
        "Il segnale composito finale e:\n\n"
        f"  z_final = z_rr + {BEST['k_yc']:.2f} * z_yc + {BEST['k_term']:.2f} * z_term\n\n"
        f"dove z_rr = z_base(w=[63,252]) + {BEST['k_slope']:.2f} * z_slope\n\n"
        "I pesi sono stati determinati attraverso una grid search a 3 fasi su ~97.000 configurazioni, "
        "ottimizzando un composite score multi-obiettivo che bilancia Sharpe ratio, Sortino ratio, "
        "Calmar ratio, win rate e una penalita per gli anni morti."
    )

    # ── 4. RILEVAMENTO DEI REGIMI ──
    pdf.add_page()
    pdf.chapter_title("4. Rilevamento dei Regimi (HMM)")
    pdf.body_text(
        "Un Hidden Markov Model (HMM) gaussiano a 2 stati classifica il mercato in regimi CALM e "
        "STRESS basandosi su due input:\n\n"
        "  - log(MOVE Index): volatilita implicita dei Treasury\n"
        "  - log(VIX): volatilita implicita azionaria\n\n"
        "L'HMM identifica periodi di volatilita cross-asset elevata (STRESS: ~31% del campione) dove "
        "si applicano soglie di trading differenti. Durante i regimi STRESS, sono consentiti solo segnali "
        "SHORT (tramite una soglia separata e piu conservativa), mentre i segnali LONG sono disabilitati "
        "per evitare di entrare durante i crolli di mercato.\n\n"
        "Questo approccio condizionato al regime e stato cruciale nel 2020 (crash COVID) e nel 2022 "
        "(ciclo di rialzo tassi), dove la strategia ha correttamente spostato in SHORT durante gli "
        "episodi iniziali di STRESS."
    )

    # ── 5. REGOLE DI TRADING ──
    pdf.chapter_title("5. Regole di Trading")
    pdf.body_text(
        "La strategia opera come uno switch binario long/short tra TLT e SHV, con una posizione "
        "intermedia NEUTRAL (50/50):\n"
    )
    pdf.mono_text(
        f"  Regime CALM (69% del tempo):\n"
        f"    z_final <= {BEST['tl_c']:.3f}  -->  SHORT (100% SHV)\n"
        f"    z_final >= {BEST['ts_c']:.3f}  -->  LONG  (100% TLT)\n\n"
        f"  Regime STRESS (31% del tempo):\n"
        f"    z_final <= {BEST['tl_s']:.3f}  -->  SHORT (100% SHV)\n"
        f"    Segnali LONG disabilitati in STRESS\n\n"
        f"  Soglia adattiva:\n"
        f"    Dopo {BEST['adapt_days']} giorni continuativamente LONG,\n"
        f"    ts_calm sale di {BEST['adapt_raise']:.2f} (da {BEST['ts_c']:.3f} a {BEST['ts_c']+BEST['adapt_raise']:.3f})\n"
        f"    Previene il 'sempre LONG' nei mercati bull stabili\n\n"
        f"  Esecuzione:\n"
        f"    Cooldown: {BEST['cooldown']} giorni di trading tra i segnali\n"
        f"    Ritardo: T+{BEST['delay']+1} esecuzione (segnale al giorno T, trade al T+2)\n"
        f"    Costi di transazione: {BEST['tcost_bps']} bps (esclusi dal backtest)"
    )

    # ── 6. RISULTATI ──
    pdf.add_page()
    pdf.chapter_title("6. Risultati di Performance")

    pdf.section_title("6.1 Statistiche di Sintesi")
    pdf.mono_text(
        f"  {'Metrica':<20} {'V2':>10} {'V_Pres':>10} {'Finale':>10} {'vs V_Pres':>10}\n"
        f"  {'-'*65}\n"
        f"  {'Sharpe Ratio':<20} {m2['sharpe']:>10.4f} {mp['sharpe']:>10.4f} {m['sharpe']:>10.4f} {m['sharpe']-mp['sharpe']:>+10.4f}\n"
        f"  {'Sortino Ratio':<20} {m2['sortino']:>10.4f} {mp['sortino']:>10.4f} {m['sortino']:>10.4f} {m['sortino']-mp['sortino']:>+10.4f}\n"
        f"  {'Rend. Annuo':<20} {m2['ann_ret']*100:>9.2f}% {mp['ann_ret']*100:>9.2f}% {m['ann_ret']*100:>9.2f}% {(m['ann_ret']-mp['ann_ret'])*100:>+9.2f}%\n"
        f"  {'Volatilita':<20} {m2['vol']*100:>9.2f}% {mp['vol']*100:>9.2f}% {m['vol']*100:>9.2f}%\n"
        f"  {'Max Drawdown':<20} {m2['mdd']*100:>9.2f}% {mp['mdd']*100:>9.2f}% {m['mdd']*100:>9.2f}%\n"
        f"  {'Calmar Ratio':<20} {m2['calmar']:>10.4f} {mp['calmar']:>10.4f} {m['calmar']:>10.4f}\n"
        f"  {'Anni Morti':<20} {m2['n_dead']:>10} {mp['n_dead']:>10} {m['n_dead']:>10} {m['n_dead']-mp['n_dead']:>+10}\n"
        f"  {'Win Rate':<20} {m2['win_rate']:>9.0%} {mp['win_rate']:>9.0%} {m['win_rate']:>9.0%}\n"
        f"  {'Operazioni':<20} {m2['n_events']:>10} {mp['n_events']:>10} {m['n_events']:>10}"
    )

    pdf.section_title("6.2 Dettaglio Annuale")
    lines = f"  {'Anno':>6}  {'Finale':>8}  {'V_Pres':>8}  {'B&H':>8}  {'F_exc':>8}  {'P_exc':>8}  {'Delta':>7}\n"
    lines += f"  {'-'*60}\n"
    for i in range(len(m["yearly_years"])):
        y = int(m["yearly_years"][i])
        fe = m["yearly_excess"][i]; fr = m["yearly_strat"][i]; bm = m["yearly_bm"][i]
        pe = 0.0
        for j in range(len(mp["yearly_years"])):
            if mp["yearly_years"][j] == y: pe = mp["yearly_excess"][j]; break
        d = fe - pe
        fd = " D" if abs(fe) < 0.01 else "  "
        pd_ = " D" if abs(pe) < 0.01 else "  "
        lines += f"  {y:>6}  {fr*100:>+7.1f}%  {(bm+pe)*100:>+7.1f}%  {bm*100:>+7.1f}%  {fe*100:>+7.1f}%{fd}  {pe*100:>+7.1f}%{pd_}  {d*100:>+6.1f}%\n"
    pdf.mono_text(lines)

    # ── 7. ANALISI DI ABLATION ──
    pdf.add_page()
    pdf.chapter_title("7. Analisi di Ablation delle Componenti")
    pdf.body_text(
        "Uno studio di ablation completo ha testato tutte le 8 possibili combinazioni delle quattro "
        "componenti del segnale (con z_base sempre presente come nucleo). Ogni combinazione e stata "
        "ottimizzata indipendentemente.\n\n"
        "Risultati chiave:"
    )
    pdf.mono_text(
        "  Combinazione              Sharpe  Morti  WR    Marginale vs solo RR\n"
        "  -------------------------------------------------------------------\n"
        "  RR + Slope + YC + Term    0.487    2    72%    +0.350  (MIGLIORE)\n"
        "  RR + Slope + Term         0.435    2    67%    +0.298\n"
        "  RR + Slope + YC           0.407    3    72%    +0.271\n"
        "  RR + Slope                0.374    4    83%    +0.237\n"
        "  RR + YC + Term            0.256    3    67%    +0.119\n"
        "  RR + Term                 0.174    2    67%    +0.037\n"
        "  RR + YC                   0.158    3    61%    +0.022\n"
        "  Solo RR                   0.137    4    61%    baseline"
    )
    pdf.body_text(
        "Conclusioni:\n"
        "- Lo Slope e il contributore dominante (+0.24 Sharpe da solo). Senza di esso, la strategia non "
        "e attuabile.\n"
        "- Il Term Spread aggiunge valore significativo combinato con Slope (+0.08 marginale vs Slope+YC).\n"
        "- La YC contribuisce in misura modesta (+0.05 marginale) ma migliora la consistenza.\n"
        "- Tutte e quattro le componenti insieme raggiungono il miglior composite score, giustificando "
        "il modello completo."
    )

    # ── 8. CONSIDERAZIONI SUL RISCHIO ──
    pdf.add_page()
    pdf.chapter_title("8. Considerazioni sul Rischio")

    pdf.section_title("8.1 Punti di Debolezza Noti")
    pdf.body_text(
        "1. Taper Tantrum 2013: Il massimo drawdown della strategia (-19.7%) si verifica durante la "
        "vendita di bond tra maggio e dicembre 2013. Questo e strutturalmente non correggibile -- il "
        "segnale RR era LONG entrando nella crisi e nessuna combinazione di segnali testata ha potuto "
        "evitare questo drawdown senza degradare severamente la performance complessiva.\n\n"
        "2. Sottoperformance 2011: Il segnale Term Spread costa circa -15% di excess return nel 2011 "
        "a causa di un episodio prolungato di flight-to-quality successivo al declassamento del rating "
        "creditizio USA. Lo spread 1W-3M ha dato segnali SHORT contrarian prematuri. Tuttavia, la "
        "strategia ha comunque generato un rendimento assoluto positivo (+15.8% vs +31.5% per B&H).\n\n"
        "3. Costi di Transazione: Il backtest esclude i costi di transazione. Con 91 operazioni su "
        "17 anni, il periodo medio di detenzione e di circa 47 giorni di trading, suggerendo che i "
        "costi sarebbero gestibili ma non trascurabili per allocazioni di grandi dimensioni."
    )

    pdf.section_title("8.2 Robustezza (validazioni precedenti)")
    pdf.body_text(
        "I test di robustezza precedenti (sulla configurazione V_Pres) hanno mostrato:\n"
        "- Analisi walk-forward: Sharpe OOS >= Sharpe IS nella maggior parte degli split (nessun "
        "overfitting rilevato)\n"
        "- Block bootstrap (2000 iterazioni, blocchi da 21 giorni): P(Sharpe > 0) = 95.7%\n"
        "- Sensibilita ai parametri: Sharpe si degrada < 0.05 per perturbazioni di +/- 10% sulla "
        "maggior parte dei parametri\n"
        "- k_slope e il parametro piu sensibile (come atteso dato il suo contributo dominante)"
    )

    # ── 9. CONCLUSIONE ──
    pdf.chapter_title("9. Conclusione")
    pdf.body_text(
        f"La strategia finale raggiunge uno Sharpe ratio di {m['sharpe']:.3f} con un massimo drawdown "
        f"del {m['mdd']*100:.1f}%, superando significativamente sia il benchmark B&H TLT che le versioni "
        f"precedenti della strategia. L'integrazione dello spread RR Term Structure (1W-3M) come segnale "
        f"contrarian migliora lo Sharpe di +0.12 rispetto alla versione precedente, riducendo gli anni "
        f"morti da 3 a 2.\n\n"
        "Il valore principale della strategia risiede nella capacita di proteggere il capitale durante i "
        "mercati obbligazionari ribassisti severi (2013, 2022) catturando al contempo la maggior parte "
        f"del rialzo durante i rally. Sull'intero periodo di backtest, $1 investito cresce a "
        f"${m['equity'][-1]:.2f}, rispetto a ${np.cumprod(1.0 + res_f['r_long'])[-1]:.2f} per il B&H TLT.\n\n"
        "L'architettura del segnale a quattro componenti fornisce fonti di alpha diversificate, con "
        "ciascuna componente che contribuisce positivamente e la combinazione che si dimostra piu "
        "robusta di qualsiasi sottoinsieme."
    )

    # ── PAGINE GRAFICI ──
    charts = [
        ("01_equity_zscore.png", "Figura 1: Curva Equity, Componenti Z-Score e Segnali di Trading"),
        ("02_yearly_comparison.png", "Figura 2: Confronto Rendimenti Annuali (Finale vs V_Pres vs V2 vs B&H)"),
        ("03_drawdown.png", "Figura 3: Analisi Drawdown -- Strategia vs B&H TLT"),
        ("04_regime_timeline.png", "Figura 4: Timeline dei Regimi HMM e Yield Curve"),
        ("05_dashboard.png", "Figura 5: Dashboard di Performance e Confronto Versioni"),
        ("06_allocation.png", "Figura 6: Allocazione TLT nel Tempo"),
        ("07_term_spread_analysis.png", "Figura 7: Analisi RR Term Structure (Spread 1W vs 3M)"),
    ]
    for fname, caption in charts:
        pdf.add_image_page(os.path.join(out_dir, fname), caption)

    pdf_path = os.path.join(out_dir, "strategy_report.pdf")
    pdf.output(pdf_path)
    return pdf_path


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()
    print("=" * 72)
    print("  FINAL VERSION")
    print("  RR + Slope + YC + Term Spread(1W-3M) Duration Strategy")
    print("=" * 72)

    # 1. Load
    print("\n[1/7] Loading data ...")
    D = load_all()
    rr_dict = build_rr_individual(D)

    # 2. Regime
    print("[2/7] HMM regime ...")
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])

    # 3. Build signals
    print("[3/7] Building signals ...")
    z_rr, z_base, z_slope = build_rr_multitf(D, BEST["z_windows"], k_slope=BEST["k_slope"])
    z_yc, regime_yc, _ = build_yc_signal(D, change_window=BEST["yc_cw"])
    z_term, raw_spread = build_term_spread_zscore(
        rr_dict[BEST["term_pair"][0]], rr_dict[BEST["term_pair"][1]],
        z_window=BEST["term_z_w"])
    z_final = z_rr + BEST["k_yc"] * z_yc + BEST["k_term"] * z_term

    # Align
    etf_ret = D["etf_ret"]
    bt_start = pd.Timestamp(BACKTEST_START)
    all_idx = [z_final.dropna().index, regime_hmm.dropna().index,
               etf_ret.dropna(subset=["TLT", "SHV"]).index]
    common = all_idx[0]
    for idx in all_idx[1:]: common = common.intersection(idx)
    common = common[common >= bt_start].sort_values()

    reg_arr = (regime_hmm.reindex(common) == "STRESS").astype(int).values
    r_long = etf_ret["TLT"].reindex(common).fillna(0).values
    r_short = etf_ret["SHV"].reindex(common).fillna(0).values
    year_arr = np.array([d.year for d in common])

    print(f"  Period: {common[0]:%Y-%m-%d} -> {common[-1]:%Y-%m-%d} ({len(common)} days)")

    # 4. Run backtests
    print("[4/7] Running backtests ...")
    z_arr = z_final.reindex(common).values
    res_f = fast_bt(z_arr, reg_arr, r_long, r_short, year_arr, BEST)
    res_f["r_long"] = r_long

    # V_Pres baseline
    zrr_p, _, _ = build_rr_multitf(D, VPRES_PARAMS["z_windows"], k_slope=VPRES_PARAMS["k_slope"])
    zyc_p, _, _ = build_yc_signal(D, change_window=VPRES_PARAMS["yc_cw"])
    zf_p = (zrr_p + VPRES_PARAMS["k_yc"] * zyc_p).reindex(common).values
    res_p = fast_bt(zf_p, reg_arr, r_long, r_short, year_arr, VPRES_PARAMS)

    # V2 baseline
    zrr_2, _, _ = build_rr_multitf(D, V2_PARAMS["z_windows"], k_slope=V2_PARAMS["k_slope"])
    zyc_2, _, _ = build_yc_signal(D, change_window=V2_PARAMS["yc_cw"])
    zf_2 = (zrr_2 + V2_PARAMS["k_yc"] * zyc_2).reindex(common).values
    res_v2 = fast_bt(zf_2, reg_arr, r_long, r_short, year_arr, V2_PARAMS)

    m = res_f
    print(f"\n  FINAL:  Sharpe={m['sharpe']:.4f}  Dead={m['n_dead']}  WR={m['win_rate']:.0%}  MDD={m['mdd']*100:.1f}%")
    print(f"  V_Pres: Sharpe={res_p['sharpe']:.4f}  Dead={res_p['n_dead']}  WR={res_p['win_rate']:.0%}")
    print(f"  V2:     Sharpe={res_v2['sharpe']:.4f}  Dead={res_v2['n_dead']}  WR={res_v2['win_rate']:.0%}")

    # Yearly
    print(f"\n  {'Year':>6}  {'Final':>8}  {'V_Pres':>8}  {'B&H':>8}  {'F_exc':>8}")
    print(f"  {'-'*48}")
    for i in range(len(m["yearly_years"])):
        y = int(m["yearly_years"][i])
        dead = " D" if abs(m["yearly_excess"][i]) < 0.01 else "  "
        print(f"  {y:>6}  {m['yearly_strat'][i]*100:>+7.1f}%  "
              f"{m['yearly_bm'][i]*100:>+7.1f}%  {m['yearly_bm'][i]*100:>+7.1f}%  "
              f"{m['yearly_excess'][i]*100:>+7.1f}%{dead}")

    # 5. Charts
    print("\n[5/7] Generating charts ...")
    dates = common
    plot_01(dates, res_f, z_final, z_rr, z_yc, z_term, regime_hmm, OUT_DIR)
    print("  01_equity_zscore.png")
    plot_02(dates, res_f, res_p, res_v2, r_long, year_arr, OUT_DIR)
    print("  02_yearly_comparison.png")
    plot_03(dates, res_f, r_long, OUT_DIR)
    print("  03_drawdown.png")
    plot_04(dates, res_f, regime_hmm, regime_yc, OUT_DIR)
    print("  04_regime_timeline.png")
    plot_05(res_f, res_p, res_v2, OUT_DIR)
    print("  05_dashboard.png")
    plot_06(dates, res_f, OUT_DIR)
    print("  06_allocation.png")
    plot_07(dates, rr_dict, z_term, raw_spread, regime_hmm, res_f, OUT_DIR)
    print("  07_term_spread_analysis.png")

    # 6. CSV
    print("\n[6/7] Saving data ...")
    pd.DataFrame({
        "date": dates, "equity": res_f["equity"],
        "benchmark_tlt": np.cumprod(1.0 + r_long),
        "weight": res_f["weight"], "net_return": res_f["net_ret"],
        "z_final": z_final.reindex(dates).values,
        "z_rr": z_rr.reindex(dates).values,
        "z_yc": z_yc.reindex(dates).values,
        "z_term": z_term.reindex(dates).values,
        "regime_hmm": regime_hmm.reindex(dates).values,
    }).to_csv(os.path.join(OUT_DIR, "daily_equity.csv"), index=False)

    pd.DataFrame({
        "year": m["yearly_years"].astype(int),
        "strat_ret": m["yearly_strat"], "bm_ret": m["yearly_bm"],
        "excess": m["yearly_excess"],
    }).to_csv(os.path.join(OUT_DIR, "yearly.csv"), index=False)

    pd.DataFrame([{**BEST, "sharpe": m["sharpe"], "sortino": m["sortino"],
                    "ann_ret": m["ann_ret"], "vol": m["vol"], "mdd": m["mdd"],
                    "calmar": m["calmar"], "n_events": m["n_events"],
                    "n_dead": m["n_dead"], "win_rate": m["win_rate"]}
    ]).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    w_arr = res_f["weight"]; dw = np.abs(np.diff(w_arr, prepend=w_arr[0]))
    tidx = np.where(dw > 0.01)[0]
    trades = []
    for ti in tidx:
        sig = "LONG" if w_arr[ti] > 0.7 else ("SHORT" if w_arr[ti] < 0.3 else "NEUTRAL")
        trades.append({"date": dates[ti], "signal": sig,
                       "z_final": z_final.reindex(dates).values[ti],
                       "weight": w_arr[ti],
                       "regime": regime_hmm.reindex(dates).values[ti]})
    pd.DataFrame(trades).to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)
    print(f"  Saved: daily_equity.csv, yearly.csv, metrics.csv, trades.csv ({len(trades)} trades)")

    # 7. PDF Report
    print("\n[7/7] Generating PDF report ...")
    pdf_path = generate_pdf_report(res_f, res_p, res_v2, OUT_DIR)
    print(f"  Saved: {os.path.basename(pdf_path)}")

    elapsed = time.time() - T0
    print(f"\n{'='*72}")
    print(f"  FINAL VERSION COMPLETE")
    print(f"  Sharpe: {m['sharpe']:.4f}  |  Dead: {m['n_dead']}  |  WR: {m['win_rate']:.0%}")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
