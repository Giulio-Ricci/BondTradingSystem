"""
versione_finale.py - VERSIONE FINALE della strategia RR Duration.

Configurazione ottimale fissata. Esegui questo script per riprodurre
esattamente la miglior equity line trovata.

Segnale composito: ALL|std|w63 + 10Y-2Y Slope Momentum (k=0.50)
Thresholds: tl_calm=-3.25, ts_calm=2.75, tl_stress=-4.00, cooldown=5

Risultati attesi:
  Sharpe:     0.325
  Sortino:    0.399
  Ann.Return: 5.39%
  Vol:        10.4%
  Max DD:     -23.6%
  Calmar:     0.228
  Excess:     +7.0% vs B&H TLT
  Events:     52
  Eq Final:   $2.45 (da $1.00)
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

from config import SAVE_DIR, RR_TENORS, BACKTEST_START, RISK_FREE_RATE
from data_loader import load_all
from regime import fit_hmm_regime

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETRI FINALI (FROZEN)
# ═══════════════════════════════════════════════════════════════════════════

PARAMS = {
    # --- Segnale base: RR z-score ---
    "rr_tenors":         ["1W", "1M", "3M", "6M"],   # tutti i tenors, media
    "z_type":            "std",                        # z-score standard (non MOVE-adjusted)
    "z_window":          63,                           # finestra rolling per media/std (~3 mesi)
    "z_min_periods":     20,                           # min osservazioni per rolling

    # --- Componente slope momentum ---
    "slope_long_yield":  "US_10Y",                     # gamba lunga dello slope
    "slope_short_yield": "US_2Y",                      # gamba corta dello slope
    "slope_mom_window":  126,                          # finestra per delta slope (~6 mesi)
    "slope_zscore_window": 252,                        # finestra per z-score del delta slope
    "slope_zscore_min_periods": 63,                    # min periods per z-score slope
    "k_slope":           0.50,                         # peso del slope momentum nel composito

    # --- Soglie di trading ---
    "tl_calm":           -3.25,                        # SHORT threshold in regime CALM
    "ts_calm":           +2.75,                        # LONG threshold in regime CALM
    "tl_stress":         -4.00,                        # SHORT threshold in regime STRESS
    "ts_stress":         99,                           # LONG in STRESS: disabilitato
    "allow_long_stress": False,                        # no LONG in STRESS

    # --- Esecuzione ---
    "cooldown":          5,                            # giorni minimi tra segnali consecutivi
    "delay":             1,                            # ritardo esecuzione (T+2)
    "tcost_bps":         5,                            # costo di transazione per rebalance (bps)

    # --- Allocazione ---
    "w_initial":         0.50,                         # allocazione iniziale (50% TLT / 50% SHV)
    "w_long":            1.00,                         # peso TLT dopo segnale LONG
    "w_short":           0.00,                         # peso TLT dopo segnale SHORT (=100% SHV)

    # --- Strumenti ---
    "long_etf":          "TLT",                        # ETF duration long (20+ anni)
    "short_etf":         "SHV",                        # ETF duration short (< 1 anno)

    # --- Regime HMM ---
    "hmm_n_seeds":       10,
    "hmm_n_iter":        300,
}

# Output
OUT_DIR = os.path.join(SAVE_DIR, "versione_finale")
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# COSTRUZIONE DEL SEGNALE
# ═══════════════════════════════════════════════════════════════════════════

def build_composite_signal(D, params):
    """
    Costruisce il segnale composito:
        z_composite = z_base_rr + k * z_slope_momentum

    Dove:
        z_base_rr = z-score rolling della media dei Risk Reversals (1W,1M,3M,6M)
        z_slope_momentum = z-score del delta 126gg dello slope 10Y-2Y
    """
    options = D["options"]
    yields = D["yields"]
    move = D["move"]

    # ── Step 1: Calcolo RR blend (media di tutti i tenors) ──
    rr_parts = []
    for tenor in params["rr_tenors"]:
        put_col, call_col = RR_TENORS[tenor]
        rr = (pd.to_numeric(options[put_col], errors="coerce")
              - pd.to_numeric(options[call_col], errors="coerce"))
        rr_parts.append(rr)
    rr_blend = pd.concat(rr_parts, axis=1).mean(axis=1)

    # ── Step 2: Z-score standard del RR blend ──
    w = params["z_window"]
    mp = params["z_min_periods"]
    mu = rr_blend.rolling(w, min_periods=mp).mean()
    sig = rr_blend.rolling(w, min_periods=mp).std()
    z_base = (rr_blend - mu) / sig.replace(0, np.nan)

    # ── Step 3: Slope momentum z-score ──
    col_l = params["slope_long_yield"]
    col_s = params["slope_short_yield"]
    slope = yields[col_l] - yields[col_s]

    mom_w = params["slope_mom_window"]
    delta_slope = slope.diff(mom_w)

    sz_w = params["slope_zscore_window"]
    sz_mp = params["slope_zscore_min_periods"]
    mu_slope = delta_slope.rolling(sz_w, min_periods=sz_mp).mean()
    sig_slope = delta_slope.rolling(sz_w, min_periods=sz_mp).std()
    z_slope = (delta_slope - mu_slope) / sig_slope.replace(0, np.nan)

    # ── Step 4: Composito ──
    k = params["k_slope"]
    z_composite = z_base + k * z_slope

    return z_composite, z_base, z_slope


# ═══════════════════════════════════════════════════════════════════════════
# BACKTEST
# ═══════════════════════════════════════════════════════════════════════════

def run_backtest(z, regime, etf_ret, params, start):
    """
    Esegue il backtest completo con output dettagliato.

    Returns dict con: equity, net_ret, weight, events, metrics, yearly
    """
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

    # ── Generazione segnali e pesi ──
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

    # ── Execution delay ──
    shift = 1 + delay
    w_del = np.empty(n)
    w_del[:shift] = w_init
    w_del[shift:] = weight[:n - shift]

    # ── Rendimenti portafoglio ──
    gross = w_del * r_long + (1.0 - w_del) * r_short
    dw = np.empty(n)
    dw[0] = abs(w_del[0] - w_init)
    dw[1:] = np.abs(np.diff(w_del))
    net = gross - dw * (tcost / 10_000)

    # ── Equity curve ──
    equity = pd.Series(np.cumprod(1.0 + net), index=common)

    # ── Metriche ──
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

    metrics = {
        "sharpe": sharpe, "sortino": sortino, "ann_ret": ann_ret,
        "vol": vol, "mdd": mdd, "calmar": calmar, "excess": excess,
        "total_ret": total_ret, "eq_final": equity.iloc[-1],
        "bm_ann": bm_ann, "n_events": len(events), "n_days": n, "years": years,
    }

    # ── Yearly breakdown ──
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
        "net_ret": pd.Series(net, index=common),
        "weight": pd.Series(w_del, index=common),
        "events": pd.DataFrame(events),
        "metrics": metrics,
        "yearly": pd.DataFrame(yearly_rows),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    T0 = time.time()

    print("=" * 70)
    print("  VERSIONE FINALE — RR Duration Strategy")
    print("  Composite: ALL|std|w63 + 10Y-2Y Slope Momentum (k=0.50)")
    print("=" * 70)

    # ── 1. Caricamento dati ──
    print("\n[1/5] Caricamento dati ...")
    D = load_all()

    # ── 2. Regime HMM ──
    print("\n[2/5] Regime HMM ...")
    regime = fit_hmm_regime(D["move"], D["vix"])

    # ── 3. Costruzione segnale composito ──
    print("\n[3/5] Costruzione segnale composito ...")
    z_composite, z_base, z_slope = build_composite_signal(D, PARAMS)

    n_valid = z_composite.dropna().shape[0]
    print(f"  z_base (RR ALL|std|w63):      {z_base.dropna().shape[0]} valori")
    print(f"  z_slope (10Y-2Y momentum):    {z_slope.dropna().shape[0]} valori")
    print(f"  z_composite:                  {n_valid} valori")

    # ── 4. Backtest ──
    print("\n[4/5] Backtest ...")
    start = pd.Timestamp(BACKTEST_START)
    result = run_backtest(z_composite, regime, D["etf_ret"], PARAMS, start)

    m = result["metrics"]
    print(f"\n  {'='*60}")
    print(f"  RISULTATI")
    print(f"  {'='*60}")
    print(f"  Sharpe Ratio:     {m['sharpe']:.4f}")
    print(f"  Sortino Ratio:    {m['sortino']:.4f}")
    print(f"  Ann. Return:      {m['ann_ret']:.4f}  ({m['ann_ret']*100:.2f}%)")
    print(f"  Volatility:       {m['vol']:.4f}  ({m['vol']*100:.2f}%)")
    print(f"  Max Drawdown:     {m['mdd']:.4f}  ({m['mdd']*100:.2f}%)")
    print(f"  Calmar Ratio:     {m['calmar']:.4f}")
    print(f"  Excess vs TLT:    {m['excess']:+.4f}  ({m['excess']*100:+.2f}%)")
    print(f"  Total Return:     {m['total_ret']:.4f}  ({m['total_ret']*100:.2f}%)")
    print(f"  Equity Finale:    ${m['eq_final']:.3f}")
    print(f"  Benchmark Ann:    {m['bm_ann']:.4f}  ({m['bm_ann']*100:.2f}%)")
    print(f"  Eventi:           {m['n_events']}")
    print(f"  Giorni:           {m['n_days']}")
    print(f"  Anni:             {m['years']:.1f}")

    # Yearly
    print(f"\n  Breakdown annuale:")
    print(f"  {'Anno':>6}  {'Strategia':>10}  {'B&H TLT':>10}  {'Excess':>10}  {'Vol':>8}  {'Ev':>4}")
    print(f"  {'-'*56}")
    yearly = result["yearly"]
    for _, yr in yearly.iterrows():
        marker = " *" if yr["excess"] > 0 else ""
        print(f"  {int(yr['year']):>6}  {yr['strat_ret']:>+10.4f}  {yr['bm_ret']:>+10.4f}  "
              f"{yr['excess']:>+10.4f}  {yr['vol']:>8.4f}  {int(yr['n_events']):>4}{marker}")
    pos = (yearly["excess"] > 0).sum()
    print(f"\n  Anni positivi: {pos}/{len(yearly)} ({pos/len(yearly)*100:.0f}%)")

    # Events
    ev_df = result["events"]
    if len(ev_df) > 0:
        print(f"\n  Lista operazioni ({len(ev_df)} eventi):")
        print(f"  {'#':>3}  {'Data':<12}  {'Segnale':<7}  {'z':>7}  {'Regime':<7}")
        print(f"  {'-'*44}")
        for i, (_, ev) in enumerate(ev_df.iterrows()):
            print(f"  {i+1:>3}  {ev['date'].strftime('%Y-%m-%d'):<12}  "
                  f"{ev['signal']:<7}  {ev['z']:>+7.3f}  {ev['regime']:<7}")

    # ── 5. Grafici ──
    print("\n[5/5] Grafici ...")

    equity = result["equity"]
    benchmark = result["benchmark"]
    dates = equity.index
    z_vals = z_composite.reindex(dates).values
    reg_series = regime.reindex(dates)

    # ── Grafico principale: Equity + Z-score ──
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 11),
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08},
        sharex=True,
    )

    # Regime shading
    stress_mask = (reg_series.values == "STRESS")
    i = 0
    while i < len(dates):
        if stress_mask[i]:
            j = i
            while j < len(dates) and stress_mask[j]:
                j += 1
            for ax in (ax1, ax2):
                ax.axvspan(dates[i], dates[min(j-1, len(dates)-1)],
                           color="#FFE0E0", alpha=0.35, zorder=0)
            i = j
        else:
            i += 1

    # TOP: Equity
    ax1.plot(dates, equity.values, color="#1a5276", linewidth=1.8, zorder=4)
    ax1.plot(dates, benchmark.values, color="#aab7b8", linewidth=1.2,
             linestyle="--", alpha=0.7, zorder=3)
    ax1.fill_between(dates, equity.values, benchmark.values,
                     where=(equity.values >= benchmark.values),
                     color="#27ae60", alpha=0.08, zorder=1)
    ax1.fill_between(dates, equity.values, benchmark.values,
                     where=(equity.values < benchmark.values),
                     color="#e74c3c", alpha=0.08, zorder=1)

    for _, ev in ev_df.iterrows():
        dt = ev["date"]
        if dt in equity.index:
            eq_val = equity.loc[dt]
            marker = "^" if ev["signal"] == "LONG" else "v"
            color = "#27ae60" if ev["signal"] == "LONG" else "#e74c3c"
            ax1.scatter(dt, eq_val, marker=marker, s=70, c=color,
                        edgecolors="black", linewidths=0.5, zorder=6)

    ax1.set_ylabel("Growth of $1", fontsize=12, fontweight="bold")
    ax1.set_title("VERSIONE FINALE — RR Duration Strategy (Composite: ALL|std|w63 + 10Y-2Y Slope)",
                  fontsize=14, fontweight="bold", pad=12)
    ax1.grid(True, alpha=0.2)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.2f}"))
    ax1.set_ylim(bottom=0.60)

    legend_handles = [
        Line2D([0], [0], color="#1a5276", linewidth=2, label=f"Strategy (Sh={m['sharpe']:.3f})"),
        Line2D([0], [0], color="#aab7b8", linewidth=1.2, linestyle="--", label="B&H TLT"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#27ae60",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5, label="LONG"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#e74c3c",
               markersize=10, markeredgecolor="black", markeredgewidth=0.5, label="SHORT"),
        plt.Rectangle((0, 0), 1, 1, fc="#FFE0E0", alpha=0.5, label="STRESS regime"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)

    ann_text = (f"Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}  |  "
                f"Ann.Ret: {m['ann_ret']*100:.2f}%\n"
                f"Vol: {m['vol']*100:.1f}%  |  MDD: {m['mdd']*100:.1f}%  |  "
                f"Calmar: {m['calmar']:.3f}  |  Events: {m['n_events']}")
    ax1.text(0.98, 0.03, ann_text, transform=ax1.transAxes,
             fontsize=8, fontfamily="monospace", ha="right", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       alpha=0.85, edgecolor="#cccccc"))

    # BOTTOM: Z-score
    tl_c = PARAMS["tl_calm"]
    ts_c = PARAMS["ts_calm"]
    tl_s = PARAMS["tl_stress"]

    ax2.plot(dates, z_vals, color="#2c3e50", linewidth=0.7, alpha=0.85, zorder=4)
    ax2.fill_between(dates, z_vals, 0, where=(z_vals > 0),
                     color="#27ae60", alpha=0.08, zorder=1)
    ax2.fill_between(dates, z_vals, 0, where=(z_vals < 0),
                     color="#e74c3c", alpha=0.08, zorder=1)

    ax2.axhline(tl_c, color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.axhline(ts_c, color="#27ae60", linewidth=1.2, linestyle="--", alpha=0.7)
    ax2.axhline(tl_s, color="#c0392b", linewidth=1.0, linestyle=":", alpha=0.6)
    ax2.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    for _, ev in ev_df.iterrows():
        dt = ev["date"]
        zv = ev["z"]
        marker = "^" if ev["signal"] == "LONG" else "v"
        color = "#27ae60" if ev["signal"] == "LONG" else "#e74c3c"
        ax2.scatter(dt, zv, marker=marker, s=90, c=color,
                    edgecolors="black", linewidths=0.6, zorder=6)

    ax2.set_ylabel("Composite Z-Score", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(-8, 8)
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    z_legend = [
        Line2D([0], [0], color="#e74c3c", linewidth=1.2, linestyle="--",
               label=f"SHORT calm ({tl_c})"),
        Line2D([0], [0], color="#27ae60", linewidth=1.2, linestyle="--",
               label=f"LONG calm ({ts_c})"),
        Line2D([0], [0], color="#c0392b", linewidth=1.0, linestyle=":",
               label=f"SHORT stress ({tl_s})"),
    ]
    ax2.legend(handles=z_legend, loc="lower left", fontsize=8, framealpha=0.9, ncol=3)

    fig.savefig(os.path.join(OUT_DIR, "equity_zscore.png"),
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Grafico salvato: {OUT_DIR}/equity_zscore.png")

    # ── Salvataggio dati ──
    # Metriche
    pd.DataFrame([{**PARAMS, **m}]).to_csv(
        os.path.join(OUT_DIR, "metrics.csv"), index=False)

    # Equity curve giornaliera
    eq_export = pd.DataFrame({
        "date": equity.index,
        "strategy": equity.values,
        "benchmark_tlt": benchmark.values,
        "weight_tlt": result["weight"].values,
        "net_return": result["net_ret"].values,
        "z_composite": z_vals,
        "regime": reg_series.values,
    })
    eq_export.to_csv(os.path.join(OUT_DIR, "daily_equity.csv"), index=False)

    # Trade list
    ev_df.to_csv(os.path.join(OUT_DIR, "trades.csv"), index=False)

    # Yearly
    yearly.to_csv(os.path.join(OUT_DIR, "yearly.csv"), index=False)

    elapsed = time.time() - T0
    print(f"\n{'='*70}")
    print(f"  VERSIONE FINALE COMPLETATA")
    print(f"  Sharpe: {m['sharpe']:.4f}  |  Equity: ${m['eq_final']:.3f}  |  "
          f"Events: {m['n_events']}")
    print(f"  Output: {OUT_DIR}/")
    print(f"  Tempo: {elapsed:.1f}s")
    print(f"{'='*70}")

    return result


if __name__ == "__main__":
    main()
