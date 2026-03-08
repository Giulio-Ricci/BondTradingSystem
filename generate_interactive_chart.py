"""
generate_interactive_chart.py - Interactive Strategy Dashboard (Plotly)

Genera un dashboard HTML interattivo con 6 pannelli sincronizzati
per l'analisi della strategia RR+YC Duration Timing V2.

Pannelli:
    1. Equity Curve (strategy vs B&H TLT) con regime shading e trade markers
    2. Z-Score Final (z_final) con soglie
    3. Z-Score RR (z_rr component)
    4. Z-Score YC (z_yc component)
    5. Drawdown (strategy e benchmark)
    6. Allocation / Weight (TLT weight step line)

Prerequisiti:
    pip install plotly

Output:
    output/report/interactive_dashboard.html
"""

import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import SAVE_DIR, BACKTEST_START
from data_loader import load_all
from regime import fit_hmm_regime
from versione_finale_v2 import build_combined_signal, run_backtest, PARAMS


# ======================================================================
# OUTPUT DIRECTORY
# ======================================================================
REPORT_DIR = os.path.join(SAVE_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)


# ======================================================================
# HELPER: regime shading rectangles
# ======================================================================

def get_stress_intervals(regime_series):
    """Extract contiguous STRESS intervals as (start, end) pairs."""
    is_stress = (regime_series == "STRESS").astype(int)
    diff = is_stress.diff().fillna(is_stress.iloc[0])
    intervals = []
    start = None
    for dt, d in diff.items():
        if d == 1:
            start = dt
        elif d == -1 and start is not None:
            intervals.append((start, dt))
            start = None
    if start is not None:
        intervals.append((start, regime_series.index[-1]))
    return intervals


# ======================================================================
# MAIN: build dashboard
# ======================================================================

def build_dashboard(result, z_final, z_rr, z_yc, regime_hmm, params, etf_ret):
    """Build Plotly interactive dashboard and return the figure."""

    equity = result["equity"]
    benchmark = result["benchmark"]
    dd = result["drawdown"]
    net_ret = result["net_ret"]
    weight = result["weight"]
    ev_df = result["events"]
    m = result["metrics"]

    dates = equity.index

    # Build benchmark equity curves for other ETFs
    BM_ETFS = {
        "SHV":  dict(color="#a0a0a0", dash="dot",  width=1.0),
        "SHY":  dict(color="#ff7f0e", dash="dash", width=1.0),
        "IEI":  dict(color="#9467bd", dash="dash", width=1.0),
        "IEF":  dict(color="#2ca02c", dash="dash", width=1.0),
        "TLH":  dict(color="#e377c2", dash="dash", width=1.0),
        "GOVT": dict(color="#8c564b", dash="dash", width=1.0),
    }
    bm_equities = {}
    for etf in BM_ETFS:
        if etf in etf_ret.columns:
            r = etf_ret[etf].reindex(dates).fillna(0).values
            bm_equities[etf] = pd.Series(np.cumprod(1.0 + r), index=dates)

    # Reindex signals to backtest dates
    z_f = z_final.reindex(dates)
    z_r = z_rr.reindex(dates)
    z_y = z_yc.reindex(dates)
    reg = regime_hmm.reindex(dates)

    # Benchmark drawdown
    bm_cummax = benchmark.cummax()
    bm_dd = benchmark / bm_cummax - 1

    # Stress intervals for shading
    stress_intervals = get_stress_intervals(reg)

    # Threshold values
    tl_calm = params["tl_calm"]
    ts_calm = params["ts_calm"]
    tl_stress = params["tl_stress"]

    # Trade events
    if len(ev_df) > 0:
        ev_long = ev_df[ev_df["signal"] == "LONG"]
        ev_short = ev_df[ev_df["signal"] == "SHORT"]
    else:
        ev_long = pd.DataFrame(columns=["date", "signal", "z", "regime"])
        ev_short = pd.DataFrame(columns=["date", "signal", "z", "regime"])

    # ------------------------------------------------------------------
    # Create subplots: 6 rows, shared X
    # ------------------------------------------------------------------
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.40, 0.12, 0.10, 0.10, 0.14, 0.14],
        subplot_titles=(
            "Equity Curve",
            "Z-Score Final (z_final)",
            "Z-Score RR (z_rr)",
            "Z-Score YC (z_yc)",
            "Drawdown",
            "Allocation (TLT Weight)",
        ),
    )

    # ==================================================================
    # PANEL 1: Equity Curve
    # ==================================================================

    # Strategy equity
    fig.add_trace(go.Scatter(
        x=dates, y=equity.values,
        name="Strategy",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Strategy: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Benchmark B&H TLT
    fig.add_trace(go.Scatter(
        x=dates, y=benchmark.values,
        name="B&H TLT",
        line=dict(color="#888888", width=1.5, dash="dash"),
        hovertemplate="B&H TLT: $%{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # Other ETF benchmarks
    for etf, style in BM_ETFS.items():
        if etf in bm_equities:
            fig.add_trace(go.Scatter(
                x=dates, y=bm_equities[etf].values,
                name=f"B&H {etf}",
                line=dict(color=style["color"], width=style["width"],
                          dash=style["dash"]),
                visible=True,
                hovertemplate=f"B&H {etf}: $%{{y:.2f}}<extra></extra>",
            ), row=1, col=1)

    # Trade markers on equity: LONG
    if len(ev_long) > 0:
        eq_at_long = equity.reindex(ev_long["date"].values)
        fig.add_trace(go.Scatter(
            x=ev_long["date"].values,
            y=eq_at_long.values,
            mode="markers",
            name="LONG",
            marker=dict(symbol="triangle-up", size=12, color="#2ca02c",
                        line=dict(width=1, color="darkgreen")),
            hovertemplate=(
                "LONG<br>Date: %{x|%Y-%m-%d}<br>"
                "Equity: $%{y:.2f}<extra></extra>"
            ),
        ), row=1, col=1)

    # Trade markers on equity: SHORT
    if len(ev_short) > 0:
        eq_at_short = equity.reindex(ev_short["date"].values)
        fig.add_trace(go.Scatter(
            x=ev_short["date"].values,
            y=eq_at_short.values,
            mode="markers",
            name="SHORT",
            marker=dict(symbol="triangle-down", size=12, color="#d62728",
                        line=dict(width=1, color="darkred")),
            hovertemplate=(
                "SHORT<br>Date: %{x|%Y-%m-%d}<br>"
                "Equity: $%{y:.2f}<extra></extra>"
            ),
        ), row=1, col=1)

    # ==================================================================
    # PANEL 2: Z-Score Final
    # ==================================================================

    fig.add_trace(go.Scatter(
        x=dates, y=z_f.values,
        name="z_final",
        line=dict(color="#1a3a6a", width=1.5),
        hovertemplate="z_final: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    # Threshold lines
    fig.add_hline(y=tl_calm, line=dict(color="red", width=1, dash="dash"),
                  annotation_text=f"tl_calm={tl_calm}", annotation_position="bottom right",
                  row=2, col=1)
    fig.add_hline(y=ts_calm, line=dict(color="green", width=1, dash="dash"),
                  annotation_text=f"ts_calm={ts_calm}", annotation_position="top right",
                  row=2, col=1)
    fig.add_hline(y=tl_stress, line=dict(color="red", width=1, dash="dot"),
                  annotation_text=f"tl_stress={tl_stress}", annotation_position="bottom left",
                  row=2, col=1)

    # Trade markers on z_final
    if len(ev_long) > 0:
        fig.add_trace(go.Scatter(
            x=ev_long["date"].values,
            y=ev_long["z"].values,
            mode="markers",
            name="LONG (z)",
            marker=dict(symbol="triangle-up", size=9, color="#2ca02c"),
            showlegend=False,
            hovertemplate="LONG z=%{y:.3f}<extra></extra>",
        ), row=2, col=1)

    if len(ev_short) > 0:
        fig.add_trace(go.Scatter(
            x=ev_short["date"].values,
            y=ev_short["z"].values,
            mode="markers",
            name="SHORT (z)",
            marker=dict(symbol="triangle-down", size=9, color="#d62728"),
            showlegend=False,
            hovertemplate="SHORT z=%{y:.3f}<extra></extra>",
        ), row=2, col=1)

    # ==================================================================
    # PANEL 3: Z-Score RR
    # ==================================================================

    fig.add_trace(go.Scatter(
        x=dates, y=z_r.values,
        name="z_rr",
        line=dict(color="#001f5c", width=1.2),
        hovertemplate="z_rr: %{y:.3f}<extra></extra>",
    ), row=3, col=1)

    fig.add_hline(y=0, line=dict(color="#999", width=0.5), row=3, col=1)

    # ==================================================================
    # PANEL 4: Z-Score YC
    # ==================================================================

    fig.add_trace(go.Scatter(
        x=dates, y=z_y.values,
        name="z_yc",
        line=dict(color="#7b2d8e", width=1.2),
        hovertemplate="z_yc: %{y:.3f}<extra></extra>",
    ), row=4, col=1)

    fig.add_hline(y=0, line=dict(color="#999", width=0.5), row=4, col=1)

    # ==================================================================
    # PANEL 5: Drawdown
    # ==================================================================

    fig.add_trace(go.Scatter(
        x=dates, y=(dd.values * 100),
        name="Strategy DD",
        fill="tozeroy",
        line=dict(color="#d62728", width=1),
        fillcolor="rgba(214, 39, 40, 0.3)",
        hovertemplate="Strategy DD: %{y:.1f}%<extra></extra>",
    ), row=5, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=(bm_dd.values * 100),
        name="B&H TLT DD",
        line=dict(color="#888888", width=1, dash="dash"),
        hovertemplate="TLT DD: %{y:.1f}%<extra></extra>",
    ), row=5, col=1)

    # ==================================================================
    # PANEL 6: Allocation / Weight
    # ==================================================================

    w = weight.values
    w_dates = weight.index

    # Step line for weight
    fig.add_trace(go.Scatter(
        x=w_dates, y=w,
        name="TLT Weight",
        line=dict(color="#1f77b4", width=1.5, shape="hv"),
        hovertemplate="TLT Weight: %{y:.0%}<extra></extra>",
    ), row=6, col=1)

    # Color-coded fill: green when TLT (w>=0.5), red when SHV (w<0.5)
    # We do this with two filled traces
    w_tlt = np.where(w >= 0.5, w, np.nan)
    w_shv = np.where(w < 0.5, w, np.nan)

    fig.add_trace(go.Scatter(
        x=w_dates, y=w_tlt,
        fill="tozeroy",
        fillcolor="rgba(44, 160, 44, 0.2)",
        line=dict(width=0, shape="hv"),
        showlegend=False,
        hoverinfo="skip",
    ), row=6, col=1)

    fig.add_trace(go.Scatter(
        x=w_dates, y=w_shv,
        fill="tozeroy",
        fillcolor="rgba(214, 39, 40, 0.2)",
        line=dict(width=0, shape="hv"),
        showlegend=False,
        hoverinfo="skip",
    ), row=6, col=1)

    # Reference line at 0.5
    fig.add_hline(y=0.5, line=dict(color="#999", width=0.5, dash="dot"),
                  row=6, col=1)

    # ==================================================================
    # STRESS regime shading across all panels
    # ==================================================================

    for s_start, s_end in stress_intervals:
        for row in range(1, 7):
            fig.add_vrect(
                x0=s_start, x1=s_end,
                fillcolor="rgba(255, 182, 193, 0.25)",
                layer="below",
                line_width=0,
                row=row, col=1,
            )

    # ==================================================================
    # LAYOUT
    # ==================================================================

    # Build header metrics string
    header_text = (
        f"Sharpe: {m['sharpe']:.3f}  |  "
        f"Sortino: {m['sortino']:.3f}  |  "
        f"Ann.Ret: {m['ann_ret']*100:.2f}%  |  "
        f"Vol: {m['vol']*100:.1f}%  |  "
        f"Max DD: {m['mdd']*100:.1f}%  |  "
        f"Calmar: {m['calmar']:.3f}  |  "
        f"Trades: {m['n_events']}  |  "
        f"Period: {m['years']:.1f}y"
    )

    fig.update_layout(
        title=dict(
            text=(
                f"<b>RR+YC Duration Strategy V2 - Interactive Dashboard</b>"
                f"<br><span style='font-size:13px;color:#555'>{header_text}</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=18),
        ),
        height=1200,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=12),
        margin=dict(l=60, r=30, t=110, b=60),
    )

    # Y-axis labels
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="z_final", row=2, col=1)
    fig.update_yaxes(title_text="z_rr", row=3, col=1)
    fig.update_yaxes(title_text="z_yc", row=4, col=1)
    fig.update_yaxes(title_text="DD (%)", row=5, col=1)
    fig.update_yaxes(title_text="Weight", row=6, col=1, range=[-0.05, 1.05])

    # Light grid
    for row in range(1, 7):
        fig.update_yaxes(
            gridcolor="rgba(200,200,200,0.4)",
            gridwidth=0.5,
            zeroline=False,
            row=row, col=1,
        )
        fig.update_xaxes(
            gridcolor="rgba(200,200,200,0.2)",
            gridwidth=0.5,
            row=row, col=1,
        )

    # Spike lines for crosshair alignment
    for row in range(1, 7):
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=0.5,
            spikecolor="#999",
            spikedash="dot",
            row=row, col=1,
        )

    # Range selector buttons on bottom x-axis (row 6)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            x=0,
            y=1.06,
            font=dict(size=11),
        ),
        row=6, col=1,
    )

    # Subplot titles style
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=13, color="#333")
        ann["xanchor"] = "left"
        ann["x"] = 0.01

    return fig


# ======================================================================
# ENTRY POINT
# ======================================================================

def main():
    T0 = time.time()
    print("=" * 72)
    print("  INTERACTIVE DASHBOARD GENERATION - RR+YC Duration Strategy V2")
    print("=" * 72)

    # 1. Load data
    print("\n[1/4] Caricamento dati ...")
    D = load_all()

    # 2. Regime HMM
    print("[2/4] Regime HMM ...")
    regime_hmm = fit_hmm_regime(D["move"], D["vix"])

    # 3. Build signals and run backtest
    print("[3/4] Segnale combinato + Backtest ...")
    z_final, z_rr, z_yc, z_base, z_slope, regime_yc, yc_score = \
        build_combined_signal(D, PARAMS)

    start = pd.Timestamp(BACKTEST_START)
    result = run_backtest(z_final, regime_hmm, D["etf_ret"], PARAMS, start)

    m = result["metrics"]
    print(f"  Sharpe: {m['sharpe']:.4f}  |  Ann.Ret: {m['ann_ret']*100:.2f}%  |  "
          f"MDD: {m['mdd']*100:.1f}%  |  Events: {m['n_events']}")

    # 4. Build interactive dashboard
    print("[4/4] Generazione dashboard interattivo ...")
    fig = build_dashboard(result, z_final, z_rr, z_yc, regime_hmm, PARAMS,
                          D["etf_ret"])

    out_path = os.path.join(REPORT_DIR, "interactive_dashboard.html")
    fig.write_html(
        out_path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "rr_yc_strategy_dashboard",
                "height": 1200,
                "width": 1800,
                "scale": 2,
            },
        },
    )

    elapsed = time.time() - T0
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n{'='*72}")
    print(f"  DASHBOARD GENERATO: {out_path}")
    print(f"  Dimensione: {size_mb:.1f} MB  |  Tempo: {elapsed:.1f}s")
    print(f"{'='*72}")

    return out_path


if __name__ == "__main__":
    main()
