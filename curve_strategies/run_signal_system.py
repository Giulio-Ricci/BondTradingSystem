"""
run_signal_system.py - Orchestrator for the signal-driven curve strategy system.

Replaces Markowitz/ML with slope + RR signal-based strategy selection.

Usage:
    python -m curve_strategies.run_signal_system
    python curve_strategies/run_signal_system.py

Steps:
    1. Load all data + backtest 25 strategies
    2. Build returns matrix + correlation matrix
    3. Compute all signals (slope, curvature, RR, overlay)
    4. CurveSignalSystem.backtest() - in-sample
    5. CurveSignalSystem.walk_forward(2012-2025)
    6. explain_signals()
    7. Generate CSV output
    8. Build 6-tab Plotly dashboard
"""

import os
import sys
import time
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure parent dir is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BACKTEST_START, SAVE_DIR
from data_loader import load_all

from curve_strategies.strategy_definitions import STRATEGY_POOL
from curve_strategies.strategy_backtest import backtest_all, compute_metrics
from curve_strategies.portfolio_combinations import (
    build_returns_matrix, compute_correlation_matrix,
)
from curve_strategies.curve_signal_system import (
    CurveSignalSystem,
    compute_segment_signals, compute_curvature_signals, compute_global_overlays,
    SEGMENT_STRATEGIES, BUTTERFLY_SEGMENTS,
)


# ── Output directory ─────────────────────────────────────────────────────
OUT_DIR = os.path.join(SAVE_DIR, "signal_system")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 1 - SIGNAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def build_signal_dashboard_tab(seg_signals, overlay_signals, threshold=0.5):
    """Panels per segment: z_mom, z_level, spread. Highlight > threshold."""
    # Select segments that have data
    segments = []
    for seg in SEGMENT_STRATEGIES:
        if f"{seg}_z_mom" in seg_signals.columns:
            segments.append(seg)

    n_segs = len(segments)
    if n_segs == 0:
        fig = go.Figure()
        fig.update_layout(title="No signal data available")
        return fig

    fig = make_subplots(
        rows=n_segs, cols=2,
        shared_xaxes=True,
        vertical_spacing=0.03,
        horizontal_spacing=0.08,
        subplot_titles=[f"{s} Z-Scores" for s in segments]
                       + [f"{s} Spread" for s in segments],
        column_widths=[0.55, 0.45],
    )

    colors_mom = "#1f77b4"
    colors_level = "#ff7f0e"
    colors_spread = "#2ca02c"

    for i, seg in enumerate(segments):
        row = i + 1

        # Left panel: z_mom and z_level
        z_mom = seg_signals[f"{seg}_z_mom"]
        z_level = seg_signals[f"{seg}_z_level"]

        fig.add_trace(go.Scatter(
            x=z_mom.index, y=z_mom.values,
            name=f"{seg} z_mom", line=dict(color=colors_mom, width=1.2),
            showlegend=(i == 0),
            legendgroup="z_mom",
            hovertemplate=f"{seg} z_mom: %{{y:.2f}}<extra></extra>",
        ), row=row, col=1)

        fig.add_trace(go.Scatter(
            x=z_level.index, y=z_level.values,
            name=f"{seg} z_level", line=dict(color=colors_level, width=1.0),
            showlegend=(i == 0),
            legendgroup="z_level",
            hovertemplate=f"{seg} z_level: %{{y:.2f}}<extra></extra>",
        ), row=row, col=1)

        # Threshold lines
        fig.add_hline(y=threshold, line=dict(color="gray", width=0.5, dash="dot"),
                      row=row, col=1)
        fig.add_hline(y=-threshold, line=dict(color="gray", width=0.5, dash="dot"),
                      row=row, col=1)

        fig.update_yaxes(title_text=seg, row=row, col=1)

        # Right panel: raw spread
        spread_col = f"{seg}_spread"
        if spread_col in seg_signals.columns:
            spread = seg_signals[spread_col]
            fig.add_trace(go.Scatter(
                x=spread.index, y=spread.values,
                name=f"{seg} spread", line=dict(color=colors_spread, width=1.0),
                showlegend=False,
                hovertemplate=f"{seg} spread: %{{y:.3f}}<extra></extra>",
            ), row=row, col=2)

    fig.update_layout(
        title="<b>Curve Segment Signals</b>",
        height=250 * n_segs + 100,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(font=dict(size=10)),
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.03),
        rangeselector=dict(buttons=[
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        row=n_segs, col=1,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 2 - ACTIVE STRATEGIES HEATMAP
# ══════════════════════════════════════════════════════════════════════════

def build_active_strategies_tab(weights_df):
    """Heatmap: days x strategies, color = weight."""
    # Keep only strategies that are ever active
    active_cols = weights_df.columns[weights_df.abs().max() > 1e-6]
    if len(active_cols) == 0:
        fig = go.Figure()
        fig.update_layout(title="No active strategies")
        return fig

    w = weights_df[active_cols]

    # Downsample for performance (weekly)
    w_weekly = w.resample("W").last().fillna(0)

    fig = go.Figure(data=go.Heatmap(
        z=w_weekly.values.T,
        x=w_weekly.index,
        y=list(active_cols),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Weight"),
        hovertemplate="Date: %{x}<br>Strategy: %{y}<br>Weight: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Active Strategies Over Time</b> (weekly snapshot)",
        height=max(400, 30 * len(active_cols) + 200),
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Strategy",
        yaxis=dict(autorange="reversed"),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 3 - SEGMENT PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════

def build_segment_performance_tab(returns_matrix, seg_signals):
    """Per segment: steepener vs flattener equity, colored by signal direction."""
    segments_with_data = []
    for seg in SEGMENT_STRATEGIES:
        if f"{seg}_z_mom" in seg_signals.columns:
            segments_with_data.append(seg)

    n = len(segments_with_data)
    if n == 0:
        fig = go.Figure()
        fig.update_layout(title="No segment data")
        return fig

    fig = make_subplots(
        rows=n, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[f"{s} Segment" for s in segments_with_data],
    )

    colors = {"steepener": "#1f77b4", "flattener": "#ff7f0e"}

    for i, seg in enumerate(segments_with_data):
        row = i + 1
        mapping = SEGMENT_STRATEGIES[seg]

        for direction in ["steepener", "flattener"]:
            strat_name = mapping.get(direction)
            if strat_name and strat_name in returns_matrix.columns:
                equity = (1 + returns_matrix[strat_name]).cumprod()
                fig.add_trace(go.Scatter(
                    x=equity.index, y=equity.values,
                    name=f"{strat_name}",
                    line=dict(color=colors[direction], width=1.5),
                    showlegend=(i == 0),
                    legendgroup=direction,
                    hovertemplate=f"{strat_name}: $%{{y:.3f}}<extra></extra>",
                ), row=row, col=1)

        fig.update_yaxes(title_text="Equity", row=row, col=1)

    fig.update_layout(
        title="<b>Segment Performance</b> (Steepener vs Flattener Equity)",
        height=250 * n + 100,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.03),
        row=n, col=1,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 4 - SYSTEM EQUITY
# ══════════════════════════════════════════════════════════════════════════

def build_system_equity_tab(system_result, returns_matrix, system_metrics):
    """System equity vs benchmarks (EW all, B&H TLT, B&H IEF)."""
    fig = go.Figure()

    # Signal system equity
    equity = system_result["equity"]
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Signal System",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="Signal: $%{y:.3f}<extra></extra>",
    ))

    # EW benchmark (all 25 strategies)
    ew_ret = returns_matrix.mean(axis=1)
    ew_equity = (1 + ew_ret).cumprod()
    ew_equity = ew_equity.reindex(equity.index)
    fig.add_trace(go.Scatter(
        x=ew_equity.index, y=ew_equity.values,
        name="EW 25 Strats",
        line=dict(color="#7f7f7f", width=1.5, dash="dash"),
        hovertemplate="EW: $%{y:.3f}<extra></extra>",
    ))

    # B&H benchmarks
    for bh_name, color in [("bh_tlt", "#d62728"), ("bh_ief", "#2ca02c")]:
        if bh_name in returns_matrix.columns:
            bh_eq = (1 + returns_matrix[bh_name]).cumprod()
            bh_eq = bh_eq.reindex(equity.index)
            fig.add_trace(go.Scatter(
                x=bh_eq.index, y=bh_eq.values,
                name=bh_name.upper().replace("_", " "),
                line=dict(color=color, width=1.2, dash="dot"),
                hovertemplate=f"{bh_name}: $%{{y:.3f}}<extra></extra>",
            ))

    header = (
        f"Sharpe: {system_metrics.get('sharpe', 0):.3f}  |  "
        f"Ann.Ret: {system_metrics.get('ann_ret', 0)*100:.2f}%  |  "
        f"Vol: {system_metrics.get('vol', 0)*100:.1f}%  |  "
        f"MDD: {system_metrics.get('mdd', 0)*100:.1f}%"
    )

    fig.update_layout(
        title=dict(
            text=(f"<b>Signal System Equity</b>"
                  f"<br><span style='font-size:13px;color:#555'>{header}</span>"),
            x=0.5, xanchor="center",
        ),
        height=700,
        hovermode="x unified",
        yaxis_title="Equity ($)",
        template="plotly_white",
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True, thickness=0.04),
        ),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 5 - RISK MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════

def build_risk_management_tab(system_result):
    """Exposure, drawdown, turnover, realized vol vs target."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            "Total Exposure",
            "Drawdown",
            "Turnover",
            "Realized Volatility (63d rolling)",
        ),
    )

    equity = system_result["equity"]
    daily_ret = system_result["daily_ret"]
    exposure = system_result["exposure"]
    turnover_s = system_result["turnover"]

    # Panel 1: Exposure
    fig.add_trace(go.Scatter(
        x=exposure.index, y=exposure.values,
        name="Exposure", line=dict(color="#1f77b4", width=1),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.2)",
        hovertemplate="Exposure: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="Exposure", row=1, col=1)

    # Panel 2: Drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        name="Drawdown", line=dict(color="#d62728", width=1),
        fill="tozeroy", fillcolor="rgba(214,39,40,0.2)",
        hovertemplate="DD: %{y:.2%}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=-0.10, line=dict(color="black", width=0.5, dash="dash"),
                  annotation_text="DD threshold", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)

    # Panel 3: Turnover
    fig.add_trace(go.Scatter(
        x=turnover_s.index, y=turnover_s.values,
        name="Turnover", line=dict(color="#ff7f0e", width=0.8),
        hovertemplate="Turnover: %{y:.3f}<extra></extra>",
    ), row=3, col=1)
    fig.update_yaxes(title_text="Turnover", row=3, col=1)

    # Panel 4: Realized vol (63d rolling annualized)
    rvol = daily_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    fig.add_trace(go.Scatter(
        x=rvol.index, y=rvol.values,
        name="Realized Vol", line=dict(color="#9467bd", width=1.2),
        hovertemplate="RVol: %{y:.1%}<extra></extra>",
    ), row=4, col=1)
    fig.add_hline(y=0.15, line=dict(color="black", width=0.5, dash="dash"),
                  annotation_text="Target vol", row=4, col=1)
    fig.update_yaxes(title_text="Vol (ann.)", tickformat=".0%", row=4, col=1)

    fig.update_layout(
        title="<b>Risk Management Dashboard</b>",
        height=900,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.03),
        row=4, col=1,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 6 - WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════

def build_walk_forward_tab(wf_df):
    """Bar chart of OOS Sharpe by year + metrics table."""
    if len(wf_df) == 0:
        fig = go.Figure()
        fig.update_layout(title="Walk-Forward: No data")
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        vertical_spacing=0.12,
        specs=[[{"type": "bar"}], [{"type": "table"}]],
        subplot_titles=("OOS Sharpe by Year", "Detailed Metrics"),
    )

    colors = ["#2ca02c" if s > 0 else "#d62728"
              for s in wf_df["sharpe"]]

    fig.add_trace(go.Bar(
        x=wf_df["year"].astype(str),
        y=wf_df["sharpe"],
        marker_color=colors,
        text=[f"{s:.2f}" for s in wf_df["sharpe"]],
        textposition="outside",
        hovertemplate="Year: %{x}<br>Sharpe: %{y:.3f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    avg_sharpe = wf_df["sharpe"].mean()
    fig.add_hline(y=avg_sharpe,
                  line=dict(color="#1f77b4", width=2, dash="dash"),
                  annotation_text=f"Avg: {avg_sharpe:.3f}",
                  row=1, col=1)
    fig.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=1)

    # Add return annotations
    for _, row in wf_df.iterrows():
        fig.add_annotation(
            x=str(int(row["year"])),
            y=row["sharpe"],
            text=f"R:{row['ann_ret']*100:.1f}%",
            showarrow=False,
            yshift=-15 if row["sharpe"] > 0 else 15,
            font=dict(size=8, color="#666"),
        )

    # Table
    table_cols = ["year", "sharpe", "ann_ret", "vol", "mdd", "hit_rate"]
    available_cols = [c for c in table_cols if c in wf_df.columns]
    header_text = [c.replace("_", " ").title() for c in available_cols]

    # Add IS Sharpe and best params if available
    if "is_sharpe" in wf_df.columns:
        available_cols.append("is_sharpe")
        header_text.append("IS Sharpe")
    if "best_threshold" in wf_df.columns:
        available_cols.append("best_threshold")
        header_text.append("Threshold")
    if "best_w_mom" in wf_df.columns:
        available_cols.append("best_w_mom")
        header_text.append("W Mom")

    cell_values = []
    for col in available_cols:
        vals = wf_df[col].values
        if col == "year":
            cell_values.append([str(int(v)) for v in vals])
        elif col in ("ann_ret", "vol", "mdd", "hit_rate"):
            cell_values.append([f"{v:.2%}" for v in vals])
        else:
            cell_values.append([f"{v:.3f}" for v in vals])

    fig.add_trace(go.Table(
        header=dict(values=header_text,
                    fill_color="#2c3e50",
                    font=dict(color="white", size=11),
                    align="center"),
        cells=dict(values=cell_values,
                   fill_color="white",
                   font=dict(size=10),
                   align="center"),
    ), row=2, col=1)

    n_pos = (wf_df["sharpe"] > 0).sum()
    fig.update_layout(
        title=dict(
            text=(f"<b>Walk-Forward Validation</b>"
                  f"<br><span style='font-size:13px;color:#555'>"
                  f"Avg OOS Sharpe: {avg_sharpe:.3f}  |  "
                  f"Positive: {n_pos}/{len(wf_df)}"
                  f"</span>"),
            x=0.5, xanchor="center",
        ),
        height=800,
        template="plotly_white",
        showlegend=False,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD BUILDER
# ══════════════════════════════════════════════════════════════════════════

def _write_html(fig, path):
    """Write a Plotly figure to HTML."""
    fig.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png", "height": 900, "width": 1600, "scale": 2,
            },
        },
    )


def _build_tabbed_html(tab_figs, out_path):
    """Build a single HTML file with CSS/JS tab navigation using iframes."""
    out_dir = os.path.dirname(out_path)

    tabs = [
        ("Signal Dashboard",     "tab1_signals.html"),
        ("Active Strategies",    "tab2_active.html"),
        ("Segment Performance",  "tab3_segments.html"),
        ("System Equity",        "tab4_equity.html"),
        ("Risk Management",      "tab5_risk.html"),
        ("Walk-Forward",         "tab6_walkforward.html"),
    ]

    # Save individual tab HTMLs
    for (_, fname), fig in zip(tabs, tab_figs):
        _write_html(fig, os.path.join(out_dir, fname))

    # Build button and iframe HTML
    buttons_html = ""
    for i, (label, _) in enumerate(tabs):
        active = " active" if i == 0 else ""
        buttons_html += (
            f'    <button class="tab-btn{active}" '
            f'onclick="switchTab(event, {i})">{label}</button>\n'
        )

    iframes_html = ""
    for i, (_, fname) in enumerate(tabs):
        display = "block" if i == 0 else "none"
        iframes_html += (
            f'<iframe id="frame{i}" src="{fname}" '
            f'class="tab-frame" style="display:{display}"></iframe>\n'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Curve Signal System - Dashboard</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
    .header {{
        background: linear-gradient(135deg, #1a5c3a, #27ae60);
        color: white; padding: 20px 30px; text-align: center;
    }}
    .header h1 {{ margin: 0; font-size: 24px; }}
    .header p {{ margin: 5px 0 0; opacity: 0.8; font-size: 14px; }}
    .tab-bar {{
        display: flex; background: #2c3e50; padding: 0;
        border-bottom: 3px solid #1a5c3a;
    }}
    .tab-btn {{
        padding: 12px 20px; cursor: pointer; color: #bdc3c7;
        font-size: 13px; font-weight: bold; border: none;
        background: transparent; transition: all 0.2s;
    }}
    .tab-btn:hover {{ background: #34495e; color: white; }}
    .tab-btn.active {{ background: #1a5c3a; color: white; border-bottom: 3px solid #2ecc71; }}
    .tab-frame {{
        width: 100%; height: calc(100vh - 110px); border: none;
        background: white;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Curve Signal System</h1>
    <p>Signal-Driven Strategy Selection | Slope + RR Signals | Walk-Forward Validation</p>
</div>

<div class="tab-bar">
{buttons_html}</div>

{iframes_html}

<script>
function switchTab(evt, idx) {{
    var frames = document.querySelectorAll('.tab-frame');
    for (var i = 0; i < frames.length; i++) {{
        frames[i].style.display = 'none';
    }}
    var buttons = document.querySelectorAll('.tab-btn');
    for (var i = 0; i < buttons.length; i++) {{
        buttons[i].classList.remove('active');
    }}
    document.getElementById('frame' + idx).style.display = 'block';
    evt.currentTarget.classList.add('active');
}}
</script>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_full_dashboard(seg_signals, overlay_signals, system_result,
                         system_metrics, weights_df, returns_matrix,
                         wf_df, signal_threshold=0.5, out_dir=None):
    """Build and save the full 6-tab signal system dashboard.

    Args:
        seg_signals: segment signals DataFrame
        overlay_signals: global overlay signals DataFrame
        system_result: CurveSignalSystem.backtest() output
        system_metrics: system performance metrics
        weights_df: daily weights DataFrame
        returns_matrix: (T x N) strategy returns
        wf_df: walk-forward results DataFrame
        signal_threshold: threshold used (for display)
        out_dir: output directory

    Returns:
        Path to main dashboard HTML
    """
    if out_dir is None:
        out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("  Building dashboard tabs ...")
    fig1 = build_signal_dashboard_tab(seg_signals, overlay_signals, signal_threshold)
    fig2 = build_active_strategies_tab(weights_df)
    fig3 = build_segment_performance_tab(returns_matrix, seg_signals)
    fig4 = build_system_equity_tab(system_result, returns_matrix, system_metrics)
    fig5 = build_risk_management_tab(system_result)
    fig6 = build_walk_forward_tab(wf_df)

    tab_figs = [fig1, fig2, fig3, fig4, fig5, fig6]

    main_path = os.path.join(out_dir, "signal_system_dashboard.html")
    _build_tabbed_html(tab_figs, main_path)

    print(f"  Dashboard saved: {main_path}")
    return main_path


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  CURVE SIGNAL SYSTEM")
    print("  Signal-driven strategy selection | Slope + RR | Walk-Forward")
    print("=" * 72)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA + BACKTEST ALL 25 STRATEGIES
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/8] Loading data ...")
    D = load_all()
    etf_ret = D["etf_ret"]
    yields = D["yields"]
    rr_1m = D.get("rr_1m")
    move = D.get("move")
    vix = D.get("vix")

    print("\n  Backtesting 25 strategies ...")
    summary_df, results = backtest_all(STRATEGY_POOL, etf_ret,
                                       start=BACKTEST_START, tcost_bps=0)
    print(f"  {len(summary_df)} strategies backtested")
    print(f"  Top 5 by Sharpe:")
    for _, r in summary_df.head(5).iterrows():
        print(f"    {r['name']:20s}  Sharpe={r['sharpe']:.3f}  "
              f"Ret={r['ann_ret']:+.2%}  MDD={r['mdd']:.1%}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: BUILD RETURNS MATRIX + CORRELATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/8] Building returns matrix ...")
    strategy_returns = {name: res["daily_ret"] for name, res in results.items()}
    returns_matrix = build_returns_matrix(strategy_returns)
    print(f"  Shape: {returns_matrix.shape} "
          f"({returns_matrix.index[0].date()} to {returns_matrix.index[-1].date()})")

    corr_matrix = compute_correlation_matrix(returns_matrix)
    corr_matrix.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))
    print(f"  Correlation matrix saved")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: (SKIPPED) enumerate_combinations - optional, for comparison
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/8] Skipping combination enumeration (optional)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: COMPUTE ALL SIGNALS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/8] Computing signals ...")
    seg_signals = compute_segment_signals(yields)
    curv_signals = compute_curvature_signals(yields)
    overlay_signals = compute_global_overlays(rr_1m, move, vix)

    print(f"  Segment signals: {seg_signals.shape}")
    print(f"  Curvature signals: {curv_signals.shape}")
    print(f"  Overlay signals: {overlay_signals.shape}")

    # Check signal quality
    for col in seg_signals.columns:
        if "_z_" in col:
            valid = seg_signals[col].dropna()
            if len(valid) > 0:
                print(f"    {col}: range [{valid.min():.2f}, {valid.max():.2f}], "
                      f"mean={valid.mean():.3f}, {len(valid)} valid")

    seg_signals.to_csv(os.path.join(OUT_DIR, "segment_signals.csv"))
    print(f"  Signals saved to CSV")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: IN-SAMPLE BACKTEST
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5/8] Signal system in-sample backtest ...")
    system = CurveSignalSystem(
        returns_matrix=returns_matrix,
        yields=yields,
        rr_1m=rr_1m,
        move=move,
        vix=vix,
        strategy_pool=STRATEGY_POOL,
    )

    is_result = system.backtest()
    is_metrics = is_result["metrics"]

    print(f"\n  === IN-SAMPLE RESULTS ===")
    print(f"  Sharpe:   {is_metrics.get('sharpe', 0):.3f}")
    print(f"  Ann.Ret:  {is_metrics.get('ann_ret', 0):+.2%}")
    print(f"  Vol:      {is_metrics.get('vol', 0):.2%}")
    print(f"  MDD:      {is_metrics.get('mdd', 0):.2%}")
    print(f"  Hit Rate: {is_metrics.get('hit_rate', 0):.1%}")
    print(f"  Days:     {is_metrics.get('n_days', 0)}")

    avg_active = is_result["active_strategies"].mean()
    avg_exposure = is_result["exposure"].mean()
    avg_turnover = is_result["turnover"].mean()
    print(f"  Avg active strategies: {avg_active:.1f}")
    print(f"  Avg exposure: {avg_exposure:.2f}")
    print(f"  Avg turnover: {avg_turnover:.3f}")

    # Save backtest results
    is_result["daily_ret"].to_csv(
        os.path.join(OUT_DIR, "signal_backtest.csv"), header=True)
    is_result["weights"].to_csv(
        os.path.join(OUT_DIR, "signal_weights_daily.csv"))

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n[6/8] Walk-forward validation (2012-2025) ...")
    t_wf = time.time()
    wf_df = system.walk_forward(start_year=2012, end_year=2025)

    if len(wf_df) > 0:
        avg_oos_sharpe = wf_df["sharpe"].mean()
        n_positive = (wf_df["sharpe"] > 0).sum()
        print(f"\n  === WALK-FORWARD SUMMARY ===")
        print(f"  Avg OOS Sharpe: {avg_oos_sharpe:.3f}")
        print(f"  Positive years: {n_positive}/{len(wf_df)}")
        print(f"  Best year:  {wf_df.loc[wf_df['sharpe'].idxmax(), 'year']:.0f} "
              f"(Sharpe={wf_df['sharpe'].max():.3f})")
        print(f"  Worst year: {wf_df.loc[wf_df['sharpe'].idxmin(), 'year']:.0f} "
              f"(Sharpe={wf_df['sharpe'].min():.3f})")
        print(f"  WF time: {time.time() - t_wf:.0f}s")

        wf_df.to_csv(os.path.join(OUT_DIR, "walk_forward_results.csv"),
                      index=False)
    else:
        print("  No walk-forward results (insufficient data)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: EXPLAIN SIGNALS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[7/8] Explaining signals ...")
    explanation = system.explain_signals()
    if len(explanation) > 0:
        explanation.to_csv(os.path.join(OUT_DIR, "signal_explanation.csv"),
                           index=False)
        print(f"  {len(explanation)} active days explained")
        print(f"  Most common active strategies:")
        # Count strategy appearances
        all_strats = []
        for strats in explanation["active_strategies"]:
            all_strats.extend([s.strip() for s in strats.split(",")])
        from collections import Counter
        top_strats = Counter(all_strats).most_common(8)
        for strat, count in top_strats:
            pct = count / len(explanation) * 100
            print(f"    {strat:25s}  {count:5d} days ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    print("\n[8/8] Generating dashboard ...")
    dashboard_path = build_full_dashboard(
        seg_signals=seg_signals,
        overlay_signals=overlay_signals,
        system_result=is_result,
        system_metrics=is_metrics,
        weights_df=is_result["weights"],
        returns_matrix=returns_matrix,
        wf_df=wf_df,
        out_dir=OUT_DIR,
    )

    # ══════════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {OUT_DIR}")
    print(f"  Dashboard: {dashboard_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
