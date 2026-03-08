"""
run_regime_system.py - Orchestrator for the multi-mode regime-adaptive system.

Runs 6 signal modes in parallel, selects the best based on rolling
performance + YC regime filter, and produces a 6-tab interactive dashboard.

Usage:
    python -m curve_strategies.run_regime_system
    python curve_strategies/run_regime_system.py

Steps:
    1. Load all data + backtest 25 strategies
    2. Build returns matrix + compute regimes (YC + HMM)
    3. Compute all mode returns (6 modes in parallel)
    4. Compute mode-regime matrix (which mode works where)
    5. Adaptive backtest (in-sample)
    6. Walk-forward validation (2012-2025)
    7. Explain daily decisions
    8. CSV output + 6-tab dashboard
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
from curve_strategies.strategy_backtest import (
    backtest_all, compute_metrics, compute_yearly_breakdown,
)
from curve_strategies.portfolio_combinations import build_returns_matrix
from curve_strategies.regime_analysis import compute_yc_regime
from curve_strategies.curve_regime_system import (
    CurveRegimeSystem, SIGNAL_MODES,
)


# ── Output directory ─────────────────────────────────────────────────────
OUT_DIR = os.path.join(SAVE_DIR, "regime_system")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 1 - MODE EQUITY CURVES
# ══════════════════════════════════════════════════════════════════════════

def build_mode_equity_tab(mode_returns, adaptive_result):
    """Equity curves of all 6 modes + adaptive, overlaid."""
    fig = go.Figure()

    mode_colors = {
        "MR_level": "#1f77b4", "TF_level": "#ff7f0e",
        "MOM_TF": "#2ca02c", "MOM_MR": "#d62728",
        "BLEND_TF": "#9467bd", "BLEND_MR": "#8c564b",
    }

    for mode_name, mode_ret in mode_returns.items():
        equity = (1 + mode_ret).cumprod()
        m = compute_metrics(mode_ret)
        label = (f"{mode_name} (S={m.get('sharpe', 0):.2f}, "
                 f"R={m.get('ann_ret', 0):+.1%})")
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            name=label,
            line=dict(color=mode_colors.get(mode_name, "#666"),
                      width=1.2),
            hovertemplate=(f"{mode_name}: $%{{y:.3f}}<extra></extra>"),
        ))

    # Adaptive system
    adaptive_eq = adaptive_result["equity"]
    am = adaptive_result["metrics"]
    fig.add_trace(go.Scatter(
        x=adaptive_eq.index, y=adaptive_eq.values,
        name=(f"ADAPTIVE (S={am.get('sharpe', 0):.2f}, "
              f"R={am.get('ann_ret', 0):+.1%})"),
        line=dict(color="black", width=3),
        hovertemplate="Adaptive: $%{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Mode Equity Curves</b> (all 6 modes + adaptive system)",
        height=700,
        template="plotly_white",
        hovermode="x unified",
        yaxis_title="Equity ($)",
        legend=dict(font=dict(size=10), x=0.01, y=0.99),
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
# DASHBOARD: TAB 2 - MODE-REGIME HEATMAP
# ══════════════════════════════════════════════════════════════════════════

def build_mode_regime_tab(mode_regime_matrix):
    """Heatmap of Sharpe ratios: modes (rows) x regimes (cols)."""
    matrix = mode_regime_matrix.copy()

    # Annotations text
    text_matrix = []
    for i in range(len(matrix)):
        row_text = []
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if np.isnan(val):
                row_text.append("")
            else:
                row_text.append(f"{val:.2f}")
        text_matrix.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=list(matrix.columns),
        y=list(matrix.index),
        colorscale="RdYlGn",
        zmid=0,
        zmin=-2, zmax=2,
        colorbar=dict(title="Sharpe"),
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(size=14, color="black"),
        hovertemplate=("Mode: %{y}<br>Regime: %{x}"
                       "<br>Sharpe: %{z:.3f}<extra></extra>"),
    ))

    fig.update_layout(
        title=dict(
            text=("<b>Mode-Regime Performance Matrix</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  "Sharpe ratio by signal mode and YC regime</span>"),
            x=0.5, xanchor="center",
        ),
        height=500,
        template="plotly_white",
        xaxis_title="YC Regime",
        yaxis_title="Signal Mode",
        yaxis=dict(autorange="reversed"),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 3 - ADAPTIVE SYSTEM
# ══════════════════════════════════════════════════════════════════════════

def build_adaptive_system_tab(adaptive_result):
    """3 panels: equity + mode timeline + exposure."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(
            "Adaptive System Equity",
            "Active Mode Over Time",
            "Total Exposure",
        ),
    )

    equity = adaptive_result["equity"]
    mode_tl = adaptive_result["mode_timeline"]
    exposure = adaptive_result["exposure"]
    metrics = adaptive_result["metrics"]

    # Panel 1: Equity
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Equity",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Equity: $%{y:.3f}<extra></extra>",
    ), row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)

    # Panel 2: Mode timeline as colored scatter
    mode_names_all = list(SIGNAL_MODES.keys()) + ["FLAT"]
    mode_color_map = {
        "MR_level": "#1f77b4", "TF_level": "#ff7f0e",
        "MOM_TF": "#2ca02c", "MOM_MR": "#d62728",
        "BLEND_TF": "#9467bd", "BLEND_MR": "#8c564b",
        "FLAT": "#cccccc",
    }
    # Convert mode names to numeric for scatter
    mode_to_num = {m: i for i, m in enumerate(mode_names_all)}
    mode_nums = [mode_to_num.get(m, len(mode_names_all) - 1)
                 for m in mode_tl.values]
    mode_colors = [mode_color_map.get(m, "#999") for m in mode_tl.values]

    # Downsample for performance
    step = max(1, len(mode_tl) // 2000)
    ds_idx = list(range(0, len(mode_tl), step))

    fig.add_trace(go.Scatter(
        x=[mode_tl.index[i] for i in ds_idx],
        y=[mode_nums[i] for i in ds_idx],
        mode="markers",
        marker=dict(
            color=[mode_colors[i] for i in ds_idx],
            size=4,
        ),
        name="Mode",
        hovertemplate=[f"{mode_tl.values[i]}<extra></extra>"
                       for i in ds_idx],
        showlegend=False,
    ), row=2, col=1)

    fig.update_yaxes(
        tickvals=list(range(len(mode_names_all))),
        ticktext=mode_names_all,
        title_text="Mode",
        row=2, col=1,
    )

    # Panel 3: Exposure
    fig.add_trace(go.Scatter(
        x=exposure.index, y=exposure.values,
        name="Exposure",
        line=dict(color="#ff7f0e", width=1),
        fill="tozeroy", fillcolor="rgba(255,127,14,0.15)",
        hovertemplate="Exposure: %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=3, col=1)
    fig.update_yaxes(title_text="Exposure", row=3, col=1)

    header = (
        f"Sharpe: {metrics.get('sharpe', 0):.3f}  |  "
        f"Ann.Ret: {metrics.get('ann_ret', 0)*100:.2f}%  |  "
        f"Vol: {metrics.get('vol', 0)*100:.1f}%  |  "
        f"MDD: {metrics.get('mdd', 0)*100:.1f}%"
    )

    fig.update_layout(
        title=dict(
            text=(f"<b>Adaptive Regime System</b>"
                  f"<br><span style='font-size:13px;color:#555'>"
                  f"{header}</span>"),
            x=0.5, xanchor="center",
        ),
        height=900,
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.03),
        row=3, col=1,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 4 - YEARLY BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════

def build_yearly_breakdown_tab(mode_returns, adaptive_result):
    """Table: annual Sharpe for each mode + adaptive. Highlight best."""
    mode_names = list(mode_returns.keys())

    # Compute yearly Sharpe for each mode
    yearly_data = {}
    for mn, mr in mode_returns.items():
        yb = compute_yearly_breakdown(mr)
        yearly_data[mn] = yb.set_index("year")["sharpe"]

    # Adaptive yearly
    adaptive_yb = compute_yearly_breakdown(adaptive_result["daily_ret"])
    yearly_data["ADAPTIVE"] = adaptive_yb.set_index("year")["sharpe"]

    # Build combined DataFrame
    all_years = sorted(set().union(
        *[set(s.index) for s in yearly_data.values()]))
    combined = pd.DataFrame(index=all_years)
    for name, series in yearly_data.items():
        combined[name] = series.reindex(all_years)

    # Build table
    cols = list(combined.columns)
    header_vals = ["Year"] + cols
    cell_values = [[str(int(y)) for y in combined.index]]

    for col in cols:
        vals = combined[col].values
        formatted = []
        for v in vals:
            if np.isnan(v):
                formatted.append("")
            else:
                formatted.append(f"{v:.2f}")
        cell_values.append(formatted)

    # Color cells: green if positive, red if negative
    fill_colors = [["white"] * len(combined)]  # Year column
    for col in cols:
        vals = combined[col].values
        colors = []
        for v in vals:
            if np.isnan(v):
                colors.append("white")
            elif v > 0.5:
                colors.append("rgba(46,204,113,0.4)")
            elif v > 0:
                colors.append("rgba(46,204,113,0.15)")
            elif v > -0.5:
                colors.append("rgba(231,76,60,0.15)")
            else:
                colors.append("rgba(231,76,60,0.4)")
        fill_colors.append(colors)

    # Bold the best mode per year
    font_weights = [["normal"] * len(combined)]  # Year column
    for ci, col in enumerate(cols):
        weights = []
        for yi in range(len(combined)):
            row_vals = combined.iloc[yi].values
            best_val = np.nanmax(row_vals)
            if (not np.isnan(combined.iloc[yi][col])
                    and abs(combined.iloc[yi][col] - best_val) < 1e-6):
                weights.append("bold")
            else:
                weights.append("normal")
        font_weights.append(weights)

    fig = go.Figure(data=go.Table(
        header=dict(
            values=header_vals,
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="center",
        ),
        cells=dict(
            values=cell_values,
            fill_color=fill_colors,
            font=dict(size=11),
            align="center",
        ),
    ))

    fig.update_layout(
        title=dict(
            text=("<b>Yearly Sharpe Breakdown</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  "Annual Sharpe by mode | Best highlighted</span>"),
            x=0.5, xanchor="center",
        ),
        height=max(500, 35 * len(combined) + 200),
        template="plotly_white",
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD: TAB 5 - RISK DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def build_risk_dashboard_tab(adaptive_result):
    """4 panels: exposure, drawdown, turnover, realized vol."""
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

    equity = adaptive_result["equity"]
    daily_ret = adaptive_result["daily_ret"]
    exposure = adaptive_result["exposure"]
    turnover_s = adaptive_result["turnover"]

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

    # Panel 4: Realized vol
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
    """Bar chart of OOS Sharpe + metrics table + mode info."""
    if len(wf_df) == 0:
        fig = go.Figure()
        fig.update_layout(title="Walk-Forward: No data")
        return fig

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
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

    # Return annotations
    for _, row in wf_df.iterrows():
        fig.add_annotation(
            x=str(int(row["year"])),
            y=row["sharpe"],
            text=f"R:{row['ann_ret']*100:.1f}%",
            showarrow=False,
            yshift=-15 if row["sharpe"] > 0 else 15,
            font=dict(size=8, color="#666"),
        )

    # Table with all metrics
    table_cols = ["year", "sharpe", "ann_ret", "vol", "mdd", "hit_rate"]
    available_cols = [c for c in table_cols if c in wf_df.columns]
    header_text = [c.replace("_", " ").title() for c in available_cols]

    # Add extra columns
    extra_cols = [
        ("is_sharpe", "IS Sharpe"),
        ("dominant_mode", "Dominant Mode"),
        ("best_lookback", "Lookback"),
        ("best_threshold", "Threshold"),
        ("best_regime_filter", "Regime Flt"),
    ]
    for col, label in extra_cols:
        if col in wf_df.columns:
            available_cols.append(col)
            header_text.append(label)

    cell_values = []
    for col in available_cols:
        vals = wf_df[col].values
        if col == "year":
            cell_values.append([str(int(v)) for v in vals])
        elif col in ("ann_ret", "vol", "mdd", "hit_rate"):
            cell_values.append([f"{v:.2%}" for v in vals])
        elif col == "dominant_mode":
            cell_values.append([str(v) for v in vals])
        elif col == "best_regime_filter":
            cell_values.append([str(v) for v in vals])
        else:
            cell_values.append([f"{v:.3f}" if not isinstance(v, str)
                                else v for v in vals])

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
            text=(f"<b>Walk-Forward Validation (Regime System)</b>"
                  f"<br><span style='font-size:13px;color:#555'>"
                  f"Avg OOS Sharpe: {avg_sharpe:.3f}  |  "
                  f"Positive: {n_pos}/{len(wf_df)}"
                  f"</span>"),
            x=0.5, xanchor="center",
        ),
        height=850,
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
        ("Mode Equity",          "tab1_mode_equity.html"),
        ("Mode-Regime Matrix",   "tab2_mode_regime.html"),
        ("Adaptive System",      "tab3_adaptive.html"),
        ("Yearly Breakdown",     "tab4_yearly.html"),
        ("Risk Dashboard",       "tab5_risk.html"),
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
<title>Curve Regime System - Dashboard</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
    .header {{
        background: linear-gradient(135deg, #8e44ad, #3498db);
        color: white; padding: 20px 30px; text-align: center;
    }}
    .header h1 {{ margin: 0; font-size: 24px; }}
    .header p {{ margin: 5px 0 0; opacity: 0.8; font-size: 14px; }}
    .tab-bar {{
        display: flex; background: #2c3e50; padding: 0;
        border-bottom: 3px solid #8e44ad;
    }}
    .tab-btn {{
        padding: 12px 20px; cursor: pointer; color: #bdc3c7;
        font-size: 13px; font-weight: bold; border: none;
        background: transparent; transition: all 0.2s;
    }}
    .tab-btn:hover {{ background: #34495e; color: white; }}
    .tab-btn.active {{ background: #8e44ad; color: white; border-bottom: 3px solid #e74c3c; }}
    .tab-frame {{
        width: 100%; height: calc(100vh - 110px); border: none;
        background: white;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Curve Regime System</h1>
    <p>Multi-Mode Regime-Adaptive Strategy Selection | 6 Modes | Walk-Forward Validation</p>
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


def build_full_dashboard(mode_returns, mode_regime_matrix, adaptive_result,
                         wf_df, out_dir=None):
    """Build and save the full 6-tab regime system dashboard.

    Args:
        mode_returns: dict of {mode_name: pd.Series}
        mode_regime_matrix: DataFrame of Sharpe per (mode, regime)
        adaptive_result: adaptive_backtest() output
        wf_df: walk-forward results DataFrame
        out_dir: output directory

    Returns:
        Path to main dashboard HTML
    """
    if out_dir is None:
        out_dir = OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    print("  Building dashboard tabs ...")
    fig1 = build_mode_equity_tab(mode_returns, adaptive_result)
    fig2 = build_mode_regime_tab(mode_regime_matrix)
    fig3 = build_adaptive_system_tab(adaptive_result)
    fig4 = build_yearly_breakdown_tab(mode_returns, adaptive_result)
    fig5 = build_risk_dashboard_tab(adaptive_result)
    fig6 = build_walk_forward_tab(wf_df)

    tab_figs = [fig1, fig2, fig3, fig4, fig5, fig6]

    main_path = os.path.join(out_dir, "regime_system_dashboard.html")
    _build_tabbed_html(tab_figs, main_path)

    print(f"  Dashboard saved: {main_path}")
    return main_path


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  CURVE REGIME SYSTEM")
    print("  Multi-Mode Regime-Adaptive Strategy Selection")
    print("  6 modes | YC Regime Filter | Walk-Forward Validation")
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
    # STEP 2: BUILD RETURNS MATRIX + COMPUTE REGIMES
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/8] Building returns matrix + computing regimes ...")
    strategy_returns = {name: res["daily_ret"] for name, res in results.items()}
    returns_matrix = build_returns_matrix(strategy_returns)
    print(f"  Shape: {returns_matrix.shape} "
          f"({returns_matrix.index[0].date()} to {returns_matrix.index[-1].date()})")

    yc_regime = compute_yc_regime(yields)
    yc_regime_aligned = yc_regime.reindex(returns_matrix.index, method="ffill")

    # Show regime distribution
    regime_counts = yc_regime_aligned.value_counts()
    print(f"  YC Regime distribution:")
    for reg, count in regime_counts.items():
        pct = count / len(yc_regime_aligned) * 100
        print(f"    {reg:15s}  {count:5d} days ({pct:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: COMPUTE ALL MODE RETURNS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/8] Computing mode returns (6 modes) ...")
    t_modes = time.time()

    system = CurveRegimeSystem(
        returns_matrix=returns_matrix,
        yields=yields,
        rr_1m=rr_1m,
        move=move,
        vix=vix,
        strategy_pool=STRATEGY_POOL,
    )

    mode_returns = system.compute_all_mode_returns()

    print(f"  Mode returns computed in {time.time() - t_modes:.1f}s")
    print(f"\n  === MODE PERFORMANCE (FULL SAMPLE) ===")
    for mode_name, mode_ret in mode_returns.items():
        m = compute_metrics(mode_ret)
        desc = SIGNAL_MODES[mode_name]["description"]
        print(f"  {mode_name:12s} ({desc:25s})  "
              f"Sharpe={m.get('sharpe', 0):+.3f}  "
              f"Ret={m.get('ann_ret', 0):+.2%}  "
              f"Vol={m.get('vol', 0):.1%}  "
              f"MDD={m.get('mdd', 0):.1%}")

    # Save mode returns
    mode_ret_df = pd.DataFrame(mode_returns)
    mode_ret_df.to_csv(os.path.join(OUT_DIR, "mode_returns.csv"))
    print(f"  Mode returns saved to CSV")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: MODE-REGIME MATRIX
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/8] Computing mode-regime matrix ...")
    mode_regime_matrix = system.compute_mode_regime_matrix(yc_regime_aligned)

    print(f"\n  === MODE-REGIME SHARPE MATRIX ===")
    print(mode_regime_matrix.to_string(float_format="%.2f"))

    mode_regime_matrix.to_csv(os.path.join(OUT_DIR, "mode_regime_matrix.csv"))

    # Identify best modes per regime
    print(f"\n  Best mode per regime:")
    for col in mode_regime_matrix.columns:
        if col == "FULL":
            continue
        best_mode = mode_regime_matrix[col].idxmax()
        best_val = mode_regime_matrix[col].max()
        print(f"    {col:15s} -> {best_mode:12s} (Sharpe={best_val:+.2f})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: ADAPTIVE BACKTEST (IN-SAMPLE)
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5/8] Adaptive backtest (in-sample) ...")
    t_bt = time.time()
    is_result = system.adaptive_backtest()
    is_metrics = is_result["metrics"]

    print(f"  Backtest completed in {time.time() - t_bt:.1f}s")
    print(f"\n  === ADAPTIVE SYSTEM IS RESULTS ===")
    print(f"  Sharpe:   {is_metrics.get('sharpe', 0):.3f}")
    print(f"  Ann.Ret:  {is_metrics.get('ann_ret', 0):+.2%}")
    print(f"  Vol:      {is_metrics.get('vol', 0):.2%}")
    print(f"  MDD:      {is_metrics.get('mdd', 0):.2%}")
    print(f"  Hit Rate: {is_metrics.get('hit_rate', 0):.1%}")

    # Mode distribution
    mode_tl = is_result["mode_timeline"]
    mode_dist = mode_tl.value_counts()
    print(f"\n  Mode distribution:")
    for mode, count in mode_dist.items():
        pct = count / len(mode_tl) * 100
        print(f"    {mode:12s}  {count:5d} days ({pct:.1f}%)")

    avg_exposure = is_result["exposure"].mean()
    avg_turnover = is_result["turnover"].mean()
    print(f"  Avg exposure: {avg_exposure:.2f}")
    print(f"  Avg turnover: {avg_turnover:.3f}")

    # Save
    is_result["daily_ret"].to_csv(
        os.path.join(OUT_DIR, "adaptive_backtest.csv"), header=True)
    is_result["weights"].to_csv(
        os.path.join(OUT_DIR, "adaptive_weights_daily.csv"))
    is_result["mode_timeline"].to_csv(
        os.path.join(OUT_DIR, "adaptive_mode_timeline.csv"), header=True)

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
        if len(wf_df) > 0:
            best_row = wf_df.loc[wf_df["sharpe"].idxmax()]
            worst_row = wf_df.loc[wf_df["sharpe"].idxmin()]
            print(f"  Best year:  {best_row['year']:.0f} "
                  f"(Sharpe={best_row['sharpe']:.3f})")
            print(f"  Worst year: {worst_row['year']:.0f} "
                  f"(Sharpe={worst_row['sharpe']:.3f})")

        # Mode frequency in OOS
        if "dominant_mode" in wf_df.columns:
            mode_freq = wf_df["dominant_mode"].value_counts()
            print(f"\n  OOS dominant mode frequency:")
            for mode, cnt in mode_freq.items():
                print(f"    {mode:12s}  {cnt} years")

        print(f"  WF time: {time.time() - t_wf:.0f}s")

        wf_df.to_csv(os.path.join(OUT_DIR, "walk_forward_results.csv"),
                      index=False)
    else:
        print("  No walk-forward results (insufficient data)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: EXPLAIN DAILY DECISIONS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[7/8] Explaining daily decisions ...")
    explanation = system.explain(is_result)
    if len(explanation) > 0:
        explanation.to_csv(os.path.join(OUT_DIR, "daily_explanation.csv"),
                           index=False)
        print(f"  {len(explanation)} days explained")

        # Summary stats
        flat_days = (explanation["active_mode"] == "FLAT").sum()
        active_days = len(explanation) - flat_days
        print(f"  Active days: {active_days} ({active_days/len(explanation)*100:.1f}%)")
        print(f"  Flat days:   {flat_days} ({flat_days/len(explanation)*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    print("\n[8/8] Generating dashboard ...")
    dashboard_path = build_full_dashboard(
        mode_returns=mode_returns,
        mode_regime_matrix=mode_regime_matrix,
        adaptive_result=is_result,
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
