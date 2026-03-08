"""
interactive_dashboard.py - Plotly interactive dashboard for curve strategies.

Generates an HTML dashboard with tab navigation:
    Tab 1: All Strategy Equity Curves
    Tab 2: Meta-System Dashboard (equity + regime timeline + allocations)
    Tab 3: Regime Heatmap (Sharpe per strategy x regime)
    Tab 4: Walk-Forward OOS Sharpe by year

Output: output/curve_strategies/curve_strategies_dashboard.html
"""

import os
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Color palettes ───────────────────────────────────────────────────────
TYPE_COLORS = {
    "spread_steepener": "#1f77b4",
    "spread_flattener": "#ff7f0e",
    "butterfly":        "#2ca02c",
    "directional":      "#d62728",
    "barbell":          "#9467bd",
}

REGIME_COLORS = {
    "BULL_STEEP/CALM":    "#2ecc71",
    "BULL_STEEP/STRESS":  "#82e0aa",
    "BULL_FLAT/CALM":     "#27ae60",
    "BULL_FLAT/STRESS":   "#a9dfbf",
    "BEAR_STEEP/CALM":    "#e74c3c",
    "BEAR_STEEP/STRESS":  "#f1948a",
    "BEAR_FLAT/CALM":     "#c0392b",
    "BEAR_FLAT/STRESS":   "#f5b7b1",
    "UNDEFINED/CALM":     "#bdc3c7",
    "UNDEFINED/STRESS":   "#95a5a6",
}


def _get_type_color(strategy: dict) -> str:
    """Get color based on strategy type + direction."""
    stype = strategy["type"]
    direction = strategy.get("direction", "")
    if stype == "spread" and direction == "steepener":
        return TYPE_COLORS["spread_steepener"]
    elif stype == "spread" and direction == "flattener":
        return TYPE_COLORS["spread_flattener"]
    return TYPE_COLORS.get(stype, "#7f7f7f")


# ══════════════════════════════════════════════════════════════════════════
# TAB 1: ALL STRATEGY EQUITY CURVES
# ══════════════════════════════════════════════════════════════════════════

def build_equity_curves_tab(results: dict, strategies: list) -> go.Figure:
    """All 25 equity curves normalized to $1, colored by strategy type."""
    fig = go.Figure()

    for strat in strategies:
        name = strat["name"]
        if name not in results:
            continue
        equity = results[name]["equity"]
        color = _get_type_color(strat)
        width = 2.0 if strat["type"] == "directional" else 1.2

        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            name=strat["display"],
            line=dict(color=color, width=width),
            hovertemplate=f"{strat['display']}: $%{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        title="<b>All Strategy Equity Curves</b> (normalized to $1)",
        height=700,
        hovermode="x unified",
        yaxis_title="Equity ($)",
        xaxis_title="Date",
        template="plotly_white",
        legend=dict(font=dict(size=9), itemsizing="constant"),
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
# TAB 2: META-SYSTEM DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def build_meta_system_tab(meta_result: dict, meta_metrics: dict) -> go.Figure:
    """3-panel meta-system dashboard: equity, regime timeline, allocations."""
    equity = meta_result["equity"]
    regime = meta_result.get("regime_timeline", pd.Series(dtype=str))
    alloc_df = meta_result["allocations"]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.45, 0.15, 0.40],
        subplot_titles=(
            "Meta-System Equity",
            "Regime Timeline",
            "Strategy Allocation Weights",
        ),
    )

    # Panel 1: Equity
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Meta-System",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="Meta: $%{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Panel 2: Regime timeline as colored bars
    if len(regime) > 0:
        unique_regimes = sorted(regime.unique())
        for reg_name in unique_regimes:
            mask = regime == reg_name
            dates_in = regime.index[mask]
            if len(dates_in) == 0:
                continue
            y_vals = np.ones(len(dates_in))
            color = REGIME_COLORS.get(reg_name, "#bdc3c7")
            fig.add_trace(go.Scatter(
                x=dates_in, y=y_vals,
                mode="markers",
                name=reg_name,
                marker=dict(color=color, size=3, symbol="square"),
                showlegend=True,
                hovertemplate=f"{reg_name}<extra></extra>",
            ), row=2, col=1)

    # Panel 3: Allocation stacked area
    if len(alloc_df) > 0:
        # Only show strategies that ever have non-zero weight
        active = alloc_df.columns[alloc_df.max() > 0.01]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                  "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]
        for i, col in enumerate(active):
            fig.add_trace(go.Scatter(
                x=alloc_df.index,
                y=alloc_df[col].values,
                name=col,
                stackgroup="alloc",
                line=dict(width=0),
                fillcolor=colors[i % len(colors)],
                hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>",
            ), row=3, col=1)

    # Layout
    header = (
        f"Sharpe: {meta_metrics.get('sharpe', 0):.3f}  |  "
        f"Ann.Ret: {meta_metrics.get('ann_ret', 0)*100:.2f}%  |  "
        f"MDD: {meta_metrics.get('mdd', 0)*100:.1f}%"
    )
    fig.update_layout(
        title=dict(
            text=(f"<b>Meta-System Dashboard</b>"
                  f"<br><span style='font-size:13px;color:#555'>{header}</span>"),
            x=0.5, xanchor="center",
        ),
        height=900,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(font=dict(size=9)),
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Regime", row=2, col=1,
                     showticklabels=False, range=[0.5, 1.5])
    fig.update_yaxes(title_text="Weight", row=3, col=1, range=[0, 1.05])

    # Range slider on bottom
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04),
        rangeselector=dict(buttons=[
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        row=3, col=1,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# TAB 3: REGIME HEATMAP
# ══════════════════════════════════════════════════════════════════════════

def build_regime_heatmap_tab(regime_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of Sharpe per strategy x regime (colorscale RdYlGn)."""
    # Replace NaN with 0 for display
    z = regime_matrix.fillna(0).values
    strategies = regime_matrix.index.tolist()
    regimes = regime_matrix.columns.tolist()

    # Annotation text
    text = []
    for i in range(len(strategies)):
        row_text = []
        for j in range(len(regimes)):
            val = regime_matrix.iloc[i, j]
            if pd.isna(val):
                row_text.append("N/A")
            else:
                row_text.append(f"{val:.2f}")
        text.append(row_text)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=regimes,
        y=strategies,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=9),
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Sharpe"),
        hovertemplate="Strategy: %{y}<br>Regime: %{x}<br>Sharpe: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Regime-Strategy Performance Matrix</b> (Sharpe Ratio)",
        height=max(500, 25 * len(strategies) + 200),
        template="plotly_white",
        xaxis=dict(title="Regime", tickangle=45),
        yaxis=dict(title="Strategy", autorange="reversed"),
        margin=dict(l=150, b=120),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# TAB 4: WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════════

def build_walk_forward_tab(wf_df: pd.DataFrame) -> go.Figure:
    """Bar chart of OOS Sharpe by year for walk-forward validation."""
    if len(wf_df) == 0:
        fig = go.Figure()
        fig.update_layout(title="Walk-Forward: No data")
        return fig

    colors = ["#2ca02c" if s > 0 else "#d62728" for s in wf_df["sharpe"]]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=wf_df["year"].astype(str),
        y=wf_df["sharpe"],
        marker_color=colors,
        text=[f"{s:.2f}" for s in wf_df["sharpe"]],
        textposition="outside",
        hovertemplate=(
            "Year: %{x}<br>"
            "Sharpe: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    avg_sharpe = wf_df["sharpe"].mean()
    fig.add_hline(y=avg_sharpe, line=dict(color="#1f77b4", width=2, dash="dash"),
                  annotation_text=f"Avg: {avg_sharpe:.3f}")
    fig.add_hline(y=0, line=dict(color="black", width=0.5))

    # Add return info as secondary annotations
    for _, row in wf_df.iterrows():
        fig.add_annotation(
            x=str(int(row["year"])),
            y=row["sharpe"],
            text=f"R:{row['ann_ret']*100:.1f}%",
            showarrow=False,
            yshift=-15 if row["sharpe"] > 0 else 15,
            font=dict(size=8, color="#666"),
        )

    fig.update_layout(
        title=dict(
            text=(f"<b>Walk-Forward OOS Sharpe</b>"
                  f"<br><span style='font-size:13px;color:#555'>"
                  f"Average Sharpe: {avg_sharpe:.3f}  |  "
                  f"Positive: {(wf_df['sharpe'] > 0).sum()}/{len(wf_df)}"
                  f"</span>"),
            x=0.5, xanchor="center",
        ),
        height=500,
        template="plotly_white",
        xaxis_title="Year",
        yaxis_title="OOS Sharpe Ratio",
        showlegend=False,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# MASTER DASHBOARD BUILDER
# ══════════════════════════════════════════════════════════════════════════

def build_full_dashboard(results: dict, strategies: list,
                         meta_result: dict, meta_metrics: dict,
                         regime_matrix: pd.DataFrame,
                         wf_df: pd.DataFrame,
                         out_dir: str) -> str:
    """Build and save the full multi-tab dashboard + individual HTMLs.

    Args:
        results: {strategy_name: backtest_result}
        strategies: list of strategy dicts
        meta_result: MetaSystem.backtest() output
        meta_metrics: meta-system performance metrics
        regime_matrix: Sharpe per (strategy, regime)
        wf_df: walk-forward results DataFrame
        out_dir: output directory

    Returns:
        Path to main dashboard HTML
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build individual tabs
    fig1 = build_equity_curves_tab(results, strategies)
    fig2 = build_meta_system_tab(meta_result, meta_metrics)
    fig3 = build_regime_heatmap_tab(regime_matrix)
    fig4 = build_walk_forward_tab(wf_df)

    # Save individual HTMLs
    _write_html(fig1, os.path.join(out_dir, "tab1_equity_curves.html"))
    _write_html(fig2, os.path.join(out_dir, "tab2_meta_system.html"))
    _write_html(fig3, os.path.join(out_dir, "tab3_regime_heatmap.html"))
    _write_html(fig4, os.path.join(out_dir, "tab4_walk_forward.html"))

    # Build combined dashboard with tab navigation
    main_path = os.path.join(out_dir, "curve_strategies_dashboard.html")
    _build_tabbed_html(fig1, fig2, fig3, fig4, main_path)

    return main_path


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


def _build_tabbed_html(fig1, fig2, fig3, fig4, out_path):
    """Build a single HTML file with CSS/JS tab navigation using iframes.

    Uses iframes to isolate each Plotly figure, preventing JavaScript
    conflicts when multiple figures share the same page.
    """
    out_dir = os.path.dirname(out_path)

    # Tab file names (already saved by caller)
    tabs = [
        ("Equity Curves", "tab1_equity_curves.html"),
        ("Meta-System", "tab2_meta_system.html"),
        ("Regime Heatmap", "tab3_regime_heatmap.html"),
        ("Walk-Forward", "tab4_walk_forward.html"),
    ]

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
<title>Yield Curve Relative Value - Dashboard</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
    .header {{
        background: linear-gradient(135deg, #1a3a6a, #2980b9);
        color: white; padding: 20px 30px; text-align: center;
    }}
    .header h1 {{ margin: 0; font-size: 24px; }}
    .header p {{ margin: 5px 0 0; opacity: 0.8; font-size: 14px; }}
    .tab-bar {{
        display: flex; background: #2c3e50; padding: 0;
        border-bottom: 3px solid #1a3a6a;
    }}
    .tab-btn {{
        padding: 12px 24px; cursor: pointer; color: #bdc3c7;
        font-size: 14px; font-weight: bold; border: none;
        background: transparent; transition: all 0.2s;
    }}
    .tab-btn:hover {{ background: #34495e; color: white; }}
    .tab-btn.active {{ background: #1a3a6a; color: white; border-bottom: 3px solid #3498db; }}
    .tab-frame {{
        width: 100%; height: calc(100vh - 110px); border: none;
        background: white;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Yield Curve Relative Value Trading System</h1>
    <p>25 Strategies | Regime Analysis | Meta-System | Walk-Forward Validation</p>
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
