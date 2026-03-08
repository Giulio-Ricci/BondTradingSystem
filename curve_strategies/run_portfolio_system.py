"""
run_portfolio_system.py - Orchestrator for the Portfolio Optimization & Trading System.

Usage:
    python -m curve_strategies.run_portfolio_system
    python curve_strategies/run_portfolio_system.py

10-step pipeline:
    1.  Load data + backtest all 25 strategies
    2.  Build returns matrix + correlation matrix
    3.  Enumerate all C(25,k) combinations (k=2..5, ~68K)
    4.  Batch optimize top 100 combinations (3 Markowitz methods)
    5.  Rolling Markowitz on full universe
    6.  Compute all trading indicators
    7.  PortfolioTradingSystem backtest
    8.  Walk-forward validation (2012-2025)
    9.  Performance attribution
    10. Generate CSV + 6-tab dashboard
"""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BACKTEST_START, SAVE_DIR
from data_loader import load_all

from curve_strategies.strategy_definitions import STRATEGY_POOL
from curve_strategies.strategy_backtest import backtest_all, compute_metrics
from curve_strategies.portfolio_combinations import (
    build_returns_matrix, compute_correlation_matrix,
    enumerate_combinations, filter_top_combinations,
)
from curve_strategies.markowitz_optimizer import (
    rolling_markowitz, compute_efficient_frontier,
    batch_optimize_top_combinations, estimate_covariance,
)
from curve_strategies.trading_indicators import compute_all_indicators
from curve_strategies.portfolio_trading_system import PortfolioTradingSystem
from curve_strategies.ml_optimizer import MLPortfolioSystem

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Output directory ─────────────────────────────────────────────────────
OUT_DIR = os.path.join(SAVE_DIR, "portfolio_system")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD BUILDER (6 tabs)
# ══════════════════════════════════════════════════════════════════════════

def _write_html(fig, path):
    """Write a Plotly figure to standalone HTML."""
    fig.write_html(
        path, include_plotlyjs=True, full_html=True,
        config={"displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png", "height": 900, "width": 1600, "scale": 2}},
    )


def build_tab1_combination_frontier(combo_df, frontier_pts, out_dir):
    """Tab 1: Scatter vol vs ret for all combos + efficient frontier."""
    fig = go.Figure()

    # Color by k
    k_colors = {2: "#1f77b4", 3: "#ff7f0e", 4: "#2ca02c", 5: "#d62728"}

    for k in sorted(combo_df["k"].unique()):
        sub = combo_df[combo_df["k"] == k]
        fig.add_trace(go.Scatter(
            x=sub["vol"] * 100,
            y=sub["ann_ret"] * 100,
            mode="markers",
            name=f"k={k} ({len(sub):,})",
            marker=dict(color=k_colors.get(k, "#7f7f7f"), size=3, opacity=0.3),
            hovertemplate=(
                "Strategies: %{customdata}<br>"
                "Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<br>"
                "<extra></extra>"
            ),
            customdata=sub["strategies"].values,
        ))

    # Efficient frontier
    if len(frontier_pts) > 0:
        fig.add_trace(go.Scatter(
            x=frontier_pts[:, 0] * 100,
            y=frontier_pts[:, 1] * 100,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="black", width=3),
        ))

    fig.update_layout(
        title="<b>Portfolio Combinations</b> (EW, k=2..5) + Efficient Frontier",
        height=700,
        template="plotly_white",
        xaxis_title="Annualized Volatility (%)",
        yaxis_title="Annualized Return (%)",
        legend=dict(font=dict(size=10)),
    )

    _write_html(fig, os.path.join(out_dir, "tab1_combination_frontier.html"))
    return fig


def build_tab2_correlation(corr_matrix, out_dir):
    """Tab 2: Correlation heatmap + hierarchical clustering."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title="Corr"),
        texttemplate="%{z:.2f}",
        textfont=dict(size=7),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Corr: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title="<b>Strategy Correlation Matrix</b> (hierarchically clustered)",
        height=max(600, 25 * len(corr_matrix) + 200),
        template="plotly_white",
        xaxis=dict(tickangle=45, tickfont=dict(size=8)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=8)),
        margin=dict(l=120, b=120),
    )

    _write_html(fig, os.path.join(out_dir, "tab2_correlation.html"))
    return fig


def build_tab3_rolling_markowitz(markowitz_weights_df, markowitz_equity, out_dir):
    """Tab 3: Stacked area of Markowitz weights + equity curve."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.4, 0.6],
        subplot_titles=("Portfolio Equity", "Strategy Weights (Rolling Markowitz)"),
    )

    # Panel 1: equity
    fig.add_trace(go.Scatter(
        x=markowitz_equity.index, y=markowitz_equity.values,
        name="Rolling Markowitz",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="$%{y:.3f}<extra></extra>",
    ), row=1, col=1)

    # Panel 2: stacked area weights
    active_cols = markowitz_weights_df.columns[markowitz_weights_df.max() > 0.01]
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
        "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    for i, col in enumerate(active_cols):
        fig.add_trace(go.Scatter(
            x=markowitz_weights_df.index,
            y=markowitz_weights_df[col].values,
            name=col,
            stackgroup="weights",
            line=dict(width=0),
            fillcolor=colors[i % len(colors)],
            hovertemplate=f"{col}: %{{y:.1%}}<extra></extra>",
        ), row=2, col=1)

    fig.update_layout(
        title="<b>Rolling Markowitz Optimization</b>",
        height=900, template="plotly_white",
        legend=dict(font=dict(size=8)),
    )
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Weight", row=2, col=1, range=[0, 1.05])
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04),
        rangeselector=dict(buttons=[
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        row=2, col=1,
    )

    _write_html(fig, os.path.join(out_dir, "tab3_rolling_markowitz.html"))
    return fig


def build_tab4_indicators(indicators, returns_matrix, out_dir):
    """Tab 4: Multi-panel indicator dashboard."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.25, 0.25, 0.25, 0.25],
        subplot_titles=(
            "Composite Scores (top 5 strategies)",
            "Cross-Strategy Consensus",
            "Volatility Regime (vol_z)",
            "Rolling Sharpe 126d (top 5)",
        ),
    )

    scores = indicators["composite_scores"]
    consensus = indicators["consensus"]
    vol_z = indicators["vol"]["vol_z"]
    sharpe_126 = indicators["sharpe"]["sharpe_126"]

    # Top 5 by average score
    avg_scores = scores.mean()
    top5 = avg_scores.nlargest(5).index.tolist()

    colors5 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Panel 1: composite scores for top 5
    for i, name in enumerate(top5):
        fig.add_trace(go.Scatter(
            x=scores.index, y=scores[name].values,
            name=f"Score: {name}",
            line=dict(color=colors5[i], width=1.2),
        ), row=1, col=1)

    # Panel 2: consensus (average across all strategies by type)
    from curve_strategies.trading_indicators import STRATEGY_TYPES
    groups = {}
    for col in consensus.columns:
        g = STRATEGY_TYPES.get(col, "other")
        if g not in groups:
            groups[g] = []
        groups[g].append(col)

    group_colors = {"steepener": "#1f77b4", "flattener": "#ff7f0e",
                    "butterfly": "#2ca02c", "directional": "#d62728",
                    "barbell": "#9467bd"}
    for g, members in groups.items():
        avg_consensus = consensus[members].mean(axis=1)
        fig.add_trace(go.Scatter(
            x=avg_consensus.index, y=avg_consensus.values,
            name=f"Cons: {g}",
            line=dict(color=group_colors.get(g, "#7f7f7f"), width=1.2),
        ), row=2, col=1)

    # Panel 3: vol_z (average)
    avg_vol_z = vol_z.mean(axis=1)
    fig.add_trace(go.Scatter(
        x=avg_vol_z.index, y=avg_vol_z.values,
        name="Avg vol_z",
        line=dict(color="#1f77b4", width=1.5),
    ), row=3, col=1)
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-1.5, line_dash="dash", line_color="green", row=3, col=1)

    # Panel 4: rolling Sharpe 126 for top 5
    for i, name in enumerate(top5):
        fig.add_trace(go.Scatter(
            x=sharpe_126.index, y=sharpe_126[name].values,
            name=f"Sh126: {name}",
            line=dict(color=colors5[i], width=1),
        ), row=4, col=1)

    fig.update_layout(
        title="<b>Trading Indicators</b>",
        height=1000, template="plotly_white",
        legend=dict(font=dict(size=8)),
        showlegend=True,
    )
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.04),
        row=4, col=1,
    )

    _write_html(fig, os.path.join(out_dir, "tab4_indicators.html"))
    return fig


def build_tab5_system_performance(system_result, ew_equity, bh_tlt_equity,
                                  best_single_equity, out_dir,
                                  basic_result=None):
    """Tab 5: System equity vs benchmarks."""
    fig = go.Figure()

    # ML System (primary)
    eq = system_result["equity"]
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        name="ML Portfolio System",
        line=dict(color="#1f77b4", width=2.5),
        hovertemplate="ML System: $%{y:.3f}<extra></extra>",
    ))

    # Basic Markowitz (comparison)
    if basic_result is not None:
        eq_basic = basic_result["equity"]
        fig.add_trace(go.Scatter(
            x=eq_basic.index, y=eq_basic.values,
            name="Basic Markowitz",
            line=dict(color="#9467bd", width=1.5, dash="dash"),
            hovertemplate="Basic: $%{y:.3f}<extra></extra>",
        ))

    # EW benchmark
    fig.add_trace(go.Scatter(
        x=ew_equity.index, y=ew_equity.values,
        name="Equal Weight (all 25)",
        line=dict(color="#ff7f0e", width=1.5, dash="dash"),
        hovertemplate="EW: $%{y:.3f}<extra></extra>",
    ))

    # B&H TLT
    if bh_tlt_equity is not None:
        fig.add_trace(go.Scatter(
            x=bh_tlt_equity.index, y=bh_tlt_equity.values,
            name="B&H TLT",
            line=dict(color="#d62728", width=1.5, dash="dot"),
            hovertemplate="TLT: $%{y:.3f}<extra></extra>",
        ))

    # Best single strategy
    if best_single_equity is not None:
        fig.add_trace(go.Scatter(
            x=best_single_equity.index, y=best_single_equity.values,
            name="Best Single Strategy",
            line=dict(color="#2ca02c", width=1.5, dash="dashdot"),
            hovertemplate="Best: $%{y:.3f}<extra></extra>",
        ))

    m = system_result["metrics"]
    header = (
        f"ML Sharpe: {m.get('sharpe', 0):.3f}  |  "
        f"Ann.Ret: {m.get('ann_ret', 0)*100:.2f}%  |  "
        f"MDD: {m.get('mdd', 0)*100:.1f}%  |  "
        f"Calmar: {m.get('calmar', 0):.3f}"
    )

    fig.update_layout(
        title=dict(
            text=(f"<b>Portfolio System Performance</b>"
                  f"<br><span style='font-size:13px;color:#555'>{header}</span>"),
            x=0.5, xanchor="center",
        ),
        height=700,
        template="plotly_white",
        yaxis_title="Equity ($1 start)",
        xaxis_title="Date",
        hovermode="x unified",
        legend=dict(font=dict(size=10)),
        xaxis=dict(
            rangeselector=dict(buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ]),
            rangeslider=dict(visible=True, thickness=0.04),
        ),
    )

    _write_html(fig, os.path.join(out_dir, "tab5_system_performance.html"))
    return fig


def build_tab6_walkforward_attribution(wf_df, attribution_df, out_dir):
    """Tab 6: Walk-forward OOS Sharpe bars + performance attribution."""
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.12,
        row_heights=[0.5, 0.5],
        subplot_titles=(
            "Walk-Forward OOS Sharpe by Year",
            "Performance Attribution (Annual)",
        ),
    )

    # Panel 1: WF Sharpe bars
    if len(wf_df) > 0:
        colors = ["#2ca02c" if s > 0 else "#d62728" for s in wf_df["sharpe"]]
        fig.add_trace(go.Bar(
            x=wf_df["year"].astype(str),
            y=wf_df["sharpe"],
            marker_color=colors,
            text=[f"{s:.2f}" for s in wf_df["sharpe"]],
            textposition="outside",
            name="OOS Sharpe",
            hovertemplate="Year: %{x}<br>Sharpe: %{y:.3f}<extra></extra>",
        ), row=1, col=1)

        avg_sharpe = wf_df["sharpe"].mean()
        fig.add_hline(y=avg_sharpe, line=dict(color="#1f77b4", width=2, dash="dash"),
                      annotation_text=f"Avg: {avg_sharpe:.3f}", row=1, col=1)
        fig.add_hline(y=0, line=dict(color="black", width=0.5), row=1, col=1)

    # Panel 2: Attribution stacked bars
    if len(attribution_df) > 0:
        years_str = attribution_df["year"].astype(str)

        fig.add_trace(go.Bar(
            x=years_str, y=attribution_df["markowitz_contrib"] * 100,
            name="Markowitz", marker_color="#1f77b4",
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=years_str, y=attribution_df["tilt_contrib"] * 100,
            name="Indicator Tilt", marker_color="#ff7f0e",
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=years_str, y=attribution_df["tcost_drag"] * 100,
            name="TCost Drag", marker_color="#d62728",
        ), row=2, col=1)

    fig.update_layout(
        title="<b>Walk-Forward Validation & Performance Attribution</b>",
        height=900, template="plotly_white",
        barmode="group",
        legend=dict(font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Return Contribution (%)", row=2, col=1)

    _write_html(fig, os.path.join(out_dir, "tab6_walkforward_attribution.html"))
    return fig


def build_portfolio_dashboard(out_dir):
    """Build the 6-tab master dashboard HTML with iframe navigation."""
    tabs = [
        ("Combination Frontier", "tab1_combination_frontier.html"),
        ("Correlation & Clustering", "tab2_correlation.html"),
        ("Rolling Markowitz", "tab3_rolling_markowitz.html"),
        ("Trading Indicators", "tab4_indicators.html"),
        ("System Performance", "tab5_system_performance.html"),
        ("Walk-Forward + Attribution", "tab6_walkforward_attribution.html"),
    ]

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
<title>Portfolio Optimization & Trading System</title>
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
        padding: 12px 20px; cursor: pointer; color: #bdc3c7;
        font-size: 13px; font-weight: bold; border: none;
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
    <h1>Portfolio Optimization & Trading System</h1>
    <p>25 Curve Strategies | Markowitz Optimization | Indicator-Based Tilt | Walk-Forward Validation</p>
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

    main_path = os.path.join(out_dir, "portfolio_system_dashboard.html")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write(html)
    return main_path


# ══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 72)
    print("  PORTFOLIO OPTIMIZATION & TRADING SYSTEM")
    print("  25 Curve Strategies | Markowitz | Indicators | Walk-Forward")
    print("=" * 72)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: LOAD DATA + BACKTEST ALL STRATEGIES
    # ══════════════════════════════════════════════════════════════════════
    print("\n[1/10] Loading data and backtesting all strategies ...")
    D = load_all()
    etf_ret = D["etf_ret"]

    summary_df, results = backtest_all(STRATEGY_POOL, etf_ret, BACKTEST_START, tcost_bps=5)
    strategy_returns = {name: results[name]["daily_ret"] for name in results}

    print(f"  {len(STRATEGY_POOL)} strategies backtested")
    print(f"  Best:  {summary_df.iloc[0]['display']} (Sharpe={summary_df.iloc[0]['sharpe']:.3f})")
    print(f"  Worst: {summary_df.iloc[-1]['display']} (Sharpe={summary_df.iloc[-1]['sharpe']:.3f})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: BUILD RETURNS MATRIX + CORRELATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n[2/10] Building returns matrix and correlation matrix ...")
    returns_matrix = build_returns_matrix(strategy_returns)
    print(f"  Returns matrix: {returns_matrix.shape}")

    corr_matrix = compute_correlation_matrix(returns_matrix)
    corr_matrix.to_csv(os.path.join(OUT_DIR, "correlation_matrix.csv"))
    print(f"  -> correlation_matrix.csv ({corr_matrix.shape})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: ENUMERATE COMBINATIONS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[3/10] Enumerating portfolio combinations (k=2..5) ...")
    combo_df = enumerate_combinations(returns_matrix, max_k=5, min_k=2)
    combo_df.to_csv(os.path.join(OUT_DIR, "combination_results.csv"), index=False)
    print(f"  -> combination_results.csv ({len(combo_df):,} rows)")

    top200 = filter_top_combinations(combo_df, top_n=200, metric="sharpe")
    top200.to_csv(os.path.join(OUT_DIR, "combination_top200.csv"), index=False)
    print(f"  -> combination_top200.csv ({len(top200)} rows)")
    print(f"  Best EW combo: Sharpe={top200.iloc[0]['sharpe']:.4f} "
          f"({top200.iloc[0]['strategies']})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: BATCH OPTIMIZE TOP COMBINATIONS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[4/10] Batch optimizing top 100 combinations (3 Markowitz methods) ...")
    markowitz_df = batch_optimize_top_combinations(
        returns_matrix, top200, top_n=100,
        methods=["min_variance", "max_sharpe", "risk_parity"],
    )
    markowitz_df.to_csv(os.path.join(OUT_DIR, "markowitz_optimized.csv"), index=False)
    print(f"  -> markowitz_optimized.csv ({len(markowitz_df)} rows)")

    # Best optimized combo
    best_opt = markowitz_df.sort_values("sharpe", ascending=False).iloc[0]
    print(f"  Best optimized: {best_opt['method']} Sharpe={best_opt['sharpe']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: ROLLING MARKOWITZ ON FULL UNIVERSE
    # ══════════════════════════════════════════════════════════════════════
    print("\n[5/10] Running rolling Markowitz on full 25-strategy universe ...")
    markowitz_weights_df, markowitz_port_ret = rolling_markowitz(
        returns_matrix, method="max_sharpe",
        min_window=504, rebalance_freq=21, expanding=True,
    )

    markowitz_weights_df.to_csv(os.path.join(OUT_DIR, "markowitz_rolling_weights.csv"))
    markowitz_metrics = compute_metrics(markowitz_port_ret)
    print(f"  Rolling Markowitz: Sharpe={markowitz_metrics['sharpe']:.4f}, "
          f"Ret={markowitz_metrics['ann_ret']:+.2%}, "
          f"MDD={markowitz_metrics['mdd']:.2%}")
    print(f"  -> markowitz_rolling_weights.csv")

    # Efficient frontier
    mu_full = returns_matrix.values.mean(axis=0)
    cov_full = estimate_covariance(returns_matrix, method="ledoit_wolf")
    frontier_pts = compute_efficient_frontier(mu_full, cov_full, n_points=50)
    if len(frontier_pts) > 0:
        frontier_df = pd.DataFrame(frontier_pts, columns=["vol", "ret"])
        frontier_df.to_csv(os.path.join(OUT_DIR, "efficient_frontier.csv"), index=False)
        print(f"  -> efficient_frontier.csv ({len(frontier_df)} points)")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: COMPUTE ALL TRADING INDICATORS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[6/10] Computing trading indicators ...")
    indicators = compute_all_indicators(returns_matrix, STRATEGY_POOL)

    # Save daily indicator values (composite scores)
    indicators["composite_scores"].to_csv(
        os.path.join(OUT_DIR, "indicator_daily.csv"))
    print(f"  -> indicator_daily.csv ({indicators['composite_scores'].shape})")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: ML-ENHANCED PORTFOLIO SYSTEM BACKTEST
    # ══════════════════════════════════════════════════════════════════════
    print("\n[7/11] Running ML-enhanced portfolio system backtest ...")
    yields = D["yields"]
    vix = D.get("vix")
    move = D.get("move")

    ml_system = MLPortfolioSystem(
        returns_matrix=returns_matrix,
        strategy_pool=STRATEGY_POOL,
        yields=yields,
        vix=vix,
        move=move,
        markowitz_method="max_sharpe",
        min_train_window=504,
        retrain_freq=63,
        rebalance_freq=21,
        target_vol=0.15,
        max_leverage=2.0,
        dd_threshold=-0.10,
        dd_reduction=0.5,
        top_n_strategies=10,
        allow_short=False,
        tcost_bps=5,
        ml_weight=0.5,
        n_estimators=100,
    )

    ml_result = ml_system.backtest()
    m_ml = ml_result["metrics"]
    print(f"  ML System: Sharpe={m_ml['sharpe']:.4f}, Ret={m_ml['ann_ret']:+.2%}, "
          f"MDD={m_ml['mdd']:.2%}, Calmar={m_ml['calmar']:.4f}")

    # Save ML system backtest
    ml_bt = pd.DataFrame({
        "date": ml_result["daily_ret"].index,
        "daily_ret": ml_result["daily_ret"].values,
        "equity": ml_result["equity"].values,
        "turnover": ml_result["turnover"].values,
        "exposure": ml_result["exposure"].values,
    })
    ml_bt.to_csv(os.path.join(OUT_DIR, "ml_system_backtest.csv"), index=False)
    print(f"  -> ml_system_backtest.csv")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 8: ALSO RUN BASIC SYSTEM FOR COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print("\n[8/11] Running basic Markowitz system (comparison) ...")
    system = PortfolioTradingSystem(
        returns_matrix=returns_matrix,
        strategy_pool=STRATEGY_POOL,
        markowitz_method="max_sharpe",
        indicator_weights=None,
        min_estimation_window=504,
        rebalance_freq=21,
        tilt_strength=0.0,  # pure Markowitz (WF showed tilt=0 is best)
        top_n_strategies=10,
        tcost_bps=5,
    )
    basic_result = system.backtest()
    m_basic = basic_result["metrics"]
    print(f"  Basic:  Sharpe={m_basic['sharpe']:.4f}, Ret={m_basic['ann_ret']:+.2%}, "
          f"MDD={m_basic['mdd']:.2%}")

    system_bt = pd.DataFrame({
        "date": basic_result["daily_ret"].index,
        "daily_ret": basic_result["daily_ret"].values,
        "equity": basic_result["equity"].values,
        "turnover": basic_result["turnover"].values,
    })
    system_bt.to_csv(os.path.join(OUT_DIR, "system_backtest.csv"), index=False)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 9: ML WALK-FORWARD VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    print("\n[9/11] ML walk-forward validation (2012-2025) ...")
    wf_df = ml_system.walk_forward(start_year=2012, end_year=2025)
    wf_df.to_csv(os.path.join(OUT_DIR, "walk_forward_results.csv"), index=False)

    avg_oos = wf_df["sharpe"].mean()
    pos_years = (wf_df["sharpe"] > 0).sum()
    print(f"\n  ML Walk-Forward Summary:")
    print(f"    Avg OOS Sharpe: {avg_oos:.3f}")
    print(f"    Positive years: {pos_years}/{len(wf_df)}")
    for _, r in wf_df.iterrows():
        marker = "+" if r["sharpe"] > 0 else " "
        print(f"    {marker} {r['year']:.0f}: Sharpe={r['sharpe']:>+.3f}  "
              f"Ret={r['ann_ret']:>+.2%}  MDD={r['mdd']:.2%}")
    print(f"  -> walk_forward_results.csv")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 10: FEATURE IMPORTANCE
    # ══════════════════════════════════════════════════════════════════════
    print("\n[10/11] Computing feature importance ...")
    feat_imp = ml_system.get_feature_importance()
    if len(feat_imp) > 0:
        feat_imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"))
        print(f"  -> feature_importance.csv")
        print(f"  Top 10 features:")
        for i, (fname, row) in enumerate(feat_imp.head(10).iterrows()):
            print(f"    {i+1}. {fname}: {row['mean']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP 11: GENERATE DASHBOARD
    # ══════════════════════════════════════════════════════════════════════
    print("\n[11/11] Generating 6-tab interactive dashboard ...")

    # Benchmarks for Tab 5
    ew_ret = returns_matrix.mean(axis=1)
    ew_equity = (1 + ew_ret).cumprod()
    ew_equity.name = "EW_all_25"

    bh_tlt_equity = None
    if "bh_tlt" in results:
        bh_tlt_equity = results["bh_tlt"]["equity"]

    best_single_name = summary_df.iloc[0]["name"]
    best_single_equity = results[best_single_name]["equity"] if best_single_name in results else None

    # Markowitz equity for tab 3
    markowitz_equity = (1 + markowitz_port_ret).cumprod()

    # Build all 6 tabs
    print("  Building Tab 1: Combination Frontier ...")
    build_tab1_combination_frontier(combo_df, frontier_pts, OUT_DIR)

    print("  Building Tab 2: Correlation & Clustering ...")
    build_tab2_correlation(corr_matrix, OUT_DIR)

    print("  Building Tab 3: Rolling Markowitz ...")
    build_tab3_rolling_markowitz(markowitz_weights_df, markowitz_equity, OUT_DIR)

    print("  Building Tab 4: Trading Indicators ...")
    build_tab4_indicators(indicators, returns_matrix, OUT_DIR)

    print("  Building Tab 5: System Performance (ML vs Basic vs Benchmarks) ...")
    # Update Tab 5 to show ML system as primary
    build_tab5_system_performance(ml_result, ew_equity, bh_tlt_equity,
                                  best_single_equity, OUT_DIR,
                                  basic_result=basic_result)

    print("  Building Tab 6: Walk-Forward ...")
    build_tab6_walkforward_attribution(wf_df, pd.DataFrame(), OUT_DIR)

    print("  Building master dashboard ...")
    dashboard_path = build_portfolio_dashboard(OUT_DIR)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n{'=' * 72}")
    print(f"  COMPLETE  ({elapsed:.1f}s)")
    print(f"{'=' * 72}")

    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Dashboard: {dashboard_path}")

    print(f"\n  KEY RESULTS:")
    print(f"    Combinations enumerated:   {len(combo_df):,}")
    print(f"    Best EW combo Sharpe:      {top200.iloc[0]['sharpe']:.4f}")
    print(f"    Rolling Markowitz Sharpe:   {markowitz_metrics['sharpe']:.4f}")
    print(f"    Basic System Sharpe:       {m_basic['sharpe']:.4f}")
    print(f"    ML System Sharpe:          {m_ml['sharpe']:.4f}")
    print(f"    ML WF Avg OOS Sharpe:      {avg_oos:.4f}")
    print(f"    ML WF Positive years:      {pos_years}/{len(wf_df)}")

    print(f"\n{'=' * 72}")

    return {
        "returns_matrix": returns_matrix,
        "combo_df": combo_df,
        "ml_result": ml_result,
        "basic_result": basic_result,
        "wf_df": wf_df,
    }


if __name__ == "__main__":
    main()
