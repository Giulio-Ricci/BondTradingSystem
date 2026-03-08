"""
build_regime_charts.py - Eight regime analysis charts in a tabbed dashboard.

Chart 1: Mode equity curves (all 6 + adaptive) - clean, readable
Chart 2: Monthly best-mode selector - cumulative equity
Chart 3: Regime-colored bars per slope segment + TLT overlay
Chart 4: Strategy x Regime Sharpe heatmap (YC + HMM + Full)
Chart 5: Risk-Return scatterplot per YC regime (2x2)
Chart 6: Strategy Sharpe by 10Y rate level tercile
Chart 7: Correlation clustermap with hierarchical clustering
Chart 8: Rolling best/worst strategy timeline (quarterly)

Usage:
    python -m curve_strategies.build_regime_charts
"""

import os
import sys
import time
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import BACKTEST_START, SAVE_DIR
from data_loader import load_all
from curve_strategies.strategy_definitions import STRATEGY_POOL
from curve_strategies.strategy_backtest import backtest_all, compute_metrics
from curve_strategies.portfolio_combinations import build_returns_matrix
from curve_strategies.curve_regime_system import CurveRegimeSystem, SIGNAL_MODES
from curve_strategies.regime_analysis import compute_yc_regime
from regime import compute_regime
from scipy.cluster.hierarchy import linkage, leaves_list

OUT_DIR = os.path.join(SAVE_DIR, "regime_system")
os.makedirs(OUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# REGIME PER SEGMENT
# ══════════════════════════════════════════════════════════════════════════

SEGMENT_DEFS = {
    "2s5s":  ("US_2Y", "US_5Y"),
    "2s10s": ("US_2Y", "US_10Y"),
    "5s10s": ("US_5Y", "US_10Y"),
    "10s30s": ("US_10Y", "US_30Y"),
    "2s30s": ("US_2Y", "US_30Y"),
}

REGIME_COLORS = {
    "BULL_STEEP": "#00e676",   # bright green
    "BEAR_STEEP": "#ff5252",   # red
    "BULL_FLAT":  "#ff9800",   # orange
    "BEAR_FLAT":  "#40c4ff",   # cyan
    "UNDEFINED":  "#9e9e9e",   # grey
}


def compute_segment_regime(yields, short_col, long_col, window=21):
    """Compute regime for a specific yield curve segment."""
    level = (yields[short_col] + yields[long_col]) / 2
    spread = yields[long_col] - yields[short_col]

    dl = level.diff(window)
    ds = spread.diff(window)

    r = pd.Series("UNDEFINED", index=yields.index)
    r[(dl < 0) & (ds > 0)] = "BULL_STEEP"
    r[(dl > 0) & (ds > 0)] = "BEAR_STEEP"
    r[(dl < 0) & (ds < 0)] = "BULL_FLAT"
    r[(dl > 0) & (ds < 0)] = "BEAR_FLAT"
    return r


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY GROUPING AND COLORS
# ══════════════════════════════════════════════════════════════════════════

STRATEGY_GROUPS = {
    "Steepeners": {
        "prefix": "steep_",
        "palette": ["#1565c0", "#1976d2", "#1e88e5", "#2196f3",
                     "#42a5f5", "#64b5f6", "#90caf9", "#bbdefb"],
    },
    "Flatteners": {
        "prefix": "flat_",
        "palette": ["#e65100", "#f57c00", "#ff9800", "#ffb74d"],
    },
    "Butterflies": {
        "prefix": "bfly_",
        "palette": ["#1b5e20", "#2e7d32", "#388e3c",
                     "#43a047", "#66bb6a", "#81c784"],
    },
    "Directional": {
        "prefix": "bh_",
        "palette": ["#b71c1c", "#c62828", "#d32f2f", "#e53935"],
    },
    "Barbell": {
        "prefix": "barbell_",
        "palette": ["#6a1b9a", "#7b1fa2", "#8e24aa"],
    },
}


def _classify_strategy(name):
    """Return group name for a strategy."""
    for grp, cfg in STRATEGY_GROUPS.items():
        if name.startswith(cfg["prefix"]):
            return grp
    return "Other"


GROUP_COLORS = {
    "Steepeners": "#1976d2",
    "Flatteners": "#ff9800",
    "Butterflies": "#388e3c",
    "Directional": "#d32f2f",
    "Barbell": "#8e24aa",
    "Other": "#666666",
}


def _display_name(s):
    """Short display name for strategy."""
    d = s.replace("steep_", "S: ").replace("flat_", "F: ")
    d = d.replace("bfly_", "B: ").replace("bh_", "D: ")
    d = d.replace("barbell_", "BB: ").replace("_", " ")
    return d


def _order_strategies(strat_names):
    """Return (ordered_strats, group_labels) sorted by strategy group."""
    group_order = ["Steepeners", "Flatteners", "Butterflies",
                   "Directional", "Barbell"]
    ordered = []
    labels = []
    for grp in group_order:
        prefix = STRATEGY_GROUPS[grp]["prefix"]
        members = [s for s in strat_names if s.startswith(prefix)]
        ordered.extend(members)
        labels.extend([grp] * len(members))
    for s in strat_names:
        if s not in ordered:
            ordered.append(s)
            labels.append("Other")
    return ordered, labels


# ══════════════════════════════════════════════════════════════════════════
# CHART 1: ALL 25 STRATEGIES - SMALL MULTIPLES BY GROUP
# ══════════════════════════════════════════════════════════════════════════

def build_chart1_strategy_equity(returns_matrix):
    """Small-multiples: one panel per strategy group, clean and readable."""

    strat_names = returns_matrix.columns.tolist()

    # Group strategies
    groups = {}
    for sn in strat_names:
        grp = _classify_strategy(sn)
        groups.setdefault(grp, []).append(sn)

    group_order = ["Steepeners", "Flatteners", "Butterflies",
                   "Directional", "Barbell"]
    group_order = [g for g in group_order if g in groups]

    n_groups = len(group_order)

    fig = make_subplots(
        rows=n_groups, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=[0.22, 0.16, 0.22, 0.20, 0.20][:n_groups],
        subplot_titles=[f"{g}  ({len(groups[g])} strategies)"
                        for g in group_order],
    )

    for gi, grp in enumerate(group_order):
        row = gi + 1
        strats = groups[grp]
        palette = STRATEGY_GROUPS.get(grp, {}).get(
            "palette", ["#666"] * len(strats))

        for si, sn in enumerate(strats):
            ret = returns_matrix[sn]
            eq = (1 + ret).cumprod()
            m = compute_metrics(ret)
            color = palette[si % len(palette)]

            # Display name: clean up
            display = sn.replace("steep_", "").replace("flat_", "")
            display = display.replace("bfly_", "").replace("bh_", "")
            display = display.replace("barbell_", "")

            label = f"{display} (S={m.get('sharpe', 0):+.2f})"

            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                name=label,
                line=dict(color=color, width=1.8),
                legendgroup=grp,
                hovertemplate=f"{sn}: $%{{y:.3f}}<extra></extra>",
            ), row=row, col=1)

        # $1 reference line
        fig.add_hline(y=1.0, line=dict(color="rgba(0,0,0,0.2)", width=0.8,
                                        dash="dot"), row=row, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=row, col=1)

    fig.update_layout(
        height=350 * n_groups,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            font=dict(size=10, family="Consolas"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ddd", borderwidth=1,
            groupclick="togglegroup",
        ),
        title=dict(
            text=("<b>All 25 Strategy Equity Curves</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  "Grouped by type: Steepeners | Flatteners | "
                  "Butterflies | Directional | Barbell</span>"),
            x=0.5, xanchor="center",
        ),
        margin=dict(t=80, b=50),
    )

    # Range selector on bottom
    bottom_xaxis = f"xaxis{n_groups}" if n_groups > 1 else "xaxis"
    fig.layout[bottom_xaxis].update(
        rangeselector=dict(buttons=[
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(count=5, label="5Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ]),
        rangeslider=dict(visible=True, thickness=0.03),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# CHART 2: MONTHLY RETURNS HEATMAP + BEST STRATEGY ROTATION
# ══════════════════════════════════════════════════════════════════════════

def _ret_to_color(v, vmin=-8, vmax=8):
    """Map return % to RdYlGn-like color string."""
    import math
    if v != v or math.isnan(v):  # NaN
        return "#2a2a3e"
    t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    # Red → Yellow → Green
    if t < 0.5:
        r = 220
        g = int(60 + 340 * t)
        b = int(50 + 40 * t)
    else:
        r = int(220 - 190 * (t - 0.5) * 2)
        g = int(180 + 40 * (t - 0.5) * 2)
        b = int(70 - 30 * (t - 0.5) * 2)
    return f"rgb({r},{g},{b})"


def _text_color(v):
    """White text for extreme values, dark for middle."""
    import math
    if v != v or math.isnan(v):
        return "#666"
    if abs(v) > 5:
        return "#fff"
    return "#1a1a1a"


def build_chart2_monthly_best(returns_matrix):
    """Pure HTML table: monthly returns per strategy.

    - Sticky Y-axis (strategy names stay visible on scroll)
    - Best performer each month highlighted in magenta
    - RdYlGn cell coloring, values in every cell
    """

    strat_names = returns_matrix.columns.tolist()

    # Monthly returns
    monthly_matrix = returns_matrix.resample("ME").apply(
        lambda x: (1 + x).prod() - 1)

    # ── Order strategies by group ────────────────────────────────
    group_order = ["Steepeners", "Flatteners", "Butterflies",
                   "Directional", "Barbell"]
    ordered_strats = []
    group_labels = []
    for grp in group_order:
        prefix = STRATEGY_GROUPS[grp]["prefix"]
        members = [s for s in strat_names if s.startswith(prefix)]
        ordered_strats.extend(members)
        group_labels.extend([grp] * len(members))
    for s in strat_names:
        if s not in ordered_strats:
            ordered_strats.append(s)
            group_labels.append("Other")

    monthly_ordered = monthly_matrix[ordered_strats]

    # ── Display names ────────────────────────────────────────────
    display_names = []
    for s in ordered_strats:
        d = s.replace("steep_", "Steep ").replace("flat_", "Flat ")
        d = d.replace("bfly_", "Bfly ").replace("bh_", "B&H ")
        d = d.replace("barbell_", "Barbell ").replace("_", " ")
        display_names.append(d)

    # ── Prepare data ─────────────────────────────────────────────
    z_vals = monthly_ordered.values.T * 100  # rows=strats, cols=months
    n_strats = len(ordered_strats)
    n_months = len(monthly_ordered)

    month_labels = [d.strftime("%b-%y") for d in monthly_ordered.index]

    # Best strategy per month (column-wise argmax)
    best_idx_per_month = z_vals.argmax(axis=0)  # shape (n_months,)

    # Group boundaries
    group_boundaries = set()
    prev_grp = group_labels[0]
    for i, grp in enumerate(group_labels):
        if grp != prev_grp:
            group_boundaries.add(i)
            prev_grp = grp

    # ── Build HTML table ────────────────────────────────────────
    cell_w = 54
    label_w = 160

    rows_html = []
    # Header row
    hdr = (f'<tr><th class="sticky-col hdr-corner"'
           f' style="width:{label_w}px;">Strategy</th>')
    for ml in month_labels:
        hdr += (f'<th class="hdr-cell" style="width:{cell_w}px;">'
                f'{ml}</th>')
    hdr += '</tr>'
    rows_html.append(hdr)

    # Data rows
    for i in range(n_strats):
        is_boundary = i in group_boundaries
        border_top = "border-top:3px solid #e91e63;" if is_boundary else ""
        grp_color = STRATEGY_GROUPS.get(
            group_labels[i], {}).get("color", "#888")

        row = (f'<tr><td class="sticky-col row-label"'
               f' style="{border_top}'
               f' border-left:4px solid {grp_color};">'
               f'{display_names[i]}</td>')

        for j in range(n_months):
            v = z_vals[i, j]
            is_best = (best_idx_per_month[j] == i)
            bg = _ret_to_color(v)
            tc = _text_color(v)
            txt = f"{v:+.1f}" if abs(v) >= 0.05 else ""

            if is_best:
                cell_style = (
                    f"background:{bg}; color:#fff;"
                    f" outline:2px solid #ff00ff;"
                    f" outline-offset:-2px;"
                    f" box-shadow:inset 0 0 8px rgba(255,0,255,0.5);"
                    f" font-weight:bold; {border_top}"
                )
            else:
                cell_style = (
                    f"background:{bg}; color:{tc}; {border_top}"
                )

            row += f'<td class="data-cell" style="{cell_style}">{txt}</td>'

        row += '</tr>'
        rows_html.append(row)

    table_html = '\n'.join(rows_html)
    total_w = label_w + n_months * cell_w + 40

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{
    margin: 0; padding: 0; background: #0f0f23;
    font-family: Consolas, 'Courier New', monospace;
    color: #ccc;
}}
.title-bar {{
    background: linear-gradient(135deg, #e91e63, #9c27b0);
    color: white; padding: 14px 24px; text-align: center;
}}
.title-bar h2 {{ margin: 0; font-size: 18px; }}
.title-bar p {{ margin: 4px 0 0; opacity: 0.85; font-size: 12px; }}
.legend {{
    display: flex; align-items: center; gap: 18px;
    padding: 8px 24px; background: #1a1a2e; font-size: 12px;
    border-bottom: 1px solid #333; flex-wrap: wrap;
}}
.legend-item {{
    display: flex; align-items: center; gap: 5px;
}}
.legend-swatch {{
    width: 16px; height: 16px; border-radius: 3px;
    display: inline-block;
}}
.scroll-wrapper {{
    width: 100%; overflow-x: auto; overflow-y: auto;
    max-height: calc(100vh - 110px);
    -webkit-overflow-scrolling: touch;
}}
table {{
    border-collapse: separate;
    border-spacing: 0;
    width: {total_w}px;
}}
.sticky-col {{
    position: sticky;
    left: 0;
    z-index: 10;
    background: #1a1a2e;
}}
.hdr-corner {{
    z-index: 20 !important;
    position: sticky !important;
    left: 0; top: 0;
    background: #1a1a2e !important;
    color: #e91e63;
    font-size: 12px;
    padding: 6px 8px;
    text-align: left;
    border-bottom: 2px solid #e91e63;
}}
.hdr-cell {{
    position: sticky;
    top: 0;
    z-index: 5;
    background: #1a1a2e;
    color: #aaa;
    font-size: 9px;
    padding: 4px 2px;
    text-align: center;
    border-bottom: 2px solid #e91e63;
    white-space: nowrap;
}}
.row-label {{
    font-size: 11px;
    padding: 4px 8px;
    white-space: nowrap;
    color: #ddd;
    border-right: 2px solid #333;
    min-width: {label_w}px;
    max-width: {label_w}px;
}}
.data-cell {{
    font-size: 10px;
    text-align: center;
    padding: 3px 2px;
    min-width: {cell_w}px;
    max-width: {cell_w}px;
    white-space: nowrap;
    box-sizing: border-box;
}}
tr:hover .row-label {{
    background: #2a2a4e !important;
    color: #fff;
}}
</style></head><body>

<div class="title-bar">
    <h2>Monthly Returns Matrix</h2>
    <p>All 46 strategies | Values in % | Scroll horizontally
    | <span style="color:#ff00ff;font-weight:bold;">
    Magenta = best performer of the month</span></p>
</div>
<div class="legend">
    <span style="color:#888;">Groups:</span>
    <div class="legend-item">
        <span class="legend-swatch" style="background:#1565c0;"></span>
        Steepeners</div>
    <div class="legend-item">
        <span class="legend-swatch" style="background:#c62828;"></span>
        Flatteners</div>
    <div class="legend-item">
        <span class="legend-swatch" style="background:#2e7d32;"></span>
        Butterflies</div>
    <div class="legend-item">
        <span class="legend-swatch" style="background:#ef6c00;"></span>
        Directional</div>
    <div class="legend-item">
        <span class="legend-swatch" style="background:#6a1b9a;"></span>
        Barbell</div>
    <span style="color:#555;margin-left:12px;">|</span>
    <div class="legend-item">
        <span class="legend-swatch"
         style="background:#333;outline:2px solid #ff00ff;"></span>
        Best of month</div>
</div>
<div class="scroll-wrapper">
<table>
{table_html}
</table>
</div>

</body></html>"""

    # Return HTML string (not a Plotly figure)
    return html


# ══════════════════════════════════════════════════════════════════════════
# CHART 3: REGIME BARS PER SLOPE + TLT
# ══════════════════════════════════════════════════════════════════════════

def build_chart3_regime_bars(yields, etf_ret):
    """Regime-colored bars for each slope segment + TLT equity.

    Inspired by the reference chart: bars colored by regime showing
    spread level in bps. Multiple panels stacked vertically.
    """

    segments = list(SEGMENT_DEFS.keys())
    n_panels = len(segments) + 1  # +1 for TLT

    fig = make_subplots(
        rows=n_panels, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.18] * len(segments) + [0.10],
        subplot_titles=[f"{seg.upper()} Spread" for seg in segments]
                       + ["TLT Equity"],
    )

    # Downsample for performance (weekly)
    yields_w = yields.resample("W").last().dropna(how="all")

    for panel_i, seg in enumerate(segments):
        short_col, long_col = SEGMENT_DEFS[seg]
        if short_col not in yields.columns or long_col not in yields.columns:
            continue

        spread = (yields_w[long_col] - yields_w[short_col]) * 100  # bps
        regime = compute_segment_regime(yields_w, short_col, long_col)

        row = panel_i + 1

        # Create one bar trace per regime for proper legend
        for reg_name, reg_color in REGIME_COLORS.items():
            mask = regime == reg_name
            if not mask.any():
                continue

            spread_masked = spread.copy()
            spread_masked[~mask] = None

            fig.add_trace(go.Bar(
                x=spread_masked.index,
                y=spread_masked.values,
                marker_color=reg_color,
                name=reg_name.replace("_", " ").title(),
                showlegend=(panel_i == 0),
                legendgroup=reg_name,
                hovertemplate=(f"{seg.upper()} %{{y:.0f}}bps "
                               f"({reg_name})<extra></extra>"),
                width=6 * 86400000,  # 6 days in ms for weekly bars
            ), row=row, col=1)

        fig.update_yaxes(
            title_text=f"{seg.upper()} (bps)",
            title_font=dict(size=10),
            tickfont=dict(size=9),
            row=row, col=1,
        )

    # TLT equity panel
    tlt_col = "TLT"
    if tlt_col in etf_ret.columns:
        tlt_eq = (1 + etf_ret[tlt_col]).cumprod()
        tlt_eq_w = tlt_eq.resample("W").last()
        fig.add_trace(go.Scatter(
            x=tlt_eq_w.index, y=tlt_eq_w.values,
            name="TLT",
            line=dict(color="#e91e63", width=2),
            fill="tozeroy", fillcolor="rgba(233,30,99,0.1)",
            showlegend=True,
            hovertemplate="TLT: $%{y:.2f}<extra></extra>",
        ), row=n_panels, col=1)
        fig.update_yaxes(
            title_text="TLT ($)",
            title_font=dict(size=10),
            tickfont=dict(size=9),
            row=n_panels, col=1,
        )

    fig.update_layout(
        height=300 * len(segments) + 250,
        template="plotly_dark",
        hovermode="x unified",
        barmode="overlay",
        bargap=0,
        legend=dict(
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#555", borderwidth=1,
            orientation="h",
            x=0.5, xanchor="center", y=1.02,
        ),
        title=dict(
            text=("<b>Yield Curve Regime by Slope Segment</b>"
                  "<br><span style='font-size:13px;color:#aaa'>"
                  "Spread in bps | Colored by local segment regime "
                  "(21d changes) | TLT overlay</span>"),
            x=0.5, xanchor="center",
            font=dict(color="white"),
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#16213e",
    )

    # Range selector on bottom panel
    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(count=3, label="3Y", step="year", stepmode="backward"),
                dict(count=5, label="5Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#2c3e50",
            font=dict(color="white"),
        ),
        rangeslider=dict(visible=True, thickness=0.03),
        row=n_panels, col=1,
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# CHART 4: STRATEGY × REGIME SHARPE HEATMAP
# ══════════════════════════════════════════════════════════════════════════

def build_chart4_regime_heatmap(returns_matrix, yc_regime, hmm_regime):
    """Heatmap: annualized Sharpe for each (strategy, regime) pair."""
    ordered, grp_labels = _order_strategies(returns_matrix.columns.tolist())

    yc_cols = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]
    hmm_cols = ["CALM", "STRESS"]
    all_cols = yc_cols + hmm_cols + ["Full Sample"]

    common = returns_matrix.index
    yc = yc_regime.reindex(common)
    hmm = hmm_regime.reindex(common)

    n_s = len(ordered)
    n_c = len(all_cols)
    z = np.full((n_s, n_c), np.nan)

    for i, sn in enumerate(ordered):
        ret = returns_matrix[sn]
        for j, reg in enumerate(yc_cols):
            r = ret[yc == reg].dropna()
            if len(r) >= 20:
                z[i, j] = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
        for j, reg in enumerate(hmm_cols):
            r = ret[hmm == reg].dropna()
            if len(r) >= 20:
                z[i, len(yc_cols) + j] = r.mean() / (r.std() + 1e-10) * np.sqrt(252)
        # Full sample
        r = ret.dropna()
        if len(r) >= 20:
            z[i, -1] = r.mean() / (r.std() + 1e-10) * np.sqrt(252)

    display = [_display_name(s) for s in ordered]
    text = [[f"{v:.2f}" if not np.isnan(v) else "" for v in row]
            for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=all_cols, y=display,
        colorscale="RdYlGn", zmid=0,
        text=text, texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate=(
            "Strategy: %{y}<br>Regime: %{x}<br>"
            "Sharpe: %{z:.2f}<extra></extra>"),
        colorbar=dict(title="Sharpe"),
    ))

    # Group separator lines
    for i in range(1, len(grp_labels)):
        if grp_labels[i] != grp_labels[i - 1]:
            fig.add_hline(y=i - 0.5,
                          line=dict(color="white", width=2))

    # YC / HMM / Full separators
    fig.add_vline(x=3.5, line=dict(color="white", width=2))
    fig.add_vline(x=5.5, line=dict(color="white", width=2))

    fig.update_layout(
        height=max(600, 22 * n_s),
        width=900,
        template="plotly_white",
        title=dict(
            text=("<b>Strategy × Regime Sharpe Heatmap</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  "Annualized Sharpe | YC regimes | HMM regimes | "
                  "Full sample</span>"),
            x=0.5, xanchor="center",
        ),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top"),
        margin=dict(l=160, r=80, t=100, b=30),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# CHART 5: RISK-RETURN SCATTERPLOT PER REGIME
# ══════════════════════════════════════════════════════════════════════════

def build_chart5_risk_return_scatter(returns_matrix, yc_regime):
    """Risk-return scatter with 2x2 panels (one per YC regime)."""
    strat_names = returns_matrix.columns.tolist()
    yc_labels = ["BULL_STEEP", "BULL_FLAT", "BEAR_STEEP", "BEAR_FLAT"]

    common = returns_matrix.index
    yc = yc_regime.reindex(common)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[r.replace("_", " ").title() for r in yc_labels],
        horizontal_spacing=0.12,
        vertical_spacing=0.12,
    )

    shown_groups = set()
    for ri, regime in enumerate(yc_labels):
        row, col = ri // 2 + 1, ri % 2 + 1
        mask = yc == regime

        for sn in strat_names:
            ret = returns_matrix[sn][mask].dropna()
            if len(ret) < 20:
                continue

            grp = _classify_strategy(sn)
            vol = ret.std() * np.sqrt(252) * 100
            ann_ret = ret.mean() * 252 * 100
            sharpe = ret.mean() / (ret.std() + 1e-10) * np.sqrt(252)
            eq = (1 + ret).cumprod()
            mdd = ((eq - eq.cummax()) / eq.cummax()).min() * 100

            color = GROUP_COLORS.get(grp, "#666")
            sz = max(6, min(25, abs(sharpe) * 10))

            show_leg = grp not in shown_groups
            if show_leg:
                shown_groups.add(grp)

            fig.add_trace(go.Scatter(
                x=[vol], y=[ann_ret], mode="markers",
                marker=dict(color=color, size=sz, opacity=0.8,
                            line=dict(width=1, color="white")),
                name=grp, legendgroup=grp, showlegend=show_leg,
                hovertemplate=(
                    f"<b>{sn}</b><br>"
                    f"Vol: {vol:.1f}%<br>"
                    f"Return: {ann_ret:.1f}%<br>"
                    f"Sharpe: {sharpe:.2f}<br>"
                    f"MDD: {mdd:.1f}%<extra></extra>"),
            ), row=row, col=col)

        # Reference: y=0
        fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.3)",
                      width=1, dash="dash"), row=row, col=col)

        # Sharpe=1 diagonal: Return% = Vol%
        fig.add_trace(go.Scatter(
            x=[0, 30], y=[0, 30], mode="lines",
            line=dict(color="rgba(0,0,0,0.15)", width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=row, col=col)

        fig.update_xaxes(title_text="Vol (%)", row=row, col=col)
        fig.update_yaxes(title_text="Return (%)", row=row, col=col)

    fig.update_layout(
        height=800,
        template="plotly_white",
        title=dict(
            text=("<b>Risk-Return Scatterplot by YC Regime</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  "Marker size = |Sharpe| | Dotted = Sharpe=1"
                  "</span>"),
            x=0.5, xanchor="center",
        ),
        margin=dict(t=100, b=50),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# CHART 6: PERFORMANCE PER RATE LEVEL (10Y TERCILES)
# ══════════════════════════════════════════════════════════════════════════

def build_chart6_rate_level(returns_matrix, yields):
    """Heatmap: Sharpe per strategy x 10Y yield tercile."""
    ordered, grp_labels = _order_strategies(
        returns_matrix.columns.tolist())

    y10 = yields["US_10Y"].reindex(returns_matrix.index).dropna()
    common = y10.index.intersection(returns_matrix.index)
    y10 = y10.reindex(common)

    p33 = y10.quantile(0.333)
    p66 = y10.quantile(0.667)

    rate_label = pd.Series("Mid", index=common)
    rate_label[y10 <= p33] = "Low"
    rate_label[y10 >= p66] = "High"

    col_labels = [f"Low (<{p33:.2f}%)",
                  f"Mid ({p33:.2f}-{p66:.2f}%)",
                  f"High (>{p66:.2f}%)"]
    regime_vals = ["Low", "Mid", "High"]

    n_s = len(ordered)
    z = np.full((n_s, 3), np.nan)

    for i, sn in enumerate(ordered):
        ret = returns_matrix[sn].reindex(common)
        for j, reg in enumerate(regime_vals):
            r = ret[rate_label == reg].dropna()
            if len(r) >= 20:
                z[i, j] = r.mean() / (r.std() + 1e-10) * np.sqrt(252)

    display = [_display_name(s) for s in ordered]
    text = [[f"{v:.2f}" if not np.isnan(v) else "" for v in row]
            for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=col_labels, y=display,
        colorscale="RdYlGn", zmid=0,
        text=text, texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate=(
            "Strategy: %{y}<br>Rate: %{x}<br>"
            "Sharpe: %{z:.2f}<extra></extra>"),
        colorbar=dict(title="Sharpe"),
    ))

    for i in range(1, len(grp_labels)):
        if grp_labels[i] != grp_labels[i - 1]:
            fig.add_hline(y=i - 0.5,
                          line=dict(color="white", width=2))

    fig.update_layout(
        height=max(600, 22 * n_s),
        width=700,
        template="plotly_white",
        title=dict(
            text=("<b>Strategy Sharpe by 10Y Rate Level</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  "Terciles of US 10Y yield level</span>"),
            x=0.5, xanchor="center",
        ),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top"),
        margin=dict(l=160, r=80, t=100, b=30),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# CHART 7: CORRELATION CLUSTERMAP
# ══════════════════════════════════════════════════════════════════════════

def build_chart7_correlation(returns_matrix):
    """Clustered correlation heatmap of strategy returns."""
    corr = returns_matrix.corr()

    # Hierarchical clustering for reordering
    dist = 1.0 - np.abs(corr.values)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    dist = np.clip(dist, 0.0, None)

    n = len(corr)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(dist[i, j])
    condensed = np.array(condensed)

    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    names_ordered = [corr.columns[i] for i in order]
    corr_ordered = corr.loc[names_ordered, names_ordered]

    display = [_display_name(s) for s in names_ordered]
    z = corr_ordered.values
    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=display, y=display,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=text, texttemplate="%{text}",
        textfont=dict(size=7),
        hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Corr"),
    ))

    # Group separator lines
    groups_ordered = [_classify_strategy(s) for s in names_ordered]
    for i in range(1, len(groups_ordered)):
        if groups_ordered[i] != groups_ordered[i - 1]:
            fig.add_hline(y=i - 0.5,
                          line=dict(color="white", width=1.5))
            fig.add_vline(x=i - 0.5,
                          line=dict(color="white", width=1.5))

    fig.update_layout(
        height=max(700, 18 * n),
        width=max(700, 18 * n),
        template="plotly_white",
        title=dict(
            text=("<b>Strategy Correlation Clustermap</b>"
                  "<br><span style='font-size:13px;color:#555'>"
                  "Ward clustering | Blue = negative, Red = positive"
                  "</span>"),
            x=0.5, xanchor="center",
        ),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(side="top", tickangle=-45),
        margin=dict(l=160, r=50, t=100, b=50),
    )

    return fig


# ══════════════════════════════════════════════════════════════════════════
# CHART 8: ROLLING BEST STRATEGY TIMELINE
# ══════════════════════════════════════════════════════════════════════════

def build_chart8_rolling_best(returns_matrix):
    """HTML table: rolling 6M Sharpe top/bottom 3 per quarter."""
    # Vectorized rolling Sharpe (much faster than .apply)
    window = 126
    min_p = 63
    r_mean = returns_matrix.rolling(window, min_periods=min_p).mean()
    r_std = returns_matrix.rolling(window, min_periods=min_p).std()
    rolling_sharpe = r_mean / (r_std + 1e-10) * np.sqrt(252)

    # Resample to quarterly
    quarterly = rolling_sharpe.resample("QE").last().dropna(how="all")

    rank_labels = ["Top 1", "Top 2", "Top 3",
                   "Bottom 3", "Bottom 2", "Bottom 1"]
    n_ranks = len(rank_labels)
    n_q = len(quarterly)

    q_labels = [f"{d.year}Q{(d.month - 1) // 3 + 1}"
                for d in quarterly.index]

    # Compute rankings per quarter
    cells = [[None] * n_q for _ in range(n_ranks)]

    for qi in range(n_q):
        row = quarterly.iloc[qi].dropna().sort_values(ascending=False)
        if len(row) < 6:
            continue
        for k in range(3):
            sn = row.index[k]
            cells[k][qi] = (sn, row.iloc[k], _classify_strategy(sn))
        for k in range(3):
            idx = -(3 - k)
            sn = row.index[idx]
            cells[3 + k][qi] = (sn, row.iloc[idx],
                                _classify_strategy(sn))

    # ── Build HTML ───────────────────────────────────────────────
    cell_w = 110
    label_w = 90

    rows_html = []
    # Header
    hdr = (f'<tr><th class="sticky-col hdr-corner"'
           f' style="width:{label_w}px;">Rank</th>')
    for ql in q_labels:
        hdr += (f'<th class="hdr-cell" style="width:{cell_w}px;">'
                f'{ql}</th>')
    hdr += '</tr>'
    rows_html.append(hdr)

    # Data rows
    for ri in range(n_ranks):
        sep = "border-top:3px solid #e91e63;" if ri == 3 else ""
        row_html = (f'<tr><td class="sticky-col row-label"'
                    f' style="{sep}">{rank_labels[ri]}</td>')

        for qi in range(n_q):
            cell = cells[ri][qi]
            if cell is None:
                row_html += (f'<td class="data-cell"'
                             f' style="{sep}">-</td>')
            else:
                sn, sv, grp = cell
                color = GROUP_COLORS.get(grp, "#666")
                short = sn.replace("steep_", "S:")
                short = short.replace("flat_", "F:")
                short = short.replace("bfly_", "B:")
                short = short.replace("bh_", "D:")
                short = short.replace("barbell_", "BB:")
                vc = "#4caf50" if sv >= 0 else "#f44336"
                cs = f"border-left:4px solid {color};{sep}"
                row_html += (
                    f'<td class="data-cell" style="{cs}">'
                    f'<span class="sn">{short}</span>'
                    f'<br><span style="color:{vc};'
                    f'font-size:9px;">{sv:+.2f}</span></td>')

        row_html += '</tr>'
        rows_html.append(row_html)

    table_html = '\n'.join(rows_html)
    total_w = label_w + n_q * cell_w + 40

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{
    margin: 0; padding: 0; background: #0f0f23;
    font-family: Consolas, 'Courier New', monospace;
    color: #ccc;
}}
.title-bar {{
    background: linear-gradient(135deg, #e91e63, #9c27b0);
    color: white; padding: 14px 24px; text-align: center;
}}
.title-bar h2 {{ margin: 0; font-size: 18px; }}
.title-bar p {{ margin: 4px 0 0; opacity: 0.85; font-size: 12px; }}
.legend {{
    display: flex; align-items: center; gap: 14px;
    padding: 8px 24px; background: #1a1a2e; font-size: 11px;
    border-bottom: 1px solid #333; flex-wrap: wrap;
}}
.legend-item {{
    display: flex; align-items: center; gap: 4px;
}}
.legend-swatch {{
    width: 14px; height: 14px; border-radius: 3px;
    display: inline-block;
}}
.scroll-wrapper {{
    width: 100%; overflow-x: auto; overflow-y: auto;
    max-height: calc(100vh - 130px);
    -webkit-overflow-scrolling: touch;
}}
table {{
    border-collapse: separate;
    border-spacing: 0;
    width: {total_w}px;
}}
.sticky-col {{
    position: sticky;
    left: 0;
    z-index: 10;
    background: #1a1a2e;
}}
.hdr-corner {{
    z-index: 20 !important;
    position: sticky !important;
    left: 0; top: 0;
    background: #1a1a2e !important;
    color: #e91e63;
    font-size: 12px;
    padding: 6px 8px;
    text-align: left;
    border-bottom: 2px solid #e91e63;
}}
.hdr-cell {{
    position: sticky;
    top: 0;
    z-index: 5;
    background: #1a1a2e;
    color: #aaa;
    font-size: 10px;
    padding: 4px 2px;
    text-align: center;
    border-bottom: 2px solid #e91e63;
    white-space: nowrap;
}}
.row-label {{
    font-size: 12px;
    padding: 6px 10px;
    white-space: nowrap;
    color: #ddd;
    border-right: 2px solid #333;
    font-weight: bold;
    min-width: {label_w}px;
    max-width: {label_w}px;
}}
.data-cell {{
    font-size: 10px;
    text-align: center;
    padding: 4px 3px;
    min-width: {cell_w}px;
    max-width: {cell_w}px;
    white-space: nowrap;
    box-sizing: border-box;
    vertical-align: middle;
}}
.sn {{
    font-weight: bold;
    color: #eee;
    font-size: 9px;
}}
tr:hover td {{
    background: #2a2a4e !important;
}}
</style></head><body>

<div class="title-bar">
    <h2>Rolling Best &amp; Worst Strategies</h2>
    <p>6M rolling Sharpe | Top 3 and Bottom 3 per quarter
    | Colored by strategy group</p>
</div>
<div class="legend">
    <span style="color:#888;">Groups:</span>"""

    for grp, color in GROUP_COLORS.items():
        if grp == "Other":
            continue
        html += (f'\n    <div class="legend-item">'
                 f'<span class="legend-swatch"'
                 f' style="background:{color};"></span>'
                 f' {grp}</div>')

    html += f"""
</div>
<div class="scroll-wrapper">
<table>
{table_html}
</table>
</div>

</body></html>"""

    return html


# ══════════════════════════════════════════════════════════════════════════
# HTML WRITER + TABBED DASHBOARD
# ══════════════════════════════════════════════════════════════════════════

def _write_html(fig, path, scrollable=False):
    if scrollable:
        # For wide charts: wrap in scrollable container
        inner_html = fig.to_html(
            include_plotlyjs=True, full_html=False,
            config={"displaylogo": False},
        )
        full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ margin: 0; padding: 0; background: #fafafa; }}
.scroll-container {{
    width: 100%; overflow-x: auto; overflow-y: auto;
    -webkit-overflow-scrolling: touch;
}}
</style></head><body>
<div class="scroll-container">{inner_html}</div>
</body></html>"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(full_html)
    else:
        fig.write_html(
            path, include_plotlyjs=True, full_html=True,
            config={
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png", "height": 1200, "width": 1800,
                    "scale": 2,
                },
            },
        )


def _build_tabbed_html(tab_figs, out_path):
    out_dir = os.path.dirname(out_path)

    tabs = [
        ("Strategy Equity",     "regime_chart1_modes.html",     False),
        ("Monthly Matrix",      "regime_chart2_monthly.html",   True),
        ("Regime Bars + TLT",   "regime_chart3_bars.html",      False),
        ("Strategy\u00d7Regime", "regime_chart4_heatmap.html",  False),
        ("Risk-Return",         "regime_chart5_scatter.html",   False),
        ("Rate Level",          "regime_chart6_rates.html",     False),
        ("Correlations",        "regime_chart7_corr.html",      False),
        ("Rolling Best",        "regime_chart8_rolling.html",   True),
    ]

    for (_, fname, _flag), fig_or_html in zip(tabs, tab_figs):
        fpath = os.path.join(out_dir, fname)
        if isinstance(fig_or_html, str):
            # Raw HTML string (e.g. chart2 pure-HTML table)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(fig_or_html)
        else:
            _write_html(fig_or_html, fpath)

    buttons_html = ""
    for i, (label, fname, _scroll) in enumerate(tabs):
        active = " active" if i == 0 else ""
        buttons_html += (
            f'    <button class="tab-btn{active}" '
            f'onclick="switchTab(event, {i})">{label}</button>\n'
        )

    iframes_html = ""
    for i, (_label, fname, _scroll) in enumerate(tabs):
        display = "block" if i == 0 else "none"
        iframes_html += (
            f'<iframe id="frame{i}" src="{fname}" '
            f'class="tab-frame" style="display:{display}"></iframe>\n'
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Regime Analysis Charts</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #0a0a1a; }}
    .header {{
        background: linear-gradient(135deg, #e91e63, #9c27b0);
        color: white; padding: 18px 30px; text-align: center;
    }}
    .header h1 {{ margin: 0; font-size: 22px; }}
    .header p {{ margin: 4px 0 0; opacity: 0.85; font-size: 13px; }}
    .tab-bar {{
        display: flex; background: #1a1a2e; padding: 0;
        border-bottom: 3px solid #e91e63;
    }}
    .tab-btn {{
        padding: 12px 24px; cursor: pointer; color: #aaa;
        font-size: 14px; font-weight: bold; border: none;
        background: transparent; transition: all 0.2s;
    }}
    .tab-btn:hover {{ background: #2a2a3e; color: white; }}
    .tab-btn.active {{ background: #e91e63; color: white; }}
    .tab-frame {{
        width: 100%; height: calc(100vh - 105px); border: none;
        background: #1a1a2e;
    }}
</style>
</head>
<body>
<div class="header">
    <h1>Regime Analysis Deep Dive</h1>
    <p>Equity | Monthly | Regime Bars | Strategy&times;Regime | Risk-Return | Rate Level | Correlations | Rolling Best</p>
</div>
<div class="tab-bar">
{buttons_html}</div>
{iframes_html}
<script>
function switchTab(evt, idx) {{
    var frames = document.querySelectorAll('.tab-frame');
    for (var i = 0; i < frames.length; i++) frames[i].style.display = 'none';
    var buttons = document.querySelectorAll('.tab-btn');
    for (var i = 0; i < buttons.length; i++) buttons[i].classList.remove('active');
    document.getElementById('frame' + idx).style.display = 'block';
    evt.currentTarget.classList.add('active');
}}
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 60)
    print("  REGIME ANALYSIS CHARTS")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────
    print("\n[1/6] Loading data ...")
    D = load_all()
    etf_ret = D["etf_ret"]
    yields = D["yields"]
    rr_1m = D.get("rr_1m")
    move = D.get("move")
    vix = D.get("vix")

    # ── Backtest + returns matrix ────────────────────────────────
    print("[2/6] Backtesting strategies ...")
    summary_df, results = backtest_all(STRATEGY_POOL, etf_ret,
                                       start=BACKTEST_START, tcost_bps=0)
    strategy_returns = {n: r["daily_ret"] for n, r in results.items()}
    returns_matrix = build_returns_matrix(strategy_returns)
    print(f"  Returns matrix: {returns_matrix.shape}")

    # ── Build base charts ────────────────────────────────────────
    print("[3/6] Building base charts (1-3) ...")

    print("  Chart 1: Strategy equity curves (grouped) ...")
    fig1 = build_chart1_strategy_equity(returns_matrix)

    print("  Chart 2: Monthly best strategy selector ...")
    fig2 = build_chart2_monthly_best(returns_matrix)

    print("  Chart 3: Regime bars per slope + TLT ...")
    fig3 = build_chart3_regime_bars(yields, etf_ret)

    # ── Compute regimes ──────────────────────────────────────────
    print("[4/6] Computing regimes (YC + HMM) ...")
    yc_regime = compute_yc_regime(yields)
    D = compute_regime(D)
    hmm_regime = D["regime"]
    print(f"  YC regime distribution: {yc_regime.value_counts().to_dict()}")
    print(f"  HMM regime distribution: {hmm_regime.value_counts().to_dict()}")

    # ── Build analytical charts ──────────────────────────────────
    print("[5/6] Building analytical charts (4-8) ...")

    print("  Chart 4: Strategy x Regime heatmap ...")
    fig4 = build_chart4_regime_heatmap(returns_matrix, yc_regime,
                                       hmm_regime)

    print("  Chart 5: Risk-Return scatterplot ...")
    fig5 = build_chart5_risk_return_scatter(returns_matrix, yc_regime)

    print("  Chart 6: Rate level analysis ...")
    fig6 = build_chart6_rate_level(returns_matrix, yields)

    print("  Chart 7: Correlation clustermap ...")
    fig7 = build_chart7_correlation(returns_matrix)

    print("  Chart 8: Rolling best strategy timeline ...")
    fig8 = build_chart8_rolling_best(returns_matrix)

    # ── Save dashboard ───────────────────────────────────────────
    print("[6/6] Saving dashboard (8 tabs) ...")
    tab_figs = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8]
    main_path = os.path.join(OUT_DIR, "regime_charts_dashboard.html")
    _build_tabbed_html(tab_figs, main_path)

    elapsed = time.time() - t0
    print(f"\n  DONE in {elapsed:.0f}s")
    print(f"  Dashboard: {main_path}")
    print("=" * 60)

    return main_path


if __name__ == "__main__":
    main()
