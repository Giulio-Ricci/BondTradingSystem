"""All equity curves on a single plot - every combination tested."""

import sys, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))

from config import SAVE_DIR, BACKTEST_START, RISK_FREE_RATE, DEFAULT_PARAMS
from data_loader import load_all
from signals import compute_zscore_standard, compute_zscore_move_adj, compute_all_signals
from regime import compute_regime
from backtest import backtest

SAVE = os.path.join(SAVE_DIR, "robust")
os.makedirs(SAVE, exist_ok=True)


def build_equity(net_ret):
    """Build equity curve from net returns."""
    return (1 + net_ret).cumprod() * 1e7


def run_backtest_get_equity(z, regime, etf_ret, params, start=BACKTEST_START):
    """Run backtest and return equity series."""
    res = backtest(z, regime, etf_ret, params=params, start=start)
    if len(res["net_ret"]) == 0:
        return None, res["metrics"]
    return build_equity(res["net_ret"]), res["metrics"]


def custom_strategy_equity(z, regime, etf_ret, logic_fn, default_w=0.5, start=BACKTEST_START):
    """Run a custom strategy logic and return equity + metrics."""
    common = z.dropna().index.intersection(regime.dropna().index) \
               .intersection(etf_ret.dropna(subset=["TLT", "SHV"]).index)
    common = common[common >= start].sort_values()
    if len(common) < 20:
        return None, {}

    weight = pd.Series(np.nan, index=common)
    events = []
    state = {"last_event": pd.Timestamp("1900-01-01"), "last_signal": None}

    for dt in common:
        w = logic_fn(dt, z.loc[dt], regime.loc[dt], state, weight)
        if w is not None:
            weight.loc[dt] = w

    weight = weight.ffill().fillna(default_w)
    w_del = weight.shift(2).fillna(default_w)

    r_long = etf_ret["TLT"].reindex(common).fillna(0)
    r_short = etf_ret["SHV"].reindex(common).fillna(0)
    net = w_del * r_long + (1 - w_del) * r_short
    rebal = w_del.diff().abs().fillna(0)
    net = net - rebal * (5 / 10_000)

    eq = build_equity(net)
    yrs = len(net) / 252
    total = eq.iloc[-1] / eq.iloc[0] - 1
    ann = (1 + total) ** (1 / max(yrs, 0.01)) - 1
    vol = net.std() * np.sqrt(252)
    sh = (ann - RISK_FREE_RATE) / vol if vol > 0 else 0
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    bm = etf_ret["TLT"].reindex(common).fillna(0)
    bm_eq = (1 + bm).cumprod()
    bm_ann = (1 + bm_eq.iloc[-1] - 1) ** (1 / max(yrs, 0.01)) - 1
    excess = ann - bm_ann

    return eq, {"sharpe": sh, "excess": excess, "mdd": mdd, "ann_ret": ann, "vol": vol}


def main():
    print("Loading data...")
    D = load_all()
    D = compute_all_signals(D)
    D = compute_regime(D)

    regime = D["regime"]
    etf_ret = D["etf_ret"]
    rr_1m = D["rr_1m"]

    params_base = {**DEFAULT_PARAMS, "tl_calm": -3.5, "ts_calm": 3.0, "tl_stress": -4.0}

    curves = {}  # name -> (equity_series, metrics_dict)

    # =========================================================================
    # GROUP 1: Z-SCORE VARIANTS (with fixed thresholds)
    # =========================================================================
    print("Computing z-score variants...")

    # z_std windows
    for w in [63, 126, 250, 504]:
        z = compute_zscore_standard(rr_1m, window=w)
        eq, m = run_backtest_get_equity(z, regime, etf_ret, params_base)
        curves[f"z_std w={w}d"] = (eq, m)

    # z_move 250d
    eq, m = run_backtest_get_equity(D["z_move"], regime, etf_ret, params_base)
    curves["z_move 250d"] = (eq, m)

    # Expanding z-score
    mu_exp = rr_1m.expanding(min_periods=250).mean()
    sig_exp = rr_1m.expanding(min_periods=250).std()
    z_exp = (rr_1m - mu_exp) / sig_exp.replace(0, np.nan)
    eq, m = run_backtest_get_equity(z_exp, regime, etf_ret, params_base)
    curves["z_expanding"] = (eq, m)

    # Rank-based (probit) 250d
    z_rank = rr_1m.rolling(250, min_periods=63).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100.0, raw=False)
    z_rank_probit = pd.Series(stats.norm.ppf(z_rank.clip(0.001, 0.999).values),
                              index=z_rank.index)
    eq, m = run_backtest_get_equity(z_rank_probit, regime, etf_ret, params_base)
    curves["z_rank probit 250d"] = (eq, m)

    # Rank 10/90 thresholds
    params_rank = {**DEFAULT_PARAMS, "tl_calm": -1.28, "ts_calm": 1.28, "tl_stress": -1.65}
    eq, m = run_backtest_get_equity(z_rank_probit, regime, etf_ret, params_rank)
    curves["z_rank 10/90 thr"] = (eq, m)

    # =========================================================================
    # GROUP 2: THRESHOLD VARIANTS (on z_std 250d)
    # =========================================================================
    print("Computing threshold variants...")
    z_std = D["z_std"]

    # Different fixed thresholds
    threshold_sets = [
        ("tl=-2.0 ts=2.0", -2.0, 2.0, -3.0),
        ("tl=-3.0 ts=2.5", -3.0, 2.5, -4.0),
        ("tl=-3.5 ts=3.0 (base)", -3.5, 3.0, -4.0),
        ("tl=-4.0 ts=3.5", -4.0, 3.5, -5.0),
    ]
    for label, tlc, tsc, tls in threshold_sets:
        p = {**DEFAULT_PARAMS, "tl_calm": tlc, "ts_calm": tsc, "tl_stress": tls}
        eq, m = run_backtest_get_equity(z_std, regime, etf_ret, p)
        curves[f"Fixed {label}"] = (eq, m)

    # Adaptive 5/95
    print("Computing adaptive thresholds...")
    roll_lo = z_std.rolling(504, min_periods=250).quantile(0.05)
    roll_hi = z_std.rolling(504, min_periods=250).quantile(0.95)

    def adaptive_logic(dt, zv, reg, state, weight):
        if pd.isna(zv):
            return None
        try:
            lo = roll_lo.loc[dt]
            hi = roll_hi.loc[dt]
        except KeyError:
            return None
        if pd.isna(lo) or pd.isna(hi):
            return None
        days = (dt - state["last_event"]).days
        signal = None
        if reg == "CALM":
            if zv <= lo: signal = "SHORT"
            elif zv >= hi: signal = "LONG"
        else:
            if zv <= lo * 1.3: signal = "SHORT"
        if signal and days < 5:
            signal = None
        if signal == "LONG":
            state["last_event"] = dt
            return 1.0
        elif signal == "SHORT":
            state["last_event"] = dt
            return 0.0
        return None

    eq, m = custom_strategy_equity(z_std, regime, etf_ret, adaptive_logic)
    curves["Adaptive 5/95"] = (eq, m)

    # =========================================================================
    # GROUP 3: SHORT-ONLY VARIANTS
    # =========================================================================
    print("Computing SHORT-only variants...")

    for hold in [21, 63, 126]:
        def make_short_only(hold_days):
            def logic(dt, zv, reg, state, weight):
                if pd.isna(zv):
                    return None
                days = (dt - state["last_event"]).days
                # Auto-return after holding
                if state["last_signal"] == "SHORT" and days == hold_days:
                    state["last_signal"] = "LONG"
                    return 1.0
                tl = -3.5 if reg == "CALM" else -4.0
                if zv <= tl and days >= 5:
                    state["last_event"] = dt
                    state["last_signal"] = "SHORT"
                    return 0.0
                return None
            return logic
        eq, m = custom_strategy_equity(z_std, regime, etf_ret,
                                       make_short_only(hold), default_w=1.0)
        curves[f"SHORT-only hold={hold}d"] = (eq, m)

    # =========================================================================
    # GROUP 4: FIXED HOLDING PERIOD
    # =========================================================================
    print("Computing fixed holding variants...")

    for hold in [5, 21, 63]:
        def make_fixed_hold(hold_days):
            def logic(dt, zv, reg, state, weight):
                if pd.isna(zv):
                    return None
                days = (dt - state["last_event"]).days
                if state["last_signal"] is not None and days == hold_days:
                    state["last_signal"] = None
                    return 0.5
                if days >= hold_days or state["last_signal"] is None:
                    tl = -3.5 if reg == "CALM" else -4.0
                    ts = 3.0 if reg == "CALM" else 99
                    if zv <= tl:
                        state["last_event"] = dt
                        state["last_signal"] = "SHORT"
                        return 0.0
                    elif zv >= ts:
                        state["last_event"] = dt
                        state["last_signal"] = "LONG"
                        return 1.0
                return None
            return logic
        eq, m = custom_strategy_equity(z_std, regime, etf_ret, make_fixed_hold(hold))
        curves[f"Fixed hold={hold}d"] = (eq, m)

    # =========================================================================
    # BENCHMARKS
    # =========================================================================
    print("Computing benchmarks...")

    # B&H TLT
    idx_ref = list(curves.values())[0][0].index
    bm_tlt = build_equity(etf_ret["TLT"].reindex(idx_ref).fillna(0))
    curves["B&H TLT"] = (bm_tlt, {"sharpe": -99, "excess": 0, "mdd": 0})

    # 50/50
    bm_5050 = build_equity(
        0.5 * etf_ret["TLT"].reindex(idx_ref).fillna(0) +
        0.5 * etf_ret["SHV"].reindex(idx_ref).fillna(0))
    curves["50/50 TLT/SHV"] = (bm_5050, {"sharpe": -99, "excess": 0, "mdd": 0})

    # B&H SHV (cash)
    bm_shv = build_equity(etf_ret["SHV"].reindex(idx_ref).fillna(0))
    curves["B&H SHV (cash)"] = (bm_shv, {"sharpe": -99, "excess": 0, "mdd": 0})

    # =========================================================================
    # PRINT SUMMARY TABLE
    # =========================================================================
    print("\n" + "=" * 90)
    print(f"  {'Variant':<30} {'Sharpe':>8} {'Ann Ret':>8} {'Vol':>8} {'Excess':>8} {'MDD':>8}")
    print("  " + "-" * 80)
    sorted_curves = sorted(curves.items(),
                           key=lambda x: x[1][1].get("sharpe", -99), reverse=True)
    for name, (eq, m) in sorted_curves:
        sh = m.get("sharpe", -99)
        if sh == -99:
            print(f"  {name:<30} {'---':>8} {'---':>8} {'---':>8} {'---':>8} {'---':>8}")
        else:
            print(f"  {name:<30} {sh:>8.3f} {m.get('ann_ret',0):>8.3f} "
                  f"{m.get('vol',0):>8.3f} {m.get('excess',0):>8.4f} {m.get('mdd',0):>8.4f}")
    print("=" * 90)

    # =========================================================================
    # MEGA PLOT
    # =========================================================================
    print("\nGenerating plot...")

    # Color scheme by group
    group_colors = {
        # Z-score variants: blues
        "z_std w=63d": "#1f77b4",
        "z_std w=126d": "#4a9bd9",
        "z_std w=250d": "#7fbfee",
        "z_std w=504d": "#b3d9f7",
        "z_move 250d": "#d62728",       # RED - best performer
        "z_expanding": "#17becf",
        "z_rank probit 250d": "#bcbd22",
        "z_rank 10/90 thr": "#dbdb8d",
        # Threshold variants: greens
        "Fixed tl=-2.0 ts=2.0": "#2ca02c",
        "Fixed tl=-3.0 ts=2.5": "#53c653",
        "Fixed tl=-3.5 ts=3.0 (base)": "#80d980",
        "Fixed tl=-4.0 ts=3.5": "#adebad",
        "Adaptive 5/95": "#98df8a",
        # SHORT-only: oranges
        "SHORT-only hold=21d": "#ff7f0e",
        "SHORT-only hold=63d": "#ffbb78",
        "SHORT-only hold=126d": "#ffd9b3",
        # Fixed holding: purples
        "Fixed hold=5d": "#9467bd",
        "Fixed hold=21d": "#b89fd4",
        "Fixed hold=63d": "#d4c4e0",
        # Benchmarks: grays
        "B&H TLT": "#555555",
        "50/50 TLT/SHV": "#999999",
        "B&H SHV (cash)": "#cccccc",
    }

    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot benchmarks first (behind)
    for name in ["B&H SHV (cash)", "50/50 TLT/SHV", "B&H TLT"]:
        if name in curves and curves[name][0] is not None:
            eq = curves[name][0]
            c = group_colors.get(name, "#888888")
            ax.plot(eq.index, eq.values, color=c, lw=2.0, ls="--", alpha=0.7, label=name)

    # Plot all strategies (sorted by Sharpe so best is on top)
    for name, (eq, m) in sorted_curves:
        if eq is None or name in ["B&H TLT", "50/50 TLT/SHV", "B&H SHV (cash)"]:
            continue
        c = group_colors.get(name, "#666666")
        sh = m.get("sharpe", 0)
        # Highlight the best
        if name == "z_move 250d":
            ax.plot(eq.index, eq.values, color=c, lw=3.0, alpha=1.0,
                    label=f"** {name} (Sh={sh:.3f}) **", zorder=10)
        else:
            ax.plot(eq.index, eq.values, color=c, lw=1.0, alpha=0.6,
                    label=f"{name} (Sh={sh:.3f})")

    ax.set_yscale("log")
    ax.set_ylabel("NAV ($)", fontsize=12)
    ax.set_title("ALL STRATEGY VARIANTS - Equity Curves Comparison", fontsize=14, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.2)

    # Legend outside
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, framealpha=0.9)

    path = os.path.join(SAVE, "ALL_equity_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    # =========================================================================
    # GROUPED PLOTS (one per group, cleaner)
    # =========================================================================

    groups = {
        "Z-Score Construction": [
            "z_std w=63d", "z_std w=126d", "z_std w=250d", "z_std w=504d",
            "z_move 250d", "z_expanding", "z_rank probit 250d",
        ],
        "Threshold Variants": [
            "Fixed tl=-2.0 ts=2.0", "Fixed tl=-3.0 ts=2.5",
            "Fixed tl=-3.5 ts=3.0 (base)", "Fixed tl=-4.0 ts=3.5",
            "Adaptive 5/95", "z_move 250d",
        ],
        "Strategy Structure": [
            "z_std w=250d",
            "SHORT-only hold=21d", "SHORT-only hold=63d", "SHORT-only hold=126d",
            "Fixed hold=5d", "Fixed hold=21d", "Fixed hold=63d", "z_move 250d",
        ],
    }

    for group_name, members in groups.items():
        fig, ax = plt.subplots(figsize=(16, 8))

        # Benchmarks
        for bm in ["B&H TLT", "50/50 TLT/SHV"]:
            if bm in curves and curves[bm][0] is not None:
                eq = curves[bm][0]
                ax.plot(eq.index, eq.values, color=group_colors.get(bm, "#888"),
                        lw=1.5, ls="--", alpha=0.6, label=bm)

        for name in members:
            if name not in curves or curves[name][0] is None:
                continue
            eq = curves[name][0]
            m = curves[name][1]
            sh = m.get("sharpe", 0)
            c = group_colors.get(name, "#666666")
            lw = 3.0 if name == "z_move 250d" else 1.5
            al = 1.0 if name == "z_move 250d" else 0.8
            ax.plot(eq.index, eq.values, color=c, lw=lw, alpha=al,
                    label=f"{name} (Sh={sh:.3f})")

        ax.set_yscale("log")
        ax.set_ylabel("NAV ($)", fontsize=11)
        ax.set_title(f"{group_name} - Equity Comparison", fontsize=13, fontweight="bold")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

        fname = group_name.lower().replace(" ", "_").replace("-", "_")
        path = os.path.join(SAVE, f"equity_{fname}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
