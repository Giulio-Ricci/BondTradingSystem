"""
strategy_definitions.py - Pool of 25 yield curve relative value strategies.

Strategy types:
    A) 8 Steepener spreads  (long front, short back)
    B) 4 Flattener spreads  (long back, short front)
    C) 6 Butterfly          (3 structures x 2 directions)
    D) 4 Directional        (baseline B&H)
    E) 3 Barbell            (duration-neutral long/short mix)
"""

# ── Approximate durations for Treasury ETFs ──────────────────────────────
ETF_DURATIONS = {
    "SHV": 0.4,
    "SHY": 1.9,
    "IEI": 4.4,
    "IEF": 7.5,
    "TLH": 13.0,
    "TLT": 17.5,
    "GOVT": 6.5,
}


def _dn_weight(long_etf: str, short_etf: str) -> float:
    """Duration-neutral weight for the short leg.

    For spread long A / short B: weight_B = -(dur_A / dur_B).
    """
    return -(ETF_DURATIONS[long_etf] / ETF_DURATIONS[short_etf])


def _normalize_weights(weights: dict) -> dict:
    """Normalize weights so sum(|w|) = 2.0 for comparability."""
    total = sum(abs(w) for w in weights.values())
    if total < 1e-10:
        return weights
    factor = 2.0 / total
    return {k: v * factor for k, v in weights.items()}


def _make_spread(name, display, long_etf, short_etf, direction, desc):
    """Build a duration-neutral 2-leg spread strategy dict."""
    raw_w_short = _dn_weight(long_etf, short_etf)
    raw_weights = {long_etf: 1.0, short_etf: raw_w_short}
    weights = _normalize_weights(raw_weights)
    return {
        "name": name,
        "display": display,
        "type": "spread",
        "direction": direction,
        "legs": [long_etf, short_etf],
        "weights": weights,
        "description": desc,
    }


def _make_butterfly(name, display, wing1, body, wing2, direction, desc):
    """Build a duration-neutral 3-leg butterfly strategy dict.

    Positive butterfly = long wings, short body (profit from curvature increase).
    Negative butterfly = short wings, long body (profit from curvature decrease).

    Body weight for DN: w_body = -(d_wing1 + d_wing2) / d_body
    """
    d1 = ETF_DURATIONS[wing1]
    d2 = ETF_DURATIONS[wing2]
    db = ETF_DURATIONS[body]

    if direction == "positive":
        # Long wings, short body
        raw_weights = {
            wing1: 1.0,
            body: -(d1 + d2) / db,
            wing2: 1.0,
        }
    else:
        # Short wings, long body
        raw_weights = {
            wing1: -1.0,
            body: (d1 + d2) / db,
            wing2: -1.0,
        }

    weights = _normalize_weights(raw_weights)
    return {
        "name": name,
        "display": display,
        "type": "butterfly",
        "direction": direction,
        "legs": [wing1, body, wing2],
        "weights": weights,
        "description": desc,
    }


def _make_directional(name, display, weights_raw, desc):
    """Build a directional (B&H) strategy dict."""
    weights = _normalize_weights(weights_raw)
    return {
        "name": name,
        "display": display,
        "type": "directional",
        "direction": "long",
        "legs": list(weights_raw.keys()),
        "weights": weights,
        "description": desc,
    }


def _make_barbell(name, display, short_etf, long_etf, desc):
    """Build a duration-neutral barbell (equal risk from both ends)."""
    d_short = ETF_DURATIONS[short_etf]
    d_long = ETF_DURATIONS[long_etf]
    # Both positive weights, ratio for DN: w_short/w_long = d_long/d_short
    raw_weights = {short_etf: d_long / d_short, long_etf: 1.0}
    weights = _normalize_weights(raw_weights)
    return {
        "name": name,
        "display": display,
        "type": "barbell",
        "direction": "neutral",
        "legs": [short_etf, long_etf],
        "weights": weights,
        "description": desc,
    }


# ══════════════════════════════════════════════════════════════════════════
# STRATEGY POOL (25 strategies)
# ══════════════════════════════════════════════════════════════════════════

STRATEGY_POOL = []

# ── A) 8 Steepener spreads (long front, short back) ──────────────────────
_steepeners = [
    # Adjacent
    ("steep_2s5s",   "Steep 2s5s",   "SHY", "IEI", "steepener", "Steepener: long SHY / short IEI (2s5s adjacent)"),
    ("steep_5s10s",  "Steep 5s10s",  "IEI", "IEF", "steepener", "Steepener: long IEI / short IEF (5s10s adjacent)"),
    ("steep_10s20s", "Steep 10s20s", "IEF", "TLH", "steepener", "Steepener: long IEF / short TLH (10s20s adjacent)"),
    ("steep_20s30s", "Steep 20s30s", "TLH", "TLT", "steepener", "Steepener: long TLH / short TLT (20s30s adjacent)"),
    # Wide
    ("steep_2s10s",  "Steep 2s10s",  "SHY", "IEF", "steepener", "Steepener: long SHY / short IEF (2s10s wide)"),
    ("steep_2s30s",  "Steep 2s30s",  "SHY", "TLT", "steepener", "Steepener: long SHY / short TLT (2s30s wide)"),
    ("steep_5s30s",  "Steep 5s30s",  "IEI", "TLT", "steepener", "Steepener: long IEI / short TLT (5s30s wide)"),
    ("steep_10s30s", "Steep 10s30s", "IEF", "TLT", "steepener", "Steepener: long IEF / short TLT (10s30s wide)"),
]
for name, display, long_e, short_e, direction, desc in _steepeners:
    STRATEGY_POOL.append(_make_spread(name, display, long_e, short_e, direction, desc))

# ── B) 4 Flattener spreads (long back, short front) ─────────────────────
_flatteners = [
    ("flat_10s2s",  "Flat 10s2s",  "IEF", "SHY", "flattener", "Flattener: long IEF / short SHY (10s2s)"),
    ("flat_30s2s",  "Flat 30s2s",  "TLT", "SHY", "flattener", "Flattener: long TLT / short SHY (30s2s)"),
    ("flat_30s5s",  "Flat 30s5s",  "TLT", "IEI", "flattener", "Flattener: long TLT / short IEI (30s5s)"),
    ("flat_30s10s", "Flat 30s10s", "TLT", "IEF", "flattener", "Flattener: long TLT / short IEF (30s10s)"),
]
for name, display, long_e, short_e, direction, desc in _flatteners:
    STRATEGY_POOL.append(_make_spread(name, display, long_e, short_e, direction, desc))

# ── C) 6 Butterfly (3 structures x 2 directions) ────────────────────────
_butterflies = [
    ("bfly_2s5s10s_pos",  "Bfly 2-5-10 +", "SHY", "IEI", "IEF", "positive",
     "Positive butterfly 2s5s10s: long wings (SHY+IEF), short body (IEI)"),
    ("bfly_2s5s10s_neg",  "Bfly 2-5-10 -", "SHY", "IEI", "IEF", "negative",
     "Negative butterfly 2s5s10s: short wings (SHY+IEF), long body (IEI)"),
    ("bfly_2s10s30s_pos", "Bfly 2-10-30 +", "SHY", "IEF", "TLT", "positive",
     "Positive butterfly 2s10s30s: long wings (SHY+TLT), short body (IEF)"),
    ("bfly_2s10s30s_neg", "Bfly 2-10-30 -", "SHY", "IEF", "TLT", "negative",
     "Negative butterfly 2s10s30s: short wings (SHY+TLT), long body (IEF)"),
    ("bfly_5s10s30s_pos", "Bfly 5-10-30 +", "IEI", "IEF", "TLT", "positive",
     "Positive butterfly 5s10s30s: long wings (IEI+TLT), short body (IEF)"),
    ("bfly_5s10s30s_neg", "Bfly 5-10-30 -", "IEI", "IEF", "TLT", "negative",
     "Negative butterfly 5s10s30s: short wings (IEI+TLT), long body (IEF)"),
]
for name, display, w1, body, w2, direction, desc in _butterflies:
    STRATEGY_POOL.append(_make_butterfly(name, display, w1, body, w2, direction, desc))

# ── D) 4 Directional (baseline) ─────────────────────────────────────────
STRATEGY_POOL.append(_make_directional(
    "bh_tlt", "B&H TLT", {"TLT": 1.0},
    "Buy & Hold TLT (long duration baseline)"))
STRATEGY_POOL.append(_make_directional(
    "bh_shv", "B&H SHV", {"SHV": 1.0},
    "Buy & Hold SHV (cash-like baseline)"))
STRATEGY_POOL.append(_make_directional(
    "bh_ief", "B&H IEF", {"IEF": 1.0},
    "Buy & Hold IEF (intermediate baseline)"))
STRATEGY_POOL.append(_make_directional(
    "bh_50_50", "50/50 TLT/SHV", {"TLT": 1.0, "SHV": 1.0},
    "50/50 TLT + SHV (balanced baseline)"))

# ── E) 3 Barbell ────────────────────────────────────────────────────────
STRATEGY_POOL.append(_make_barbell(
    "barbell_shy_tlt", "Barbell SHY+TLT", "SHY", "TLT",
    "Duration-neutral barbell: SHY + TLT"))
STRATEGY_POOL.append(_make_barbell(
    "barbell_shy_ief", "Barbell SHY+IEF", "SHY", "IEF",
    "Duration-neutral barbell: SHY + IEF"))
STRATEGY_POOL.append(_make_barbell(
    "barbell_iei_tlt", "Barbell IEI+TLT", "IEI", "TLT",
    "Duration-neutral barbell: IEI + TLT"))


# ══════════════════════════════════════════════════════════════════════════
# LOOKUP FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def get_strategy(name: str) -> dict:
    """Get a strategy by name. Raises KeyError if not found."""
    for s in STRATEGY_POOL:
        if s["name"] == name:
            return s
    raise KeyError(f"Strategy '{name}' not found in pool")


def list_strategies(stype: str = None) -> list:
    """List strategy names, optionally filtered by type.

    Args:
        stype: filter by type ('spread', 'butterfly', 'directional', 'barbell')
    """
    if stype is None:
        return [s["name"] for s in STRATEGY_POOL]
    return [s["name"] for s in STRATEGY_POOL if s["type"] == stype]


# ── Sanity check ─────────────────────────────────────────────────────────
assert len(STRATEGY_POOL) == 25, f"Expected 25 strategies, got {len(STRATEGY_POOL)}"
