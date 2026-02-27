
"""Remove obviously arbitrageable quotes (static no-arbitrage sanity checks).

Goal: keep the empirical section clean and defensible:
- Remove quotes violating basic bounds
- Remove severe vertical-spread / butterfly / calendar-arb violations in MID

This is NOT a full surface-arbitrage repair; it is a conservative filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _lower_upper_bounds(df: pd.DataFrame, r: float, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Theoretical bounds under no-arbitrage for European options (continuous carry)."""
    S = df["UNDERLYING_LAST"].to_numpy(dtype=float)
    K = df["STRIKE"].to_numpy(dtype=float)
    T = df["T"].to_numpy(dtype=float)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)

    is_call = (df["option_type"].to_numpy() == "call")
    # Call: max(0, S e^{-qT} - K e^{-rT}) <= C <= S e^{-qT}
    # Put : max(0, K e^{-rT} - S e^{-qT}) <= P <= K e^{-rT}
    lower = np.where(is_call, np.maximum(0.0, S * disc_q - K * disc_r), np.maximum(0.0, K * disc_r - S * disc_q))
    upper = np.where(is_call, S * disc_q, K * disc_r)
    return lower, upper


def filter_bounds(df: pd.DataFrame, r: float, q: float, tol: float = 1e-8) -> pd.DataFrame:
    """Drop rows violating basic option price bounds for BID/ASK/MID."""
    out = df.copy()
    lower, upper = _lower_upper_bounds(out, r=r, q=q)

    bid = out["BID"].to_numpy(dtype=float)
    ask = out["ASK"].to_numpy(dtype=float)
    mid = out["MID"].to_numpy(dtype=float)

    ok = (
        (mid + tol >= lower) & (mid - tol <= upper) &
        (bid + tol >= lower) & (ask - tol <= upper)
    )
    return out.loc[ok].reset_index(drop=True)


def filter_vertical_monotonicity(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Drop severe violations of monotonicity in strike per (QUOTE_DATE, EXPIRE_DATE, option_type).

    Calls must be non-increasing in K; puts must be non-decreasing in K.
    Uses MID only; we drop rows that deviate materially from a monotone projection.
    """
    keep = np.ones(len(df), dtype=bool)
    df = df.copy().reset_index(drop=True)

    gcols = ["QUOTE_DATE", "EXPIRE_DATE", "option_type"]
    for _, g in df.groupby(gcols, sort=False):
        idx = g.index.to_numpy()
        gg = g.sort_values("STRIKE")
        mids = gg["MID"].to_numpy(dtype=float)
        is_call = (gg["option_type"].iloc[0] == "call")

        if is_call:
            proj = np.minimum.accumulate(mids)  # enforce non-increasing (cummin)
        else:
            proj = np.maximum.accumulate(mids)  # enforce non-decreasing (cummax)

        rel = np.abs(mids - proj) / np.maximum(1.0, np.abs(mids))
        bad = rel > tol
        keep[gg.index.to_numpy()] &= ~bad

    return df.loc[keep].reset_index(drop=True)


def filter_butterfly_convexity(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """Drop severe butterfly-arbitrage signals (lack of convexity in strike) per slice.

    For European calls/puts, price is convex in strike K (second difference >= 0).
    We check MID on an equally-spaced-ish grid; for irregular spacing, we use a
    simple discrete second-difference on neighboring strikes.
    """
    df = df.copy().reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)

    gcols = ["QUOTE_DATE", "EXPIRE_DATE", "option_type"]
    for _, g in df.groupby(gcols, sort=False):
        gg = g.sort_values("STRIKE")
        Ks = gg["STRIKE"].to_numpy(dtype=float)
        V = gg["MID"].to_numpy(dtype=float)
        if len(gg) < 5:
            continue

        # discrete second difference on consecutive points
        # convex if V_{i+1} - 2V_i + V_{i-1} >= -tol
        d2 = V[2:] - 2 * V[1:-1] + V[:-2]
        bad_mid = d2 < -tol * np.maximum(1.0, np.abs(V[1:-1]))
        bad_idx = gg.index.to_numpy()[1:-1][bad_mid]
        keep[bad_idx] = False

    return df.loc[keep].reset_index(drop=True)


def filter_calendar_monotonicity(df: pd.DataFrame, tol: float = 1e-6, strike_round: float = 0.5) -> pd.DataFrame:
    """Drop severe calendar-arbitrage signals: option value non-decreasing with maturity T for fixed K.

    We group by (QUOTE_DATE, option_type, rounded STRIKE) and enforce MID increases with T.
    """
    df = df.copy().reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)
    df["_Kround"] = (df["STRIKE"] / strike_round).round() * strike_round

    gcols = ["QUOTE_DATE", "option_type", "_Kround"]
    for _, g in df.groupby(gcols, sort=False):
        gg = g.sort_values("T")
        mids = gg["MID"].to_numpy(dtype=float)
        proj = np.maximum.accumulate(mids)  # non-decreasing in T
        rel = np.abs(mids - proj) / np.maximum(1.0, np.abs(mids))
        bad = rel > tol
        keep[gg.index.to_numpy()] &= ~bad

    df = df.drop(columns=["_Kround"])
    return df.loc[keep].reset_index(drop=True)


def remove_arbitrageable_quotes(
    df: pd.DataFrame,
    r: float,
    q: float,
    tol_bounds: float = 1e-8,
    tol_mono: float = 1e-6,
    tol_convex: float = 1e-6,
    tol_calendar: float = 1e-6,
) -> pd.DataFrame:
    """Apply a conservative sequence of no-arbitrage filters."""
    out = filter_bounds(df, r=r, q=q, tol=tol_bounds)
    out = filter_vertical_monotonicity(out, tol=tol_mono)
    out = filter_butterfly_convexity(out, tol=tol_convex)
    out = filter_calendar_monotonicity(out, tol=tol_calendar)
    return out.reset_index(drop=True)
