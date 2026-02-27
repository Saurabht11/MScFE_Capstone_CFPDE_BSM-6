"""Black–Scholes–Merton call pricing + Greeks used for calibration weights."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class BSMParams:
    sigma: float = 0.16
    r: float = 0.035
    q: float = 0.021


def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))


def _d2(d1: float, T: float, sigma: float) -> float:
    return d1 - sigma * sqrt(T)


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")

    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, T, sigma)

    if option_type.lower() != "call":
        raise ValueError("Call-only pipeline: option_type must be 'call'")
    return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """dPrice/dSigma for calls."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = _d1(S, K, T, r, q, sigma)
    return S * exp(-q * T) * norm.pdf(d1) * sqrt(T)


def vectorized_price(S, K, T, r, q, sigma, option_type_arr):
    """Vectorized call-pricing wrapper (numpy arrays) for speed."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    option_type_arr = np.asarray(option_type_arr)

    out = np.full_like(S, fill_value=np.nan, dtype=float)
    ok = (S > 0) & (K > 0) & (T > 0) & (sigma > 0)

    if not np.any(ok):
        return out

    d1 = (np.log(S[ok] / K[ok]) + (r - q + 0.5 * sigma**2) * T[ok]) / (sigma * np.sqrt(T[ok]))
    d2 = d1 - sigma * np.sqrt(T[ok])

    is_call = option_type_arr[ok].astype(str) == "call"

    out_ok = np.full_like(d1, np.nan, dtype=float)
    out_ok[is_call] = S[ok][is_call] * np.exp(-q * T[ok][is_call]) * norm.cdf(d1[is_call]) - K[ok][is_call] * np.exp(-r * T[ok][is_call]) * norm.cdf(d2[is_call])

    out[ok] = out_ok
    return out
