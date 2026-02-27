"""Caputo–Fabrizio time-fractional Black–Scholes PDE solver (implicit / semi-implicit).

We implement the same numerical style as your stable CF scheme:
- Time variable: tau = T - t (tau=0 at maturity, tau=T at valuation)
- Memory state: Z
- Alpha parameter: alpha in (0, 1)
- Implicit solve per time step using banded tridiagonal system.

This is intended for **EOD option pricing comparisons** (not HFT).
Grid sizes default to your 'fast profile' (I=28, J=70).

References:
- Your CF implicit solver notebook/PDF (class CFStableBS)
- Caputo–Fabrizio derivative formulation used in the report
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp
from typing import Literal

import numpy as np
from scipy.linalg import solve_banded


OptionType = Literal["call"]


@dataclass(frozen=True)
class CFPDEParams:
    sigma: float = 0.16
    r: float = 0.035
    q: float = 0.021
    I: int = 28
    J: int = 70


@lru_cache(maxsize=128)
def _build_operator(I: int, sigma: float, r: float, q: float):
    """Build tridiagonal coefficients a,b,c for i=1..I-1.

    Using the standard BS operator in S-space with uniform grid S_i = i*dS, the dS cancels
    from coefficients when written in terms of index i (as in your CFStableBS code).
    """
    a = np.zeros(I + 1, dtype=float)
    b = np.zeros(I + 1, dtype=float)
    c = np.zeros(I + 1, dtype=float)
    for i in range(1, I):
        a[i] = 0.5 * sigma**2 * i**2 - 0.5 * (r - q) * i
        b[i] = -sigma**2 * i**2 - r
        c[i] = 0.5 * sigma**2 * i**2 + 0.5 * (r - q) * i
    return a, b, c


def solve_cf_surface(
    Smax: float,
    K: float,
    T: float,
    params: CFPDEParams,
    alpha: float,
    option_type: OptionType,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve CF-PDE and return (S_grid, V_grid at t=0)."""
    if option_type != "call":
        raise ValueError("Call-only pipeline: option_type must be 'call'")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if T <= 0:
        raise ValueError("T must be positive")
    if Smax <= 0 or K <= 0:
        raise ValueError("Smax and K must be positive")

    I, J = params.I, params.J
    dt = T / J
    S = np.linspace(0.0, Smax, I + 1)

    sigma, r, q = params.sigma, params.r, params.q
    a, b, c = _build_operator(I, sigma, r, q)

    # terminal payoff at maturity (tau=0)
    V = np.maximum(S - K, 0.0)

    Z = np.zeros_like(V)
    lam = alpha / (1.0 - alpha)

    # banded matrix storage (3, I-1)
    ab = np.zeros((3, I - 1), dtype=float)

    for j in range(J):
        tau = (j + 1) * dt

        # matrix coefficients for this alpha, dt
        for i in range(1, I):
            ab[0, i - 1] = -(1.0 - alpha) * dt * c[i]   # upper
            ab[1, i - 1] = 1.0 - (1.0 - alpha) * dt * b[i]  # diag
            ab[2, i - 1] = -(1.0 - alpha) * dt * a[i]   # lower

        rhs = V[1:I] + alpha * dt * Z[1:I]

        # boundary conditions at tau (call)
        V0 = 0.0
        VI = Smax * exp(-q * tau) - K * exp(-r * tau)

        # incorporate boundaries into RHS
        rhs[0] -= -(1.0 - alpha) * dt * a[1] * V0
        rhs[-1] -= -(1.0 - alpha) * dt * c[I - 1] * VI

        V_new = V.copy()
        V_new[1:I] = solve_banded((1, 1), ab, rhs)
        V_new[0] = V0
        V_new[I] = VI

        # memory update
        Z_new = (Z + (V_new - V)) / (1.0 + lam * dt)

        V, Z = V_new, Z_new

    return S, V


def price_cf(
    S0: float,
    K: float,
    T: float,
    params: CFPDEParams,
    alpha: float,
    option_type: OptionType,
    Smax: float | None = None,
) -> float:
    """Price a single option contract by solving CF-PDE and interpolating V(S0)."""
    if Smax is None:
        Smax = max(3.0 * S0, 2.0 * K)

    S_grid, V_grid = solve_cf_surface(Smax=Smax, K=K, T=T, params=params, alpha=alpha, option_type=option_type)
    # linear interpolation on grid
    return float(np.interp(S0, S_grid, V_grid))
