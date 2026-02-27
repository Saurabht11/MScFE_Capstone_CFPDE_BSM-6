"""Alpha calibration (per DTE bucket).

We calibrate alpha by minimizing weighted SSE against MID prices.

Weight modes:
- 'spread'      : w = 1 / spread^2
- 'vega'        : w = vega
- 'vega_spread' : w = vega / spread^2


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..models import bsm as bsm_mod
from ..models import cf_pde as cf_mod


WeightMode = Literal["spread", "vega", "vega_spread"]


@dataclass(frozen=True)
class AlphaBounds:
    lo: float = 0.10
    hi: float = 0.99


@dataclass(frozen=True)
class CalibResult:
    alpha: float
    obj: float
    n_rows: int
    weight_mode: str


def _weights(df: pd.DataFrame, bsm_params: bsm_mod.BSMParams, mode: WeightMode) -> np.ndarray:
    spread = np.asarray(df["spread"], dtype=float)
    spread = np.where(spread <= 0, np.nan, spread)
    eps = 1e-10

    if mode == "spread":
        w = 1.0 / np.maximum(spread, eps) ** 2
    else:
        vega = np.array(
            [
                bsm_mod.vega(S=float(s), K=float(k), T=float(t), r=bsm_params.r, q=bsm_params.q, sigma=bsm_params.sigma)
                for s, k, t in zip(df["UNDERLYING_LAST"], df["STRIKE"], df["T"])
            ],
            dtype=float,
        )
        if mode == "vega":
            w = np.maximum(vega, 0.0)
        elif mode == "vega_spread":
            w = np.maximum(vega, 0.0) / np.maximum(spread, eps) ** 2
        else:
            raise ValueError("Unknown weight mode")

    w = np.where(np.isfinite(w), w, 0.0)
    return w


def objective_alpha(
    alpha: float,
    df: pd.DataFrame,
    cf_params_base: cf_mod.CFPDEParams,
    cf_params_short: cf_mod.CFPDEParams,
    short_dte_max: int,
    weights: np.ndarray,
) -> float:
    """
    Weighted MSE objective (lower is better).

    Uses cf_params_short when DTE <= short_dte_max, else cf_params_base.
    """
    preds = []
    for s0, k, t, opt_type, dte in zip(df["UNDERLYING_LAST"], df["STRIKE"], df["T"], df["option_type"], df["DTE"]):
        params = cf_params_short if (np.isfinite(dte) and float(dte) <= float(short_dte_max)) else cf_params_base
        try:
            p = cf_mod.price_cf(
                S0=float(s0),
                K=float(k),
                T=float(t),
                params=params,
                alpha=float(alpha),
                option_type=str(opt_type),
            )
        except Exception:
            p = np.nan
        preds.append(p)

    preds = np.asarray(preds, dtype=float)
    mid = np.asarray(df["MID"], dtype=float)

    ok = np.isfinite(preds) & np.isfinite(mid) & (weights > 0)
    if not np.any(ok):
        return float("inf")

    err2 = (preds[ok] - mid[ok]) ** 2
    w = weights[ok]
    return float(np.sum(w * err2) / np.sum(w))


def calibrate_alpha(
    df_bucket: pd.DataFrame,
    bsm_params: bsm_mod.BSMParams,
    cf_params_base: cf_mod.CFPDEParams,
    cf_params_short: cf_mod.CFPDEParams,
    short_dte_max: int = 30,
    bounds: AlphaBounds = AlphaBounds(),
    weight_mode: WeightMode = "spread",
    coarse_points: int = 5,
) -> CalibResult:
    """
    Calibrate alpha on a bucket (single DataFrame) with:
      1) coarse grid search
      2) bounded scalar refinement

    Returns weighted MSE objective value.
    """
    df_bucket = df_bucket.copy().reset_index(drop=True)
    if len(df_bucket) < 5:
        return CalibResult(alpha=float("nan"), obj=float("inf"), n_rows=len(df_bucket), weight_mode=weight_mode)

    weights = _weights(df_bucket, bsm_params=bsm_params, mode=weight_mode)

    grid = np.linspace(bounds.lo, bounds.hi, int(coarse_points))
    grid_obj = []
    for a in grid:
        grid_obj.append(objective_alpha(a, df_bucket, cf_params_base, cf_params_short, short_dte_max, weights))
    grid_obj = np.asarray(grid_obj, dtype=float)

    # robust fallback
    try:
        best_idx = int(np.nanargmin(grid_obj))
    except Exception:
        best_idx = 0

    # refine within full bounds (simple, robust)
    res = minimize_scalar(
        lambda a: objective_alpha(a, df_bucket, cf_params_base, cf_params_short, short_dte_max, weights),
        bounds=(bounds.lo, bounds.hi),
        method="bounded",
        options={"xatol": 5e-3, "maxiter": 24},
    )

    if res.success and np.isfinite(res.x):
        return CalibResult(alpha=float(res.x), obj=float(res.fun), n_rows=len(df_bucket), weight_mode=weight_mode)

    # if refinement fails, return best grid point
    a0 = float(grid[best_idx])
    return CalibResult(alpha=a0, obj=float(grid_obj[best_idx]), n_rows=len(df_bucket), weight_mode=weight_mode)
