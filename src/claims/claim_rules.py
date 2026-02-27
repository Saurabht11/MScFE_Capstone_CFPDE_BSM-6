"""Validation framework for CF-PDE pricing comparisons against BSM.

This module evaluates whether the Caputo-Fabrizio time-fractional PDE model
achieves superior pricing performance over the classical Black-Scholes-Merton
benchmark using direct metric comparison and calibration stability assessment.

Performance validation criteria:

1. Primary Error Reduction: MAE_CF < MAE_BSM AND RMSE_CF < RMSE_BSM
   - Quantifies improvement in both average and squared pricing errors

2. Bid-Ask Spread Alignment: InsideBidAsk_CF > InsideBidAsk_BSM and 
   median normalized error improves
   - Ensures predicted prices are closer to observed mid-quote

3. Calibration Stability: Alpha estimates demonstrate low boundary hit rate and
   low coefficient of variation across time periods
   - Validates that fractional parameters remain well-identified and stable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClaimConfig:
    # Alpha stability thresholds
    bound_lo: float = 0.10
    bound_hi: float = 0.99
    max_boundary_hit_rate: float = 0.20
    max_cv: float = 0.35  # Coefficient of variation: std/abs(mean)




def alpha_stability_summary(alpha_hist: pd.DataFrame, cfg: ClaimConfig) -> pd.DataFrame:
    """Quantify stability of calibrated alpha parameters over time.
    
    Assesses whether fractional-order parameters remain well-identified and stable
    by computing boundary hit rates and coefficient of variation per DTE bucket.
    
    Parameters
    ----------
    alpha_hist : pd.DataFrame
        Time series of calibrated alpha values with columns: QUOTE_DATE, dte_bucket, alpha
    cfg : ClaimConfig
        Configuration containing boundary and stability thresholds
        
    Returns
    -------
    pd.DataFrame
        Stability summary with columns: dte_bucket, alpha_mean, alpha_std, 
        boundary_hit_rate, n_days, alpha_cv, stable
    """
    df = alpha_hist.copy()
    df = df[np.isfinite(df["alpha"])]

    def boundary_hit(a: float) -> int:
        return int((a <= cfg.bound_lo + 1e-6) or (a >= cfg.bound_hi - 1e-6))

    df["boundary_hit"] = df["alpha"].apply(boundary_hit)

    g = df.groupby("dte_bucket", dropna=False)
    out = g.agg(
        alpha_mean=("alpha", "mean"),
        alpha_std=("alpha", "std"),
        boundary_hit_rate=("boundary_hit", "mean"),
        n_days=("alpha", "count"),
    ).reset_index()

    out["alpha_cv"] = out["alpha_std"] / out["alpha_mean"].abs().replace(0, np.nan)
    out["stable"] = (out["boundary_hit_rate"] <= cfg.max_boundary_hit_rate) & (out["alpha_cv"] <= cfg.max_cv)
    return out


def build_claim_table(
    eval_rows: pd.DataFrame,
    alpha_hist: Optional[pd.DataFrame] = None,
    cfg: ClaimConfig = ClaimConfig(),
) -> pd.DataFrame:
    """Evaluate pricing improvements across DTE buckets using multi-gate validation.
    
    Compares CF-PDE and BSM performance across three validation criteria:
    (1) error metrics (MAE, RMSE), (2) bid-ask spread alignment, and 
    (3) calibration stability. CF is considered to add value when all gates pass.
    
    Parameters
    ----------
    eval_rows : pd.DataFrame
        Test set evaluation data with columns: dte_bucket, ABS_ERR_CF, ABS_ERR_BSM,
        ERR_CF, ERR_BSM, InsideBidAsk_CF, InsideBidAsk_BSM, NormErr_CF, NormErr_BSM
    alpha_hist : Optional[pd.DataFrame]
        Time series of calibrated alphas (required for stability gate). 
        Columns: QUOTE_DATE, dte_bucket, alpha
    cfg : ClaimConfig
        Configuration for stability thresholds
        
    Returns
    -------
    pd.DataFrame
        Claim summary with columns: dte_bucket, n, MAE_CF, MAE_BSM, RMSE_CF, RMSE_BSM,
        Inside_CF, Inside_BSM, MedNorm_CF, MedNorm_BSM, Gate_1_MAE_RMSE, 
        Gate_2_SpreadMetrics, Gate_5_AlphaStable, CF_Adds_Value
    """
    for bucket, dfb in eval_rows.groupby("dte_bucket", dropna=False):
        dfb = dfb.copy()

        # core metrics
        mae_cf = float(dfb["ABS_ERR_CF"].mean())
        mae_bsm = float(dfb["ABS_ERR_BSM"].mean())
        rmse_cf = float(np.sqrt(np.mean(dfb["ERR_CF"].dropna().values ** 2))) if dfb["ERR_CF"].notna().any() else float("nan")
        rmse_bsm = float(np.sqrt(np.mean(dfb["ERR_BSM"].dropna().values ** 2))) if dfb["ERR_BSM"].notna().any() else float("nan")

        inside_cf = float(dfb["InsideBidAsk_CF"].mean())
        inside_bsm = float(dfb["InsideBidAsk_BSM"].mean())
        mednorm_cf = float(dfb["NormErr_CF"].median())
        mednorm_bsm = float(dfb["NormErr_BSM"].median())



        # validation gates
        gate_1 = (mae_cf < mae_bsm) and (rmse_cf < rmse_bsm)
        gate_2 = (inside_cf > inside_bsm) and (mednorm_cf < mednorm_bsm)

        rows.append(
            dict(
                dte_bucket=bucket,
                n=len(dfb),
                MAE_CF=mae_cf,
                MAE_BSM=mae_bsm,
                RMSE_CF=rmse_cf,
                RMSE_BSM=rmse_bsm,
                Inside_CF=inside_cf,
                Inside_BSM=inside_bsm,
                MedNorm_CF=mednorm_cf,
                MedNorm_BSM=mednorm_bsm,
                Gate_1_MAE_RMSE=gate_1,
                Gate_2_SpreadMetrics=gate_2,
            )
        )

    claim = pd.DataFrame(rows)

    # alpha stability gate (rolling only)
    if alpha_hist is not None and not alpha_hist.empty:
        stab = alpha_stability_summary(alpha_hist, cfg=cfg)
        claim = claim.merge(stab[["dte_bucket", "stable", "boundary_hit_rate", "alpha_cv"]], on="dte_bucket", how="left")
        claim["Gate_5_AlphaStable"] = claim["stable"].fillna(False)
    else:
        claim["Gate_5_AlphaStable"] = False

    claim["CF_Adds_Value"] = (
        claim["Gate_1_MAE_RMSE"]
        & claim["Gate_2_SpreadMetrics"]
        & claim["Gate_5_AlphaStable"]
    )

    # tidy
    return claim.sort_values("dte_bucket")
