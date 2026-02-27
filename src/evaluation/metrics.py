"""Evaluation utilities (basic + explainable).

This pipeline focuses on a single primary comparison metric:
- RMSE of model price vs MID, on the out-of-sample test month.

We still keep contract-level outputs (P_CF, P_BSM, MID, ERR_*) for auditing.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from ..models import bsm as bsm_mod
from ..models import cf_pde as cf_mod


def rmse_arr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x * x)))


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _mae_arr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(np.abs(x)))


def add_predictions_basic(
    df: pd.DataFrame,
    alpha_by_bucket: Dict[str, float],
    bsm_params: bsm_mod.BSMParams,
    cf_params_selector: Callable[[float], cf_mod.CFPDEParams],
) -> pd.DataFrame:
    """
    Add:
      - P_BSM (closed form)
      - P_CF  (CF-PDE using per-bucket alpha; PDE grid chosen by DTE)
      - ERR_* columns
    """
    df = df.copy().reset_index(drop=True)

    # BSM (fast)
    df["P_BSM"] = bsm_mod.vectorized_price(
        S=df["UNDERLYING_LAST"].values,
        K=df["STRIKE"].values,
        T=df["T"].values,
        r=bsm_params.r,
        q=bsm_params.q,
        sigma=bsm_params.sigma,
        option_type_arr=df["option_type"].values,
    )

    # CF-PDE (loop)
    P_cf = np.full(len(df), np.nan, dtype=float)
    alpha_used = np.full(len(df), np.nan, dtype=float)

    for i, row in enumerate(df.itertuples(index=False)):
        bucket = str(getattr(row, "dte_bucket"))
        alpha = alpha_by_bucket.get(bucket, np.nan)
        alpha_used[i] = alpha
        if not np.isfinite(alpha):
            continue

        dte = float(getattr(row, "DTE"))
        params = cf_params_selector(dte)

        try:
            P_cf[i] = cf_mod.price_cf(
                S0=float(getattr(row, "UNDERLYING_LAST")),
                K=float(getattr(row, "STRIKE")),
                T=float(getattr(row, "T")),
                params=params,
                alpha=float(alpha),
                option_type=str(getattr(row, "option_type")),
            )
        except Exception:
            P_cf[i] = np.nan

    df["alpha"] = alpha_used
    df["P_CF"] = P_cf

    df["ERR_CF"] = df["P_CF"] - df["MID"]
    df["ERR_BSM"] = df["P_BSM"] - df["MID"]
    return df


def rmse_summaries(eval_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      1) metrics_by_bucket: dte_bucket, n, alpha_mean, RMSE_CF, RMSE_BSM, RMSE_impr
      2) metrics_daily: QUOTE_DATE, RMSE_CF, RMSE_BSM, RMSE_impr
      3) metrics_overall: single-row overall RMSEs
    """
    df = eval_rows.copy()
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"]).dt.normalize()

    # by bucket
    g = df.groupby("dte_bucket", dropna=False)
    metrics_by_bucket = g.agg(
        n=("contract_id", "count"),
        alpha_mean=("alpha", "mean"),
        RMSE_CF=("ERR_CF", lambda s: rmse_arr(s.values)),
        RMSE_BSM=("ERR_BSM", lambda s: rmse_arr(s.values)),
    ).reset_index()
    metrics_by_bucket["RMSE_impr"] = metrics_by_bucket["RMSE_BSM"] - metrics_by_bucket["RMSE_CF"]
    metrics_by_bucket = metrics_by_bucket.sort_values("dte_bucket").reset_index(drop=True)

    # daily overall (aggregate across buckets)
    gd = df.groupby("QUOTE_DATE", dropna=False)
    metrics_daily = gd.agg(
        n=("contract_id", "count"),
        RMSE_CF=("ERR_CF", lambda s: rmse_arr(s.values)),
        RMSE_BSM=("ERR_BSM", lambda s: rmse_arr(s.values)),
    ).reset_index()
    metrics_daily["RMSE_impr"] = metrics_daily["RMSE_BSM"] - metrics_daily["RMSE_CF"]
    metrics_daily = metrics_daily.sort_values("QUOTE_DATE").reset_index(drop=True)

    # overall
    overall = pd.DataFrame([{
        "n": int(df["contract_id"].count()),
        "RMSE_CF": rmse_arr(df["ERR_CF"].values),
        "RMSE_BSM": rmse_arr(df["ERR_BSM"].values),
    }])
    overall["RMSE_impr"] = overall["RMSE_BSM"] - overall["RMSE_CF"]

    return metrics_by_bucket, metrics_daily, overall


def price_comparison_summaries(eval_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      1) by_bucket: observed/model means + bias/MAE/RMSE by DTE bucket
      2) daily: observed/model means + bias/MAE/RMSE by day
      3) overall: one-row global summary
    """
    df = eval_rows.copy()
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"]).dt.normalize()

    def _summary_block(g: pd.DataFrame) -> pd.Series:
        obs = np.asarray(g["MID"], dtype=float)
        p_bsm = np.asarray(g["P_BSM"], dtype=float)
        p_cf = np.asarray(g["P_CF"], dtype=float)
        err_bsm = p_bsm - obs
        err_cf = p_cf - obs
        return pd.Series(
            {
                "n": int(len(g)),
                "OBS_mean": _nanmean(obs),
                "BSM_mean": _nanmean(p_bsm),
                "CF_mean": _nanmean(p_cf),
                "BIAS_BSM": _nanmean(err_bsm),
                "BIAS_CF": _nanmean(err_cf),
                "MAE_BSM": _mae_arr(err_bsm),
                "MAE_CF": _mae_arr(err_cf),
                "RMSE_BSM": rmse_arr(err_bsm),
                "RMSE_CF": rmse_arr(err_cf),
            }
        )

    by_bucket_rows = []
    for bucket, g in df.groupby("dte_bucket", dropna=False):
        row = _summary_block(g).to_dict()
        row["dte_bucket"] = bucket
        by_bucket_rows.append(row)
    by_bucket = pd.DataFrame(by_bucket_rows)
    by_bucket = by_bucket.sort_values("dte_bucket").reset_index(drop=True)

    daily_rows = []
    for qd, g in df.groupby("QUOTE_DATE", dropna=False):
        row = _summary_block(g).to_dict()
        row["QUOTE_DATE"] = qd
        daily_rows.append(row)
    daily = pd.DataFrame(daily_rows)
    daily = daily.sort_values("QUOTE_DATE").reset_index(drop=True)

    overall = _summary_block(df).to_frame().T.reset_index(drop=True)

    return by_bucket, daily, overall
