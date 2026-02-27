"""Data preparation: wide → long, filters, buckets.

Input (wide):
- combined_options_data.csv
  QUOTE_DATE, EXPIRE_DATE, STRIKE, UNDERLYING_LAST, DTE,
  C_BID, C_ASK, C_MID

Output (long):
- prepared_long.parquet (recommended)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


OptionType = Literal["call"]


@dataclass(frozen=True)
class PrepConfig:
    max_rel_spread: float = 0.45
    min_moneyness: float = 0.90
    max_moneyness: float = 1.10
    min_dte: int = 7
    max_dte: int = 550
    otm_only: bool = True


DTE_BUCKETS = [
    (7, 14, "1w_2w"),
    (15, 30, "2w_1m"),
    (31, 60, "1m_2m"),
    (61, 90, "2m_3m"),
    (91, 180, "3m_6m"),
    (181, 270, "6m_9m"),
    (271, 365, "9m_12m"),
    (366, 10_000, "12m_plus"),
]


def assign_dte_bucket(
    dte_or_df: pd.Series | pd.DataFrame,
    dte_col: str = "DTE",
    out_col: str = "dte_bucket",
) -> pd.Series | pd.DataFrame:
    """Assign granular DTE buckets.

    Supports two call styles:
      1) assign_dte_bucket(series) -> Series of bucket labels
      2) assign_dte_bucket(df)     -> df with a new column `out_col`

    This dual-mode behavior keeps the pipeline robust when callers pass
    a full DataFrame vs. a single DTE Series.
    """

    if isinstance(dte_or_df, pd.DataFrame):
        df = dte_or_df.copy()
        dte = pd.to_numeric(df[dte_col], errors="coerce").astype(float)
        buckets = pd.Series(index=df.index, dtype="object")
        for lo, hi, label in DTE_BUCKETS:
            buckets[(dte >= lo) & (dte <= hi)] = label
        df[out_col] = buckets
        return df

    dte = pd.to_numeric(dte_or_df, errors="coerce").astype(float)
    buckets = pd.Series(index=dte.index, dtype="object")
    for lo, hi, label in DTE_BUCKETS:
        buckets[(dte >= lo) & (dte <= hi)] = label
    return buckets


def wide_to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    cols = ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "UNDERLYING_LAST", "DTE", "C_BID", "C_ASK"]
    if "C_MID" in df_wide.columns:
        cols.append("C_MID")
    calls = df_wide[cols].copy()

    if "C_MID" not in calls.columns:
        calls["C_MID"] = (pd.to_numeric(calls["C_BID"], errors="coerce") + pd.to_numeric(calls["C_ASK"], errors="coerce")) / 2.0

    df = calls.rename(columns={"C_BID": "BID", "C_ASK": "ASK", "C_MID": "MID"})
    df["option_type"] = "call"

    # Types
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"]).dt.date
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"]).dt.date
    df["STRIKE"] = pd.to_numeric(df["STRIKE"], errors="coerce")
    df["UNDERLYING_LAST"] = pd.to_numeric(df["UNDERLYING_LAST"], errors="coerce")
    df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce").astype("Int64")
    df["BID"] = pd.to_numeric(df["BID"], errors="coerce")
    df["ASK"] = pd.to_numeric(df["ASK"], errors="coerce")
    df["MID"] = pd.to_numeric(df["MID"], errors="coerce")

    # Derived fields
    df["spread"] = df["ASK"] - df["BID"]
    df["rel_spread"] = df["spread"] / df["MID"]
    df["moneyness"] = df["STRIKE"] / df["UNDERLYING_LAST"]
    df["T"] = df["DTE"].astype(float) / 365.0

    # Buckets
    df["dte_bucket"] = assign_dte_bucket(df["DTE"].astype(float))

    return df


def apply_filters(df: pd.DataFrame, cfg: PrepConfig) -> pd.DataFrame:
    # Basic quote sanity
    mask = (
        (df["ASK"] > 0)
        & (df["BID"] >= 0)
        & (df["ASK"] >= df["BID"])
        & (df["MID"] > 0)
        & df["STRIKE"].notna()
        & df["UNDERLYING_LAST"].notna()
        & df["DTE"].notna()
    )

    # Project filters
    mask &= df["rel_spread"] <= cfg.max_rel_spread
    mask &= (df["moneyness"] >= cfg.min_moneyness) & (df["moneyness"] <= cfg.max_moneyness)
    mask &= (df["DTE"] >= cfg.min_dte) & (df["DTE"] <= cfg.max_dte)
    mask &= df["dte_bucket"].notna()

    if cfg.otm_only:
        # Call-only pipeline: keep OTM/ATM calls.
        mask &= df["STRIKE"] >= df["UNDERLYING_LAST"]

    out = df.loc[mask].copy()

    # Helpful IDs
    out["contract_id"] = (
        out["QUOTE_DATE"].astype(str)
        + "|"
        + out["EXPIRE_DATE"].astype(str)
        + "|"
        + out["option_type"].astype(str)
        + "|"
        + out["STRIKE"].round(2).astype(str)
    )

    return out.reset_index(drop=True)


def scope_last_years(df: pd.DataFrame, years: int = 2) -> pd.DataFrame:
    df = df.copy()
    df["QUOTE_DATE_dt"] = pd.to_datetime(df["QUOTE_DATE"])
    max_date = df["QUOTE_DATE_dt"].max()
    cutoff = (max_date - pd.DateOffset(years=years)).normalize()
    df = df[df["QUOTE_DATE_dt"] >= cutoff]
    df = df.drop(columns=["QUOTE_DATE_dt"])
    return df.reset_index(drop=True)


def prepare_long_dataset(
    in_csv: str | Path,
    out_parquet: str | Path,
    cfg: PrepConfig = PrepConfig(),
    scope_years: int = 2,
) -> pd.DataFrame:
    in_csv = Path(in_csv)
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    df_wide = pd.read_csv(in_csv)
    df_long = wide_to_long(df_wide)
    df_long = apply_filters(df_long, cfg)

    # Hard-cut scope (no leakage rule)
    if scope_years is not None and scope_years > 0:
        df_long = scope_last_years(df_long, years=scope_years)

    df_long.to_parquet(out_parquet, index=False)
    return df_long


def scope_last_months(df: pd.DataFrame, months: int) -> pd.DataFrame:
    """Hard-cut dataset to only last `months` months based on max QUOTE_DATE.

    This is the paper-friendly 'no leakage' scope rule (all steps use only this subset).
    """
    if months is None or months <= 0:
        return df
    out = df.copy()
    out["QUOTE_DATE"] = pd.to_datetime(out["QUOTE_DATE"])
    max_dt = out["QUOTE_DATE"].max()
    if pd.isna(max_dt):
        return out
    cutoff = (max_dt - pd.DateOffset(months=int(months))).normalize()
    return out[out["QUOTE_DATE"] >= cutoff].reset_index(drop=True)
