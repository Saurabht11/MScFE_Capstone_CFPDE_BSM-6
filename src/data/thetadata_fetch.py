"""ThetaData (ThetaTerminal v3) SPX EOD downloader.

This module exposes reusable functions for downloading and assembling
SPX EOD data into the project-wide wide-format schema.

Key output:
- combined_options_data.csv (wide format):
  QUOTE_DATE, EXPIRE_DATE, STRIKE, UNDERLYING_LAST, DTE,
  C_BID, C_ASK, C_MID
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, Tuple

import httpx
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


@dataclass(frozen=True)
class ThetaConfig:
    base_url: str = "http://localhost:25503/v3"
    symbol: str = "SPX"
    strike_range: int = 60
    max_dte: int = 550
    timeout_index_s: int = 120
    timeout_option_s: int = 300


def _yyyymmdd(d: date) -> str:
    return d.strftime("%Y%m%d")


def month_chunks(start: date, end: date) -> Iterator[Tuple[date, date]]:
    """Yield [chunk_start, chunk_end] split by month to avoid large request failures."""
    cur = date(start.year, start.month, 1)
    if cur < start:
        cur = start
    while cur <= end:
        nxt = date(cur.year, cur.month, 1) + relativedelta(months=1)
        chunk_end = min(end, nxt - relativedelta(days=1))
        yield cur, chunk_end
        cur = nxt


def fetch_csv_stream(url: str, params: dict, timeout_s: int = 180) -> pd.DataFrame:
    """Stream CSV line-by-line and return as DataFrame."""
    rows = []
    header = None

    with httpx.stream("GET", url, params=params, timeout=timeout_s) as resp:
        if resp.status_code != 200:
            try:
                err_text = resp.read().decode("utf-8", errors="replace")
            except Exception:
                err_text = "<could not read response body>"

            raise RuntimeError(
                f"HTTP {resp.status_code} from ThetaTerminal.\n"
                f"URL: {resp.request.url}\n"
                f"Response:\n{err_text}"
            )

        for line in resp.iter_lines():
            if not line:
                continue
            if header is None:
                header = next(csv.reader([line]))
                continue
            rows.append(next(csv.reader([line])))

    if header is None:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=header)


def safe_to_date(series: pd.Series) -> pd.Series:
    """Parse mixed ISO8601 timestamps/dates safely."""
    try:
        return pd.to_datetime(series, format="mixed", errors="coerce").dt.date
    except TypeError:
        return pd.to_datetime(series, errors="coerce").dt.date


def get_index_eod(cfg: ThetaConfig, start: date, end: date) -> pd.DataFrame:
    """GET /v3/index/history/eod -> QUOTE_DATE, UNDERLYING_LAST"""
    url = f"{cfg.base_url}/index/history/eod"
    params = {
        "symbol": cfg.symbol,
        "start_date": _yyyymmdd(start),
        "end_date": _yyyymmdd(end),
        "format": "csv",
    }
    df = fetch_csv_stream(url, params=params, timeout_s=cfg.timeout_index_s)
    if df.empty:
        return df

    if "created" not in df.columns or "close" not in df.columns:
        raise RuntimeError(f"Unexpected index EOD schema. Columns: {list(df.columns)}")

    df["QUOTE_DATE"] = safe_to_date(df["created"])
    df["UNDERLYING_LAST"] = pd.to_numeric(df["close"], errors="coerce")

    df = df.dropna(subset=["QUOTE_DATE", "UNDERLYING_LAST"])
    df = df.drop_duplicates(subset=["QUOTE_DATE"])

    return df[["QUOTE_DATE", "UNDERLYING_LAST"]].sort_values("QUOTE_DATE")


def get_option_eod(cfg: ThetaConfig, start: date, end: date) -> pd.DataFrame:
    """GET /v3/option/history/eod -> long format: QUOTE_DATE, EXPIRE_DATE, STRIKE, RIGHT, BID, ASK"""
    url = f"{cfg.base_url}/option/history/eod"
    params = {
        "symbol": cfg.symbol,
        "expiration": "*",
        "strike": "*",
        "right": "call",
        "start_date": _yyyymmdd(start),
        "end_date": _yyyymmdd(end),
        "format": "csv",
        "strike_range": int(cfg.strike_range),
        "max_dte": int(cfg.max_dte),
    }

    df = fetch_csv_stream(url, params=params, timeout_s=cfg.timeout_option_s)
    if df.empty:
        return df

    required = {"created", "expiration", "strike", "right", "bid", "ask"}
    if not required.issubset(df.columns):
        raise RuntimeError(
            f"Unexpected option EOD schema. Missing {required - set(df.columns)}. "
            f"Columns: {list(df.columns)}"
        )

    df["QUOTE_DATE"] = safe_to_date(df["created"])
    df["EXPIRE_DATE"] = safe_to_date(df["expiration"])
    df["STRIKE"] = pd.to_numeric(df["strike"], errors="coerce")
    df["BID"] = pd.to_numeric(df["bid"], errors="coerce")
    df["ASK"] = pd.to_numeric(df["ask"], errors="coerce")
    df["RIGHT"] = df["right"].astype(str).str.lower()

    df = df.dropna(subset=["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"])
    df = df[df["RIGHT"] == "call"]

    return df[["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "RIGHT", "BID", "ASK"]]


def _build_combined_options_csv_core(
    cfg: ThetaConfig,
    start: date,
    end: date,
    out_csv: str | Path,
) -> pd.DataFrame:
    """Build and save wide-format combined options dataset."""

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) Underlying EOD close
    idx_parts = [get_index_eod(cfg, cs, ce) for cs, ce in month_chunks(start, end)]
    idx = pd.concat(idx_parts, ignore_index=True) if idx_parts else pd.DataFrame()
    if idx.empty:
        raise RuntimeError("No index EOD data returned. Is ThetaTerminal running?")

    idx = idx.drop_duplicates(subset=["QUOTE_DATE"]).sort_values("QUOTE_DATE")

    # 2) Options EOD (by month)
    opt_parts = []
    chunks = list(month_chunks(start, end))
    for cs, ce in tqdm(chunks, desc=f"Downloading {cfg.symbol} option EOD by month"):
        dfm = get_option_eod(cfg, cs, ce)
        if not dfm.empty:
            opt_parts.append(dfm)

    if not opt_parts:
        raise RuntimeError("No option EOD data returned.")

    opt = pd.concat(opt_parts, ignore_index=True)

    # 3) Long → wide (calls only)
    calls = (
        opt[opt["RIGHT"] == "call"]
        .rename(columns={"BID": "C_BID", "ASK": "C_ASK"})
        .drop(columns=["RIGHT"])
    )
    combined = calls.copy()

    # 4) Join underlying
    combined = pd.merge(combined, idx, on="QUOTE_DATE", how="left")

    # 5) Mid prices
    combined["C_MID"] = (combined["C_BID"] + combined["C_ASK"]) / 2.0

    # 6) DTE
    combined["DTE"] = (
        pd.to_datetime(combined["EXPIRE_DATE"]) - pd.to_datetime(combined["QUOTE_DATE"])
    ).dt.days

    # 7) Basic cleaning (do NOT apply your modeling filters here; those belong in prep step)
    combined = combined.dropna(subset=["UNDERLYING_LAST", "QUOTE_DATE", "EXPIRE_DATE", "STRIKE"])
    combined = combined[(combined["DTE"] >= 0) & (combined["DTE"] <= cfg.max_dte)]

    bad_call = (
        combined["C_BID"].notna()
        & combined["C_ASK"].notna()
        & (combined["C_ASK"] < combined["C_BID"])
    )
    combined = combined[~bad_call]

    combined = combined.sort_values(["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"])
    combined.to_csv(out_csv, index=False)

    return combined


# -------------------------------------------------------------------
# Public wrapper (stable signature used by pipeline.py / notebooks)
# -------------------------------------------------------------------
def build_combined_options_csv(
    out_csv: str | Path,
    base_url: str = "http://localhost:25503/v3",
    symbol: str = "SPX",
    start_date: date | None = None,
    end_date: date | None = None,
    strike_range: int = 60,
    max_dte: int = 550,
    timeout_index_s: int = 120,
    timeout_option_s: int = 300,
) -> pd.DataFrame:
    """
    Download index + option EOD data from ThetaTerminal v3 and save as wide CSV.

    Output CSV schema (wide, call-only):
      QUOTE_DATE, EXPIRE_DATE, STRIKE, UNDERLYING_LAST, DTE,
      C_BID, C_ASK, C_MID
    """
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date must be provided")

    cfg = ThetaConfig(
        base_url=base_url,
        symbol=symbol,
        strike_range=int(strike_range),
        max_dte=int(max_dte),
        timeout_index_s=int(timeout_index_s),
        timeout_option_s=int(timeout_option_s),
    )
    return _build_combined_options_csv_core(cfg=cfg, start=start_date, end=end_date, out_csv=out_csv)
