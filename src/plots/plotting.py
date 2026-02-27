"""Plotting utilities (matplotlib only, report-ready).

This project keeps plots minimal and interpretable:
- RMSE comparison by DTE bucket (CF vs BSM)
- RMSE improvement by DTE bucket (BSM - CF)
- Daily RMSE time series on test month (CF vs BSM)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def save_rmse_improvement_by_bucket(metrics_df: pd.DataFrame, out_png: str | Path, title: str):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = metrics_df.copy()
    if "dte_bucket" not in df.columns:
        raise ValueError("Expected dte_bucket column")

    df = df.sort_values("dte_bucket")

    plt.figure(figsize=(10, 4))
    plt.bar(df["dte_bucket"].astype(str), df["RMSE_impr"].astype(float))
    plt.axhline(0, linewidth=1)
    plt.xlabel("DTE bucket")
    plt.ylabel("RMSE_BSM − RMSE_CF")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_rmse_comparison_by_bucket(metrics_df: pd.DataFrame, out_png: str | Path, title: str):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = metrics_df.copy().sort_values("dte_bucket")
    x = range(len(df))

    plt.figure(figsize=(10, 4))
    plt.bar([i - 0.2 for i in x], df["RMSE_BSM"].astype(float), width=0.4, label="BSM")
    plt.bar([i + 0.2 for i in x], df["RMSE_CF"].astype(float), width=0.4, label="CF-PDE")
    plt.xticks(list(x), df["dte_bucket"].astype(str), rotation=0)
    plt.xlabel("DTE bucket")
    plt.ylabel("RMSE vs MID")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_daily_rmse_timeseries(metrics_daily: pd.DataFrame, out_png: str | Path, title: str):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = metrics_daily.copy()
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df = df.sort_values("QUOTE_DATE")

    plt.figure(figsize=(10, 4))
    plt.plot(df["QUOTE_DATE"], df["RMSE_BSM"], label="BSM")
    plt.plot(df["QUOTE_DATE"], df["RMSE_CF"], label="CF-PDE")
    plt.xlabel("Date")
    plt.ylabel("RMSE vs MID")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_price_means_by_bucket(price_by_bucket: pd.DataFrame, out_png: str | Path, title: str):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = price_by_bucket.copy().sort_values("dte_bucket")
    x = range(len(df))

    plt.figure(figsize=(11, 4))
    plt.bar([i - 0.25 for i in x], df["OBS_mean"].astype(float), width=0.25, label="Observed MID")
    plt.bar([i for i in x], df["BSM_mean"].astype(float), width=0.25, label="BSM")
    plt.bar([i + 0.25 for i in x], df["CF_mean"].astype(float), width=0.25, label="CF-PDE")
    plt.xticks(list(x), df["dte_bucket"].astype(str), rotation=0)
    plt.xlabel("DTE bucket")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_daily_price_timeseries(price_daily: pd.DataFrame, out_png: str | Path, title: str):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df = price_daily.copy()
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df = df.sort_values("QUOTE_DATE")

    plt.figure(figsize=(11, 4))
    plt.plot(df["QUOTE_DATE"], df["OBS_mean"], label="Observed MID")
    plt.plot(df["QUOTE_DATE"], df["BSM_mean"], label="BSM")
    plt.plot(df["QUOTE_DATE"], df["CF_mean"], label="CF-PDE")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(title)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
