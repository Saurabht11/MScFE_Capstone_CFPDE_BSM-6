"""Plotting utilities (matplotlib only).

Provided figures:
- RMSE comparison by DTE bucket (CF vs BSM)
- RMSE improvement by DTE bucket (BSM - CF)
- Daily RMSE time series on test month (CF vs BSM)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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


def save_pipeline_diagram(
    out_png: str | Path,
    total_period: str,
    total_rows: int,
    total_dates: int,
    calib_period: str,
    calib_rows: int,
    calib_dates: int,
    test_period: str,
    test_rows: int,
    test_dates: int,
):
    """Create a formal publication-style pipeline diagram."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    txt_main = "#0f172a"
    txt_muted = "#475569"
    arrow = "#334155"
    lane_specs = [
        ("Data Preparation", 0.03, "#edf2f7", "#1d4e89"),
        ("Calibration", 0.355, "#f8fafc", "#92400e"),
        ("Evaluation", 0.68, "#effcf6", "#0f766e"),
    ]

    lane_w = 0.29
    lane_y = 0.12
    lane_h = 0.78

    for label, x, face, edge in lane_specs:
        lane = FancyBboxPatch(
            (x, lane_y),
            lane_w,
            lane_h,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
            alpha=0.75,
        )
        ax.add_patch(lane)
        ax.text(
            x + 0.015,
            lane_y + lane_h - 0.035,
            label.upper(),
            fontsize=11,
            fontweight="bold",
            color=edge,
            ha="left",
            va="center",
        )

    def stage_box(x: float, y: float, w: float, h: float, title: str, body: str, edge: str):
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            linewidth=1.4,
            edgecolor=edge,
            facecolor="white",
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h * 0.67,
            title,
            fontsize=10.5,
            fontweight="bold",
            color=txt_main,
            ha="center",
            va="center",
        )
        ax.text(
            x + w / 2,
            y + h * 0.35,
            body,
            fontsize=9,
            color=txt_muted,
            ha="center",
            va="center",
        )

    def flow_polyline(points: list[tuple[float, float]], dashed: bool = False):
        """Draw a clean orthogonal connector with one arrowhead at the end."""
        if len(points) < 2:
            return

        linestyle = (0, (4, 3)) if dashed else "-"
        for i in range(len(points) - 2):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=arrow,
                linewidth=1.4,
                linestyle=linestyle,
                solid_capstyle="round",
                zorder=5,
            )

        arr = FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color=arrow,
            connectionstyle="arc3,rad=0.0",
            linestyle=linestyle,
        )
        ax.add_patch(arr)

    box_w = 0.245
    box_h = 0.12
    ys = [0.70, 0.51, 0.32]

    x_data = lane_specs[0][1] + 0.022
    x_cal = lane_specs[1][1] + 0.022
    x_eval = lane_specs[2][1] + 0.022

    stage_box(
        x_data,
        ys[0],
        box_w,
        box_h,
        "Raw Market Data",
        f"ThetaData SPX Calls\n{total_rows:,} rows · {total_dates} quote dates",
        lane_specs[0][3],
    )
    stage_box(
        x_data,
        ys[1],
        box_w,
        box_h,
        "ETL and Validation",
        "Wide-to-long transform\nQuote-quality controls",
        lane_specs[0][3],
    )
    stage_box(
        x_data,
        ys[2],
        box_w,
        box_h,
        "Filtering and Bucketing",
        "Moneyness, DTE, spread filters\n8 DTE buckets",
        lane_specs[0][3],
    )

    stage_box(
        x_cal,
        ys[0],
        box_w,
        box_h,
        "Calibration Set (Month 1)",
        f"{calib_period}\n{calib_rows:,} rows · {calib_dates} quote dates",
        lane_specs[1][3],
    )
    stage_box(
        x_cal,
        ys[1],
        box_w,
        box_h,
        "Alpha Calibration",
        "Estimate CF alpha by DTE bucket\nCached for reproducibility",
        lane_specs[1][3],
    )
    stage_box(
        x_cal,
        ys[2],
        box_w,
        box_h,
        "Locked Parameters",
        "BSM: sigma, r, q\nCF-PDE: short/base grids + alpha",
        lane_specs[1][3],
    )

    stage_box(
        x_eval,
        ys[0],
        box_w,
        box_h,
        "Test Set (Month 2)",
        f"{test_period}\n{test_rows:,} rows · {test_dates} quote dates",
        lane_specs[2][3],
    )
    stage_box(
        x_eval,
        ys[1],
        box_w,
        box_h,
        "Out-of-Sample Pricing",
        "Run CF-PDE and BSM on test month\nApply calibrated alpha map",
        lane_specs[2][3],
    )
    stage_box(
        x_eval,
        ys[2],
        box_w,
        box_h,
        "Report Outputs",
        "RMSE summaries, comparisons,\nfigures and publication tables",
        lane_specs[2][3],
    )

    for x in (x_data, x_cal, x_eval):
        cx = x + box_w / 2
        flow_polyline([(cx, ys[0]), (cx, ys[1] + box_h)])
        flow_polyline([(cx, ys[1]), (cx, ys[2] + box_h)])

    split_x, split_y, split_w, split_h = 0.45, 0.84, 0.10, 0.05
    split_box = FancyBboxPatch(
        (split_x, split_y),
        split_w,
        split_h,
        boxstyle="round,pad=0.005,rounding_size=0.008",
        linewidth=1.2,
        edgecolor="#64748b",
        facecolor="#f8fafc",
    )
    ax.add_patch(split_box)
    ax.text(
        split_x + split_w / 2,
        split_y + split_h / 2,
        "1M / 1M Split",
        fontsize=8.8,
        fontweight="bold",
        color=txt_main,
        ha="center",
        va="center",
    )

    split_mid_y = split_y + split_h / 2
    left_start_y = ys[2] + box_h / 2
    route_x = x_data + box_w + 0.02
    route_y = 0.895
    flow_polyline(
        [
            (x_data + box_w, left_start_y),
            (route_x, left_start_y),
            (route_x, route_y),
            (split_x, route_y),
            (split_x, split_mid_y),
        ]
    )

    cal_x = x_cal + box_w / 2
    eval_x = x_eval + box_w / 2
    split_left_x = split_x + split_w * 0.35
    split_right_x = split_x + split_w * 0.65
    split_drop_y = ys[0] + box_h + 0.01

    flow_polyline(
        [
            (split_left_x, split_y),
            (split_left_x, split_drop_y),
            (cal_x, split_drop_y),
            (cal_x, ys[0] + box_h),
        ]
    )
    flow_polyline(
        [
            (split_right_x, split_y),
            (split_right_x, split_drop_y),
            (eval_x, split_drop_y),
            (eval_x, ys[0] + box_h),
        ]
    )

    mid_y = ys[1] + box_h / 2
    flow_polyline(
        [
            (x_cal + box_w, mid_y),
            (x_cal + box_w + 0.015, mid_y),
            (x_eval - 0.015, mid_y),
            (x_eval, mid_y),
        ],
        dashed=True,
    )

    ax.text(
        0.5,
        0.955,
        "Figure M1: End-to-End Analytics Pipeline",
        fontsize=18,
        fontweight="bold",
        color=txt_main,
        ha="center",
        va="center",
    )
    ax.text(
        0.5,
        0.925,
        f"Analysis window: {total_period} | Split: 1 month calibration + 1 month out-of-sample test",
        fontsize=11,
        color=txt_muted,
        ha="center",
        va="center",
    )

    footer = FancyBboxPatch(
        (0.22, 0.045),
        0.56,
        0.045,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=1.0,
        edgecolor="#64748b",
        facecolor="#f8fafc",
    )
    ax.add_patch(footer)
    ax.text(
        0.5,
        0.067,
        "Models: CF-PDE (Caputo-Fabrizio time-fractional) vs BSM (Black-Scholes-Merton closed form)",
        fontsize=9.5,
        color=txt_main,
        ha="center",
        va="center",
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
