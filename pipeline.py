"""
MScFE Capstone — empirical replication pipeline for call-option pricing.

The pipeline compares a Caputo–Fabrizio time-fractional PDE model with
classical Black–Scholes–Merton (BSM) under a fixed design:
- Use only the most recent 2 months of quotes (hard cut by max(QUOTE_DATE)).
- Calibrate α by DTE bucket on month 1.
- Evaluate out-of-sample on month 2.
- Use BSM closed form with constant (sigma, r, q) as benchmark.
- Report RMSE-focused model comparison outputs.

Operational settings:
- Higher PDE resolution for short maturities (DTE <= 30): I=60, J=200.
- Wider moneyness coverage and no OTM-only restriction.
- Calibration and evaluation caching for reproducibility and runtime control.

Outputs (default): results/
  - results/prepared_long_2m.parquet
  - results/calibration/alpha_by_bucket.csv
  - results/test/metrics_rmse_by_bucket.csv
  - results/test/metrics_rmse_daily.csv
  - results/test/price_comparison_by_bucket.csv
  - results/test/price_comparison_daily.csv
  - results/test/price_comparison_overall.csv
  - results/plots/*.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.thetadata_fetch import build_combined_options_csv
from src.data.prep import PrepConfig, wide_to_long, apply_filters, scope_last_months
from src.data.arbitrage import remove_arbitrageable_quotes
from src.models.bsm import BSMParams
from src.models.cf_pde import CFPDEParams
from src.calibration.alpha_calibration import calibrate_alpha, AlphaBounds
from src.evaluation.metrics import add_predictions_basic, rmse_summaries, price_comparison_summaries
from src.plots import plotting as plot_mod


# -----------------------
# Configuration
# -----------------------
@dataclass(frozen=True)
class ProjectConfig:
    # Scope + split
    total_months: int = 2
    calib_months: int = 1

    # Data filters (mitigate weak alpha-identifiability in short maturities)
    option_type: str = "call"
    max_rel_spread: float = 0.45
    min_moneyness: float = 0.85
    max_moneyness: float = 1.15
    min_dte: int = 7
    max_dte: int = 550
    otm_only: bool = False  # disabled to increase signal (esp. short maturities)

    # BSM constants
    sigma: float = 0.16
    r: float = 0.035
    q: float = 0.021

    # CF PDE grids
    I_base: int = 28
    J_base: int = 70
    I_short: int = 60
    J_short: int = 200
    short_dte_max: int = 30  # 1w–1m bucket range

    # Alpha calibration
    alpha_lo: float = 0.10
    alpha_hi: float = 0.99
    weight_mode: str = "spread"  # spread | vega | vega_spread
    coarse_points: int = 3
    max_calib_rows_per_bucket: int = 120
    short_bucket_ensemble_runs: int = 1

    # Test evaluation runtime cap (optional; set None to use all)
    max_test_rows_per_bucket: Optional[int] = 300

    # Caching
    force_recalibrate: bool = False

    # If False and cached eval rows exist (newer than alpha), skip re-pricing test
    force_reprice_test: bool = False

    # Disable if matplotlib backend/file permissions are problematic in headless runs
    enable_plots: bool = True


def _ensure_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"])
    return df


def split_calib_test(df_long_scoped: pd.DataFrame, total_months: int, calib_months: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the last `total_months` window into:
      - calibration: first `calib_months` month(s)
      - test: remaining months
    """
    df = _ensure_dates(df_long_scoped)

    max_dt = df["QUOTE_DATE"].max().normalize()
    start_total = (max_dt - pd.DateOffset(months=int(total_months))).normalize()
    start_test = (max_dt - pd.DateOffset(months=int(total_months - calib_months))).normalize()

    calib = df[(df["QUOTE_DATE"] >= start_total) & (df["QUOTE_DATE"] < start_test)].reset_index(drop=True)
    test = df[df["QUOTE_DATE"] >= start_test].reset_index(drop=True)
    return calib, test


def _sample_cap(df: pd.DataFrame, group_col: str, max_rows: Optional[int], seed: int = 42) -> pd.DataFrame:
    """Cap rows per group for runtime control (reproducible)."""
    if max_rows is None or max_rows <= 0:
        return df
    parts = []
    rng = np.random.default_rng(seed)
    for key, g in df.groupby(group_col, dropna=False):
        if len(g) <= max_rows:
            parts.append(g)
        else:
            idx = rng.choice(g.index.values, size=max_rows, replace=False)
            parts.append(g.loc[idx])
    return pd.concat(parts, ignore_index=True).reset_index(drop=True)


def _cf_params_for_row(cfg: ProjectConfig, dte: float) -> CFPDEParams:
    """Higher resolution only for short maturities (DTE <= short_dte_max)."""
    if np.isfinite(dte) and float(dte) <= float(cfg.short_dte_max):
        return CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_short, J=cfg.J_short)
    return CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_base, J=cfg.J_base)


def _dataset_summary(df: pd.DataFrame) -> tuple[str, int, int]:
    """Return period label, row count, and quote-date count."""
    n_rows = int(len(df))
    if n_rows == 0 or "QUOTE_DATE" not in df.columns:
        return "N/A", n_rows, 0

    qd = pd.to_datetime(df["QUOTE_DATE"], errors="coerce").dropna()
    if len(qd) == 0:
        return "N/A", n_rows, 0

    period = f"{qd.min().date().strftime('%b %d, %Y')} to {qd.max().date().strftime('%b %d, %Y')}"
    return period, n_rows, int(qd.nunique())


def _calibration_cache_meta(cfg: ProjectConfig, calib: pd.DataFrame) -> dict:
    return {
        "total_months": cfg.total_months,
        "calib_months": cfg.calib_months,
        "filters": {
            "option_type": cfg.option_type,
            "max_rel_spread": cfg.max_rel_spread,
            "min_moneyness": cfg.min_moneyness,
            "max_moneyness": cfg.max_moneyness,
            "min_dte": cfg.min_dte,
            "max_dte": cfg.max_dte,
            "otm_only": cfg.otm_only,
        },
        "bsm": {"sigma": cfg.sigma, "r": cfg.r, "q": cfg.q},
        "cf_grid": {
            "I_base": cfg.I_base, "J_base": cfg.J_base,
            "I_short": cfg.I_short, "J_short": cfg.J_short,
            "short_dte_max": cfg.short_dte_max,
        },
        "alpha": {
            "lo": cfg.alpha_lo,
            "hi": cfg.alpha_hi,
            "weight_mode": cfg.weight_mode,
            "coarse_points": cfg.coarse_points,
            "max_calib_rows_per_bucket": cfg.max_calib_rows_per_bucket,
            "short_bucket_ensemble_runs": cfg.short_bucket_ensemble_runs,
        },
        "calib_quote_date_min": str(pd.to_datetime(calib["QUOTE_DATE"]).min().date()) if len(calib) else None,
        "calib_quote_date_max": str(pd.to_datetime(calib["QUOTE_DATE"]).max().date()) if len(calib) else None,
    }


def load_or_calibrate_alpha(df_calib: pd.DataFrame, out_dir: Path, cfg: ProjectConfig) -> pd.DataFrame:
    """
    Calibrate alpha by DTE bucket (or load cached results).

    Saves:
      results/calibration/alpha_by_bucket.csv
      results/calibration/alpha_meta.json
    """
    calib_dir = out_dir / "calibration"
    calib_dir.mkdir(parents=True, exist_ok=True)
    alpha_csv = calib_dir / "alpha_by_bucket.csv"
    alpha_meta = calib_dir / "alpha_meta.json"

    if alpha_csv.exists() and alpha_meta.exists() and (not cfg.force_recalibrate):
        # Load cached calibration
        return pd.read_csv(alpha_csv)

    # Calibrate per bucket
    bsm_params = BSMParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q)
    bounds = AlphaBounds(lo=cfg.alpha_lo, hi=cfg.alpha_hi)

    calib = df_calib.copy().reset_index(drop=True)

    # runtime cap
    calib = _sample_cap(calib, group_col="dte_bucket", max_rows=cfg.max_calib_rows_per_bucket, seed=42)

    out_rows = []
    SHORT_BUCKETS = {"1w_2w", "2w_1m"}
    ensemble_runs = max(1, int(cfg.short_bucket_ensemble_runs))
    for bucket, g in calib.groupby("dte_bucket", dropna=False):
        bucket_str = str(bucket)
        # For very short maturities, alpha can be weakly identified; use a small subsample ensemble
        # and take the median alpha to improve robustness.
        if bucket_str in SHORT_BUCKETS and ensemble_runs > 1:
            alphas = []
            objs = []
            seeds = (11, 22, 33, 44, 55)
            for seed in seeds[:ensemble_runs]:
                gg = g.sample(n=min(len(g), max(400, min(1200, len(g)))), random_state=seed) if len(g) > 0 else g
                res_i = calibrate_alpha(
                    gg,
                    bsm_params=bsm_params,
                    cf_params_base=CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_base, J=cfg.J_base),
                    cf_params_short=CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_short, J=cfg.J_short),
                    short_dte_max=cfg.short_dte_max,
                    bounds=bounds,
                    weight_mode=cfg.weight_mode,
                    coarse_points=max(cfg.coarse_points, 5),
                )
                if np.isfinite(res_i.alpha):
                    alphas.append(float(res_i.alpha))
                    objs.append(float(res_i.obj))
            if len(alphas) == 0:
                res = calibrate_alpha(
                    g,
                    bsm_params=bsm_params,
                    cf_params_base=CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_base, J=cfg.J_base),
                    cf_params_short=CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_short, J=cfg.J_short),
                    short_dte_max=cfg.short_dte_max,
                    bounds=bounds,
                    weight_mode=cfg.weight_mode,
                    coarse_points=max(cfg.coarse_points, 5),
                )
                alpha_hat = float(res.alpha)
                obj_hat = float(res.obj)
            else:
                alpha_hat = float(np.median(alphas))
                obj_hat = float(np.median(objs))
            out_rows.append(dict(dte_bucket=bucket_str, alpha=alpha_hat, obj=obj_hat, n_rows=int(len(g)), weight_mode=str(cfg.weight_mode)))
            continue

        # Other buckets: standard calibration (fast + explainable)
        res = calibrate_alpha(
            g,
            bsm_params=bsm_params,
            cf_params_base=CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_base, J=cfg.J_base),
            cf_params_short=CFPDEParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q, I=cfg.I_short, J=cfg.J_short),
            short_dte_max=cfg.short_dte_max,
            bounds=bounds,
            weight_mode=cfg.weight_mode,
            coarse_points=cfg.coarse_points,
        )
        out_rows.append(dict(dte_bucket=bucket_str, alpha=float(res.alpha), obj=float(res.obj), n_rows=int(res.n_rows), weight_mode=str(res.weight_mode)))

    alpha_df = pd.DataFrame(out_rows).sort_values("dte_bucket").reset_index(drop=True)

    alpha_df.to_csv(alpha_csv, index=False)
    with open(alpha_meta, "w") as f:
        json.dump(_calibration_cache_meta(cfg, df_calib), f, indent=2)

    return alpha_df


def run_project_pipeline(
    wide_csv: str | Path,
    out_dir: str | Path = "results",
    cfg: ProjectConfig = ProjectConfig(),
) -> dict[str, Path]:
    if str(cfg.option_type).lower() != "call":
        raise ValueError("This pipeline is configured for call options only. Set option_type='call'.")

    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    test_dir = out_dir / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # 1) Prepare long dataset
    # -----------------------
    df_wide = pd.read_csv(wide_csv)
    df_long = wide_to_long(df_wide)
    df_long = df_long[df_long["option_type"] == cfg.option_type].reset_index(drop=True)

    prep_cfg = PrepConfig(
        max_rel_spread=cfg.max_rel_spread,
        min_moneyness=cfg.min_moneyness,
        max_moneyness=cfg.max_moneyness,
        min_dte=cfg.min_dte,
        max_dte=cfg.max_dte,
        otm_only=cfg.otm_only,
    )
    n_raw = len(df_long)
    df_long = apply_filters(df_long, prep_cfg)
    n_after_basic_filters = len(df_long)

    # Remove obvious static-arbitrage / inconsistent quotes (defensive cleaning)
    df_long = remove_arbitrageable_quotes(df_long, r=cfg.r, q=cfg.q)
    n_after_no_arb = len(df_long)
    (out_dir / 'tables').mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{'stage':'raw_wide_to_long','n_rows':n_raw},{'stage':'after_basic_filters','n_rows':n_after_basic_filters},{'stage':'after_no_arbitrage_filters','n_rows':n_after_no_arb}]).to_csv(out_dir/'tables'/'cleaning_row_counts.csv', index=False)


    # Hard cut: last 2 months only
    df_long = scope_last_months(df_long, months=cfg.total_months)

    prepared_path = out_dir / "prepared_long_2m.parquet"
    df_long.to_parquet(prepared_path, index=False)

    # -----------------------
    # 2) Split calib/test
    # -----------------------
    df_calib, df_test = split_calib_test(df_long, total_months=cfg.total_months, calib_months=cfg.calib_months)
    total_period, total_rows, total_dates = _dataset_summary(df_long)
    calib_period, calib_rows, calib_dates = _dataset_summary(df_calib)
    test_period, test_rows, test_dates = _dataset_summary(df_test)

    calib_path = out_dir / "calibration_set_1m.parquet"
    test_path = out_dir / "test_set_1m.parquet"
    df_calib.to_parquet(calib_path, index=False)
    df_test.to_parquet(test_path, index=False)

    # -----------------------
    # 3) Calibrate alpha (cached)
    # -----------------------
    alpha_df = load_or_calibrate_alpha(df_calib, out_dir=out_dir, cfg=cfg)
    alpha_path = out_dir / "calibration" / "alpha_by_bucket.csv"

    # -----------------------
    # 4) Evaluate on test (RMSE only)
    # -----------------------
    # runtime cap on test too (optional)
    df_test_eval = _sample_cap(df_test, group_col="dte_bucket", max_rows=cfg.max_test_rows_per_bucket, seed=7)

    bsm_params = BSMParams(sigma=cfg.sigma, r=cfg.r, q=cfg.q)

    alpha_map = {str(r["dte_bucket"]): float(r["alpha"]) for _, r in alpha_df.iterrows()}

    eval_rows_path = test_dir / "eval_rows_test.parquet"
    alpha_mtime = alpha_path.stat().st_mtime if alpha_path.exists() else 0.0

    # Reuse cached evaluations when alpha has not changed.
    if eval_rows_path.exists() and (not cfg.force_reprice_test) and (eval_rows_path.stat().st_mtime >= alpha_mtime):
        eval_rows = pd.read_parquet(eval_rows_path)
    else:
        eval_rows = add_predictions_basic(
            df_test_eval,
            alpha_by_bucket=alpha_map,
            bsm_params=bsm_params,
            cf_params_selector=lambda dte: _cf_params_for_row(cfg, dte),
        )
        eval_rows.to_parquet(eval_rows_path, index=False)

    metrics_by_bucket, metrics_daily, metrics_overall = rmse_summaries(eval_rows)
    price_by_bucket, price_daily, price_overall = price_comparison_summaries(eval_rows)

    metrics_bucket_path = test_dir / "metrics_rmse_by_bucket.csv"
    metrics_daily_path = test_dir / "metrics_rmse_daily.csv"
    metrics_overall_path = test_dir / "metrics_rmse_overall.csv"
    price_bucket_path = test_dir / "price_comparison_by_bucket.csv"
    price_daily_path = test_dir / "price_comparison_daily.csv"
    price_overall_path = test_dir / "price_comparison_overall.csv"

    metrics_by_bucket.to_csv(metrics_bucket_path, index=False)
    metrics_daily.to_csv(metrics_daily_path, index=False)
    metrics_overall.to_csv(metrics_overall_path, index=False)
    price_by_bucket.to_csv(price_bucket_path, index=False)
    price_daily.to_csv(price_daily_path, index=False)
    price_overall.to_csv(price_overall_path, index=False)

    # -----------------------
    # 5) Plots
    # -----------------------
    if cfg.enable_plots:
        plot_mod.save_rmse_comparison_by_bucket(
            metrics_df=metrics_by_bucket,
            out_png=plots_dir / "rmse_by_bucket_test.png",
            title="Test month: RMSE comparison (CF-PDE vs BSM) by DTE bucket",
        )

        plot_mod.save_rmse_improvement_by_bucket(
            metrics_df=metrics_by_bucket,
            out_png=plots_dir / "rmse_improvement_by_bucket_test.png",
            title="Test month: RMSE improvement (BSM − CF) by DTE bucket",
        )

        plot_mod.save_daily_rmse_timeseries(
            metrics_daily=metrics_daily,
            out_png=plots_dir / "rmse_timeseries_test.png",
            title="Test month: Daily RMSE (CF-PDE vs BSM)",
        )

        plot_mod.save_price_means_by_bucket(
            price_by_bucket=price_by_bucket,
            out_png=plots_dir / "price_means_by_bucket_test.png",
            title="Test month: Observed vs BSM vs CF mean price by DTE bucket",
        )

        plot_mod.save_daily_price_timeseries(
            price_daily=price_daily,
            out_png=plots_dir / "price_timeseries_test.png",
            title="Test month: Observed vs BSM vs CF daily mean price",
        )

        plot_mod.save_pipeline_diagram(
            out_png=plots_dir / "pipeline_diagram.png",
            total_period=total_period,
            total_rows=total_rows,
            total_dates=total_dates,
            calib_period=calib_period,
            calib_rows=calib_rows,
            calib_dates=calib_dates,
            test_period=test_period,
            test_rows=test_rows,
            test_dates=test_dates,
        )

    return {
        "prepared_long_2m": prepared_path,
        "calibration_set_1m": calib_path,
        "test_set_1m": test_path,
        "alpha_by_bucket": alpha_path,
        "eval_rows_test": eval_rows_path,
        "metrics_rmse_by_bucket": metrics_bucket_path,
        "metrics_rmse_daily": metrics_daily_path,
        "metrics_rmse_overall": metrics_overall_path,
        "price_comparison_by_bucket": price_bucket_path,
        "price_comparison_daily": price_daily_path,
        "price_comparison_overall": price_overall_path,
        "plots_dir": plots_dir,
    }


# -----------------------
# Optional: ThetaData fetch
# -----------------------
def fetch_last_months_from_thetadata(
    out_csv: str | Path,
    months: int = 2,
    base_url: str = "http://localhost:25503/v3",
    symbol: str = "SPX",
    strike_range: int = 60,
    max_dte: int = 550,
) -> Path:
    """Download last `months` months of ThetaData SPX EOD into wide CSV."""
    end = date.today()
    start = (pd.Timestamp(end) - pd.DateOffset(months=int(months))).date()

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    return build_combined_options_csv(
        out_csv=out_csv,
        base_url=base_url,
        symbol=symbol,
        start_date=start,
        end_date=end,
        strike_range=strike_range,
        max_dte=max_dte,
    )


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--wide_csv", type=str, required=True, help="Path to combined_options_data.csv")
    p.add_argument("--out_dir", type=str, default="results")
    args = p.parse_args()

    out = run_project_pipeline(args.wide_csv, out_dir=args.out_dir, cfg=ProjectConfig())
    print("Wrote outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
