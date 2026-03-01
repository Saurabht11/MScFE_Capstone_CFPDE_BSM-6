# MScFE Capstone: CF-PDE vs BSM (Call Options) - Research Summary
**Snapshot Date:** March 1, 2026  
**Scope:** SPX EOD call options only

## Executive Summary
This repository evaluates a Caputo-Fabrizio time-fractional PDE (CF-PDE) against Black-Scholes-Merton (BSM) on a strict 2-month split:
- Month 1: calibration
- Month 2: out-of-sample test

Using current checked-in outputs:
- Test sample size: **2,400** contracts (300 per DTE bucket due runtime cap)
- RMSE (BSM): **80.43**
- RMSE (CF-PDE): **31.09**
- RMSE improvement: **49.34** absolute (**61.35%** lower RMSE vs BSM)
- Mean pricing bias (BSM): **-40.58**
- Mean pricing bias (CF-PDE): **-16.89**

## Data Window and Split
- Min quote date: **2025-12-26**
- Max quote date: **2026-02-25**
- Quote dates: **41**
- Prepared long dataset rows: **42,069**
- Calibration rows (Month 1): **19,059**
- Test rows (Month 2): **23,010**

## Methodology (Current Code Defaults)
From `pipeline.py` and `src/`:
- Option type: call-only
- Filtering: spread, moneyness, DTE + no-arbitrage filters
- DTE buckets: 8 buckets (`1w_2w` ... `12m_plus`)
- Calibration target: spread-weighted SSE
- Alpha bounds: [0.10, 0.99]
- Base grid (longer DTE): I=28, J=70
- Short grid (DTE<=30): I=60, J=200
- Benchmark parameters: sigma=0.16, r=0.035, q=0.021

Note: notebook runs can override defaults (for example row caps and PDE grid choices), so results reflect the exact config used at run time.

## Current Artifacts in `results/`
### Core datasets
- `results/prepared_long_2m.parquet`
- `results/prepared_long_2m.csv`
- `results/calibration_set_1m.parquet`
- `results/calibration_set_1m.csv`
- `results/test_set_1m.parquet`
- `results/test_set_1m.csv`

### Calibration outputs
- `results/calibration/alpha_by_bucket.csv`
- `results/calibration/alpha_meta.json`

### Test evaluation outputs
- `results/test/eval_rows_test.parquet`
- `results/test/eval_rows_test.csv`
- `results/test/metrics_rmse_by_bucket.csv`
- `results/test/metrics_rmse_daily.csv`
- `results/test/metrics_rmse_overall.csv`
- `results/test/price_comparison_by_bucket.csv`
- `results/test/price_comparison_daily.csv`
- `results/test/price_comparison_overall.csv`

### Tables
- `results/tables/cleaning_row_counts.csv`
- `results/tables/counts_by_bucket.csv`
- `results/tables/data_coverage_summary.csv`
- `results/tables/filter_retention_summary.csv`

### Plots currently present
- `results/plots/alpha_and_error_analysis.png`
- `results/plots/daily_price_evolution_test_month.png`
- `results/plots/daily_rmse_evolution.png`
- `results/plots/mae_mape_accuracy_by_bucket.png`
- `results/plots/moneyness_hist.png`
- `results/plots/pipeline_diagram.png`
- `results/plots/price_heatmaps_by_date_dte.png`
- `results/plots/price_rmse_comparison_by_bucket.png`
- `results/plots/rel_spread_hist.png`
- `results/plots/scatter_observed_vs_predicted.png`

## Conclusion
On the current checked-in run, CF-PDE materially outperforms BSM on out-of-sample call pricing RMSE and reduces systematic underpricing bias. The repository structure is centered on:
- reusable pipeline code in `pipeline.py` + `src/`
- reproducible outputs in `results/`
- notebook-driven analysis and publication figures
