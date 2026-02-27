# MScFE Capstone Project: CF Time-Fractional PDE vs Black–Scholes (BSM) — Call Options

This repository provides a **project-ready, reproducible** empirical pipeline:

**ThetaData (SPX EOD, calls) → Data prep → α calibration (Month 1) → Out-of-sample test (Month 2) → Observed vs BSM vs CF comparison + RMSE benchmark → Plots**

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate  # windows
pip install -r requirements.txt
```

## 2) Start ThetaTerminal v3 (only if you want to fetch)

In a separate terminal:

```bash
java -jar ThetaTerminalv3.jar --creds-file=creds.txt
```

Default base URL: `http://localhost:25503/v3`.

## 3) Run (recommended: notebook)

Open and run:

- `notebooks/run_experiment_2m.ipynb` (Kernel → Restart & Run All)

### Outputs (written to `results/`)
- `results/prepared_long_2m.parquet`
- `results/calibration/alpha_by_bucket.csv`  *(cached; reused automatically)*
- `results/test/metrics_rmse_by_bucket.csv`
- `results/test/metrics_rmse_daily.csv`
- `results/test/metrics_rmse_overall.csv`
- `results/test/price_comparison_by_bucket.csv`
- `results/test/price_comparison_daily.csv`
- `results/test/price_comparison_overall.csv`
- Plots: `results/plots/*.png`

## 4) Experimental design (basic + explainable)

- **Scope:** use **only the last 2 months** of **call option** quotes (hard cut by `max(QUOTE_DATE)`).
- **Split:**
  - **Calibration:** first month (α per DTE bucket)
  - **Test:** second month (out-of-sample)
- **Benchmark:** BSM closed-form with constant `(σ, r, q)` (analytical solution of the classical BS PDE).
- **Comparison metrics:** observed MID vs BSM vs CF (mean/bias/MAE/RMSE), with RMSE highlighted for benchmark comparison.

### Practical adjustments to improve α identifiability (short maturities)
- **Higher PDE resolution only for short maturities (DTE ≤ 30):** `I=60, J=200`.
- **Broader signal:** widen moneyness band and disable OTM-only filter (reduces “near-noise-floor” effect for very short-dated OTM options).
- **Runtime control:** cap calibration rows per bucket (configurable) and cache α results to avoid recalibrating repeatedly.

If you already have `data/combined_options_data.csv`, set `FETCH=False` in the notebook and run directly.


## Methodology (aligned to research report)
- **Data scope:** Extract and analyze **only the last 2 months** of SPX EOD **call option** quotes from ThetaData.
- **Split:** Month-1 = calibration, Month-2 = out-of-sample test.
- **Cleaning:** basic quote sanity + spread/moneyness/DTE filters, then **static no-arbitrage filters** (bounds, vertical monotonicity, butterfly convexity, calendar monotonicity).
- **Calibration:** calibrate **α per DTE bucket** on Month-1 using spread-weighted SSE.
- **Short buckets robustness:** for `1w_2w` and `2w_1m`, calibrate α using a small subsample ensemble and take the **median α**.
- **Numerical resolution:** use higher PDE resolution for short maturities (DTE≤30): I=60, J=200; otherwise I=28, J=70.
- **Benchmark:** BSM closed-form (analytical solution to classical BS PDE) with constant σ,r,q.
- **Metric:** report **RMSE** (overall + by DTE bucket) for CF-PDE vs BSM.
