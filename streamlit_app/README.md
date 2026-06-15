# ⚡ Austrian Day-Ahead Forecaster — Live Demo

Interactive Streamlit demo for the model in
[`AIMLDS7/ML_Dayahead_XGBoost_energy_price_forecaster_Austria`]([https://github.com/AIMLDS7/ML_Dayahead_XGBoost_energy_price_forecaster_Austria](https://aimlds7-xgboost-dayahead.streamlit.app/)).

**What it does:** Forecasts Austrian day-ahead electricity prices 1/3/7/14 days ahead
using gradient-boosted regression with rolling lag features and quantile
confidence intervals.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Deploy to Streamlit Cloud (free, ~3 min)

1. Push this folder to a GitHub repo (e.g. a new
   `at-dayahead-forecaster-demo` repo, or drop the `app.py` into your existing
   `ML_Dayahead_XGBoost_energy_price_forecaster_Austria` repo under a
   `streamlit_app/` folder).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"** → connect GitHub → pick the repo → set main file path
   to `app.py` → **Deploy**.
4. You'll get a URL like `https://aimlds7-at-dayahead-forecaster.streamlit.app`.

## Why synthetic data?

The real EPEX auction data lives in the private research repo (NDA).
This demo uses a deterministic synthetic generator that mimics:

- Annual seasonality (winter peak)
- Weekly seasonality (weekend dip)
- Daily peaks (morning ramp + evening peak)
- Solar "duck curve" (midday PV depression)
- Rare price spikes (gas-crisis style)
- Negative prices (oversupply)

It's reproducible, NDA-safe, and gives reviewers a real feel for the
pipeline without exposing proprietary data.

## What's in the demo

- Hero KPIs: last actual, forecast mean, forecast high/low
- Historical + forecast chart with 50–95% confidence band
- Daily summary table (low / mean / high per day)
- Backtest scatter (predicted vs actual)
- Feature importance bar
- Pattern explorer (hour of day / day of week / month)

## Caveats

The synthetic generator is intentionally simple — model performance numbers
here will be better than real-world EPEX data. Treat this as a **demo of the
pipeline**, not a benchmark.
