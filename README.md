# ML_Dayahead_XGBoost_energy_price_forecaster_Austria
ML regression model for day-ahead energy price forecasting. Built with XGBoost, Plotly, and ipywidgets.
<div align="center">

```
██╗  ██╗ ██████╗ ██████╗  ██████╗  ██████╗ ███████╗████████╗
╚██╗██╔╝██╔════╝ ██╔══██╗██╔═══██╗██╔═══██╗██╔════╝╚══██╔══╝
 ╚███╔╝ ██║  ███╗██████╔╝██║   ██║██║   ██║███████╗   ██║   
 ██╔██╗ ██║   ██║██╔══██╗██║   ██║██║   ██║╚════██║   ██║   
██╔╝ ██╗╚██████╔╝██████╔╝╚██████╔╝╚██████╔╝███████║   ██║   
╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝  ╚═════╝╚══════╝   ╚═╝   
```

# ⚡ XGBoost Day-Ahead Energy Price Forecaster

**Predict electricity market prices with machine learning — hour by hour, day by day.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## 🔮 What Is This?

A fully interactive Jupyter notebook that trains an **XGBoost regression model** on historical day-ahead electricity auction data and forecasts future hourly prices (€/MWh) — up to 7 days ahead.

No command line. No config files. Just open the notebook, pick your dates, hit **Run Forecast**, and get results.

---

## ✨ Features at a Glance

| | Feature |
|---|---|
| 🧠 | XGBoost regressor trained fresh on every run |
| 📅 | Interactive date range selector for training data |
| ⏱️ | Forecast horizons: **1 day**, **3 days**, or **7 days** |
| 📊 | Dual interactive Plotly charts (daily overview + full hourly curve) |
| 🗓️ | Daily summary table: cheapest and priciest hour per day |
| 📥 | One-click **CSV download** of forecast results |
| 🔁 | Iterative multi-step forecasting using lag feedback |

---

## 🖥️ Demo Flow

```
┌─────────────────────────────────────────────────────────┐
│  📅 Select Date Range      ⏱️ Choose Horizon             │
│  [ From: 2025-04-01 ]      ○ 1 day                      │
│  [ To:   2025-05-15 ]      ● 3 days                     │
│                            ○ 7 days                     │
│              [ 🚀 Run Forecast ]                         │
└─────────────────────────────────────────────────────────┘
              ↓  trains XGBoost  ↓
┌─────────────────────────────────────────────────────────┐
│  date        │ hour_low │ price_low │ hour_high │ price_high │
│  2025-05-16  │    03    │  38.2 €   │    19     │  112.4 €   │
│  2025-05-17  │    04    │  41.7 €   │    20     │  108.9 €   │
│  2025-05-18  │    03    │  36.5 €   │    18     │  119.2 €   │
└─────────────────────────────────────────────────────────┘
              ↓  renders charts  ↓  exports CSV  ↓
```

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install xgboost pandas numpy plotly ipywidgets openpyxl
```

### 2 — Point to your data file

Open the notebook and update this line:

```python
file_path = r"D:\ML XGboost\Merged_Hourly_DayAhead_2020_2025.xlsx"
```

### 3 — Run the notebook

```
Kernel → Restart & Run All
```

### 4 — Use the UI

Select your training date range, pick a forecast horizon, and click **Run Forecast**.

---

## 📂 Input Data Format

The notebook reads a single `.xlsx` file. Required columns:

| Column | Description |
|---|---|
| `Time from [CET/CEST]` | Hourly timestamp in CET/CEST (DST `A`/`B` suffixes handled automatically) |
| `Price MC Auction [EUR/MWh]` | Day-ahead market clearing auction price |

> Data is resampled to clean 1-hour frequency. Missing values are filled via **linear interpolation**.

---

## 🧠 Model Architecture

### Feature Engineering

The model learns from 8 hand-crafted time-series features:

```
┌──────────────┬─────────────────────────────────────────┐
│ Feature      │ Description                             │
├──────────────┼─────────────────────────────────────────┤
│ hour         │ Hour of day (0–23)                      │
│ dayofweek    │ Day of week (0 = Monday)                │
│ is_weekend   │ Binary: 1 if Saturday or Sunday         │
│ lag_1        │ Price 1 hour ago                        │
│ lag_2        │ Price 2 hours ago                       │
│ lag_24       │ Price same hour, yesterday              │
│ lag_48       │ Price same hour, 2 days ago             │
│ lag_168      │ Price same hour, last week              │
└──────────────┴─────────────────────────────────────────┘
```

### Hyperparameters

```python
XGBRegressor(
    n_estimators  = 200,   # number of boosting rounds
    max_depth     = 5,     # tree depth
    learning_rate = 0.1,   # shrinkage
    random_state  = 42
)
```

### Iterative Forecasting

Predictions are generated step-by-step. Each forecasted value is fed back as a lag input for the next step — simulating real-world forecasting conditions.

```
t+1 → predicted → used as lag_1 for t+2
t+2 → predicted → used as lag_1 for t+3
...
```

> ⚠️ Prediction errors compound over longer horizons. 1-day forecasts are more reliable than 7-day forecasts.

---

## 📈 Output

### Chart 1 — Daily Low/High Overview

Dual y-axis chart showing:
- Which **hour** of the day has the lowest / highest price
- What those **prices** are (€/MWh)

### Chart 2 — Full Hourly Forecast Curve

A continuous line chart of every forecasted hourly price across the entire horizon.

### Exported CSV

`xgboost_forecast.csv` — daily summary table:

```
date, hour_low, price_low, hour_high, price_high
2025-05-16, 3, 38.2, 19, 112.4
...
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `xgboost` | ≥ 1.6 | Gradient boosted regression model |
| `pandas` | ≥ 1.3 | Data loading, resampling, feature engineering |
| `numpy` | ≥ 1.21 | Numerical operations |
| `plotly` | ≥ 5.0 | Interactive charts |
| `ipywidgets` | ≥ 7.6 | Jupyter UI (date pickers, buttons) |
| `openpyxl` | ≥ 3.0 | Reading `.xlsx` files |

> `ipywidgets` requires **Jupyter Notebook** or **JupyterLab**. Widgets will not render in static GitHub previews.

---

## 📁 Project Structure

```
📦 ML_XGBoost_Forecaster/
├── 📓 ML_Regressor_XGBoost_Model.ipynb   ← Main notebook
└── 📊 Merged_Hourly_DayAhead_2020_2025.xlsx  ← Input data (not included)
```

---

## ⚠️ Limitations

- **No model persistence** — the model is retrained on every run
- **Error compounding** — iterative lag feedback amplifies errors over longer horizons
- **No external signals** — weather, grid load, fuel prices, and cross-border flows are not modeled
- **Minimum data** — at least ~500 hourly records (~3 weeks) are required for training

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

<div align="center">

**Built with ⚡ XGBoost · 🐼 Pandas · 📊 Plotly · 🪐 Jupyter**

*Forecast smarter. Trade better.*

</div>
