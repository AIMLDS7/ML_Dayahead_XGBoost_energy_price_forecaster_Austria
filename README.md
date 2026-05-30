<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=⚡%20Energy%20Price%20Forecaster&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=XGBoost%20%2B%20LSTM%20%7C%20Day-Ahead%20Electricity%20Market%20Intelligence&descAlignY=60&descSize=16" width="100%"/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-189AB4?style=flat-square)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=flat-square&logo=plotly&logoColor=white)](https://plotly.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)]()

> **Dual-model time series forecasting engine for European day-ahead electricity markets.**  
> Train XGBoost and LSTM in parallel, compare accuracy metrics, and export forecasts — all from a zero-config interactive Jupyter UI.

</div>

---

## Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Model Details](#-model-details)
- [Feature Engineering](#-feature-engineering)
- [Performance Benchmarks](#-performance-benchmarks)
- [Quick Start](#-quick-start)
- [Data Format](#-data-format)
- [UI Walkthrough](#️-ui-walkthrough)
- [Technical Decisions](#-technical-decisions)
- [Changelog](#-changelog)
- [Limitations & Future Work](#️-limitations--future-work)

---

## 🔭 Overview

This project implements a **production-quality, dual-model forecasting system** for European day-ahead electricity prices, using data from hourly market clearing auctions (€/MWh).

Most open-source energy forecasters pick one model and call it done. This one trains both **XGBoost** (gradient-boosted trees, fast, interpretable) and **LSTM** (deep recurrent network, sequence-aware) on identical data splits, then surfaces a head-to-head comparison of RMSE, MAE, and MAPE — letting the data decide which model wins on your specific market window.

**Why both models?**

| Scenario | Better Model |
|---|---|
| Short-term horizon (1–2 days), stable seasonality | XGBoost |
| Longer horizon, complex sequential dependencies | LSTM |
| Novel price spikes or regime changes | Neither — but LSTM degrades more gracefully |

---

## 🏗 Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM DATA FLOW                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐   │
│   │  .xlsx / .csv│───▶│  Preprocessing    │───▶│  Feature Store     │   │
│   │  Raw Market  │    │  - Parse CET/CEST │    │  - 12 features     │   │
│   │  Clearing    │    │  - 1h resample    │    │  - Lag engineering │   │
│   │  Data        │    │  - Interpolation  │    │  - Cyclical encod. │   │
│   └──────────────┘    └───────────────────┘    └────────┬───────────┘   │
│                                                         │                │
│                              ┌──────────────────────────┘                │
│                              │  85/15 time-based split (no shuffle)      │
│                              │                                           │
│              ┌───────────────▼──────────┐  ┌──────────────────────────┐ │
│              │  XGBoost Pipeline        │  │  LSTM Pipeline           │ │
│              │  n_est=200 depth=5 lr=.05│  │  64 units tanh window=168│ │
│              │  Iterative lag feedback  │  │  Recursive prediction    │ │
│              └───────────────┬──────────┘  └──────────────┬───────────┘ │
│                              │                             │             │
│                    ┌─────────▼─────────────────────────────▼──────────┐ │
│                    │    Evaluation & Comparison Layer                  │ │
│                    │    RMSE  │  MAE  │  MAPE  │  Plotly Charts       │ │
│                    └─────────────────────────────────────────────────-┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Model Details

### Model 1 — XGBoost Regressor

XGBoost frames price forecasting as a **tabular regression problem**. Each row in the training set is a snapshot of engineered lag features at time `t`, and the label is `price[t]`.

```python
XGBRegressor(
    n_estimators  = 200,     # boosting rounds
    max_depth     = 5,       # controls bias-variance tradeoff
    learning_rate = 0.05,    # shrinkage; lower = more robust
    n_jobs        = -1,      # full CPU parallelism
    random_state  = 42
)
```

**Forecasting strategy:** Recursive multi-step. Each predicted value is appended to the price history and used as input for the next step — simulating real-time autoregressive inference.

```
train_end ──▶ t+1 predicted ──▶ appended ──▶ t+2 predicted ──▶ ...
```

> Error compounds with horizon length. 1-day forecasts are significantly more reliable than 7-day.

---

### Model 2 — LSTM (Long Short-Term Memory)

LSTM treats forecasting as a **sequence-to-one regression problem**. A sliding window of the last 168 hours (exactly 1 week) is passed as input, and the model predicts the next hour's price.

```
Architecture:
  Input shape: (batch, 168 timesteps, 4 features)
        │
  LSTM(64, activation='tanh')      — learns temporal patterns
        │
  Dense(1)                         — outputs next price (scaled)
        │
  Inverse transform via MinMaxScaler
```

**Input features per timestep:**

| Feature | Construction |
|---|---|
| `price` | MinMaxScaler-normalized €/MWh |
| `hour_sin` | `sin(2π × hour / 24)` |
| `hour_cos` | `cos(2π × hour / 24)` |
| `dow_sin` | `sin(2π × weekday / 7)` |

**Training config:**
```python
model.fit(
    X_train, y_train,
    epochs          = 50,
    batch_size      = 32,
    validation_data = (X_test, y_test),
    callbacks       = [EarlyStopping(patience=5, restore_best_weights=True)],
    shuffle         = False    # ← CRITICAL for time series — never shuffle
)
```

---

## 🔬 Feature Engineering

XGBoost feature set (12 features total):

```
┌────────────────────┬──────────────────────────────────────────────────────┐
│ Feature            │ Description & Rationale                              │
├────────────────────┼──────────────────────────────────────────────────────┤
│ hour               │ Hour of day (0–23). Captures intraday demand cycle   │
│ dow                │ Day of week (0=Mon). Captures weekday/weekend splits  │
│ month              │ Month. Captures seasonal heating/cooling demand       │
│ is_weekend         │ Binary flag. Weekend prices behave fundamentally      │
│                    │ differently (lower industrial load)                   │
│ hour_sin           │ sin(2π·h/24) — cyclical encoding so hour 23 and     │
│ hour_cos           │ cos(2π·h/24)   hour 0 are adjacent, not 23 apart    │
│ lag_1              │ Price 1h ago    ─┐                                   │
│ lag_2              │ Price 2h ago     │ Short-memory autoregressive        │
│ lag_3              │ Price 3h ago    ─┘ features                          │
│ lag_24             │ Price same hour yesterday — strong daily periodicity │
│ lag_48             │ Price same hour 2 days ago                           │
│ lag_168            │ Price same hour last week — weekly seasonality       │
│ roll_24            │ 24h rolling mean (shifted 1). Smoothed recent level  │
│ roll_168           │ 168h rolling mean (shifted 1). Weekly baseline       │
└────────────────────┴──────────────────────────────────────────────────────┘
```

**Why cyclical encoding?**  
A raw `hour` feature treats 23 and 0 as far apart (distance = 23). After sin/cos encoding, they are adjacent on the unit circle — which better reflects the continuous nature of the daily price cycle.

---

## 📊 Performance Benchmarks

*Benchmarks computed on a held-out 15% test slice (time-based, no shuffle) of 2020–2024 Austrian day-ahead clearing data (~43,800 hourly records).*

| Metric | XGBoost | LSTM | Lower is better |
|---|---|---|---|
| **RMSE** (€/MWh) | ~8–12 | ~9–14 | ↓ |
| **MAE** (€/MWh) | ~5–9 | ~6–10 | ↓ |
| **MAPE** (%) | ~10–15% | ~12–18% | ↓ |

> Ranges reflect variability across different training window selections (2020–2021 data behaves very differently from 2021–2022 during the energy crisis). Exact numbers depend on your chosen date range in the UI.

**Key finding:** XGBoost consistently outperforms LSTM on this dataset. The likely reason is that energy price drivers are primarily **feature-based** (hour of day, day of week, recent lag) rather than **long-sequence-based** — which plays to XGBoost's strengths. LSTM's advantage materializes on datasets with longer-range temporal dependencies.

---

## 🚀 Quick Start

### 1 — Clone / download

```bash
git clone https://github.com/AIMLDS7/ML_Dayahead_XGBoost_energy_price_forecaster_Austria.git
cd ML_Dayahead_XGBoost_energy_price_forecaster_Austria
```

### 2 — Install dependencies

```bash
pip install xgboost pandas numpy plotly ipywidgets scikit-learn tensorflow openpyxl
```

Or with a lockfile:

```bash
pip install xgboost>=2.0 pandas>=1.5 numpy>=1.23 plotly>=5.0 \
            ipywidgets>=7.6 scikit-learn>=1.2 tensorflow>=2.12 openpyxl>=3.0
```

### 3 — Launch

```bash
jupyter notebook "Price_Forecaster_ML_and_XG_boost.ipynb"
```

Then: **Kernel → Restart & Run All**

### 4 — (Optional) Set a default data path

Edit the top of the notebook:

```python
DEFAULT_PATH = r"path/to/your/Merged_Hourly_DayAhead_2020_2025.xlsx"
```

You can also paste the path directly into the **File Path** text box in the UI at runtime — no code edit required.

---

## 📂 Data Format

The notebook auto-detects column names. It looks for any column whose name contains `time`, `date`, or `timestamp` for the index, and any column containing `price`, `eur`, or `mwh` for the target.

**Canonical format (Austrian EXAA/EPEX data):**

| Column | Type | Notes |
|---|---|---|
| `Time from [CET/CEST]` | string | Hourly timestamps. `A`/`B` DST suffixes (e.g. `2023-10-29 02A:00`) are stripped automatically |
| `Price MC Auction [EUR/MWh]` | float | Day-ahead market clearing price |

**Accepted file types:** `.xlsx`, `.xls`, `.csv`

**Pre-processing pipeline:**
1. Parse timestamps, strip DST `A`/`B` suffixes
2. Coerce price column to numeric, drop NaN rows
3. Group duplicates by mean (handles DST overlap hours)
4. Resample to clean 1-hour frequency
5. Linear interpolation for gaps ≤ 3 hours
6. Drop remaining NaNs

**Minimum data requirement:** ≥ 500 records (~3 weeks). Recommended: ≥ 8,760 records (1 full year) for reliable seasonality capture.

---

## 🖥️ UI Walkthrough

```
┌────────────────────────────────────────────────────────────────────┐
│ ⚡ Price Forecaster                                                │
│    XGBoost + LSTM • Model Comparison                               │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  📂 Path: [ E:\...\Merged_Hourly_DayAhead_2020_2025.xlsx ] [Load] │
│                                                                    │
│  ✅ 43,824 records  |  2020-01-01 → 2024-12-31  |  Avg €72.4/MWh  │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  From: [ 2024-01-01 ]   To: [ 2024-12-31 ]                        │
│  Horizon:  |──────●────────|  3 days                              │
│  Model:    ● Both (Compare)  ○ XGBoost Only  ○ LSTM Only          │
│                                                                    │
│                       [ ▶  Run Forecast ]                          │
├────────────────────────────────────────────────────────────────────┤
│  ████████████████████████████████████████  100%  Training done    │
│                                                                    │
│  ┌───────────────────┐  ┌───────────────────┐                     │
│  │ XGBoost           │  │ LSTM              │                     │
│  │ RMSE  €8.42       │  │ RMSE  €9.17       │                     │
│  │ MAE   €5.91       │  │ MAE   €6.44       │                     │
│  │ MAPE  11.2%       │  │ MAPE  12.8%       │                     │
│  └───────────────────┘  └───────────────────┘                     │
│                                                                    │
│  [Plotly interactive forecast chart]                               │
│                                                                    │
│  date        hour_low  price_low  hour_high  price_high           │
│  2025-05-16     03      €38.2        19        €112.4             │
│  2025-05-17     04      €41.7        20        €108.9             │
│  2025-05-18     03      €36.5        18        €119.2             │
│                                                                    │
│                              [ ⬇ Export CSV ]                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Technical Decisions

**Why `shuffle=False` in LSTM training?**  
Shuffling training batches is standard for i.i.d. data, but time series observations are temporally dependent. Shuffling would allow the model to "see the future" during training — a form of data leakage that inflates validation metrics. Setting `shuffle=False` ensures every training step only has access to observations prior to the validation window.

**Why 85/15 time-based split instead of random split?**  
Same reason: the test set must come strictly after the training set in time, or the model implicitly leaks future information through lagged features.

**Why `lag_168` (1 week lookback) alongside shorter lags?**  
Electricity markets exhibit strong **weekly seasonality** — Monday morning prices in winter are highly correlated with the same hour 7 days prior. Without this feature, the model would only capture daily and sub-daily cycles, missing the workweek/weekend structure.

**Why cyclical sin/cos encoding for `hour` and `dow`?**  
Integer encoding (hour = 0..23) implies that hour 0 and hour 23 are 23 units apart — but they're actually adjacent. Projecting onto a unit circle (sin + cos) preserves this topology and improves model performance on intraday patterns.

**Why 168-step window for LSTM?**  
One full week of hourly data (7 × 24 = 168 timesteps). This ensures the LSTM can capture both daily and weekly cycles in a single input sequence — the minimum window that contains a complete periodic structure for electricity markets.

---

## 📋 Changelog

### v2.0 — May 2025
- Added **LSTM model** with time-encoded input features
- Added **model comparison UI** (side-by-side RMSE / MAE / MAPE)
- Fixed data leakage: `shuffle=False` enforced across both models
- Extended XGBoost feature set: `lag_3`, `month`, `hour_sin`, `hour_cos`, `roll_168` added
- Learning rate tuned: `0.1 → 0.05` for more stable convergence
- Added `EarlyStopping` with `restore_best_weights=True` to LSTM training

### v1.0 — 2024
- Initial release: XGBoost-only forecaster
- Interactive Jupyter UI with ipywidgets
- Plotly charts and CSV export

---

## ⚠️ Limitations & Future Work

**Current limitations:**

| Limitation | Impact |
|---|---|
| No model persistence | Model retrained on every run (~15–90s depending on hardware) |
| Recursive forecast error compounding | 7-day forecasts significantly less accurate than 1-day |
| No exogenous features | Weather, grid load, cross-border flows, fuel prices not modeled |
| No uncertainty quantification | Point forecasts only; no prediction intervals |
| Austria-specific data format | Other markets may need minor column-name adjustments |

**Planned improvements:**

- [ ] Feature importance visualisation (SHAP values for XGBoost)
- [ ] Probabilistic forecasting — prediction intervals via quantile regression
- [ ] Exogenous inputs: temperature, hydro reservoir levels, wind/solar generation
- [ ] Model persistence: `xgb.save_model()` + `model.save()` so retraining is optional
- [ ] Backtesting harness: rolling-window evaluation across multiple time periods
- [ ] Export to ONNX for deployment outside Jupyter

---

## 📦 Dependencies

| Package | Version | Role |
|---|---|---|
| `xgboost` | ≥ 2.0 | Gradient boosted regression |
| `tensorflow` | ≥ 2.12 | LSTM neural network |
| `scikit-learn` | ≥ 1.2 | MinMaxScaler, evaluation metrics |
| `pandas` | ≥ 1.5 | Data loading, resampling, feature engineering |
| `numpy` | ≥ 1.23 | Numerical operations |
| `plotly` | ≥ 5.0 | Interactive charts |
| `ipywidgets` | ≥ 7.6 | Jupyter UI (sliders, date pickers, buttons) |
| `openpyxl` | ≥ 3.0 | Reading `.xlsx` files |

> `ipywidgets` requires **Jupyter Notebook** or **JupyterLab**. Widgets do not render in static GitHub previews or VS Code's notebook renderer without the widget extension.

---

## 📁 Repository Structure

```
📦 ML_Dayahead_XGBoost_energy_price_forecaster_Austria/
│
├── 📓 Price_Forecaster_ML_and_XG_boost.ipynb   ← v2: XGBoost + LSTM comparison
├── 📓 ML Regressor XGBoost Model.ipynb          ← v1: XGBoost only (archived)
│
├── 📊 Merged_Hourly_DayAhead_2020_2025.xlsx     ← Sample dataset (Austrian EXAA)
│
├── 📄 README.md
└── 📄 LICENSE
```

---

## 🤝 Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss the approach.

If you're running this on a different European market (DE, FR, NL, etc.) and need column-name adjustments, open an issue with a sample of your data format.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

**Built with ⚡ XGBoost · 🧠 LSTM/TensorFlow · 🐼 Pandas · 📊 Plotly · 🪐 Jupyter**

*Forecast smarter. Trade better.*

[![GitHub](https://img.shields.io/badge/GitHub-AIMLDS7-181717?style=flat-square&logo=github)](https://github.com/AIMLDS7)

</div>
