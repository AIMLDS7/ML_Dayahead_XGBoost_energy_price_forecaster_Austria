# ⚡ Day-Ahead Electricity Price Forecaster

**XGBoost + LSTM with Interactive Model Comparison**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-189AB4?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](./LICENSE)

---

Predict hourly electricity market prices (€/MWh) up to 7 days ahead using two ML models — **XGBoost** and **LSTM** — with a fully interactive Jupyter UI. Train both models simultaneously and compare their accuracy metrics side by side, all without writing a single line of code at runtime.

---

## ✨ What's New in v2

| | v1 (XGBoost only) | v2 (this notebook) |
|---|---|---|
| Models | XGBoost | XGBoost + LSTM |
| Comparison | ❌ | ✅ Side-by-side RMSE / MAE / MAPE |
| LSTM features | ❌ | ✅ Time-encoded (sin/cos) inputs |
| Data leakage fix | Partial | ✅ `shuffle=False`, time-based split |
| Feature engineering | 7 features | 10 features (lag + rolling + cyclical) |

---

## 🖥️ UI Preview

```
┌────────────────────────────────────────────────────────────┐
│  📂 File Path:  [ Merged_Hourly_DayAhead_2020_2025.xlsx ]  │
│                                        [ Load  ]           │
│  ✅ 43,824 records | 2020-01-01 → 2024-12-31 | Avg €72.4  │
├────────────────────────────────────────────────────────────┤
│  From: [ 2024-01-01 ]   To: [ 2024-12-31 ]                │
│  Horizon:  |──●────|  3 days                               │
│  Model:  ● Both (Compare)  ○ XGBoost Only  ○ LSTM Only     │
│                     [ ▶  Run Forecast ]                    │
└────────────────────────────────────────────────────────────┘
         ↓  trains both models  ↓
┌──────────────────────┐  ┌──────────────────────┐
│ XGBoost              │  │ LSTM                 │
│ RMSE: €8.42          │  │ RMSE: €9.17          │
│ MAE:  €5.91          │  │ MAE:  €6.44          │
│ MAPE: 11.2%          │  │ MAPE: 12.8%          │
└──────────────────────┘  └──────────────────────┘
         ↓  interactive Plotly charts + CSV export  ↓
```

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
pip install xgboost pandas numpy plotly ipywidgets scikit-learn tensorflow openpyxl
```

### 2 — Open the notebook

```
Price_Forecaster_ML_and_XG_boost.ipynb
```

### 3 — Point to your data

Update the default path at the top of the notebook:

```python
DEFAULT_PATH = r"path/to/your/Merged_Hourly_DayAhead_2020_2025.xlsx"
```

Or just paste the path into the **File Path** text box in the UI at runtime.

### 4 — Run all cells, then use the UI

```
Kernel → Restart & Run All
```

Select a date range, choose a forecast horizon (1–7 days), pick a model, and click **Run Forecast**.

---

## 📂 Input Data Format

A single `.xlsx` or `.csv` file with at least two columns:

| Column | Description |
|---|---|
| `Time from [CET/CEST]` (or any column containing `time`, `date`, `timestamp`) | Hourly timestamps. CET/CEST `A`/`B` DST suffixes are handled automatically. |
| `Price MC Auction [EUR/MWh]` (or any column containing `price`, `eur`, `mwh`) | Day-ahead market clearing price in €/MWh |

Data is resampled to a clean 1-hour frequency. Gaps of up to 3 hours are filled via linear interpolation.

---

## 🧠 Model Details

### XGBoost

**Features (10 total):**

| Feature | Description |
|---|---|
| `hour`, `dow`, `month` | Calendar components |
| `is_weekend` | Binary flag |
| `hour_sin`, `hour_cos` | Cyclical hour encoding |
| `lag_1`, `lag_2`, `lag_3` | Recent price lags |
| `lag_24`, `lag_48`, `lag_168` | Same-hour: yesterday, 2 days ago, last week |
| `roll_24`, `roll_168` | 24h and 168h rolling mean |

**Training:** 85/15 time-based split (no shuffle), 200 estimators, depth 5, LR 0.05.

---

### LSTM

**Architecture:** Single LSTM layer (64 units) → Dense(1)

**Input:** 168-hour (1 week) sliding window with 4 features: `price`, `hour_sin`, `hour_cos`, `dow_sin`

**Training:** 85/15 time-based split, `shuffle=False`, EarlyStopping (patience=5), up to 50 epochs.

**Forecast:** Recursive multi-step — each predicted price is fed back into the sequence for the next step.

---

## 📊 Output

- **Model comparison cards** — RMSE, MAE, MAPE for each selected model  
- **Plotly forecast chart** — historical prices + XGBoost and/or LSTM forecasts  
- **Daily summary table** — cheapest and priciest hour per day  
- **CSV download** — one-click export of all forecast values  

---

## ⚠️ Limitations

- Model is retrained fresh on every run (no persistence/saving)
- Recursive lag feedback compounds errors — 1-day forecasts are more reliable than 7-day
- No external features: weather, grid load, fuel prices, cross-border flows are not modeled
- LSTM training can take 1–3 minutes depending on data size and hardware
- Minimum ~500 hourly records required for training

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `xgboost` | Gradient boosted regression |
| `tensorflow` | LSTM neural network |
| `pandas`, `numpy` | Data processing |
| `scikit-learn` | Preprocessing, metrics |
| `plotly` | Interactive charts |
| `ipywidgets` | Jupyter UI components |
| `openpyxl` | Reading `.xlsx` files |

> `ipywidgets` requires **Jupyter Notebook** or **JupyterLab**. Widgets will not render in static GitHub previews.

---

## 📁 Repository Structure

```
📦 ML_Dayahead_XGBoost_energy_price_forecaster_Austria/
├── 📓 Price_Forecaster_ML_and_XG_boost.ipynb   ← v2: XGBoost + LSTM
├── 📓 ML Regressor XGBoost Model.ipynb          ← v1: XGBoost only
├── 📊 Merged_Hourly_DayAhead_2020_2025.xlsx     ← Sample data
├── 📄 README.md
└── 📄 LICENSE
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

**Built with ⚡ XGBoost · 🧠 LSTM · 🐼 Pandas · 📊 Plotly · 🪐 Jupyter**

*Forecast smarter. Trade better.*
