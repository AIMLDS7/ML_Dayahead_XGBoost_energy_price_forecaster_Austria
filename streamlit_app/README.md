<div align="center">

# ⚡ Austrian Day-Ahead Forecaster

**Gradient-boosted regression · 1/3/7/14-day horizon · 80% confidence bands**

[![🚀 Live Demo](https://img.shields.io/badge/🚀_Live_Demo-aimlds7--xgboost--dayahead-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://aimlds7-xgboost-dayahead.streamlit.app)
[![GitHub Repo](https://img.shields.io/badge/📂_Main_Repo-AIMLDS7-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AIMLDS7/ML_Dayahead_XGBoost_energy_price_forecaster_Austria)

![Demo Preview](https://raw.githubusercontent.com/AIMLDS7/ML_Dayahead_XGBoost_energy_price_forecaster_Austria/main/streamlit_app/preview.png)

Forecasts Austrian day-ahead electricity prices 1–14 days ahead using gradient-boosted regression with rolling lag features and quantile confidence intervals. Click the red **Live Demo** button above — no install required.

</div>

---

## 🧠 What's inside

- **Hero KPIs** — last actual, forecast mean, forecast high/low
- **Interactive forecast chart** — historical prices + 1/3/7/14-day forecast with 50–95% confidence band
- **Daily summary table** — low / mean / high price per day with intraday spread
- **Backtest scatter** — predicted vs actual on holdout set
- **Feature importance bar** — which inputs drive the forecast
- **Pattern explorer** — price by hour of day / day of week / month

## 🎛 Sidebar controls

- **Forecast horizon** — 1, 3, 7, or 14 days
- **Confidence level** — 50% to 95%
- **History shown** — 30 to 180 days of context

## 🛠 Tech stack

`Python` · `Streamlit` · `scikit-learn` (`GradientBoostingRegressor`) · `Pandas` · `NumPy` · `Plotly`

## 🚀 Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
