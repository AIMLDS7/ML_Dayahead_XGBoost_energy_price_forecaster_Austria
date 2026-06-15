"""
Austrian Day-Ahead Electricity Price Forecaster — Streamlit Demo
----------------------------------------------------------------
Live demo of the model in: AIMLDS7/ML_Dayahead_XGBoost_energy_price_forecaster_Austria
Uses synthetic EPEX-style data so the demo is NDA-safe and reproducible.

Run locally:    streamlit run app.py
Deploy:         share.streamlit.io (free, ~3 min)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AT Day-Ahead Forecaster · Live Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Synthetic data generator (mimics Austrian EPEX day-ahead patterns) ─────
@st.cache_data
def generate_austrian_day_ahead(seed: int = 42, n_days: int = 730) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    hour = np.asarray(timestamps.hour, dtype=int)
    dow = np.asarray(timestamps.dayofweek, dtype=int)
    doy = np.asarray(timestamps.dayofyear, dtype=int)

    base = 78.0
    annual = 28 * np.sin(2 * np.pi * (doy - 80) / 365)             # winter peak
    weekly = -10 * (dow >= 5).astype(float)                          # weekend dip
    morning_peak = 18 * np.exp(-((hour - 9) ** 2) / 5)              # 08-10 ramp
    evening_peak = 24 * np.exp(-((hour - 19) ** 2) / 6)             # 18-21 peak
    solar_duck = -10 * np.exp(-((hour - 13) ** 2) / 10)             # midday PV duck
    spike_mask = rng.random(len(timestamps)) < 0.004                # rare spikes
    spikes = spike_mask * rng.exponential(95, len(timestamps))
    noise = rng.normal(0, 7, len(timestamps))

    prices = base + annual + weekly + morning_peak + evening_peak + solar_duck + spikes + noise
    prices = np.maximum(np.asarray(prices, dtype=float), -5.0)

    return pd.DataFrame({
        "timestamp": timestamps,
        "price": prices.round(2),
        "hour": hour,
        "dow": dow,
        "month": timestamps.month,
        "is_weekend": (dow >= 5).astype(int),
    })


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["lag_24"] = out["price"].shift(24)
    out["lag_48"] = out["price"].shift(48)
    out["lag_168"] = out["price"].shift(168)
    out["roll_mean_24"] = out["price"].rolling(24).mean()
    out["roll_mean_168"] = out["price"].rolling(168).mean()
    out["roll_std_24"] = out["price"].rolling(24).std()
    return out.dropna().reset_index(drop=True)


FEATURES = ["hour", "dow", "month", "is_weekend", "lag_24", "lag_48", "lag_168",
            "roll_mean_24", "roll_mean_168", "roll_std_24"]


@st.cache_resource
def train_models(df_feat: pd.DataFrame, confidence: float):
    split = int(len(df_feat) * 0.85)
    X_train, X_test = df_feat[FEATURES].iloc[:split], df_feat[FEATURES].iloc[split:]
    y_train, y_test = df_feat["price"].iloc[:split], df_feat["price"].iloc[split:]

    alpha_low = (1 - confidence) / 2
    alpha_hi = 1 - alpha_low

    model_low = GradientBoostingRegressor(loss="quantile", alpha=alpha_low,
                                           n_estimators=120, max_depth=5,
                                           learning_rate=0.08, random_state=42)
    model_med = GradientBoostingRegressor(n_estimators=120, max_depth=5,
                                          learning_rate=0.08, random_state=42)
    model_hi = GradientBoostingRegressor(loss="quantile", alpha=alpha_hi,
                                          n_estimators=120, max_depth=5,
                                          learning_rate=0.08, random_state=42)

    model_low.fit(X_train, y_train)
    model_med.fit(X_train, y_train)
    model_hi.fit(X_train, y_train)

    y_pred = model_med.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "models": (model_low, model_med, model_hi),
        "metrics": {"MAE (€/MWh)": round(mae, 2), "R²": round(r2, 3)},
        "test_idx": df_feat["timestamp"].iloc[split:].reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "y_pred": pd.Series(y_pred, name="forecast"),
        "feat_importance": pd.Series(model_med.feature_importances_, index=FEATURES, name="importance").sort_values(ascending=False),
    }


def forecast_future(df_feat: pd.DataFrame, models, horizon_days: int) -> pd.DataFrame:
    """Iterative multi-step forecast with rolling lag updates."""
    model_low, model_med, model_hi = models
    df = df_feat.copy()
    last_ts = df["timestamp"].iloc[-1]
    future_ts = pd.date_range(last_ts + pd.Timedelta(hours=1),
                              periods=horizon_days * 24, freq="h")
    rows = []
    for ts in future_ts:
        row = {
            "timestamp": ts,
            "hour": ts.hour,
            "dow": ts.dayofweek,
            "month": ts.month,
            "is_weekend": int(ts.dayofweek >= 5),
        }
        # Use last known 24/48/168-hour-ago values
        if len(df) >= 24:
            row["lag_24"] = df["price"].iloc[-24]
        else:
            row["lag_24"] = df["price"].iloc[-1]
        row["lag_48"] = df["price"].iloc[-48] if len(df) >= 48 else row["lag_24"]
        row["lag_168"] = df["price"].iloc[-168] if len(df) >= 168 else row["lag_24"]
        row["roll_mean_24"] = df["price"].iloc[-24:].mean()
        row["roll_mean_168"] = df["price"].iloc[-168:].mean()
        row["roll_std_24"] = df["price"].iloc[-24:].std()

        X_row = pd.DataFrame([row])[FEATURES]
        row["p10"] = model_low.predict(X_row)[0]
        row["p50"] = model_med.predict(X_row)[0]
        row["p90"] = model_hi.predict(X_row)[0]
        # Update df for next iteration (use p50 as the "realised" value)
        row["price"] = row["p50"]
        rows.append(row)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return pd.DataFrame(rows)


# ── Hero ────────────────────────────────────────────────────────────────────
df_raw = generate_austrian_day_ahead()
df_feat = add_features(df_raw)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Forecast Settings")
    horizon = st.select_slider("Forecast horizon", options=[1, 3, 7, 14], value=7)
    confidence = st.slider("Confidence level", 0.50, 0.95, 0.90, 0.05)
    history_days = st.slider("History shown (days)", 30, 180, 90)
    st.markdown("---")
    st.markdown("**Model:** Gradient-boosted regression  \n"
                "**Features:** 10 (lags + rolling stats + calendar)")
    st.markdown("---")
    st.caption("Synthetic EPEX-style data — NDA-safe demo.  \n"
               "Real model trained on 2020-2025 EPEX auction data is in the repo.")

# Train models with current confidence level
bundle = train_models(df_feat, confidence)
future = forecast_future(df_feat, bundle["models"], horizon)

# Top KPI row
last_actual = df_raw["price"].iloc[-1]
fc_mean = future["p50"].mean()
fc_max = future["p50"].max()
fc_min = future["p50"].min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last actual", f"€{last_actual:.0f}/MWh")
c2.metric(f"{horizon}-day mean forecast", f"€{fc_mean:.0f}/MWh",
          delta=f"{fc_mean - last_actual:+.0f}")
c3.metric("Forecast high", f"€{fc_max:.0f}/MWh")
c4.metric("Forecast low", f"€{fc_min:.0f}/MWh")

st.markdown("---")

# ── Main forecast chart ─────────────────────────────────────────────────────
st.markdown(f"### 📈 Price history + {horizon}-day forecast (€{confidence*100:.0f}% CI)")
history_window = df_raw.tail(history_days * 24)
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=history_window["timestamp"], y=history_window["price"],
    name="Historical", line=dict(color="#9ca3af", width=1.5),
    hovertemplate="%{x|%a %d %b %H:%M}<br>€%{y:.1f}/MWh<extra></extra>",
))

# CI band first (so it's behind the line)
fig.add_trace(go.Scatter(
    x=future["timestamp"], y=future["p90"],
    name=f"Upper {confidence*100:.0f}%", line=dict(width=0),
    showlegend=False, hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=future["timestamp"], y=future["p10"],
    name=f"{confidence*100:.0f}% CI", fill="tonexty",
    fillcolor="rgba(240, 180, 41, 0.20)", line=dict(width=0),
    hoverinfo="skip",
))
fig.add_trace(go.Scatter(
    x=future["timestamp"], y=future["p50"],
    name="Forecast (median)", line=dict(color="#f0b429", width=2.5),
    hovertemplate="%{x|%a %d %b %H:%M}<br>€%{y:.1f}/MWh<extra></extra>",
))

fig.update_layout(
    height=460, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified", xaxis_title="", yaxis_title="€/MWh",
    legend=dict(orientation="h", y=1.08, x=0),
    margin=dict(l=10, r=10, t=30, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ── Daily summary table ─────────────────────────────────────────────────────
st.markdown("### 🗓️ Daily forecast summary")
daily = future.copy()
daily["date"] = daily["timestamp"].dt.date
daily["day"] = daily["timestamp"].dt.strftime("%a %d %b")
summary = daily.groupby("day").agg(
    Low=("p10", "min"),
    Mean=("p50", "mean"),
    High=("p90", "max"),
).round(1).reset_index()
summary.columns = ["Day", "Low (€/MWh)", "Mean (€/MWh)", "High (€/MWh)"]

def color_negative(val):
    return "color: #ef4444" if val < 0 else "color: #f0b429"

st.dataframe(
    summary.style.map(color_negative, subset=["Low (€/MWh)", "Mean (€/MWh)", "High (€/MWh)"]),
    use_container_width=True, hide_index=True,
)

# ── Model performance + feature importance ──────────────────────────────────
col_a, col_b = st.columns([1, 1])

with col_a:
    st.markdown("### 🎯 Backtest performance")
    metrics_df = pd.DataFrame({
        "Metric": list(bundle["metrics"].keys()),
        "Value": list(bundle["metrics"].values()),
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    st.caption("Holdout: last 15% of synthetic data.")

    fig2 = go.Figure()
    sample_n = min(500, len(bundle["y_test"]))
    fig2.add_trace(go.Scatter(
        x=bundle["y_test"].iloc[:sample_n], y=bundle["y_pred"].iloc[:sample_n],
        mode="markers", marker=dict(color="#f0b429", size=5, opacity=0.6),
        hovertemplate="Actual: €%{x:.0f}<br>Predicted: €%{y:.0f}<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=[bundle["y_test"].min(), bundle["y_test"].max()],
        y=[bundle["y_test"].min(), bundle["y_test"].max()],
        mode="lines", line=dict(color="#9ca3af", dash="dash"),
        name="Perfect forecast",
    ))
    fig2.update_layout(
        height=340, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Actual (€/MWh)", yaxis_title="Predicted (€/MWh)",
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown("### 🔬 Feature importance")
    fig3 = px.bar(
        bundle["feat_importance"].reset_index(),
        x="importance", y="index", orientation="h",
        color_discrete_sequence=["#f0b429"],
    )
    fig3.update_layout(
        height=340, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importance", yaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Pattern explorer**")
    pattern_choice = st.selectbox("Show price by", ["Hour of day", "Day of week", "Month"])
    df_pattern = df_raw.copy()
    if pattern_choice == "Hour of day":
        agg = df_pattern.groupby("hour")["price"].agg(["mean", "std"]).reset_index()
        x_col, x_label = "hour", "Hour of day"
    elif pattern_choice == "Day of week":
        agg = df_pattern.groupby("dow")["price"].agg(["mean", "std"]).reset_index()
        x_col, x_label = "dow", "Day of week (0=Mon)"
    else:
        agg = df_pattern.groupby("month")["price"].agg(["mean", "std"]).reset_index()
        x_col, x_label = "month", "Month"
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=agg[x_col], y=agg["mean"],
        error_y=dict(type="data", array=agg["std"], color="rgba(240,180,41,0.3)"),
        mode="lines+markers", line=dict(color="#f0b429", width=2),
        marker=dict(size=8),
    ))
    fig4.update_layout(
        height=220, template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title=x_label, yaxis_title="€/MWh",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig4, use_container_width=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#9ca3af; font-size:14px;'>
    ⚡ <b>Energy Data Science Demo</b> · Synthetic data for reproducibility ·
    <a href='https://github.com/AIMLDS7/ML_Dayahead_XGBoost_energy_price_forecaster_Austria'
       style='color:#f0b429;'>View the real model on GitHub →</a>
    </div>
    """,
    unsafe_allow_html=True,
)
