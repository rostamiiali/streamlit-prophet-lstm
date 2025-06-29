import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- Data Loading and Train/Test Split ---
# Forecast Range
forecast_horizon = st.slider("Select forecast horizon (months)", 6, 24, 12)

# Option to Upload or Generate Data
upload_option = st.radio("Choose input method:", ("Upload CSV", "Use Sample Data"))

if upload_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your time series CSV file (columns: ds, y)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df['ds'] = pd.to_datetime(df['ds'])
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    # Generate realistic sample data
    np.random.seed(42)
    date_rng = pd.date_range(start='2018-01-01', end='2025-01-01', freq='MS')
    seasonal = 3000 * np.sin(2 * np.pi * date_rng.month / 12)
    trend = np.linspace(25000, 35000, len(date_rng))
    noise = np.random.normal(0, 800, len(date_rng))
    data = np.clip(trend + seasonal + noise, 10000, 45000)
    df = pd.DataFrame({'ds': date_rng, 'y': data})
    df['ds'] = pd.to_datetime(df['ds'])

# Split into train/test based on forecast_horizon
df = df.sort_values('ds').reset_index(drop=True)
train_size = len(df) - forecast_horizon
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# Prophet Model (Expert-tuned)
prophet = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    seasonality_mode='multiplicative',
    daily_seasonality=False
)
prophet.add_seasonality(name='monthly', period=30.5, fourier_order=10)
prophet.fit(train_df)
future = prophet.make_future_dataframe(periods=forecast_horizon, freq='MS')
forecast = prophet.predict(future)
forecast_df = forecast[['ds', 'yhat']].set_index('ds')
test_df.set_index('ds', inplace=True)
forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0).rolling(window=2, min_periods=1).mean()
merged = forecast_df.join(test_df, how='inner')
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
mae = mean_absolute_error(merged['y'], merged['yhat'])
st.write(f"### Prophet Forecast RMSE: {rmse:.2f}")
st.write(f"### Prophet Forecast MAE: {mae:.2f}")
fig1 = prophet.plot(forecast)
st.pyplot(fig1)

# Show Prophet forecast components (trend, yearly seasonality, monthly seasonality)
st.subheader("📊 Prophet Forecast Components")
fig_comp = prophet.plot_components(forecast)
st.pyplot(fig_comp)

# Display prediction intervals for the next periods
st.subheader("🔮 Prophet Forecast with Confidence Intervals")
ci_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds')
st.line_chart(ci_df)

st.write("---")
st.subheader("🔍 SARIMAX Forecasting (Optimized)")

# Train SARIMAX on the training set
sarima_endog = train_df['y']
sarima_model = SARIMAX(
    sarima_endog,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_results = sarima_model.fit(disp=False)

# Forecast with confidence intervals
forecast_res = sarima_results.get_forecast(steps=forecast_horizon)
forecast_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int(alpha=0.05)
lower_series = conf_int.iloc[:, 0]
upper_series = conf_int.iloc[:, 1]

# Align to test index and cap negatives
sarima_forecast = pd.Series(forecast_mean.values, index=test_df.index).clip(lower=0)
sarima_lower = pd.Series(lower_series.values, index=test_df.index).clip(lower=0)
sarima_upper = pd.Series(upper_series.values, index=test_df.index).clip(lower=0)

# Compute metrics
sarima_rmse = np.sqrt(mean_squared_error(test_df['y'], sarima_forecast))
sarima_mae = mean_absolute_error(test_df['y'], sarima_forecast)
st.write(f"### SARIMAX Forecast RMSE: {sarima_rmse:.2f}")
st.write(f"### SARIMAX Forecast MAE: {sarima_mae:.2f}")

# Plot actual vs forecast with confidence band
st.write("### SARIMAX Forecast vs Actual with 95% CI")
fig_sarima, ax_sarima = plt.subplots()
ax_sarima.plot(test_df.index, test_df['y'], label='Actual', color='black')
ax_sarima.plot(test_df.index, sarima_forecast, label='Forecast', linestyle='--', color='orange')
ax_sarima.fill_between(test_df.index, sarima_lower, sarima_upper, color='orange', alpha=0.3)
ax_sarima.set_title("SARIMAX Forecast vs Actual")
ax_sarima.set_xlabel("Date")
ax_sarima.set_ylabel("Vaccinations")
ax_sarima.legend()
st.pyplot(fig_sarima)

st.write("---")
st.subheader("📊 Holt-Winters Forecasting")
holt_train = train_df['y']
holt_model = ExponentialSmoothing(holt_train, trend="add", seasonal="add", seasonal_periods=12).fit(smoothing_level=0.7, smoothing_slope=0.2, smoothing_seasonal=0.2)
holt_forecast = holt_model.forecast(steps=forecast_horizon)
holt_forecast = pd.Series(holt_forecast.values, index=test_df.index).clip(lower=0).rolling(window=2, min_periods=1).ffill().bfill()
holt_rmse = np.sqrt(mean_squared_error(test_df['y'], holt_forecast))
holt_mae = mean_absolute_error(test_df['y'], holt_forecast)
st.write(f"### Holt-Winters Forecast RMSE: {holt_rmse:.2f}")
st.write(f"### Holt-Winters Forecast MAE: {holt_mae:.2f}")
fig_hw, ax_hw = plt.subplots()
ax_hw.plot(test_df.index, test_df['y'], label='Actual', color='black')
ax_hw.plot(test_df.index, holt_forecast, label='Holt-Winters Forecast', linestyle='--', color='brown')
st.pyplot(fig_hw)