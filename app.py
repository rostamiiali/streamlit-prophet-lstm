from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
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

st.subheader("🔮 NeuralProphet Forecasting (Enhanced)")

neural_prophet = NeuralProphet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    seasonality_mode="multiplicative",
    learning_rate=1.0,
    epochs=100,
    country_holidays=None
)

neural_prophet.add_seasonality(name="monthly", period=30.5, fourier_order=10)

metrics = neural_prophet.fit(train_df, freq="MS")
future = neural_prophet.make_future_dataframe(train_df, periods=forecast_horizon, n_historic_predictions=True)
forecast = neural_prophet.predict(future)

# Evaluation
forecast_df = forecast[['ds', 'yhat1']].set_index('ds')
test_df.set_index('ds', inplace=True)
forecast_df = forecast_df.rolling(window=2, min_periods=1).mean()
merged = forecast_df.join(test_df, how='inner')
neural_rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat1']))
neural_mae = mean_absolute_error(merged['y'], merged['yhat1'])

st.write(f"### NeuralProphet Forecast RMSE: {neural_rmse:.2f}")
st.write(f"### NeuralProphet Forecast MAE: {neural_mae:.2f}")

# Plotting
st.subheader("📈 NeuralProphet Forecast vs Actual")
fig_np, ax_np = plt.subplots()
ax_np.plot(merged.index, merged['y'], label='Actual', color='black')
ax_np.plot(merged.index, merged['yhat1'], label='NeuralProphet Forecast', linestyle='--', color='purple')
ax_np.set_title("NeuralProphet Forecast vs Actual")
ax_np.set_xlabel("Date")
ax_np.set_ylabel("Value")
ax_np.legend()
st.pyplot(fig_np)

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

# SARIMAX Components Visualization
st.subheader("🧩 SARIMAX Forecast Components")
# Placeholder decomposition for SARIMAX (true components unavailable)
sarimax_trend = sarima_results.fittedvalues.rolling(window=3, min_periods=1).mean()
sarimax_resid = sarima_results.resid

fig_sarimax_comp, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(sarimax_trend, label="Trend")
axs[0].set_title("SARIMAX Trend (Rolling Avg)")
axs[1].plot(sarimax_resid, label="Residuals", color='green')
axs[1].set_title("SARIMAX Residuals")
plt.tight_layout()
st.pyplot(fig_sarimax_comp)

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

# --- Best Practice Holt-Winters Exponential Smoothing (HWES) ---
st.write("---")
st.subheader("📊 Holt-Winters Exponential Smoothing (Best Practice)")

# Prepare training series
hwes_series = train_df['y']

# Fit HWES model with optimized parameters
hwes_model = ExponentialSmoothing(
    hwes_series,
    trend="add",
    seasonal="mul",
    seasonal_periods=12,
    damped_trend=True,
    initialization_method="estimated"
)
hwes_fit = hwes_model.fit(optimized=True)

# Forecast
hwes_forecast = hwes_fit.forecast(steps=forecast_horizon)
hwes_forecast = pd.Series(hwes_forecast.values, index=test_df.index).clip(lower=0)

# Evaluation
hwes_rmse = np.sqrt(mean_squared_error(test_df['y'], hwes_forecast))
hwes_mae = mean_absolute_error(test_df['y'], hwes_forecast)

st.write(f"### HWES Forecast RMSE: {hwes_rmse:.2f}")
st.write(f"### HWES Forecast MAE: {hwes_mae:.2f}")

# HWES Components Visualization
st.subheader("🧩 HWES Forecast Components")
fig_hwes_comp, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
axs[0].plot(hwes_fit.level, label="Level")
axs[0].set_title("HWES Level")
axs[1].plot(hwes_fit.season, label="Season", color='orange')
axs[1].set_title("HWES Season")
axs[2].plot(hwes_fit.resid, label="Residuals", color='green')
axs[2].set_title("HWES Residuals")
plt.tight_layout()
st.pyplot(fig_hwes_comp)

# Plotting
st.write("### HWES Forecast vs Actual")
fig_hwes, ax_hwes = plt.subplots()
ax_hwes.plot(test_df.index, test_df['y'], label='Actual', color='black')
ax_hwes.plot(test_df.index, hwes_forecast, label='HWES Forecast', linestyle='--', color='blue')
ax_hwes.set_title("Holt-Winters Forecast vs Actual")
ax_hwes.set_xlabel("Date")
ax_hwes.set_ylabel("Value")
ax_hwes.legend()
st.pyplot(fig_hwes)

# --- XGBoost Forecasting ---
import xgboost as xgb

st.write("---")
st.subheader("📈 XGBoost Forecasting (Lag-Based Supervised Learning)")

def create_lagged_features(df, lags=12):
    df_lagged = df.copy()
    for lag in range(1, lags + 1):
        df_lagged[f"lag_{lag}"] = df_lagged['y'].shift(lag)
    df_lagged['month'] = df_lagged['ds'].dt.month
    df_lagged = df_lagged.dropna().reset_index(drop=True)
    return df_lagged

df_lagged = create_lagged_features(df, lags=12)

# Train/test split
xgb_train = df_lagged[df_lagged['ds'] < test_df.index[0]]
xgb_test = df_lagged[df_lagged['ds'] >= test_df.index[0]]

features = [col for col in xgb_train.columns if col not in ['ds', 'y']]
X_train, y_train = xgb_train[features], xgb_train['y']
X_test, y_test = xgb_test[features], xgb_test['y']

# Train model
xgb_model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=3)
xgb_model.fit(X_train, y_train)

# Forecast
xgb_pred = xgb_model.predict(X_test)
xgb_pred = pd.Series(xgb_pred, index=xgb_test['ds'])

# Evaluation
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_mae = mean_absolute_error(y_test, xgb_pred)

st.write(f"### XGBoost Forecast RMSE: {xgb_rmse:.2f}")
st.write(f"### XGBoost Forecast MAE: {xgb_mae:.2f}")

# XGBoost Plotting
st.write("### XGBoost Forecast vs Actual")
fig_xgb, ax_xgb = plt.subplots()
ax_xgb.plot(y_test.index, y_test.values, label='Actual', color='black')
ax_xgb.plot(y_test.index, xgb_pred.values, label='XGBoost Forecast', linestyle='--', color='teal')
ax_xgb.set_title("XGBoost Forecast vs Actual")
ax_xgb.set_xlabel("Date")
ax_xgb.set_ylabel("Value")
ax_xgb.legend()
st.pyplot(fig_xgb)