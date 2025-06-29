import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Title
st.title("📈 Flu Vaccination Forecasting with Prophet")

# Forecast horizon slider
forecast_horizon = st.slider("Forecast horizon (months)", min_value=1, max_value=24, value=12)

# Data input option
data_option = st.radio("Data source:", ("Upload CSV", "Use Sample Data"))

# Load data
if data_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV (columns: ds, y)", type=["csv"])
    if uploaded_file is None:
        st.warning("Please upload a CSV file.")
        st.stop()
    df = pd.read_csv(uploaded_file)
    st.success("File loaded successfully.")
else:
    # Generate sample monthly data from 2018-01 to present
    rng = pd.date_range(start="2018-01-01", end=pd.Timestamp.today(), freq="MS")
    np.random.seed(42)
    seasonal = 3000 * np.sin(2 * np.pi * rng.month / 12)
    trend = np.linspace(20000, 40000, len(rng))
    noise = np.random.normal(0, 1000, len(rng))
    values = np.clip(trend + seasonal + noise, a_min=0, a_max=None)
    df = pd.DataFrame({"ds": rng, "y": values})
    st.info("Using generated sample data.")

# Preprocess
df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values("ds").reset_index(drop=True)

# Show data
st.subheader("Raw data")
st.dataframe(df.tail(10))

# Split train/test
train = df.iloc[:-forecast_horizon].copy()
test = df.iloc[-forecast_horizon:].copy()

# Fit Prophet
m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
m.fit(train)

# Forecast
future = m.make_future_dataframe(periods=forecast_horizon, freq="MS")
forecast = m.predict(future)

# Extract forecasted values
fc = forecast.set_index("ds")["yhat"].iloc[-forecast_horizon:]
fc = fc.clip(lower=0)

# Metrics
y_true = test.set_index("ds")["y"]
rmse = np.sqrt(mean_squared_error(y_true, fc))
mae = mean_absolute_error(y_true, fc)

# Display metrics
st.subheader("Forecast Accuracy")
st.write(f"RMSE: **{rmse:.2f}**")
st.write(f"MAE: **{mae:.2f}**")

# Plot
st.subheader("Forecast Plot")
fig, ax = plt.subplots()
ax.plot(train["ds"], train["y"], label="Train", color="black")
ax.plot(test["ds"], test["y"], label="Actual", color="blue")
ax.plot(fc.index, fc.values, label="Forecast", linestyle="--", color="red")
ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel("Value")
st.pyplot(fig)

st.write("---")
st.subheader("🔍 SARIMA Forecasting (Optimized)")

import statsmodels.api as sm

# Prepare the series at a fixed monthly frequency
df_sarima = df.copy()
df_sarima['ds'] = pd.to_datetime(df_sarima['ds'])
df_sarima = df_sarima.set_index('ds').asfreq('MS')
y = df_sarima['y'].astype(float)

# Split into train/test
train_sarima, test_sarima = y.iloc[:-forecast_horizon], y.iloc[-forecast_horizon:]

# Fit SARIMAX with seasonal period 12
sarima_model = sm.tsa.statespace.SARIMAX(
    train_sarima,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_res = sarima_model.fit(disp=False)

# Forecast out of sample
sarima_pred = sarima_res.get_forecast(steps=forecast_horizon)
sarima_fc = sarima_pred.predicted_mean.clip(lower=0)
sarima_ci = sarima_pred.conf_int(alpha=0.05)

# Compute metrics
sarima_rmse = np.sqrt(mean_squared_error(test_sarima, sarima_fc))
sarima_mae = mean_absolute_error(test_sarima, sarima_fc)
st.write(f"### SARIMA Forecast RMSE: {sarima_rmse:.2f}")
st.write(f"### SARIMA Forecast MAE: {sarima_mae:.2f}")

# Plot SARIMA results
fig_sarima, ax_sarima = plt.subplots()
ax_sarima.plot(train_sarima.index, train_sarima, label='Train', color='black')
ax_sarima.plot(test_sarima.index, test_sarima, label='Actual', color='blue')
ax_sarima.plot(sarima_fc.index, sarima_fc.values, label='SARIMA Forecast', linestyle='--', color='orange')
ax_sarima.fill_between(sarima_ci.index, sarima_ci.iloc[:,0], sarima_ci.iloc[:,1], color='orange', alpha=0.2)
ax_sarima.set_title("SARIMA Forecast vs Actual")
ax_sarima.set_xlabel("Date")
ax_sarima.set_ylabel("Vaccinations")
ax_sarima.legend()
st.pyplot(fig_sarima)

# Holt-Winters Model (Robust)
# ... (rest of the code remains unchanged)