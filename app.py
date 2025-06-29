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