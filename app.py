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

# --- Improved Transformer-based Forecasting ---
from torch.nn import TransformerEncoder, TransformerEncoderLayer

st.write("---")
st.subheader("🚀 Transformer Forecasting (Optimized Architecture with Positional Encoding)")

# Fit scaler only on training data to avoid leakage
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df[['y']].values).flatten()
scaled_full = scaler.transform(df[['y']].values).flatten()

def create_sequences(data, input_len, output_len):
    xs, ys = [], []
    for i in range(len(data) - input_len - output_len):
        x = data[i:(i+input_len)]
        y = data[(i+input_len):(i+input_len+output_len)]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

input_len = 12
output_len = forecast_horizon
X_seq, y_seq = create_sequences(scaled_train, input_len, output_len)

X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(-1)
y_seq_tensor = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(-1)

# Final sequence for testing
test_input = torch.tensor(scaled_full[-(input_len + output_len):-output_len], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=2, num_layers=2, dim_feedforward=64):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(d_model, output_len)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc_out(x)

model = TransformerTimeSeries()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train model
epochs = 80
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    out = model(X_seq_tensor)
    loss = loss_fn(out, y_seq_tensor.squeeze())
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        st.text(f"Transformer Epoch {epoch}: loss={loss.item():.4f}")

# Predict
model.eval()
with torch.no_grad():
    pred = model(test_input).squeeze().numpy()

# Rescale prediction
pred_rescaled = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
true_rescaled = test_df['y'].values[:forecast_horizon]

# Evaluation
transformer_rmse = np.sqrt(mean_squared_error(true_rescaled, pred_rescaled))
transformer_mae = mean_absolute_error(true_rescaled, pred_rescaled)

st.write(f"### Transformer Forecast RMSE: {transformer_rmse:.2f}")
st.write(f"### Transformer Forecast MAE: {transformer_mae:.2f}")

# Plotting
st.write("### Transformer Forecast vs Actual")
fig_tf, ax_tf = plt.subplots()
ax_tf.plot(test_df.index[:forecast_horizon], true_rescaled, label='Actual', color='black')
ax_tf.plot(test_df.index[:forecast_horizon], pred_rescaled, label='Transformer Forecast', linestyle='--', color='purple')
ax_tf.set_title("Transformer Forecast vs Actual")
ax_tf.set_xlabel("Date")
ax_tf.set_ylabel("Value")
ax_tf.legend()
st.pyplot(fig_tf)