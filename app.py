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

st.write("---")
st.subheader("🔗 ARIMA + Prophet Hybrid")
prophet_residuals = train_df['y'] - prophet.predict(train_df)['yhat']
arima_model = ARIMA(prophet_residuals, order=(1,0,0)).fit()
arima_forecast = arima_model.forecast(steps=forecast_horizon)
hybrid_forecast = forecast_df['yhat'][-forecast_horizon:] + arima_forecast.values
hybrid_forecast = pd.Series(hybrid_forecast, index=test_df.index).clip(lower=0).rolling(window=2, min_periods=1).mean()
combined_df = pd.DataFrame({'ds': test_df.index, 'Hybrid': hybrid_forecast, 'Actual': test_df['y']})
hybrid_rmse = np.sqrt(mean_squared_error(combined_df['Actual'], combined_df['Hybrid']))
hybrid_mae = mean_absolute_error(combined_df['Actual'], combined_df['Hybrid'])
st.write(f"### Hybrid Forecast RMSE: {hybrid_rmse:.2f}")
st.write(f"### Hybrid Forecast MAE: {hybrid_mae:.2f}")
fig_hybrid, ax_hybrid = plt.subplots()
ax_hybrid.plot(combined_df['ds'], combined_df['Actual'], label='Actual', color='black')
ax_hybrid.plot(combined_df['ds'], combined_df['Hybrid'], label='Hybrid Forecast', linestyle='--', color='purple')
st.pyplot(fig_hybrid)

st.write("---")
st.subheader("🔍 SARIMA Forecasting (Optimized)")
sarima_train = train_df['y']
sarima_model = SARIMAX(
    sarima_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
sarima_forecast = sarima_model.forecast(steps=forecast_horizon)
sarima_forecast = pd.Series(sarima_forecast.values, index=test_df.index).clip(lower=0).rolling(window=2, min_periods=1).ffill().bfill()
sarima_rmse = np.sqrt(mean_squared_error(test_df['y'], sarima_forecast))
sarima_mae = mean_absolute_error(test_df['y'], sarima_forecast)
st.write(f"### SARIMA Forecast RMSE: {sarima_rmse:.2f}")
st.write(f"### SARIMA Forecast MAE: {sarima_mae:.2f}")
fig_sarima, ax_sarima = plt.subplots()
ax_sarima.plot(test_df.index, test_df['y'], label='Actual', color='black')
ax_sarima.plot(test_df.index, sarima_forecast, label='SARIMA Forecast', linestyle='--', color='orange')
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

st.write("---")
st.subheader("📉 LSTM Forecasting (PyTorch)")

from sklearn.preprocessing import MinMaxScaler

# Add cyclical month-of-year features
df_lstm = df.copy()
df_lstm['month'] = df_lstm['ds'].dt.month
df_lstm['month_sin'] = np.sin(2 * np.pi * df_lstm['month'] / 12)
df_lstm['month_cos'] = np.cos(2 * np.pi * df_lstm['month'] / 12)
features = ['y', 'month_sin', 'month_cos']

# Scale features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_lstm[features])

# Prepare sequences
sequence_length = 12
X = []
y = []
for i in range(len(scaled) - sequence_length):
    X.append(scaled[i : i + sequence_length])
    y.append(scaled[i + sequence_length, 0])  # target is the y-value
X = np.array(X)
y = np.array(y)

# Split train/test
train_X, train_y = X[:-forecast_horizon], y[:-forecast_horizon]
test_X, test_y = X[-forecast_horizon:], y[-forecast_horizon:]

# Convert to tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_y = torch.tensor(train_y, dtype=torch.float32).unsqueeze(-1)
train_ds = TensorDataset(train_X, train_y)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# LSTM model definition
class LSTMForecast(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMForecast()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Train
model.train()
for epoch in range(100):
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 20 == 0:
        st.write(f"Epoch {epoch}, loss={total_loss/len(train_loader):.4f}")

# Forecast
model.eval()
with torch.no_grad():
    input_seq = torch.tensor(train_X[-sequence_length:], dtype=torch.float32).unsqueeze(0)
    preds = []
    for _ in range(forecast_horizon):
        out = model(input_seq)
        preds.append(out.item())
        # append next input (drop first, add new features for predicted y)
        next_month = (df_lstm['month'].iloc[-forecast_horizon + len(preds)-1] % 12) + 1
        sin, cos = np.sin(2*np.pi*next_month/12), np.cos(2*np.pi*next_month/12)
        new_feat = torch.tensor([[out.item(), sin, cos]], dtype=torch.float32)
        input_seq = torch.cat([input_seq[:, 1:, :], new_feat.unsqueeze(0)], dim=1)

# Inverse scale predictions
preds = np.array(preds).reshape(-1, 1)
dummy = np.zeros((len(preds), 2))
scaled_preds = np.concatenate([preds, dummy], axis=1)
inv = scaler.inverse_transform(scaled_preds)[:, 0]

# Compute metrics
lstm_rmse = np.sqrt(mean_squared_error(test_y, inv))
lstm_mae = mean_absolute_error(test_y, inv)
st.write(f"### LSTM Forecast RMSE: {lstm_rmse:.2f}")
st.write(f"### LSTM Forecast MAE: {lstm_mae:.2f}")

# Plot
fig_lstm, ax_lstm = plt.subplots()
test_dates = df_lstm['ds'].iloc[-forecast_horizon:]
ax_lstm.plot(test_dates, df_lstm['y'].iloc[-forecast_horizon:], label='Actual', color='blue')
ax_lstm.plot(test_dates, inv, label='LSTM Forecast', linestyle='--', color='green')
ax_lstm.set_title("LSTM Forecast vs Actual")
ax_lstm.set_xlabel("Date")
ax_lstm.set_ylabel("Vaccinations")
ax_lstm.legend()
st.pyplot(fig_lstm)