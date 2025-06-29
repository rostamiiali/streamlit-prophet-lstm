# LSTM Forecasting (PyTorch) with cyclical time features
st.write("---")
st.subheader("📉 LSTM Forecasting (PyTorch)")

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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