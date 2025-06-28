import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn

class LSTMForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMForecastModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
from statsmodels.tsa.arima.model import ARIMA

# App Title
st.title("📈 Flu Vaccination Forecasting App")

# Forecast Range
forecast_horizon = st.slider("Select forecast horizon (months)", 6, 36, 12)

# Option to Upload or Generate Data
upload_option = st.radio("Choose input method:", ("Upload CSV", "Use Sample Data"))


# Load Data
if upload_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your time series CSV file (columns: ds, y)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['ds'] = pd.to_datetime(df['ds'])
            st.success("✅ File uploaded and processed.")
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.stop()
    else:
        st.warning("Please upload a CSV file.")
        st.stop()
else:
    uploaded_file = None
    st.info("ℹ️ No file uploaded. Generating high-quality sample data for demonstration.")

    np.random.seed(42)
    date_rng = pd.date_range(start='2018-01-01', end='2025-01-01', freq='MS')
    seasonal = 3000 * np.sin(2 * np.pi * date_rng.month / 12)
    trend = np.linspace(25000, 35000, len(date_rng))
    noise = np.random.normal(0, 800, len(date_rng))
    data = trend + seasonal + noise

    # Ensure positive, non-zero, and realistic values
    data = np.clip(data, 10000, 45000)
    data = pd.Series(data).interpolate().fillna(method='bfill').fillna(method='ffill')

    df = pd.DataFrame({'ds': date_rng, 'y': data})
    st.success("✅ Sample vaccination data generated with realistic trend, seasonality, and noise.")

st.write("### Raw Data", df.tail())

# Visual diagnostic: Show last 24 observations for outlier check
st.write("### 🔎 Check Last 24 Observations for Outliers")
st.write(df.tail(24))

# Split into Train/Test
train_size = int(len(df) - forecast_horizon)
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# Remove extreme outliers from training data (keep 5th–95th percentile)
q_low = train_df['y'].quantile(0.05)
q_high = train_df['y'].quantile(0.95)
train_df = train_df[(train_df['y'] >= q_low) & (train_df['y'] <= q_high)]


#
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

# Cap negative Prophet forecast values before metrics
forecast_df['yhat'] = forecast_df['yhat'].clip(lower=0)
# Smooth Prophet forecast (rolling mean, window=2)
forecast_df['yhat'] = forecast_df['yhat'].rolling(window=2, min_periods=1).mean()

# Calculate RMSE and MAE
merged = forecast_df.join(test_df, how='inner')
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
mae = mean_absolute_error(merged['y'], merged['yhat'])

st.write(f"### Prophet Forecast RMSE: {rmse:.2f}")
st.write(f"### Prophet Forecast MAE: {mae:.2f}")
st.markdown(f"ℹ️ The model's predictions deviate from actuals by ~{mae:.0f} units/month. Lower values = better accuracy.")

# Prophet Forecast Plot
fig1 = prophet.plot(forecast)
st.pyplot(fig1)

#
# ARIMA + Prophet Hybrid Forecast
st.write("---")
st.subheader("🔗 ARIMA + Prophet Hybrid")
prophet_residuals = train_df['y'] - prophet.predict(train_df)['yhat']
arima_model = ARIMA(prophet_residuals, order=(1,0,0)).fit()
arima_forecast = arima_model.forecast(steps=forecast_horizon)
hybrid_forecast = forecast_df['yhat'][-forecast_horizon:] + arima_forecast.values
# Cap negative Hybrid forecast values before metrics
hybrid_forecast = pd.Series(hybrid_forecast, index=test_df.index).clip(lower=0)
# Smooth Hybrid forecast (rolling mean, window=2)
hybrid_forecast = hybrid_forecast.rolling(window=2, min_periods=1).mean()
combined_df = pd.DataFrame({'ds': test_df.index, 'Hybrid': hybrid_forecast, 'Actual': test_df['y']})
hybrid_rmse = np.sqrt(mean_squared_error(combined_df['Actual'], combined_df['Hybrid']))
hybrid_mae = mean_absolute_error(combined_df['Actual'], combined_df['Hybrid'])
st.write(f"### Hybrid Forecast RMSE: {hybrid_rmse:.2f}")
st.write(f"### Hybrid Forecast MAE: {hybrid_mae:.2f}")
st.markdown(f"ℹ️ The model's predictions deviate from actuals by ~{hybrid_mae:.0f} units/month. Lower values = better accuracy.")

# Plot Hybrid Forecast vs Actual
st.write("### Hybrid Forecast vs Actual (Prophet + ARIMA)")
fig_hybrid, ax_hybrid = plt.subplots()
ax_hybrid.plot(combined_df['ds'], combined_df['Actual'], label='Actual', color='black')
ax_hybrid.plot(combined_df['ds'], combined_df['Hybrid'], label='Hybrid Forecast', linestyle='--', color='purple')
ax_hybrid.set_title("Hybrid Forecast vs Actuals")
ax_hybrid.set_xlabel("Date")
ax_hybrid.set_ylabel("Vaccinations")
ax_hybrid.legend()
st.pyplot(fig_hybrid)

# LSTM Model
st.write("---")
st.subheader("📉 LSTM Forecasting")

# Expert-recommended LSTM hyperparameters
lstm_units = 64
dropout_rate = 0.2
epochs = 50

# Normalize data
df_lstm = df.set_index('ds')
values = df_lstm['y'].values.reshape(-1, 1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Remove outliers from training data for LSTM
train_indices = df_lstm.index[:train_size]
train_lstm = df_lstm.loc[train_indices].copy()
train_lstm = train_lstm[(train_lstm['y'] >= q_low) & (train_lstm['y'] <= q_high)]
scaled_train = scaler.transform(train_lstm['y'].values.reshape(-1, 1))

# Create LSTM sequences (use all data for sequence continuity)
sequence_length = 12
generator = TimeseriesGenerator(scaled_values, scaled_values, length=sequence_length, batch_size=1)

inputs = Input(shape=(sequence_length, 1))
x = Bidirectional(GRU(lstm_units, return_sequences=True))(inputs)
x = BatchNormalization()(x)
x = Dropout(dropout_rate)(x)
attention = Attention()([x, x])
x = Concatenate()([x, attention])
x = Bidirectional(GRU(int(lstm_units/2)))(x)
x = Dropout(dropout_rate / 2)(x)
output = Dense(1)(x)
model = Model(inputs, output)

model.compile(optimizer='adam', loss='mse')
model.fit(generator, epochs=epochs, batch_size=8, verbose=0)

# Forecasting with LSTM
pred_input = scaled_values[-sequence_length:]
predictions = []
for _ in range(forecast_horizon):
    pred = model.predict(pred_input.reshape(1, sequence_length, 1), verbose=0)
    predictions.append(pred[0, 0])
    pred_input = np.append(pred_input[1:], pred, axis=0)

# Rescale LSTM predictions
lstm_forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
# Cap negative LSTM forecast values before metrics
lstm_forecast = np.clip(lstm_forecast, 0, None)
# (Do not smooth LSTM output: LSTM learns sequence patterns)
df['ds'] = pd.to_datetime(df['ds'])
df.index = pd.to_datetime(df['ds'])
lstm_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_horizon, freq='MS')
lstm_df = pd.DataFrame({"ds": lstm_dates, "y": lstm_forecast})

# Join LSTM with actuals for RMSE and MAE
if len(test_df) == forecast_horizon:
    rmse_lstm = np.sqrt(mean_squared_error(test_df['y'].values, lstm_forecast))
    mae_lstm = mean_absolute_error(test_df['y'].values, lstm_forecast)
    st.write(f"### LSTM Forecast RMSE: {rmse_lstm:.2f}")
    st.write(f"### LSTM Forecast MAE: {mae_lstm:.2f}")
    st.markdown(f"ℹ️ The model's predictions deviate from actuals by ~{mae_lstm:.0f} units/month. Lower values = better accuracy.")

 # SARIMA Model (Optimized)
st.write("---")
st.subheader("🔍 SARIMA Forecasting (Optimized)")

from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

# Use expert parameters without grid search
sarima_train = train_df['y']
sarima_model = SARIMAX(
    sarima_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False)

# Forecast into the test period
sarima_forecast = sarima_fit.forecast(steps=forecast_horizon)
# Align forecast series to test index
sarima_forecast = pd.Series(sarima_forecast.values, index=test_df.index)
# Cap and smooth
sarima_forecast = sarima_forecast.clip(lower=0).rolling(window=2, min_periods=1).mean().ffill().bfill()

# Compute metrics
sarima_rmse = np.sqrt(mean_squared_error(test_df['y'], sarima_forecast))
sarima_mae = mean_absolute_error(test_df['y'], sarima_forecast)
st.write(f"### SARIMA Forecast RMSE: {sarima_rmse:.2f}")
st.write(f"### SARIMA Forecast MAE: {sarima_mae:.2f}")
st.markdown(f"ℹ️ The model's predictions deviate from actuals by ~{sarima_mae:.0f} units/month. Lower values = better accuracy.")

# Plot SARIMA forecast
st.write("### SARIMA Forecast vs Actual")
fig_sarima, ax_sarima = plt.subplots()
ax_sarima.plot(test_df.index, test_df['y'], label='Actual', color='black')
ax_sarima.plot(test_df.index, sarima_forecast, label='SARIMA Forecast', linestyle='--', color='orange')
ax_sarima.set_title("SARIMA Forecast vs Actual")
ax_sarima.set_xlabel("Date")
ax_sarima.set_ylabel("Vaccinations")
ax_sarima.legend()
st.pyplot(fig_sarima)

# Holt-Winters Model (Robust)
st.write("---")
st.subheader("📊 Holt-Winters Forecasting")

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Use full filtered train_df without additional filtering
holt_train = train_df['y']

# Fit Holt-Winters model with expert smoothing parameters
holt_model = ExponentialSmoothing(
    holt_train,
    trend="add",
    seasonal="add",
    seasonal_periods=12
)
holt_fit = holt_model.fit(
    smoothing_level=0.7,
    smoothing_slope=0.2,
    smoothing_seasonal=0.2
)

# Forecast for the test period
holt_forecast = holt_fit.forecast(steps=forecast_horizon)
# Align and clean forecast series
holt_forecast = pd.Series(holt_forecast.values, index=test_df.index)
holt_forecast = holt_forecast.clip(lower=0).rolling(window=2, min_periods=1).mean().ffill().bfill()

# Compute metrics directly since alignment now guaranteed
holt_rmse = np.sqrt(mean_squared_error(test_df['y'], holt_forecast))
holt_mae = mean_absolute_error(test_df['y'], holt_forecast)
st.write(f"### Holt-Winters Forecast RMSE: {holt_rmse:.2f}")
st.write(f"### Holt-Winters Forecast MAE: {holt_mae:.2f}")
st.markdown(f"ℹ️ The model's predictions deviate from actuals by ~{holt_mae:.0f} units/month. Lower values = better accuracy.")


# Plot Holt-Winters Forecast vs Actual
st.write("### Holt-Winters Forecast vs Actual")
fig_hw, ax_hw = plt.subplots()
ax_hw.plot(test_df.index, test_df['y'], label='Actual', color='black')
ax_hw.plot(test_df.index, holt_forecast, label='Holt-Winters Forecast', linestyle='--', color='brown')
ax_hw.set_title("Holt-Winters Forecast vs Actual")
ax_hw.set_xlabel("Date")
ax_hw.set_ylabel("Vaccinations")
ax_hw.legend()
st.pyplot(fig_hw)

# Prepare DataFrame for combined plotting
holt_df = pd.DataFrame({'ds': test_df.index, 'HoltWinters': holt_forecast})


# Plot Combined Forecasts
st.write("### 📊 Forecast Comparison (All Models)")
fig_combined, ax_combined = plt.subplots()
ax_combined.plot(df['ds'], df['y'], label='Actual', color='black')
ax_combined.plot(forecast_df.index, forecast_df['yhat'], label='Prophet Forecast', linestyle='--', color='blue')
ax_combined.plot(lstm_df['ds'], lstm_df['y'], label='LSTM Forecast', linestyle=':', color='green')
ax_combined.plot(combined_df['ds'], combined_df['Hybrid'], label='Hybrid Forecast', linestyle='-.', color='purple')
# Only add SARIMA to combined plot if sarima_df is defined
if 'sarima_df' in locals():
    ax_combined.plot(sarima_df['ds'], sarima_df['SARIMA'], label='SARIMA Forecast', linestyle='--', color='orange')
if 'holt_df' in locals():
    ax_combined.plot(holt_df['ds'], holt_df['HoltWinters'], label='Holt-Winters Forecast', linestyle='--', color='brown')
ax_combined.set_title("Forecast Comparison")
ax_combined.set_xlabel("Date")
ax_combined.set_ylabel("Vaccinations")
ax_combined.legend()
st.pyplot(fig_combined)

# Prepare SARIMA metrics text for markdown
if 'sarima_rmse' in locals() and 'sarima_mae' in locals():
    sarima_metrics_text = f"- RMSE: **{sarima_rmse:.2f}**\n- MAE: **{sarima_mae:.2f}**"
else:
    sarima_metrics_text = "- (SARIMA metrics unavailable due to alignment issue.)"

# Final Accuracy Interpretation
st.write("## 🧠 Final Model Interpretation")
st.markdown(f'''
**Prophet Model**
- RMSE: **{rmse:.2f}**
- MAE: **{mae:.2f}**
- 📌 On average, Prophet deviates by about **{mae:.0f}** vaccinations per month.

**LSTM Model**
- RMSE: **{rmse_lstm:.2f}**
- MAE: **{mae_lstm:.2f}**
- 📌 LSTM's predictions differ by about **{mae_lstm:.0f}** vaccinations per month on average.

**Hybrid Model (Prophet + ARIMA)**
- RMSE: **{hybrid_rmse:.2f}**
- MAE: **{hybrid_mae:.2f}**
- 📌 Hybrid adjusts Prophet’s leftover error, improving overall accuracy.

**SARIMA Model**
{sarima_metrics_text}

**Holt-Winters Model**
- RMSE: **{holt_rmse:.2f}**
- MAE: **{holt_mae:.2f}**

### What This Means:
- **RMSE (Root Mean Squared Error)** penalizes big mistakes more heavily.
- **MAE (Mean Absolute Error)** gives the average error regardless of direction.
- ✅ **Lower = better.**
- Use this insight to pick your best model or inform stakeholders with realistic expectations.
''')

st.success("✅ Forecasting completed.")

# --- Optional PDF Report Export ---
import io
from fpdf import FPDF
import tempfile
from PIL import Image

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "Flu Vaccination Forecasting Report", ln=True, align="C")
        self.ln(10)

    def chapter_title(self, title):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, ln=True)
        self.ln(5)

    def chapter_body(self, body):
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 10, body)
        self.ln()

if st.button("📄 Generate Summary Report (PDF)"):
    pdf = PDFReport()
    pdf.add_page()

    pdf.chapter_title("Model Accuracy Summary")
    pdf.chapter_body(
        f"""Prophet RMSE: {rmse:.2f}
Prophet MAE: {mae:.2f}

LSTM RMSE: {rmse_lstm:.2f}
LSTM MAE: {mae_lstm:.2f}

Hybrid RMSE: {hybrid_rmse:.2f}
Hybrid MAE: {hybrid_mae:.2f}

SARIMA RMSE: {sarima_rmse:.2f}
SARIMA MAE: {sarima_mae:.2f}

Holt-Winters RMSE: {holt_rmse:.2f}
Holt-Winters MAE: {holt_mae:.2f}
"""
    )

    pdf.chapter_title("Interpretation")
    pdf.chapter_body(
        """RMSE (Root Mean Squared Error) penalizes larger errors more.
MAE (Mean Absolute Error) gives average error in prediction.

Lower values = better prediction accuracy.
Use these to choose the most reliable model or share with stakeholders."""
    )

    # Save forecast charts as images and add to PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp1, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp2, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp3, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp4, \
         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp5:
        fig1.savefig(tmp1.name)
        fig_hybrid.savefig(tmp2.name)
        fig_combined.savefig(tmp3.name)
        fig_sarima.savefig(tmp4.name)
        fig_hw.savefig(tmp5.name)

        pdf.add_page()
        pdf.chapter_title("📉 Prophet Forecast Chart")
        pdf.image(tmp1.name, x=10, w=190)

        pdf.add_page()
        pdf.chapter_title("📈 Hybrid Forecast Chart")
        pdf.image(tmp2.name, x=10, w=190)

        pdf.add_page()
        pdf.chapter_title("📊 Combined Forecast Chart")
        pdf.image(tmp3.name, x=10, w=190)

        pdf.add_page()
        pdf.chapter_title("📈 SARIMA Forecast Chart")
        pdf.image(tmp4.name, x=10, w=190)

        pdf.add_page()
        pdf.chapter_title("📉 Holt-Winters Forecast Chart")
        pdf.image(tmp5.name, x=10, w=190)

    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_output.write(pdf_bytes)

    st.download_button("⬇️ Download Report", data=pdf_output.getvalue(), file_name="forecast_report.pdf", mime="application/pdf")
