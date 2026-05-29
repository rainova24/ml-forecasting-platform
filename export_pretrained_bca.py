import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# 1. Pastikan folder pretrained_models ada
base_dir = os.path.join("backend", "pretrained_models")
os.makedirs(base_dir, exist_ok=True)

# 2. Baca Data
df = pd.read_csv("dataset_bca_siap_uji.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
df = df.sort_values('Date').reset_index(drop=True)

print("Mengekspor Model MLR (Regresi Linear Berganda)...")
# ==========================================
# EXPORT MODEL 1: MLR (Multiple Linear Regression)
# ==========================================
mlr_dir = os.path.join(base_dir, "bca-mlr")
os.makedirs(mlr_dir, exist_ok=True)

# Fitur MLR: Open, High, Low -> Target: Close
df_mlr = df.copy()
df_mlr['Open_X'] = df_mlr['Open'].shift(1)
df_mlr['High_X'] = df_mlr['High'].shift(1)
df_mlr['Low_X'] = df_mlr['Low'].shift(1)
df_mlr.dropna(inplace=True)

X_mlr = df_mlr[['Open_X', 'High_X', 'Low_X']].values
y_mlr = df_mlr['Close'].values

model_mlr = LinearRegression()
model_mlr.fit(X_mlr, y_mlr)

# Simpan MLR model
joblib.dump(model_mlr, os.path.join(mlr_dir, "model.pkl"))

# Buat Metadata MLR
mlr_metadata = {
    "description": "Model Regresi Linear Berganda memprediksi harga Close BCA berdasarkan harga Open, High, dan Low dari hari sebelumnya.",
    "model_type": "MLR",
    "input_schema": {
        "type": "array",
        "description": "Array berisi 3 angka: [Open_kemarin, High_kemarin, Low_kemarin]",
        "example": [[9200, 9300, 9150]]
    },
    "output_schema": {
        "type": "array",
        "description": "Prediksi harga Close"
    }
}
with open(os.path.join(mlr_dir, "metadata.json"), "w") as f:
    json.dump(mlr_metadata, f, indent=4)


print("\nMengekspor Model LSTM (Deep Learning)...")
# ==========================================
# EXPORT MODEL 2: LSTM
# ==========================================
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input
    
    lstm_dir = os.path.join(base_dir, "bca-lstm")
    os.makedirs(lstm_dir, exist_ok=True)

    # Skala Data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']].values)

    # Window Size = 60
    window_size = 60
    x_lstm, y_lstm = [], []
    for i in range(window_size, len(scaled_data)):
        x_lstm.append(scaled_data[i-window_size:i, 0])
        y_lstm.append(scaled_data[i, 0])

    x_lstm, y_lstm = np.array(x_lstm), np.array(y_lstm)
    x_lstm = np.reshape(x_lstm, (x_lstm.shape[0], x_lstm.shape[1], 1))

    # Bangun & Latih Model
    model_lstm = Sequential([
        Input(shape=(x_lstm.shape[1], 1)),
        LSTM(50, return_sequences=True),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(x_lstm, y_lstm, batch_size=32, epochs=5, verbose=0) 

    # Simpan LSTM model (.keras) & Scaler (.pkl)
    model_lstm.save(os.path.join(lstm_dir, "model.keras"))
    joblib.dump(scaler, os.path.join(lstm_dir, "scaler.pkl"))

    # Buat Metadata LSTM
    lstm_metadata = {
        "description": "Deep Learning LSTM model memprediksi harga Close BCA berdasarkan sekuens 60 hari harga Close ke belakang.",
        "model_type": "LSTM",
        "input_schema": {
            "type": "array",
            "description": "List 1D yang berisi 60 harga Close berturut-turut",
            "example": {"past_60_days": [9000, 9100, 9050, "...", 9250]}
        },
        "output_schema": {
            "type": "array",
            "description": "Prediksi harga Close untuk hari ke-61"
        }
    }
    with open(os.path.join(lstm_dir, "metadata.json"), "w") as f:
        json.dump(lstm_metadata, f, indent=4)
    print("✓ LSTM Berhasil Diekspor")

except ImportError as e:
    print(f"⚠️ Melewati ekspor LSTM karena CPU tidak mendukung TensorFlow AVX: {e}")

# ==========================================
# EXPORT MODEL 3: ARIMA
# ==========================================
print("Mengekspor Model ARIMA...")
from statsmodels.tsa.arima.model import ARIMA
arima_dir = os.path.join(base_dir, "bca-arima")
os.makedirs(arima_dir, exist_ok=True)

# Train ARIMA (p=5, d=1, q=0) contoh sederhana
y_ts = df['Close'].values
model_arima = ARIMA(y_ts, order=(5,1,0))
fitted_arima = model_arima.fit()
joblib.dump(fitted_arima, os.path.join(arima_dir, "model.pkl"))

arima_metadata = {
    "description": "Model ARIMA (AutoRegressive Integrated Moving Average) untuk prediksi runtun waktu saham BCA.",
    "model_type": "ARIMA",
    "input_schema": {
        "type": "object",
        "description": "Jumlah langkah ke depan (steps) yang ingin diprediksi.",
        "example": {"steps": 1}
    }
}
with open(os.path.join(arima_dir, "metadata.json"), "w") as f:
    json.dump(arima_metadata, f, indent=4)

# ==========================================
# EXPORT MODEL 4: Holt-Winters (Exponential Smoothing)
# ==========================================
print("Mengekspor Model Holt-Winters...")
from statsmodels.tsa.holtwinters import ExponentialSmoothing
hw_dir = os.path.join(base_dir, "bca-holtwinters")
os.makedirs(hw_dir, exist_ok=True)

model_hw = ExponentialSmoothing(y_ts, trend='add', seasonal=None)
fitted_hw = model_hw.fit()
joblib.dump(fitted_hw, os.path.join(hw_dir, "model.pkl"))

hw_metadata = {
    "description": "Model Holt-Winters (Exponential Smoothing) untuk menangkap tren harga saham BCA.",
    "model_type": "Holt-Winters",
    "input_schema": {
        "type": "object",
        "description": "Jumlah langkah ke depan (steps) yang ingin diprediksi.",
        "example": {"steps": 1}
    }
}
with open(os.path.join(hw_dir, "metadata.json"), "w") as f:
    json.dump(hw_metadata, f, indent=4)

# ==========================================
# EXPORT MODEL 5: Prophet
# ==========================================
print("Mengekspor Model Facebook Prophet...")
from prophet import Prophet
prophet_dir = os.path.join(base_dir, "bca-prophet")
os.makedirs(prophet_dir, exist_ok=True)

df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
model_prophet = Prophet()
model_prophet.fit(df_prophet)
joblib.dump(model_prophet, os.path.join(prophet_dir, "model.pkl"))

prophet_metadata = {
    "description": "Model Facebook Prophet yang handal mendeteksi musim (seasonality) dan tren (trend) pada harga BCA.",
    "model_type": "Prophet",
    "input_schema": {
        "type": "object",
        "description": "Tanggal target yang ingin diprediksi",
        "example": {"ds": "2026-06-01"}
    }
}
with open(os.path.join(prophet_dir, "metadata.json"), "w") as f:
    json.dump(prophet_metadata, f, indent=4)

print("\nSELESAI! Seluruh 5 Model BCA (MLR, LSTM, ARIMA, Holt-Winters, Prophet) telah diekspor!")
print("Silakan cek folder backend/pretrained_models/ dan restart uvicorn!")
