from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import io
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import Library Algoritma
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

app = FastAPI(title="Forecasting Experiment API")

# Mencegah error CORS agar Frontend (React) bisa memanggil API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/experiment")
async def run_experiment(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    target_col: str = Form(...),
    date_col: str = Form(...),
    feature_cols: str = Form("") # Khusus MLR (pisahkan dengan koma)
):
    try:
        # 1. Membaca CSV dari Upload Frontend
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validasi Kolom
        if date_col not in df.columns or target_col not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Kolom '{date_col}' atau '{target_col}' tidak ditemukan di CSV."})
            
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        
        # Mengambil 80% Train, 20% Test
        training_len = int(np.ceil(len(df) * 0.8))
        train_df = df.iloc[:training_len].copy()
        test_df = df.iloc[training_len:].copy()
        
        train_dates = train_df[date_col].dt.strftime('%Y-%m-%d').tolist()
        test_dates = test_df[date_col].dt.strftime('%Y-%m-%d').tolist()
        
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        
        predictions = []
        
        # ==========================================
        # LOGIKA PERCABANGAN 5 ALGORITMA
        # ==========================================
        
        if model_type == "ARIMA":
            # ARIMA(0,1,1)
            model = ARIMA(y_train, order=(0, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(y_test))
            predictions = forecast.tolist()
            
        elif model_type == "Holt-Winters":
            # Additive HW
            model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=21)
            model_fit = model.fit()
            forecast = model_fit.forecast(len(y_test))
            predictions = forecast.tolist()
            
        elif model_type == "Prophet":
            # Facebook Prophet
            prophet_train = train_df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
            prophet_test = test_df[[date_col]].rename(columns={date_col: 'ds'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            model.fit(prophet_train)
            forecast = model.predict(prophet_test)
            predictions = forecast['yhat'].tolist()
            
        elif model_type == "MLR":
            # Multiple Linear Regression (Multivariate Time-Shifting)
            if not feature_cols:
                return JSONResponse(status_code=400, content={"error": "Pilih minimal 1 fitur pendukung untuk MLR."})
                
            fitur_list = [f.strip() for f in feature_cols.split(',')]
            df_mlr = df.copy()
            df_mlr['Target_Y'] = df_mlr[target_col]
            
            # Shift fitur H-1
            for f in fitur_list:
                df_mlr[f'{f}_X'] = df_mlr[f].shift(1)
            
            df_mlr.dropna(inplace=True)
            
            # Re-split karena dropna
            train_len_mlr = int(np.ceil(len(df_mlr) * 0.8))
            X_cols = [f'{f}_X' for f in fitur_list]
            
            X_train_mlr = df_mlr.iloc[:train_len_mlr][X_cols].values
            Y_train_mlr = df_mlr.iloc[:train_len_mlr]['Target_Y'].values
            X_test_mlr = df_mlr.iloc[train_len_mlr:][X_cols].values
            Y_test_mlr = df_mlr.iloc[train_len_mlr:]['Target_Y'].values
            
            # Update Test Dates for chart (karena berkurang 1 baris di awal)
            test_dates = df_mlr.iloc[train_len_mlr:][date_col].dt.strftime('%Y-%m-%d').tolist()
            y_test = Y_test_mlr
            
            model = LinearRegression()
            model.fit(X_train_mlr, Y_train_mlr)
            preds = model.predict(X_test_mlr)
            predictions = preds.tolist()
            
        elif model_type == "LSTM":
            # Deep Learning (Scaled)
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(df[[target_col]].values)
            
            train_data = scaled_data[:training_len]
            
            # Buat struktur X_train, y_train (Window 60 hari)
            x_train_lstm, y_train_lstm = [], []
            for i in range(60, len(train_data)):
                x_train_lstm.append(train_data[i-60:i, 0])
                y_train_lstm.append(train_data[i, 0])
                
            x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)
            x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))
            
            # Bangun Model Cepat (Hanya 5 Epoch agar API tidak Timeout kelamaan)
            model = Sequential([
                Input(shape=(x_train_lstm.shape[1], 1)),
                LSTM(50, return_sequences=True),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train_lstm, y_train_lstm, batch_size=32, epochs=5, verbose=0)
            
            # Buat Test Data
            test_data_scaled = scaled_data[training_len - 60:]
            x_test_lstm = []
            for i in range(60, len(test_data_scaled)):
                x_test_lstm.append(test_data_scaled[i-60:i, 0])
                
            x_test_lstm = np.array(x_test_lstm)
            x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))
            
            preds = model.predict(x_test_lstm)
            preds = scaler.inverse_transform(preds)
            predictions = preds.flatten().tolist()
            
        else:
            return JSONResponse(status_code=400, content={"error": "Metode tidak dikenali."})
            
        # ==========================================
        # KALKULASI METRIK
        # ==========================================
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        mape = float(mean_absolute_percentage_error(y_test, predictions) * 100)
        
        # Kirim Balasan ke Web Frontend
        return {
            "status": "success",
            "model_type": model_type,
            "metrics": {
                "RMSE": round(rmse, 2),
                "MAE": round(mae, 2),
                "MAPE": round(mape, 2)
            },
            "chart_data": {
                "dates": test_dates,
                "actual": y_test.tolist(),
                "predicted": predictions
            }
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Menjalankan server (Untuk keperluan testing manual)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
