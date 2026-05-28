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
    feature_cols: str = Form(""),
    time_resample: str = Form("none"), # Parameter baru: none, daily, weekly, monthly
    missing_values: str = Form("drop") # Parameter baru: drop, mean
):
    try:
        # 1. Membaca CSV dari Upload Frontend
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validasi Kolom
        if date_col not in df.columns or target_col not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Kolom '{date_col}' atau '{target_col}' tidak ditemukan di CSV."})
            
        # ==========================================
        # 2. PREPROCESSING DINAMIS 
        # ==========================================
        
        # A. Auto-Translate Bulan Indonesia ke Inggris untuk Pandas Datetime
        bulan_indo = {
            'Januari': 'January', 'Februari': 'February', 'Maret': 'March',
            'Mei': 'May', 'Juni': 'June', 'Juli': 'July', 'Agustus': 'August',
            'Oktober': 'October', 'Nopember': 'November', 'Desember': 'December'
        }
        
        if df[date_col].dtype == 'object':
            # Ignore case secara sederhana dengan replace per kata
            for id_month, en_month in bulan_indo.items():
                df[date_col] = df[date_col].str.replace(id_month, en_month, regex=False)
                df[date_col] = df[date_col].str.replace(id_month.lower(), en_month, regex=False)
                
        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        
        # Menjadikan Tanggal sebagai index untuk memudahkan proses Resampling
        df.set_index(date_col, inplace=True)
        
        # B. Penanganan Missing Values Awal (Kolom Angka Saja)
        num_cols = df.select_dtypes(include=[np.number]).columns
        if missing_values == "mean":
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        else:
            df.dropna(inplace=True)
            
        # C. Time Resampling Otomatis (Harian, Mingguan, Bulanan)
        if time_resample == "daily":
            df = df.resample('D').mean()
        elif time_resample == "weekly":
            df = df.resample('W').mean()
        elif time_resample == "monthly":
            df = df.resample('ME').mean() # Pandas 'ME' = Month End
            
        # D. Penanganan Missing Values Kedua (Efek Samping Resampling yang mungkin menghasilkan hari libur kosong)
        if time_resample != "none":
            if missing_values == "mean":
                df = df.fillna(df.mean())
            else:
                df.dropna(inplace=True)
                
        # Kembalikan kolom Tanggal dari Index agar bisa dipakai algoritma
        df.reset_index(inplace=True)
        
        # ==========================================
        # 3. SPLITTING DATA (80% / 20%)
        # ==========================================
        
        training_len = int(np.ceil(len(df) * 0.8))
        train_df = df.iloc[:training_len].copy()
        test_df = df.iloc[training_len:].copy()
        
        train_dates = train_df[date_col].dt.strftime('%Y-%m-%d').tolist()
        test_dates = test_df[date_col].dt.strftime('%Y-%m-%d').tolist()
        
        y_train = train_df[target_col].values
        y_test = test_df[target_col].values
        
        predictions = []
        
        # ==========================================
        # 4. LOGIKA PERCABANGAN 5 ALGORITMA
        # ==========================================
        
        if model_type == "ARIMA":
            # ARIMA(0,1,1)
            model = ARIMA(y_train, order=(0, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(y_test))
            predictions = forecast.tolist()
            
        elif model_type == "Holt-Winters":
            # Mematikan seasonal jika datanya terlalu sedikit akibat resampling bulanan
            seasonal_opt = 'add' if len(y_train) > 42 else None
            period_opt = 21 if seasonal_opt else None
            
            model = ExponentialSmoothing(y_train, trend='add', seasonal=seasonal_opt, seasonal_periods=period_opt)
            model_fit = model.fit()
            forecast = model_fit.forecast(len(y_test))
            predictions = forecast.tolist()
            
        elif model_type == "Prophet":
            prophet_train = train_df[[date_col, target_col]].rename(columns={date_col: 'ds', target_col: 'y'})
            prophet_test = test_df[[date_col]].rename(columns={date_col: 'ds'})
            model = Prophet(daily_seasonality=False, yearly_seasonality=True)
            model.fit(prophet_train)
            forecast = model.predict(prophet_test)
            predictions = forecast['yhat'].tolist()
            
        elif model_type == "MLR":
            if not feature_cols:
                return JSONResponse(status_code=400, content={"error": "Pilih minimal 1 fitur pendukung untuk MLR."})
                
            fitur_list = [f.strip() for f in feature_cols.split(',')]
            df_mlr = df.copy()
            df_mlr['Target_Y'] = df_mlr[target_col]
            
            # Shift fitur H-1
            for f in fitur_list:
                df_mlr[f'{f}_X'] = df_mlr[f].shift(1)
            
            df_mlr.dropna(inplace=True)
            
            train_len_mlr = int(np.ceil(len(df_mlr) * 0.8))
            X_cols = [f'{f}_X' for f in fitur_list]
            
            X_train_mlr = df_mlr.iloc[:train_len_mlr][X_cols].values
            Y_train_mlr = df_mlr.iloc[:train_len_mlr]['Target_Y'].values
            X_test_mlr = df_mlr.iloc[train_len_mlr:][X_cols].values
            Y_test_mlr = df_mlr.iloc[train_len_mlr:]['Target_Y'].values
            
            test_dates = df_mlr.iloc[train_len_mlr:][date_col].dt.strftime('%Y-%m-%d').tolist()
            y_test = Y_test_mlr
            
            model = LinearRegression()
            model.fit(X_train_mlr, Y_train_mlr)
            preds = model.predict(X_test_mlr)
            predictions = preds.tolist()
            
        elif model_type == "LSTM":
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(df[[target_col]].values)
            
            train_data = scaled_data[:training_len]
            
            # Menyesuaikan Window Size jika data terlalu sedikit akibat resampling
            window_size = 60 if len(train_data) > 120 else 5
            
            x_train_lstm, y_train_lstm = [], []
            for i in range(window_size, len(train_data)):
                x_train_lstm.append(train_data[i-window_size:i, 0])
                y_train_lstm.append(train_data[i, 0])
                
            if len(x_train_lstm) == 0:
                return JSONResponse(status_code=400, content={"error": "Data terlalu sedikit untuk LSTM setelah Resampling."})

            x_train_lstm, y_train_lstm = np.array(x_train_lstm), np.array(y_train_lstm)
            x_train_lstm = np.reshape(x_train_lstm, (x_train_lstm.shape[0], x_train_lstm.shape[1], 1))
            
            model = Sequential([
                Input(shape=(x_train_lstm.shape[1], 1)),
                LSTM(50, return_sequences=True),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train_lstm, y_train_lstm, batch_size=32, epochs=5, verbose=0)
            
            test_data_scaled = scaled_data[training_len - window_size:]
            x_test_lstm = []
            for i in range(window_size, len(test_data_scaled)):
                x_test_lstm.append(test_data_scaled[i-window_size:i, 0])
                
            x_test_lstm = np.array(x_test_lstm)
            x_test_lstm = np.reshape(x_test_lstm, (x_test_lstm.shape[0], x_test_lstm.shape[1], 1))
            
            preds = model.predict(x_test_lstm)
            preds = scaler.inverse_transform(preds)
            predictions = preds.flatten().tolist()
            
        else:
            return JSONResponse(status_code=400, content={"error": "Metode tidak dikenali."})
            
        # ==========================================
        # 5. KALKULASI METRIK
        # ==========================================
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        mape = float(mean_absolute_percentage_error(y_test, predictions) * 100)
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
