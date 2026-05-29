from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import io
import json
import os
import pickle
import glob
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional

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
    missing_values: str = Form("drop"), # Parameter baru: drop, mean
    one_hot_encode_cols: str = Form(""),
    window_size: int = Form(0),
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

        def create_lagged_features(df, columns, window_size):
            for i in range(1, window_size + 1):
                df[f"min_lag_{i}"] = df[target_col].shift(i)
            
            df_clean = df.dropna(subset=[f'min_lag_{i}' for i in range(1, window_size + 1)]).reset_index(drop=True)
            return df_clean
        
        def encode_categorical_features(df, columns):
            one_hot_encode_features = []
            if columns:
                for col in columns.split(","):
                    col = col.strip() # Hapus spasi ekstra
                    if col and col in df.columns:
                        categorical_columns = pd.get_dummies(df[col], prefix=col, dtype=int)
                        df = pd.concat([df, categorical_columns], axis=1)
                        one_hot_encode_features.extend(categorical_columns.columns.tolist())
            return df, one_hot_encode_features
            
        
                
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
        
        elif model_type == "Prophet-Add-Regressor":
            # Add external regressors
            df, one_hot_encode_features = encode_categorical_features(df, one_hot_encode_cols)
            
            print("[PROPHET ADD REGRESSOR] Category Features: ", one_hot_encode_features)
            if not one_hot_encode_features:
                return JSONResponse(status_code=400, content={"error": "Pilih minimal 1 fitur one hot encoding untuk Prophet-Add Regressor."})

            # Ensure all dummy columns exist in both train and test sets
            for col in one_hot_encode_features:
                if col not in train_df.columns:
                    train_df[col] = 0
                if col not in test_df.columns:
                    test_df[col] = 0

            # Prepare dataframes with dummy regressor columns
            prophet_train = train_df[[date_col, target_col] + one_hot_encode_features].rename(columns={date_col: 'ds', target_col: 'y'})
            prophet_test = test_df[[date_col] + one_hot_encode_features].rename(columns={date_col: 'ds'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05)
            
            for col in one_hot_encode_features:
                model.add_regressor(col)
                
            model.fit(prophet_train)
            forecast = model.predict(prophet_test)
            predictions = forecast['yhat'].tolist()
            
        elif model_type == "MLR":
            # Require at least one source of features
            if not feature_cols and not one_hot_encode_cols:
                return JSONResponse(status_code=400, content={"error": "Pilih minimal 1 fitur pendukung (numeric atau one-hot) untuk MLR."})

            if window_size > 0:
                df = create_lagged_features(df, [target_col], window_size)

            if one_hot_encode_cols:
                df, one_hot_encode_features = encode_categorical_features(df, one_hot_encode_cols)
            else:
                one_hot_encode_features = []

            if feature_cols:
                fitur_list = [f.strip() for f in feature_cols.split(',')]
            else:
                fitur_list = []

            df_mlr = df.copy()
            df_mlr["Target_Y"] = df_mlr[target_col]

            # Create lagged versions for numeric features
            for f in fitur_list:
                if f not in df_mlr.columns:
                    continue
                if not np.issubdtype(df_mlr[f].dtype, np.number):
                    continue
                df_mlr[f"{f}_X"] = df_mlr[f].shift(1)

            # Drop rows with NaNs introduced by shifting
            df_mlr.dropna(inplace=True)

            # Build feature matrix: lagged numeric features + one-hot dummy columns
            X_cols = [f"{f}_X" for f in fitur_list] + one_hot_encode_features

            # Split into train/test
            train_len_mlr = int(np.ceil(len(df_mlr) * 0.8))
            X_train_mlr = df_mlr.iloc[:train_len_mlr][X_cols].values
            Y_train_mlr = df_mlr.iloc[:train_len_mlr]["Target_Y"].values
            X_test_mlr = df_mlr.iloc[train_len_mlr:][X_cols].values
            Y_test_mlr = df_mlr.iloc[train_len_mlr:]["Target_Y"].values

            test_dates = df_mlr.iloc[train_len_mlr:][date_col].dt.strftime("%Y-%m-%d").tolist()
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

# ==========================================
# PRE-TRAINED MODEL ENDPOINTS
# ==========================================

# Directory where pre-trained model artifacts live
PRETRAINED_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models")


class PredictRequest(BaseModel):
    model_name: str
    input: Any  # Accepts any JSON structure


@app.get("/api/pretrained/models")
async def list_pretrained_models():
    """Return a list of available pre-trained model names stored on the server."""
    try:
        if not os.path.isdir(PRETRAINED_DIR):
            return {"models": []}

        # A model entry is a sub-directory (or metadata .json) inside PRETRAINED_DIR
        models = []
        for entry in sorted(os.listdir(PRETRAINED_DIR)):
            entry_path = os.path.join(PRETRAINED_DIR, entry)
            meta_path = os.path.join(entry_path, "metadata.json")
            if os.path.isdir(entry_path) and os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                models.append({
                    "name": entry,
                    "description": meta.get("description", ""),
                    "model_type": meta.get("model_type", "unknown"),
                    "input_schema": meta.get("input_schema", {}),
                    "output_schema": meta.get("output_schema", {}),
                })
        return {"models": models}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/pretrained/predict")
async def predict_pretrained(request: PredictRequest):
    """
    Run inference on a pre-trained model stored on the server.
    Accepts arbitrary JSON input and returns JSON output.

    Expected layout for PRETRAINED_DIR/<model_name>/:
        metadata.json   – model description, type, and schema hints
        model.*         – the actual artifact (.keras / .pkl / etc.)
        scaler.pkl      – (optional) MinMaxScaler or similar
    """
    try:
        model_dir = os.path.join(PRETRAINED_DIR, request.model_name)
        if not os.path.isdir(model_dir):
            return JSONResponse(
                status_code=404,
                content={"error": f"Model '{request.model_name}' not found on server."}
            )

        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return JSONResponse(
                status_code=500,
                content={"error": "metadata.json missing from model directory."}
            )

        with open(meta_path, "r") as f:
            meta = json.load(f)

        model_type = meta.get("model_type", "").upper()
        raw_input = request.input  # plain Python object (dict / list)

        # ---- Locate the model artifact ----
        keras_files = glob.glob(os.path.join(model_dir, "model.keras"))
        pkl_files   = glob.glob(os.path.join(model_dir, "model.pkl"))
        h5_files    = glob.glob(os.path.join(model_dir, "model.h5"))

        result_output: Any = None

        # ---- Keras / deep-learning models (LSTM, etc.) ----
        if keras_files or h5_files:
            artifact_path = (keras_files or h5_files)[0]
            model = load_model(artifact_path)

            if model_type == "DUAL-INPUT LSTM":
                # Ensure input format
                if not isinstance(raw_input, dict) or "past_14_days_minutes" not in raw_input or "static_features" not in raw_input:
                    return JSONResponse(status_code=400, content={"error": "Input must be a JSON object containing 'past_14_days_minutes' and 'static_features'."})
                
                scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
                scaler_exog_path = os.path.join(model_dir, "scaler_exog.pkl")
                
                if not os.path.exists(scaler_y_path) or not os.path.exists(scaler_exog_path):
                    return JSONResponse(status_code=500, content={"error": "scaler_y.pkl and scaler_exog.pkl are required for Dual-Input LSTM."})
                
                try:
                    scaler_y = joblib.load(scaler_y_path)
                except Exception:
                    with open(scaler_y_path, "rb") as f:
                        scaler_y = pickle.load(f)
                        
                try:
                    scaler_exog = joblib.load(scaler_exog_path)
                except Exception:
                    with open(scaler_exog_path, "rb") as f:
                        scaler_exog = pickle.load(f)
                
                # Process past_14_days_minutes -> scale -> shape (1, 14, 1)
                past_14 = np.array(raw_input["past_14_days_minutes"], dtype=np.float32).reshape(-1, 1)
                past_14_scaled = scaler_y.transform(past_14)
                lstm_input = past_14_scaled.reshape(1, past_14.shape[0], 1)
                
                # Process static_features -> scale -> shape (1, num_features)
                static_dict = raw_input["static_features"]
                static_arr = np.array(list(static_dict.values()), dtype=np.float32).reshape(1, -1)
                static_arr_scaled = scaler_exog.transform(static_arr)
                
                # Predict dual inputs
                raw_pred = model.predict([lstm_input, static_arr_scaled])
                
                # Inverse scale
                raw_pred_inv = scaler_y.inverse_transform(raw_pred)
                result_output = raw_pred_inv.tolist()

            else:
                # Optionally load scaler
                scaler_path = os.path.join(model_dir, "scaler.pkl")
                scaler = None
                if os.path.exists(scaler_path):
                    try:
                        scaler = joblib.load(scaler_path)
                    except Exception:
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)

                # raw_input is expected to be a list-of-lists (2-D) or flat list
                arr = np.array(raw_input, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                # Scale if scaler provided
                if scaler is not None:
                    arr = scaler.transform(arr)

                # Reshape for LSTM: (samples, timesteps, features)
                if model_type == "LSTM":
                    if arr.ndim == 2:
                        arr = arr.reshape(arr.shape[0], arr.shape[1], 1)

                raw_pred = model.predict(arr)

                # Inverse-scale if scaler provided
                if scaler is not None:
                    raw_pred = scaler.inverse_transform(raw_pred)

                result_output = raw_pred.tolist()

        # ---- Sklearn / Prophet / pickle models ----
        elif pkl_files:
            try:
                model = joblib.load(pkl_files[0])
            except Exception:
                with open(pkl_files[0], "rb") as f:
                    model = pickle.load(f)

            if model_type == "PROPHET":
                # Prophet expects a DataFrame
                if isinstance(raw_input, dict):
                    df_input = pd.DataFrame([raw_input])
                elif isinstance(raw_input, list):
                    df_input = pd.DataFrame(raw_input)
                else:
                    return JSONResponse(status_code=400, content={"error": "Invalid input format for Prophet."})
                
                # Convert date column if present
                if 'ds' in df_input.columns:
                    df_input['ds'] = pd.to_datetime(df_input['ds'])

                forecast = model.predict(df_input)
                result_output = forecast['yhat'].tolist()
            else:
                arr = np.array(raw_input, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

                raw_pred = model.predict(arr)
                result_output = raw_pred.tolist()

        else:
            return JSONResponse(
                status_code=500,
                content={"error": "No model artifact (model.keras / model.h5 / model.pkl) found in model directory."}
            )

        return {
            "status": "success",
            "model_name": request.model_name,
            "model_type": meta.get("model_type", "unknown"),
            "description": meta.get("description", ""),
            "output": result_output,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/pretrained/evaluate")
async def evaluate_pretrained_with_dataset(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    target_col: str = Form("Close"),
    date_col: str = Form("Date"),
    test_size: float = Form(0.2),
):
    """
    Evaluate a pre-trained model using the last test_size fraction of a CSV dataset.
    Returns MAE, RMSE, R², MAPE metrics plus actual vs predicted chart data.
    """
    try:
        # 1. Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if date_col not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Kolom tanggal '{date_col}' tidak ditemukan dalam CSV."})
        if target_col not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Kolom target '{target_col}' tidak ditemukan dalam CSV."})

        # 2. Date preprocessing (same logic as /api/experiment)
        bulan_indo = {
            'Januari': 'January', 'Februari': 'February', 'Maret': 'March',
            'Mei': 'May', 'Juni': 'June', 'Juli': 'July', 'Agustus': 'August',
            'Oktober': 'October', 'Nopember': 'November', 'Desember': 'December'
        }
        if df[date_col].dtype == 'object':
            for id_month, en_month in bulan_indo.items():
                df[date_col] = df[date_col].str.replace(id_month, en_month, regex=False)
                df[date_col] = df[date_col].str.replace(id_month.lower(), en_month, regex=False)

        df[date_col] = pd.to_datetime(df[date_col])
        df.sort_values(date_col, inplace=True)
        df.dropna(subset=[target_col], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # 3. Load model metadata
        model_dir = os.path.join(PRETRAINED_DIR, model_name)
        if not os.path.isdir(model_dir):
            return JSONResponse(status_code=404, content={"error": f"Model '{model_name}' tidak ditemukan di server."})

        meta_path = os.path.join(model_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return JSONResponse(status_code=500, content={"error": "metadata.json tidak ditemukan di direktori model."})

        with open(meta_path, "r") as f:
            meta = json.load(f)
        model_type = meta.get("model_type", "").upper()

        # 4. Check for custom preprocess.py
        preprocess_path = os.path.join(model_dir, "preprocess.py")
        if os.path.exists(preprocess_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("preprocess", preprocess_path)
            preprocess_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(preprocess_module)
            
            # The custom preprocess script should return (X_test, y_test, test_dates)
            X_test, y_test, test_dates = preprocess_module.preprocess_evaluate(df, target_col, date_col, test_size)
            test_size_clamped = test_size  # Handled by preprocess.py
            
            # Locate and load the model
            keras_files = glob.glob(os.path.join(model_dir, "model.keras"))
            h5_files    = glob.glob(os.path.join(model_dir, "model.h5"))
            pkl_files   = glob.glob(os.path.join(model_dir, "model.pkl"))
            
            if keras_files or h5_files:
                artifact_path = (keras_files or h5_files)[0]
                loaded_model  = load_model(artifact_path)
                
                # Check for scaler to inverse transform predictions
                scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
                scaler_path   = os.path.join(model_dir, "scaler.pkl")
                
                preds_scaled = loaded_model.predict(X_test, verbose=0)
                
                if os.path.exists(scaler_y_path):
                    try:
                        scaler_y = joblib.load(scaler_y_path)
                    except Exception:
                        with open(scaler_y_path, "rb") as f:
                            scaler_y = pickle.load(f)
                    preds_inv = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))
                elif os.path.exists(scaler_path):
                    try:
                        scaler = joblib.load(scaler_path)
                    except Exception:
                        with open(scaler_path, "rb") as f:
                            scaler = pickle.load(f)
                    preds_inv = scaler.inverse_transform(preds_scaled.reshape(-1, 1))
                else:
                    preds_inv = preds_scaled
                    
                predictions = preds_inv.flatten().tolist()
                
            elif pkl_files:
                try:
                    loaded_model = joblib.load(pkl_files[0])
                except Exception:
                    with open(pkl_files[0], "rb") as f:
                        loaded_model = pickle.load(f)
                        
                if model_type == "PROPHET":
                    forecast = loaded_model.predict(X_test)
                    predictions = forecast['yhat'].tolist()
                elif model_type == "PROPHET-REGRESSOR":
                    forecast = loaded_model.predict(X_test)
                    predictions = forecast['yhat'].tolist()
                else:
                    predictions = loaded_model.predict(X_test).tolist()
            else:
                return JSONResponse(status_code=500, content={"error": "Tidak ada artefak model ditemukan."})
                
        else:
            # --- FALLBACK TO GENERIC LOGIC IF NO PREPROCESS.PY EXISTS ---
            # 4. Train/Test split — use the last test_size fraction as the test set
            n = len(df)
            test_size_clamped = max(0.05, min(0.5, test_size))
            test_start_idx = int(n * (1.0 - test_size_clamped))
    
            if test_start_idx < 1 or test_start_idx >= n:
                return JSONResponse(status_code=400, content={"error": "Ukuran data terlalu kecil atau test_size tidak valid."})
    
            test_df    = df.iloc[test_start_idx:].copy()
            test_dates = test_df[date_col].dt.strftime('%Y-%m-%d').tolist()
            y_test     = test_df[target_col].values.copy()
            predictions = []
    
            # 5. Locate model artifact
            keras_files = glob.glob(os.path.join(model_dir, "model.keras"))
            h5_files    = glob.glob(os.path.join(model_dir, "model.h5"))
            pkl_files   = glob.glob(os.path.join(model_dir, "model.pkl"))
    
            if keras_files or h5_files:
                artifact_path = (keras_files or h5_files)[0]
                loaded_model  = load_model(artifact_path)
    
                if model_type == "DUAL-INPUT LSTM":
                    # ── Dual-Input LSTM evaluation ──
                    scaler_y_path    = os.path.join(model_dir, "scaler_y.pkl")
                    scaler_exog_path = os.path.join(model_dir, "scaler_exog.pkl")
    
                    if not os.path.exists(scaler_y_path):
                        return JSONResponse(status_code=400, content={"error": "scaler_y.pkl diperlukan untuk evaluasi Dual-Input LSTM."})
    
                    try:
                        scaler_y = joblib.load(scaler_y_path)
                    except Exception:
                        with open(scaler_y_path, "rb") as f:
                            scaler_y = pickle.load(f)
    
                    # Determine window size from metadata example
                    window_size = 14
                    try:
                        example = meta.get("input_schema", {}).get("example", {})
                        seq_key = next((k for k in example if isinstance(example[k], list)), None)
                        if seq_key:
                            window_size = len(example[seq_key])
                    except Exception:
                        pass
    
                    # Get exog feature count from scaler
                    n_exog      = 1
                    scaler_exog = None
                    if os.path.exists(scaler_exog_path):
                        try:
                            scaler_exog = joblib.load(scaler_exog_path)
                            n_exog = scaler_exog.n_features_in_
                        except Exception:
                            pass
    
                    all_values = df[target_col].values.reshape(-1, 1).astype(np.float32)
                    all_scaled = scaler_y.transform(all_values)
    
                    # Build batched sequences (efficient: no per-sample model.predict)
                    sequences, valid_test_indices = [], []
                    for i in range(test_start_idx, n):
                        if i < window_size:
                            continue
                        sequences.append(all_scaled[i - window_size:i, 0])
                        valid_test_indices.append(i - test_start_idx)
    
                    if not sequences:
                        return JSONResponse(status_code=400, content={"error": "Data tidak cukup untuk membangun urutan input LSTM. Kurangi test size atau tambah lebih banyak data."})
    
                    x_seq    = np.array(sequences).reshape(len(sequences), window_size, 1)
                    x_static = np.zeros((len(sequences), n_exog), dtype=np.float32)
                    if scaler_exog is not None:
                        x_static = scaler_exog.transform(x_static)
    
                    preds_scaled = loaded_model.predict([x_seq, x_static], verbose=0)
                    preds_inv    = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1))
                    predictions  = preds_inv.flatten().tolist()
                    y_test       = y_test[valid_test_indices]
                    test_dates   = [test_dates[k] for k in valid_test_indices]
    
                else:
                    # ── Standard Keras / LSTM evaluation ──
                    scaler_path = os.path.join(model_dir, "scaler.pkl")
                    scaler = None
                    if os.path.exists(scaler_path):
                        try:
                            scaler = joblib.load(scaler_path)
                        except Exception:
                            with open(scaler_path, "rb") as f:
                                scaler = pickle.load(f)
    
                    all_values = df[target_col].values.reshape(-1, 1).astype(np.float32)
                    all_scaled = scaler.transform(all_values) if scaler is not None else all_values
    
                    window_size = 60 if test_start_idx > 120 else max(5, min(14, test_start_idx - 1))
    
                    sequences, valid_test_indices = [], []
                    for i in range(test_start_idx, n):
                        if i < window_size:
                            continue
                        sequences.append(all_scaled[i - window_size:i, 0])
                        valid_test_indices.append(i - test_start_idx)
    
                    if not sequences:
                        return JSONResponse(status_code=400, content={"error": "Data tidak cukup untuk membangun urutan input LSTM."})
    
                    x_arr        = np.array(sequences).reshape(len(sequences), window_size, 1)
                    preds_scaled = loaded_model.predict(x_arr, verbose=0)
                    preds_arr    = scaler.inverse_transform(preds_scaled) if scaler is not None else preds_scaled
                    predictions  = preds_arr.flatten().tolist()
                    y_test       = y_test[valid_test_indices]
                    test_dates   = [test_dates[k] for k in valid_test_indices]
    
            elif pkl_files:
                try:
                    loaded_model = joblib.load(pkl_files[0])
                except Exception:
                    with open(pkl_files[0], "rb") as f:
                        loaded_model = pickle.load(f)
    
                if model_type == "PROPHET":
                    prophet_test = test_df[[date_col]].rename(columns={date_col: 'ds'})
                    prophet_test['ds'] = pd.to_datetime(prophet_test['ds'])
                    forecast    = loaded_model.predict(prophet_test)
                    predictions = forecast['yhat'].tolist()
                else:
                    # Generic sklearn: use all numeric columns except target
                    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
                    if not numeric_cols:
                        return JSONResponse(status_code=400, content={"error": "Tidak ada kolom fitur numerik untuk evaluasi model sklearn."})
                    X_test      = test_df[numeric_cols].fillna(0).values
                    predictions = loaded_model.predict(X_test).tolist()
            else:
                return JSONResponse(status_code=500, content={"error": "Tidak ada artefak model ditemukan (model.keras / model.h5 / model.pkl)."})
        # --- END FALLBACK LOGIC ---

        # 6. Calculate metrics
        y_arr = np.array(y_test,      dtype=np.float64)
        p_arr = np.array(predictions, dtype=np.float64)

        if len(y_arr) == 0 or len(p_arr) == 0:
            return JSONResponse(status_code=500, content={"error": "Tidak ada data prediksi yang dihasilkan."})
        if len(y_arr) != len(p_arr):
            return JSONResponse(status_code=500, content={"error": f"Jumlah prediksi ({len(p_arr)}) tidak sesuai data aktual ({len(y_arr)})."})

        mae_val  = float(mean_absolute_error(y_arr, p_arr))
        rmse_val = float(np.sqrt(mean_squared_error(y_arr, p_arr)))
        mape_val = float(mean_absolute_percentage_error(y_arr, p_arr) * 100)
        ss_res   = np.sum((y_arr - p_arr) ** 2)
        ss_tot   = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2_val   = float(1.0 - ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {
            "status":          "success",
            "model_name":      model_name,
            "model_type":      meta.get("model_type", "unknown"),
            "test_size_pct":   round(test_size_clamped * 100, 1),
            "n_test_samples":  int(len(y_arr)),
            "metrics": {
                "MAE":  round(mae_val,  4),
                "RMSE": round(rmse_val, 4),
                "R2":   round(r2_val,   4),
                "MAPE": round(mape_val, 2),
            },
            "chart_data": {
                "dates":     test_dates,
                "actual":    y_arr.tolist(),
                "predicted": p_arr.tolist(),
            }
        }


    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
