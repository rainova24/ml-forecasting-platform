import pandas as pd
import numpy as np
import os
import joblib
import pickle

def preprocess_evaluate(df: pd.DataFrame, target_col: str, date_col: str, test_size: float):
    """
    Preprocess data for Dual-Input LSTM evaluation.
    Requires past_14_days sequences and static features (day_of_week + category dummies).
    Precisely aligned with the Corrected_LSTM.ipynb notebook pipeline.
    """
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    df = df.copy()
    
    # 1. Date preprocessing & Indonesian Month Translation (from .ipynb)
    if df[date_col].dtype == object:
        month_map = {
            'Januari': 'January', 'Februari': 'February', 'Maret': 'March', 'April': 'April',
            'Mei': 'May', 'Juni': 'June', 'Juli': 'July', 'Agustus': 'August',
            'September': 'September', 'Oktober': 'October', 'November': 'November', 'Desember': 'December'
        }
        df[date_col] = df[date_col].astype(str).replace(month_map, regex=True)
        
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    n = len(df)
    test_size_clamped = max(0.05, min(0.5, test_size))
    test_start_idx = int(n * (1.0 - test_size_clamped))
    
    if test_start_idx < 1:
        raise ValueError("Data terlalu sedikit untuk test size ini.")
        
    # 2. Extract and Align Static Features to match training structure exactly
    # Notebook order: ['day_of_week', 'category_Interrupted', 'category_Learning', 'category_Normal']
    static_cols = ['day_of_week', 'category_Interrupted', 'category_Learning', 'category_Normal']
    
    df['day_of_week'] = df[date_col].dt.dayofweek
    
    # Replicate pd.get_dummies(..., drop_first=True) columns robustly
    for col in ['category_Interrupted', 'category_Learning', 'category_Normal']:
        cat_name = col.split('_')[1]
        if 'category' in df.columns:
            df[col] = (df['category'] == cat_name).astype(int)
        else:
            df[col] = 0
    
    # 3. Load Scalers
    scaler_y_path = os.path.join(model_dir, "scaler_y.pkl")
    scaler_exog_path = os.path.join(model_dir, "scaler_exog.pkl")
    
    if not os.path.exists(scaler_y_path):
        raise FileNotFoundError("scaler_y.pkl tidak ditemukan untuk evaluasi Dual-Input LSTM.")
        
    try:
        scaler_y = joblib.load(scaler_y_path)
    except Exception:
        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)
            
    scaler_exog = None
    if os.path.exists(scaler_exog_path):
        try:
            scaler_exog = joblib.load(scaler_exog_path)
        except Exception:
            with open(scaler_exog_path, "rb") as f:
                scaler_exog = pickle.load(f)
                
    # 4. Scale Target Series (for LSTM sequence matrix)
    all_values = df[target_col].values.reshape(-1, 1).astype(np.float32)
    all_scaled = scaler_y.transform(all_values)
    
    window_size = 14
    
    # 5. Build batched sequences and static features for test set
    # Matching create_dual_input_dataset(test_df) behavior:
    # The lookback loop begins 'window_size' indices inside the test partition.
    sequences = []
    static_feats = []
    valid_test_indices = []
    
    for i in range(test_start_idx + window_size, n):
        # Sequence input: scaled historical values
        sequences.append(all_scaled[i - window_size:i, 0])
        # Context input: static features for current step 'i'
        static_feats.append(df.iloc[i][static_cols].values.astype(np.float32))
        valid_test_indices.append(i)
        
    if not sequences:
        raise ValueError("Data tidak cukup untuk membangun urutan 14 hari LSTM. Kurangi persentase test_size atau tambah jumlah baris dataset.")
        
    x_seq = np.array(sequences).reshape(len(sequences), window_size, 1)
    x_static = np.array(static_feats)
    
    # Transform using training static scaler
    if scaler_exog is not None:
        x_static = scaler_exog.transform(x_static)
            
    # Combine inputs for the Keras Dual-Input structural layout
    X_test = [x_seq, x_static]
    
    # 6. Prepare actual target values and formatted evaluation timestamps
    y_test = df.iloc[valid_test_indices][target_col].values
    test_dates = df.iloc[valid_test_indices][date_col].dt.strftime('%Y-%m-%d').tolist()
    
    return X_test, y_test, test_dates