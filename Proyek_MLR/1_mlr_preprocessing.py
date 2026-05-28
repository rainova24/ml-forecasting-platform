import pandas as pd
import numpy as np
import pickle

print("=== TAHAP 1: PREPROCESSING MULTIVARIATE (MLR) ===\n")

# ==============================================================================
# 1. MEMUAT DATASET
# ==============================================================================
file_path = 'dataset_bca_siap_uji.csv'
print(f"1. Memuat data dari {file_path}...")
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
df.sort_index(inplace=True)

# ==============================================================================
# 2. MENCEGAH DATA LEAKAGE (MENGGESER FITUR KE HARI KEMARIN / H-1)
# ==============================================================================
print("2. Menerapkan Logika 'Time-Shifting' untuk mencegah Kecurangan (Data Leakage)...")

# Kita ingin menebak 'Close' HARI INI (Variabel Target Y)
df['Target_Y'] = df['Close']

# Berbekal 5 Fitur Pendukung HARI KEMARIN (Variabel Fitur X)
# Fungsi .shift(1) akan mendorong data 1 hari ke bawah
df['Open_X'] = df['Open'].shift(1)
df['High_X'] = df['High'].shift(1)
df['Low_X'] = df['Low'].shift(1)
df['Close_X'] = df['Close'].shift(1)
df['Volume_X'] = df['Volume'].shift(1)

# Karena kita menggeser data 1 hari ke bawah, baris paling pertama otomatis akan kosong (NaN).
# Kita harus membuang baris kosong tersebut agar mesin MLR tidak kebingungan (Error).
df.dropna(inplace=True)
print("   -> Fitur X berhasil digeser 1 hari ke belakang. Baris kosong telah dibuang.")

# ==============================================================================
# 3. PEMBAGIAN DATA (TRAIN & TEST SPLIT)
# ==============================================================================
print("3. Memisahkan Data Latih (Train) dan Data Uji (Test)...")

# Variabel X (5 Kolom Fitur Historis H-1)
X = df[['Open_X', 'High_X', 'Low_X', 'Close_X', 'Volume_X']].values

# Variabel Y (1 Kolom Target HARI INI)
Y = df['Target_Y'].values
tanggal = df.index

# Membagi secara kronologis (80% Train, 20% Test)
training_data_len = int(np.ceil(len(df) * .8))

X_train = X[:training_data_len]
X_test = X[training_data_len:]

Y_train = Y[:training_data_len]
Y_test = Y[training_data_len:]

train_dates = tanggal[:training_data_len]
test_dates = tanggal[training_data_len:]

print(f"\nDistribusi Matriks Data:")
print(f"- Total Data Siap Olah : {len(df)} hari")
print(f"- Dimensi Fitur X (X_Train) : {X_train.shape} (Hari, Kolom/Variabel)")
print(f"- Data Uji (Y_Test) 20% : {len(Y_test)} hari")

# ==============================================================================
# 4. MENYIMPAN HASIL PREPROCESSING
# ==============================================================================
print("\n4. Menyimpan susunan matriks data untuk Tahap Pelatihan...")
mlr_processed_data = {
    'X_train': X_train,
    'X_test': X_test,
    'Y_train': Y_train,
    'Y_test': Y_test,
    'train_dates': train_dates,
    'test_dates': test_dates
}

with open('mlr_processed_data.pkl', 'wb') as f:
    pickle.dump(mlr_processed_data, f)
    
print("-> Matriks berhasil diekspor ke: 'mlr_processed_data.pkl'")
print("Tahap prapemrosesan Regresi Berganda (Tahap 1) SELESAI.")
