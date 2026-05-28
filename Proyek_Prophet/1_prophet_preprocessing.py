import pandas as pd
import numpy as np
import pickle

print("=== TAHAP 1: DATA SPLITTING & PREPROCESSING PROPHET ===\n")

# ==============================================================================
# 1. MEMUAT DATASET
# ==============================================================================
file_path = 'dataset_bca_siap_uji.csv'
print(f"Memuat data dari {file_path}...")
df = pd.read_csv(file_path)


# ==============================================================================
# 2. PENYESUAIAN STANDAR ALGORITMA PROPHET
# ==============================================================================
# Aturan baku dari Facebook/Meta: Algoritma Prophet menolak membaca data 
# jika nama kolom waktunya bukan 'ds' (Date Stamp) dan nilai targetnya bukan 'y'.
df_prophet = df[['Date', 'Close']].copy()
df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Pastikan tipe datanya adalah format waktu yang dikenali kalender
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
df_prophet.sort_values('ds', inplace=True)

print("Nama kolom berhasil diubah menjadi format 'ds' (Waktu) dan 'y' (Harga).")


# ==============================================================================
# 3. PEMBAGIAN DATA (TRAIN & TEST SPLIT)
# ==============================================================================
# Membagi secara kronologis (80% Train, 20% Test) agar adil dengan algoritma lainnya
training_data_len = int(np.ceil(len(df_prophet) * .8))

train_data = df_prophet.iloc[:training_data_len]
test_data = df_prophet.iloc[training_data_len:]

print(f"\nDistribusi Data:")
print(f"- Total data tersedia: {len(df_prophet)} hari")
print(f"- Data Latih (Train) 80%: {len(train_data)} hari")
print(f"- Data Uji (Test) 20%: {len(test_data)} hari")


# ==============================================================================
# 4. MENYIMPAN HASIL PREPROCESSING
# ==============================================================================
# Disimpan dalam bentuk CSV karena Prophet membacanya lewat Pandas DataFrame
train_data.to_csv('prophet_train_data.csv', index=False)
test_data.to_csv('prophet_test_data.csv', index=False)

print("\n-> Data siap pakai telah berhasil diekspor ke format standar Prophet!")
print("Tahap prapemrosesan Prophet (Tahap 1) SELESAI.")
