import pandas as pd
import numpy as np
import pickle

print("=== TAHAP 1: DATA SPLITTING & PREPROCESSING HOLT-WINTERS ===\n")

# ==============================================================================
# 1. MEMUAT DATASET YANG SIAP UJI
# ==============================================================================
file_path = 'dataset_bca_siap_uji.csv'
print(f"Memuat data dari {file_path}...")
# Memastikan kolom Date dibaca sebagai format waktu (Datetime)
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
df.sort_index(inplace=True)

# Mengambil HANYA kolom Close karena Holt-Winters klasik adalah Univariate
data = df.filter(['Close'])


# ==============================================================================
# 2. PEMBAGIAN DATA (TRAIN & TEST SPLIT)
# ==============================================================================
# Membagi data secara berurutan/kronologis (80% Train, 20% Test) 
# Persentase ini disamakan persis dengan metode ARIMA dan LSTM sebelumnya
training_data_len = int(np.ceil(len(data) * .8))

train_data = data.iloc[:training_data_len]
test_data = data.iloc[training_data_len:]

print(f"\nDistribusi Data:")
print(f"- Total data tersedia: {len(data)} hari")
print(f"- Data Latih (Train) 80%: {len(train_data)} hari")
print(f"- Data Uji (Test) 20%: {len(test_data)} hari")

# ==============================================================================
# 3. PENYESUAIAN INDEKS WAKTU (SANGAT PENTING UNTUK STATSMODELS)
# ==============================================================================
# Modul statsmodels sangat rewel soal index waktu (frekuensi). 
# Karena pasar modal libur di akhir pekan, kita ubah frekuensinya menjadi 'B' (Business Days)
# Namun, untuk menghindari error tanggal merah (libur nasional), kita akan membuang info 
# tanggal selama proses matematis, dan mengembalikan tanggalnya saat divisualisasikan nanti.

y_train = train_data['Close'].values
y_test = test_data['Close'].values
train_dates = train_data.index
test_dates = test_data.index

# ==============================================================================
# 4. MENYIMPAN HASIL PREPROCESSING (EXPORT)
# ==============================================================================
# Menyimpan array dan index tanggal ke dalam format Pickle
hw_processed_data = {
    'y_train': y_train,
    'y_test': y_test,
    'train_dates': train_dates,
    'test_dates': test_dates
}

with open('hw_processed_data.pkl', 'wb') as f:
    pickle.dump(hw_processed_data, f)
    
print("\n-> Data siap pakai telah berhasil diekspor ke 'hw_processed_data.pkl'!")
print("Tahap prapemrosesan Holt-Winters (Tahap 1) SELESAI.")
