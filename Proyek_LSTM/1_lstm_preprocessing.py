import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

print("=== TAHAP 1: DATA PREPROCESSING & WINDOWING LSTM ===\n")

# ==============================================================================
# 1. MEMUAT DATASET YANG SIAP UJI
# ==============================================================================
file_path = 'dataset_bca_siap_uji.csv'
print(f"Memuat data dari {file_path}...")
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
df.sort_index(inplace=True)

# Mengambil HANYA kolom Close karena kita menggunakan pendekatan Univariate 
# (Sama dengan ARIMA agar perbandingannya adil / apples-to-apples)
data = df.filter(['Close'])
dataset = data.values


# ==============================================================================
# 2. PEMBAGIAN DATA (TRAIN & TEST SPLIT)
# ==============================================================================
# Membagi data secara berurutan/kronologis (80% Train, 20% Test) sama seperti ARIMA
training_data_len = int(np.ceil(len(dataset) * .8))

print(f"\nDistribusi Data:")
print(f"- Total data tersedia: {len(dataset)} hari")
print(f"- Data Latih (Train) 80%: {training_data_len} hari")
print(f"- Data Uji (Test) 20%: {len(dataset) - training_data_len} hari")


# ==============================================================================
# 3. SCALING (NORMALISASI) DATA
# ==============================================================================
# Jaringan Saraf Tiruan (Neural Network) sangat sensitif terhadap rentang angka yang besar
# Kita menekan (scale) semua harga saham ke rentang (0) hingga (1)
print("\nMelakukan scaling data ke rentang 0-1 dengan MinMaxScaler...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Menyimpan objek scaler untuk dipakai di tahap akhir (mengembalikan harga ke Rupiah)
with open('lstm_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("-> Objek scaler berhasil disimpan sebagai 'lstm_scaler.pkl'")


# ==============================================================================
# 4. MEMBANGUN SEQUENCES (SLIDING WINDOW) UNTUK DATA LATIH
# ==============================================================================
# Memutuskan jumlah observasi hari ke belakang yang akan dicerna oleh memori LSTM
time_step = 60 # Model melihat pola 60 hari terakhir untuk menebak harga esok hari
print(f"\nMembangun matriks Sequences (X, y) dengan time step = {time_step} hari...")

train_data = scaled_data[0:int(training_data_len), :]

x_train = []
y_train = []

for i in range(time_step, len(train_data)):
    x_train.append(train_data[i-time_step:i, 0]) # Fitur masukan (60 harga masa lalu)
    y_train.append(train_data[i, 0]) # Label target (harga pada hari ke-61)

# Mengubah Python List menjadi Numpy Array untuk komputasi matematis
x_train, y_train = np.array(x_train), np.array(y_train)

# [SANGAT PENTING] LSTM mewajibkan data input berwujud matriks 3 Dimensi:
# Format: [Jumlah Sampel (Samples), Langkah Waktu (Time Steps), Jumlah Fitur (Features)]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(f"-> Bentuk (Shape) X_train 3 Dimensi: {x_train.shape}")
print(f"-> Bentuk (Shape) y_train: {y_train.shape}")


# ==============================================================================
# 5. MEMBANGUN SEQUENCES (SLIDING WINDOW) UNTUK DATA UJI
# ==============================================================================
# Data uji dimulai mundur (training_data_len - time_step) hari 
# agar observasi PERTAMA di Data Uji memiliki konteks histori 60 hari dari Data Latih
test_data = scaled_data[training_data_len - time_step: , :]

x_test = []
# y_test dibiarkan dalam wujud harga aktual Rupiah asli (bukan skala 0-1) untuk evaluasi MAPE nanti
y_test = dataset[training_data_len:, :] 

for i in range(time_step, len(test_data)):
    x_test.append(test_data[i-time_step:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(f"-> Bentuk (Shape) X_test 3 Dimensi: {x_test.shape}")
print(f"-> Bentuk (Shape) y_test (Harga Aktual): {y_test.shape}")


# ==============================================================================
# 6. MENYIMPAN HASIL MATRIKS (EXPORT)
# ==============================================================================
# Kita bungkus semua array beserta kalender/tanggalnya ke dalam format Dictionary
processed_data = {
    'x_train': x_train,
    'y_train': y_train,
    'x_test': x_test,
    'y_test': y_test,
    'time_step': time_step,
    'train_dates': data.index[:training_data_len],
    'test_dates': data.index[training_data_len:]
}

# Menyimpan bungkusan matriks siap komputasi agar file script training nanti bersih
with open('lstm_processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)
    
print("\nSeluruh matriks input siap suap telah berhasil diekspor ke 'lstm_processed_data.pkl'!")
print("Tahap prapemrosesan LSTM (Tahap 1) SELESAI.")
