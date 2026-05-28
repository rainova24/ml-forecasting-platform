import pandas as pd

# ==============================================================================
# FEATURE ENGINEERING UNTUK MACHINE LEARNING
# ==============================================================================

# 1. Memuat Dataset
input_filename = 'dataset_bca_siap_uji.csv'
print(f"Memuat data dari {input_filename}...\n")

# Memastikan Date menjadi tipe datetime dan dijadikan index
df = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')

# Memastikan urutan data berdasarkan waktu dari yang paling lampau ke terbaru
df.sort_index(inplace=True)

print("Memulai proses Feature Engineering...")

# ==============================================================================
# TAHAP 1: Pembuatan Lag Features (Fitur Masa Lalu)
# ==============================================================================
# Menggunakan fungsi shift() untuk menggeser data ke bawah.
# Tujuannya agar model dapat belajar dari harga t-1, t-2, hingga t-5.
for i in range(1, 6):
    df[f'Lag_{i}'] = df['Close'].shift(i)

# ==============================================================================
# TAHAP 2: Pembuatan Rolling Features (Fitur Tren Berjalan)
# ==============================================================================
# MA_5: Rata-rata pergerakan harga selama 5 hari bursa terakhir (representasi 1 minggu)
df['MA_5'] = df['Close'].rolling(window=5).mean()

# MA_20: Rata-rata pergerakan harga selama 20 hari bursa terakhir (representasi 1 bulan)
df['MA_20'] = df['Close'].rolling(window=20).mean()

# ==============================================================================
# TAHAP 3: Pembersihan Sisa Transformasi
# ==============================================================================
# Karena shift(5) dan rolling(20) membutuhkan data historis sebelumnya,
# maka 19 baris pertama pasti akan memiliki setidaknya satu elemen NaN.
# Kita menghapusnya menggunakan dropna() agar model (SVR/XGBoost) tidak error.
df = df.dropna()

# ==============================================================================
# TAHAP 4: Verifikasi dan Ekspor
# ==============================================================================
print("\n=== 5 Baris Pertama Data dengan Fitur Baru ===")
# Menampilkan beberapa kolom utama dan fitur baru untuk verifikasi visual
kolom_verifikasi = ['Close', 'Lag_1', 'Lag_5', 'MA_5', 'MA_20']
print(df[kolom_verifikasi].head())

# Menyimpan hasil akhir ke file CSV baru
output_filename = 'dataset_bca_ml_features.csv'
df.to_csv(output_filename)

print(f"\nDataset feature engineering berhasil disimpan ke: {output_filename}")
print("Data kini siap digunakan untuk melatih model Machine Learning pembanding (SVR/XGBoost)!")
