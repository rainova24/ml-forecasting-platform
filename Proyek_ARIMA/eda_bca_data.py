import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur gaya plot bawaan dari seaborn agar grafik terlihat lebih rapi dan profesional
sns.set_theme(style="darkgrid")

# 1. Memuat file dataset dan memastikan kolom Date menjadi tipe datetime serta diset sebagai index
input_filename = 'dataset_bca_lengkap.csv'
print(f"Memuat data dari {input_filename}...")

# parse_dates=['Date'] secara otomatis mengonversi string ke datetime
# index_col='Date' mengatur kolom Date sebagai index utama dataframe
df = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')

# 2. Pengecekan Missing Values sebelum diproses
print("\n=== Laporan Missing Values (Sebelum Interpolasi) ===")
print(df.isnull().sum())

# 3. Penanganan Missing Values menggunakan Interpolasi Waktu
# Metode 'time' sangat ideal untuk time series karena mengestimasi nilai kosong 
# berdasarkan rentang/jarak waktu riil antar index (tanggal).
print("\nMelakukan penanganan missing values dengan interpolasi waktu (method='time')...")
df = df.interpolate(method='time')

# Memastikan jika ada baris pertama yang masih NaN (karena interpolasi tidak bisa ke belakang),
# kita bisa menambahkan backward fill sebagai jaring pengaman terakhir.
df = df.bfill()

# 4. Pengecekan ulang Missing Values setelah diinterpolasi
print("\n=== Laporan Missing Values (Setelah Interpolasi) ===")
print(df.isnull().sum())

# 5. Visualisasi Exploratory Data Analysis (EDA)
print("\nMembuat plot pergerakan Harga Tutup (Close)...")
plt.figure(figsize=(14, 7))

# Membuat plot garis untuk kolom Close
plt.plot(df.index, df['Close'], color='dodgerblue', linewidth=1.5, label='Harga Tutup (BBCA.JK)')

# Menambahkan judul dan label sumbu yang informatif
plt.title('Pergerakan Harga Saham Bank Central Asia (BBCA)', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Tahun', fontsize=12, labelpad=10)
plt.ylabel('Harga dalam Rupiah (IDR)', fontsize=12, labelpad=10)

# Menambahkan grid, legenda, dan memastikan layout pas
plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# Menyimpan grafik sebagai file gambar (PNG)
plot_filename = 'grafik_bbca_close.png'
plt.savefig(plot_filename, dpi=300)
print(f"Visualisasi berhasil dibuat dan disimpan sebagai: {plot_filename}")

# Menampilkan grafik ke layar pengguna
plt.show()

# 6. Menyimpan dataset akhir yang sudah bersih ke CSV baru
output_filename = 'dataset_bca_siap_uji.csv'
df.to_csv(output_filename)
print(f"\nDataset siap uji telah berhasil disimpan ke: {output_filename}")
print("Tahap EDA dan pembersihan data selesai!")
