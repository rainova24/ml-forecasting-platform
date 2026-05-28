import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur gaya plot agar lebih rapi
sns.set_theme(style="darkgrid")

# ==============================================================================
# 1. MEMUAT DATASET
# ==============================================================================
# Menggunakan dataset yang sudah siap dan memiliki fitur tambahan
input_filename = 'dataset_bca_ml_features.csv'
print(f"Memuat data dari {input_filename}...\n")

df = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')
df.sort_index(inplace=True)

# ==============================================================================
# 2. PEMBAGIAN KRONOLOGIS (TRAIN-TEST SPLIT)
# ==============================================================================
print("=== MELAKUKAN TRAIN-TEST SPLIT (80% Train, 20% Test) ===")

# Menghitung titik potong (index) untuk 80% data
split_idx = int(len(df) * 0.8)

# Melakukan pembagian berbasis indeks (slicing)
# Ini sangat mutlak untuk Time Series agar tidak merusak urutan waktu
train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]

# Menyimpan hasil pembagian agar bisa dipakai di skrip pemodelan berikutnya
train_data.to_csv('dataset_bca_train.csv')
test_data.to_csv('dataset_bca_test.csv')

# ==============================================================================
# 3. VERIFIKASI DATA (Dimensi & Tanggal)
# ==============================================================================
print("\n[Verifikasi Dimensi Data]")
print(f"Total Baris Keseluruhan : {len(df)}")
print(f"Dimensi Data Train      : {train_data.shape}")
print(f"Dimensi Data Test       : {test_data.shape}")

print("\n[Verifikasi Rentang Waktu - Memastikan Tidak Ada Overlap]")
print(f"Data Train : {train_data.index.min().strftime('%Y-%m-%d')} s/d {train_data.index.max().strftime('%Y-%m-%d')}")
print(f"Data Test  : {test_data.index.min().strftime('%Y-%m-%d')} s/d {test_data.index.max().strftime('%Y-%m-%d')}")

# ==============================================================================
# 4. VISUALISASI UNTUK LAPORAN
# ==============================================================================
print("\nMenyiapkan grafik visualisasi Train-Test Split...")
plt.figure(figsize=(14, 7))

# Plotting Data Train (menggunakan warna biru)
plt.plot(train_data.index, train_data['Close'], color='dodgerblue', linewidth=1.5, label='Data Train (80%)')

# Plotting Data Test (menggunakan warna oranye)
plt.plot(test_data.index, test_data['Close'], color='darkorange', linewidth=1.5, label='Data Test (20%)')

# Menambahkan garis vertikal sebagai penanda visual titik potong
split_date = train_data.index.max()
plt.axvline(x=split_date, color='black', linestyle='--', linewidth=1.5, alpha=0.6, label='Titik Pemisah (Split Point)')

# Menambahkan Judul dan Label sesuai spesifikasi
plt.title('Visualisasi Train-Test Split (80/20) - Harga Saham BCA', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Tahun', fontsize=12, labelpad=10)
plt.ylabel('Harga dalam Rupiah (IDR)', fontsize=12, labelpad=10)

# Menambahkan legend dan memastikan layout rapi
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()

# Menyimpan grafik untuk kebutuhan laporan
plot_filename = 'plot_train_test_split.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Grafik perbandingan telah disimpan ke: {plot_filename}")

# Menampilkan grafik ke layar
plt.show()

print("\nTahap pembagian Train-Test Split berhasil diselesaikan!")
