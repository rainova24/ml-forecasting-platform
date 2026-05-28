import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

# Mengatur gaya plotting bawaan seaborn agar rapi
sns.set_theme(style="darkgrid")

def find_optimal_d(series, max_d=2):
    """
    Fungsi untuk mencari nilai orde differencing (d) yang optimal 
    berdasarkan Augmented Dickey-Fuller (ADF) test (alpha = 0.05).
    """
    d = 0
    current_series = series.copy().dropna()
    
    # 1. Hitung p-value awal (sebelum differencing, d=0)
    # adfuller() mengembalikan tuple, index [1] adalah p-value
    p_value = adfuller(current_series)[1]
    print(f"Nilai p-value awal (d=0): {p_value:.6f}")
    
    # 2. Looping selama data tidak stasioner (p-value >= 0.05) dan d belum mencapai batas max_d
    while p_value >= 0.05 and d < max_d:
        d += 1
        # Mengaplikasikan differencing (selisih nilai saat ini dengan nilai sebelumnya)
        current_series = current_series.diff().dropna()
        
        # Lakukan uji ADF ulang pada data yang sudah di-differencing
        p_value = adfuller(current_series)[1]
        print(f"  -> Uji ADF pada d={d}, p-value: {p_value:.6f}")
        
    # Peringatan jika sudah mencapai max_d tetapi p-value masih >= 0.05
    if p_value >= 0.05 and d == max_d:
        print(f"\n[PERINGATAN] Data belum stasioner sepenuhnya hingga batas maksimal (d={max_d}).")
        
    return d, current_series

# ==============================================================================
# EKSEKUSI PROGRAM
# ==============================================================================

# 1. Memuat Dataset
input_filename = 'dataset_bca_siap_uji.csv'
print(f"Memuat data dari {input_filename}...\n")
df = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')

# Memastikan data terurut berdasarkan tanggal (wajib untuk operasi .diff())
df.sort_index(inplace=True)

# 2. Menjalankan Fungsi Pencarian Nilai d Optimal
print("=== PROSES PENCARIAN NILAI d ===")
optimal_d, diff_series = find_optimal_d(df['Close'], max_d=2)

# 3. Laporan Terminal
print("\n=== KESIMPULAN ===")
print(f"Proses differencing dilakukan sebanyak: {optimal_d} kali")
print(f"Nilai d yang optimal untuk ARIMA adalah: {optimal_d}")

# 4. Visualisasi Data Asli vs Differencing (untuk laporan)
print("\nMenyiapkan grafik perbandingan data asli vs data differencing...")
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

# Subplot Atas: Pergerakan Data Asli
axes[0].plot(df.index, df['Close'], color='dodgerblue', linewidth=1.5)
axes[0].set_title('Pergerakan Data Asli (Harga Tutup BCA)', fontsize=16, fontweight='bold', pad=10)
axes[0].set_ylabel('Harga dalam Rupiah (IDR)', fontsize=12)

# Subplot Bawah: Pergerakan Data Setelah Differencing
axes[1].plot(diff_series.index, diff_series, color='tomato', linewidth=1.5)
axes[1].set_title(f'Pergerakan Data Setelah di-differencing (d = {optimal_d})', fontsize=16, fontweight='bold', pad=10)
axes[1].set_ylabel('Perubahan Harga (Selisih)', fontsize=12)
axes[1].set_xlabel('Tahun', fontsize=12)

# Merapikan jarak antar subplot agar tidak bertumpuk
plt.tight_layout(pad=3.0)

# Menyimpan hasil visualisasi menjadi gambar PNG
plot_filename = f'plot_differencing_d{optimal_d}.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Grafik perbandingan telah disimpan ke: {plot_filename}")

# Menampilkan grafik ke layar pengguna
plt.show()

print("\nTahap penentuan nilai d selesai!")
