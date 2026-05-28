import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
import warnings

# Mengabaikan warning agar terminal tetap bersih
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. PERSIAPAN DATA (Memuat variabel dari file CSV yang sudah kita buat sebelumnya)
# ==============================================================================
# Meskipun Anda mengasumsikan variabel sudah ada di memori, 
# kita tetap perlu memuatnya dari file agar script ini bisa berjalan mandiri (standalone).
print("Memuat data hasil prediksi dan menghitung nilai MAPE...")
df = pd.read_csv('tabel_hasil_prediksi_arima.csv', parse_dates=['Date'], index_col='Date')

# Mengekstrak variabel yang dibutuhkan ke dalam format Series Pandas
data_test = df['Aktual_Test']
prediksi = df['Prediksi_ARIMA']
batas_bawah = df['Batas_Bawah_95%']
batas_atas = df['Batas_Atas_95%']

# Menghitung ulang MAPE secara instan untuk dimasukkan ke dalam grafik
mape_value = mean_absolute_percentage_error(data_test, prediksi) * 100


# ==============================================================================
# 2. PENGATURAN KANVAS (FIGURE SETUP)
# ==============================================================================
# Menggunakan style matplotlib yang rapi, modern, dan bernuansa akademis
plt.style.use('seaborn-v0_8-whitegrid')

# Membuat figure berukuran lebar dan proporsional untuk laporan (14x7 inch)
fig, ax = plt.subplots(figsize=(14, 7))


# ==============================================================================
# 3. PEMBUATAN GARIS & AREA (PLOTTING)
# ==============================================================================
# a) Plot Garis Harga Aktual (Data Testing)
# Menggunakan warna hijau tua eksklusif (#1a9850), dengan marker lingkaran 'o' kecil
# untuk menunjukkan detail harga dari hari ke hari (membantu analisis visual di laporan)
ax.plot(data_test.index, data_test.values, color='#1a9850', linewidth=2, 
        marker='o', markersize=4, label='Data Aktual (Test)')

# b) Plot Garis Prediksi ARIMA
# Menggunakan warna merah terang (#d73027) dengan garis putus-putus
ax.plot(prediksi.index, prediksi.values, color='#d73027', linewidth=2, 
        linestyle='--', label='Prediksi ARIMA')

# c) Area Arsiran 95% Confidence Interval
# Menggunakan warna merah muda/merah pucat yang transparan (alpha=0.15)
ax.fill_between(data_test.index, batas_bawah.values, batas_atas.values, 
                color='#d73027', alpha=0.15, label='95% Confidence Interval')


# ==============================================================================
# 4. ANOTASI & ESTETIKA GRAFIK
# ==============================================================================
# Menambahkan judul yang elegan, besar, dan tebal (bold)
ax.set_title('Komparasi Harga Aktual vs Prediksi Model ARIMA (Data Testing) - Saham BCA', 
             fontsize=18, fontweight='bold', pad=20, color='#222222')

# Label sumbu X dan Y
ax.set_xlabel('Tanggal', fontsize=12, labelpad=12, fontweight='500')
ax.set_ylabel('Harga Tutup (IDR)', fontsize=12, labelpad=12, fontweight='500')

# Mempercantik garis grid agar lebih halus
ax.grid(True, linestyle='-', alpha=0.6)

# Memasang Legend di lokasi terbaik yang tidak menutupi garis (loc='best')
# frameon=True untuk memberikan kotak bingkai, shadow=True untuk efek dimensi
ax.legend(loc='best', fontsize=11, frameon=True, shadow=True, edgecolor='#cccccc')

# FITUR SPESIAL: Membuat Kotak Teks untuk nilai akurasi MAPE
# Posisi menggunakan koordinat relatif (0.02, 0.04) -> Kiri Bawah
text_str = f'Metrik Evaluasi:\nMAPE: {mape_value:.2f}%'
# Mengatur styling kotak (background putih solid, sudut membulat, dan garis pinggir tebal)
props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='#1a9850', alpha=0.95, linewidth=1.5)
ax.text(0.02, 0.04, text_str, transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', bbox=props, fontweight='bold', color='#111111')


# ==============================================================================
# 5. OUTPUT PENYIMPANAN DAN TAMPILAN
# ==============================================================================
# Memastikan semua elemen terpasang pas dan tidak terpotong (tight_layout)
plt.tight_layout()

# Menyimpan hasil plotting ke file gambar PNG dengan Resolusi Tinggi (dpi=300)
output_filename = 'plot_prediksi_vs_aktual.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Grafik High-Quality berhasil digambar dan disimpan sebagai: {output_filename}")

# Menampilkan hasil visualisasi ke layar pengguna
plt.show()
