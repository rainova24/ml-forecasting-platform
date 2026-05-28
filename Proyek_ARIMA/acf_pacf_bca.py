import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Mengatur tema plotting menjadi whitegrid agar garis korelasi dan batas signifikansi lebih jelas
sns.set_theme(style="whitegrid")

# ==============================================================================
# 1. PERSIAPAN DATA
# ==============================================================================
input_filename = 'dataset_bca_train.csv'
print(f"Memuat data latih (Train Data) dari {input_filename}...\n")

# Hanya menggunakan Data Latih untuk estimasi (mencegah data leakage)
df_train = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')
df_train.sort_index(inplace=True)

# Menerapkan Differencing untuk mencapai stasioneritas
# Asumsi nilai d optimal adalah 1 (berdasarkan mayoritas data saham)
# Ubah nilai 'd_optimal' ini jika output uji ADF Anda sebelumnya menyarankan angka lain.
d_optimal = 1
print(f"Mengaplikasikan differencing tingkat d={d_optimal} pada data latih...")

# Menghitung selisih dan membuang NaN yang dihasilkan oleh proses differencing
train_diff = df_train['Close'].diff(d_optimal).dropna()

# ==============================================================================
# 2. PEMBUATAN VISUALISASI ACF DAN PACF
# ==============================================================================
print("Membangun grafik ACF dan PACF...")

# Membuat Figure dengan 2 subplot (Atas untuk ACF, Bawah untuk PACF)
# Ukuran 14x10 sudah sangat proporsional untuk screenshot laporan
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))

# Subplot 1: Plot ACF (Membantu menentukan parameter MA / q)
# Area biru muda adalah Confidence Interval (biasanya 95%). Garis yang melewati area biru berarti signifikan.
plot_acf(train_diff, ax=axes[0], lags=40, alpha=0.05)
axes[0].set_title('Plot ACF (Estimasi q) pada Data Latih Stasioner', fontsize=16, fontweight='bold')
axes[0].set_ylabel('Korelasi (Auto-Correlation)', fontsize=12)

# Subplot 2: Plot PACF (Membantu menentukan parameter AR / p)
# Menggunakan method='ywm' (Yule-Walker dengan penyesuaian) untuk menghindari peringatan (warning)
plot_pacf(train_diff, ax=axes[1], lags=40, alpha=0.05, method='ywm')
axes[1].set_title('Plot PACF (Estimasi p) pada Data Latih Stasioner', fontsize=16, fontweight='bold')
axes[1].set_ylabel('Korelasi Parsial (Partial Auto-Correlation)', fontsize=12)
axes[1].set_xlabel('Lags (Jumlah Hari Ke Belakang)', fontsize=12)

# Merapikan jarak antar grafik agar judul dan label tidak tumpang tindih
plt.tight_layout(pad=3.0)

# ==============================================================================
# 3. PENYIMPANAN DAN OUTPUT
# ==============================================================================
# Menyimpan gambar dengan resolusi tinggi (dpi=300) yang cocok untuk disisipkan di Word/PDF
plot_filename = 'plot_acf_pacf_bca.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"\nGrafik ACF & PACF berkualitas tinggi telah disimpan ke: {plot_filename}")

# Menampilkan grafik ke layar pengguna
plt.show()

print("Tahap estimasi parameter p dan q selesai!")
