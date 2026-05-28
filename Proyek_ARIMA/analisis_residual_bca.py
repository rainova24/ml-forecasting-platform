import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro
import warnings

# Mengabaikan warning agar terminal tetap bersih
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. PERSIAPAN DATA DAN EKSTRAKSI RESIDUAL
# ==============================================================================
# Memuat data latih kembali karena residual didapatkan dari proses fitting (in-sample)
df_train = pd.read_csv('dataset_bca_train.csv', parse_dates=['Date'], index_col='Date')
y_train = df_train['Close']

# Melatih ulang model ARIMA dengan parameter terbaik (0, 1, 1)
p, d, q = 0, 1, 1
model = ARIMA(y_train, order=(p, d, q))
model_fit = model.fit()

# Mengekstrak data residual (selisih antara aktual train vs prediksi in-sample)
residual = model_fit.resid

# [PENTING] Membuang baris pertama (sejumlah d) dari residual 
# Karena pada saat differencing (d=1), baris pertama tidak bisa dihitung (biasanya 0) 
# dan dapat merusak uji normalitas/white noise.
residual = residual.iloc[d:]


# ==============================================================================
# 2. UJI STATISTIK FORMAL
# ==============================================================================
print("\n=== UJI STATISTIK DIAGNOSTIK RESIDUAL ===")

# a) Uji Ljung-Box (Uji White Noise / Autokorelasi)
# H0: Data merupakan White Noise (Tidak ada autokorelasi pada lag tertentu)
# Jika p-value > 0.05, kita GAGAL menolak H0 (Artinya data benar White Noise)
ljung_box_result = acorr_ljungbox(residual, lags=[10])
p_value_lb = ljung_box_result['lb_pvalue'].iloc[0]

print(f"\n1. Uji Ljung-Box (White Noise Test)")
print(f"   P-Value: {p_value_lb:.4f}")
if p_value_lb > 0.05:
    print("   Kesimpulan: Residual bersifat White Noise (Model Valid)")
else:
    print("   Kesimpulan: Peringatan: Residual masih memiliki autokorelasi")


# b) Uji Shapiro-Wilk (Uji Normalitas)
# H0: Data berdistribusi Normal
# Jika p-value > 0.05, kita GAGAL menolak H0 (Artinya data berdistribusi normal)
stat_sw, p_value_sw = shapiro(residual)

print(f"\n2. Uji Shapiro-Wilk (Normalitas Test)")
print(f"   P-Value: {p_value_sw:.4f}")
if p_value_sw > 0.05:
    print("   Kesimpulan: Residual Berdistribusi Normal")
else:
    print("   Kesimpulan: Residual tidak berdistribusi normal sempurna")

print("=========================================\n")


# ==============================================================================
# 3. VISUALISASI DIAGNOSTIK RESIDUAL (4-IN-1)
# ==============================================================================
print("Menyiapkan grafik diagnostik 4-in-1...")

# Mengatur tema plot menjadi rapi dan profesional
sns.set_theme(style="whitegrid")

# Membuat figure berukuran besar (12x8) dengan susunan 2 baris x 2 kolom
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Memberikan Judul Utama pada Figure
fig.suptitle('Diagnostik Residual Model ARIMA', fontsize=18, fontweight='bold', y=1.02)


# --- Subplot 1 (Kiri Atas): Line Plot Residual ---
# Tujuan: Melihat pergerakan error, rata-ratanya harus di angka 0 dan rentangnya stabil (varians konstan)
axes[0, 0].plot(residual.index, residual.values, color='dodgerblue', linewidth=1)
axes[0, 0].axhline(y=0, color='crimson', linestyle='--', linewidth=2) # Garis merah di 0
axes[0, 0].set_title('Pergerakan Residual (Stabilitas Varians)', fontweight='bold')
axes[0, 0].set_ylabel('Nilai Error')


# --- Subplot 2 (Kanan Atas): Histogram & KDE ---
# Tujuan: Memastikan distribusi error membentuk kurva lonceng yang simetris di angka 0
sns.histplot(residual, kde=True, ax=axes[0, 1], color='mediumpurple', edgecolor='white', bins=30)
axes[0, 1].set_title('Distribusi Residual (Normalitas)', fontweight='bold')
axes[0, 1].set_xlabel('Nilai Residual')
axes[0, 1].set_ylabel('Frekuensi')


# --- Subplot 3 (Kiri Bawah): Q-Q Plot ---
# Tujuan: Evaluasi visual normalitas ekstrem di bagian ekor (tails)
# Parameter line='s' membuat garis diagonal merah referensi
sm.qqplot(residual, line='s', ax=axes[1, 0], color='dodgerblue', alpha=0.5)
axes[1, 0].set_title('Q-Q Plot (Distribusi Ekor/Outlier)', fontweight='bold')


# --- Subplot 4 (Kanan Bawah): Plot ACF Residual ---
# Tujuan: Membuktikan tidak ada pola/informasi yang tersisa (semua bar dalam area biru)
plot_acf(residual, ax=axes[1, 1], lags=30, alpha=0.05, color='dodgerblue')
axes[1, 1].set_title('Plot ACF Residual (White Noise)', fontweight='bold')
axes[1, 1].set_xlabel('Lags (Hari)')
axes[1, 1].set_ylabel('Korelasi')


# Merapikan susunan grafik agar judul dan sumbu tidak saling tumpang tindih
plt.tight_layout()

# Menyimpan grafik sebagai file gambar berkualitas tinggi
output_filename = 'plot_diagnostik_residual_manual.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Visualisasi diagnostik berhasil disimpan ke: {output_filename}")

# Menampilkan grafik ke layar
plt.show()

print("\nTahap validasi matematis (Analisis Residual) selesai!")
