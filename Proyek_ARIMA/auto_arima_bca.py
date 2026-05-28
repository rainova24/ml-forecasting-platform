import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pmdarima import auto_arima

# Mengatur tema plotting menjadi whitegrid
sns.set_theme(style="whitegrid")

# ==============================================================================
# 1. PERSIAPAN DATA
# ==============================================================================
input_filename = 'dataset_bca_train.csv'
print(f"Memuat data latih (Train Data) dari {input_filename}...\n")

df_train = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')
df_train.sort_index(inplace=True)

# Mengambil kolom Close untuk pelatihan model
y_train = df_train['Close']


# ==============================================================================
# 2. PENCARIAN MODEL DENGAN AUTO-ARIMA
# ==============================================================================
print("=== MEMULAI PENCARIAN PARAMETER AUTO-ARIMA ===")
print("Sedang mengevaluasi nilai AIC untuk berbagai kombinasi (p, d, q)...\n")

# Konfigurasi parameter auto_arima sesuai permintaan:
# - start_p=0, start_q=0 : Memulai pencarian dari ordo AR dan MA = 0.
# - max_p=5, max_q=5     : Batas atas pencarian ordo AR dan MA adalah 5.
# - d=None               : Membiarkan fungsi menentukan orde differencing (d) secara otomatis.
# - test='adf'           : Menggunakan uji Augmented Dickey-Fuller (ADF) untuk memvalidasi nilai d.
# - seasonal=False       : Menggunakan ARIMA standar (bukan SARIMA/Seasonal ARIMA).
# - trace=True           : Mencetak setiap iterasi model beserta skor AIC-nya ke terminal.
# - error_action='ignore': Melewati iterasi jika suatu kombinasi parameter gagal konvergen (error).
# - suppress_warnings=True: Menyembunyikan peringatan/warning agar output terminal rapi.
best_model = auto_arima(
    y_train,
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    d=None,
    test='adf',
    seasonal=False,
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True  # Pendekatan stepwise membuat komputasi pencarian jauh lebih cepat
)


# ==============================================================================
# 3. OUTPUT RINGKASAN & ANALISIS RESIDUAL
# ==============================================================================
print("\n=== RINGKASAN MODEL TERBAIK (BEST MODEL) ===")
# Cetak ringkasan statistik dari model terbaik (koefisien, p-value, AIC, BIC, dll.)
print(best_model.summary())

print("\nMenyiapkan grafik diagnostik residual...")
# Membuat plot diagnostik untuk menganalisis sisa error (residual) dari model
# Terdiri dari 4 grafik: Standardized residual, Histogram, Q-Q plot, dan Correlogram (ACF dari residual)
fig = best_model.plot_diagnostics(figsize=(10, 8))

# Menyimpan hasil visualisasi menjadi gambar PNG
plot_filename = 'plot_diagnostics_arima.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Grafik diagnostik telah berhasil disimpan sebagai: {plot_filename}")

# Menampilkan grafik ke layar
plt.show()

print("\nTahap penentuan model Auto-ARIMA selesai!")
