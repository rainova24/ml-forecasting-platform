import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Mengabaikan warning statsmodels agar terminal tetap rapi
warnings.filterwarnings('ignore')

# Mengatur tema plot agar profesional
sns.set_theme(style="darkgrid")

# ==============================================================================
# 1. PERSIAPAN DATA (TRAIN & TEST)
# ==============================================================================
print("Memuat data latih (Train) dan data uji (Test)...")
df_train = pd.read_csv('dataset_bca_train.csv', parse_dates=['Date'], index_col='Date')
df_test = pd.read_csv('dataset_bca_test.csv', parse_dates=['Date'], index_col='Date')

# Memisahkan kolom target (Harga Tutup / Close)
y_train = df_train['Close']
y_test = df_test['Close']


# ==============================================================================
# 2. MELATIH KEMBALI MODEL ARIMA TERBAIK
# ==============================================================================
# Menggunakan parameter optimal dari hasil Auto-ARIMA sebelumnya
p, d, q = 0, 1, 1
print(f"Melatih ulang model ARIMA({p},{d},{q}) dengan data Train...")

model = ARIMA(y_train, order=(p, d, q))
model_fit = model.fit()


# ==============================================================================
# 3. PROSES FORECASTING (PREDIKSI) PADA DATA UJI (TEST)
# ==============================================================================
print("\nMemulai proses Forecasting (Prediksi) ke masa depan...")

# Meminta model memprediksi ke depan sebanyak jumlah observasi yang ada pada Data Test
# Nilai steps disesuaikan dengan panjang (length) dari y_test
forecast_result = model_fit.get_forecast(steps=len(y_test))

# Mengekstrak nilai prediksi (nilai rata-rata ekspektasi / mean forecast)
prediksi = forecast_result.predicted_mean
# Menyelaraskan index prediksi agar sama persis dengan index Data Test (untuk memudahkan visualisasi)
prediksi.index = y_test.index

# Mengekstrak batas bawah dan batas atas dari tingkat kepercayaan (Confidence Interval 95%)
conf_int = forecast_result.conf_int(alpha=0.05) # alpha=0.05 berarti 95% Confidence Interval
batas_bawah = conf_int.iloc[:, 0]
batas_atas = conf_int.iloc[:, 1]
batas_bawah.index = y_test.index
batas_atas.index = y_test.index

# Menggabungkan Aktual, Prediksi, dan Confidence Interval ke dalam satu DataFrame baru
df_hasil = pd.DataFrame({
    'Aktual_Test': y_test,
    'Prediksi_ARIMA': prediksi,
    'Batas_Bawah_95%': batas_bawah,
    'Batas_Atas_95%': batas_atas
})

# Menyimpan tabel hasil ke CSV untuk keperluan analisis/lampiran laporan
hasil_filename = 'tabel_hasil_prediksi_arima.csv'
df_hasil.to_csv(hasil_filename)
print(f"Tabel komparasi prediksi berhasil disimpan ke: {hasil_filename}")


# ==============================================================================
# 4. VISUALISASI HASIL PREDIKSI VS AKTUAL
# ==============================================================================
print("\nMenyiapkan grafik visualisasi Forecasting...")
plt.figure(figsize=(14, 7))

# Plot 1: Garis Harga Close dari Data Train historis (Warna Biru)
# Mengambil 150 hari terakhir saja dari Data Train agar grafik tidak terlalu padat ke belakang
plt.plot(y_train.index[-150:], y_train.values[-150:], color='dodgerblue', linewidth=2, label='Data Latih Historis (150 Hari Terakhir)')

# Plot 2: Garis Harga Close dari Data Test aktual (Warna Hijau)
plt.plot(y_test.index, y_test.values, color='mediumseagreen', linewidth=2, label='Data Aktual (Test)')

# Plot 3: Garis Prediksi dari model ARIMA (Warna Merah Putus-putus)
plt.plot(prediksi.index, prediksi.values, color='crimson', linestyle='--', linewidth=2, label='Prediksi ARIMA')

# Plot 4: Area Arsiran untuk 95% Confidence Interval (Warna Abu/Pink transparan)
plt.fill_between(
    y_test.index,
    batas_bawah.values,
    batas_atas.values,
    color='pink',
    alpha=0.3,
    label='95% Confidence Interval'
)

# Menambahkan judul, label sumbu, dan legend yang profesional
plt.title('Hasil Prediksi ARIMA vs Data Aktual - Harga Saham BCA', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Tahun', fontsize=12, labelpad=10)
plt.ylabel('Harga dalam Rupiah (IDR)', fontsize=12, labelpad=10)
plt.legend(loc='upper left', fontsize=11)

# Merapikan layout plot
plt.tight_layout()

# Menyimpan grafik sebagai gambar PNG resolusi tinggi
plot_filename = 'plot_forecasting_arima.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"Grafik forecasting telah disimpan sebagai: {plot_filename}")

# Menampilkan grafik langsung ke layar
plt.show()

print("\nTahap The Grand Finale (Forecasting) selesai!")
