import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("=== TAHAP 3: FORECASTING & EVALUASI HOLT-WINTERS ===\n")

# ==============================================================================
# 1. MEMUAT KOMPONEN MODEL DAN DATA UJI
# ==============================================================================
print("1. Memuat objek model dan data uji (Test Data)...")
with open('hw_bca_model.pkl', 'rb') as f:
    model_fit = pickle.load(f)

with open('hw_processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

y_test = data['y_test']
test_dates = data['test_dates']


# ==============================================================================
# 2. MELAKUKAN PREDIKSI (FORECASTING)
# ==============================================================================
print("2. Model meramalkan harga saham ke depan menyusuri periode Data Uji...")
# Menginstruksikan model untuk meramal sejauh 'x' hari ke depan (sebanyak data uji)
predictions = model_fit.forecast(steps=len(y_test))


# ==============================================================================
# 3. MENGHITUNG METRIK EVALUASI (KOMPARASI DENGAN ARIMA & LSTM)
# ==============================================================================
print("3. Mengalkulasi tingkat keakuratan dan metrik error...")
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions) * 100

print("\n=======================================================")
print("=== HASIL EVALUASI MODEL STATISTIK (HOLT-WINTERS) ===")
print("=======================================================")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"Mean Absolute Error (MAE)      : {mae:.2f}")
print(f"Mean Absolute Percentage Error : {mape:.2f}%")
print("=======================================================")


# ==============================================================================
# 4. MENYIMPAN HASIL PREDIKSI KE TABEL CSV
# ==============================================================================
df_hasil = pd.DataFrame({
    'Tanggal': test_dates,
    'Harga_Aktual': y_test,
    'Prediksi_HoltWinters': predictions
})
df_hasil.set_index('Tanggal', inplace=True)
df_hasil.to_csv('tabel_hasil_prediksi_hw.csv')
print("\n-> Data detail prediksi berhasil diekspor ke: 'tabel_hasil_prediksi_hw.csv'")


# ==============================================================================
# 5. VISUALISASI GRAFIK KOMPARATIF (AKTUAL VS PREDIKSI)
# ==============================================================================
print("4. Membuat lukisan grafik perbandingan Harga Aktual vs Prediksi...")
plt.figure(figsize=(14, 7))

# Plot Garis Aktual (Hijau)
plt.plot(df_hasil.index, df_hasil['Harga_Aktual'], color='#1a9850', linewidth=2, label='Data Aktual (Test Data)')

# Plot Garis Prediksi Holt-Winters (Oranye Putus-putus)
plt.plot(df_hasil.index, df_hasil['Prediksi_HoltWinters'], color='#ff7f00', linewidth=2, linestyle='--', label='Prediksi Holt-Winters')

plt.title(f'Hasil Peramalan Harga Saham BCA Menggunakan Holt-Winters\n(Tingkat Kesalahan / MAPE: {mape:.2f}%)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Tanggal / Waktu', fontsize=12)
plt.ylabel('Harga Penutupan (IDR)', fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()

plot_filename = 'plot_prediksi_vs_aktual_hw.png'
plt.savefig(plot_filename, dpi=300)
print(f"-> Grafik komparatif berhasil disimpan sebagai: {plot_filename}")

# Tampilkan di layar
plt.show()

print("\nTahap Evaluasi Akhir Holt-Winters (Tahap 3) SELESAI!")
