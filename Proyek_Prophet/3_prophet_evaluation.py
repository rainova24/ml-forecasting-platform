import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from prophet.serialize import model_from_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

# Mengabaikan warning tampilan agar console rapi
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("=== TAHAP 3: FORECASTING & EVALUASI PROPHET ===\n")

# ==============================================================================
# 1. MEMUAT KOMPONEN MODEL DAN DATA UJI
# ==============================================================================
print("1. Memuat objek model Prophet dan Data Uji (Test Data)...")
with open('prophet_bca_model.json', 'r') as fin:
    model = model_from_json(json.load(fin))

test_data = pd.read_csv('prophet_test_data.csv')
test_data['ds'] = pd.to_datetime(test_data['ds'])

# Array harga aktual dan tanggal untuk kunci jawaban evaluasi
y_test = test_data['y'].values
test_dates = test_data['ds'].values


# ==============================================================================
# 2. MELAKUKAN PREDIKSI (FORECASTING)
# ==============================================================================
print("2. Prophet sedang meramal rentang waktu masa depan berdasarkan kalender...")
# Prophet butuh DataFrame dengan setidaknya kolom 'ds' (tanggal) untuk diprediksi
future = test_data[['ds']].copy()
forecast = model.predict(future)

# Kolom 'yhat' adalah prediksi harga utamanya (titik tengah peramalan)
predictions = forecast['yhat'].values


# ==============================================================================
# 3. MENGHITUNG METRIK EVALUASI INDUSTRI
# ==============================================================================
print("3. Mengalkulasi nilai akurasi dan metrik persentase kesalahan (MAPE)...")
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions) * 100

print("\n=======================================================")
print("=== HASIL EVALUASI MODEL FACEBOOK PROPHET ===")
print("=======================================================")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"Mean Absolute Error (MAE)      : {mae:.2f}")
print(f"Mean Absolute Percentage Error : {mape:.2f}%")
print("=======================================================")


# ==============================================================================
# 4. MENYIMPAN HASIL PREDIKSI KE CSV
# ==============================================================================
df_hasil = pd.DataFrame({
    'Tanggal': test_data['ds'],
    'Harga_Aktual': y_test,
    'Prediksi_Prophet': predictions,
    # Prophet menghasilkan Rentang Ketidakpastian (Upper & Lower Bound)
    'Batas_Bawah_Pasti': forecast['yhat_lower'].values, 
    'Batas_Atas_Pasti': forecast['yhat_upper'].values
})
df_hasil.set_index('Tanggal', inplace=True)
df_hasil.to_csv('tabel_hasil_prediksi_prophet.csv')
print("\n-> Data detail prediksi dan rentang ketidakpastian diekspor ke: 'tabel_hasil_prediksi_prophet.csv'")


# ==============================================================================
# 5. VISUALISASI GRAFIK KOMPARATIF (PITA KETIDAKPASTIAN PROPHET)
# ==============================================================================
print("4. Membuat lukisan grafik perbandingan Harga Aktual vs Prophet...")
plt.figure(figsize=(14, 7))

# Plot Harga Aktual (Hijau)
plt.plot(df_hasil.index, df_hasil['Harga_Aktual'], color='#1a9850', linewidth=2, label='Data Aktual (Test Data)')

# Plot Tebakan Utama Prophet (Biru Putus-putus)
plt.plot(df_hasil.index, df_hasil['Prediksi_Prophet'], color='#3182bd', linewidth=2, linestyle='--', label='Prediksi Prophet (yhat)')

# Plot Uncertainty Interval (Pita Biru Transparan khas Prophet)
plt.fill_between(df_hasil.index, df_hasil['Batas_Bawah_Pasti'], df_hasil['Batas_Atas_Pasti'], color='#3182bd', alpha=0.2, label='Uncertainty Interval (Rentang Wajar)')

plt.title(f'Hasil Peramalan Saham BCA Menggunakan Meta Prophet\n(Tingkat Kesalahan / MAPE: {mape:.2f}%)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Tanggal / Waktu', fontsize=12)
plt.ylabel('Harga Penutupan (IDR)', fontsize=12)
plt.legend(loc='lower left', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()

plot_filename = 'plot_prediksi_vs_aktual_prophet.png'
plt.savefig(plot_filename, dpi=300)
print(f"-> Grafik komparatif khas Prophet berhasil disimpan sebagai: {plot_filename}")

# Tampilkan di layar
plt.show()

print("\nTahap Evaluasi Akhir Prophet (Tahap 3) SELESAI!")
