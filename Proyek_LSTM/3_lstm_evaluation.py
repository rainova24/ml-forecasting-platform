from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("=== TAHAP 3: FORECASTING & EVALUASI LSTM ===\n")

# ==============================================================================
# 1. MEMUAT KOMPONEN MODEL DAN DATA UJI
# ==============================================================================
print("1. Memuat model LSTM, objek Scaler, dan Data Uji (Test Data)...")
model = load_model('lstm_bca_model.keras')

with open('lstm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('lstm_processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Matriks 3 Dimensi untuk ujian model
x_test = data['x_test']

# Harga aktual (asli) untuk kunci jawaban
y_test = data['y_test'] 
test_dates = data['test_dates']


# ==============================================================================
# 2. MELAKUKAN PREDIKSI (FORECASTING)
# ==============================================================================
print("\n2. Meminta model memprediksi masa depan berdasarkan Data Uji...")
# Hasil tebakan model masih berupa angka desimal 0-1 (karena data inputnya di-scale)
predictions_scaled = model.predict(x_test)

# Inverse Transform: Mengubah kembali angka desimal (0-1) menjadi wujud harga Rupiah
predictions = scaler.inverse_transform(predictions_scaled)


# ==============================================================================
# 3. MENGHITUNG METRIK EVALUASI (KOMPARASI DENGAN ARIMA)
# ==============================================================================
print("3. Menghitung tingkat akurasi dan metrik error industri...")
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions) * 100

print("\n=======================================================")
print("=== HASIL EVALUASI MODEL DEEP LEARNING (LSTM) ===")
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
    'Harga_Aktual': y_test.flatten(),
    'Prediksi_LSTM': predictions.flatten()
})
df_hasil.set_index('Tanggal', inplace=True)
df_hasil.to_csv('tabel_hasil_prediksi_lstm.csv')
print("\n-> Data detail prediksi berhasil disimpan ke: 'tabel_hasil_prediksi_lstm.csv'")


# ==============================================================================
# 5. VISUALISASI GRAFIK KOMPARATIF (AKTUAL VS PREDIKSI)
# ==============================================================================
print("4. Membuat visualisasi plot Harga Aktual vs Prediksi LSTM...")
plt.figure(figsize=(14, 7))

# Plot Garis Aktual (Hijau)
plt.plot(df_hasil.index, df_hasil['Harga_Aktual'], color='#1a9850', linewidth=2, label='Data Aktual (Test Data)')

# Plot Garis Prediksi LSTM (Merah Putus-putus)
plt.plot(df_hasil.index, df_hasil['Prediksi_LSTM'], color='#d73027', linewidth=2, linestyle='--', label='Prediksi LSTM')

plt.title(f'Hasil Peramalan Harga Saham BCA Menggunakan LSTM\n(Tingkat Kesalahan / MAPE: {mape:.2f}%)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Tanggal / Waktu', fontsize=12)
plt.ylabel('Harga Penutupan (IDR)', fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()

# Simpan grafik
plot_filename = 'plot_prediksi_vs_aktual_lstm.png'
plt.savefig(plot_filename, dpi=300)
print(f"-> Grafik komparatif berhasil disimpan sebagai: {plot_filename}")

# Tampilkan di layar
plt.show()

print("\nTahap Evaluasi Akhir LSTM (Tahap 3) SELESAI!")
