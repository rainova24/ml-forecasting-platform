import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

# Mengabaikan warning tampilan agar console rapi
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("=== TAHAP 3: FORECASTING & EVALUASI MLR ===\n")

# ==============================================================================
# 1. MEMUAT KOMPONEN MODEL DAN DATA UJI
# ==============================================================================
print("1. Memuat objek model regresi dan Matriks Uji (X_Test)...")
with open('mlr_bca_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('mlr_processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_test = data['X_test']
Y_test = data['Y_test']
test_dates = data['test_dates']


# ==============================================================================
# 2. MELAKUKAN PREDIKSI (FORECASTING)
# ==============================================================================
print("2. Model menebak harga penutupan berdasarkan 5 fitur H-1 (Multivariate)...")
# Alih-alih menebak dari udara kosong, MLR memegang matriks X_test
# lalu mengalikannya dengan rumus yang ditemukan di Tahap 2
predictions = model.predict(X_test)


# ==============================================================================
# 3. MENGHITUNG METRIK EVALUASI INDUSTRI
# ==============================================================================
print("3. Mengalkulasi persentase kemelesetan tebakan (MAPE, RMSE, MAE)...")
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
mae = mean_absolute_error(Y_test, predictions)
mape = mean_absolute_percentage_error(Y_test, predictions) * 100

print("\n=======================================================")
print("=== HASIL EVALUASI MODEL REGRESI LINEAR BERGANDA ===")
print("=======================================================")
print(f"Root Mean Squared Error (RMSE) : {rmse:.2f}")
print(f"Mean Absolute Error (MAE)      : {mae:.2f}")
print(f"Mean Absolute Percentage Error : {mape:.2f}%")
print("=======================================================")


# ==============================================================================
# 4. MENYIMPAN HASIL PREDIKSI KE CSV
# ==============================================================================
df_hasil = pd.DataFrame({
    'Tanggal': test_dates,
    'Harga_Aktual': Y_test,
    'Prediksi_MLR': predictions
})
df_hasil.set_index('Tanggal', inplace=True)
df_hasil.to_csv('tabel_hasil_prediksi_mlr.csv')
print("\n-> Data detail prediksi berhasil diekspor ke: 'tabel_hasil_prediksi_mlr.csv'")


# ==============================================================================
# 5. VISUALISASI GRAFIK KOMPARATIF (AKTUAL VS MLR)
# ==============================================================================
print("4. Membuat lukisan grafik perbandingan Harga Aktual vs Prediksi MLR...")
plt.figure(figsize=(14, 7))

# Plot Harga Aktual (Hijau)
plt.plot(df_hasil.index, df_hasil['Harga_Aktual'], color='#1a9850', linewidth=2, label='Data Aktual (Test Data)')

# Plot Prediksi MLR (Ungu Putus-putus)
plt.plot(df_hasil.index, df_hasil['Prediksi_MLR'], color='#984ea3', linewidth=2, linestyle='--', label='Prediksi Regresi Berganda (MLR)')

plt.title(f'Hasil Peramalan Harga Saham BCA Menggunakan Regresi Linear Berganda\n(Tingkat Kesalahan / MAPE: {mape:.2f}%)', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Tanggal / Waktu', fontsize=12)
plt.ylabel('Harga Penutupan (IDR)', fontsize=12)
plt.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

plt.tight_layout()

plot_filename = 'plot_prediksi_vs_aktual_mlr.png'
plt.savefig(plot_filename, dpi=300)
print(f"-> Grafik komparatif berhasil disimpan sebagai: {plot_filename}")

# Tampilkan di layar
plt.show()

print("\nTahap Evaluasi Akhir Regresi Berganda (Tahap 3) SELESAI!")
