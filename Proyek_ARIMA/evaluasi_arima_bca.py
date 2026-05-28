import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# ==============================================================================
# 1. MEMUAT DATA HASIL PREDIKSI
# ==============================================================================
input_filename = 'tabel_hasil_prediksi_arima.csv'
print(f"Memuat data hasil prediksi dari {input_filename}...\n")

# Membaca file CSV yang disimpan pada tahap forecasting sebelumnya
df_hasil = pd.read_csv(input_filename, index_col='Date')

# Mengekstrak kolom data aktual dan prediksi untuk dihitung error-nya
aktual = df_hasil['Aktual_Test']
prediksi = df_hasil['Prediksi_ARIMA']


# ==============================================================================
# 2. PERHITUNGAN METRIK EVALUASI
# ==============================================================================
print("Mengkalkulasi metrik evaluasi performa model...")

# a) Root Mean Squared Error (RMSE)
# Menghitung rata-rata kuadrat error, lalu diakarkan (sqrt) agar kembali ke satuan awal (Rupiah)
# Sangat sensitif terhadap nilai error yang besar/outlier
rmse = np.sqrt(mean_squared_error(aktual, prediksi))

# b) Mean Absolute Error (MAE)
# Menghitung rata-rata selisih absolut (tanpa melihat minus/plus) antara prediksi dan aktual
# Menggambarkan rata-rata selisih harga dalam satuan Rupiah
mae = mean_absolute_error(aktual, prediksi)

# c) Mean Absolute Percentage Error (MAPE)
# Menghitung persentase error. Dikalikan 100 agar outputnya menjadi persentase murni (%)
mape = mean_absolute_percentage_error(aktual, prediksi) * 100


# ==============================================================================
# 3. FORMAT OUTPUT UNTUK LAPORAN
# ==============================================================================
# Menyusun ketiga metrik tersebut ke dalam Pandas DataFrame sederhana agar rapi
df_metrik = pd.DataFrame({
    'Metrik': [
        'Root Mean Squared Error (RMSE)', 
        'Mean Absolute Error (MAE)', 
        'Mean Absolute Percentage Error (MAPE)'
    ],
    # Memformat angka: .2f artinya 2 angka di belakang koma. MAPE ditambah simbol %
    'Nilai': [
        f"{rmse:.2f}", 
        f"{mae:.2f}", 
        f"{mape:.2f}%"
    ]
})

print("\n=== TABEL EVALUASI PERFORMA MODEL ARIMA ===")
# Cetak tabel ke terminal, to_string(index=False) digunakan untuk menyembunyikan angka index (0, 1, 2)
print(df_metrik.to_string(index=False))

# Garis pemisah untuk kerapian
print("\n" + "=" * 55)

# Interpretasi otomatis yang siap disalin ke dalam paragraf laporan
print(f"Interpretasi: Rata-rata prediksi model ARIMA meleset sebesar {mape:.2f}% dari harga aslinya di dunia nyata.")

print("=" * 55)
print("\nTahap evaluasi kuantitatif selesai!")
