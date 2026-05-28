import pandas as pd
import json
from prophet import Prophet
from prophet.serialize import model_to_json
import warnings
import logging

# Menyembunyikan peringatan/log bawaan dari Prophet dan cmdstanpy agar layar tetap bersih
warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
logging.getLogger('prophet').setLevel(logging.ERROR)

print("=== TAHAP 2: MEMBANGUN DAN MELATIH MODEL PROPHET ===\n")

# ==============================================================================
# 1. MEMUAT DATA LATIH
# ==============================================================================
print("1. Memuat Data Latih (Train Data) dalam format baku (ds, y)...")
train_data = pd.read_csv('prophet_train_data.csv')


# ==============================================================================
# 2. INISIALISASI ARSITEKTUR PROPHET & HARI LIBUR
# ==============================================================================
print("2. Membangun model algoritma Facebook Prophet...")

# Mematikan daily_seasonality karena data kita berwujud harian, bukan per-jam/menit.
# Menghidupkan yearly_seasonality karena saham sering memiliki siklus tahunan 
# (seperti Window Dressing di bulan Desember atau pembagian Dividen di kuartal 1-2).
model = Prophet(daily_seasonality=False, yearly_seasonality=True)

# INI ADALAH FITUR UNGGULAN: Menyuntikkan Kalender Libur Nasional Indonesia (ID)
# Prophet akan otomatis mengunduh dan mencatat tanggal Idul Fitri, Kemerdekaan, dsb.
model.add_country_holidays(country_name='ID')
print("   -> [AKTIF] Kalender Libur Nasional Indonesia (ID) telah disuntikkan ke otak model.")


# ==============================================================================
# 3. PELATIHAN MODEL (FITTING)
# ==============================================================================
print("\n3. Memulai proses pelatihan (Fitting) ke Data Latih...")
print("   (Prophet sedang membedah trend, musiman tahunan, dan hari libur secara bersamaan...)")

# Model menelan data dan menyesuaikan kurva regresinya
model.fit(train_data)

# Mengintip daftar hari libur yang berhasil dideteksi oleh Prophet
print("\n=== RINGKASAN HARI LIBUR NASIONAL YANG TERDETEKSI ===")
print("Contoh 5 hari libur yang akan diperhatikan oleh model:")
for holiday_name in model.train_holiday_names.head(5).tolist():
    print(f" - {holiday_name}")


# ==============================================================================
# 4. EKSPOR MODEL (SERIALIZATION)
# ==============================================================================
print("\n4. Menyimpan arsitektur model dan bobot (Serialisasi format JSON)...")
model_filename = 'prophet_bca_model.json'

# Prophet menyarankan format JSON untuk penyimpanan modelnya
with open(model_filename, 'w') as fout:
    json.dump(model_to_json(model), fout)

print(f"-> Objek Model Prophet berhasil diekspor ke: {model_filename}")
print("Tahap Pelatihan Prophet (Tahap 2) SELESAI!")
