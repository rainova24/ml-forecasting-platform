import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Mengabaikan warning selama proses fitting
warnings.filterwarnings('ignore')

print("=== SCRIPT PENYIMPANAN MODEL (DEPLOYMENT PREP) ===")

# 1. MEMUAT DATASET TERBAIK
# Untuk sistem production di dunia nyata, model harus dilatih menggunakan seluruh 
# data yang tersedia agar ia mendapatkan update tren paling akhir.
file_data = 'dataset_bca_siap_uji.csv'
print(f"Memuat seluruh dataset dari {file_data}...")
df = pd.read_csv(file_data, parse_dates=['Date'], index_col='Date')

# Memastikan data urut berdasarkan waktu
df.sort_index(inplace=True)
y = df['Close']

# 2. MELATIH MODEL (FITTING)
# Menggunakan kombinasi parameter terbaik dari evaluasi sebelumnya: ARIMA(0, 1, 1)
print("Melatih model ARIMA(0, 1, 1) menggunakan seluruh data...")
model = ARIMA(y, order=(0, 1, 1))
model_fit = model.fit()

# 3. MENYIMPAN OBJEK MODEL
# Menyimpan model yang sudah pintar ini ke dalam file berformat .pkl (Pickle)
model_filename = 'arima_bca_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model_fit, file)
print(f"✅ Model ARIMA berhasil diekspor ke: {model_filename}")

# 4. MENYIMPAN DATA HISTORIS UNTUK GRAFIK UI
# Mengambil 100 hari terakhir dari data aktual sebagai titik awal grafik di Streamlit
history_data = y.tail(100)
history_filename = 'history_bca.pkl'
with open(history_filename, 'wb') as file:
    pickle.dump(history_data, file)
print(f"✅ Data historis (100 hari) berhasil diekspor ke: {history_filename}")

print("\nTahap ekspor selesai. Anda siap beralih ke Streamlit!")
