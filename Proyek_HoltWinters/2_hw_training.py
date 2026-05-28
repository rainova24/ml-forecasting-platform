import pickle
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Mengabaikan peringatan internal dari statsmodels terkait optimasi
warnings.filterwarnings("ignore")

print("=== TAHAP 2: MEMBANGUN DAN MELATIH MODEL HOLT-WINTERS ===\n")

# ==============================================================================
# 1. MEMUAT DATA LATIH
# ==============================================================================
print("1. Memuat array harga (Data Latih)...")
with open('hw_processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

y_train = data['y_train']


# ==============================================================================
# 2. KONFIGURASI ARSITEKTUR HOLT-WINTERS
# ==============================================================================
print("2. Membangun model pemulusan eksponensial (Holt-Winters)...")
print("   - Trend: Additive (Penjumlahan linear)")
print("   - Seasonality: Additive")
print("   - Siklus Musiman (Seasonal Periods): 21 Hari (Siklus Bulanan Bursa)")

# Inisialisasi model
model = ExponentialSmoothing(
    y_train, 
    trend='add',       # Tren linier (tidak meledak eksponensial)
    seasonal='add',    # Komponen musiman linier
    seasonal_periods=21 # Pola berulang setiap 21 hari (1 bulan bursa)
)


# ==============================================================================
# 3. PELATIHAN MODEL (FITTING)
# ==============================================================================
print("\n3. Memulai proses pelatihan (mencari bobot matematis yang optimal)...")
# Algoritma akan melakukan optimasi iteratif untuk mencari nilai Alpha, Beta, dan Gamma
model_fit = model.fit(optimized=True)

# Mencetak bobot yang ditemukan oleh algoritma untuk dianalisis
print("\n=== RINGKASAN BOBOT PEMULUSAN (SMOOTHING PARAMETERS) ===")
# Alpha: Seberapa responsif model terhadap pergerakan harga terbaru (Makin besar = makin reaktif)
print(f"Alpha (Level) : {model_fit.params['smoothing_level']:.4f}")
# Beta: Seberapa kuat model mengikuti kemiringan tren jangka panjang
print(f"Beta (Trend)  : {model_fit.params['smoothing_trend']:.4f}")
# Gamma: Seberapa kaku model mengikuti siklus musimannya
print(f"Gamma (Musim) : {model_fit.params['smoothing_seasonal']:.4f}")


# ==============================================================================
# 4. EKSPOR MODEL
# ==============================================================================
model_filename = 'hw_bca_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_fit, f)

print(f"\n-> Objek Model Holt-Winters berhasil diekspor ke: {model_filename}")
print("Tahap Pelatihan Holt-Winters (Tahap 2) SELESAI!")
