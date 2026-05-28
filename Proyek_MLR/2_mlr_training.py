import pickle
from sklearn.linear_model import LinearRegression

print("=== TAHAP 2: PELATIHAN MODEL MULTIVARIATE (MLR) ===\n")

# ==============================================================================
# 1. MEMUAT DATA LATIH (X DAN Y)
# ==============================================================================
print("1. Memuat matriks 5 Dimensi (Data Latih/X_Train)...")
with open('mlr_processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = data['X_train']
Y_train = data['Y_train']


# ==============================================================================
# 2. INISIALISASI & PELATIHAN ALGORITMA (FITTING)
# ==============================================================================
print("2. Menginisialisasi algoritma Multiple Linear Regression...")
model = LinearRegression()

print("3. Memulai komputasi matematika regresi (Mencari korelasi antar variabel)...")
# Algoritma sedang mencari persamaan garis lurus yang melewati 5 dimensi fitur secara optimal
model.fit(X_train, Y_train)


# ==============================================================================
# 3. BEDAH BOBOT KOEFISIEN MATEMATIS (UNTUK LAPORAN)
# ==============================================================================
print("\n=== PERSAMAAN REGRESI BERGANDA YANG DITEMUKAN ===")
print("Rumus Regresi : Y = (X1*B1) + (X2*B2) + (X3*B3) + (X4*B4) + (X5*B5) + Konstanta (Intercept)\n")

# Nama-nama kolom sesuai urutan saat kita memisahkan kolom di Tahap 1
kolom_fitur = ['Open_X', 'High_X', 'Low_X', 'Close_X', 'Volume_X']
bobot_koefisien = model.coef_

# Mencetak pengaruh masing-masing kolom terhadap harga penutupan hari ini
print("[Bobot Korelasi / Pengaruh]")
for nama, bobot in zip(kolom_fitur, bobot_koefisien):
    # Jika bobot positif, berarti variabel tsb ikut mendorong harga naik
    # Jika bobot negatif, berarti variabel tsb menekan harga turun
    print(f"Koefisien {nama:<10} : {bobot:10.5f}")

print("-" * 35)
print(f"Nilai Konstanta (Intercept): {model.intercept_:10.5f}")


# ==============================================================================
# 4. EKSPOR MODEL
# ==============================================================================
model_filename = 'mlr_bca_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

print(f"\n-> Objek Model Regresi Linear Berganda berhasil diekspor ke: {model_filename}")
print("Tahap Pelatihan MLR (Tahap 2) SELESAI!")
