import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings

# Mengabaikan peringatan internal
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("=== TAHAP 2: MEMBANGUN DAN MELATIH MODEL LSTM ===\n")

# ==============================================================================
# 1. MEMUAT MATRIKS DATA DARI TAHAP 1
# ==============================================================================
print("1. Memuat matriks 3 Dimensi yang sudah disiapkan...")
with open('lstm_processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

x_train = data['x_train']
y_train = data['y_train']
# Perhatikan bahwa kita SAMA SEKALI tidak menyentuh x_test dan y_test di script ini 
# untuk menjaga keutuhan (menghindari data leakage) saat evaluasi nanti.


# ==============================================================================
# 2. MEMBANGUN ARSITEKTUR DEEP LEARNING
# ==============================================================================
print("\n2. Membangun arsitektur Jaringan Saraf Tiruan (LSTM)...")
model = Sequential()

# Layer 1: LSTM (Dengan 50 neuron/unit seluler)
# return_sequences=True wajib dipakai jika layer selanjutnya juga merupakan layer LSTM
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# Dropout 20%: Sengaja 'mematikan' 20% neuron secara acak untuk mencegah model menghafal jawaban (Overfitting)
model.add(Dropout(0.2)) 

# Layer 2: LSTM (Dengan 50 neuron lagi)
# return_sequences=False karena setelah ini kita akan masuk ke layer biasa (Dense), bukan LSTM lagi
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Layer 3: Dense (Layer klasikal untuk menggabungkan inti informasi dari LSTM)
model.add(Dense(units=25))

# Layer 4: Output (Hanya 1 neuron karena kita butuh 1 angka spesifik yaitu tebakan harga esok harinya)
model.add(Dense(units=1))


# ==============================================================================
# 3. KOMPILASI MODEL
# ==============================================================================
# Menggunakan optimizer 'Adam' (Standar industri AI modern yang pintar mengatur kecepatan belajar)
# Menggunakan 'mean_squared_error' sebagai target kerugian (Loss) yang harus diminimalkan
model.compile(optimizer='adam', loss='mean_squared_error')
print("\n=== Ringkasan Arsitektur Model (Summary) ===")
model.summary()


# ==============================================================================
# 4. MEMULAI PROSES PELATIHAN (TRAINING)
# ==============================================================================
print("\n3. Memulai proses belajar model (Training). Ini akan memakan waktu sejenak...")
# batch_size=32 : Model mencerna 32 data historis sekaligus sebelum memperbarui parameternya
# epochs=50     : Model akan mengulang membaca seluruh buku materi latihannya sebanyak 50 kali
# validation_split=0.1 : Mengambil 10% bagian belakang dari Train Data untuk ujian pantauan Loss
history = model.fit(x_train, y_train, batch_size=32, epochs=50, validation_split=0.1)


# ==============================================================================
# 5. VISUALISASI KURVA PEMBELAJARAN (LEARNING CURVE)
# ==============================================================================
print("\n4. Membuat visualisasi Kurva Pembelajaran (Loss Curve)...")
plt.figure(figsize=(10, 6))

# Mem-plot pergerakan tingkat error (Loss) selama 50 kali putaran (Epoch)
plt.plot(history.history['loss'], color='dodgerblue', linewidth=2, label='Training Loss (Data Latih)')
plt.plot(history.history['val_loss'], color='crimson', linewidth=2, label='Validation Loss (Data Validasi Pantauan)')

plt.title('Kurva Pembelajaran (Loss) - Model LSTM', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Jumlah Putaran Belajar (Epoch)', fontsize=12, labelpad=10)
plt.ylabel('Tingkat Error (Mean Squared Error)', fontsize=12, labelpad=10)
plt.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
plt.tight_layout()

# Menyimpan grafik kurva agar bisa dilampirkan ke Laporan Tugas Besar
plot_filename = 'plot_lstm_loss.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"-> Grafik Learning Curve berhasil disimpan sebagai: {plot_filename}")

# Tampilkan di layar
plt.show()


# ==============================================================================
# 6. EKSPOR MODEL (SAVE)
# ==============================================================================
# Menyimpan otak buatan (Network) yang sudah cerdas ini ke dalam format '.keras'
model_filename = 'lstm_bca_model.keras'
model.save(model_filename)
print(f"-> Otak AI model LSTM berhasil diekspor ke: {model_filename}")

print("\nTahap Pelatihan Model LSTM (Tahap 2) SELESAI!")
