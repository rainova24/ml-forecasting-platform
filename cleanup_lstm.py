import os
import shutil

# Folder saat ini
current_dir = os.getcwd()

# Membuat folder baru khusus LSTM
lstm_dir = os.path.join(current_dir, 'Proyek_LSTM')
if not os.path.exists(lstm_dir):
    os.makedirs(lstm_dir)
    print(f"Folder '{lstm_dir}' berhasil dibuat.")

# Daftar file yang berkaitan murni dengan proyek LSTM
# File dataset asli dibiarkan tetap di root agar bisa dipakai oleh Holt-Winters
files_to_move = [
    "1_lstm_preprocessing.py", 
    "2_lstm_training.py", 
    "3_lstm_evaluation.py",
    "lstm_scaler.pkl", 
    "lstm_processed_data.pkl", 
    "lstm_bca_model.keras",
    "plot_lstm_loss.png", 
    "plot_prediksi_vs_aktual_lstm.png", 
    "tabel_hasil_prediksi_lstm.csv"
]

# Memindahkan file-file ke dalam folder Proyek_LSTM
berhasil_dipindah = 0
for file in files_to_move:
    src_path = os.path.join(current_dir, file)
    dst_path = os.path.join(lstm_dir, file)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        berhasil_dipindah += 1

print(f"Selesai! {berhasil_dipindah} file terkait LSTM telah dirapikan ke dalam folder 'Proyek_LSTM'.")
print("Workspace Anda kini bersih dan siap untuk menyambut algoritma Holt-Winters!")
