import os
import shutil

# Folder saat ini
current_dir = os.getcwd()

# Membuat folder baru khusus Prophet
prophet_dir = os.path.join(current_dir, 'Proyek_Prophet')
if not os.path.exists(prophet_dir):
    os.makedirs(prophet_dir)
    print(f"Folder '{prophet_dir}' berhasil dibuat.")

# Daftar file eksperimen Prophet
files_to_move = [
    "1_prophet_preprocessing.py", 
    "2_prophet_training.py", 
    "3_prophet_evaluation.py",
    "prophet_train_data.csv", 
    "prophet_test_data.csv", 
    "prophet_bca_model.json",
    "plot_prediksi_vs_aktual_prophet.png", 
    "tabel_hasil_prediksi_prophet.csv"
]

# Memindahkan file-file
berhasil_dipindah = 0
for file in files_to_move:
    src_path = os.path.join(current_dir, file)
    dst_path = os.path.join(prophet_dir, file)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        berhasil_dipindah += 1

print(f"Selesai! {berhasil_dipindah} file terkait Prophet telah dirapikan ke dalam folder 'Proyek_Prophet'.")
