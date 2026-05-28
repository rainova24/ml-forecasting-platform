import os
import shutil

# Mendapatkan direktori saat ini
current_dir = os.getcwd()

# Membuat folder baru khusus MLR
mlr_dir = os.path.join(current_dir, 'Proyek_MLR')
if not os.path.exists(mlr_dir):
    os.makedirs(mlr_dir)
    print(f"Folder '{mlr_dir}' berhasil dibuat.")

# Daftar file eksperimen Regresi Linear Berganda (MLR)
files_to_move = [
    "1_mlr_preprocessing.py", 
    "2_mlr_training.py", 
    "3_mlr_evaluation.py",
    "mlr_processed_data.pkl", 
    "mlr_bca_model.pkl",
    "plot_prediksi_vs_aktual_mlr.png", 
    "tabel_hasil_prediksi_mlr.csv"
]

# Memindahkan file-file ke folder baru
berhasil_dipindah = 0
for file in files_to_move:
    src_path = os.path.join(current_dir, file)
    dst_path = os.path.join(mlr_dir, file)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        berhasil_dipindah += 1

print(f"Beres! {berhasil_dipindah} file terkait MLR telah diringkas ke dalam folder 'Proyek_MLR'.")
