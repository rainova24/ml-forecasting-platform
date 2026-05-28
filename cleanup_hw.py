import os
import shutil

# Folder saat ini
current_dir = os.getcwd()

# Membuat folder baru khusus Holt-Winters
hw_dir = os.path.join(current_dir, 'Proyek_HoltWinters')
if not os.path.exists(hw_dir):
    os.makedirs(hw_dir)
    print(f"Folder '{hw_dir}' berhasil dibuat.")

# Daftar file yang berkaitan murni dengan proyek Holt-Winters
files_to_move = [
    "1_hw_preprocessing.py", 
    "2_hw_training.py", 
    "3_hw_evaluation.py",
    "hw_processed_data.pkl", 
    "hw_bca_model.pkl",
    "plot_prediksi_vs_aktual_hw.png", 
    "tabel_hasil_prediksi_hw.csv"
]

# Memindahkan file-file
berhasil_dipindah = 0
for file in files_to_move:
    src_path = os.path.join(current_dir, file)
    dst_path = os.path.join(hw_dir, file)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        berhasil_dipindah += 1

print(f"Selesai! {berhasil_dipindah} file terkait Holt-Winters telah dirapikan ke dalam folder 'Proyek_HoltWinters'.")
