import os
import shutil

# Folder saat ini
current_dir = os.getcwd()

# Membuat folder baru khusus ARIMA
arima_dir = os.path.join(current_dir, 'Proyek_ARIMA')
if not os.path.exists(arima_dir):
    os.makedirs(arima_dir)
    print(f"Folder '{arima_dir}' berhasil dibuat.")

# Daftar file yang berkaitan dengan proyek ARIMA
# Catatan: dataset_bca_siap_uji.csv, dataset_bca_lengkap.csv, dan dataset_bca_ml_features.csv 
# TETAP ditinggalkan di luar agar bisa dipakai untuk eksperimen ML lainnya (SVR, XGBoost, dll).
files_to_move = [
    "fetch_bca_data.py", "eda_bca_data.py", "ts_analysis_bca.py", "find_d_bca.py",
    "train_test_split_bca.py", "acf_pacf_bca.py", "auto_arima_bca.py", 
    "fit_arima_bca.py", "forecast_arima_bca.py", "evaluasi_arima_bca.py", 
    "high_res_plot_arima.py", "save_model.py", "app.py", "analisis_residual_bca.py",
    "dataset_bca_test.csv", "dataset_bca_train.csv", "dataset_prediksi_bca.csv",
    "tabel_hasil_prediksi_arima.csv", "arima_bca_model.pkl", "history_bca.pkl",
    "dekomposisi_bbca.png", "grafik_bbca_close.png", "plot_acf_pacf_bca.png", 
    "plot_diagnostics_arima.png", "plot_diagnostik_residual_manual.png", 
    "plot_differencing_d1.png", "plot_forecasting_arima.png", "plot_prediksi_vs_aktual.png", 
    "plot_train_test_split.png", "plot_fitted_arima.html", "plot_interaktif_bbca.html",
    "arima.ipynb"
]

# Memindahkan file-file ke dalam folder Proyek_ARIMA
berhasil_dipindah = 0
for file in files_to_move:
    src_path = os.path.join(current_dir, file)
    dst_path = os.path.join(arima_dir, file)
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        berhasil_dipindah += 1

print(f"Selesai! {berhasil_dipindah} file terkait ARIMA telah dirapikan ke dalam folder 'Proyek_ARIMA'.")
print("Data mentah dan feature engineering Anda tetap aman di folder utama.")
