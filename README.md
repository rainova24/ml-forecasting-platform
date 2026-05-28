# Forecasting Experiment Platform (Full-Stack Web App)
Platform eksperimen Machine Learning untuk membandingkan 5 metode peramalan (Forecasting): LSTM, ARIMA, Holt-Winters, Prophet, dan MLR secara interaktif melalui antarmuka web.

## Persiapan Instalasi untuk Anggota Tim
Untuk menjalankan platform ini di komputer Anda masing-masing, Anda harus menyalakan 2 mesin secara bersamaan (Mesin Python/Backend dan Mesin Web/Frontend).

Pastikan Anda sudah menginstal **Node.js** dan **Python 3.10+**.

### 1. Menyalakan Backend (Mesin AI Python)
Buka terminal baru, arahkan ke folder proyek ini, dan jalankan perintah:
```bash
# Instal semua modul Machine Learning yang dibutuhkan
pip install -r requirements.txt

# Masuk ke folder backend
cd backend

# Nyalakan Server API
uvicorn main:app --reload
```
*(Jangan tutup terminal ini. Biarkan server berjalan di `http://127.0.0.1:8000`)*

### 2. Menyalakan Frontend (Antarmuka Web)
Buka terminal KEDUA, arahkan ke folder proyek ini, lalu jalankan:
```bash
# Masuk ke folder frontend
cd frontend

# Install seluruh pustaka UI (React, Axios, Recharts)
npm install

# Jalankan website
npm run dev
```
Buka link `http://localhost:5173/` yang muncul di terminal Anda melalui browser!

## Panduan Penggunaan Eksperimen
1. Upload file CSV dataset milik Anda sendiri.
2. Isi nama kolom yang menjadi Target (misal: `Close`, `Harga`, dll).
3. Isi nama kolom yang berisi Tanggal (misal: `Date`, `Tanggal`, dll).
4. Pilih algoritma peramalan yang ingin dicoba.
5. Jika memilih `MLR`, masukkan daftar fitur pendukung di pisahkan dengan koma (misal: `Open,High,Low,Volume`).
6. Klik **Jalankan Eksperimen** dan tunggu hingga grafik serta evaluasi metrik (MAPE, RMSE, MAE) muncul!
