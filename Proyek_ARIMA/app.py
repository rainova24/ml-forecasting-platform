import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. KONFIGURASI HALAMAN & DESKRIPSI
# ==============================================================================
# Mengatur layout aplikasi agar melebar penuh (wide) dan memberi judul tab browser
st.set_page_config(layout="wide", page_title="Prediksi Saham BCA")

# Judul utama di bagian atas antarmuka
st.title("📈 Dashboard Prediksi Harga Saham BCA (BBCA.JK) dengan Algoritma ARIMA")

# Membuat area expander (bisa di-klik untuk buka/tutup) berisi latar belakang proyek
with st.expander("ℹ️ Tentang Proyek & Dataset", expanded=True):
    st.write("""
    Aplikasi web ini merupakan implementasi Tugas Besar Machine Learning.
    Model kecerdasan buatan di balik aplikasi ini dibangun menggunakan algoritma **ARIMA (0, 1, 1)** 
    yang terbukti memiliki performa *Good Forecasting* (MAPE ~19.9%) pada pengujian *out-of-sample*.
    
    Data historis ditarik secara langsung dari Bursa Efek Indonesia (melalui Yahoo Finance). 
    Model akan memproyeksikan pergerakan harga di masa depan berdasarkan pola waktu yang telah dipelajarinya.
    """)

st.markdown("---")


# ==============================================================================
# 2. INPUT PENGGUNA (INTERAKTIVITAS)
# ==============================================================================
# Membuat sidebar di sebelah kiri untuk meletakkan alat kontrol pengguna
st.sidebar.header("Konfigurasi Prediksi")

# Slider interaktif untuk memilih rentang hari prediksi (1 sampai 30 hari)
hari_prediksi = st.sidebar.slider(
    "Pilih periode masa depan yang ingin diprediksi (Hari):", 
    min_value=1, max_value=30, value=7
)

# Tombol pemicu eksekusi prediksi
tombol_prediksi = st.sidebar.button("🚀 Mulai Prediksi", use_container_width=True)

st.sidebar.markdown("<br><br><small>Dibuat untuk Tugas Besar Machine Learning</small>", unsafe_allow_html=True)


# ==============================================================================
# 3. LOGIKA PREDIKSI & LOAD MODEL
# ==============================================================================
# Logika ini HANYA akan berjalan apabila pengguna menekan tombol "Mulai Prediksi"
if tombol_prediksi:
    try:
        # Mencoba memuat (load) model dan histori data dari file pickle
        with open('arima_bca_model.pkl', 'rb') as f:
            model_fit = pickle.load(f)
            
        with open('history_bca.pkl', 'rb') as f:
            history_data = pickle.load(f)
            
        # Menampilkan indikator loading berputar saat model sedang berpikir
        with st.spinner('Model sedang melakukan kalkulasi prediksi tingkat tinggi...'):
            
            # Melakukan peramalan (forecasting) menggunakan objek model ARIMA
            forecast = model_fit.get_forecast(steps=hari_prediksi)
            
            # Mengekstrak harga rata-rata dan batas 95% tingkat kepercayaan
            pred_mean = forecast.predicted_mean
            conf_int = forecast.conf_int(alpha=0.05)
            
            # Mendapatkan tanggal terakhir dari data historis
            last_date = history_data.index[-1]
            
            # Mengenerate tanggal kalender untuk masa depan (menggunakan BDays / Business Days)
            # Ini sangat penting agar prediksi tidak jatuh di hari Sabtu/Minggu (hari bursa tutup)
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=hari_prediksi)
            
            # Mengganti index numeric hasil prediksi menjadi index datetime
            pred_mean.index = future_dates
            conf_int.index = future_dates
            batas_bawah = conf_int.iloc[:, 0]
            batas_atas = conf_int.iloc[:, 1]
            
            
            # ==============================================================================
            # 4. VISUALISASI INTERAKTIF (PLOTLY)
            # ==============================================================================
            st.subheader(f"📊 Hasil Prediksi {hari_prediksi} Hari Bursa Kedepan")
            
            # Inisiasi kanvas Plotly
            fig = go.Figure()
            
            # [PLOT 1] Garis Historis Asli (100 Hari Terakhir)
            fig.add_trace(go.Scatter(
                x=history_data.index, y=history_data.values,
                mode='lines+markers', name='Data Historis (100 Hari Terakhir)',
                line=dict(color='#1f77b4', width=2), # Biru tua
                marker=dict(size=4)
            ))
            
            # [TRIK VISUAL] Menyambungkan titik historis terakhir ke titik prediksi pertama
            # Ini dilakukan agar grafik di layar tidak terlihat terpotong di tengah-tengah
            sambungan_x = [history_data.index[-1], future_dates[0]]
            sambungan_y = [history_data.values[-1], pred_mean.values[0]]
            fig.add_trace(go.Scatter(
                x=sambungan_x, y=sambungan_y,
                mode='lines', showlegend=False,
                line=dict(color='#d62728', width=2, dash='dash')
            ))
            
            # [PLOT 2] Garis Prediksi Masa Depan
            fig.add_trace(go.Scatter(
                x=future_dates, y=pred_mean.values,
                mode='lines+markers', name='Prediksi ARIMA',
                line=dict(color='#d62728', width=2, dash='dash'), # Merah putus-putus
                marker=dict(size=6)
            ))
            
            # [PLOT 3] Area Arsiran Confidence Interval 95%
            # Logika fill pada plotly membutuhkan penggabungan batas atas dan batas bawah yang diputar mundur (reversed)
            x_area = list(future_dates) + list(future_dates)[::-1]
            y_area = list(batas_atas.values) + list(batas_bawah.values)[::-1]
            
            fig.add_trace(go.Scatter(
                x=x_area, y=y_area,
                fill='toself', fillcolor='rgba(128, 128, 128, 0.2)', # Abu-abu transparan
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip", showlegend=True, name='95% Confidence Interval'
            ))
            
            # Menyempurnakan layout, label, grid, dan posisi legend
            fig.update_layout(
                xaxis_title='Tanggal',
                yaxis_title='Harga Tutup (IDR)',
                hovermode='x unified', # Munculkan tooltip rapi saat mouse digerakkan
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor='white',
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
            
            # Merender grafik plotly ke dalam aplikasi Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            
            # ==============================================================================
            # 5. MENAMPILKAN TABEL DETAIL HARGA
            # ==============================================================================
            st.markdown("### 📋 Tabel Detail Perkiraan Harga")
            df_table = pd.DataFrame({
                'Prediksi Harga (IDR)': pred_mean,
                'Batas Bawah 95% (IDR)': batas_bawah,
                'Batas Atas 95% (IDR)': batas_atas
            })
            # Mengganti format index agar tidak menampilkan jam (00:00:00)
            df_table.index = df_table.index.strftime('%Y-%m-%d')
            # Memformat angka di tabel menjadi mata uang Rupiah
            st.dataframe(df_table.style.format("Rp {:,.2f}"), use_container_width=True)
            
    # Penanganan Error jika user langsung klik tombol sebelum menjalankan save_model.py
    except FileNotFoundError:
        st.error("⚠️ File sistem ('arima_bca_model.pkl' atau 'history_bca.pkl') tidak ditemukan di direktori Anda! Silakan buka terminal dan jalankan `python save_model.py` terlebih dahulu.")
else:
    # Teks yang muncul saat halaman pertama kali dibuka
    st.info("👈 Silakan atur konfigurasi di panel sebelah kiri dan klik **Mulai Prediksi** untuk melihat hasilnya.")
