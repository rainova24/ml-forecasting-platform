import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Mengabaikan warning yang mungkin muncul dari perhitungan statsmodels agar terminal tetap rapi
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. PERSIAPAN DATA LATIH & SETUP PARAMETER
# ==============================================================================
input_filename = 'dataset_bca_train.csv'
print(f"Memuat data latih dari {input_filename}...\n")

# Memuat data dan memastikan index berbentuk waktu (DatetimeIndex)
df_train = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')
df_train.sort_index(inplace=True)

# Memisahkan seri waktu utama yang akan dipelajari oleh model
y_train = df_train['Close']

# ------------------------------------------------------------------------------
# [PENTING] Masukkan parameter optimal (p, d, q) hasil dari Auto-ARIMA di sini!
# ------------------------------------------------------------------------------
# Silakan ubah angka di bawah ini sesuai dengan Best Model yang tercetak sebelumnya.
# Sebagai contoh default (placeholder), saya isi dengan 1, 1, 1.
p = 0  
d = 1  
q = 1  

print(f"Mengonfigurasi model ARIMA dengan orde: AR(p)={p}, Diff(d)={d}, MA(q)={q}...\n")


# ==============================================================================
# 2. PROSES PELATIHAN (FITTING) MODEL
# ==============================================================================
print("Sedang melatih (fitting) model ARIMA ke data latih...")

# Inisialisasi model ARIMA dari statsmodels
model = ARIMA(y_train, order=(p, d, q))

# Mengeksekusi proses pelatihan (fitting)
model_fit = model.fit()

# Mencetak ringkasan evaluasi statistik model ke terminal
print("\n=== RINGKASAN MODEL ARIMA ===")
print(model_fit.summary())


# ==============================================================================
# 3. VISUALISASI EVALUASI DATA LATIH (ACTUAL VS FITTED VALUES)
# ==============================================================================
print("\nMenyiapkan visualisasi interaktif Actual vs Fitted Values...")

# Ekstraksi nilai prediksi in-sample (Fitted Values) yang dihasilkan model terhadap data latih
fitted_values = model_fit.fittedvalues

# Membuat objek figure menggunakan plotly
fig = go.Figure()

# Plot Garis 1: Data Latih Aktual (Biru)
fig.add_trace(go.Scatter(
    x=y_train.index, 
    y=y_train.values,
    mode='lines',
    name='Data Latih Aktual',
    line=dict(color='dodgerblue', width=2)
))

# Plot Garis 2: Fitted Values / Hasil 'Belajar' Model (Merah/Oranye, Dashed)
# Kita mulai plot dari observasi ke-(d) karena baris pertama biasanya bernilai 0 akibat proses differencing
fig.add_trace(go.Scatter(
    x=fitted_values.index[d:], 
    y=fitted_values.values[d:],
    mode='lines',
    name='Fitted Values (Prediksi Model)',
    line=dict(color='tomato', width=2, dash='dash')
))

# Menyempurnakan layout, judul, dan label sumbu
fig.update_layout(
    title=dict(
        text='Perbandingan Data Latih Aktual vs Fitted Values ARIMA',
        font=dict(size=20, color='black'),
        x=0.5, # Mengatur posisi judul di tengah
        xanchor='center'
    ),
    xaxis_title='Tahun',
    yaxis_title='Harga dalam Rupiah (IDR)',
    legend=dict(
        yanchor="top", y=0.99, 
        xanchor="left", x=0.01,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="lightgrey", borderwidth=1
    ),
    hovermode='x unified', # Menampilkan info kedua garis secara vertikal saat kursor diarahkan
    plot_bgcolor='white',
    paper_bgcolor='white',
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    yaxis=dict(showgrid=True, gridcolor='lightgrey')
)

# Menyimpan plot ke file HTML (sangat berguna untuk laporan berbasis web atau lampiran interaktif)
html_filename = 'plot_fitted_arima.html'
fig.write_html(html_filename)
print(f"Visualisasi interaktif telah berhasil disimpan ke: {html_filename}")

# ==============================================================================
# EKSEKUSI PENAMPILAN GRAFIK
# ==============================================================================
# HAPUS TANDA KUTIP (Comment) di bawah ini jika Anda ingin grafiknya langsung terbuka di browser
# fig.show()

print("\nTahap pelatihan model ARIMA selesai!")
