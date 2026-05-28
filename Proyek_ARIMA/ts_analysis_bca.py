import pandas as pd
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Memuat dataset dari tahap sebelumnya
input_filename = 'dataset_bca_siap_uji.csv'
print(f"Memuat data dari {input_filename}...\n")

# Memastikan kolom Date di-parsing menjadi datetime dan dijadikan sebagai index
df = pd.read_csv(input_filename, parse_dates=['Date'], index_col='Date')

# Memastikan baris data terurut berdasarkan tanggal (krusial untuk data time series)
df.sort_index(inplace=True)

# ==============================================================================
# 1. VISUALISASI TREN HARGA INTERAKTIF
# ==============================================================================
print("[1/3] Memproses visualisasi interaktif dengan Plotly...")

# Membuat line plot menggunakan plotly.express
fig = px.line(df, x=df.index, y='Close', 
              title='Pergerakan Harga Tutup Saham BBCA (Interaktif)',
              labels={'Date': 'Waktu (Tahun)', 'Close': 'Harga dalam Rupiah (IDR)'})

# Menambahkan fitur range slider di sumbu X untuk memudahkan zoom-in/zoom-out
fig.update_xaxes(rangeslider_visible=True)

# Menyimpan plot interaktif ke file HTML agar bisa dibuka di browser manapun
html_filename = 'plot_interaktif_bbca.html'
fig.write_html(html_filename)
print(f"  -> File interaktif berhasil disimpan sebagai: {html_filename}")

# Menampilkan visualisasi secara langsung (jika lingkungan eksekusi mendukung)
# fig.show()  # <-- DINONAKTIFKAN agar terminal tidak kebanjiran kode JS


# ==============================================================================
# 2. UJI STASIONERITAS (AUGMENTED DICKEY-FULLER / ADF TEST)
# ==============================================================================
print("\n[2/3] Melakukan Uji Stasioneritas (ADF Test) pada kolom 'Close'...")

# Eksekusi ADF test, pastikan membuang NaN (meski kita sudah bersihkan sebelumnya)
adf_result = adfuller(df['Close'].dropna())

# Mengekstrak masing-masing nilai dari output adfuller
adf_statistic = adf_result[0]
p_value = adf_result[1]
critical_values = adf_result[4]

# Menampilkan laporan ke terminal
print("-" * 45)
print("Hasil Uji Augmented Dickey-Fuller (ADF):")
print(f"ADF Statistic   : {adf_statistic:.4f}")
print(f"p-value         : {p_value:.4f}")
print("Critical Values :")
for key, value in critical_values.items():
    print(f"   {key} : {value:.4f}")
print("-" * 45)

# Logika pengambilan kesimpulan berdasarkan nilai p-value (alpha = 0.05)
if p_value < 0.05:
    print("Kesimpulan: Data Stasioner (Tolak H0)")
else:
    print("Kesimpulan: Data Tidak Stasioner (Gagal Tolak H0)")


# ==============================================================================
# 3. DEKOMPOSISI TIME SERIES
# ==============================================================================
print("\n[3/3] Melakukan Dekomposisi Time Series...")

# Menggunakan period=252 yang merepresentasikan asumsi jumlah hari bursa aktif dalam satu tahun
# Model additive digunakan karena variasi seasonal cenderung konstan, tidak eksponensial.
decomposition = seasonal_decompose(df['Close'].dropna(), model='additive', period=252)

# Mengatur ukuran plot matplotlib menjadi lebih besar agar keempat subplot terlihat jelas
plt.rcParams.update({'figure.figsize': (14, 10)})

# Melakukan plotting
fig_decomp = decomposition.plot()
fig_decomp.suptitle('Dekomposisi Time Series Harga Tutup Saham BBCA', fontsize=18, fontweight='bold', y=1.02)

# Menyimpan plot ke dalam file gambar
decomp_filename = 'dekomposisi_bbca.png'
plt.savefig(decomp_filename, bbox_inches='tight', dpi=300)
print(f"  -> Plot dekomposisi disimpan sebagai: {decomp_filename}")

# Menampilkan grafik ke layar
plt.show()

print("\nSemua proses analisis Time Series selesai!")
