import yfinance as yf
import pandas as pd
from datetime import datetime

# 1. Menentukan ticker saham dan rentang waktu (1 Januari 2019 hingga hari ini)
ticker_symbol = 'BBCA.JK'
start_date = '2019-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

print(f"Mengunduh data historis {ticker_symbol} dari {start_date} hingga {end_date}...")

# 2. Mengunduh data historis menggunakan yfinance
df = yf.download(ticker_symbol, start=start_date, end=end_date)

# Mengatasi penyesuaian jika library yfinance versi terbaru mengembalikan MultiIndex columns
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# 3. Memilih dan mengurutkan kolom sesuai spesifikasi
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# 4. Memastikan index Date diformat sebagai datetime murni tanpa timezone (tz-naive)
df.index = pd.to_datetime(df.index).tz_localize(None)
df.index.name = 'Date'

# 5. Pengecekan dan penanganan Missing Values
print("\nPengecekan Missing Values awal:")
print(df.isnull().sum())

# Mengisi missing values menggunakan Forward Fill (ffill) dilanjutkan Backward Fill (bfill)
df = df.ffill().bfill()

# 6. Menyimpan dataframe akhir ke dalam file CSV
csv_filename = 'dataset_bca_lengkap.csv'
df.to_csv(csv_filename)
print(f"\nData berhasil dibersihkan dan disimpan ke dalam file: {csv_filename}\n")

# 7. Verifikasi Dataset Akhir
print("=== 5 Baris Pertama Data ===")
print(df.head())

print("\n=== Informasi Dataset ===")
df.info()
