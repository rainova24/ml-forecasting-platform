import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Membaca dataset
df = pd.read_csv('dataset_bca_siap_uji.csv')

# Menghitung korelasi khusus untuk kolom angka
kolom_angka = df.select_dtypes(include=['number'])
korelasi = kolom_angka.corr()

print("=== MATRIKS KORELASI DATASET BCA ===")
print(korelasi.to_string())
print("====================================")

# Membuat visualisasi Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(korelasi, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title('Heatmap Korelasi Matriks - Saham BCA')
plt.tight_layout()

# Menyimpan gambar
plt.savefig('korelasi_bca.png')
print("\nVisualisasi Heatmap telah disimpan sebagai 'korelasi_bca.png'.")
plt.show()
