# Mall Customer Segmentation — Kelompok 3 SPK

Proyek ini merupakan sistem pendukung keputusan (SPK) untuk segmentasi pelanggan mall menggunakan kombinasi algoritma **K-Means Clustering** dan **Logistic Regression**, dilengkapi dengan antarmuka prediksi berbasis **Streamlit**.

---

## Struktur Proyek

```
├── Notebook/
│   ├── Kelompok_3_SPK.ipynb      # Notebook analisis utama
│   └── Mall_Customers.csv        # Dataset pelanggan
├── app.py                        # Aplikasi web Streamlit
├── logistic_regression_pipeline.pkl  # Model tersimpan (pipeline)
├── requirements.txt              # Dependensi Python
└── .gitignore
```

---

## Dataset

Dataset yang digunakan adalah **Mall Customers** dengan 200 baris dan fitur sebagai berikut:

| Fitur | Tipe | Keterangan |
|-------|------|------------|
| `CustomerID` | Integer | ID unik pelanggan (dihapus saat preprocessing) |
| `Genre` | Kategorikal | Gender pelanggan (Female / Male) |
| `Age` | Integer | Usia pelanggan |
| `Annual Income (k$)` | Integer | Pendapatan tahunan dalam ribu dolar |
| `Spending Score (1-100)` | Integer | Skor pengeluaran yang ditetapkan mall |

### Karakteristik Dataset
- Tidak ada *missing value* maupun data duplikat → dataset bersih
- Distribusi gender: **56% Female**, **44% Male**
- Usia rata-rata pelanggan: **~39 tahun** (mayoritas 20–40 tahun)
- Korelasi antar variabel numerik umumnya lemah, kecuali Age vs Spending Score yang sedikit negatif (~-0.33)

---

## Metodologi

### 1. Eksplorasi Data (EDA)
Analisis awal meliputi distribusi histogram tiap fitur, scatter plot antar variabel, dan heatmap korelasi. Temuan kunci: scatter plot *Annual Income* vs *Spending Score* memperlihatkan **~5 kluster alami** yang menjadi sinyal kuat kesesuaian data untuk K-Means Clustering.

### 2. Feature Engineering
Ditambahkan fitur baru bernama **Propensity to Spend**, yaitu rasio antara *Spending Score* dan *Annual Income*:

```
Propensity to Spend = Spending Score / Annual Income
```

Nilai tinggi berarti pelanggan cenderung konsumtif relatif terhadap pendapatannya; nilai rendah berarti hemat.

### 3. Preprocessing
- **Label Encoding**: `Genre` diubah menjadi biner (Female = 0, Male = 1)
- **Standarisasi**: Semua fitur dinormalisasi menggunakan `StandardScaler` (mean=0, std=1) agar tidak ada fitur yang mendominasi proses clustering

### 4. Penentuan Jumlah Cluster Optimal
Tiga metode evaluasi digunakan secara bersamaan:

| Metrik | Tujuan | Nilai Lebih Baik |
|--------|--------|-----------------|
| **Silhouette Score** | Mengukur seberapa baik pemisahan cluster | Mendekati 1 |
| **Davies-Bouldin Index (DBI)** | Mengukur compactness & separasi cluster | Lebih rendah |
| **Calinski-Harabasz Score (CH)** | Mengukur rasio dispersi antar vs dalam cluster | Lebih tinggi |

Perbandingan kandidat K:

| Aspek | K=6 | K=7 | K=8 |
|-------|-----|-----|-----|
| Silhouette | 0.301 | 0.303 | **0.372** |
| DBI | **1.084** | 1.118 | **0.922** |
| CH Score | 70.639 | 70.424 | **81.995** |
| Distribusi Cluster | Cukup seimbang | **Paling seimbang** | Mulai terfragmentasi |
| Cluster terbesar | 55 anggota | 39 anggota | 34 anggota |
| Kemudahan interpretasi | Tinggi | **Tinggi** | Sedang |

> **Keputusan: K = 7** dipilih sebagai jumlah cluster optimal karena memberikan keseimbangan terbaik antara kualitas metrik, distribusi anggota yang merata, dan kemudahan interpretasi segmentasi.

---

## Hasil Segmentasi (K=7)

| Cluster | Profil | Gender | Usia | Income | Spending Score | Karakteristik |
|---------|--------|--------|------|--------|----------------|---------------|
| **0** | Pria Senior Moderat | Male | ~56 thn | ~$50k | Rendah (~39) | Berbelanja berdasarkan kebutuhan, kurang konsumtif |
| **1** | Muda Pengeluaran Seimbang | Campuran (↑ Female) | ~26 thn | ~$56k | Sedang (~47) | Segmen umum, potensial untuk program loyalitas |
| **2** | Anak Muda Konsumtif | Female | ~26 thn | Rendah (~$23k) | Tinggi (~79) | *Impulsive buyer*, responsif terhadap promosi |
| **3** | Premium / VIP | Campuran | ~33 thn | Tinggi (~$87k) | Sangat tinggi (~82) | *High value customer*, prioritas retensi |
| **4** | Berpenghasilan Tinggi, Hemat | Campuran (↑ Female) | ~40 thn | Sangat tinggi (~$90k) | Sangat rendah (~16) | Daya beli tinggi namun jarang belanja, potensi besar |
| **5** | Pria Muda Konsumtif | Male | ~25 thn | Rendah–sedang (~$31k) | Tinggi (~66) | Cocok untuk produk lifestyle dan hiburan |
| **6** | Wanita Senior Konservatif | Female | ~51 thn | ~$47k | Rendah (~40) | Belanja berdasarkan nilai & manfaat produk |

---

## Model Klasifikasi

Setelah label cluster dihasilkan oleh K-Means, model **Logistic Regression** dilatih untuk memprediksi segmen pelanggan baru tanpa perlu menjalankan ulang clustering.

### Konfigurasi Pipeline
```python
Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        C=1,
        penalty="l1",       # Lasso regularization
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    ))
])
```

### Pembagian Data
- **Train set**: 90% data (stratified)
- **Test set**: 10% data (stratified)

### Hasil Evaluasi Model

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | Tercantum dalam output notebook |
| **Precision** (weighted) | Tercantum dalam output notebook |
| **Recall** (weighted) | Tercantum dalam output notebook |
| **F1-Score** (weighted) | Tercantum dalam output notebook |
| **ROC-AUC** (OvR weighted) | Tercantum dalam output notebook |

Model disimpan sebagai file `logistic_regression_pipeline.pkl` menggunakan `joblib` untuk digunakan langsung oleh aplikasi Streamlit.

---

## Aplikasi Web (Streamlit)

Aplikasi prediksi interaktif dapat dijalankan secara lokal:

### Instalasi

```bash
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
streamlit run app.py
```

### Cara Penggunaan
1. Buka aplikasi di browser (default: `http://localhost:8501`)
2. Masukkan data pelanggan di panel samping kiri:
   - **Gender**: Female / Male
   - **Usia** (tahun)
   - **Pendapatan Tahunan** (k$)
   - **Skor Pengeluaran** (1–100)
3. Klik tombol **Analisis Segmen**
4. Sistem akan menampilkan:
   - Nomor cluster pelanggan
   - Nilai *Propensity to Spend*
   - Deskripsi karakteristik segmen

---

## Dependensi

```txt
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
plotly
```

---

## Kesimpulan

1. **Dataset Mall Customers** terbukti sangat cocok untuk analisis segmentasi pelanggan karena memiliki pola kluster alami yang terlihat jelas pada scatter plot pendapatan vs skor belanja.

2. **K = 7** dipilih sebagai jumlah kluster optimal berdasarkan evaluasi tiga metrik sekaligus (Silhouette, DBI, CH Score), dengan mempertimbangkan keseimbangan distribusi anggota dan kemudahan interpretasi bisnis.

3. **Tujuh segmen pelanggan** berhasil diidentifikasi dengan profil yang berbeda-beda, mulai dari segmen VIP berpenghasilan tinggi (Cluster 3) hingga pelanggan hemat berdaya beli tinggi (Cluster 4) yang merupakan target konversi yang potensial.

4. **Model Logistic Regression** digunakan sebagai *classifier* untuk memprediksi segmen pelanggan baru secara efisien, menggabungkan *StandardScaler* dan regularisasi L1 dalam satu pipeline yang portabel.

5. **Aplikasi Streamlit** memungkinkan pengguna non-teknis melakukan prediksi segmen secara real-time hanya dengan memasukkan data dasar pelanggan.

---

---

*Model didasarkan pada algoritma K-Means (k=7) & Logistic Regression — Scikit-Learn*
