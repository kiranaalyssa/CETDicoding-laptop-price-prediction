# Laporan Proyek Machine Learning - Kirana Alyssa Putri
## Domain Proyek
**Predictive Analysis untuk Pengadaan Laptop**
Proyek ini berada dalam domain Predictive Analysis (analisis prediktif), yang berfokus pada kebutuhan strategis tim Pengadaan Barang dan Jasa. Tujuan utamanya adalah membangun model regresi yang mampu memprediksi harga wajar (Price) sebuah laptop berdasarkan spesifikasi teknisnya.

## Latar Belakang
Menentukan harga wajar laptop dalam proses pengadaan sering menjadi tantangan, karena harga dipengaruhi oleh berbagai variabel spesifikasi, seperti brand, processor tier, kapasitas RAM, dan GPU.

Proyek ini hadir sebagai solusi berbasis data untuk:
* Efisiensi Anggaran: Menetapkan benchmark harga wajar agar anggaran pengadaan tidak overbudget atau underbudget.
* Negosiasi: Memberikan landasan data yang kuat saat bernegosiasi dengan vendor.
* Strategi Pengadaan: Mengidentifikasi fitur yang paling signifikan memengaruhi harga untuk memilih spesifikasi laptop yang cost-effective.

Masalah regresi ini diselesaikan dengan membandingkan tiga model Machine Learning: KNN, Random Forest, dan Gradient Boosting, untuk menemukan model dengan Mean Squared Error (MSE) terendah.

## Business Understanding
### Problem Statements
1. Bagaimana cara mengidentifikasi faktor spesifikasi teknis (prosesor, RAM, penyimpanan, GPU) yang paling signifikan memengaruhi harga laptop?
2. Bagaimana membangun model prediktif yang akurat untuk estimasi harga wajar laptop sebagai alat bantu dalam proses pengadaan?

### Goals
1. Memahami korelasi antara spesifikasi teknis laptop dan harga jualnya.
2. Menemukan model regresi terbaik (dengan MSE terendah) untuk dijadikan alat validasi harga pengadaan.

### Solution Statements
Untuk mencapai tujuan proyek, tiga model diuji dan dievaluasi menggunakan Mean Squared Error (MSE) sebagai ukuran kesalahan prediksi:
* K-Nearest Neighbor (KNN)
Model ini bekerja dengan mencari data yang paling mirip (tetangga terdekat) dengan data baru yang akan diprediksi. Prediksi harga laptop didasarkan pada harga laptop-laptop serupa di dataset.
* Random Forest (RF)
Random Forest adalah model ensemble yang menggabungkan banyak pohon keputusan. Model ini kuat dalam menangani data yang kompleks dan non-linear, sehingga dapat menangkap pola-pola yang sulit terlihat dengan model sederhana.
* Gradient Boosting (Boosting)
Model ini juga termasuk ensemble, namun dibangun secara berurutan. Setiap model berikutnya mencoba memperbaiki kesalahan dari model sebelumnya, sehingga hasil prediksinya cenderung lebih akurat.

## Data Understanding
Pada tahap Data Understanding, fokus utamanya adalah memahami struktur, isi, dan karakteristik dataset sebelum dilakukan proses persiapan data maupun pemodelan. Pengetahuan mendalam terkait data sangat penting agar setiap langkah berikutnya yaitu preprocessing dan modeling dapat dilakukan dengan tepat.

### Sumber Dataset
Dataset diambil dari Kaggle dengan detail berikut:
Jikadara, B. (2024). Brand Laptops Dataset. Kaggle.
URL dataset: https://www.kaggle.com/datasets/bhavikjikadara/brand-laptops-dataset/data

Dataset ini merupakan hasil scraping dan pengumpulan data laptop dari marketplace Smartprix, yang diperbarui terakhir pada 14 Januari 2024. Dataset ini berisi informasi mengenai harga dan spesifikasi berbagai merek laptop yang tersedia di pasaran, sehingga cocok digunakan sebagai bahan analisis dalam proyek ini.

### Jumlah Baris dan Kolom Dataset
Berdasarkan eksplorasi awal menggunakan notebook:
* Jumlah baris (entries): 991
* Jumlah kolom (fitur): 22
  
Jumlah baris tersebut sudah cukup mewakili variasi laptop di pasaran, mencakup perbedaan merek, rentang harga, dan spesifikasi teknis. Sementara itu, jumlah fitur yang tersedia juga memadai untuk menggambarkan faktor-faktor yang memengaruhi harga laptop tanpa menambah kompleksitas analisis secara berlebihan.

### Kondisi Dataset
Mengevaluasi kondisi data merupakan bagian penting agar kita memahami apa saja yang perlu ditangani pada tahap Data Preparation. Berikut ringkasannya:

1. Missing Values
Berdasarkan pemeriksaan menggunakan df.isnull().sum(), dataset tidak memiliki missing value.
Ini memudahkan proses preprocessing karena tidak diperlukan imputasi atau penghapusan baris berdasarkan missing data.

2. Inkonsistensi Format
Nama kolom pada dataset awal memiliki inkonsistensi dalam penulisan, seperti penggunaan huruf kapital dan format penamaan yang tidak seragam. Untuk menghindari potensi kesalahan saat pemanggilan kolom dan memastikan konsistensi pada proses analisis, seluruh nama kolom kemudian distandarisasi menjadi huruf kecil menggunakan fungsi .lower().

3. Outlier
Outlier terdeteksi pada beberapa fitur numerik, terutama:
* Price → wajar karena terdapat laptop flagship yang sangat mahal.
* RAM dan Storage → variasi kapasitas laptop memang besar.
Outlier tidak selalu salah. Pada konteks produk laptop, outlier sering mencerminkan kategori produk yang berbeda. Namun, beberapa outlier ekstrem perlu dikendalikan agar tidak mendistorsi model.

### Uraian Seluruh Fitur Dataset
| **Fitur**                     | **Penjelasan (Bahasa Indonesia)**                                               |
|------------------------------|----------------------------------------------------------------------------------|
| Brand                        | Nama merek laptop.                                                               |
| Model                        | Seri atau model spesifik dari laptop.                                            |
| Price                        | Harga laptop dalam mata uang Rupee India.                                        |
| Rating                       | Penilaian atau rating berdasarkan spesifikasi atau performa laptop.              |
| Processor brand              | Merek prosesor yang digunakan pada laptop.                                       |
| Processor tier               | Kategori atau tingkat performa dari prosesor.                                    |
| Number of Cores              | Jumlah inti pemrosesan pada prosesor.                                            |
| Number of Threads            | Jumlah thread yang dapat diproses oleh prosesor.                                 |
| Ram memory                   | Besaran kapasitas RAM pada laptop.                                               |
| Primary storage type         | Jenis penyimpanan utama (misalnya HDD atau SSD).                                 |
| Primary storage capacity     | Kapasitas penyimpanan utama.                                                     |
| Secondary storage type       | Jenis penyimpanan tambahan apabila tersedia.                                     |
| Secondary storage capacity   | Kapasitas penyimpanan tambahan.                                                  |
| GPU brand                    | Merek kartu grafis (GPU) yang digunakan.                                         |
| GPU type                     | Jenis atau tipe GPU yang terpasang.                                              |
| Is Touch screen              | Menunjukkan apakah laptop memiliki fitur layar sentuh.                           |
| Display size                 | Ukuran layar laptop dalam inci.                                                  |
| Resolution width             | Resolusi lebar layar.                                                            |
| Resolution height            | Resolusi tinggi layar.                                                           |
| OS                           | Sistem operasi yang terpasang pada laptop.                                       |
| Year of warranty             | Lama garansi yang diberikan, umumnya dalam tahun.                                |

## Data Cleaning
Tahap ini bertujuan untuk memastikan dataset siap digunakan dalam analisis dan proses prediksi harga laptop. Langkah-langkah utama meliputi:
* Standarisasi nama kolom:

`df.columns = df.columns.str.lower()`

Semua nama kolom diubah menjadi huruf kecil agar penamaan konsisten dan lebih mudah diakses saat eksplorasi, visualisasi, dan preprocessing.

* Menghapus kolom tidak relevan:

`df = df.drop(columns=['index'])`

Kolom index dihapus karena tidak mengandung informasi terkait harga laptop maupun fitur teknis seperti RAM, brand, processor, atau storage.

## Handling Missing Value dan Outlier
Dataset ini tidak memiliki missing value, namun terdapat beberapa nilai ekstrem (outlier) pada fitur seperti harga, storage, RAM, dan spesifikasi hardware lainnya. Nilai ekstrem ini merupakan variasi alami dari laptop dengan spesifikasi berbeda, sehingga tetap penting untuk prediksi harga.

Sebagai solusi, digunakan winsorization berbasis IQR (Interquartile Range), yaitu menyesuaikan nilai yang berada di bawah atau di atas batas IQR dengan nilai ambang batasnya. Pendekatan ini menjaga informasi penting dalam data, sekaligus mengurangi pengaruh outlier ekstrem terhadap model.

## Data Preparation
### Encoding
Fitur kategorikal diubah menjadi nilai numerik agar dapat diproses oleh algoritma machine learning, yang hanya menerima input berupa angka. Teknik yang digunakan antara lain One-Hot Encoding untuk fitur seperti brand, processor_tier, dan gpu_type.

### Train–Test Split
Dataset dibagi menjadi data latih dan data uji menggunakan rasio 80:20.

### Scaling
Fitur numerik distandarisasi menggunakan StandardScaler agar semua fitur memiliki skala yang sama.
* Tujuan: mencegah fitur dengan rentang nilai besar (misal RAM atau storage) mendominasi proses pembelajaran model.
* Manfaat: model menjadi lebih stabil dan prediksi lebih akurat.

## Modeling
Tiga algoritma diuji: KNN, Random Forest, Gradient Boosting.

**Pemilihan Model Terbaik**
* Random Forest dipilih sebagai model terbaik karena: Memberikan MSE terendah pada data test.
* Tahan terhadap outlier dan noise, umum pada harga pasar laptop.
* Mampu menggeneralisasi dengan baik tanpa overfitting berlebihan.

## Evaluation
Metrik yang digunakan: Mean Squared Error (MSE). MSE mengukur rata-rata kuadrat kesalahan prediksi, memberi penalti besar pada outlier. Dalam pengadaan, prediksi yang terlalu meleset bisa menyebabkan kegagalan tender, sehingga MSE sangat relevan.

**Kesimpulan:**
Random Forest unggul dengan Test MSE terendah, menunjukkan prediksi paling dekat dengan harga aktual. Model ini menjadi alat prediksi harga wajar yang andal untuk proses pengadaan.





