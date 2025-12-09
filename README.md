# Laporan Proyek Machine Learning - Kirana Alyssa Putri
## Domain Proyek
**Predictive Analysis untuk Pengadaan Laptop**
Proyek ini berada dalam domain Predictive Analysis (analisis prediktif), yang berfokus pada kebutuhan strategis tim Pengadaan Barang dan Jasa. Tujuan utamanya adalah membangun model regresi yang mampu memprediksi harga wajar (Price) sebuah laptop berdasarkan spesifikasi teknisnya.

## Latar Belakang dan Relevansi
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
Untuk mencapai goals, tiga model diuji dan dievaluasi menggunakan MSE:
1. K-Nearest Neighbor (KNN): Model berbasis kemiripan data.
2. Random Forest (RF): Model ensemble yang tangguh terhadap non-linearitas data.
3. Gradient Boosting (Boosting): Model ensemble yang meningkatkan performa secara sekuensial.

Model dengan MSE terkecil akan menjadi acuan harga prediksi.

## Data Understanding
Dataset yang digunakan: **Brand Laptops Dataset** dari Kaggle.  
Sumber: Jikadara, B. (2024). *Brand Laptops Dataset*. Kaggle. Diakses dari [https://www.kaggle.com/datasets/bhavikjikadara/brand-laptops-dataset/data]

Dataset Brand Laptops Dataset ini berisi 991 entri unik laptop. Dataset ini telah dibersihkan secara menyeluruh dan mencakup 22 fitur seperti nama laptop, harga dalam rupee India, prosesor, GPU, dan lain-lain. Dataset ini bersumber dari website ‘Smartprix’ dan diperbarui terakhir pada 14 Januari 2024 oleh author dataset tersebut.

## Data Preparation


## Modeling
Tiga algoritma diuji: KNN, Random Forest, Gradient Boosting.

**Pemilihan Model Terbaik**
* Random Forest dipilih sebagai model terbaik karena: Memberikan MSE terendah pada data test.
* Tahan terhadap outlier dan noise, umum pada harga pasar laptop.
* Mampu menggeneralisasi dengan baik tanpa overfitting berlebihan.

## Evaluation
Metrik yang digunakan: Mean Squared Error (MSE)

MSE mengukur rata-rata kuadrat kesalahan prediksi, memberi penalti besar pada outlier.

Dalam pengadaan, prediksi yang terlalu meleset bisa menyebabkan kegagalan tender, sehingga MSE sangat relevan.

Kesimpulan:
Random Forest unggul dengan Test MSE terendah, menunjukkan prediksi paling dekat dengan harga aktual. Model ini menjadi alat prediksi harga wajar yang andal untuk proses pengadaan.





