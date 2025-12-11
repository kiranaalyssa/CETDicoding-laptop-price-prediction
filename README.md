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

## Data Preparation

### Handling Missing Values
Karena dataset tidak memiliki missing value, tidak diperlukan teknik imputasi. Semua baris dapat dipertahankan.

## Handling Outlier
Terdapat beberapa nilai ekstrem (outlier) pada fitur seperti harga, storage, RAM, dan spesifikasi hardware lainnya. Nilai ekstrem ini merupakan variasi alami dari laptop dengan spesifikasi berbeda, sehingga tetap penting untuk prediksi harga.

Sebagai solusi, digunakan winsorization berbasis **IQR (Interquartile Range)**, yaitu menyesuaikan nilai yang berada di bawah atau di atas batas IQR dengan nilai ambang batasnya. Pendekatan ini menjaga informasi penting dalam data, sekaligus mengurangi pengaruh outlier ekstrem terhadap model.

### Encoding
Fitur kategorikal diubah menjadi nilai numerik agar dapat diproses oleh algoritma machine learning, yang hanya menerima input berupa angka. Teknik yang digunakan antara lain One-Hot Encoding untuk fitur seperti brand, is_touch_screen, gpu_type, dan lainnya.

### Train–Test Split
Dataset ini dibagi menjadi data latih dan data uji menggunakan rasio 80:20. Dengan total 991 entri, pembagian tersebut menghasilkan kurang lebih 792 data untuk pelatihan (training set) dan 199 data untuk pengujian (test set).

Rasio 80:20 dipilih karena merupakan standar yang umum digunakan dalam proyek machine learning untuk dataset berukuran menengah. Proporsi 80% memberikan jumlah data yang cukup besar bagi model untuk mempelajari pola dan hubungan antarfitur secara optimal, sementara 20% sisanya tetap menyediakan porsi data yang memadai untuk menilai performa model secara objektif pada data yang tidak dilibatkan dalam proses pelatihan.

### Scaling
Fitur numerik distandarisasi menggunakan StandardScaler agar semua fitur memiliki skala yang sama.
* Tujuan: mencegah fitur dengan rentang nilai besar (misal RAM atau Storage) mendominasi proses pembelajaran model.
* Manfaat: model menjadi lebih stabil dan prediksi lebih akurat.

## Model Development
### Model 1 – K-Nearest Neighbors (KNN) Regressor
K-Nearest Neighbors (KNN) adalah algoritma berbasis instance-based learning atau lazy learner. Artinya, model tidak membangun fungsi atau aturan khusus saat pelatihan, tetapi menyimpan seluruh data latih dan baru melakukan perhitungan ketika diminta melakukan prediksi.

* Proses kerja KNN dalam regresi adalah sebagai berikut:
1. Ketika model menerima satu data baru yang ingin diprediksi, ia akan menghitung jarak antara data tersebut dengan seluruh data latih.
2. Jarak yang digunakan biasanya Euclidean Distance (karena p=2, default).
3. Setelah semua jarak dihitung, model memilih k tetangga terdekat sesuai nilai n_neighbors.
4. Nilai target (harga laptop) diprediksi dengan menghitung rata-rata dari nilai target milik tetangga-tetangga terdekat tersebut.
5. Semakin kecil jaraknya, semakin besar pengaruh datapoint tersebut terhadap prediksi.
Karena KNN melihat kedekatan data (mirip atau tidak mirip), scaling data sangat penting agar fitur dengan rentang besar tidak mendominasi jarak.

* Parameter Model
`knn = KNeighborsRegressor(n_neighbors=10)`
Penjelasan parameter:

n_neighbors = 10
Menentukan jumlah tetangga terdekat yang digunakan untuk melakukan prediksi.
Dengan nilai 10, prediksi dilakukan dengan menghitung rata-rata harga dari 10 laptop yang paling mirip.
Nilai k yang lebih besar membuat model lebih halus (less variance), tetapi terlalu besar dapat mengaburkan pola lokal.

### Model 2 - Random Forest Regressor
Random Forest adalah algoritma ensemble berbasis bagging yang menggabungkan banyak Decision Tree untuk menghasilkan prediksi yang lebih stabil dan akurat. 

* Proses kerja RF dalam regresi adalah sebagai berikut:
1. Model membangun banyak pohon keputusan (Decision Trees).
2. Setiap pohon dilatih pada subset data yang berbeda yang diambil secara acak (bootstrap sampling).
3. Selain bootstrap, setiap pohon juga menggunakan subset fitur acak saat membangun setiap node.
4. Pada saat prediksi, masing-masing pohon memberikan prediksinya, kemudian hasil akhirnya adalah rata-rata prediksi seluruh pohon (untuk regresi).
5. Proses ini membuat Random Forest tahan terhadap overfitting, lebih stabil, dan mampu menangkap pola non-linear.

* Parameter Model
`RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)`

Penjelasan parameter:
1. n_estimators = 50
Jumlah pohon dalam hutan. Semakin banyak pohon, semakin stabil hasilnya.
Dalam kasus ini digunakan 50 pohon—cukup untuk mendapatkan hasil yang baik tanpa waktu komputasi yang terlalu besar.

2. max_depth = 16
Batas kedalaman maksimal setiap pohon.
Batas ini mencegah pohon tumbuh terlalu dalam sehingga mengurangi risiko overfitting.

3. random_state = 55
Memberikan hasil yang konsisten pada setiap running.

4. n_jobs = -1
Menginstruksikan model untuk menggunakan seluruh core CPU agar proses training lebih cepat.

### Model 3 - AdaBoost Regressor (Boosting)
AdaBoost (Adaptive Boosting) adalah algoritma boosting yang bekerja secara sekuensial, di mana setiap model baru dibuat untuk memperbaiki kesalahan dari model sebelumnya.

* Proses kerja RF dalam regresi adalah sebagai berikut:
1. Model pertama (weak learner) dilatih menggunakan seluruh data.
2. Setelah model pertama selesai, datapoint yang diprediksi salah akan diberikan bobot lebih besar, sehingga model berikutnya fokus memperbaiki kesalahan tersebut.
3. Model kedua dilatih dengan mempertimbangkan bobot baru tersebut.
4. Proses ini terus diulang untuk sejumlah iterasi (sesuai default atau parameter).
5. Prediksi akhir merupakan kombinasi berbobot dari seluruh weak learners yang dibangun.
Karena sifatnya sekuensial, boosting sangat baik dalam meningkatkan akurasi, tetapi lebih sensitif terhadap noise.

* Parameter Model
`boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)`

Penjelasan Parameter

1. learning_rate = 0.05
Mengatur kontribusi setiap weak learner (pohon kecil) terhadap model akhir.
Nilai rendah (0.05) membuat setiap model memberi kontribusi kecil sehingga proses boosting lebih halus dan mengurangi risiko overfitting.

2. random_state = 55
Untuk konsistensi hasil.

## Evaluation
Metrik yang digunakan: Mean Squared Error (MSE). MSE mengukur rata-rata kuadrat kesalahan prediksi, memberi penalti besar pada outlier. Dalam pengadaan, prediksi yang terlalu meleset bisa menyebabkan kegagalan tender, sehingga MSE sangat relevan.

### Hasil MSE terhadap Algoritma
Hasil evaluasi model berdasarkan nilai Mean Squared Error (MSE) ditunjukkan pada tabel berikut:
| **Model**   | **Train MSE**       | **Test MSE**        |
|-------------|----------------------|----------------------|
| KNN         | 229552.334894        | 210297.871642        |
| RF          | 32035.624644         | 132791.787976        |
| Boosting    | 226245.065429        | 235836.559536        |

### Interpretasi Hasil MSE
1. K-Nearest Neighbors (KNN)
* MSE train: 229552.334894	
* MSE test: 210297.871642
Selisih train–test tidak terlalu besar, menandakan model relatif stabil.
Namun, nilai MSE yang tinggi menunjukkan bahwa akurasi model kurang baik, kemungkinan karena KNN sensitif terhadap skala data dan tidak mampu menangkap pola kompleks dalam dataset laptop.

2. Random Forest (RF)
* MSE train jauh lebih rendah (32035.624644), menunjukkan model mampu mempelajari pola dengan baik.
* MSE test (132791.787976) juga relatif lebih rendah dibanding model lain.
Meskipun terdapat gap antara train dan test, nilainya masih dalam batas wajar untuk algoritma ensemble yang memang cenderung fit lebih baik pada data latih.
Secara keseluruhan, model ini memberikan performa terbaik di antara ketiganya.

3. AdaBoost (Boosting)
* MSE train (226245.065429) dan test (235836.559536) cukup tinggi, menandakan model tidak optimal dalam belajar pola pada data.
* Gap kecil antara train dan test menunjukkan model tidak overfitting, namun hasilnya tetap kurang akurat.
AdaBoost cenderung kurang kuat pada dataset tanpa noise rendah atau tanpa tuning parameter lebih lanjut.

## Hasil Prediksi
| Index | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|-------|--------|---------------|--------------|--------------------|
| 213   | 144990 | 139155.0      | 115338.7     | 110793.4           |
| 331   | 20999  | 37804.6       | 24054.9      | 37464.9            |
| 501   | 45490  | 59446.6       | 61693.2      | 64064.4            |
| 309   | 67288  | 59446.6       | 60126.1      | 64064.4            |
| 88    | 11990  | 33073.7       | 18007.9      | 37464.9            |
| 535   | 86990  | 91124.0       | 80188.0      | 81795.6            |
| 280   | 24990  | 39156.9       | 28269.2      | 37558.5            |
| 107   | 85990  | 98745.7       | 135662.5     | 133228.4           |
| 59    | 60990  | 83991.0       | 76581.1      | 67710.3            |
| 506   | 47490  | 45925.0       | 43380.2      | 39569.7            |

Tabel di atas menampilkan perbandingan antara nilai harga asli (y_true) dengan hasil prediksi dari tiga model: KNN, Random Forest, dan Boosting. Dari data ini, dapat dilakukan analisis performa masing-masing model berdasarkan kedekatan prediksi terhadap nilai aktual.

### 1. KNN (K-Nearest Neighbors)
KNN menghasilkan prediksi yang cenderung lebih moderat dan berada cukup dekat dengan nilai aktual pada beberapa titik data, terutama pada:
* Index 213: prediksi 139.155 vs 144.990 (cukup dekat)
* Index 535: prediksi 91.124 vs 86.990 (dekat)
Namun pada kasus harga rendah seperti index 331 (20.999), KNN overestimate cukup jauh (hasil 37.804), menunjukkan kelemahan KNN dalam mempelajari pola harga rendah.

### 2. Random Forest
RF umumnya menghasilkan nilai prediksi yang lebih dekat ke nilai aktual dibanding dua model lainnya.
Contoh kedekatan prediksi:
* Index 331: 24.054 vs 20.999 → sangat dekat
* Index 506: 43.380 vs 47.490 → dekat
* Index 280: 28.269 vs 24.990 → dekat
  
Namun ada beberapa titik di mana RF memberikan deviasi besar, terutama pada harga menengah–tinggi:
* Index 107: prediksi jauh lebih tinggi dibanding nilai aktual (overestimate)
Tetap saja, secara keseluruhan RF menunjukkan performa paling stabil.

### 3. Boosting (AdaBoost)
Boosting menghasilkan prediksi yang sering kali lebih tinggi dari nilai aktual (overestimate), terutama:
* Index 331 (harga rendah) → prediksi 37.464 (lebih tinggi)
* Index 107 (harga 85.990) → prediksi 133.228 (melenceng cukup jauh)

Di beberapa titik model cukup akurat:
* Index 506: 39.569 vs 47.490 (relatif dekat)
Namun secara umum, Boosting yang kamu gunakan tampak kurang mampu menangkap pola kompleks harga laptop tanpa tuning lebih lanjut.

### Kesimpulan dari Tabel Prediksi
Berdasarkan perbandingan nilai aktual dan hasil prediksi:
**Random Forest** memberikan prediksi paling mendekati nilai aktual di sebagian besar data. Hal ini sejalan dengan hasil MSE sebelumnya, di mana Random Forest menghasilkan MSE terendah pada data uji. KNN memberikan performa yang cukup, namun hasilnya kurang akurat pada data dengan harga yang sangat rendah atau sangat tinggi. Sementara itu, Boosting menunjukkan selisih prediksi yang paling besar dan cenderung memprediksi harga lebih tinggi dari nilai sebenarnya, sehingga kurang cocok digunakan pada dataset ini.

## Kesimpulan Akhir
* Berdasarkan keseluruhan proses analisis dan pemodelan, proyek ini berhasil menghasilkan model prediktif yang dapat digunakan untuk memperkirakan harga wajar laptop berdasarkan spesifikasi teknisnya. Dari tiga algoritma yang diuji, Random Forest menunjukkan performa terbaik dengan nilai MSE terendah pada data uji. Hal ini menunjukkan bahwa model mampu memberikan prediksi yang lebih mendekati harga aktual dibandingkan dua model lainnya.

Hasil ini sekaligus menjawab Problem Statement yang diajukan di awal.
Melalui analisis fitur dan hasil pemodelan, faktor-faktor seperti processor tier, RAM, GPU type, dan brand terbukti memiliki kontribusi yang signifikan dalam memengaruhi harga laptop. Selain itu, model prediktif yang dibangun telah mampu memberikan estimasi harga yang cukup akurat sehingga dapat dimanfaatkan sebagai alat pendukung keputusan dalam proses pengadaan.

Dengan demikian, proyek ini berhasil mencapai tujuan (Goals), yaitu:
1. Memahami hubungan antara spesifikasi teknis laptop dan harga jualnya.
2. Menentukan model regresi terbaik untuk memprediksi harga wajar laptop dalam konteks pengadaan.

* Apakah proyek ini berhasil menjadi solusi?
Ya, proyek ini berhasil memberikan solusi yang relevan dan aplikatif.
Model prediksi harga yang dihasilkan dapat membantu tim Pengadaan Barang dan Jasa dalam:
1. Menetapkan benchmark harga wajar sebelum proses pembelian.
2. Mengurangi risiko overbudget maupun underbudget.
3. Mendukung proses negosiasi harga dengan vendor menggunakan data objektif.
4. Memahami spesifikasi mana yang paling memengaruhi kenaikan harga, sehingga dapat memilih laptop yang optimal dan cost-effective.

Secara keseluruhan, proyek ini tidak hanya menjawab kebutuhan analitis, tetapi juga memberikan dampak bagi proses pengadaan melalui pemanfaatan model Machine Learning yang efektif dan dapat diandalkan.
