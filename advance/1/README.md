# Laporan Proyek Machine Learning - I

## Domain Proyek

Diabetes adalah penyakit kronis yang dapat menyebabkan kematian, apabila tidak ditangani dengan baik. 
Menurut International Diabetes Federation (IDF), terdapat sekitar 537 juta orang dewasa di seluruh dunia setara dengan 1 dari setiap 10 individu yang menderita diabetes. Penyakit ini juga mencatatkan sebagai penyebab 6,7 juta kematian per tahun, atau setara dengan 1 kematian setiap 5 detik. Berdasarkan data tersebut, penanganan  dan pencegahan merupakan sebuah urgensi.

Penggunaan teknologi machine learning untuk menganalisis data secara akurat dalam mengidentifikasi risiko diabetes merupakan salah satu bentuk inovasi di bidang kesehatan digital. Dengan kemampuan dalam mengenali pola dari data medis yang kompleks, pendekatan ini diharapkan dapat menurunkan angka kematian serta meningkatkan kualitas hidup penderita diabetes melalui deteksi dan intervensi yang lebih cepat dan tepat sasaran.

## Business Understanding

### Problem Statements
- Bagaimana korelasi fitur `glucose` dan `BMI` dengan kondisi pasien diabetes?
- Bagaimana mengatasi ketidakseimbangan kelas pada data diabetes?
- Bagaimana cara memprediksi risiko kematian pada penderita diabetes?
### Goals
- Mengetahui dan memahami korelasi fitur `glucose` dan `BMI` dengan kondisi pasien diabetes.
- Mengurangi bias model terhadap kelas mayoritas agar performa klasifikasi menjadi lebih seimbang dan representatif.
- Membuat model machine learning yang dapat memprediksi risiko kematian pada penderita diabetes.
### Solution Statement
- Melakukan multivariate analysis untuk mengetahui korelasi fitur `glucose` dan `BMI` dengan kondisi pasien diabetes.
- Menggunakan SMOTE untuk mengatasi ketidakseimbangan kelas pada data diabetes.
- Melakukan analysis pada data untuk menentukan fitur yang tepat, serta melakukan hypertuning pada algoritma yang digunakan untuk membuat model.
- 
## Data Understanding
Data diambil dari https://www.kaggle.com/datasets/joebeachcapital/diabetes-factors. Dataset memiliki 768 baris dan 9 kolom.

### Variabel-variabel pada dataset:
1. Pregnancies (int64) :Jumlah kehamilan yang pernah dialami oleh pasien.
2. Glucose (int64): Konsentrasi glukosa plasma dua jam dalam tes toleransi glukosa oral.
3. BloodPressure (int64): Tekanan darah diastolik (mm Hg).
4. SkinThickness (int64): Ketebalan lipatan kulit triceps (dalam mm), sebagai indikator lemak tubuh.
5. Insulin (int64): Kadar insulin serum 2 jam (mu U/ml).
6. BMI (float64): Indeks massa tubuh (berat badan dalam kg dibagi tinggi badan dalam m²).
7. DiabetesPedigreeFunction (float64): Fungsi silsilah genetik yang menunjukkan kemungkinan keturunan diabetes berdasarkan riwayat keluarga.
8. Age (int64):Usia pasien (dalam tahun).
9. Outcome (object): Status akhir berupa dead atau alive

### Missing Data dan Outlier:
![Box Plot Features](.\images\box-plot-features.png)
- Tidak terdapat missing value pada dataset.
- Terdapat data dengan nilai 0 pada Glucose, Insulin, SkinThickness, BMI, dan BloodPressure, yang merupakan data yang tidak valid.
- Terdapat outlier pada beberapa fitur, namun dipertahankan karena dianggap sebagai data valid yang mengandung informasi penting dan dapat berpengaruh terhadap variabel target.

### Univariate  Analysis
![hist-features.png](.\images\hist-features.png)![Box Plot Features](.\images\box-plot-features.png)
- Kehamilan: Sebagian besar wanita memiliki jumlah kehamilan antara 0–3 kali.
- Glukosa: Mayoritas nilai glukosa berada di kisaran 100–130 dengan beberapa nilai tinggi.
- Tekanan Darah: Nilai tekanan darah umumnya berkisar antara 60–80.
- BMI: Sebagian besar BMI berada di rentang 25–35, menunjukkan banyak yang overweight atau obesitas.
- Distribusi usia condong ke kiri dengan dominasi usia 20-30 tahun.
- Distribusi DiabetesPedigreeFunction sangat condong ke kiri (right-skewed), dengan sebagian besar nilai berada antara 0.0 dan 0.5.

### Multivariate  Analysis
![correlation-matrix-features.png](.\images\correlation-matrix-features.png)
- Berdasarkan correlation matrix, tidak ada fitur yang memiliki korelasi tinggi satu sama lain.

![box-plot-by-outcome.png](.\images\box-plot-by-outcome.png)
- Glucose: Median pada pasien meninggal lebih tinggi dibandingkan yang masih hidup. Namun terdapat beberapa outlier pada passien yang masih hidup.
- BMI: Median BMI pada pasien meniggal juga terlihat sedikit lebih tinggi dibandingkan yang hidup.
- Pregnancies: Median jumlah kehamilan sedikit lebih tinggi pada pasien yang meninggal, tetapi distribusinya relatif mirip antara kedua kelompok.
- BloodPressure: Median kelompok yang masih hidup sedikit lebih tinggi, namun tidak menunjukkan perbedaan yang signifikan.
- DiabetesPedigreeFunction: Distribusi antar kelompok hampir sama; perbedaan tidak mencolok, tetapi tetap relevan sebagai indikator faktor genetik.
- Age: Pasien yang meninggal cenderung berusia lebih tua dibandingkan yang masih hidup, menjadikan usia sebagai salah satu fitur penting dalam klasifikasi.


## Data Preparation
- Menghapus fitur Insulin dan SkinThickness, kerena memiliki data tidak valid dengan jumlah besar.
- Menghapus baris yang memiliki nilai 0 untuk BMI, Glucose dan BloodPressure.
- Mehapus baris yang memiliki outlier pada fitur BloodPressure  yang memiliki nilai di bawah kuartil pertama (Q1) karena nilai tidak wajar.
- Mengelompokan (binning data) fitur `Age` dan `BMI` berdasarkan rentang umur dan BMI, dengan bantuan `LabelEncoder`. Langkah ini dilakukan agar pola hubungan antara usia atau indeks massa tubuh dengan risiko kematian pada penderita diabetes menjadi lebih mudah dianalisis dan diinterpretasikan, baik secara statistik maupun visual. 
- Men-encode `Outcome` yang merupakan target menjadi 0 dan 1. Hal in perlu dilakukan untuk mengubah nilai target menjadi format numerik agar dapat digunakan oleh algoritma machine learning. Dalam konteks ini, label seperti "alive" diubah menjadi 0 dan "dead" menjadi 1, sehingga model dapat mengenali dan memproses informasi target dengan benar dalam proses pelatihan dan prediksi.
- Menentukan fitur yang digunakan untuk model, dalam hal ini, fitur yang dipilih meliputi Pregnancies, Glucose, DiabetesPedigreeFunction, AgeGroupEncoded, dan BMICategoryEncoded.
Hal ini bertujuan untuk mengurangi kompleksitas model dan memastikan bahwa model hanya menggunakan fitur yang relevan untuk memprediksi risiko diabetes.
- Melakukan resampling menggunakan SMOTE untuk mengatasi masalah ketidakseimbangan kelas pada target dataset, sehingga model memiliki distribusi kelas yang lebih seimbang dan menghasilkan prediksi yang lebih akurat.
- Memisahkan data menjadi training dan test dengan rasio 80:20 menggunakan metode `train_test_split`.
- Melakukan standarisasi data menggunakan `StandardScaler`, hal ini dapat membantu model untuk bekerja secara optimal, jika menggunakan algoritma yang sensitif terhadap skala.

## Modeling
### Algoritma yang digunakan:
Random Forest

Salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Algoritma ini termasuk metode ensemble yang membangun banyak decision tree dan menggabungkan hasilnya untuk meningkatkan akurasi dan kestabilan prediksi.

Cara kerjanya melibatkan dua teknik utama: bagging dan pemilihan fitur secara acak. Setiap pohon dilatih dengan subset data yang dipilih secara acak (dengan pengembalian), dan pada setiap split, hanya sebagian fitur yang dipertimbangkan secara acak. Hasil prediksi akhir ditentukan dengan voting mayoritas (untuk klasifikasi) atau rata-rata (untuk regresi), sehingga mengurangi risiko overfitting dan membuat model lebih andal.

Parameter yang digunakan: 
- random_state: 42.  Digunakan untuk memastikan hasil yang konsisten dan dapat direproduksi setiap kali model dijalankan, karena proses pembentukan pohon dalam Random Forest bersifat acak.

### Hypertunning Parameter
Proses hypertunning dilakukan dengan bantuan `GridSearchCV` untuk mencari parameter terbaik.
- n_estimators: jumlah pohon dalam hutan, mengontrol stabilitas prediksi.
- max_depth: kedalaman maksimum pohon, mencegah overfitting atau underfitting.
- min_samples_split dan min_samples_leaf: mengatur jumlah minimum sampel yang dibutuhkan untuk membagi node dan membentuk daun, membantu mengontrol kompleksitas pohon.
- bootstrap: menentukan apakah sampel bootstrap digunakan saat membangun pohon.
Parameter yang digunakan:
```
'n_estimators': [100, 300],
'max_depth': [None, 10, 30],
'min_samples_split': [2, 9],
'min_samples_leaf': [2, 9],
'bootstrap': [True, False]
```
Parameter terpilih:
```
'bootstrap': True, 
'max_depth': None, 
'min_samples_leaf': 2, 
'min_samples_split': 9, 
'n_estimators': 100
```

## Evaluation
Metrik yang digunakan pada proyek ini adalah akurasi, precision, recall, dan F1 score.
Akurasi digunakan untuk mengukur proporsi prediksi yang benar dari keseluruhan data. Precision digunakan untuk mengevaluasi seberapa banyak dari prediksi positif yang benar-benar positif.
Recall digunakan untuk mengukur seberapa banyak dari kasus positif yang berhasil dideteksi oleh model.
F1 Score merupakan harmonic mean dari precision dan recall, dan digunakan sebagai metrik utama ketika ada trade-off antara keduanya.

```
Accuracy     : 0.8095
Precision    : 0.8444
Recall       : 0.7755
F1 Score     : 0.8085
```
Berdasakan hasil evaluasi, model memiliki tingkat `precision` yang cukup bagus, 
yang mana hal tersebut penting dalam bidang kesehatan untuk memberikan informasi yang akurat tentang risiko kematian pada penderita diabetes.
Selain itu, nilai dan accuracy diatas 80% menunjukan model mampu mengklasifikasikan data dengan cukup baik.
