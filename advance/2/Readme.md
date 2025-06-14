# Laporan Proyek Machine Learning - Nama Anda

## Project Overview
### Latar belakang
Cepatnya perkembangan ponsel serta banyaknyak jenis ponsel dengan spesifikasi, harga, dan fitur yang beragam, 
seringkali membuat binggung calon pembeli dalam memilih ponsel yang sesuai dengan preferensi mereka. 
Terutama bagi pengguna yang awam akan teknologi.

Sistem rekomendasi dapat menjadi sebuah solusi bagi pengguna dalam memilih ponsel yang sesuai dengan preferensi.
 Dengan memanfaatkan data preferensi pengguna serta karakteristik produk, sistem ini dapat menyarankan pilihan ponsel yang relevan, sehingga proses pengambilan keputusan menjadi lebih efisien dan terarah.

## Business Understanding

### Problem Statements
Bagaimana cara membentuk sistem yang dapat memeberi rekomendasi ponsel bagi pengguna?

### Goals
Membangun sistem rekomendasi berdasarkan spesifikasi ponsel dan `rating` ponsel oleh penguna.

### Solution statements
  - Membangun sistem berbasis content-based filtering, dengan menggunakan `consine similarity` untuk membentuk metrik kemiripan spesifikasi ponsel.
  - Membangun sistem berbasis collaborative-based filtering, menggunakan data `rating` ponsel yang diproses menggunakan machine learning.

## Data Understanding
Dataset yang digunakan berasal dari:
https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations
Dataset terdiri dari 3 buat file sebagai berikut:

### Data ponsel (cellphones data.csv)
Data ponsel terdiri atas 33 baris data dan 14 kolom.
1. **cellphone_id**
   ID unik yang digunakan untuk mengidentifikasi setiap ponsel dalam dataset.
2. **brand**
   Merek dari ponsel.
3. **model**
   Nama atau tipe model ponsel.
4. **operating system**
   Sistem operasi yang digunakan oleh ponsel.
5. **internal memory**
   Kapasitas penyimpanan internal (ROM) ponsel dalam satuan gigabyte (GB).
6. **RAM**
   Kapasitas memori utama (RAM) dalam satuan gigabyte (GB).
7. **performance**
   Skor performa keseluruhan ponsel.
8. **main camera** Resolusi kamera belakang utama ponsel dalam satuan megapiksel (MP).
9. **selfie camera**
   Resolusi kamera depan (selfie) ponsel dalam satuan megapiksel (MP).
10. **battery size**
    Kapasitas baterai ponsel dalam satuan miliampere-hour (mAh).
11. **screen size**
    Ukuran layar ponsel dalam satuan inci, yang menunjukkan besar layar secara diagonal.
12. **weight**
    Berat ponsel dalam satuan gram (g).
13. **price**
    Harga ponsel dalam satuan dolar.
14. **release date**
    Tanggal resmi peluncuran ponsel di pasaran.

### Data pengguna (users.csv)
Data pengguna terdiri atas 99 baris data dan 4 kolom.
1. **user_id**
   ID unik yang digunakan untuk mengidentifikasi setiap pengguna dalam sistem.
2. **age**
   Usia pengguna dalam satuan tahun. 
3. **gender**
   Jenis kelamin pengguna, biasanya dikategorikan sebagai “Male”, “Female”, atau label lainnya.
4. **occupation**
   Pekerjaan pengguna. Fitur ini bersifat kategorikal.

### Data penilaian (ratings.csv)
Data penilaian terdiri atas 990 baris data dan 3 kolom.
1. **user_id**
   ID unik pengguna yang memberikan penilaian terhadap ponsel. Merujuk pada kolom `user_id` di dataset pengguna.
2. **cellphone\_id**
   ID unik ponsel yang dinilai oleh pengguna. Merujuk pada kolom `cellphone_id` di dataset ponsel.
3. **rating**
   Nilai penilaian (rating) yang diberikan pengguna terhadap ponsel dalam skala 1 - 10.

Namun pada proyek kali ini kita hanya menggunakan data ponsel dan data penilaian saja.

## Exploratory Data Analysis
### Data Ponsel
![eda-cellphone.png](.\images\eda-cellphone.png)
![eda-cellphone-cat.png](.\images\eda-cellphone-cat.png)
1. Data menunjukkan bahwa mayoritas ponsel memiliki kapasitas memori internal sebesar 128 GB.
2. Dari segi RAM, ponsel dengan kapasitas 8 GB paling banyak ditemukan.
3. Nilai performa dan harga ponsel cukup bervariasi, tanpa adanya dominasi yang mencolok dari satu model tertentu.
4. Kamera utama yang paling umum digunakan memiliki resolusi 50 MP.
5. Untuk kamera depan (selfie camera), resolusi yang paling banyak digunakan adalah 32 MP.
6. Ukuran baterai yang paling sering ditemukan adalah 5000 mAh. 7 .Ukuran layar paling banyak berada pada rentang 6,0 hingga 7,0 inci.
7. Berat ponsel bervariasi, namun paling banyak berada pada kisaran 200–210 gram.
8. Samsung merupakan merek dengan jumlah tipe ponsel terbanyak dalam data yang dianalisis.

### Data Penilaian(Rating)
![eda-ratings.png](.\images\eda-ratings.png)
1. User ID: Aktivitas pengguna bervariasi; sebagian sangat aktif, sebagian hanya memberi sedikit ulasan.
2. Cellphone ID: Jumlah ulasan per ponsel cukup merata, tiap model mendapat ulasan yang seimbang.
3. Rating: Mayoritas rating berada di kisaran 6–9, menunjukkan kecenderungan penilaian positif dari pengguna.Namun terdapat outlier untuk nilai diatas 10.

## Data Preparation
- Menghapus baris pada data rating yang memiliki rating diatas 10, karena merupakan outlier.
- Mengganti nilai data gender yang bernilai '-Select Gender-' (karena tidak diisi/kosong) dengan 'Unspecified'.

### Content Filtering
1. Scaling data untuk numerikal fitur ('performance', 'internal memory', 'RAM', 'main camera',
    'selfie camera', 'battery size', 'screen size', 'weight', 'price') pada dataset cellphones. Disini kita menggunakan `StandarScaler` untuk melakukan scaling.
Hal ini dilakukan untuk memastikan bahwa semua fitur numerik memiliki skala yang sama, sehingga model pembelajaran mesin tidak bias terhadap fitur dengan nilai yang lebih besar.
2. Menggunakan One-Hot Encoding untuk fitur kategorikal (OS dan brand), karena jumlah kategorinya terbatas (10 brand dan 2 OS).
Pendekatan ini dipilih karena One-Hot Encoding efektif untuk fitur dengan jumlah kategori yang relatif sedikit. Diperlukan proses encoding untuk data kategorikal agar data dapat diproses oleh algoritma machine learning.

### Collaborative Filtering
1. Menyandikan (encode) fitur user_id dan ‘cellphone_id’ ke dalam indeks integer, hal ini dilakukan agar nilai dari kedua fitur tersebut menjadi nilai integer yang berurutan, sehingga lebih efisien dan mudah diproses oleh algoritma machine learning.
2. Memetakan user_id dan ‘cellphone_id’ ke dataframe yang berkaitan, agar dapat dengan mudah mangakses nilai id asli dan hasil encoding.
3. Mengubah nilai rating menjadi float. Ini dilakukan agar nilai rating dapat diproses dengan tepat dalam perhitungan matematis oleh model.
4. Mengacak data pada rating. Pengacakan dilakukan untuk mencegah bias urutan saat membagi data menjadi data latih dan data uji, sehingga model dapat belajar secara lebih umum dan tidak overfitting pada pola urutan tertentu.
5. Mengubah nilai rating menjadi skala 0 hingga 1. Hal ini bertujuan untuk menyamakan skala nilai rating agar model lebih stabil dalam proses pelatihan dan hasilnya lebih konsisten.
6. Membagi data train dan validasi dengan komposisi 80:20. Bertujuan untuk melatih model pada sebagian besar data, sementara sisanya digunakan untuk mengevaluasi kinerja model secara objektif.


## Modeling
### Content Filtering
Sistem content filtering akan memberikan `top-N recommendation` berdasarkan kemiripan pada fitur ponsel.
Sistem dibentuk menggunakan data fitur ponsel untuk membentuk matrix kemiripan dengan menggunakan `cosine similarity`. 
**Kelebihan**:
- Tidak bergantung pada data pengguna lain.
- Dapat menangani item baru dengan baik, jika fitur/metadata lengkap dan tersedia.
**Kekurangan**:
- Membutuhkan deskripsi fitur yang lengkap dan relevan.
- Tidak mampu mengangani tren yang sedang terjadi.
Percobaan rekomedasi:
- Preferensi item yang ingin dicari
![requested-recom.png](.\images\requested-recom.png)
- Hasil rekomendasi
![result-content.png](.\images\result-content.png)

### Collaborative Filtering
Pada sistem ini menggunakan data pengguna lain berupa penilaian(rating) sebagai sumber data untuk dipelajari

Pada metode sistem rekomendasi collaborative filtering, mengolah data interaksi pengguna, dalam proyek ini berupa penilaian (rating), 
untuk memprediksi preferensi pengguna lain. 
Pada proyek ini menggunakan Keras Model class untuk membuat class Recommender Net. Model aakn menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. 
**Kelebihan**:
- Tidak memerlukan detail fitur item.
- Mampu merekomendasikan item yang tidak mirip secara konten, tetapi disukai oleh pengguna dengan preferensi serupa.
- Dapat memberikan rekomendasi yang lebih luas dibanding content filtering.

**Kekurangan**:
- Sulit untuk memberi rekomendasi pada pengguna dan item baru.
- Jika matriks pengguna–item sedikit, sulit menemukan pola yang kuat untuk rekomendasi.

Hasil rekomendasi:
![colla-result.png](.\images\colla-result.png)

## Evaluation
### Content Filtering
Untuk mengevaluasi sistem Content-Based Filtering, digunakan metrik Precision@K, yaitu seberapa banyak item relevan yang berhasil direkomendasikan dari total K item yang ditampilkan, dibandingkan dengan ground truth yang telah ditentukan sebelumnya.

Pada sistem ini, hasil evaluasi menghasilkan Precision@K sebesar **0.56**. Nilai ini masih tergolong kurang memuaskan untuk digunakan sebagai sistem rekomendasi yang andal. Namun, perlu dicatat bahwa hasil ini dapat dipengaruhi oleh adanya bias dalam penentuan ground truth, misalnya jika ground truth tidak sepenuhnya mewakili preferensi pengguna sebenarnya atau terlalu sempit cakupannya.

### Collaborative Filtering
Untuk mengevaluasi sistem Collaborative-Based FIltering, dapat mengguakan Root Mean Squared Error (RMSE). 
RMSE dihasilkan dengan menghitung rata-rata kesalahan  prediksi model dengan data yang asli.  Semakin kecil nilai RMSE, semakin dekat nilai yang diprediksi dan diamati.
Pada sistem ini, hasil evaluasi RMSE menghasilkan sebesar **0.2807** pada epoch terakhir, yang mana hal tersebut cukup baik  dan mampu merepresentasikan pola rating pengguna dengan cukup akurat.