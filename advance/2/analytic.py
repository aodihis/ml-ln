# -*- coding: utf-8 -*-
"""dicoding-319-2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TcLLCwZxz1ybH2qRPVnyJTmG2u8lstCX

# Import packages
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

"""# Data Loading
- Mengunduh dan meng-ekstrak dataset.
- dataset yang digunakan: meirnizri/cellphones-recommendations
"""

from google.colab import files
files.upload()

# Create kaggle folder and move file
os.makedirs("/root/.kaggle", exist_ok=True)
!mv kaggle.json /root/.kaggle/kaggle.json

# Set permissions
!chmod 600 /root/.kaggle/kaggle.json

!kaggle datasets download -d meirnizri/cellphones-recommendations
!unzip cellphones-recommendations.zip

"""# Data Understanding
Pada proyek kali ini kita memiliki 3 file terpisah sebagai dataset.
- Membaca masing-masing file dataset.
- Melakukan pengecekan nilai unik pada tiap dataset.
"""

cp_df = pd.read_csv('cellphones data.csv')
ratings_df = pd.read_csv('cellphones ratings.csv')
users_df = pd.read_csv('cellphones users.csv')

cp_df.head()

ratings_df.head()

users_df.head()

print('Jumlah data cellphones: ', len(cp_df.cellphone_id.unique()))
print('Jumlah data users: ', len(users_df.user_id.unique()))

"""# Exploratory Data Analysis

## Univariate Exploratory Data Analysis

### Cellphones Data
- Mengecek fitur pada dataset.
- Mengecek perseberan fitur numerik.
- Mengecek persebaran fitur categorikal (brand)
"""

cp_df.info()

cp_df.drop('cellphone_id', axis=1).hist(bins=50, figsize=(10,7))
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot for Brand
cp_df['brand'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Distribution of Brand')
axes[0].set_xlabel('Brand')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Plot for OS
cp_df['operating system'].value_counts().plot(kind='bar', ax=axes[1], color='orange')
axes[1].set_title('Distribution of OS')
axes[1].set_xlabel('OS')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

"""1. Data menunjukkan bahwa mayoritas ponsel memiliki kapasitas memori internal sebesar 128 GB.
2. Dari segi RAM, ponsel dengan kapasitas 8 GB paling banyak ditemukan.
3. Nilai performa dan harga ponsel cukup bervariasi, tanpa adanya dominasi yang mencolok dari satu model tertentu.
4. Kamera utama yang paling umum digunakan memiliki resolusi 50 MP.
5. Untuk kamera depan (selfie camera), resolusi yang paling banyak digunakan adalah 32 MP.
6. Ukuran baterai yang paling sering ditemukan adalah 5000 mAh.
7 .Ukuran layar paling banyak berada pada rentang 6,0 hingga 7,0 inci.
8. Berat ponsel bervariasi, namun paling banyak berada pada kisaran 200–210 gram.
9. Samsung merupakan merek dengan jumlah tipe ponsel terbanyak dalam data yang dianalisis.

### Ratings Data
- Mengecek fitur pada dataset.
- Mengecek perseberan fitur numerik.
"""

ratings_df.info()

ratings_df.hist(bins=50, figsize=(10,7))
plt.show()

"""1. User ID: Aktivitas pengguna bervariasi; sebagian sangat aktif, sebagian hanya memberi sedikit ulasan.
2. Cellphone ID: Jumlah ulasan per ponsel cukup merata, tiap model mendapat ulasan yang seimbang.
3. Rating: Mayoritas rating berada di kisaran 6–9, menunjukkan kecenderungan penilaian positif dari pengguna.Namun terdapat outlier untuk nilai diatas 10.

### User Data
- Mengecek fitur pada dataset.
- Mengecek perseberan fitur numerik.
- Mengecek persebaran fitur categorikal.
"""

users_df.info()

users_df.drop('user_id', axis=1).hist(bins=50, figsize=(8,4))
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
users_df['gender'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)

plt.subplot(1, 2, 2)
users_df['occupation'].value_counts().plot(kind='bar', color='salmon')
plt.title('Distribution of Occupation')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=90)

plt.tight_layout()
plt.show()

"""- Persebaran umur pada dataset users cukup bervariasi, namun didominasi oleh pengguna berumur 25 tahun.
- Persebaran jenis kelamin tidak berbeda jauh antara laki-laki dan perempuan. Tetapi terdapat outlier pada beberapa data.
- Information Technology merupakan pekerjaan yang paling umum di antara pengguna, diikuti oleh Manager dan Developer.

# Data Prepartion

## Handling Outlier
- Menghapus baris pada data rating yang memiliki rating diatas 10.
- Mengganti nilai data gender yang bernilai '-Select Gender-' dengan 'Unspecified'.
"""

ratings_df = ratings_df[ratings_df['rating'] <= 10]

users_df['gender'] = users_df['gender'].replace('-Select Gender-', 'Unspecified')

"""### Content Filtering Prep
- Scaling data untuk numerikal fitur pada dataset cellphones.
- Menggunakan One-Hot encoding untuk kategorikal fitur. (Disini kita hanya memiliki 10 brand dan 2 OS saja)
"""

numerical_features = [
    'performance', 'internal memory', 'RAM', 'main camera',
    'selfie camera', 'battery size', 'screen size', 'weight', 'price'
]
scaler = StandardScaler()
X_numeric = scaler.fit_transform(cp_df[numerical_features])
X_categorical = pd.get_dummies(cp_df[['brand', 'operating system']])
content_df = np.hstack([X_numeric, X_categorical.values])
content_df

"""## Collaborative filtering Prep
- Menyandikan (encode) fitur user_id dan ‘cellphone_id’ ke dalam indeks integer.
- Memetakan user_id dan ‘cellphone_id’ ke dataframe yang berkaitan.
- Mengubah nilai rating menjadi float.
- Mengacak data pada rating.
- Mengubah nilai rating menjadi skala 0 hingga 1.
- Membagi data train dan validasi dengan komposisi 80:20
"""

cl_rating_df = ratings_df.copy()
# Encoding data user_id
user_ids = cl_rating_df['user_id'].unique().tolist()
print('list user_id: ', user_ids)

user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded user_id : ', user_to_user_encoded)

user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded number to user_ids: ', user_encoded_to_user)

# encoding data cellphones
cellphone_ids = cl_rating_df['cellphone_id'].unique().tolist()
print('list cellphone_ids:', cellphone_ids)

cp_to_cp_encoded = {x: i for i, x in enumerate(cellphone_ids)}
print('encoded cellphone_id : ', cp_to_cp_encoded)

cp_encoded_to_cp = {i: x for i, x in enumerate(cellphone_ids)}
print('encoded number to cellphone_id: ', cp_encoded_to_cp)

cl_rating_df['user'] = cl_rating_df['user_id'].map(user_to_user_encoded)
cl_rating_df['cellphone'] = cl_rating_df['cellphone_id'].map(cp_to_cp_encoded)

cl_rating_df['rating'] = cl_rating_df['rating'].values.astype(np.float32)

min_rating = min(cl_rating_df['rating'])

max_rating = max(cl_rating_df['rating'])

num_users = len(user_to_user_encoded)
print(num_users)

# Mendapatkan jumlah resto
num_cp = len(cp_to_cp_encoded)
print(num_cp)
print(min_rating, max_rating)

cl_rating_df = cl_rating_df.sample(frac=1, random_state=42)
cl_rating_df

x = cl_rating_df[['user', 'cellphone']].values
y = cl_rating_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.8 * cl_rating_df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

"""# Modeling and Result"""

item_features = ['model', 'brand',  'operating system', 'internal memory', 'RAM', 'performance' ,'main camera', 'selfie camera', 'battery size', 'screen size', 'weight','price']

"""## Content Filtering
- Menghitung kesamaan (similarity) mengunakan fungsi cosine_similarity.
- Membuat dataframe berdasarkan hasil cosine dengan baris dan kolom berupa model.
- Membuat fungsi untuk mencari rekomendasi.
- Melakukan percobaan pada salah satu model hp.
"""

similarity_matrix = cosine_similarity(content_df)
similarity_matrix

content_sim_df = pd.DataFrame(similarity_matrix, index=cp_df['model'], columns=cp_df['model'])
print('Shape:', content_sim_df.shape)
content_sim_df.sample(5, axis=1).sample(10, axis=0)

def cellphones_recommendations(model, similarity_data=content_sim_df, items=cp_df[item_features], k=5):
    index = similarity_data.loc[:, model].to_numpy().argpartition(range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(model, errors='ignore')
    return pd.DataFrame(closest).merge(items).head(k)

cp_df[cp_df.model.eq('iPhone 13')]

content_recom = cellphones_recommendations('iPhone 13')
content_recom

"""## Colaborative Filtering
- Membuat class RecommenderNet dengan keras Model class.
- Melakukan proses `compile` pada model.
- Melakuka proses `training`.
- Mecoba melakukan proses rekomendasi

"""

class RecommenderNet(tf.keras.Model):

  # Insialisasi fungsi
  def __init__(self, num_users, num_cp, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_cp = num_cp
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding(
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1)
    self.cp_embedding = layers.Embedding(
        num_cp,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.cp_bias = layers.Embedding(num_cp, 1)

  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0])
    user_bias = self.user_bias(inputs[:, 0])
    cp_vector = self.cp_embedding(inputs[:, 1])
    cp_bias = self.cp_bias(inputs[:, 1])

    dot_user_cp = tf.tensordot(user_vector, cp_vector, 2)

    x = dot_user_cp + user_bias + cp_bias

    return tf.nn.sigmoid(x)

model = RecommenderNet(num_users, num_cp, 50)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 8,
    epochs = 100,
    validation_data = (x_val, y_val)
)

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

user_id = ratings_df.user_id.sample(1).iloc[0]
owned_cp_by_user = ratings_df[ratings_df.user_id == user_id]

cp_not_owned = cl_rating_df[~cl_rating_df['cellphone_id'].isin(owned_cp_by_user.cellphone_id.values)]['cellphone_id']
cp_not_owned = list(
    set(cp_not_owned)
    .intersection(set(cp_to_cp_encoded.keys()))
)

cp_not_owned_encoded = [[cp_to_cp_encoded[x]] for x in cp_not_owned]
user_encoder = user_to_user_encoded.get(user_id)
user_cp_array = np.hstack(
    ([[user_encoder]] * len(cp_not_owned_encoded), cp_not_owned_encoded)
)

ratings = model.predict(user_cp_array).flatten()
top_indices = ratings.argsort()[-10:][::-1]
recommended_cellphone_ids = [
    cp_encoded_to_cp[i[0]] for i in np.array(cp_not_owned_encoded)[top_indices]
]
recommended_cellphones = cp_df[cp_df['cellphone_id'].isin(recommended_cellphone_ids)][item_features].drop_duplicates()
print(recommended_cellphones)

"""# Evaluation

## Content Filtering
- Mengunakan percision@k untuk menguji sistem dengan pre-defined ground truth.
"""

# Evaluation function: Precision@K
def precision_at_k(recommend_fn, ground_truth, k=5):
    precisions = []
    for item in ground_truth:
        recommended = recommend_fn(item)[:k]
        relevant = ground_truth[item]
        if not relevant:
            continue
        precision = len(set(recommended['model']) & set(relevant)) / k
        precisions.append(precision)
    return sum(precisions) / len(precisions)

ground_truth = {
    'iPhone SE (2022)': ['iPhone 13 Mini', 'iPhone 13', 'iPhone XR'],
    'iPhone 13 Mini': ['iPhone SE (2022)', 'iPhone 13', 'Galaxy S22'],
    'iPhone 13': ['iPhone 13 Mini', 'iPhone SE (2022)', 'iPhone XR'],
    'iPhone 13 Pro': ['iPhone 13', 'Xperia Pro', 'iPhone 13 Pro Max'],
    'iPhone 13 Pro Max': ['Galaxy Z Fold 3', 'Xperia Pro', 'iPhone 13 Pro']
}


precision_score = precision_at_k(cellphones_recommendations, ground_truth, k=5)
precision_score

"""## Colaborative filtering
- Menggunakan rmse untuk mengevaluasi model, disini menggunakan nilai rmse pada epoch terakhir pada saat traning.
"""

last_val_rmse = history.history['val_root_mean_squared_error'][-1]
print(f"Last validation RMSE: {last_val_rmse:.4f}")