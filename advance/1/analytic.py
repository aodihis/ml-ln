#!/usr/bin/env python
# coding: utf-8

# In[1]:


from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# # Data Loading
# - Menggunakan dataset : joebeachcapital/diabetes-factors
# - Dataset diunduh dari Kaggle dan disimpan di dalam folder data.
# - Selanjutnya, data dibaca menggunakan pandas dan ditampilkan beberapa baris pertama untuk mendapatkan gambaran awal isi dataset.

# In[2]:


api = KaggleApi()
api.authenticate()
api.dataset_download_files('joebeachcapital/diabetes-factors', path='data', unzip=True)


# In[3]:


df = pd.read_csv('data/diabetes-vid.csv')
df


# Dataset ini terdiri dari total 768 baris data, dengan kolom-kolom sebagai berikut:
# Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, dan Outcome.

# # Exploratory Data Analysis

# ### Variable Info
# - Melihat jenis-jenis fitur yang ada pada dataset.
# - Menampilkan statistik deskriptif.

# In[4]:


df.info()


# Jenis kolom pada dataset tersebut adalah sebagai berikut:
# 1. Pregnancies (int64) :Jumlah kehamilan yang pernah dialami oleh pasien.
# 2. Glucose (int64): Konsentrasi glukosa plasma dua jam dalam tes toleransi glukosa oral.
# 3. BloodPressure (int64): Tekanan darah diastolik (mm Hg).
# 4. SkinThickness (int64): Ketebalan lipatan kulit triceps (dalam mm), sebagai indikator lemak tubuh.
# 5. Insulin (int64): Kadar insulin serum 2 jam (mu U/ml).
# 6. BMI (float64): Indeks massa tubuh (berat badan dalam kg dibagi tinggi badan dalam m²).
# 7. DiabetesPedigreeFunction (float64): Fungsi silsilah genetik yang menunjukkan kemungkinan keturunan diabetes berdasarkan riwayat keluarga.
# 8. Age (int64):Usia pasien (dalam tahun).
# 9. Outcome (object): Status akhir berupa dead atau alive

# In[5]:


df.describe()


# ### Handling Missing Values
# - Melakukan pengecekan terhadap missing values pada dataset
# - Melakukan penangana terhadap missing values jika ada

# In[6]:


df.isnull().sum()


# In[7]:


(df == 0).sum()


# In[8]:


df.drop(columns=['Insulin', 'SkinThickness'], inplace=True)
df = df[df['BMI'] != 0]
df = df[df['Glucose'] != 0]
df = df[df['BloodPressure'] != 0]
df.shape


# Tidak terdapat data kosong pada dataset. Namun terdapat data dengan nilai 0 pada beberapa fitur.
# - Hapus fitur Insulin dan SkinThickness, kerena memiliki data tidak valid dengan jumlah besar.
# - Hapus baris yang memiliki nilai 0 untuk BMI, Glucose dan BloodPressure.

# ### Handling Outliers
# - Menampilkan boxplot untuk melihat outlier untuk setiap fitur numerik
# 

# In[9]:


numeric_cols = df.select_dtypes(include='number').columns
target = df['Outcome']
nrows = (len(numeric_cols) + 1)//2
fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(8, 6))
ax = ax.flatten()
for i, col in enumerate(numeric_cols):
    sns.boxplot(x=df[col], ax=ax[i])
    ax[i].set_title(col)
    ax[i].set_xlabel(col)
plt.tight_layout()
plt.show()


# In[10]:


Q1 = df['BloodPressure'].quantile(0.25)
Q3 = df['BloodPressure'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['BloodPressure'] >= lower_bound)]


# In[11]:


df.shape


# Di sini, kita dapat melihat adanya outlier pada beberapa fitur dalam data. Namun, sebagian besar outlier tetap dipertahankan karena dianggap sebagai data valid yang mengandung informasi penting dan dapat berpengaruh terhadap variabel target.
# 
# Pengecualian dilakukan pada fitur BloodPressure, di mana nilai-nilai yang berada di bawah kuartil pertama (Q1) dianggap tidak wajar, sehingga dihapus dari dataset.

# ### Univariate  Analysis

# In[12]:


df.hist(bins=50, figsize=(20,15))
plt.show()


# - Kehamilan: Sebagian besar wanita memiliki jumlah kehamilan antara 0–3 kali.
# - Glukosa: Mayoritas nilai glukosa berada di kisaran 100–130 dengan beberapa nilai tinggi.
# - Tekanan Darah: Nilai tekanan darah umumnya berkisar antara 60–80.
# - BMI: Sebagian besar BMI berada di rentang 25–35, menunjukkan banyak yang overweight atau obesitas.
# - Distribusi usia condong ke kiri dengan dominasi usia 20-30 tahun.
# - Distribusi DiabetesPedigreeFunction sangat condong ke kiri (right-skewed), dengan sebagian besar nilai berada antara 0.0 dan 0.5.
# 
# 
# 

# ### Mulivariate  Analysis

# In[13]:


nrows = ((len(numeric_cols)+1)//2)
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10,6))  # smaller figure size
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.barplot(x='Outcome', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'Average {col} by Outcome')
    axes[i].set_xlabel('Outcome')
    axes[i].set_ylabel(col)

plt.tight_layout()
plt.show()


# In[14]:


nrows = ((len(numeric_cols)+1)//2)
fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(10,6))  # smaller figure size
axes = axes.flatten()
for i,col in enumerate(numeric_cols):
    sns.boxplot(x='Outcome', y=col, data=df, ax=axes[i])
    axes[i].set_title(f'Average {col} by Outcome')
    axes[i].set_xlabel('Outcome')
    axes[i].set_ylabel(col)
plt.tight_layout()
plt.show()


# - Glucose: Median pada pasien meninggal lebih tinggi dibandingkan yang masih hidup. Namun terdapat beberapa outlier pada passien yang masih hidup.
# - BMI: Median BMI pada pasien meniggal juga terlihat sedikit lebih tinggi dibandingkan yang hidup.
# - Pregnancies: Median jumlah kehamilan sedikit lebih tinggi pada pasien yang meninggal, tetapi distribusinya relatif mirip antara kedua kelompok.
# - BloodPressure: Median kelompok yang masih hidup sedikit lebih tinggi, namun tidak menunjukkan perbedaan yang signifikan.
# - DiabetesPedigreeFunction: Distribusi antar kelompok hampir sama; perbedaan tidak mencolok, tetapi tetap relevan sebagai indikator faktor genetik.
# - Age: Pasien yang meninggal cenderung berusia lebih tua dibandingkan yang masih hidup, menjadikan usia sebagai salah satu fitur penting dalam klasifikasi.

# In[15]:


plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix ", size=20)
plt.show()


# Tidak ada fitur yang memiliki nilai korelasi yang sangat tinggi.

# ## Data Preparation
# - Binning fitur Age dan BMI.
# - Men-encode Outcome menjadi 0 dan  1.
# - Menentuka fitur yang digunakan untuk model, disini kita tidak akan menggunakan fitur `BloodPressure`.
# - Melakukan resampling menggunakan SMOTE untuk mengatasi ketidakseimbangan data.
# - Memisahkan data menjadi training dan test.
# - Melakukan standarisasi pada data training dan test.

# In[16]:


from sklearn.preprocessing import LabelEncoder

bins = [0, 20, 30, 40, 50, 60, 999]
labels = ['<20', '21–30', '31–40', '41–50', '51–60', '60+']
age_encoder = LabelEncoder()
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
df['AgeGroupEncoded'] = age_encoder.fit_transform(df['AgeGroup'])

bins = [0, 18.5, 24.9, 29.9, 100]
labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
bmi_encoder = LabelEncoder()
df['BMICategory'] = pd.cut(df['BMI'], bins=bins, labels=labels)
df['BMICategoryEncoded'] = bmi_encoder.fit_transform(df['BMICategory'])


# In[17]:


status = {'alive': 0, 'dead': 1}
reversed_status = {0: 'alive', 1: 'dead'}
df['Outcome'] = df['Outcome'].map(status)
df


# In[18]:


feature_cols = [
    'Pregnancies', 'Glucose',
    'DiabetesPedigreeFunction',
    'AgeGroupEncoded', 'BMICategoryEncoded'
]


# In[19]:


from imblearn.over_sampling import SMOTE
df_model = df[feature_cols + ['Outcome']].copy()
X = df_model.drop('Outcome', axis=1)
y = df_model['Outcome']

smote = SMOTE()
X_resampled,y_resampled=smote.fit_resample(X,y,)
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2,random_state=123)


# In[20]:


scaler = StandardScaler()
X_train[feature_cols] = scaler.fit_transform(X_train.loc[:, feature_cols])
X_train[feature_cols].head()


# In[21]:


X_test[feature_cols] = scaler.transform(X_test.loc[:, feature_cols])
X_test[feature_cols].head()


# ## Modeling
# - Membuat model Random Forest dengan bantuan GridSearch untuk Hypertunning.

# In[22]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 9],
    'min_samples_leaf': [2, 9],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, n_jobs=-1, scoring='precision')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)


# In[23]:


best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)


# ## Evaluation
# - Menghitung nilai akurasi, presisi, recall, dan f1 score.

# In[24]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# In[25]:


predict = X_test.iloc[7:15].copy()
pred_dict = {'y_true': y_test[7:15], 'y_pred': best_rf.predict(predict)}

pd.DataFrame(pred_dict)

