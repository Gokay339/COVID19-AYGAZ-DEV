# KÜTÜPHANELERİMİZ

import joblib
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
# VERİ SETİ HAKKINDA BİLGİLER
"""
1. Veri Setini Ekleme: Verimi yükledim ve analize başlamak için hazır hale getirdim.
2. Eksik Verileri Ekleme: Eksik veriler olup olmadığını kontrol ettim ve bunları tamamlamak için uygun yöntemleri (silme veya ortalama ile doldurma) kullandım.
3. grabcolnames Fonksiyonu ile Kategorik ve Sayısal Verileri Bulma: Veri setindeki sayısal ve kategorik değişkenleri belirlemek için bu fonksiyonu kullandım.
4. NaN Verilere Bakma: Veri setimde NaN değerlerini kontrol ettim ve duruma göre bu değerleri ya sildim ya da ortalama değerlerle doldurdum.
5. catsummary ve numsummary Değerlerine Bakma: Kategorik ve sayısal veriler için özet istatistiklere göz attım ve önemli bilgiler edindim.
6. Korelasyona Bakma: Değişkenler arasındaki korelasyonu inceledim ve hangi değişkenlerin daha güçlü ilişkiler sunduğunu değerlendirdim.
7. Aykırı Değerlere Bakma ve Gerekli İşlemleri Yapma: Veri setimdeki aykırı verilere göz attım ve gerekirse bu değerleri düzelttim.
8. OneHotEncoder ve StandardScaler Uygulama: Kategorik verileri one-hot encoding ile dönüştürdüm ve sayısal verileri ölçeklendirmek için StandardScaler kullandım.
9. Modeli Oluşturma: Veri setimi hazırladıktan sonra bir model oluşturup eğittim."""
---




# ***1.DENEME - BOŞ VERİLERE ORTALAMA DEĞER VEREREK DOLDURMA***

df = pd.read_csv("/kaggle/input/covid-19/Covid Data.csv",
                 nrows=200000)  # Yaklaşık 1m satır var ama ben ilk 200k satırı aldım
df.head(5)


# Bu kod ile kategorik , sayısal ve kategorik gibi gözüken sayısal verileri buluyoruz
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
"""
1.
cat_cols
Kategorik(string / object
tipi) sütunları
içerir.

Örneğin, "City"
sütunu(Ankara, İstanbul, İzmir)
ve
"Gender"
sütunu(Male, Female).

2.
num_but_cat
Sayısal
olup, benzersiz
değer
sayısı
düşük
olduğu
için
kategorik
gibi
davranan
sütunları
içerir.

Örneğin, "Age_Group"
sütunu(1, 2, 3
grupları) veya
"Binary_Flag"
sütunu(0
ve
1
değerleri).

3.
cat_but_car
Kategorik
görünümlü(object
tipi) ancak
benzersiz
değer
sayısı
yüksek
olduğu
için
kardinal
olan
sütunları
içerir..

Örneğin, "Customer_ID"
sütunu(her
müşteri
için
eşsiz
bir
ID) veya
"Invoice_Number"
sütunu(her
fatura
için
benzersiz
bir
numara).

4.
num_cols
Sayısal
sütunları
içerir, ancak
sayısal
olup
kategorik
davranan
sütunları
çıkarır.

Örneğin, "Salary"
sütunu(5000, 7000, 10000)
ve
"Age"
sütunu(25, 30, 35)."""

# NAN VERİ EKLEYELİM
import pandas as pd
import numpy as np
import random


def add_random_missing_values(dataframe: pd.DataFrame,
                              missing_rate: float = 0.05) -> pd.DataFrame:
    """Turns random values to NaN in a DataFrame.

    To use this function, you need to import pandas, numpy and random libraries.

    Args:
        dataframe (pd.DataFrame): DataFrame to be processed.
        missing_rate (float): Percentage of missing value rate in float format. Defaults 0.05

    Returns:
        df_missing (pd.DataFrame): Processed DataFrame object.

    """
    # Get copy of dataframe
    df_missing = dataframe.copy()

    # Obtain size of dataframe and number total number of missing values
    df_size = dataframe.size
    num_missing = int(df_size * missing_rate)

    # Get random row and column indexes to turn them NaN
    for _ in range(num_missing):
        row_idx = random.randint(0, dataframe.shape[0] - 1)
        col_idx = random.randint(0, dataframe.shape[1] - 1)

        df_missing.iat[row_idx, col_idx] = np.nan

    return df_missing


df = add_random_missing_values(df, missing_rate=0.05)
print(df.isnull().sum())

from sklearn.impute import SimpleImputer

# Sayısal verileri doldurmak için median yöntemini kullanıyoruz
num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Kategorik verileri doldurmak içi most-frequent yöntemini kullanıyoruz
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Tarih verileri doldurmak içi most-frequent yöntemini kullanıyoruz
cat_but_car_imputer = SimpleImputer(strategy="most_frequent")
df[cat_but_car] = cat_but_car_imputer.fit_transform(df[cat_but_car])


# KATEGORİK DEĞİŞKENİN SINIF FREKANSINI VE SINIFLARIN ORANINI VERİR , PLOT TRUE OLURSA GRAFİK VERİR
# KATEGORİK DEĞİŞKENLERİ ÖZETLEMEK İÇİN KULLANILIR
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# SAYISAL DEĞİŞKENLERİ ÖZETLEMEK İÇİN KULLANILIR
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# Hedef değişken (target) ile sayısal bir değişken (numerical_col) arasındaki ilişkiyi özetler.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Kategorik bir sütundaki her bir kategori için hedef değişkenin ortalamasını
# hesaplayarak hedef değişken ile kategorik değişken arasındaki ilişkiyi özetler.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


# Sayısal değişkenlerin incelenmesi
for col in num_cols:
    num_summary(df[num_cols], col, plot=True)

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df[cat_cols], col)


# Korelasyonlarına Bakalım
def correlation_matrix(df, cols):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu")
    plt.show()


correlation_matrix(df, num_cols)

# Veri tiplerine bakalım
print(df.dtypes)

---


# AYKIRI DEĞERLER

# aykırı değerlerin sınır değerlerini bulmaya yarar
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)  # Quartile 1 (Q1)
    quartile3 = dataframe[col_name].quantile(q3)  # Quartile 3 (Q3)
    interquartile_range = quartile3 - quartile1  # IQR (Interquartile Range)

    up_limit = quartile3 + 1.5 * interquartile_range  # Upper limit
    low_limit = quartile1 - 1.5 * interquartile_range  # Lower limit

    return low_limit, up_limit


# aykırı değerleri sınır değerine eşitler
def replace_with_thresholds(dataframe, veriable):
    low_limit, up_limit = outlier_thresholds(dataframe, veriable)
    dataframe.loc[(dataframe[veriable] < low_limit), veriable] = low_limit
    dataframe.loc[(dataframe[veriable] > up_limit), veriable] = up_limit


# aykırı değer var mı ? Kontrol edelim
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Aykırı değer var mı ?  kontrol edelim
for col in num_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))

replace_with_thresholds(df, "AGE")
# Aykırı değerleri sınır değerlerine eşitleyelim - istersek silebiliriz ama ben sınıra eşitmeke istedim

# Aykırı değer var mı ?  kontrol edelim
for col in num_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))

# Aykırı değer var mı ?  kontrol edelim
for col in cat_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))

replace_with_thresholds(df, "MEDICAL_UNIT")
replace_with_thresholds(df, "COPD")
replace_with_thresholds(df, "PNEUMONIA")
replace_with_thresholds(df, "DIABETES")
replace_with_thresholds(df, "ASTHMA")
replace_with_thresholds(df, "INMSUPR")
replace_with_thresholds(df, "HIPERTENSION")  # AYKIRI DEĞERLERİ SINIR DEĞERİNE EŞİTLİYOR
replace_with_thresholds(df, "OTHER_DISEASE")
replace_with_thresholds(df, "CARDIOVASCULAR")
replace_with_thresholds(df, "OBESITY")
replace_with_thresholds(df, "RENAL_CHRONIC")
replace_with_thresholds(df, "TOBACCO")

for col in cat_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))

---
# AYKIRI DEĞERLERİ HALLETTİK ŞİMDİ NORMAL İŞLEMLERE GEÇEBİLİRİZ

# Classification datamızda birsürü değişken var ve bu yüzden sınıflandırma yapamayız ancak tahmin işlemleri yaparız
# Bunu düzeltmek için yeni değişken oluşturalım
# 1: COVID Pozitif (1, 2, 3)
# 0: COVID Negatif/Belirsiz (4 ve üzeri)

df["COVID_STATUS"] = np.where(df["CLASIFFICATION_FINAL"] <= 3, 1, 0)
# Anlamı 3 veya daha küçükse 1 atanır  ama 4 veya daha büyükse 0 atanır
# COVID_STATUS değişkenini kontrol edelim
print(df['COVID_STATUS'].value_counts())


# Onehot encoder amacı kategorik verileri sayısallaştırmak
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    # One-Hot Encoding işlemi
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)
df = one_hot_encoder(df, cat_but_car, drop_first=True)

# StandardScaler'ın amacı, sayısal verilerin dağılımlarını standartlaştırmak yani ortalama (mean) ve standart sapma (standard deviation)
# kullanarak verileri standart normal dağılıma dönüştürür
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=df[num_cols].columns, index=df.index)

# Numy dizisini , Pandas DataFrame'e dönüştürürken orijinal satır indekslerini korumak için index=df.index parametresini kulanabiliriz

print(df['COVID_STATUS'].value_counts())  # eşsiz değerlerin sayısına bakalım

print(df.isnull().sum())  # Boş veri var mı bakalım

print(df.nunique())  # Eşsiz değerlerin sayısına bakalım

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

X = df.drop(["COVID_STATUS"], axis=1)
y = df["COVID_STATUS"]

model = LogisticRegression(max_iter=500)  # İterasyon modelin parametrelerini (ağırlıklarını) güncelleme süreci
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print("Boş Verilerin Yerine Ortalama Değerler İle Doldurduğumuzda Çıkan Sonuç : ", np.mean(cv_scores))

---
---
---

# ***2.DENEME - BOŞ VERİLERİ SİLEREK İŞLEM YAPMA***

df1 = pd.read_csv("/kaggle/input/covid-19/Covid Data.csv",
                  nrows=200000)  # Yaklaşık 1m satır var ama ben ilk 200k satırı aldım
df1.head(5)


# Bu kod ile kategorik , sayısal ve kategorik gibi gözüken sayısal verileri buluyoruz
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df1, cat_th=5, car_th=20)


# Nan verileri ekleyelim
def add_random_missing_values(dataframe: pd.DataFrame,
                              missing_rate: float = 0.05) -> pd.DataFrame:
    """Turns random values to NaN in a DataFrame.

    To use this function, you need to import pandas, numpy and random libraries.

    Args:
        dataframe (pd.DataFrame): DataFrame to be processed.
        missing_rate (float): Percentage of missing value rate in float format. Defaults 0.05

    Returns:
        df_missing (pd.DataFrame): Processed DataFrame object.

    """
    # Get copy of dataframe
    df_missing = dataframe.copy()

    # Obtain size of dataframe and number total number of missing values
    df_size = dataframe.size
    num_missing = int(df_size * missing_rate)

    # Get random row and column indexes to turn them NaN
    for _ in range(num_missing):
        row_idx = random.randint(0, dataframe.shape[0] - 1)
        col_idx = random.randint(0, dataframe.shape[1] - 1)

        df_missing.iat[row_idx, col_idx] = np.nan

    return df_missing


df1 = add_random_missing_values(df1, missing_rate=0.05)
print(df1.isnull().sum())

# BOŞ VERİLERİ EKLEDİK AMA BEN SİLEREK DENEMEK İSTİYORUM
# Yalnızca özelliklerdeki eksik verileri silme
# COVID_STATUS haricindeki tüm sütunlarda eksik verileri silme
df1 = df1.dropna()


# KATEGORİK DEĞİŞKENİN SINIF FREKANSINI VE SINIFLARIN ORANINI VERİR , PLOT TRUE OLURSA GRAFİK VERİR
# KATEGORİK DEĞİŞKENLERİ ÖZETLEMEK İÇİN KULLANILIR
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# SAYISAL DEĞİŞKENLERİ ÖZETLEMEK İÇİN KULLANILIR
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# Hedef değişken (target) ile sayısal bir değişken (numerical_col) arasındaki ilişkiyi özetler.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Kategorik bir sütundaki her bir kategori için hedef değişkenin ortalamasını
# hesaplayarak hedef değişken ile kategorik değişken arasındaki ilişkiyi özetler.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in num_cols:
    num_summary(df1[num_cols], col, plot=True)

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df1[cat_cols], col)


# Korelasyonlarına Bakalım
def correlation_matrix(df, cols):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu")
    plt.show()


correlation_matrix(df1, num_cols)

print(df1.isnull().sum())

---


# AYKIRI DEĞERLERE BAKALIM TEKRARDAN

# aykırı değerlerin sınır değerlerini bulmaya yarar
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)  # Quartile 1 (Q1)
    quartile3 = dataframe[col_name].quantile(q3)  # Quartile 3 (Q3)
    interquartile_range = quartile3 - quartile1  # IQR (Interquartile Range)

    up_limit = quartile3 + 1.5 * interquartile_range  # Upper limit
    low_limit = quartile1 - 1.5 * interquartile_range  # Lower limit

    return low_limit, up_limit


# aykırı değerleri sınır değerine eşitler
def replace_with_thresholds(dataframe, veriable):
    low_limit, up_limit = outlier_thresholds(dataframe, veriable)
    dataframe.loc[(dataframe[veriable] < low_limit), veriable] = low_limit
    dataframe.loc[(dataframe[veriable] > up_limit), veriable] = up_limit


# aykırı değer var mı ? Kontrol edelim
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Aykırı değer var mı ?  kontrol edelim
for col in num_cols:
    print(col, check_outlier(df1, col, 0.25, 0.75))

# Aykırı değer var mı ?  kontrol edelim
for col in cat_cols:
    print(col, check_outlier(df1, col, 0.25, 0.75))

replace_with_thresholds(df1, "AGE")

replace_with_thresholds(df1, "MEDICAL_UNIT")
replace_with_thresholds(df1, "COPD")
replace_with_thresholds(df1, "PNEUMONIA")
replace_with_thresholds(df1, "DIABETES")
replace_with_thresholds(df1, "ASTHMA")
replace_with_thresholds(df1, "INMSUPR")
replace_with_thresholds(df1, "HIPERTENSION")  # AYKIRI DEĞERLERİ SINIR DEĞERİNE EŞİTLİYOR
replace_with_thresholds(df1, "OTHER_DISEASE")
replace_with_thresholds(df1, "CARDIOVASCULAR")
replace_with_thresholds(df1, "OBESITY")
replace_with_thresholds(df1, "RENAL_CHRONIC")
replace_with_thresholds(df1, "TOBACCO")

for col in cat_cols:
    print(col, check_outlier(df1, col, 0.25, 0.75))

---


# Kategorik verileri sayısallaştıralım
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df1 = one_hot_encoder(df1, cat_cols, drop_first=True)
df1 = one_hot_encoder(df1, cat_but_car, drop_first=True)

from sklearn.preprocessing import StandardScaler

# Kısaca sayısal verilerin alanlarını küçültmeeye yarar ki daha iyi sınıflandırma yapabilelim
X_scaled = StandardScaler().fit_transform(df1[num_cols])
df1[num_cols] = pd.DataFrame(df1, columns=df1[num_cols].columns, index=df1.index)
#   pd.DataFrame(df1  buraya X_scaled yazarsam NaN veriler çıkıyor ama df1 yazınca çıkmıyor nedeninin anlayamadım
#   dropna ile sildiğim için yapıyor sanırım


# index=df1.index: DataFrame'in indekslerini koruyarak hatalı bir eşleştirme olmasını önler
# çünkü bu kod olmazsa yeni boş veriler çıkıyor

print(df1[num_cols].isnull().sum())  # Numarik sütunlardaki eksik verileri kontrol et

# CLASIFFICATION_FINAL sütununu kullanarak COVID_STATUS hedef değişkenini oluşturma
df1["COVID_STATUS"] = np.where(df1["CLASIFFICATION_FINAL"] <= 3, 1,
                               0)  # ANLAMI 1 VE 3 ARASINDA OLMAYAN DEĞERLER 0 OLARAK ATANICAK
# 1: COVID Pozitif (1, 2, 3)
# 0: COVID Negatif/Belirsiz (4 ve üzeri)

# eşsiz değerlerin sayısına bakalım
print(df1["COVID_STATUS"].value_counts())

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

X = df1.drop(["COVID_STATUS"], axis=1)
y = df1["COVID_STATUS"]
model = LogisticRegression(max_iter=500)
cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print("Verileri Sildiğimiz Zaman Çıkan Sonuç :", np.mean(cv_scores))

#

---
---
# **VERİ SETİMİZDEKİ VERİLERİN ORTALAMA DEĞERLERİNİ ALARAK VE**
# **BOŞ DEĞERLERİNİ SİLEREK 2 MODEL YAPTIK**

# **ŞİMDİ ANA MODELİMİZE GEÇEBİLİRİZ**

df2 = pd.read_csv("/kaggle/input/covid-19/Covid Data.csv",
                  nrows=5000)  # Yaklaşık 1m satır var ama ben ilk 20bin satırı aldım
df2.head(5)

# VERİSETİNDEN 5000 SATIR ALIYORUM

"""5000
ALMAMIN
SEBEBİ
BU
MODELDE
3 - 4
FARKLI
MODELİN
SONUÇLARINI
ALIYORUM
"""

# Bu kod ile kategorik , sayısal ve kategorik gibi gözüken sayısal verileri buluyoruz
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df2, cat_th=5, car_th=20)

# NAN VERİ EKLEYELİM
import pandas as pd
import numpy as np
import random


def add_random_missing_values(dataframe: pd.DataFrame,
                              missing_rate: float = 0.05) -> pd.DataFrame:
    """Turns random values to NaN in a DataFrame.

    To use this function, you need to import pandas, numpy and random libraries.

    Args:
        dataframe (pd.DataFrame): DataFrame to be processed.
        missing_rate (float): Percentage of missing value rate in float format. Defaults 0.05

    Returns:
        df_missing (pd.DataFrame): Processed DataFrame object.

    """
    # Get copy of dataframe
    df_missing = dataframe.copy()

    # Obtain size of dataframe and number total number of missing values
    df_size = dataframe.size
    num_missing = int(df_size * missing_rate)

    # Get random row and column indexes to turn them NaN
    for _ in range(num_missing):
        row_idx = random.randint(0, dataframe.shape[0] - 1)
        col_idx = random.randint(0, dataframe.shape[1] - 1)

        df_missing.iat[row_idx, col_idx] = np.nan

    return df_missing


df2 = add_random_missing_values(df2, missing_rate=0.05)
print(df2.isnull().sum())

df2.head(15)

from sklearn.impute import SimpleImputer

# Sayısal veriler için SimpleImputer kullanarak medyan ile doldurma
num_imputer = SimpleImputer(strategy="median")
df2[num_cols] = num_imputer.fit_transform(df2[num_cols])

# Kategorik veriler için SimpleImputer kullanarak mod ile doldurma
cat_imputer = SimpleImputer(strategy="most_frequent")
df2[cat_cols] = cat_imputer.fit_transform(df2[cat_cols])

# Tarih verileri doldurmak içi most-frequent yöntemini kullanıyoruz
cat_but_car_imputer = SimpleImputer(strategy="most_frequent")
df2[cat_but_car] = cat_but_car_imputer.fit_transform(df2[cat_but_car])


# KATEGORİK DEĞİŞKENİN SINIF FREKANSINI VE SINIFLARIN ORANINI VERİR , PLOT TRUE OLURSA GRAFİK VERİR
# KATEGORİK DEĞİŞKENLERİ ÖZETLEMEK İÇİN KULLANILIR
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


# SAYISAL DEĞİŞKENLERİ ÖZETLEMEK İÇİN KULLANILIR
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# Hedef değişken (target) ile sayısal bir değişken (numerical_col) arasındaki ilişkiyi özetler.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Kategorik bir sütundaki her bir kategori için hedef değişkenin ortalamasını
# hesaplayarak hedef değişken ile kategorik değişken arasındaki ilişkiyi özetler.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in num_cols:
    num_summary(df2[num_cols], col, plot=True)

# Kategorik değişkenlerin incelenmesi
for col in cat_cols:
    cat_summary(df2[cat_cols], col)


# Korelasyonlarına Bakalım
def correlation_matrix(df, cols):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="RdBu")
    plt.show()


correlation_matrix(df2, num_cols)

---


# AYKIRI DEĞERLERE BAKALIM

# aykırı değerlerin sınır değerlerini bulmaya yarar
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)  # Quartile 1 (Q1)
    quartile3 = dataframe[col_name].quantile(q3)  # Quartile 3 (Q3)
    interquartile_range = quartile3 - quartile1  # IQR (Interquartile Range)

    up_limit = quartile3 + 1.5 * interquartile_range  # Upper limit
    low_limit = quartile1 - 1.5 * interquartile_range  # Lower limit

    return low_limit, up_limit


# aykırı değerleri sınır değerine eşitler
def replace_with_thresholds(dataframe, veriable):
    low_limit, up_limit = outlier_thresholds(dataframe, veriable)
    dataframe.loc[(dataframe[veriable] < low_limit), veriable] = low_limit
    dataframe.loc[(dataframe[veriable] > up_limit), veriable] = up_limit


# aykırı değer var mı ? Kontrol edelim
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


# Aykırı değer var mı ?  kontrol edelim
for col in num_cols:
    print(col, check_outlier(df2, col, 0.25, 0.75))

# Aykırı değer var mı ?  kontrol edelim
for col in cat_cols:
    print(col, check_outlier(df2, col, 0.25, 0.75))

replace_with_thresholds(df2, "AGE")

replace_with_thresholds(df2, "MEDICAL_UNIT")
replace_with_thresholds(df2, "COPD")
replace_with_thresholds(df2, "PNEUMONIA")
replace_with_thresholds(df2, "DIABETES")
replace_with_thresholds(df2, "ASTHMA")
replace_with_thresholds(df2, "INMSUPR")
replace_with_thresholds(df2, "HIPERTENSION")  # AYKIRI DEĞERLERİ SINIR DEĞERİNE EŞİTLİYOR
replace_with_thresholds(df2, "OTHER_DISEASE")
replace_with_thresholds(df2, "CARDIOVASCULAR")
replace_with_thresholds(df2, "OBESITY")
replace_with_thresholds(df2, "RENAL_CHRONIC")
replace_with_thresholds(df2, "TOBACCO")

for col in cat_cols:
    print(col, check_outlier(df2, col, 0.25, 0.75))

---
# MODELE DÖNELİM

# CLASIFFICATION_FINAL sütununu kullanarak COVID_STATUS hedef değişkenini oluşturma
# 1: COVID Pozitif (1, 2, 3)
# 0: COVID Negatif/Belirsiz (4 ve üzeri)

df2["COVID_STATUS"] = np.where(df2["CLASIFFICATION_FINAL"] <= 3, 1,
                               0)  # ANLAMI 1 VE 3 ARASINDA OLMAYAN DEĞERLER 0 OLARAK ATANICAK

# COVID_STATUS değişkenini kontrol etmek için
print(df2['COVID_STATUS'].value_counts())


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df2 = one_hot_encoder(df2, cat_cols, drop_first=True)
df2 = one_hot_encoder(df2, cat_but_car, drop_first=True)

X_scaled = StandardScaler().fit_transform(df2[num_cols])
df2[num_cols] = pd.DataFrame(X_scaled, columns=df2[num_cols].columns, index=df2.index)

print(df2['COVID_STATUS'].value_counts())

print(df2.isnull().sum())

y = df2["COVID_STATUS"]
X = df2.drop(["COVID_STATUS"], axis=1)


def base_models(X, y, scoring="roc_auc"):
    print("Modeller İşleniyor Lütfen Bekleyin...")
    classifiers = [("LR", LogisticRegression()),
                   ("KNN", KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("RF", RandomForestClassifier()),
                   ("Adaboost", AdaBoostClassifier()),
                   ("GBM", GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss'))]
    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=10, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")


# MODELLERİ HİPERPARAMETRE EDELİM
# VEYA BUNUN YERİNE GRİDSEARCH GİBİ ALGORİTMALARI KULLANABİLİRİZ

# log_params = {
#    "penalty": ["l1", "l2", "elasticnet"],  # Cezalandırma türleri (overfitting engellemek amacıyla kullanılır)
#    "C": [0.01, 0.1, 1, 10],                 # Regularizasyon güçleri - Model, veriye tam uyum sağlamaktan kaçınır, böylece overfitting (aşırı uyum) riski azalır.
#    "solver": ["lbfgs", "liblinear", "saga"],      # Optimizasyon çözücüleri - modelin optimizasyon algoritmasını belirler
#    "max_iter": [500]   }           # Maksimum iterasyon sayıları - modelin maksimum iterasyon sayısını belirtir


knn_params = {"n_neighbors": range(2, 30)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [  # ("LR",LogisticRegression(),log_params),
    ("KNN", KNeighborsClassifier(), knn_params),
    ("CART", DecisionTreeClassifier(), cart_params),
    ("RF", RandomForestClassifier(), rf_params),
    ("XGBoost", XGBClassifier(eval_metric='logloss'), xgboost_params),
    # ('LightGBM', LGBMClassifier(), lightgbm_params)
]


def hyperparameter_optimization(X, y, cv=10, scoring="roc_auc"):
    print("Hiperparametre Optimize Ediliyor....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Önce): {round(cv_results['test_score'].mean(), 4)}")

        # Hiperparametre optimizasyonu: Modelin performansını artırmak için belirlenen params (parametre aralıkları) üzerinde bir arama yapar.
        # En iyi parametre kombinasyonunu belirler ve modeli bu parametrelerle eğitir.
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Sonra): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


def voting_classifier(best_models, X, y):
    print("Modellerin Tahminlerini Birleştirerek Toplu Tahmin Yapabiliriz...")
    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]), ('RF', best_models["RF"]),
                                              # ('LightGBM', best_models["LightGBM"])
                                              ],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf


base_models(X, y)
best_models = hyperparameter_optimization(X, y)
voting_clf = voting_classifier(best_models, X, y)

---
---
# OVERFİTTİNG VAR MI EMİN OLMAK İÇİN UFAK BİR MODEL YAPIYORUM
# HEM EĞİTİM HEM TEST VERİLERİNE BAKIYORUM

# Crossvalidate gibi yapılarla overfitting önleyebiliriz ama yine de emin olmak için böyle küçük birşey yapıyorum

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

model = RandomForestClassifier()

cv_train_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
print(f"CV Doğruluk (Eğitim): {cv_train_accuracy.mean() * 100}")

cv_test_accuracy = cross_val_score(model, X_test, y_test, cv=cv, scoring="accuracy")
print(f"CV Doğruluk (Test): {cv_test_accuracy.mean() * 100}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Doğruluk: {test_accuracy * 100}")

# **Tahmin Yapmak İçin Böyle Birşey Yaptım**

import warnings

warnings.filterwarnings("ignore")

# Modelleri böyle eğiterek sonuçlarını öğrenmek istiyorum

best_models["KNN"].fit(X, y)  # KNN modelini eğitiyoruz
best_models["CART"].fit(X, y)  # CART modelini de eğitiyoruz
best_models["XGBoost"].fit(X, y)  # XGBoost modelni de eğitiyoruz

# 0: COVID hastalığı taşıyor.
# 1 ve üzeri: COVID taşımıyor veya testin kesin olmayan bir sonucu.


# 50. kişinin verilerini alalım
X_50 = X.iloc[49, :].values.reshape(1, -1)  # 50. kişi (reshape ile 2D hale getiriyoruz)

# Voting Classifier tahmini yapalım
voting_prediction = voting_clf.predict(X_50)
print(f"50. kişinin COVID durumu tahmini (Voting Classifier): {voting_prediction[0]}")

# KNN modeline göre tahmin yapalım
knn_prediction = best_models["KNN"].predict(X_50)
print(f"50. kişinin COVID durumu tahmini (KNN modeline göre): {knn_prediction[0]}")

# CART modeline göre tahmin yapalım
cart_prediction = best_models["CART"].predict(X_50)
print(f"50. kişinin COVID durumu tahmini (CART modeline göre): {cart_prediction[0]}")

# XGBoost modeline göre tahmin yapalım
xgboost_prediction = best_models["XGBoost"].predict(X_50)
print(f"50. kişinin COVID durumu tahmini (XGBoost modeline göre): {xgboost_prediction[0]}")

print("""\n\n\n 0: COVID hastalığı taşıyor.
 1 ve üzeri: COVID taşımıyor veya testin kesinlik sonucu yok.""")
