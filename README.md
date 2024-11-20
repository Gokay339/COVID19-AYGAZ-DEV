# COVID19-AYGAZ-DEV

## Proje Adı :  COVID-19 Risk Tahmini



## Proje Hakkında
Bu proje, Kaggle'dan alınan veri seti ile COVID-19 komplikasyon riskini tahmin etmek amacıyla hazırlanmıştır.  
Veri setine [buradan ulaşabilirsiniz](https://www.kaggle.com/datasets/meirnizri/covid19-dataset).

Kaggle projeme ulaşmak için Tıklayın [buradan ulaşabilirsiniz](https://www.kaggle.com/code/gokay339/aygaz-dev).

## Kullanılan Kütüphaneler
- **Pandas**
- **Numpy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **StratifiedKFold**
- **VotingClassifier**
- **RandomForestClassifier**
- **KNeighborsClassifier**
- **LogisticRegression**
- **SVC**
- **GridSearchCV**
- **AdaboostClassifier**
- **GradientBoostingClassifier**
- **XGBClassifier**
- **StandardScaler**
- **OneHotEncoder**

## Sonuçlar
Farklı modellerin test edilmesi sonucunda elde edilen doğruluk ve ROC-AUC skorları aşağıdaki gibidir:

- **Random Forest Modeli:** %85 doğruluk ile en iyi sonucu verdi.
- **Boş Verilerin Yerine Ortalama Değerler İle Doldurduğumuzda Çıkan Sonuç:** 0.9991
- **Verileri Sildiğimiz Zaman Çıkan Sonuç:** 0.9910

Boş verilerin yerine ortalama değerler ile doldurma yöntemini denedim ve bu yöntem, sonuçları daha yüksek bir doğruluk oranına taşımayı başardı.

### Farklı Modellerin ROC-AUC Değerleri:
- **Logistic Regression (LR):** 1.0
- **K-Nearest Neighbors (KNN):** 0.9854
- **Support Vector Classifier (SVC):** 1.0
- **Random Forest (RF):** 1.0
- **Adaboost:** 1.0
- **Gradient Boosting Machine (GBM):** 1.0
- **XGBoost:** 1.0

### Model Hiperparametre Optimizasyonu Sonucunda En İyi Parametreler:
- **KNN:** `n_neighbors = 5`
- **CART (Decision Tree):** `max_depth = 1`, `min_samples_split = 2`
- **RF (Random Forest):** `max_depth = 15`, `max_features = 'sqrt'`, `min_samples_split = 15`, `n_estimators = 200`
- **XGBoost:** `colsample_bytree = 0.5`, `learning_rate = 0.1`, `max_depth = 5`, `n_estimators = 100`

### Modellerin Tahminlerini Birleştirerek Yapılan Toplu Tahminlerin Sonuçları:
- **Accuracy:** 0.9936
- **F1Score:** 0.9964
- **ROC_AUC:** 1.0

### Örnek Tahminler:
- 50. kişinin COVID durumu tahmini (Voting Classifier): 0
- 50. kişinin COVID durumu tahmini (KNN modeline göre): 0
- 50. kişinin COVID durumu tahmini (CART modeline göre): 0
- 50. kişinin COVID durumu tahmini (XGBoost modeline göre): 0
 
###
Kaggle Linki : 
