import warnings

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,roc_auc_score, roc_curve, mean_squared_error
from pandas.plotting import scatter_matrix

df = pd.read_csv('diabetes.csv')

head = df.head() #Veri setinin ilk kısmı listelendi
print(head)
print('\n')

shape = df.shape #Veri şekli satır ve sütun sayısı olarak belirtildi
print ("Shape of data {}" . format(shape))
print ("Number of rows: {}" . format(shape[0]))
print ("Number of columns: {}" . format(shape[1]))
print('\n')

columns = df.columns #Sütunları belirtiuor
print(columns)
print('\n')

dtypes = df.dtypes #Sütündaki verilerin türünü belirtiyor
print(dtypes)
print('\n')

info = df.info() #Sütunlardaki verilerin tipini ve null olmayan değerleri yazdırır
print(info)
print('\n')

describe = df.describe() #Verilerin tabloya nasıl dağıldığını gösteirir(non-null toplam sayı, ortalam, standart sapma, ...)
print(describe)
print('\n')

df = df.drop_duplicates() #Yinelenen verileri kontrol ediyor varsa siliyor

nullSum = df.isnull().sum() #Veri setinde null değerlirin sayısını belirtiyor
print(nullSum)
print('\n')

nunique = df.nunique() #Her sütun için benzersiz değerlerin sayısını döndürür
print(nunique)
print('\n')

#Pregnancies'in 0 olması mümkün, diğerleri için kontrol sağlanıyor
print('Glucose:', df[df['Glucose']==0].shape[0])
print('BloodPressure:', df[df['BloodPressure']==0].shape[0])
print('SkinThickness:', df[df['SkinThickness']==0].shape[0])
print('Insulin:', df[df['Insulin']==0].shape[0])
print('BMI:', df[df['BMI']==0].shape[0])
print('\n')

#0 olan değerler sütunun ortalam değerleriyle değiştiriliyor
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())#Normal dağılım
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())#Normal dağılım
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())#Çarpık dağılım
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())#Çarpık dağılım
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())#Çarpık dağılım

scatter_matrix(df, figsize=(20,20)) #Dağılım matris
plt.show()

sns.heatmap(df.corr(), annot=True) #Korelasyon matris
plt.title("Correlation Matrix", weight='bold')
plt.show()

df.hist(bins = 10, figsize = (10, 10)) #Her veri seti özelliği için histogram
plt.show()

df.Pregnancies.value_counts().plot(kind = "pie") #Hamilelik pasta grafiği
plt.show()

df.Outcome.value_counts().plot(kind = "pie") #Diyabet pasta grafiği
plt.show()

sns.countplot(x=df["Outcome"]) #Diyabet hastalığının sonucu grafikleştirildi
plt.show()

#Verilerin birbiriyle ilişkisini inceleyen çeşitli grafikler elde edildi
plt.figure(figsize=[10, 4], dpi=100)
plt.scatter(df["Pregnancies"], df["Outcome"], color="red")
plt.title("Hamilelik ve Diyabet İlişkisi", weight='bold')
plt.xticks(rotation=90)
plt.xlabel('Pregnancies')
plt.ylabel('Diabetes')
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.xticks(rotation=90)
ax = sns.countplot(x=df.Pregnancies, hue=df.Outcome)
ax.set_title('Hamileliğe Göre Diyabet Olup Olmaması')
plt.show()

plt.figure(figsize=[10, 4], dpi=100)
plt.scatter(df["Pregnancies"], df["Age"], color="red")
plt.title("Hamilelik ve Yaş İlişkisi", weight='bold')
plt.xticks(rotation=90)
plt.xlabel('Pregnancies')
plt.ylabel('Age')
plt.grid()
plt.show()

plt.figure(figsize=[10, 4], dpi=100)
plt.scatter(df["Age"], df["Outcome"], color="red")
plt.title("Yaş ve Diyabet İlişkisi", weight='bold')
plt.xticks(rotation=90)
plt.xlabel('Age')
plt.ylabel('Diabetes')
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.xticks(rotation=90)
ax = sns.countplot(x=df.Age, hue=df.Outcome)
ax.set_title('Yaşa Göre Diyabet Olup Olmaması')
plt.show()

plt.figure(figsize=[10, 4], dpi=100)
plt.scatter(df["Glucose"], df["Outcome"], color="red")
plt.title("Glukoz ve Diyabet İlişkisi", weight='bold')
plt.xticks(rotation=90)
plt.xlabel('Glucose')
plt.ylabel('Diabetes')
plt.grid()
plt.show()

plt.figure(figsize=(10,4))
plt.xticks(rotation=90)
ax = sns.countplot(x=df.Glucose, hue=df.Outcome)
ax.set_title('Glukoza Göre Diyabet Olup Olmaması')
plt.show()

# Test ve eğitim verilerine ayırma
X = df.drop('Outcome', axis=1) #Outcome sütunu hariç test verileri
Y = df['Outcome'] #Eğitim verileri
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0) #%80 test ve %20 train olarak bölme

#Bölünen verilerin kaç sütun ve satır oluştuğunu veriyor
train = X_train.shape, y_train.shape
print(train)
test = X_test.shape, y_test.shape
print(test)
print('\n')

#Bölünen verilerin sütun ve satırlarının verilerini yazıyor
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.fit_transform(X_train)
print(X_train)
X_test = sc.transform(X_test)
print(X_test)

#K Nearest Neighbours(KNN - K En Yakın Komşu)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

knn = KNeighborsClassifier(n_neighbors=1) #KNN modeli oluşturuldu
knn.fit(X_train, y_train)

#Eğitim ve Test verilerinin puanı değerlendirildi
print('\nTraining accuracy score: %.3f' %knn.score(X_train, y_train))
print('Test accuracy score: %.3f' %knn.score(X_test, y_test))

y_pred1 = knn.predict(X_test)
y_pred1

KNN_accuracy_score = accuracy_score(y_test, y_pred1) #Model doğruluğunu hesaplama
print("\nK=1 KNN Classifier Accuracy Score:", KNN_accuracy_score)

print("K=1 KNN Classification Report:\n", classification_report(y_test, y_pred1))

KNNcm = confusion_matrix(y_test, y_pred1) #Confusion matrix elde edildi
print("\nK=1 KNN Confusion Matrix:\n", KNNcm)

sns.heatmap(KNNcm, annot=True) #Confusion matrix görsel halinde elde edildi
plt.title('K=1 KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#En iyi K değerinin hata oranını veren Elbow Method
error_rate= []

for i in (range(1,40)):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K Values')
plt.ylabel('Error Rate')
plt.show()

#K'nin en iyi değerini almak için çapraz doğrulamayı kullanma
k_values= [i for i in range(1,40)]
scores= []
scaler = StandardScaler()

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, scaler.fit_transform(X), Y, cv=5)
    scores.append(np.mean(score))

sns.lineplot(x= k_values, y= scores, marker= 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()

#Parametre değerleri atandı
hyperparameters = dict(n_neighbors = list(range(1,40)), p=[1,2],
                       weights = ['uniform', 'distance'],
                       metric = [ 'manhattan', 'euclidean', 'minkowski'])

#Model oluşturuldu
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1',error_score=0)

best_model = grid_search.fit(X_train,y_train)

#En iyi hiperparametre değeri
print('\nBest metric:', best_model.best_estimator_.get_params()['metric'])
print('Best p:', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])

#K=24 için
knn24 = KNeighborsClassifier(n_neighbors=24)

knn24.fit(X_train,y_train)
y_pred2 = knn24.predict(X_test)

print('\n**K=24**')

#Eğitim ve Test verilerinin puanı K=24'e göre değerlendirildi
print('\nTraining accuracy score: %.3f' %knn24.score(X_train, y_train))
print('Test accuracy score: %.3f' %knn24.score(X_test, y_test))

print("\nK=24 KNN Classifier Accuracy Score:", accuracy_score(y_test, y_pred2))
print("K=24 KNN Classification Report:\n", classification_report(y_test,y_pred2))
print("\nK=24 KNN Confusion Matrix:\n", confusion_matrix(y_test,y_pred2))

sns.heatmap(confusion_matrix(y_test,y_pred2), annot=True) #Confusion matrix görsel halinde elde edildi
plt.title('K=24 KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#K değerlerinin etkisini test setinde görmek için
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred1, marker= '*', s=100,edgecolors='blue')
plt.title("Predicted values with k=1", fontsize=20)

plt.subplot(1,2,2)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred2, marker= '*', s=100,edgecolors='blue')
plt.title("Predicted values with k=24", fontsize=20)
plt.show()

#Bazı verileri diyabet verilerine göre veri görselleştirmesi
sns.scatterplot(x=df['Pregnancies'],y=df['Glucose'], hue=df['Outcome'])
plt.show()

sns.scatterplot(x=df['Glucose'],y=df['Insulin'], hue=df['Outcome'])
plt.show()

sns.scatterplot(x=df['BloodPressure'],y=df['SkinThickness'], hue=df['Outcome'])
plt.show()

#NAİVE BAYES
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

NB = GaussianNB() #Navie bayes modeli oluşturuldu
NB = NB.fit(X_train,y_train)
y_predNB= NB.predict(X_test)
y_predNB

NB_accuracy_score = accuracy_score(y_test, y_predNB) #Model doğruluğunu hesaplama
print("\nNaive Bayes Classifier Accuracy Score:", NB_accuracy_score)

print("Naive Bayes Classification Report:\n", classification_report(y_test, y_predNB))

NBcm = confusion_matrix(y_test, y_predNB) #Confusion matrix elde edildi
print("\nNaive Bayes Confusion Matrix:\n", NBcm)

sns.heatmap(NBcm, annot=True) #Confusion matrix görsel halinde elde edildi
plt.title('Naive Bayes Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

print('\nTrue Positives(TP) = ', NBcm[0,0])
print('True Negatives(TN) = ', NBcm[1,1])
print('False Positives(FP) = ', NBcm[0,1])
print('False Negatives(FN) = ', NBcm[1,0])

TP = NBcm[0,0]
TN = NBcm[1,1]
FP = NBcm[0,1]
FN = NBcm[1,0]

#Sınıflandırmanın doğruluk oranı
print('\nClassification accuracy : {0:0.4f}'.format((TP + TN) / float(TP + TN + FP + FN)))

#Sınıflandırmanın hata oranı
print('Classification error : {0:0.4f}'.format((FP + FN) / float(TP + TN + FP + FN)))

#Precision, Recall, TP, FP ve Specificity oranları elde ettim
precision = TP / float(TP + FP)
print('\nPrecision : {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))

true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))

false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))

specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))

#İki sınıfın tahmin edilen ilk 10 verisini yazdırma
y_pred_prob = NB.predict_proba(X_test)[0:10]
print("\n", y_pred_prob)

#Olasılıkları veri çerçevesinde saklama
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Diabet(-)', 'Diabet(+)'])
print("\n", y_pred_prob_df)

#1. sınıf için olaslıklıları yazdırma
diabet = NB.predict_proba(X_test)[0:10, 1]
print("\n", diabet)

#1. sınıf için olasılıkları tutma
y_predNB1 = NB.predict_proba(X_test)[:, 1]
y_predNB1

#Tahmin edilen olasılıkların histogramını çizme
plt.rcParams['font.size'] = 12
plt.hist(y_predNB1, bins = 10)
plt.title('Diyabet Hastalarının Tahmini Olasılıklarının Histogramı')
plt.xlim(0,1)
plt.xlabel('Tahmini diyabet olasılığı')
plt.ylabel('Tahmin')
plt.show()

#ROC Eğrisi(Alıcı Çalışma Karakteristiği), ikili sınıflandırıcılarda kullanılan bir araçtır
model_roc_auc_NB = roc_auc_score(y_test, y_predNB1)
fpr, tpr, thresholds = roc_curve(y_test, NB.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Naive Bayes Classifier (area = %f)' %model_roc_auc_NB)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC curve for Gaussian Naive Bayes Classifier for Diabetes')
plt.rcParams['font.size'] = 12
plt.legend(loc='lower right')
plt.show()

#K-Fold Cross Validation
#10 Katlı Çapraz Doğrulama Uygulaması
scores = cross_val_score(NB, X_train, y_train, cv = 10, scoring='accuracy')
print('\nCross-validation scores:{}'.format(scores))

#Average cross-validation puanı
print('Average cross-validation score: {:.4f}'.format(scores.mean()))

#SUPPORT VECTOR MACHİNE (SVM)
from sklearn import svm
from sklearn.model_selection import cross_val_score

SVM = svm.SVC(kernel='linear') #SVM modeli oluşturuldu
SVM = SVM.fit(X_train, y_train)
y_predSVM = SVM.predict(X_test)
y_predSVM

#Model doğruluğunu hesaplama
print('\nSVM Classifier Accuracy Score:', accuracy_score(y_test, y_predSVM))

print("SVM Classification Report:\n", classification_report(y_test, y_predSVM))

SVMcm = confusion_matrix(y_test, y_predSVM) #Confusion matrix elde edildi
print('\nSVM Confusion Matrix:\n', SVMcm)

sns.heatmap(SVMcm, annot=True) #Confusion matrix görsel halinde elde edildi
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#ROC Eğrisi(Alıcı Çalışma Karakteristiği), ikili sınıflandırıcılarda kullanılan bir araçtır
fpr, tpr, thresholds = roc_curve(y_test, y_predSVM)
plt.figure()
plt.plot(fpr, tpr)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.rcParams['font.size'] = 12
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for SVM Classifier for Diabetes')
plt.show()

ROC_AUC = roc_auc_score(y_test, y_predSVM)
print('\nSVM ROC AUC : {:.4f}'.format(ROC_AUC))

#Doğrusal çekirdek ve C=1.0 ile sınıflandırıcıyı başlat
linear_svm1 = svm.SVC(kernel='linear', C=1.0) #Model oluşturuldu
linear_svm1.fit(X_train,y_train)
y_predsvm1 = linear_svm1.predict(X_test)
print('\nModel accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_predsvm1)))

#Doğrusal çekirdek ve C=100.0 ile sınıflandırıcıyı başlat
linear_svm100=svm.SVC(kernel='linear', C=100.0) #Model oluşturuldu
linear_svm100.fit(X_train, y_train)
y_predsvm100=linear_svm100.predict(X_test)
print('Model accuracy score with linear kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_predsvm100)))

#Doğrusal çekirdek ve C=1000.0 ile sınıflandırıcıyı başlat
linear_svm1000=svm.SVC(kernel='linear', C=1000.0) #Model oluşturuldu
linear_svm1000.fit(X_train, y_train)
y_predsvm1000=linear_svm1000.predict(X_test)
print('Model accuracy score with linear kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_predsvm1000)))

#Eğitim veri seti için tahmin yürütüldü
y_pred_train = linear_svm1.predict(X_train)
y_pred_train

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

#Eğitim ve test setindeki puanları yazdır
print('Training set score: {:.4f}'.format(linear_svm1.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(linear_svm1.score(X_test, y_test)))

#Cross validated ROC için doğruluk skoru elde edildi
Cross_validated_ROC_AUC = cross_val_score(linear_svm1, X_train, y_train, cv=10, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))

#LOGISTIC REGRESSION (LOJİSTİK REGRESYON)
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf

logreg = LogisticRegression(random_state=16) #Lojistik regresyon modeli oluşturuldu
logreg = logreg.fit(X_train,y_train)
y_predLOG = logreg.predict(X_test)
y_predLOG

#Model doğruluğunu hesaplama
print('\nLogistic Regression Classifier Accuracy Score:', accuracy_score(y_test, y_predLOG))

print("Logistic Regression Classification Report:\n", classification_report(y_test, y_predLOG))

LOGcm = confusion_matrix(y_test, y_predLOG) #Confusion matrix elde edildi
print('\nLogistic Regression Confusion Matrix:\n', LOGcm)

sns.heatmap(LOGcm, annot=True) #Confusion matrix görsel halinde elde edildi
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#ROC Eğrisi(Alıcı Çalışma Karakteristiği), ikili sınıflandırıcılarda kullanılan bir araçtır
model_roc_auc_LOC = roc_auc_score(y_test, y_predLOG)
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression Classifier (area = %f)' %model_roc_auc_LOC)
plt.plot([0,1], [0,1], 'k--' )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC curve for Gaussian Logistic Regression Classifier for Diabetes')
plt.rcParams['font.size'] = 12
plt.legend(loc='lower right')
plt.show()

#Regresyon modelinin özetini oluşturma
result = smf.ols(formula='Glucose ~ Outcome', data=df).fit()

#Uygun yöntemlerle elde ettiğim değerlerin ayrıntılı raporları
result.summary()
print('\n', result.summary())

result.summary2()
print('\n', result.summary2())

#RANDOM FOREST(RASTGELE ORMAN)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

RF = RandomForestClassifier() #Karar ağacı modeli oluşturuldu
RF = RF.fit(X_train,y_train)
y_predRF = RF.predict(X_test)
y_predRF

#Model doğruluğunu hesaplama
print('\nRandom Forest Classifier Accuracy Score:', accuracy_score(y_test, y_predRF))

print("Random Forest Classification Report:\n", classification_report(y_test, y_predRF))

RFcm = confusion_matrix(y_test, y_predRF) #Confusion matrix elde edildi
print('\nRandom Forest Confusion Matrix:\n', RFcm)

sns.heatmap(RFcm, annot=True) #Confusion matrix görsel halinde elde edildi
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#Diyabet hastalarının özelliklerini tahmin ederek en büyüğü ve ne küçüğü gözlemlemek
importances = RF.feature_importances_
columns = X.columns
i = 0

print("\n")
while i < len(columns):
    print("The importance of feature", columns[i], "is %", round(importances[i] * 100, 2), ".")
    i += 1

#Tahminleri görselleştirme
feature_list = list(X.columns)
x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, feature_list)
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.show()

#Bir aralık içindeki parametreleri rastgele arama
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

#En iyi hiperparametreleri bulmak için oluşturuldu
rand_search = RandomizedSearchCV(RF,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=5)

rand_search.fit(X_train, y_train)
best_rf = rand_search.best_estimator_ #En iyi model için bir değişken oluşturma ve yazdırmma
print('\nBest hyperparameters:',  rand_search.best_params_)

#DECISION TREE (KARAR AĞAÇLARI)
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree

dtc = DecisionTreeClassifier(criterion='entropy', random_state=42) #Karar ağacı modeli oluşturuldu
dtc = dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)
y_pred

accuracy_score = accuracy_score(y_test, y_pred) #Model doğruluğunu hesaplama
print('\nDecision Tree Classifier Accuracy Score:', accuracy_score)

print("Decision Tree Classification Report:\n", classification_report(y_test, y_pred))

DTcm = confusion_matrix(y_test, y_pred) #Confusion matrix elde edildi
print('\nDecision Tree Confusion Matrix:\n', DTcm)

sns.heatmap(DTcm, annot=True) #Confusion matrix görsel halinde elde edildi
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

#ROC Eğrisi (Alıcı Çalışma Karakteristiği), ikili sınıflandırıcılarda kullanılan bir araçtır
model_roc_auc = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, dtc.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Decision Tree Classifier (area = %f)' %model_roc_auc)
plt.plot([0, 1], [0, 1])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for Gaussian Decision Tree Classifier for Diabetes')
plt.legend(loc='lower right')
plt.show()

#Bir modelin ne kadar önemli olduğunu derecelendirme. Toplamları her zaman 1 olacaktır.
loans_features = [x for i,x in enumerate(X.columns)
                  if i!=len(X.columns)]
print('\nFeature importances:\n{}'.format(dtc.feature_importances_))

#En önemli özellikleri sıralıyor
feature_labels = np.array(['Pregnancies','Glucose','BloodPressure','SkinThickness',
                           'Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome'])
importance = dtc.feature_importances_
feature_indexes_by_importance = importance.argsort()[::-1]
for index in feature_indexes_by_importance:
    print('{}-{:.2f}%'.format(feature_labels[index], (importance[index] *100.0)))

#Özelliklerin görselleştirilmesi
def plot_feature_importances_loans(model):
    plt.figure(figsize=(8, 6))
    n_features = len(X.columns)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), loans_features)
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    plt.ylim(-1, n_features)

plot_feature_importances_loans(dtc)
plt.show()

#Karar ağaçlarının metin gösterimi
text_representation = tree.export_text(dtc)
print('\n', text_representation)

#Karar ağacının görselleştirilmesi
tree.plot_tree(dtc)
plt.savefig('DT_image')
plt.show()

fig = plt.figure(figsize=(10, 10))
_ = plot_tree(dtc,
              filled=True,
              rounded=True,
              class_names=["alpha", "beta"])

fig.savefig("tree.png")
plt.show()

#DECISION TREE REGRESSİON (KARAR AĞACI REGRESYON)
from sklearn.tree import DecisionTreeRegressor

A = df.iloc[:, 1:2].values
B = df.iloc[:, 2].values

dtr = DecisionTreeRegressor(random_state=0) #Karar ağacı regresyon modeli oluşturuldu
dtr = dtr.fit(A, B)

y_pred_dtr = dtr.predict([[160]])
print("\nDecision Tree Regression Predicted: % d\n" % y_pred_dtr)

X_grid = np.arange(min(A), max(A), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(A, B, color = 'red')
plt.plot(X_grid, dtr.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Glucose')
plt.ylabel('Blood Pressure')
plt.show()

#SUPPORT VECTOR REGRESSİON (DESTEK VEKTÖRÜ REGRESYONU)
from sklearn.svm import SVR

model_regressor = SVR(kernel = 'rbf')
model_regressor.fit(A, B)

y_pred_SVR = model_regressor.predict(np.array([6.5]).reshape(-1,1))
print('\nSVR prediction:', y_pred_SVR)

#y_predSVR = sc_Y.inverse_transform(model_regressor.predict(sc_X.transform(np.array([6.5]).reshape(-1,1))))
plt.scatter(A, B, color = 'red')
plt.plot(A, model_regressor.predict(A), color='blue')
plt.title('SVR Model')
plt.xlabel('Glucose')
plt.ylabel('Outcome')
plt.show()

svr_rbf = SVR(kernel='rbf',gamma='scale', C=1.0, epsilon=0.1)
svr_rbf.fit(X_train, y_train)

Score= svr_rbf.score(X_test,y_test)

print('\nSupport Vector Regression SVR Accuracy Score: ', Score)
print("RMSE for RBF kernelized SVR:",np.sqrt(mean_squared_error(y_test,svr_rbf.predict(X_test))))

#LİNEAR REGRESSİON (DOĞRUSAL REGRESYON)
from sklearn import linear_model

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual errors")
plt.show()

print('\nCoefficients: ', reg.coef_)
print('Variance score: {}'.format(reg.score(X_test, y_test)))