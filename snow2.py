import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("snow2.csv", header=None, skiprows=1)
data.columns = ["MinTemp", "Snowfall", "YR", "MO", "DA", "Rain"]

bins = [0, 30, 60, 90]
labels = ['Low', 'Medium', 'High']
data['Snowfall_binned'] = pd.cut(data['Snowfall'], bins=bins, labels=labels)

X = data[['MinTemp','Rain']]
y = data['Snowfall_binned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


logistic_regression = LogisticRegression(solver='lbfgs')
X_train = X_train.copy()
scaler = MinMaxScaler()
X_train[['MinTemp','Rain']] = scaler.fit_transform(X_train[['MinTemp','Rain']])

#X_train = pd.get_dummies(X_train, columns=['Rain'], prefix = ['Rain'])
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)
score = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, labels=['Low', 'Medium', 'High'], average='weighted')
print('Logistic Regression - Score:', score, 'Recall:', recall)

linear_svm = SVC(kernel='linear')
linear_svm.fit(X_train, y_train)
y_pred = linear_svm.predict(X_test)
score = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, labels=['Low', 'Medium', 'High'], average='weighted')
print('SVM with Linear Kernel - Score:', score, 'Recall:', recall)

rbf_svm = SVC(kernel='rbf')
rbf_svm.fit(X_train, y_train)
y_pred = rbf_svm.predict(X_test)
score = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, labels=['Low', 'Medium', 'High'], average='weighted')
print('SVM with RBF Kernel - Score:', score, 'Recall:', recall)

plt.plot(data['MinTemp'], label='MinTemp')
plt.plot(data['Snowfall'], label='Snowfall')
plt.plot(data['Rain'], label='Rain')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.show()

data['Snowfall_binned'].value_counts().plot(kind='bar', color = ['b', 'g', 'r'])
plt.xlabel('Snowfall Bin')
plt.ylabel('Count')
plt.show()

sns.scatterplot(x= range(len(data)), y='Snowfall', hue='Snowfall_binned', data=data)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

high_snow = data[data['Snowfall_binned'] == 'High']
print(high_snow[['DA', 'MO', 'YR', 'MinTemp', 'Snowfall', 'Rain']])


