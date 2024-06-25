# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:39:42 2023

@author: MP
"""
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('phishing.csv')

X = df.iloc[:, :31]
y = df['CLASS_LABEL']

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

model = SVC()
model.fit(X_pca, y)

print(pca.explained_variance_ratio_)

kfold = KFold(n_splits=10)
scores = cross_val_score(model, X_pca, y, cv=kfold)

print(scores)

predictions = model.predict(X_pca)

report = classification_report(y, predictions)
print(report)

matrix = confusion_matrix(y, predictions)
print(matrix)

plt.bar(range(1, 11), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

colors = ['red','blue']
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color=colors[0], label='Class 0')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color=colors[1], label='Class 1')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.show()

sns.heatmap(matrix,annot=True, fmt='d',cmap='YlGnBu', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

plt.plot(scores)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.show()


