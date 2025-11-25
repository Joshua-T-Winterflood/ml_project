# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:48:53 2025

@author: samir
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


file_path = os.path.join(os.getcwd(), "UCI_Heart_Disease_Dataset_Combined.csv")
df = pd.read_csv(file_path)
    
print("Form:", df.shape)


print(df.dtypes)

print("\nMissing Values per column:")
print(df.isna().sum())



# seperate target

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain/Test Shapes:")
print(X_train.shape, X_test.shape)



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Logistic Regression (L2 = Ridge)

model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500)
model_l2.fit(X_train_scaled, y_train)

y_pred_l2 = model_l2.predict(X_test_scaled)
y_proba_l2 = model_l2.predict_proba(X_test_scaled)[:, 1]

print("\n===== L2-Regularisierung (Ridge) =====")
print("Accuracy:", accuracy_score(y_test, y_pred_l2))
print("Precision:", precision_score(y_test, y_pred_l2))
print("Recall:", recall_score(y_test, y_pred_l2))
print("F1-Score:", f1_score(y_test, y_pred_l2))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_l2))


# Logistic Regression (L1 = Lasso → Feature Selection)

model_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
model_l1.fit(X_train_scaled, y_train)

y_pred_l1 = model_l1.predict(X_test_scaled)
y_proba_l1 = model_l1.predict_proba(X_test_scaled)[:, 1]

print("\n===== L1-Regularisierung (Lasso) =====")
print("Accuracy:", accuracy_score(y_test, y_pred_l1))
print("Precision:", precision_score(y_test, y_pred_l1))
print("Recall:", recall_score(y_test, y_pred_l1))
print("F1-Score:", f1_score(y_test, y_pred_l1))
print("ROC-AUC:", roc_auc_score(y_test, y_proba_l1))


#Feature Importance Plot (für L1 und L2)

coeff_l2 = model_l2.coef_[0]
coeff_l1 = model_l1.coef_[0]
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, coeff_l2, color="steelblue")
plt.title("Feature Importance – Logistic Regression (L2-Regularisierung)")
plt.xlabel("Koeffizient")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.barh(features, coeff_l1, color="darkred")
plt.title("Feature Importance – Logistic Regression (L1-Regularisierung)")
plt.xlabel("Koeffizient")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# Confusion Matrix for L2

cm = confusion_matrix(y_test, y_pred_l2)




plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")                 # Heatmap
plt.colorbar()                               # Farbleiste

# Werte in die Zellen schreiben
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center", color="black")

plt.title("Confusion Matrix (L2 Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(cm.shape[1]))
plt.yticks(np.arange(cm.shape[0]))
plt.show()