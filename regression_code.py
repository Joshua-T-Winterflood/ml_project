# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 10:48:53 2025

@author: samir
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,  roc_curve
)

# Load data
file_path = os.path.join(os.getcwd(), "UCI_Heart_Disease_Dataset_Combined.csv")
df = pd.read_csv(file_path)

print("Form:", df.shape)
print(df.dtypes)
print("\nMissing Values per column:")
print(df.isna().sum())

# Split target and features
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTrain/Test Shapes:")
print(X_train.shape, X_test.shape)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Specificity
def specificity_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# Unified metrics reporter
def report_all_metrics(y_true, y_pred, y_proba, prefix=""):
    print(f"\n--- {prefix} ---")
    print(f"Accuracy:       {accuracy_score(y_true, y_pred):.3f}")
    print(f"BalancedAcc:    {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision:      {precision_score(y_true, y_pred):.3f}")
    print(f"Recall/Sens:    {recall_score(y_true, y_pred):.3f}")
    print(f"Specificity:    {specificity_score(y_true, y_pred):.3f}")
    print(f"F1 Score:       {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:        {roc_auc_score(y_true, y_proba):.3f}")

# Confusion matrix plot
def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Negative", "Positive"])
    ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    plt.title(title)
    plt.tight_layout()
    plt.show()


# ROC plot
def plot_roc_curve(y_true, y_proba, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
    plt.plot([0,1], [0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# L2 Logistic Regression
model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500)
model_l2.fit(X_train_scaled, y_train)

y_pred_l2 = model_l2.predict(X_test_scaled)
y_proba_l2 = model_l2.predict_proba(X_test_scaled)[:, 1]

report_all_metrics(y_test, y_pred_l2, y_proba_l2, prefix="L2 Logistic Regression")

# L1 Logistic Regression
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
model_l1.fit(X_train_scaled, y_train)

y_pred_l1 = model_l1.predict(X_test_scaled)
y_proba_l1 = model_l1.predict_proba(X_test_scaled)[:, 1]

report_all_metrics(y_test, y_pred_l1, y_proba_l1, prefix="L1 Logistic Regression")

# Feature importance plots
coeff_l2 = model_l2.coef_[0]
coeff_l1 = model_l1.coef_[0]
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, coeff_l2)
plt.title("Feature Importance – Logistic Regression (L2)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.barh(features, coeff_l1)
plt.title("Feature Importance – Logistic Regression (L1)")
plt.tight_layout()
plt.show()

# Confusion matrices
cm_l2 = confusion_matrix(y_test, y_pred_l2)
plot_confusion_matrix(cm_l2, title="Confusion Matrix – L2 Logistic Regression")
plot_roc_curve(y_test, y_proba_l2, title="ROC Curve – L2 Logistic Regression")

cm_l1 = confusion_matrix(y_test, y_pred_l1)
plot_confusion_matrix(cm_l1, title="Confusion Matrix – L1 Logistic Regression")
plot_roc_curve(y_test, y_proba_l1, title="ROC Curve – L1 Logistic Regression")
