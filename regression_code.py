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
from sklearn.metrics import confusion_matrix

from utils import report_all_metrics, plot_confusion_matrix, plot_roc_curve

def main():
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

    # Figures Logging
    plt.figure(figsize=(10,6))
    plt.barh(features, coeff_l2)
    plt.title("Feature Importance – Logistic Regression (L2)")
    plt.tight_layout()
    path = os.path.join(os.getcwd(), "results", "Regression", "Feature Importance")
    filename = "features_importance_L2.png"
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename))

    plt.figure(figsize=(10,6))
    plt.barh(features, coeff_l1)
    plt.title("Feature Importance – Logistic Regression (L1)")
    plt.tight_layout()
    path = os.path.join(os.getcwd(), "results", "Regression", "Feature Importance")
    filename = "features_importance_L1.png"
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, filename))

    # Confusion matrices
    cm_l2 = confusion_matrix(y_test, y_pred_l2)
    plot_confusion_matrix(cm_l2, os.path.join(os.getcwd(), "results", "Regression"), title="Confusion Matrix – L2 Logistic Regression")
    plot_roc_curve(y_test, y_proba_l2, os.path.join(os.getcwd(), "results", "Regression"), title="ROC Curve – L2 Logistic Regression")

    cm_l1 = confusion_matrix(y_test, y_pred_l1)
    plot_confusion_matrix(cm_l1, os.path.join(os.getcwd(), "results", "Regression"), title="Confusion Matrix – L1 Logistic Regression")
    plot_roc_curve(y_test, y_proba_l1, os.path.join(os.getcwd(), "results", "Regression"), title="ROC Curve – L1 Logistic Regression")


if __name__ == "__main__":
    main()