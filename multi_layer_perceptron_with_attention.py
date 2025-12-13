from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools
import os
import logging
import sys

from utils import report_all_metrics, plot_confusion_matrix, plot_roc_curve
from mlp import Heart_Disease_NN

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def log(func):
    def inner(params):
        logging.info(f"Starting function : {func.__name__} with params : {params[:2]}")
        try:
            func(params)
            logging.info(f"Ended function {func.__name__} with params : {params[:2]}")
        except Exception as e:
            logging.info(f"{func.__name__} failed due to :\n {e}")
    return inner

@log
def process_values(params):

    mlp_dir = os.path.join(os.getcwd(), "results", "MLP_Attention", f"{params[0]}_{params[1]}")
    if(os.path.exists(mlp_dir)):
        logging.info(f"Images already computed, skipping recomputation ...")
        return

    os.makedirs(mlp_dir, exist_ok=True)

    mlp = Heart_Disease_NN()

    mlp.fit(params[2], params[4])
    mlp.eval()

    X_test_scaled = params[3]
    feature_names = params[6]
    
    X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        # forward muss return_attention unterstützen (mlp.py Änderung!)
        _, attn_weights = mlp(X_tensor, return_attention=True)
    
    # attn_weights: (batch, heads=1, 10, 10)
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.mean(dim=1)  # mean over heads

    
    attn_map = attn_weights.mean(dim=0).cpu().numpy()  # mean over batch -> (10, 10)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attn_map,
        xticklabels=feature_names,
        yticklabels=feature_names,
        square=True,
        cbar_kws={"label": "Attention weight"}
    )
    plt.title("Self-Attention Map – Feature Interactions")
    plt.xlabel("Attended feature (keys)")
    plt.ylabel("Query feature")
    plt.tight_layout()
    
    # Speichern in deinen Ergebnisordner
    heatmap_path = os.path.join(mlp_dir, "Attention")
    os.makedirs(heatmap_path, exist_ok=True)
    plt.savefig(os.path.join(heatmap_path, "attention_heatmap.png"), dpi=200)
    plt.close()
    y_pred_mlp = mlp.predict(params[3])
    y_proba_mlp = mlp.predict_proba(params[3])[:, 1]

    # report_all_metrics(params[5], y_pred_mlp, y_proba_mlp, prefix="MLP Classifier")


    cm_mlp = confusion_matrix(params[5], y_pred_mlp)
    plot_confusion_matrix(
        cm_mlp,
        os.path.join(mlp_dir, "Confusion Matrix"), 
        title="Confusion Matrix – MLP Classifier"
    )

    plot_roc_curve(
        params[5],
        y_proba_mlp,
        os.path.join(mlp_dir, "ROC Curve"),         
        title="ROC Curve – MLP Classifier"
    )

    #torch.save(mlp.state_dict(), os.path.join(mlp_dir, "mlp_state_dict.pt"))

def worker(params):
    set_up_logging()
    process_values(params)

def set_up_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )

def main():
    df = pd.read_csv(os.path.join(os.getcwd(), "UCI_Heart_Disease_Dataset_Combined.csv"))
    x = df.drop("HeartDisease", axis=1)
    feature_names = x.columns.tolist()

    y = df["HeartDisease"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    #Apply Scaling
    scaler = StandardScaler()
    x_train_scaled, x_test_scaled = scaler.fit_transform(x_train), scaler.transform(x_test)

    learning_rate_inits = [0.0001, 0.001, 0.01, 0.1]
    max_iterations = [200, 400, 600, 800, 1000]

    combinations = itertools.product(
    learning_rate_inits, max_iterations,
    [x_train_scaled], [x_test_scaled],
    [y_train], [y_test],
    [feature_names]
)


    for combination in combinations:
        worker(combination)


if __name__ == "__main__":
    main()

