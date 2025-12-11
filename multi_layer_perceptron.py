from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import itertools
import os
import logging
import sys

from utils import plot_confusion_matrix, plot_roc_curve

def log(func):
    def inner(params):
        logging.info(f"Starting function : {func.__name__} with params : {params[:3]}")
        try:
            func(params)
            logging.info(f"Ended function {func.__name__} with params : {params[:3]}")
        except Exception as e:
            logging.info(f"{func.__name__} failed due to :\n {e}")
    return inner

@log
def process_values(params):

    mlp_dir = os.path.join(os.getcwd(), "results", "MLP", f"{params[0]}_{params[1]}_{params[2]}")
    os.makedirs(mlp_dir, exist_ok=True)

    mlp = MLPClassifier(
        hidden_layer_sizes=params[0],
        learning_rate_init=params[1],
        max_iter=params[2],
        random_state=42,
        verbose=False
    )

    mlp.fit(params[3], params[5])

    y_pred_mlp = mlp.predict(params[4])
    y_proba_mlp = mlp.predict_proba(params[4])[:, 1]

    #report_all_metrics(params[6], y_pred_mlp, y_proba_mlp, prefix="MLP Classifier")

    cm_mlp = confusion_matrix(params[6], y_pred_mlp)
    plot_confusion_matrix(
        cm_mlp,
        os.path.join(mlp_dir, "Confusion Matrix"), 
        title="Confusion Matrix – MLP Classifier"
    )

    plot_roc_curve(
        params[6],
        y_proba_mlp,
        os.path.join(mlp_dir, "ROC Curve"),         
        title="ROC Curve – MLP Classifier"
    )

    # Save raw MLP weights
    np.save(os.path.join(mlp_dir, "mlp_weights.npy"), mlp.coefs_)
    np.save(os.path.join(mlp_dir, "mlp_biases.npy"), mlp.intercepts_)
    
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
    y = df["HeartDisease"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

    scaler = StandardScaler()
    x_train_scaled, x_test_scaled = scaler.fit_transform(x_train), scaler.transform(x_test)

    hidden_layer_sizes = [[10, 5, 2, 1]]
    learning_rate_inits = [0.0001, 0.001, 0.01, 0.1]
    max_iterations = [200, 400, 600, 800, 1000]

    combinations = itertools.product(hidden_layer_sizes, learning_rate_inits, max_iterations, [x_train_scaled], [x_test_scaled], [y_train], [y_test])

    for combination in combinations:
        worker(combination)

if __name__ == "__main__":
    main()

