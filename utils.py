from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,  roc_curve
)
import matplotlib.pyplot as plt
import os

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
def plot_confusion_matrix(cm, path_to_saving_directory, title="Confusion Matrix"):
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
    filename = f"{title}.png"
    os.makedirs(path_to_saving_directory, exist_ok=True)
    plt.savefig(os.path.join(path_to_saving_directory, filename))

# ROC plot
def plot_roc_curve(y_true, y_proba, path_to_saving_directory, title="ROC Curve"):
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
    filename = f"{title}.png"
    os.makedirs(path_to_saving_directory, exist_ok=True)
    plt.savefig(os.path.join(path_to_saving_directory, filename))