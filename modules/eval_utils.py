# eval_utils.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay, accuracy_score, f1_score, recall_score, precision_score
)

def evaluate_and_plot(model, loader, device, plot_dir="plots", log_file="logs/eval_log.txt"):
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y_true.extend(y.numpy())

            y_pred.extend(torch.sigmoid(model(x)).cpu().numpy().flatten())
            # y_pred.extend(model(x))


    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_bin = (y_pred > 0.5).astype(int)

    # === Metrics ===
    acc  = accuracy_score(y_true, y_bin)
    prec = precision_score(y_true, y_bin)
    rec  = recall_score(y_true, y_bin)
    f1   = f1_score(y_true, y_bin)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_val = auc(fpr, tpr)
    pr, rc, _ = precision_recall_curve(y_true, y_pred)

    # === Equal Error Rate (EER) ===
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

    # === Log ===
    with open(log_file, "a") as f:
        f.write(
            f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, "
            f"F1={f1:.4f}, AUC={auc_val:.4f}, EER={eer:.4f} (thr={eer_threshold:.4f})\n"
        )

    # === Plots ===
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}, EER={eer:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, "roc_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(rc, pr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig(os.path.join(plot_dir, "pr_curve.png"))
    plt.close()

    cm = confusion_matrix(y_true, y_bin, normalize="true")
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(plot_dir, "conf_matrix.png"))
    plt.close()

    print(f"[Eval] Acc={acc:.4f}, F1={f1:.4f}, AUC={auc_val:.4f}, EER={eer:.4f}")

