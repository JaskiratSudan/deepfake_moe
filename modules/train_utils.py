# train_utils.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_curve


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    eer_threshold = thresholds[idx]
    return eer, eer_threshold


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    lr=1e-4,
    epochs=10,
    patience=5,
    save_path="best_classifier.pt",
    log_func=print
):
    """Train a binary classifier with early stopping and EER-based validation."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_eer = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        train_acc = total_correct / total_samples if total_samples > 0 else 0.0
        val_eer = evaluate_eer(model, val_loader, device)

        log_func(f"Epoch {epoch + 1}: Loss={total_loss:.4f}, Acc={train_acc:.4f}, Val EER={val_eer:.4f}")

        if not np.isnan(val_eer) and val_eer < best_eer:
            best_eer = val_eer
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            log_func(f"  🔸 Saved new best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log_func("⏹️ Early stopping triggered.")
                break

    log_func(f"✅ Best Validation EER: {best_eer:.4f}")


def evaluate_eer(model, loader, device):
    """Compute EER for model on a given dataset."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            y_pred.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid crash if dataset is empty or only one class
    if len(np.unique(y_true)) < 2:
        return np.nan

    eer, _ = compute_eer(y_true, y_pred)
    return eer
