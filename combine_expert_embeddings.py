import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

from modules.classifier_head import LinearHead
from modules.train_utils import train_classifier
from modules.eval_utils import evaluate_and_plot

# ---------------- CONFIG ----------------
CACHE_DIR = "cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100

EXPERTS = ["style", "linguistic", "hubert"]
COMBINED_NAME = "combined_model"
LOG_DIR = f"logs/{COMBINED_NAME}"
PLOT_DIR = f"plots/{COMBINED_NAME}"
CKPT_DIR = f"model_checkpoints/{COMBINED_NAME}"

for d in [LOG_DIR, PLOT_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)


# ---------------- DATASET ----------------
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------- UTILS ----------------
def load_embeddings(expert, split):
    emb_path = os.path.join(CACHE_DIR, expert, f"{split}_embeddings.npy")
    lab_path = os.path.join(CACHE_DIR, expert, f"{split}_labels.npy")

    if not (os.path.exists(emb_path) and os.path.exists(lab_path)):
        print(f"[WARN] Missing data for {expert} ({split}) — skipping.")
        return None, None

    X = np.load(emb_path)
    y = np.load(lab_path)

    # Flatten if multi-dimensional
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return X, y


# ---------------- MAIN PIPELINE ----------------
def combine_embeddings(splits=["train", "dev", "eval"]):
    combined = {}

    for split in splits:
        emb_list, labels = [], None
        dims = []

        print(f"\n=== Processing split: {split} ===")
        for expert in EXPERTS:
            X, y = load_embeddings(expert, split)
            if X is None:
                continue

            emb_list.append(X)
            dims.append(X.shape[1])
            if labels is None:
                labels = y

        if len(emb_list) == 0:
            raise RuntimeError(f"No embeddings found for split {split}.")

        # Match number of samples
        min_samples = min([x.shape[0] for x in emb_list])
        emb_list = [x[:min_samples] for x in emb_list]
        labels = labels[:min_samples]

        # Determine lowest dimension
        min_dim = min(dims)
        print(f"[INFO] Lowest embedding dim = {min_dim}")

        # Reduce higher-dimensional experts to match lowest dim
        aligned_embs = []
        for X in emb_list:
            if X.shape[1] > min_dim:
                print(f"  ↳ Reducing {X.shape[1]} → {min_dim} using PCA")
                X = PCA(n_components=min_dim, random_state=42).fit_transform(X)
            aligned_embs.append(X)

        # Concatenate across experts
        combined_emb = np.concatenate(aligned_embs, axis=1)
        print(f"[INFO] Combined embedding shape ({split}): {combined_emb.shape}")

        combined[split] = (combined_emb, labels)

    return combined


# ---------------- TRAIN + EVAL ----------------
def main():
    combined = combine_embeddings()

    # Prepare datasets
    X_train, y_train = combined["train"]
    X_dev, y_dev = combined["dev"]
    X_eval, y_eval = combined["eval"]

    train_ds = NumpyDataset(X_train, y_train)
    dev_ds = NumpyDataset(X_dev, y_dev)
    eval_ds = NumpyDataset(X_eval, y_eval)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)
    eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE)

    # Model
    input_dim = X_train.shape[1]
    model = LinearHead(input_dim).to(DEVICE)

    print("\n[TRAIN] Training combined classifier...")
    save_path = os.path.join(CKPT_DIR, "best_classifier.pt")
    train_classifier(model, train_loader, dev_loader, DEVICE, lr=LR, epochs=EPOCHS, patience=15, save_path=save_path)

    print("\n[EVAL] Evaluating combined classifier...")
    log_file = os.path.join(LOG_DIR, "eval_log.txt")
    metrics = evaluate_and_plot(model, eval_loader, DEVICE, plot_dir=PLOT_DIR, log_file=log_file)

    print("\n✅ Training complete. Results:")
    print(metrics)


if __name__ == "__main__":
    main()
