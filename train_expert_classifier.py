# scripts/train_expert_classifier.py
"""
Train per-expert binary classifier heads on cached embeddings (no CLI args).
- Z-score normalization (fit on train, saved in checkpoint)
- pos_weight for class imbalance
- Early stopping on DEV EER
- Cosine LR with warmup
- Saves: model_checkpoints/<expert>/best_classifier.pt
"""

import os
import numpy as np
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Optional

# ===========================
# USER CONFIG (edit here)
# ===========================
EXPERTS = ["style", "linguistic", "hubert", "wavelm", "emotion2vec"]  # which experts to train

# per-expert head choice: "linear" | "resmlp" | "cosine"
HEAD_TYPE_PER_EXPERT = {
    "style": "resmlp",
    "linguistic": "resmlp",
    "hubert": "cosine",
    "wavelm": "cosine",
    "emotion2vec": "cosine",
}

CACHE_DIR = "cache"
CKPT_DIR  = "model_checkpoints"
DATASET_NAME_FOR_TRAIN = "ASVspoof2019"  # we train heads on ASVspoof2019 train/dev caches

BATCH_SIZE   = 256
EPOCHS       = 50
LR           = 1e-3
WEIGHT_DECAY = 5e-3
WARMUP_STEPS = 200
PATIENCE     = 8
NUM_WORKERS  = 4
SEED         = 1337
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Optional: override pos_weight per expert (None -> auto from train labels)
POS_WEIGHT_OVERRIDES = {
    # "style": 1.5,
    # "linguistic": None,
    # "hubert": None,
    # "wavelm": None,
}

# ===========================
# Imports from your module
# ===========================
try:
    from modules.classifier_head import get_head
except Exception:
    from classifier_head import get_head


# ===========================
# Helpers
# ===========================
def set_seed(seed: int = 1337):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def compute_eer(y_true: np.ndarray, y_score: np.ndarray):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(fpr[idx]), float(thr[idx])

def evaluate_metrics(y_true: np.ndarray, y_score: np.ndarray):
    from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
    eer, th = compute_eer(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    y_hat = (y_score >= th).astype(int)
    return {
        "eer": eer, "eer_threshold": th, "auc": auc_val,
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
        "acc": accuracy_score(y_true, y_hat),
    }

def zscore_fit(X):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-6
    return mu.astype(np.float32), sigma.astype(np.float32)

def zscore_apply(X, mu, sigma):
    return (X - mu) / sigma

class NpDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y.astype(np.float32)).float().view(-1, 1)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

@dataclass
class TrainConfig:
    expert: str
    head_type: str
    cache_dir: str = CACHE_DIR
    ckpt_dir: str  = CKPT_DIR
    batch_size: int = BATCH_SIZE
    epochs: int     = EPOCHS
    lr: float       = LR
    weight_decay: float = WEIGHT_DECAY
    warmup_steps: int   = WARMUP_STEPS
    patience: int       = PATIENCE
    num_workers: int    = NUM_WORKERS
    seed: int           = SEED
    device: str         = DEVICE
    pos_weight: Optional[float] = None


def prepare_data(cfg: TrainConfig, split: str):
    root = os.path.join(cfg.cache_dir, DATASET_NAME_FOR_TRAIN, cfg.expert)
    X = np.load(os.path.join(root, f"{split}_embeddings.npy"))
    y = np.load(os.path.join(root, f"{split}_labels.npy")).astype(int).reshape(-1)
    return X, y

def get_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        # cosine decay after warmup
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    running = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        running += float(loss.item()) * Xb.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def infer_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for Xb, yb in loader:
        Xb = Xb.to(device, non_blocking=True)
        logits = model(Xb).squeeze(1)
        p = torch.sigmoid(logits).cpu().numpy()
        probs.append(p)
        labels.append(yb.cpu().numpy().reshape(-1))
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0)

def train_one_expert(expert: str, head_type: str):
    print(f"\n===== Training expert: {expert} (head={head_type}) =====")
    cfg = TrainConfig(
        expert=expert,
        head_type=head_type,
        pos_weight=POS_WEIGHT_OVERRIDES.get(expert, None),
    )

    os.makedirs(os.path.join(cfg.ckpt_dir, cfg.expert), exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- Load data ---
    Xtr, ytr = prepare_data(cfg, "train")
    Xdv, ydv = prepare_data(cfg, "dev")

    # z-score (fit on train)
    mu, sigma = zscore_fit(Xtr)
    Xtr = zscore_apply(Xtr, mu, sigma)
    Xdv = zscore_apply(Xdv, mu, sigma)

    in_dim = Xtr.shape[1]
    model = get_head(cfg.head_type, input_dim=in_dim).to(device)

    # pos_weight
    if cfg.pos_weight is None:
        pos = (ytr == 1).sum()
        neg = (ytr == 0).sum()
        pw = float(neg) / float(max(pos, 1))
    else:
        pw = float(cfg.pos_weight)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_ds = NpDataset(Xtr, ytr)
    dev_ds   = NpDataset(Xdv, ydv)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    dev_loader   = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, pin_memory=True)

    total_steps = cfg.epochs * len(train_loader)
    scheduler = get_scheduler(opt, total_steps=total_steps, warmup_steps=cfg.warmup_steps)

    best_eer = 1e9
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        loss_tr = train_one_epoch(model, train_loader, opt, scheduler, criterion, device)

        # DEV
        p_dev, y_dev = infer_probs(model, dev_loader, device)
        mets = evaluate_metrics(y_dev, p_dev)
        print(f"[{expert:>10s}][Epoch {epoch:02d}] "
              f"loss={loss_tr:.4f} | EER={mets['eer']*100:.2f}% AUC={mets['auc']:.3f} "
              f"F1={mets['f1']:.3f} Acc={mets['acc']:.3f}")

        if mets["eer"] < best_eer:
            best_eer = mets["eer"]; no_improve = 0
            best_state = {
                "state_dict": model.state_dict(),
                "head_type": cfg.head_type,
                "input_dim": in_dim,
                "norm_mean": mu,
                "norm_std": sigma,
                "pos_weight": pw,
                "metrics_dev": mets,
                "config": asdict(cfg),
            }
            ckpt_path = os.path.join(cfg.ckpt_dir, cfg.expert, "best_classifier.pt")
            torch.save(best_state, ckpt_path)
            print(f"  🔸 Saved new best to {ckpt_path} -> {best_eer*100:.2f}% EER")
        else:
            no_improve += 1

        if no_improve >= cfg.patience:
            print(f"Early stopping (no improve {cfg.patience} epochs). Best EER={best_eer*100:.2f}%")
            break

    if best_state is None:
        # still save last
        best_state = {
            "state_dict": model.state_dict(),
            "head_type": head_type,
            "input_dim": in_dim,
            "norm_mean": mu,
            "norm_std": sigma,
            "pos_weight": pw,
            "config": asdict(cfg),
        }
        torch.save(best_state, os.path.join(cfg.ckpt_dir, cfg.expert, "best_classifier.pt"))

def main():
    for exp in EXPERTS:
        head = HEAD_TYPE_PER_EXPERT.get(exp, "resmlp")
        train_one_expert(exp, head)

if __name__ == "__main__":
    main()
