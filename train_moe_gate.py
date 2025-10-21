# scripts/train_moe_gate.py
import os
import numpy as np
import torch
from torch import nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

from modules.classifier_head import LinearHead
from modules.moe_head import LateFusionBinaryMoE

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERT_NAMES = ["style", "linguistic", "hubert", "wavelm"]

CACHE_DIR = "cache"
CKPT_DIR  = "model_checkpoints"
OUT_DIR   = "logs/moe"
PLOT_DIR  = "plots/moe"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

BATCH_SIZE = 256
LR = 5e-4
EPOCHS = 60
PATIENCE = 10
SEED = 1337

def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------- Utilities ----------
def load_cached_embeddings(expert: str, split: str):
    epath = os.path.join(CACHE_DIR, expert, f"{split}_embeddings.npy")
    lpath = os.path.join(CACHE_DIR, expert, f"{split}_labels.npy")
    if not (os.path.exists(epath) and os.path.exists(lpath)):
        raise FileNotFoundError(f"Cache missing for {expert} {split}: {epath}, {lpath}")
    embs = np.load(epath)
    labs = np.load(lpath)
    return embs, labs

def load_head(expert: str, input_dim: int):
    head = LinearHead(input_dim=input_dim)
    ckpt = os.path.join(CKPT_DIR, expert, "best_classifier.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing classifier checkpoint for {expert}: {ckpt}")
    state = torch.load(ckpt, map_location="cpu")
    head.load_state_dict(state)
    head.eval()
    return head.to(DEVICE)

@torch.no_grad()
def compute_logits_for_split(experts, split: str):
    """
    Returns:
        logits_matrix: (N, E) numpy float32 of raw logits
        labels:       (N,) numpy int/float labels (0/1)

    Assumes all experts' caches for this split were created with the same
    sample order (i.e., dataloader used shuffle=False).
    """
    embs_list, labels_list, heads = [], [], []
    for expert in experts:
        embs, labs = load_cached_embeddings(expert, split)
        embs_list.append(embs)
        labels_list.append(labs)
        heads.append(load_head(expert, embs.shape[1]))

    # ---- alignment check ----
    for labs in labels_list[1:]:
        if not np.array_equal(labels_list[0], labs):
            raise ValueError(f"[ALIGNMENT ERROR] Label mismatch across experts for split='{split}'. "
                             f"Re-extract with shuffle=False for every expert.")
    labels = labels_list[0].astype(np.float32)
    N = embs_list[0].shape[0]
    E = len(experts)

    # batched inference
    logits_all = np.zeros((N, E), dtype=np.float32)
    bs = 2048
    for i in range(0, N, bs):
        j = min(i + bs, N)
        for e_idx, (embs, head) in enumerate(zip(embs_list, heads)):
            x = torch.tensor(embs[i:j], dtype=torch.float32, device=DEVICE)
            out = head(x)                     # [b,1] logits
            logits_all[i:j, e_idx] = out.squeeze(1).cpu().numpy()

    return logits_all, labels

# --------- Dataset ----------
class MoELogitsDataset(Dataset):
    """Serves vector of expert logits (E,) and label."""
    def __init__(self, logits_matrix: np.ndarray, labels: np.ndarray):
        assert logits_matrix.shape[0] == labels.shape[0]
        self.X = torch.tensor(logits_matrix, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --------- Metrics / Plots ----------
def evaluate_logits(y_true, y_logit, plot_prefix=None, title_prefix="MoE"):
    """
    y_logit: raw logits (before sigmoid)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = 1.0 / (1.0 + np.exp(-np.asarray(y_logit)))
    y_bin = (y_score > 0.5).astype(int)

    acc  = accuracy_score(y_true, y_bin)
    prec = precision_score(y_true, y_bin, zero_division=0)
    rec  = recall_score(y_true, y_bin, zero_division=0)
    f1   = f1_score(y_true, y_bin, zero_division=0)

    fpr, tpr, thr = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_idx]
    eer_thr = thr[eer_idx]

    if plot_prefix:
        import matplotlib.pyplot as plt
        pr, rc, _ = precision_recall_curve(y_true, y_score)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={auc_val:.3f}, EER={eer:.3f}")
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"{title_prefix} ROC"); plt.legend()
        plt.savefig(f"{plot_prefix}_roc.png"); plt.close()

        plt.figure()
        plt.plot(rc, pr)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title(f"{title_prefix} Precision-Recall")
        plt.savefig(f"{plot_prefix}_pr.png"); plt.close()

        cm = confusion_matrix(y_true, y_bin, normalize="true")
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay(cm).plot()
        plt.title(f"{title_prefix} Confusion Matrix")
        plt.savefig(f"{plot_prefix}_cm.png"); plt.close()

    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "auc": float(auc_val), "eer": float(eer),
        "eer_threshold": float(eer_thr)
    }

# --------- Training loop ----------
def train_gate(moe, train_loader, val_loader, epochs=EPOCHS, lr=LR, patience=PATIENCE):
    opt = torch.optim.Adam(moe.parameters(), lr=lr)
    crit = nn.BCEWithLogitsLoss()
    best_val_eer = float("inf")
    patience_ctr = 0
    best_state = None

    for ep in range(1, epochs + 1):
        moe.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X = X.to(DEVICE)                 # [B,E]
            y = y.to(DEVICE).unsqueeze(1)    # [B,1]
            opt.zero_grad()
            fused_logit, gate_probs, aux_reg = moe(X)
            loss = crit(fused_logit, y) + aux_reg
            loss.backward()
            opt.step()
            epoch_loss += loss.item()

        # ---- Validation EER on DEV ----
        moe.eval()
        val_logits, val_labels = [], []
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(DEVICE)
                fl, _, _ = moe(Xv)
                val_logits.append(fl.squeeze(1).cpu().numpy())
                val_labels.append(yv.numpy())
        val_logits = np.concatenate(val_logits)
        val_labels = np.concatenate(val_labels)
        fpr, tpr, _ = roc_curve(val_labels, 1/(1+np.exp(-val_logits)))
        fnr = 1 - tpr
        val_eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

        print(f"[MoE] Epoch {ep:03d}  Loss={epoch_loss/len(train_loader):.4f}  DEV EER={val_eer:.4f}")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            patience_ctr = 0
            best_state = {k: v.detach().cpu().clone() for k, v in moe.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("[MoE] Early stopping.")
                break

    if best_state is not None:
        moe.load_state_dict(best_state)
    return moe, best_val_eer

# --------- Simple baselines ----------
def simple_fusions(logits_matrix):
    """
    Return dict of baseline fused raw logits:
      - mean:   mean over experts
      - max:    elementwise max over experts
      - median: elementwise median over experts
    """
    return {
        "mean":   logits_matrix.mean(axis=1),
        "max":    logits_matrix.max(axis=1),
        "median": np.median(logits_matrix, axis=1),
    }

# ---------------- Main ----------------
if __name__ == "__main__":
    set_seed()

    # 1) Build per-expert logits for all splits from cached embeddings + trained heads
    print("[MoE] Computing per-expert logits for TRAIN...")
    train_logits, train_labels = compute_logits_for_split(EXPERT_NAMES, "train")
    print("[MoE] Computing per-expert logits for DEV...")
    dev_logits, dev_labels       = compute_logits_for_split(EXPERT_NAMES, "dev")
    print("[MoE] Computing per-expert logits for EVAL...")
    eval_logits, eval_labels     = compute_logits_for_split(EXPERT_NAMES, "eval")

    # 2) Datasets / loaders
    train_ds = MoELogitsDataset(train_logits, train_labels)
    dev_ds   = MoELogitsDataset(dev_logits, dev_labels)
    eval_ds  = MoELogitsDataset(eval_logits, eval_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    dev_loader   = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    eval_loader  = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 3) Init MoE gate (logits as inputs). If you see collapse, set entropy_reg=1e-3.
    moe = LateFusionBinaryMoE(
        num_experts=len(EXPERT_NAMES),
        hidden=128,
        use_logits_as_gate_input=True,
        temperature_init=1.0,
        entropy_reg=0.0,
    ).to(DEVICE)

    # 4) Train on TRAIN, early stop on DEV
    moe, best_dev_eer = train_gate(moe, train_loader, dev_loader, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    print(f"[MoE] Best DEV EER: {best_dev_eer:.4f}")

    # 5) Evaluate on EVAL
    moe.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for X, y in eval_loader:
            X = X.to(DEVICE)
            fl, _, _ = moe(X)
            all_logits.append(fl.squeeze(1).cpu().numpy())
            all_labels.append(y.numpy())
    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    moe_metrics = evaluate_logits(all_labels, all_logits, plot_prefix=os.path.join(PLOT_DIR, "moe_eval"), title_prefix="MoE (EVAL)")
    print(f"[MoE][EVAL] Acc={moe_metrics['accuracy']:.4f}, F1={moe_metrics['f1']:.4f}, "
          f"AUC={moe_metrics['auc']:.4f}, EER={moe_metrics['eer']:.4f}")

    # 6) Compare to individual experts and simple fusions on EVAL
    results = []
    # per-expert
    for i, name in enumerate(EXPERT_NAMES):
        m = evaluate_logits(all_labels, eval_logits[:, i], plot_prefix=None, title_prefix=f"{name} (EVAL)")
        m["method"] = name
        results.append(m)
    # simple fusions
    for tag, vec in simple_fusions(eval_logits).items():
        m = evaluate_logits(all_labels, vec, plot_prefix=None, title_prefix=f"{tag} fusion (EVAL)")
        m["method"] = f"{tag}_fusion"
        results.append(m)
    # MoE
    m = dict(moe_metrics); m["method"] = "moe"
    results.append(m)

    # 7) Save table (formatted)
    cols = ["method","accuracy","f1","auc","eer","eer_threshold","precision","recall"]
    df = pd.DataFrame(results)[cols]

    # format helpers
    fmt_pct = lambda x: f"{x*100:.2f}"     # percentages with 2 decimals
    fmt3    = lambda x: f"{x:.3f}"         # plain numbers with 3 decimals

    df_fmt = pd.DataFrame({
        "method":         df["method"],
        "accuracy (%)":   df["accuracy"].apply(fmt_pct),
        "eer (%)":        df["eer"].apply(fmt_pct),
        "auc":            df["auc"].apply(fmt3),
        "f1":             df["f1"].apply(fmt3),
        "precision":      df["precision"].apply(fmt3),
        "recall":         df["recall"].apply(fmt3),
        "eer_threshold":  df["eer_threshold"].apply(fmt3),
    })

    out_csv = os.path.join(OUT_DIR, "expert_vs_moe_eval.csv")
    df_fmt.to_csv(out_csv, index=False)

    print("\n[MoE] EVAL Comparison")
    print(df_fmt.to_string(index=False))
    print(f"[MoE] Wrote comparison table to {out_csv}")

