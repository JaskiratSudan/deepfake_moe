# scripts/eval_simple_fusions.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

from modules.classifier_head import LinearHead

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERT_NAMES = ["style", "linguistic", "hubert", "wavelm"]  # adjust if needed

CACHE_DIR = "cache"
CKPT_DIR  = "model_checkpoints"
OUT_DIR   = "logs/fusion"
os.makedirs(OUT_DIR, exist_ok=True)

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
        logits_matrix: (N, E) float32 raw logits
        labels:        (N,)  float32 labels (0/1)

    Assumes caches for this split were created with shuffle=False for all experts.
    """
    embs_list, labels_list, heads = [], [], []
    for expert in experts:
        embs, labs = load_cached_embeddings(expert, split)
        embs_list.append(embs)
        labels_list.append(labs)
        heads.append(load_head(expert, embs.shape[1]))

    # alignment check
    for labs in labels_list[1:]:
        if not np.array_equal(labels_list[0], labs):
            raise ValueError(f"[ALIGNMENT ERROR] Label mismatch across experts for split='{split}'. "
                             f"Re-extract with shuffle=False consistently.")
    labels = labels_list[0].astype(np.float32)
    N = embs_list[0].shape[0]
    E = len(experts)

    logits_all = np.zeros((N, E), dtype=np.float32)
    bs = 2048
    for i in range(0, N, bs):
        j = min(i + bs, N)
        for e_idx, (embs, head) in enumerate(zip(embs_list, heads)):
            x = torch.tensor(embs[i:j], dtype=torch.float32, device=DEVICE)
            out = head(x)                                # [b,1] logits
            logits_all[i:j, e_idx] = out.squeeze(1).cpu().numpy()

    return logits_all, labels

# --------- Metrics ----------
def evaluate_logits(y_true, y_logit):
    """
    y_logit: raw logits (before sigmoid)
    Returns dict of metrics (accuracy, eer, auc, f1, precision, recall, eer_threshold)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = 1.0 / (1.0 + np.exp(-np.asarray(y_logit)))  # sigmoid
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

    return {
        "accuracy": float(acc),
        "eer": float(eer),
        "auc": float(auc_val),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "eer_threshold": float(eer_thr),
    }

def simple_fusions(logits_matrix):
    """
    Return dict of baseline fused raw logits (no learning):
      - mean:   mean over experts
      - max:    elementwise max over experts
      - median: elementwise median over experts
    """
    return {
        "mean_fusion":   logits_matrix.mean(axis=1),
        "max_fusion":    logits_matrix.max(axis=1),
        "median_fusion": np.median(logits_matrix, axis=1),
    }

# ---------------- Main ----------------
if __name__ == "__main__":
    set_seed()

    # 1) Per-expert logits for EVAL
    print("[FUSION] Computing per-expert logits for EVAL...")
    eval_logits, eval_labels = compute_logits_for_split(EXPERT_NAMES, "eval")

    # 2) Evaluate per-expert and simple fusions
    results = []

    # per-expert baselines
    for i, name in enumerate(EXPERT_NAMES):
        m = evaluate_logits(eval_labels, eval_logits[:, i])
        m["method"] = name
        results.append(m)

    # simple, non-learned fusions
    for tag, vec in simple_fusions(eval_logits).items():
        m = evaluate_logits(eval_labels, vec)
        m["method"] = tag
        results.append(m)

    # 3) Save/print formatted table
    import pandas as pd
    cols = ["method","accuracy","eer","auc","f1","precision","recall","eer_threshold"]
    df = pd.DataFrame(results)[cols]

    # formatting: accuracy & EER as percentages (2 decimals); others 3 decimals
    fmt_pct = lambda x: f"{x*100:.2f}"
    fmt3    = lambda x: f"{x:.3f}"

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

    out_csv = os.path.join(OUT_DIR, "eval_simple_fusions.csv")
    df_fmt.to_csv(out_csv, index=False)

    print("\n[Simple Fusions] EVAL Comparison")
    print(df_fmt.to_string(index=False))
    print(f"[Simple Fusions] Wrote CSV to {out_csv}")
