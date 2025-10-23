# ============================================================
# evaluate_simple_fusion.py — ASVspoof2019 (eval) + ITW (eval)
# Robust to PyTorch 2.6 weights_only change; no CLI, config at top.
# ============================================================
import os
import glob
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
)

# ---------------- Config (edit here) ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERT_NAMES = ["style", "linguistic", "hubert", "wavelm", "emotion2vec"]  # which experts to include
CACHE_DIR = "cache"
CKPT_DIR  = "model_checkpoints"
OUT_DIR   = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Try these patterns to locate each expert's checkpoint
CANDIDATE_PATTERNS = [
    "{root}/{exp}/best_classifier.pt",
    "{root}/{exp}/best_classifier.pth",
    "{root}/{exp}/best.pt",
    "{root}/{exp}_classifier/best_classifier.pt",
    "{root}/{exp}_classifier/best.pt",
    "{root}/experts/{exp}/best_classifier.pt",
    "{root}/experts/{exp}/best.pt",
]

# Optional: datasets to evaluate (must have caches under cache/<dataset>/<expert>/eval_*.npy)
EVAL_DATASETS = ["ASVspoof2019", "ITW"]

# ---------- Head factory import (robust) ----------
HeadFactory = None
try:
    from modules.classifier_head import get_head as HeadFactory
except Exception:
    try:
        from classifier_head import get_head as HeadFactory
    except Exception:
        HeadFactory = None

def _default_linear(input_dim: int):
    import torch.nn as nn
    return nn.Linear(input_dim, 1)

# --------------------- Utils ---------------------
def fmt3(x):  return f"{x:.3f}"
def fmt_pct(x): return f"{100*x:.2f}"

def compute_eer(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(fpr[idx]), float(thr[idx])

def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray):
    eer, th = compute_eer(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr, tpr)
    y_hat = (y_score >= th).astype(int)
    return {
        "eer": eer,
        "eer_threshold": th,
        "auc": auc_val,
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "precision": precision_score(y_true, y_hat, zero_division=0),
        "recall": recall_score(y_true, y_hat, zero_division=0),
        "acc": accuracy_score(y_true, y_hat),
    }

def extract_state_dict(obj):
    # If the object already looks like a state_dict (tensor values), return it.
    if isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj
    # Common wrappers
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    raise TypeError("Unrecognized checkpoint format for state_dict extraction.")

def clean_state_dict_keys(sd: dict):
    # Properly strip 'module.' once; do not duplicate keys.
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out

def find_checkpoint_for_expert(expert: str) -> str:
    for pat in CANDIDATE_PATTERNS:
        path = pat.format(root=CKPT_DIR, exp=expert)
        if os.path.exists(path):
            return path
    cands = []
    for ext in ("*.pt", "*.pth"):
        for p in glob.glob(os.path.join(CKPT_DIR, "**", ext), recursive=True):
            base = os.path.basename(p).lower()
            if expert.lower() in p.lower() and "best" in base:
                cands.append(p)
    return sorted(cands)[0] if cands else ""

def load_cached(dataset_name: str, expert: str, split="eval"):
    emb = os.path.join(CACHE_DIR, dataset_name, expert, f"{split}_embeddings.npy")
    lab = os.path.join(CACHE_DIR, dataset_name, expert, f"{split}_labels.npy")
    if not (os.path.exists(emb) and os.path.exists(lab)):
        return None, None
    X = np.load(emb)
    y = np.load(lab).astype(int).reshape(-1)
    return X, y

def instantiate_head_from_ckpt(ckpt: dict, input_dim: int):
    head_type = (ckpt.get("head_type") or "linear").lower()
    if HeadFactory is not None:
        try:
            return HeadFactory(head_type, input_dim=input_dim)
        except Exception:
            pass
    return _default_linear(input_dim)

def _torch_load_compat(path: str):
    """
    Force weights_only=False for PyTorch >=2.6, and stay compatible with older versions.
    """
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        # Older PyTorch without weights_only arg
        return torch.load(path, map_location=DEVICE)

def run_classifier_probs(X: np.ndarray, ckpt_path: str) -> np.ndarray:
    """
    Load checkpoint (with weights_only=False), rebuild head, apply saved z-score, return probs (N,).
    """
    raw = _torch_load_compat(ckpt_path)
    sd = clean_state_dict_keys(extract_state_dict(raw))

    mu = raw.get("norm_mean", None)
    sigma = raw.get("norm_std", None)
    if mu is not None and sigma is not None:
        Xn = (X - mu) / sigma
    else:
        Xn = X

    input_dim = Xn.shape[1]
    head = instantiate_head_from_ckpt(raw, input_dim).to(DEVICE)
    head.load_state_dict(sd, strict=False)  # permissive loading across head variants
    head.eval()

    with torch.no_grad():
        logits = head(torch.from_numpy(Xn).float().to(DEVICE)).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

def evaluate_dataset(dataset_name: str, experts: list[str]) -> pd.DataFrame:
    scores = {}
    y_ref = None
    available = []

    for exp in experts:
        X, y = load_cached(dataset_name, exp, split="eval")
        if X is None:
            print(f"[{dataset_name}] Missing embeddings for expert='{exp}', skipping.")
            continue
        if y_ref is None:
            y_ref = y
        elif len(y) != len(y_ref):
            print(f"[WARN] {dataset_name}:{exp} length mismatch ({len(y)} vs {len(y_ref)}), skipping.")
            continue

        ckpt = find_checkpoint_for_expert(exp)
        if not ckpt:
            print(f"[{dataset_name}] No checkpoint found for expert '{exp}', skipping.")
            continue

        probs = run_classifier_probs(X, ckpt)
        scores[exp] = probs
        available.append(exp)

    if not available:
        return pd.DataFrame()

    mat = np.stack([scores[e] for e in available], axis=1)  # (N, E)
    fusion_defs = {
        "mean_fusion":   mat.mean(axis=1),
        "max_fusion":    mat.max(axis=1),
        "median_fusion": np.median(mat, axis=1),
    }

    rows = []
    for method, s in fusion_defs.items():
        m = evaluate_scores(y_ref, s)
        rows.append({"dataset": dataset_name, "method": method, **m})

    for e in available:
        m = evaluate_scores(y_ref, scores[e])
        rows.append({"dataset": dataset_name, "method": f"{e}_alone", **m})

    return pd.DataFrame(rows)

# --------------------- Main ---------------------
if __name__ == "__main__":
    parts = []
    for dname in EVAL_DATASETS:
        df_d = evaluate_dataset(dname, EXPERT_NAMES)
        if not df_d.empty:
            parts.append(df_d)

    if not parts:
        print("[ERROR] Nothing evaluated (missing caches or checkpoints).")
        raise SystemExit(1)

    df = pd.concat(parts, ignore_index=True)
    df_fmt = pd.DataFrame({
        "dataset":       df["dataset"],
        "method":        df["method"],
        "eer (%)":       df["eer"].apply(fmt_pct),
        "auc":           df["auc"].apply(fmt3),
        "f1":            df["f1"].apply(fmt3),
        "precision":     df["precision"].apply(fmt3),
        "recall":        df["recall"].apply(fmt3),
        "acc":           df["acc"].apply(fmt3),
        "eer_threshold": df["eer_threshold"].apply(fmt3),
    })

    out_csv = os.path.join(OUT_DIR, "simple_fusions_ASV19_and_ITW.csv")
    df_fmt.to_csv(out_csv, index=False)

    print("\n[Simple Fusions] Combined results")
    print(df_fmt.to_string(index=False))
    print(f"\n[✓] Wrote CSV to {out_csv}")
