import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# ============================================================
# CONFIG
# ============================================================
CACHE_DIR = "cache/"
SPLIT_NAME = "eval"   # "train", "dev", or "eval"
SAVE_FIG = True
FIG_PATH = f"plots/expert_correlation_{SPLIT_NAME}.png"
N_SAMPLES = 2000       

# ============================================================
# Load cached embeddings
# ============================================================
def load_cached_embeddings(cache_dir, split_name, n_samples=None):
    experts, emb_arrays = [], []

    for expert_name in sorted(os.listdir(cache_dir)):
        expert_dir = os.path.join(cache_dir, expert_name)
        emb_path = os.path.join(expert_dir, f"{split_name}_embeddings.npy")

        if not os.path.exists(emb_path):
            print(f"[WARN] No cached embeddings for {expert_name}, skipping.")
            continue

        embs = np.load(emb_path)
        embs = embs.reshape(len(embs), -1)
        if n_samples is not None:
            embs = embs[:n_samples]

        # Normalize
        embs = embs - embs.mean(axis=0, keepdims=True)
        embs = embs / (embs.std(axis=0, keepdims=True) + 1e-8)

        experts.append(expert_name)
        emb_arrays.append(embs)
        print(f"[LOAD] {expert_name}: {embs.shape}")

    return experts, emb_arrays

# ============================================================
# Dimension alignment using UMAP
# ============================================================
def align_dimensions_with_umap(emb_arrays):
    dims = [e.shape[1] for e in emb_arrays]
    min_dim = min(dims)

    print(f"[INFO] Aligning all embeddings to {min_dim} dims using UMAP...")
    aligned = []
    for embs in emb_arrays:
        if embs.shape[1] == min_dim:
            aligned.append(embs)
        else:
            reducer = umap.UMAP(n_components=min_dim, random_state=42)
            reduced = reducer.fit_transform(embs)
            aligned.append(reduced)
    return aligned

# ============================================================
# Compute correlation matrix
# ============================================================
def compute_inter_expert_correlation(emb_arrays, experts):
    n = len(experts)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            e1 = emb_arrays[i].mean(axis=0)
            e2 = emb_arrays[j].mean(axis=0)
            corr = np.corrcoef(e1, e2)[0, 1]
            corr_matrix[i, j] = corr
    return corr_matrix

# ============================================================
# Custom colormap: red = ±1, black = 0
# ============================================================
from matplotlib.colors import LinearSegmentedColormap
red_black = LinearSegmentedColormap.from_list("red_black", [(0, "red"), (0.5, "black"), (1, "red")])

# ============================================================
# Plot correlation heatmap
# ============================================================
def plot_correlation_heatmap(corr_matrix, experts, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap=red_black,
        vmin=-1,
        vmax=1,
        xticklabels=experts,
        yticklabels=experts,
        square=True,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("Inter-Expert Embedding Correlation")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=400)
        print(f"[SAVE] Correlation heatmap saved at: {save_path}")
    plt.show()

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    experts, emb_arrays = load_cached_embeddings(CACHE_DIR, SPLIT_NAME, N_SAMPLES)
    if len(emb_arrays) < 2:
        raise RuntimeError("Need embeddings from at least 2 experts to compute correlation.")

    emb_arrays = align_dimensions_with_umap(emb_arrays)
    corr_matrix = compute_inter_expert_correlation(emb_arrays, experts)
    plot_correlation_heatmap(corr_matrix, experts, FIG_PATH if SAVE_FIG else None)

    print("\n[INFO] Pairwise Correlation Matrix:")
    for i, name_i in enumerate(experts):
        for j, name_j in enumerate(experts):
            print(f"{name_i} vs {name_j}: {corr_matrix[i, j]:.3f}")
