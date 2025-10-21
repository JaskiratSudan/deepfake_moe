# dataset_utils.py

import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

# ============================================================
# Extract and Cache Embeddings
# ============================================================
def extract_and_cache_embeddings(encoder, dataloader, cache_dir, split_name, device):
    """
    Extract embeddings from a pretrained encoder and cache them.

    Args:
        encoder (nn.Module): Pretrained encoder (e.g., Style or Linguistic).
        dataloader (DataLoader): DataLoader for the split.
        cache_dir (str): Directory to save cached embeddings.
        split_name (str): Dataset split name ("train", "dev", "eval").
        device (torch.device): CUDA or CPU device.

    Returns:
        (embeddings_path, labels_path)
    """
    os.makedirs(cache_dir, exist_ok=True)

    emb_path = os.path.join(cache_dir, f"{split_name}_embeddings.npy")
    lab_path = os.path.join(cache_dir, f"{split_name}_labels.npy")

    # ---- Check for existing cache ----
    if os.path.exists(emb_path) and os.path.exists(lab_path):
        print(f"[CACHE] Found existing embeddings for '{split_name}' — skipping extraction.")
        return emb_path, lab_path

    encoder.eval()
    all_embs, all_labels = [], []
    num_samples = len(dataloader.dataset)
    print(f"\n[INFO] Extracting embeddings for '{split_name}' split ({num_samples} samples)...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting {split_name}", ncols=90):
            # Handle variable batch tuple formats
            if len(batch) == 2:
                wavs, labels = batch
            elif len(batch) == 3:
                wavs, labels, _ = batch  # ignore any third item like 'path' or 'id'
            else:
                raise ValueError(f"Unexpected batch format ({len(batch)} elements)")

            wavs = wavs.to(device)

            # Forward pass
            features = encoder(wavs)

            # Normalize shape → average across temporal/layer dimensions
            if features.dim() == 4:
                # shape: (B, L, H, T)
                features = features.mean(dim=(1, 3))  # → (B, H)
            elif features.dim() == 3:
                # shape: (B, T, H)
                features = features.mean(dim=1)       # → (B, H)
            elif features.dim() == 2:
                # Already flattened: (B, H)
                pass
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")

            all_embs.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # ---- Stack all results ----
    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ---- Save cache ----
    np.save(emb_path, all_embs)
    np.save(lab_path, all_labels)
    print(f"[INFO] Saved cached embeddings for '{split_name}' at:\n"
          f"       {emb_path}\n       {lab_path}\n"
          f"       → Total: {len(all_embs)} samples | Feature dim: {all_embs.shape[1]}")

    return emb_path, lab_path


# ============================================================
# Cached Embedding Dataset
# ============================================================
class CachedEmbeddingDataset(Dataset):
    """
    Dataset for loading cached embeddings and labels from disk.
    """
    def __init__(self, emb_path, lab_path):
        self.embs = torch.tensor(np.load(emb_path), dtype=torch.float32)
        self.labels = torch.tensor(np.load(lab_path), dtype=torch.float32)

        assert len(self.embs) == len(self.labels), \
            f"Mismatch: {len(self.embs)} embeddings vs {len(self.labels)} labels."

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]
