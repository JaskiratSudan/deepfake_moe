# ============================================================
# extract_all_experts.py
# ============================================================
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your encoders and utils
from encoders.experts import (
    StyleEncoder, LinguisticEncoder,
    WaveLMEncoder, HuBERTEncoder, Emotion2VecEncoder
)
from modules.dataset_utils import extract_and_cache_embeddings, CachedEmbeddingDataset
from modules.data_loader import ASVspoof2019Dataset  

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache_root = "cache"
split_name = "train"  # choose "train", "dev", or "eval"
batch_size = 8
num_workers = 4
num_samples = None  # optionally limit samples for quick tests

# ---------------- DATASET LOADING ----------------
datasets_info = {
    "train": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt",
        "num_samples": num_samples,
    },
    "dev": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt",
        "num_samples": num_samples,
    },
    "eval": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt",
        "num_samples": num_samples,
    },
}

print(f"[LOAD] Preparing {split_name} dataset...")
info = datasets_info[split_name]
optional_args = {k: v for k, v in info.items() if k not in ["root_dir", "protocol_file"]}
dataset = ASVspoof2019Dataset(root_dir=info["root_dir"], protocol_file=info["protocol_file"], **optional_args)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print(f"[DATA] Loaded {len(dataset)} samples from {split_name} split.\n")

# ---------------- EXPERT DEFINITIONS ----------------
EXPERTS = {
    "style": StyleEncoder,
    "linguistic": LinguisticEncoder,
    "wavelm": WaveLMEncoder,
    "hubert": HuBERTEncoder,
    "emotion2vec": Emotion2VecEncoder,
}

# ---------------- EXTRACTION LOOP ----------------
for name, model_cls in EXPERTS.items():
    print(f"\n======================\n[EXPERT] {name}\n======================")

    expert_cache = os.path.join(cache_root, name)
    os.makedirs(expert_cache, exist_ok=True)

    emb_path = os.path.join(expert_cache, f"{split_name}_embeddings.npy")
    lab_path = os.path.join(expert_cache, f"{split_name}_labels.npy")

    # Skip if already cached
    if os.path.exists(emb_path) and os.path.exists(lab_path):
        print(f"[CACHE] Found cached {name} embeddings → Skipping extraction.")
        continue

    # Load model lazily
    print(f"[LOAD] Loading {name} model...")
    model = model_cls().to(device)

    # Extract and cache embeddings
    print(f"[RUN] Extracting {name} embeddings...")
    extract_and_cache_embeddings(
        encoder=model,
        dataloader=dataloader,
        cache_dir=expert_cache,
        split_name=split_name,
        device=device,
    )

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print(f"[DONE] Finished {name} extraction.\n")

print("\n✅ All experts processed successfully.")
