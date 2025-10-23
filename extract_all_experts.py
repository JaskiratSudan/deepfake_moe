# ============================================================
# extract_all_experts.py  (updated to also extract ITW)
# ============================================================
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Encoders and dataset/utils
from encoders.experts import (
    StyleEncoder, LinguisticEncoder,
    WaveLMEncoder, HuBERTEncoder, Emotion2VecEncoder
)
from modules.dataset_utils import extract_and_cache_embeddings
from modules.data_loader import ASVspoof2019Dataset, InTheWildDataset

# ---------------- Config ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True

# Where to cache embeddings
CACHE_DIR = "cache"

# ASVspoof2019 paths (leave as-is if yours are already right)
ASV19_ROOTS = {
    "train": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac",  # e.g., "/scratch"
    "dev":   "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac",               # e.g., "/scratch"
    "eval":  "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac",               # e.g., "/scratch"
}

ASV19_PROTOCOLS = {
    "train": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt",
    "dev":   "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt",
    "eval":  "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt",
}

# ✅ ITW paths (SET THESE to your actual locations)
ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"               # folder containing wav/flac files
ITW_META_CSV = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"   # CSV with at least ['file','label']

EXPERTS = {
    "style":      StyleEncoder,
    "linguistic": LinguisticEncoder,
    "hubert":     HuBERTEncoder,
    "wavelm":     WaveLMEncoder,
    "emotion2vec": Emotion2VecEncoder,
}

def make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    for name, EncClass in EXPERTS.items():
        print(f"\n==================== {name.upper()} ====================")
        model = EncClass(freeze_model=True).to(DEVICE)
        model.eval()

        # --------- ASVspoof2019 (train/dev/eval with per-split roots) ----------
        asv_cache_root = ensure_dir(os.path.join(CACHE_DIR, "ASVspoof2019", name))
        for split, protocol_file in ASV19_PROTOCOLS.items():
            root_dir = ASV19_ROOTS.get(split, "")
            if not root_dir:
                print(f"[WARN] Missing ASV19 root for split '{split}', skipping.")
                continue
            if not os.path.exists(protocol_file):
                print(f"[WARN] Missing ASV19 protocol for split '{split}': {protocol_file}")
                continue

            print(f"[ASVspoof2019] {name} → split={split}")
            ds = ASVspoof2019Dataset(protocol_file=protocol_file, root_dir=root_dir)
            dl = make_loader(ds)
            extract_and_cache_embeddings(
                encoder=model,
                dataloader=dl,
                cache_dir=asv_cache_root,
                split_name=split,
                device=DEVICE,
            )

        # --------- InTheWild (eval only) ----------
        itw_cache_root = ensure_dir(os.path.join(CACHE_DIR, "ITW", name))
        if os.path.exists(ITW_META_CSV):
            print(f"[ITW] {name} → split=eval")
            itw_ds = InTheWildDataset(root_dir=ITW_ROOT, protocol_file=ITW_META_CSV, subset="all")
            itw_dl = make_loader(itw_ds)
            extract_and_cache_embeddings(
                encoder=model,
                dataloader=itw_dl,
                cache_dir=itw_cache_root,
                split_name="eval",   # keep as 'eval' for symmetry
                device=DEVICE,
            )
        else:
            print(f"[WARN] ITW metadata CSV not found at {ITW_META_CSV}; skipping ITW for {name}")

        del model
        torch.cuda.empty_cache()

    print("\n✅ All experts processed successfully.")
