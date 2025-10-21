# train_expert_classifier.py


import os
import importlib
import torch
from torch.utils.data import DataLoader
import pandas as pd

from modules.dataset_utils import extract_and_cache_embeddings, CachedEmbeddingDataset
from modules.classifier_head import LinearHead
from modules.train_utils import train_classifier
from modules.eval_utils import evaluate_and_plot
from modules.data_loader import ASVspoof2019Dataset

# ------------------- Config -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "cache/"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50

datasets_info = {
    "train": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt",
    },
    "dev": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt",
    },
    "eval": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt",
    }
}

EXPERT_NAMES = ["style", "linguistic", "hubert", "wavelm", "emotion2vec"]
LOG_DIR = "logs/"
os.makedirs(LOG_DIR, exist_ok=True)


# ------------------- Dynamic Expert Loader -------------------
def get_expert_class(expert_name: str):
    """
    Dynamically fetch expert class from encoders/experts.py
    Handles special capitalization like HuBERT.
    """
    mod = importlib.import_module("encoders.experts")

    # Map expert_name to correct class names
    class_map = {
        "style": "StyleEncoder",
        "linguistic": "LinguisticEncoder",
        "hubert": "HuBERTEncoder",
        "wavelm": "WaveLMEncoder",
        "emotion2vec": "Emotion2VecEncoder"
    }

    if expert_name.lower() not in class_map:
        raise ValueError(f"[ERROR] No expert class mapping for '{expert_name}'")

    cls_name = class_map[expert_name.lower()]
    return getattr(mod, cls_name)

# ------------------- Pipeline -------------------
def run_expert_pipeline(expert_name, dataloaders_raw):
    print(f"\n=== Running pipeline for expert: {expert_name} ===")
    try:
        expert_cls = get_expert_class(expert_name)
    except ValueError as e:
        print(f"[WARN] Skipping {expert_name} due to error: {e}")
        return None

    expert = expert_cls(freeze_model=True).to(DEVICE)

    # Create cache, plot, and log directories per expert
    enc_cache_dir = os.path.join(CACHE_DIR, expert_name)
    os.makedirs(enc_cache_dir, exist_ok=True)
    plot_dir = os.path.join("plots", expert_name)
    log_file = os.path.join("logs", f"{expert_name}/eval_log.txt")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Extract or load embeddings
    cached_paths = {}
    for split, loader in dataloaders_raw.items():
        emb_path, lab_path = extract_and_cache_embeddings(expert, loader, enc_cache_dir, split, DEVICE)
        cached_paths[split] = (emb_path, lab_path)

    cached_datasets = {
        split: CachedEmbeddingDataset(emb_path, lab_path)
        for split, (emb_path, lab_path) in cached_paths.items()
    }

    loaders = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        for split, ds in cached_datasets.items()
    }

    # Train classifier
    input_dim = cached_datasets["train"].embs.shape[1]
    classifier = LinearHead(input_dim=input_dim).to(DEVICE)

    print("[INFO] Training classifier...")
    train_classifier(classifier, loaders["train"], loaders["dev"], DEVICE, lr=LR, epochs=EPOCHS,
                     save_path=f"model_checkpoints/{expert_name}/best_classifier.pt")

    # Evaluate and save plots/logs per expert
    print("[INFO] Evaluating classifier...")
    metrics = evaluate_and_plot(classifier, loaders["eval"], DEVICE,
                                plot_dir=plot_dir,
                                log_file=log_file)

    # Handle None return from evaluate_and_plot
    if metrics is None:
        metrics = {}

    metrics["expert"] = expert_name
    return metrics


# ------------------- Main -------------------
if __name__ == "__main__":
    # Load datasets
    datasets = {}
    for split, info in datasets_info.items():
        ds = ASVspoof2019Dataset(root_dir=info["root_dir"], protocol_file=info["protocol_file"])
        datasets[split] = ds
        print(f"[INFO] Loaded {split}: {len(ds)} samples")

    dataloaders_raw = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        for split, ds in datasets.items()
    }

    # Run all experts
    summary = []
    metrics_csv = os.path.join(LOG_DIR, "evaluation_metrics.csv")

    # Create CSV with header if it doesn't exist
    if not os.path.exists(metrics_csv):
        pd.DataFrame(columns=["expert", "accuracy", "eer", "auc", "f1", "precision", "recall"]).to_csv(metrics_csv, index=False)

    for expert_name in EXPERT_NAMES:
        try:
            metrics = run_expert_pipeline(expert_name, dataloaders_raw)
            if metrics:
                summary.append(metrics)

                # ✅ Append metrics to the global CSV file
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(metrics_csv, mode="a", index=False, header=False)

        except Exception as e:
            print(f"[WARN] Skipping {expert_name} due to error: {e}")

    # Save summary table (redundant backup of all results)
    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(LOG_DIR, "expert_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✅ Summary table saved at: {summary_path}")
    print(f"✅ Cumulative metrics saved at: {metrics_csv}")
