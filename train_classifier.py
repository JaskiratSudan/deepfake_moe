import argparse
import importlib
import os
import torch
from torch.utils.data import DataLoader

from modules.classifier_head import LinearHead
from modules.dataset_utils import extract_and_cache_embeddings, CachedEmbeddingDataset
from modules.train_utils import train_classifier
from modules.eval_utils import evaluate_and_plot
from modules.data_loader import ASVspoof2019Dataset  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "cache/"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 50
num_samples = None

# Example dataset definition (num_samples optional)
datasets_info = {
    "train": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt",
        "num_samples": num_samples
    },
    "dev": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt",
        "num_samples": num_samples
    },
    "eval": {
        "root_dir": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac",
        "protocol_file": "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt",
        "num_samples": num_samples
    }
}


# ============================================================
# 🔹 Dynamic Encoder Importer
# ============================================================
def get_encoder_class(encoder_arg: str):
    """Dynamically import an encoder class from the encoders package."""
    base = encoder_arg
    if base.lower().endswith("_encoder"):
        base = base[: -len("_encoder")]
    elif base.lower().endswith("encoder"):
        base = base[: -len("encoder")]
    base = base.strip().lower()

    module_name = f"encoders.{base}_encoder"
    parts = base.split("_")
    class_name = "".join(p.capitalize() for p in parts) + "Encoder"

    try:
        mod = importlib.import_module(module_name)
        encoder_cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not import encoder '{encoder_arg}'. Expected '{module_name}.{class_name}'.\nDetails: {e}"
        )
    return encoder_cls


# ============================================================
# 🔹 Main Pipeline Function
# ============================================================
def run_pipeline_with_encoder(encoder_cls, enc_name: str, dataloaders_raw):
    encoder = encoder_cls(freeze_model=True).to(DEVICE)

    # Per-encoder cache directory
    enc_cache_dir = os.path.join(CACHE_DIR, enc_name)
    os.makedirs(enc_cache_dir, exist_ok=True)

    print("\n[INFO] Extracting or loading cached embeddings...")
    cached_paths = {}
    for split, loader in dataloaders_raw.items():
        emb_path, lab_path = extract_and_cache_embeddings(
            encoder, loader, enc_cache_dir, split, DEVICE
        )
        cached_paths[split] = (emb_path, lab_path)

    cached_datasets = {
        split: CachedEmbeddingDataset(emb_path, lab_path)
        for split, (emb_path, lab_path) in cached_paths.items()
    }

    loaders = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"))
        for split, ds in cached_datasets.items()
    }

    input_dim = cached_datasets["train"].embs.shape[1]
    classifier = LinearHead(input_dim=input_dim).to(DEVICE)

    print("\n[INFO] Training classifier...")
    train_classifier(classifier, loaders["train"], loaders["dev"], DEVICE, lr=LR, epochs=EPOCHS, save_path=f"model_checkpoints/{enc_name}/best_classifier.pt")

    print("\n[INFO] Evaluating classifier...")
    evaluate_and_plot(classifier, loaders["eval"], DEVICE, plot_dir=f"plots/{enc_name}", log_file=f"logs/{enc_name}/eval_log.txt")

# ============================================================
# 🔹 Entry Point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", "-e", default="style", help="Encoder to use (e.g. 'style' or 'linguistic')")
    args = parser.parse_args()

    enc_arg = args.encoder

    print("\n[INFO] Loading ASVspoof2019 datasets...")

    datasets = {}
    for split, info in datasets_info.items():
        # Extract only required args
        root_dir = info["root_dir"]
        protocol_file = info["protocol_file"]

        # Optional arguments (e.g. num_samples)
        optional_args = {k: v for k, v in info.items() if k not in ["root_dir", "protocol_file"]}

        ds = ASVspoof2019Dataset(root_dir=root_dir, protocol_file=protocol_file, **optional_args)
        datasets[split] = ds

        sample_count = len(ds) if hasattr(ds, "__len__") else "?"
        print(f"  ➤ Loaded '{split}' split with {sample_count} samples (args: {optional_args})")

    dataloaders_raw = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"))
        for split, ds in datasets.items()
    }

    try:
        encoder_cls = get_encoder_class(enc_arg)
    except ImportError as e:
        print(f"Error: {e}")
        print("Available encoders are modules in 'encoders/' named '<name>_encoder.py' with a '<Name>Encoder' class.")
        raise

    print(f"\n=== Running pipeline with encoder: {enc_arg} ===")
    run_pipeline_with_encoder(encoder_cls, enc_arg, dataloaders_raw)

    print("\n✅ Training + Evaluation completed successfully!")
