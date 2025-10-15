# data_loader.py
# This version is updated to include a wrapper for applying augmentations.

import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
import soundfile
import librosa

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend.utils")

def pad_collate_fn(batch):
    """
    Pads audio waveforms in a batch to the length of the longest waveform.
    Assumes each item in the batch is a (waveform, label) tuple.
    """
    waveforms, labels = zip(*batch)
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(list(waveforms), batch_first=True, padding_value=0.0)
    labels = torch.stack(list(labels))
    return padded_waveforms, labels

def pad_collate_fn_aug(batch):
    """
    Collate function for the DatasetWithAugmentation. It handles batches of
    (original_waveform, augmented_waveform, label) tuples.
    """
    originals, augmenteds, labels = zip(*batch)
    padded_originals = torch.nn.utils.rnn.pad_sequence(list(originals), batch_first=True, padding_value=0.0)
    padded_augmenteds = torch.nn.utils.rnn.pad_sequence(list(augmenteds), batch_first=True, padding_value=0.0)
    labels = torch.stack(list(labels))
    return padded_originals, padded_augmenteds, labels

    # Add this new function to data_loader.py

def pad_collate_fn_speaker(batch):
    """
    Pads audio waveforms and handles speaker IDs.
    Assumes each item in the batch is a (waveform, label, speaker_id) tuple.
    """
    # Unpack three items now
    waveforms, labels, speakers = zip(*batch)
    
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(list(waveforms), batch_first=True, padding_value=0.0)
    labels = torch.stack(list(labels))
    
    # Speakers are usually strings, so we return them as a tuple
    return padded_waveforms, labels, speakers

class BaseAudioDataset(Dataset):
    """
    A base class for audio datasets to handle common processing tasks.
    Tracks how many files were successfully decoded vs. failed.
    """
    loaded_count = 0
    failed_count = 0

    def __init__(self, target_sample_rate: int = 16000, max_duration_seconds: int = 5, **kwargs):
        self.target_sample_rate = target_sample_rate
        self.max_duration_seconds = max_duration_seconds

    def _process_audio(self, audio_path: Path) -> torch.Tensor:
        try:
            waveform, sample_rate = librosa.load(
                audio_path, sr=self.target_sample_rate, mono=True
            )
            waveform = torch.from_numpy(waveform).float()
            BaseAudioDataset.loaded_count += 1
        except Exception as e:
            tqdm.write(f"[WARNING] Corrupted file: {audio_path}. Error: {e}")
            BaseAudioDataset.failed_count += 1
            if self.max_duration_seconds is not None:
                return torch.zeros(self.max_duration_seconds * self.target_sample_rate)
            else:
                return torch.zeros(self.target_sample_rate)

        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        if self.max_duration_seconds is not None:
            target_len = self.max_duration_seconds * self.target_sample_rate
            current_len = waveform.shape[0]
            if current_len > target_len:
                waveform = waveform[:target_len]
            elif current_len < target_len:
                waveform = F.pad(waveform, (0, target_len - current_len))

        return waveform

    @classmethod
    def print_summary(cls):
        total = cls.loaded_count + cls.failed_count
        print(f"\n[DATASET SUMMARY] Loaded: {cls.loaded_count}, Failed: {cls.failed_count}, Total: {total}")


class ASVspoof2019Dataset(BaseAudioDataset):
    """PyTorch Dataset for ASVspoof2019 with speaker IDs."""
    def __init__(self, protocol_file: str, root_dir: str = "", num_samples: int = None, **kwargs):
        super().__init__(**kwargs)
        self.root_dir = Path(root_dir)
        self.data = []

        with open(protocol_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                audio_path = parts[0]
                label_str = parts[2]      # bonafide/spoof
                speaker_id = parts[4]     # p240, etc.

                audio_path = audio_path.split('/').pop()

                label = 1 if label_str == "bonafide" else 0
                full_path = self.root_dir / audio_path
                self.data.append((full_path, label, speaker_id))

        if num_samples is not None:
            self.data = self.data[:num_samples]

        if not self.data:
            raise RuntimeError(f"No audio files found from protocol {protocol_file}.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label, speaker_id = self.data[idx]
        waveform = self._process_audio(audio_path)
        return waveform, torch.tensor(label, dtype=torch.long), speaker_id

class RAVDESSDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the RAVDESS dataset."""
    def __init__(self, root_dir: str, **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)
        
        self.root_dir = Path(root_dir)
        self.audio_files = sorted(list(self.root_dir.glob('**/Actor_*/*.wav')))
        
        if num_samples is not None:
            self.audio_files = self.audio_files[:num_samples]
            
        if not self.audio_files:
            raise RuntimeError(f"No .wav files found in {root_dir}.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1, dtype=torch.long)
        return waveform, label

class CommonVoiceDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the Common Voice dataset."""
    def __init__(self, root_dir: str, **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        self.audio_files = sorted(list(self.root_dir.glob('**/*.wav')))
        
        if num_samples is not None:
            self.audio_files = self.audio_files[:num_samples]
            
        if not self.audio_files:
            raise RuntimeError(f"No .wav files found in {root_dir}.")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1, dtype=torch.long)
        return waveform, label

# class ASVspoof2021Dataset(BaseAudioDataset):
#     """A PyTorch Dataset for loading audio from the ASVspoof 2021 DF eval dataset."""
#     def __init__(self, root_dir: str, protocol_file: str, subset='all', **kwargs):
#         num_samples = kwargs.pop("num_samples", None)
#         super().__init__(**kwargs)
        
#         self.root_dir = Path(root_dir)
#         self.audio_folder = self.root_dir / "flac"
        
#         if not Path(protocol_file).exists():
#             raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
            
#         col_names = ['speaker_id', 'filename', 'compression', 'source', 'attack_id', 'label', 'trim', 'set', 'vocoder_type', 'col10', 'col11', 'col12', 'col13']
#         protocol_df = pd.read_csv(protocol_file, sep='\s+', header=None, engine='python', names=col_names)
        
#         protocol_df.dropna(subset=['filename'], inplace=True)

#         original_count = len(protocol_df)
#         protocol_df['exists'] = protocol_df['filename'].apply(lambda fname: (self.audio_folder / f"{fname}.flac").exists())
#         protocol_df = protocol_df[protocol_df['exists']]
#         if len(protocol_df) < original_count:
#             print(f"[INFO] ASVspoof2021: Filtered out {original_count - len(protocol_df)} missing audio files.")

#         if subset == 'bonafide':
#             self.protocol = protocol_df[protocol_df['label'] == 'bonafide'].reset_index(drop=True)
#         elif subset == 'spoof':
#             self.protocol = protocol_df[protocol_df['label'] != 'bonafide'].reset_index(drop=True)
#         else:
#             self.protocol = protocol_df
        
#         if num_samples is not None:
#             self.protocol = self.protocol.sample(frac=1, random_state=42).reset_index(drop=True).head(num_samples)

#         if len(self.protocol) == 0:
#             raise RuntimeError(f"Found 0 audio files after filtering for subset '{subset}'.")

#     def __len__(self):
#         return len(self.protocol)

#     def __getitem__(self, idx):
#         row = self.protocol.iloc[idx]
#         audio_path = self.audio_folder / f"{row['filename']}.flac"
#         waveform = self._process_audio(audio_path)
#         label = torch.tensor(1 if row['label'] == 'bonafide' else 0, dtype=torch.long)
#         return waveform, label

class ASVspoof2021Dataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from the ASVspoof 2021 DF eval dataset using ok_files.txt."""
    def __init__(self, root_dir: str, ok_files: str, protocol_file: str, subset="all", **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)
        
        self.root_dir = Path(root_dir)
        self.audio_folder = self.root_dir / "flac"

        # Load ok_files list
        with open(ok_files, "r") as f:
            ok_list = [line.strip() for line in f if line.strip()]
        ok_set = set([Path(x).stem for x in ok_list])  # keep just the stems like DF_E_2000011

        # Load protocol file for labels
        col_names = [
            "speaker_id", "filename", "compression", "source", "attack_id",
            "label", "trim", "set", "vocoder_type", "col10", "col11", "col12", "col13"
        ]
        protocol_df = pd.read_csv(protocol_file, sep="\s+", header=None, engine="python", names=col_names)

        # Filter by ok_files
        protocol_df = protocol_df[protocol_df["filename"].isin(ok_set)]

        if subset == "bonafide":
            self.protocol = protocol_df[protocol_df["label"] == "bonafide"].reset_index(drop=True)
        elif subset == "spoof":
            self.protocol = protocol_df[protocol_df["label"] != "bonafide"].reset_index(drop=True)
        else:
            self.protocol = protocol_df.reset_index(drop=True)

        if num_samples is not None:
            self.protocol = self.protocol.sample(frac=1, random_state=42).reset_index(drop=True).head(num_samples)

        if len(self.protocol) == 0:
            raise RuntimeError(f"Found 0 audio files after filtering with ok_files and subset='{subset}'.")

        print(f"[INFO] Loaded {len(self.protocol)} samples (subset={subset}).")

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]
        audio_path = self.audio_folder / f"{row['filename']}.flac"
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1 if row["label"] == "bonafide" else 0, dtype=torch.long)
        return waveform, label

class InTheWildDataset(BaseAudioDataset):
    """A PyTorch Dataset for loading audio from an In-the-Wild dataset."""
    def __init__(self, root_dir: str, protocol_file: str, subset='all', **kwargs):
        num_samples = kwargs.pop("num_samples", None)
        super().__init__(**kwargs)

        self.root_dir = Path(root_dir)
        # Assumes audio files are in a 'wavs' subdirectory. Change if needed.
        self.audio_folder = self.root_dir 
        
        if not Path(protocol_file).exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

        protocol_df = pd.read_csv(protocol_file)
        
        # Standardize labels: 'bona-fide' -> 'bonafide'
        protocol_df['label'] = protocol_df['label'].replace('bona-fide', 'bonafide')

        original_count = len(protocol_df)
        protocol_df['exists'] = protocol_df['file'].apply(lambda fname: (self.audio_folder / fname).exists())
        protocol_df = protocol_df[protocol_df['exists']]
        if len(protocol_df) < original_count:
            print(f"[INFO] InTheWild: Filtered out {original_count - len(protocol_df)} missing audio files.")

        if subset == 'bonafide':
            self.protocol = protocol_df[protocol_df['label'] == 'bonafide'].reset_index(drop=True)
        elif subset == 'spoof':
            self.protocol = protocol_df[protocol_df['label'] == 'spoof'].reset_index(drop=True)
        else:
            self.protocol = protocol_df
        
        if num_samples is not None:
            self.protocol = self.protocol.sample(frac=1, random_state=42).reset_index(drop=True).head(num_samples)

        if len(self.protocol) == 0:
            raise RuntimeError(f"Found 0 audio files after filtering for subset '{subset}'.")

    def __len__(self):
        return len(self.protocol)

    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]
        audio_path = self.audio_folder / row['file']
        waveform = self._process_audio(audio_path)
        label = torch.tensor(1 if row['label'] == 'bonafide' else 0, dtype=torch.long)
        return waveform, label


# Example of how to use the new InTheWildDataset
if __name__ == '__main__':

    # Path to directory with .flac files
    audio_dir = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021/ASVspoof2021_DF_eval/flac"

    # Path to metadata list (txt file with file names or paths)
    metadata_list = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021/DF_keys/DF/CM/trial_metadata.txt"

    missing_files = []
    corrupted_files = []
    ok_files = []

    with open(metadata_list, "r") as f:
        file_list = [line.strip() for line in f if line.strip()]

    for fname in tqdm(file_list, desc="Checking audio files"):
        fpath = os.path.join(audio_dir, fname)

        # Check if the file exists
        if not os.path.exists(fpath):
            missing_files.append(fpath)
            continue

        # Try loading with torchaudio
        try:
            waveform, sr = torchaudio.load(fpath)
            ok_files.append(fpath)
        except Exception as e:
            corrupted_files.append(fpath)

    print(f"\nTotal in metadata: {len(file_list)}")
    print(f"OK files: {len(ok_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")

    # Save lists for inspection
    with open("missing_files.txt", "w") as f:
        f.write("\n".join(missing_files))

    with open("corrupted_files.txt", "w") as f:
        f.write("\n".join(corrupted_files))