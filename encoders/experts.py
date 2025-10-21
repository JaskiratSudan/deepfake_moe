# experts.py

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, HubertModel
from funasr import AutoModel
import numpy as np

# ------------------ Style Expert ------------------
class StyleEncoder(nn.Module):
    def __init__(self, freeze_model=True):
        super().__init__()
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_model:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False

    def forward(self, waveform):
        outputs = self.wav2vec2(waveform, output_hidden_states=True)
        style_layers = outputs.hidden_states[1:12]
        stacked = torch.stack(style_layers, dim=0)
        style_features = stacked.permute(1, 0, 3, 2)  # (B, L, H, T)
        return style_features


# ------------------ Linguistic Expert ------------------
class LinguisticEncoder(nn.Module):
    def __init__(self, freeze_model=True):
        super().__init__()
        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_model:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False

    def forward(self, waveform):
        outputs = self.wav2vec2(waveform, output_hidden_states=True)
        linguistic_layers = outputs.hidden_states[15:23]
        stacked = torch.stack(linguistic_layers, dim=0)
        linguistic_features = stacked.permute(1, 0, 3, 2)
        return linguistic_features


# ------------------ WaveLM Expert ------------------
class WaveLMEncoder(nn.Module):
    def __init__(self, freeze_model=True):
        super().__init__()
        model_name = "microsoft/wavlm-base-plus"
        self.wavlm = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_model:
            for p in self.wavlm.parameters():
                p.requires_grad = False

    def forward(self, waveform):
        outputs = self.wavlm(waveform, output_hidden_states=True)
        layers = outputs.hidden_states  # mid–high level speech features
        stacked = torch.stack(layers, dim=0)
        features = stacked.permute(1, 0, 3, 2)
        return features


# ------------------ HuBERT Expert ------------------
class HuBERTEncoder(nn.Module):
    def __init__(self, freeze_model=True):
        super().__init__()
        model_name = "facebook/hubert-large-ls960-ft"
        self.hubert = HubertModel.from_pretrained(model_name)

        if freeze_model:
            for p in self.hubert.parameters():
                p.requires_grad = False

    def forward(self, waveform):
        outputs = self.hubert(waveform, output_hidden_states=True)
        layers = outputs.hidden_states
        stacked = torch.stack(layers, dim=0)
        features = stacked.permute(1, 0, 3, 2)
        return features


# ------------------ Emotion2Vec Expert ------------------
class Emotion2VecEncoder(nn.Module):
    """
    FunASR Emotion2Vec wrapper using generate() for safe extraction.
    Returns (B, T, H) embeddings without pooling.
    """
    def __init__(self, model_name="emotion2vec_base", freeze_model=True):
        super().__init__()
        print("[LOAD] Loading Emotion2Vec model...")
        self.model = AutoModel(model=f"emotion2vec/{model_name}", hub="hf")
        self.model.model.eval()  # freeze

    def forward(self, waveforms):
        """
        waveforms: Tensor (B, T)
        Returns: Tensor (B, T', H)
        """
        all_embs = []
        device = waveforms.device

        for wav in waveforms:
            wav_np = wav.cpu().numpy()
            rec_result = self.model.generate(wav_np, granularity="utterance")
            emb = np.array(rec_result[0]["feats"], dtype=np.float32)  # (T', H) or (H,)
            emb_tensor = torch.tensor(emb, dtype=torch.float32)

            # ensure 2D
            if emb_tensor.ndim == 1:
                emb_tensor = emb_tensor.unsqueeze(0)  # (1, H)

            all_embs.append(emb_tensor.to(device))

        # pad to max length
        max_len = max(e.shape[0] for e in all_embs)
        padded = torch.stack([
            torch.nn.functional.pad(e, (0,0,0,max_len - e.shape[0])) for e in all_embs
        ])
        return padded  # (B, T, H)
