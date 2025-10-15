# linguistic_encoder.py
# Defines the LinguisticEncoder module for the SLIM model.
# Uses a pre-trained Wav2vec 2.0 XLSR ASR model to extract content features.

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class LinguisticEncoder(nn.Module):
    """
    PyTorch module for the linguistic encoder.
    """
    def __init__(self, freeze_model=True):
        """
        Initializes the LinguisticEncoder.
        """
        super(LinguisticEncoder, self).__init__()

        model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-english"

        # Load the base Wav2Vec2Model to get hidden-state embeddings.
        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        if freeze_model:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.
        """
        # Pass the waveform through the model, requesting all hidden states.
        outputs = self.wav2vec2(waveform, output_hidden_states=True)

        # corresponds to indices 15 through 22.
        linguistic_layers = outputs.hidden_states[15:23]

        # Stack the layer outputs into a new dimension and average them.
        # stacked_layers = torch.stack(linguistic_layers, dim=0)
        # linguistic_features = torch.mean(stacked_layers, dim=0)

        stacked_layers = torch.stack(linguistic_layers, dim=0)
        linguistic_features = stacked_layers.permute(1, 0, 3, 2)

        return linguistic_features

# Example usage script.
if __name__ == '__main__':
    # Create a dummy audio input tensor.
    batch_size = 2
    sample_rate = 16000
    duration_seconds = 5
    dummy_audio = torch.randn(batch_size, sample_rate * duration_seconds)
    print(f"\nInput audio shape: {dummy_audio.shape}")

    print("\nInitializing LinguisticEncoder...")
    try:
        linguistic_encoder = LinguisticEncoder(freeze_model=True)
        linguistic_encoder.eval()
        print("LinguisticEncoder initialized successfully.")

        print("\nPerforming forward pass...")
        with torch.no_grad():
            features = linguistic_encoder(dummy_audio)
        
        print("Forward pass successful.")
        print(f"\nOutput feature shape: {features.shape}\n")

    except Exception as e:
        print(f"\nAn error occurred during the example usage: {e}")