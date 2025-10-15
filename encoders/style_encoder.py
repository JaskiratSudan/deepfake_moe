# style_encoder.py
# Defines the StyleEncoder module for the SLIM model.
# Uses a pre-trained Wav2vec 2.0 XLSR SER model to extract style features.

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class StyleEncoder(nn.Module):
    """
    PyTorch module for the style encoder.
    """
    def __init__(self, freeze_model=True):
        """
        Initializes the StyleEncoder.
        """
        super(StyleEncoder, self).__init__()

        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"

        try:
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        # Freeze model parameters to use it as a fixed feature extractor.
        if freeze_model:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.
        """
        # Pass the waveform through the model, requesting all hidden states.
        outputs = self.wav2vec2(waveform, output_hidden_states=True)

        # This corresponds to layers 0 through 10's output.
        style_layers = outputs.hidden_states[1:12]

        # Stack the layer outputs into a new dimension and average them.
        # stacked_layers = torch.stack(style_layers, dim=0)
        # style_features = torch.mean(stacked_layers, dim=0)

        stacked_layers = torch.stack(style_layers, dim=0)
        style_features = stacked_layers.permute(1, 0, 3, 2)

        return style_features

if __name__ == '__main__':
    # Create a dummy audio input tensor.
    batch_size = 2
    sample_rate = 16000
    duration_seconds = 5
    dummy_audio = torch.randn(batch_size, sample_rate * duration_seconds)
    print(f"\nInput audio shape: {dummy_audio.shape}")

    print("\nInitializing StyleEncoder...")
    try:
        style_encoder = StyleEncoder(freeze_model=True)
        style_encoder.eval()
        print("StyleEncoder initialized successfully.")

        print("\nPerforming forward pass...")
        with torch.no_grad():
            features = style_encoder(dummy_audio)
        
        print("Forward pass successful.")
        print(f"\nOutput feature shape: {features.shape}\n")


    except Exception as e:
        print(f"\nAn error occurred during the example usage: {e}")