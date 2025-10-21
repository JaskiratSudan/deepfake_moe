# classifier_head.py
import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
