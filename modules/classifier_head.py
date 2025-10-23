# modules/classifier_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LinearHead",
    "ResidualMLPHead",
    "CosineMarginHead",
    "get_head",
]

class LinearHead(nn.Module):
    """
    Your original MLP head (kept for compatibility).
    """
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


class ResidualMLPHead(nn.Module):
    """
    Stronger MLP: LayerNorm + GELU + Dropout + residual projection.
    Default dims: 1024 -> 512 -> 128 -> 1
    """
    def __init__(self, input_dim=1024, hidden1=512, hidden2=128, p_drop=0.2):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)
        self.drop = nn.Dropout(p_drop)
        self.act  = nn.GELU()
        # projection for residual if needed
        self.proj1 = nn.Linear(input_dim, hidden1) if hidden1 != input_dim else nn.Identity()

        # init
        nn.init.xavier_uniform_(self.fc1.weight); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.out.weight); nn.init.zeros_(self.out.bias)
        if isinstance(self.proj1, nn.Linear):
            nn.init.xavier_uniform_(self.proj1.weight); nn.init.zeros_(self.proj1.bias)

    def forward(self, x):
        x = self.in_norm(x)
        h = self.act(self.fc1(x))
        h = self.drop(h)
        h = h + self.proj1(x)     # residual
        h = self.act(self.fc2(h))
        h = self.drop(h)
        return self.out(h)


class CosineMarginHead(nn.Module):
    """
    Cosine classifier with additive margin for binary classification.
      logit = s * (cos(x, w) - m)
    Good defaults: s in [16..32], m in [0.20..0.35]
    """
    def __init__(self, input_dim=1024, s=24.0, m=0.25, use_ln=True, p_drop=0.1):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(float(s)))
        self.m = float(m)
        self.ln = nn.LayerNorm(input_dim) if use_ln else nn.Identity()
        self.dropout = nn.Dropout(p_drop) if p_drop and p_drop > 0 else nn.Identity()
        # learnable weight vector for positive class
        self.weight = nn.Parameter(torch.randn(input_dim, 1))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        x = self.ln(x)
        x = self.dropout(x)
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=0)
        cosine = x @ w                      # (B,1)
        logits = self.s * (cosine - self.m) # (B,1)
        return logits


# ---------- Factory ----------
def get_head(head_type: str, input_dim: int, **kwargs) -> nn.Module:
    head_type = (head_type or "linear").lower()
    if head_type in ["linear", "mlp"]:
        return LinearHead(input_dim=input_dim)
    if head_type in ["resmlp", "residual", "residual_mlp"]:
        return ResidualMLPHead(
            input_dim=input_dim,
            hidden1=kwargs.get("hidden1", 512),
            hidden2=kwargs.get("hidden2", 128),
            p_drop=kwargs.get("p_drop", 0.2),
        )
    if head_type in ["cosine", "cosinemargin", "cosine_margin"]:
        return CosineMarginHead(
            input_dim=input_dim,
            s=kwargs.get("s", 24.0),
            m=kwargs.get("m", 0.25),
            use_ln=kwargs.get("use_ln", True),
            p_drop=kwargs.get("p_drop", 0.1),
        )
    raise ValueError(f"Unknown head_type '{head_type}'")
