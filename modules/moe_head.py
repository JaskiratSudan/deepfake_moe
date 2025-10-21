# modules/moe_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionBinaryMoE(nn.Module):
    """
    Gate learns mixture weights over E experts' binary logits.
    Inputs:
      - either a list of tensors [B,1] (length E) or a tensor [B,E]
      - optional gate_features [B,D] if you want the gate to condition on embeddings
    Output:
      - fused_logit [B,1]    (still a logit; pass through sigmoid at eval)
      - gate_probs [B,E]
    """
    def __init__(
        self,
        num_experts: int,
        gate_input_dim: int = None,      # default: E (concat logits)
        hidden: int = 128,
        use_logits_as_gate_input: bool = True,
        temperature_init: float = 1.0,
        entropy_reg: float = 0.0         # optional regularizer to avoid collapse
    ):
        super().__init__()
        self.num_experts = num_experts
        self.use_logits_as_gate_input = use_logits_as_gate_input
        self.entropy_reg = entropy_reg

        if gate_input_dim is None:
            gate_input_dim = num_experts  # concat of E binary logits

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_experts),
        )

        # temperature for experts' logits (learned)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))
        # per-expert bias over the single class-logit
        self.bias = nn.Parameter(torch.zeros(num_experts, 1))

    def forward(self, expert_logits, gate_features=None):
        """
        expert_logits: list([B,1]) of length E OR tensor [B,E]
        gate_features: [B,D] if provided; else we use expert logits as gate input
        """
        if isinstance(expert_logits, (list, tuple)):
            logits = torch.stack(expert_logits, dim=1)  # [B,E,1]
        else:
            # [B,E] -> [B,E,1]
            logits = expert_logits.unsqueeze(-1)

        B, E, _ = logits.shape
        assert E == self.num_experts, f"Expected {self.num_experts} experts, got {E}"

        # Scale + bias experts
        T = torch.exp(self.log_temperature).clamp_min(1e-3)
        logits_scaled = logits / T + self.bias.unsqueeze(0)  # [B,E,1]

        # Gate input
        if gate_features is None and self.use_logits_as_gate_input:
            gate_in = logits.squeeze(-1)  # [B,E]
        elif gate_features is not None:
            gate_in = gate_features       # [B,D]
        else:
            raise ValueError("Provide gate_features or set use_logits_as_gate_input=True")

        gate_scores = self.gate(gate_in)         # [B,E]
        gate_probs = F.softmax(gate_scores, -1)  # [B,E]

        # Mixture of experts (weighted sum of logits)
        fused_logit = torch.sum(gate_probs.unsqueeze(-1) * logits_scaled, dim=1)  # [B,1]

        # Optional entropy regularization (return as aux)
        if self.entropy_reg > 0:
            ent = -(gate_probs * (gate_probs.clamp_min(1e-8)).log()).sum(dim=1).mean()
            aux_reg = -self.entropy_reg * ent
        else:
            aux_reg = torch.tensor(0.0, device=fused_logit.device)

        return fused_logit, gate_probs, aux_reg
