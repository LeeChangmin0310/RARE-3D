# rldoom/models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class QHead(nn.Module):
    """Standard linear Q-value head: features -> Q(s, a)."""

    def __init__(self, feature_dim: int, num_actions: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_actions)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, D)
        return self.fc(feat)


class DuelingQHead(nn.Module):
    """Dueling head: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))."""

    def __init__(self, feature_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        # Value stream
        self.v_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.v_fc2 = nn.Linear(hidden_dim, 1)

        # Advantage stream
        self.a_fc1 = nn.Linear(feature_dim, hidden_dim)
        self.a_fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B, D)
        v = F.relu(self.v_fc1(feat))
        v = self.v_fc2(v)  # (B, 1)

        a = F.relu(self.a_fc1(feat))
        a = self.a_fc2(a)  # (B, A)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)  # (B, A)
        return q


class PolicyHead(nn.Module):
    """Discrete policy head (categorical)."""

    def __init__(self, feature_dim: int, num_actions: int):
        super().__init__()
        self.logits = nn.Linear(feature_dim, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)


class ValueHead(nn.Module):
    """State-value head."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.v = nn.Linear(feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.v(x).squeeze(-1)
