# rldoom/models/cnn_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoomCNN(nn.Module):
    """Shared CNN backbone for all agents."""

    def __init__(self, in_channels: int = 4, feature_dim: int = 512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, feature_dim)  # 84x84 → 7x7 after convs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
