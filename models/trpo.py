import torch
import torch.nn as nn
import torch.nn.functional as F

class TRPO(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(TRPO, self).__init__()

        # Convolutional encoder
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2)

        # For input (4, 100, 120), conv3 output is (128, 4, 5) => 128*4*5 = 2560
        conv_output_dim = 128 * 4 * 5

        # Policy stream (actor)
        self.fc1 = nn.Linear(conv_output_dim, 512)
        self.policy = nn.Linear(512, num_actions)  # Action probabilities

    def forward(self, x):
        """Forward pass returning policy."""
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten

        x = F.elu(self.fc1(x))
        policy = F.softmax(self.policy(x), dim=-1)  # Action probabilities
        return policy
