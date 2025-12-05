# rldoom/agents/reinforce.py

# Monte Carlo Policy Gradient (REINFORCE) Agent for Doom Environment
# rldoom/agents/reinforce.py
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rldoom.agents.base import Agent
from rldoom.models.cnn_backbone import DoomCNN
from rldoom.models.heads import PolicyHead


class ReinforceAgent(Agent):
    """Monte Carlo policy gradient (REINFORCE) with CNN policy network."""

    def __init__(self, obs_shape, num_actions, cfg, device):
        super().__init__(obs_shape, num_actions, cfg, device)

        c, _, _ = obs_shape
        self.backbone = DoomCNN(in_channels=c, feature_dim=cfg.feature_dim).to(device)
        self.policy_head = PolicyHead(cfg.feature_dim, num_actions).to(device)

        self.optimizer = optim.Adam(
            list(self.backbone.parameters()) + list(self.policy_head.parameters()),
            lr=cfg.lr,
        )

        self.gamma = cfg.gamma
        self.ent_coef = getattr(cfg, "ent_coef", 0.0)

        self.log_probs = []
        self.rewards = []
        self._done = False

    def _distribution(self, obs_t: torch.Tensor):
        feat = self.backbone(obs_t)
        logits = self.policy_head(feat)
        return torch.distributions.Categorical(logits=logits)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        dist = self._distribution(obs_t)

        if deterministic:
            action = int(dist.probs.argmax(dim=1).item())
            self.last_log_prob = dist.log_prob(torch.tensor([action], device=self.device))[0]
        else:
            action_t = dist.sample()
            self.last_log_prob = dist.log_prob(action_t)[0]
            action = int(action_t.item())

        return action

    def observe(self, transition):
        """
        Receive one transition:
        (obs, action, reward, next_obs, done).
        We only need reward (and optionally done) for REINFORCE.
        """
        obs, action, reward, next_obs, done = transition
        # store reward for this time step
        self.rewards.append(float(reward))

    def update(self) -> Dict[str, float]:
        if not self._done:
            return {}

        # Compute returns
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.append(G)
        returns.reverse()
        returns = torch.as_tensor(returns, device=self.device, dtype=torch.float32)

        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.backbone.parameters(), self.cfg.grad_clip)
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        self._done = False

        return {"loss": float(loss.item())}
