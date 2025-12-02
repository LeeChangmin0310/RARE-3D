# rldoom/buffers/rollout_buffer.py
from typing import Tuple
import numpy as np
import torch


class RolloutBuffer:
    """On-policy rollout buffer for REINFORCE / A2C / A3C / PPO / TRPO."""

    def __init__(
        self,
        rollout_len: int,
        obs_shape: Tuple[int, ...],
        num_actions: int,
        device: torch.device,
    ):
        self.rollout_len = int(rollout_len)
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device

        self.reset()

    def reset(self) -> None:
        """Clear stored trajectory."""
        self.obses = []
        self.actions = []
        self.rewards = []
        self.next_obses = []
        self.dones = []
        self._ready = False

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store one step of the trajectory."""
        self.obses.append(obs.copy())
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_obses.append(next_obs.copy())
        self.dones.append(bool(done))

        if len(self.obses) >= self.rollout_len or done:
            self._ready = True

    def is_ready(self) -> bool:
        """Return True if buffer has enough data for one update."""
        return self._ready

    def get(self):
        """Return tensors (T, ...) and do not reset automatically."""
        assert self._ready, "RolloutBuffer not ready yet"

        obs = torch.as_tensor(np.stack(self.obses, axis=0), device=self.device)
        actions = torch.as_tensor(self.actions, device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(self.rewards, device=self.device, dtype=torch.float32)
        next_obs = torch.as_tensor(np.stack(self.next_obses, axis=0), device=self.device)
        dones = torch.as_tensor(self.dones, device=self.device, dtype=torch.float32)

        return obs, actions, rewards, next_obs, dones

'''
# rldoom/buffers/rollout_buffer.py
from typing import List, Tuple
import numpy as np
import torch


class RolloutBuffer:
    """Simple on-policy rollout buffer for A2C-style methods."""

    def __init__(self, rollout_len, obs_shape, num_actions, device):
        """
        Args:
            rollout_len (int): number of steps per rollout.
            obs_shape (tuple): observation shape (C, H, W).
            num_actions (int): number of discrete actions.
        """
        self.rollout_len = int(rollout_len)
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device

        self.reset()

    def reset(self):
        """Clear internal storage."""
        self.obs_list: List[np.ndarray] = []
        self.next_obs_list: List[np.ndarray] = []
        self.actions_list: List[int] = []
        self.rewards_list: List[float] = []
        self.dones_list: List[float] = []

    def add(self, obs, action, reward, next_obs, done):
        """Store one transition."""
        self.obs_list.append(obs)
        self.next_obs_list.append(next_obs)
        self.actions_list.append(int(action))
        self.rewards_list.append(float(reward))
        self.dones_list.append(float(done))

        # Rollout length is a soft limit; buffer may be flushed by agent earlier.
        if len(self.obs_list) > self.rollout_len:
            # Keep only the most recent rollout_len steps
            self.obs_list = self.obs_list[-self.rollout_len:]
            self.next_obs_list = self.next_obs_list[-self.rollout_len:]
            self.actions_list = self.actions_list[-self.rollout_len:]
            self.rewards_list = self.rewards_list[-self.rollout_len:]
            self.dones_list = self.dones_list[-self.rollout_len:]

    def is_ready(self) -> bool:
        """Return True if at least rollout_len transitions are stored."""
        return len(self.obs_list) >= self.rollout_len

    def get(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stacked rollout as torch tensors (T, ...)."""
        obs = torch.from_numpy(
            np.stack(self.obs_list, axis=0).astype(np.float32)
        ).to(self.device)  # (T,C,H,W)
        next_obs = torch.from_numpy(
            np.stack(self.next_obs_list, axis=0).astype(np.float32)
        ).to(self.device)
        actions = torch.from_numpy(
            np.array(self.actions_list, dtype=np.int64)
        ).to(self.device)  # (T,)
        rewards = torch.from_numpy(
            np.array(self.rewards_list, dtype=np.float32)
        ).to(self.device)
        dones = torch.from_numpy(
            np.array(self.dones_list, dtype=np.float32)
        ).to(self.device)

        return obs, actions, rewards, next_obs, dones
'''