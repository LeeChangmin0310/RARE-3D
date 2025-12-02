# rldoom/buffers/replay_buffer.py
from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
    """Simple replay buffer for off-policy methods (DQN / DDQN / Rainbow)."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], device: torch.device):
        self.capacity = int(capacity)
        self.device = device

        self.obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.idx = 0
        self.full = False

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition in the buffer."""
        self.obses[self.idx] = obs
        self.next_obses[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int):
        """Sample a random minibatch of transitions."""
        size = self.size
        assert size >= batch_size, "Not enough samples in replay buffer"

        idxs = np.random.randint(0, size, size=batch_size)

        obs = torch.as_tensor(self.obses[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obs = torch.as_tensor(self.next_obses[idxs], device=self.device)
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return obs, actions, rewards, next_obs, dones
'''
# rldoom/buffers/replay_buffer.py
import numpy as np
import torch


class ReplayBuffer:
    """Simple FIFO replay buffer for off-policy value-based methods (DQN, DDQN)."""

    def __init__(self, capacity, obs_shape, device):
        """
        Args:
            capacity (int): maximum number of transitions stored.
            obs_shape (tuple): observation shape (C, H, W).
            device (torch.device): target device for sampled tensors.
        """
        self.capacity = int(capacity)
        self.device = device

        self.obs_buf = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add a transition to the buffer."""
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample a batch of transitions as torch tensors."""
        batch_size = min(batch_size, self.size)
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = torch.from_numpy(self.obs_buf[idxs]).to(self.device)          # (B,C,H,W)
        next_obs = torch.from_numpy(self.next_obs_buf[idxs]).to(self.device)
        actions = torch.from_numpy(self.action_buf[idxs]).to(self.device)   # (B,)
        rewards = torch.from_numpy(self.reward_buf[idxs]).to(self.device)   # (B,)
        dones = torch.from_numpy(self.done_buf[idxs]).to(self.device)       # (B,)

        return obs, actions, rewards, next_obs, dones
    
'''