# rldoom/buffers/prioritized_replay.py
from typing import Tuple
import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Simple proportional PER (no sum-tree, O(N) sampling, good enough here)."""

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        device: torch.device,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
    ):
        self.capacity = int(capacity)
        self.device = device
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)

        self.obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.frame = 1

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.idx

    def _beta(self) -> float:
        """Anneal beta from beta_start to 1.0 over beta_frames."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with max priority so it is likely to be sampled soon."""
        self.obses[self.idx] = obs
        self.next_obses[self.idx] = next_obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = float(done)

        max_prio = self.priorities.max() if self.size > 0 else 1.0
        self.priorities[self.idx] = max_prio

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True

    def sample(self, batch_size: int):
        """Sample a minibatch with PER, return indices and importance weights."""
        size = self.size
        assert size >= batch_size, "Not enough samples in PER buffer"

        prios = self.priorities[:size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(size, batch_size, p=probs)

        obs = torch.as_tensor(self.obses[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        next_obs = torch.as_tensor(self.next_obses[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)

        self.frame += 1
        beta = self._beta()

        # Importance sampling weights
        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)

        return obs, actions, rewards, next_obs, dones, indices, weights

    def update_priorities(self, indices, new_priorities) -> None:
        """Update priorities for sampled transitions."""
        new_priorities = np.abs(new_priorities) + 1e-6
        self.priorities[indices] = new_priorities
'''
# rldoom/buffers/prioritized_replay.py
import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Proportional prioritized replay buffer (simplified, array-based)."""

    def __init__(self, capacity, obs_shape, device, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Args:
            alpha (float): how much prioritization is used (0 = uniform).
            beta_start (float): initial value of importance-sampling exponent.
            beta_frames (int): number of frames over which beta is annealed to 1.0.
        """
        self.capacity = int(capacity)
        self.device = device
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, int(beta_frames))

        self.obs_buf = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.action_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.reward_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)

        # priorities
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.frame = 1  # for beta schedule

    @property
    def beta(self):
        """Linearly annealed beta schedule."""
        t = min(self.frame, self.beta_frames)
        frac = t / self.beta_frames
        return self.beta_start + frac * (1.0 - self.beta_start)

    def add(self, obs, action, reward, next_obs, done):
        """Add transition with max priority so far (or 1 if buffer is empty)."""
        max_prio = self.priorities.max() if self.size > 0 else 1.0

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.done_buf[self.ptr] = float(done)
        self.priorities[self.ptr] = max_prio

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Sample transitions according to priorities."""
        batch_size = min(batch_size, self.size)
        if self.size == 0:
            raise ValueError("Cannot sample from an empty PrioritizedReplayBuffer.")

        prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        idxs = np.random.choice(self.size, batch_size, p=probs)

        self.frame += 1
        beta = self.beta

        # importance-sampling weights
        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = torch.from_numpy(weights.astype(np.float32)).to(self.device)

        obs = torch.from_numpy(self.obs_buf[idxs]).to(self.device)
        next_obs = torch.from_numpy(self.next_obs_buf[idxs]).to(self.device)
        actions = torch.from_numpy(self.action_buf[idxs]).to(self.device)
        rewards = torch.from_numpy(self.reward_buf[idxs]).to(self.device)
        dones = torch.from_numpy(self.done_buf[idxs]).to(self.device)

        return idxs, obs, actions, rewards, next_obs, dones, weights

    def update_priorities(self, idxs, new_priorities):
        """Update priorities after learning step."""
        new_priorities = np.asarray(new_priorities, dtype=np.float32)
        assert new_priorities.shape[0] == len(idxs)
        for idx, prio in zip(idxs, new_priorities):
            self.priorities[idx] = max(prio, 1e-6)  # avoid zeros
'''