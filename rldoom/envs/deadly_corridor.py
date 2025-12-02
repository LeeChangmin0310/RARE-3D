# rldoom/envs/deadly_corridor.py
import numpy as np
from collections import deque
from typing import Tuple, List

from vizdoom import DoomGame
import cv2


class DeadlyCorridorEnv:
    """VizDoom Deadly Corridor wrapper with preprocessing & frame stacking."""

    def __init__(
        self,
        cfg_path: str = "doom_files/deadly_corridor.cfg",
        wad_path: str = "doom_files/deadly_corridor.wad",
        frame_size: int = 84,
        stack_size: int = 4,
        frame_skip: int = 4,
    ):
        self.game = DoomGame()
        self.game.load_config(cfg_path)
        self.game.set_doom_scenario_path(wad_path)
        self.game.init()

        # 7 discrete actions (same as notebooks)
        self.possible_actions: List[List[int]] = np.identity(7, dtype=np.int32).tolist()

        self.frame_size = frame_size
        self.stack_size = stack_size
        self.frame_skip = frame_skip

        self.frames = deque(
            [np.zeros((frame_size, frame_size), dtype=np.float32)
             for _ in range(stack_size)],
            maxlen=stack_size,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Convert RGB frame (H,W,3) -> gray, crop, resize, normalize."""
        # Original: (C,H,W) so transpose first if needed
        if frame.ndim == 3 and frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)  # (H,W,3)

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Crop: (0, -60, -40, 60) in the notebooks ⇒ [top:bottom, left:right]
        h, w = gray.shape
        top, bottom = 0, h - 60
        left, right = 40, w - 60
        cropped = gray[top:bottom, left:right]

        resized = cv2.resize(cropped, (self.frame_size, self.frame_size),
                             interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        return norm

    def _stack(self, new_frame: np.ndarray, new_episode: bool) -> np.ndarray:
        """Update and return stacked frames (stack_size, H, W)."""
        if new_episode:
            self.frames = deque(
                [np.zeros_like(new_frame, dtype=np.float32)
                 for _ in range(self.stack_size)],
                maxlen=self.stack_size,
            )
            for _ in range(self.stack_size):
                self.frames.append(new_frame)
        else:
            self.frames.append(new_frame)

        stacked = np.stack(self.frames, axis=0)
        return stacked

    def reset(self) -> np.ndarray:
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        frame = self._preprocess(state)
        stacked = self._stack(frame, new_episode=True)
        return stacked

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Frame-skip step."""
        action = self.possible_actions[action_idx]
        total_reward = 0.0
        done = False

        for _ in range(self.frame_skip):
            reward = self.game.make_action(action)
            total_reward += reward
            done = self.game.is_episode_finished()
            if done:
                break

        if done:
            next_state = np.zeros(
                (self.frame_size, self.frame_size), dtype=np.float32
            )
            stacked = self._stack(next_state, new_episode=False)
        else:
            state = self.game.get_state().screen_buffer
            frame = self._preprocess(state)
            stacked = self._stack(frame, new_episode=False)

        info = {}
        return stacked, float(total_reward), bool(done), info

    def close(self):
        self.game.close()
