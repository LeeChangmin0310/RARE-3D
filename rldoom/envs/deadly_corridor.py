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

        # Initialize frame stack with zeros
        self.frames = deque(
            [np.zeros((frame_size, frame_size), dtype=np.float32)
             for _ in range(stack_size)],
            maxlen=stack_size,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert Doom screen_buffer to (frame_size, frame_size) grayscale float32 in [0,1].

        Handles different shapes:
          - (H, W)               : already grayscale
          - (C, H, W), C in {1,3}: channel-first, possibly RGB
          - (H, W, C), C in {1,3}: channel-last, possibly RGB
        """
        if frame is None:
            raise RuntimeError("VizDoom returned None screen_buffer")

        # If channel-first, move channels to last dimension.
        # Typical shapes from VizDoom: (C, H, W) with C=1 or 3.
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = frame.transpose(1, 2, 0)  # (H, W, C)

        # Now handle according to number of dimensions/channels
        if frame.ndim == 2:
            # (H, W): already grayscale
            gray = frame
        elif frame.ndim == 3:
            h, w, c = frame.shape
            if c == 3:
                # (H, W, 3): RGB -> GRAY
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif c == 1:
                # (H, W, 1): squeeze channel
                gray = frame[:, :, 0]
            else:
                raise ValueError(f"Unexpected number of channels: {c} in frame.shape={frame.shape}")
        else:
            raise ValueError(f"Unexpected frame ndim: {frame.ndim}, shape={frame.shape}")

        # Crop (same as your original logic)
        h, w = gray.shape
        top, bottom = 0, h - 60
        left, right = 40, w - 60
        cropped = gray[top:bottom, left:right]

        # Resize to target frame_size
        resized = cv2.resize(
            cropped,
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_AREA,
        )

        # Normalize to [0, 1] float32
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
        """Start a new episode and return initial stacked observation."""
        self.game.new_episode()
        state = self.game.get_state()
        if state is None:
            raise RuntimeError("Game state is None right after new_episode()")
        frame = self._preprocess(state.screen_buffer)
        stacked = self._stack(frame, new_episode=True)
        return stacked

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Frame-skip step with reward accumulation."""
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
            # If episode finished, push zeros as next frame
            next_frame = np.zeros(
                (self.frame_size, self.frame_size), dtype=np.float32
            )
            stacked = self._stack(next_frame, new_episode=False)
        else:
            state = self.game.get_state()
            if state is None:
                # Safety check: treat as done with zero frame
                next_frame = np.zeros(
                    (self.frame_size, self.frame_size), dtype=np.float32
                )
            else:
                next_frame = self._preprocess(state.screen_buffer)
            stacked = self._stack(next_frame, new_episode=False)

        info = {}
        return stacked, float(total_reward), bool(done), info

    def close(self):
        self.game.close()
