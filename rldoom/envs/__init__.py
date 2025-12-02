# rldoom/envs/__init__.py
from .deadly_corridor import DeadlyCorridorEnv


def make_env(cfg):
    """Create Deadly Corridor env from config."""
    env = DeadlyCorridorEnv(
        cfg_path=cfg.cfg_path,
        wad_path=cfg.wad_path,
        frame_size=cfg.frame_size,
        stack_size=cfg.stack_size,
        frame_skip=cfg.frame_skip,
    )
    return env
