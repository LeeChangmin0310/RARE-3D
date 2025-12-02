# rldoom/utils/logger.py
from typing import Dict, Any
import os

try:
    import wandb
except ImportError:
    wandb = None


class Logger:
    """Minimal logger with optional wandb logging."""

    def __init__(self, cfg):
        self.cfg = cfg

        # 1) Decide whether to use wandb
        #    - cfg.use_wandb must be True
        #    - wandb package must be installed
        self.use_wandb = bool(getattr(cfg, "use_wandb", False) and wandb is not None)

        # 2) Read settings from environment first, then fall back to cfg
        project = os.getenv("WANDB_PROJECT", getattr(cfg, "wandb_project", None))
        entity = os.getenv("WANDB_ENTITY", getattr(cfg, "wandb_entity", None))
        wandb_dir = os.getenv("WANDB_DIR", None)

        if wandb_dir:
            os.makedirs(wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = wandb_dir  # wandb respects this

        # Set WANDB_API_KEY at .env -> shell

        self.run = None
        if self.use_wandb and project is not None:
            self.run = wandb.init(
                project=project,
                entity=entity,
                name=f"{cfg.algo}_seed{cfg.seed}",
                dir=wandb_dir,
                config={
                    "algo": cfg.algo,
                    "seed": cfg.seed,
                    "frame_size": cfg.frame_size,
                    "stack_size": cfg.stack_size,
                    "gamma": cfg.gamma,
                },
            )

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics to stdout and optionally to wandb."""
        line = f"[step={step}] " + " ".join(
            f"{k}={v:.4f}" for k, v in metrics.items()
        )
        print(line)

        if self.use_wandb and self.run is not None:
            wandb.log(metrics, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()
