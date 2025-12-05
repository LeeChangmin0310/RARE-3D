# rldoom/trainers/offpolicy.py
from typing import Dict, Any
from tqdm import trange
import os
import shutil

from rldoom.envs import make_env


def train_offpolicy(agent, cfg, logger):
    """
    Generic off-policy training loop
    (DQN / DDQN / DDDQN / Rainbow).

    - Each environment step:
        * collect transition
        * store into replay buffer
        * call agent.update() for TD learning
    - Logging is done once per episode, using the latest metrics
      returned by agent.update() inside that episode.
    - Checkpoints are saved periodically via Agent.save().
    """
    env = make_env(cfg)
    global_step = 0

    # Ensure checkpoint directory exists
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Main training loop over episodes
    for ep in trange(cfg.train_episodes, desc=f"{cfg.algo} train", dynamic_ncols=True):
        obs = env.reset()
        episode_return = 0.0
        episode_len = 0

        # Will store the last non-empty metrics from agent.update()
        last_metrics: Dict[str, Any] = {}

        while True:
            # 1) Select action from the current policy (epsilon-greedy, etc.)
            action = agent.act(obs, deterministic=False)

            # 2) Step the environment
            next_obs, reward, done, info = env.step(action)

            # 3) Store transition into the replay buffer
            transition = (obs, action, reward, next_obs, done)
            agent.observe(transition)

            # 4) Perform one off-policy TD update (if buffer is ready)
            step_metrics: Dict[str, Any] = agent.update()
            if step_metrics:
                # Keep the most recent loss/value_loss, etc. for this episode
                last_metrics = step_metrics

            # 5) Bookkeeping
            episode_return += reward
            episode_len += 1
            global_step += 1
            obs = next_obs

            # 6) Episode termination condition
            if done or episode_len >= cfg.max_steps_per_episode:
                break

        # -------- episode-level logging --------
        ep_idx = ep + 1  # 1-based episode index (aligns with on-policy loop)

        log_dict: Dict[str, float] = {
            "episode": float(ep_idx),
            "return": float(episode_return),
            "length": float(episode_len),
            "global_step": float(global_step),
        }
        # Merge the last metrics from TD updates (e.g., loss, value_loss)
        for k, v in last_metrics.items():
            log_dict[k] = float(v)

        # Use ep_idx as wandb step so that x-axis = episode (same as on-policy)
        logger.log_metrics(log_dict, step=ep_idx)

        # -------- checkpointing --------
        if (
            ep_idx % cfg.checkpoint_interval == 0
            or ep_idx == cfg.train_episodes
        ):
            ckpt_name = f"{cfg.algo}_seed{cfg.seed}_ep{ep_idx:06d}.pth"
            ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
            agent.save(ckpt_path)

            latest_path = os.path.join(
                cfg.checkpoint_dir,
                f"{cfg.algo}_seed{cfg.seed}_latest.pth",
            )
            shutil.copy2(ckpt_path, latest_path)

    env.close()
    logger.close()
