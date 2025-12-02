# rldoom/trainers/onpolicy.py
from typing import Dict, Any
from tqdm import trange

from rldoom.envs import make_env


def train_onpolicy(agent, cfg, logger):
    """Generic on-policy training loop (REINFORCE / A2C / A3C / PPO / TRPO)."""
    env = make_env(cfg)
    global_step = 0

    for ep in trange(cfg.train_episodes, desc=f"{cfg.algo} train", dynamic_ncols=True):
        obs = env.reset()
        episode_return = 0.0
        episode_len = 0

        if hasattr(agent, "on_episode_start"):
            agent.on_episode_start()

        while True:
            action = agent.act(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)

            transition = (obs, action, reward, next_obs, done)
            agent.observe(transition)

            episode_return += reward
            episode_len += 1
            global_step += 1
            obs = next_obs

            if done or episode_len >= cfg.max_steps_per_episode:
                break

        metrics: Dict[str, Any] = agent.update()

        log_dict: Dict[str, float] = {
            "return": episode_return,
            "length": episode_len,
            "global_step": float(global_step),
        }
        for k, v in metrics.items():
            log_dict[k] = float(v)

        logger.log_metrics(log_dict, step=ep)

    env.close()
    logger.close()
