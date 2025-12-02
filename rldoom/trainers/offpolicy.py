# rldoom/trainers/offpolicy.py
from typing import Dict, Any
from tqdm import trange

from rldoom.envs import make_env



def train_offpolicy(agent, cfg, logger):
    """Generic off-policy training loop (DQN / DDQN / Rainbow)."""
    env = make_env(cfg)
    global_step = 0

    for ep in trange(cfg.train_episodes, desc=f"{cfg.algo} train", dynamic_ncols=True):
        obs = env.reset()
        episode_return = 0.0
        episode_len = 0
        last_metrics: Dict[str, Any] = {}

        for t in range(cfg.max_steps_per_episode):
            action = agent.act(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)

            transition = (obs, action, reward, next_obs, done)
            agent.observe(transition)
            metrics = agent.update()

            if metrics:
                last_metrics = metrics

            episode_return += reward
            episode_len += 1
            global_step += 1
            obs = next_obs

            if done:
                break

        log_dict: Dict[str, float] = {
            "return": episode_return,
            "length": episode_len,
            "global_step": float(global_step),
        }
        for k, v in last_metrics.items():
            log_dict[k] = float(v)

        logger.log_metrics(log_dict, step=ep)

    env.close()
    logger.close()
