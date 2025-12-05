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

    - 매 step마다: act -> env.step -> agent.observe() -> agent.update()
    - agent.update()가 빈 dict가 아닌 metrics를 리턴하면,
      그 즉시 logger.log_metrics로 loss/value_loss를 찍는다.
    - 에피소드가 끝날 때 return/length/global_step도 따로 로깅.
    """
    env = make_env(cfg)
    global_step = 0

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for ep_idx in trange(cfg.train_episodes, desc=f"{cfg.algo} train", dynamic_ncols=True):
        obs = env.reset()
        episode_return = 0.0
        episode_len = 0

        while True:
            # 1) Select action
            action = agent.act(obs, deterministic=False)

            # 2) Step environment
            next_obs, reward, done, info = env.step(action)

            # 3) Give transition to agent
            transition = (obs, action, reward, next_obs, done)
            agent.observe(transition)

            # 4) One off-policy update step
            step_metrics: Dict[str, Any] = agent.update()

            # 5) Bookkeeping
            episode_return += reward
            episode_len += 1
            global_step += 1
            obs = next_obs

            # --- per-step logging of losses (if any) ---
            # 여기서 loss/value_loss가 실제로 wandb에 만들어진다.
            if step_metrics:
                log_step_dict: Dict[str, float] = {
                    "global_step": float(global_step),
                    "episode": float(ep_idx + 1),
                    "return_so_far": float(episode_return),
                    "length_so_far": float(episode_len),
                }
                for k, v in step_metrics.items():
                    log_step_dict[k] = float(v)

                # x축을 global_step으로 해서 step 단위 loss를 찍음
                logger.log_metrics(log_step_dict, step=global_step)

            if done or episode_len >= cfg.max_steps_per_episode:
                break

        # -------- episode-level logging (return 등) --------
        ep_num = ep_idx + 1
        ep_log: Dict[str, float] = {
            "episode": float(ep_num),
            "return": float(episode_return),
            "length": float(episode_len),
            "global_step": float(global_step),
        }
        logger.log_metrics(ep_log, step=ep_num)

        # -------- checkpointing --------
        if (ep_num % cfg.checkpoint_interval == 0) or (ep_num == cfg.train_episodes):
            ckpt_name = f"{cfg.algo}_seed{cfg.seed}_ep{ep_num:06d}.pth"
            ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
            agent.save(ckpt_path)

            latest_path = os.path.join(
                cfg.checkpoint_dir,
                f"{cfg.algo}_seed{cfg.seed}_latest.pth",
            )
            shutil.copy2(ckpt_path, latest_path)

    env.close()
    logger.close()
