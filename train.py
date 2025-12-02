# train.py
import argparse
import os

import torch

from rldoom.configs import make_config
from rldoom.utils.logger import Logger
from rldoom.utils.seeding import set_seed

from rldoom.agents.dqn import DQNAgent
from rldoom.agents.ddqn import DDQNAgent
from rldoom.agents.dddqn import DDDQNAgent
from rldoom.agents.rainbow import RainbowAgent
from rldoom.agents.reinforce import ReinforceAgent
from rldoom.agents.a2c import A2CAgent
from rldoom.agents.a3c import A3CAgent
from rldoom.agents.ppo import PPOAgent
from rldoom.agents.trpo import TRPOAgent

from rldoom.trainers.offpolicy import train_offpolicy
from rldoom.trainers.onpolicy import train_onpolicy


def build_agent(algo: str, obs_shape, num_actions: int, cfg, device):
    """Factory for all supported agents."""
    if algo == "dqn":
        return DQNAgent(obs_shape, num_actions, cfg, device)
    if algo == "ddqn":
        return DDQNAgent(obs_shape, num_actions, cfg, device)
    if algo == "dddqn":
        return DDDQNAgent(obs_shape, num_actions, cfg, device)
    if algo == "rainbow":
        return RainbowAgent(obs_shape, num_actions, cfg, device)
    if algo == "reinforce":
        return ReinforceAgent(obs_shape, num_actions, cfg, device)
    if algo == "a2c":
        return A2CAgent(obs_shape, num_actions, cfg, device)
    if algo == "a3c":
        return A3CAgent(obs_shape, num_actions, cfg, device)
    if algo == "ppo":
        return PPOAgent(obs_shape, num_actions, cfg, device)
    if algo == "trpo":
        return TRPOAgent(obs_shape, num_actions, cfg, device)
    raise ValueError(f"Unknown algorithm: {algo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "ddqn", "dddqn", "rainbow",
                                 "reinforce", "a2c", "a3c", "ppo", "trpo"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Build config from YAML + selected algo + seed
    cfg = make_config(args.algo, args.seed)

    # Seeding
    set_seed(cfg.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Observation shape (C, H, W)
    obs_shape = (cfg.stack_size, cfg.frame_size, cfg.frame_size)
    # Deadly Corridor has 7 discrete actions
    num_actions = 7

    # Agent
    agent = build_agent(cfg.algo, obs_shape, num_actions, cfg, device)

    # Logger (handles wandb / tensorboard inside)
    logger = Logger(cfg)

    # Choose trainer by algo_type from YAML
    if cfg.algo_type == "offpolicy":
        train_offpolicy(agent, cfg, logger)
    elif cfg.algo_type == "onpolicy":
        train_onpolicy(agent, cfg, logger)
    else:
        raise ValueError(f"Unknown algo_type: {cfg.algo_type}")


if __name__ == "__main__":
    # Make sure we run from project root, so 'rldoom' is importable
    main()
