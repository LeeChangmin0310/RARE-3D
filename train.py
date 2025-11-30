# train.py

import os
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from config import Config
from envs.doom_env import DoomEnv

from models.dddqn import DuelingDQN
from models.dqn import DQN
from models.ppo import PPO
from models.a3c import A3C
from models.trpo import TRPO

from memory.per_memory import PERMemory

from dotenv import load_dotenv

load_dotenv()

WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
WANDB_DIR = os.getenv("WANDB_DIR")

def select_action(q_net, state, epsilon, num_actions, device):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        action_idx = np.random.randint(0, num_actions)
        return action_idx

    state_t = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        q_values = q_net(state_t)
    action_idx = int(torch.argmax(q_values, dim=1).item())
    return action_idx

def select_action_ppo(q_net, state, epsilon, num_actions, device):
    """Epsilon-greedy action selection for PPO and other models."""
    if np.random.rand() < epsilon:
        # Explore: random action
        action_idx = np.random.randint(0, num_actions)
        return action_idx

    state_t = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        # Get the policy (probability distribution) and value from the network
        policy, value = q_net(state_t)

    action_probs = policy.squeeze(0)  # (A,)
    action_idx = torch.multinomial(action_probs, 1).item()  # 샘플링을 통해 액션 선택
    return action_idx

def select_action_a3c(q_net, state, epsilon, num_actions, device):
    """Epsilon-greedy action selection."""
    if np.random.rand() < epsilon:
        action_idx = np.random.randint(0, num_actions)
        return action_idx

    state_t = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        # q_net가 정책과 가치를 튜플로 반환할 수 있기 때문에 정책만 사용
        policy, value = q_net(state_t)

    if policy is None:
        raise ValueError("Policy output from q_net is None. Ensure q_net returns both policy and value.")

    action_probs = policy.squeeze(0)  # (A,)
    action_idx = torch.multinomial(action_probs, 1).item()  # 샘플링을 통해 액션 선택
    return action_idx

def select_action_trpo(q_net, state, epsilon, num_actions, device):
    """Epsilon-greedy action selection for TRPO."""
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, C, H, W)
    with torch.no_grad():
        # Get the policy (probability distribution) and value from the network
        policy = q_net(state_t)

    action_probs = policy.squeeze(0)  # (A,)
    action_idx = torch.multinomial(action_probs, 1).item()  # 샘플링을 통해 액션 선택
    return action_idx

def soft_update(target_net, online_net, tau):
    """Soft or hard update of target network parameters."""
    if tau >= 1.0:
        # Hard update
        target_net.load_state_dict(online_net.state_dict())
        return

    # Soft update
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(
            tau * online_param.data + (1.0 - tau) * target_param.data
        )

def train():
    cfg = Config()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.logs_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize environment
    env = DoomEnv(
        config_path=cfg.config_path,
        scenario_path=cfg.scenario_path,
        frame_height=cfg.frame_height,
        frame_width=cfg.frame_width,
        stack_size=cfg.stack_size,
    )
    num_actions = len(env.possible_actions)

    # Initialize networks
    if cfg.algorithm == "dddqn":
        online_net = DuelingDQN(cfg.stack_size, num_actions).to(device)
        target_net = DuelingDQN(cfg.stack_size, num_actions).to(device)
    elif cfg.algorithm == "dqn":
        online_net = DQN(cfg.stack_size, num_actions).to(device)
        target_net = DQN(cfg.stack_size, num_actions).to(device)
    elif cfg.algorithm == "ppo":
        online_net = PPO(cfg.stack_size, num_actions).to(device)
        target_net = PPO(cfg.stack_size, num_actions).to(device)
    elif cfg.algorithm == "a3c":
        online_net = A3C(cfg.stack_size, num_actions).to(device)
        target_net = A3C(cfg.stack_size, num_actions).to(device)
    elif cfg.algorithm == "trpo":
        online_net = TRPO(cfg.stack_size, num_actions).to(device)
        target_net = TRPO(cfg.stack_size, num_actions).to(device)
        old_policy_net = TRPO(cfg.stack_size, num_actions).to(device)

    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(online_net.parameters(), lr=cfg.learning_rate)
    mse_loss = nn.MSELoss(reduction="none")

    # Replay memory
    memory = PERMemory(cfg.memory_size)

    # wandb
    wandb.init(
        project=cfg.wandb_project,
        entity=os.environ.get("WANDB_ENTITY", None),
        name=cfg.wandb_run_name,
        config={
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "batch_size": cfg.batch_size,
            "total_episodes": cfg.total_episodes,
            "max_steps_per_episode": cfg.max_steps_per_episode,
            "memory_size": cfg.memory_size,
            "pretrain_length": cfg.pretrain_length,
        },
    )

    # Pre-fill replay buffer with random policy
    print("[INFO] Pre-filling replay buffer...")
    state = env.reset()
    for _ in trange(cfg.pretrain_length):
        action_idx = np.random.randint(0, num_actions)
        next_state, reward, done = env.step(action_idx)
        memory.store((state, action_idx, reward, next_state, done))

        if done:
            state = env.reset()
        else:
            state = next_state

    global_step = 0
    epsilon = cfg.eps_start
    best_reward = -np.inf  # track best episode reward

    pbar = trange(1, cfg.total_episodes + 1, desc="Training episodes")
    for episode in pbar:
        state = env.reset()
        episode_reward = 0.0
        losses = []

        for step in range(cfg.max_steps_per_episode):
            global_step += 1

            # Exponential epsilon decay
            epsilon = max(
                cfg.eps_end,
                cfg.eps_end
                + (cfg.eps_start - cfg.eps_end)
                * np.exp(-cfg.eps_decay * global_step),
            )

            if cfg.algorithm == "ppo":
                action_idx = select_action_ppo(online_net, state, epsilon, num_actions, device)
            elif cfg.algorithm == "a3c":
                action_idx = select_action_a3c(online_net, state, epsilon, num_actions, device)
            elif cfg.algorithm == "trpo":
                action_idx = select_action_trpo(online_net, state, epsilon, num_actions, device)
            else:
                action_idx = select_action(online_net, state, epsilon, num_actions, device)

            next_state, reward, done = env.step(action_idx)
            episode_reward += reward

            memory.store((state, action_idx, reward, next_state, done))
            state = next_state

            # Learning phase
            if global_step > cfg.learn_start:
                idxs, batch, is_weights = memory.sample(cfg.batch_size)

                states = np.stack([b[0] for b in batch], axis=0)
                actions = np.array([b[1] for b in batch], dtype=np.int64)
                rewards = np.array([b[2] for b in batch], dtype=np.float32)
                next_states = np.stack([b[3] for b in batch], axis=0)
                dones = np.array([b[4] for b in batch], dtype=np.float32)

                states_t = torch.from_numpy(states).to(device)
                next_states_t = torch.from_numpy(next_states).to(device)
                actions_t = torch.from_numpy(actions).to(device)
                rewards_t = torch.from_numpy(rewards).to(device)
                dones_t = torch.from_numpy(dones).to(device)
                is_weights_t = torch.from_numpy(is_weights).to(device)

                if cfg.algorithm == "dddqn" or cfg.algorithm == "dqn":
                    # Current Q-values
                    q_values = online_net(states_t)  # (B, A)
                    q_values = q_values.gather(
                        1, actions_t.unsqueeze(1)
                    ).squeeze(1)  # (B,)
                    # Double DQN target
                    with torch.no_grad():
                        next_q_online = online_net(next_states_t)
                        next_actions = torch.argmax(next_q_online, dim=1)

                        next_q_target = target_net(next_states_t)
                        next_q = next_q_target.gather(
                            1, next_actions.unsqueeze(1)
                        ).squeeze(1)
                        targets = rewards_t + cfg.gamma * (1.0 - dones_t) * next_q
                    # PER-weighted loss
                    loss_per_sample = mse_loss(q_values, targets)
                    loss = (is_weights_t.squeeze(1) * loss_per_sample).mean()
                
                elif cfg.algorithm == "ppo":
                    # PPO loss calculation (with clipping)
                    policy, value = online_net(states_t)
                    next_policy, next_value = target_net(next_states_t)
                    # PPO loss using advantage (clipped)
                    log_probs = torch.log(policy.gather(1, actions_t.unsqueeze(1)))
                    value_loss = 0.5 * (rewards_t - value).pow(2)
                    advantage = rewards_t - value.detach()
                    policy_loss = -log_probs * advantage.detach()
                    loss = policy_loss.mean() + value_loss.mean()

                elif cfg.algorithm == "a3c":
                    policy, value = online_net(states_t)
                    # A3C 손실 계산
                    log_probs = torch.log(policy.gather(1, actions_t.unsqueeze(1)))
                    value_loss = 0.5 * (rewards_t - value).pow(2)
                    advantage = rewards_t - value.detach()
                    policy_loss = -log_probs * advantage.detach()
                    total_loss = policy_loss.mean() + value_loss.mean()
                    loss = total_loss

                elif cfg.algorithm == "trpo":
                    policy = online_net(states_t)  # Get the policy (probability distribution)
                    # TRPO loss calculation (with KL divergence)
                    log_probs = torch.log(policy.gather(1, actions_t.unsqueeze(1)))  # Log probabilities of chosen actions
                    old_policy = old_policy_net(states_t)  # Old policy (for KL divergence)
                    # Compute KL divergence between the new and old policy
                    kl_div = torch.mean(torch.sum(old_policy * (torch.log(old_policy) - log_probs), dim=1))
                    # TRPO loss: policy loss + KL divergence
                    policy_loss = -log_probs
                    # Total loss for TRPO
                    loss = policy_loss.mean() + cfg.kl_coeff * kl_div

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(online_net.parameters(), cfg.grad_clip)
                optimizer.step()

                losses.append(loss.item())

                if cfg.algorithm == "dddqn" or cfg.algorithm == "dqn":
                    # Update priorities
                    abs_errors = torch.abs(q_values - targets).detach().cpu().numpy()
                    memory.update_batch(idxs, abs_errors)

                if cfg.algorithm == 'trpo':
                    old_policy_net.load_state_dict(online_net.state_dict())

                # Update target network
                if global_step % cfg.target_update_freq == 0:
                    soft_update(target_net, online_net, cfg.tau)

            if done:
                break

        avg_loss = float(np.mean(losses)) if losses else 0.0
        
        pbar.set_postfix(
            reward=f"{episode_reward:.2f}",
            loss=f"{avg_loss:.4f}",
            eps=f"{epsilon:.3f}",
        )

        if episode_reward > best_reward:
            best_reward = episode_reward
            print(
                "[BEST] EP {:05d} reward={:.2f} loss={:.4f} epsilon={:.4f}".format(
                    episode, episode_reward, avg_loss, epsilon
                )
            )

            # 2) save best checkpoint
            best_path = os.path.join(cfg.checkpoint_dir, "dddqn_best.pt")
            ckpt = {
                "episode": episode,
                "global_step": global_step,
                "online_state_dict": online_net.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epsilon": epsilon,
                "best_reward": best_reward,
            }
            torch.save(ckpt, best_path)
            # W&B summary에 최고 값 기록
            wandb.run.summary["best_reward"] = best_reward
            wandb.save(best_path, base_path=cfg.base_dir)

        # wandb logging
        wandb.log(
            {
                "train/episode_reward": episode_reward,
                "train/loss": avg_loss,
                "train/epsilon": epsilon,
                "train/episode": episode,
                "train/global_step": global_step,
            },
            step=episode,
        )

        # Checkpoint saving
        if episode % cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, "dddqn_ep_{:06d}.pt".format(episode)
            )
            latest_path = os.path.join(cfg.checkpoint_dir, "dddqn_latest.pt")

            ckpt = {
                "episode": episode,
                "global_step": global_step,
                "online_state_dict": online_net.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epsilon": epsilon,
            }

            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, latest_path)

            # Upload checkpoint to wandb (optional)
            # Use base_path to preserve folder structure in W&B
            wandb.save(ckpt_path, base_path=cfg.base_dir)
            print("[INFO] Saved checkpoint to {}".format(ckpt_path))

    env.close()
    wandb.finish()


if __name__ == "__main__":
    train()
