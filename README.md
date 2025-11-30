# Doom DDDQN: Prioritized Dueling Double DQN for VizDoom

**TL;DR.** This repo implements a **Dueling Double Deep Q-Network (DDDQN)** with **Prioritized Experience Replay (PER)** for the VizDoom **Deadly Corridor** scenario, using **PyTorch**, **wandb**, and a tmux-friendly training script.

The goal is a clean, self-contained RL toy project that can run on a shared lab server without polluting the global environment.

---

## Features

- **Environment wrapper**
  - VizDoom Deadly Corridor (`deadly_corridor.cfg`, `deadly_corridor.wad`)
  - Frame preprocessing (crop → normalize → resize)
  - Frame stacking (4 frames → shape `(C, H, W)`)

- **RL algorithm**
  - Dueling DQN (separate value + advantage streams)
  - Double DQN target (online network for argmax, target network for value)
  - Prioritized Experience Replay (SumTree implementation)

- **Training infrastructure**
  - PyTorch implementation (DDDQN + PER)
  - Hard / soft target network update
  - Checkpointing (`checkpoints/`)
  - wandb logging (episode reward, loss, epsilon, etc.)
  - tmux-friendly shell scripts (`scripts/run_train.sh`, optional `run_eval.sh`)

---

## Environment

Tested on a shared lab server:

- **OS:** Ubuntu 20.04 / 22.04
- **GPU:** NVIDIA RTX A5000 / RTX 3090 (CUDA 12.x driver)
- **Python:** 3.9 (via conda env `doomrl`)
- **Frameworks:** PyTorch, VizDoom, wandb

All dependencies are installed only inside the `doomrl` conda environment so that the system / lab environment is not modified.

---

## Installation

Clone the repository and create a dedicated conda environment:

```bash
git clone <THIS_REPO_URL> RLDoom
cd RLDoom

# Create dedicated env (do not touch base / system env)
conda create -n doomrl python=3.9 -y
conda activate doomrl

# Basic libs (can also be installed via requirements.txt)
pip install -r requirements.txt
# or, manually:
# pip install numpy scikit-image tqdm matplotlib vizdoom wandb torch torchvision
````

Make sure the environment’s Python is used:

```bash
which python
# -> something like: /home/<USER>/anaconda3/envs/doomrl/bin/python
```

---

## VizDoom assets

This project assumes the following files exist in the **repo root**:

* `deadly_corridor.cfg`
* `deadly_corridor.wad`

You can copy them from the VizDoom examples or from Thomas Simonini’s Doom tutorial resources.

`config.py` currently expects:

```python
class Config:
    config_path = "deadly_corridor.cfg"
    scenario_path = "deadly_corridor.wad"
```

If you prefer `assets/` or another directory, adjust these paths accordingly.

---

## Repository Structure

```text
RLDoom/
  config.py               # Global hyperparameters & paths
  train.py                # Main training script (DDDQN + PER + wandb)
  eval.py                 # Evaluation script for trained agent
  requirements.txt        # Python dependencies

  envs/
    __init__.py
    doom_env.py           # VizDoom wrapper + preprocessing + frame stacking

  models/
    __init__.py
    dddqn.py              # Dueling DQN (value + advantage streams)

  memory/
    __init__.py
    sumtree.py            # SumTree data structure for PER
    per_memory.py         # Prioritized Experience Replay buffer

  scripts/
    run_train.sh          # tmux-friendly training launcher (GPU + env setup)
    run_eval.sh           # (optional) evaluation launcher

  checkpoints/            # Saved PyTorch checkpoints (created at runtime)
  logs/                   # Console logs + wandb local cache (created at runtime)

  deadly_corridor.cfg     # VizDoom config (expected)
  deadly_corridor.wad     # VizDoom scenario (expected)
  README.md
```

---

## Configuration

Global configuration is centralized in `config.py`:

```python
class Config:
    # Doom assets
    config_path = "deadly_corridor.cfg"
    scenario_path = "deadly_corridor.wad"

    # Frame & stack
    frame_height = 100
    frame_width = 120
    stack_size = 4

    # Training
    learning_rate = 2.5e-4
    gamma = 0.95
    total_episodes = 1000
    max_steps_per_episode = 3000

    # Epsilon-greedy
    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 5e-5

    # Replay memory
    memory_size = 100000
    pretrain_length = 50000
    learn_start = 1000

    # Target network
    target_update_freq = 1000
    tau = 1.0  # 1.0 = hard update; <1.0 = soft update

    # Optimizer
    grad_clip = 10.0
    batch_size = 64

    # Paths
    checkpoint_dir = "<REPO_ROOT>/checkpoints"
    logs_dir = "<REPO_ROOT>/logs"
    checkpoint_interval = 50

    # wandb
    wandb_project = "doom-dddqn"
    wandb_run_name = "dddqn_deadly_corridor"
```

You can tune hyperparameters here without touching the training code.

---

## Training

### 1. Prepare wandb

Once inside the `doomrl` environment:

```bash
conda activate doomrl
wandb login
# paste your API key
```

### 2. Training via script (recommended)

Edit `scripts/run_train.sh` to match your environment (conda path, repo path, GPU index).
Example:

```bash
#!/usr/bin/env bash
set -e

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

# Go to project root
cd /home/<USER>/disk1/bci_intern/AAAI2026/RLDoom

# Use only GPU 3 (visible as device 0 inside the process)
export CUDA_VISIBLE_DEVICES=3

# wandb local cache
export WANDB_DIR="${PWD}/logs/wandb"
mkdir -p "$WANDB_DIR"

mkdir -p logs

python train.py 2>&1 | tee logs/train.log
```

Make it executable:

```bash
chmod +x scripts/run_train.sh
```

Run it inside tmux:

```bash
tmux new -s doomrl
bash scripts/run_train.sh

# Detach:  Ctrl+b, d
# Reattach: tmux attach -t doomrl
```

Training will:

* Pre-fill replay memory with random actions
* Train DDDQN with PER
* Log metrics to wandb
* Save checkpoints to `checkpoints/dddqn_ep_XXXXXX.pt` and `checkpoints/dddqn_latest.pt`

---

## Evaluation

You can evaluate a trained agent with the latest checkpoint:

```bash
conda activate doomrl
cd /home/<USER>/disk1/bci_intern/AAAI2026/RLDoom

python eval.py
# or specify a checkpoint:
# python eval.py --ckpt checkpoints/dddqn_ep_000500.pt
```

If you prefer a script, you can use `scripts/run_eval.sh`:

```bash
#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

cd /home/<USER>/disk1/bci_intern/AAAI2026/RLDoom

export CUDA_VISIBLE_DEVICES=3

mkdir -p logs
python eval.py 2>&1 | tee logs/eval.log
```

(Use your actual path and GPU index.)

---

## Logging & Checkpoints

* **Console logs:**
  `logs/train.log`, `logs/eval.log` (from `tee` in the scripts).

* **wandb logs:**

  * Project name: `doom-dddqn` (configurable in `Config.wandb_project`)
  * Metrics: episode reward, mean loss, epsilon, global step, etc.
  * Local cache: `logs/wandb` (via `WANDB_DIR`)

* **Checkpoints:**

  * Saved every `checkpoint_interval` episodes (default: 50)
  * Format: `checkpoints/dddqn_ep_000050.pt`, `checkpoints/dddqn_latest.pt`
  * Each checkpoint contains:

    * Online network weights
    * Target network weights
    * Optimizer state
    * Episode index, global step, epsilon

---

## Method Overview

* **Input state:** stack of 4 grayscale frames
  Shape: `(C=4, H=100, W=120)` after crop + normalize + resize.

* **Network:** Dueling DQN (`models/dddqn.py`)

  * Conv encoder (3 conv layers)
  * Flatten
  * Two streams:

    * Value stream: `V(s)`
    * Advantage stream: `A(s, a)`
  * Aggregation:
    `Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)`

* **Double DQN target:**

  * Online network chooses `argmax_a Q(s', a)`
  * Target network evaluates that action’s Q-value
  * Target:
    `y = r + γ (1 − done) Q_target(s', argmax_a Q_online(s', a))`

* **Prioritized Experience Replay:**

  * SumTree (`memory/sumtree.py`) stores priorities and experiences.
  * Memory (`memory/per_memory.py`) implements:

    * Sampling by priority
    * Importance-sampling weights for unbiased updates
    * Priority updates using TD error

* **Optimization:**

  * Loss: PER-weighted MSE between current Q and target
  * Optimizer: RMSprop
  * Gradient clipping: `max_grad_norm = 10.0`
  * Target network update: every `target_update_freq` global steps, using hard or soft update (`tau`)

---

## Credits & References

This project is inspired by:

* Thomas Simonini, **“Dueling Double Deep Q-Learning with PER — Doom Deadly Corridor”**
  Deep Reinforcement Learning Course (TensorFlow notebook).
* Mnih et al., **“Human-level control through deep reinforcement learning,”** Nature, 2015.
* Wang et al., **“Dueling Network Architectures for Deep Reinforcement Learning,”** ICML, 2016.
* Van Hasselt et al., **“Deep Reinforcement Learning with Double Q-learning,”** AAAI, 2016.
* Schaul et al., **“Prioritized Experience Replay,”** ICLR, 2016.
* VizDoom: **“ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning”**

All original Doom assets belong to their respective copyright holders.

---

## Contributors

<table>
  <tr>
    <td align="center" valign="top" width="160">
      <a href="https://github.com/LeeChangmin0310">
        <img src="https://github.com/LeeChangmin0310.png?size=120" width="96" height="96" alt="LeeChangmin0310 avatar"/><br/>
        <sub><b>Changmin Lee</b></sub><br/>
        <sub>@LeeChangmin0310</sub><br/>
        <sub>Maintainer</sub>
      </a>
    </td>
    <td align="center" valign="top" width="160">
      <a href="https://github.com/suyeonmyeong">
        <img src="https://github.com/suyeonmyeong.png?size=120" width="96" height="96" alt="suyeonmyeong avatar"/><br/>
        <sub><b>Suyeon Myung</b></sub><br/>
        <sub>@suyeonmyeong</sub><br/>
        <sub>Core Contributor</sub>
      </a>
    </td>
    <td align="center" valign="top" width="160">
      <a href="https://github.com/maeng00">
        <img src="https://github.com/maeng00.png?size=120" width="96" height="96" alt="maeng00 avatar"/><br/>
        <sub><b>Ui-Hyun Maeng</b></sub><br/>
        <sub>@maeng00</sub><br/>
        <sub>Core Contributor</sub>
      </a>
    </td>
  </tr>
</table>


---
