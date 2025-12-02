#!/usr/bin/env bash
set -e

source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi

export CUDA_VISIBLE_DEVICES=3

ALGO=${1:-dqn}
SEED=${2:-0}
CKPT_PATH=${3:-"checkpoints/${ALGO}_seed${SEED}.pth"}

python -u train.py --algo "${ALGO}" --seed "${SEED}" \
  --mode eval --checkpoint "${CKPT_PATH}"
