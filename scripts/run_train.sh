#!/usr/bin/env bash
set -e

# Activate conda environment
# Use Anaconda installation detected at ~/anaconda3
source ~/anaconda3/etc/profile.d/conda.sh
conda activate doomrl

# Go to project root
cd /home/cia/disk1/bci_intern/AAAI2026/RLDoom

# Use only GPU 3
export CUDA_VISIBLE_DEVICES=3

# Set local wandb directory
export WANDB_DIR="${PWD}/logs/wandb"
mkdir -p "$WANDB_DIR"

# Create logs directory if it does not exist
mkdir -p logs

# Run training and save console output
python train.py 2>&1 | tee logs/train.log
