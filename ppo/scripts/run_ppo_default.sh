#!/bin/bash
# Run PPO training with default hyperparameters

python scripts/train_ppo.py \
    --env_name MountainCarContinuous-v0 \
    --total_timesteps 400000 \
    --timesteps_per_batch 4096 \
    --max_timesteps_per_episode 256 \
    --discount_factor 0.99 \
    --n_epochs 10 \
    --clip_ratio 0.2 \
    --lr 0.0005
