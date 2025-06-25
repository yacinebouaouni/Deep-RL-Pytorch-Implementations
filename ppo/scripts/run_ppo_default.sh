#!/bin/bash
# Run PPO training with default hyperparameters

python scripts/train_ppo.py \
    --env_name MountainCarContinuous-v0 \
    --total_timesteps 100000 \
    --timesteps_per_batch  2048 \
    --max_timesteps_per_episode 256 \
    --discount_factor 0.99 \
    --n_epochs 10 \
    --clip_ratio 0.1 \
    --lr 0.001 \
    --coeff_loss_vf 0.5 \
    --coeff_loss_entropy 0.01 \
