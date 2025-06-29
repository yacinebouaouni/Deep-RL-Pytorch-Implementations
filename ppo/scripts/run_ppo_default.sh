#!/bin/bash
# Run PPO training with default hyperparameters

python scripts/train_ppo.py \
    --env_name LunarLander-v3 \
    --total_timesteps 500000 \
    --timesteps_per_batch  4096 \
    --max_timesteps_per_episode 1000 \
    --discount_factor 0.99 \
    --n_epochs 5 \
    --clip_ratio 0.2 \
    --lr 5e-4 \
    --coeff_loss_vf 0.5 \
    --coeff_loss_entropy 0.01 \
    --coeff_gae_lambda 0.95
