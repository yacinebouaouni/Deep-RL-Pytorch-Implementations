#!/bin/bash
# Hyperparameter tuning script for PPO
# Runs ~300 combinations for total_timesteps=100000

ENV_NAME="MountainCarContinuous-v0"
TOTAL_TIMESTEPS=100000

# Reduced parameter grids to get close to 300 combinations
TIMESTEPS_PER_BATCH=(2048 4096 8192)
MAX_TIMESTEPS_PER_EPISODE=(256 999)
DISCOUNT_FACTOR=(0.90 0.95 0.99)
N_EPOCHS=(20)
CLIP_RATIO=(0.1 0.2)
LR=(0.0001 0.0004 0.001)
COEFF_LOSS_VF=(0.3 0.7)
COEFF_LOSS_ENTROPY=(0.005 0.01)

count=0
for tpb in "${TIMESTEPS_PER_BATCH[@]}"; do
  for mte in "${MAX_TIMESTEPS_PER_EPISODE[@]}"; do
    for gamma in "${DISCOUNT_FACTOR[@]}"; do
      for ne in "${N_EPOCHS[@]}"; do
        for cr in "${CLIP_RATIO[@]}"; do
          for lr in "${LR[@]}"; do
            for vf in "${COEFF_LOSS_VF[@]}"; do
              for ent in "${COEFF_LOSS_ENTROPY[@]}"; do
                ((count++))
                echo "Run $count: tpb=$tpb, mte=$mte, gamma=$gamma, ne=$ne, cr=$cr, lr=$lr, vf=$vf, ent=$ent"
                python scripts/train_ppo.py \
                  --env_name $ENV_NAME \
                  --total_timesteps $TOTAL_TIMESTEPS \
                  --timesteps_per_batch $tpb \
                  --max_timesteps_per_episode $mte \
                  --discount_factor $gamma \
                  --n_epochs $ne \
                  --clip_ratio $cr \
                  --lr $lr \
                  --coeff_loss_vf $vf \
                  --coeff_loss_entropy $ent
                # Optionally, break after 300 runs
                if [[ $count -ge 300 ]]; then
                  exit 0
                fi
              done
            done
          done
        done
      done
    done
  done

done
