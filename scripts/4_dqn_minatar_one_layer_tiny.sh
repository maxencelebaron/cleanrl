#!/bin/bash
COMMAND="uv run cleanrl/feature_rank/4_dqn_minatar_tiny_one_layer.py \
    --track \
    --wandb-project-name "growing_network_for_feature_collapse" \
    --wandb-entity "tekoumaxencelebaron-cole-normale-sup-rieure-paris-saclay" \
    --capture-video \
    --total-timesteps 5_000_000 \
    --buffer-size 100_000 \
    --start-e 1 \
    --end-e 0.01 \
    --exploration-fraction 0.05 \
    --learning-starts 10_000 \
    --train-frequency 1 \
    --feature-rank-n-states 2000 \
    --return-window-size 100 \
    --compute-final-feature-rank"

uv run python scripts/submit.py \
    --env-ids "MinAtar/Breakout-v1" \
    --seeds 1 2 3 4 5 \
    --command "$COMMAND" \
    --gres "gpu:1" \
    --time "7-00:00:00" \
    "$@"

# "MinAtar/Asterix-v1" "MinAtar/Freeway-v1" "MinAtar/Seaquest-v1" "MinAtar/SpaceInvaders-v1"\
