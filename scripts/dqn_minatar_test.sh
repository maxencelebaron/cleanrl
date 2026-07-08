#!/bin/bash
COMMAND="uv run cleanrl/feature_rank/dqn_minatar_grow_hidden_only.py \
    --capture-video \
    --total-timesteps 150_000 \
    --buffer-size 20_000 \
    --batch_size 128 \
    --start-e 1 \
    --end-e 0.1 \
    --exploration-fraction 0.02 \
    --feature-rank-n-states 1000 \
    --return-window-size 100 \
    --learning-starts 5_000 \
    --compute-final-feature-rank \
    --plasticity-n-steps 100 \
    --plasticity-n-tasks 5 \
    --growth-adjust-steps 100"

uv run python scripts/submit.py \
    --env-ids "MinAtar/Asterix-v1" \
    --seeds 1 \
    --command "$COMMAND" \
    --gres "gpu:1" \
    --time "7-00:00:00" \
    "$@"

# "MinAtar/Asterix-v1" "MinAtar/Freeway-v1" "MinAtar/Seaquest-v1" "MinAtar/SpaceInvaders-v1"\
