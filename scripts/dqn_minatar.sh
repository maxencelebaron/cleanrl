#!/bin/bash
COMMAND="uv run cleanrl/feature_rank/dqn_minatar.py \
    --capture-video \
    --total-timesteps 5_000_000 \
    --buffer-size 50_000 \
    --start-e 1 \
    --end-e 0.1 \
    --exploration-fraction 0.02 \
    --learning-starts 20_000 \
    --train-frequency 1 \
    --feature-rank-n-states 2000 \
    --return-window-size 100 \
    --compute-final-feature-rank"

uv run python scripts/submit.py \
    --env-ids "MinAtar/Asterix-v1" \
    --seeds 1 \
    --command "$COMMAND" \
    --gres "gpu:1" \
    --time "7-00:00:00" \
    "$@"
