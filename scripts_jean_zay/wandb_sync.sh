#!/bin/bash
set -euo pipefail

source .venv/bin/activate

SYNCED=0
FAILED=0
LOG_FILE="wandb_sync.log"

echo "Starting W&B sync at $(date)" | tee -a "$LOG_FILE"

for run_dir in wandb/offline-run-*; do
    run_name=$(basename "$run_dir")
    echo "Syncing $run_name..." | tee -a "$LOG_FILE"
    
    if wandb sync "$run_dir" \
        --include-synced \
        --include-offline \
        --mark-synced \
        --skip-console \
        >> "$LOG_FILE" 2>&1; then
        ((SYNCED++))
        echo "  Success" | tee -a "$LOG_FILE"
    else
        ((FAILED++))
        echo "  Failed" | tee -a "$LOG_FILE"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "Completed: $SYNCED synced, $FAILED failed" | tee -a "$LOG_FILE"
echo "Log saved to $LOG_FILE"
