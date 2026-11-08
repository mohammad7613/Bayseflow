# Usage: fish scripts/train.fish [--epochs 20 --batches-per-epoch 100 ...]
# Defaults are an execution smoke test, NOT a scientifically trained posterior.
source (path dirname (status filename))/common.fish
set -gx MODEL_DC_SIMULATION_SEED 2023
set -gx MODEL_DC_NETWORK_SEED 2023

model_dc_cli train --models model_DC --batches-per-epoch 2 --batch-size 4 --epochs 1 $argv
or exit $status
model_dc_cli checkpoints list
