# Usage: fish scripts/resume.fish run-001 [--epochs 2 ...]
# --mode new preserves the original artifact and records its parent relationship.
# Pass --mode old explicitly if you intend to update the selected artifact.
source (path dirname (status filename))/common.fish
if test (count $argv) -lt 1
    echo "Usage: fish scripts/resume.fish ARTIFACT_REFERENCE [training options]" >&2
    exit 2
end
set -l artifact_ref $argv[1]
set -e argv[1]
set -gx MODEL_DC_SIMULATION_SEED 2024
set -gx MODEL_DC_NETWORK_SEED 2024

model_dc_cli resume --artifacts "$artifact_ref" --mode new \
    --batches-per-epoch 2 --batch-size 4 --epochs 1 $argv
or exit $status
model_dc_cli checkpoints list
