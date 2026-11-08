# Usage: fish scripts/mixed_recovery.fish run-001 [--n-posterior-samples 1000]
source (path dirname (status filename))/common.fish
if test (count $argv) -lt 1
    echo "Usage: fish scripts/mixed_recovery.fish ARTIFACT_REFERENCE [evaluation options]" >&2
    exit 2
end
set -l artifact_ref $argv[1]
set -e argv[1]
model_dc_cli evaluate --artifacts "$artifact_ref" --scenario mixed-tta \
    --options-file "$MODEL_DC_WORKSPACE_ROOT/configs/mixed_recovery.yaml" \
    --n-posterior-samples 32 $argv
or exit $status
model_dc_python "$MODEL_DC_WORKSPACE_ROOT/scripts/plot_recovery.py" \
    --artifact "$artifact_ref" --scenario mixed-tta
