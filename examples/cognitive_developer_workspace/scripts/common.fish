# Shared paths and runtime settings. Source this file; run the other scripts.
# Absolute paths keep outputs here even if you invoke a script from elsewhere.
set -l script_dir (path dirname (status filename))
set -g MODEL_DC_WORKSPACE_ROOT (path resolve "$script_dir/..")

set -gx KERAS_BACKEND torch
set -gx MPLBACKEND Agg
set -gx MPLCONFIGDIR "$MODEL_DC_WORKSPACE_ROOT/.cache/matplotlib"
set -gx KERAS_HOME "$MODEL_DC_WORKSPACE_ROOT/.cache/keras"
set -gx PYTHONDONTWRITEBYTECODE 1
set -gx PYTHONHASHSEED 2023
# This example deliberately tests CPU. Hiding CUDA also keeps lazily created
# tensors on CPU with the current package's Torch/Keras defaults.
set -gx CUDA_VISIBLE_DEVICES -1

function model_dc_cli
    # Reuse the parent project's existing uv environment; install nothing here.
    command uv --directory "$MODEL_DC_WORKSPACE_ROOT" --offline --no-cache run --no-sync bayesflow-cli \
        --plugin-path "$MODEL_DC_WORKSPACE_ROOT" \
        --device cpu \
        --checkpoint-dir "$MODEL_DC_WORKSPACE_ROOT/checkpoints" \
        --results-dir "$MODEL_DC_WORKSPACE_ROOT/results" \
        --logs-dir "$MODEL_DC_WORKSPACE_ROOT/logs" \
        $argv
end

function model_dc_python
    command uv --directory "$MODEL_DC_WORKSPACE_ROOT" --offline --no-cache run --no-sync python $argv
end
