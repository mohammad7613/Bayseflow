# List registrations and saved runs without training or evaluating anything.
source (path dirname (status filename))/common.fish
model_dc_cli models list
or exit $status
model_dc_cli evaluations list
or exit $status
model_dc_cli checkpoints list
