# Scientific contract: existing seven-parameter model_DC

## Source and scope

The source is `examples/article_workspace/plugins/reproduce_models.py`, specifically
`prior_DC` and `simulator_DC`. It is NOT the four-parameter simplest model used
by the older package-specific mixed-TTA evaluator. No CPP observation is present.

The tests compare the new prior and homogeneous trial kernel against those
unchanged source functions using identical RNG states. The full training data
stream is not claimed to be bit-identical: the wrapper owns its RNG explicitly
and organizes calls differently from the old SequentialSimulator.

| Parameter order | Prior |
| --- | --- |
| tau | Uniform(0, 1) |
| b0 | Uniform(0.1, 2) |
| k | Uniform(1, 5) |
| mu_ndt | Uniform(0.2, 0.6) |
| sigma_ndt | Uniform(0.06, 0.1) |
| alpha | Uniform(0, 1) |
| theta | Uniform(0, 1) |

One participant has one shared vector of these seven parameters.

## Trial algorithm preserved from the source

1. Select nominal TTA from 2.5, 3.0, 3.5, or 4.0 seconds.
2. Add Uniform(0, 0.1) trial jitter to obtain actual TTA.
3. Initialize evidence and time at zero.
4. Calculate `boundary = b0 / (1 + exp(-k * (actual_tta - t - tau)))` once.
5. While `abs(evidence) < boundary` and `t < actual_tta`, update
   `evidence += alpha * (actual_tta - t - theta) * dt + sqrt(dt) * Normal(0, 1)`;
   increase time by `dt=0.005` seconds.
6. Draw nondecision time from Normal(mu_ndt, sigma_ndt), rejecting negative draws.
7. Encode the response as -1 if evidence is negative or time exceeds actual TTA;
   otherwise encode reaction time as time + nondecision time.

Despite the inherited 'collapsing boundary' name, the source does NOT update the
boundary during the accumulation loop. It also uses tau and theta directly in
these expressions, rather than multiplying them by TTA. These choices are
preserved for faithful code reproduction; revising them requires an explicit
scientific change/version and new validation, not a cleanup edit.

## Shapes and adaptation

For N participants and T trials:

- Raw observations `x`: (N, T, 1), response/RT values in the source encoding.
- Raw covariates `tta_per_trial`: (N, T, 1), actual jittered TTA in seconds.
- `number_of_trials`: (N, 1), bookkeeping; not an extra network feature.
- Training parameter fields: each (N, 1).
- Evaluation truths: each (N,), kept outside posterior conditions.
- Adapter summary inputs: (N, T, 2), ordered [response/RT, actual TTA].
- Adapter inference targets: (N, 7), in the parameter order above.
- Engine-normalized draws: each parameter (N, S).

The new adapter omits the old trial-count broadcast because its output was
unused by the networks. It uses only built-in serializable BayesFlow transforms
and does not sample a prior merely to obtain the parameter names.

## Training and evaluation designs

Training reproduces the source's default 60 trials and one nominal condition per
batch, with participant-specific parameters and trial-level jitter. Homogeneous
evaluation selects one nominal condition per participant. Mixed evaluation uses
15 trials at each of four nominal conditions, optionally shuffling the complete
schedule before simulation. No condition-specific posterior draws are pooled.

Both scenarios run one posterior inference for each complete participant dataset.
The homogeneous-trained checkpoint's mixed-design recovery is a generalization
stress test. Accurate mixed-design inference is not guaranteed by concatenating
TTA to each trial; train and validate appropriate designs before real-data use.

## Source mapping

| Previous function/concept | New workspace location |
| --- | --- |
| prior_DC | cognitive_models_codes/science.py::sample_prior |
| simulator_DC | cognitive_models_codes/science.py::simulate_dataset |
| meta/default trial design | cognitive_models_codes/designs.py + simulators.py |
| get_adapter_trialwise_dc | cognitive_models_codes/adapters.py::make_adapter |
| ModelSpec registration | models_registeration.py |
| Recovery experiment generation | cognitive_models_codes/scenarios.py |
| EvaluationSpec registration | evaluations_registerations.py |

Training and numerical recovery still execute through the existing installed
CLI/package. The workspace's registered resume policy is documented in README.
