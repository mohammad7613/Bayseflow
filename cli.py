import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", message="When using torch backend")

from torch import device,cuda
import copy
from bayesflow_models.discovery import discover_model_specs
import bayesflow_models.workflow as wf
from typing import Union, Any

from bayesflow_models.discovery import discover_plugins

import yaml

from collections.abc import Mapping

from bayesflow_models.interfaces import (
    ModelSpec,
    Workflow,
)


from bayesflow_models.evaluation.interfaces import EvaluationSpec







DEFAULT_CONFIG = {
    "device": device('cuda' if cuda.is_available() else 'cpu'),
    "training": {
        "n_sim": 100,          # Simulations per epoch
        "epochs": 10,          # Total epochs to train
        "batch_size": 32,
        "resume_epochs": 3,    # Additional epochs when resuming
    },
    "recovery": {
        "n_test_sims": 50,    # Simulations for recovery evaluation
        "n_posterior_samples": 50,  # MCMC samples for posterior
        "mode": "visualize" # visualize | save_only
    },
    "paths": {
        "checkpoints": "trained_model1/checkpoints",
        "results": "results",
        "real_data": "real_data",
        "logs": "logs"
    }
}

def setup_directories(config: dict):
    """Ensure all required directories exist"""
    assert config["paths"] != None
    for key, path in config["paths"].items():
        Path(path).mkdir(parents=True, exist_ok=True)
    print("✓ Directories setup complete")


def ensure_directories(config: dict, path_keys: tuple[str, ...],) -> None:
    paths = config.get("paths")

    if not isinstance(paths, dict):
        raise ValueError(
            "Runtime configuration must contain "
            "a paths dictionary."
        )

    for key in path_keys:
        if key not in paths:
            raise ValueError(
                f"Missing configured path: {key}"
            )

        Path(paths[key]).mkdir(
            parents=True,
            exist_ok=True,
        )


def build_runtime_config(args):
    config = copy.deepcopy(DEFAULT_CONFIG)

    cwd = Path.cwd()

    config["paths"]["checkpoints"] = str(cwd / "checkpoints")
    config["paths"]["results"] = str(cwd / "results")
    config["paths"]["logs"] = str(cwd / "logs")

    if args.device is not None:
        config["device"] = device(args.device)

    if args.checkpoint_dir is not None:
        config["paths"]["checkpoints"] = args.checkpoint_dir
    if args.results_dir is not None:
        config["paths"]["results"] = args.results_dir
    if args.logs_dir is not None:
        config["paths"]["logs"] = args.logs_dir

    if args.command == "train":
        config["training"]["n_sim"] = args.n_sim
        config["training"]["epochs"] = args.epochs
        config["training"]["batch_size"] = args.batch_size

    if args.command == "resume":
        config["training"]["resume_epochs"] = args.epochs
        config["training"]["batch_size"] = args.batch_size
        config["training"]["n_sim"] = args.n_sim

    if args.command in {"recovery", "mixed-recovery"}:
        config["recovery"]["n_test_sims"] = args.n_test_sims
        config["recovery"]["n_posterior_samples"] = args.n_posterior_samples
        if hasattr(args, "mode"):
            config["recovery"]["mode"] = args.mode
        if hasattr(args, "trials_per_tta"):
            config["recovery"]["trials_per_tta"] = args.trials_per_tta
        if hasattr(args, "shuffle_tta"):
            config["recovery"]["shuffle_tta"] = args.shuffle_tta
        if hasattr(args, "seed"):
            config["recovery"]["seed"] = args.seed
    
    return config



## Helper function for terminal-based of visualization of data stored in format of list of dictionary
def get_value(obj: Union[dict,object], key: str, default=None)-> Any:
    """Unified getter for dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def display_header(headers: list[str], header_widths: list):
    header = "  ".join(h.ljust(w) for h, w in zip(headers, header_widths))
    print(header)

def dict_to_list(dict: dict)-> list:
    a = []
    for k,v in dict.items():
        a.append(v)
    return a

def extract_header_widths(headers: list[str],records: list[Union[dict,object]], keys:list[str])-> list[int]:
    header_widths = [max(len(header), max(len(get_value(r,keys[i])) for r in records)) for i,header in enumerate(headers)]
    return header_widths
def display_rows(records: list[Union[dict,object]],header_widths: list[int], keys:list[str]):
    for row in records:
        text = "  ".join(get_value(row,k).ljust(w) for k, w in zip(keys, header_widths))
        print(text)


def display_table(headers: list[str], records: list[dict], keys: list[str]):
    header_widths = extract_header_widths(headers=headers,records=records,keys=keys)
    # header_widths = [max(len(header), max(len(r[keys[i]]) for r in records)) for i,header in enumerate(headers)]
    display_header(headers=headers,header_widths= header_widths)
    display_rows(records=records,header_widths=header_widths, keys=keys)


def print_checkpoints_table(records: list[dict]) -> None:
      if not records:
          print("No checkpoints found.")
          return

      headers = ["ALIAS", "SPEC NAME", "STATUS", "CREATED","RESUME"]
      keys = ["alias","spec_name","status","created_display","resumed_display"]
      display_table(headers=headers,records=records,keys=keys)

def print_models_table(specs: dict) -> None:
      if not specs:
          print(
                "No model plugins found. "
                "Mount plugins in the working directory or pass --plugin-path DIRECTORY."
            )
          return

      headers = ["SPEC NAME","FAMILY","DESCRIPTION"]
      keys = ["name","family","description"]
      records = dict_to_list(specs)
      display_table(headers=headers, records=records, keys=keys)



def handle_checkpoints_list(config):
      records = wf.list_checkpoint_records(config)
      print_checkpoints_table(records)

def handle_models_list(specs):
    print_models_table(specs=specs)




def build_parser():
    parser = argparse.ArgumentParser(prog="bayesflow-cli")

    parser.add_argument("--device", "-d", default=device('cpu'), help="Select the device of the torch")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument(
    "--plugin-path",
    action="append",
    default=[],
    metavar="DIRECTORY",
    help=(
        "Directory containing trusted ModelSpec or Workflow plugins. "
        "May be passed multiple times. Defaults to the current directory."
    ),)

    # parser.add_argument("--config", type=str,help="You can input your config assuming that you import your workflow plugin")

    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="Model registry commands")
    models_sub = models_parser.add_subparsers(dest="models_command", required=True)
    models_sub.add_parser("list", help="List available model specs")

    checkpoints_parser = subparsers.add_parser("checkpoints", help="it allow monitoring or processing checkpoints files")
    checkpoints_sub = checkpoints_parser.add_subparsers(dest="checkpoints_command",required=True)
    checkpoints_sub.add_parser("list", help="show all information of the save checkpoint files")


    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--models", nargs="+", required=True)
    train_parser.add_argument("--batches-per-epoch","--n-sim", dest= "n_sim",type=int, default=100)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch-size", type=int, default=32)

    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("--artifacts", nargs="+", required=True)
    resume_parser.add_argument("--batches-per-epoch","--n-sim", dest="n_sim", type=int, default=100)
    resume_parser.add_argument("--epochs", type=int, default=3)
    resume_parser.add_argument("--batch-size", type=int, default=32)
    resume_parser.add_argument("--mode",choices=["new","old"],default="old")


    evaluations_parser = subparsers.add_parser(
    "evaluations",
    help="Inspect workspace evaluation scenarios.",)

    evaluations_sub = evaluations_parser.add_subparsers(dest="evaluations_command",required=True,)
    evaluations_sub.add_parser("list",help="List discovered model/scenario pairs.")

    evaluate_parser = subparsers.add_parser("evaluate",help="Evaluate saved artifacts using a workspace scenario.",)
    
    evaluate_parser.add_argument("--artifacts",nargs="+",required=True,help="Checkpoint artifact IDs or aliases.",)
    evaluate_parser.add_argument("--scenario",required=True,help="Name of a discovered EvaluationSpec.",)
    evaluate_parser.add_argument("--options-file",type=Path,help="YAML mapping of scenario-specific options.",)
    evaluate_parser.add_argument("--n-posterior-samples",type=int,default=1000,help="Posterior draws per evaluation dataset.",)
    evaluate_parser.add_argument("--mode", choices=["save_only", "visualize"], default="save_only")
    
    
    recovery_parser = subparsers.add_parser("recovery")
    recovery_parser.add_argument("--artifacts", nargs="+", required=True)
    recovery_parser.add_argument("--n-test-sims", type=int, default=50)
    recovery_parser.add_argument("--n-posterior-samples", type=int, default=50)
    recovery_parser.add_argument("--mode", choices=["save_only", "visualize"], default="save_only")
    



    mixed_recovery_parser = subparsers.add_parser(
        "mixed-recovery",
        help="Stress-test a trained artifact on synthetic mixed-TTA trial sequences.",
    )
    mixed_recovery_parser.add_argument("--artifacts", nargs="+", required=True)
    mixed_recovery_parser.add_argument("--n-test-sims", type=int, default=100)
    mixed_recovery_parser.add_argument("--n-posterior-samples", type=int, default=1000)
    mixed_recovery_parser.add_argument("--trials-per-tta", type=int, default=15)
    mixed_recovery_parser.add_argument("--no-shuffle-tta", dest="shuffle_tta", action="store_false")
    mixed_recovery_parser.set_defaults(shuffle_tta=True)
    mixed_recovery_parser.add_argument("--seed", type=int, default=2026)
    mixed_recovery_parser.add_argument("--mode", choices=["save_only", "visualize"], default="save_only")

    return parser



def require_model(
    specs: Mapping[str, ModelSpec],
    model_name: str,
) -> ModelSpec:
    try:
        return specs[model_name]
    except KeyError:
        available = ", ".join(sorted(specs)) or "none"

        raise ValueError(
            f"Unknown model {model_name!r}. "
            f"Available models: {available}. "
            "Check the mounted workspace or --plugin-path."
        ) from None


def require_evaluation(
    evaluations: Mapping[tuple[str, str], EvaluationSpec],
    *,
    model_name: str,
    scenario_name: str,
) -> EvaluationSpec:
    """Select a scenario registered for this checkpoint's model."""
    key = (model_name, scenario_name)

    try:
        return evaluations[key]
    except KeyError:
        available = ", ".join(
            sorted(
                scenario
                for model, scenario in evaluations
                if model == model_name
            )
        ) or "none"

        raise ValueError(
            f"Unknown evaluation {scenario_name!r} "
            f"for model {model_name!r}. "
            f"Available evaluations: {available}. "
            "Check --plugin-path and the workspace registrations."
        ) from None



def load_evaluation_options(path: Path | None) -> dict[str, Any]:
    """Read configuration as data; the scenario validates its own settings."""
    if path is None:
        return {}

    with path.open("r", encoding="utf-8") as stream:
        options = yaml.safe_load(stream)

    if not isinstance(options, dict):
        raise ValueError(
            "The evaluation options file must contain a YAML mapping."
        )

    if not all(isinstance(key, str) for key in options):
        raise ValueError("Evaluation option names must be strings.")

    return options



def require_workflow(
      workflows: Mapping[str, Workflow],
      workflow_name: str,
      *,
      model_name: str | None = None,
  ) -> Workflow:
      try:
          return workflows[workflow_name]
      except KeyError:
          available = (
              ", ".join(sorted(workflows))
              or "none"
          )

          model_context = (
              f" required by model {model_name!r}"
              if model_name is not None
              else ""
          )

          raise ValueError(
              f"Unknown workflow {workflow_name!r}"
              f"{model_context}. "
              f"Available workflows: {available}."
          ) from None




def main():
    parser = build_parser()
    args = parser.parse_args()
    ## New order 
    if (args.command == "models" and args.models_command == "list"):
        registry = discover_plugins( plugin_paths=args.plugin_path)

        handle_models_list(specs=registry.models)
        return

    config = build_runtime_config(args)

    if (args.command == "checkpoints" and args.checkpoints_command == "list"):
        handle_checkpoints_list(config)
        return

    registry = discover_plugins(
        plugin_paths=args.plugin_path
    )

    specs = registry.models
    workflows = registry.workflows
    evaluations = registry.evaluations

    if args.command == "evaluations":
        if not registry.evaluations:
            print("No evaluation scenarios found. Check --plugin-path.")
            return

        for key, evaluation_spec in sorted(evaluations.items()):
            model_name, scenario_name = key
            print(
                f"{model_name}\t{scenario_name}\t"
                f"{evaluation_spec.description}"
            )
        return





      
    ## Ensure directory should be checked if it is necessary for the command inside each brand. 
    if args.command == "train":
        ensure_directories(config, path_keys=("checkpoints", "results", "logs"))
        for spec_name in args.models:

            spec = require_model(specs, spec_name)
            workflow = require_workflow(workflows, spec.workflow,model_name=spec.name)
            workflow.train_fn(spec,config)
      
    if args.command == "resume":
        ensure_directories(config, path_keys=("checkpoints", "results", "logs"))
        for ref in args.artifacts:
            metadata = wf.resolve_artifact_ref(ref, config)
            # spec_name = metadata["spec_name"]

            spec = require_model(specs,metadata["spec_name"])
            workflow = require_workflow(workflows,spec.workflow,model_name=spec.name)
            config_added_alia = copy.deepcopy(config)
            config_added_alia["metadata"] = metadata
            config_added_alia["mode"] = args.mode
            workflow = workflows[spec.workflow]
            workflow.resume_fn(spec, config_added_alia)  


    if args.command == "recovery":
        ensure_directories(
            config,
            path_keys=("checkpoints", "results", "logs"),
        )

        for ref in args.artifacts:
            metadata = wf.resolve_artifact_ref(ref, config)

            spec = require_model(specs, metadata["spec_name"])
            workflow = require_workflow(
                workflows,
                spec.workflow,
                model_name=spec.name,
            )

            recovery_config = copy.deepcopy(config)
            recovery_config["metadata"] = metadata
            recovery_config["mode"] = args.mode

            workflow.recovery_fn(spec, recovery_config)

    if args.command == "mixed-recovery":
        ensure_directories(
            config,
            path_keys=("checkpoints", "results", "logs"),
        )
        from examples.cognitive_developer_workspace.cognitive_models_codes.mixed_tta_evaluation import evaluate_mixed_tta_artifact_with_bf_recovery

        for ref in args.artifacts:
            metadata = wf.resolve_artifact_ref(ref, config)

            spec = require_model(specs,metadata["spec_name"])

            config["mode"] = args.mode
            report = evaluate_mixed_tta_artifact_with_bf_recovery(
                artifact_ref=ref,
                spec=spec,
                config=config,
                n_test_sims=args.n_test_sims,
                n_posterior_samples=args.n_posterior_samples,
                trials_per_tta=args.trials_per_tta,
                shuffle_tta=args.shuffle_tta,
                seed=args.seed,
            )
            print(f"Mixed-TTA recovery for {report['artifact_id']}")

    if args.command == "evaluate":
        # Validate the common execution option before loading a model.
        if args.n_posterior_samples < 1:
            parser.error("--n-posterior-samples must be at least 1.")

        try:
            options = load_evaluation_options(args.options_file)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            parser.error(str(exc))

        # This common engine setting is controlled by the CLI flag.
        # It overrides any identically named value in the YAML file.
        options["n_posterior_samples"] = args.n_posterior_samples

        # Resolve all selections before starting potentially expensive work.
        selected = []
        try:
            for artifact_ref in args.artifacts:
                metadata = wf.resolve_artifact_ref(artifact_ref, config)

                model_spec = require_model(
                    registry.models,
                    metadata["spec_name"],
                )
                evaluation_spec = require_evaluation(
                    registry.evaluations,
                    model_name=model_spec.name,
                    scenario_name=args.scenario,
                )
                selected.append(
                    (artifact_ref, model_spec, evaluation_spec)
                )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

        # Import when executing evaluation, rather than for command parsing.
        from bayesflow_models.evaluation.fixed_engine import run_evaluation

        for artifact_ref, model_spec, evaluation_spec in selected:
            if args.mode == "visualize":
               visualize = True
            else:
                visualize = False   
            
            report = run_evaluation(
                artifact_ref=artifact_ref,
                model_spec=model_spec,
                evaluation_spec=evaluation_spec,
                config=copy.deepcopy(config),
                options=copy.deepcopy(options),
                visualize=visualize
                
            )
            print(
                f"Evaluation {evaluation_spec.name!r} completed "
                f"for {report['artifact_id']}."
            )

        return
           

   

if __name__ == "__main__":
    main()
   
