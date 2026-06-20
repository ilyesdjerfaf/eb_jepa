"""
Mesh JEPA Experiment Runner

Orchestrates the full pipeline from a single YAML config:
  preprocess → train (HKS + XYZ) → eval (HKS + XYZ)

Usage:
    uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml all
    uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml preprocess
    uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml train
    uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml eval
    uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml train --feature_type hks
    uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml --force preprocess
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

RESERVED_NAMES = ["default", "quick_test", "experiment", "test", "unnamed"]


def load_experiment_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_experiment_name(cfg, paths, force=False):
    """Check that the experiment name has been set intentionally.

    Prevents accidental overwrites when someone forgets to change the name.
    """
    name = cfg["experiment"]["name"]

    if name in RESERVED_NAMES:
        print(f"\n  ERROR: Experiment name '{name}' is a reserved/default name.")
        print(
            f'  Please set a unique name in your YAML: experiment.name: "your_name_here"'
        )
        sys.exit(1)

    # Check if outputs already exist from a previous run with same name
    if not force:
        existing = []
        # Only check processed_dir if it's auto-generated (not an explicit override)
        has_explicit_processed = "processed_dir" in cfg.get("preprocessing", {})
        if not has_explicit_processed:
            processed_dir = Path(paths["processed_dir"])
            if processed_dir.exists() and (processed_dir / "manifest.csv").exists():
                existing.append(f"  - {processed_dir}")
        for ft in cfg["experiment"]["feature_types"]:
            model_path = Path(paths["models"][ft])
            if model_path.exists():
                existing.append(f"  - {model_path}")
            eval_dir = Path(paths["eval_dirs"][ft])
            if (eval_dir / "results.npy").exists():
                existing.append(f"  - {eval_dir}")

        if existing:
            print(f"\n  WARNING: Experiment '{name}' already has outputs:")
            for e in existing:
                print(e)
            print(f"\n  Options:")
            print(
                f"    1. Change experiment.name in your YAML to start a fresh experiment"
            )
            print(f"    2. Use --force to overwrite existing outputs")
            print(f"\n  Aborting to prevent accidental overwrite.")
            sys.exit(1)


def resolve_paths(cfg):
    """Resolve all derived paths from experiment config."""
    name = cfg["experiment"]["name"]
    raw_dir = cfg["preprocessing"]["raw_data_dir"]

    # Allow explicit processed_dir override (useful for quick_test reusing existing data)
    processed_dir = cfg["preprocessing"].get(
        "processed_dir", f"datasets/dfaust/processed_{name}"
    )

    paths = {
        "name": name,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "models": {},
        "eval_dirs": {},
    }

    for ft in cfg["experiment"]["feature_types"]:
        paths["models"][ft] = f"checkpoints/mesh_jepa/{name}/{ft}/final.pth.tar"
        paths["eval_dirs"][ft] = f"results/{name}/{ft}"

    return paths


def run_cmd(cmd, description):
    """Run a command with logging."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  $ {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2])
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)

    print(f"\n  Done in {elapsed:.1f}s")
    return elapsed


def step_preprocess(cfg, paths, force=False):
    """Run preprocessing."""
    out_dir = Path(paths["processed_dir"])
    if out_dir.exists() and not force:
        manifest = out_dir / "manifest.csv"
        if manifest.exists():
            print(f"\n  Preprocessing already done: {out_dir}")
            print(f"  (use --force to rerun)")
            return 0.0

    prep = cfg["preprocessing"]
    cmd = [
        sys.executable,
        "-m",
        "examples.mesh_jepa.preprocess",
        "--data_dir",
        prep["raw_data_dir"],
        "--out_dir",
        paths["processed_dir"],
        "--n_eigen",
        str(prep["n_eigen"]),
        "--n_hks",
        str(prep["n_hks"]),
        "--temporal_stride",
        str(prep["temporal_stride"]),
        "--actions",
        *prep["actions"],
    ]

    return run_cmd(cmd, f"PREPROCESSING → {paths['processed_dir']}")


def step_train(cfg, paths, feature_type, force=False):
    """Run training for a specific feature type."""
    model_path = Path(paths["models"][feature_type])
    if model_path.exists() and not force:
        print(f"\n  Model already exists: {model_path}")
        print(f"  (use --force to retrain)")
        return 0.0

    # Check preprocessing done
    processed_dir = Path(paths["processed_dir"])
    if not (processed_dir / "manifest.csv").exists():
        print(f"\n  ERROR: Preprocessed data not found at {processed_dir}")
        print(f"  Run 'preprocess' step first.")
        sys.exit(1)

    t = cfg["training"]
    exp = cfg["experiment"]

    # Build the model output dir
    model_dir = model_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)

    # Write a temporary training config
    train_cfg = {
        "meta": {"seed": exp["seed"], "device": exp["device"]},
        "data": {
            "data_dir": paths["processed_dir"],
            "feature_type": feature_type,
            "seq_len": t["seq_len"],
            "batch_size": t["batch_size"],
            "num_workers": t["num_workers"],
            "train_subjects": t["train_subjects"],
            "test_subjects": t["test_subjects"],
        },
        "model": {
            "width": t["width"],
            "depth": t["depth"],
            "henc": t["henc"],
            "hpre": t["hpre"],
            "n_eigen": t["n_eigen"],
            "steps": t["steps"],
            "predictor_layers": t["predictor_layers"],
            "dropout": t.get("dropout", True),
            "grad_clip": t.get("grad_clip", None),
        },
        "loss": {
            "std_coeff": t["std_coeff"],
            "cov_coeff": t["cov_coeff"],
            "proj_spec": t["proj_spec"],
        },
        "optim": {
            "epochs": t["epochs"],
            "lr": t["lr"],
            "weight_decay": t["weight_decay"],
        },
        "logging": {
            "log_wandb": t["log_wandb"],
            "log_every": t["log_every"],
            "save_every": t["save_every"],
        },
    }

    # Write temp config
    tmp_cfg_path = model_dir / "train_config.yaml"
    with open(tmp_cfg_path, "w") as f:
        yaml.dump(train_cfg, f, default_flow_style=False)

    cmd = [
        sys.executable,
        "-m",
        "examples.mesh_jepa.main",
        "--fname",
        str(tmp_cfg_path),
        "--folder",
        str(model_dir),
    ]

    return run_cmd(cmd, f"TRAINING [{feature_type.upper()}] → {model_dir}")


def step_eval(cfg, paths, feature_type, force=False):
    """Run evaluation for a specific feature type."""
    eval_dir = Path(paths["eval_dirs"][feature_type])
    if eval_dir.exists() and (eval_dir / "results.npy").exists() and not force:
        print(f"\n  Evaluation already done: {eval_dir}")
        print(f"  (use --force to rerun)")
        return 0.0

    model_path = Path(paths["models"][feature_type])
    if not model_path.exists():
        print(f"\n  ERROR: Model not found at {model_path}")
        print(f"  Run 'train' step first.")
        sys.exit(1)

    ev = cfg["eval"]
    exp = cfg["experiment"]

    cmd = [
        sys.executable,
        "-m",
        "examples.mesh_jepa.eval",
        "--model_path",
        str(model_path),
        "--data_dir",
        paths["processed_dir"],
        "--output_dir",
        str(eval_dir),
        "--batch_size",
        str(ev["batch_size"]),
        "--num_workers",
        str(ev["num_workers"]),
        "--device",
        exp["device"],
    ]

    return run_cmd(cmd, f"EVALUATION [{feature_type.upper()}] → {eval_dir}")


def step_summary(cfg, paths):
    """Print final summary of all results."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT SUMMARY: {paths['name']}")
    print(f"{'='*60}")

    print(f"\n  Preprocessed data: {paths['processed_dir']}")

    for ft in cfg["experiment"]["feature_types"]:
        print(f"\n  --- {ft.upper()} ---")
        print(f"  Model:  {paths['models'][ft]}")
        print(f"  Eval:   {paths['eval_dirs'][ft]}")

        results_path = Path(paths["eval_dirs"][ft]) / "results.npy"
        if results_path.exists():
            results = dict(np.load(results_path, allow_pickle=True).item())
            print(
                f"  Probe Accuracy (test): {results.get('encoder_probe_test_acc', 'N/A'):.1%}"
            )
            print(
                f"  Effective Rank:        {results.get('effective_rank', 'N/A'):.1f}"
            )
            if "timing" in results:
                print(
                    f"  Inference Latency:     {results['timing'].get('total_inference_time_ms', 'N/A'):.1f} ms"
                )

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Mesh JEPA Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Targets:
  preprocess  Run preprocessing (shared for all feature types)
  train       Train models for all feature types
  eval        Evaluate all trained models
  all         Run full pipeline: preprocess → train → eval
  summary     Print results summary

Examples:
  uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml all
  uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml train --feature_type hks
  uv run python -m examples.mesh_jepa.run_experiment --config experiments/default.yaml --force preprocess
        """,
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment YAML"
    )
    parser.add_argument(
        "target",
        choices=["preprocess", "train", "eval", "all", "summary"],
        help="Which step to run",
    )
    parser.add_argument(
        "--feature_type",
        type=str,
        default=None,
        help="Run only for this feature type (default: all in config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs exist",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Include preprocessing when running 'all' (skipped by default if data exists)",
    )
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg)

    # Validate experiment name (skip for summary)
    if args.target != "summary":
        validate_experiment_name(cfg, paths, force=args.force)

    feature_types = (
        [args.feature_type] if args.feature_type else cfg["experiment"]["feature_types"]
    )

    timings = {}

    if args.target == "preprocess" or (args.target == "all" and args.preprocess):
        timings["preprocess"] = step_preprocess(cfg, paths, force=args.force)
    elif args.target == "all":
        # Check preprocessed data exists when skipping preprocessing
        processed_dir = Path(paths["processed_dir"])
        if not (processed_dir / "manifest.csv").exists():
            print(f"\n  ERROR: Preprocessed data not found at {processed_dir}")
            print(f"  Either:")
            print(
                f"    - Run with --preprocess flag: run_experiment.py --config ... all --preprocess"
            )
            print(
                f"    - Or run preprocessing separately: run_experiment.py --config ... preprocess"
            )
            sys.exit(1)

    if args.target in ("train", "all"):
        for ft in feature_types:
            timings[f"train_{ft}"] = step_train(cfg, paths, ft, force=args.force)

    if args.target in ("eval", "all"):
        for ft in feature_types:
            timings[f"eval_{ft}"] = step_eval(cfg, paths, ft, force=args.force)

    if args.target in ("summary", "all"):
        step_summary(cfg, paths)

    # Print timing summary
    if timings:
        print(f"\n  Timing:")
        for step, t in timings.items():
            print(f"    {step}: {t:.1f}s")
        print(f"    TOTAL: {sum(timings.values()):.1f}s")


if __name__ == "__main__":
    main()
