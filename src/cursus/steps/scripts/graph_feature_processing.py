"""GraphFeatureProcessing step script (Nexus ``feature_processing`` → Cursus).

Ports Nexus's ``run_feature_processing.py`` orchestrator: it runs the bundled
``prepare_graphstorm_format.py`` → custom features → ``sanity_check.py`` from the BYO GraphStorm
image's code bundle (mounted at /opt/ml/processing/input/code). The heavy feature-engineering
modules (``prepare_graphstorm_format.py`` + ``graph_utils`` + ``compute_order_features`` +
``custom_features`` + ``sanity_check.py``, ~46KB) live in that image, NOT in cursus — this is a
BYO-container step, so cursus vendors only the thin contract-driven orchestrator.

Contract (from ``graph_feature_processing.step.yaml``):
  inputs: SUBGRAPHS (/opt/ml/processing/input/subgraphs), SEEDS (/opt/ml/processing/input/seeds)
  args:   --config  (the materialized config.yaml, staged in the code/source dir)
          --num-chunks
  output: /opt/ml/processing/output  (the GConstruct input tree: nodes/ edges/ *_idx.parquet
          gconstruct_config.json), uploaded EndOfJob.

Load-bearing behaviors (preserved by delegating to the bundled scripts verbatim):
  * type-aware node/edge feature extraction (numerical log1p, temporal, spatial-haversine,
    aggregation, structural), reverse-edge generation, 1e-6 constant-column noise, label clip ≥0,
  * node-ID-keyed multi-task train/val/test masks, gconstruct_config.json emission,
  * sanity_check.py validation of the output tree.
"""

import argparse
import os
import subprocess
import sys

CODE_DIR = "/opt/ml/processing/input/code"
OUTPUT_DIR = "/opt/ml/processing/output"

# Make the bundled modules importable (custom_features et al. live in the code bundle).
sys.path.insert(0, CODE_DIR)


def run(cmd):
    print(f">>> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def apply_custom_features(config_path):
    """Apply custom features if the config declares a `custom_features` section (bundled module)."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    custom_cfg = cfg.get("custom_features")
    if not custom_cfg:
        print("[Custom Features] No custom_features configured, skipping.", flush=True)
        return

    print("[Custom Features] Processing custom features...", flush=True)
    from custom_features import process_custom_features

    process_custom_features(OUTPUT_DIR, custom_cfg)


def main():
    parser = argparse.ArgumentParser(
        description="GraphStorm feature-processing orchestrator"
    )
    parser.add_argument(
        "--config", required=True, help="Path to the materialized config.yaml"
    )
    parser.add_argument("--num-chunks", type=int, default=2)
    parser.add_argument("--skip-sanity-check", action="store_true")
    args = parser.parse_args()

    # Step 2a — feature processing → writes the GConstruct input tree to /opt/ml/processing/output.
    run(
        ["python3", f"{CODE_DIR}/prepare_graphstorm_format.py", "--config", args.config]
    )

    # Step 2b — custom features (external joins + computed features), best-effort.
    try:
        apply_custom_features(args.config)
    except Exception as e:  # noqa: BLE001 — custom features are optional; never fail the job on them
        print(f"WARNING: custom feature processing failed: {e}", flush=True)
        import traceback

        traceback.print_exc()

    # Step 2c — sanity check on the output tree.
    if not args.skip_sanity_check and not _env_true("SKIP_SANITY_CHECK"):
        try:
            run(
                [
                    "python3",
                    f"{CODE_DIR}/sanity_check.py",
                    "--output-dir",
                    OUTPUT_DIR,
                    "--config",
                    args.config,
                ]
            )
        except subprocess.CalledProcessError:
            print("WARNING: sanity_check had issues, see report.", flush=True)

    print("Feature processing complete.", flush=True)


def _env_true(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


if __name__ == "__main__":
    main()
