"""GraphConstruction step script (Nexus ``run_gconstruct.sh`` → Cursus).

Ports the Nexus ``run_gconstruct.sh`` wrapper into a contract-driven Python entry point. Runs
inside the BYO GraphStorm image (GraphStorm + DGL are baked into the image, not pip-provisioned).

Responsibilities (verbatim port of the shell wrapper):
  1. Path rewrite — GraphFeatureProcessing (Step 2) wrote ``/opt/ml/processing/output/...`` paths
     into ``gconstruct_config.json``; here those files mount at ``/opt/ml/processing/input/data/...``.
     Rewrite the config in place and chdir into the data dir so any relative paths resolve.
  2. GraphStorm bug patch — inject ``hard_edge_neg_ops = []`` after ``edges = {}`` in the container's
     ``construct_graph.py`` (an unbound-variable bug), idempotently. Prefer pinning a fixed GraphStorm
     version in the Dockerfile so this becomes unnecessary; the guard stays for image-version drift.
  3. Run ``python3 -m graphstorm.gconstruct.construct_graph`` with the fixed ``--conf-file`` /
     ``--output-dir`` (constant container paths) plus the config-sourced ``--graph-name`` /
     ``--num-processes`` / ``--num-parts`` / ``[--skip-nonexist-edges]`` — multi-process partitioning
     on a single instance. Emits a partitioned DGL heterograph.

Contract args (from ``graph_construction.step.yaml`` job_arguments): --graph-name, --num-processes,
--num-parts (+ --skip-nonexist-edges appended by the config's gconstruct_arguments when set).
"""

import argparse
import os
import subprocess
import sys

DATA_DIR = "/opt/ml/processing/input/data"
OUTPUT_DIR = "/opt/ml/processing/output"
CONF_FILE = os.path.join(DATA_DIR, "gconstruct_config.json")
CONSTRUCT_FILE = (
    "/usr/local/lib/graphstorm/python/graphstorm/gconstruct/construct_graph.py"
)


def rewrite_config_paths():
    """Rewrite the Step-2 output paths in gconstruct_config.json to the Step-3 input mount."""
    if not os.path.exists(CONF_FILE):
        raise FileNotFoundError(f"gconstruct config not found at {CONF_FILE}")
    with open(CONF_FILE) as f:
        text = f.read()
    rewritten = text.replace(
        "/opt/ml/processing/output/", "/opt/ml/processing/input/data/"
    )
    if rewritten != text:
        with open(CONF_FILE, "w") as f:
            f.write(rewritten)
        print(f">>> rewrote output→input paths in {CONF_FILE}", flush=True)
    # chdir so any remaining relative paths in the config resolve against the data dir.
    os.chdir(DATA_DIR)


def patch_graphstorm_bug():
    """Inject `hard_edge_neg_ops = []` after `edges = {}` in construct_graph.py (idempotent).

    Mirrors the sed in run_gconstruct.sh. Best-effort — a pinned GraphStorm version makes it moot,
    and it must never fail the job if the file/layout differs in a newer image.
    """
    try:
        if not os.path.exists(CONSTRUCT_FILE):
            return
        with open(CONSTRUCT_FILE) as f:
            lines = f.readlines()
        if any("hard_edge_neg_ops = []" in ln for ln in lines):
            return  # already patched
        out = []
        for ln in lines:
            out.append(ln)
            if ln.rstrip("\n") == "    edges = {}":
                out.append("    hard_edge_neg_ops = []\n")
        with open(CONSTRUCT_FILE, "w") as f:
            f.writelines(out)
        print(">>> patched hard_edge_neg_ops in construct_graph.py", flush=True)
    except Exception as e:  # noqa: BLE001 — never fail the job on the best-effort patch
        print(f"WARNING: gconstruct bug patch skipped: {e}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GraphStorm gconstruct wrapper")
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--num-processes", default="20")
    parser.add_argument("--num-parts", default="1")
    parser.add_argument("--skip-nonexist-edges", action="store_true")
    args = parser.parse_args()

    rewrite_config_paths()
    patch_graphstorm_bug()

    cmd = [
        "python3",
        "-m",
        "graphstorm.gconstruct.construct_graph",
        "--conf-file",
        CONF_FILE,
        "--output-dir",
        OUTPUT_DIR,
        "--graph-name",
        args.graph_name,
        "--num-processes",
        str(args.num_processes),
        "--num-parts",
        str(args.num_parts),
    ]
    if args.skip_nonexist_edges:
        cmd.append("--skip-nonexist-edges")

    print(f">>> {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)
    print("Graph construction complete.", flush=True)


if __name__ == "__main__":
    main()
