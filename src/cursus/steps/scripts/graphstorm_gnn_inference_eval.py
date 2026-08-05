"""GraphStormGNNInferenceEval step entry point (Nexus ``run_evaluation.py``).

The REAL evaluator is the bundled ``run_evaluation.py`` (~486 lines) that ships in the BYO
GraphStorm image's ``code`` bundle, invoked via the interface's
``container_entrypoint: [bash, /opt/ml/processing/input/code/entrypoint.sh]``. It:
  * selects a checkpoint (auto / latest / epoch-N-iter-M) from the mounted model,
  * converts a multi-task checkpoint to single-task when needed,
  * auto-tunes GPU workers by probing live VRAM with ``nvidia-smi`` (``--auto-tune``),
  * runs the online-inference simulator over the eval subgraph S3 folder (``--subgraph-s3-uri``,
    streamed directly from S3 — NOT a mounted channel),
  * computes ROC-AUC / PR-AUC / Recall@Precision and writes ``predictions/*.parquet`` +
    ``evaluation_report.md`` + ``plots/*.png`` to /opt/ml/processing/output.

graphstorm/dgl/torch are baked into the image, so this step vendors NO evaluation code into cursus.
This module is the cursus-declared ``entry_point`` (informational + a defensive delegator): if ever
run as a plain SageMaker script rather than via the ContainerEntrypoint, it execs the bundled
evaluator so behavior is identical.

Contract args (from the interface job_arguments): --subgraph-s3-uri, --query-type, --checkpoint,
--model-type (+ --auto-tune appended by the builder when config.auto_tune). Env: ID_FIELD,
LABEL_FIELDS (required), QUERY_TYPE / MAX_WORKERS_PER_GPU / MAX_RUNTIME_SECONDS / AUTO_TUNE.
"""

import os
import sys

_BUNDLED_EVALUATOR = "/opt/ml/processing/input/code/run_evaluation.py"


def main():
    if os.path.exists(_BUNDLED_EVALUATOR):
        # Defensive delegation: hand off to the real bundled evaluator verbatim. Normal runs go
        # through the ContainerEntrypoint (entrypoint.sh) and never reach here.
        os.execv(sys.executable, [sys.executable, _BUNDLED_EVALUATOR, *sys.argv[1:]])
    raise SystemExit(
        "graphstorm_gnn_inference_eval: the real evaluator is the bundled run_evaluation.py invoked "
        "via the interface's container_entrypoint (BYO GraphStorm image). Bundled evaluator not found "
        f"at {_BUNDLED_EVALUATOR} — ensure the code bundle is mounted."
    )


if __name__ == "__main__":
    main()
