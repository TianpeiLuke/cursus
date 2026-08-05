"""GraphStormGNNTraining step entry point (Nexus ``train.py``).

The REAL training launcher is the bundled ``train.py`` (611 lines) that ships in the BYO GraphStorm
image's ``code`` channel, invoked directly via the interface's
``container_entrypoint: [python3, /opt/ml/input/data/code/train.py]`` (the ContainerEntrypoint
bypass of the SageMaker training toolkit). It:
  * discovers the partition-config JSON from the ``graph`` channel and the training YAML from the
    ``config`` channel,
  * applies HPO dot-path overrides passed as ``gsf.*`` env vars,
  * auto-tunes batch size from GPU VRAM (``nvidia-smi``) + graph metadata (``BATCH_SIZE_OVERRIDE``),
  * launches ``graphstorm.run.gs_multi_task_learning`` (TRAINING_MODE=multi_task) or
    ``graphstorm.run.gs_node_classification``,
  * writes DGL checkpoints + ``best_checkpoint.txt`` to /opt/ml/model and predictions to
    /opt/ml/output/data.

graphstorm/dgl/torch are baked into the image (``PYTHONPATH=/usr/local/lib/graphstorm/python``), so
this step vendors NO training code into cursus. This module is the cursus-declared ``entry_point``
(informational + a defensive delegator): if ever run as a plain SageMaker script (not via the
ContainerEntrypoint), it execs the bundled launcher so behavior is identical.
"""

import os
import sys

# The bundled launcher location inside the BYO image's code channel.
_BUNDLED_LAUNCHER = "/opt/ml/input/data/code/train.py"


def main():
    if os.path.exists(_BUNDLED_LAUNCHER):
        # Defensive delegation: hand off to the real bundled launcher verbatim. Normal runs go
        # through the ContainerEntrypoint and never reach here.
        os.execv(sys.executable, [sys.executable, _BUNDLED_LAUNCHER, *sys.argv[1:]])
    raise SystemExit(
        "graphstorm_gnn_training: the real launcher is the bundled train.py invoked via the "
        "interface's container_entrypoint (BYO GraphStorm image). Bundled launcher not found at "
        f"{_BUNDLED_LAUNCHER} — ensure the code channel is mounted."
    )


if __name__ == "__main__":
    main()
