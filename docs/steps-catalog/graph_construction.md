# GraphConstruction

**GraphStorm gconstruct — build a partitioned DGL heterograph from the node/edge parquets + gconstruct schema emitted by GraphFeatureProcessing. BYO GraphStorm image.**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `run_gconstruct.py` |
| **Build-time requirement** | `none` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/graph_construction.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `byo_container` |
| **Image** | BYO container — `config.image_uri` is passed VERBATIM to `AppSpecification.ImageUri` / `AlgorithmSpecification.TrainingImage` (no `image_uris.retrieve`; the framework deps live in the image's Dockerfile) |
| **Container entrypoint** | `['bash', '/opt/ml/processing/input/code/entrypoint.sh']` (ContainerEntrypoint bypass — the image runs its own entrypoint instead of the SageMaker toolkit) |

## Functionality

GraphStorm gconstruct wrapper (ports Nexus run_gconstruct.sh): rewrites the Step-2 output paths in gconstruct_config.json to the Step-3 input mount, patches a known GraphStorm gconstruct unbound-variable bug (idempotent guard), runs python3 -m graphstorm.gconstruct.construct_graph with multi-process partitioning, then an optional graph-integrity sanity check. Emits a partitioned DGL heterograph (not parquet).

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `graph_data` | `processing_output` | yes | [GraphFeatureProcessing](graph_feature_processing.md) |

## Outputs

| Output | Type |
|--------|------|
| `dgl_graph` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [GraphStormGNNTraining](graphstorm_gnn_training.md)
- [GraphStormGNNTuning](graphstorm_gnn_tuning.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `graphstorm` | `>=0.3` |
| `dgl` | `>=1.1` |
| `pyarrow` | `>=12.0.0` |
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `pyyaml` | `>=6.0` |
| `tqdm` | `>=4.0` |

---

← [Back to the Step Catalog](index.md)
