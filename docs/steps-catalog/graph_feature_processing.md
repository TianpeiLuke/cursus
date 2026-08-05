# GraphFeatureProcessing

**Turn per-seed subgraph pickles + labelled seeds into the GraphStorm GConstruct input (per-type node/edge parquets, reverse edges, node-ID-keyed masks, gconstruct_config.json). BYO GraphStorm image.**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `graph_feature_processing.py` |
| **Build-time requirement** | `none` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/graph_feature_processing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `byo_container` |
| **Image** | BYO container — `config.image_uri` is passed VERBATIM to `AppSpecification.ImageUri` / `AlgorithmSpecification.TrainingImage` (no `image_uris.retrieve`; the framework deps live in the image's Dockerfile) |
| **Container entrypoint** | `['bash', '/opt/ml/processing/input/code/entrypoint.sh']` (ContainerEntrypoint bypass — the image runs its own entrypoint instead of the SageMaker toolkit) |

## Functionality

Graph feature-processing orchestrator (ports Nexus run_feature_processing.py): runs prepare_graphstorm_format.py (type-aware node/edge feature extraction, reverse-edge generation, node-ID-keyed multi-task masks, gconstruct_config.json) then custom features then sanity_check.py, all from the BYO GraphStorm image's bundled code. Output is the GraphStorm GConstruct input tree.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `SUBGRAPHS` | `processing_output` | yes | [GraphSubgraphExtraction](graph_subgraph_extraction.md) |
| `SEEDS` | `processing_output` | yes | [GraphSubgraphExtraction](graph_subgraph_extraction.md), [CradleDataLoading](cradle_data_loading.md) |

## Outputs

| Output | Type |
|--------|------|
| `graph_data` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [GraphConstruction](graph_construction.md)
- [GraphStormGNNTraining](graphstorm_gnn_training.md)
- [GraphStormGNNTuning](graphstorm_gnn_tuning.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `polars` | `>=0.20.0` |
| `pandas` | `>=1.3.0` |
| `numpy` | `>=1.21.0` |
| `pyyaml` | `>=6.0` |
| `tqdm` | `>=4.65.0` |

---

← [Back to the Step Catalog](index.md)
