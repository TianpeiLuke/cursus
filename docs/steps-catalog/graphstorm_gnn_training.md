# GraphStormGNNTraining

**GraphStorm/DGL R-GCN GNN training on a partitioned heterograph, run in a bring-your-own GraphStorm ECR container.**

| | |
|---|---|
| **SageMaker step type** | `Training` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `graphstorm_gnn_training.py` |
| **Build-time requirement** | `none` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/graphstorm_gnn_training.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `byo_container` |
| **Image** | BYO container — `config.training_image_uri` is passed VERBATIM to `AppSpecification.ImageUri` / `AlgorithmSpecification.TrainingImage` (no `image_uris.retrieve`; the framework deps live in the image's Dockerfile) |
| **Container entrypoint** | `['python3', '/opt/ml/input/data/code/train.py']` (ContainerEntrypoint bypass — the image runs its own entrypoint instead of the SageMaker toolkit) |

## Functionality

GraphStorm R-GCN node-classification / multi-task training over a partitioned DGL heterograph. Discovers the partition-config JSON and training YAML from the graph/config channels, applies HPO dot-path overrides, auto-tunes batch size from GPU VRAM + graph metadata, then launches graphstorm.run.gs_multi_task_learning (or gs_node_classification). The real entry is the bundled train.py via ContainerEntrypoint; graphstorm/dgl/torch are baked into the BYO image.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `graph_data` | `processing_output` | yes | [GraphConstruction](graph_construction.md) |
| `training_config` | `hyperparameters` | yes | [GraphFeatureProcessing](graph_feature_processing.md), [GraphConstruction](graph_construction.md) |
| `code` | `processing_output` | no | [DummyDataLoading](dummy_data_loading.md), ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `model_output` | `model_artifacts` |
| `prediction_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [GraphStormGNNInferenceEval](graphstorm_gnn_inference_eval.md)

---

← [Back to the Step Catalog](index.md)
