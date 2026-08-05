# GraphStormGNNTuning

**GraphStorm/DGL R-GCN hyperparameter tuning — a HyperparameterTuner search over the GNN training estimator, run in a bring-your-own GraphStorm ECR container.**

| | |
|---|---|
| **SageMaker step type** | `Tuning` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `graphstorm_gnn_training.py` |
| **Build-time requirement** | `none` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/graphstorm_gnn_tuning.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `byo_container` |
| **Image** | BYO container — `config.training_image_uri` is passed VERBATIM to `AppSpecification.ImageUri` / `AlgorithmSpecification.TrainingImage` (no `image_uris.retrieve`; the framework deps live in the image's Dockerfile) |
| **Container entrypoint** | `['python3', '/opt/ml/input/data/code/train.py']` (ContainerEntrypoint bypass — the image runs its own entrypoint instead of the SageMaker toolkit) |

## Functionality

GraphStorm R-GCN hyperparameter tuning. Wraps the same GraphStorm training estimator the GraphStormGNNTraining step builds (byo_container, verbatim TrainingImage) in a SageMaker HyperparameterTuner, searching the configured search_space over the objective metric (regex- scraped from the container's stdout via metric_definitions, as no SDK-managed metrics exist for a custom image). Emits N training trials and selects the best; downstream steps read the winner via get_top_model_s3_uri / properties.BestTrainingJob. The estimator, channels, and container entrypoint are identical to GraphStormGNNTraining — only the search wrapper is added.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `graph_data` | `processing_output` | yes | [GraphConstruction](graph_construction.md) |
| `training_config` | `hyperparameters` | yes | [GraphFeatureProcessing](graph_feature_processing.md), [GraphConstruction](graph_construction.md) |
| `code` | `processing_output` | no | [DummyDataLoading](dummy_data_loading.md), ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `best_model` | `model_artifacts` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

---

← [Back to the Step Catalog](index.md)
