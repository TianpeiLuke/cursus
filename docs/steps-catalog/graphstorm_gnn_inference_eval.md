# GraphStormGNNInferenceEval

**GraphStorm/DGL GNN out-of-time inference + evaluation on a custom GraphStorm GPU container (online-inference simulator → ROC-AUC / PR-AUC / Recall@Precision report + plots).**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `graphstorm_gnn_inference_eval.py` |
| **Build-time requirement** | `none` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/graphstorm_gnn_inference_eval.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `byo_container` |
| **Image** | BYO container — `config.image_uri` is passed VERBATIM to `AppSpecification.ImageUri` / `AlgorithmSpecification.TrainingImage` (no `image_uris.retrieve`; the framework deps live in the image's Dockerfile) |
| **Container entrypoint** | `['bash', '/opt/ml/processing/input/code/entrypoint.sh']` (ContainerEntrypoint bypass — the image runs its own entrypoint instead of the SageMaker toolkit) |

## Functionality

GraphStorm GNN inference + out-of-time evaluation (ports Nexus run_evaluation.py): selects a checkpoint, converts multi-task→single-task if needed, auto-tunes GPU workers via live nvidia-smi VRAM probing, runs the online-inference simulator over the eval subgraph S3 folder, then computes ROC-AUC / PR-AUC / Recall@Precision and emits a report + plots. subgraph_s3_uri is a raw S3 folder streamed by the simulator (a job-arg, not a mounted channel).

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `model_input` | `model_artifacts` | yes | [GraphStormGNNTraining](graphstorm_gnn_training.md) |
| `eval_seeds` | `processing_output` | yes | [GraphSubgraphExtraction](graph_subgraph_extraction.md), [CradleDataLoading](cradle_data_loading.md) |

## Outputs

| Output | Type |
|--------|------|
| `eval_results` | `processing_output` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

## Framework requirements

| Package | Version |
|---------|---------|
| `graphstorm` | `>=0.4` |
| `dgl` | `==1.1.3` |
| `torch` | `==2.1.0` |
| `polars` | `>=0.20.0` |
| `scikit-learn` | `>=1.3.0` |
| `matplotlib` | `>=3.7.0` |
| `numpy` | `>=1.21.0` |

---

← [Back to the Step Catalog](index.md)
