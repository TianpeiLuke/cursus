# GraphSubgraphExtraction

**Point-in-time k-hop subgraph pull from a property-graph DB for seed order IDs (BYO GraphStorm image, VPC-bound).**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | source (no inputs — originates data) |
| **Container entry point** | `graph_subgraph_extraction.py` |
| **Build-time requirement** | `none` (SAIS SDK — fatal on load if absent) |
| **Interface file** | `steps/interfaces/graph_subgraph_extraction.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `byo_container` |
| **Image** | BYO container — `config.image_uri` is passed VERBATIM to `AppSpecification.ImageUri` / `AlgorithmSpecification.TrainingImage` (no `image_uris.retrieve`; the framework deps live in the image's Dockerfile) |
| **Container entrypoint** | `['bash', '/opt/ml/processing/input/code/entrypoint.sh']` (ContainerEntrypoint bypass — the image runs its own entrypoint instead of the SageMaker toolkit) |
| **Network** | `network_mode: config` — per-step VPC: attaches the step's own `subnets` / `security_group_ids` (overrides the session-wide default), for reaching a VPC-only data source |

## Functionality

Graph subgraph extraction script that opens an authenticated property-graph session pool and runs a point-in-time k-hop traversal per seed order ID, writing one pickled subgraph per seed directly to S3. Runs the custom GraphStorm image via a ContainerEntrypoint bypass; VPC-bound to reach the graph cluster.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `seeds` | `processing_output` | no | [StratifiedSampling](stratified_sampling.md), [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), ProcessingStep |
| `code` | `processing_output` | no | [DummyDataLoading](dummy_data_loading.md), ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `subgraphs` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [GraphFeatureProcessing](graph_feature_processing.md)
- [GraphStormGNNInferenceEval](graphstorm_gnn_inference_eval.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `nebula3-python` | `>=3.4` |
| `polars` | `>=0.20` |
| `pandas` | `>=1.5` |
| `boto3` | `>=1.28` |
| `tqdm` | `>=4.65` |

---

← [Back to the Step Catalog](index.md)
