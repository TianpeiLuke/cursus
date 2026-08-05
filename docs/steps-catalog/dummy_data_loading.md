# DummyDataLoading

**Dummy data loading step that processes user-provided data instead of calling Cradle services**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `dummy_data_loading.py` |
| **Interface file** | `steps/interfaces/dummy_data_loading.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `sklearn` |

## Functionality

Dummy data loading script. Drop-in replacement for CradleDataLoading that processes user-provided data instead of calling Cradle services.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `INPUT_DATA` | `processing_output` | yes | DataUploadStep, S3DataStep, LocalDataStep |

## Outputs

| Output | Type |
|--------|------|
| `DATA` | `processing_output` |
| `METADATA` | `processing_output` |
| `SIGNATURE` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [GraphStormGNNTraining](graphstorm_gnn_training.md)
- [GraphSubgraphExtraction](graph_subgraph_extraction.md)
- [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md)
- [TSAPreprocessing](tsa_preprocessing.md)
- [TSATabularPreprocessing](tsa_tabular_preprocessing.md)
- [TabularPreprocessing](tabular_preprocessing.md)
- [TemporalSequenceNormalization](temporal_sequence_normalization.md)
- [TemporalSplitPreprocessing](temporal_split_preprocessing.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `pyarrow` | `>=4.0.0` |

---

← [Back to the Step Catalog](index.md)
