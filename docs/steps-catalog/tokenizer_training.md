# TokenizerTraining

**BPE tokenizer training step for customer name data with automatic vocabulary size tuning**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `tokenizer_training.py` |
| **Interface file** | `steps/interfaces/tokenizer_training.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

Tokenizer training script. Trains custom BPE tokenizer optimized for customer name data with automatic vocabulary size tuning.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_data` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `model_artifacts_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [PyTorchTraining](pytorch_training.md)
- [SlipboxKnowledgeRouting](slipbox_knowledge_routing.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `pandas` | `>=1.3.0` |
| `tokenizers` | `>=0.13.0` |

---

← [Back to the Step Catalog](index.md)
