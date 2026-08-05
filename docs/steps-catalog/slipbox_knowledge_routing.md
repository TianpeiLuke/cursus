# SlipboxKnowledgeRouting

**Slipbox knowledge routing step that hosts the DKS knowledge+ruleset corpus and runs compile→index→route internally, emitting a compiled prompt ruleset plus per-record routed rule names and routing confidence for downstream Bedrock processing**

| | |
|---|---|
| **SageMaker step type** | `Processing` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `slipbox_knowledge_routing.py` |
| **Interface file** | `steps/interfaces/slipbox_knowledge_routing.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `framework` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

Slipbox knowledge routing script that COMPILES the DKS rule_*.md corpus into an in-memory prompt ruleset (prompts.json + tool schema), INDEXES the pattern_*/behavior_* corpus with an offline SentenceTransformer encoder into an in-memory routing index, and ROUTES each input record via cosine similarity + activation top-k to a set of routed rule names with a routing_confidence score. An internal consistency gate asserts the index linked_rules are a subset of the compiled rule_names. Emits the prompt_ruleset and the routed_records for downstream BedrockProcessing.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `records` | `processing_output` | yes | [TabularPreprocessing](tabular_preprocessing.md), [CradleDataLoading](cradle_data_loading.md), ProcessingStep |
| `knowledge_corpus` | `custom_property` | no | [DummyDataLoading](dummy_data_loading.md) |
| `embedding_model` | `model_artifacts` | no | [PyTorchModel](pytorch_model.md), [TokenizerTraining](tokenizer_training.md) |

## Outputs

| Output | Type |
|--------|------|
| `prompt_ruleset` | `processing_output` |
| `routed_records` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [BedrockBatchProcessing](bedrock_batch_processing.md)
- [BedrockProcessing](bedrock_processing.md)
- [LabelRulesetExecution](label_ruleset_execution.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `boto3` | `>=1.34` |
| `pandas` | `>=2.0` |
| `pyarrow` | `>=15.0` |
| `pyyaml` | `>=6.0` |
| `sentence-transformers` | `>=3.0` |
| `numpy` | `>=1.26` |

---

← [Back to the Step Catalog](index.md)
