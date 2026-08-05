# SOPAInstructionTuning

**SOPA Stage 2 instruction fine-tuning step for BLIP2-based model (Q-Former + Phi-3 LLM) with tabular-to-text instruction following**

| | |
|---|---|
| **SageMaker step type** | `Training` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `SOPA_instruction_tuning.py` |
| **Interface file** | `steps/interfaces/sopa_instruction_tuning.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `estimator` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

SOPA Stage 2 instruction fine-tuning script for AFN Return MDR model that fine-tunes a BLIP2-based model (Q-Former + Phi-3 LLM) for tabular-to-text instruction following. Loads a pre-trained Phi-3 LLM (frozen), a Stage 0 tabular autoencoder (frozen encoder), and a Stage 1 Q-Former (trainable), then trains a projection layer (llm_proj) and the Q-Former to align tabular embeddings with the LLM input space. Supports three tasks (return_risk, customer_risk, refund_decision) and saves best-only and final model checkpoints (stage2_{task}_best.pth, stage2_{task}_final.pth). All configuration is provided via argparse arguments; no environment variables are required.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `data_path` | `training_data` | no | [TabularPreprocessing](tabular_preprocessing.md), ProcessingStep, DataLoad |
| `llm_model_path` | `model_artifacts` | no | ProcessingStep, DataLoad |
| `tabular_encoder_path` | `model_artifacts` | no | ProcessingStep, DataLoad |
| `stage1_qformer_path` | `model_artifacts` | no | ProcessingStep, DataLoad |

## Outputs

| Output | Type |
|--------|------|
| `model_output` | `model_artifacts` |
| `checkpoints_output` | `processing_output` |

## Consumers (downstream steps)

_No cataloged step lists this step as a compatible source (it may be a terminal/sink step, or consumed via a generic source name)._

## Framework requirements

| Package | Version |
|---------|---------|
| `torch` | `>=1.10.0` |
| `pandas` | `>=1.3.0` |
| `transformers` | `>=4.28.0` |
| `packaging` | `>=20.0` |

---

← [Back to the Step Catalog](index.md)
