# TSATraining

**TSA (Temporal Self-Attention) model training step for PyTorch-based temporal attention models**

| | |
|---|---|
| **SageMaker step type** | `Training` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `tsa_training.py` |
| **Interface file** | `steps/interfaces/tsa_training.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `estimator` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

TSA (Temporal Self-Attention) training script for AFN Return Kickout model that:
    1. Loads pre-processed training data from TSA preprocessing output (4 numpy arrays)
    2. Builds temporal attention-based neural network model (OrderFeatureAttentionClassifier)
    3. Supports distributed training with PyTorch DDP (DistributedDataParallel)
    4. Trains model with configurable hyperparameters including focal loss support
    5. Implements OneCycleLR learning rate scheduling
    6. Saves training checkpoints periodically
    7. Generates training loss plots for monitoring
    8. Saves trained model with all artifacts following standard pattern (model.tar.gz)
    9. Supports region-specific hyperparameters (NA, EU, FE) via REGION environment variable

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_path` | `training_data` | yes | [TSAPreprocessing](tsa_preprocessing.md), [TemporalSequenceNormalization](temporal_sequence_normalization.md), [TemporalFeatureEngineering](temporal_feature_engineering.md), [TabularPreprocessing](tabular_preprocessing.md), ProcessingStep, DataLoad |
| `hyperparameters_s3_uri` | `hyperparameters` | no | HyperparameterPrep, ProcessingStep |

## Outputs

| Output | Type |
|--------|------|
| `model_output` | `model_artifacts` |
| `evaluation_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [TSAModelCalibration](tsa_model_calibration.md)
- [TSAModelEval](tsa_model_eval.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `torch` | `>=1.10.0` |
| `numpy` | `>=1.19.0` |
| `pandas` | `>=1.2.0` |
| `matplotlib` | `>=3.0.0` |
| `pydantic` | `>=2.0.0,<3.0.0` |

---

← [Back to the Step Catalog](index.md)
