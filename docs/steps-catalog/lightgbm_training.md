# LightGBMTraining

**LightGBM model training step using built-in algorithm**

| | |
|---|---|
| **SageMaker step type** | `Training` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `lightgbm_training.py` |
| **Interface file** | `steps/interfaces/lightgbm_training.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `estimator` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

LightGBM training for tabular classification with risk table mapping and numerical imputation. Supports native categorical features, binary/multiclass, class weights, pre-computed artifacts, and comprehensive evaluation metrics.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_path` | `training_data` | yes | [TabularPreprocessing](tabular_preprocessing.md), [BedrockProcessing](bedrock_processing.md), [StratifiedSampling](stratified_sampling.md), [RiskTableMapping](risk_table_mapping.md), [MissingValueImputation](missing_value_imputation.md), ProcessingStep, DataLoad |
| `hyperparameters_s3_uri` | `hyperparameters` | no | HyperparameterPrep, ProcessingStep |
| `model_artifacts_input` | `processing_output` | no | [LightGBMTraining](lightgbm_training.md), [MissingValueImputation](missing_value_imputation.md), [RiskTableMapping](risk_table_mapping.md), [FeatureSelection](feature_selection.md) |

## Outputs

| Output | Type |
|--------|------|
| `model_output` | `model_artifacts` |
| `evaluation_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [DummyTraining](dummy_training.md)
- [LightGBMMTModelEval](lightgbmmt_model_eval.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMModelEval](lightgbm_model_eval.md)
- [LightGBMModelInference](lightgbm_model_inference.md)
- [LightGBMTraining](lightgbm_training.md)
- [ModelCalibration](model_calibration.md)
- [Payload](payload.md)
- [PercentileModelCalibration](percentile_model_calibration.md)
- [TSAModelCalibration](tsa_model_calibration.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `lightgbm` | `>=3.0.0` |
| `scikit-learn` | `>=0.23.2,<1.0.0` |
| `pandas` | `>=1.2.0,<2.0.0` |
| `pyarrow` | `>=4.0.0,<6.0.0` |
| `boto3` | `>=1.26.0` |
| `pydantic` | `>=2.0.0,<3.0.0` |
| `matplotlib` | `>=3.0.0` |
| `numpy` | `>=1.19.0` |

---

← [Back to the Step Catalog](index.md)
