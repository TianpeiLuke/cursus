# XgboostMtTraining

**XGBoost multi-task training with one_output_per_tree strategy**

| | |
|---|---|
| **SageMaker step type** | `Training` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `xgboost_mt_training.py` |
| **Interface file** | `steps/interfaces/xgboost_mt_training.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `estimator` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

XgboostMt multi-task training for multi-label tabular classification with adaptive task weighting and knowledge distillation. Supports shared tree structures, JS-divergence weight adaptation, and per-task evaluation metrics.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_path` | `training_data` | yes | [TabularPreprocessing](tabular_preprocessing.md), [StratifiedSampling](stratified_sampling.md), ProcessingStep, DataLoad, [TemporalSplitPreprocessing](temporal_split_preprocessing.md) |
| `hyperparameters_s3_uri` | `hyperparameters` | no | HyperparameterPrep, ProcessingStep |
| `model_artifacts_input` | `processing_output` | no | [XgboostMtTraining](xgboost_mt_training.md), [MissingValueImputation](missing_value_imputation.md), [RiskTableMapping](risk_table_mapping.md), [FeatureSelection](feature_selection.md) |

## Outputs

| Output | Type |
|--------|------|
| `model_output` | `model_artifacts` |
| `evaluation_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [Package](package.md)
- [Payload](payload.md)
- [XgboostMtModelEval](xgboost_mt_model_eval.md)
- [XgboostMtTraining](xgboost_mt_training.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `xgboost` | `>=2.0.0` |
| `scikit-learn` | `>=0.23.2,<1.0.0` |
| `pandas` | `>=1.2.0,<2.0.0` |
| `pyarrow` | `>=4.0.0,<6.0.0` |
| `boto3` | `>=1.26.0` |
| `pydantic` | `>=2.0.0,<3.0.0` |
| `scipy` | `>=1.7.0` |
| `numpy` | `>=1.19.0` |
| `matplotlib` | `>=3.0.0` |

---

← [Back to the Step Catalog](index.md)
