# PyTorchTraining

**PyTorch model training step**

| | |
|---|---|
| **SageMaker step type** | `Training` |
| **Node type** | internal (consumes upstream, produces downstream) |
| **Container entry point** | `pytorch_training.py` |
| **Interface file** | `steps/interfaces/pytorch_training.step.yaml` |

## Compute

| | |
|---|---|
| **Compute kind** | `estimator` |
| **SDK class** | `PyTorch` (SageMaker DLC via `image_uris.retrieve`) |

## Functionality

PyTorch Lightning training for multimodal (text+tabular) models. Supports BERT, CNN, LSTM, multimodal variants. Handles binary/multiclass classification with early stopping, checkpointing, ONNX export, and streaming mode for memory-efficient loading.

## Inputs (dependencies)

| Input | Type | Required | Compatible producers |
|-------|------|----------|----------------------|
| `input_path` | `training_data` | yes | [TabularPreprocessing](tabular_preprocessing.md), [BedrockProcessing](bedrock_processing.md), [StratifiedSampling](stratified_sampling.md), [RiskTableMapping](risk_table_mapping.md), [MissingValueImputation](missing_value_imputation.md), [LabelRulesetExecution](label_ruleset_execution.md), ProcessingStep, DataLoad |
| `hyperparameters_s3_uri` | `hyperparameters` | no | HyperparameterPrep, ProcessingStep |
| `model_artifacts_input` | `processing_output` | no | [PyTorchTraining](pytorch_training.md), [TokenizerTraining](tokenizer_training.md), [MissingValueImputation](missing_value_imputation.md), [RiskTableMapping](risk_table_mapping.md), [FeatureSelection](feature_selection.md) |

## Outputs

| Output | Type |
|--------|------|
| `model_output` | `model_artifacts` |
| `evaluation_output` | `processing_output` |

## Consumers (downstream steps)

Steps that declare this step as a compatible input source:

- [DummyTraining](dummy_training.md)
- [LightGBMMTModelInference](lightgbmmt_model_inference.md)
- [LightGBMModelEval](lightgbm_model_eval.md)
- [LightGBMModelInference](lightgbm_model_inference.md)
- [ModelCalibration](model_calibration.md)
- [Package](package.md)
- [Payload](payload.md)
- [PercentileModelCalibration](percentile_model_calibration.md)
- [PyTorchModel](pytorch_model.md)
- [PyTorchModelEval](pytorch_model_eval.md)
- [PyTorchModelInference](pytorch_model_inference.md)
- [PyTorchTraining](pytorch_training.md)
- [TSAModelCalibration](tsa_model_calibration.md)
- [TSAModelEval](tsa_model_eval.md)
- [XGBoostModelEval](xgboost_model_eval.md)
- [XGBoostModelInference](xgboost_model_inference.md)

## Framework requirements

| Package | Version |
|---------|---------|
| `torch` | `==2.1.0` |
| `transformers` | `==4.37.2` |
| `lightning` | `==2.1.3` |
| `torchmetrics` | `==1.7.1` |
| `scikit-learn` | `==1.3.2` |
| `pandas` | `==2.1.4` |
| `pyarrow` | `==14.0.2` |
| `pydantic` | `==2.11.2` |
| `onnx` | `==1.15.0` |
| `onnxruntime` | `==1.17.0` |

---

← [Back to the Step Catalog](index.md)
