"""
Original Central registry for all pipeline step names - BACKUP.
Single source of truth for step naming across config, builders, and specifications.
"""

from typing import Dict, List

# Core step name registry - canonical names used throughout the system
STEP_NAMES = {
    "Base": {
        "config_class": "BasePipelineConfig",
        "builder_step_name": "StepBuilderBase",
        "spec_type": "Base",
        "sagemaker_step_type": "Base",  # Special case
        "description": "Base pipeline configuration",
    },
    # Processing Steps (keep Processing as-is)
    "Processing": {
        "config_class": "ProcessingStepConfigBase",
        "builder_step_name": "ProcessingStepBuilder",
        "spec_type": "Processing",
        "sagemaker_step_type": "Processing",
        "description": "Base processing step",
    },
    # Data Loading Steps
    "CradleDataLoading": {
        "config_class": "CradleDataLoadingConfig",
        "builder_step_name": "CradleDataLoadingStepBuilder",
        "spec_type": "CradleDataLoading",
        "sagemaker_step_type": "CradleDataLoading",
        "description": "Cradle data loading step",
    },
    "DummyDataLoading": {
        "config_class": "DummyDataLoadingConfig",
        "builder_step_name": "DummyDataLoadingStepBuilder",
        "spec_type": "DummyDataLoading",
        "sagemaker_step_type": "Processing",
        "description": "Dummy data loading step that processes user-provided data instead of calling Cradle services",
    },
    # Processing Steps
    "TabularPreprocessing": {
        "config_class": "TabularPreprocessingConfig",
        "builder_step_name": "TabularPreprocessingStepBuilder",
        "spec_type": "TabularPreprocessing",
        "sagemaker_step_type": "Processing",
        "description": "Tabular data preprocessing step",
    },
    "StratifiedSampling": {
        "config_class": "StratifiedSamplingConfig",
        "builder_step_name": "StratifiedSamplingStepBuilder",
        "spec_type": "StratifiedSampling",
        "sagemaker_step_type": "Processing",
        "description": "Stratified sampling step with multiple allocation strategies for class imbalance, causal analysis, and variance optimization",
    },
    "RiskTableMapping": {
        "config_class": "RiskTableMappingConfig",
        "builder_step_name": "RiskTableMappingStepBuilder",
        "spec_type": "RiskTableMapping",
        "sagemaker_step_type": "Processing",
        "description": "Risk table mapping step for categorical features",
    },
    "MissingValueImputation": {
        "config_class": "MissingValueImputationConfig",
        "builder_step_name": "MissingValueImputationStepBuilder",
        "spec_type": "MissingValueImputation",
        "sagemaker_step_type": "Processing",
        "description": "Missing value imputation step using statistical methods (mean, median, mode, constant) with pandas-safe values",
    },
    "CurrencyConversion": {
        "config_class": "CurrencyConversionConfig",
        "builder_step_name": "CurrencyConversionStepBuilder",
        "spec_type": "CurrencyConversion",
        "sagemaker_step_type": "Processing",
        "description": "Currency conversion processing step",
    },
    # Training Steps
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",
        "builder_step_name": "PyTorchTrainingStepBuilder",
        "spec_type": "PyTorchTraining",
        "sagemaker_step_type": "Training",
        "description": "PyTorch model training step",
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost model training step",
    },
    "LightGBMTraining": {
        "config_class": "LightGBMTrainingConfig",
        "builder_step_name": "LightGBMTrainingStepBuilder",
        "spec_type": "LightGBMTraining",
        "sagemaker_step_type": "Training",
        "description": "LightGBM model training step using built-in algorithm",
    },
    "DummyTraining": {
        "config_class": "DummyTrainingConfig",
        "builder_step_name": "DummyTrainingStepBuilder",
        "spec_type": "DummyTraining",
        "sagemaker_step_type": "Processing",
        "description": "Training step that uses a pretrained model",
    },
    # Evaluation Steps
    "XGBoostModelEval": {
        "config_class": "XGBoostModelEvalConfig",
        "builder_step_name": "XGBoostModelEvalStepBuilder",
        "spec_type": "XGBoostModelEval",
        "sagemaker_step_type": "Processing",
        "description": "XGBoost model evaluation step",
    },
    "XGBoostModelInference": {
        "config_class": "XGBoostModelInferenceConfig",
        "builder_step_name": "XGBoostModelInferenceStepBuilder",
        "spec_type": "XGBoostModelInference",
        "sagemaker_step_type": "Processing",
        "description": "XGBoost model inference step for prediction generation without metrics",
    },
    "ModelMetricsComputation": {
        "config_class": "ModelMetricsComputationConfig",
        "builder_step_name": "ModelMetricsComputationStepBuilder",
        "spec_type": "ModelMetricsComputation",
        "sagemaker_step_type": "Processing",
        "description": "Model metrics computation step for comprehensive performance evaluation",
    },
    "ModelWikiGenerator": {
        "config_class": "ModelWikiGeneratorConfig",
        "builder_step_name": "ModelWikiGeneratorStepBuilder",
        "spec_type": "ModelWikiGenerator",
        "sagemaker_step_type": "Processing",
        "description": "Model wiki generator step for automated documentation creation",
    },
    # Model Steps
    "PyTorchModel": {
        "config_class": "PyTorchModelConfig",
        "builder_step_name": "PyTorchModelStepBuilder",
        "spec_type": "PyTorchModel",
        "sagemaker_step_type": "CreateModel",
        "description": "PyTorch model creation step",
    },
    "XGBoostModel": {
        "config_class": "XGBoostModelConfig",
        "builder_step_name": "XGBoostModelStepBuilder",
        "spec_type": "XGBoostModel",
        "sagemaker_step_type": "CreateModel",
        "description": "XGBoost model creation step",
    },
    # Model Processing Steps
    "ModelCalibration": {
        "config_class": "ModelCalibrationConfig",
        "builder_step_name": "ModelCalibrationStepBuilder",
        "spec_type": "ModelCalibration",
        "sagemaker_step_type": "Processing",
        "description": "Calibrates model prediction scores to accurate probabilities",
    },
    # Deployment Steps
    "Package": {
        "config_class": "PackageConfig",
        "builder_step_name": "PackageStepBuilder",
        "spec_type": "Package",
        "sagemaker_step_type": "Processing",
        "description": "Model packaging step",
    },
    "Registration": {
        "config_class": "RegistrationConfig",
        "builder_step_name": "RegistrationStepBuilder",
        "spec_type": "Registration",
        "sagemaker_step_type": "MimsModelRegistrationProcessing",
        "description": "Model registration step",
    },
    "Payload": {
        "config_class": "PayloadConfig",
        "builder_step_name": "PayloadStepBuilder",
        "spec_type": "Payload",
        "sagemaker_step_type": "Processing",
        "description": "Payload testing step",
    },
    # Utility Steps
    "HyperparameterPrep": {
        "config_class": "HyperparameterPrepConfig",
        "builder_step_name": "HyperparameterPrepStepBuilder",
        "spec_type": "HyperparameterPrep",
        "sagemaker_step_type": "Lambda",  # Special classification
        "description": "Hyperparameter preparation step",
    },
    # Transform Steps
    "BatchTransform": {
        "config_class": "BatchTransformStepConfig",
        "builder_step_name": "BatchTransformStepBuilder",
        "spec_type": "BatchTransform",
        "sagemaker_step_type": "Transform",
        "description": "Batch transform step",
    },
}

# Generate the mappings that existing code expects
CONFIG_STEP_REGISTRY = {
    info["config_class"]: step_name for step_name, info in STEP_NAMES.items()
}

BUILDER_STEP_NAMES = {
    step_name: info["builder_step_name"] for step_name, info in STEP_NAMES.items()
}

# Generate step specification types
SPEC_STEP_TYPES = {
    step_name: info["spec_type"] for step_name, info in STEP_NAMES.items()
}
