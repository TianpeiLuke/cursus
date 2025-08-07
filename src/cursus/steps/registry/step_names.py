"""
Central registry for all pipeline step names.
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
        "description": "Base pipeline configuration"
    },

    # Processing Steps (keep Processing as-is)
    "Processing": {
        "config_class": "ProcessingStepConfigBase",
        "builder_step_name": "ProcessingStepBuilder",
        "spec_type": "Processing",
        "sagemaker_step_type": "Processing",
        "description": "Base processing step"
    },

    # Data Loading Steps
    "CradleDataLoading": {
        "config_class": "CradleDataLoadConfig",
        "builder_step_name": "CradleDataLoadingStepBuilder",
        "spec_type": "CradleDataLoading",
        "sagemaker_step_type": "CradleDataLoading",
        "description": "Cradle data loading step"
    },

    # Processing Steps
    "TabularPreprocessing": {
        "config_class": "TabularPreprocessingConfig",
        "builder_step_name": "TabularPreprocessingStepBuilder",
        "spec_type": "TabularPreprocessing",
        "sagemaker_step_type": "Processing",
        "description": "Tabular data preprocessing step"
    },
    "RiskTableMapping": {
        "config_class": "RiskTableMappingConfig",
        "builder_step_name": "RiskTableMappingStepBuilder",
        "spec_type": "RiskTableMapping",
        "sagemaker_step_type": "Processing",
        "description": "Risk table mapping step for categorical features"
    },
    "CurrencyConversion": {
        "config_class": "CurrencyConversionConfig",
        "builder_step_name": "CurrencyConversionStepBuilder",
        "spec_type": "CurrencyConversion",
        "sagemaker_step_type": "Processing",
        "description": "Currency conversion processing step"
    },
    
    # Training Steps
    "PyTorchTraining": {
        "config_class": "PyTorchTrainingConfig",
        "builder_step_name": "PyTorchTrainingStepBuilder",
        "spec_type": "PyTorchTraining",
        "sagemaker_step_type": "Training",
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig",
        "builder_step_name": "XGBoostTrainingStepBuilder",
        "spec_type": "XGBoostTraining",
        "sagemaker_step_type": "Training",
        "description": "XGBoost model training step"
    },
    "DummyTraining": {
        "config_class": "DummyTrainingConfig",
        "builder_step_name": "DummyTrainingStepBuilder",
        "spec_type": "DummyTraining",
        "sagemaker_step_type": "Processing",
        "description": "Training step that uses a pretrained model"
    },
    
    # Evaluation Steps
    "XGBoostModelEval": {
        "config_class": "XGBoostModelEvalConfig",
        "builder_step_name": "XGBoostModelEvalStepBuilder",
        "spec_type": "XGBoostModelEval",
        "sagemaker_step_type": "Processing",
        "description": "XGBoost model evaluation step"
    },
    
    # Model Steps
    "PyTorchModel": {
        "config_class": "PyTorchModelConfig",
        "builder_step_name": "PyTorchModelStepBuilder",
        "spec_type": "PyTorchModel",
        "sagemaker_step_type": "CreateModel",
        "description": "PyTorch model creation step"
    },
    "XGBoostModel": {
        "config_class": "XGBoostModelConfig",
        "builder_step_name": "XGBoostModelStepBuilder",
        "spec_type": "XGBoostModel",
        "sagemaker_step_type": "CreateModel",
        "description": "XGBoost model creation step"
    },
    
    # Model Processing Steps
    "ModelCalibration": {
        "config_class": "ModelCalibrationConfig",
        "builder_step_name": "ModelCalibrationStepBuilder",
        "spec_type": "ModelCalibration",
        "sagemaker_step_type": "Processing",
        "description": "Calibrates model prediction scores to accurate probabilities"
    },
    
    # Deployment Steps
    "Package": {
        "config_class": "PackageConfig",
        "builder_step_name": "PackageStepBuilder",
        "spec_type": "Package",
        "sagemaker_step_type": "Processing",
        "description": "Model packaging step"
    },
    "Registration": {
        "config_class": "RegistrationConfig",
        "builder_step_name": "RegistrationStepBuilder",
        "spec_type": "Registration",
        "sagemaker_step_type": "MimsModelRegistrationProcessing",
        "description": "Model registration step"
    },
    "Payload": {
        "config_class": "PayloadConfig",
        "builder_step_name": "PayloadStepBuilder",
        "spec_type": "Payload",
        "sagemaker_step_type": "Processing",
        "description": "Payload testing step"
    },
    
    # Utility Steps
    "HyperparameterPrep": {
        "config_class": "HyperparameterPrepConfig",
        "builder_step_name": "HyperparameterPrepStepBuilder",
        "spec_type": "HyperparameterPrep",
        "sagemaker_step_type": "Lambda",  # Special classification
        "description": "Hyperparameter preparation step"
    },
    
    # Transform Steps
    "BatchTransform": {
        "config_class": "BatchTransformStepConfig",
        "builder_step_name": "BatchTransformStepBuilder",
        "spec_type": "BatchTransform",
        "sagemaker_step_type": "Transform",
        "description": "Batch transform step"
    }
}

# Generate the mappings that existing code expects
CONFIG_STEP_REGISTRY = {
    info["config_class"]: step_name 
    for step_name, info in STEP_NAMES.items()
}

BUILDER_STEP_NAMES = {
    step_name: info["builder_step_name"]
    for step_name, info in STEP_NAMES.items()
}

# Generate step specification types
SPEC_STEP_TYPES = {
    step_name: info["spec_type"]
    for step_name, info in STEP_NAMES.items()
}

# Helper functions
def get_config_class_name(step_name: str) -> str:
    """Get config class name for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["config_class"]

def get_builder_step_name(step_name: str) -> str:
    """Get builder step class name for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["builder_step_name"]

def get_spec_step_type(step_name: str) -> str:
    """Get step_type value for StepSpecification."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["spec_type"]

def get_spec_step_type_with_job_type(step_name: str, job_type: str = None) -> str:
    """Get step_type with optional job_type suffix."""
    base_type = get_spec_step_type(step_name)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type

def get_step_name_from_spec_type(spec_type: str) -> str:
    """Get canonical step name from spec_type."""
    # Handle job type variants (e.g., "TabularPreprocessing_Training" -> "TabularPreprocessing")
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    
    reverse_mapping = {info["spec_type"]: step_name 
                      for step_name, info in STEP_NAMES.items()}
    return reverse_mapping.get(base_spec_type, spec_type)

def get_all_step_names() -> List[str]:
    """Get all canonical step names."""
    return list(STEP_NAMES.keys())

# Validation functions
def validate_step_name(step_name: str) -> bool:
    """Validate that a step name exists in the registry."""
    return step_name in STEP_NAMES

def validate_spec_type(spec_type: str) -> bool:
    """Validate that a spec_type exists in the registry."""
    # Handle job type variants
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    return base_spec_type in [info["spec_type"] for info in STEP_NAMES.values()]

def get_step_description(step_name: str) -> str:
    """Get description for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["description"]

def list_all_step_info() -> Dict[str, Dict[str, str]]:
    """Get complete step information for all registered steps."""
    return STEP_NAMES.copy()

# SageMaker Step Type Classification Functions
def get_sagemaker_step_type(step_name: str) -> str:
    """Get SageMaker step type for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["sagemaker_step_type"]

def get_steps_by_sagemaker_type(sagemaker_type: str) -> List[str]:
    """Get all step names that create a specific SageMaker step type."""
    return [
        step_name for step_name, info in STEP_NAMES.items()
        if info["sagemaker_step_type"] == sagemaker_type
    ]

def get_all_sagemaker_step_types() -> List[str]:
    """Get all unique SageMaker step types."""
    return list(set(info["sagemaker_step_type"] for info in STEP_NAMES.values()))

def validate_sagemaker_step_type(sagemaker_type: str) -> bool:
    """Validate that a SageMaker step type exists in the registry."""
    valid_types = {"Processing", "Training", "Transform", "CreateModel", "RegisterModel", "Base", "Utility"}
    return sagemaker_type in valid_types

def get_sagemaker_step_type_mapping() -> Dict[str, List[str]]:
    """Get mapping of SageMaker step types to step names."""
    mapping = {}
    for step_name, info in STEP_NAMES.items():
        sagemaker_type = info["sagemaker_step_type"]
        if sagemaker_type not in mapping:
            mapping[sagemaker_type] = []
        mapping[sagemaker_type].append(step_name)
    return mapping
