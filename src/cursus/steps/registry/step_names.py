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

def get_canonical_name_from_file_name(file_name: str) -> str:
    """
    Get canonical step name from file name using registry-first algorithmic conversion.
    
    This function uses a deterministic algorithm that tries multiple interpretations
    of the file name against the registry, eliminating the need for hard-coded mappings.
    
    Args:
        file_name: File-based name (e.g., "model_evaluation_xgb", "dummy_training", "xgboost_training")
        
    Returns:
        Canonical step name (e.g., "XGBoostModelEval", "DummyTraining", "XGBoostTraining")
        
    Raises:
        ValueError: If file name cannot be mapped to a canonical name
    """
    if not file_name:
        raise ValueError("File name cannot be empty")
    
    parts = file_name.split('_')
    job_type_suffixes = ['training', 'validation', 'testing', 'calibration']
    
    # Strategy 1: Try full name as PascalCase (handles cases like "xgboost_training" -> "XGBoostTraining")
    full_pascal = ''.join(word.capitalize() for word in parts)
    if full_pascal in STEP_NAMES:
        return full_pascal
    
    # Strategy 2: Try without last part if it's a job type suffix
    if len(parts) > 1 and parts[-1] in job_type_suffixes:
        base_parts = parts[:-1]
        base_pascal = ''.join(word.capitalize() for word in base_parts)
        if base_pascal in STEP_NAMES:
            return base_pascal
    
    # Strategy 3: Handle special abbreviations and patterns
    # Convert known abbreviations to full names
    abbreviation_map = {
        'xgb': 'XGBoost',
        'xgboost': 'XGBoost',  # Add full xgboost mapping
        'pytorch': 'PyTorch',
        'mims': '',  # Remove MIMS prefix
        'tabular': 'Tabular',
        'preprocess': 'Preprocessing'
    }
    
    # Apply abbreviation expansion
    expanded_parts = []
    for part in parts:
        if part in abbreviation_map:
            expansion = abbreviation_map[part]
            if expansion:  # Only add non-empty expansions
                expanded_parts.append(expansion)
        else:
            expanded_parts.append(part.capitalize())
    
    # Try expanded version
    if expanded_parts:
        expanded_pascal = ''.join(expanded_parts)
        if expanded_pascal in STEP_NAMES:
            return expanded_pascal
        
        # Try expanded version without job type suffix
        if len(expanded_parts) > 1 and parts[-1] in job_type_suffixes:
            expanded_base = ''.join(expanded_parts[:-1])
            if expanded_base in STEP_NAMES:
                return expanded_base
    
    # Strategy 4: Handle compound names (like "model_evaluation_xgb")
    if len(parts) >= 3:
        # Try different combinations for compound names
        combinations_to_try = [
            # For "model_evaluation_xgb" -> "XGBoostModelEval"
            (parts[-1], parts[0], parts[1]),  # xgb, model, evaluation -> XGBoost, Model, Eval
            # For other patterns
            (parts[0], parts[1], parts[-1]),  # model, evaluation, xgb
        ]
        
        for combo in combinations_to_try:
            # Apply abbreviation expansion to combination
            expanded_combo = []
            for part in combo:
                if part in abbreviation_map:
                    expansion = abbreviation_map[part]
                    if expansion:
                        expanded_combo.append(expansion)
                else:
                    # Special handling for "evaluation" -> "Eval"
                    if part == 'evaluation':
                        expanded_combo.append('Eval')
                    else:
                        expanded_combo.append(part.capitalize())
            
            combo_pascal = ''.join(expanded_combo)
            if combo_pascal in STEP_NAMES:
                return combo_pascal
    
    # Strategy 5: Fuzzy matching against registry entries
    # Calculate similarity scores for all registry entries
    best_match = None
    best_score = 0.0
    
    for canonical_name in STEP_NAMES.keys():
        score = _calculate_name_similarity(file_name, canonical_name)
        if score > best_score and score >= 0.8:  # High threshold for fuzzy matching
            best_score = score
            best_match = canonical_name
    
    if best_match:
        return best_match
    
    # If all strategies fail, provide detailed error message
    tried_variations = [
        full_pascal,
        ''.join(word.capitalize() for word in parts[:-1]) if len(parts) > 1 and parts[-1] in job_type_suffixes else None,
        ''.join(expanded_parts) if expanded_parts else None
    ]
    tried_variations = [v for v in tried_variations if v]  # Remove None values
    
    raise ValueError(
        f"Cannot map file name '{file_name}' to canonical name. "
        f"Tried variations: {tried_variations}. "
        f"Available canonical names: {sorted(STEP_NAMES.keys())}"
    )

def _calculate_name_similarity(file_name: str, canonical_name: str) -> float:
    """
    Calculate similarity score between file name and canonical name.
    
    Args:
        file_name: File-based name (e.g., "xgboost_training")
        canonical_name: Canonical name (e.g., "XGBoostTraining")
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Convert both to lowercase for comparison
    file_lower = file_name.lower().replace('_', '')
    canonical_lower = canonical_name.lower()
    
    # Exact match after normalization
    if file_lower == canonical_lower:
        return 1.0
    
    # Check if file name is contained in canonical name
    if file_lower in canonical_lower:
        return 0.9
    
    # Check if canonical name contains most of the file name parts
    file_parts = file_name.lower().split('_')
    matches = sum(1 for part in file_parts if part in canonical_lower)
    
    if matches == len(file_parts):
        return 0.85
    elif matches >= len(file_parts) * 0.8:
        return 0.8
    else:
        return matches / len(file_parts) * 0.7

def validate_file_name(file_name: str) -> bool:
    """Validate that a file name can be mapped to a canonical name."""
    try:
        get_canonical_name_from_file_name(file_name)
        return True
    except ValueError:
        return False
