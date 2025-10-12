"""
Unified Alignment Tester - Validation Ruleset Configuration

This module defines validation rules for different SageMaker step types,
controlling which validation levels are applied and how they behave.
"""

from typing import Dict, List, Set, Optional
from enum import Enum
from dataclasses import dataclass

class ValidationLevel(Enum):
    """Validation levels in the alignment tester."""
    SCRIPT_CONTRACT = 1      # Level 1: Script ↔ Contract
    CONTRACT_SPEC = 2        # Level 2: Contract ↔ Specification  
    SPEC_DEPENDENCY = 3      # Level 3: Specification ↔ Dependencies (Universal)
    BUILDER_CONFIG = 4       # Level 4: Builder ↔ Configuration

class StepTypeCategory(Enum):
    """Categories of step types based on validation requirements."""
    SCRIPT_BASED = "script_based"           # Full 4-level validation
    CONTRACT_BASED = "contract_based"       # Skip Level 1, need 2-4
    NON_SCRIPT = "non_script"              # Skip Levels 1-2, need 3-4
    CONFIG_ONLY = "config_only"            # Only Level 4 needed
    EXCLUDED = "excluded"                   # No validation needed

@dataclass
class ValidationRuleset:
    """Validation ruleset for a specific step type."""
    step_type: str
    category: StepTypeCategory
    enabled_levels: Set[ValidationLevel]
    level_4_validator_class: Optional[str] = None  # Step-type-specific Level 4 validator
    skip_reason: Optional[str] = None              # Reason for skipping levels
    examples: List[str] = None                     # Example step names

# ============================================================================
# VALIDATION RULESET CONFIGURATION
# ============================================================================

VALIDATION_RULESETS: Dict[str, ValidationRuleset] = {
    
    # ========================================================================
    # SCRIPT-BASED STEPS (Full 4-Level Validation)
    # ========================================================================
    "Processing": ValidationRuleset(
        step_type="Processing",
        category=StepTypeCategory.SCRIPT_BASED,
        enabled_levels={
            ValidationLevel.SCRIPT_CONTRACT,
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="ProcessingStepBuilderValidator",
        examples=[
            "TabularPreprocessing",
            "StratifiedSampling",
            "CurrencyConversion", 
            "RiskTableMapping",
            "MissingValueImputation",
            "ModelCalibration",
            "DummyTraining",
            "XGBoostModelEval",
            "XGBoostModelInference",
            "ModelCalibration",
            "ModelMetricsComputation",
            "ModelWikiGenerator",
            "Package",
            "Payload"
        ]
    ),
    
    "Training": ValidationRuleset(
        step_type="Training",
        category=StepTypeCategory.SCRIPT_BASED,
        enabled_levels={
            ValidationLevel.SCRIPT_CONTRACT,
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="TrainingStepBuilderValidator",
        examples=[
            "XGBoostTraining",
            "PyTorchTraining"
        ]
    ),
    
    # ========================================================================
    # CONTRACT-BASED STEPS (Skip Level 1, Need Levels 2-4)
    # ========================================================================
    "CradleDataLoading": ValidationRuleset(
        step_type="CradleDataLoading",
        category=StepTypeCategory.CONTRACT_BASED,
        enabled_levels={
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="ProcessingStepBuilderValidator",  # Uses processing validator
        skip_reason="No script in cursus/steps/scripts",
        examples=["CradleDataLoading"]
    ),
    
    "MimsModelRegistrationProcessing": ValidationRuleset(
        step_type="MimsModelRegistrationProcessing",
        category=StepTypeCategory.CONTRACT_BASED,
        enabled_levels={
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="ProcessingStepBuilderValidator",  # Uses processing validator
        skip_reason="No script in cursus/steps/scripts",
        examples=["Registration"]
    ),
    
    # ========================================================================
    # NON-SCRIPT STEPS (Skip Levels 1-2, Focus on 3-4)
    # ========================================================================
    "CreateModel": ValidationRuleset(
        step_type="CreateModel",
        category=StepTypeCategory.NON_SCRIPT,
        enabled_levels={
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="CreateModelStepBuilderValidator",
        skip_reason="No script or contract - SageMaker model creation",
        examples=["XGBoostModel", "PyTorchModel"]
    ),
    
    "Transform": ValidationRuleset(
        step_type="Transform",
        category=StepTypeCategory.NON_SCRIPT,
        enabled_levels={
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="TransformStepBuilderValidator",
        skip_reason="Uses existing model - no custom script",
        examples=["BatchTransform"]
    ),
    
    "RegisterModel": ValidationRuleset(
        step_type="RegisterModel",
        category=StepTypeCategory.NON_SCRIPT,
        enabled_levels={
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="RegisterModelStepBuilderValidator",
        skip_reason="SageMaker service operation - no custom code",
        examples=["Registration"]
    ),
    
    # ========================================================================
    # CONFIGURATION-ONLY STEPS (Only Level 4 Needed)
    # ========================================================================
    "Lambda": ValidationRuleset(
        step_type="Lambda",
        category=StepTypeCategory.CONFIG_ONLY,
        enabled_levels={
            ValidationLevel.SPEC_DEPENDENCY,  # Universal Level 3 requirement
            ValidationLevel.BUILDER_CONFIG
        },
        level_4_validator_class="LambdaStepBuilderValidator",
        skip_reason="Lambda function - different execution model, but still needs dependency validation",
        examples=["HyperparameterPrep"]
    ),
    
    # ========================================================================
    # EXCLUDED STEPS (No Validation Needed)
    # ========================================================================
    "Base": ValidationRuleset(
        step_type="Base",
        category=StepTypeCategory.EXCLUDED,
        enabled_levels=set(),  # No validation levels
        skip_reason="Base configurations - no builder to validate",
        examples=["Base"]
    ),
    
    "Utility": ValidationRuleset(
        step_type="Utility",
        category=StepTypeCategory.EXCLUDED,
        enabled_levels=set(),  # No validation levels
        skip_reason="Special case - doesn't create SageMaker steps directly",
        examples=["HyperparameterPrep"]
    ),
}

# ============================================================================
# CONFIGURATION API
# ============================================================================

def get_validation_ruleset(sagemaker_step_type: str) -> Optional[ValidationRuleset]:
    """Get validation ruleset for a SageMaker step type."""
    return VALIDATION_RULESETS.get(sagemaker_step_type)

def is_validation_level_enabled(sagemaker_step_type: str, level: ValidationLevel) -> bool:
    """Check if a validation level is enabled for a step type."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return False
    return level in ruleset.enabled_levels

def get_enabled_validation_levels(sagemaker_step_type: str) -> Set[ValidationLevel]:
    """Get all enabled validation levels for a step type."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return set()
    return ruleset.enabled_levels

def get_level_4_validator_class(sagemaker_step_type: str) -> Optional[str]:
    """Get the Level 4 validator class for a step type."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return None
    return ruleset.level_4_validator_class

def is_step_type_excluded(sagemaker_step_type: str) -> bool:
    """Check if a step type is excluded from validation."""
    ruleset = get_validation_ruleset(sagemaker_step_type)
    if not ruleset:
        return False
    return ruleset.category == StepTypeCategory.EXCLUDED

def get_step_types_by_category(category: StepTypeCategory) -> List[str]:
    """Get all step types in a specific category."""
    return [
        step_type for step_type, ruleset in VALIDATION_RULESETS.items()
        if ruleset.category == category
    ]

def get_all_step_types() -> List[str]:
    """Get all configured step types."""
    return list(VALIDATION_RULESETS.keys())

def validate_step_type_configuration() -> List[str]:
    """Validate the configuration for consistency issues."""
    issues = []
    
    # Check that all step types have valid categories
    for step_type, ruleset in VALIDATION_RULESETS.items():
        if not isinstance(ruleset.category, StepTypeCategory):
            issues.append(f"Invalid category for {step_type}: {ruleset.category}")
        
        # Check that excluded steps have no enabled levels
        if ruleset.category == StepTypeCategory.EXCLUDED and ruleset.enabled_levels:
            issues.append(f"Excluded step {step_type} should have no enabled levels")
        
        # Check that Level 3 is enabled for non-excluded steps (universal requirement)
        if (ruleset.category != StepTypeCategory.EXCLUDED and 
            ValidationLevel.SPEC_DEPENDENCY not in ruleset.enabled_levels):
            issues.append(f"Step {step_type} missing universal Level 3 validation")
    
    return issues

# ============================================================================
# REGISTRY INTEGRATION
# ============================================================================

def get_validation_ruleset_for_step_name(step_name: str, workspace_id: str = None) -> Optional[ValidationRuleset]:
    """Get validation ruleset for a step name using registry integration."""
    try:
        from ....registry.step_names import get_sagemaker_step_type
        sagemaker_step_type = get_sagemaker_step_type(step_name, workspace_id)
        return get_validation_ruleset(sagemaker_step_type)
    except (ImportError, ValueError) as e:
        # Fallback to default Processing validation if registry lookup fails
        return get_validation_ruleset("Processing")

def is_validation_level_enabled_for_step_name(step_name: str, level: ValidationLevel, workspace_id: str = None) -> bool:
    """Check if a validation level is enabled for a step name using registry integration."""
    try:
        from ....registry.step_names import get_sagemaker_step_type
        sagemaker_step_type = get_sagemaker_step_type(step_name, workspace_id)
        return is_validation_level_enabled(sagemaker_step_type, level)
    except (ImportError, ValueError):
        # Fallback to Processing validation if registry lookup fails
        return is_validation_level_enabled("Processing", level)

def get_enabled_validation_levels_for_step_name(step_name: str, workspace_id: str = None) -> Set[ValidationLevel]:
    """Get all enabled validation levels for a step name using registry integration."""
    try:
        from ....registry.step_names import get_sagemaker_step_type
        sagemaker_step_type = get_sagemaker_step_type(step_name, workspace_id)
        return get_enabled_validation_levels(sagemaker_step_type)
    except (ImportError, ValueError):
        # Fallback to Processing validation if registry lookup fails
        return get_enabled_validation_levels("Processing")

def is_step_name_excluded(step_name: str, workspace_id: str = None) -> bool:
    """Check if a step name is excluded from validation using registry integration."""
    try:
        from ....registry.step_names import get_sagemaker_step_type
        sagemaker_step_type = get_sagemaker_step_type(step_name, workspace_id)
        return is_step_type_excluded(sagemaker_step_type)
    except (ImportError, ValueError):
        # Fallback to not excluded if registry lookup fails
        return False
