"""
Simple validation utilities for step definition standardization.

This module provides lightweight validation for new step creation following
the simplified approach from the redundancy analysis. It focuses on essential
validation without over-engineering.

Based on: Hybrid Registry Standardization Enforcement Implementation Plan
Redundancy Target: 15-20% (vs 30-35% in original design)
Implementation Size: ~50-100 lines (vs 1,200+ in original design)
"""

import re
from typing import Dict, List, Any, Optional


# Essential validation patterns (simplified)
PASCAL_CASE_PATTERN = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
VALID_SAGEMAKER_TYPES = {
    'Processing', 'Training', 'Transform', 'CreateModel', 'RegisterModel', 
    'Base', 'Utility', 'Lambda', 'CradleDataLoading', 'MimsModelRegistrationProcessing'
}


def validate_new_step_definition(step_data: Dict[str, Any]) -> List[str]:
    """
    Validate new step definition with essential checks only.
    
    This function provides the core validation logic identified as essential
    in the redundancy analysis, focusing on preventing naming violations
    during new step creation.
    
    Args:
        step_data: Dictionary containing step definition data
        
    Returns:
        List of error messages (empty if validation passes)
    """
    errors = []
    
    # Validate step name (PascalCase)
    name = step_data.get('name', '')
    if not name:
        errors.append("Step name is required")
    elif not PASCAL_CASE_PATTERN.match(name):
        corrected = to_pascal_case(name)
        errors.append(
            f"Step name '{name}' must be PascalCase. "
            f"Example: '{corrected}' (suggested correction)"
        )
    
    # Validate config class naming
    config_class = step_data.get('config_class', '')
    if config_class and not config_class.endswith('Config'):
        corrected = f"{to_pascal_case(name)}Config"
        errors.append(
            f"Config class '{config_class}' must end with 'Config'. "
            f"Example: '{corrected}' (suggested correction)"
        )
    
    # Validate builder name - must end with 'StepBuilder' (not just 'Builder')
    builder_name = step_data.get('builder_step_name', '')
    if builder_name:
        if not builder_name.endswith('StepBuilder'):
            corrected = f"{to_pascal_case(name)}StepBuilder"
            errors.append(
                f"Builder name '{builder_name}' must end with 'StepBuilder'. "
                f"Example: '{corrected}' (suggested correction)"
            )
        else:
            # Check if the base name part is PascalCase
            base_name = builder_name.replace('StepBuilder', '')
            if base_name and not PASCAL_CASE_PATTERN.match(base_name):
                corrected = f"{to_pascal_case(name)}StepBuilder"
                errors.append(
                    f"Builder name '{builder_name}' base must be PascalCase. "
                    f"Example: '{corrected}' (suggested correction)"
                )
    
    # Validate SageMaker step type
    sagemaker_type = step_data.get('sagemaker_step_type', '')
    if sagemaker_type and sagemaker_type not in VALID_SAGEMAKER_TYPES:
        valid_types_str = ', '.join(sorted(VALID_SAGEMAKER_TYPES))
        errors.append(
            f"SageMaker step type '{sagemaker_type}' is invalid. "
            f"Valid types: {valid_types_str}"
        )
    
    return errors


def auto_correct_step_definition(step_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Auto-correct step definition with simple regex-based fixes.
    
    This function applies the simple auto-correction approach identified
    in the redundancy analysis, using regex patterns to fix common
    naming violations.
    
    Args:
        step_data: Dictionary containing step definition data
        
    Returns:
        Corrected step data dictionary
    """
    corrected_data = step_data.copy()
    
    # Auto-correct step name to PascalCase
    name = step_data.get('name', '')
    if name and not PASCAL_CASE_PATTERN.match(name):
        corrected_data['name'] = to_pascal_case(name)
    
    # Auto-correct config class name
    config_class = step_data.get('config_class', '')
    if config_class and not config_class.endswith('Config'):
        corrected_name = to_pascal_case(corrected_data.get('name', name))
        corrected_data['config_class'] = f"{corrected_name}Config"
    elif config_class and not to_pascal_case(config_class.replace('Config', '')):
        # Fix PascalCase in config class name
        base_name = config_class.replace('Config', '').replace('Configuration', '')
        corrected_base = to_pascal_case(base_name) if base_name else to_pascal_case(corrected_data.get('name', name))
        corrected_data['config_class'] = f"{corrected_base}Config"
    
    # Auto-correct builder name
    builder_name = step_data.get('builder_step_name', '')
    if builder_name and not builder_name.endswith('StepBuilder'):
        corrected_name = to_pascal_case(corrected_data.get('name', name))
        corrected_data['builder_step_name'] = f"{corrected_name}StepBuilder"
    
    return corrected_data


def to_pascal_case(text: str) -> str:
    """
    Convert text to PascalCase using simple regex patterns.
    
    This utility function provides the essential PascalCase conversion
    identified as necessary in the redundancy analysis.
    
    Args:
        text: Input text to convert
        
    Returns:
        PascalCase version of the text
    """
    if not text:
        return text
    
    # Handle snake_case, kebab-case, and space-separated words
    words = re.split(r'[_\-\s]+', text)
    
    # Handle camelCase by splitting on capital letters
    if len(words) == 1 and any(c.isupper() for c in text[1:]):
        # Split camelCase: myCustomStep -> ['my', 'Custom', 'Step']
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)', text)
    
    return ''.join(word.capitalize() for word in words if word)


def get_validation_errors_with_suggestions(step_data: Dict[str, Any]) -> List[str]:
    """
    Get validation errors with helpful suggestions and examples.
    
    This function provides the clear error messages with examples
    identified as essential for developer experience in the redundancy analysis.
    
    Args:
        step_data: Dictionary containing step definition data
        
    Returns:
        List of detailed error messages with suggestions
    """
    errors = validate_new_step_definition(step_data)
    
    if not errors:
        return []
    
    # Add helpful context to errors
    detailed_errors = []
    for error in errors:
        detailed_errors.append(f"❌ {error}")
    
    # Add general guidance
    if any("PascalCase" in error for error in errors):
        detailed_errors.append(
            "💡 PascalCase examples: 'CradleDataLoading', 'XGBoostTraining', 'PyTorchModel'"
        )
    
    if any("Config" in error for error in errors):
        detailed_errors.append(
            "💡 Config class examples: 'CradleDataLoadConfig', 'XGBoostTrainingConfig'"
        )
    
    if any("StepBuilder" in error for error in errors):
        detailed_errors.append(
            "💡 Builder name examples: 'CradleDataLoadingStepBuilder', 'XGBoostTrainingStepBuilder'"
        )
    
    return detailed_errors


def register_step_with_validation(step_name: str, step_data: Dict[str, Any], 
                                 existing_steps: Dict[str, Any], 
                                 mode: str = "warn") -> List[str]:
    """
    Register step with simple standardization validation.
    
    This function provides the minimal registry integration identified
    as essential in the redundancy analysis.
    
    Args:
        step_name: Name of the step to register
        step_data: Step definition data
        existing_steps: Dictionary of existing steps (for duplicate checking)
        mode: Validation mode ("warn", "strict", "auto_correct")
        
    Returns:
        List of warnings/messages
        
    Raises:
        ValueError: If validation fails in strict mode
    """
    warnings = []
    
    # Prepare step data for validation
    validation_data = step_data.copy()
    validation_data['name'] = step_name
    
    # Check for duplicate step names
    if step_name in existing_steps:
        duplicate_msg = f"Step '{step_name}' already exists in registry"
        if mode == "strict":
            raise ValueError(duplicate_msg)
        warnings.append(f"⚠️  {duplicate_msg}")
    
    # Validate step definition
    errors = validate_new_step_definition(validation_data)
    
    if not errors:
        return warnings
    
    # Handle validation errors based on mode
    if mode == "strict":
        error_msg = f"Step definition validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)
    
    elif mode == "auto_correct":
        # Apply auto-corrections
        corrected_data = auto_correct_step_definition(validation_data)
        
        # Re-validate corrected data
        remaining_errors = validate_new_step_definition(corrected_data)
        
        if not remaining_errors:
            warnings.append(f"✅ Auto-corrected {len(errors)} validation issues for step '{step_name}'")
            for error in errors:
                warnings.append(f"  - Fixed: {error}")
        else:
            warnings.extend([f"⚠️  Validation issue: {error}" for error in remaining_errors])
    
    else:  # "warn" mode
        warnings.extend([f"⚠️  Validation issue: {error}" for error in errors])
    
    return warnings
