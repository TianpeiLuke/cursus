"""
Step-Type-Specific Builder Validation Rules

This module defines validation rules specific to different SageMaker step types.
Each step type has unique methods and requirements beyond the universal builder interface.

Based on analysis of actual step builders in the cursus codebase.
"""

from typing import Dict, Any, List, Optional, Set
from enum import Enum


class StepTypeCategory(Enum):
    """Categories of step types based on their validation requirements."""
    SCRIPT_BASED = "script_based"      # Full 4-level validation (Training, Processing)
    CONTRACT_BASED = "contract_based"   # Skip Level 1, need 2-4 (CradleDataLoading, etc.)
    NON_SCRIPT = "non_script"          # Skip Levels 1-2, need 3-4 (CreateModel, Transform, etc.)
    CONFIG_ONLY = "config_only"        # Only Level 4 needed (Lambda)
    EXCLUDED = "excluded"              # No validation needed (Base, Utility)


# Metadata for the validation rules
VALIDATION_RULES_METADATA = {
    "version": "1.0.0",
    "description": "Step-type-specific validation rules based on actual codebase analysis",
    "last_updated": "2025-10-02"
}

STEP_TYPE_SPECIFIC_VALIDATION_RULES = {
    # Training Steps - Full script-based validation
    "Training": {
        "sagemaker_step_class": "TrainingStep",
        "category": StepTypeCategory.SCRIPT_BASED,
        "description": "SageMaker Training steps that train ML models",
        "examples": ["XGBoostTraining", "PyTorchTraining"],
        
        "required_methods": {
            "_create_estimator": {
                "signature": "_create_estimator(self, output_path=None) -> Estimator",
                "description": "Create SageMaker Estimator instance for training job",
                "return_type": "Estimator",  # XGBoost, PyTorch, TensorFlow, etc.
                "required": True,
                "purpose": "Create the estimator that defines training job configuration",
                "implementation_notes": "All training builders implement this to create framework-specific estimators",
                "common_patterns": [
                    "Use config.effective_source_dir for source directory",
                    "Set framework version, instance type, role from config",
                    "Generate job name using _generate_job_name()",
                    "Include environment variables from _get_environment_variables()"
                ]
            },
            "_get_outputs": {
                "signature": "_get_outputs(self, outputs: Dict[str, Any]) -> str",
                "description": "Generate output path for model artifacts, used by _create_estimator",
                "return_type": "str",
                "required": True,
                "purpose": "Provide S3 path where model artifacts will be stored, passed to estimator",
                "implementation_notes": "Training steps must override this. Result is used by _create_estimator(output_path=...)",
                "common_patterns": [
                    "Generate default path using: base_output_path = self._get_base_output_path(); output_path = Join(on='/', values=[base_output_path, step_identifier])",
                    "Return single S3 path string",
                    "Used internally by _create_estimator method"
                ]
            }
        },
        
        "method_return_types": {
            "_get_inputs": "Dict[str, TrainingInput]",
            "_get_outputs": "str",  # Output path for model artifacts
            "create_step": "TrainingStep"
        },
        
        "validation_specifics": {
            "inputs_must_be_training_channels": True,
            "outputs_must_be_s3_path": True,
            "estimator_must_have_role": True,
            "supports_hyperparameters": True,
            "supports_caching": True
        },
        
        "common_input_patterns": [
            "Create TrainingInput objects for each data channel",
            "Handle train/val/test data splits",
            "Support hyperparameters_s3_uri channel (optional)",
            "Map logical names to training channels"
        ],
        
        "common_output_patterns": [
            "Return single S3 output path",
            "SageMaker automatically creates model.tar.gz and output.tar.gz",
            "Use Join() for parameter-compatible paths"
        ]
    },
    
    # Processing Steps - Full script-based validation
    "Processing": {
        "sagemaker_step_class": "ProcessingStep", 
        "category": StepTypeCategory.SCRIPT_BASED,
        "description": "SageMaker Processing steps for data processing and feature engineering",
        "examples": ["TabularPreprocessing", "FeatureEngineering"],
        
        "required_methods": {
            "_create_processor": {
                "signature": "_create_processor(self) -> Processor",
                "description": "Create SageMaker Processor instance for processing job",
                "return_type": "Processor",  # ScriptProcessor, FrameworkProcessor, etc.
                "required": True,
                "purpose": "Create the processor that defines processing job configuration",
                "implementation_notes": "All processing builders implement this to create processor instances",
                "common_patterns": [
                    "Choose instance type based on config (large vs small)",
                    "Set framework version, role, volume size from config",
                    "Generate job name using _generate_job_name()",
                    "Include environment variables from _get_environment_variables()"
                ]
            },
            "_get_outputs": {
                "signature": "_get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]",
                "description": "Transform logical outputs to ProcessingOutput objects",
                "return_type": "List[ProcessingOutput]",
                "required": True,
                "purpose": "Create ProcessingOutput objects with specific S3 destinations",
                "implementation_notes": "Processing steps must override this to define output destinations",
                "common_patterns": [
                    "Process each output in specification",
                    "Generate default paths using: base_output_path = self._get_base_output_path(); destination = Join(on='/', values=[base_output_path, step_identifier, logical_name])",
                    "Create ProcessingOutput with output_name, source, destination",
                    "Map container paths to S3 destinations"
                ]
            }
        },
        
        "optional_methods": {
            "_get_job_arguments": {
                "signature": "_get_job_arguments(self) -> List[str]",
                "description": "Constructs command-line arguments from script contract and config",
                "return_type": "List[str]",
                "required": False,
                "inherited_from": "StepBuilderBase",
                "purpose": "Override base implementation to add processing-specific arguments",
                "implementation_notes": "Processing steps often override this to add job_type or other arguments",
                "common_patterns": [
                    "Add --job_type argument from config",
                    "Add step-specific command-line parameters",
                    "Combine with base class arguments"
                ]
            }
        },
        
        "method_return_types": {
            "_get_inputs": "List[ProcessingInput]",
            "_get_outputs": "List[ProcessingOutput]", 
            "create_step": "ProcessingStep"
        },
        
        "validation_specifics": {
            "inputs_must_be_processing_inputs": True,
            "outputs_must_be_processing_outputs": True,
            "processor_must_have_role": True,
            "supports_job_arguments": True,
            "supports_caching": True
        },
        
        "common_input_patterns": [
            "Create ProcessingInput for each dependency",
            "Map logical names to container paths using contract",
            "Handle optional vs required inputs",
            "Set input_name and destination paths"
        ],
        
        "common_output_patterns": [
            "Create ProcessingOutput for each specification output",
            "Generate default destinations if not provided",
            "Map container paths to S3 destinations",
            "Set output_name and source paths"
        ]
    },
    
    # CreateModel Steps - Non-script validation (skip levels 1-2)
    "CreateModel": {
        "sagemaker_step_class": "CreateModelStep",
        "category": StepTypeCategory.NON_SCRIPT,
        "description": "SageMaker CreateModel steps for model deployment preparation",
        "examples": ["XGBoostModel", "PyTorchModel"],
        
        "required_methods": {
            "_create_model": {
                "signature": "_create_model(self, model_data: str) -> Model",
                "description": "Create SageMaker Model instance for deployment",
                "return_type": "Model",  # XGBoostModel, PyTorchModel, etc.
                "required": True,
                "purpose": "Create the model that defines model endpoint configuration",
                "implementation_notes": "All model builders implement this to create framework-specific models",
                "common_patterns": [
                    "Use model_data parameter for model artifacts",
                    "Set entry_point and source_dir for inference code",
                    "Generate or retrieve container image URI",
                    "Include environment variables for inference"
                ]
            }
        },
        
        "optional_methods": {
            "_get_image_uri": {
                "signature": "_get_image_uri(self) -> str",
                "description": "Generate appropriate SageMaker container image URI",
                "return_type": "str",
                "required": False,
                "purpose": "Create framework-specific container image URI for inference",
                "implementation_notes": "Model builders often implement this for custom image logic",
                "common_patterns": [
                    "Use image_uris.retrieve() from SageMaker SDK",
                    "Set framework, region, version from config",
                    "Handle region-specific requirements"
                ]
            }
        },
        
        "method_return_types": {
            "_get_inputs": "Dict[str, Any]",  # Processed inputs (typically model_data)
            "_get_outputs": "None",  # CreateModelStep handles outputs automatically
            "create_step": "CreateModelStep"
        },
        
        "validation_specifics": {
            "model_must_have_role": True,
            "model_must_have_image_or_package": True,
            "requires_model_data_input": True,
            "supports_caching": False,  # CreateModelStep doesn't support caching
            "outputs_handled_automatically": True
        },
        
        "common_input_patterns": [
            "Extract model_data from inputs dictionary",
            "Validate model_data is provided",
            "Process model artifacts location"
        ],
        
        "common_output_patterns": [
            "Return None - CreateModelStep provides ModelName property automatically",
            "No explicit output configuration needed",
            "Model name available via step properties"
        ]
    },
    
    # Transform Steps - Non-script validation (skip levels 1-2)
    "Transform": {
        "sagemaker_step_class": "TransformStep",
        "category": StepTypeCategory.NON_SCRIPT,
        "description": "SageMaker Transform steps for batch inference",
        "examples": ["BatchTransform", "ModelInference"],
        
        "required_methods": {
            "_create_transformer": {
                "signature": "_create_transformer(self, model_name: Union[str, Properties], output_path: Optional[str] = None) -> Transformer",
                "description": "Create SageMaker Transformer instance for batch transform job",
                "return_type": "Transformer",
                "required": True,
                "purpose": "Create the transformer that defines batch transform job configuration",
                "implementation_notes": "Transform builders implement this to create transformer instances with output path from _get_outputs"
            },
            "_get_outputs": {
                "signature": "_get_outputs(self, outputs: Dict[str, Any]) -> str",
                "description": "Generate output path for transform results, used by _create_transformer",
                "return_type": "str",
                "required": True,
                "purpose": "Provide S3 path where transform results will be stored, passed to transformer",
                "implementation_notes": "Transform steps must override this. Result is used by _create_transformer(output_path=...)",
                "common_patterns": [
                    "Generate default path using: base_output_path = self._get_base_output_path(); output_path = Join(on='/', values=[base_output_path, step_type, self.config.job_type])",
                    "Return single S3 path string",
                    "Used internally by _create_transformer method"
                ]
            }
        },
        
        "method_return_types": {
            "_get_inputs": "Tuple[TransformInput, Union[str, Properties]]",  # Returns both TransformInput and model_name
            "_get_outputs": "str",  # Output path for transform results
            "create_step": "TransformStep"
        },
        
        "validation_specifics": {
            "inputs_must_be_transform_input": True,
            "outputs_must_be_s3_path": True,
            "transformer_must_have_model": True,
            "supports_caching": True
        }
    },
    
    # RegisterModel Steps - Non-script validation (skip levels 1-2)
    "RegisterModel": {
        "sagemaker_step_class": "RegisterModel",
        "category": StepTypeCategory.NON_SCRIPT,
        "description": "SageMaker RegisterModel steps for model registry",
        "examples": ["ModelRegistration", "ModelPackageRegistration"],
        
        "required_methods": {
            "_create_model_package": {
                "signature": "_create_model_package(self) -> ModelPackage",
                "description": "Create SageMaker ModelPackage instance for registration",
                "return_type": "ModelPackage",
                "required": True,
                "purpose": "Create the model package for registration in model registry"
            }
        },
        
        "method_return_types": {
            "_get_inputs": "Dict[str, Any]",
            "_get_outputs": "None",  # RegisterModel handles outputs automatically
            "create_step": "RegisterModel"
        },
        
        "validation_specifics": {
            "model_package_must_have_group": True,
            "model_package_must_have_inference_spec": True,
            "supports_approval_workflow": True
        }
    },
    
    # Lambda Steps - Config-only validation (only level 4)
    "Lambda": {
        "sagemaker_step_class": "LambdaStep",
        "category": StepTypeCategory.CONFIG_ONLY,
        "description": "SageMaker Lambda steps for custom logic execution",
        "examples": ["CustomLogic", "DataValidation"],
        
        "required_methods": {
            "_create_lambda_function": {
                "signature": "_create_lambda_function(self) -> LambdaFunction",
                "description": "Create Lambda function configuration",
                "return_type": "LambdaFunction",
                "required": True,
                "purpose": "Create the Lambda function that will be executed"
            }
        },
        
        "method_return_types": {
            "_get_inputs": "Dict[str, Any]",
            "_get_outputs": "Dict[str, Any]",
            "create_step": "LambdaStep"
        },
        
        "validation_specifics": {
            "lambda_function_must_exist": True,
            "supports_custom_outputs": True,
            "no_script_validation_needed": True
        }
    },
    
    # Excluded step types - No validation needed
    "Base": {
        "sagemaker_step_class": None,
        "category": StepTypeCategory.EXCLUDED,
        "description": "Base configurations with no corresponding builder",
        "skip_reason": "Base configurations - no builder to validate",
        "examples": ["BaseConfig", "BaseStep"],
        "required_methods": {}  # Empty - no validation needed
    },
    
    "Utility": {
        "sagemaker_step_class": None,
        "category": StepTypeCategory.EXCLUDED,
        "description": "Utility configurations with no corresponding builder", 
        "skip_reason": "Utility configurations - no builder to validate",
        "examples": ["UtilityConfig", "HelperStep"],
        "required_methods": {}  # Empty - no validation needed
    }
}


def get_step_type_validation_rules() -> Dict[str, Any]:
    """
    Get all step-type-specific validation rules.
    
    Returns:
        Dictionary containing all step-type-specific validation rules
    """
    return STEP_TYPE_SPECIFIC_VALIDATION_RULES


def get_validation_rules_for_step_type(step_type: str) -> Optional[Dict[str, Any]]:
    """
    Get validation rules for a specific step type.
    
    Args:
        step_type: The SageMaker step type (e.g., "Training", "Processing")
        
    Returns:
        Dictionary of validation rules for the step type, or None if not found
    """
    return STEP_TYPE_SPECIFIC_VALIDATION_RULES.get(step_type)


def get_required_methods_for_step_type(step_type: str) -> Dict[str, Any]:
    """
    Get required methods for a specific step type.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        Dictionary of required methods, empty if step type not found
    """
    rules = get_validation_rules_for_step_type(step_type)
    if rules and "required_methods" in rules:
        return rules["required_methods"]
    return {}


def get_optional_methods_for_step_type(step_type: str) -> Dict[str, Any]:
    """
    Get optional methods for a specific step type.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        Dictionary of optional methods, empty if step type not found
    """
    rules = get_validation_rules_for_step_type(step_type)
    if rules and "optional_methods" in rules:
        return rules["optional_methods"]
    return {}


def get_all_methods_for_step_type(step_type: str) -> Dict[str, Any]:
    """
    Get all methods (required + optional) for a specific step type.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        Dictionary combining required and optional methods
    """
    required = get_required_methods_for_step_type(step_type)
    optional = get_optional_methods_for_step_type(step_type)
    
    all_methods = {}
    all_methods.update(required)
    all_methods.update(optional)
    
    return all_methods


def get_step_types_by_category(category: StepTypeCategory) -> List[str]:
    """
    Get all step types in a specific category.
    
    Args:
        category: The step type category
        
    Returns:
        List of step type names in the category
    """
    step_types = []
    
    for step_type, rules in STEP_TYPE_SPECIFIC_VALIDATION_RULES.items():
        if rules.get("category") == category:
            step_types.append(step_type)
    
    return step_types


def is_step_type_excluded(step_type: str) -> bool:
    """
    Check if a step type is excluded from validation.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        True if step type is excluded, False otherwise
    """
    rules = get_validation_rules_for_step_type(step_type)
    if not rules:
        return False  # Unknown step types are not excluded
    return rules.get("category") == StepTypeCategory.EXCLUDED


def get_step_type_category(step_type: str) -> Optional[StepTypeCategory]:
    """
    Get the category for a step type.
    
    Args:
        step_type: The SageMaker step type
        
    Returns:
        StepTypeCategory enum value, or None if step type not found
    """
    rules = get_validation_rules_for_step_type(step_type)
    return rules.get("category") if rules else None


def validate_step_type_compliance(builder_class: type, step_type: str) -> List[str]:
    """
    Validate that a builder class complies with step-type-specific rules.
    
    Args:
        builder_class: The builder class to validate
        step_type: The SageMaker step type
        
    Returns:
        List of validation issues (empty if compliant)
    """
    issues = []
    
    # Check if step type is excluded
    if is_step_type_excluded(step_type):
        return []  # No validation needed for excluded step types
    
    # Get step-type-specific rules
    rules = get_validation_rules_for_step_type(step_type)
    if not rules:
        issues.append(f"No validation rules found for step type: {step_type}")
        return issues
    
    # Check required methods
    required_methods = rules.get("required_methods", {})
    for method_name, method_spec in required_methods.items():
        if not hasattr(builder_class, method_name):
            issues.append(f"Builder {builder_class.__name__} missing required {step_type} method: {method_name}")
    
    return issues


def get_validation_summary() -> Dict[str, Any]:
    """
    Get a summary of all step types and their validation requirements.
    
    Returns:
        Dictionary with validation summary statistics
    """
    summary = {
        "total_step_types": len(STEP_TYPE_SPECIFIC_VALIDATION_RULES),
        "by_category": {},
        "validation_coverage": {}
    }
    
    # Count by category
    for category in StepTypeCategory:
        step_types = get_step_types_by_category(category)
        summary["by_category"][category.value] = {
            "count": len(step_types),
            "step_types": step_types
        }
    
    # Validation coverage
    total_methods = 0
    for step_type, rules in STEP_TYPE_SPECIFIC_VALIDATION_RULES.items():
        if rules.get("category") != StepTypeCategory.EXCLUDED:
            required_count = len(rules.get("required_methods", {}))
            optional_count = len(rules.get("optional_methods", {}))
            total_methods += required_count + optional_count
    
    summary["validation_coverage"]["total_step_specific_methods"] = total_methods
    
    return summary
