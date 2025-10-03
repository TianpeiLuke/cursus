"""
Universal Builder Validation Rules

This module defines the universal validation rules that ALL step builders must implement,
regardless of their specific SageMaker step type. These rules focus on method interface
compliance rather than configuration field validation.

Based on analysis of actual step builders in the cursus codebase.
"""

from typing import Dict, Any, List, Optional
from enum import Enum


class UniversalMethodCategory(Enum):
    """Categories of universal methods that all builders must implement."""
    REQUIRED_ABSTRACT = "required_abstract"  # Must be implemented by all builders
    REQUIRED_OVERRIDE = "required_override"   # Must override base class method
    INHERITED_OPTIONAL = "inherited_optional" # Can optionally override base class method
    INHERITED_FINAL = "inherited_final"       # Cannot override base class method


UNIVERSAL_BUILDER_VALIDATION_RULES = {
    "version": "1.0.0",
    "description": "Universal validation rules for all step builders based on actual codebase analysis",
    "last_updated": "2025-10-02",
    
    "required_methods": {
        "validate_configuration": {
            "signature": "validate_configuration(self) -> None",
            "description": "Validate configuration requirements specific to this step type",
            "return_type": "None",
            "category": UniversalMethodCategory.REQUIRED_ABSTRACT,
            "required": True,
            "raises": ["ValueError"],
            "purpose": "Ensure builder configuration is valid before step creation",
            "implementation_notes": "All builders implement this with step-specific validation logic",
            "examples": [
                "Validate required config attributes exist",
                "Validate config values are within acceptable ranges",
                "Validate dependencies between config fields"
            ]
        },
        
        "_get_inputs": {
            "signature": "_get_inputs(self, inputs: Dict[str, Any]) -> Any",
            "description": "Transform logical inputs to step-specific input format",
            "return_type": "Any",  # Varies by step type - see step-specific rules
            "category": UniversalMethodCategory.REQUIRED_ABSTRACT,
            "required": True,
            "purpose": "Convert generic input dictionary to step-type-specific input objects",
            "implementation_notes": "Return type varies: Dict[str, TrainingInput] for Training, List[ProcessingInput] for Processing, etc.",
            "common_patterns": [
                "Process specification dependencies",
                "Map logical names to container paths using script contract",
                "Create step-specific input objects (TrainingInput, ProcessingInput, etc.)"
            ]
        },
        
        
        "create_step": {
            "signature": "create_step(self, **kwargs: Any) -> Step",
            "description": "Create the final SageMaker pipeline step",
            "return_type": "Step",  # Specific subclass varies by step type
            "category": UniversalMethodCategory.REQUIRED_ABSTRACT,
            "required": True,
            "purpose": "Orchestrate step creation using all other methods",
            "common_kwargs": [
                "inputs: Dict[str, Any]",
                "outputs: Dict[str, Any]", 
                "dependencies: List[Step]",
                "enable_caching: bool"
            ],
            "implementation_notes": "All builders follow similar pattern: extract inputs, create step-specific objects, return configured step",
            "common_patterns": [
                "Extract and process inputs using _get_inputs()",
                "Extract and process outputs using _get_outputs()",
                "Create step-specific objects (estimator, processor, model)",
                "Handle dependencies and caching",
                "Return configured SageMaker step"
            ]
        }
    },
    
    "inherited_methods": {
        "_get_environment_variables": {
            "signature": "_get_environment_variables(self) -> Dict[str, str]",
            "description": "Create environment variables from script contract and config",
            "return_type": "Dict[str, str]",
            "category": UniversalMethodCategory.INHERITED_OPTIONAL,
            "inherited_from": "StepBuilderBase",
            "can_override": True,
            "required": False,
            "purpose": "Generate environment variables for step execution",
            "implementation_notes": "Many builders override this to add step-specific environment variables, but not all"
        },
        
        "_get_job_arguments": {
            "signature": "_get_job_arguments(self) -> Optional[List[str]]",
            "description": "Constructs command-line arguments from script contract",
            "return_type": "Optional[List[str]]",
            "category": UniversalMethodCategory.INHERITED_OPTIONAL,
            "inherited_from": "StepBuilderBase",
            "can_override": True,
            "required": False,
            "purpose": "Generate command-line arguments for script execution",
            "implementation_notes": "Processing steps often override this, Training steps typically use base implementation"
        },
        
        "_get_outputs": {
            "signature": "_get_outputs(self, outputs: Dict[str, Any]) -> Any",
            "description": "Transform logical outputs to step-specific output format",
            "return_type": "Any",  # Varies by step type - see step-specific rules
            "category": UniversalMethodCategory.INHERITED_OPTIONAL,
            "inherited_from": "StepBuilderBase",
            "can_override": True,
            "required": False,
            "purpose": "Convert generic output dictionary to step-type-specific output format",
            "implementation_notes": "Only certain step types need to override this. See step-type-specific rules for requirements.",
            "common_patterns": [
                "Check specification and contract availability",
                "Process each output in specification",
                "Generate default paths using: base_output_path = self._get_base_output_path(); destination = Join(on='/', values=[base_output_path, step_identifier, logical_name])",
                "Use Join() for parameter-compatible paths",
                "Return step-type-appropriate format"
            ]
        },
        
        "_get_cache_config": {
            "signature": "_get_cache_config(self, enable_caching: bool = True) -> CacheConfig",
            "description": "Get cache configuration for step",
            "return_type": "CacheConfig",
            "category": UniversalMethodCategory.INHERITED_FINAL,
            "inherited_from": "StepBuilderBase",
            "can_override": False,
            "required": False,
            "purpose": "Configure step caching behavior"
        },
        
        "_generate_job_name": {
            "signature": "_generate_job_name(self, step_type: Optional[str] = None) -> str",
            "description": "Generate standardized job name for SageMaker jobs",
            "return_type": "str",
            "category": UniversalMethodCategory.INHERITED_FINAL,
            "inherited_from": "StepBuilderBase",
            "can_override": False,
            "required": False,
            "purpose": "Create unique, valid SageMaker job names"
        },
        
        "_get_step_name": {
            "signature": "_get_step_name(self, include_job_type: bool = True) -> str",
            "description": "Get standard step name from builder class name",
            "return_type": "str",
            "category": UniversalMethodCategory.INHERITED_FINAL,
            "inherited_from": "StepBuilderBase",
            "can_override": False,
            "required": False,
            "purpose": "Extract step name from builder class for registry lookup"
        },
        
        "_get_base_output_path": {
            "signature": "_get_base_output_path(self) -> str",
            "description": "Get base output path for step outputs",
            "return_type": "str",
            "category": UniversalMethodCategory.INHERITED_FINAL,
            "inherited_from": "StepBuilderBase",
            "can_override": False,
            "required": False,
            "purpose": "Provide consistent base path for step outputs"
        }
    },
    
    "required_constructor_params": {
        "config": {
            "type": "BasePipelineConfig",
            "description": "Step configuration object",
            "required": True,
            "purpose": "Provide step-specific configuration"
        },
        "spec": {
            "type": "Optional[StepSpecification]",
            "description": "Step specification for specification-driven implementation",
            "required": False,
            "purpose": "Enable specification-driven step creation",
            "implementation_notes": "Most builders now use specifications"
        },
        "sagemaker_session": {
            "type": "Optional[PipelineSession]",
            "description": "SageMaker session",
            "required": False,
            "purpose": "Manage AWS SageMaker interactions"
        },
        "role": {
            "type": "Optional[str]",
            "description": "IAM role ARN",
            "required": False,
            "purpose": "Provide AWS execution permissions"
        },
        "registry_manager": {
            "type": "Optional[RegistryManager]",
            "description": "Registry manager for dependency injection",
            "required": False,
            "purpose": "Enable registry-based dependency resolution"
        },
        "dependency_resolver": {
            "type": "Optional[UnifiedDependencyResolver]",
            "description": "Dependency resolver for dependency injection",
            "required": False,
            "purpose": "Enable automatic dependency resolution"
        }
    },
    
    "validation_rules": {
        "inheritance": {
            "must_inherit_from": "StepBuilderBase",
            "description": "All step builders must inherit from StepBuilderBase",
            "validation_method": "isinstance(builder_class, type) and issubclass(builder_class, StepBuilderBase)"
        },
        
        "method_signatures": {
            "validate_parameter_names": True,
            "validate_return_types": True,
            "validate_parameter_types": True,
            "description": "Validate method signatures match expected patterns",
            "strict_mode": False  # Allow some flexibility in parameter names
        },
        
        "abstract_methods": {
            "must_implement_all": True,
            "description": "All abstract methods from base class must be implemented",
            "required_abstract_methods": [
                "validate_configuration",
                "_get_inputs", 
                "create_step"
            ]
        },
        
        "method_categories": {
            "required_abstract": {
                "description": "Methods that must be implemented by all builders",
                "validation_level": "ERROR",
                "methods": ["validate_configuration", "_get_inputs", "create_step"]
            },
            "inherited_optional": {
                "description": "Methods that can optionally override base class",
                "validation_level": "INFO",
                "methods": ["_get_environment_variables", "_get_job_arguments", "_get_outputs"]
            },
            "inherited_final": {
                "description": "Methods that should not be overridden",
                "validation_level": "WARNING",
                "methods": ["_get_cache_config", "_generate_job_name", "_get_step_name", "_get_base_output_path"]
            }
        }
    },
    
    "common_implementation_patterns": {
        "initialization": {
            "description": "Common patterns in __init__ method",
            "patterns": [
                "Validate config type with isinstance()",
                "Load and validate step specification",
                "Call super().__init__() with all parameters",
                "Store typed config reference"
            ]
        },
        
        "validation": {
            "description": "Common patterns in validate_configuration",
            "patterns": [
                "Check required attributes exist with hasattr()",
                "Validate attribute values are not None or empty",
                "Validate enum values are in acceptable ranges",
                "Raise ValueError with descriptive messages"
            ]
        },
        
        "input_processing": {
            "description": "Common patterns in _get_inputs",
            "patterns": [
                "Check specification and contract availability",
                "Process each dependency in specification",
                "Map logical names to container paths using contract",
                "Create step-specific input objects",
                "Handle optional vs required inputs"
            ]
        },
        
        "output_processing": {
            "description": "Common patterns in _get_outputs", 
            "patterns": [
                "Check specification and contract availability",
                "Process each output in specification",
                "Generate default paths if not provided",
                "Use Join() for parameter-compatible paths",
                "Return step-type-appropriate format"
            ]
        },
        
        "step_creation": {
            "description": "Common patterns in create_step",
            "patterns": [
                "Extract common parameters from kwargs",
                "Handle dependency input extraction",
                "Process inputs and outputs using helper methods",
                "Create step-specific objects (estimator, processor, model)",
                "Configure step with dependencies and caching",
                "Attach specification to step for reference"
            ]
        }
    }
}


def get_universal_validation_rules() -> Dict[str, Any]:
    """
    Get the universal builder validation rules.
    
    Returns:
        Dictionary containing all universal validation rules
    """
    return UNIVERSAL_BUILDER_VALIDATION_RULES


def get_required_methods() -> Dict[str, Any]:
    """
    Get only the required methods that all builders must implement.
    
    Returns:
        Dictionary of required methods with their specifications
    """
    return UNIVERSAL_BUILDER_VALIDATION_RULES["required_methods"]


def get_inherited_methods() -> Dict[str, Any]:
    """
    Get the inherited methods from StepBuilderBase.
    
    Returns:
        Dictionary of inherited methods with their specifications
    """
    return UNIVERSAL_BUILDER_VALIDATION_RULES["inherited_methods"]


def get_validation_rules() -> Dict[str, Any]:
    """
    Get the validation rules for universal builder compliance.
    
    Returns:
        Dictionary of validation rules and criteria
    """
    return UNIVERSAL_BUILDER_VALIDATION_RULES["validation_rules"]


def validate_universal_compliance(builder_class: type) -> List[str]:
    """
    Validate that a builder class complies with universal rules.
    
    Args:
        builder_class: The builder class to validate
        
    Returns:
        List of validation issues (empty if compliant)
    """
    issues = []
    
    # Handle None input
    if builder_class is None:
        issues.append("Builder class cannot be None")
        return issues
    
    # Handle non-class input
    if not isinstance(builder_class, type):
        issues.append(f"Expected a class, got {type(builder_class).__name__}")
        return issues
    
    # Check inheritance
    try:
        from ....core.base.builder_base import StepBuilderBase
        if not issubclass(builder_class, StepBuilderBase):
            issues.append(f"Builder {builder_class.__name__} must inherit from StepBuilderBase")
    except ImportError:
        issues.append("Cannot validate inheritance - StepBuilderBase not available")
    except TypeError:
        issues.append(f"Cannot check inheritance for {builder_class}")
    
    # Check required methods
    required_methods = get_required_methods()
    for method_name, method_spec in required_methods.items():
        if not hasattr(builder_class, method_name):
            issues.append(f"Builder {builder_class.__name__} missing required method: {method_name}")
    
    return issues
