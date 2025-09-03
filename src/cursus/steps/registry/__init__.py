"""
Backward compatibility shim for registry imports.
All functionality has moved to cursus.registry.

This module provides backward compatibility for existing imports while
the codebase is being migrated to use the new location.
"""
import warnings

# Issue deprecation warning for old import path
warnings.warn(
    "Importing from cursus.steps.registry is deprecated. "
    "Use cursus.registry instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new registry location
from ...registry.step_names import *
from ...registry.builder_registry import *
from ...registry.hyperparameter_registry import *
from ...registry.exceptions import *

# Re-export all public API for backward compatibility
__all__ = [
    # Core registry data structures
    'STEP_NAMES',
    'CONFIG_STEP_REGISTRY', 
    'BUILDER_STEP_NAMES',
    'SPEC_STEP_TYPES',
    
    # Registry classes
    'StepBuilderRegistry',
    'HYPERPARAMETER_REGISTRY',
    
    # Helper functions from step_names
    'get_config_class_name',
    'get_builder_step_name',
    'get_spec_step_type',
    'get_spec_step_type_with_job_type',
    'get_step_name_from_spec_type',
    'get_all_step_names',
    'validate_step_name',
    'validate_spec_type',
    'get_step_description',
    'list_all_step_info',
    'get_sagemaker_step_type',
    'get_steps_by_sagemaker_type',
    'get_all_sagemaker_step_types',
    'validate_sagemaker_step_type',
    'get_sagemaker_step_type_mapping',
    'get_canonical_name_from_file_name',
    'validate_file_name',
    
    # Registry management from builder_registry
    'get_global_registry',
    'register_global_builder',
    'list_global_step_types',
    
    # Hyperparameter registry functions
    'get_all_hyperparameter_classes',
    'get_hyperparameter_class_by_model_type',
    'get_module_path',
    'get_all_hyperparameter_info',
    'validate_hyperparameter_class',
    
    # Exceptions
    'RegistryError'
]
