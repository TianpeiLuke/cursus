"""
Pipeline Registry Module.

This module contains registry components for tracking step types, specifications,
hyperparameters, and other metadata used in the pipeline system. It helps ensure 
consistency in step naming and configuration.
"""

from .exceptions import RegistryError

from .builder_registry import (
    StepBuilderRegistry,
    get_global_registry,
    register_global_builder,
    list_global_step_types
)

from .step_names import (
    STEP_NAMES,
    CONFIG_STEP_REGISTRY,
    BUILDER_STEP_NAMES,
    SPEC_STEP_TYPES,
    get_config_class_name,
    get_builder_step_name,
    get_spec_step_type,
    get_spec_step_type_with_job_type,
    get_step_name_from_spec_type,
    get_all_step_names,
    validate_step_name,
    validate_spec_type,
    get_step_description,
    list_all_step_info
)

from .hyperparameter_registry import (
    HYPERPARAMETER_REGISTRY,
    get_all_hyperparameter_classes,
    get_hyperparameter_class_by_model_type,
    get_module_path,
    get_all_hyperparameter_info,
    validate_hyperparameter_class
)

__all__ = [
    # Exceptions
    "RegistryError",
    
    # Builder registry
    "StepBuilderRegistry",
    "get_global_registry",
    "register_global_builder",
    "list_global_step_types",
    
    # Step names and registry
    "STEP_NAMES",
    "CONFIG_STEP_REGISTRY",
    "BUILDER_STEP_NAMES",
    "SPEC_STEP_TYPES",
    "get_config_class_name",
    "get_builder_step_name",
    "get_spec_step_type",
    "get_spec_step_type_with_job_type",
    "get_step_name_from_spec_type",
    "get_all_step_names",
    "validate_step_name",
    "validate_spec_type",
    "get_step_description",
    "list_all_step_info",
    
    # Hyperparameter registry
    "HYPERPARAMETER_REGISTRY",
    "get_all_hyperparameter_classes",
    "get_hyperparameter_class_by_model_type",
    "get_module_path",
    "get_all_hyperparameter_info",
    "validate_hyperparameter_class"
]
