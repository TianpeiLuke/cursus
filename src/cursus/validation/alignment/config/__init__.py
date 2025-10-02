"""
Validation Alignment Configuration Module

This module provides centralized configuration for the validation alignment system,
controlling which validation levels are applied to different SageMaker step types.
"""

from .validation_ruleset import (
    # Enums
    ValidationLevel,
    StepTypeCategory,
    
    # Data classes
    ValidationRuleset,
    
    # Configuration data
    VALIDATION_RULESETS,
    
    # Core API functions
    get_validation_ruleset,
    is_validation_level_enabled,
    get_enabled_validation_levels,
    get_level_4_validator_class,
    is_step_type_excluded,
    get_step_types_by_category,
    get_all_step_types,
    validate_step_type_configuration,
    
    # Registry integration functions
    get_validation_ruleset_for_step_name,
    is_validation_level_enabled_for_step_name,
    get_enabled_validation_levels_for_step_name,
    is_step_name_excluded,
)

# Import universal builder validation rules
from .universal_builder_rules import (
    UNIVERSAL_BUILDER_VALIDATION_RULES,
    UniversalMethodCategory,
    get_universal_validation_rules,
    get_required_methods,
    get_inherited_methods,
    get_validation_rules,
    validate_universal_compliance,
)

# Import step-type-specific validation rules
from .step_type_specific_rules import (
    STEP_TYPE_SPECIFIC_VALIDATION_RULES,
    get_step_type_validation_rules,
    get_validation_rules_for_step_type,
    get_required_methods_for_step_type,
    get_optional_methods_for_step_type,
    get_all_methods_for_step_type,
    get_step_types_by_category as get_step_types_by_category_specific,
    is_step_type_excluded as is_step_type_excluded_specific,
    get_step_type_category as get_step_type_category_specific,
    validate_step_type_compliance,
    get_validation_summary,
)

__all__ = [
    # Enums
    "ValidationLevel",
    "StepTypeCategory",
    "UniversalMethodCategory",
    
    # Data classes
    "ValidationRuleset",
    
    # Configuration data
    "VALIDATION_RULESETS",
    "UNIVERSAL_BUILDER_VALIDATION_RULES",
    "STEP_TYPE_SPECIFIC_VALIDATION_RULES",
    
    # Core API functions
    "get_validation_ruleset",
    "is_validation_level_enabled",
    "get_enabled_validation_levels",
    "get_level_4_validator_class",
    "is_step_type_excluded",
    "get_step_types_by_category",
    "get_all_step_types",
    "validate_step_type_configuration",
    
    # Registry integration functions
    "get_validation_ruleset_for_step_name",
    "is_validation_level_enabled_for_step_name",
    "get_enabled_validation_levels_for_step_name",
    "is_step_name_excluded",
    
    # Universal builder validation
    "get_universal_validation_rules",
    "get_required_methods",
    "get_inherited_methods",
    "get_validation_rules",
    "validate_universal_compliance",
    
    # Step-type-specific validation
    "get_step_type_validation_rules",
    "get_validation_rules_for_step_type",
    "get_required_methods_for_step_type",
    "get_optional_methods_for_step_type",
    "get_all_methods_for_step_type",
    "get_step_types_by_category_specific",
    "is_step_type_excluded_specific",
    "get_step_type_category_specific",
    "validate_step_type_compliance",
    "get_validation_summary",
]
