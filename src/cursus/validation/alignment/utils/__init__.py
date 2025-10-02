"""
Utilities and Models Module

This module contains utility functions, data models, and configuration classes
for the alignment validation system. It provides common functionality used
across all validation components.

Components:
- alignment_utils.py: Core utility functions and helper classes
- core_models.py: Core data models and type definitions
- level3_validation_config.py: Level 3 validation configuration and modes
- script_analysis_models.py: Models for script analysis results
- utils.py: General utility functions and helpers

Utility Features:
- Common data structures and enums
- Validation configuration management
- Helper functions for alignment operations
- Type definitions and model classes
- Configuration validation and defaults
"""

# Core models and enums - now from consolidated validation_models
from .validation_models import (
    ValidationLevel,
    ValidationStatus,
    IssueLevel,
    RuleType,
    ValidationIssue,
    ValidationResult,
    ValidationSummary,
    MethodValidationInfo,
    StepValidationContext,
    create_validation_issue,
    create_validation_result,
    merge_validation_results,
    filter_issues_by_level,
    group_issues_by_method,
    format_validation_summary,
)

# Note: Dependency classification was removed during consolidation
# from ..validators.dependency_classifier import (
#     DependencyPattern,
#     DependencyPatternClassifier,
# )

# File resolution
from ....step_catalog.adapters.file_resolver import FlexibleFileResolverAdapter as FlexibleFileResolver

# Step type detection - using registry functions instead of redundant factories
from ....registry.step_names import (
    get_sagemaker_step_type,
    get_canonical_name_from_file_name,
)
from ....step_catalog import StepCatalog

# Utility functions
from .utils import (
    normalize_path,
    extract_logical_name_from_path,
    is_sagemaker_path,
    format_alignment_issue,
    group_issues_by_severity,
    get_highest_severity,
    validate_environment_setup,
    get_validation_summary_stats,
)

# Level 3 validation configuration
from ..core.level3_validation_config import (
    Level3ValidationConfig,
    ValidationMode,
)

__all__ = [
    # Core validation models - consolidated
    "ValidationLevel",
    "ValidationStatus",
    "IssueLevel",
    "RuleType",
    "ValidationIssue",
    "ValidationResult",
    "ValidationSummary",
    "MethodValidationInfo",
    "StepValidationContext",
    "create_validation_issue",
    "create_validation_result",
    "merge_validation_results",
    "filter_issues_by_level",
    "group_issues_by_method",
    "format_validation_summary",
    
    
    # File resolution
    "FlexibleFileResolver",
    
    # Step type detection - registry functions
    "get_sagemaker_step_type",
    "get_canonical_name_from_file_name",
    "StepCatalog",
    
    # Level 3 configuration
    "Level3ValidationConfig",
    "ValidationMode",
    
    # General utilities
    "normalize_path",
    "extract_logical_name_from_path",
    "is_sagemaker_path",
    "format_alignment_issue",
    "group_issues_by_severity",
    "get_highest_severity",
    "validate_environment_setup",
    "get_validation_summary_stats",
]
