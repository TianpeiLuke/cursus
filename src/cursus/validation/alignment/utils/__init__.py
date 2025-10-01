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

# Core models and enums
from .core_models import (
    SeverityLevel,
    AlignmentLevel,
    AlignmentIssue,
    StepTypeAwareAlignmentIssue,
    ValidationResult,
    create_alignment_issue,
    create_step_type_aware_alignment_issue,
)

# Script analysis models
from .script_analysis_models import (
    PathReference,
    EnvVarAccess,
    ImportStatement,
    ArgumentDefinition,
    PathConstruction,
    FileOperation,
)

# Dependency classification
from ..validators.dependency_classifier import (
    DependencyPattern,
    DependencyPatternClassifier,
)

# File resolution
from ....step_catalog.adapters.file_resolver import FlexibleFileResolverAdapter as FlexibleFileResolver

# Step type detection
from ..factories.step_type_detection import (
    detect_step_type_from_registry,
    detect_framework_from_imports,
    detect_step_type_from_script_patterns,
    get_step_type_context,
)

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
    # Core utilities
    "SeverityLevel",
    "AlignmentLevel", 
    "AlignmentIssue",
    "StepTypeAwareAlignmentIssue",
    "create_alignment_issue",
    "create_step_type_aware_alignment_issue",
    
    # Script analysis models
    "PathReference",
    "EnvVarAccess",
    "ImportStatement",
    "ArgumentDefinition",
    "PathConstruction",
    "FileOperation",
    
    # Dependency classification
    "DependencyPattern",
    "DependencyPatternClassifier",
    
    # File resolution
    "FlexibleFileResolver",
    
    # Step type detection
    "detect_step_type_from_registry",
    "detect_framework_from_imports",
    "detect_step_type_from_script_patterns",
    "get_step_type_context",
    
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
