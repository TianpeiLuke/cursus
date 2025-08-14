"""
Alignment validation utilities - Main import aggregator.

This module provides a clean interface to all alignment validation utilities
by importing from the specialized modules. This maintains backward compatibility
while organizing code into focused modules.
"""

# Core models and enums
from .core_models import (
    SeverityLevel,
    AlignmentLevel,
    AlignmentIssue,
    StepTypeAwareAlignmentIssue,
    create_alignment_issue,
    create_step_type_aware_alignment_issue
)

# Script analysis models
from .script_analysis_models import (
    PathReference,
    EnvVarAccess,
    ImportStatement,
    ArgumentDefinition,
    PathConstruction,
    FileOperation
)

# Dependency classification
from .dependency_classifier import (
    DependencyPattern,
    DependencyPatternClassifier
)

# File resolution
from .file_resolver import (
    FlexibleFileResolver
)

# Step type detection
from .step_type_detection import (
    detect_step_type_from_registry,
    detect_framework_from_imports,
    detect_step_type_from_script_patterns,
    get_step_type_context
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
    get_validation_summary_stats
)

# Re-export everything for backward compatibility
__all__ = [
    # Core models
    'SeverityLevel',
    'AlignmentLevel', 
    'AlignmentIssue',
    'StepTypeAwareAlignmentIssue',
    'create_alignment_issue',
    'create_step_type_aware_alignment_issue',
    
    # Script analysis models
    'PathReference',
    'EnvVarAccess',
    'ImportStatement',
    'ArgumentDefinition',
    'PathConstruction',
    'FileOperation',
    
    # Dependency classification
    'DependencyPattern',
    'DependencyPatternClassifier',
    
    # File resolution
    'FlexibleFileResolver',
    
    # Step type detection
    'detect_step_type_from_registry',
    'detect_framework_from_imports',
    'detect_step_type_from_script_patterns',
    'get_step_type_context',
    
    # Utility functions
    'normalize_path',
    'extract_logical_name_from_path',
    'is_sagemaker_path',
    'format_alignment_issue',
    'group_issues_by_severity',
    'get_highest_severity',
    'validate_environment_setup',
    'get_validation_summary_stats'
]
