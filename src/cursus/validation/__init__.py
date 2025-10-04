"""
Cursus Validation Framework

This module provides validation capabilities for step builders, alignment testing,
and runtime validation of pipeline components.
"""

# Import runtime testing components that actually exist
from .runtime import (
    RuntimeTester,
    ScriptTestResult,
    DataCompatibilityResult,
    PipelineTestingSpecBuilder,
    WorkspaceAwarePipelineTestingSpecBuilder,
)

# Import alignment testing components
from .alignment import unified_alignment_tester

# Import builder testing components
from .builders import universal_test

# Export available functions and classes
__all__ = [
    # Runtime testing components
    "RuntimeTester",
    "ScriptTestResult", 
    "DataCompatibilityResult",
    "PipelineTestingSpecBuilder",
    "WorkspaceAwarePipelineTestingSpecBuilder",
    # Alignment testing
    "unified_alignment_tester",
    # Builder testing
    "universal_test",
]


def get_validation_info() -> dict:
    """
    Get information about available validation components.
    
    Returns:
        Dictionary with validation framework information
    """
    return {
        "runtime_testing": "Available - RuntimeTester and related components",
        "alignment_testing": "Available - unified_alignment_tester module", 
        "builder_testing": "Available - universal_test module",
        "available_classes": [
            "RuntimeTester",
            "ScriptTestResult",
            "DataCompatibilityResult", 
            "PipelineTestingSpecBuilder",
            "WorkspaceAwarePipelineTestingSpecBuilder",
        ],
        "available_modules": [
            "unified_alignment_tester",
            "universal_test",
        ],
    }
