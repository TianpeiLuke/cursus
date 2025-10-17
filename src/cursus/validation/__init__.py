"""
Cursus Validation Framework

This module provides comprehensive validation capabilities for pipeline components,
including step builders, alignment testing, runtime validation, and script testing.
"""

# Import alignment testing components
from .alignment import unified_alignment_tester

# Import builder testing components
from .builders import universal_test

# Import simplified script testing framework
from .script_testing import (
    run_dag_scripts,
    ScriptTestingInputCollector,
    ResultFormatter,
    ScriptTestResult as ScriptExecutionResult,  # Renamed to avoid conflict
    execute_single_script,
    install_script_dependencies,
    quick_test_dag,
    get_script_testing_info
)

# Export available functions and classes
__all__ = [
    # Alignment testing
    "unified_alignment_tester",
    # Builder testing
    "universal_test",
    # Script testing framework (simplified)
    "run_dag_scripts",
    "ScriptTestingInputCollector",
    "ResultFormatter",
    "ScriptExecutionResult",
    "execute_single_script",
    "install_script_dependencies",
    "quick_test_dag",
    "get_script_testing_info",
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
