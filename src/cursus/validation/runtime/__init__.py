"""
Simplified Pipeline Runtime Testing

Validates script functionality and data transfer consistency for pipeline development.
Based on validated user story: "examine the script's functionality and their data 
transfer consistency along the DAG, without worrying about the resolution of 
step-to-step or step-to-script dependencies."
"""

# Simplified runtime testing components
from .runtime_testing import RuntimeTester
from .runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
    RuntimeTestingConfiguration
)
from .runtime_spec_builder import PipelineTestingSpecBuilder

# Main API exports - Simplified to user requirements only
__all__ = [
    'RuntimeTester',
    'ScriptTestResult', 
    'DataCompatibilityResult'
]
