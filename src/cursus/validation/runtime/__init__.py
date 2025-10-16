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
    RuntimeTestingConfiguration,
)
from .runtime_spec_builder import PipelineTestingSpecBuilder

# NEW: Interactive factory (primary user interface)
from .interactive_factory import InteractiveRuntimeTestingFactory
from .workspace_aware_spec_builder import WorkspaceAwarePipelineTestingSpecBuilder
from ...step_catalog.adapters.contract_adapter import ContractDiscoveryManagerAdapter as ContractDiscoveryManager, ContractDiscoveryResult

# Enhanced logical name matching components
from .logical_name_matching import (
    PathSpec,
    PathMatch,
    MatchType,
    EnhancedScriptExecutionSpec,
    PathMatcher,
    TopologicalExecutor,
    LogicalNameMatchingTester,
    EnhancedDataCompatibilityResult,
)

# Inference handler testing components
from .runtime_inference import (
    InferenceHandlerSpec,
    InferenceTestResult,
    InferencePipelineTestingSpec,
)

# Main API exports - Simplified to user requirements only
__all__ = [
    # Primary Interactive Interface
    "InteractiveRuntimeTestingFactory",
    
    # Core Components
    "RuntimeTester",
    "ScriptTestResult",
    "DataCompatibilityResult",
    "PipelineTestingSpecBuilder",
    "WorkspaceAwarePipelineTestingSpecBuilder",
    "ContractDiscoveryManager",
    "ContractDiscoveryResult",
    # Enhanced logical name matching exports
    "PathSpec",
    "PathMatch",
    "MatchType",
    "EnhancedScriptExecutionSpec",
    "PathMatcher",
    "TopologicalExecutor",
    "LogicalNameMatchingTester",
    "EnhancedDataCompatibilityResult",
    # Inference handler testing exports
    "InferenceHandlerSpec",
    "InferenceTestResult",
    "InferencePipelineTestingSpec",
]
