"""
Pipeline Runtime Testing System

A comprehensive testing framework for validating pipeline script functionality,
data flow compatibility, and end-to-end execution.
"""

# Core components
from .core.pipeline_script_executor import PipelineScriptExecutor
from .core.script_import_manager import ScriptImportManager
from .core.data_flow_manager import DataFlowManager

# Data management
from .data.synthetic_data_generator import SyntheticDataGenerator
from .data.local_data_manager import LocalDataManager

# Utilities
from .utils.result_models import TestResult, ExecutionResult
from .utils.execution_context import ExecutionContext

# Testing components
from .testing.pipeline_dag_resolver import PipelineDAGResolver, PipelineExecutionPlan
from .testing.pipeline_executor import PipelineExecutor, PipelineExecutionResult, StepExecutionResult
from .testing.data_compatibility_validator import DataCompatibilityValidator, DataCompatibilityReport, DataSchemaInfo

# S3 Integration components
from .integration.s3_data_downloader import S3DataDownloader, S3DataSource, DownloadResult
from .integration.workspace_manager import WorkspaceManager, WorkspaceConfig, CacheEntry
from .integration.real_data_tester import (
    RealDataTester,
    RealDataTestScenario,
    RealDataTestResult,
    ProductionValidationRule
)

# Main API exports
__all__ = [
    # Core components
    'PipelineScriptExecutor',
    'ScriptImportManager', 
    'DataFlowManager',
    
    # Data management
    'SyntheticDataGenerator',
    'LocalDataManager',
    
    # Utilities
    'TestResult',
    'ExecutionResult',
    'ExecutionContext',
    
    # Pipeline testing
    'PipelineDAGResolver',
    'PipelineExecutionPlan',
    'PipelineExecutor',
    'PipelineExecutionResult',
    'StepExecutionResult',
    'DataCompatibilityValidator',
    'DataCompatibilityReport',
    'DataSchemaInfo',
    
    # S3 Integration
    'S3DataDownloader',
    'S3DataSource',
    'DownloadResult',
    'WorkspaceManager',
    'WorkspaceConfig',
    'CacheEntry',
    'RealDataTester',
    'RealDataTestScenario',
    'RealDataTestResult',
    'ProductionValidationRule'
]
