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
from .data.base_synthetic_data_generator import BaseSyntheticDataGenerator
from .data.default_synthetic_data_generator import DefaultSyntheticDataGenerator
from .data.local_data_manager import LocalDataManager
from .data.enhanced_data_flow_manager import EnhancedDataFlowManager
from .data.s3_output_registry import S3OutputInfo, ExecutionMetadata, S3OutputPathRegistry

# Configuration
from .config.default_config import DefaultConfig, default_config

# Jupyter Integration
from .jupyter.notebook_interface import NotebookInterface, NotebookSession
from .jupyter.visualization import VisualizationReporter
from .jupyter.debugger import InteractiveDebugger
from .jupyter.templates import NotebookTemplateManager
from .jupyter.advanced import AdvancedNotebookFeatures

# Production Components
from .production.e2e_validator import EndToEndValidator, E2ETestScenario, E2ETestResult
from .production.performance_optimizer import PerformanceOptimizer, PerformanceMetrics, OptimizationRecommendation
from .production.health_checker import HealthChecker, SystemHealthReport
from .production.deployment_validator import DeploymentValidator, DeploymentConfig

# Utilities
from .utils.result_models import TestResult, ExecutionResult
from .utils.execution_context import ExecutionContext

# Testing components
from ...api.dag.pipeline_dag_resolver import PipelineDAGResolver, PipelineExecutionPlan
# from .testing.pipeline_executor import PipelineExecutor, PipelineExecutionResult, StepExecutionResult
# from .testing.data_compatibility_validator import DataCompatibilityValidator, DataCompatibilityReport, DataSchemaInfo

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
    'BaseSyntheticDataGenerator',
    'DefaultSyntheticDataGenerator',
    'LocalDataManager',
    'EnhancedDataFlowManager',
    'S3OutputInfo',
    'ExecutionMetadata',
    'S3OutputPathRegistry',
    
    # Configuration
    'DefaultConfig',
    'default_config',
    
    # Jupyter Integration
    'NotebookInterface',
    'NotebookSession',
    'VisualizationReporter',
    'InteractiveDebugger',
    'NotebookTemplateManager',
    'AdvancedNotebookFeatures',
    
    # Production Components
    'EndToEndValidator',
    'E2ETestScenario',
    'E2ETestResult',
    'PerformanceOptimizer',
    'PerformanceMetrics',
    'OptimizationRecommendation',
    'HealthChecker',
    'SystemHealthReport',
    'DeploymentValidator',
    'DeploymentConfig',
    
    # Utilities
    'TestResult',
    'ExecutionResult',
    'ExecutionContext',
    
    # Pipeline testing
    'PipelineDAGResolver',
    'PipelineExecutionPlan',
    # 'PipelineExecutor',
    # 'PipelineExecutionResult',
    # 'StepExecutionResult',
    # 'DataCompatibilityValidator',
    # 'DataCompatibilityReport',
    # 'DataSchemaInfo',
    
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
