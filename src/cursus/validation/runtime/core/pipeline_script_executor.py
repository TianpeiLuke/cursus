"""Pipeline Script Executor for orchestrating script execution testing."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import argparse

from ..utils.result_models import TestResult, ExecutionResult
from ..utils.execution_context import ExecutionContext
from ..utils.error_handling import ScriptExecutionError, ScriptImportError, ConfigurationError
from .script_import_manager import ScriptImportManager
from .data_flow_manager import DataFlowManager
from ..data.local_data_manager import LocalDataManager
from ....workspace.core import WorkspaceComponentRegistry

logger = logging.getLogger(__name__)

class PipelineScriptExecutor:
    """Main orchestrator for pipeline script execution testing"""
    
    def __init__(self, workspace_dir: str = "./development/projects/project_alpha", workspace_root: str = None):
        """Initialize executor with workspace directory and optional workspace root for component discovery"""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.script_manager = ScriptImportManager()
        self.data_manager = DataFlowManager(str(self.workspace_dir))
        self.local_data_manager = LocalDataManager(str(self.workspace_dir), workspace_root)
        self.execution_history = []
        
        # Workspace-aware component registry for script discovery
        self.workspace_root = workspace_root or str(Path.cwd())
        self.workspace_registry = WorkspaceComponentRegistry(self.workspace_root)
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"PipelineScriptExecutor initialized with workspace: {self.workspace_dir}")
        logger.info(f"Workspace component registry initialized for: {self.workspace_root}")
    
    def test_script_isolation(self, script_name: str, 
                             data_source: str = "synthetic", 
                             developer_id: str = None) -> TestResult:
        """Test single script in isolation with specified data source and optional developer context"""
        
        logger.info(f"Starting isolation test for script: {script_name} (developer: {developer_id or 'any'})")
        
        try:
            # Phase 5: Enhanced implementation with workspace-aware discovery
            if data_source not in ["synthetic", "local"]:
                raise ConfigurationError(f"Data source '{data_source}' not yet implemented")
            
            # Discover script path with workspace awareness
            script_path = self._discover_script_path(script_name, developer_id)
            
            # Import script main function
            main_func = self.script_manager.import_script_main(script_path)
            
            # Prepare execution context with data source support and workspace context
            context = self._prepare_basic_execution_context(script_name, data_source, developer_id)
            
            # Execute script
            execution_result = self.script_manager.execute_script_main(main_func, context)
            
            # Record execution with workspace context
            self.execution_history.append({
                "script_name": script_name,
                "script_path": script_path,
                "developer_id": developer_id,
                "execution_result": execution_result,
                "workspace_context": {
                    "workspace_root": self.workspace_root,
                    "discovery_method": "workspace" if developer_id else "fallback"
                }
            })
            
            # Create test result
            test_result = TestResult(
                script_name=script_name,
                status="PASS" if execution_result.success else "FAIL",
                execution_time=execution_result.execution_time,
                memory_usage=execution_result.memory_usage,
                error_message=execution_result.error_message,
                recommendations=self._generate_basic_recommendations(execution_result)
            )
            
            logger.info(f"Isolation test completed for {script_name}: {test_result.status}")
            return test_result
            
        except (ScriptImportError, ScriptExecutionError, ConfigurationError) as e:
            # Handle specific runtime testing errors with appropriate categorization
            logger.error(f"Isolation test failed for {script_name} ({type(e).__name__}): {str(e)}")
            error_type = type(e).__name__
            recommendations = self._generate_error_specific_recommendations(e, error_type)
            
            return TestResult(
                script_name=script_name,
                status="FAIL",
                execution_time=0.0,
                memory_usage=0,
                error_message=f"{error_type}: {str(e)}",
                recommendations=recommendations
            )
        except FileNotFoundError as e:
            # Handle script discovery failures
            logger.error(f"Script discovery failed for {script_name}: {str(e)}")
            return TestResult(
                script_name=script_name,
                status="FAIL",
                execution_time=0.0,
                memory_usage=0,
                error_message=f"Script not found: {str(e)}",
                recommendations=[
                    f"Verify script '{script_name}' exists in one of the expected locations",
                    "Check script name spelling and path configuration",
                    "Ensure script file has .py extension"
                ]
            )
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error during isolation test for {script_name}: {str(e)}")
            return TestResult(
                script_name=script_name,
                status="FAIL",
                execution_time=0.0,
                memory_usage=0,
                error_message=f"Unexpected error: {str(e)}",
                recommendations=[f"Check system logs for details: {str(e)}"]
            )
    
    def test_pipeline_e2e(self, pipeline_dag: Dict, 
                         data_source: str = "synthetic") -> Dict[str, TestResult]:
        """Test complete pipeline end-to-end with data flow validation"""
        # Phase 1: Not implemented - stub for API compatibility
        logger.error("Pipeline end-to-end testing not implemented in Phase 1")
        raise NotImplementedError("Pipeline end-to-end testing will be implemented in Phase 2")
    
    def _discover_script_path(self, script_name: str, developer_id: str = None) -> str:
        """Workspace-aware script path discovery with fallback to basic discovery"""
        
        # Phase 5: Try workspace-aware discovery first
        try:
            components = self.workspace_registry.discover_components(developer_id)
            
            # Look for script in workspace components
            script_key = f"{developer_id}:{script_name}" if developer_id else None
            
            # Search in discovered scripts
            for key, script_info in components['scripts'].items():
                if script_key and key == script_key:
                    logger.info(f"Found script via workspace registry: {script_info['file_path']}")
                    return script_info['file_path']
                elif not developer_id and script_info['step_name'] == script_name:
                    logger.info(f"Found script via workspace registry (any developer): {script_info['file_path']}")
                    return script_info['file_path']
            
            logger.debug(f"Script {script_name} not found in workspace components, trying fallback")
            
        except Exception as e:
            logger.warning(f"Workspace script discovery failed for {script_name}: {e}")
        
        # Fallback to original basic discovery
        possible_paths = [
            f"src/cursus/steps/scripts/{script_name}.py",
            f"cursus/steps/scripts/{script_name}.py",
            f"scripts/{script_name}.py",
            f"dockers/xgboost_atoz/scripts/{script_name}.py",
            f"dockers/xgboost_atoz/{script_name}.py",
            f"dockers/pytorch_bsm_ext/scripts/{script_name}.py",
            f"dockers/pytorch_bsm_ext/{script_name}.py",
            f"dockers/xgboost_pda/scripts/{script_name}.py",
            f"dockers/xgboost_pda/{script_name}.py"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                logger.info(f"Found script via fallback discovery: {path}")
                return path
                
        raise FileNotFoundError(f"Script not found: {script_name}")
    
    def _prepare_basic_execution_context(self, script_name: str, data_source: str = "synthetic", developer_id: str = None) -> ExecutionContext:
        """Prepare basic execution context with data source support and workspace awareness"""
        
        input_dir = self.workspace_dir / "inputs" / script_name
        output_dir = self.workspace_dir / "outputs" / script_name
        
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle local data source with workspace context
        if data_source == "local":
            # Use LocalDataManager to prepare local data with workspace awareness
            local_data_paths = self.local_data_manager.get_data_for_script(script_name, developer_id)
            if local_data_paths:
                # Copy local data to input directory
                self.local_data_manager.prepare_data_for_execution(script_name, str(input_dir), developer_id)
                logger.info(f"Prepared local data for script {script_name}: {len(local_data_paths)} files")
            else:
                logger.warning(f"No local data found for script: {script_name} (developer: {developer_id or 'any'})")
        
        # Basic job args for Phase 1
        job_args = argparse.Namespace()
        job_args.verbose = True
        
        return ExecutionContext(
            input_paths={"input": str(input_dir)},
            output_paths={"output": str(output_dir)},
            environ_vars=os.environ.copy(),  # Use current environment
            job_args=job_args
        )
    
    def _generate_basic_recommendations(self, execution_result: ExecutionResult) -> list:
        """Generate basic recommendations - Phase 1 implementation"""
        recommendations = []
        
        if execution_result.execution_time > 60:
            recommendations.append("Consider optimizing script performance - execution time > 60s")
            
        if execution_result.memory_usage > 1024:  # 1GB
            recommendations.append("Consider optimizing memory usage - peak usage > 1GB")
            
        if not execution_result.success and execution_result.error_message:
            recommendations.append(f"Address execution error: {execution_result.error_message}")
            
        return recommendations
    
    def _generate_error_specific_recommendations(self, error: Exception, error_type: str) -> list:
        """Generate error-specific recommendations based on error type"""
        recommendations = []
        
        if error_type == "ScriptImportError":
            recommendations.extend([
                "Check if the script has syntax errors or missing dependencies",
                "Verify the script has a 'main' function defined",
                "Ensure all required imports are available in the environment",
                "Check if the script file is readable and properly formatted"
            ])
        elif error_type == "ScriptExecutionError":
            recommendations.extend([
                "Review script logic for runtime errors",
                "Check input data format and availability",
                "Verify script has proper error handling",
                "Ensure all required environment variables are set"
            ])
        elif error_type == "ConfigurationError":
            recommendations.extend([
                "Check data source configuration",
                "Verify workspace directory permissions",
                "Review runtime testing configuration settings",
                "Ensure all required parameters are provided"
            ])
        else:
            recommendations.append(f"Review error details for {error_type}: {str(error)}")
        
        return recommendations
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.workspace_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "execution.log"),
                logging.StreamHandler()
            ]
        )
