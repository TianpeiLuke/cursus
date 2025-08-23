"""Pipeline Script Executor for orchestrating script execution testing."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import argparse

from ..utils.result_models import TestResult, ExecutionResult
from ..utils.execution_context import ExecutionContext
from ..utils.error_handling import ScriptExecutionError
from .script_import_manager import ScriptImportManager
from .data_flow_manager import DataFlowManager
from ..data.local_data_manager import LocalDataManager

logger = logging.getLogger(__name__)

class PipelineScriptExecutor:
    """Main orchestrator for pipeline script execution testing"""
    
    def __init__(self, workspace_dir: str = "./pipeline_testing"):
        """Initialize executor with workspace directory"""
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        self.script_manager = ScriptImportManager()
        self.data_manager = DataFlowManager(str(self.workspace_dir))
        self.local_data_manager = LocalDataManager(str(self.workspace_dir))
        self.execution_history = []
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"PipelineScriptExecutor initialized with workspace: {self.workspace_dir}")
    
    def test_script_isolation(self, script_name: str, 
                             data_source: str = "synthetic") -> TestResult:
        """Test single script in isolation with specified data source"""
        
        logger.info(f"Starting isolation test for script: {script_name}")
        
        try:
            # Phase 1: Basic implementation with synthetic and local data
            if data_source not in ["synthetic", "local"]:
                raise NotImplementedError(f"Data source '{data_source}' not yet implemented")
            
            # Discover script path (basic implementation)
            script_path = self._discover_script_path(script_name)
            
            # Import script main function
            main_func = self.script_manager.import_script_main(script_path)
            
            # Prepare execution context with data source support
            context = self._prepare_basic_execution_context(script_name, data_source)
            
            # Execute script
            execution_result = self.script_manager.execute_script_main(main_func, context)
            
            # Record execution
            self.execution_history.append({
                "script_name": script_name,
                "script_path": script_path,
                "execution_result": execution_result
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
            
        except Exception as e:
            logger.error(f"Isolation test failed for {script_name}: {str(e)}")
            return TestResult(
                script_name=script_name,
                status="FAIL",
                execution_time=0.0,
                memory_usage=0,
                error_message=str(e),
                recommendations=[f"Check script implementation: {str(e)}"]
            )
    
    def test_pipeline_e2e(self, pipeline_dag: Dict, 
                         data_source: str = "synthetic") -> Dict[str, TestResult]:
        """Test complete pipeline end-to-end with data flow validation"""
        # Phase 1: Not implemented - stub for API compatibility
        logger.error("Pipeline end-to-end testing not implemented in Phase 1")
        raise NotImplementedError("Pipeline end-to-end testing will be implemented in Phase 2")
    
    def _discover_script_path(self, script_name: str) -> str:
        """Basic script path discovery - Phase 1 implementation"""
        # Simple path resolution - will be enhanced in later phases
        possible_paths = [
            f"src/cursus/steps/scripts/{script_name}.py",
            f"cursus/steps/scripts/{script_name}.py",
            f"scripts/{script_name}.py",
            f"dockers/xgboost_atoz/scripts/{script_name}.py",
            f"dockers/pytorch_bsm_ext/scripts/{script_name}.py",
            f"dockers/xgboost_pda/scripts/{script_name}.py"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
                
        raise FileNotFoundError(f"Script not found: {script_name}")
    
    def _prepare_basic_execution_context(self, script_name: str, data_source: str = "synthetic") -> ExecutionContext:
        """Prepare basic execution context with data source support - Phase 1 implementation"""
        
        input_dir = self.workspace_dir / "inputs" / script_name
        output_dir = self.workspace_dir / "outputs" / script_name
        
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle local data source
        if data_source == "local":
            # Use LocalDataManager to prepare local data
            local_data_paths = self.local_data_manager.get_data_for_script(script_name)
            if local_data_paths:
                # Copy local data to input directory
                self.local_data_manager.prepare_data_for_execution(script_name, str(input_dir))
                logger.info(f"Prepared local data for script {script_name}: {len(local_data_paths)} files")
            else:
                logger.warning(f"No local data found for script: {script_name}")
        
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
