"""Real data testing system using S3 pipeline outputs."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import json
import logging

from .s3_data_downloader import S3DataDownloader, S3DataSource
from .workspace_manager import WorkspaceManager, WorkspaceConfig
from ..core.pipeline_script_executor import PipelineScriptExecutor
# from ..testing.data_compatibility_validator import DataCompatibilityValidator

class RealDataTestScenario(BaseModel):
    """Test scenario using real pipeline data."""
    scenario_name: str
    pipeline_name: str
    s3_data_source: S3DataSource
    test_steps: List[str]
    validation_rules: Dict[str, Any] = Field(default_factory=dict)

class RealDataTestResult(BaseModel):
    """Result of real data testing."""
    scenario_name: str
    success: bool
    step_results: Dict[str, Any] = Field(default_factory=dict)
    data_validation_results: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    error_details: Optional[str] = None

class ProductionValidationRule(BaseModel):
    """Rule for validating production data."""
    rule_name: str
    rule_type: str  # 'statistical', 'schema', 'business_logic'
    parameters: Dict[str, Any]
    severity: str  # 'error', 'warning', 'info'

class RealDataTester:
    """Tests pipeline scripts using real production data."""
    
    def __init__(self, workspace_dir: str = "./pipeline_testing"):
        """Initialize with workspace directory.
        
        Args:
            workspace_dir: Directory for test workspace
        """
        self.workspace_dir = Path(workspace_dir)
        self.s3_downloader = S3DataDownloader(workspace_dir=workspace_dir)
        self.script_executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
        # self.data_validator = DataCompatibilityValidator()
        self.logger = logging.getLogger(__name__)
        
        # Set up workspace manager
        config = WorkspaceConfig(base_dir=Path(workspace_dir))
        self.workspace_manager = WorkspaceManager(config)
    
    def discover_test_scenarios(self, bucket: str, pipeline_name: str, 
                              limit: int = 5) -> List[RealDataTestScenario]:
        """Discover available test scenarios from S3.
        
        Args:
            bucket: S3 bucket name
            pipeline_name: Name of the pipeline
            limit: Maximum number of scenarios to discover
            
        Returns:
            List of discovered test scenarios
        """
        # Discover available data sources
        data_sources = self.s3_downloader.discover_pipeline_data(bucket, pipeline_name)
        
        # Limit the number of sources
        data_sources = data_sources[:limit]
        
        # Create scenarios
        scenarios = []
        for data_source in data_sources:
            scenario = RealDataTestScenario(
                scenario_name=f"{pipeline_name}_{data_source.execution_id}",
                pipeline_name=pipeline_name,
                s3_data_source=data_source,
                test_steps=list(data_source.step_outputs.keys()),
                validation_rules=self._create_default_validation_rules(data_source.step_outputs.keys())
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def create_test_scenario(self, pipeline_name: str, bucket: str,
                           execution_id: Optional[str] = None,
                           test_steps: Optional[List[str]] = None) -> RealDataTestScenario:
        """Create a test scenario from S3 pipeline data.
        
        Args:
            pipeline_name: Name of pipeline to test
            bucket: S3 bucket name
            execution_id: Optional specific execution ID
            test_steps: Optional list of steps to test
            
        Returns:
            Test scenario object
        """
        # Discover available data
        data_sources = self.s3_downloader.discover_pipeline_data(
            bucket, pipeline_name, execution_id
        )
        
        if not data_sources:
            raise ValueError(f"No data found for pipeline {pipeline_name}")
        
        # Use the most recent execution
        data_source = data_sources[0]
        
        # Default to testing all available steps
        if test_steps is None:
            test_steps = list(data_source.step_outputs.keys())
        else:
            # Validate steps exist
            for step in test_steps:
                if step not in data_source.step_outputs:
                    raise ValueError(f"Step {step} not found in pipeline data")
        
        return RealDataTestScenario(
            scenario_name=f"{pipeline_name}_{data_source.execution_id}",
            pipeline_name=pipeline_name,
            s3_data_source=data_source,
            test_steps=test_steps,
            validation_rules=self._create_default_validation_rules(test_steps)
        )
    
    def _create_default_validation_rules(self, step_names: List[str]) -> Dict[str, Any]:
        """Create default validation rules for the steps.
        
        Args:
            step_names: List of step names
            
        Returns:
            Dictionary of validation rules by step
        """
        rules = {}
        
        for step_name in step_names:
            rules[step_name] = {
                "expected_outputs": [],  # Will be populated during execution
                "quality_thresholds": {
                    "execution_time_seconds": 300,  # 5 minutes max
                    "memory_usage_mb": 2048,  # 2GB max
                }
            }
        
        return rules
    
    def execute_test_scenario(self, scenario: RealDataTestScenario) -> RealDataTestResult:
        """Execute a real data test scenario.
        
        Args:
            scenario: Test scenario to execute
            
        Returns:
            Test result object
        """
        step_results = {}
        data_validation_results = {}
        performance_metrics = {}
        
        try:
            # Set up workspace for this scenario
            scenario_workspace = f"real_data_{scenario.scenario_name}"
            self.workspace_manager.setup_workspace(scenario_workspace)
            
            # Download required data
            for step_name in scenario.test_steps:
                self.logger.info(f"Testing step: {step_name}")
                
                # Download step data
                download_results = self.s3_downloader.download_step_data(
                    scenario.s3_data_source, step_name
                )
                
                # Validate downloads
                failed_downloads = [
                    key for key, result in download_results.items() 
                    if not result.success
                ]
                
                if failed_downloads:
                    return RealDataTestResult(
                        scenario_name=scenario.scenario_name,
                        success=False,
                        error_details=f"Failed to download data for {step_name}: {failed_downloads}"
                    )
                
                # Prepare step inputs from downloaded data
                step_inputs = self._prepare_step_inputs_from_s3(
                    step_name, download_results
                )
                
                # Execute step with real data
                step_result = self._execute_step_with_real_data(
                    step_name, step_inputs, scenario
                )
                
                step_results[step_name] = step_result
                
                # Validate step outputs against real data expectations
                validation_result = self._validate_step_against_real_data(
                    step_name, step_result, scenario
                )
                
                data_validation_results[step_name] = validation_result
                
                # Collect performance metrics
                performance_metrics[step_name] = {
                    'execution_time': step_result.execution_time,
                    'memory_usage': step_result.memory_usage,
                    'data_size_processed': self._estimate_data_size_processed(download_results)
                }
                
                # Check step success
                if step_result.status != "PASS":
                    return RealDataTestResult(
                        scenario_name=scenario.scenario_name,
                        success=False,
                        step_results=step_results,
                        data_validation_results=data_validation_results,
                        performance_metrics=performance_metrics,
                        error_details=f"Step {step_name} failed: {step_result.error_message}"
                    )
            
            # All steps passed
            return RealDataTestResult(
                scenario_name=scenario.scenario_name,
                success=True,
                step_results=step_results,
                data_validation_results=data_validation_results,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            self.logger.exception(f"Error executing test scenario: {e}")
            return RealDataTestResult(
                scenario_name=scenario.scenario_name,
                success=False,
                step_results=step_results,
                data_validation_results=data_validation_results,
                performance_metrics=performance_metrics,
                error_details=str(e)
            )
    
    def _prepare_step_inputs_from_s3(self, step_name: str, 
                                   download_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare step inputs from downloaded S3 data.
        
        Args:
            step_name: Name of step
            download_results: Results from S3 download
            
        Returns:
            Dictionary of inputs for the step
        """
        inputs = {}
        
        for s3_key, result in download_results.items():
            if result.success and result.local_path:
                # Determine input type based on file extension
                file_path = result.local_path
                
                if file_path.suffix.lower() == '.csv':
                    inputs[file_path.stem] = str(file_path)
                elif file_path.suffix.lower() == '.json':
                    with open(file_path) as f:
                        inputs[file_path.stem] = json.load(f)
                elif file_path.suffix.lower() in ['.parquet', '.pq']:
                    inputs[file_path.stem] = str(file_path)
                else:
                    # Generic file path
                    inputs[file_path.stem] = str(file_path)
        
        return inputs
    
    def _execute_step_with_real_data(self, step_name: str, step_inputs: Dict[str, Any], 
                                   scenario: RealDataTestScenario) -> Any:
        """Execute a step with real data inputs.
        
        Args:
            step_name: Name of step to execute
            step_inputs: Inputs for the step
            scenario: Test scenario
            
        Returns:
            Step execution result
        """
        # Look for the script path in the S3 data structure
        script_path = self._infer_script_path(step_name, scenario)
        
        # Execute the script with real data
        result = self.script_executor.test_script_isolation(
            script_path, 
            input_data=step_inputs
        )
        
        return result
    
    def _infer_script_path(self, step_name: str, scenario: RealDataTestScenario) -> str:
        """Infer script path from step name and scenario.
        
        Args:
            step_name: Name of step
            scenario: Test scenario
            
        Returns:
            Path to script
        """
        # Try to infer script name from step name
        # This is a simple heuristic - in real systems there would be more sophisticated mapping
        if step_name == "data_preprocessing":
            return "data_preprocessing.py"
        elif step_name == "model_training":
            return "model_training.py"
        elif step_name == "model_evaluation":
            return "model_evaluation.py"
        elif "calibration" in step_name.lower():
            return "model_calibration.py"
        else:
            # Default to using step name as script name
            return f"{step_name}.py"
    
    def _validate_step_against_real_data(self, step_name: str, step_result: Any,
                                       scenario: RealDataTestScenario) -> Dict[str, Any]:
        """Validate step results against real data expectations.
        
        Args:
            step_name: Name of step
            step_result: Result from step execution
            scenario: Test scenario
            
        Returns:
            Validation result
        """
        validation_rules = scenario.validation_rules.get(step_name, {})
        validation_results = {
            'passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check quality metrics
        quality_thresholds = validation_rules.get('quality_thresholds', {})
        
        # Check execution time
        max_execution_time = quality_thresholds.get('execution_time_seconds', 300)
        if step_result.execution_time > max_execution_time:
            validation_results['warnings'].append(
                f"Execution time exceeds threshold: {step_result.execution_time:.2f}s > {max_execution_time}s"
            )
        
        # Check memory usage
        max_memory_usage = quality_thresholds.get('memory_usage_mb', 2048)
        if step_result.memory_usage > max_memory_usage:
            validation_results['warnings'].append(
                f"Memory usage exceeds threshold: {step_result.memory_usage}MB > {max_memory_usage}MB"
            )
        
        # Check for errors
        if step_result.error_message:
            validation_results['issues'].append(
                f"Error in step execution: {step_result.error_message}"
            )
            validation_results['passed'] = False
        
        return validation_results
    
    def _estimate_data_size_processed(self, download_results: Dict[str, Any]) -> int:
        """Estimate the total size of data processed.
        
        Args:
            download_results: Results from S3 download
            
        Returns:
            Estimated data size in bytes
        """
        total_size = 0
        for _, result in download_results.items():
            if result.success and result.size_bytes:
                total_size += result.size_bytes
                
        return total_size
