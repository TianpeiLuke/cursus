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
from ..execution.data_compatibility_validator import DataCompatibilityValidator, DataSchemaInfo
from ....registry.step_names import STEP_NAMES, get_canonical_name_from_file_name
from ....core.deps.specification_registry import SpecificationRegistry

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
    
    def __init__(self, workspace_dir: str = "./development/projects/project_alpha", 
                 specification_registry: Optional[SpecificationRegistry] = None):
        """Initialize with workspace directory.
        
        Args:
            workspace_dir: Directory for test workspace
            specification_registry: Optional specification registry for script discovery
        """
        self.workspace_dir = Path(workspace_dir)
        self.s3_downloader = S3DataDownloader(workspace_dir=workspace_dir)
        self.script_executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
        self.data_validator = DataCompatibilityValidator()
        self.logger = logging.getLogger(__name__)
        
        # Set up workspace manager
        config = WorkspaceConfig(base_dir=Path(workspace_dir))
        self.workspace_manager = WorkspaceManager(config)
        
        # Set up specification registry for script discovery
        self.specification_registry = specification_registry
    
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
        """Infer script path from step name using registry and script contracts.
        
        Args:
            step_name: Name of step
            scenario: Test scenario
            
        Returns:
            Path to script
        """
        try:
            # First, try to get canonical step name from the step registry
            canonical_name = get_canonical_name_from_file_name(step_name)
            self.logger.info(f"Mapped step '{step_name}' to canonical name '{canonical_name}'")
            
            # Get step information from registry
            if canonical_name in STEP_NAMES:
                step_info = STEP_NAMES[canonical_name]
                
                # Try to get script path from step specification/contract
                # This would require loading the step specification and getting the script contract
                # For now, we'll use a mapping based on the canonical name
                script_path = self._get_script_path_from_canonical_name(canonical_name)
                if script_path:
                    self.logger.info(f"Found script path for '{canonical_name}': {script_path}")
                    return script_path
                    
        except ValueError as e:
            self.logger.warning(f"Could not map step name '{step_name}' to canonical name: {e}")
        
        # Fallback to enhanced heuristic mapping
        return self._fallback_script_mapping(step_name)
    
    def _get_script_path_from_canonical_name(self, canonical_name: str) -> Optional[str]:
        """Get script path from canonical step name using specification registry.
        
        Args:
            canonical_name: Canonical step name from registry
            
        Returns:
            Script path if found, None otherwise
        """
        # Try to get script path from specification registry if available
        if self.specification_registry:
            try:
                # Look for step specification by canonical name
                specification = self.specification_registry.get_specification(canonical_name)
                if specification and specification.script_contract:
                    script_path = specification.script_contract.entry_point
                    self.logger.info(f"Found script path from specification registry: {script_path}")
                    return script_path
                else:
                    self.logger.debug(f"No script contract found for canonical name '{canonical_name}' in specification registry")
            except Exception as e:
                self.logger.warning(f"Error accessing specification registry for '{canonical_name}': {e}")
        
        # If no specification registry or no script contract found, try to derive from canonical name
        # Convert canonical name to snake_case script name
        script_name = self._canonical_name_to_script_name(canonical_name)
        if script_name:
            self.logger.info(f"Derived script name from canonical name '{canonical_name}': {script_name}")
            return script_name
        
        return None
    
    def _canonical_name_to_script_name(self, canonical_name: str) -> Optional[str]:
        """Convert canonical name to script name using naming conventions.
        
        Args:
            canonical_name: Canonical step name (e.g., "XGBoostTraining")
            
        Returns:
            Script name (e.g., "xgboost_training.py") or None if conversion fails
        """
        if not canonical_name:
            return None
        
        # Convert PascalCase to snake_case
        import re
        
        # Insert underscores before uppercase letters (except the first one)
        snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', canonical_name)
        
        # Convert to lowercase
        snake_case = snake_case.lower()
        
        # Add .py extension
        script_name = f"{snake_case}.py"
        
        self.logger.debug(f"Converted canonical name '{canonical_name}' to script name '{script_name}'")
        return script_name
    
    def _fallback_script_mapping(self, step_name: str) -> str:
        """Fallback script mapping for step names that couldn't be canonicalized.
        
        Args:
            step_name: Original step name
            
        Returns:
            Script path
        """
        # Enhanced fallback mapping
        step_lower = step_name.lower()
        
        # Pattern-based mapping
        if "preprocess" in step_lower or "tabular" in step_lower:
            return "tabular_preprocessing.py"
        elif "xgboost" in step_lower and "train" in step_lower:
            return "xgboost_training.py"
        elif "xgboost" in step_lower and ("eval" in step_lower or "evaluation" in step_lower):
            return "xgboost_model_eval.py"
        elif "xgboost" in step_lower and "model" in step_lower:
            return "xgboost_model.py"
        elif "pytorch" in step_lower and "train" in step_lower:
            return "pytorch_training.py"
        elif "pytorch" in step_lower and "model" in step_lower:
            return "pytorch_model.py"
        elif "calibrat" in step_lower:
            return "model_calibration.py"
        elif "package" in step_lower:
            return "package.py"
        elif "register" in step_lower or "registration" in step_lower:
            return "registration.py"
        elif "payload" in step_lower:
            return "payload.py"
        elif "transform" in step_lower:
            return "batch_transform.py"
        elif "cradle" in step_lower or "data_load" in step_lower:
            return "cradle_data_loading.py"
        elif "risk" in step_lower and "table" in step_lower:
            return "risk_table_mapping.py"
        elif "currency" in step_lower:
            return "currency_conversion.py"
        elif "dummy" in step_lower and "train" in step_lower:
            return "dummy_training.py"
        else:
            # Default to using step name as script name
            self.logger.warning(f"No specific script mapping found for step '{step_name}', using default")
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
            'warnings': [],
            'schema_validation': {},
            'data_quality_metrics': {}
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
        
        # Enhanced data validation using DataCompatibilityValidator
        try:
            schema_validation = self._validate_output_schemas(step_name, step_result)
            validation_results['schema_validation'] = schema_validation
            
            # Add schema issues to main validation results
            if schema_validation.get('issues'):
                validation_results['issues'].extend(schema_validation['issues'])
                validation_results['passed'] = False
            
            if schema_validation.get('warnings'):
                validation_results['warnings'].extend(schema_validation['warnings'])
            
            # Data quality validation
            quality_metrics = self._validate_data_quality(step_name, step_result)
            validation_results['data_quality_metrics'] = quality_metrics
            
            # Add quality issues
            if quality_metrics.get('issues'):
                validation_results['issues'].extend(quality_metrics['issues'])
                validation_results['passed'] = False
                
            if quality_metrics.get('warnings'):
                validation_results['warnings'].extend(quality_metrics['warnings'])
                
        except Exception as e:
            validation_results['warnings'].append(f"Error during enhanced validation: {str(e)}")
        
        return validation_results
    
    def _validate_output_schemas(self, step_name: str, step_result: Any) -> Dict[str, Any]:
        """Validate output file schemas using DataCompatibilityValidator.
        
        Args:
            step_name: Name of step
            step_result: Result from step execution
            
        Returns:
            Schema validation results
        """
        schema_validation = {
            'issues': [],
            'warnings': [],
            'schemas_analyzed': {}
        }
        
        # Get output files from step result
        output_files = getattr(step_result, 'output_files', {})
        
        for file_name, file_path in output_files.items():
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists():
                    # Analyze file schema
                    schema_info = self.data_validator.analyze_file(file_path_obj)
                    schema_validation['schemas_analyzed'][file_name] = {
                        'format': schema_info.data_format,
                        'columns': len(schema_info.columns),
                        'rows': schema_info.num_rows,
                        'column_types': schema_info.column_types
                    }
                    
                    # Basic schema validation rules
                    if schema_info.num_rows == 0:
                        schema_validation['warnings'].append(
                            f"Output file {file_name} is empty"
                        )
                    
                    if len(schema_info.columns) == 0:
                        schema_validation['issues'].append(
                            f"Output file {file_name} has no columns"
                        )
                        
                else:
                    schema_validation['issues'].append(
                        f"Output file {file_name} does not exist: {file_path}"
                    )
                    
            except Exception as e:
                schema_validation['warnings'].append(
                    f"Error analyzing schema for {file_name}: {str(e)}"
                )
        
        return schema_validation
    
    def _validate_data_quality(self, step_name: str, step_result: Any) -> Dict[str, Any]:
        """Validate data quality metrics.
        
        Args:
            step_name: Name of step
            step_result: Result from step execution
            
        Returns:
            Data quality validation results
        """
        quality_metrics = {
            'issues': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Get output files from step result
        output_files = getattr(step_result, 'output_files', {})
        
        for file_name, file_path in output_files.items():
            try:
                file_path_obj = Path(file_path)
                if file_path_obj.exists() and file_path_obj.suffix.lower() == '.csv':
                    # Analyze CSV data quality
                    df = pd.read_csv(file_path_obj, nrows=1000)  # Sample for performance
                    
                    file_metrics = {
                        'completeness': self._calculate_completeness(df),
                        'uniqueness': self._calculate_uniqueness(df),
                        'consistency': self._calculate_consistency(df)
                    }
                    
                    quality_metrics['metrics'][file_name] = file_metrics
                    
                    # Check quality thresholds
                    if file_metrics['completeness'] < 0.95:
                        quality_metrics['warnings'].append(
                            f"Low data completeness in {file_name}: {file_metrics['completeness']:.2%}"
                        )
                    
                    if file_metrics['uniqueness'] < 0.90:
                        quality_metrics['warnings'].append(
                            f"Low data uniqueness in {file_name}: {file_metrics['uniqueness']:.2%}"
                        )
                        
            except Exception as e:
                quality_metrics['warnings'].append(
                    f"Error analyzing data quality for {file_name}: {str(e)}"
                )
        
        return quality_metrics
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness (non-null ratio).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Completeness ratio (0.0 to 1.0)
        """
        if df.empty:
            return 0.0
        
        total_cells = df.size
        non_null_cells = df.count().sum()
        
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
    def _calculate_uniqueness(self, df: pd.DataFrame) -> float:
        """Calculate data uniqueness (unique values ratio).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Uniqueness ratio (0.0 to 1.0)
        """
        if df.empty:
            return 0.0
        
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates())
        
        return unique_rows / total_rows if total_rows > 0 else 0.0
    
    def _calculate_consistency(self, df: pd.DataFrame) -> float:
        """Calculate data consistency (format consistency).
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Consistency ratio (0.0 to 1.0)
        """
        if df.empty:
            return 0.0
        
        # Simple consistency check: ratio of valid data types
        consistent_columns = 0
        total_columns = len(df.columns)
        
        for column in df.columns:
            try:
                # Check if column has consistent data type
                if df[column].dtype != 'object':
                    consistent_columns += 1
                else:
                    # For object columns, check if values are consistent
                    non_null_values = df[column].dropna()
                    if len(non_null_values) > 0:
                        # Check if all values have similar type
                        first_type = type(non_null_values.iloc[0])
                        consistent_type_ratio = sum(
                            isinstance(val, first_type) for val in non_null_values
                        ) / len(non_null_values)
                        
                        if consistent_type_ratio > 0.95:
                            consistent_columns += 1
                            
            except Exception:
                # If we can't analyze, assume inconsistent
                pass
        
        return consistent_columns / total_columns if total_columns > 0 else 0.0
    
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
