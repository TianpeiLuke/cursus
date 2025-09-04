"""Pipeline executor for end-to-end testing."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import logging
import os
from pydantic import BaseModel, Field
from datetime import datetime

from ....api.dag.pipeline_dag_resolver import PipelineDAGResolver, PipelineExecutionPlan
from ....api.dag.workspace_dag import WorkspaceAwareDAG
from .data_compatibility_validator import DataCompatibilityValidator, DataCompatibilityReport
from ..core.pipeline_script_executor import PipelineScriptExecutor
from ..data.enhanced_data_flow_manager import EnhancedDataFlowManager
from ..data.s3_output_registry import S3OutputPathRegistry, S3OutputInfo, ExecutionMetadata
from ..utils.result_models import TestResult

class StepExecutionResult(BaseModel):
    """Result of a single step execution."""
    step_name: str
    status: str  # SUCCESS, FAILURE, SKIPPED
    execution_time: float
    memory_usage: int
    error_message: Optional[str] = None
    outputs: Dict[str, Any] = Field(default_factory=dict)
    data_validation_report: Optional[DataCompatibilityReport] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class PipelineExecutionResult(BaseModel):
    """Result of pipeline execution."""
    success: bool
    completed_steps: List[StepExecutionResult] = Field(default_factory=list)
    execution_plan: Optional[PipelineExecutionPlan] = None
    error: Optional[str] = None
    total_duration: float = 0.0
    memory_peak: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)

class PipelineExecutor:
    """Executes entire pipeline with data flow validation."""
    
    def __init__(self, workspace_dir: str = "./development/projects/project_alpha", testing_mode: str = "pre_execution", workspace_root: str = None):
        """Initialize with workspace directory and testing mode.
        
        Args:
            workspace_dir: Directory for test workspace
            testing_mode: "pre_execution" or "post_execution" testing mode
            workspace_root: Optional workspace root for workspace-aware execution
        """
        self.workspace_dir = Path(workspace_dir)
        self.testing_mode = testing_mode
        self.workspace_root = workspace_root
        
        # Initialize script executor with workspace awareness
        self.script_executor = PipelineScriptExecutor(
            workspace_dir=workspace_dir, 
            workspace_root=workspace_root
        )
        
        self.data_validator = DataCompatibilityValidator(workspace_root)
        self.enhanced_data_flow_manager = EnhancedDataFlowManager(workspace_dir, testing_mode)
        self.s3_output_registry = S3OutputPathRegistry()
        self.execution_results = {}
        self.logger = logging.getLogger(__name__)
        self.resolver = None  # Will be set during pipeline execution
        
        # Workspace-aware execution tracking
        self.workspace_execution_context = {
            'workspace_root': workspace_root,
            'cross_workspace_dependencies': [],
            'developer_execution_stats': {}
        }
    
    def execute_pipeline(self, dag, data_source: str = "synthetic", 
                        config_path: Optional[str] = None,
                        available_configs: Optional[Dict[str, Any]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> PipelineExecutionResult:
        """Execute complete pipeline with data flow validation.
        
        Args:
            dag: PipelineDAG or WorkspaceAwareDAG object representing the pipeline
            data_source: Source of data for testing ("synthetic" or "s3")
            config_path: Path to configuration file (optional)
            available_configs: Pre-loaded configuration instances (optional)
            metadata: Configuration metadata for enhanced resolution (optional)
            
        Returns:
            PipelineExecutionResult object with execution results
        """
        start_time = time.time()
        memory_peak = 0
        
        try:
            # Phase 5: Handle WorkspaceAwareDAG with enhanced workspace context
            is_workspace_dag = isinstance(dag, WorkspaceAwareDAG)
            
            if is_workspace_dag:
                self.logger.info("Executing workspace-aware pipeline")
                self._prepare_workspace_execution_context(dag)
            
            # Create resolver with enhanced configuration support
            resolver = PipelineDAGResolver(
                dag=dag,
                config_path=config_path,
                available_configs=available_configs,
                metadata=metadata
            )
            self.resolver = resolver  # Store resolver for use in other methods
            execution_plan = resolver.create_execution_plan()
            
            # Log configuration resolution info if available
            if resolver.config_resolver:
                config_preview = resolver.get_config_resolution_preview()
                if config_preview:
                    self.logger.info(f"Configuration resolution preview: {config_preview}")
            
            # Enhanced diagnostic information using resolver capabilities
            self.logger.info("Pipeline dependency analysis:")
            for step_name in execution_plan.execution_order:
                dependencies = resolver.get_step_dependencies(step_name)
                dependent_steps = resolver.get_dependent_steps(step_name)
                self.logger.debug(f"  {step_name}: deps={dependencies}, dependents={dependent_steps}")
                
                # Log contract discovery status
                contract = resolver._discover_step_contract(step_name)
                if contract:
                    self.logger.debug(f"  {step_name}: contract found - {contract.entry_point}")
                else:
                    self.logger.debug(f"  {step_name}: no contract found, using fallback")
            
            # Validate DAG integrity
            integrity_issues = resolver.validate_dag_integrity()
            if integrity_issues:
                error_msg = f"Pipeline DAG has integrity issues: {integrity_issues}"
                self.logger.error(error_msg)
                return PipelineExecutionResult(
                    success=False,
                    error=error_msg,
                    execution_plan=execution_plan,
                    total_duration=time.time() - start_time,
                    memory_peak=memory_peak
                )
            
            self.logger.info(f"Starting pipeline execution with {len(execution_plan.execution_order)} steps")
            self.logger.info(f"Execution order: {', '.join(execution_plan.execution_order)}")
            
            # Execute steps in topological order
            results = []
            step_outputs = {}
            
            for step_name in execution_plan.execution_order:
                try:
                    # Phase 5: Get workspace context for step if available
                    workspace_step_info = None
                    developer_id = None
                    
                    if is_workspace_dag:
                        workspace_step_info = dag.get_workspace_step(step_name)
                        if workspace_step_info:
                            developer_id = workspace_step_info['developer_id']
                            self.logger.info(f"Executing workspace step: {step_name} (developer: {developer_id})")
                        else:
                            self.logger.info(f"Executing step: {step_name} (non-workspace)")
                    else:
                        self.logger.info(f"Executing step: {step_name}")
                    
                    # Prepare step inputs from previous outputs
                    step_inputs = self._prepare_step_inputs(
                        step_name, execution_plan, step_outputs
                    )
                    
                    # Execute step with workspace context
                    step_result = self._execute_step(
                        step_name, 
                        execution_plan.step_configs.get(step_name, {}),
                        step_inputs,
                        data_source,
                        developer_id=developer_id,
                        workspace_step_info=workspace_step_info
                    )
                    
                    # Update peak memory usage
                    memory_peak = max(memory_peak, step_result.memory_usage)
                    
                    # Validate outputs with next steps
                    data_validation_report = self._validate_step_outputs(
                        step_name, step_result.outputs, execution_plan
                    )
                    step_result.data_validation_report = data_validation_report
                    
                    # Store outputs for next steps
                    step_outputs[step_name] = step_result.outputs
                    results.append(step_result)
                    
                    # Check if step failed
                    if step_result.status != "SUCCESS":
                        self.logger.error(f"Step {step_name} failed: {step_result.error_message}")
                        return PipelineExecutionResult(
                            success=False,
                            error=f"Pipeline failed at step {step_name}: {step_result.error_message}",
                            completed_steps=results,
                            execution_plan=execution_plan,
                            total_duration=time.time() - start_time,
                            memory_peak=memory_peak
                        )
                    
                except Exception as e:
                    self.logger.exception(f"Error executing step {step_name}")
                    return PipelineExecutionResult(
                        success=False,
                        error=f"Pipeline failed at step {step_name}: {str(e)}",
                        completed_steps=results,
                        execution_plan=execution_plan,
                        total_duration=time.time() - start_time,
                        memory_peak=memory_peak
                    )
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            self.logger.info(f"Pipeline execution completed successfully in {total_duration:.2f} seconds")
            
            return PipelineExecutionResult(
                success=True,
                completed_steps=results,
                execution_plan=execution_plan,
                total_duration=total_duration,
                memory_peak=memory_peak
            )
            
        except Exception as e:
            self.logger.exception("Pipeline execution failed")
            end_time = time.time()
            return PipelineExecutionResult(
                success=False,
                error=f"Pipeline execution failed: {str(e)}",
                total_duration=end_time - start_time,
                memory_peak=memory_peak
            )
    
    def _prepare_step_inputs(self, step_name: str, 
                           execution_plan: PipelineExecutionPlan, 
                           step_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare inputs for a step from previous step outputs using enhanced data flow mapping.
        
        Args:
            step_name: Name of the step
            execution_plan: PipelineExecutionPlan object
            step_outputs: Dictionary of step outputs from previous steps
            
        Returns:
            Dictionary with step inputs
        """
        inputs = {}
        
        # Get input mapping from enhanced data flow map
        input_mapping = execution_plan.data_flow_map.get(step_name, {})
        
        if input_mapping:
            # Use contract-based data flow mapping
            for input_key, output_ref in input_mapping.items():
                if ":" in output_ref:
                    src_step, output_key = output_ref.split(":", 1)
                    if src_step in step_outputs and output_key in step_outputs[src_step]:
                        inputs[input_key] = step_outputs[src_step][output_key]
                        self.logger.debug(f"Mapped {src_step}:{output_key} -> {step_name}:{input_key}")
                    else:
                        self.logger.warning(f"Could not find output {output_ref} for input {input_key}")
        else:
            # Fallback to generic dependency-based mapping
            dependencies = execution_plan.dependencies.get(step_name, [])
            for i, dep_step in enumerate(dependencies):
                if dep_step in step_outputs:
                    # Use first available output from dependency
                    dep_outputs = step_outputs[dep_step]
                    if dep_outputs:
                        first_output_key = next(iter(dep_outputs.keys()))
                        inputs[f"input_{i}"] = dep_outputs[first_output_key]
                        self.logger.debug(f"Fallback mapping: {dep_step}:{first_output_key} -> {step_name}:input_{i}")
        
        self.logger.info(f"Prepared {len(inputs)} inputs for step {step_name}: {list(inputs.keys())}")
        return inputs
    
    def _execute_step(self, step_name: str, step_config: Dict[str, Any], 
                    step_inputs: Dict[str, Any], 
                    data_source: str,
                    developer_id: str = None,
                    workspace_step_info: Dict[str, Any] = None) -> StepExecutionResult:
        """Execute a single step with inputs and workspace context.
        
        Args:
            step_name: Name of the step
            step_config: Configuration for the step
            step_inputs: Inputs for the step
            data_source: Source of data for testing
            developer_id: Optional developer ID for workspace-aware execution
            workspace_step_info: Optional workspace step information
            
        Returns:
            StepExecutionResult object with execution results
        """
        start_time = time.time()
        
        try:
            # Enhancement 2: Use contract for enhanced validation and setup
            contract = None
            if self.resolver:
                contract = self.resolver._discover_step_contract(step_name)
                if contract:
                    # Validate required environment variables
                    missing_env_vars = [var for var in contract.required_env_vars 
                                       if var not in os.environ]
                    if missing_env_vars:
                        self.logger.warning(f"Missing required env vars for {step_name}: {missing_env_vars}")
                        # In a real implementation, you might want to fail here or set defaults
                    
                    # Log optional environment variables status
                    for var, default in contract.optional_env_vars.items():
                        if var not in os.environ:
                            self.logger.debug(f"Optional env var {var} not set for {step_name}, default: {default}")
            
            # Get script path using enhanced contract discovery
            script_path = self._get_script_path(step_name, step_config)
            
            # Create input/output paths for the step
            input_path = self.workspace_dir / "inputs" / step_name
            output_path = self.workspace_dir / "outputs" / step_name
            input_path.mkdir(parents=True, exist_ok=True)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Enhancement 3: Use contract paths for better input/output setup
            if contract:
                # Use contract-defined input/output paths
                input_paths = {}
                for input_name, input_path_template in contract.expected_input_paths.items():
                    # Map contract paths to workspace paths
                    workspace_input_path = self.workspace_dir / "inputs" / step_name / input_name
                    workspace_input_path.mkdir(parents=True, exist_ok=True)
                    input_paths[input_name] = str(workspace_input_path)
                
                output_paths = {}
                for output_name, output_path_template in contract.expected_output_paths.items():
                    # Map contract paths to workspace paths
                    workspace_output_path = self.workspace_dir / "outputs" / step_name / output_name
                    workspace_output_path.mkdir(parents=True, exist_ok=True)
                    output_paths[output_name] = str(workspace_output_path)
                
                self.logger.debug(f"Using contract-based paths for {step_name}: inputs={list(input_paths.keys())}, outputs={list(output_paths.keys())}")
            else:
                # Fallback to generic paths
                input_paths = {
                    "input": str(input_path)
                }
                
                output_paths = {
                    "output": str(output_path)
                }
            
            # Add any specific input/output paths from step_config (override contract paths if needed)
            if "input_paths" in step_config:
                input_paths.update(step_config["input_paths"])
            
            if "output_paths" in step_config:
                output_paths.update(step_config["output_paths"])
            
            # Execute the script with workspace context
            result = self.script_executor.test_script_isolation(
                script_path, 
                data_source=data_source,
                developer_id=developer_id
            )
            
            end_time = time.time()
            
            # Build step execution result with workspace context
            step_result = StepExecutionResult(
                step_name=step_name,
                status="SUCCESS" if result.status == "PASS" else "FAILURE",
                execution_time=end_time - start_time,
                memory_usage=result.memory_usage,
                error_message=result.error_message,
                outputs={
                    "output": {
                        "format": "unknown",
                        "path": str(output_path),
                        "size": 0
                    }
                }
            )
            
            # Phase 5: Track workspace execution statistics
            if developer_id and workspace_step_info:
                if developer_id not in self.workspace_execution_context['developer_execution_stats']:
                    self.workspace_execution_context['developer_execution_stats'][developer_id] = {
                        'steps_executed': 0,
                        'total_execution_time': 0.0,
                        'successful_steps': 0,
                        'failed_steps': 0
                    }
                
                stats = self.workspace_execution_context['developer_execution_stats'][developer_id]
                stats['steps_executed'] += 1
                stats['total_execution_time'] += step_result.execution_time
                
                if step_result.status == "SUCCESS":
                    stats['successful_steps'] += 1
                else:
                    stats['failed_steps'] += 1
            
            return step_result
            
        except Exception as e:
            end_time = time.time()
            
            return StepExecutionResult(
                step_name=step_name,
                status="FAILURE",
                execution_time=end_time - start_time,
                memory_usage=0,
                error_message=str(e)
            )
    
    def _validate_step_outputs(self, step_name: str, step_outputs: Dict[str, Any], 
                             execution_plan: PipelineExecutionPlan) -> Optional[DataCompatibilityReport]:
        """Validate step outputs against next steps' requirements.
        
        Args:
            step_name: Name of the step
            step_outputs: Outputs from the step
            execution_plan: PipelineExecutionPlan object
            
        Returns:
            DataCompatibilityReport object if validation was performed, None otherwise
        """
        # Find steps that depend on this one
        dependent_steps = []
        for other_step, deps in execution_plan.dependencies.items():
            if step_name in deps:
                dependent_steps.append(other_step)
        
        if not dependent_steps:
            # No dependent steps to validate against
            return None
        
        # For now, just validate against the first dependent step
        # In a real implementation, we would validate against all dependent steps
        next_step = dependent_steps[0]
        next_step_config = execution_plan.step_configs.get(next_step, {})
        
        # Get input spec for next step
        input_spec = {
            "required_files": list(next_step_config.get("inputs", {}).keys()),
            "file_formats": {
                key: "unknown" for key in next_step_config.get("inputs", {})
            },
            "schemas": {}
        }
        
        # Validate outputs against input spec with workspace context
        producer_workspace_info = None
        consumer_workspace_info = None
        
        # Get workspace info if available
        if hasattr(self, 'workspace_execution_context') and self.workspace_execution_context.get('workspace_root'):
            # Try to get workspace info for the steps (simplified for this implementation)
            producer_workspace_info = {
                'step_name': step_name,
                'developer_id': 'unknown',  # Would be determined from DAG in real implementation
                'step_type': 'unknown'
            }
            consumer_workspace_info = {
                'step_name': next_step,
                'developer_id': 'unknown',  # Would be determined from DAG in real implementation
                'step_type': 'unknown'
            }
        
        return self.data_validator.validate_step_transition(
            {"files": step_outputs},
            input_spec,
            producer_workspace_info,
            consumer_workspace_info
        )
    
    def _get_script_path(self, step_name: str, step_config: Dict[str, Any]) -> str:
        """Get script path from step configuration using enhanced contract discovery.
        
        Args:
            step_name: Name of the step
            step_config: Configuration for the step
            
        Returns:
            Path to the script
        """
        # Enhancement 1: Use contract discovery from resolver
        if self.resolver:
            contract = self.resolver._discover_step_contract(step_name)
            if contract and contract.entry_point:
                self.logger.debug(f"Using contract entry point for {step_name}: {contract.entry_point}")
                return contract.entry_point
        
        # Existing fallback logic
        if "script_path" in step_config:
            return step_config["script_path"]
        
        # If no script path specified, use a default
        return "model_calibration.py"  # Placeholder
    
    def _prepare_workspace_execution_context(self, workspace_dag: WorkspaceAwareDAG) -> None:
        """Prepare workspace execution context for cross-workspace dependency tracking.
        
        Args:
            workspace_dag: WorkspaceAwareDAG instance
        """
        try:
            # Analyze cross-workspace dependencies
            validation_result = workspace_dag.validate_workspace_dependencies()
            self.workspace_execution_context['cross_workspace_dependencies'] = validation_result.get('cross_workspace_dependencies', [])
            
            # Log cross-workspace dependency information
            if self.workspace_execution_context['cross_workspace_dependencies']:
                self.logger.info(f"Found {len(self.workspace_execution_context['cross_workspace_dependencies'])} cross-workspace dependencies")
                for dep in self.workspace_execution_context['cross_workspace_dependencies']:
                    self.logger.debug(f"  {dep['dependent_step']} ({dep['dependent_developer']}) -> {dep['dependency_step']} ({dep['dependency_developer']})")
            else:
                self.logger.info("No cross-workspace dependencies detected")
            
            # Initialize developer execution stats
            developers = workspace_dag.get_developers()
            for developer_id in developers:
                self.workspace_execution_context['developer_execution_stats'][developer_id] = {
                    'steps_executed': 0,
                    'total_execution_time': 0.0,
                    'successful_steps': 0,
                    'failed_steps': 0
                }
            
            self.logger.info(f"Prepared workspace execution context for {len(developers)} developers")
            
        except Exception as e:
            self.logger.error(f"Error preparing workspace execution context: {e}")
            # Continue execution even if workspace context preparation fails
    
    def get_workspace_execution_summary(self) -> Dict[str, Any]:
        """Get summary of workspace execution statistics.
        
        Returns:
            Dictionary containing workspace execution summary
        """
        if not self.workspace_root:
            return {"error": "No workspace context available"}
        
        summary = {
            'workspace_root': self.workspace_root,
            'cross_workspace_dependencies': len(self.workspace_execution_context.get('cross_workspace_dependencies', [])),
            'developer_stats': self.workspace_execution_context.get('developer_execution_stats', {}),
            'overall_stats': {
                'total_developers': len(self.workspace_execution_context.get('developer_execution_stats', {})),
                'total_steps_executed': 0,
                'total_execution_time': 0.0,
                'overall_success_rate': 0.0
            }
        }
        
        # Calculate overall statistics
        total_successful = 0
        total_failed = 0
        
        for developer_id, stats in summary['developer_stats'].items():
            summary['overall_stats']['total_steps_executed'] += stats['steps_executed']
            summary['overall_stats']['total_execution_time'] += stats['total_execution_time']
            total_successful += stats['successful_steps']
            total_failed += stats['failed_steps']
        
        total_steps = total_successful + total_failed
        if total_steps > 0:
            summary['overall_stats']['overall_success_rate'] = total_successful / total_steps
        
        return summary
