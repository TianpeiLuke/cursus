"""Pipeline executor for end-to-end testing."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import time
import logging
from pydantic import BaseModel, Field
from datetime import datetime

from .pipeline_dag_resolver import PipelineDAGResolver, PipelineExecutionPlan
from .data_compatibility_validator import DataCompatibilityValidator, DataCompatibilityReport
from ..core.pipeline_script_executor import PipelineScriptExecutor
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
    
    def __init__(self, workspace_dir: str = "./pipeline_testing"):
        """Initialize with workspace directory."""
        self.workspace_dir = Path(workspace_dir)
        self.script_executor = PipelineScriptExecutor(workspace_dir=workspace_dir)
        self.data_validator = DataCompatibilityValidator()
        self.execution_results = {}
        self.logger = logging.getLogger(__name__)
    
    def execute_pipeline(self, dag, data_source: str = "synthetic") -> PipelineExecutionResult:
        """Execute complete pipeline with data flow validation.
        
        Args:
            dag: PipelineDAG object representing the pipeline
            data_source: Source of data for testing ("synthetic" or "s3")
            
        Returns:
            PipelineExecutionResult object with execution results
        """
        start_time = time.time()
        memory_peak = 0
        
        try:
            resolver = PipelineDAGResolver(dag)
            execution_plan = resolver.create_execution_plan()
            
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
                    self.logger.info(f"Executing step: {step_name}")
                    
                    # Prepare step inputs from previous outputs
                    step_inputs = self._prepare_step_inputs(
                        step_name, execution_plan, step_outputs
                    )
                    
                    # Execute step
                    step_result = self._execute_step(
                        step_name, 
                        execution_plan.step_configs.get(step_name, {}),
                        step_inputs,
                        data_source
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
        """Prepare inputs for a step from previous step outputs.
        
        Args:
            step_name: Name of the step
            execution_plan: PipelineExecutionPlan object
            step_outputs: Dictionary of step outputs from previous steps
            
        Returns:
            Dictionary with step inputs
        """
        inputs = {}
        
        # Get dependencies from execution plan
        dependencies = execution_plan.dependencies.get(step_name, [])
        
        # Get input mapping from data flow map
        input_mapping = execution_plan.data_flow_map.get(step_name, {})
        
        for dep_step in dependencies:
            if dep_step in step_outputs:
                dep_outputs = step_outputs[dep_step]
                
                # Map outputs to inputs based on data flow map
                for input_key, output_ref in input_mapping.items():
                    if ":" in output_ref:
                        src_step, output_key = output_ref.split(":", 1)
                        if src_step == dep_step and output_key in dep_outputs:
                            inputs[input_key] = dep_outputs[output_key]
        
        return inputs
    
    def _execute_step(self, step_name: str, step_config: Dict[str, Any], 
                    step_inputs: Dict[str, Any], 
                    data_source: str) -> StepExecutionResult:
        """Execute a single step with inputs.
        
        Args:
            step_name: Name of the step
            step_config: Configuration for the step
            step_inputs: Inputs for the step
            data_source: Source of data for testing
            
        Returns:
            StepExecutionResult object with execution results
        """
        start_time = time.time()
        
        try:
            # For now, we use the script executor to run the script
            # In the future, this will use more sophisticated step execution logic
            script_path = self._get_script_path(step_config)
            
            # Create input/output paths for the step
            input_path = self.workspace_dir / "inputs" / step_name
            output_path = self.workspace_dir / "outputs" / step_name
            input_path.mkdir(parents=True, exist_ok=True)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare inputs
            # In a real implementation, this would copy or generate actual files
            # For now, we just simulate this
            input_paths = {
                "input": str(input_path)
            }
            
            output_paths = {
                "output": str(output_path)
            }
            
            # Add any specific input/output paths from step_config
            if "input_paths" in step_config:
                input_paths.update(step_config["input_paths"])
            
            if "output_paths" in step_config:
                output_paths.update(step_config["output_paths"])
            
            # Execute the script
            result = self.script_executor.test_script_isolation(script_path)
            
            end_time = time.time()
            
            # Build step execution result
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
        
        # Validate outputs against input spec
        return self.data_validator.validate_step_transition(
            {"files": step_outputs},
            input_spec
        )
    
    def _get_script_path(self, step_config: Dict[str, Any]) -> str:
        """Get script path from step configuration.
        
        Args:
            step_config: Configuration for the step
            
        Returns:
            Path to the script
        """
        # In a real implementation, this would resolve the script path
        # based on the step configuration
        # For now, we just use a placeholder
        if "script_path" in step_config:
            return step_config["script_path"]
        
        # If no script path specified, use a default
        return "model_calibration.py"  # Placeholder
