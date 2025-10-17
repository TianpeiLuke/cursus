"""
Script Assembler

Assembles and executes scripts using DAG structure with dependency resolution.
Mirrors PipelineAssembler but executes scripts instead of creating SageMaker pipeline steps.
Uses maximum component reuse from existing cursus infrastructure.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import sys
import importlib.util
import argparse
import time
import logging
from datetime import datetime

# Direct imports from existing cursus components - MAXIMUM REUSE
from ...step_catalog import StepCatalog
from ...core.deps.dependency_resolver import create_dependency_resolver
from ..base.script_execution_plan import ScriptExecutionPlan
from ..base.script_execution_spec import ScriptExecutionSpec
from ..base.script_test_result import ScriptTestResult

logger = logging.getLogger(__name__)


class ScriptAssembler:
    """
    Assembles and executes scripts using DAG structure with dependency resolution.
    
    Mirrors PipelineAssembler but executes scripts instead of creating 
    SageMaker pipeline steps. Uses maximum component reuse from existing
    cursus infrastructure.
    
    This class follows the same patterns as PipelineAssembler:
    1. Takes an execution plan with DAG structure
    2. Executes components in topological order
    3. Resolves dependencies between components
    4. Collects and returns execution results
    
    Attributes:
        execution_plan: ScriptExecutionPlan defining what to execute
        step_catalog: StepCatalog for enhanced script discovery and validation
        dependency_resolver: Dependency resolver for script I/O connections
        script_results: Dictionary tracking script execution results
        script_outputs: Dictionary tracking script output paths for dependency resolution
    """
    
    def __init__(
        self,
        execution_plan: ScriptExecutionPlan,
        step_catalog: Optional[StepCatalog] = None,
        dependency_resolver: Optional[Any] = None,
    ):
        """
        Initialize the Script Assembler.
        
        Args:
            execution_plan: ScriptExecutionPlan to execute
            step_catalog: Optional step catalog for enhanced script operations
            dependency_resolver: Optional dependency resolver for I/O connections
        """
        self.execution_plan = execution_plan
        self.step_catalog = step_catalog or StepCatalog()
        
        # DIRECT REUSE: Use existing dependency resolver (mirrors PipelineAssembler)
        self.dependency_resolver = dependency_resolver or create_dependency_resolver()
        
        # Track script execution state (mirrors PipelineAssembler state tracking)
        self.script_results: Dict[str, ScriptTestResult] = {}
        self.script_outputs: Dict[str, Dict[str, str]] = {}
        
        logger.info(f"Initialized ScriptAssembler for execution plan with {len(execution_plan.script_specs)} scripts")
    
    def execute_dag_scripts(self) -> Dict[str, Any]:
        """
        Execute all scripts in DAG order with dependency resolution.
        
        This method mirrors generate_pipeline in PipelineAssembler but executes
        scripts instead of creating SageMaker pipeline steps.
        
        Returns:
            Dictionary with comprehensive execution results:
            {
                "pipeline_success": bool,
                "script_results": Dict[str, ScriptTestResult],
                "data_flow_results": Dict[str, Any],
                "execution_order": List[str],
                "errors": List[str],
                "execution_summary": Dict[str, Any]
            }
        """
        logger.info("Starting DAG script execution")
        start_time = datetime.now()
        
        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "execution_order": self.execution_plan.execution_order,
            "errors": [],
            "execution_summary": {
                "total_scripts": len(self.execution_plan.script_specs),
                "successful_scripts": 0,
                "failed_scripts": 0,
                "start_time": start_time.isoformat(),
            }
        }
        
        # Execute scripts in topological order (MIRRORS PipelineAssembler)
        for node_name in self.execution_plan.execution_order:
            try:
                logger.info(f"Executing script for node: {node_name}")
                
                # 1. Resolve script inputs from dependencies (MIRRORS PipelineAssembler)
                resolved_inputs = self._resolve_script_inputs(node_name)
                
                # 2. Execute script with resolved inputs
                script_result = self._execute_script(node_name, resolved_inputs)
                
                # 3. Store results and outputs
                self.script_results[node_name] = script_result
                results["script_results"][node_name] = script_result
                
                if script_result.success:
                    results["execution_summary"]["successful_scripts"] += 1
                    # 4. Register outputs for dependency resolution (MIRRORS PipelineAssembler)
                    self._register_script_outputs(node_name, script_result)
                    logger.info(f"✅ Script {node_name} executed successfully in {script_result.execution_time:.2f}s")
                else:
                    results["execution_summary"]["failed_scripts"] += 1
                    results["pipeline_success"] = False
                    results["errors"].append(f"Script {node_name} failed: {script_result.error_message}")
                    logger.error(f"❌ Script {node_name} failed: {script_result.error_message}")
                    
            except Exception as e:
                results["execution_summary"]["failed_scripts"] += 1
                results["pipeline_success"] = False
                error_msg = f"Error executing {node_name}: {str(e)}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
                
                # Create failure result for tracking
                failure_result = ScriptTestResult.create_failure(
                    script_name=self.execution_plan.script_specs[node_name].script_name,
                    step_name=node_name,
                    execution_time=0.0,
                    error_message=str(e)
                )
                self.script_results[node_name] = failure_result
                results["script_results"][node_name] = failure_result
        
        # Test data flow between connected scripts (MIRRORS PipelineAssembler validation)
        self._test_data_flow_connections(results)
        
        # Finalize execution summary
        end_time = datetime.now()
        results["execution_summary"].update({
            "end_time": end_time.isoformat(),
            "total_execution_time": (end_time - start_time).total_seconds(),
            "success_rate": results["execution_summary"]["successful_scripts"] / results["execution_summary"]["total_scripts"] if results["execution_summary"]["total_scripts"] > 0 else 0.0,
        })
        
        logger.info(f"DAG script execution completed: {results['execution_summary']['successful_scripts']}/{results['execution_summary']['total_scripts']} successful")
        return results
    
    def _resolve_script_inputs(self, node_name: str) -> Dict[str, Any]:
        """
        Resolve script inputs from dependency outputs.
        
        Mirrors input resolution in PipelineAssembler using UnifiedDependencyResolver.
        
        Args:
            node_name: Name of the node to resolve inputs for
            
        Returns:
            Dictionary with resolved input paths and parameters
        """
        script_spec = self.execution_plan.script_specs[node_name]
        resolved_inputs = script_spec.input_paths.copy()
        
        # Get dependencies for this node (DIRECT REUSE)
        dependencies = self.execution_plan.dag.get_dependencies(node_name)
        
        if dependencies:
            logger.debug(f"Resolving dependencies for {node_name}: {dependencies}")
            
            for dep_node in dependencies:
                if dep_node in self.script_outputs:
                    # DIRECT REUSE: Use dependency resolver to match outputs to inputs
                    dep_outputs = self.script_outputs[dep_node]
                    
                    # Apply semantic matching to connect outputs to inputs
                    matches = self._find_semantic_matches(dep_outputs, script_spec.input_paths)
                    
                    for input_name, output_path in matches.items():
                        resolved_inputs[input_name] = output_path
                        logger.debug(f"Resolved {node_name}.{input_name} -> {output_path} (from {dep_node})")
        
        return resolved_inputs
    
    def _find_semantic_matches(
        self, 
        dep_outputs: Dict[str, str], 
        target_inputs: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Find semantic matches between dependency outputs and target inputs.
        
        This method uses simple semantic matching that could be enhanced
        with the SemanticMatcher from cursus/core for more sophisticated matching.
        
        Args:
            dep_outputs: Dictionary of dependency output paths
            target_inputs: Dictionary of target input paths
            
        Returns:
            Dictionary mapping input names to matched output paths
        """
        matches = {}
        
        for input_name, input_path in target_inputs.items():
            for output_name, output_path in dep_outputs.items():
                # Simple semantic matching (could use SemanticMatcher for enhancement)
                if self._paths_semantically_match(output_name, input_name):
                    matches[input_name] = output_path
                    break
        
        return matches
    
    def _paths_semantically_match(self, output_name: str, input_name: str) -> bool:
        """
        Simple semantic matching for path names.
        
        This could be enhanced to use the SemanticMatcher from cursus/core
        for more sophisticated matching.
        """
        # Simple matching: exact match or contains relationship
        return (
            output_name == input_name or
            output_name in input_name or
            input_name in output_name or
            output_name.replace('_', '') == input_name.replace('_', '') or
            # Common semantic patterns
            (output_name.endswith('_data') and input_name.startswith('input_')) or
            (output_name.startswith('processed_') and input_name.endswith('_input'))
        )
    
    def _execute_script(self, node_name: str, resolved_inputs: Dict[str, str]) -> ScriptTestResult:
        """
        Execute individual script with resolved inputs.
        
        Mirrors step instantiation in PipelineAssembler but executes Python scripts
        instead of creating SageMaker steps.
        
        Args:
            node_name: Name of the DAG node
            resolved_inputs: Resolved input paths for the script
            
        Returns:
            ScriptTestResult with execution results
        """
        script_spec = self.execution_plan.script_specs[node_name]
        start_time = time.time()
        
        try:
            # Create main parameters (mirrors step builder parameter extraction)
            main_params = {
                "input_paths": resolved_inputs,
                "output_paths": script_spec.output_paths,
                "environ_vars": script_spec.environ_vars,
                "job_args": argparse.Namespace(**script_spec.job_args) if script_spec.job_args else argparse.Namespace(),
            }
            
            # Execute script (mirrors step.create_step())
            result = self._execute_script_main_function(script_spec, main_params)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Script execution failed for {node_name}: {e}")
            
            return ScriptTestResult.create_failure(
                script_name=script_spec.script_name,
                step_name=node_name,
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _execute_script_main_function(
        self, 
        script_spec: ScriptExecutionSpec, 
        main_params: Dict[str, Any]
    ) -> ScriptTestResult:
        """
        Execute script's main function with parameters.
        
        Args:
            script_spec: Script execution specification
            main_params: Parameters to pass to main function
            
        Returns:
            ScriptTestResult with execution results
        """
        script_path = Path(script_spec.script_path)
        
        # Check if script exists
        if not script_path.exists():
            return ScriptTestResult.create_failure(
                script_name=script_spec.script_name,
                step_name=script_spec.step_name,
                execution_time=0.0,
                error_message=f"Script file not found: {script_path}",
                has_main_function=False
            )
        
        try:
            # Load the script module
            spec = importlib.util.spec_from_file_location(script_spec.script_name, script_path)
            if spec is None or spec.loader is None:
                return ScriptTestResult.create_failure(
                    script_name=script_spec.script_name,
                    step_name=script_spec.step_name,
                    execution_time=0.0,
                    error_message="Failed to load script module",
                    has_main_function=False
                )
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if main function exists
            if not hasattr(module, 'main'):
                return ScriptTestResult.create_failure(
                    script_name=script_spec.script_name,
                    step_name=script_spec.step_name,
                    execution_time=0.0,
                    error_message="Script does not have a main() function",
                    has_main_function=False
                )
            
            # Execute main function
            main_function = getattr(module, 'main')
            
            # Try different parameter patterns
            try:
                # Try with all parameters
                main_function(**main_params)
            except TypeError:
                try:
                    # Try with just the essential parameters
                    main_function(
                        main_params["input_paths"],
                        main_params["output_paths"],
                        main_params["environ_vars"],
                        main_params["job_args"]
                    )
                except TypeError:
                    try:
                        # Try with no parameters
                        main_function()
                    except TypeError as e:
                        return ScriptTestResult.create_failure(
                            script_name=script_spec.script_name,
                            step_name=script_spec.step_name,
                            execution_time=0.0,
                            error_message=f"main() function parameter mismatch: {e}",
                            has_main_function=True
                        )
            
            # Check for output files
            output_files = []
            for output_name, output_path in script_spec.output_paths.items():
                if Path(output_path).exists():
                    output_files.append(output_path)
            
            # Create success result
            result = ScriptTestResult.create_success(
                script_name=script_spec.script_name,
                step_name=script_spec.step_name,
                execution_time=0.0,  # Will be set by caller
                output_files=output_files
            )
            
            return result
            
        except Exception as e:
            return ScriptTestResult.create_failure(
                script_name=script_spec.script_name,
                step_name=script_spec.step_name,
                execution_time=0.0,
                error_message=f"Script execution error: {e}",
                has_main_function=True
            )
    
    def _register_script_outputs(self, node_name: str, script_result: ScriptTestResult) -> None:
        """
        Register script outputs for dependency resolution.
        
        Args:
            node_name: Name of the node that was executed
            script_result: Result of script execution
        """
        script_spec = self.execution_plan.script_specs[node_name]
        
        # Register output paths for dependency resolution
        self.script_outputs[node_name] = script_spec.output_paths.copy()
        
        logger.debug(f"Registered outputs for {node_name}: {list(script_spec.output_paths.keys())}")
    
    def _test_data_flow_connections(self, results: Dict[str, Any]) -> None:
        """
        Test data flow between connected scripts.
        
        This method validates that outputs from dependency scripts are properly
        connected to inputs of dependent scripts.
        
        Args:
            results: Results dictionary to update with data flow information
        """
        data_flow_results = {
            "connections_tested": 0,
            "successful_connections": 0,
            "failed_connections": [],
            "connection_details": {}
        }
        
        for node_name in self.execution_plan.execution_order:
            dependencies = self.execution_plan.dag.get_dependencies(node_name)
            
            if dependencies:
                for dep_node in dependencies:
                    if dep_node in self.script_outputs and node_name in self.script_results:
                        connection_key = f"{dep_node} -> {node_name}"
                        data_flow_results["connections_tested"] += 1
                        
                        # Test if dependency outputs exist and are accessible
                        dep_outputs = self.script_outputs[dep_node]
                        connection_success = True
                        connection_details = {
                            "dependency_outputs": dep_outputs,
                            "output_files_exist": {},
                            "connection_successful": True
                        }
                        
                        for output_name, output_path in dep_outputs.items():
                            file_exists = Path(output_path).exists()
                            connection_details["output_files_exist"][output_name] = file_exists
                            if not file_exists:
                                connection_success = False
                        
                        if connection_success:
                            data_flow_results["successful_connections"] += 1
                        else:
                            data_flow_results["failed_connections"].append(connection_key)
                            connection_details["connection_successful"] = False
                        
                        data_flow_results["connection_details"][connection_key] = connection_details
        
        results["data_flow_results"] = data_flow_results
        logger.info(f"Data flow testing: {data_flow_results['successful_connections']}/{data_flow_results['connections_tested']} connections successful")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the script execution results.
        
        Returns:
            Dictionary with execution summary
        """
        total_scripts = len(self.script_results)
        successful_scripts = sum(1 for result in self.script_results.values() if result.success)
        
        return {
            "total_scripts": total_scripts,
            "successful_scripts": successful_scripts,
            "failed_scripts": total_scripts - successful_scripts,
            "success_rate": successful_scripts / total_scripts if total_scripts > 0 else 0.0,
            "execution_order": self.execution_plan.execution_order,
            "script_results": {
                name: result.get_execution_summary() 
                for name, result in self.script_results.items()
            }
        }
    
    def __str__(self) -> str:
        """String representation of the assembler."""
        return f"ScriptAssembler(scripts={len(self.execution_plan.script_specs)}, executed={len(self.script_results)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ScriptAssembler("
            f"scripts={len(self.execution_plan.script_specs)}, "
            f"executed={len(self.script_results)}, "
            f"execution_order={self.execution_plan.execution_order})"
        )
