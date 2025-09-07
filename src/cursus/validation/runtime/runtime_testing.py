"""
Simplified Pipeline Runtime Testing

Validates script functionality and data transfer consistency for pipeline development.
Based on validated user story: "examine the script's functionality and their data 
transfer consistency along the DAG, without worrying about the resolution of 
step-to-step or step-to-script dependencies."

Refactored implementation with PipelineTestingSpecBuilder and ScriptExecutionSpec
for user-centric approach with local persistence.
"""

import importlib.util
import json
import os
import time
import argparse
import pandas as pd
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import from separate model files
from .runtime_models import (
    ScriptTestResult, 
    DataCompatibilityResult, 
    ScriptExecutionSpec, 
    PipelineTestingSpec, 
    RuntimeTestingConfiguration
)
from .runtime_spec_builder import PipelineTestingSpecBuilder

# Import PipelineDAG for integration
from ...api.dag.base_dag import PipelineDAG


class RuntimeTester:
    """Core testing engine that uses PipelineTestingSpecBuilder for parameter extraction"""
    
    def __init__(self, config_or_workspace_dir):
        # Support both new RuntimeTestingConfiguration and old string workspace_dir for backward compatibility
        if isinstance(config_or_workspace_dir, RuntimeTestingConfiguration):
            self.config = config_or_workspace_dir
            self.pipeline_spec = config_or_workspace_dir.pipeline_spec
            self.workspace_dir = Path(config_or_workspace_dir.pipeline_spec.test_workspace_root)
            
            # Create builder instance for parameter extraction
            self.builder = PipelineTestingSpecBuilder(
                test_data_dir=config_or_workspace_dir.pipeline_spec.test_workspace_root
            )
        else:
            # Backward compatibility: treat as workspace directory string
            workspace_dir = str(config_or_workspace_dir)
            self.config = None
            self.pipeline_spec = None
            self.workspace_dir = Path(workspace_dir)
            
            # Create builder instance for parameter extraction
            self.builder = PipelineTestingSpecBuilder(test_data_dir=workspace_dir)
    
    def test_script_with_spec(self, script_spec: ScriptExecutionSpec, main_params: Dict[str, Any]) -> ScriptTestResult:
        """Test script functionality using ScriptExecutionSpec"""
        start_time = time.time()
        
        try:
            script_path = self._find_script_path(script_spec.script_name)
            
            # Import script using standard Python import
            spec = importlib.util.spec_from_file_location("script", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for main function with correct signature
            has_main = hasattr(module, 'main') and callable(module.main)
            
            if not has_main:
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message="Script missing main() function",
                    execution_time=time.time() - start_time,
                    has_main_function=False
                )
            
            # Validate main function signature matches script development guide
            sig = inspect.signature(module.main)
            expected_params = ['input_paths', 'output_paths', 'environ_vars', 'job_args']
            actual_params = list(sig.parameters.keys())
            
            if not all(param in actual_params for param in expected_params):
                return ScriptTestResult(
                    script_name=script_spec.script_name,
                    success=False,
                    error_message="Main function signature doesn't match script development guide",
                    execution_time=time.time() - start_time,
                    has_main_function=True
                )
            
            # Create test directories based on ScriptExecutionSpec
            test_dir = Path(script_spec.output_paths["data_output"])
            test_dir.mkdir(parents=True, exist_ok=True)
            
            # Use ScriptExecutionSpec input data path or generate sample data
            input_data_path = script_spec.input_paths.get("data_input")
            if not input_data_path or not Path(input_data_path).exists():
                # Generate sample data for testing
                sample_data = self._generate_sample_data()
                input_data_path = test_dir / "input_data.csv"
                pd.DataFrame(sample_data).to_csv(input_data_path, index=False)
            
            # EXECUTE THE MAIN FUNCTION with ScriptExecutionSpec parameters
            module.main(**main_params)
            
            return ScriptTestResult(
                script_name=script_spec.script_name,
                success=True,
                error_message=None,
                execution_time=time.time() - start_time,
                has_main_function=True
            )
            
        except Exception as e:
            return ScriptTestResult(
                script_name=script_spec.script_name,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                has_main_function=has_main if 'has_main' in locals() else False
            )
    
    def test_data_compatibility_with_specs(self, spec_a: ScriptExecutionSpec, spec_b: ScriptExecutionSpec) -> DataCompatibilityResult:
        """Test data compatibility between scripts using ScriptExecutionSpecs"""
        
        try:
            # Execute script A using its ScriptExecutionSpec
            main_params_a = self.builder.get_script_main_params(spec_a)
            script_a_result = self.test_script_with_spec(spec_a, main_params_a)
            
            if not script_a_result.success:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[f"Script A failed: {script_a_result.error_message}"]
                )
            
            # Check if script A produced output
            output_dir_a = Path(spec_a.output_paths["data_output"])
            output_files = list(output_dir_a.glob("*.csv"))
            
            if not output_files:
                return DataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=["Script A did not produce output data"]
                )
            
            # Use script A's output as script B's input
            # Create a modified spec_b with script A's output as input
            modified_spec_b = ScriptExecutionSpec(
                script_name=spec_b.script_name,
                step_name=spec_b.step_name,
                script_path=spec_b.script_path,
                input_paths={"data_input": str(output_files[0])},  # Use script A's output
                output_paths=spec_b.output_paths,
                environ_vars=spec_b.environ_vars,
                job_args=spec_b.job_args
            )
            
            # Test script B with script A's output
            main_params_b = self.builder.get_script_main_params(modified_spec_b)
            script_b_result = self.test_script_with_spec(modified_spec_b, main_params_b)
            
            # Analyze compatibility
            compatibility_issues = []
            if not script_b_result.success:
                compatibility_issues.append(f"Script B failed with script A output: {script_b_result.error_message}")
            
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=script_b_result.success,
                compatibility_issues=compatibility_issues,
                data_format_a="csv",
                data_format_b="csv"
            )
            
        except Exception as e:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Compatibility test failed: {str(e)}"]
            )
    
    def test_pipeline_flow_with_spec(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
        """Test end-to-end pipeline flow using PipelineTestingSpec and PipelineDAG"""
        
        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "errors": []
        }
        
        try:
            dag = pipeline_spec.dag
            script_specs = pipeline_spec.script_specs
            
            if not dag.nodes:
                results["pipeline_success"] = False
                results["errors"].append("No nodes found in pipeline DAG")
                return results
            
            # Test each script individually first using ScriptExecutionSpec
            for node_name in dag.nodes:
                if node_name not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(f"No ScriptExecutionSpec found for node: {node_name}")
                    continue
                    
                script_spec = script_specs[node_name]
                main_params = self.builder.get_script_main_params(script_spec)
                
                script_result = self.test_script_with_spec(script_spec, main_params)
                results["script_results"][node_name] = script_result
                
                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Script {node_name} failed: {script_result.error_message}")
            
            # Test data flow between connected scripts using DAG edges
            for edge in dag.edges:
                script_a, script_b = edge
                
                if script_a not in script_specs or script_b not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Missing ScriptExecutionSpec for edge: {script_a} -> {script_b}")
                    continue
                
                spec_a = script_specs[script_a]
                spec_b = script_specs[script_b]
                
                # Test data compatibility using ScriptExecutionSpecs
                compat_result = self.test_data_compatibility_with_specs(spec_a, spec_b)
                results["data_flow_results"][f"{script_a}->{script_b}"] = compat_result
                
                if not compat_result.compatible:
                    results["pipeline_success"] = False
                    results["errors"].extend(compat_result.compatibility_issues)
            
            return results
            
        except Exception as e:
            results["pipeline_success"] = False
            results["errors"].append(f"Pipeline flow test failed: {str(e)}")
            return results
    
    # Keep existing methods for backward compatibility
    def test_script(self, script_name: str, sample_data: Optional[Dict] = None) -> ScriptTestResult:
        """Test single script functionality by ACTUALLY EXECUTING IT - USER REQUIREMENT 1 & 2"""
        start_time = time.time()
        
        try:
            script_path = self._find_script_path(script_name)
            
            # Import script using standard Python import
            spec = importlib.util.spec_from_file_location("script", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check for main function with correct signature
            has_main = hasattr(module, 'main') and callable(module.main)
            
            if not has_main:
                return ScriptTestResult(
                    script_name=script_name,
                    success=False,
                    error_message="Script missing main() function",
                    execution_time=time.time() - start_time,
                    has_main_function=False
                )
            
            # Validate main function signature matches script development guide
            sig = inspect.signature(module.main)
            expected_params = ['input_paths', 'output_paths', 'environ_vars', 'job_args']
            actual_params = list(sig.parameters.keys())
            
            if not all(param in actual_params for param in expected_params):
                return ScriptTestResult(
                    script_name=script_name,
                    success=False,
                    error_message="Main function signature doesn't match script development guide",
                    execution_time=time.time() - start_time,
                    has_main_function=True
                )
            
            # ACTUALLY EXECUTE THE SCRIPT with test data
            test_dir = self.workspace_dir / f"test_{script_name}"
            test_dir.mkdir(exist_ok=True)
            
            # Create test input data
            if sample_data is None:
                sample_data = self._generate_sample_data()
            
            input_data_path = test_dir / "input_data.csv"
            output_data_path = test_dir / "output_data.csv"
            
            pd.DataFrame(sample_data).to_csv(input_data_path, index=False)
            
            # Prepare execution parameters following script development guide
            input_paths = {"data_input": str(input_data_path)}
            output_paths = {"data_output": str(test_dir)}
            environ_vars = {"LABEL_FIELD": "label"}
            job_args = argparse.Namespace(job_type="testing")
            
            # EXECUTE THE MAIN FUNCTION
            module.main(input_paths, output_paths, environ_vars, job_args)
            
            return ScriptTestResult(
                script_name=script_name,
                success=True,
                error_message=None,
                execution_time=time.time() - start_time,
                has_main_function=True
            )
            
        except Exception as e:
            return ScriptTestResult(
                script_name=script_name,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                has_main_function=has_main if 'has_main' in locals() else False
            )
    
    def test_data_compatibility(self, script_a: str, script_b: str, 
                               sample_data: Dict) -> DataCompatibilityResult:
        """Test data compatibility between scripts - USER REQUIREMENT 3"""
        
        try:
            # Create test environment for script A
            test_dir_a = self.workspace_dir / f"test_{script_a}"
            test_dir_a.mkdir(exist_ok=True)
            
            # Generate test data for script A
            input_data_path = test_dir_a / "input_data.csv"
            output_data_path = test_dir_a / "output_data.csv"
            
            # Create sample input data
            pd.DataFrame(sample_data).to_csv(input_data_path, index=False)
            
            # Execute script A to generate output
            script_a_result = self._execute_script_with_data(script_a, 
                                                            str(input_data_path), 
                                                            str(output_data_path))
            
            if not script_a_result.success:
                return DataCompatibilityResult(
                    script_a=script_a,
                    script_b=script_b,
                    compatible=False,
                    compatibility_issues=[f"Script A failed: {script_a_result.error_message}"]
                )
            
            # Check if script A produced output
            if not output_data_path.exists():
                return DataCompatibilityResult(
                    script_a=script_a,
                    script_b=script_b,
                    compatible=False,
                    compatibility_issues=["Script A did not produce output data"]
                )
            
            # Load script A output
            output_data_a = pd.read_csv(output_data_path)
            
            # Create test environment for script B
            test_dir_b = self.workspace_dir / f"test_{script_b}"
            test_dir_b.mkdir(exist_ok=True)
            
            # Use script A output as script B input
            input_data_b_path = test_dir_b / "input_data.csv"
            output_data_a.to_csv(input_data_b_path, index=False)
            
            # Test script B with script A's output
            script_b_result = self._execute_script_with_data(script_b,
                                                            str(input_data_b_path),
                                                            str(test_dir_b / "output_data.csv"))
            
            # Analyze compatibility
            compatibility_issues = []
            if not script_b_result.success:
                compatibility_issues.append(f"Script B failed with script A output: {script_b_result.error_message}")
            
            return DataCompatibilityResult(
                script_a=script_a,
                script_b=script_b,
                compatible=script_b_result.success,
                compatibility_issues=compatibility_issues,
                data_format_a="csv",
                data_format_b="csv"
            )
            
        except Exception as e:
            return DataCompatibilityResult(
                script_a=script_a,
                script_b=script_b,
                compatible=False,
                compatibility_issues=[f"Compatibility test failed: {str(e)}"]
            )
    
    def test_pipeline_flow(self, pipeline_config: Dict) -> Dict[str, Any]:
        """Test end-to-end pipeline flow - USER REQUIREMENT 4"""
        
        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "errors": []
        }
        
        try:
            steps = pipeline_config.get("steps", {})
            if not steps:
                results["pipeline_success"] = False
                results["errors"].append("No steps found in pipeline configuration")
                return results
            
            # Test each script individually first
            for step_name in steps:
                script_result = self.test_script(step_name)
                results["script_results"][step_name] = script_result
                
                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Script {step_name} failed: {script_result.error_message}")
            
            # Test data flow between connected scripts
            step_list = list(steps.keys())
            for i in range(len(step_list) - 1):
                script_a = step_list[i]
                script_b = step_list[i + 1]
                
                # Generate sample data for testing
                sample_data = self._generate_sample_data()
                
                # Test data compatibility
                compat_result = self.test_data_compatibility(script_a, script_b, sample_data)
                results["data_flow_results"][f"{script_a}->{script_b}"] = compat_result
                
                if not compat_result.compatible:
                    results["pipeline_success"] = False
                    results["errors"].extend(compat_result.compatibility_issues)
            
            return results
            
        except Exception as e:
            results["pipeline_success"] = False
            results["errors"].append(f"Pipeline flow test failed: {str(e)}")
            return results
    
    def _find_script_path(self, script_name: str) -> str:
        """Simple script discovery - ESSENTIAL UTILITY"""
        possible_paths = [
            f"src/cursus/steps/scripts/{script_name}.py",
            f"scripts/{script_name}.py",
            f"dockers/xgboost_atoz/scripts/{script_name}.py",
            f"dockers/pytorch_bsm_ext/scripts/{script_name}.py"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                return path
        
        raise FileNotFoundError(f"Script not found: {script_name}")
    
    def _execute_script_with_data(self, script_name: str, input_path: str, 
                                 output_path: str) -> ScriptTestResult:
        """Execute script with test data - ESSENTIAL FOR DATA FLOW TESTING"""
        start_time = time.time()
        
        try:
            script_path = self._find_script_path(script_name)
            
            # Import script
            spec = importlib.util.spec_from_file_location("script", script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Prepare execution parameters following script development guide
            input_paths = {"data_input": input_path}
            output_paths = {"data_output": str(Path(output_path).parent)}
            environ_vars = {"LABEL_FIELD": "label"}  # Basic environment
            job_args = argparse.Namespace(job_type="testing")
            
            # Create output directory
            Path(output_paths["data_output"]).mkdir(parents=True, exist_ok=True)
            
            # Execute main function
            module.main(input_paths, output_paths, environ_vars, job_args)
            
            return ScriptTestResult(
                script_name=script_name,
                success=True,
                execution_time=time.time() - start_time,
                has_main_function=True
            )
            
        except Exception as e:
            return ScriptTestResult(
                script_name=script_name,
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time,
                has_main_function=False
            )
    
    def _generate_sample_data(self) -> Dict:
        """Generate simple sample data for testing"""
        return {
            "feature1": [1, 2, 3, 4, 5],
            "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "label": [0, 1, 0, 1, 0]
        }
