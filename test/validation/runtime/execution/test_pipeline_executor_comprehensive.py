"""Comprehensive unit tests for enhanced pipeline executor with PipelineDAGResolver integration."""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from datetime import datetime

from src.cursus.api.dag import PipelineDAG
from src.cursus.api.dag.pipeline_dag_resolver import PipelineExecutionPlan
from src.cursus.validation.runtime.execution.pipeline_executor import (
    PipelineExecutor, 
    PipelineExecutionResult, 
    StepExecutionResult
)
from src.cursus.core.base.contract_base import ScriptContract


class TestPipelineExecutorComprehensive:
    """Comprehensive test suite for enhanced pipeline executor functionality."""
    
    def setup_method(self):
        """Set up test fixtures with temporary workspace."""
        # Create temporary workspace
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_dir = Path(self.temp_dir) / "test_workspace"
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test DAGs
        self.simple_dag = PipelineDAG(
            nodes=["StepA"],
            edges=[]
        )
        
        self.linear_dag = PipelineDAG(
            nodes=["TabularPreprocessing", "XGBoostTraining", "XGBoostModelEval"],
            edges=[("TabularPreprocessing", "XGBoostTraining"), ("XGBoostTraining", "XGBoostModelEval")]
        )
        
        self.complex_dag = PipelineDAG(
            nodes=["CradleDataLoading", "TabularPreprocessing", "XGBoostTraining", "XGBoostModelEval", "ModelCalibration"],
            edges=[
                ("CradleDataLoading", "TabularPreprocessing"),
                ("TabularPreprocessing", "XGBoostTraining"),
                ("XGBoostTraining", "XGBoostModelEval"),
                ("XGBoostTraining", "ModelCalibration")
            ]
        )
        
        # Create executor
        self.executor = PipelineExecutor(workspace_dir=str(self.workspace_dir), testing_mode="pre_execution")
        
        # Create mock contracts
        self.preprocessing_contract = ScriptContract(
            entry_point="tabular_preprocessing.py",
            expected_input_paths={"DATA": "/opt/ml/processing/input/data"},
            expected_output_paths={"processed_data": "/opt/ml/processing/output"},
            required_env_vars=["LABEL_FIELD", "TRAIN_RATIO"],
            optional_env_vars={"CATEGORICAL_COLUMNS": "", "NUMERICAL_COLUMNS": ""}
        )
        
        self.training_contract = ScriptContract(
            entry_point="xgboost_training.py",
            expected_input_paths={"input_path": "/opt/ml/processing/input/data"},
            expected_output_paths={"model_output": "/opt/ml/processing/output/model", "evaluation_output": "/opt/ml/processing/output/evaluation"},
            required_env_vars=[],
            optional_env_vars={}
        )
        
        self.evaluation_contract = ScriptContract(
            entry_point="xgboost_evaluation.py",
            expected_input_paths={"model_path": "/opt/ml/processing/input/model", "test_data": "/opt/ml/processing/input/test_data"},
            expected_output_paths={"evaluation_results": "/opt/ml/processing/output/evaluation"},
            required_env_vars=["EVALUATION_METRICS"],
            optional_env_vars={}
        )
    
    def teardown_method(self):
        """Clean up temporary workspace."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def create_mock_resolver(self, dag, contracts=None, config_resolver=None, integrity_issues=None):
        """Create a mock resolver with specified behavior."""
        mock_resolver = Mock()
        
        # Create real execution plan based on DAG
        execution_order = list(dag.nodes)
        step_configs = {node: {} for node in dag.nodes}
        dependencies = {
            node: [src for src, dst in dag.edges if dst == node]
            for node in dag.nodes
        }
        
        # Create data flow map
        data_flow_map = {}
        for node in dag.nodes:
            deps = dependencies[node]
            if deps and contracts and node in contracts and contracts[node] is not None:
                # Use contract-based mapping
                contract = contracts[node]
                inputs = {}
                for input_name in contract.expected_input_paths.keys():
                    if deps:
                        dep_contract = contracts.get(deps[0])
                        if dep_contract and dep_contract.expected_output_paths:
                            output_name = list(dep_contract.expected_output_paths.keys())[0]
                            inputs[input_name] = f"{deps[0]}:{output_name}"
                data_flow_map[node] = inputs
            else:
                # Fallback mapping
                data_flow_map[node] = {f"input_{i}": f"{dep}:output" for i, dep in enumerate(deps)}
        
        # Create real PipelineExecutionPlan object
        execution_plan = PipelineExecutionPlan(
            execution_order=execution_order,
            step_configs=step_configs,
            dependencies=dependencies,
            data_flow_map=data_flow_map
        )
        
        # Configure resolver behavior
        mock_resolver.create_execution_plan.return_value = execution_plan
        mock_resolver.validate_dag_integrity.return_value = integrity_issues or {}
        mock_resolver.config_resolver = config_resolver
        mock_resolver.get_config_resolution_preview.return_value = None
        
        # Mock dependency methods
        def mock_get_dependencies(step_name):
            return dependencies.get(step_name, [])
        
        def mock_get_dependents(step_name):
            return [node for node, deps in dependencies.items() if step_name in deps]
        
        mock_resolver.get_step_dependencies.side_effect = mock_get_dependencies
        mock_resolver.get_dependent_steps.side_effect = mock_get_dependents
        
        # Mock contract discovery
        def mock_discover_contract(step_name):
            return contracts.get(step_name) if contracts else None
        
        mock_resolver._discover_step_contract.side_effect = mock_discover_contract
        
        return mock_resolver
    
    def create_mock_script_result(self, status="PASS", memory_usage=100, error_message=None):
        """Create a mock script execution result."""
        mock_result = Mock()
        mock_result.status = status
        mock_result.memory_usage = memory_usage
        mock_result.error_message = error_message
        return mock_result
    
    # Basic Functionality Tests
    
    def test_simple_pipeline_execution_success(self):
        """Test successful execution of a simple single-step pipeline."""
        contracts = {"StepA": self.preprocessing_contract}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                result = self.executor.execute_pipeline(self.simple_dag)
                
                assert result.success is True
                assert len(result.completed_steps) == 1
                assert result.completed_steps[0].step_name == "StepA"
                assert result.completed_steps[0].status == "SUCCESS"
                assert result.total_duration > 0
                assert result.execution_plan is not None
    
    def test_linear_pipeline_execution_success(self):
        """Test successful execution of a linear three-step pipeline."""
        contracts = {
            "TabularPreprocessing": self.preprocessing_contract,
            "XGBoostTraining": self.training_contract,
            "XGBoostModelEval": self.evaluation_contract
        }
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.linear_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                result = self.executor.execute_pipeline(self.linear_dag)
                
                assert result.success is True
                assert len(result.completed_steps) == 3
                
                # Verify execution order
                step_names = [step.step_name for step in result.completed_steps]
                assert step_names == ["TabularPreprocessing", "XGBoostTraining", "XGBoostModelEval"]
                
                # Verify all steps succeeded
                for step in result.completed_steps:
                    assert step.status == "SUCCESS"
    
    def test_complex_dag_with_branching(self):
        """Test execution of complex DAG with branching dependencies."""
        contracts = {
            "CradleDataLoading": ScriptContract(
                entry_point="cradle_data_loading.py",
                expected_input_paths={},
                expected_output_paths={"output_data": "/opt/ml/processing/output/data"},
                required_env_vars=[],
                optional_env_vars={}
            ),
            "TabularPreprocessing": self.preprocessing_contract,
            "XGBoostTraining": self.training_contract,
            "XGBoostModelEval": self.evaluation_contract,
            "ModelCalibration": ScriptContract(
                entry_point="model_calibration.py",
                expected_input_paths={"model_input": "/opt/ml/processing/input/model"},
                expected_output_paths={"calibrated_model": "/opt/ml/processing/output/model"},
                required_env_vars=[],
                optional_env_vars={}
            )
        }
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.complex_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                result = self.executor.execute_pipeline(self.complex_dag)
                
                assert result.success is True
                assert len(result.completed_steps) == 5
                
                # Verify topological ordering
                step_names = [step.step_name for step in result.completed_steps]
                assert step_names == ["CradleDataLoading", "TabularPreprocessing", "XGBoostTraining", "XGBoostModelEval", "ModelCalibration"]
    
    # Contract Discovery Integration Tests
    
    def test_contract_discovery_script_path_resolution(self):
        """Test that script paths are resolved using contract discovery."""
        contracts = {
            "TabularPreprocessing": self.preprocessing_contract,
            "XGBoostTraining": self.training_contract,
            "XGBoostModelEval": None  # Test fallback behavior
        }
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.linear_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                result = self.executor.execute_pipeline(self.linear_dag)
                
                # Verify script paths were resolved correctly
                script_calls = mock_script_exec.call_args_list
                assert len(script_calls) == 3
                
                # First two should use contract entry points
                assert script_calls[0][0][0] == "tabular_preprocessing.py"
                assert script_calls[1][0][0] == "xgboost_training.py"
                # Third should use fallback
                assert script_calls[2][0][0] == "model_calibration.py"
    
    def test_environment_variable_validation(self):
        """Test environment variable validation from contracts."""
        contracts = {"StepA": self.preprocessing_contract}  # Has required env vars
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                # Test with missing required env vars
                with patch.dict('os.environ', {}, clear=True):
                    with patch.object(self.executor.logger, 'warning') as mock_warning:
                        result = self.executor.execute_pipeline(self.simple_dag)
                        
                        # Should still succeed but log warning
                        assert result.success is True
                        mock_warning.assert_called_with("Missing required env vars for StepA: ['LABEL_FIELD', 'TRAIN_RATIO']")
                
                # Test with all required env vars present
                with patch.dict('os.environ', {'LABEL_FIELD': 'target', 'TRAIN_RATIO': '0.8'}):
                    with patch.object(self.executor.logger, 'warning') as mock_warning:
                        result = self.executor.execute_pipeline(self.simple_dag)
                        
                        # Should succeed without warnings
                        assert result.success is True
                        mock_warning.assert_not_called()
    
    def test_contract_based_input_output_paths(self):
        """Test that input/output paths are created based on contract specifications."""
        contracts = {"StepA": self.preprocessing_contract}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                result = self.executor.execute_pipeline(self.simple_dag)
                
                # Verify execution succeeded
                assert result.success is True
                
                # Verify workspace directories were created
                step_input_dir = self.workspace_dir / "inputs" / "StepA"
                step_output_dir = self.workspace_dir / "outputs" / "StepA"
                assert step_input_dir.exists()
                assert step_output_dir.exists()
                
                # Verify contract-specific directories were created
                data_input_dir = step_input_dir / "DATA"
                processed_output_dir = step_output_dir / "processed_data"
                assert data_input_dir.exists()
                assert processed_output_dir.exists()
    
    # Error Handling Tests
    
    def test_dag_integrity_validation_failure(self):
        """Test handling of DAG integrity validation failures."""
        integrity_issues = {
            "cycles": ["Cycle detected: StepA -> StepB -> StepA"],
            "dangling_dependencies": ["Edge references non-existent node: StepC"]
        }
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, integrity_issues=integrity_issues)
            mock_resolver_class.return_value = mock_resolver
            
            result = self.executor.execute_pipeline(self.simple_dag)
            
            assert result.success is False
            assert "integrity issues" in result.error
            assert len(result.completed_steps) == 0
    
    def test_step_execution_failure(self):
        """Test handling of step execution failures."""
        contracts = {"StepA": self.preprocessing_contract}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                # Mock script failure
                mock_script_exec.return_value = self.create_mock_script_result(
                    status="FAIL", 
                    error_message="Script execution failed"
                )
                
                result = self.executor.execute_pipeline(self.simple_dag)
                
                assert result.success is False
                assert "Pipeline failed at step StepA" in result.error
                assert len(result.completed_steps) == 1
                assert result.completed_steps[0].status == "FAILURE"
    
    def test_step_execution_exception(self):
        """Test handling of exceptions during step execution."""
        contracts = {"StepA": self.preprocessing_contract}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                # Mock script exception
                mock_script_exec.side_effect = Exception("Unexpected error")
                
                result = self.executor.execute_pipeline(self.simple_dag)
                
                assert result.success is False
                assert "Pipeline failed at step StepA: Unexpected error" in result.error
                assert len(result.completed_steps) == 1
                assert result.completed_steps[0].status == "FAILURE"
                assert result.completed_steps[0].error_message == "Unexpected error"
    
    def test_pipeline_execution_exception(self):
        """Test handling of exceptions during pipeline setup."""
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            # Mock resolver creation failure
            mock_resolver_class.side_effect = Exception("Resolver creation failed")
            
            result = self.executor.execute_pipeline(self.simple_dag)
            
            assert result.success is False
            assert "Pipeline execution failed: Resolver creation failed" in result.error
            assert len(result.completed_steps) == 0
    
    # Data Flow and Dependencies Tests
    
    def test_data_flow_mapping_with_contracts(self):
        """Test data flow mapping using contract-based channel definitions."""
        contracts = {
            "TabularPreprocessing": self.preprocessing_contract,
            "XGBoostTraining": self.training_contract
        }
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.linear_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                # Mock step outputs to test data flow
                step_outputs = {}
                
                def mock_execute_step(step_name, step_config, step_inputs, data_source):
                    # Simulate step execution and output generation
                    outputs = {}
                    if step_name == "TabularPreprocessing":
                        outputs = {"processed_data": {"path": "/test/output", "format": "csv"}}
                    elif step_name == "XGBoostTraining":
                        outputs = {"model_output": {"path": "/test/model", "format": "pkl"}}
                    
                    step_outputs[step_name] = outputs
                    
                    return StepExecutionResult(
                        step_name=step_name,
                        status="SUCCESS",
                        execution_time=1.0,
                        memory_usage=100,
                        outputs=outputs
                    )
                
                with patch.object(self.executor, '_execute_step', side_effect=mock_execute_step):
                    result = self.executor.execute_pipeline(self.linear_dag)
                    
                    assert result.success is True
                    
                    # Verify data flow was established
                    # This would be verified by checking the step inputs in a real implementation
                    assert len(result.completed_steps) == 3
    
    def test_fallback_data_flow_mapping(self):
        """Test fallback data flow mapping when contracts are not available."""
        # No contracts provided - should use fallback mapping
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.linear_dag, contracts=None)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                result = self.executor.execute_pipeline(self.linear_dag)
                
                assert result.success is True
                assert len(result.completed_steps) == 3
                
                # Verify fallback script paths were used
                script_calls = mock_script_exec.call_args_list
                for call in script_calls:
                    assert call[0][0] == "model_calibration.py"  # Fallback script
    
    # Configuration and Metadata Tests
    
    def test_configuration_resolution_integration(self):
        """Test integration with configuration resolution."""
        config_resolver = Mock()
        config_preview = {"step_configs": {"StepA": {"param1": "value1"}}}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, config_resolver=config_resolver)
            mock_resolver.get_config_resolution_preview.return_value = config_preview
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                with patch.object(self.executor.logger, 'info') as mock_info:
                    result = self.executor.execute_pipeline(
                        self.simple_dag,
                        config_path="/test/config.json",
                        metadata={"version": "1.0"}
                    )
                    
                    assert result.success is True
                    
                    # Verify configuration preview was logged
                    mock_info.assert_any_call(f"Configuration resolution preview: {config_preview}")
    
    # Performance and Memory Tests
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking across pipeline execution."""
        contracts = {"StepA": self.preprocessing_contract}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result(memory_usage=500)
                
                result = self.executor.execute_pipeline(self.simple_dag)
                
                assert result.success is True
                assert result.memory_peak == 500
                assert result.completed_steps[0].memory_usage == 500
    
    def test_execution_time_tracking(self):
        """Test execution time tracking for steps and overall pipeline."""
        contracts = {"StepA": self.preprocessing_contract}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                result = self.executor.execute_pipeline(self.simple_dag)
                
                assert result.success is True
                assert result.total_duration > 0
                assert result.completed_steps[0].execution_time > 0
                assert isinstance(result.timestamp, datetime)
                assert isinstance(result.completed_steps[0].timestamp, datetime)
    
    # Logging and Diagnostics Tests
    
    def test_diagnostic_logging(self):
        """Test comprehensive diagnostic logging."""
        contracts = {
            "TabularPreprocessing": self.preprocessing_contract,
            "XGBoostTraining": self.training_contract,
            "XGBoostModelEval": None  # Test mixed contract availability
        }
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.linear_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                with patch.object(self.executor.logger, 'debug') as mock_debug:
                    result = self.executor.execute_pipeline(self.linear_dag)
                    
                    assert result.success is True
                    
                    # Verify diagnostic logging occurred
                    debug_calls = [call.args[0] for call in mock_debug.call_args_list]
                    
                    # Should log contract discovery status
                    contract_logs = [log for log in debug_calls if "contract found" in log or "no contract found" in log]
                    assert len(contract_logs) >= 3  # One for each step
                    
                    # Should log script path resolution
                    script_logs = [log for log in debug_calls if "Using contract entry point" in log]
                    assert len(script_logs) >= 2  # For steps with contracts
    
    def test_step_validation_reporting(self):
        """Test step output validation and reporting."""
        contracts = {"StepA": self.preprocessing_contract}
        
        with patch('src.cursus.validation.runtime.execution.pipeline_executor.PipelineDAGResolver') as mock_resolver_class:
            mock_resolver = self.create_mock_resolver(self.simple_dag, contracts)
            mock_resolver_class.return_value = mock_resolver
            
            with patch.object(self.executor.script_executor, 'test_script_isolation') as mock_script_exec:
                mock_script_exec.return_value = self.create_mock_script_result()
                
                with patch.object(self.executor.data_validator, 'validate_step_transition') as mock_validate:
                    mock_report = Mock()
                    mock_validate.return_value = mock_report
                    
                    result = self.executor.execute_pipeline(self.simple_dag)
                    
                    assert result.success is True
                    # Validation report should be attached to step result
                    # (In this case, no dependent steps, so validation might not be called)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
