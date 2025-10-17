"""
Unit tests for script testing base classes.

Tests the core base classes: ScriptExecutionSpec, ScriptExecutionPlan, and ScriptTestResult.
Validates maximum component reuse and integration with existing cursus infrastructure.

Following pytest best practices:
1. Read source code first - COMPLETED
2. Use correct import paths (no src prefix for pip install .)
3. Mock at correct locations based on actual imports
4. Match test expectations to implementation reality
5. Use proper fixtures and isolation
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Correct imports for pip install . (no src prefix)
from cursus.script_testing.base import (
    ScriptExecutionSpec,
    ScriptExecutionPlan,
    ScriptTestResult,
    PipelineDAG,
    StepCatalog,
)


class TestScriptExecutionSpec:
    """Test ScriptExecutionSpec class functionality."""
    
    def test_basic_creation(self):
        """Test basic ScriptExecutionSpec creation."""
        spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="TestStep_training",
            script_path="/path/to/test_script.py"
        )
        
        assert spec.script_name == "test_script"
        assert spec.step_name == "TestStep_training"
        assert spec.script_path == "/path/to/test_script.py"
        assert isinstance(spec.input_paths, dict)
        assert isinstance(spec.output_paths, dict)
        assert isinstance(spec.environ_vars, dict)
        assert isinstance(spec.job_args, dict)
        assert isinstance(spec.last_updated, datetime)
    
    def test_validation_errors(self):
        """Test validation errors for invalid inputs."""
        # Empty script_name should fail
        with pytest.raises(ValueError, match="script_name cannot be empty"):
            ScriptExecutionSpec(
                script_name="",
                step_name="TestStep_training",
                script_path="/path/to/script.py"
            )
        
        # Empty step_name should fail
        with pytest.raises(ValueError, match="step_name cannot be empty"):
            ScriptExecutionSpec(
                script_name="test_script",
                step_name="",
                script_path="/path/to/script.py"
            )
        
        # Empty script_path should fail
        with pytest.raises(ValueError, match="script_path cannot be empty"):
            ScriptExecutionSpec(
                script_name="test_script",
                step_name="TestStep_training",
                script_path=""
            )
    
    def test_step_name_conversion(self):
        """Test step name to script name conversion."""
        # Test PascalCase conversion
        result = ScriptExecutionSpec._convert_step_name_to_script_name("TestStep_training")
        assert result == "test_step_training"
        
        # Test simple PascalCase
        result = ScriptExecutionSpec._convert_step_name_to_script_name("SimpleStep")
        assert result == "simple_step"
        
        # Test already snake_case
        result = ScriptExecutionSpec._convert_step_name_to_script_name("already_snake_case")
        assert result == "already_snake_case"
    
    def test_create_from_node_name(self):
        """Test creating spec from node name with fallback discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = ScriptExecutionSpec.create_from_node_name(
                step_name="TestStep_training",
                test_workspace_dir=temp_dir
            )
            
            assert spec.script_name == "test_step_training"
            assert spec.step_name == "TestStep_training"
            assert spec.script_path == str(Path(temp_dir) / "scripts" / "test_step_training.py")
            assert "input_data" in spec.input_paths
            assert "output_data" in spec.output_paths
    
    # CRITICAL: Mock at source location, not importing module
    # Source shows: from ...step_catalog import StepCatalog
    @patch('cursus.step_catalog.StepCatalog')
    def test_create_from_step_catalog(self, mock_step_catalog_class):
        """Test creating spec using step catalog integration."""
        # Mock step catalog and its methods
        mock_step_catalog = Mock()
        mock_step_info = Mock()
        mock_script_metadata = Mock()
        mock_script_metadata.path = Path("/workspace/scripts/test_script.py")
        mock_step_info.file_components = {"script": mock_script_metadata}
        mock_step_catalog.resolve_pipeline_node.return_value = mock_step_info
        mock_step_catalog.load_contract_class.return_value = None
        
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = ScriptExecutionSpec.create_from_step_catalog(
                step_name="TestStep_training",
                step_catalog=mock_step_catalog,
                test_workspace_dir=temp_dir
            )
            
            assert spec.script_name == "test_script"
            assert spec.step_name == "TestStep_training"
            assert spec.script_path == "/workspace/scripts/test_script.py"
            
            # Verify step catalog was used
            mock_step_catalog.resolve_pipeline_node.assert_called_once_with("TestStep_training")
            mock_step_catalog.load_contract_class.assert_called_once_with("TestStep_training")
    
    def test_contract_aware_paths(self):
        """Test contract-aware path resolution."""
        mock_step_catalog = Mock()
        mock_contract = Mock()
        mock_contract.get_input_paths.return_value = {"train_data": "path", "model_config": "path"}
        mock_contract.get_output_paths.return_value = {"trained_model": "path", "metrics": "path"}
        mock_step_catalog.load_contract_class.return_value = mock_contract
        
        with tempfile.TemporaryDirectory() as temp_dir:
            input_paths, output_paths = ScriptExecutionSpec._get_contract_aware_paths(
                "TestStep_training", mock_step_catalog, temp_dir
            )
            
            assert "train_data" in input_paths
            assert "model_config" in input_paths
            assert "trained_model" in output_paths
            assert "metrics" in output_paths
            
            # Verify paths are in test workspace
            assert temp_dir in input_paths["train_data"]
            assert temp_dir in output_paths["trained_model"]
    
    def test_validate_paths_exist(self):
        """Test path existence validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test script file
            script_path = Path(temp_dir) / "test_script.py"
            script_path.write_text("# test script")
            
            # Create input file
            input_dir = Path(temp_dir) / "input"
            input_dir.mkdir()
            input_file = input_dir / "input.json"
            input_file.write_text("{}")
            
            spec = ScriptExecutionSpec(
                script_name="test_script",
                step_name="TestStep_training",
                script_path=str(script_path),
                input_paths={"input_data": str(input_file)},
                output_paths={"output_data": str(Path(temp_dir) / "output" / "output.json")}
            )
            
            # Test validation
            result = spec.validate_paths_exist(check_inputs=True, check_outputs=False)
            assert result["script_exists"] is True
            assert len(result["missing_inputs"]) == 0
            assert result["all_valid"] is True
            
            # Test with missing output
            result = spec.validate_paths_exist(check_inputs=True, check_outputs=True)
            assert result["script_exists"] is True
            assert len(result["missing_outputs"]) == 1
            assert result["all_valid"] is False
    
    def test_get_main_params(self):
        """Test getting main function parameters."""
        spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="TestStep_training",
            script_path="/path/to/script.py",
            input_paths={"input": "/path/to/input"},
            output_paths={"output": "/path/to/output"},
            environ_vars={"ENV_VAR": "value"},
            job_args={"arg1": "value1", "arg2": 42}
        )
        
        params = spec.get_main_params()
        
        assert params["input_paths"] == {"input": "/path/to/input"}
        assert params["output_paths"] == {"output": "/path/to/output"}
        assert params["environ_vars"] == {"ENV_VAR": "value"}
        assert hasattr(params["job_args"], "arg1")
        assert params["job_args"].arg1 == "value1"
        assert params["job_args"].arg2 == 42
    
    def test_update_paths_from_dependencies(self):
        """Test updating input paths from dependency outputs."""
        spec = ScriptExecutionSpec(
            script_name="test_script",
            step_name="TestStep_training",
            script_path="/path/to/script.py",
            input_paths={"input_data": "/default/path", "model_config": "/default/config"}
        )
        
        dependency_outputs = {
            "DataPrep_processing": {
                "processed_data": "/new/processed_data.json",
                "data_summary": "/new/summary.json"
            }
        }
        
        spec.update_paths_from_dependencies(dependency_outputs)
        
        # Should update input_data to match processed_data
        assert spec.input_paths["input_data"] == "/new/processed_data.json"
        # model_config should remain unchanged
        assert spec.input_paths["model_config"] == "/default/config"
    
    def test_save_and_load_from_file(self):
        """Test saving and loading specification from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = ScriptExecutionSpec(
                script_name="test_script",
                step_name="TestStep_training",
                script_path="/path/to/script.py",
                input_paths={"input": "/path/to/input"},
                output_paths={"output": "/path/to/output"},
                user_notes="Test specification"
            )
            
            # Save to file
            spec_file = Path(temp_dir) / "test_spec.json"
            saved_path = spec.save_to_file(str(spec_file))
            
            assert saved_path == spec_file
            assert spec_file.exists()
            
            # Load from file
            loaded_spec = ScriptExecutionSpec.load_from_file(str(spec_file))
            
            assert loaded_spec.script_name == spec.script_name
            assert loaded_spec.step_name == spec.step_name
            assert loaded_spec.script_path == spec.script_path
            assert loaded_spec.input_paths == spec.input_paths
            assert loaded_spec.output_paths == spec.output_paths
            assert loaded_spec.user_notes == spec.user_notes
    
    def test_load_from_file_not_found(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Specification file not found"):
            ScriptExecutionSpec.load_from_file("/nonexistent/file.json")


class TestScriptExecutionPlan:
    """Test ScriptExecutionPlan class functionality."""
    
    @pytest.fixture
    def mock_dag(self):
        """Create a mock DAG for testing."""
        mock_dag = Mock(spec=PipelineDAG)
        mock_dag.nodes = ["Node1", "Node2"]
        mock_dag.edges = [("Node1", "Node2")]
        mock_dag.topological_sort.return_value = ["Node1", "Node2"]
        mock_dag.get_dependencies.return_value = []
        return mock_dag
    
    @pytest.fixture
    def sample_script_specs(self):
        """Create sample script specs for testing."""
        spec1 = ScriptExecutionSpec(
            script_name="node1_script",
            step_name="Node1",
            script_path="/path/to/node1.py"
        )
        spec2 = ScriptExecutionSpec(
            script_name="node2_script", 
            step_name="Node2",
            script_path="/path/to/node2.py"
        )
        return {"Node1": spec1, "Node2": spec2}
    
    def test_basic_creation(self, mock_dag, sample_script_specs):
        """Test basic ScriptExecutionPlan creation."""
        plan = ScriptExecutionPlan(
            dag=mock_dag,
            script_specs=sample_script_specs,
            execution_order=["Node1", "Node2"],
            test_workspace_dir="/test/workspace"
        )
        
        assert plan.dag == mock_dag
        assert len(plan.script_specs) == 2
        assert plan.execution_order == ["Node1", "Node2"]
        assert plan.test_workspace_dir == "/test/workspace"
        assert isinstance(plan.created_at, datetime)
    
    def test_validate_execution_plan(self, mock_dag):
        """Test execution plan validation."""
        # Create script specs with existing script
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "test_script.py"
            script_path.write_text("# test script")
            
            spec1 = ScriptExecutionSpec(
                script_name="test_script",
                step_name="Node1",
                script_path=str(script_path)
            )
            
            plan = ScriptExecutionPlan(
                dag=mock_dag,
                script_specs={"Node1": spec1},
                execution_order=["Node1", "Node2"],  # Mismatched order
                test_workspace_dir=temp_dir
            )
            
            validation = plan.validate_execution_plan()
            
            # Should have warnings about execution order
            assert len(validation["warnings"]) > 0
            assert any("execution order" in warning.lower() for warning in validation["warnings"])
    
    def test_preview_execution(self, mock_dag, sample_script_specs):
        """Test execution plan preview."""
        # Configure mock DAG for dependencies
        mock_dag.get_dependencies.side_effect = lambda node: ["Node1"] if node == "Node2" else []
        
        # Update script specs with paths
        sample_script_specs["Node1"].output_paths = {"output": "/path/to/output1"}
        sample_script_specs["Node2"].input_paths = {"input": "/path/to/input2"}
        
        plan = ScriptExecutionPlan(
            dag=mock_dag,
            script_specs=sample_script_specs,
            execution_order=["Node1", "Node2"],
            test_workspace_dir="/test/workspace"
        )
        
        preview = plan.preview_execution()
        
        assert "execution_summary" in preview
        assert preview["execution_summary"]["total_scripts"] == 2
        assert preview["execution_summary"]["execution_order"] == ["Node1", "Node2"]
        
        assert "script_preview" in preview
        assert "Node1" in preview["script_preview"]
        assert "Node2" in preview["script_preview"]
        
        assert "dependency_analysis" in preview
        assert "Node2" in preview["dependency_analysis"]
        assert preview["dependency_analysis"]["Node2"]["depends_on"] == ["Node1"]
    
    def test_save_and_load_from_file(self, sample_script_specs):
        """Test saving and loading execution plan from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock DAG with model_dump method
            mock_dag = Mock(spec=PipelineDAG)
            mock_dag.model_dump.return_value = {"nodes": ["Node1"], "edges": []}
            
            plan = ScriptExecutionPlan(
                dag=mock_dag,
                script_specs={"Node1": sample_script_specs["Node1"]},
                execution_order=["Node1"],
                test_workspace_dir="/test/workspace"
            )
            
            # Save to file
            plan_file = Path(temp_dir) / "test_plan.json"
            saved_path = plan.save_to_file(str(plan_file))
            
            assert saved_path == plan_file
            assert plan_file.exists()
            
            # Verify file contents
            with open(plan_file, 'r') as f:
                plan_data = json.load(f)
            
            assert "dag" in plan_data
            assert "script_specs" in plan_data
            assert "execution_order" in plan_data
            assert plan_data["execution_order"] == ["Node1"]
    
    def test_load_from_file_not_found(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Execution plan file not found"):
            ScriptExecutionPlan.load_from_file("/nonexistent/plan.json")
    
    def test_get_script_dependencies(self, mock_dag, sample_script_specs):
        """Test getting script dependencies."""
        mock_dag.get_dependencies.return_value = ["Node1"]
        
        plan = ScriptExecutionPlan(
            dag=mock_dag,
            script_specs=sample_script_specs,
            execution_order=["Node1", "Node2"],
            test_workspace_dir="/test/workspace"
        )
        
        dependencies = plan.get_script_dependencies("Node2")
        assert dependencies == ["Node1"]
        mock_dag.get_dependencies.assert_called_once_with("Node2")
    
    def test_get_script_dependents(self, mock_dag, sample_script_specs):
        """Test getting script dependents."""
        # Mock get_dependencies to return different values for different nodes
        def mock_get_deps(node):
            if node == "Node2":
                return ["Node1"]
            return []
        
        mock_dag.get_dependencies.side_effect = mock_get_deps
        
        plan = ScriptExecutionPlan(
            dag=mock_dag,
            script_specs=sample_script_specs,
            execution_order=["Node1", "Node2"],
            test_workspace_dir="/test/workspace"
        )
        
        dependents = plan.get_script_dependents("Node1")
        assert dependents == ["Node2"]


class TestScriptTestResult:
    """Test ScriptTestResult class functionality."""
    
    def test_create_success(self):
        """Test creating successful test result."""
        result = ScriptTestResult.create_success(
            script_name="test_script",
            step_name="TestStep_training",
            execution_time=1.5,
            output_files=["/path/to/output1.json", "/path/to/output2.json"],
            custom_metadata="test_value"
        )
        
        assert result.script_name == "test_script"
        assert result.step_name == "TestStep_training"
        assert result.success is True
        assert result.execution_time == 1.5
        assert result.output_files == ["/path/to/output1.json", "/path/to/output2.json"]
        assert result.error_message is None
        assert result.metadata["custom_metadata"] == "test_value"
        assert isinstance(result.execution_timestamp, datetime)
    
    def test_create_failure(self):
        """Test creating failed test result."""
        result = ScriptTestResult.create_failure(
            script_name="test_script",
            step_name="TestStep_training",
            execution_time=0.5,
            error_message="Script execution failed",
            has_main_function=False,
            error_code=1
        )
        
        assert result.script_name == "test_script"
        assert result.step_name == "TestStep_training"
        assert result.success is False
        assert result.execution_time == 0.5
        assert result.error_message == "Script execution failed"
        assert result.has_main_function is False
        assert result.metadata["error_code"] == 1
    
    def test_add_framework_info(self):
        """Test adding framework information."""
        result = ScriptTestResult.create_success(
            script_name="test_script",
            step_name="TestStep_training",
            execution_time=1.0
        )
        
        result.add_framework_info("pytorch", version="1.9.0", gpu_available=True)
        
        assert result.framework_info["detected_framework"] == "pytorch"
        assert result.framework_info["version"] == "1.9.0"
        assert result.framework_info["gpu_available"] is True
    
    def test_add_builder_consistency_info(self):
        """Test adding builder consistency information."""
        result = ScriptTestResult.create_success(
            script_name="test_script",
            step_name="TestStep_training",
            execution_time=1.0
        )
        
        result.add_builder_consistency_info(
            consistent=False,
            missing_inputs=["train_data"],
            extra_outputs=["debug_info"]
        )
        
        assert result.builder_consistency["builder_consistent"] is False
        assert result.builder_consistency["missing_inputs"] == ["train_data"]
        assert result.builder_consistency["extra_outputs"] == ["debug_info"]
    
    def test_add_warning(self):
        """Test adding warning messages."""
        result = ScriptTestResult.create_success(
            script_name="test_script",
            step_name="TestStep_training",
            execution_time=1.0
        )
        
        result.add_warning("Performance warning: slow execution")
        result.add_warning("Data warning: missing validation")
        
        assert len(result.warnings) == 2
        assert "Performance warning" in result.warnings[0]
        assert "Data warning" in result.warnings[1]
    
    def test_get_output_summary(self):
        """Test getting output summary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some output files
            output1 = Path(temp_dir) / "output1.json"
            output1.write_text("{}")
            
            result = ScriptTestResult.create_success(
                script_name="test_script",
                step_name="TestStep_training",
                execution_time=1.0,
                output_files=[str(output1), "/nonexistent/output2.json"]
            )
            
            summary = result.get_output_summary()
            
            assert summary["total_outputs"] == 2
            assert len(summary["outputs_exist"]) == 2
            assert summary["outputs_exist"][0]["exists"] is True
            assert summary["outputs_exist"][1]["exists"] is False
            assert len(summary["missing_outputs"]) == 1
            assert "/nonexistent/output2.json" in summary["missing_outputs"]
    
    def test_is_successful_with_outputs(self):
        """Test checking success with output validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create output file
            output_file = Path(temp_dir) / "output.json"
            output_file.write_text("{}")
            
            # Test successful result with existing outputs
            result = ScriptTestResult.create_success(
                script_name="test_script",
                step_name="TestStep_training",
                execution_time=1.0,
                output_files=[str(output_file)]
            )
            
            assert result.is_successful_with_outputs() is True
            
            # Test successful result with missing outputs
            result.output_files = ["/nonexistent/file.json"]
            assert result.is_successful_with_outputs() is False
            
            # Test failed result
            result = ScriptTestResult.create_failure(
                script_name="test_script",
                step_name="TestStep_training",
                execution_time=0.5,
                error_message="Failed"
            )
            
            assert result.is_successful_with_outputs() is False
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        result = ScriptTestResult.create_success(
            script_name="test_script",
            step_name="TestStep_training",
            execution_time=2.5
        )
        
        # Add framework performance info
        result.add_framework_info("pytorch", performance_metrics={"gpu_utilization": 85.5})
        
        metrics = result.get_performance_metrics()
        
        assert metrics["execution_time_seconds"] == 2.5
        assert "execution_timestamp" in metrics
        assert metrics["gpu_utilization"] == 85.5
    
    def test_get_execution_summary(self):
        """Test getting execution summary."""
        result = ScriptTestResult.create_success(
            script_name="test_script",
            step_name="TestStep_training",
            execution_time=1.0
        )
        
        result.add_warning("Test warning")
        result.add_framework_info("pytorch")
        result.add_builder_consistency_info(consistent=True)
        
        summary = result.get_execution_summary()
        
        assert summary["script_name"] == "test_script"
        assert summary["step_name"] == "TestStep_training"
        assert summary["success"] is True
        assert summary["execution_time"] == 1.0
        assert summary["warnings_count"] == 1
        assert summary["framework"] == "pytorch"
        assert summary["builder_consistent"] is True


# Integration tests for component interaction
class TestBaseClassesIntegration:
    """Test integration between base classes."""
    
    def test_spec_to_plan_integration(self):
        """Test creating execution plan from specs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock DAG
            mock_dag = Mock(spec=PipelineDAG)
            mock_dag.nodes = ["TestStep"]
            mock_dag.topological_sort.return_value = ["TestStep"]
            mock_dag.get_dependencies.return_value = []
            
            # Create spec
            spec = ScriptExecutionSpec.create_from_node_name(
                step_name="TestStep",
                test_workspace_dir=temp_dir
            )
            
            # Create plan
            plan = ScriptExecutionPlan(
                dag=mock_dag,
                script_specs={"TestStep": spec},
                execution_order=["TestStep"],
                test_workspace_dir=temp_dir
            )
            
            # Verify integration
            assert len(plan.script_specs) == 1
            assert "TestStep" in plan.script_specs
            assert plan.script_specs["TestStep"].script_name == "test_step"
    
    def test_plan_to_result_integration(self):
        """Test creating results from plan execution."""
        # Create a simple result
        result = ScriptTestResult.create_success(
            script_name="test_script",
            step_name="TestStep",
            execution_time=1.0,
            output_files=["/path/to/output.json"]
        )
        
        # Verify result structure matches plan expectations
        assert result.script_name == "test_script"
        assert result.step_name == "TestStep"
        assert result.success is True
        assert len(result.output_files) == 1


if __name__ == "__main__":
    pytest.main([__file__])
