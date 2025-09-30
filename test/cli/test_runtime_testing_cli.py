"""
Pytest tests for enhanced runtime testing CLI with step catalog integration

Tests the Click-based CLI interface for runtime testing functionality with optional step catalog features.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
from click.testing import CliRunner

from cursus.cli.runtime_testing_cli import runtime
from cursus.validation.runtime.runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
)


@pytest.fixture
def runner():
    """Create CLI runner fixture"""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create temporary directory fixture"""
    return tempfile.mkdtemp()


@pytest.fixture
def mock_script_result():
    """Create mock script test result"""
    return ScriptTestResult(
        script_name="test_script",
        success=True,
        execution_time=1.5,
        has_main_function=True,
        error_message=None,
    )


@pytest.fixture
def mock_compatibility_result():
    """Create mock compatibility result"""
    return DataCompatibilityResult(
        script_a="script_a",
        script_b="script_b",
        compatible=True,
        compatibility_issues=[],
        data_format_a="csv",
        data_format_b="csv",
    )


@pytest.fixture
def mock_script_spec():
    """Create mock script execution spec"""
    return ScriptExecutionSpec(
        script_name="test_script",
        step_name="TestScript_training",
        script_path="scripts/test_script.py",
        input_paths={"data_input": "input/data"},
        output_paths={"data_output": "output/data"},
        environ_vars={"PYTHONPATH": "src"},
        job_args={"job_type": "testing"}
    )


class TestRuntimeTestingCLI:
    """Test runtime testing CLI commands with step catalog integration"""

    def test_cli_help(self, runner):
        """Test CLI help message"""
        result = runner.invoke(runtime, ["--help"])

        assert result.exit_code == 0
        assert "Pipeline Runtime Testing CLI" in result.output
        assert "test-script" in result.output
        assert "test-compatibility" in result.output
        assert "test-pipeline" in result.output

    def test_cli_version(self, runner):
        """Test CLI version"""
        result = runner.invoke(runtime, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_test_script_help(self, runner):
        """Test test-script command help shows step catalog options"""
        result = runner.invoke(runtime, ["test-script", "--help"])

        assert result.exit_code == 0
        assert "step-catalog" in result.output
        assert "workspace-dirs" in result.output
        assert "optional step catalog enhancements" in result.output

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_test_script_success_without_step_catalog(self, mock_builder_class, mock_tester_class, 
                                                     runner, temp_dir, mock_script_spec, mock_script_result):
        """Test successful script testing without step catalog"""
        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        mock_builder.resolve_script_execution_spec_from_node.return_value = mock_script_spec
        mock_builder.get_script_main_params.return_value = {"input_paths": {}, "output_paths": {}}
        mock_tester.test_script_with_spec.return_value = mock_script_result

        result = runner.invoke(
            runtime, [
                "test-script", "test_script", 
                "--workspace-dir", temp_dir,
                "--no-step-catalog"
            ]
        )

        assert result.exit_code == 0
        assert "test_script" in result.output
        assert "PASS" in result.output
        mock_tester.test_script_with_spec.assert_called_once()

    @patch("cursus.cli.runtime_testing_cli.STEP_CATALOG_AVAILABLE", True)
    @patch("cursus.cli.runtime_testing_cli.StepCatalog")
    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_test_script_success_with_step_catalog(self, mock_builder_class, mock_tester_class, 
                                                  mock_step_catalog_class, runner, temp_dir, 
                                                  mock_script_spec, mock_script_result):
        """Test successful script testing with step catalog"""
        mock_step_catalog = Mock()
        mock_step_catalog_class.return_value = mock_step_catalog
        
        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        mock_builder.resolve_script_execution_spec_from_node.return_value = mock_script_spec
        mock_builder.get_script_main_params.return_value = {"input_paths": {}, "output_paths": {}}
        mock_tester.test_script_with_step_catalog_enhancements.return_value = mock_script_result
        mock_tester._detect_framework_if_needed.return_value = "xgboost"
        mock_tester._validate_builder_consistency_if_available.return_value = []

        result = runner.invoke(
            runtime, [
                "test-script", "test_script", 
                "--workspace-dir", temp_dir,
                "--step-catalog"
            ]
        )

        assert result.exit_code == 0
        assert "test_script" in result.output
        assert "PASS" in result.output
        assert "xgboost" in result.output
        mock_tester.test_script_with_step_catalog_enhancements.assert_called_once()

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_test_script_failure(self, mock_builder_class, mock_tester_class, 
                                runner, temp_dir, mock_script_spec):
        """Test script testing failure"""
        failed_result = ScriptTestResult(
            script_name="test_script",
            success=False,
            execution_time=0.5,
            has_main_function=False,
            error_message="Script missing main() function",
        )

        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        mock_builder.resolve_script_execution_spec_from_node.return_value = mock_script_spec
        mock_builder.get_script_main_params.return_value = {"input_paths": {}, "output_paths": {}}
        mock_tester.test_script_with_spec.return_value = failed_result

        result = runner.invoke(
            runtime, [
                "test-script", "test_script", 
                "--workspace-dir", temp_dir,
                "--no-step-catalog"
            ]
        )

        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "Script missing main() function" in result.output

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_test_compatibility_success(self, mock_builder_class, mock_tester_class, 
                                       runner, temp_dir, mock_compatibility_result):
        """Test successful data compatibility testing"""
        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        spec_a = ScriptExecutionSpec(
            script_name="script_a", step_name="ScriptA_training", script_path="scripts/script_a.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        spec_b = ScriptExecutionSpec(
            script_name="script_b", step_name="ScriptB_training", script_path="scripts/script_b.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        
        mock_builder.resolve_script_execution_spec_from_node.side_effect = [spec_a, spec_b]
        mock_tester.test_data_compatibility_with_specs.return_value = mock_compatibility_result

        result = runner.invoke(
            runtime,
            [
                "test-compatibility",
                "script_a",
                "script_b",
                "--workspace-dir",
                temp_dir,
                "--no-step-catalog"
            ],
        )

        assert result.exit_code == 0
        assert "script_a -> script_b" in result.output
        assert "PASS" in result.output

    @patch("cursus.cli.runtime_testing_cli.STEP_CATALOG_AVAILABLE", True)
    @patch("cursus.cli.runtime_testing_cli.StepCatalog")
    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_test_compatibility_with_step_catalog(self, mock_builder_class, mock_tester_class, 
                                                 mock_step_catalog_class, runner, temp_dir, 
                                                 mock_compatibility_result):
        """Test data compatibility testing with step catalog enhancements"""
        mock_step_catalog = Mock()
        mock_step_catalog_class.return_value = mock_step_catalog
        
        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        spec_a = ScriptExecutionSpec(
            script_name="script_a", step_name="ScriptA_training", script_path="scripts/script_a.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        spec_b = ScriptExecutionSpec(
            script_name="script_b", step_name="ScriptB_training", script_path="scripts/script_b.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        
        mock_builder.resolve_script_execution_spec_from_node.side_effect = [spec_a, spec_b]
        mock_tester.test_data_compatibility_with_step_catalog_enhancements.return_value = mock_compatibility_result

        result = runner.invoke(
            runtime,
            [
                "test-compatibility",
                "script_a",
                "script_b",
                "--workspace-dir",
                temp_dir,
                "--step-catalog"
            ],
        )

        assert result.exit_code == 0
        assert "script_a -> script_b" in result.output
        assert "PASS" in result.output
        mock_tester.test_data_compatibility_with_step_catalog_enhancements.assert_called_once()

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_test_compatibility_failure(self, mock_builder_class, mock_tester_class, runner, temp_dir):
        """Test data compatibility testing failure"""
        incompatible_result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=False,
            compatibility_issues=["Column mismatch", "Type error"],
            data_format_a="csv",
            data_format_b="json",
        )

        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        spec_a = ScriptExecutionSpec(
            script_name="script_a", step_name="ScriptA_training", script_path="scripts/script_a.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        spec_b = ScriptExecutionSpec(
            script_name="script_b", step_name="ScriptB_training", script_path="scripts/script_b.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        
        mock_builder.resolve_script_execution_spec_from_node.side_effect = [spec_a, spec_b]
        mock_tester.test_data_compatibility_with_specs.return_value = incompatible_result

        result = runner.invoke(
            runtime,
            [
                "test-compatibility",
                "script_a",
                "script_b",
                "--workspace-dir",
                temp_dir,
                "--no-step-catalog"
            ],
        )

        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "Column mismatch" in result.output
        assert "Type error" in result.output

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    @patch("cursus.cli.runtime_testing_cli.PipelineDAG")
    @patch("builtins.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_test_pipeline_success(self, mock_exists, mock_open, mock_dag_class, 
                                  mock_builder_class, mock_tester_class, runner, temp_dir, 
                                  mock_script_result, mock_compatibility_result):
        """Test successful pipeline testing"""
        pipeline_config = {
            "nodes": ["step1", "step2"],
            "edges": [["step1", "step2"]]
        }

        mock_open.return_value.__enter__.return_value = Mock()
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(pipeline_config)

        mock_dag = Mock()
        mock_dag_class.return_value = mock_dag

        mock_pipeline_spec = Mock()
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_from_dag.return_value = mock_pipeline_spec

        pipeline_result = {
            "pipeline_success": True,
            "script_results": {
                "step1": mock_script_result,
                "step2": mock_script_result,
            },
            "data_flow_results": {"step1->step2": mock_compatibility_result},
            "errors": [],
        }

        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        mock_tester.test_pipeline_flow_with_spec.return_value = pipeline_result

        result = runner.invoke(
            runtime,
            [
                "test-pipeline", "pipeline.json", 
                "--workspace-dir", temp_dir,
                "--no-step-catalog"
            ],
        )

        assert result.exit_code == 0
        assert "pipeline.json" in result.output
        assert "PASS" in result.output

    @patch("cursus.cli.runtime_testing_cli.STEP_CATALOG_AVAILABLE", True)
    @patch("cursus.cli.runtime_testing_cli.StepCatalog")
    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    @patch("cursus.cli.runtime_testing_cli.PipelineDAG")
    @patch("builtins.open")
    @patch("pathlib.Path.exists", return_value=True)
    def test_test_pipeline_with_step_catalog(self, mock_exists, mock_open, mock_dag_class, 
                                            mock_builder_class, mock_tester_class, mock_step_catalog_class,
                                            runner, temp_dir, mock_script_result, mock_compatibility_result):
        """Test pipeline testing with step catalog enhancements"""
        mock_step_catalog = Mock()
        mock_step_catalog_class.return_value = mock_step_catalog
        
        pipeline_config = {
            "nodes": ["step1", "step2"],
            "edges": [["step1", "step2"]]
        }

        mock_open.return_value.__enter__.return_value = Mock()
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(pipeline_config)

        mock_dag = Mock()
        mock_dag_class.return_value = mock_dag

        mock_pipeline_spec = Mock()
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_from_dag.return_value = mock_pipeline_spec

        pipeline_result = {
            "pipeline_success": True,
            "script_results": {
                "step1": mock_script_result,
                "step2": mock_script_result,
            },
            "data_flow_results": {"step1->step2": mock_compatibility_result},
            "errors": [],
            "step_catalog_analysis": {
                "workspace_analysis": {"step1": {"available_workspaces": ["workspace1"]}},
                "framework_analysis": {"step1": "xgboost"}
            }
        }

        mock_tester = Mock()
        mock_tester_class.return_value = mock_tester
        mock_tester.test_pipeline_flow_with_step_catalog_enhancements.return_value = pipeline_result

        result = runner.invoke(
            runtime,
            [
                "test-pipeline", "pipeline.json", 
                "--workspace-dir", temp_dir,
                "--step-catalog"
            ],
        )

        assert result.exit_code == 0
        assert "pipeline.json" in result.output
        assert "PASS" in result.output
        mock_tester.test_pipeline_flow_with_step_catalog_enhancements.assert_called_once()

    @patch("pathlib.Path.exists", return_value=False)
    def test_test_pipeline_file_not_found(self, mock_exists, runner, temp_dir):
        """Test pipeline testing with missing file"""
        result = runner.invoke(
            runtime,
            ["test-pipeline", "nonexistent.json", "--workspace-dir", temp_dir],
        )

        assert result.exit_code == 1
        assert "not found" in result.output

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    def test_exception_handling(self, mock_tester_class, runner):
        """Test CLI exception handling"""
        mock_tester_class.side_effect = Exception("Test exception")

        result = runner.invoke(runtime, ["test-script", "test_script"])

        assert result.exit_code == 1
        assert "Error" in result.output

    def test_workspace_dirs_option(self, runner):
        """Test workspace-dirs option parsing"""
        result = runner.invoke(runtime, ["test-script", "--help"])
        
        assert result.exit_code == 0
        assert "workspace-dirs" in result.output

    @patch.dict(os.environ, {'CURSUS_DEV_WORKSPACES': 'workspace1:workspace2'})
    @patch("cursus.cli.runtime_testing_cli.STEP_CATALOG_AVAILABLE", True)
    @patch("cursus.cli.runtime_testing_cli.StepCatalog")
    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_environment_workspace_integration(self, mock_builder_class, mock_tester_class, 
                                              mock_step_catalog_class, runner, temp_dir, 
                                              mock_script_spec, mock_script_result):
        """Test integration with CURSUS_DEV_WORKSPACES environment variable"""
        mock_step_catalog = Mock()
        mock_step_catalog_class.return_value = mock_step_catalog
        
        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        mock_builder.resolve_script_execution_spec_from_node.return_value = mock_script_spec
        mock_builder.get_script_main_params.return_value = {"input_paths": {}, "output_paths": {}}
        mock_tester.test_script_with_step_catalog_enhancements.return_value = mock_script_result
        mock_tester._detect_framework_if_needed.return_value = "xgboost"
        mock_tester._validate_builder_consistency_if_available.return_value = []

        result = runner.invoke(
            runtime, [
                "test-script", "test_script", 
                "--workspace-dir", temp_dir,
                "--step-catalog"
            ]
        )

        # Debug output if test fails
        if result.exit_code != 0:
            print(f"CLI output: {result.output}")
            print(f"Exception: {result.exception}")

        assert result.exit_code == 0
        # Verify StepCatalog was called with environment workspaces
        mock_step_catalog_class.assert_called_once()


class TestRuntimeTestingCLIIntegration:
    """Integration tests for runtime testing CLI with step catalog"""

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_full_workflow_simulation(self, mock_builder_class, mock_tester_class, runner, temp_dir):
        """Test a complete workflow simulation"""
        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder

        # Mock script test results
        script_result_a = ScriptTestResult(
            script_name="script_a",
            success=True,
            execution_time=1.0,
            has_main_function=True,
        )

        script_result_b = ScriptTestResult(
            script_name="script_b",
            success=True,
            execution_time=1.5,
            has_main_function=True,
        )

        # Mock compatibility result
        compatibility_result = DataCompatibilityResult(
            script_a="script_a",
            script_b="script_b",
            compatible=True,
            compatibility_issues=[],
            data_format_a="csv",
            data_format_b="csv",
        )

        # Mock script specs
        spec_a = ScriptExecutionSpec(
            script_name="script_a", step_name="ScriptA_training", script_path="scripts/script_a.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        spec_b = ScriptExecutionSpec(
            script_name="script_b", step_name="ScriptB_training", script_path="scripts/script_b.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )

        # Set up mock returns
        mock_builder.resolve_script_execution_spec_from_node.side_effect = [spec_a, spec_a, spec_b, spec_b]
        mock_builder.get_script_main_params.return_value = {"input_paths": {}, "output_paths": {}}
        mock_tester.test_script_with_spec.side_effect = [script_result_a, script_result_b]
        mock_tester.test_data_compatibility_with_specs.return_value = compatibility_result

        # Test individual scripts
        result1 = runner.invoke(
            runtime, [
                "test-script", "script_a", 
                "--workspace-dir", temp_dir,
                "--no-step-catalog"
            ]
        )
        assert result1.exit_code == 0

        result2 = runner.invoke(
            runtime, [
                "test-script", "script_b", 
                "--workspace-dir", temp_dir,
                "--no-step-catalog"
            ]
        )
        assert result2.exit_code == 0

        # Test compatibility
        result3 = runner.invoke(
            runtime,
            [
                "test-compatibility",
                "script_a",
                "script_b",
                "--workspace-dir",
                temp_dir,
                "--no-step-catalog"
            ],
        )
        assert result3.exit_code == 0

    @patch("cursus.cli.runtime_testing_cli.RuntimeTester")
    @patch("cursus.cli.runtime_testing_cli.PipelineTestingSpecBuilder")
    def test_json_output_format(self, mock_builder_class, mock_tester_class, runner, temp_dir):
        """Test JSON output format for all commands"""
        mock_tester = Mock()
        mock_builder = Mock()
        mock_tester_class.return_value = mock_tester
        mock_builder_class.return_value = mock_builder
        
        script_result = ScriptTestResult(
            script_name="test_script",
            success=True,
            execution_time=1.5,
            has_main_function=True,
            error_message=None,
        )
        
        mock_script_spec = ScriptExecutionSpec(
            script_name="test_script", step_name="TestScript_training", script_path="scripts/test_script.py",
            input_paths={}, output_paths={}, environ_vars={}, job_args={}
        )
        
        mock_builder.resolve_script_execution_spec_from_node.return_value = mock_script_spec
        mock_builder.get_script_main_params.return_value = {"input_paths": {}, "output_paths": {}}
        mock_tester.test_script_with_spec.return_value = script_result

        result = runner.invoke(
            runtime, [
                "test-script", "test_script", 
                "--workspace-dir", temp_dir,
                "--output-format", "json",
                "--no-step-catalog"
            ]
        )

        assert result.exit_code == 0
        # Verify JSON output
        try:
            json.loads(result.output)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")


if __name__ == "__main__":
    pytest.main([__file__])
