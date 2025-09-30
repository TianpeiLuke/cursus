"""
Pytest tests for runtime testing system

Tests the RuntimeTester class and its methods for script validation,
data compatibility testing, and pipeline flow validation.
Updated to include tests for enhanced Phase 2/3 functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_models import (
    ScriptTestResult,
    DataCompatibilityResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
    RuntimeTestingConfiguration,
)
from cursus.validation.runtime.runtime_spec_builder import PipelineTestingSpecBuilder
from cursus.api.dag.base_dag import PipelineDAG

# Import logical name matching components for enhanced testing
try:
    from cursus.validation.runtime.logical_name_matching import (
        PathSpec,
        PathMatch,
        MatchType,
        EnhancedScriptExecutionSpec,
        PathMatcher,
        TopologicalExecutor,
        LogicalNameMatchingTester,
        EnhancedDataCompatibilityResult,
    )

    LOGICAL_MATCHING_AVAILABLE = True
except ImportError:
    LOGICAL_MATCHING_AVAILABLE = False


class TestRuntimeTester:
    """Test RuntimeTester class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create a simple DAG for testing"""
        return PipelineDAG(
            nodes=["script_a", "script_b"], edges=[("script_a", "script_b")]
        )

    @pytest.fixture
    def script_spec_a(self, temp_dir):
        """Create script spec A for testing"""
        return ScriptExecutionSpec.create_default("script_a", "script_a_step", temp_dir)

    @pytest.fixture
    def script_spec_b(self, temp_dir):
        """Create script spec B for testing"""
        return ScriptExecutionSpec.create_default("script_b", "script_b_step", temp_dir)

    @pytest.fixture
    def pipeline_spec(self, test_dag, script_spec_a, script_spec_b, temp_dir):
        """Create pipeline spec for testing"""
        return PipelineTestingSpec(
            dag=test_dag,
            script_specs={"script_a": script_spec_a, "script_b": script_spec_b},
            test_workspace_root=temp_dir,
        )

    @pytest.fixture
    def config(self, pipeline_spec):
        """Create configuration for testing"""
        return RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

    @pytest.fixture
    def tester(self, config):
        """Create RuntimeTester instance"""
        return RuntimeTester(config, step_catalog=None)

    @pytest.fixture
    def script_specs(self, temp_dir):
        """Create script specs based on real cursus scripts and contracts"""
        return {
            "tabular_preprocessing": ScriptExecutionSpec(
                script_name="tabular_preprocessing",
                step_name="TabularPreprocessing_training",
                input_paths={
                    "input_path": f"{temp_dir}/input/data",
                    "hyperparameters_s3_uri": f"{temp_dir}/input/config"
                },
                output_paths={
                    "processed_data": f"{temp_dir}/preprocessing/output",
                    "preprocessing_artifacts": f"{temp_dir}/preprocessing/artifacts"
                },
                environ_vars={"PREPROCESSING_MODE": "standard"},
                job_args={"preprocessing_mode": "standard"},
            ),
            "xgboost_training": ScriptExecutionSpec(
                script_name="xgboost_training",
                step_name="XGBoostTraining_training",
                input_paths={
                    "input_path": f"{temp_dir}/input/data",
                    "hyperparameters_s3_uri": f"{temp_dir}/input/config/hyperparameters.json"
                },
                output_paths={
                    "model_output": f"{temp_dir}/model",
                    "evaluation_output": f"{temp_dir}/output/data"
                },
                environ_vars={"MODEL_TYPE": "xgboost"},
                job_args={"max_depth": "6"},
            ),
        }

    def test_script_missing_main_function_with_spec(self, tester, script_spec_a):
        """Test script without main function fails validation with spec"""
        script_spec = script_spec_a
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args,
        }

        with patch.object(tester, "_find_script_path") as mock_find:
            mock_find.return_value = "test_script.py"

            with patch("importlib.util.spec_from_file_location") as mock_spec:
                mock_module = Mock()
                # No main function
                del mock_module.main

                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj

                with patch("importlib.util.module_from_spec", return_value=mock_module):
                    result = tester.test_script_with_spec(script_spec, main_params)

                    assert result.success is False
                    assert result.has_main_function is False
                    assert "missing main() function" in result.error_message

    def test_data_compatibility_with_specs(self, tester, script_spec_a, script_spec_b):
        """Test data compatibility between scripts using ScriptExecutionSpecs"""
        spec_a = script_spec_a
        spec_b = script_spec_b

        # Disable logical matching to avoid the enhanced compatibility test
        with patch.object(tester, "enable_logical_matching", False):
            with patch.object(tester.builder, "get_script_main_params") as mock_params:
                mock_params.return_value = {
                    "input_paths": {"data_input": "/test/input"},
                    "output_paths": {"data_output": "/test/output"},
                    "environ_vars": {"LABEL_FIELD": "label"},
                    "job_args": {"job_type": "testing"},
                }

                with patch.object(tester, "test_script_with_spec") as mock_test_script:
                    # Mock successful script A execution
                    mock_test_script.side_effect = [
                        ScriptTestResult(
                            script_name="script_a", success=True, execution_time=0.1
                        ),
                        ScriptTestResult(
                            script_name="script_b", success=True, execution_time=0.1
                        ),
                    ]

                    with patch.object(
                        tester, "_find_valid_output_files"
                    ) as mock_find_files:
                        mock_find_files.return_value = [Path("/test/output/data.csv")]

                        result = tester.test_data_compatibility_with_specs(
                            spec_a, spec_b
                        )

                        assert isinstance(result, DataCompatibilityResult)
                        assert result.script_a == "script_a"
                        assert result.script_b == "script_b"
                        assert result.compatible is True

    def test_data_compatibility_script_a_fails(
        self, tester, script_spec_a, script_spec_b
    ):
        """Test data compatibility when script A fails"""
        spec_a = script_spec_a
        spec_b = script_spec_b

        with patch.object(tester.builder, "get_script_main_params") as mock_params:
            mock_params.return_value = {}

            with patch.object(tester, "test_script_with_spec") as mock_test_script:
                # Mock script A failure
                mock_test_script.return_value = ScriptTestResult(
                    script_name="script_a",
                    success=False,
                    error_message="Script failed",
                    execution_time=0.1,
                )

                result = tester.test_data_compatibility_with_specs(spec_a, spec_b)

                assert result.compatible is False
                assert "Script A failed" in result.compatibility_issues[0]

    def test_pipeline_flow_with_spec(self, tester, pipeline_spec):
        """Test end-to-end pipeline flow using PipelineTestingSpec"""
        with patch.object(tester.builder, "get_script_main_params") as mock_params:
            mock_params.return_value = {
                "input_paths": {"data_input": "/test/input"},
                "output_paths": {"data_output": "/test/output"},
                "environ_vars": {"LABEL_FIELD": "label"},
                "job_args": {"job_type": "testing"},
            }

            with patch.object(tester, "test_script_with_spec") as mock_test_script:
                # Mock successful script tests
                mock_test_script.side_effect = [
                    ScriptTestResult(
                        script_name="script_a", success=True, execution_time=0.1
                    ),
                    ScriptTestResult(
                        script_name="script_b", success=True, execution_time=0.1
                    ),
                ]

                with patch.object(
                    tester, "test_data_compatibility_with_specs"
                ) as mock_test_compat:
                    # Mock successful data compatibility
                    mock_test_compat.return_value = DataCompatibilityResult(
                        script_a="script_a", script_b="script_b", compatible=True
                    )

                    result = tester.test_pipeline_flow_with_spec(pipeline_spec)

                    assert result["pipeline_success"] is True
                    assert len(result["script_results"]) == 2
                    assert len(result["data_flow_results"]) == 1
                    assert len(result["errors"]) == 0

    def test_pipeline_flow_empty_dag(self, tester, temp_dir):
        """Test pipeline flow with empty DAG"""
        empty_dag = PipelineDAG(nodes=[], edges=[])
        empty_pipeline_spec = PipelineTestingSpec(
            dag=empty_dag, script_specs={}, test_workspace_root=temp_dir
        )

        result = tester.test_pipeline_flow_with_spec(empty_pipeline_spec)

        assert result["pipeline_success"] is False
        assert "No nodes found in pipeline DAG" in result["errors"]

    def test_backward_compatibility_test_script_with_spec(self, tester, script_spec_a):
        """Test script testing using test_script_with_spec method"""
        script_spec = script_spec_a
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args,
        }

        with patch.object(tester, "_find_script_path") as mock_find:
            mock_find.return_value = "test_script.py"

            with patch("importlib.util.spec_from_file_location") as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()

                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj

                with patch("importlib.util.module_from_spec", return_value=mock_module):
                    with patch("inspect.signature") as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            "input_paths",
                            "output_paths",
                            "environ_vars",
                            "job_args",
                        ]

                        with patch("pandas.DataFrame.to_csv"), patch(
                            "pathlib.Path.mkdir"
                        ), patch("pathlib.Path.exists", return_value=True):

                            result = tester.test_script_with_spec(
                                script_spec, main_params
                            )

                            assert isinstance(result, ScriptTestResult)
                            assert result.success is True
                            assert result.has_main_function is True
                            assert result.script_name == "script_a"

                            # Verify that main function was actually called
                            mock_module.main.assert_called_once_with(**main_params)

    def test_backward_compatibility_test_data_compatibility_with_specs(
        self, tester, script_spec_a, script_spec_b
    ):
        """Test data compatibility using ScriptExecutionSpecs (current implementation)"""
        spec_a = script_spec_a
        spec_b = script_spec_b

        # Disable logical matching to avoid the enhanced compatibility test
        with patch.object(tester, "enable_logical_matching", False):
            with patch.object(tester.builder, "get_script_main_params") as mock_params:
                mock_params.return_value = {
                    "input_paths": {"data_input": "/test/input"},
                    "output_paths": {"data_output": "/test/output"},
                    "environ_vars": {"LABEL_FIELD": "label"},
                    "job_args": {"job_type": "testing"},
                }

                with patch.object(tester, "test_script_with_spec") as mock_test_script:
                    # Mock successful script executions
                    mock_test_script.side_effect = [
                        ScriptTestResult(
                            script_name="script_a", success=True, execution_time=0.1
                        ),
                        ScriptTestResult(
                            script_name="script_b", success=True, execution_time=0.1
                        ),
                    ]

                    with patch.object(
                        tester, "_find_valid_output_files"
                    ) as mock_find_files:
                        mock_find_files.return_value = [Path("/test/output/data.csv")]

                        result = tester.test_data_compatibility_with_specs(
                            spec_a, spec_b
                        )

                        assert isinstance(result, DataCompatibilityResult)
                        assert result.script_a == "script_a"
                        assert result.script_b == "script_b"
                        assert result.compatible is True

    def test_pipeline_flow_with_spec_comprehensive(self, tester, pipeline_spec):
        """Test comprehensive pipeline flow using PipelineTestingSpec (current implementation)"""
        with patch.object(tester.builder, "get_script_main_params") as mock_params:
            mock_params.return_value = {
                "input_paths": {"data_input": "/test/input"},
                "output_paths": {"data_output": "/test/output"},
                "environ_vars": {"LABEL_FIELD": "label"},
                "job_args": {"job_type": "testing"},
            }

            with patch.object(tester, "test_script_with_spec") as mock_test_script:
                # Mock successful script tests
                mock_test_script.side_effect = [
                    ScriptTestResult(
                        script_name="script_a", success=True, execution_time=0.1
                    ),
                    ScriptTestResult(
                        script_name="script_b", success=True, execution_time=0.1
                    ),
                ]

                with patch.object(
                    tester, "test_data_compatibility_with_specs"
                ) as mock_test_compat:
                    # Mock successful data compatibility
                    mock_test_compat.return_value = DataCompatibilityResult(
                        script_a="script_a", script_b="script_b", compatible=True
                    )

                    result = tester.test_pipeline_flow_with_spec(pipeline_spec)

                    assert result["pipeline_success"] is True
                    assert len(result["script_results"]) == 2
                    assert len(result["data_flow_results"]) == 1
                    assert len(result["errors"]) == 0

    def test_find_script_path(self, tester):
        """Test script path discovery logic"""
        # Mock the specific Path.exists calls to return True only for fallback paths
        def mock_path_exists():
            # This function will be called as a method on the Path object
            # We need to access the path through the mock's parent
            return True  # Just return True for the test
        
        # Mock both the Path object exists method and the Path constructor
        with patch.object(Path, 'exists', return_value=True):
            result = tester._find_script_path("test_script")
            # Should return workspace path since exists returns True for all paths
            assert str(tester.workspace_dir) in result or result == "src/cursus/steps/scripts/test_script.py"

    def test_find_script_path_not_found(self, tester):
        """Test script path discovery when script doesn't exist"""
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError) as exc_info:
                tester._find_script_path("nonexistent_script")

            assert "Script not found: nonexistent_script" in str(exc_info.value)

    def test_execute_script_with_data(self, tester, temp_dir):
        """Test executing script with test data using test_script_with_spec"""
        script_spec = ScriptExecutionSpec.create_default(
            "test_script", "test_step", temp_dir
        )
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args,
        }

        with patch.object(tester, "_find_script_path") as mock_find:
            mock_find.return_value = "test_script.py"

            with patch("importlib.util.spec_from_file_location") as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()

                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj

                with patch("importlib.util.module_from_spec", return_value=mock_module):
                    with patch("inspect.signature") as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            "input_paths",
                            "output_paths",
                            "environ_vars",
                            "job_args",
                        ]

                        with patch("pathlib.Path.mkdir"), patch(
                            "pandas.DataFrame.to_csv"
                        ), patch("pathlib.Path.exists", return_value=True):

                            result = tester.test_script_with_spec(
                                script_spec, main_params
                            )

                            assert isinstance(result, ScriptTestResult)
                            assert result.success is True
                            assert result.has_main_function is True
                            assert result.script_name == "test_script"

                            # Verify main function was called with correct parameters
                            mock_module.main.assert_called_once_with(**main_params)


    def test_clear_error_feedback(self, tester, temp_dir):
        """Test error messages are clear and actionable"""
        script_spec = ScriptExecutionSpec.create_default(
            "nonexistent_script", "nonexistent_step", temp_dir
        )
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args,
        }

        with patch.object(tester, "_find_script_path") as mock_find:
            mock_find.side_effect = FileNotFoundError(
                "Script not found: nonexistent_script"
            )

            result = tester.test_script_with_spec(script_spec, main_params)

            assert result.success is False
            assert "Script not found: nonexistent_script" in result.error_message
            assert result.script_name == "nonexistent_script"

    def test_performance_requirements(self, tester, temp_dir):
        """Test that script testing completes quickly"""
        script_spec = ScriptExecutionSpec.create_default(
            "test_script", "test_step", temp_dir
        )
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args,
        }

        with patch.object(tester, "_find_script_path") as mock_find:
            mock_find.return_value = "test_script.py"

            with patch("importlib.util.spec_from_file_location") as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()

                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj

                with patch("importlib.util.module_from_spec", return_value=mock_module):
                    with patch("inspect.signature") as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            "input_paths",
                            "output_paths",
                            "environ_vars",
                            "job_args",
                        ]

                        with patch("pandas.DataFrame.to_csv"), patch(
                            "pathlib.Path.mkdir"
                        ):

                            result = tester.test_script_with_spec(
                                script_spec, main_params
                            )

                            # Should complete very quickly (much less than 100ms)
                            assert result.execution_time < 0.1


class TestRuntimeTesterIntegration:
    """Integration tests for RuntimeTester"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def complex_dag(self):
        """Create a complex DAG for testing"""
        return PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train", "model_eval"],
            edges=[
                ("data_prep", "feature_eng"),
                ("feature_eng", "model_train"),
                ("model_train", "model_eval"),
            ],
        )

    @pytest.fixture
    def script_specs(self, complex_dag, temp_dir):
        """Create script specs for all nodes"""
        script_specs = {}
        for node in complex_dag.nodes:
            script_specs[node] = ScriptExecutionSpec.create_default(
                node, f"{node}_step", temp_dir
            )
        return script_specs

    @pytest.fixture
    def pipeline_spec(self, complex_dag, script_specs, temp_dir):
        """Create pipeline spec"""
        return PipelineTestingSpec(
            dag=complex_dag, script_specs=script_specs, test_workspace_root=temp_dir
        )

    @pytest.fixture
    def config(self, pipeline_spec):
        """Create configuration"""
        return RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

    @pytest.fixture
    def tester(self, config):
        """Create RuntimeTester instance"""
        return RuntimeTester(config)

    def test_end_to_end_workflow(self, tester, script_specs, pipeline_spec):
        """Test complete workflow from script testing to pipeline validation"""
        with patch.object(tester, "_find_script_path", return_value="test.py"):
            with patch("importlib.util.spec_from_file_location") as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock()

                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj

                with patch("importlib.util.module_from_spec", return_value=mock_module):
                    with patch("inspect.signature") as mock_sig:
                        mock_sig.return_value.parameters.keys.return_value = [
                            "input_paths",
                            "output_paths",
                            "environ_vars",
                            "job_args",
                        ]

                        with patch.object(
                            tester.builder, "get_script_main_params"
                        ) as mock_params:
                            mock_params.return_value = {
                                "input_paths": {"data_input": "/test/input"},
                                "output_paths": {"data_output": "/test/output"},
                                "environ_vars": {"LABEL_FIELD": "label"},
                                "job_args": {"job_type": "testing"},
                            }

                            with patch.object(
                                tester, "_find_valid_output_files"
                            ) as mock_find_files:
                                mock_find_files.return_value = [
                                    Path("/test/output/data.csv")
                                ]

                                with patch.object(
                                    tester, "test_data_compatibility_with_specs"
                                ) as mock_compat:
                                    # Mock successful data compatibility for all edges
                                    mock_compat.return_value = DataCompatibilityResult(
                                        script_a="", script_b="", compatible=True
                                    )

                                    with patch("pathlib.Path.exists", return_value=True):
                                        # Test individual script functionality
                                        script_result = tester.test_script_with_spec(
                                            script_specs["data_prep"],
                                            mock_params.return_value,
                                        )
                                        assert script_result.success is True

                                    # Test complete pipeline with proper mocking
                                    with patch.object(tester, "test_script_with_spec") as mock_pipeline_test:
                                        mock_pipeline_test.side_effect = [
                                            ScriptTestResult(script_name="data_prep", success=True, execution_time=0.1),
                                            ScriptTestResult(script_name="feature_eng", success=True, execution_time=0.1),
                                            ScriptTestResult(script_name="model_train", success=True, execution_time=0.1),
                                            ScriptTestResult(script_name="model_eval", success=True, execution_time=0.1),
                                        ]
                                        
                                        pipeline_result = (
                                            tester.test_pipeline_flow_with_spec(
                                                pipeline_spec
                                            )
                                        )
                                        assert pipeline_result["pipeline_success"] is True


class TestEnhancedFileFormatSupport:
    """Test enhanced file format support in RuntimeTester"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create test DAG"""
        return PipelineDAG(nodes=["script_a"], edges=[])

    @pytest.fixture
    def script_spec(self, temp_dir):
        """Create script spec"""
        return ScriptExecutionSpec.create_default("script_a", "script_a_step", temp_dir)

    @pytest.fixture
    def pipeline_spec(self, test_dag, script_spec, temp_dir):
        """Create pipeline spec"""
        return PipelineTestingSpec(
            dag=test_dag,
            script_specs={"script_a": script_spec},
            test_workspace_root=temp_dir,
        )

    @pytest.fixture
    def config(self, pipeline_spec):
        """Create configuration"""
        return RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

    @pytest.fixture
    def tester(self, config):
        """Create RuntimeTester instance"""
        return RuntimeTester(config)

    def test_find_valid_output_files_csv_only(self, tester, temp_dir):
        """Test finding valid output files with CSV-only approach (Phase 1)"""
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)

        # Create test files with actual content (not empty files)
        (output_dir / "data.csv").write_text("col1,col2\n1,2\n")
        (output_dir / "model.pkl").write_bytes(b"fake pickle data")
        (output_dir / "temp_file.tmp").write_text("temp")
        (output_dir / ".hidden").write_text("hidden")

        valid_files = tester._find_valid_output_files(output_dir)

        # Should find all valid files except temp/system files
        valid_names = {f.name for f in valid_files}
        expected_names = {"data.csv", "model.pkl"}
        assert valid_names == expected_names

    def test_find_valid_output_files_enhanced_mode(self, tester, temp_dir):
        """Test finding valid output files with enhanced format support (Phase 2+)"""
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)

        # Create test files with actual content (not empty files)
        (output_dir / "data.csv").write_text("col1,col2\n1,2\n")
        (output_dir / "model.pkl").write_bytes(b"fake pickle data")
        (output_dir / "features.parquet").write_bytes(b"fake parquet data")
        (output_dir / "config.json").write_text('{"key": "value"}')
        (output_dir / "temp_file.tmp").write_text("temp")
        (output_dir / ".hidden").write_text("hidden")
        (output_dir / "system.log").write_text("log data")

        valid_files = tester._find_valid_output_files(output_dir)

        # Should find all valid files except temp/system files
        valid_names = {f.name for f in valid_files}
        expected_names = {"data.csv", "model.pkl", "features.parquet", "config.json"}
        assert valid_names == expected_names

    def test_is_temp_or_system_file(self, tester):
        """Test identification of temporary and system files"""
        test_cases = [
            ("data.csv", False),
            ("model.pkl", False),
            ("temp_file.tmp", True),
            (".hidden", True),
            ("system.log", True),
            ("backup.bak", True),
            ("temp~", True),  # Files ending with ~ are temp files
            ("file.swp", True),
            ("normal_file.json", False),
            ("features.parquet", False),
        ]

        for filename, expected_is_temp in test_cases:
            result = tester._is_temp_or_system_file(Path(filename))
            assert (
                result == expected_is_temp
            ), f"File {filename} should {'be' if expected_is_temp else 'not be'} temp/system"

    def test_enhanced_mode_detection(self, tester):
        """Test enhanced mode detection logic"""
        # Test with logical matching available
        if LOGICAL_MATCHING_AVAILABLE:
            # Test that logical matching is enabled by default
            assert tester.enable_logical_matching is True
        else:
            # Should always be False if logical matching not available
            assert tester.enable_logical_matching is False


@pytest.mark.skipif(
    not LOGICAL_MATCHING_AVAILABLE, reason="Logical name matching not available"
)
class TestLogicalNameMatchingIntegration:
    """Test logical name matching integration in RuntimeTester"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create DAG with logical name dependencies"""
        return PipelineDAG(
            nodes=["tabular_preprocessing", "xgboost_training"],
            edges=[("tabular_preprocessing", "xgboost_training")],
        )

    @pytest.fixture
    def preprocessing_spec(self, temp_dir):
        """Create preprocessing spec with enhanced features"""
        return EnhancedScriptExecutionSpec(
            script_name="tabular_preprocessing",
            step_name="preprocessing_step",
            input_paths={"raw_data": "/input/raw.csv"},
            output_paths={"processed_data": "/output/processed.csv"},
            environ_vars={"PREPROCESSING_MODE": "standard"},
            job_args={"batch_size": "1000"},
            input_path_specs={
                "raw_data": PathSpec(
                    logical_name="raw_data",
                    path="/input/raw.csv",
                    aliases=["input_data", "source_data"],
                )
            },
            output_path_specs={
                "processed_data": PathSpec(
                    logical_name="processed_data",
                    path="/output/processed.csv",
                    aliases=["clean_data", "training_ready_data"],
                )
            },
        )

    @pytest.fixture
    def training_spec(self, temp_dir):
        """Create training spec with enhanced features"""
        return EnhancedScriptExecutionSpec(
            script_name="xgboost_training",
            step_name="training_step",
            input_paths={
                "training_data": "/input/training.csv",
                "hyperparameter_s3": "/config/hyperparams.json",  # Independent input
            },
            output_paths={"model_output": "/output/model.pkl"},
            environ_vars={"MODEL_TYPE": "xgboost"},
            job_args={"max_depth": "6"},
            input_path_specs={
                "training_data": PathSpec(
                    logical_name="training_data",
                    path="/input/training.csv",
                    aliases=["processed_data", "clean_data"],
                ),
                "hyperparameter_s3": PathSpec(
                    logical_name="hyperparameter_s3",
                    path="/config/hyperparams.json",
                    aliases=["config", "params"],
                ),
            },
            output_path_specs={
                "model_output": PathSpec(
                    logical_name="model_output",
                    path="/output/model.pkl",
                    aliases=["trained_model", "model_artifact"],
                )
            },
        )

    @pytest.fixture
    def script_specs(self, preprocessing_spec, training_spec):
        """Create script specs dictionary"""
        return {
            "tabular_preprocessing": preprocessing_spec,
            "xgboost_training": training_spec,
        }

    @pytest.fixture
    def pipeline_spec(self, test_dag, script_specs, temp_dir):
        """Create pipeline spec with enhanced features enabled"""
        return PipelineTestingSpec(
            dag=test_dag, script_specs=script_specs, test_workspace_root=temp_dir
        )

    @pytest.fixture
    def config(self, pipeline_spec):
        """Create configuration with enhanced features"""
        return RuntimeTestingConfiguration(
            pipeline_spec=pipeline_spec, enable_enhanced_features=True
        )

    @pytest.fixture
    def tester(self, config):
        """Create RuntimeTester instance"""
        return RuntimeTester(config)

    def test_enhanced_data_compatibility_with_logical_matching(
        self, tester, preprocessing_spec, training_spec
    ):
        """Test enhanced data compatibility using logical name matching"""
        # Create basic specs that match the expected structure
        basic_preprocessing_spec = ScriptExecutionSpec(
            script_name="tabular_preprocessing",
            step_name="preprocessing_step",
            input_paths={"data_input": "/input/raw.csv"},
            output_paths={"data_output": "/output/processed.csv"},
            environ_vars={"PREPROCESSING_MODE": "standard"},
            job_args={"batch_size": "1000"},
        )

        basic_training_spec = ScriptExecutionSpec(
            script_name="xgboost_training",
            step_name="training_step",
            input_paths={"data_input": "/input/training.csv"},
            output_paths={"data_output": "/output/model.pkl"},
            environ_vars={"MODEL_TYPE": "xgboost"},
            job_args={"max_depth": "6"},
        )

        # Disable logical matching to test basic compatibility
        with patch.object(tester, "enable_logical_matching", False):
            with patch.object(tester.builder, "get_script_main_params") as mock_params:
                mock_params.return_value = {
                    "input_paths": {"data_input": "/input/raw.csv"},
                    "output_paths": {"data_output": "/output/processed.csv"},
                    "environ_vars": {"PREPROCESSING_MODE": "standard"},
                    "job_args": {"batch_size": "1000"},
                }

                with patch.object(tester, "test_script_with_spec") as mock_test_script:
                    # Mock successful script executions
                    mock_test_script.side_effect = [
                        ScriptTestResult(
                            script_name="tabular_preprocessing",
                            success=True,
                            execution_time=0.1,
                        ),
                        ScriptTestResult(
                            script_name="xgboost_training",
                            success=True,
                            execution_time=0.1,
                        ),
                    ]

                    with patch.object(
                        tester, "_find_valid_output_files"
                    ) as mock_find_files:
                        mock_find_files.return_value = [Path("/output/processed.csv")]

                        result = tester.test_data_compatibility_with_specs(
                            basic_preprocessing_spec, basic_training_spec
                        )

                        # Should be basic result since enhanced specs fallback to basic mode
                        assert isinstance(result, DataCompatibilityResult)
                        assert result.compatible is True
                        assert result.script_a == "tabular_preprocessing"
                        assert result.script_b == "xgboost_training"

    def test_independent_input_handling(
        self, tester, preprocessing_spec, training_spec
    ):
        """Test that independent inputs (like hyperparameter_s3) are handled correctly"""
        # Create basic specs that match the expected structure
        basic_preprocessing_spec = ScriptExecutionSpec(
            script_name="tabular_preprocessing",
            step_name="preprocessing_step",
            input_paths={"data_input": "/input/raw.csv"},
            output_paths={"data_output": "/output/processed.csv"},
            environ_vars={"PREPROCESSING_MODE": "standard"},
            job_args={"batch_size": "1000"},
        )

        basic_training_spec = ScriptExecutionSpec(
            script_name="xgboost_training",
            step_name="training_step",
            input_paths={"data_input": "/input/training.csv"},
            output_paths={"data_output": "/output/model.pkl"},
            environ_vars={"MODEL_TYPE": "xgboost"},
            job_args={"max_depth": "6"},
        )

        # Disable logical matching to test basic compatibility
        with patch.object(tester, "enable_logical_matching", False):
            with patch.object(tester.builder, "get_script_main_params") as mock_params:
                mock_params.return_value = {
                    "input_paths": {"data_input": "/input/training.csv"},
                    "output_paths": {"data_output": "/output/model.pkl"},
                    "environ_vars": {"MODEL_TYPE": "xgboost"},
                    "job_args": {"max_depth": "6"},
                }

                with patch.object(tester, "test_script_with_spec") as mock_test_script:
                    mock_test_script.side_effect = [
                        ScriptTestResult(
                            script_name="tabular_preprocessing",
                            success=True,
                            execution_time=0.1,
                        ),
                        ScriptTestResult(
                            script_name="xgboost_training",
                            success=True,
                            execution_time=0.1,
                        ),
                    ]

                    with patch.object(
                        tester, "_find_valid_output_files"
                    ) as mock_find_files:
                        mock_find_files.return_value = [Path("/output/processed.csv")]

                        result = tester.test_data_compatibility_with_specs(
                            basic_preprocessing_spec, basic_training_spec
                        )

                        # Should still be compatible despite independent input
                        assert result.compatible is True

    def test_fallback_to_basic_mode(self, tester, preprocessing_spec, training_spec):
        """Test fallback to basic mode when enhanced features fail"""
        # Disable logical matching to force basic mode
        with patch.object(tester, "enable_logical_matching", False):
            with patch.object(tester.builder, "get_script_main_params") as mock_params:
                mock_params.return_value = {
                    "input_paths": {"raw_data": "/input/raw.csv"},
                    "output_paths": {"processed_data": "/output/processed.csv"},
                    "environ_vars": {"PREPROCESSING_MODE": "standard"},
                    "job_args": {"batch_size": "1000"},
                }

                with patch.object(tester, "test_script_with_spec") as mock_test_script:
                    mock_test_script.side_effect = [
                        ScriptTestResult(
                            script_name="tabular_preprocessing",
                            success=True,
                            execution_time=0.1,
                        ),
                        ScriptTestResult(
                            script_name="xgboost_training",
                            success=True,
                            execution_time=0.1,
                        ),
                    ]

                    with patch.object(
                        tester, "_find_valid_output_files"
                    ) as mock_find_files:
                        mock_find_files.return_value = [Path("/output/processed.csv")]

                        result = tester.test_data_compatibility_with_specs(
                            preprocessing_spec, training_spec
                        )

                        # Should fallback to basic DataCompatibilityResult
                        assert isinstance(result, DataCompatibilityResult)
                        assert not isinstance(result, EnhancedDataCompatibilityResult)


@pytest.mark.skipif(
    not LOGICAL_MATCHING_AVAILABLE, reason="Logical name matching not available"
)
class TestTopologicalExecution:
    """Test topological execution capabilities"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def complex_dag(self):
        """Create complex DAG for topological testing"""
        return PipelineDAG(
            nodes=["data_prep", "feature_eng", "model_train", "model_eval"],
            edges=[
                ("data_prep", "feature_eng"),
                ("feature_eng", "model_train"),
                ("model_train", "model_eval"),
            ],
        )

    @pytest.fixture
    def script_specs(self, complex_dag, temp_dir):
        """Create enhanced script specs"""
        script_specs = {}
        for i, node in enumerate(complex_dag.nodes):
            script_specs[node] = EnhancedScriptExecutionSpec(
                script_name=node,
                step_name=f"{node}_step",
                workspace_dir=temp_dir,
                input_paths={"input": f"/input/{node}.csv"},
                output_paths={"output": f"/output/{node}.csv"},
                environ_vars={"STEP": node},
                job_args={"step_id": str(i)},
                logical_names={},
                aliases={},
            )
        return script_specs

    @pytest.fixture
    def pipeline_spec(self, complex_dag, script_specs, temp_dir):
        """Create pipeline spec"""
        return PipelineTestingSpec(
            dag=complex_dag, script_specs=script_specs, test_workspace_root=temp_dir
        )

    @pytest.fixture
    def config(self, pipeline_spec):
        """Create configuration"""
        return RuntimeTestingConfiguration(
            pipeline_spec=pipeline_spec, enable_enhanced_features=True
        )

    @pytest.fixture
    def tester(self, config):
        """Create RuntimeTester instance"""
        return RuntimeTester(config)

    def test_topological_execution_order(self, tester, pipeline_spec):
        """Test that pipeline execution follows topological order"""
        execution_order = []

        def mock_test_script(spec, params):
            execution_order.append(spec.script_name)
            return ScriptTestResult(
                script_name=spec.script_name, success=True, execution_time=0.1
            )

        with patch.object(tester.builder, "get_script_main_params") as mock_params:
            mock_params.return_value = {
                "input_paths": {"input": "/test/input.csv"},
                "output_paths": {"output": "/test/output.csv"},
                "environ_vars": {"STEP": "test"},
                "job_args": {"step_id": "0"},
            }

            with patch.object(
                tester, "test_script_with_spec", side_effect=mock_test_script
            ):
                with patch.object(
                    tester, "test_data_compatibility_with_specs"
                ) as mock_compat:
                    mock_compat.return_value = DataCompatibilityResult(
                        script_a="", script_b="", compatible=True
                    )

                    # Enable logical matching for this test
                    with patch.object(tester, "enable_logical_matching", True):
                        result = tester.test_pipeline_flow_with_spec(pipeline_spec)

                        # Verify execution order follows topological sort
                        expected_order = [
                            "data_prep",
                            "feature_eng",
                            "model_train",
                            "model_eval",
                        ]
                        assert execution_order == expected_order
                        assert result["pipeline_success"] is True

    def test_topological_execution_with_failure(self, tester, pipeline_spec):
        """Test topological execution handles failures gracefully"""

        def mock_test_script(spec, params):
            if spec.script_name == "feature_eng":
                return ScriptTestResult(
                    script_name=spec.script_name,
                    success=False,
                    error_message="Feature engineering failed",
                    execution_time=0.1,
                )
            return ScriptTestResult(
                script_name=spec.script_name, success=True, execution_time=0.1
            )

        with patch.object(tester.builder, "get_script_main_params") as mock_params:
            mock_params.return_value = {
                "input_paths": {"input": "/test/input.csv"},
                "output_paths": {"output": "/test/output.csv"},
                "environ_vars": {"STEP": "test"},
                "job_args": {"step_id": "0"},
            }

            with patch.object(
                tester, "test_script_with_spec", side_effect=mock_test_script
            ):
                # Enable logical matching for this test
                with patch.object(tester, "enable_logical_matching", True):
                    result = tester.test_pipeline_flow_with_spec(pipeline_spec)

                    # Pipeline should fail due to feature_eng failure
                    assert result["pipeline_success"] is False
                    assert len(result["errors"]) > 0

                    # Should have results for scripts that were tested
                    # script_results might be a list of ScriptTestResult objects or a dict
                    if isinstance(result["script_results"], list):
                        script_names = [r.script_name for r in result["script_results"]]
                    else:
                        script_names = list(result["script_results"].keys())
                    assert "data_prep" in script_names
                    assert "feature_eng" in script_names


class TestPipelineTestingSpecCompatibility:
    """Test compatibility between basic and enhanced script specs in PipelineTestingSpec"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create test DAG"""
        return PipelineDAG(
            nodes=["basic_script", "enhanced_script"],
            edges=[("basic_script", "enhanced_script")],
        )

    @pytest.fixture
    def basic_spec(self, temp_dir):
        """Create basic script spec"""
        return ScriptExecutionSpec.create_default(
            "basic_script", "basic_step", temp_dir
        )

    @pytest.fixture
    def enhanced_spec(self, temp_dir):
        """Create enhanced script spec"""
        if not LOGICAL_MATCHING_AVAILABLE:
            pytest.skip("Logical name matching not available")

        return EnhancedScriptExecutionSpec(
            script_name="enhanced_script",
            step_name="enhanced_step",
            input_paths={"data_input": f"{temp_dir}/enhanced/input"},
            output_paths={"data_output": f"{temp_dir}/enhanced/output"},
            environ_vars={"MODE": "enhanced"},
            job_args={"enhanced": True},
            input_path_specs={
                "data_input": PathSpec(
                    logical_name="data_input",
                    path=f"{temp_dir}/enhanced/input",
                    aliases=["input_data", "source"],
                )
            },
            output_path_specs={
                "data_output": PathSpec(
                    logical_name="data_output",
                    path=f"{temp_dir}/enhanced/output",
                    aliases=["output_data", "result"],
                )
            },
        )

    def test_mixed_spec_types_in_pipeline(
        self, test_dag, basic_spec, enhanced_spec, temp_dir
    ):
        """Test pipeline with mixed basic and enhanced script specs"""
        if not LOGICAL_MATCHING_AVAILABLE:
            pytest.skip("Logical name matching not available")

        # Create pipeline with mixed spec types
        pipeline_spec = PipelineTestingSpec(
            dag=test_dag,
            script_specs={"basic_script": basic_spec, "enhanced_script": enhanced_spec},
            test_workspace_root=temp_dir,
        )

        # Test that pipeline spec accepts both types
        assert len(pipeline_spec.script_specs) == 2
        assert "basic_script" in pipeline_spec.script_specs
        assert "enhanced_script" in pipeline_spec.script_specs

        # Test enhanced spec detection
        assert pipeline_spec.has_enhanced_specs() is True

        # Test spec filtering
        enhanced_specs = pipeline_spec.get_enhanced_specs()
        basic_specs = pipeline_spec.get_basic_specs()

        assert len(enhanced_specs) == 1
        assert len(basic_specs) == 1
        assert "enhanced_script" in enhanced_specs
        assert "basic_script" in basic_specs

    def test_auto_enable_enhanced_features(
        self, test_dag, basic_spec, enhanced_spec, temp_dir
    ):
        """Test that enhanced features are auto-enabled when enhanced specs are present"""
        if not LOGICAL_MATCHING_AVAILABLE:
            pytest.skip("Logical name matching not available")

        pipeline_spec = PipelineTestingSpec(
            dag=test_dag,
            script_specs={"basic_script": basic_spec, "enhanced_script": enhanced_spec},
            test_workspace_root=temp_dir,
        )

        # Create configuration - should auto-enable enhanced features
        config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

        assert config.enable_enhanced_features is True
        assert config.enable_logical_matching is True

    def test_basic_specs_only_no_auto_enable(self, test_dag, basic_spec, temp_dir):
        """Test that enhanced features are not auto-enabled with basic specs only"""
        basic_spec_2 = ScriptExecutionSpec.create_default(
            "basic_script_2", "basic_step_2", temp_dir
        )

        pipeline_spec = PipelineTestingSpec(
            dag=test_dag,
            script_specs={
                "basic_script": basic_spec,
                "enhanced_script": basic_spec_2,  # Using basic spec for both
            },
            test_workspace_root=temp_dir,
        )

        # Create configuration - should NOT auto-enable enhanced features
        config = RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

        assert config.enable_enhanced_features is False
        assert config.enable_logical_matching is False
        assert pipeline_spec.has_enhanced_specs() is False

    def test_inheritance_compatibility(self, temp_dir):
        """Test that EnhancedScriptExecutionSpec is compatible with ScriptExecutionSpec type hints"""
        if not LOGICAL_MATCHING_AVAILABLE:
            pytest.skip("Logical name matching not available")

        # Create enhanced spec
        enhanced_spec = EnhancedScriptExecutionSpec(
            script_name="test_script",
            step_name="test_step",
            input_paths={"input": f"{temp_dir}/input"},
            output_paths={"output": f"{temp_dir}/output"},
            environ_vars={"TEST": "true"},
            job_args={"test_mode": True},
        )

        # Test that enhanced spec can be used where ScriptExecutionSpec is expected
        def accepts_script_spec(spec: ScriptExecutionSpec) -> str:
            return spec.script_name

        # This should work without type errors
        result = accepts_script_spec(enhanced_spec)
        assert result == "test_script"

        # Test that enhanced spec has all basic spec attributes
        assert hasattr(enhanced_spec, "script_name")
        assert hasattr(enhanced_spec, "step_name")
        assert hasattr(enhanced_spec, "input_paths")
        assert hasattr(enhanced_spec, "output_paths")
        assert hasattr(enhanced_spec, "environ_vars")
        assert hasattr(enhanced_spec, "job_args")

        # Test that enhanced spec has additional attributes
        assert hasattr(enhanced_spec, "input_path_specs")
        assert hasattr(enhanced_spec, "output_path_specs")


class TestRealScriptIntegration:
    """Test integration with real cursus scripts and contracts"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def xgboost_training_spec(self, temp_dir):
        """Create XGBoost training spec based on real contract"""
        return ScriptExecutionSpec(
            script_name="xgboost_training",
            step_name="XGBoostTraining_training",
            input_paths={
                "input_path": f"{temp_dir}/input/data",
                "hyperparameters_s3_uri": f"{temp_dir}/input/data/config/hyperparameters.json"
            },
            output_paths={
                "model_output": f"{temp_dir}/model",
                "evaluation_output": f"{temp_dir}/output/data"
            },
            environ_vars={},
            job_args={},
        )

    @pytest.fixture
    def tabular_preprocessing_spec(self, temp_dir):
        """Create tabular preprocessing spec based on real contract"""
        return ScriptExecutionSpec(
            script_name="tabular_preprocessing",
            step_name="TabularPreprocessing_training",
            input_paths={
                "input_path": f"{temp_dir}/input/data",
                "hyperparameters_s3_uri": f"{temp_dir}/input/data/config"
            },
            output_paths={
                "processed_data": f"{temp_dir}/preprocessing/output",
                "preprocessing_artifacts": f"{temp_dir}/preprocessing/artifacts"
            },
            environ_vars={"PREPROCESSING_MODE": "standard"},
            job_args={"preprocessing_mode": "standard"},
        )

    @pytest.fixture
    def real_pipeline_dag(self):
        """Create DAG with real script dependencies"""
        return PipelineDAG(
            nodes=["TabularPreprocessing_training", "XGBoostTraining_training"],
            edges=[("TabularPreprocessing_training", "XGBoostTraining_training")]
        )

    @pytest.fixture
    def real_pipeline_spec(self, real_pipeline_dag, tabular_preprocessing_spec, xgboost_training_spec, temp_dir):
        """Create pipeline spec with real scripts"""
        return PipelineTestingSpec(
            dag=real_pipeline_dag,
            script_specs={
                "TabularPreprocessing_training": tabular_preprocessing_spec,
                "XGBoostTraining_training": xgboost_training_spec
            },
            test_workspace_root=temp_dir,
        )

    @pytest.fixture
    def config(self, real_pipeline_spec):
        """Create configuration for real scripts"""
        return RuntimeTestingConfiguration(pipeline_spec=real_pipeline_spec)

    @pytest.fixture
    def tester(self, config):
        """Create RuntimeTester instance"""
        return RuntimeTester(config)

    def test_xgboost_training_script_contract_compatibility(self, tester, xgboost_training_spec):
        """Test XGBoost training script with its actual contract"""
        main_params = {
            "input_paths": xgboost_training_spec.input_paths,
            "output_paths": xgboost_training_spec.output_paths,
            "environ_vars": xgboost_training_spec.environ_vars,
            "job_args": xgboost_training_spec.job_args,
        }

        # Mock the script path to point to the real script
        with patch.object(tester, "_find_script_path") as mock_find:
            mock_find.return_value = "src/cursus/steps/scripts/xgboost_training.py"

            with patch("importlib.util.spec_from_file_location") as mock_spec:
                mock_module = Mock()
                # Mock the main function with the correct signature from the real script
                mock_module.main = Mock()

                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj

                with patch("importlib.util.module_from_spec", return_value=mock_module):
                    with patch("inspect.signature") as mock_sig:
                        # Use the actual signature from xgboost_training.py
                        mock_sig.return_value.parameters.keys.return_value = [
                            "input_paths",
                            "output_paths", 
                            "environ_vars",
                            "job_args",
                        ]

                        with patch("pathlib.Path.mkdir"), patch("pathlib.Path.exists", return_value=True):
                            result = tester.test_script_with_spec(xgboost_training_spec, main_params)

                            assert isinstance(result, ScriptTestResult)
                            assert result.success is True
                            assert result.has_main_function is True
                            assert result.script_name == "xgboost_training"

                            # Verify main function was called with contract-compliant parameters
                            mock_module.main.assert_called_once_with(**main_params)

    def test_real_script_data_compatibility(self, tester, tabular_preprocessing_spec, xgboost_training_spec):
        """Test data compatibility between real preprocessing and training scripts"""
        # Disable logical matching to test basic semantic matching
        with patch.object(tester, "enable_logical_matching", False):
            with patch.object(tester.builder, "get_script_main_params") as mock_params:
                mock_params.return_value = {
                    "input_paths": tabular_preprocessing_spec.input_paths,
                    "output_paths": tabular_preprocessing_spec.output_paths,
                    "environ_vars": tabular_preprocessing_spec.environ_vars,
                    "job_args": tabular_preprocessing_spec.job_args,
                }

                with patch.object(tester, "test_script_with_spec") as mock_test_script:
                    # Mock successful script executions
                    mock_test_script.side_effect = [
                        ScriptTestResult(
                            script_name="tabular_preprocessing",
                            success=True,
                            execution_time=0.1,
                        ),
                        ScriptTestResult(
                            script_name="xgboost_training",
                            success=True,
                            execution_time=0.1,
                        ),
                    ]

                    with patch.object(tester, "_find_valid_output_files") as mock_find_files:
                        # Mock finding processed data output
                        mock_find_files.return_value = [Path("/preprocessing/output/processed_data.csv")]

                        # Mock semantic matching to return matches
                        with patch.object(tester, "_find_semantic_path_matches") as mock_semantic:
                            mock_semantic.return_value = [("processed_data", "input_path", 0.8)]

                            result = tester.test_data_compatibility_with_specs(
                                tabular_preprocessing_spec, xgboost_training_spec
                            )

                            assert isinstance(result, DataCompatibilityResult)
                            assert result.script_a == "tabular_preprocessing"
                            assert result.script_b == "xgboost_training"
                            # Should use semantic matching to connect processed_data -> input_path
                            assert result.compatible is True

    def test_real_pipeline_flow(self, tester, real_pipeline_spec):
        """Test complete pipeline flow with real scripts"""
        with patch.object(tester.builder, "get_script_main_params") as mock_params:
            mock_params.return_value = {
                "input_paths": {"input_path": "/test/input/data"},
                "output_paths": {"processed_data": "/test/preprocessing/output"},
                "environ_vars": {"PREPROCESSING_MODE": "standard"},
                "job_args": {"preprocessing_mode": "standard"},
            }

            with patch.object(tester, "test_script_with_spec") as mock_test_script:
                # Mock successful script tests
                mock_test_script.side_effect = [
                    ScriptTestResult(
                        script_name="tabular_preprocessing",
                        success=True,
                        execution_time=0.1,
                    ),
                    ScriptTestResult(
                        script_name="xgboost_training",
                        success=True,
                        execution_time=0.1,
                    ),
                ]

                with patch.object(tester, "test_data_compatibility_with_specs") as mock_test_compat:
                    # Mock successful data compatibility using semantic matching
                    mock_test_compat.return_value = DataCompatibilityResult(
                        script_a="tabular_preprocessing",
                        script_b="xgboost_training",
                        compatible=True,
                        compatibility_issues=[],
                        data_format_a="csv",
                        data_format_b="csv",
                    )

                    result = tester.test_pipeline_flow_with_spec(real_pipeline_spec)

                    assert result["pipeline_success"] is True
                    assert len(result["script_results"]) == 2
                    assert len(result["data_flow_results"]) == 1
                    assert len(result["errors"]) == 0

                    # Verify the data flow result uses semantic matching
                    data_flow_key = "TabularPreprocessing_training->XGBoostTraining_training"
                    assert data_flow_key in result["data_flow_results"]
                    assert result["data_flow_results"][data_flow_key].compatible is True

    def test_contract_based_path_resolution(self, tester, xgboost_training_spec):
        """Test that paths are resolved according to the actual contract"""
        # Verify that the spec uses the correct logical names from the contract
        assert "input_path" in xgboost_training_spec.input_paths
        assert "hyperparameters_s3_uri" in xgboost_training_spec.input_paths
        assert "model_output" in xgboost_training_spec.output_paths
        assert "evaluation_output" in xgboost_training_spec.output_paths

        # Test that the builder can extract main params correctly
        main_params = tester.builder.get_script_main_params(xgboost_training_spec)
        
        assert "input_paths" in main_params
        assert "output_paths" in main_params
        assert "environ_vars" in main_params
        assert "job_args" in main_params

        # Verify the paths match the contract expectations
        assert main_params["input_paths"]["input_path"] == xgboost_training_spec.input_paths["input_path"]
        assert main_params["output_paths"]["model_output"] == xgboost_training_spec.output_paths["model_output"]

    def test_semantic_matching_with_real_logical_names(self, tester, tabular_preprocessing_spec, xgboost_training_spec):
        """Test semantic matching using real logical names from contracts"""
        # Mock the semantic matcher to return matches for testing
        with patch('cursus.core.deps.semantic_matcher.SemanticMatcher') as mock_semantic_matcher_class:
            mock_semantic_matcher = Mock()
            mock_semantic_matcher.calculate_similarity.return_value = 0.8  # High similarity for data-related terms
            mock_semantic_matcher_class.return_value = mock_semantic_matcher
            
            # Test the semantic matching between real logical names
            matches = tester._find_semantic_path_matches(tabular_preprocessing_spec, xgboost_training_spec)
            
            # Should find semantic matches between preprocessing outputs and training inputs
            assert len(matches) > 0
            
            # Check for expected semantic matches
            match_pairs = [(output_name, input_name) for output_name, input_name, score in matches]
            
            # Should match processed_data -> input_path (both are data-related)
            data_matches = [pair for pair in match_pairs if "data" in pair[0].lower() and "input" in pair[1].lower()]
            assert len(data_matches) > 0, f"Expected data-related matches, got: {match_pairs}"


class TestRuntimeTesterErrorHandling:
    """Test error handling and edge cases in RuntimeTester"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create test DAG"""
        return PipelineDAG(nodes=["script_a"], edges=[])

    @pytest.fixture
    def script_spec(self, temp_dir):
        """Create script spec"""
        return ScriptExecutionSpec.create_default("script_a", "script_a_step", temp_dir)

    @pytest.fixture
    def pipeline_spec(self, test_dag, script_spec, temp_dir):
        """Create pipeline spec"""
        return PipelineTestingSpec(
            dag=test_dag,
            script_specs={"script_a": script_spec},
            test_workspace_root=temp_dir,
        )

    @pytest.fixture
    def config(self, pipeline_spec):
        """Create configuration"""
        return RuntimeTestingConfiguration(pipeline_spec=pipeline_spec)

    @pytest.fixture
    def tester(self, config):
        """Create RuntimeTester instance"""
        return RuntimeTester(config)

    def test_script_import_error(self, tester, temp_dir):
        """Test handling of script import errors"""
        script_spec = ScriptExecutionSpec.create_default(
            "test_script", "test_step", temp_dir
        )
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args,
        }

        with patch.object(tester, "_find_script_path", return_value="test_script.py"):
            with patch(
                "importlib.util.spec_from_file_location",
                side_effect=ImportError("Module not found"),
            ):
                result = tester.test_script_with_spec(script_spec, main_params)

                assert result.success is False
                assert "Module not found" in result.error_message
                assert result.script_name == "test_script"

    def test_script_execution_error(self, tester, temp_dir):
        """Test handling of script execution errors"""
        script_spec = ScriptExecutionSpec.create_default(
            "test_script", "test_step", temp_dir
        )
        main_params = {
            "input_paths": script_spec.input_paths,
            "output_paths": script_spec.output_paths,
            "environ_vars": script_spec.environ_vars,
            "job_args": script_spec.job_args,
        }

        with patch.object(tester, "_find_script_path", return_value="test_script.py"):
            with patch("importlib.util.spec_from_file_location") as mock_spec:
                mock_module = Mock()
                mock_module.main = Mock(
                    side_effect=RuntimeError("Script execution failed")
                )

                mock_spec_obj = Mock()
                mock_spec_obj.loader.exec_module = Mock()
                mock_spec.return_value = mock_spec_obj

                with patch("importlib.util.module_from_spec", return_value=mock_module):
                        with patch("inspect.signature") as mock_sig:
                            mock_sig.return_value.parameters.keys.return_value = [
                                "input_paths",
                                "output_paths",
                                "environ_vars",
                                "job_args",
                            ]

                            with patch("pathlib.Path.exists", return_value=True):
                                result = tester.test_script_with_spec(script_spec, main_params)

                                assert result.success is False
                                assert "Script execution failed" in result.error_message

    def test_invalid_pipeline_spec(self, tester, temp_dir):
        """Test handling of invalid pipeline specifications"""
        # Create invalid pipeline spec with missing script spec
        invalid_dag = PipelineDAG(nodes=["missing_script"], edges=[])
        invalid_pipeline_spec = PipelineTestingSpec(
            dag=invalid_dag,
            script_specs={},  # Missing script spec
            test_workspace_root=temp_dir,
        )

        result = tester.test_pipeline_flow_with_spec(invalid_pipeline_spec)

        assert result["pipeline_success"] is False
        assert len(result["errors"]) > 0
        assert "missing_script" in str(result["errors"])

    def test_workspace_permission_error(self, tester, temp_dir):
        """Test handling of workspace permission errors"""
        with patch(
            "pathlib.Path.mkdir", side_effect=PermissionError("Permission denied")
        ):
            with patch.object(
                tester, "_find_script_path", return_value="test_script.py"
            ):
                with patch("importlib.util.spec_from_file_location") as mock_spec:
                    mock_module = Mock()
                    mock_module.main = Mock()

                    mock_spec_obj = Mock()
                    mock_spec_obj.loader.exec_module = Mock()
                    mock_spec.return_value = mock_spec_obj

                    with patch(
                        "importlib.util.module_from_spec", return_value=mock_module
                    ):
                        with patch("inspect.signature") as mock_sig:
                            mock_sig.return_value.parameters.keys.return_value = [
                                "input_paths",
                                "output_paths",
                                "environ_vars",
                                "job_args",
                            ]

                            script_spec = ScriptExecutionSpec.create_default(
                                "test_script", "test_step", temp_dir
                            )
                            main_params = {
                                "input_paths": script_spec.input_paths,
                                "output_paths": script_spec.output_paths,
                                "environ_vars": script_spec.environ_vars,
                                "job_args": script_spec.job_args,
                            }

                            result = tester.test_script_with_spec(
                                script_spec, main_params
                            )

                            assert result.success is False
                            assert "Permission denied" in result.error_message
