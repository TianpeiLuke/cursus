"""
Pytest tests for inference testing system

Tests the RuntimeTester inference methods and InferenceHandlerSpec functionality.
Includes tests with real examples from package/payload steps and XGBoost inference handler.
"""

import pytest
import tempfile
import json
import tarfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
import pandas as pd
import numpy as np

from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_inference import (
    InferenceHandlerSpec,
    InferenceTestResult,
    InferencePipelineTestingSpec,
)
from cursus.validation.runtime.runtime_models import (
    ScriptTestResult,
    ScriptExecutionSpec,
    PipelineTestingSpec,
    RuntimeTestingConfiguration,
)
from cursus.api.dag.base_dag import PipelineDAG


class TestInferenceHandlerSpec:
    """Test InferenceHandlerSpec data model"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_inference_handler_spec_creation(self, temp_dir):
        """Test basic InferenceHandlerSpec creation"""
        spec = InferenceHandlerSpec(
            handler_name="test_handler",
            step_name="TestStep",
            packaged_model_path=f"{temp_dir}/model.tar.gz",
            payload_samples_path=f"{temp_dir}/payload_samples/"
        )
        
        assert spec.handler_name == "test_handler"
        assert spec.step_name == "TestStep"
        assert spec.packaged_model_path == f"{temp_dir}/model.tar.gz"
        assert spec.payload_samples_path == f"{temp_dir}/payload_samples/"
        assert spec.supported_content_types == ["application/json", "text/csv"]
        assert spec.supported_accept_types == ["application/json", "text/csv"]

    def test_inference_handler_spec_create_default(self, temp_dir):
        """Test InferenceHandlerSpec.create_default method"""
        spec = InferenceHandlerSpec.create_default(
            handler_name="xgboost_inference",
            step_name="ModelServing_inference",
            packaged_model_path=f"{temp_dir}/model.tar.gz",
            payload_samples_path=f"{temp_dir}/payload_samples/"
        )
        
        assert spec.handler_name == "xgboost_inference"
        assert spec.step_name == "ModelServing_inference"
        assert "inference_inputs" in spec.model_paths["extraction_root"]
        assert "code" in spec.code_paths["inference_code_dir"]
        assert spec.environ_vars["INFERENCE_MODE"] == "testing"

    def test_inference_handler_spec_validation_missing_files(self, temp_dir):
        """Test validation with missing files"""
        spec = InferenceHandlerSpec(
            handler_name="test_handler",
            step_name="TestStep",
            packaged_model_path=f"{temp_dir}/nonexistent.tar.gz",
            payload_samples_path=f"{temp_dir}/nonexistent_samples/"
        )
        
        errors = spec.validate_configuration()
        assert len(errors) >= 2  # Should have errors for missing files
        assert any("does not exist" in error for error in errors)

    def test_inference_handler_spec_validation_invalid_format(self, temp_dir):
        """Test validation with invalid file format"""
        # Create a non-tar.gz file
        invalid_file = Path(temp_dir) / "model.txt"
        invalid_file.write_text("not a tar.gz file")
        
        spec = InferenceHandlerSpec(
            handler_name="test_handler",
            step_name="TestStep",
            packaged_model_path=str(invalid_file),
            payload_samples_path=f"{temp_dir}/payload_samples/"
        )
        
        errors = spec.validate_configuration()
        assert any("must be a .tar.gz file" in error for error in errors)

    def test_inference_handler_spec_convenience_methods(self, temp_dir):
        """Test convenience methods for path access"""
        spec = InferenceHandlerSpec.create_default(
            handler_name="test_handler",
            step_name="TestStep",
            packaged_model_path=f"{temp_dir}/model.tar.gz",
            payload_samples_path=f"{temp_dir}/payload_samples/"
        )
        
        assert spec.get_packaged_model_path() == f"{temp_dir}/model.tar.gz"
        assert spec.get_payload_samples_path() == f"{temp_dir}/payload_samples/"
        assert spec.get_extraction_root_path() is not None
        assert spec.get_inference_code_path() is not None


class TestInferenceTestResult:
    """Test InferenceTestResult data model"""

    def test_inference_test_result_creation(self):
        """Test basic InferenceTestResult creation"""
        result = InferenceTestResult(
            handler_name="test_handler",
            overall_success=True,
            total_execution_time=1.5
        )
        
        assert result.handler_name == "test_handler"
        assert result.overall_success is True
        assert result.total_execution_time == 1.5
        assert result.errors == []
        assert result.warnings == []

    def test_inference_test_result_success_rate_calculation(self):
        """Test success rate calculation"""
        result = InferenceTestResult(
            handler_name="test_handler",
            overall_success=True,
            total_execution_time=1.5,
            model_fn_result={"success": True},
            input_fn_results=[{"success": True}, {"success": False}],
            predict_fn_results=[{"success": True}],
            output_fn_results=[{"success": True}, {"success": True}],
            end_to_end_results=[{"success": True}]
        )
        
        # Total: 1 + 2 + 1 + 2 + 1 = 7 tests
        # Successful: 1 + 1 + 1 + 2 + 1 = 6 tests
        # Success rate: 6/7 ≈ 0.857
        success_rate = result.get_overall_success_rate()
        assert abs(success_rate - 6/7) < 0.001

    def test_inference_test_result_empty_results(self):
        """Test success rate calculation with empty results"""
        result = InferenceTestResult(
            handler_name="test_handler",
            overall_success=False,
            total_execution_time=0.0
        )
        
        success_rate = result.get_overall_success_rate()
        assert success_rate == 0.0


class TestInferencePipelineTestingSpec:
    """Test InferencePipelineTestingSpec data model"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def test_dag(self):
        """Create test DAG"""
        return PipelineDAG(
            nodes=["script_step", "inference_step"],
            edges=[("script_step", "inference_step")]
        )

    @pytest.fixture
    def script_spec(self, temp_dir):
        """Create script spec"""
        return ScriptExecutionSpec.create_default("test_script", "script_step", temp_dir)

    @pytest.fixture
    def handler_spec(self, temp_dir):
        """Create inference handler spec"""
        return InferenceHandlerSpec.create_default(
            handler_name="test_handler",
            step_name="inference_step",
            packaged_model_path=f"{temp_dir}/model.tar.gz",
            payload_samples_path=f"{temp_dir}/payload_samples/"
        )

    def test_inference_pipeline_spec_creation(self, test_dag, script_spec, handler_spec, temp_dir):
        """Test InferencePipelineTestingSpec creation"""
        pipeline_spec = InferencePipelineTestingSpec(
            dag=test_dag,
            script_specs={"script_step": script_spec},
            test_workspace_root=temp_dir
        )
        
        pipeline_spec.add_inference_handler("inference_step", handler_spec)
        
        assert pipeline_spec.has_inference_handlers() is True
        assert len(pipeline_spec.get_inference_handler_names()) == 1
        assert "inference_step" in pipeline_spec.get_inference_handler_names()

    def test_mixed_step_types(self, test_dag, script_spec, handler_spec, temp_dir):
        """Test mixed step types functionality"""
        pipeline_spec = InferencePipelineTestingSpec(
            dag=test_dag,
            script_specs={"script_step": script_spec},
            test_workspace_root=temp_dir
        )
        pipeline_spec.add_inference_handler("inference_step", handler_spec)
        
        step_types = pipeline_spec.get_mixed_step_types()
        assert step_types["script_step"] == "script"
        assert step_types["inference_step"] == "inference"

    def test_mixed_pipeline_validation(self, test_dag, script_spec, handler_spec, temp_dir):
        """Test mixed pipeline validation"""
        pipeline_spec = InferencePipelineTestingSpec(
            dag=test_dag,
            script_specs={"script_step": script_spec},
            test_workspace_root=temp_dir
        )
        pipeline_spec.add_inference_handler("inference_step", handler_spec)
        
        errors = pipeline_spec.validate_mixed_pipeline()
        assert len(errors) == 0  # Should be valid
        assert pipeline_spec.is_valid_mixed_pipeline() is True

    def test_mixed_pipeline_validation_conflicts(self, test_dag, script_spec, handler_spec, temp_dir):
        """Test mixed pipeline validation with conflicts"""
        pipeline_spec = InferencePipelineTestingSpec(
            dag=test_dag,
            script_specs={"conflict_step": script_spec},
            test_workspace_root=temp_dir
        )
        # Add handler with same name as script
        handler_spec.step_name = "conflict_step"
        pipeline_spec.add_inference_handler("conflict_step", handler_spec)
        
        errors = pipeline_spec.validate_mixed_pipeline()
        assert len(errors) > 0
        assert any("conflicts" in error for error in errors)


class TestRuntimeTesterInferenceMethods:
    """Test RuntimeTester inference testing methods"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def tester(self, temp_dir):
        """Create RuntimeTester instance"""
        return RuntimeTester(temp_dir)

    @pytest.fixture
    def mock_handler_module(self):
        """Create mock inference handler module"""
        module = Mock()
        module.model_fn = Mock(return_value={"model": "mock_model", "config": {"feature_columns": ["f1", "f2"]}})
        module.input_fn = Mock(return_value=pd.DataFrame([[1, 2], [3, 4]], columns=["f1", "f2"]))
        module.predict_fn = Mock(return_value=np.array([[0.3, 0.7], [0.8, 0.2]]))
        module.output_fn = Mock(return_value='{"predictions": [{"class": 1}, {"class": 0}]}')
        return module

    @pytest.fixture
    def handler_spec(self, temp_dir):
        """Create inference handler spec"""
        return InferenceHandlerSpec.create_default(
            handler_name="test_handler",
            step_name="TestStep",
            packaged_model_path=f"{temp_dir}/model.tar.gz",
            payload_samples_path=f"{temp_dir}/payload_samples/"
        )

    def test_test_inference_function_model_fn(self, tester, mock_handler_module):
        """Test individual model_fn testing"""
        test_params = {"model_dir": "/test/model"}
        
        result = tester.test_inference_function(
            mock_handler_module, "model_fn", test_params
        )
        
        assert result["success"] is True
        assert result["function_name"] == "model_fn"
        assert "result" in result
        assert "validation" in result
        mock_handler_module.model_fn.assert_called_once_with(**test_params)

    def test_test_inference_function_input_fn(self, tester, mock_handler_module):
        """Test individual input_fn testing"""
        test_params = {"request_body": "1,2\n3,4", "request_content_type": "text/csv"}
        
        result = tester.test_inference_function(
            mock_handler_module, "input_fn", test_params
        )
        
        assert result["success"] is True
        assert result["function_name"] == "input_fn"
        mock_handler_module.input_fn.assert_called_once_with(**test_params)

    def test_test_inference_function_error(self, tester, mock_handler_module):
        """Test individual function testing with error"""
        mock_handler_module.predict_fn.side_effect = ValueError("Prediction failed")
        test_params = {"input_data": [[1, 2]], "model": "mock_model"}
        
        result = tester.test_inference_function(
            mock_handler_module, "predict_fn", test_params
        )
        
        assert result["success"] is False
        assert result["function_name"] == "predict_fn"
        assert "Prediction failed" in result["error"]

    def test_validate_function_result(self, tester):
        """Test function result validation"""
        # Test model_fn validation
        model_result = {"model": "test", "config": {}}
        validation = tester._validate_function_result("model_fn", model_result, {})
        assert validation["function_type"] == "model_fn"
        assert validation["has_model_artifacts"] is True
        assert validation["is_dict"] is True

        # Test input_fn validation
        input_result = pd.DataFrame([[1, 2]])
        validation = tester._validate_function_result("input_fn", input_result, {})
        assert validation["function_type"] == "input_fn"
        assert validation["has_processed_input"] is True
        assert "DataFrame" in validation["input_type"]

    def test_extract_packaged_model(self, tester, temp_dir):
        """Test packaged model extraction"""
        # Create a mock tar.gz file
        model_dir = Path(temp_dir) / "model_content"
        model_dir.mkdir()
        
        # Create model files
        (model_dir / "model.pkl").write_bytes(b"fake model data")
        code_dir = model_dir / "code"
        code_dir.mkdir()
        (code_dir / "inference.py").write_text("# inference handler")
        
        # Create tar.gz
        tar_path = Path(temp_dir) / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_dir, arcname=".")
        
        # Test extraction with proper cleanup
        try:
            extraction_paths = tester._extract_packaged_model(str(tar_path), "test_extraction")
            
            assert "extraction_root" in extraction_paths
            assert "inference_code" in extraction_paths
            assert "handler_file" in extraction_paths
            
            # Verify extracted files exist
            extraction_root = Path(extraction_paths["extraction_root"])
            assert (extraction_root / "model.pkl").exists()
            assert (extraction_root / "code" / "inference.py").exists()
        finally:
            # Ensure cleanup
            tester._cleanup_extraction_directory("test_extraction")

    def test_load_handler_module(self, tester, temp_dir):
        """Test inference handler module loading"""
        # Create a mock inference handler file
        handler_file = Path(temp_dir) / "test_inference.py"
        handler_content = '''
def model_fn(model_dir):
    return {"model": "test_model"}

def input_fn(request_body, request_content_type):
    return {"processed": "data"}

def predict_fn(input_data, model):
    return [0.5, 0.5]

def output_fn(predictions, accept):
    return "test_output"
'''
        handler_file.write_text(handler_content)
        
        # Test loading
        module = tester._load_handler_module(str(handler_file))
        
        assert hasattr(module, "model_fn")
        assert hasattr(module, "input_fn")
        assert hasattr(module, "predict_fn")
        assert hasattr(module, "output_fn")
        
        # Test function calls
        assert module.model_fn("/test") == {"model": "test_model"}
        assert module.input_fn("data", "json") == {"processed": "data"}

    def test_load_payload_samples(self, tester, temp_dir):
        """Test payload samples loading"""
        # Create payload samples directory structure
        payload_dir = Path(temp_dir) / "payload_samples"
        csv_dir = payload_dir / "csv_samples"
        json_dir = payload_dir / "json_samples"
        csv_dir.mkdir(parents=True)
        json_dir.mkdir(parents=True)
        
        # Create sample files
        (csv_dir / "sample1.csv").write_text("1.0,2.0,3.0")
        (csv_dir / "sample2.csv").write_text("4.0,5.0,6.0")
        (json_dir / "sample1.json").write_text('{"f1": 1.0, "f2": 2.0}')
        (json_dir / "sample2.json").write_text('{"f1": 3.0, "f2": 4.0}')
        
        # Test loading
        samples = tester._load_payload_samples(str(payload_dir))
        
        assert len(samples) == 4
        
        # Check CSV samples
        csv_samples = [s for s in samples if s["content_type"] == "text/csv"]
        assert len(csv_samples) == 2
        assert any("1.0,2.0,3.0" in s["data"] for s in csv_samples)
        
        # Check JSON samples
        json_samples = [s for s in samples if s["content_type"] == "application/json"]
        assert len(json_samples) == 2
        assert any('"f1": 1.0' in s["data"] for s in json_samples)

    def test_cleanup_extraction_directory(self, tester, temp_dir):
        """Test extraction directory cleanup"""
        # Create test directory
        test_dir = Path(temp_dir) / "test_extraction"
        test_dir.mkdir()
        (test_dir / "test_file.txt").write_text("test content")
        
        assert test_dir.exists()
        
        # Test cleanup
        tester._cleanup_extraction_directory(str(test_dir))
        
        assert not test_dir.exists()


class TestInferenceTestingErrorHandling:
    """Test error handling in inference testing"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def tester(self, temp_dir):
        """Create RuntimeTester instance"""
        return RuntimeTester(temp_dir)

    def test_comprehensive_error_scenarios(self, tester, temp_dir):
        """Test comprehensive error handling scenarios"""
        handler_spec = InferenceHandlerSpec(
            handler_name="error_test_handler",
            step_name="ErrorTestStep",
            packaged_model_path="/invalid/path/model.tar.gz",
            payload_samples_path="/invalid/path/samples/"
        )
        
        result = tester.test_inference_pipeline(handler_spec)
        
        assert result["pipeline_success"] is False
        assert len(result["errors"]) > 0
        assert any("does not exist" in error or "No such file" in error for error in result["errors"])

    def test_error_message_quality(self, tester, temp_dir):
        """Test that error messages are informative and actionable"""
        handler_spec = InferenceHandlerSpec(
            handler_name="quality_test_handler",
            step_name="QualityTestStep",
            packaged_model_path="/completely/invalid/path.tar.gz",
            payload_samples_path="/also/invalid/path/"
        )
        
        result = tester.test_inference_pipeline(handler_spec)
        
        assert result["pipeline_success"] is False
        assert len(result["errors"]) > 0
        
        # Verify error message quality
        error_message = result["errors"][0]
        assert len(error_message) > 20  # Should be descriptive
        assert "quality_test_handler" in str(result) or "QualityTestStep" in str(result)

    def test_resource_cleanup_comprehensive(self, tester, temp_dir):
        """Test comprehensive resource cleanup scenarios"""
        handler_spec = InferenceHandlerSpec(
            handler_name="cleanup_test_handler",
            step_name="CleanupTestStep",
            packaged_model_path=f"{temp_dir}/model.tar.gz",
            payload_samples_path=f"{temp_dir}/payload_samples/"
        )
        
        # Test cleanup with various error types
        error_scenarios = [
            FileNotFoundError("File not found"),
            PermissionError("Permission denied"),
            MemoryError("Out of memory"),
            ValueError("Invalid value"),
            RuntimeError("Runtime error")
        ]
        
        for error in error_scenarios:
            with patch.object(tester, '_extract_packaged_model') as mock_extract:
                mock_extract.side_effect = error
                
                with patch.object(tester, '_cleanup_extraction_directory') as mock_cleanup:
                    result = tester.test_inference_pipeline(handler_spec)
                    
                    # Cleanup should always be called
                    mock_cleanup.assert_called()
                    assert result["pipeline_success"] is False
                    assert len(result["errors"]) > 0
