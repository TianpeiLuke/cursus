"""
Integration tests for inference testing system with real examples

Tests integration with real package/payload steps and XGBoost inference handler.
Validates compatibility with actual workflow: Package → Payload → Inference.
"""

import pytest
import tempfile
import json
import tarfile
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from cursus.validation.runtime.runtime_testing import RuntimeTester
from cursus.validation.runtime.runtime_inference import InferenceHandlerSpec


class TestRealWorldIntegration:
    """Test integration with real package/payload steps and XGBoost inference handler"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def tester(self, temp_dir):
        """Create RuntimeTester instance"""
        return RuntimeTester(temp_dir)

    def test_package_step_output_structure(self, temp_dir):
        """Test that package step output structure matches expectations"""
        # Create package step output structure
        package_output = Path(temp_dir) / "package_output"
        package_output.mkdir()
        
        # Create model.tar.gz with expected structure
        model_content = package_output / "model_content"
        model_content.mkdir()
        
        # Model files at root level
        (model_content / "xgboost_model.bst").write_bytes(b"fake xgboost model")
        (model_content / "risk_table_map.pkl").write_bytes(b"fake risk table")
        (model_content / "impute_dict.pkl").write_bytes(b"fake impute dict")
        (model_content / "feature_columns.txt").write_text("0,feature1\n1,feature2\n")
        (model_content / "hyperparameters.json").write_text('{"model_type": "xgboost"}')
        
        # Inference code in code/ subdirectory
        code_dir = model_content / "code"
        code_dir.mkdir()
        (code_dir / "inference.py").write_text("# XGBoost inference handler")
        
        # Optional calibration directory
        calibration_dir = model_content / "calibration"
        calibration_dir.mkdir()
        (calibration_dir / "calibration_model.pkl").write_bytes(b"fake calibration")
        
        # Create tar.gz
        tar_path = package_output / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_content, arcname=".")
        
        # Verify structure matches expectations
        assert tar_path.exists()
        
        # Test extraction with proper cleanup
        tester = RuntimeTester(temp_dir)
        try:
            extraction_paths = tester._extract_packaged_model(str(tar_path), "test_extraction")
            
            extraction_root = Path(extraction_paths["extraction_root"])
            assert (extraction_root / "xgboost_model.bst").exists()
            assert (extraction_root / "code" / "inference.py").exists()
            assert (extraction_root / "calibration" / "calibration_model.pkl").exists()
        finally:
            # Ensure cleanup
            tester._cleanup_extraction_directory("test_extraction")

    def test_payload_step_output_structure(self, temp_dir):
        """Test that payload step output structure matches expectations"""
        # Create payload step output structure
        payload_output = Path(temp_dir) / "payload_output"
        payload_output.mkdir()
        
        # Create payload.tar.gz with expected structure
        payload_content = payload_output / "payload_content"
        csv_dir = payload_content / "csv_samples"
        json_dir = payload_content / "json_samples"
        csv_dir.mkdir(parents=True)
        json_dir.mkdir(parents=True)
        
        # Create sample files as payload step would
        (csv_dir / "payload_text_csv_0.csv").write_text("1.0,2.0,DEFAULT_TEXT")
        (csv_dir / "payload_text_csv_1.csv").write_text("3.0,4.0,DEFAULT_TEXT")
        (json_dir / "payload_application_json_0.json").write_text('{"f1": 1.0, "f2": 2.0, "f3": "DEFAULT_TEXT"}')
        (json_dir / "payload_application_json_1.json").write_text('{"f1": 3.0, "f2": 4.0, "f3": "DEFAULT_TEXT"}')
        
        # Create tar.gz
        tar_path = payload_output / "payload.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for file_path in payload_content.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(payload_content)
                    tar.add(file_path, arcname=arcname)
        
        # Verify structure matches expectations
        assert tar_path.exists()
        
        # Test loading
        tester = RuntimeTester(temp_dir)
        samples = tester._load_payload_samples(str(payload_content))
        
        assert len(samples) == 4
        csv_samples = [s for s in samples if s["content_type"] == "text/csv"]
        json_samples = [s for s in samples if s["content_type"] == "application/json"]
        assert len(csv_samples) == 2
        assert len(json_samples) == 2

    def test_xgboost_inference_handler_structure(self, temp_dir):
        """Test XGBoost inference handler function signatures"""
        # Create mock XGBoost inference handler
        handler_file = Path(temp_dir) / "xgboost_inference.py"
        handler_content = '''
def model_fn(model_dir):
    """Load XGBoost model and preprocessing artifacts"""
    return {
        "model": "mock_xgboost_model",
        "risk_processors": {},
        "numerical_processor": {},
        "config": {"feature_columns": ["f1", "f2"]},
        "calibrator": None
    }

def input_fn(request_body, request_content_type, context=None):
    """Process input data"""
    import pandas as pd
    if request_content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(request_body), header=None)
    elif request_content_type == "application/json":
        import json
        data = json.loads(request_body)
        return pd.DataFrame([data])
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Generate predictions"""
    import numpy as np
    # Mock prediction
    return {
        "raw_predictions": np.array([[0.3, 0.7], [0.8, 0.2]]),
        "calibrated_predictions": np.array([[0.25, 0.75], [0.85, 0.15]])
    }

def output_fn(prediction_output, accept="application/json"):
    """Format output"""
    import json
    if accept == "application/json":
        return json.dumps({"predictions": [{"class": 1}, {"class": 0}]}), "application/json"
    elif accept == "text/csv":
        return "0.7,0.75,class-1\\n0.2,0.15,class-0\\n", "text/csv"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
        handler_file.write_text(handler_content)
        
        # Test loading and function signatures
        tester = RuntimeTester(temp_dir)
        module = tester._load_handler_module(str(handler_file))
        
        # Verify all required functions exist
        assert hasattr(module, "model_fn")
        assert hasattr(module, "input_fn")
        assert hasattr(module, "predict_fn")
        assert hasattr(module, "output_fn")
        
        # Test function calls with expected signatures
        model_result = module.model_fn("/test/model")
        assert "model" in model_result
        assert "config" in model_result
        
        input_result = module.input_fn("1,2", "text/csv")
        assert hasattr(input_result, "shape")  # Should be DataFrame
        
        predict_result = module.predict_fn(input_result, model_result)
        assert "raw_predictions" in predict_result
        assert "calibrated_predictions" in predict_result

    def test_registration_step_input_format_compatibility(self, temp_dir):
        """Test compatibility with registration step input format"""
        # Create registration step input structure (from MIMS registration contract)
        registration_input = Path(temp_dir) / "registration_input"
        packaged_model_dir = registration_input / "PackagedModel"
        payload_samples_dir = registration_input / "GeneratedPayloadSamples"
        packaged_model_dir.mkdir(parents=True)
        payload_samples_dir.mkdir(parents=True)
        
        # Create packaged model (from package step)
        (packaged_model_dir / "model.tar.gz").write_bytes(b"fake packaged model")
        
        # Create payload samples (from payload step)
        (payload_samples_dir / "payload.tar.gz").write_bytes(b"fake payload samples")
        
        # Test that InferenceHandlerSpec can use these paths
        handler_spec = InferenceHandlerSpec(
            handler_name="registration_inference",
            step_name="RegistrationStep",
            packaged_model_path=str(packaged_model_dir / "model.tar.gz"),
            payload_samples_path=str(payload_samples_dir)
        )
        
        assert handler_spec.handler_name == "registration_inference"
        assert handler_spec.packaged_model_path == str(packaged_model_dir / "model.tar.gz")
        assert handler_spec.payload_samples_path == str(payload_samples_dir)

    def test_end_to_end_package_payload_inference_workflow(self, temp_dir):
        """Test complete workflow: Package → Payload → Inference"""
        tester = RuntimeTester(temp_dir)
        
        # Step 1: Create package step output
        package_output = self._create_package_step_output(temp_dir)
        
        # Step 2: Create payload step output  
        payload_output = self._create_payload_step_output(temp_dir)
        
        # Step 3: Create inference handler spec
        handler_spec = InferenceHandlerSpec.create_default(
            handler_name="xgboost_inference",
            step_name="ModelServing_inference",
            packaged_model_path=str(package_output / "model.tar.gz"),
            payload_samples_path=str(payload_output)
        )
        
        # Step 4: Test inference pipeline with mocked components
        with patch.object(tester, '_extract_packaged_model') as mock_extract:
            mock_extract.return_value = {
                "extraction_root": f"{temp_dir}/extracted",
                "handler_file": f"{temp_dir}/extracted/code/inference.py"
            }
            
            with patch.object(tester, '_load_handler_module') as mock_load_handler:
                mock_handler = self._create_mock_xgboost_handler()
                mock_load_handler.return_value = mock_handler
                
                with patch.object(tester, '_load_payload_samples') as mock_load_samples:
                    mock_load_samples.return_value = [
                        {"sample_name": "csv_sample", "content_type": "text/csv", "data": "1.0,2.0,DEFAULT_TEXT"},
                        {"sample_name": "json_sample", "content_type": "application/json", "data": '{"f1": 1.0, "f2": 2.0}'}
                    ]
                    
                    with patch.object(tester, '_cleanup_extraction_directory'):
                        result = tester.test_inference_pipeline(handler_spec)
                        
                        # Verify successful end-to-end testing
                        assert result["pipeline_success"] is True
                        assert "model_fn" in result["function_results"]
                        assert "input_fn" in result["function_results"]
                        assert "predict_fn" in result["function_results"]
                        assert "output_fn" in result["function_results"]
                        assert "end_to_end" in result["function_results"]

    def test_package_step_contract_compatibility(self, temp_dir):
        """Test compatibility with package step contract from src/cursus/steps/scripts/package.py"""
        # Test that our inference testing can handle the exact output format from package.py
        package_output = self._create_package_step_output(temp_dir)
        
        # Verify the structure matches what package.py produces
        tar_path = package_output / "model.tar.gz"
        assert tar_path.exists()
        
        # Test extraction with RuntimeTester
        tester = RuntimeTester(temp_dir)
        try:
            extraction_paths = tester._extract_packaged_model(str(tar_path), "package_test")
            
            # Verify all expected files are present
            extraction_root = Path(extraction_paths["extraction_root"])
            expected_files = [
                "xgboost_model.bst",
                "risk_table_map.pkl", 
                "impute_dict.pkl",
                "feature_columns.txt",
                "hyperparameters.json",
                "code/inference.py"
            ]
            
            for expected_file in expected_files:
                assert (extraction_root / expected_file).exists(), f"Missing expected file: {expected_file}"
        finally:
            # Ensure cleanup
            tester._cleanup_extraction_directory("package_test")

    def test_payload_step_contract_compatibility(self, temp_dir):
        """Test compatibility with payload step contract from src/cursus/steps/scripts/payload.py"""
        # Test that our inference testing can handle the exact output format from payload.py
        payload_output = self._create_payload_step_output(temp_dir)
        
        # Test loading with RuntimeTester
        tester = RuntimeTester(temp_dir)
        samples = tester._load_payload_samples(str(payload_output))
        
        # Verify samples match payload.py output format
        assert len(samples) > 0
        
        # Check that we have both CSV and JSON samples
        content_types = {sample["content_type"] for sample in samples}
        assert "text/csv" in content_types
        assert "application/json" in content_types
        
        # Verify sample data format
        csv_samples = [s for s in samples if s["content_type"] == "text/csv"]
        json_samples = [s for s in samples if s["content_type"] == "application/json"]
        
        assert len(csv_samples) > 0
        assert len(json_samples) > 0
        
        # Check CSV format
        csv_sample = csv_samples[0]
        assert "DEFAULT_TEXT" in csv_sample["data"]
        
        # Check JSON format
        json_sample = json_samples[0]
        json_data = json.loads(json_sample["data"])
        assert "f1" in json_data
        assert "f2" in json_data

    def test_xgboost_handler_real_signature_compatibility(self, temp_dir):
        """Test compatibility with real XGBoost handler from dockers/xgboost_atoz/"""
        # Create handler that matches the real XGBoost inference handler signature
        handler_file = Path(temp_dir) / "real_xgboost_inference.py"
        
        # Use the actual function signatures from the real handler
        handler_content = '''
def model_fn(model_dir):
    """Load XGBoost model and preprocessing artifacts - real signature"""
    return {
        "model": "mock_xgboost_booster",
        "risk_processors": {"categorical_feature": "mock_processor"},
        "numerical_processor": "mock_numerical_processor",
        "feature_importance": {"feature1": 0.5, "feature2": 0.3},
        "config": {
            "is_multiclass": False,
            "num_classes": 2,
            "feature_columns": ["feature1", "feature2"]
        },
        "version": "1.0.0",
        "calibrator": None
    }

def input_fn(request_body, request_content_type, context=None):
    """Process input data - real signature with optional context"""
    import pandas as pd
    from io import StringIO
    
    if request_content_type == "text/csv":
        return pd.read_csv(StringIO(request_body), header=None, index_col=None)
    elif request_content_type == "application/json":
        import json
        if "\\n" in request_body:
            # Multi-record JSON (NDJSON)
            records = [json.loads(line) for line in request_body.strip().splitlines() if line.strip()]
            return pd.DataFrame(records)
        else:
            json_obj = json.loads(request_body)
            if isinstance(json_obj, dict):
                return pd.DataFrame([json_obj])
            elif isinstance(json_obj, list):
                return pd.DataFrame(json_obj)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Generate predictions - real signature"""
    import numpy as np
    # Mock the real prediction format with raw and calibrated predictions
    return {
        "raw_predictions": np.array([[0.3, 0.7], [0.8, 0.2]]),
        "calibrated_predictions": np.array([[0.25, 0.75], [0.85, 0.15]])
    }

def output_fn(prediction_output, accept="application/json"):
    """Format output - real signature"""
    import json
    import numpy as np
    
    # Handle the real prediction output format
    if isinstance(prediction_output, dict):
        raw_predictions = prediction_output.get("raw_predictions")
        calibrated_predictions = prediction_output.get("calibrated_predictions")
    else:
        raw_predictions = prediction_output
        calibrated_predictions = prediction_output
    
    if accept == "application/json":
        # Real JSON format
        predictions = []
        for i, (raw_probs, cal_probs) in enumerate(zip(raw_predictions, calibrated_predictions)):
            predictions.append({
                "legacy-score": str(raw_probs[1]),
                "calibrated-score": str(cal_probs[1]),
                "custom-output-label": f"class-{1 if raw_probs[1] > raw_probs[0] else 0}"
            })
        return json.dumps({"predictions": predictions}), "application/json"
    elif accept == "text/csv":
        # Real CSV format
        lines = []
        for raw_probs, cal_probs in zip(raw_predictions, calibrated_predictions):
            raw_score = round(float(raw_probs[1]), 4)
            cal_score = round(float(cal_probs[1]), 4)
            prediction = "class-1" if raw_probs[1] > raw_probs[0] else "class-0"
            lines.append(f"{raw_score:.4f},{cal_score:.4f},{prediction}")
        return "\\n".join(lines) + "\\n", "text/csv"
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
'''
        handler_file.write_text(handler_content)
        
        # Test loading and function signatures
        tester = RuntimeTester(temp_dir)
        module = tester._load_handler_module(str(handler_file))
        
        # Verify all required functions exist with real signatures
        assert hasattr(module, "model_fn")
        assert hasattr(module, "input_fn")
        assert hasattr(module, "predict_fn")
        assert hasattr(module, "output_fn")
        
        # Test function calls with real signatures
        model_result = module.model_fn("/test/model")
        assert "model" in model_result
        assert "config" in model_result
        assert "version" in model_result
        assert "feature_importance" in model_result
        
        # Test input_fn with context parameter
        input_result = module.input_fn("1,2", "text/csv", context=None)
        assert hasattr(input_result, "shape")  # Should be DataFrame
        
        # Test predict_fn with real format
        predict_result = module.predict_fn(input_result, model_result)
        assert "raw_predictions" in predict_result
        assert "calibrated_predictions" in predict_result
        
        # Test output_fn with real format
        output_result, content_type = module.output_fn(predict_result, "application/json")
        assert content_type == "application/json"
        output_data = json.loads(output_result)
        assert "predictions" in output_data
        assert "legacy-score" in output_data["predictions"][0]
        assert "calibrated-score" in output_data["predictions"][0]
        assert "custom-output-label" in output_data["predictions"][0]

    def _create_package_step_output(self, temp_dir):
        """Helper to create package step output structure"""
        package_output = Path(temp_dir) / "package_output"
        package_output.mkdir()
        
        model_content = package_output / "model_content"
        model_content.mkdir()
        
        # Create XGBoost model files
        (model_content / "xgboost_model.bst").write_bytes(b"fake xgboost model")
        (model_content / "risk_table_map.pkl").write_bytes(b"fake risk table")
        (model_content / "impute_dict.pkl").write_bytes(b"fake impute dict")
        (model_content / "feature_columns.txt").write_text("0,feature1\n1,feature2\n")
        (model_content / "hyperparameters.json").write_text('{"model_type": "xgboost"}')
        
        # Create inference code
        code_dir = model_content / "code"
        code_dir.mkdir()
        (code_dir / "inference.py").write_text("# XGBoost inference handler")
        
        # Create tar.gz
        tar_path = package_output / "model.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(model_content, arcname=".")
        
        return package_output

    def _create_payload_step_output(self, temp_dir):
        """Helper to create payload step output structure"""
        payload_output = Path(temp_dir) / "payload_output"
        csv_dir = payload_output / "csv_samples"
        json_dir = payload_output / "json_samples"
        csv_dir.mkdir(parents=True)
        json_dir.mkdir(parents=True)
        
        # Create payload samples
        (csv_dir / "sample1.csv").write_text("1.0,2.0,DEFAULT_TEXT")
        (json_dir / "sample1.json").write_text('{"f1": 1.0, "f2": 2.0, "f3": "DEFAULT_TEXT"}')
        
        return payload_output

    def _create_mock_xgboost_handler(self):
        """Helper to create mock XGBoost handler"""
        mock_handler = Mock()
        mock_handler.model_fn.return_value = {
            "model": "mock_xgboost_model",
            "config": {"feature_columns": ["f1", "f2", "f3"]},
            "calibrator": None
        }
        mock_handler.input_fn.return_value = pd.DataFrame([[1.0, 2.0, "DEFAULT_TEXT"]])
        mock_handler.predict_fn.return_value = {
            "raw_predictions": np.array([[0.3, 0.7]]),
            "calibrated_predictions": np.array([[0.25, 0.75]])
        }
        mock_handler.output_fn.return_value = ('{"predictions": [{"class": 1}]}', "application/json")
        return mock_handler
