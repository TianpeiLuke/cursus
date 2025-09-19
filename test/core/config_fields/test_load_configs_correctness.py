"""
Test for load_configs correctness using real configuration files.

This test verifies that the load_configs function can properly load and reconstruct
all configuration objects from a saved JSON file, specifically testing with the
config_NA_xgboost_AtoZ.json file.
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Type, Any

from cursus.core.config_fields import load_configs
from cursus.steps.configs.utils import build_complete_config_classes
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestLoadConfigsCorrectness:
    """Test suite for load_configs correctness with real configuration files."""
    
    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        return Path(__file__).parent.parent.parent.parent
    
    @pytest.fixture
    def config_file_path(self, project_root):
        """Path to the test configuration file."""
        return str(project_root / "pipeline_config" / "config_NA_xgboost_AtoZ_v2" / "config_NA_xgboost_AtoZ.json")
    
    @pytest.fixture
    def config_file_data(self, config_file_path):
        """Load the raw JSON data from the config file."""
        with open(config_file_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def expected_config_types(self, config_file_data):
        """Extract expected config types from the JSON metadata."""
        return config_file_data["metadata"]["config_types"]
    
    @pytest.fixture
    def config_classes(self):
        """Build the complete set of config classes."""
        return build_complete_config_classes()
    
    def test_config_file_exists(self, config_file_path):
        """Test that the configuration file exists."""
        assert Path(config_file_path).exists(), f"Config file not found: {config_file_path}"
    
    def test_config_file_structure(self, config_file_data):
        """Test that the configuration file has the expected structure."""
        # Check top-level structure
        assert "metadata" in config_file_data, "Missing 'metadata' section"
        assert "configuration" in config_file_data, "Missing 'configuration' section"
        
        # Check metadata structure
        metadata = config_file_data["metadata"]
        assert "config_types" in metadata, "Missing 'config_types' in metadata"
        assert "created_at" in metadata, "Missing 'created_at' in metadata"
        
        # Check configuration structure
        configuration = config_file_data["configuration"]
        assert "shared" in configuration, "Missing 'shared' section in configuration"
        assert "specific" in configuration, "Missing 'specific' section in configuration"
    
    def test_expected_config_count(self, expected_config_types):
        """Test that we expect to load the correct number of configurations."""
        expected_count = len(expected_config_types)
        assert expected_count == 12, f"Expected 12 configs, but metadata shows {expected_count}"
        
        # Verify specific expected config names
        expected_names = {
            "Base", "CradleDataLoading_calibration", "CradleDataLoading_training",
            "ModelCalibration_calibration", "Package", "Payload", "Processing",
            "Registration", "TabularPreprocessing_calibration", "TabularPreprocessing_training",
            "XGBoostModelEval_calibration", "XGBoostTraining"
        }
        actual_names = set(expected_config_types.keys())
        assert actual_names == expected_names, f"Config names mismatch. Expected: {expected_names}, Got: {actual_names}"
    
    def test_config_class_availability(self, config_classes, expected_config_types):
        """Test that all required config classes are available."""
        expected_class_names = set(expected_config_types.values())
        available_class_names = set(config_classes.keys())
        
        print(f"Expected config classes: {sorted(expected_class_names)}")
        print(f"Available config classes: {sorted(available_class_names)}")
        
        missing_classes = expected_class_names - available_class_names
        if missing_classes:
            pytest.fail(f"Missing config classes: {sorted(missing_classes)}")
    
    def test_load_configs_basic_functionality(self, config_file_path, config_classes):
        """Test basic load_configs functionality."""
        try:
            loaded_configs = load_configs(config_file_path, config_classes)
            assert loaded_configs is not None, "load_configs returned None"
            assert isinstance(loaded_configs, dict), f"Expected dict, got {type(loaded_configs)}"
        except Exception as e:
            pytest.fail(f"load_configs failed with exception: {e}")
    
    def test_load_configs_return_structure(self, config_file_path, config_classes):
        """Test that load_configs returns the expected structure."""
        loaded_configs = load_configs(config_file_path, config_classes)
        
        # The new load_configs returns {"shared": {...}, "specific": {...}}
        # But the utils.py wrapper should convert it to {step_name: config_object}
        
        # Check if we got the expected format (step_name -> config_object)
        if "shared" in loaded_configs and "specific" in loaded_configs:
            pytest.fail(
                "load_configs returned raw data structure instead of config objects. "
                f"Got keys: {list(loaded_configs.keys())}"
            )
        
        # Should be step names mapping to config objects
        for step_name, config_obj in loaded_configs.items():
            assert isinstance(step_name, str), f"Step name should be string, got {type(step_name)}"
            assert hasattr(config_obj, '__dict__'), f"Config object {step_name} should be a class instance"
    
    def test_load_configs_completeness(self, config_file_path, config_classes, expected_config_types):
        """Test that load_configs loads all expected configurations."""
        loaded_configs = load_configs(config_file_path, config_classes)
        
        expected_step_names = set(expected_config_types.keys())
        actual_step_names = set(loaded_configs.keys())
        
        print(f"Expected step names: {sorted(expected_step_names)}")
        print(f"Actually loaded step names: {sorted(actual_step_names)}")
        
        missing_steps = expected_step_names - actual_step_names
        extra_steps = actual_step_names - expected_step_names
        
        if missing_steps:
            pytest.fail(f"Missing configurations: {sorted(missing_steps)}")
        
        if extra_steps:
            print(f"Warning: Extra configurations loaded: {sorted(extra_steps)}")
        
        # Check that we loaded the expected number
        assert len(loaded_configs) >= len(expected_config_types), (
            f"Expected at least {len(expected_config_types)} configs, got {len(loaded_configs)}"
        )
    
    def test_load_configs_object_types(self, config_file_path, config_classes, expected_config_types):
        """Test that loaded configurations have the correct types."""
        loaded_configs = load_configs(config_file_path, config_classes)
        
        for step_name, expected_class_name in expected_config_types.items():
            if step_name not in loaded_configs:
                continue  # Skip if not loaded (will be caught by completeness test)
            
            config_obj = loaded_configs[step_name]
            actual_class_name = config_obj.__class__.__name__
            
            assert actual_class_name == expected_class_name, (
                f"Config {step_name}: expected {expected_class_name}, got {actual_class_name}"
            )
    
    def test_load_configs_object_attributes(self, config_file_path, config_classes):
        """Test that loaded configuration objects have expected attributes."""
        loaded_configs = load_configs(config_file_path, config_classes)
        
        # Test specific configurations for expected attributes
        if "Base" in loaded_configs:
            base_config = loaded_configs["Base"]
            assert hasattr(base_config, "pipeline_name"), "Base config missing pipeline_name"
            assert hasattr(base_config, "author"), "Base config missing author"
            assert hasattr(base_config, "bucket"), "Base config missing bucket"
        
        if "Registration" in loaded_configs:
            reg_config = loaded_configs["Registration"]
            assert hasattr(reg_config, "model_domain"), "Registration config missing model_domain"
            assert hasattr(reg_config, "model_objective"), "Registration config missing model_objective"
            assert hasattr(reg_config, "framework"), "Registration config missing framework"
        
        # Test CradleDataLoading configs if present
        cradle_configs = [name for name in loaded_configs.keys() if "CradleDataLoading" in name]
        for cradle_name in cradle_configs:
            cradle_config = loaded_configs[cradle_name]
            assert hasattr(cradle_config, "data_sources_spec"), f"{cradle_name} missing data_sources_spec"
            assert hasattr(cradle_config, "transform_spec"), f"{cradle_name} missing transform_spec"
            assert hasattr(cradle_config, "output_spec"), f"{cradle_name} missing output_spec"
    
    def test_load_configs_data_integrity(self, config_file_path, config_classes, config_file_data):
        """Test that loaded configurations contain the expected data values."""
        loaded_configs = load_configs(config_file_path, config_classes)
        
        # Test some specific data values from the shared section
        shared_data = config_file_data["configuration"]["shared"]
        
        if "Base" in loaded_configs:
            base_config = loaded_configs["Base"]
            
            # Check shared values are properly loaded
            if "author" in shared_data:
                assert base_config.author == shared_data["author"], "Author mismatch"
            if "pipeline_version" in shared_data:
                assert base_config.pipeline_version == shared_data["pipeline_version"], "Pipeline version mismatch"
            if "service_name" in shared_data:
                assert base_config.service_name == shared_data["service_name"], "Service name mismatch"
    
    def test_load_configs_specific_data(self, config_file_path, config_classes, config_file_data):
        """Test that specific configuration data is properly loaded."""
        loaded_configs = load_configs(config_file_path, config_classes)
        specific_data = config_file_data["configuration"]["specific"]
        
        # Test Registration config specific data
        if "Registration" in loaded_configs and "Registration" in specific_data:
            reg_config = loaded_configs["Registration"]
            reg_specific = specific_data["Registration"]
            
            if "model_domain" in reg_specific:
                assert reg_config.model_domain == reg_specific["model_domain"], "Registration model_domain mismatch"
            if "framework" in reg_specific:
                assert reg_config.framework == reg_specific["framework"], "Registration framework mismatch"
    
    def test_load_configs_error_handling(self, config_classes):
        """Test error handling for invalid inputs."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            load_configs("nonexistent_file.json", config_classes)
        
        # Test with invalid JSON file
        invalid_json_path = "/tmp/invalid_config.json"
        with open(invalid_json_path, 'w') as f:
            f.write("invalid json content")
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_configs(invalid_json_path, config_classes)
        finally:
            Path(invalid_json_path).unlink(missing_ok=True)
    
    def test_load_configs_performance(self, config_file_path, config_classes):
        """Test that load_configs completes in reasonable time."""
        import time
        
        start_time = time.time()
        loaded_configs = load_configs(config_file_path, config_classes)
        end_time = time.time()
        
        load_time = end_time - start_time
        assert load_time < 10.0, f"load_configs took too long: {load_time:.2f} seconds"
        
        print(f"load_configs completed in {load_time:.3f} seconds")
        print(f"Loaded {len(loaded_configs)} configurations")


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "-s"])
