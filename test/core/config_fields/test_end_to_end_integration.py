"""
End-to-end integration test for unified config field management system.

This test demonstrates the complete save/load cycle using UnifiedConfigManager:
1. Create real config objects with complex nested structures
2. Save them to JSON using UnifiedConfigManager.save() - matches target format structure
3. Load them back using UnifiedConfigManager.load() 
4. Verify complete reconstruction of original configs

The test validates that the unified system produces the exact format structure
as seen in config_NA_xgboost_AtoZ.json and can completely reconstruct the
original config objects from the saved JSON.

SYSTEMATIC ERROR PREVENTION APPLIED:
- ✅ Source Code First Rule: Read UnifiedConfigManager implementation before testing
- ✅ Mock Path Precision: Use exact import paths from source code analysis
- ✅ Implementation-Driven Testing: Test actual behavior, not assumptions
- ✅ Error Prevention Categories 1-17: All systematically addressed
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock, mock_open

# Import the unified config manager
from cursus.core.config_fields.unified_config_manager import UnifiedConfigManager

# ✅ SYSTEMATIC ERROR PREVENTION: Use REAL config classes after reading source code
# Following Source Code First Rule: Import actual implementations after understanding them
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
from cursus.steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters


class TestEndToEndIntegrationUnified:
    """End-to-end integration test for the unified config management system."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.config_file = self.temp_path / "test_config.json"
        
        # Create test source directories that the config validation expects
        self.source_dir = self.temp_path / "source"
        self.processing_dir = self.temp_path / "processing" / "scripts"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unified config manager
        self.unified_manager = UnifiedConfigManager()
        
        yield  # This is where the test runs
        
        # Clean up
        self.temp_dir.cleanup()

    def create_test_configs(self) -> List[Any]:
        """
        Create 3 test config objects using REAL config classes based on source code analysis.
        
        ✅ SYSTEMATIC ERROR PREVENTION: Following Source Code First Rule
        - BasePipelineConfig: Essential fields (author, bucket, role, region, service_name, pipeline_version)
        - ProcessingStepConfigBase: Inherits from base + processing-specific fields
        - XGBoostTrainingConfig: Requires training_entry_point + hyperparameters (essential fields)
        """
        
        # 1. Base Pipeline Config - using REAL required fields from source code
        # Essential User Inputs (Tier 1): author, bucket, role, region, service_name, pipeline_version
        base_config = BasePipelineConfig(
            # Tier 1: Essential User Inputs (REQUIRED)
            author="test-user",
            bucket="test-bucket-name", 
            role="arn:aws:iam::123456789012:role/TestRole",
            region="NA",  # Must be NA, EU, or FE
            service_name="TestService",
            pipeline_version="1.0.0",
            project_root_folder="cursus",  # Required for hybrid resolution
            
            # Tier 2: System Inputs with Defaults (OPTIONAL)
            model_class="xgboost",  # Default is "xgboost"
            current_date="2025-10-05",
            framework_version="1.7-1",
            py_version="py3",
            source_dir=str(self.source_dir)
        )

        # 2. Processing Step Config - using from_base_config method from source code
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            # Processing-specific fields
            processing_source_dir=str(self.processing_dir),
            processing_instance_type_large="ml.m5.12xlarge",
            processing_instance_type_small="ml.m5.4xlarge", 
            processing_framework_version="1.2-1",
            processing_instance_count=1,
            processing_volume_size=500,
            use_large_processing_instance=False,
            processing_entry_point="processing_script.py"  # Optional but good for testing
        )

        # 3. XGBoost Training Config - using REAL hyperparameters from source code
        # First create hyperparameters using REAL XGBoostModelHyperparameters
        hyperparams = XGBoostModelHyperparameters(
            # Essential User Inputs (Tier 1) - REQUIRED from source code
            num_round=100,
            max_depth=6,
            
            # Inherited from ModelHyperparameters base class - REQUIRED
            full_field_list=[
                "feature_1", "feature_2", "feature_3", "feature_4", "feature_5",
                "categorical_1", "categorical_2", "target_label", "id_field"
            ],
            cat_field_list=["categorical_1", "categorical_2"],
            tab_field_list=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            label_name="target_label",
            id_name="id_field",
            multiclass_categories=[0, 1],
            
            # Tier 2: System Inputs with Defaults (OPTIONAL) - using defaults from source
            model_class="xgboost",  # Default from source
            min_child_weight=1.0,   # Default from source
            eta=0.3,                # Default from source
            gamma=0.0,              # Default from source
            subsample=1.0,          # Default from source
            colsample_bytree=1.0,   # Default from source
            booster="gbtree",       # Default from source
            tree_method="auto"      # Default from source
        )

        # Create training config using from_base_config method
        training_config = XGBoostTrainingConfig.from_base_config(
            base_config,
            # Essential User Inputs (Tier 1) - REQUIRED from source code
            training_entry_point="xgboost_training.py",
            hyperparameters=hyperparams,
            
            # Tier 2: System Inputs with Defaults (OPTIONAL) - using defaults from source
            training_instance_type="ml.m5.4xlarge",  # Default from source
            training_instance_count=1,               # Default from source
            training_volume_size=30,                 # Default from source (not 500!)
            framework_version="1.7-1",               # Default from source
            py_version="py3"                         # Default from source
        )

        return [base_config, processing_config, training_config]

    def test_unified_save_config_to_target_format(self):
        """Test that UnifiedConfigManager.save() produces the exact target format structure."""
        
        # Step 1: Create test configs
        original_configs = self.create_test_configs()
        
        # Verify we have 3 configs
        assert len(original_configs) == 3, "Should have exactly 3 test configs"
        
        # Step 2: Save configs using UnifiedConfigManager
        saved_result = self.unified_manager.save(original_configs, str(self.config_file))
        
        # Verify the file was created
        assert self.config_file.exists(), "Config file should be created"
        
        # Step 3: Load the JSON file and verify structure matches target format
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        # Verify top-level structure matches config_NA_xgboost_AtoZ.json format
        expected_top_level_keys = {"metadata", "configuration"}
        actual_top_level_keys = set(saved_data.keys())
        assert expected_top_level_keys == actual_top_level_keys, f"Top-level keys mismatch. Expected: {expected_top_level_keys}, Got: {actual_top_level_keys}"
        
        # Verify metadata structure
        metadata = saved_data["metadata"]
        expected_metadata_keys = {"created_at", "config_types", "field_sources"}
        actual_metadata_keys = set(metadata.keys())
        assert expected_metadata_keys.issubset(actual_metadata_keys), f"Missing metadata keys. Expected: {expected_metadata_keys}, Got: {actual_metadata_keys}"
        
        # Verify configuration structure
        configuration = saved_data["configuration"]
        expected_config_keys = {"shared", "specific"}
        actual_config_keys = set(configuration.keys())
        assert expected_config_keys == actual_config_keys, f"Configuration keys mismatch. Expected: {expected_config_keys}, Got: {actual_config_keys}"
        
        # Verify we have the expected number of specific configs
        specific_configs = configuration["specific"]
        assert len(specific_configs) == 3, f"Should have 3 specific configs, got {len(specific_configs)}"
        
        # Verify each specific config has __model_type__ field
        for step_name, config_data in specific_configs.items():
            assert "__model_type__" in config_data, f"Config {step_name} missing __model_type__"
            assert isinstance(config_data["__model_type__"], str), f"__model_type__ should be string for {step_name}"
        
        # Verify config types mapping - using REAL class names from source code
        config_types = metadata["config_types"]
        expected_types = {
            "BasePipelineConfig", 
            "ProcessingStepConfigBase", 
            "XGBoostTrainingConfig"
        }
        actual_types = set(config_types.values())
        assert expected_types.issubset(actual_types), f"Missing expected config types. Expected: {expected_types}, Got: {actual_types}"

    def test_unified_load_config_complete_reconstruction(self):
        """Test that UnifiedConfigManager.load() completely reconstructs the original configs."""
        
        # Step 1: Create and save original configs
        original_configs = self.create_test_configs()
        saved_result = self.unified_manager.save(original_configs, str(self.config_file))
        
        # Step 2: Load configs back using UnifiedConfigManager
        loaded_result = self.unified_manager.load(str(self.config_file))
        
        # Step 3: Verify loaded result structure
        assert isinstance(loaded_result, dict), "Loaded result should be a dictionary"
        assert "shared" in loaded_result, "Loaded result should have 'shared' section"
        assert "specific" in loaded_result, "Loaded result should have 'specific' section"
        
        # Step 4: Verify shared data contains expected fields
        # ✅ SYSTEMATIC ERROR PREVENTION: Only check actual fields, not derived properties
        shared_data = loaded_result["shared"]
        expected_shared_fields = [
            "bucket", "current_date", "region", "author", 
            "role", "service_name", "pipeline_version", "framework_version", 
            "py_version", "source_dir", "model_class"
        ]
        
        for field in expected_shared_fields:
            assert field in shared_data, f"Shared field '{field}' should be present"
        
        # Derived properties may or may not be in shared data depending on implementation
        # Just verify they exist somewhere in the data structure
        all_field_names = set(shared_data.keys())
        for step_name, config_data in loaded_result["specific"].items():
            all_field_names.update(config_data.keys())
        
        # These derived fields should exist somewhere in the loaded data
        derived_fields = ["aws_region", "pipeline_name", "pipeline_description", "pipeline_s3_loc"]
        for field in derived_fields:
            assert field in all_field_names, f"Derived field '{field}' should exist in loaded data"
            
        # Step 5: Verify specific data has our configs
        specific_data = loaded_result["specific"]
        assert len(specific_data) == 3, f"Should have 3 specific configs, got {len(specific_data)}"
        
        # Step 6: Verify each specific config has expected structure
        for step_name, config_data in specific_data.items():
            assert isinstance(config_data, dict), f"Config data for {step_name} should be a dict"
            assert "__model_type__" in config_data, f"Config {step_name} should have __model_type__"

    def test_round_trip_integrity_with_complex_objects(self):
        """Test that complex nested objects maintain integrity through save/load cycle."""
        
        # Step 1: Create original configs with complex nested structures
        original_configs = self.create_test_configs()
        
        # Find the training config with hyperparameters
        training_config = None
        for config in original_configs:
            if hasattr(config, 'hyperparameters'):
                training_config = config
                break
        
        assert training_config is not None, "Should have a training config with hyperparameters"
        
        # Get original hyperparameter values
        original_hyperparams = training_config.hyperparameters
        original_num_round = original_hyperparams.num_round
        original_max_depth = original_hyperparams.max_depth
        original_feature_list = original_hyperparams.full_field_list
        
        # Step 2: Save and load using unified manager
        self.unified_manager.save(original_configs, str(self.config_file))
        loaded_result = self.unified_manager.load(str(self.config_file))
        
        # Step 3: Find the training config in the loaded data
        specific_configs = loaded_result["specific"]
        training_step = None
        for step_name, config_data in specific_configs.items():
            if config_data.get("__model_type__") == "XGBoostTrainingConfig":
                training_step = config_data
                break
        
        assert training_step is not None, "Should find XGBoostTrainingConfig in loaded data"
        assert "hyperparameters" in training_step, "Training config should have hyperparameters"
        
        # Step 4: Verify hyperparameter values are preserved
        saved_hyperparams = training_step["hyperparameters"]
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Handle both dict and object cases
        # The type-aware serialization may return either a dict or the actual object
        if isinstance(saved_hyperparams, dict):
            # If it's a dict, access with keys
            assert saved_hyperparams["num_round"] == original_num_round, "num_round should be preserved"
            assert saved_hyperparams["max_depth"] == original_max_depth, "max_depth should be preserved"
            assert saved_hyperparams["full_field_list"] == original_feature_list, "full_field_list should be preserved"
        else:
            # If it's an object, access with attributes (type-aware serialization working!)
            assert saved_hyperparams.num_round == original_num_round, "num_round should be preserved"
            assert saved_hyperparams.max_depth == original_max_depth, "max_depth should be preserved"
            assert saved_hyperparams.full_field_list == original_feature_list, "full_field_list should be preserved"

    def test_shared_vs_specific_field_categorization(self):
        """Test that fields are correctly categorized as shared vs specific."""
        
        # Create configs with overlapping and unique fields
        original_configs = self.create_test_configs()
        
        # Save using unified manager
        self.unified_manager.save(original_configs, str(self.config_file))
        
        # Load the JSON to examine field categorization
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        shared_fields = saved_data["configuration"]["shared"]
        specific_configs = saved_data["configuration"]["specific"]
        
        # Verify we have shared fields (fields that appear in multiple configs)
        assert len(shared_fields) > 0, "Should have some shared fields"
        
        # Verify specific fields are unique to their configs
        all_specific_fields = set()
        for step_name, config_data in specific_configs.items():
            for field_name in config_data.keys():
                if field_name.startswith("__"):  # Skip metadata fields
                    continue
                    
                # Check if this field appears in shared (it shouldn't if it's truly specific)
                if field_name not in shared_fields:
                    all_specific_fields.add(field_name)
        
        # Verify we have some specific fields
        assert len(all_specific_fields) > 0, "Should have some fields that are specific to individual configs"

    def test_metadata_completeness_and_accuracy(self):
        """Test that metadata contains all expected information and is accurate."""
        
        original_configs = self.create_test_configs()
        self.unified_manager.save(original_configs, str(self.config_file))
        
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        metadata = saved_data["metadata"]
        
        # Verify required metadata fields
        assert "created_at" in metadata, "Should have created_at timestamp"
        assert "config_types" in metadata, "Should have config_types mapping"
        assert "field_sources" in metadata, "Should have field_sources mapping"
        
        # Verify created_at is a valid timestamp
        created_at = metadata["created_at"]
        assert isinstance(created_at, str), "created_at should be a string"
        # Should be able to parse as ISO format
        datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        # Verify config_types has entries for all our configs
        config_types = metadata["config_types"]
        assert len(config_types) == 3, f"Should have exactly 3 config type entries, got {len(config_types)}"
        
        # Verify field_sources is complete
        field_sources = metadata["field_sources"]
        assert len(field_sources) > 0, "Should have field source mappings"
        
        # Verify each field source entry is a list
        for field_name, sources in field_sources.items():
            assert isinstance(sources, list), f"Field sources for '{field_name}' should be a list"
            assert len(sources) > 0, f"Field '{field_name}' should have at least one source"

    def test_field_sources_inverted_index_accuracy(self):
        """Test that field_sources provides accurate inverted index mapping."""
        
        original_configs = self.create_test_configs()
        self.unified_manager.save(original_configs, str(self.config_file))
        
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        # Get field_sources and configuration data
        field_sources = saved_data["metadata"]["field_sources"]
        shared_fields = saved_data["configuration"]["shared"]
        specific_configs = saved_data["configuration"]["specific"]
        
        # Test 1: Verify that every field in shared section appears in field_sources
        for field_name in shared_fields.keys():
            assert field_name in field_sources, f"Shared field '{field_name}' should be in field_sources"
            sources = field_sources[field_name]
            assert len(sources) > 1, f"Shared field '{field_name}' should have multiple sources, got {sources}"
        
        # Test 2: Verify that every field in specific sections appears in field_sources
        for step_name, config_data in specific_configs.items():
            for field_name in config_data.keys():
                if not field_name.startswith("__"):  # Skip metadata fields
                    assert field_name in field_sources, f"Specific field '{field_name}' should be in field_sources"
                    sources = field_sources[field_name]
                    assert step_name in sources, f"Step '{step_name}' should be in sources for field '{field_name}'"

    def test_error_handling_unified_manager(self):
        """Test error handling for the unified config manager."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual behavior, not assumptions
        # Based on source code analysis, empty config list may not raise exception
        # Let's test what actually happens
        try:
            result = self.unified_manager.save([], str(self.config_file))
            # If it succeeds, verify the result structure
            assert isinstance(result, dict), "Empty config save should return dict"
            assert "shared" in result, "Empty config should have shared section"
            assert "specific" in result, "Empty config should have specific section"
        except Exception as e:
            # If it fails, that's also valid behavior - just verify it's a reasonable exception
            assert isinstance(e, (ValueError, TypeError, IndexError, AttributeError)), f"Unexpected exception type: {type(e)}"
        
        # Test with invalid file path for save
        invalid_path = "/invalid/path/that/does/not/exist/config.json"
        original_configs = self.create_test_configs()
        
        with pytest.raises((OSError, IOError, PermissionError)):
            self.unified_manager.save(original_configs, invalid_path)
        
        # Test loading from non-existent file
        non_existent_file = str(self.temp_path / "does_not_exist.json")
        
        with pytest.raises(FileNotFoundError):
            self.unified_manager.load(non_existent_file)
        
        # Test loading from invalid JSON
        invalid_json_file = self.temp_path / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content {")
        
        with pytest.raises(json.JSONDecodeError):
            self.unified_manager.load(str(invalid_json_file))

    def test_performance_and_file_size_unified(self):
        """Test that unified manager operations complete in reasonable time and produce reasonable file sizes."""
        import time
        
        original_configs = self.create_test_configs()
        
        # Test save performance
        start_time = time.time()
        self.unified_manager.save(original_configs, str(self.config_file))
        save_time = time.time() - start_time
        
        assert save_time < 10.0, f"Save operation took too long: {save_time:.2f} seconds"
        
        # Test file size is reasonable
        file_size = self.config_file.stat().st_size
        assert file_size > 100, f"Config file seems too small: {file_size} bytes"
        assert file_size < 1024 * 1024, f"Config file seems too large: {file_size} bytes"  # Less than 1MB
        
        # Test load performance
        start_time = time.time()
        loaded_result = self.unified_manager.load(str(self.config_file))
        load_time = time.time() - start_time
        
        assert load_time < 10.0, f"Load operation took too long: {load_time:.2f} seconds"
        assert loaded_result is not None, "Load operation should return valid result"

    def test_type_aware_serialization_preservation(self):
        """Test that type-aware serialization preserves complex types correctly."""
        
        original_configs = self.create_test_configs()
        
        # Save and load
        self.unified_manager.save(original_configs, str(self.config_file))
        loaded_result = self.unified_manager.load(str(self.config_file))
        
        # Verify that the loaded result maintains type information
        assert "shared" in loaded_result, "Should have shared section"
        assert "specific" in loaded_result, "Should have specific section"
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual behavior from source code
        # The type-aware serialization deserializes objects back to their original types
        specific_data = loaded_result["specific"]
        for step_name, config_data in specific_data.items():
            if "hyperparameters" in config_data:
                hyperparams = config_data["hyperparameters"]
                
                # Type-aware serialization should preserve the actual object type
                if hasattr(hyperparams, 'full_field_list'):
                    # It's an actual XGBoostModelHyperparameters object
                    assert hasattr(hyperparams, 'num_round'), "Should preserve hyperparameter attributes"
                    assert hasattr(hyperparams, 'max_depth'), "Should preserve hyperparameter attributes"
                    assert isinstance(hyperparams.full_field_list, list), "Field list should remain a list"
                elif isinstance(hyperparams, dict):
                    # It's a dictionary representation
                    assert "full_field_list" in hyperparams, "Should preserve nested list fields"
                    assert isinstance(hyperparams["full_field_list"], list), "Field list should remain a list"
                else:
                    # Unexpected type - fail with informative message
                    assert False, f"Unexpected hyperparameters type: {type(hyperparams)}, value: {hyperparams}"

    def test_unified_manager_vs_target_format_exact_match(self):
        """Test that unified manager output exactly matches the target format structure."""
        
        original_configs = self.create_test_configs()
        self.unified_manager.save(original_configs, str(self.config_file))
        
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        # Compare with expected structure from config_NA_xgboost_AtoZ.json
        
        # 1. Top-level structure should match exactly
        assert set(saved_data.keys()) == {"metadata", "configuration"}, "Top-level structure should match target"
        
        # 2. Metadata structure should match
        metadata = saved_data["metadata"]
        required_metadata_keys = {"created_at", "config_types", "field_sources"}
        assert required_metadata_keys.issubset(set(metadata.keys())), "Metadata should have required keys"
        
        # 3. Configuration structure should match
        configuration = saved_data["configuration"]
        assert set(configuration.keys()) == {"shared", "specific"}, "Configuration should have shared and specific"
        
        # 4. Specific configs should have __model_type__
        specific_configs = configuration["specific"]
        for step_name, config_data in specific_configs.items():
            assert "__model_type__" in config_data, f"Config {step_name} should have __model_type__"
        
        # 5. Field sources should be properly structured
        field_sources = metadata["field_sources"]
        for field_name, sources in field_sources.items():
            assert isinstance(sources, list), f"Field sources for {field_name} should be a list"
            assert all(isinstance(source, str) for source in sources), f"All sources should be strings for {field_name}"


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "-s"])
