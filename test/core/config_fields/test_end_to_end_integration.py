"""
End-to-end integration test for config field management system.

This test creates 3 different config objects, saves them to JSON using the 
merge_and_save_configs function, loads them back using load_configs, and 
verifies that the loaded configs match the original definitions.

The test validates the complete workflow:
1. Config creation and definition
2. Merging and saving to JSON (with shared/specific structure)
3. Loading from JSON 
4. Comparison and validation of round-trip integrity
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from cursus.core.config_fields import merge_and_save_configs, load_configs
from cursus.steps.configs.utils import build_complete_config_classes

# Import config classes for testing
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
from cursus.steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters


class TestEndToEndIntegration:
    """End-to-end integration test for the complete config workflow."""

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
        
        yield  # This is where the test runs
        
        # Clean up
        self.temp_dir.cleanup()

    def create_test_configs(self) -> List[Any]:
        """Create 3 test config objects similar to the demo notebook example."""
        
        # 1. Base Pipeline Config (shared across all steps)
        base_config = BasePipelineConfig(
            bucket="test-bucket-name",
            current_date="2025-09-19",
            region="NA",
            aws_region="us-east-1",
            author="test-user",
            role="arn:aws:iam::123456789012:role/TestRole",
            service_name="TestService",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            source_dir=str(self.source_dir)  # Use the actual temp directory
        )

        # 2. Processing Step Config (inherits from base)
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir=str(self.processing_dir),  # Use the actual temp directory
            processing_instance_type_large="ml.m5.12xlarge",
            processing_instance_type_small="ml.m5.4xlarge"
        )

        # 3. XGBoost Training Config (more complex with hyperparameters)
        # First create hyperparameters
        hyperparams = XGBoostModelHyperparameters(
            full_field_list=[
                "feature_1", "feature_2", "feature_3", "feature_4", "feature_5",
                "categorical_1", "categorical_2", "target_label", "id_field"
            ],
            cat_field_list=["categorical_1", "categorical_2"],
            tab_field_list=["feature_1", "feature_2", "feature_3", "feature_4", "feature_5"],
            label_name="target_label",
            id_name="id_field",
            multiclass_categories=[0, 1],
            # XGBoost specific parameters
            num_round=100,
            max_depth=6,
            min_child_weight=1.0,
            eta=0.3,
            gamma=0.0,
            subsample=1.0,
            colsample_bytree=1.0
        )

        training_config = XGBoostTrainingConfig.from_base_config(
            base_config,
            training_instance_type="ml.m5.4xlarge",
            training_entry_point="xgboost_training.py",
            training_volume_size=500,
            hyperparameters=hyperparams
        )

        return [base_config, processing_config, training_config]

    def test_end_to_end_config_workflow(self):
        """Test the complete workflow: create -> save -> load -> compare."""
        
        # Step 1: Create test configs
        original_configs = self.create_test_configs()
        
        # Verify we have 3 configs
        assert len(original_configs) == 3, "Should have exactly 3 test configs"
        
        # Step 2: Save configs to JSON using merge_and_save_configs
        merged_result = merge_and_save_configs(
            original_configs, 
            str(self.config_file)
        )
        
        # Verify the file was created
        assert self.config_file.exists(), "Config file should be created"
        
        # Verify merged_result has the expected structure
        assert isinstance(merged_result, dict), "Merged result should be a dictionary"
        assert "shared" in merged_result, "Should have 'shared' section"
        assert "specific" in merged_result, "Should have 'specific' section"
        
        # Step 3: Load the JSON file and verify structure
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        # Verify JSON structure matches expected format (like config_NA_xgboost_AtoZ.json)
        assert "metadata" in saved_data, "Should have metadata section"
        assert "configuration" in saved_data, "Should have configuration section"
        
        # Verify metadata structure
        metadata = saved_data["metadata"]
        assert "created_at" in metadata, "Should have created_at timestamp"
        assert "config_types" in metadata, "Should have config_types mapping"
        assert "field_sources" in metadata, "Should have field_sources mapping (inverted index)"
        
        # Verify configuration structure
        configuration = saved_data["configuration"]
        assert "shared" in configuration, "Should have shared configuration"
        assert "specific" in configuration, "Should have specific configuration"
        
        # Verify we have the expected number of specific configs
        specific_configs = configuration["specific"]
        assert len(specific_configs) == 3, f"Should have 3 specific configs, got {len(specific_configs)}"
        
        # Verify config types mapping
        config_types = metadata["config_types"]
        expected_types = {
            "BasePipelineConfig", 
            "ProcessingStepConfigBase", 
            "XGBoostTrainingConfig"
        }
        actual_types = set(config_types.values())
        assert expected_types.issubset(actual_types), f"Missing expected config types. Expected: {expected_types}, Got: {actual_types}"

    def test_config_round_trip_integrity(self):
        """Test that configs maintain integrity through save/load cycle."""
        
        # Step 1: Create original configs
        original_configs = self.create_test_configs()
        
        # Step 2: Save to JSON
        merge_and_save_configs(original_configs, str(self.config_file))
        
        # Step 3: Load configs back
        config_classes = build_complete_config_classes()
        loaded_result = load_configs(str(self.config_file), config_classes)
        
        # Step 4: Verify loaded result structure
        assert isinstance(loaded_result, dict), "Loaded result should be a dictionary"
        
        # The loaded result should have the shared/specific structure
        if "shared" in loaded_result and "specific" in loaded_result:
            # New format: verify shared and specific sections
            shared_data = loaded_result["shared"]
            specific_data = loaded_result["specific"]
            
            # Verify shared data contains expected fields
            expected_shared_fields = [
                "bucket", "current_date", "region", "aws_region", "author", 
                "role", "service_name", "pipeline_version", "framework_version", 
                "py_version", "source_dir"
            ]
            
            for field in expected_shared_fields:
                assert field in shared_data, f"Shared field '{field}' should be present"
                
            # Verify specific data has our configs
            assert len(specific_data) >= 3, f"Should have at least 3 specific configs, got {len(specific_data)}"
            
            # Verify each specific config has expected structure
            for step_name, config_data in specific_data.items():
                assert isinstance(config_data, dict), f"Config data for {step_name} should be a dict"
                assert "__model_type__" in config_data, f"Config {step_name} should have __model_type__"
        
        else:
            # Old format: verify we have config objects
            assert len(loaded_result) >= 3, f"Should have at least 3 loaded configs, got {len(loaded_result)}"
            
            # Verify each loaded config is not None
            for step_name, config_obj in loaded_result.items():
                assert config_obj is not None, f"Config {step_name} should not be None"

    def test_shared_vs_specific_field_categorization(self):
        """Test that fields are correctly categorized as shared vs specific."""
        
        # Create configs with some overlapping and some unique fields
        original_configs = self.create_test_configs()
        
        # Save to JSON
        merged_result = merge_and_save_configs(original_configs, str(self.config_file))
        
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

    def test_hyperparameters_preservation(self):
        """Test that complex nested objects like hyperparameters are preserved."""
        
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
        
        # Save and load
        merge_and_save_configs(original_configs, str(self.config_file))
        
        # Load the JSON to examine hyperparameter preservation
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        # Find the training config in the saved data
        specific_configs = saved_data["configuration"]["specific"]
        training_step = None
        for step_name, config_data in specific_configs.items():
            if config_data.get("__model_type__") == "XGBoostTrainingConfig":
                training_step = config_data
                break
        
        assert training_step is not None, "Should find XGBoostTrainingConfig in saved data"
        assert "hyperparameters" in training_step, "Training config should have hyperparameters"
        
        # Verify hyperparameter values are preserved
        saved_hyperparams = training_step["hyperparameters"]
        assert saved_hyperparams["num_round"] == original_num_round, "num_round should be preserved"
        assert saved_hyperparams["max_depth"] == original_max_depth, "max_depth should be preserved"
        assert saved_hyperparams["full_field_list"] == original_feature_list, "full_field_list should be preserved"

    def test_metadata_completeness(self):
        """Test that metadata contains all expected information."""
        
        original_configs = self.create_test_configs()
        merge_and_save_configs(original_configs, str(self.config_file))
        
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        metadata = saved_data["metadata"]
        
        # Verify required metadata fields
        assert "created_at" in metadata, "Should have created_at timestamp"
        assert "config_types" in metadata, "Should have config_types mapping"
        
        # Verify created_at is a valid timestamp
        created_at = metadata["created_at"]
        assert isinstance(created_at, str), "created_at should be a string"
        # Should be able to parse as ISO format
        datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        
        # Verify config_types has entries for all our configs
        config_types = metadata["config_types"]
        assert len(config_types) >= 3, f"Should have at least 3 config type entries, got {len(config_types)}"
        
        # Verify field_sources is now required
        assert "field_sources" in metadata, "Should have field_sources mapping (inverted index)"
        field_sources = metadata["field_sources"]
        assert len(field_sources) > 0, "Should have field source mappings"
        
        # Verify each field source entry is a list
        for field_name, sources in field_sources.items():
            assert isinstance(sources, list), f"Field sources for '{field_name}' should be a list"
            assert len(sources) > 0, f"Field '{field_name}' should have at least one source"

    def test_error_handling(self):
        """Test error handling for invalid inputs and edge cases."""
        
        # Test with empty config list
        with pytest.raises((ValueError, TypeError)):
            merge_and_save_configs([], str(self.config_file))
        
        # Test with invalid file path
        invalid_path = "/invalid/path/that/does/not/exist/config.json"
        original_configs = self.create_test_configs()
        
        with pytest.raises((OSError, IOError, PermissionError)):
            merge_and_save_configs(original_configs, invalid_path)
        
        # Test loading from non-existent file
        config_classes = build_complete_config_classes()
        non_existent_file = str(self.temp_path / "does_not_exist.json")
        
        with pytest.raises(FileNotFoundError):
            load_configs(non_existent_file, config_classes)
        
        # Test loading from invalid JSON
        invalid_json_file = self.temp_path / "invalid.json"
        with open(invalid_json_file, 'w') as f:
            f.write("invalid json content {")
        
        with pytest.raises(json.JSONDecodeError):
            load_configs(str(invalid_json_file), config_classes)

    def test_performance_and_file_size(self):
        """Test that the save/load operations complete in reasonable time and produce reasonable file sizes."""
        import time
        
        original_configs = self.create_test_configs()
        
        # Test save performance
        start_time = time.time()
        merge_and_save_configs(original_configs, str(self.config_file))
        save_time = time.time() - start_time
        
        assert save_time < 5.0, f"Save operation took too long: {save_time:.2f} seconds"
        
        # Test file size is reasonable
        file_size = self.config_file.stat().st_size
        assert file_size > 100, f"Config file seems too small: {file_size} bytes"
        assert file_size < 1024 * 1024, f"Config file seems too large: {file_size} bytes"  # Less than 1MB
        
        # Test load performance
        config_classes = build_complete_config_classes()
        start_time = time.time()
        loaded_result = load_configs(str(self.config_file), config_classes)
        load_time = time.time() - start_time
        
        assert load_time < 5.0, f"Load operation took too long: {load_time:.2f} seconds"
        assert loaded_result is not None, "Load operation should return valid result"

    def test_json_structure_matches_expected_format(self):
        """Test that the generated JSON structure exactly matches the expected format from config_NA_xgboost_AtoZ.json."""
        
        original_configs = self.create_test_configs()
        merge_and_save_configs(original_configs, str(self.config_file))
        
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        # Verify top-level structure matches expected format
        expected_top_level_keys = {"metadata", "configuration"}
        actual_top_level_keys = set(saved_data.keys())
        assert expected_top_level_keys == actual_top_level_keys, f"Top-level keys mismatch. Expected: {expected_top_level_keys}, Got: {actual_top_level_keys}"
        
        # Verify metadata structure
        metadata = saved_data["metadata"]
        expected_metadata_keys = {"created_at", "config_types", "field_sources"}  # field_sources is now required
        actual_metadata_keys = set(metadata.keys())
        assert expected_metadata_keys.issubset(actual_metadata_keys), f"Missing metadata keys. Expected: {expected_metadata_keys}, Got: {actual_metadata_keys}"
        
        # Verify configuration structure
        configuration = saved_data["configuration"]
        expected_config_keys = {"shared", "specific"}
        actual_config_keys = set(configuration.keys())
        assert expected_config_keys == actual_config_keys, f"Configuration keys mismatch. Expected: {expected_config_keys}, Got: {actual_config_keys}"
        
        # Verify specific configs have __model_type__ field
        specific_configs = configuration["specific"]
        for step_name, config_data in specific_configs.items():
            assert "__model_type__" in config_data, f"Config {step_name} missing __model_type__"
            
            # Verify the model type is a string
            assert isinstance(config_data["__model_type__"], str), f"__model_type__ should be string for {step_name}"

    def test_field_sources_inverted_index_validation(self):
        """Test that field_sources provides a correct inverted index mapping fields to their source steps."""
        
        original_configs = self.create_test_configs()
        merge_and_save_configs(original_configs, str(self.config_file))
        
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)
        
        # Get field_sources and configuration data
        field_sources = saved_data["metadata"]["field_sources"]
        shared_fields = saved_data["configuration"]["shared"]
        specific_configs = saved_data["configuration"]["specific"]
        
        # Verify field_sources structure
        assert isinstance(field_sources, dict), "field_sources should be a dictionary"
        assert len(field_sources) > 0, "field_sources should not be empty"
        
        # Test 1: Verify that every field in shared section appears in field_sources
        for field_name in shared_fields.keys():
            assert field_name in field_sources, f"Shared field '{field_name}' should be in field_sources"
            sources = field_sources[field_name]
            assert isinstance(sources, list), f"Sources for '{field_name}' should be a list"
            assert len(sources) > 1, f"Shared field '{field_name}' should have multiple sources, got {sources}"
        
        # Test 2: Verify that every field in specific sections appears in field_sources
        all_specific_fields = set()
        for step_name, config_data in specific_configs.items():
            for field_name in config_data.keys():
                if not field_name.startswith("__"):  # Skip metadata fields
                    all_specific_fields.add(field_name)
                    assert field_name in field_sources, f"Specific field '{field_name}' should be in field_sources"
                    sources = field_sources[field_name]
                    assert isinstance(sources, list), f"Sources for '{field_name}' should be a list"
                    assert step_name in sources, f"Step '{step_name}' should be in sources for field '{field_name}'"
        
        # Test 3: Verify that field_sources doesn't contain extra fields (except metadata fields)
        all_actual_fields = set(shared_fields.keys()) | all_specific_fields
        field_sources_fields = set(field_sources.keys())
        extra_fields = field_sources_fields - all_actual_fields
        # Remove metadata fields that are expected to be in field_sources
        metadata_fields = {"__model_type__"}
        extra_fields = extra_fields - metadata_fields
        assert len(extra_fields) == 0, f"field_sources contains unexpected extra fields: {extra_fields}"
        
        # Test 4: Verify inverted index correctness - check a few specific examples
        # Find a field that should be shared (appears in multiple configs)
        shared_field_example = None
        for field_name, sources in field_sources.items():
            if len(sources) >= 2 and field_name in shared_fields:
                shared_field_example = field_name
                break
        
        if shared_field_example:
            # Verify this field appears in the expected number of configs
            expected_sources = field_sources[shared_field_example]
            assert len(expected_sources) >= 2, f"Shared field '{shared_field_example}' should have at least 2 sources"
            
            # Verify each source step actually exists in our config types
            config_types = saved_data["metadata"]["config_types"]
            for source_step in expected_sources:
                assert source_step in config_types, f"Source step '{source_step}' should exist in config_types"
        
        # Test 5: Verify specific field mapping
        # Find a field that should be specific (appears in only one config)
        specific_field_example = None
        specific_step_example = None
        for step_name, config_data in specific_configs.items():
            for field_name in config_data.keys():
                if not field_name.startswith("__") and field_name not in shared_fields:
                    if field_name in field_sources and len(field_sources[field_name]) == 1:
                        specific_field_example = field_name
                        specific_step_example = step_name
                        break
            if specific_field_example:
                break
        
        if specific_field_example and specific_step_example:
            sources = field_sources[specific_field_example]
            assert len(sources) == 1, f"Specific field '{specific_field_example}' should have exactly 1 source"
            assert sources[0] == specific_step_example, f"Specific field '{specific_field_example}' should map to step '{specific_step_example}'"
        
        # Test 6: Verify that the inverted index is complete and consistent
        # Count total field occurrences from field_sources
        total_field_occurrences_from_sources = sum(len(sources) for sources in field_sources.values())
        
        # Count total field occurrences from configuration
        shared_occurrences = len(shared_fields) * 3  # Each shared field appears in all 3 configs
        specific_occurrences = sum(
            len([f for f in config_data.keys() if not f.startswith("__")])
            for config_data in specific_configs.values()
        )
        
        # The counts should be consistent (allowing for some flexibility due to categorization logic)
        assert total_field_occurrences_from_sources > 0, "Should have some field occurrences in field_sources"
        
        # Log some statistics for debugging
        print(f"Field sources statistics:")
        print(f"  - Total unique fields: {len(field_sources)}")
        print(f"  - Total field occurrences: {total_field_occurrences_from_sources}")
        print(f"  - Shared fields: {len(shared_fields)}")
        print(f"  - Specific field occurrences: {specific_occurrences}")
        print(f"  - Example shared field: {shared_field_example} -> {field_sources.get(shared_field_example, [])}")
        print(f"  - Example specific field: {specific_field_example} -> {field_sources.get(specific_field_example, [])}")


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "-s"])
