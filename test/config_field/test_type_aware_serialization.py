"""
Unit tests for TypeAwareConfigSerializer class.

This module contains tests for the TypeAwareConfigSerializer class,
with particular focus on job type variant handling.
"""

import unittest
from unittest import mock
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from typing import Dict, Any, List, Optional

from pydantic import BaseModel, Field

from src.config_field_manager.type_aware_config_serializer import (
    TypeAwareConfigSerializer,
    serialize_config,
    deserialize_config
)
from src.config_field_manager.constants import SerializationMode


# Test model classes
class TestEnum(Enum):
    A = "a"
    B = "b"

class NestedModel(BaseModel):
    value: str = "nested"
    numbers: List[int] = [1, 2, 3]

class TestConfig(BaseModel):
    """Base config for testing."""
    name: str
    value: int = 42
    nested: Optional[NestedModel] = None

class JobTypeConfig(TestConfig):
    """Config with job type for testing variants."""
    job_type: Optional[str] = None
    data_type: Optional[str] = None
    mode: Optional[str] = None


class TestTypeAwareSerialization(unittest.TestCase):
    """Test cases for TypeAwareConfigSerializer."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_config = TestConfig(name="test")
        self.nested_config = TestConfig(
            name="nested_test",
            nested=NestedModel(value="custom", numbers=[4, 5, 6])
        )
        self.serializer = TypeAwareConfigSerializer()
        
        # Create configs with different job types
        self.training_config = JobTypeConfig(name="training_test", job_type="training")
        self.calibration_config = JobTypeConfig(name="calib_test", job_type="calibration")
        self.validation_config = JobTypeConfig(name="valid_test", job_type="validation")
        self.testing_config = JobTypeConfig(name="test_test", job_type="testing")
        
        # Config with multiple variant attributes
        self.complex_variant_config = JobTypeConfig(
            name="complex",
            job_type="training",
            data_type="tabular",
            mode="incremental"
        )
        
    def test_basic_serialization(self):
        """Test basic serialization of a config object."""
        result = self.serializer.serialize(self.test_config)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["value"], 42)
        
    def test_nested_serialization(self):
        """Test serialization with nested models."""
        result = self.serializer.serialize(self.nested_config)
        self.assertIsInstance(result, dict)
        self.assertIn("nested", result)
        self.assertIsInstance(result["nested"], dict)
        self.assertEqual(result["nested"]["value"], "custom")
        
    def test_generate_step_name_basic(self):
        """Test generate_step_name with a basic config."""
        with mock.patch('src.pipeline_steps.config_base.BasePipelineConfig.get_step_name') as mock_get_step_name:
            mock_get_step_name.return_value = "TestConfig"
            step_name = self.serializer.generate_step_name(self.test_config)
            self.assertEqual(step_name, "TestConfig")
            mock_get_step_name.assert_called_once_with("TestConfig")
            
    def test_generate_step_name_job_type(self):
        """Test generate_step_name with job type variants."""
        configs_and_expected = [
            (self.training_config, "JobTypeConfig_training"),
            (self.calibration_config, "JobTypeConfig_calibration"),
            (self.validation_config, "JobTypeConfig_validation"),
            (self.testing_config, "JobTypeConfig_testing")
        ]
        
        with mock.patch('src.pipeline_steps.config_base.BasePipelineConfig.get_step_name') as mock_get_step_name:
            mock_get_step_name.return_value = "JobTypeConfig"
            
            # Test instance method
            for config, expected in configs_and_expected:
                step_name = self.serializer.generate_step_name(config)
                self.assertEqual(step_name, expected)
    
    def test_generate_step_name_multiple_attributes(self):
        """Test generate_step_name with multiple variant attributes."""
        with mock.patch('src.pipeline_steps.config_base.BasePipelineConfig.get_step_name') as mock_get_step_name:
            mock_get_step_name.return_value = "JobTypeConfig"
            step_name = self.serializer.generate_step_name(self.complex_variant_config)
            self.assertEqual(step_name, "JobTypeConfig_training_tabular_incremental")
    
    def test_serialize_config_includes_step_name(self):
        """Test that serialize_config includes job type in step names."""
        with mock.patch('src.pipeline_steps.config_base.BasePipelineConfig.get_step_name') as mock_get_step_name:
            mock_get_step_name.return_value = "JobTypeConfig"
            
            # Test training job type
            result = serialize_config(self.training_config)
            self.assertIsInstance(result, dict)
            self.assertIn("_metadata", result)
            self.assertEqual(result["_metadata"]["step_name"], "JobTypeConfig_training")
            
            # Test calibration job type
            result = serialize_config(self.calibration_config)
            self.assertIn("_metadata", result)
            self.assertEqual(result["_metadata"]["step_name"], "JobTypeConfig_calibration")
    
    def test_serialize_deserialize_preserves_job_type(self):
        """Test that serialization and deserialization preserves job type."""
        # Register the JobTypeConfig class for proper deserialization
        config_classes = {"JobTypeConfig": JobTypeConfig}
        serializer = TypeAwareConfigSerializer(config_classes=config_classes)
        
        # Serialize with job type
        serialized = serializer.serialize(self.training_config)
        
        # Add metadata manually since we're using the serializer directly
        serialized["_metadata"] = {
            "step_name": "JobTypeConfig_training",
            "config_type": "JobTypeConfig"
        }
        
        # Deserialize
        deserialized = serializer.deserialize(serialized, expected_type=JobTypeConfig)
        
        # Check that job type is preserved
        self.assertIsInstance(deserialized, JobTypeConfig)
        self.assertEqual(deserialized.job_type, "training")
        
    def test_full_serialize_config_cycle(self):
        """Test full serialize_config and deserialize_config cycle with job types."""
        # Register the JobTypeConfig class
        config_classes = {"JobTypeConfig": JobTypeConfig}
        
        # Serialize with job type
        serialized = serialize_config(self.complex_variant_config)
        
        # Convert to JSON and back to simulate storage
        json_str = json.dumps(serialized)
        loaded_dict = json.loads(json_str)
        
        # Deserialize
        deserialized = deserialize_config(loaded_dict, JobTypeConfig)
        
        # Check all attributes are preserved
        self.assertEqual(deserialized.name, "complex")
        self.assertEqual(deserialized.job_type, "training")
        self.assertEqual(deserialized.data_type, "tabular")
        self.assertEqual(deserialized.mode, "incremental")


if __name__ == '__main__':
    unittest.main()
