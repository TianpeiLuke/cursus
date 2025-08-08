"""
Unit tests for ConfigClassDetector class.

This module contains comprehensive tests for the ConfigClassDetector class,
testing its ability to detect required configuration classes from JSON files.
"""

import unittest
import sys
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Type, Set
from unittest.mock import patch, mock_open, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.cursus.core.config_fields.config_class_detector import (
    ConfigClassDetector,
    detect_config_classes_from_json
)
from pydantic import BaseModel


class MockConfigA(BaseModel):
    """Mock config class A for testing."""
    name: str = "mock_a"
    value: int = 1


class MockConfigB(BaseModel):
    """Mock config class B for testing."""
    name: str = "mock_b"
    value: int = 2


class BasePipelineConfig(BaseModel):
    """Mock base pipeline config for testing."""
    pipeline_name: str = "test_pipeline"


class ProcessingStepConfigBase(BaseModel):
    """Mock processing step config base for testing."""
    step_name: str = "test_step"


class TestConfigClassDetector(unittest.TestCase):
    """Test cases for ConfigClassDetector."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample configuration data for testing
        self.sample_config_data = {
            "metadata": {
                "config_types": {
                    "step1": "MockConfigA",
                    "step2": "MockConfigB",
                    "pipeline": "BasePipelineConfig"
                }
            },
            "configuration": {
                "specific": {
                    "training_step": {
                        "__model_type__": "TrainingStepConfig",
                        "learning_rate": 0.01
                    },
                    "processing_step": {
                        "__model_type__": "ProcessingStepConfig",
                        "batch_size": 32
                    }
                }
            }
        }
        
        # Mock complete config classes
        self.mock_complete_classes = {
            "MockConfigA": MockConfigA,
            "MockConfigB": MockConfigB,
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase,
            "TrainingStepConfig": MockConfigA,  # Using MockConfigA as placeholder
            "ProcessingStepConfig": MockConfigB  # Using MockConfigB as placeholder
        }

    def test_extract_class_names_from_metadata(self):
        """Test extracting class names from metadata.config_types."""
        logger = MagicMock()
        
        result = ConfigClassDetector._extract_class_names(self.sample_config_data, logger)
        
        # Should extract from metadata.config_types
        expected_classes = {"MockConfigA", "MockConfigB", "BasePipelineConfig"}
        self.assertTrue(expected_classes.issubset(result))
        
        # Should also extract from configuration.specific.__model_type__
        self.assertIn("TrainingStepConfig", result)
        self.assertIn("ProcessingStepConfig", result)

    def test_extract_class_names_from_specific_configs(self):
        """Test extracting class names from configuration.specific.__model_type__ fields."""
        config_data = {
            "configuration": {
                "specific": {
                    "step1": {
                        "__model_type__": "StepConfig1",
                        "param": "value"
                    },
                    "step2": {
                        "__model_type__": "StepConfig2",
                        "param": "value"
                    }
                }
            }
        }
        
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names(config_data, logger)
        
        self.assertIn("StepConfig1", result)
        self.assertIn("StepConfig2", result)

    def test_extract_class_names_empty_config(self):
        """Test extracting class names from empty configuration."""
        empty_config = {}
        logger = MagicMock()
        
        result = ConfigClassDetector._extract_class_names(empty_config, logger)
        
        self.assertEqual(len(result), 0)

    def test_extract_class_names_partial_config(self):
        """Test extracting class names from partial configuration data."""
        # Config with only metadata
        metadata_only = {
            "metadata": {
                "config_types": {
                    "step1": "ConfigA"
                }
            }
        }
        
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names(metadata_only, logger)
        self.assertIn("ConfigA", result)
        
        # Config with only specific configurations
        specific_only = {
            "configuration": {
                "specific": {
                    "step1": {
                        "__model_type__": "ConfigB"
                    }
                }
            }
        }
        
        result = ConfigClassDetector._extract_class_names(specific_only, logger)
        self.assertIn("ConfigB", result)

    def test_extract_class_names_invalid_specific_config(self):
        """Test extracting class names from invalid specific configuration."""
        invalid_config = {
            "configuration": {
                "specific": {
                    "step1": "not_a_dict",  # Should be ignored
                    "step2": {
                        "no_model_type": "value"  # Should be ignored
                    },
                    "step3": {
                        "__model_type__": "ValidConfig"  # Should be extracted
                    }
                }
            }
        }
        
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names(invalid_config, logger)
        
        self.assertIn("ValidConfig", result)
        self.assertEqual(len(result), 1)

    @patch('src.cursus.core.config_fields.config_class_detector.build_complete_config_classes')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file')
    def test_detect_from_json_success(self, mock_is_file, mock_file, mock_build_classes):
        """Test successful detection from JSON file."""
        # Setup mocks
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self.sample_config_data)
        mock_build_classes.return_value = self.mock_complete_classes
        
        result = ConfigClassDetector.detect_from_json("test_config.json")
        
        # Should return only the required classes plus essential ones
        self.assertIn("MockConfigA", result)
        self.assertIn("MockConfigB", result)
        self.assertIn("BasePipelineConfig", result)
        self.assertIn("ProcessingStepConfigBase", result)  # Essential class
        
        # Verify file operations
        mock_is_file.assert_called_once()
        mock_file.assert_called_once_with("test_config.json", 'r')

    @patch('src.cursus.core.config_fields.config_class_detector.build_complete_config_classes')
    @patch('pathlib.Path.is_file')
    def test_detect_from_json_file_not_found(self, mock_is_file, mock_build_classes):
        """Test detection when JSON file is not found."""
        mock_is_file.return_value = False
        mock_build_classes.return_value = self.mock_complete_classes
        
        with self.assertLogs(level='ERROR') as log:
            result = ConfigClassDetector.detect_from_json("nonexistent.json")
        
        # Should fall back to complete classes
        self.assertEqual(result, self.mock_complete_classes)
        self.assertTrue(any("Configuration file not found" in message for message in log.output))

    @patch('src.cursus.core.config_fields.config_class_detector.build_complete_config_classes')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file')
    def test_detect_from_json_invalid_json(self, mock_is_file, mock_file, mock_build_classes):
        """Test detection with invalid JSON file."""
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = "invalid json content"
        mock_build_classes.return_value = self.mock_complete_classes
        
        with self.assertLogs(level='ERROR') as log:
            result = ConfigClassDetector.detect_from_json("invalid.json")
        
        # Should fall back to complete classes
        self.assertEqual(result, self.mock_complete_classes)
        self.assertTrue(any("Error reading or parsing configuration file" in message for message in log.output))

    @patch('src.cursus.core.config_fields.config_class_detector.build_complete_config_classes')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file')
    def test_detect_from_json_no_class_names_found(self, mock_is_file, mock_file, mock_build_classes):
        """Test detection when no class names are found in config."""
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = json.dumps({})  # Empty config
        mock_build_classes.return_value = self.mock_complete_classes
        
        with self.assertLogs(level='WARNING') as log:
            result = ConfigClassDetector.detect_from_json("empty_config.json")
        
        # Should fall back to complete classes
        self.assertEqual(result, self.mock_complete_classes)
        self.assertTrue(any("No config class names found" in message for message in log.output))

    @patch('src.cursus.core.config_fields.config_class_detector.build_complete_config_classes')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file')
    def test_detect_from_json_missing_classes(self, mock_is_file, mock_file, mock_build_classes):
        """Test detection when some required classes are missing."""
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self.sample_config_data)
        
        # Mock incomplete class set (missing some classes)
        incomplete_classes = {
            "MockConfigA": MockConfigA,
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase
            # Missing MockConfigB, TrainingStepConfig, ProcessingStepConfig
        }
        mock_build_classes.return_value = incomplete_classes
        
        with self.assertLogs(level='WARNING') as log:
            result = ConfigClassDetector.detect_from_json("test_config.json")
        
        # Should include available classes and essential ones
        self.assertIn("MockConfigA", result)
        self.assertIn("BasePipelineConfig", result)
        self.assertIn("ProcessingStepConfigBase", result)
        
        # Should log warning about missing classes
        self.assertTrue(any("Could not load" in message and "required classes" in message 
                          for message in log.output))

    @patch('src.cursus.core.config_fields.config_class_detector.ConfigClassStore')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file')
    def test_from_config_store_success(self, mock_is_file, mock_file, mock_store):
        """Test from_config_store method success case."""
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = json.dumps(self.sample_config_data)
        
        # Mock ConfigClassStore
        mock_store.get_class.side_effect = lambda name: self.mock_complete_classes.get(name)
        mock_store.registered_names.return_value = list(self.mock_complete_classes.keys())
        
        result = ConfigClassDetector.from_config_store("test_config.json")
        
        # Should return classes from ConfigClassStore
        self.assertIn("MockConfigA", result)
        self.assertIn("MockConfigB", result)
        self.assertIn("BasePipelineConfig", result)

    @patch('src.cursus.core.config_fields.config_class_detector.ConfigClassStore')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file')
    def test_from_config_store_io_error(self, mock_is_file, mock_file, mock_store):
        """Test from_config_store method with IO error."""
        mock_is_file.return_value = True
        mock_file.side_effect = IOError("File read error")
        
        # Mock ConfigClassStore fallback
        mock_store.get_class.side_effect = lambda name: self.mock_complete_classes.get(name)
        mock_store.registered_names.return_value = list(self.mock_complete_classes.keys())
        
        with self.assertLogs(level='ERROR') as log:
            result = ConfigClassDetector.from_config_store("test_config.json")
        
        # Should fall back to all registered classes
        self.assertEqual(len(result), len(self.mock_complete_classes))
        self.assertTrue(any("Error reading or parsing configuration file" in message 
                          for message in log.output))

    @patch('src.cursus.core.config_fields.config_class_detector.ConfigClassStore')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.is_file')
    def test_from_config_store_no_class_names(self, mock_is_file, mock_file, mock_store):
        """Test from_config_store method when no class names found."""
        mock_is_file.return_value = True
        mock_file.return_value.read.return_value = json.dumps({})  # Empty config
        
        # Mock ConfigClassStore
        mock_store.get_class.side_effect = lambda name: self.mock_complete_classes.get(name)
        mock_store.registered_names.return_value = list(self.mock_complete_classes.keys())
        
        with self.assertLogs(level='WARNING') as log:
            result = ConfigClassDetector.from_config_store("empty_config.json")
        
        # Should return all registered classes
        self.assertEqual(len(result), len(self.mock_complete_classes))
        self.assertTrue(any("No config class names found" in message for message in log.output))

    def test_essential_classes_constant(self):
        """Test that essential classes constant is properly defined."""
        essential_classes = ConfigClassDetector.ESSENTIAL_CLASSES
        
        self.assertIn("BasePipelineConfig", essential_classes)
        self.assertIn("ProcessingStepConfigBase", essential_classes)
        self.assertIsInstance(essential_classes, list)

    def test_field_constants(self):
        """Test that field name constants are properly defined."""
        self.assertEqual(ConfigClassDetector.MODEL_TYPE_FIELD, "__model_type__")
        self.assertEqual(ConfigClassDetector.METADATA_FIELD, "metadata")
        self.assertEqual(ConfigClassDetector.CONFIG_TYPES_FIELD, "config_types")
        self.assertEqual(ConfigClassDetector.CONFIGURATION_FIELD, "configuration")
        self.assertEqual(ConfigClassDetector.SPECIFIC_FIELD, "specific")

    @patch('src.cursus.core.config_fields.config_class_detector.ConfigClassDetector.detect_from_json')
    def test_detect_config_classes_from_json_function(self, mock_detect):
        """Test the standalone detect_config_classes_from_json function."""
        mock_detect.return_value = self.mock_complete_classes
        
        result = detect_config_classes_from_json("test_config.json")
        
        mock_detect.assert_called_once_with("test_config.json")
        self.assertEqual(result, self.mock_complete_classes)

    def test_real_file_integration(self):
        """Test with a real temporary file to ensure file operations work."""
        # Create a temporary file with test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            json.dump(self.sample_config_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Mock the build_complete_config_classes function
            with patch('src.cursus.core.config_fields.config_class_detector.build_complete_config_classes') as mock_build:
                mock_build.return_value = self.mock_complete_classes
                
                # Test the actual file reading
                result = ConfigClassDetector.detect_from_json(temp_file_path)
                
                # Should successfully read and process the file
                self.assertIsInstance(result, dict)
                self.assertIn("MockConfigA", result)
                self.assertIn("BasePipelineConfig", result)
        
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)

    def test_edge_case_nested_specific_configs(self):
        """Test edge case with deeply nested specific configurations."""
        nested_config = {
            "configuration": {
                "specific": {
                    "step1": {
                        "__model_type__": "NestedConfig1",
                        "nested": {
                            "substep": {
                                "__model_type__": "SubConfig"  # This should not be extracted
                            }
                        }
                    }
                }
            }
        }
        
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names(nested_config, logger)
        
        # Should only extract top-level __model_type__ fields
        self.assertIn("NestedConfig1", result)
        self.assertNotIn("SubConfig", result)

    def test_edge_case_malformed_metadata(self):
        """Test edge case with malformed metadata section."""
        malformed_config = {
            "metadata": {
                "config_types": "not_a_dict"  # Should be a dict
            }
        }
        
        logger = MagicMock()
        
        # Should not crash and should return empty set
        try:
            result = ConfigClassDetector._extract_class_names(malformed_config, logger)
            self.assertEqual(len(result), 0)
        except Exception as e:
            # If it raises an exception, that's also acceptable behavior
            self.assertIsInstance(e, (TypeError, AttributeError))


if __name__ == '__main__':
    unittest.main()
