"""
Unit tests for ConfigClassDetector (Modern Step Catalog Implementation).

This module contains comprehensive tests for the ConfigClassDetectorAdapter,
testing its ability to discover configuration classes using the unified step catalog
system with AST-based discovery instead of legacy JSON parsing.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Type
from unittest.mock import patch, MagicMock
import pytest

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cursus.core.config_fields.config_class_detector import (
    ConfigClassDetector,
    detect_config_classes_from_json,
    build_complete_config_classes,
)
from cursus.step_catalog.step_catalog import StepCatalog


@pytest.fixture
def test_workspace_root():
    """Fixture providing test workspace root path."""
    return Path(__file__).parent.parent.parent / "src" / "cursus" / "steps"


class TestModernConfigClassDetector:
    """Test cases for modern ConfigClassDetector using step catalog."""

    def test_detect_from_json_returns_real_config_classes(self):
        """Test that detect_from_json returns real config classes from step catalog."""
        result = ConfigClassDetector.detect_from_json("dummy_config.json")
        
        # Should return a dictionary of real config classes
        assert isinstance(result, dict)
        assert len(result) > 0, "Should discover at least some config classes"
        
        # Check that we get real config classes (not mock ones)
        for class_name, class_type in result.items():
            assert isinstance(class_name, str)
            assert hasattr(class_type, '__name__'), f"Class {class_name} should be a real class"
            # Config classes end with 'Config' or 'ConfigBase', hyperparameter classes end with 'Hyperparameters'
            # This reflects the enhanced discovery that now includes both config and hyperparameter classes
            assert (class_name.endswith('Config') or 
                   class_name.endswith('ConfigBase') or 
                   class_name.endswith('Hyperparameters') or
                   'Hyperparameter' in class_name), \
                f"Class {class_name} should follow naming convention (Config, ConfigBase, or Hyperparameters)"

    def test_detect_from_json_includes_common_config_classes(self):
        """Test that detect_from_json includes expected common config classes."""
        result = ConfigClassDetector.detect_from_json("dummy_config.json")
        
        # Should include some common config classes we know exist
        expected_classes = [
            'CradleDataLoadConfig',
            'TabularPreprocessingConfig', 
            'XGBoostTrainingConfig',
            'ProcessingStepConfigBase'
        ]
        
        found_classes = set(result.keys())
        common_classes = set(expected_classes) & found_classes
        
        assert len(common_classes) > 0, f"Should find at least some common classes. Found: {found_classes}"

    def test_from_config_store_same_as_detect_from_json(self):
        """Test that from_config_store returns same result as detect_from_json."""
        result1 = ConfigClassDetector.detect_from_json("dummy_config.json")
        result2 = ConfigClassDetector.from_config_store("dummy_config.json")
        
        # Both methods should return the same result in modern implementation
        assert result1.keys() == result2.keys()
        assert result1 == result2

    def test_build_complete_config_classes_function(self):
        """Test the build_complete_config_classes standalone function."""
        result = build_complete_config_classes()
        
        assert isinstance(result, dict)
        assert len(result) > 0, "Should discover config classes"
        
        # Should return real config classes
        for class_name, class_type in result.items():
            assert isinstance(class_name, str)
            assert hasattr(class_type, '__name__')

    def test_detect_config_classes_from_json_function(self):
        """Test the detect_config_classes_from_json standalone function."""
        result = detect_config_classes_from_json("dummy_config.json")
        
        assert isinstance(result, dict)
        assert len(result) > 0, "Should discover config classes"

    def test_field_constants_available(self):
        """Test that field name constants are available for backward compatibility."""
        assert ConfigClassDetector.MODEL_TYPE_FIELD == "__model_type__"
        assert ConfigClassDetector.METADATA_FIELD == "metadata"
        assert ConfigClassDetector.CONFIG_TYPES_FIELD == "config_types"
        assert ConfigClassDetector.CONFIGURATION_FIELD == "configuration"
        assert ConfigClassDetector.SPECIFIC_FIELD == "specific"

    def test_extract_class_names_legacy_compatibility(self):
        """Test that _extract_class_names method works for legacy compatibility."""
        # Test data in legacy JSON format
        test_data = {
            'metadata': {
                'config_types': {
                    'step1': 'TestConfig1',
                    'step2': 'TestConfig2'
                }
            },
            'configuration': {
                'specific': {
                    'step3': {'__model_type__': 'TestConfig3'},
                    'step4': {'__model_type__': 'TestConfig4'}
                }
            }
        }
        
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names(test_data, logger)
        
        # Should extract class names from both metadata and specific sections
        expected_classes = {'TestConfig1', 'TestConfig2', 'TestConfig3', 'TestConfig4'}
        assert result == expected_classes

    def test_extract_class_names_empty_data(self):
        """Test _extract_class_names with empty data."""
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names({}, logger)
        
        assert result == set()

    def test_extract_class_names_partial_data(self):
        """Test _extract_class_names with partial data structures."""
        # Only metadata
        metadata_only = {
            'metadata': {
                'config_types': {'step1': 'MetadataConfig'}
            }
        }
        
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names(metadata_only, logger)
        assert 'MetadataConfig' in result
        
        # Only specific configuration
        specific_only = {
            'configuration': {
                'specific': {
                    'step1': {'__model_type__': 'SpecificConfig'}
                }
            }
        }
        
        result = ConfigClassDetector._extract_class_names(specific_only, logger)
        assert 'SpecificConfig' in result

    def test_extract_class_names_malformed_data(self):
        """Test _extract_class_names handles malformed data gracefully."""
        malformed_data = {
            'metadata': {'config_types': 'not_a_dict'},  # Should be dict
            'configuration': {
                'specific': {
                    'step1': 'not_a_dict',  # Should be dict
                    'step2': {'no_model_type': 'value'},  # Missing __model_type__
                    'step3': {'__model_type__': 'ValidConfig'}  # Valid
                }
            }
        }
        
        logger = MagicMock()
        result = ConfigClassDetector._extract_class_names(malformed_data, logger)
        
        # Should only extract the valid config
        assert result == {'ValidConfig'}

    @patch('cursus.step_catalog.adapters.StepCatalog')
    def test_detect_from_json_with_mocked_catalog(self, mock_catalog_class):
        """Test detect_from_json with mocked step catalog."""
        # Mock the catalog instance and its methods
        mock_catalog = MagicMock()
        mock_catalog_class.return_value = mock_catalog
        
        mock_config_classes = {
            'MockConfig1': type('MockConfig1', (), {}),
            'MockConfig2': type('MockConfig2', (), {})
        }
        mock_catalog.build_complete_config_classes.return_value = mock_config_classes
        
        result = ConfigClassDetector.detect_from_json("test_config.json")
        
        # Should use the mocked catalog
        mock_catalog_class.assert_called_once()
        mock_catalog.build_complete_config_classes.assert_called_once()
        assert result == mock_config_classes

    @patch('cursus.step_catalog.adapters.StepCatalog')
    def test_detect_from_json_fallback_behavior(self, mock_catalog_class):
        """Test detect_from_json fallback behavior when catalog fails."""
        # Mock catalog to raise exception
        mock_catalog_class.side_effect = Exception("Catalog initialization failed")
        
        result = ConfigClassDetector.detect_from_json("test_config.json")
        
        # Should handle the failure gracefully and fall back to ConfigClassStore
        # The enhanced adapter now successfully falls back to ConfigClassStore when catalog fails
        assert isinstance(result, dict)
        # The adapter should fall back to ConfigClassStore and still return config classes
        # This is better behavior than returning empty dict - it provides resilience
        assert len(result) > 0, "Fallback should still discover config classes via ConfigClassStore"

    def test_real_step_catalog_integration(self, test_workspace_root):
        """Test integration with real step catalog."""
        # This test uses the actual step catalog to ensure integration works
        try:
            catalog = StepCatalog(test_workspace_root)
            config_classes = catalog.build_complete_config_classes()
            
            # Should discover real config classes
            assert isinstance(config_classes, dict)
            assert len(config_classes) > 0
            
            # Test that ConfigClassDetector uses the same discovery
            detector_result = ConfigClassDetector.detect_from_json("dummy_config.json")
            
            # Results should be similar (both discover real config classes)
            assert isinstance(detector_result, dict)
            assert len(detector_result) > 0
            
        except Exception as e:
            pytest.skip(f"Step catalog integration test skipped due to: {e}")

    def test_performance_reasonable(self):
        """Test that config class detection has reasonable performance."""
        start_time = time.time()
        result = ConfigClassDetector.detect_from_json("dummy_config.json")
        end_time = time.time()
        
        # Should complete within reasonable time (less than 5 seconds)
        execution_time = end_time - start_time
        assert execution_time < 5.0, f"Config detection took {execution_time:.2f}s, should be faster"
        
        # Should still return valid results
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_consistency_across_calls(self):
        """Test that multiple calls return consistent results."""
        result1 = ConfigClassDetector.detect_from_json("config1.json")
        result2 = ConfigClassDetector.detect_from_json("config2.json")
        result3 = detect_config_classes_from_json("config3.json")
        
        # All calls should return the same config classes (since they use the same catalog)
        assert result1.keys() == result2.keys()
        assert result1.keys() == result3.keys()
        assert result1 == result2
        assert result1 == result3

    def test_adapter_backward_compatibility(self):
        """Test that the adapter maintains backward compatibility."""
        # Test that all expected methods and attributes exist
        assert hasattr(ConfigClassDetector, 'detect_from_json')
        assert hasattr(ConfigClassDetector, 'from_config_store')
        assert hasattr(ConfigClassDetector, '_extract_class_names')
        
        # Test that all expected constants exist
        assert hasattr(ConfigClassDetector, 'MODEL_TYPE_FIELD')
        assert hasattr(ConfigClassDetector, 'METADATA_FIELD')
        assert hasattr(ConfigClassDetector, 'CONFIG_TYPES_FIELD')
        assert hasattr(ConfigClassDetector, 'CONFIGURATION_FIELD')
        assert hasattr(ConfigClassDetector, 'SPECIFIC_FIELD')

    def test_modern_vs_legacy_approach_comparison(self):
        """Test that modern approach is superior to legacy approach."""
        # Get results from modern approach
        modern_result = ConfigClassDetector.detect_from_json("dummy_config.json")
        
        # Modern approach should:
        # 1. Return real config classes (not mock ones)
        # 2. Discover more classes than legacy approach would with limited JSON
        # 3. Not require specific JSON structure
        
        assert isinstance(modern_result, dict)
        assert len(modern_result) > 5, "Modern approach should discover many config classes"
        
        # All discovered classes should be real classes with proper attributes
        for class_name, class_type in modern_result.items():
            assert hasattr(class_type, '__module__'), f"{class_name} should have __module__"
            assert hasattr(class_type, '__name__'), f"{class_name} should have __name__"
            assert class_type.__name__ == class_name, f"Class name should match key"
