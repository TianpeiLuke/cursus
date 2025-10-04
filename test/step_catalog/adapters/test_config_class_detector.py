"""
Unit tests for step_catalog.adapters.config_class_detector module.

Tests the ConfigClassDetectorAdapter and ConfigClassStoreAdapter classes that provide
backward compatibility with legacy config class detection systems.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from cursus.step_catalog.adapters.config_class_detector import (
    ConfigClassDetectorAdapter,
    ConfigClassStoreAdapter,
    build_complete_config_classes,
    detect_config_classes_from_json
)


class TestConfigClassDetectorAdapter:
    """Test ConfigClassDetectorAdapter functionality."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_init_without_workspace(self):
        """Test initialization without workspace root."""
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog:
            adapter = ConfigClassDetectorAdapter()
            
            # Should initialize StepCatalog with workspace_dirs=None
            mock_catalog.assert_called_once_with(workspace_dirs=None)
            assert adapter.logger is not None
    
    def test_init_with_workspace(self, temp_workspace):
        """Test initialization with workspace root."""
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog:
            adapter = ConfigClassDetectorAdapter(workspace_root=temp_workspace)
            
            # Should initialize StepCatalog with workspace_dirs=[workspace_root]
            mock_catalog.assert_called_once_with(workspace_dirs=[temp_workspace])
            assert adapter.logger is not None
    
    def test_constants(self):
        """Test that backward compatibility constants are defined."""
        assert ConfigClassDetectorAdapter.MODEL_TYPE_FIELD == "__model_type__"
        assert ConfigClassDetectorAdapter.METADATA_FIELD == "metadata"
        assert ConfigClassDetectorAdapter.CONFIG_TYPES_FIELD == "config_types"
        assert ConfigClassDetectorAdapter.CONFIGURATION_FIELD == "configuration"
        assert ConfigClassDetectorAdapter.SPECIFIC_FIELD == "specific"
    
    def test_detect_from_json_success(self):
        """Test successful config class detection from JSON."""
        mock_config_classes = {
            "TestConfig": Mock(),
            "AnotherConfig": Mock()
        }
        
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.build_complete_config_classes.return_value = mock_config_classes
            mock_catalog_class.return_value = mock_catalog
            
            result = ConfigClassDetectorAdapter.detect_from_json("/test/config.json")
            
            # Should create StepCatalog with workspace_dirs=None
            mock_catalog_class.assert_called_once_with(workspace_dirs=None)
            # Should call build_complete_config_classes
            mock_catalog.build_complete_config_classes.assert_called_once()
            # Should return the config classes
            assert result == mock_config_classes
    
    def test_detect_from_json_error_handling(self):
        """Test error handling in detect_from_json."""
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Test error")
            
            result = ConfigClassDetectorAdapter.detect_from_json("/test/config.json")
            
            # Should return empty dict on error
            assert result == {}
    
    def test_extract_class_names_from_metadata(self):
        """Test extracting class names from metadata section."""
        config_data = {
            "metadata": {
                "config_types": {
                    "step1": "TestConfig1",
                    "step2": "TestConfig2"
                }
            }
        }
        
        mock_logger = Mock()
        result = ConfigClassDetectorAdapter._extract_class_names(config_data, mock_logger)
        
        assert result == {"TestConfig1", "TestConfig2"}
    
    def test_extract_class_names_from_configuration(self):
        """Test extracting class names from configuration.specific section."""
        config_data = {
            "configuration": {
                "specific": {
                    "step1": {
                        "__model_type__": "TestConfig1"
                    },
                    "step2": {
                        "__model_type__": "TestConfig2"
                    }
                }
            }
        }
        
        mock_logger = Mock()
        result = ConfigClassDetectorAdapter._extract_class_names(config_data, mock_logger)
        
        assert result == {"TestConfig1", "TestConfig2"}
    
    def test_extract_class_names_combined_sources(self):
        """Test extracting class names from both metadata and configuration sections."""
        config_data = {
            "metadata": {
                "config_types": {
                    "step1": "MetadataConfig"
                }
            },
            "configuration": {
                "specific": {
                    "step2": {
                        "__model_type__": "ConfigurationConfig"
                    }
                }
            }
        }
        
        mock_logger = Mock()
        result = ConfigClassDetectorAdapter._extract_class_names(config_data, mock_logger)
        
        assert result == {"MetadataConfig", "ConfigurationConfig"}
    
    def test_extract_class_names_empty_data(self):
        """Test extracting class names from empty data."""
        config_data = {}
        
        mock_logger = Mock()
        result = ConfigClassDetectorAdapter._extract_class_names(config_data, mock_logger)
        
        assert result == set()
    
    def test_extract_class_names_invalid_structure(self):
        """Test extracting class names from invalid data structure."""
        config_data = {
            "metadata": "invalid",  # Should be dict
            "configuration": {
                "specific": "also_invalid"  # Should be dict
            }
        }
        
        mock_logger = Mock()
        result = ConfigClassDetectorAdapter._extract_class_names(config_data, mock_logger)
        
        assert result == set()
    
    def test_extract_class_names_error_handling(self):
        """Test error handling in _extract_class_names."""
        # Create data that will cause an exception
        config_data = Mock()
        config_data.__getitem__.side_effect = Exception("Test error")
        
        mock_logger = Mock()
        result = ConfigClassDetectorAdapter._extract_class_names(config_data, mock_logger)
        
        assert result == set()
        mock_logger.error.assert_called_once()
    
    def test_from_config_store(self):
        """Test from_config_store class method."""
        mock_config_classes = {"TestConfig": Mock()}
        
        with patch.object(ConfigClassDetectorAdapter, 'detect_from_json') as mock_detect:
            mock_detect.return_value = mock_config_classes
            
            result = ConfigClassDetectorAdapter.from_config_store("/test/config.json")
            
            mock_detect.assert_called_once_with("/test/config.json")
            assert result == mock_config_classes


class TestConfigClassStoreAdapter:
    """Test ConfigClassStoreAdapter functionality."""
    
    def setup_method(self):
        """Clear registry before each test."""
        ConfigClassStoreAdapter.clear()
    
    def teardown_method(self):
        """Clear registry after each test."""
        ConfigClassStoreAdapter.clear()
    
    def test_register_class_directly(self):
        """Test registering a class directly."""
        class TestConfig:
            pass
        
        result = ConfigClassStoreAdapter.register(TestConfig)
        
        assert result == TestConfig
        assert ConfigClassStoreAdapter.get_class("TestConfig") == TestConfig
    
    def test_register_class_as_decorator(self):
        """Test registering a class using decorator pattern."""
        @ConfigClassStoreAdapter.register()
        class TestConfig:
            pass
        
        assert ConfigClassStoreAdapter.get_class("TestConfig") == TestConfig
    
    def test_register_class_overwrite_warning(self):
        """Test warning when overwriting existing class."""
        class TestConfig1:
            pass
        
        class TestConfig2:
            pass
        
        # Register first class
        ConfigClassStoreAdapter.register(TestConfig1)
        
        # Register second class with same name should log warning
        with patch.object(ConfigClassStoreAdapter._logger, 'warning') as mock_warning:
            # Manually set the name to simulate same class name
            TestConfig2.__name__ = "TestConfig1"
            ConfigClassStoreAdapter.register(TestConfig2)
            
            mock_warning.assert_called_once()
            assert "already registered and is being overwritten" in mock_warning.call_args[0][0]
    
    def test_get_class_existing(self):
        """Test getting an existing class."""
        class TestConfig:
            pass
        
        ConfigClassStoreAdapter.register(TestConfig)
        result = ConfigClassStoreAdapter.get_class("TestConfig")
        
        assert result == TestConfig
    
    def test_get_class_nonexistent(self):
        """Test getting a non-existent class."""
        result = ConfigClassStoreAdapter.get_class("NonExistentConfig")
        
        assert result is None
    
    def test_get_all_classes(self):
        """Test getting all registered classes."""
        class TestConfig1:
            pass
        
        class TestConfig2:
            pass
        
        ConfigClassStoreAdapter.register(TestConfig1)
        ConfigClassStoreAdapter.register(TestConfig2)
        
        result = ConfigClassStoreAdapter.get_all_classes()
        
        assert len(result) == 2
        assert result["TestConfig1"] == TestConfig1
        assert result["TestConfig2"] == TestConfig2
        
        # Should return a copy, not the original registry
        result["TestConfig3"] = Mock()
        assert "TestConfig3" not in ConfigClassStoreAdapter._registry
    
    def test_register_many(self):
        """Test registering multiple classes at once."""
        class TestConfig1:
            pass
        
        class TestConfig2:
            pass
        
        class TestConfig3:
            pass
        
        ConfigClassStoreAdapter.register_many(TestConfig1, TestConfig2, TestConfig3)
        
        assert ConfigClassStoreAdapter.get_class("TestConfig1") == TestConfig1
        assert ConfigClassStoreAdapter.get_class("TestConfig2") == TestConfig2
        assert ConfigClassStoreAdapter.get_class("TestConfig3") == TestConfig3
    
    def test_clear(self):
        """Test clearing the registry."""
        class TestConfig:
            pass
        
        ConfigClassStoreAdapter.register(TestConfig)
        assert len(ConfigClassStoreAdapter._registry) == 1
        
        ConfigClassStoreAdapter.clear()
        assert len(ConfigClassStoreAdapter._registry) == 0
        assert ConfigClassStoreAdapter.get_class("TestConfig") is None
    
    def test_registered_names(self):
        """Test getting all registered class names."""
        class TestConfig1:
            pass
        
        class TestConfig2:
            pass
        
        ConfigClassStoreAdapter.register(TestConfig1)
        ConfigClassStoreAdapter.register(TestConfig2)
        
        result = ConfigClassStoreAdapter.registered_names()
        
        assert result == {"TestConfig1", "TestConfig2"}


class TestLegacyFunctions:
    """Test legacy functions for backward compatibility."""
    
    def test_build_complete_config_classes_success(self):
        """Test successful build_complete_config_classes function."""
        mock_config_classes = {
            "TestConfig": Mock(),
            "AnotherConfig": Mock()
        }
        
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.build_complete_config_classes.return_value = mock_config_classes
            mock_catalog_class.return_value = mock_catalog
            
            result = build_complete_config_classes("test_project")
            
            # Should create StepCatalog with workspace_dirs=None
            mock_catalog_class.assert_called_once_with(workspace_dirs=None)
            # Should call build_complete_config_classes with project_id
            mock_catalog.build_complete_config_classes.assert_called_once_with("test_project")
            # Should return the config classes
            assert result == mock_config_classes
    
    def test_build_complete_config_classes_error_fallback(self):
        """Test fallback to ConfigClassStoreAdapter when StepCatalog fails."""
        # Setup some registered classes
        class TestConfig:
            pass
        
        ConfigClassStoreAdapter.register(TestConfig)
        
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Test error")
            
            result = build_complete_config_classes()
            
            # Should fallback to ConfigClassStoreAdapter
            assert "TestConfig" in result
            assert result["TestConfig"] == TestConfig
        
        # Cleanup
        ConfigClassStoreAdapter.clear()
    
    def test_detect_config_classes_from_json(self):
        """Test detect_config_classes_from_json function."""
        mock_config_classes = {"TestConfig": Mock()}
        
        with patch.object(ConfigClassDetectorAdapter, 'detect_from_json') as mock_detect:
            mock_detect.return_value = mock_config_classes
            
            result = detect_config_classes_from_json("/test/config.json")
            
            mock_detect.assert_called_once_with("/test/config.json")
            assert result == mock_config_classes


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_complete_workflow_with_step_catalog(self):
        """Test complete workflow using StepCatalog integration."""
        mock_config_classes = {
            "ProcessingConfig": Mock(),
            "TrainingConfig": Mock(),
            "ValidationConfig": Mock()
        }
        
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.build_complete_config_classes.return_value = mock_config_classes
            mock_catalog_class.return_value = mock_catalog
            
            # Test adapter initialization
            adapter = ConfigClassDetectorAdapter()
            
            # Test detection
            result = adapter.detect_from_json("/test/config.json")
            
            # Test legacy function
            legacy_result = detect_config_classes_from_json("/test/config.json")
            
            assert result == mock_config_classes
            assert legacy_result == mock_config_classes
    
    def test_mixed_registry_and_catalog_usage(self):
        """Test scenario mixing ConfigClassStoreAdapter and catalog usage."""
        # Register some classes manually
        class ManualConfig:
            pass
        
        ConfigClassStoreAdapter.register(ManualConfig)
        
        # Mock catalog discovery
        mock_catalog_classes = {"CatalogConfig": Mock()}
        
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.build_complete_config_classes.return_value = mock_catalog_classes
            mock_catalog_class.return_value = mock_catalog
            
            # Test catalog-based detection
            catalog_result = build_complete_config_classes()
            assert catalog_result == mock_catalog_classes
            
            # Test registry-based access
            registry_result = ConfigClassStoreAdapter.get_all_classes()
            assert "ManualConfig" in registry_result
            assert registry_result["ManualConfig"] == ManualConfig
        
        # Cleanup
        ConfigClassStoreAdapter.clear()
    
    def test_error_resilience_workflow(self):
        """Test error resilience in complete workflow."""
        # Register fallback classes
        class FallbackConfig:
            pass
        
        ConfigClassStoreAdapter.register(FallbackConfig)
        
        # Test with StepCatalog failure
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog_class.side_effect = Exception("Catalog initialization failed")
            
            # Detection should return empty dict
            result = ConfigClassDetectorAdapter.detect_from_json("/test/config.json")
            assert result == {}
            
            # Build complete should fallback to registry
            complete_result = build_complete_config_classes()
            assert "FallbackConfig" in complete_result
            assert complete_result["FallbackConfig"] == FallbackConfig
        
        # Cleanup
        ConfigClassStoreAdapter.clear()


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_adapter_with_none_workspace(self):
        """Test adapter behavior with None workspace."""
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog:
            adapter = ConfigClassDetectorAdapter(workspace_root=None)
            
            mock_catalog.assert_called_once_with(workspace_dirs=None)
    
    def test_detect_from_json_with_empty_path(self):
        """Test detect_from_json with empty path."""
        with patch('cursus.step_catalog.adapters.config_class_detector.StepCatalog') as mock_catalog_class:
            mock_catalog = Mock()
            mock_catalog.build_complete_config_classes.return_value = {}
            mock_catalog_class.return_value = mock_catalog
            
            result = ConfigClassDetectorAdapter.detect_from_json("")
            
            assert result == {}
    
    def test_registry_with_invalid_class(self):
        """Test registry behavior with invalid class objects."""
        # Test with None
        result = ConfigClassStoreAdapter.register(None)
        # Should return the decorator function
        assert callable(result)
        
        # Test with non-class object
        invalid_obj = "not_a_class"
        invalid_obj.__name__ = "InvalidObj"  # Mock __name__ attribute
        
        # This should still work as the registry doesn't validate class types
        ConfigClassStoreAdapter.register(invalid_obj)
        assert ConfigClassStoreAdapter.get_class("InvalidObj") == invalid_obj
        
        # Cleanup
        ConfigClassStoreAdapter.clear()
    
    def test_extract_class_names_with_nested_exceptions(self):
        """Test _extract_class_names with deeply nested data causing exceptions."""
        # Create mock data that raises exceptions at different levels
        config_data = {
            "metadata": {
                "config_types": Mock(side_effect=Exception("Nested error"))
            }
        }
        
        mock_logger = Mock()
        result = ConfigClassDetectorAdapter._extract_class_names(config_data, mock_logger)
        
        assert result == set()
        mock_logger.error.assert_called_once()
