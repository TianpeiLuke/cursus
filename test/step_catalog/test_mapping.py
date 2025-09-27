"""
Unit tests for step_catalog.mapping module.

Tests the StepCatalogMapping class that provides StepBuilderRegistry compatibility
and handles config-to-builder resolution, legacy aliases, and pipeline construction.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Type, Any

from cursus.step_catalog.mapping import StepCatalogMapper, PipelineConstructionInterface


class TestStepCatalogMapper:
    """Test StepCatalogMapper class functionality."""
    
    @pytest.fixture
    def mapping_instance(self):
        """Create StepCatalogMapper instance for testing."""
        mock_catalog = Mock()
        return StepCatalogMapper(mock_catalog)
    
    @pytest.fixture
    def mock_step_catalog(self):
        """Create mock StepCatalog for testing."""
        mock_catalog = Mock()
        mock_catalog.list_available_steps.return_value = [
            "BatchTransform", "XGBoostTraining", "PyTorchModel"
        ]
        mock_catalog.get_step_info.return_value = Mock(
            registry_data={"config_class": "TestConfig"}
        )
        return mock_catalog
    
    def test_initialization(self, mapping_instance):
        """Test StepCatalogMapping initialization."""
        assert mapping_instance is not None
        assert hasattr(mapping_instance, 'LEGACY_ALIASES')
        assert isinstance(mapping_instance.LEGACY_ALIASES, dict)
    
    def test_legacy_aliases_defined(self, mapping_instance):
        """Test that legacy aliases are properly defined."""
        aliases = mapping_instance.LEGACY_ALIASES
        
        # Should have some common legacy aliases
        assert len(aliases) > 0
        assert isinstance(aliases, dict)
        
        # Check some expected aliases exist (based on actual implementation)
        expected_aliases = [
            "MIMSPackaging", "MIMSPayload", "ModelRegistration", 
            "PytorchTraining", "PytorchModel"
        ]
        
        # At least some of these should be present
        found_aliases = [alias for alias in expected_aliases if alias in aliases]
        assert len(found_aliases) > 0
    
    def test_get_builder_for_config_with_valid_config(self, mapping_instance):
        """Test get_builder_for_config with valid config."""
        # Mock config class
        mock_config = Mock()
        mock_config.__class__.__name__ = "BatchTransformConfig"
        
        with patch.object(mapping_instance.step_catalog, 'load_builder_class') as mock_load:
            mock_builder = Mock()
            mock_load.return_value = mock_builder
            
            result = mapping_instance.get_builder_for_config(mock_config)
            
            assert result == mock_builder
    
    def test_get_builder_for_config_with_none(self, mapping_instance):
        """Test get_builder_for_config with None config."""
        with patch.object(mapping_instance.step_catalog, 'load_builder_class', return_value=None):
            result = mapping_instance.get_builder_for_config(None)
            assert result is None
    
    def test_get_builder_for_config_with_unknown_config(self, mapping_instance):
        """Test get_builder_for_config with unknown config class."""
        mock_config = Mock()
        mock_config.__class__.__name__ = "UnknownConfig"
        
        with patch.object(mapping_instance.step_catalog, 'load_builder_class', return_value=None):
            result = mapping_instance.get_builder_for_config(mock_config)
            assert result is None
    
    def test_get_builder_for_step_type_direct(self, mapping_instance):
        """Test get_builder_for_step_type with direct step type."""
        with patch.object(mapping_instance.step_catalog, 'load_builder_class') as mock_load:
            mock_builder = Mock()
            mock_load.return_value = mock_builder
            
            result = mapping_instance.get_builder_for_step_type("BatchTransform")
            
            assert result == mock_builder
    
    def test_get_builder_for_step_type_with_legacy_alias(self, mapping_instance):
        """Test get_builder_for_step_type with legacy alias."""
        # Use existing legacy alias
        with patch.object(mapping_instance.step_catalog, 'load_builder_class') as mock_load:
            mock_builder = Mock()
            mock_load.return_value = mock_builder
            
            result = mapping_instance.get_builder_for_step_type("MIMSPackaging")
            
            assert result == mock_builder
    
    def test_get_builder_for_step_type_with_none(self, mapping_instance):
        """Test get_builder_for_step_type with None."""
        with patch.object(mapping_instance.step_catalog, 'load_builder_class', return_value=None):
            result = mapping_instance.get_builder_for_step_type(None)
            assert result is None
    
    def test_get_builder_for_step_type_with_empty_string(self, mapping_instance):
        """Test get_builder_for_step_type with empty string."""
        with patch.object(mapping_instance.step_catalog, 'load_builder_class', return_value=None):
            result = mapping_instance.get_builder_for_step_type("")
            assert result is None
    
    def test_fallback_config_to_step_type(self, mapping_instance):
        """Test _fallback_config_to_step_type method."""
        # Test standard config name transformation
        assert mapping_instance._fallback_config_to_step_type("BatchTransformConfig") == "BatchTransform"
        assert mapping_instance._fallback_config_to_step_type("XGBoostTrainingConfig") == "XGBoostTraining"
        assert mapping_instance._fallback_config_to_step_type("PyTorchModelConfig") == "PyTorchModel"
        
        # Test edge cases - based on actual implementation behavior
        assert mapping_instance._fallback_config_to_step_type("Config") == ""
        assert mapping_instance._fallback_config_to_step_type("SomeStepConfig") == "Some"  # Removes both "Step" and "Config"
        assert mapping_instance._fallback_config_to_step_type("NoConfigSuffix") == "NoConfigSuffix"
    
    def test_legacy_alias_resolution(self, mapping_instance):
        """Test legacy alias resolution functionality."""
        # Test that legacy aliases are properly resolved
        assert mapping_instance.resolve_legacy_aliases("MIMSPackaging") == "Package"
        assert mapping_instance.resolve_legacy_aliases("MIMSPayload") == "Payload"
        assert mapping_instance.resolve_legacy_aliases("ModelRegistration") == "Registration"
        assert mapping_instance.resolve_legacy_aliases("PytorchTraining") == "PyTorchTraining"
        assert mapping_instance.resolve_legacy_aliases("PytorchModel") == "PyTorchModel"
        
        # Test that non-legacy names are returned unchanged
        assert mapping_instance.resolve_legacy_aliases("BatchTransform") == "BatchTransform"
        assert mapping_instance.resolve_legacy_aliases("UnknownStep") == "UnknownStep"
    
    def test_is_step_type_supported(self, mapping_instance):
        """Test is_step_type_supported method."""
        # Mock the step catalog's _ensure_index_built and _step_index
        mapping_instance.step_catalog._ensure_index_built = Mock()
        mapping_instance.step_catalog._step_index = {"BatchTransform": Mock(), "XGBoostTraining": Mock()}
        
        # Test supported step types
        assert mapping_instance.is_step_type_supported("BatchTransform") == True
        assert mapping_instance.is_step_type_supported("XGBoostTraining") == True
        
        # Test unsupported step types
        assert mapping_instance.is_step_type_supported("UnsupportedStep") == False
        
        # Test legacy aliases
        assert mapping_instance.is_step_type_supported("MIMSPackaging") == False  # Package not in mock index
    
    def test_validate_builder_availability(self, mapping_instance):
        """Test validate_builder_availability method."""
        step_types = ["BatchTransform", "XGBoostTraining", "UnsupportedStep"]
        
        # Mock get_builder_for_step_type to return builders for some steps
        def mock_get_builder(step_type):
            if step_type in ["BatchTransform", "XGBoostTraining"]:
                return Mock()
            return None
        
        with patch.object(mapping_instance, 'get_builder_for_step_type', side_effect=mock_get_builder):
            result = mapping_instance.validate_builder_availability(step_types)
            
            assert isinstance(result, dict)
            assert result["BatchTransform"] == True
            assert result["XGBoostTraining"] == True
            assert result["UnsupportedStep"] == False


class TestPipelineConstructionInterface:
    """Test PipelineConstructionInterface class functionality."""
    
    @pytest.fixture
    def pipeline_interface(self):
        """Create PipelineConstructionInterface instance for testing."""
        mock_step_catalog = Mock()
        mock_mapper = StepCatalogMapper(mock_step_catalog)
        return PipelineConstructionInterface(mock_mapper)
    
    @pytest.fixture
    def mock_step_catalog_with_data(self):
        """Create mock StepCatalog with test data."""
        mock_catalog = Mock()
        
        # Mock step data
        mock_catalog.list_available_steps.return_value = [
            "BatchTransform", "XGBoostTraining", "PyTorchModel"
        ]
        
        # Mock step info
        def mock_get_step_info(step_name):
            if step_name == "BatchTransform":
                return Mock(registry_data={"config_class": "BatchTransformConfig"})
            elif step_name == "XGBoostTraining":
                return Mock(registry_data={"config_class": "XGBoostTrainingConfig"})
            elif step_name == "PyTorchModel":
                return Mock(registry_data={"config_class": "PyTorchModelConfig"})
            return None
        
        mock_catalog.get_step_info.side_effect = mock_get_step_info
        
        return mock_catalog
    
    def test_initialization(self, pipeline_interface):
        """Test PipelineConstructionInterface initialization."""
        assert pipeline_interface is not None
        assert hasattr(pipeline_interface, 'mapper')
        assert isinstance(pipeline_interface.mapper, StepCatalogMapper)
    
    def test_get_builder_map(self, pipeline_interface):
        """Test get_builder_map method."""
        # Mock the mapper's list_supported_step_types method
        with patch.object(pipeline_interface.mapper, 'list_supported_step_types', return_value=["BatchTransform", "XGBoostTraining"]):
            with patch.object(pipeline_interface.mapper, 'get_builder_for_step_type') as mock_get_builder:
                mock_builder = Mock()
                mock_get_builder.return_value = mock_builder
                
                result = pipeline_interface.get_builder_map()
                
                assert isinstance(result, dict)
                assert "BatchTransform" in result
                assert "XGBoostTraining" in result
                
                # All should map to the mock builder
                for step_type, builder in result.items():
                    assert builder == mock_builder
    
    def test_get_builder_map_with_none_builders(self, pipeline_interface):
        """Test get_builder_map when some builders are None."""
        # Mock the mapper's list_supported_step_types method
        with patch.object(pipeline_interface.mapper, 'list_supported_step_types', return_value=["BatchTransform", "XGBoostTraining"]):
            # Mock builder resolution to return None for some steps
            def mock_get_builder(step_type):
                if step_type == "BatchTransform":
                    return Mock()
                return None
            
            with patch.object(pipeline_interface.mapper, 'get_builder_for_step_type', side_effect=mock_get_builder):
                result = pipeline_interface.get_builder_map()
                
                assert isinstance(result, dict)
                # Should only include steps with valid builders
                assert "BatchTransform" in result
                assert "XGBoostTraining" not in result
    
    def test_validate_dag_compatibility(self, pipeline_interface):
        """Test validate_dag_compatibility method."""
        step_types = ["BatchTransform", "XGBoostTraining", "UnsupportedStep"]
        
        # Mock the mapper's validate_builder_availability method
        mock_availability = {
            "BatchTransform": True,
            "XGBoostTraining": True,
            "UnsupportedStep": False
        }
        
        with patch.object(pipeline_interface.mapper, 'validate_builder_availability', return_value=mock_availability):
            result = pipeline_interface.validate_dag_compatibility(step_types)
            
            assert isinstance(result, dict)
            assert result['compatible'] == False  # Because UnsupportedStep is missing
            assert result['missing_builders'] == ["UnsupportedStep"]
            assert result['available_builders'] == ["BatchTransform", "XGBoostTraining"]
            assert result['total_steps'] == 3
            assert result['supported_steps'] == 2
    
    def test_get_step_builder_suggestions(self, pipeline_interface):
        """Test get_step_builder_suggestions method."""
        config_class_name = "BatchTransformConfig"
        
        # Mock the mapper's list_supported_step_types method
        with patch.object(pipeline_interface.mapper, 'list_supported_step_types', return_value=["BatchTransform", "XGBoostTraining"]):
            result = pipeline_interface.get_step_builder_suggestions(config_class_name)
            
            assert isinstance(result, list)
            assert len(result) <= 5  # Should limit to top 5 suggestions
            # Should find BatchTransform as it matches the config name pattern
            assert "BatchTransform" in result


class TestIntegrationScenarios:
    """Test integration scenarios between mapping components."""
    
    def test_full_config_to_builder_resolution_flow(self):
        """Test complete flow from config to builder resolution."""
        mock_catalog = Mock()
        mapping = StepCatalogMapper(mock_catalog)
        
        # Mock config
        mock_config = Mock()
        mock_config.__class__.__name__ = "BatchTransformConfig"
        
        # Mock builder discovery
        mock_builder = Mock()
        mock_builder.__name__ = "BatchTransformStepBuilder"
        
        with patch.object(mapping.step_catalog, 'load_builder_class', return_value=mock_builder):
            result = mapping.get_builder_for_config(mock_config)
            
            assert result == mock_builder
    
    def test_legacy_alias_resolution_flow(self):
        """Test complete flow with legacy alias resolution."""
        mock_catalog = Mock()
        mapping = StepCatalogMapper(mock_catalog)
        
        # Mock builder discovery
        mock_builder = Mock()
        mock_builder.__name__ = "PackageStepBuilder"
        
        with patch.object(mapping.step_catalog, 'load_builder_class', return_value=mock_builder):
            result = mapping.get_builder_for_step_type("MIMSPackaging")  # Uses existing legacy alias
            
            assert result == mock_builder
    
    def test_pipeline_interface_with_mapping_integration(self):
        """Test PipelineConstructionInterface integration with StepCatalogMapper."""
        mock_catalog = Mock()
        mock_catalog.list_available_steps.return_value = ["TestStep"]
        mock_catalog.get_step_info.return_value = Mock(
            registry_data={"config_class": "TestConfig"}
        )
        
        mapper = StepCatalogMapper(mock_catalog)
        interface = PipelineConstructionInterface(mapper)
        
        # Test that mapper is properly integrated
        assert interface.mapper is not None
        assert isinstance(interface.mapper, StepCatalogMapper)
        
        # Test delegation to mapper
        with patch.object(interface.mapper, 'get_builder_for_step_type') as mock_method:
            interface.mapper.get_builder_for_step_type("TestStep")
            mock_method.assert_called_once_with("TestStep")


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""
    
    def test_mapping_with_malformed_config_names(self):
        """Test mapping handles malformed config class names gracefully."""
        mock_catalog = Mock()
        mapping = StepCatalogMapper(mock_catalog)
        
        # Test various malformed names - based on actual implementation behavior
        test_cases = [
            ("", ""),
            ("Config", ""),
            ("ConfigConfig", "Config"),
            ("StepConfig", ""),  # Removes both "Step" and "Config", leaving empty string
            ("VeryLongConfigNameThatShouldStillWork", "VeryLongConfigNameThatShouldStillWork")
        ]
        
        for input_name, expected_output in test_cases:
            result = mapping._fallback_config_to_step_type(input_name)
            assert result == expected_output
    
    def test_pipeline_interface_with_catalog_errors(self):
        """Test PipelineConstructionInterface handles catalog errors gracefully."""
        mock_catalog = Mock()
        mock_catalog.list_available_steps.side_effect = Exception("Catalog error")
        
        mapper = StepCatalogMapper(mock_catalog)
        interface = PipelineConstructionInterface(mapper)
        
        # Should handle errors gracefully
        result = interface.get_builder_map()
        assert isinstance(result, dict)
        assert len(result) == 0  # Should return empty dict on error
    
    def test_builder_resolution_with_import_errors(self):
        """Test builder resolution handles import errors gracefully."""
        mock_catalog = Mock()
        mapping = StepCatalogMapper(mock_catalog)
        
        with patch.object(mapping.step_catalog, 'load_builder_class', 
                         side_effect=ImportError("Module not found")):
            result = mapping.get_builder_for_step_type("TestStep")
            assert result is None
    
    def test_mapping_thread_safety(self):
        """Test that mapping operations are thread-safe."""
        import threading
        import time
        
        mock_catalog = Mock()
        mapping = StepCatalogMapper(mock_catalog)
        results = []
        errors = []
        
        def worker():
            try:
                for i in range(10):
                    # Simulate concurrent access
                    result = mapping._fallback_config_to_step_type(f"TestConfig{i}")
                    results.append(result)
                    time.sleep(0.001)  # Small delay to increase chance of race conditions
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors and expected number of results
        assert len(errors) == 0
        assert len(results) == 50  # 5 threads * 10 operations each


class TestPerformanceCharacteristics:
    """Test performance characteristics of mapping operations."""
    
    def test_large_scale_step_type_resolution(self):
        """Test performance with large number of step types."""
        import time
        
        mock_catalog = Mock()
        mapping = StepCatalogMapper(mock_catalog)
        
        # Add many legacy aliases
        for i in range(1000):
            mapping.LEGACY_ALIASES[f"LegacyStep{i}"] = f"ModernStep{i}"
        
        # Mock the step catalog to return None (simulating no builder found)
        with patch.object(mapping.step_catalog, 'load_builder_class', return_value=None):
            # Test resolution performance
            start_time = time.time()
            
            for i in range(100):
                step_name = f"LegacyStep{i % 1000}"
                mapping.get_builder_for_step_type(step_name)
            
            end_time = time.time()
            
            # Should complete quickly even with many aliases
            assert (end_time - start_time) < 0.5  # Less than 500ms
    
    def test_config_name_transformation_performance(self):
        """Test performance of config name transformations."""
        import time
        
        mock_catalog = Mock()
        mapping = StepCatalogMapper(mock_catalog)
        
        # Test many config name transformations
        config_names = [f"TestConfig{i}" for i in range(1000)]
        
        start_time = time.time()
        
        for config_name in config_names:
            mapping._fallback_config_to_step_type(config_name)
        
        end_time = time.time()
        
        # Should complete quickly
        assert (end_time - start_time) < 0.1  # Less than 100ms
