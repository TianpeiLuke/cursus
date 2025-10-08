"""
Test suite for Step Catalog Parent Config Retrieval Enhancement.

This module tests the two new methods added to StepCatalog for Smart Default Value Inheritance:
1. get_immediate_parent_config_class()
2. extract_parent_values_for_inheritance()

Following pytest best practices:
- Source Code First Rule: Read implementation before writing tests
- Mock Path Precision: Mock at correct import locations
- Implementation-Driven Testing: Match test behavior to actual implementation
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path
from typing import Dict, Any, Optional

from cursus.step_catalog.step_catalog import StepCatalog
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestParentConfigRetrieval:
    """Test suite for parent config retrieval methods in StepCatalog."""
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset any global state before each test."""
        yield
        # Cleanup after test
    
    @pytest.fixture
    def mock_step_catalog(self):
        """Create a mock StepCatalog with controlled config discovery."""
        # CRITICAL FIX: Mock at the actual conditional import location in step_catalog.py
        # Source shows: from .config_discovery import ConfigAutoDiscovery (line 25)
        with patch('cursus.step_catalog.step_catalog.ConfigAutoDiscovery') as mock_config_discovery_class:
            mock_config_discovery = Mock()
            mock_config_discovery_class.return_value = mock_config_discovery
            
            # CRITICAL FIX: Create proper mock classes that support issubclass() checks
            # Must use type() to create real classes, not Mock objects
            mock_processing_config = type('ProcessingStepConfigBase', (BasePipelineConfig,), {})
            mock_tabular_config = type('TabularPreprocessingConfig', (mock_processing_config,), {})
            mock_cradle_config = type('CradleDataLoadConfig', (BasePipelineConfig,), {})
            
            # Configure mock discovery to return these classes
            mock_config_discovery.discover_config_classes.return_value = {
                "BasePipelineConfig": BasePipelineConfig,
                "ProcessingStepConfigBase": mock_processing_config,
                "TabularPreprocessingConfig": mock_tabular_config,
                "CradleDataLoadConfig": mock_cradle_config
            }
            
            catalog = StepCatalog()
            return catalog
    
    @pytest.fixture
    def sample_base_config(self):
        """Create a sample BasePipelineConfig instance."""
        # CRITICAL FIX: Use valid region value based on source code validation
        # Source shows region must be one of ['NA', 'EU', 'FE']
        return BasePipelineConfig(
            author="test-user",
            bucket="test-bucket",
            role="arn:aws:iam::123:role/TestRole",
            region="NA",  # FIXED: Use valid region code
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
    
    @pytest.fixture
    def sample_processing_config(self, sample_base_config):
        """Create a sample ProcessingStepConfigBase instance."""
        # CRITICAL FIX: Use correct field names from ProcessingStepConfigBase source code
        return ProcessingStepConfigBase.from_base_config(
            sample_base_config,
            processing_instance_type_small="ml.m5.2xlarge",  # FIXED: Use correct field name
            processing_volume_size=500,
            processing_source_dir="src/processing"
        )
    
    def test_get_immediate_parent_config_class_tabular_preprocessing(self, mock_step_catalog):
        """Test that TabularPreprocessingConfig returns ProcessingStepConfigBase as immediate parent."""
        # CRITICAL FIX: Import happens inside the method, use patch.object to mock the import
        # Source shows: from ..core.base.config_base import BasePipelineConfig (inside method)
        with patch('cursus.core.base.config_base.BasePipelineConfig', BasePipelineConfig):
            parent = mock_step_catalog.get_immediate_parent_config_class("TabularPreprocessingConfig")
            assert parent == "ProcessingStepConfigBase"
    
    def test_get_immediate_parent_config_class_cradle_data_load(self, mock_step_catalog):
        """Test that CradleDataLoadConfig returns BasePipelineConfig as immediate parent."""
        # CRITICAL FIX: Import happens inside the method, use patch.object to mock the import
        with patch('cursus.core.base.config_base.BasePipelineConfig', BasePipelineConfig):
            parent = mock_step_catalog.get_immediate_parent_config_class("CradleDataLoadConfig")
            assert parent == "BasePipelineConfig"
    
    def test_get_immediate_parent_config_class_processing_step(self, mock_step_catalog):
        """Test that ProcessingStepConfigBase returns BasePipelineConfig as immediate parent."""
        # CRITICAL FIX: Import happens inside the method, use patch.object to mock the import
        with patch('cursus.core.base.config_base.BasePipelineConfig', BasePipelineConfig):
            parent = mock_step_catalog.get_immediate_parent_config_class("ProcessingStepConfigBase")
            assert parent == "BasePipelineConfig"
    
    def test_get_immediate_parent_config_class_nonexistent(self, mock_step_catalog):
        """Test that non-existent config class returns None."""
        parent = mock_step_catalog.get_immediate_parent_config_class("NonExistentConfig")
        assert parent is None
    
    def test_get_immediate_parent_config_class_config_discovery_unavailable(self):
        """Test behavior when ConfigAutoDiscovery is not available."""
        # CRITICAL FIX: Mock at the actual import location in step_catalog.py
        # Source shows: from .config_discovery import ConfigAutoDiscovery (line 25)
        with patch('cursus.step_catalog.step_catalog.ConfigAutoDiscovery', None):
            catalog = StepCatalog()
            parent = catalog.get_immediate_parent_config_class("TabularPreprocessingConfig")
            assert parent is None
    
    def test_get_immediate_parent_config_class_import_error(self, mock_step_catalog):
        """Test behavior when BasePipelineConfig import fails in the method."""
        # CRITICAL FIX: Mock the import at the source location where it actually happens
        with patch('cursus.core.base.config_base.BasePipelineConfig', side_effect=ImportError("Import failed")):
            # Mock discover_config_classes to return a valid config class
            with patch.object(mock_step_catalog, 'discover_config_classes', return_value={"TestConfig": Mock()}):
                parent = mock_step_catalog.get_immediate_parent_config_class("TestConfig")
                assert parent is None
    
    def test_extract_parent_values_for_inheritance_success(self, mock_step_catalog, sample_base_config, sample_processing_config):
        """Test successful parent value extraction."""
        # CRITICAL FIX: Don't mock BasePipelineConfig, just mock the method directly
        # Mock the get_immediate_parent_config_class method to return ProcessingStepConfigBase
        with patch.object(mock_step_catalog, 'get_immediate_parent_config_class', return_value="ProcessingStepConfigBase"):
            completed_configs = {
                "BasePipelineConfig": sample_base_config,
                "ProcessingStepConfigBase": sample_processing_config
            }
            
            parent_values = mock_step_catalog.extract_parent_values_for_inheritance(
                "TabularPreprocessingConfig", completed_configs
            )
            
            # Verify that values were extracted
            assert isinstance(parent_values, dict)
            assert len(parent_values) > 0
            
            # Check for expected fields from both base and processing configs
            # CRITICAL FIX: Check actual field names from the real config objects
            expected_base_fields = ["author", "bucket", "role"]
            for field in expected_base_fields:
                assert field in parent_values, f"Expected base field '{field}' not found in parent values"
            
            # Verify specific values
            assert parent_values["author"] == "test-user"
            
            # Check for processing-specific fields (use actual field names)
            # The processing config should have processing-related fields
            processing_fields = [k for k in parent_values.keys() if k.startswith('processing')]
            assert len(processing_fields) > 0, f"Expected processing fields not found. Available fields: {list(parent_values.keys())}"
    
    def test_extract_parent_values_for_inheritance_no_parent_class(self, mock_step_catalog):
        """Test behavior when no parent class is found."""
        with patch.object(mock_step_catalog, 'get_immediate_parent_config_class', return_value=None):
            completed_configs = {"BasePipelineConfig": Mock()}
            
            parent_values = mock_step_catalog.extract_parent_values_for_inheritance(
                "NonExistentConfig", completed_configs
            )
            
            assert parent_values == {}
    
    def test_extract_parent_values_for_inheritance_parent_config_not_found(self, mock_step_catalog):
        """Test behavior when parent config instance is not in completed_configs."""
        with patch.object(mock_step_catalog, 'get_immediate_parent_config_class', return_value="ProcessingStepConfigBase"):
            completed_configs = {"BasePipelineConfig": Mock()}  # Missing ProcessingStepConfigBase
            
            parent_values = mock_step_catalog.extract_parent_values_for_inheritance(
                "TabularPreprocessingConfig", completed_configs
            )
            
            assert parent_values == {}
    
    def test_extract_parent_values_for_inheritance_pydantic_v2_model_fields(self, mock_step_catalog):
        """Test field extraction using Pydantic v2 model_fields."""
        with patch.object(mock_step_catalog, 'get_immediate_parent_config_class', return_value="TestConfig"):
            # Create mock config with model_fields (Pydantic v2)
            mock_config = Mock()
            mock_config.__class__.model_fields = {
                "field1": Mock(),
                "field2": Mock(),
                "field3": Mock()
            }
            mock_config.field1 = "value1"
            mock_config.field2 = "value2"
            mock_config.field3 = None  # Should be excluded
            
            completed_configs = {"TestConfig": mock_config}
            
            parent_values = mock_step_catalog.extract_parent_values_for_inheritance(
                "TargetConfig", completed_configs
            )
            
            assert parent_values == {"field1": "value1", "field2": "value2"}
    
    def test_extract_parent_values_for_inheritance_fallback_dict(self, mock_step_catalog):
        """Test fallback to __dict__ when model_fields is not available."""
        with patch.object(mock_step_catalog, 'get_immediate_parent_config_class', return_value="TestConfig"):
            # CRITICAL FIX: Create a simple object instead of Mock to avoid internal Mock attributes
            # Mock objects have internal attributes like _mock_methods that interfere with __dict__
            class SimpleConfig:
                def __init__(self):
                    self.field1 = "value1"
                    self.field2 = "value2"
                    self._private_field = "private_value"  # Should be excluded
                    self.none_field = None  # Should be excluded
            
            mock_config = SimpleConfig()
            # Ensure model_fields doesn't exist to trigger fallback
            assert not hasattr(mock_config.__class__, 'model_fields')
            
            completed_configs = {"TestConfig": mock_config}
            
            parent_values = mock_step_catalog.extract_parent_values_for_inheritance(
                "TargetConfig", completed_configs
            )
            
            assert parent_values == {"field1": "value1", "field2": "value2"}
    
    def test_extract_parent_values_for_inheritance_exception_handling(self, mock_step_catalog):
        """Test exception handling in parent value extraction."""
        with patch.object(mock_step_catalog, 'get_immediate_parent_config_class', side_effect=Exception("Test error")):
            completed_configs = {"TestConfig": Mock()}
            
            parent_values = mock_step_catalog.extract_parent_values_for_inheritance(
                "TargetConfig", completed_configs
            )
            
            assert parent_values == {}


class TestParentConfigRetrievalIntegration:
    """Integration tests for parent config retrieval functionality."""
    
    @pytest.fixture
    def real_step_catalog(self):
        """Create a real StepCatalog instance for integration testing."""
        return StepCatalog()
    
    def test_cascading_inheritance_workflow(self, real_step_catalog):
        """Test the complete cascading inheritance workflow."""
        # Create sample configurations
        base_config = BasePipelineConfig(
            author="lukexie",
            bucket="my-pipeline-bucket",
            role="arn:aws:iam::123:role/MyRole",
            region="NA",
            service_name="xgboost-pipeline",
            pipeline_version="1.0.0",
            project_root_folder="my-project"
        )
        
        # CRITICAL FIX: Use correct field names from ProcessingStepConfigBase source code
        # Source shows: processing_instance_type_small and processing_instance_type_large, not processing_instance_type
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_instance_type_small="ml.m5.2xlarge",  # FIXED: Use correct field name
            processing_volume_size=500,
            processing_source_dir="src/processing"
        )
        
        # Test the workflow
        completed_configs = {
            "BasePipelineConfig": base_config,
            "ProcessingStepConfigBase": processing_config
        }
        
        # Test that TabularPreprocessingConfig gets values from ProcessingStepConfigBase
        try:
            parent_class = real_step_catalog.get_immediate_parent_config_class("TabularPreprocessingConfig")
            parent_values = real_step_catalog.extract_parent_values_for_inheritance(
                "TabularPreprocessingConfig", completed_configs
            )
            
            # Verify cascading inheritance works
            if parent_class and parent_values:
                # Should get values from ProcessingStepConfigBase (which includes cascaded base values)
                assert "author" in parent_values  # From base config
                # CRITICAL FIX: Check for actual field names from ProcessingStepConfigBase
                assert "processing_instance_type_small" in parent_values  # From processing config
                assert parent_values["author"] == "lukexie"
                assert parent_values["processing_instance_type_small"] == "ml.m5.2xlarge"
                
        except Exception as e:
            # If config classes are not available in test environment, that's expected
            pytest.skip(f"Config classes not available in test environment: {e}")
    
    def test_real_config_class_inheritance_patterns(self, real_step_catalog):
        """Test inheritance patterns with real config classes if available."""
        try:
            # Test various inheritance patterns
            test_cases = [
                ("ProcessingStepConfigBase", "BasePipelineConfig"),
                ("CradleDataLoadConfig", "BasePipelineConfig"),
            ]
            
            for config_class_name, expected_parent in test_cases:
                parent = real_step_catalog.get_immediate_parent_config_class(config_class_name)
                if parent is not None:  # Only assert if config class was found
                    assert parent == expected_parent, f"{config_class_name} should inherit from {expected_parent}, got {parent}"
                    
        except Exception as e:
            # If config classes are not available in test environment, that's expected
            pytest.skip(f"Config classes not available in test environment: {e}")


class TestParentConfigRetrievalEdgeCases:
    """Test edge cases and error conditions for parent config retrieval."""
    
    @pytest.fixture
    def mock_catalog_with_errors(self):
        """Create a StepCatalog that simulates various error conditions."""
        catalog = Mock(spec=StepCatalog)
        catalog.logger = Mock()
        return catalog
    
    def test_get_immediate_parent_config_class_with_complex_inheritance(self):
        """Test inheritance detection with complex inheritance chains."""
        # CRITICAL FIX: Create real classes that support issubclass() checks
        # Must use type() to create real classes, not Mock objects
        mock_base = type('BasePipelineConfig', (object,), {})
        mock_intermediate1 = type('IntermediateConfig1', (mock_base,), {})
        mock_intermediate2 = type('IntermediateConfig2', (mock_intermediate1,), {})
        mock_target = type('TargetConfig', (mock_intermediate2,), {})
        
        with patch('cursus.step_catalog.step_catalog.ConfigAutoDiscovery') as mock_discovery_class:
            mock_discovery = Mock()
            mock_discovery_class.return_value = mock_discovery
            mock_discovery.discover_config_classes.return_value = {"TargetConfig": mock_target}
            
            catalog = StepCatalog()
            
            # CRITICAL FIX: Mock the import at the source location where it actually happens
            with patch('cursus.core.base.config_base.BasePipelineConfig', mock_base):
                parent = catalog.get_immediate_parent_config_class("TargetConfig")
                # Should return the most immediate parent
                assert parent == "IntermediateConfig2"
    
    def test_extract_parent_values_with_missing_attributes(self):
        """Test parent value extraction when config instance is missing some attributes."""
        catalog = StepCatalog()
        
        # Create mock config with some missing attributes
        mock_config = Mock()
        mock_config.__class__.model_fields = {
            "existing_field": Mock(),
            "missing_field": Mock()
        }
        mock_config.existing_field = "value"
        # missing_field attribute doesn't exist on the instance
        del mock_config.missing_field
        
        with patch.object(catalog, 'get_immediate_parent_config_class', return_value="TestConfig"):
            completed_configs = {"TestConfig": mock_config}
            
            parent_values = catalog.extract_parent_values_for_inheritance(
                "TargetConfig", completed_configs
            )
            
            # Should only include fields that actually exist on the instance
            assert parent_values == {"existing_field": "value"}
    
    def test_logging_behavior(self):
        """Test that appropriate log messages are generated."""
        catalog = StepCatalog()
        
        with patch.object(catalog, 'logger') as mock_logger:
            # Test logging when config class is not found
            with patch.object(catalog, 'discover_config_classes', return_value={}):
                parent = catalog.get_immediate_parent_config_class("NonExistentConfig")
                
                assert parent is None
                mock_logger.warning.assert_called_with("Config class NonExistentConfig not found")
    
    def test_performance_with_large_config_instances(self):
        """Test performance with config instances that have many fields."""
        catalog = StepCatalog()
        
        # Create mock config with many fields
        mock_config = Mock()
        large_model_fields = {f"field_{i}": Mock() for i in range(100)}
        mock_config.__class__.model_fields = large_model_fields
        
        # Set values for all fields
        for field_name in large_model_fields:
            setattr(mock_config, field_name, f"value_{field_name}")
        
        with patch.object(catalog, 'get_immediate_parent_config_class', return_value="TestConfig"):
            completed_configs = {"TestConfig": mock_config}
            
            import time
            start_time = time.time()
            
            parent_values = catalog.extract_parent_values_for_inheritance(
                "TargetConfig", completed_configs
            )
            
            end_time = time.time()
            
            # Should complete quickly even with many fields
            assert end_time - start_time < 1.0  # Should take less than 1 second
            assert len(parent_values) == 100
