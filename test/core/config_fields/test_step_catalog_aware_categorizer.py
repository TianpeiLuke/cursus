"""
Unit tests for StepCatalogAwareConfigFieldCategorizer class.

This module contains comprehensive tests for the StepCatalogAwareConfigFieldCategorizer
class to ensure it correctly implements enhanced field categorization with workspace
and framework awareness while preserving all existing categorization rules.

Following pytest best practices:
- Read source code first to understand actual implementation
- Mock at correct import locations (not where imported TO)
- Use implementation-driven testing (test actual behavior)
- Prevent common failure categories through systematic design
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import tempfile

from cursus.core.config_fields.step_catalog_aware_categorizer import (
    StepCatalogAwareConfigFieldCategorizer,
    create_step_catalog_aware_categorizer
)
from cursus.core.config_fields.constants import CategoryType


class BaseTestConfig:
    """Base test config class for testing categorization."""

    def __init__(self, **kwargs):
        # Override with provided kwargs first
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.step_name_override = self.__class__.__name__


class SharedFieldsConfig(BaseTestConfig):
    """Config with shared fields for testing."""
    pass


class SpecificFieldsConfig(BaseTestConfig):
    """Config with specific fields for testing."""
    pass


class WorkspaceSpecificConfig(BaseTestConfig):
    """Config with workspace-specific fields for testing."""
    pass


class FrameworkSpecificConfig(BaseTestConfig):
    """Config with framework-specific fields for testing."""
    pass


class MockProcessingBase:
    """Mock base class for processing configs."""
    pass


class ProcessingConfig(MockProcessingBase, BaseTestConfig):
    """Mock processing config for testing."""
    pass


class TestStepCatalogAwareConfigFieldCategorizer:
    """
    Test cases for StepCatalogAwareConfigFieldCategorizer.
    
    Tests enhanced categorization with workspace and framework awareness
    while ensuring all existing categorization rules are preserved.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures and clean up after each test."""
        # Create test configs with various field types
        # Ensure shared_field has the same value across configs that should share it
        self.shared_config1 = SharedFieldsConfig()
        self.shared_config1.shared_field = "shared_value"
        self.shared_config1.common_field = "common_value"

        self.shared_config2 = SharedFieldsConfig()
        self.shared_config2.shared_field = "shared_value"
        self.shared_config2.common_field = "common_value"

        self.specific_config = SpecificFieldsConfig()
        self.specific_config.shared_field = "shared_value"
        self.specific_config.specific_field = "specific_value"
        self.specific_config.different_value_field = "value1"
        self.specific_config.common_field = "different_value"

        self.workspace_config = WorkspaceSpecificConfig()
        self.workspace_config.shared_field = "shared_value"
        self.workspace_config.workspace_specific_field = "workspace_value"
        self.workspace_config.project_config = "project_specific"

        self.framework_config = FrameworkSpecificConfig()
        self.framework_config.shared_field = "shared_value"
        self.framework_config.sagemaker_session = "session_value"
        self.framework_config.image_uri = "docker://image:tag"
        self.framework_config.pytorch_version = "1.9.0"

        self.processing_config = ProcessingConfig()
        self.processing_config.shared_field = "shared_value"
        self.processing_config.processing_specific = "process_value"
        self.processing_config.common_field = "common_value"

        self.configs = [
            self.shared_config1,
            self.shared_config2,
            self.specific_config,
            self.workspace_config,
            self.framework_config,
            self.processing_config
        ]

        # Create mock objects for enhanced features
        self.mock_step_catalog = Mock()
        self.mock_unified_manager = Mock()
        self.test_workspace_root = Path("/test/workspace")
        self.test_project_id = "test_project"

        yield  # This is where the test runs

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_init_basic_functionality(self, mock_serialize, mock_get_unified_manager):
        """Test that the categorizer correctly initializes with basic functionality."""
        # Setup mock serialize function
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl
        # Mock unified manager to fail initialization for basic test
        mock_get_unified_manager.side_effect = Exception("No unified manager")

        # Create categorizer without enhanced features
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            processing_step_config_base_class=MockProcessingBase
        )

        # Verify basic initialization
        assert len(categorizer.processing_configs) == 1
        assert len(categorizer.non_processing_configs) == 5
        assert self.processing_config in categorizer.processing_configs

        # Verify enhanced attributes are initialized
        assert categorizer.project_id is None
        assert categorizer.step_catalog is None
        assert categorizer.workspace_root is None
        assert categorizer.unified_manager is None
        assert isinstance(categorizer._workspace_field_mappings, dict)
        assert isinstance(categorizer._framework_field_mappings, dict)

        # Verify field info was collected (inherited functionality)
        assert "shared_field" in categorizer.field_info["sources"]
        assert len(categorizer.field_info["sources"]["shared_field"]) == 6

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_init_with_enhanced_features(self, mock_serialize, mock_get_unified_manager):
        """Test initialization with enhanced workspace and framework features."""
        # Setup mocks
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl
        mock_get_unified_manager.return_value = self.mock_unified_manager

        # Create categorizer with enhanced features
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            processing_step_config_base_class=MockProcessingBase,
            project_id=self.test_project_id,
            step_catalog=self.mock_step_catalog,
            workspace_root=self.test_workspace_root
        )

        # Verify enhanced initialization
        assert categorizer.project_id == self.test_project_id
        assert categorizer.step_catalog == self.mock_step_catalog
        assert categorizer.workspace_root == self.test_workspace_root
        assert categorizer.unified_manager == self.mock_unified_manager

        # Verify unified manager was called
        mock_get_unified_manager.assert_called_once_with(self.test_workspace_root)

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_init_unified_manager_failure(self, mock_serialize, mock_get_unified_manager):
        """Test initialization when unified manager fails to initialize."""
        # Setup mocks
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}
        mock_get_unified_manager.side_effect = Exception("Manager initialization failed")

        # Create categorizer - should handle exception gracefully
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            workspace_root=self.test_workspace_root
        )

        # Verify graceful handling
        assert categorizer.unified_manager is None
        assert isinstance(categorizer._workspace_field_mappings, dict)
        assert isinstance(categorizer._framework_field_mappings, dict)

    def test_get_framework_field_mappings(self):
        """Test that framework-specific field mappings are correctly initialized."""
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        framework_mappings = categorizer._get_framework_field_mappings()

        # Verify SageMaker-specific fields
        assert framework_mappings["sagemaker_session"] == "framework_specific"
        assert framework_mappings["role_arn"] == "framework_specific"
        assert framework_mappings["security_group_ids"] == "framework_specific"

        # Verify Docker/Container-specific fields
        assert framework_mappings["image_uri"] == "framework_specific"
        assert framework_mappings["container_entry_point"] == "framework_specific"

        # Verify Cloud provider-specific fields
        assert framework_mappings["aws_region"] == "cloud_specific"
        assert framework_mappings["azure_region"] == "cloud_specific"

        # Verify ML framework-specific fields
        assert framework_mappings["pytorch_version"] == "ml_framework"
        assert framework_mappings["tensorflow_version"] == "ml_framework"

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_get_workspace_field_mappings_no_unified_manager(self, mock_serialize):
        """Test workspace field mappings when unified manager is not available."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            project_id=self.test_project_id
        )

        workspace_mappings = categorizer._get_workspace_field_mappings()

        # Should return empty dict when no unified manager
        assert isinstance(workspace_mappings, dict)
        assert len(workspace_mappings) == 0

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_get_workspace_field_mappings_with_unified_manager(self, mock_serialize, mock_get_unified_manager):
        """Test workspace field mappings with unified manager available."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}
        
        # Setup unified manager mock
        mock_config_class = Mock()
        mock_config_class.get_workspace_field_mappings.return_value = {
            "workspace_field": "workspace_category"
        }
        
        # Mock model_fields for field annotations
        mock_field_info = Mock()
        mock_field_info.json_schema_extra = {"workspace_category": "workspace_specific"}
        mock_config_class.model_fields = {
            "annotated_field": mock_field_info
        }
        
        self.mock_unified_manager.get_config_classes.return_value = {
            "TestConfig": mock_config_class
        }
        mock_get_unified_manager.return_value = self.mock_unified_manager

        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            project_id=self.test_project_id,
            workspace_root=self.test_workspace_root
        )

        workspace_mappings = categorizer._get_workspace_field_mappings()

        # Verify workspace mappings were collected
        assert "workspace_field" in workspace_mappings
        assert workspace_mappings["workspace_field"] == "workspace_category"
        assert "annotated_field" in workspace_mappings
        assert workspace_mappings["annotated_field"] == "workspace_specific"

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_categorize_field_with_step_catalog_context_workspace_mapping(self, mock_serialize):
        """Test field categorization with workspace-specific mappings."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])
        
        # Set up workspace mappings
        categorizer._workspace_field_mappings = {
            "workspace_field": "workspace_specific"
        }

        # Test workspace-specific field categorization
        result = categorizer._categorize_field_with_step_catalog_context(
            "workspace_field", ["value"], ["TestConfig"]
        )

        assert result == "workspace_specific"

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_categorize_field_with_step_catalog_context_framework_mapping(self, mock_serialize):
        """Test field categorization with framework-specific mappings."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # Test framework-specific field categorization
        result = categorizer._categorize_field_with_step_catalog_context(
            "sagemaker_session", ["session_value"], ["TestConfig"]
        )

        assert result == "framework_specific"

        result = categorizer._categorize_field_with_step_catalog_context(
            "pytorch_version", ["1.9.0"], ["TestConfig"]
        )

        assert result == "ml_framework"

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_categorize_field_with_tier_aware_categorization(self, mock_serialize, mock_get_unified_manager):
        """Test field categorization with tier-aware logic from unified manager."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}
        
        # Setup unified manager with tier information
        self.mock_unified_manager.get_field_tiers.return_value = {
            "essential": ["essential_field"],
            "system": ["system_field"],
            "derived": ["derived_field"]
        }
        mock_get_unified_manager.return_value = self.mock_unified_manager

        # Create config with the test field
        test_config = BaseTestConfig(essential_field="value")
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[test_config],
            workspace_root=self.test_workspace_root
        )

        # Test tier-aware categorization
        result = categorizer._categorize_field_with_step_catalog_context(
            "essential_field", ["value"], ["BaseTestConfig"]
        )

        # Essential fields should be categorized as shared
        assert result == "shared"

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_categorize_field_fallback_to_base(self, mock_serialize):
        """Test that categorization falls back to base class when no enhanced rules apply."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # Mock the parent class method - the enhanced method calls base with correct signature
        with patch.object(categorizer.__class__.__bases__[0], '_categorize_field', return_value=CategoryType.SPECIFIC) as mock_base:
            result = categorizer._categorize_field_with_step_catalog_context(
                "unknown_field", ["value"], ["TestConfig"]
            )

            # The enhanced method calls base with correct signature (field_name only)
            mock_base.assert_called_once_with("unknown_field")

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_categorize_field_override(self, mock_serialize):
        """Test that _categorize_field correctly overrides base implementation."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # Note: The enhanced _categorize_field method signature is different from base
        # We need to test the enhanced categorization context method instead
        result = categorizer._categorize_field_with_step_catalog_context(
            "sagemaker_session", ["session_value"], ["TestConfig"]
        )

        assert result == "framework_specific"

        # Test workspace-specific field gets mapped to workspace category
        categorizer._workspace_field_mappings = {"workspace_field": "workspace_specific"}
        result = categorizer._categorize_field_with_step_catalog_context(
            "workspace_field", ["value"], ["TestConfig"]
        )

        assert result == "workspace_specific"

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_get_enhanced_categorization_info(self, mock_get_unified_manager):
        """Test that enhanced categorization info is correctly collected."""
        # Mock unified manager to return None to control the test expectation
        mock_get_unified_manager.side_effect = Exception("No unified manager")
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            project_id=self.test_project_id,
            step_catalog=self.mock_step_catalog,
            workspace_root=self.test_workspace_root
        )

        info = categorizer.get_enhanced_categorization_info()

        # Verify basic info
        assert info["project_id"] == self.test_project_id
        assert info["workspace_field_mappings_count"] == 0  # No mappings in basic test
        assert info["framework_field_mappings_count"] > 0  # Should have framework mappings
        assert info["unified_manager_available"] is False  # Mocked to fail
        assert info["step_catalog_available"] is True

        # Verify step catalog info
        assert "step_catalog_info" in info
        assert info["step_catalog_info"]["catalog_type"] == "Mock"

    def test_get_enhanced_categorization_info_step_catalog_exception(self):
        """Test enhanced categorization info when step catalog info extraction fails."""
        # Create a mock that raises exception when accessing attributes
        problematic_catalog = Mock()
        problematic_catalog.workspace_root = Mock(side_effect=Exception("Access failed"))

        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            step_catalog=problematic_catalog
        )

        # Should handle exception gracefully
        info = categorizer.get_enhanced_categorization_info()
        assert info["step_catalog_available"] is True
        # step_catalog_info might not be present due to exception

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_categorize_with_enhanced_metadata(self, mock_serialize):
        """Test categorization with enhanced metadata."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            project_id=self.test_project_id
        )

        # Mock the base get_categorized_fields method
        base_result = {"shared": {}, "specific": {}}
        with patch.object(categorizer, 'get_categorized_fields', return_value=base_result):
            result = categorizer.categorize_with_enhanced_metadata()

            # Verify enhanced metadata is added
            assert "enhanced_metadata" in result
            assert result["enhanced_metadata"]["project_id"] == self.test_project_id
            assert "workspace_field_mappings_count" in result["enhanced_metadata"]
            assert "framework_field_mappings_count" in result["enhanced_metadata"]

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_end_to_end_categorization_with_framework_fields(self, mock_serialize):
        """Test end-to-end categorization with framework-specific fields."""
        # Setup mock serialize function
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl

        # Create configs with framework-specific fields
        sagemaker_config = FrameworkSpecificConfig(
            shared_field="shared_value",
            sagemaker_session="session_value",
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole"
        )

        docker_config = FrameworkSpecificConfig(
            shared_field="shared_value",
            image_uri="docker://my-image:latest",
            container_entry_point=["python", "train.py"]
        )

        configs = [sagemaker_config, docker_config]

        # Create categorizer
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=configs,
            processing_step_config_base_class=MockProcessingBase
        )

        # Get categorization result
        result = categorizer.get_categorized_fields()

        # Verify structure
        assert set(result.keys()) == {"shared", "specific"}

        # Verify shared fields
        assert "shared_field" in result["shared"]
        assert result["shared"]["shared_field"] == "shared_value"

        # Verify framework-specific fields are in specific sections
        assert "FrameworkSpecificConfig" in result["specific"]
        
        # Check that framework fields are properly categorized as specific
        framework_fields = result["specific"]["FrameworkSpecificConfig"]
        assert "sagemaker_session" in framework_fields or "image_uri" in framework_fields

    def test_create_step_catalog_aware_categorizer_factory(self):
        """Test the factory function for creating categorizer instances."""
        categorizer = create_step_catalog_aware_categorizer(
            config_list=self.configs,
            processing_step_config_base_class=MockProcessingBase,
            project_id=self.test_project_id,
            step_catalog=self.mock_step_catalog,
            workspace_root=self.test_workspace_root
        )

        # Verify factory creates correct instance
        assert isinstance(categorizer, StepCatalogAwareConfigFieldCategorizer)
        assert categorizer.project_id == self.test_project_id
        assert categorizer.step_catalog == self.mock_step_catalog
        assert categorizer.workspace_root == self.test_workspace_root

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_inheritance_preservation(self, mock_serialize):
        """Test that all base class functionality is preserved."""
        # Setup mock serialize function
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl

        # Create categorizer
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            processing_step_config_base_class=MockProcessingBase
        )

        # Test inherited methods work correctly
        field_sources = categorizer.get_field_sources()
        assert isinstance(field_sources, dict)
        assert "shared_field" in field_sources

        # Test inherited categorization logic
        result = categorizer.get_categorized_fields()
        assert "shared" in result
        assert "specific" in result

        # Test inherited stats method (should not raise exception)
        categorizer.print_categorization_stats()

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_error_handling_in_enhanced_mappings(self, mock_serialize):
        """Test error handling in enhanced mapping initialization."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        # Create categorizer that will fail during enhanced mapping initialization
        with patch.object(StepCatalogAwareConfigFieldCategorizer, '_get_workspace_field_mappings', 
                         side_effect=Exception("Mapping failed")):
            categorizer = StepCatalogAwareConfigFieldCategorizer(
                config_list=[],
                project_id=self.test_project_id
            )

            # Should handle exception gracefully
            assert isinstance(categorizer._workspace_field_mappings, dict)
            assert isinstance(categorizer._framework_field_mappings, dict)

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_tier_aware_categorization_exception_handling(self, mock_serialize):
        """Test exception handling in tier-aware categorization."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        # Create categorizer with unified manager that raises exception
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])
        categorizer.unified_manager = Mock()
        categorizer.unified_manager.get_field_tiers.side_effect = Exception("Tier extraction failed")

        # Should handle exception gracefully and fall back to base categorization
        with patch.object(categorizer.__class__.__bases__[0], '_categorize_field', return_value=CategoryType.SHARED) as mock_base:
            result = categorizer._categorize_field_with_step_catalog_context(
                "test_field", ["value"], ["TestConfig"]
            )

            # Should fall back to base categorization
            mock_base.assert_called_once()

    def test_framework_field_mappings_completeness(self):
        """Test that framework field mappings cover expected categories."""
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])
        mappings = categorizer._framework_field_mappings

        # Verify different framework categories are represented
        framework_categories = set(mappings.values())
        expected_categories = {
            "framework_specific",
            "cloud_specific", 
            "ml_framework"
        }

        assert expected_categories.issubset(framework_categories)

        # Verify specific important fields are mapped
        important_fields = [
            "sagemaker_session", "image_uri", "pytorch_version", 
            "aws_region", "container_entry_point"
        ]
        
        for field in important_fields:
            assert field in mappings

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_workspace_field_mappings_with_model_fields_exception(self, mock_serialize):
        """Test workspace field mappings when model_fields access raises exception."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        # Create mock config class that raises exception on model_fields access
        mock_config_class = Mock()
        mock_config_class.get_workspace_field_mappings.return_value = {}
        
        # Make model_fields raise an exception
        type(mock_config_class).model_fields = PropertyMock(side_effect=AttributeError("No model_fields"))

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])
        categorizer.unified_manager = Mock()
        categorizer.unified_manager.get_config_classes.return_value = {
            "TestConfig": mock_config_class
        }

        # Should handle exception gracefully
        workspace_mappings = categorizer._get_workspace_field_mappings()
        assert isinstance(workspace_mappings, dict)

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_categorization_mapping_priority(self, mock_serialize):
        """Test that workspace mappings take priority over framework mappings."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])
        
        # Set up conflicting mappings (workspace should win)
        categorizer._workspace_field_mappings = {
            "sagemaker_session": "workspace_override"
        }

        # Test that workspace mapping takes priority
        result = categorizer._categorize_field_with_step_catalog_context(
            "sagemaker_session", ["value"], ["TestConfig"]
        )

        assert result == "workspace_override"


class TestStepCatalogAwareCategorizerEdgeCases:
    """Test edge cases and error conditions for StepCatalogAwareConfigFieldCategorizer."""

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_empty_config_list(self, mock_serialize):
        """Test categorizer with empty config list."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # Should handle empty list gracefully
        assert len(categorizer.processing_configs) == 0
        assert len(categorizer.non_processing_configs) == 0
        assert isinstance(categorizer._workspace_field_mappings, dict)
        assert isinstance(categorizer._framework_field_mappings, dict)

        # Should be able to get categorization (empty)
        result = categorizer.get_categorized_fields()
        assert "shared" in result
        assert "specific" in result

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_none_values_handling(self, mock_serialize, mock_get_unified_manager):
        """Test handling of None values in various parameters."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}
        # Mock unified manager to fail when workspace_root is None
        mock_get_unified_manager.side_effect = Exception("No workspace root")

        # Test with None values for optional parameters
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            processing_step_config_base_class=None,
            project_id=None,
            step_catalog=None,
            workspace_root=None
        )

        # Should handle None values gracefully
        assert categorizer.project_id is None
        assert categorizer.step_catalog is None
        assert categorizer.workspace_root is None
        assert categorizer.unified_manager is None

        # Enhanced info should handle None values
        info = categorizer.get_enhanced_categorization_info()
        assert info["project_id"] is None
        assert info["step_catalog_available"] is False

    def test_invalid_workspace_root_type(self):
        """Test handling of invalid workspace_root type."""
        # Test with string instead of Path
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            workspace_root="/invalid/string/path"  # Should be Path object
        )

        # Should handle gracefully (unified manager will likely fail to initialize)
        assert categorizer.workspace_root == "/invalid/string/path"

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_malformed_step_catalog_object(self, mock_serialize):
        """Test handling of malformed step catalog object."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}

        # Create a malformed step catalog (missing expected attributes)
        malformed_catalog = object()  # No attributes

        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[],
            step_catalog=malformed_catalog
        )

        # Should handle gracefully
        info = categorizer.get_enhanced_categorization_info()
        assert info["step_catalog_available"] is True
        # step_catalog_info extraction might fail, but shouldn't crash


class TestStepCatalogAwareCategorizerIntegration:
    """Integration tests for StepCatalogAwareConfigFieldCategorizer."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace_root = Path(temp_dir)
            
            # Create realistic directory structure
            dev_workspace = workspace_root / "dev1"
            dev_workspace.mkdir(parents=True)
            
            # Create component directories
            for component_type in ["scripts", "contracts", "specs", "builders", "configs"]:
                component_dir = dev_workspace / component_type
                component_dir.mkdir()
                
                # Create sample files
                sample_file = component_dir / f"sample_{component_type[:-1]}.py"
                sample_file.write_text(f"# Sample {component_type[:-1]} file")
            
            yield workspace_root

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_integration_with_real_workspace_structure(self, mock_serialize, mock_get_unified_manager, temp_workspace):
        """Test integration with realistic workspace structure."""
        # Setup mocks
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl
        
        # Setup unified manager mock
        mock_unified_manager = Mock()
        mock_unified_manager.get_config_classes.return_value = {}
        mock_get_unified_manager.return_value = mock_unified_manager

        # Create configs with mixed field types
        config1 = BaseTestConfig(
            shared_field="shared_value",
            workspace_path=str(temp_workspace),
            sagemaker_session="session1"
        )
        
        config2 = BaseTestConfig(
            shared_field="shared_value", 
            workspace_path=str(temp_workspace),
            pytorch_version="1.9.0"
        )

        # Create categorizer with real workspace
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[config1, config2],
            project_id="integration_test",
            workspace_root=temp_workspace
        )

        # Test categorization works end-to-end
        result = categorizer.get_categorized_fields()
        
        # Verify structure
        assert "shared" in result
        assert "specific" in result
        
        # Verify shared fields
        assert "shared_field" in result["shared"]
        assert result["shared"]["shared_field"] == "shared_value"
        
        # Verify framework-specific fields are categorized as specific
        specific_configs = result["specific"]
        assert len(specific_configs) > 0

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_performance_with_large_config_list(self, mock_serialize):
        """Test performance and correctness with large number of configs."""
        # Setup mock serialize function
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl

        # Create large number of configs
        configs = []
        for i in range(100):
            config = BaseTestConfig(
                shared_field="shared_value",
                config_id=f"config_{i}",
                framework_field="pytorch" if i % 2 == 0 else "tensorflow",
                sagemaker_session=f"session_{i % 10}"  # Framework-specific field
            )
            configs.append(config)

        # Create categorizer
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=configs,
            project_id="performance_test"
        )

        # Test categorization completes successfully
        result = categorizer.get_categorized_fields()
        
        # Verify structure is correct
        assert "shared" in result
        assert "specific" in result
        
        # Verify shared field is correctly identified
        assert "shared_field" in result["shared"]
        assert result["shared"]["shared_field"] == "shared_value"
        
        # Verify framework-specific fields are in specific sections
        assert len(result["specific"]) > 0

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_mixed_framework_and_workspace_categorization(self, mock_serialize):
        """Test categorization with both framework and workspace-specific fields."""
        # Setup mock serialize function
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl

        # Create configs with mixed field types
        sagemaker_config = BaseTestConfig(
            shared_field="shared_value",
            sagemaker_session="session_value",  # Framework-specific
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",  # Framework-specific
            project_workspace="workspace1"  # Could be workspace-specific
        )

        pytorch_config = BaseTestConfig(
            shared_field="shared_value",
            pytorch_version="1.9.0",  # Framework-specific (ML)
            cuda_version="11.1",  # Framework-specific (ML)
            project_workspace="workspace2"  # Could be workspace-specific
        )

        configs = [sagemaker_config, pytorch_config]

        # Create categorizer with workspace mappings
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=configs,
            project_id="mixed_test"
        )
        
        # Add workspace-specific mapping
        categorizer._workspace_field_mappings = {
            "project_workspace": "workspace_specific"
        }

        # Test categorization
        result = categorizer.get_categorized_fields()
        
        # Verify structure
        assert "shared" in result
        assert "specific" in result
        
        # Verify shared field
        assert "shared_field" in result["shared"]
        
        # Verify specific fields are properly categorized
        specific_configs = result["specific"]
        assert len(specific_configs) >= 1  # At least one config should have specific fields
        
        # Both configs have same class name, so they get merged into one entry
        assert "BaseTestConfig" in specific_configs
        base_config_fields = specific_configs["BaseTestConfig"]
        
        # Verify that framework-specific fields are present
        framework_fields = {"sagemaker_session", "pytorch_version", "cuda_version", "role_arn"}
        assert any(field in base_config_fields for field in framework_fields)


class TestStepCatalogAwareCategorizerCompatibility:
    """Test compatibility and backward compatibility of StepCatalogAwareConfigFieldCategorizer."""

    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_backward_compatibility_with_base_categorizer(self, mock_serialize):
        """Test that enhanced categorizer maintains backward compatibility."""
        # Setup mock serialize function (same as base categorizer tests)
        def mock_serialize_impl(config):
            result = {"_metadata": {"step_name": config.__class__.__name__}}
            for key, value in config.__dict__.items():
                if key != "step_name_override" and value is not None:
                    result[key] = value
            return result

        mock_serialize.side_effect = mock_serialize_impl

        # Create same test configs as base categorizer
        shared_config1 = BaseTestConfig(
            shared_field="shared_value", 
            common_field="common_value"
        )
        shared_config2 = BaseTestConfig(
            shared_field="shared_value", 
            common_field="common_value"
        )
        specific_config = BaseTestConfig(
            shared_field="shared_value",
            specific_field="specific_value",
            common_field="different_value"
        )

        configs = [shared_config1, shared_config2, specific_config]

        # Create enhanced categorizer
        enhanced_categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=configs,
            processing_step_config_base_class=MockProcessingBase
        )

        # Get results
        enhanced_result = enhanced_categorizer.get_categorized_fields()

        # Verify same structure as base categorizer
        assert set(enhanced_result.keys()) == {"shared", "specific"}
        
        # Verify shared fields work the same
        assert "shared_field" in enhanced_result["shared"]
        assert enhanced_result["shared"]["shared_field"] == "shared_value"
        
        # Verify specific fields work the same
        assert "BaseTestConfig" in enhanced_result["specific"]
        specific_fields = enhanced_result["specific"]["BaseTestConfig"]
        assert "specific_field" in specific_fields
        assert "common_field" in specific_fields

    def test_api_compatibility(self):
        """Test that all public API methods are available and work correctly."""
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # Test all inherited public methods are available
        assert hasattr(categorizer, 'get_categorized_fields')
        assert hasattr(categorizer, 'get_field_sources')
        assert hasattr(categorizer, 'print_categorization_stats')
        assert hasattr(categorizer, 'get_category_for_field')

        # Test new public methods are available
        assert hasattr(categorizer, 'get_enhanced_categorization_info')
        assert hasattr(categorizer, 'categorize_with_enhanced_metadata')

        # Test factory function is available
        from cursus.core.config_fields.step_catalog_aware_categorizer import create_step_catalog_aware_categorizer
        assert callable(create_step_catalog_aware_categorizer)

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    @patch('cursus.core.config_fields.config_field_categorizer.serialize_config')
    def test_enhanced_features_optional(self, mock_serialize, mock_get_unified_manager):
        """Test that enhanced features are optional and don't break basic functionality."""
        mock_serialize.return_value = {"_metadata": {"step_name": "TestConfig"}}
        # Mock unified manager to fail when no workspace_root provided
        mock_get_unified_manager.side_effect = Exception("No workspace root")

        # Create categorizer without any enhanced features
        basic_categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # Should work without enhanced features
        assert basic_categorizer.project_id is None
        assert basic_categorizer.step_catalog is None
        assert basic_categorizer.workspace_root is None
        assert basic_categorizer.unified_manager is None

        # Basic functionality should still work
        result = basic_categorizer.get_categorized_fields()
        assert "shared" in result
        assert "specific" in result

        # Enhanced info should handle None values gracefully
        info = basic_categorizer.get_enhanced_categorization_info()
        assert info["project_id"] is None
        assert info["step_catalog_available"] is False
        assert info["unified_manager_available"] is False


# Add missing import for PropertyMock
from unittest.mock import PropertyMock
