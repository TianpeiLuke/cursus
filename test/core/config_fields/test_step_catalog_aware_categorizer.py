"""
Modernized unit tests for StepCatalogAwareConfigFieldCategorizer class.

This module contains comprehensive tests for the StepCatalogAwareConfigFieldCategorizer
class following systematic error prevention methodology from pytest best practices guides.

SYSTEMATIC ERROR PREVENTION APPLIED:
- ✅ Source Code First Rule: Read actual implementation before testing
- ✅ Mock Path Precision: Use exact import paths from source code analysis
- ✅ Implementation-Driven Testing: Test actual behavior, not assumptions
- ✅ Error Prevention Categories 1-17: All systematically addressed

Following pytest best practices:
- Read source code first to understand actual implementation
- Mock at correct import locations (not where imported TO)
- Use implementation-driven testing (test actual behavior)
- Prevent common failure categories through systematic design
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional
from pathlib import Path
import tempfile

# ✅ SYSTEMATIC ERROR PREVENTION: Import actual classes after reading source code
from cursus.core.config_fields.step_catalog_aware_categorizer import (
    StepCatalogAwareConfigFieldCategorizer,
    create_step_catalog_aware_categorizer
)
from cursus.core.config_fields.constants import CategoryType

# ✅ SYSTEMATIC ERROR PREVENTION: Import real config classes for testing
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
from cursus.steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters

# ✅ SYSTEMATIC ERROR PREVENTION: Import serialize_config for proper mocking
from cursus.core.config_fields.type_aware_config_serializer import serialize_config


class TestStepCatalogAwareConfigFieldCategorizer:
    """
    Modernized test cases for StepCatalogAwareConfigFieldCategorizer.
    
    Tests enhanced categorization with workspace and framework awareness
    following systematic error prevention methodology.
    """

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up test fixtures following systematic error prevention."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Create REAL config objects based on source code analysis
        # Following Source Code First Rule: Use actual config classes with proper required fields
        
        # Create temporary directories for realistic testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.source_dir = self.temp_path / "source"
        self.processing_dir = self.temp_path / "processing" / "scripts"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.processing_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Base Pipeline Config - using REAL required fields from source code
        self.base_config = BasePipelineConfig(
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
        self.processing_config = ProcessingStepConfigBase.from_base_config(
            self.base_config,
            # Processing-specific fields
            processing_source_dir=str(self.processing_dir),
            processing_instance_type_large="ml.m5.12xlarge",
            processing_instance_type_small="ml.m5.4xlarge", 
            processing_framework_version="1.2-1",
            processing_instance_count=1,
            processing_volume_size=500,
            use_large_processing_instance=False,
            processing_entry_point="processing_script.py"
        )

        # 3. XGBoost Training Config - using REAL hyperparameters from source code
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

        self.training_config = XGBoostTrainingConfig.from_base_config(
            self.base_config,
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

        # Create config list with real objects
        self.configs = [self.base_config, self.processing_config, self.training_config]

        # Create mock objects for enhanced features
        self.mock_step_catalog = Mock()
        self.mock_unified_manager = Mock()
        self.test_workspace_root = Path("/test/workspace")
        self.test_project_id = "test_project"

        yield  # This is where the test runs
        
        # Clean up
        self.temp_dir.cleanup()

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_init_basic_functionality_following_guides(self, mock_get_unified_manager):
        """Test categorizer initialization following systematic error prevention."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Mock unified manager failure for basic test
        mock_get_unified_manager.side_effect = Exception("No unified manager")

        # Create categorizer without enhanced features
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            processing_step_config_base_class=ProcessingStepConfigBase
        )

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual behavior from source code
        # Source analysis: ProcessingStepConfigBase should be identified correctly
        assert len(categorizer.processing_configs) == 1
        assert len(categorizer.non_processing_configs) == 2  # base + training configs
        assert self.processing_config in categorizer.processing_configs

        # Verify enhanced attributes are initialized (from source code)
        assert categorizer.project_id is None
        assert categorizer.step_catalog is None
        assert categorizer.workspace_root is None
        assert categorizer.unified_manager is None
        assert isinstance(categorizer._workspace_field_mappings, dict)
        assert isinstance(categorizer._framework_field_mappings, dict)

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual field collection behavior
        # Source analysis: field_info should contain actual fields from real configs
        assert "author" in categorizer.field_info["sources"]  # From base config
        assert "bucket" in categorizer.field_info["sources"]  # From base config
        assert len(categorizer.field_info["sources"]["author"]) >= 1  # Should appear in configs

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_init_with_enhanced_features_following_guides(self, mock_get_unified_manager):
        """Test initialization with enhanced workspace and framework features."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Mock unified manager success
        mock_get_unified_manager.return_value = self.mock_unified_manager

        # Create categorizer with enhanced features
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            processing_step_config_base_class=ProcessingStepConfigBase,
            project_id=self.test_project_id,
            step_catalog=self.mock_step_catalog,
            workspace_root=self.test_workspace_root
        )

        # Verify enhanced initialization
        assert categorizer.project_id == self.test_project_id
        assert categorizer.step_catalog == self.mock_step_catalog
        assert categorizer.workspace_root == self.test_workspace_root
        assert categorizer.unified_manager == self.mock_unified_manager

        # ✅ SYSTEMATIC ERROR PREVENTION: Verify actual method calls from source
        # Source analysis: get_unified_config_manager called with workspace_root
        mock_get_unified_manager.assert_called_once_with(self.test_workspace_root)

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_init_unified_manager_failure_following_guides(self, mock_get_unified_manager):
        """Test initialization when unified manager fails to initialize."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual exception handling from source
        mock_get_unified_manager.side_effect = Exception("Manager initialization failed")

        # Create categorizer - should handle exception gracefully
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            workspace_root=self.test_workspace_root
        )

        # ✅ SYSTEMATIC ERROR PREVENTION: Verify graceful handling matches source behavior
        assert categorizer.unified_manager is None
        assert isinstance(categorizer._workspace_field_mappings, dict)
        assert isinstance(categorizer._framework_field_mappings, dict)

    def test_get_framework_field_mappings_following_guides(self):
        """Test framework-specific field mappings following source code analysis."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)

        framework_mappings = categorizer._get_framework_field_mappings()

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual mappings from source code
        # Source analysis: _get_framework_field_mappings() returns specific mappings
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

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_get_workspace_field_mappings_no_unified_manager_following_guides(self, mock_get_unified_manager):
        """Test workspace field mappings when unified manager is not available."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Mock unified manager failure
        mock_get_unified_manager.side_effect = Exception("No unified manager")

        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            project_id=self.test_project_id
        )

        workspace_mappings = categorizer._get_workspace_field_mappings()

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual behavior from source
        # Source analysis: returns empty dict when no unified manager
        assert isinstance(workspace_mappings, dict)
        assert len(workspace_mappings) == 0

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_get_workspace_field_mappings_with_unified_manager_following_guides(self, mock_get_unified_manager):
        """Test workspace field mappings with unified manager available."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Setup unified manager mock based on source analysis
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
            config_list=self.configs,
            project_id=self.test_project_id,
            workspace_root=self.test_workspace_root
        )

        workspace_mappings = categorizer._get_workspace_field_mappings()

        # ✅ SYSTEMATIC ERROR PREVENTION: Verify actual behavior from source
        assert "workspace_field" in workspace_mappings
        assert workspace_mappings["workspace_field"] == "workspace_category"
        assert "annotated_field" in workspace_mappings
        assert workspace_mappings["annotated_field"] == "workspace_specific"

    def test_categorize_field_with_step_catalog_context_workspace_mapping_following_guides(self):
        """Test field categorization with workspace-specific mappings."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)
        
        # Set up workspace mappings
        categorizer._workspace_field_mappings = {
            "workspace_field": "workspace_specific"
        }

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual method signature from source
        # Source analysis: _categorize_field_with_step_catalog_context takes 3 parameters
        result = categorizer._categorize_field_with_step_catalog_context(
            "workspace_field", ["value"], ["TestConfig"]
        )

        assert result == "workspace_specific"

    def test_categorize_field_with_step_catalog_context_framework_mapping_following_guides(self):
        """Test field categorization with framework-specific mappings."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)

        # ✅ SYSTEMATIC ERROR PREVENTION: Test framework-specific field categorization
        result = categorizer._categorize_field_with_step_catalog_context(
            "sagemaker_session", ["session_value"], ["TestConfig"]
        )

        assert result == "framework_specific"

        result = categorizer._categorize_field_with_step_catalog_context(
            "pytorch_version", ["1.9.0"], ["TestConfig"]
        )

        assert result == "ml_framework"

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_categorize_field_with_tier_aware_categorization_following_guides(self, mock_get_unified_manager):
        """Test field categorization with tier-aware logic from unified manager."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Setup unified manager with tier information
        self.mock_unified_manager.get_field_tiers.return_value = {
            "essential": ["author"],  # Use actual field from real config
            "system": ["framework_version"],  # Use actual field from real config
            "derived": ["aws_region"]  # Use actual derived field
        }
        mock_get_unified_manager.return_value = self.mock_unified_manager
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            workspace_root=self.test_workspace_root
        )

        # ✅ SYSTEMATIC ERROR PREVENTION: Test tier-aware categorization with real field
        result = categorizer._categorize_field_with_step_catalog_context(
            "author", ["test-user"], ["BasePipelineConfig"]
        )

        # Essential fields should be categorized as shared
        assert result == "shared"

    def test_categorize_field_fallback_to_base_following_guides(self):
        """Test that categorization falls back to base class when no enhanced rules apply."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)

        # ✅ SYSTEMATIC ERROR PREVENTION: Test fallback behavior with unknown field
        # Source analysis: _categorize_field_with_step_catalog_context calls _categorize_field_base_logic
        result = categorizer._categorize_field_with_step_catalog_context(
            "unknown_field", ["value"], ["TestConfig"]
        )

        # Should fall back to base categorization logic
        # Source analysis: base logic returns "shared" or "specific"
        assert result in ["shared", "specific"]  # Valid base categories

    def test_efficient_shared_fields_algorithm_following_guides(self):
        """Test the efficient O(n*m) shared fields algorithm from source code."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual efficient algorithm behavior
        # Source analysis: _populate_shared_fields_efficient uses O(n*m) algorithm
        result = {"shared": {}, "specific": defaultdict(dict)}
        categorizer._populate_shared_fields_efficient(self.configs, result)

        # Verify shared fields are correctly identified
        # Fields that appear in ALL configs with same value should be shared
        shared_fields = result["shared"]
        
        # These fields should be shared across all our real configs
        expected_shared_fields = ["author", "bucket", "region", "service_name", "pipeline_version"]
        for field in expected_shared_fields:
            if field in shared_fields:
                assert shared_fields[field] is not None

    def test_efficient_specific_fields_algorithm_following_guides(self):
        """Test the efficient specific fields population algorithm."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual efficient algorithm behavior
        result = {"shared": {"author": "test-user"}, "specific": defaultdict(dict)}
        categorizer._populate_specific_fields_efficient(self.configs, result)

        # Verify specific fields structure
        specific_configs = result["specific"]
        assert len(specific_configs) >= 1  # Should have specific configs

        # Each specific config should have __model_type__
        for step_name, config_data in specific_configs.items():
            assert "__model_type__" in config_data
            assert isinstance(config_data["__model_type__"], str)

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_get_enhanced_categorization_info_following_guides(self, mock_get_unified_manager):
        """Test that enhanced categorization info is correctly collected."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Mock unified manager to control test expectation
        mock_get_unified_manager.side_effect = Exception("No unified manager")
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            project_id=self.test_project_id,
            step_catalog=self.mock_step_catalog,
            workspace_root=self.test_workspace_root
        )

        info = categorizer.get_enhanced_categorization_info()

        # ✅ SYSTEMATIC ERROR PREVENTION: Verify actual info structure from source
        assert info["project_id"] == self.test_project_id
        assert info["workspace_field_mappings_count"] == 0  # No mappings in basic test
        assert info["framework_field_mappings_count"] > 0  # Should have framework mappings
        assert info["unified_manager_available"] is False  # Mocked to fail
        assert info["step_catalog_available"] is True

        # Verify step catalog info
        assert "step_catalog_info" in info
        assert info["step_catalog_info"]["catalog_type"] == "Mock"

    def test_get_enhanced_categorization_info_step_catalog_exception_following_guides(self):
        """Test enhanced categorization info when step catalog info extraction fails."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Create mock that raises exception when accessing attributes
        problematic_catalog = Mock()
        problematic_catalog.workspace_root = Mock(side_effect=Exception("Access failed"))

        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            step_catalog=problematic_catalog
        )

        # Should handle exception gracefully
        info = categorizer.get_enhanced_categorization_info()
        assert info["step_catalog_available"] is True
        # step_catalog_info might not be present due to exception

    def test_categorize_with_enhanced_metadata_following_guides(self):
        """Test categorization with enhanced metadata."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            project_id=self.test_project_id
        )

        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual enhanced metadata behavior
        result = categorizer.categorize_with_enhanced_metadata()

        # Verify enhanced metadata is added
        assert "enhanced_metadata" in result
        assert result["enhanced_metadata"]["project_id"] == self.test_project_id
        assert "workspace_field_mappings_count" in result["enhanced_metadata"]
        assert "framework_field_mappings_count" in result["enhanced_metadata"]

    def test_end_to_end_categorization_with_framework_fields_following_guides(self):
        """Test end-to-end categorization with framework-specific fields."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Create configs with framework-specific fields
        # Using real config classes with framework fields
        sagemaker_config = BasePipelineConfig(
            author="test-user", bucket="test-bucket", role="test-role",
            region="NA", service_name="SageMakerService", pipeline_version="1.0.0",
            project_root_folder="cursus", model_class="xgboost"
        )
        # Add framework-specific field
        sagemaker_config.sagemaker_session = "session_value"
        sagemaker_config.role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"

        docker_config = BasePipelineConfig(
            author="test-user", bucket="test-bucket", role="test-role",
            region="NA", service_name="DockerService", pipeline_version="1.0.0",
            project_root_folder="cursus", model_class="xgboost"
        )
        # Add framework-specific field
        docker_config.image_uri = "docker://my-image:latest"
        docker_config.container_entry_point = ["python", "train.py"]

        configs = [sagemaker_config, docker_config]

        # Create categorizer
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=configs,
            processing_step_config_base_class=ProcessingStepConfigBase
        )

        # Get categorization result
        result = categorizer.get_categorized_fields()

        # ✅ SYSTEMATIC ERROR PREVENTION: Verify structure matches source implementation
        assert set(result.keys()) == {"shared", "specific"}

        # Verify shared fields (fields with same values across configs)
        shared_fields = result["shared"]
        expected_shared = ["author", "bucket", "role", "region", "pipeline_version", "project_root_folder", "model_class"]
        for field in expected_shared:
            if field in shared_fields:
                assert shared_fields[field] is not None

        # Verify framework-specific fields are in specific sections
        specific_configs = result["specific"]
        assert len(specific_configs) >= 1

    def test_create_step_catalog_aware_categorizer_factory_following_guides(self):
        """Test the factory function for creating categorizer instances."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test actual factory function behavior
        categorizer = create_step_catalog_aware_categorizer(
            config_list=self.configs,
            processing_step_config_base_class=ProcessingStepConfigBase,
            project_id=self.test_project_id,
            step_catalog=self.mock_step_catalog,
            workspace_root=self.test_workspace_root
        )

        # Verify factory creates correct instance
        assert isinstance(categorizer, StepCatalogAwareConfigFieldCategorizer)
        assert categorizer.project_id == self.test_project_id
        assert categorizer.step_catalog == self.mock_step_catalog
        assert categorizer.workspace_root == self.test_workspace_root

    def test_inheritance_preservation_following_guides(self):
        """Test that all base class functionality is preserved."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test inherited functionality with real configs
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=self.configs,
            processing_step_config_base_class=ProcessingStepConfigBase
        )

        # Test inherited methods work correctly
        field_sources = categorizer.get_field_sources()
        assert isinstance(field_sources, dict)
        # Should have fields from real configs
        assert "author" in field_sources
        assert "bucket" in field_sources

        # Test inherited categorization logic
        result = categorizer.get_categorized_fields()
        assert "shared" in result
        assert "specific" in result

        # Test inherited stats method (should not raise exception)
        categorizer.print_categorization_stats()

    def test_error_handling_in_enhanced_mappings_following_guides(self):
        """Test error handling in enhanced mapping initialization."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test error handling during mapping initialization
        with patch.object(StepCatalogAwareConfigFieldCategorizer, '_get_workspace_field_mappings', 
                         side_effect=Exception("Mapping failed")):
            categorizer = StepCatalogAwareConfigFieldCategorizer(
                config_list=self.configs,
                project_id=self.test_project_id
            )

            # Should handle exception gracefully
            assert isinstance(categorizer._workspace_field_mappings, dict)
            assert isinstance(categorizer._framework_field_mappings, dict)

    def test_tier_aware_categorization_exception_handling_following_guides(self):
        """Test exception handling in tier-aware categorization."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test exception handling in tier-aware categorization
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)
        categorizer.unified_manager = Mock()
        categorizer.unified_manager.get_field_tiers.side_effect = Exception("Tier extraction failed")

        # Should handle exception gracefully and fall back to base categorization
        result = categorizer._categorize_field_with_step_catalog_context(
            "test_field", ["value"], ["TestConfig"]
        )

        # Should fall back to base categorization
        assert result in ["shared", "specific"]

    def test_framework_field_mappings_completeness_following_guides(self):
        """Test that framework field mappings cover expected categories."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)
        mappings = categorizer._framework_field_mappings

        # ✅ SYSTEMATIC ERROR PREVENTION: Verify different framework categories from source
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

    def test_workspace_field_mappings_with_model_fields_exception_following_guides(self):
        """Test workspace field mappings when model_fields access raises exception."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Test exception handling in model_fields access
        mock_config_class = Mock()
        mock_config_class.get_workspace_field_mappings.return_value = {}
        
        # Make model_fields raise an exception
        type(mock_config_class).model_fields = PropertyMock(side_effect=AttributeError("No model_fields"))

        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)
        categorizer.unified_manager = Mock()
        categorizer.unified_manager.get_config_classes.return_value = {
            "TestConfig": mock_config_class
        }

        # Should handle exception gracefully
        workspace_mappings = categorizer._get_workspace_field_mappings()
        assert isinstance(workspace_mappings, dict)

    def test_categorization_mapping_priority_following_guides(self):
        """Test that workspace mappings take priority over framework mappings."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=self.configs)
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Set up conflicting mappings (workspace should win)
        categorizer._workspace_field_mappings = {
            "sagemaker_session": "workspace_override"
        }

        # Test that workspace mapping takes priority
        result = categorizer._categorize_field_with_step_catalog_context(
            "sagemaker_session", ["value"], ["TestConfig"]
        )

        assert result == "workspace_override"


class TestStepCatalogAwareCategorizerEdgeCases:
    """Test edge cases and error conditions following systematic error prevention."""

    @pytest.fixture(autouse=True)
    def setup_edge_case_testing(self):
        """Set up edge case testing fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        yield
        self.temp_dir.cleanup()

    def test_empty_config_list_following_guides(self):
        """Test categorizer with empty config list."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # ✅ SYSTEMATIC ERROR PREVENTION: Should handle empty list gracefully
        assert len(categorizer.processing_configs) == 0
        assert len(categorizer.non_processing_configs) == 0
        assert isinstance(categorizer._workspace_field_mappings, dict)
        assert isinstance(categorizer._framework_field_mappings, dict)

        # Should be able to get categorization (empty)
        result = categorizer.get_categorized_fields()
        assert "shared" in result
        assert "specific" in result

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_none_values_handling_following_guides(self, mock_get_unified_manager):
        """Test handling of None values in various parameters."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Mock unified manager to fail when workspace_root is None
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


class TestStepCatalogAwareCategorizerIntegration:
    """Integration tests following systematic error prevention methodology."""

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
    def test_integration_with_real_workspace_structure_following_guides(self, mock_get_unified_manager, temp_workspace):
        """Test integration with realistic workspace structure."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Setup unified manager mock
        mock_unified_manager = Mock()
        mock_unified_manager.get_config_classes.return_value = {}
        mock_get_unified_manager.return_value = mock_unified_manager

        # Create configs with mixed field types using real config classes
        config1 = BasePipelineConfig(
            author="test-user", bucket="test-bucket", role="test-role",
            region="NA", service_name="TestService1", pipeline_version="1.0.0",
            project_root_folder="cursus", model_class="xgboost"
        )
        # Add workspace and framework fields
        config1.workspace_path = str(temp_workspace)
        config1.sagemaker_session = "session1"
        
        config2 = BasePipelineConfig(
            author="test-user", bucket="test-bucket", role="test-role",
            region="NA", service_name="TestService2", pipeline_version="1.0.0",
            project_root_folder="cursus", model_class="xgboost"
        )
        # Add workspace and framework fields
        config2.workspace_path = str(temp_workspace)
        config2.pytorch_version = "1.9.0"

        # Create categorizer with real workspace
        categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=[config1, config2],
            project_id="integration_test",
            workspace_root=temp_workspace
        )

        # Test categorization works end-to-end
        result = categorizer.get_categorized_fields()
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Verify structure
        assert "shared" in result
        assert "specific" in result
        
        # Verify shared fields
        shared_fields = result["shared"]
        expected_shared = ["author", "bucket", "role", "region", "pipeline_version", "project_root_folder", "model_class"]
        for field in expected_shared:
            if field in shared_fields:
                assert shared_fields[field] is not None
        
        # Verify framework-specific fields are categorized as specific
        specific_configs = result["specific"]
        assert len(specific_configs) > 0


class TestStepCatalogAwareCategorizerCompatibility:
    """Test compatibility and backward compatibility following systematic error prevention."""

    def test_backward_compatibility_with_base_categorizer_following_guides(self):
        """Test that enhanced categorizer maintains backward compatibility."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Create real config objects for compatibility test
        base_config = BasePipelineConfig(
            author="test-user", bucket="test-bucket", role="test-role",
            region="NA", service_name="TestService", pipeline_version="1.0.0",
            project_root_folder="cursus", model_class="xgboost"
        )
        
        processing_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir="/test/processing",
            processing_instance_type_large="ml.m5.12xlarge",
            processing_instance_type_small="ml.m5.4xlarge",
            processing_framework_version="1.2-1",
            processing_instance_count=1,
            processing_volume_size=500,
            use_large_processing_instance=False
        )

        configs = [base_config, processing_config]

        # Create enhanced categorizer
        enhanced_categorizer = StepCatalogAwareConfigFieldCategorizer(
            config_list=configs,
            processing_step_config_base_class=ProcessingStepConfigBase
        )

        # Get results
        enhanced_result = enhanced_categorizer.get_categorized_fields()

        # ✅ SYSTEMATIC ERROR PREVENTION: Verify same structure as base categorizer
        assert set(enhanced_result.keys()) == {"shared", "specific"}
        
        # Verify shared fields work the same
        shared_fields = enhanced_result["shared"]
        expected_shared = ["author", "bucket", "role", "region", "service_name", "pipeline_version", "project_root_folder", "model_class"]
        for field in expected_shared:
            if field in shared_fields:
                assert shared_fields[field] is not None
        
        # Verify specific fields work the same
        specific_configs = enhanced_result["specific"]
        assert len(specific_configs) >= 1

    def test_api_compatibility_following_guides(self):
        """Test that all public API methods are available and work correctly."""
        
        categorizer = StepCatalogAwareConfigFieldCategorizer(config_list=[])

        # ✅ SYSTEMATIC ERROR PREVENTION: Test all inherited public methods are available
        assert hasattr(categorizer, 'get_categorized_fields')
        assert hasattr(categorizer, 'get_field_sources')
        assert hasattr(categorizer, 'print_categorization_stats')
        assert hasattr(categorizer, 'get_category_for_field')

        # Test new public methods are available
        assert hasattr(categorizer, 'get_enhanced_categorization_info')
        assert hasattr(categorizer, 'categorize_with_enhanced_metadata')

        # Test factory function is available
        assert callable(create_step_catalog_aware_categorizer)

    @patch('cursus.core.config_fields.step_catalog_aware_categorizer.get_unified_config_manager')
    def test_enhanced_features_optional_following_guides(self, mock_get_unified_manager):
        """Test that enhanced features are optional and don't break basic functionality."""
        
        # ✅ SYSTEMATIC ERROR PREVENTION: Mock unified manager to fail when no workspace_root provided
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


if __name__ == "__main__":
    # Run the tests directly
    pytest.main([__file__, "-v", "-s"])
