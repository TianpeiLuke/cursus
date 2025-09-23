"""
Test hybrid path resolution configuration validation.

This module tests that configuration classes properly validate Tier 1 fields
required for hybrid path resolution and that hybrid resolution methods work correctly.
"""

import pytest
from pydantic import ValidationError

from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig


class TestHybridConfigValidation:
    """Test configuration validation for hybrid path resolution."""

    def test_base_configuration_creation(self):
        """Test that configuration can be created with project_root_folder."""
        # Test that configuration works with project_root_folder
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            project_root_folder="project_xgboost_pda",  # This field should be accepted
            source_dir="materials",
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Verify the field is set correctly
        assert config.project_root_folder == "project_xgboost_pda"
        assert config.source_dir == "materials"
        assert config.label_name == "is_abuse"

    def test_configuration_validation_works(self):
        """Test that configuration validation works correctly for processing steps."""
        # For processing steps, either source_dir or processing_source_dir is required
        # when processing_entry_point is provided, so this should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            TabularPreprocessingConfig(
                bucket="test-bucket",
                current_date="2025-09-22",
                region="NA",
                author="test-author",
                role="arn:aws:iam::123456789012:role/test-role",
                service_name="AtoZ",
                pipeline_version="1.0.0",
                framework_version="1.7-1",
                py_version="py3",
                project_root_folder="project_xgboost_pda",
                # source_dir omitted - should cause validation error for processing steps
                job_type="training",
                label_name="is_abuse",
                processing_entry_point="tabular_preprocessing.py"
            )
        
        # Check that the error mentions the source directory requirement
        error_str = str(exc_info.value)
        assert "source_dir" in error_str.lower() or "processing_source_dir" in error_str.lower()

    def test_field_categorization_basic(self):
        """Test that field categorization works for basic fields."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            project_root_folder="project_xgboost_pda",
            source_dir="materials",
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        categories = config.categorize_fields()
        
        # Check that basic required fields are in essential (Tier 1)
        assert "author" in categories["essential"]
        assert "bucket" in categories["essential"]
        assert "role" in categories["essential"]
        assert "region" in categories["essential"]
        assert "service_name" in categories["essential"]
        assert "pipeline_version" in categories["essential"]
        assert "label_name" in categories["essential"]
        
        # Check that fields with defaults are in system (Tier 2)
        assert "source_dir" in categories["system"]
        assert "current_date" in categories["system"]
        assert "framework_version" in categories["system"]
        
        # Check that derived properties are in derived (Tier 3)
        assert "aws_region" in categories["derived"]
        assert "pipeline_name" in categories["derived"]

    def test_project_root_folder_field_exists(self):
        """Test that project_root_folder field exists and can be set."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            project_root_folder="test_project",
            source_dir="materials",
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Verify the field exists and can be accessed
        assert hasattr(config, 'project_root_folder')
        assert config.project_root_folder == "test_project"
        
        # Note: project_root_folder is currently accepted as an extra field
        # This validates that the configuration system is ready for hybrid resolution integration

    def test_inheritance_chain_works(self):
        """Test that configuration inheritance chain works correctly."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            project_root_folder="project_xgboost_pda",
            source_dir="materials",
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Test that basic inheritance works
        assert hasattr(config, 'categorize_fields')
        assert hasattr(config, 'get_public_init_fields')
        assert hasattr(config, 'model_dump')
        
        # Test that the configuration can be serialized
        data = config.model_dump()
        assert 'project_root_folder' in data
        assert data['project_root_folder'] == "project_xgboost_pda"

    def test_configuration_ready_for_hybrid_resolution(self):
        """Test that configuration is ready for hybrid resolution integration."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            project_root_folder="project_xgboost_pda",
            source_dir="materials",
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Verify that the configuration has the necessary fields for hybrid resolution
        assert config.project_root_folder is not None
        assert config.source_dir is not None
        
        # Verify that the configuration can provide the information needed for path resolution
        public_fields = config.get_public_init_fields()
        assert 'project_root_folder' in public_fields
        assert 'source_dir' in public_fields
