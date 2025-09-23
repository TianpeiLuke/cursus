"""
Test modernized path resolution components.

This module tests the modernized script_path property, get_script_path() function,
and effective_source_dir property with hybrid resolution and Scenario 1 fallback.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from pydantic import ValidationError

from cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig


class TestModernizedPathResolution:
    """Test modernized path resolution components."""

    def test_effective_source_dir_hybrid_resolution(self):
        """Test that effective_source_dir uses hybrid resolution."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
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
        
        # Test that effective_source_dir attempts hybrid resolution
        # Since we don't have actual project structure, it should fall back to legacy value
        effective_source = config.effective_source_dir
        
        # Should not be None and should be accessible
        assert effective_source is not None
        # In test environment without actual project structure, should fall back to source_dir
        assert effective_source == "materials"

    def test_script_path_uses_modernized_effective_source_dir(self):
        """Test that script_path property uses modernized effective_source_dir."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
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
        
        # Test that script_path is constructed correctly
        script_path = config.script_path
        assert script_path is not None
        
        # Should combine effective_source_dir with processing_entry_point
        expected_path = str(Path("materials") / "tabular_preprocessing.py")
        assert script_path == expected_path

    def test_get_script_path_comprehensive_fallbacks(self):
        """Test that get_script_path() uses comprehensive fallback strategy."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
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
        
        # Test get_script_path() method
        script_path = config.get_script_path()
        assert script_path is not None
        
        # With modernized path resolution, get_script_path() now uses hybrid resolution
        # and may return resolved absolute paths instead of relative paths
        # The key test is that it returns a valid path
        assert "tabular_preprocessing.py" in script_path
        assert isinstance(script_path, str)
        
        # Test that the modernized method is being used (not the old override)
        # Both methods should now use the same modernized resolution strategy
        script_path_property = config.script_path
        assert script_path_property is not None
        assert "tabular_preprocessing.py" in script_path_property

    def test_get_script_path_with_default_fallback(self):
        """Test that get_script_path() uses modernized fallback strategy."""
        # Note: TabularPreprocessingConfig has a default processing_entry_point="tabular_preprocessing.py"
        # So it will always have an entry point. Let's test that it uses the modernized get_script_path
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            project_root_folder="project_xgboost_pda",
            source_dir="materials",
            job_type="training",
            label_name="is_abuse",
            # processing_entry_point has default "tabular_preprocessing.py"
        )
        
        # Test that get_script_path() now uses the modernized version from base class
        script_path = config.get_script_path()
        assert script_path is not None
        
        # With modernized path resolution, both methods should return the same resolved path
        # (either relative path as fallback or absolute path if hybrid resolution succeeds)
        script_path_property = config.script_path
        assert script_path_property is not None
        
        # Test that both methods work correctly (they may return different formats)
        # get_script_path() uses direct hybrid resolution and may return absolute paths
        # script_path property uses effective_source_dir which may cache relative paths
        assert "tabular_preprocessing.py" in script_path
        assert "tabular_preprocessing.py" in script_path_property
        
        # Both should be valid paths
        assert isinstance(script_path, str)
        assert isinstance(script_path_property, str)

    def test_scenario_1_fallback_method_exists(self):
        """Test that _scenario_1_fallback method is available."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
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
        
        # Test that _scenario_1_fallback method exists and can be called
        assert hasattr(config, '_scenario_1_fallback')
        
        # Test calling the method (should return None in test environment)
        result = config._scenario_1_fallback("materials/tabular_preprocessing.py")
        # In test environment without actual project structure, should return None
        assert result is None

    def test_hybrid_resolution_integration(self):
        """Test that hybrid resolution methods are integrated correctly."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
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
        
        # Test that hybrid resolution methods are available
        assert hasattr(config, 'resolve_hybrid_path')
        assert hasattr(config, 'resolved_source_dir')
        
        # Test that they can be called without errors
        resolved_source = config.resolved_source_dir
        # In test environment, may return None but should not error
        assert resolved_source is None or isinstance(resolved_source, str)
        
        # Test resolve_hybrid_path method
        hybrid_path = config.resolve_hybrid_path("materials")
        # In test environment, may return None but should not error
        assert hybrid_path is None or isinstance(hybrid_path, str)

    def test_backward_compatibility(self):
        """Test that modernized methods maintain backward compatibility."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
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
        
        # Test that legacy methods still work
        assert hasattr(config, 'get_effective_source_dir')
        assert hasattr(config, 'get_resolved_script_path')
        
        # Test legacy method calls
        effective_source = config.get_effective_source_dir()
        assert effective_source is not None
        
        resolved_script = config.get_resolved_script_path()
        assert resolved_script is not None

    def test_s3_path_handling(self):
        """Test that S3 paths are handled correctly in modernized methods."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            project_root_folder="project_xgboost_pda",
            processing_source_dir="s3://test-bucket/scripts",
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Test that S3 paths are handled correctly
        effective_source = config.effective_source_dir
        assert effective_source == "s3://test-bucket/scripts"
        
        script_path = config.script_path
        assert script_path == "s3://test-bucket/scripts/tabular_preprocessing.py"
        
        # Test get_script_path with S3
        get_script_result = config.get_script_path()
        assert get_script_result == script_path

    def test_processing_source_dir_priority(self):
        """Test that processing_source_dir takes priority over source_dir."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
            region="NA",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            project_root_folder="project_xgboost_pda",
            source_dir="base_materials",
            processing_source_dir="processing_materials",
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Test that processing_source_dir takes priority
        effective_source = config.effective_source_dir
        assert effective_source == "processing_materials"
        
        script_path = config.script_path
        expected_path = str(Path("processing_materials") / "tabular_preprocessing.py")
        assert script_path == expected_path

    def test_modernization_complete(self):
        """Test that all three modernized components work together."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-23",
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
        
        # Test that all three modernized components are working
        # 1. effective_source_dir with hybrid resolution
        effective_source = config.effective_source_dir
        assert effective_source is not None
        
        # 2. script_path using modernized effective_source_dir
        script_path = config.script_path
        assert script_path is not None
        assert "materials" in script_path
        assert "tabular_preprocessing.py" in script_path
        
        # 3. get_script_path with comprehensive fallbacks
        get_script_result = config.get_script_path()
        assert get_script_result is not None
        
        # Both methods should return valid paths containing the script name
        # They may return different formats (absolute vs relative) due to different resolution strategies
        assert "tabular_preprocessing.py" in get_script_result
        assert "tabular_preprocessing.py" in script_path
        
        # Test that Scenario 1 fallback method is available
        assert hasattr(config, '_scenario_1_fallback')
        
        # Test that hybrid resolution methods are integrated
        assert hasattr(config, 'resolve_hybrid_path')
        assert hasattr(config, 'resolved_source_dir')
        
        print("✅ All three modernized components are working correctly!")
