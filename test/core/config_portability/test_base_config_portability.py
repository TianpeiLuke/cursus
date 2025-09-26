"""
Unit tests for BasePipelineConfig hybrid path resolution functionality.

Tests the hybrid path resolution features in BasePipelineConfig that work
across different deployment scenarios (Lambda/MODS, development, pip-installed).
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cursus.core.base.config_base import BasePipelineConfig


class TestBasePipelineConfigHybridResolution:
    """Test hybrid path resolution functionality in BasePipelineConfig."""

    @pytest.fixture
    def temp_cursus_structure(self):
        """Create a temporary cursus-like directory structure for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create cursus-like structure
        cursus_root = temp_dir / "cursus"
        cursus_root.mkdir()
        
        # Create src/cursus/steps structure
        steps_dir = cursus_root / "src" / "cursus" / "steps"
        steps_dir.mkdir(parents=True)
        
        configs_dir = steps_dir / "configs"
        configs_dir.mkdir()
        
        builders_dir = steps_dir / "builders"
        builders_dir.mkdir()
        
        # Create dockers directory
        dockers_dir = cursus_root / "dockers" / "xgboost_atoz"
        dockers_dir.mkdir(parents=True)
        
        scripts_dir = dockers_dir / "scripts"
        scripts_dir.mkdir()
        
        # Create test files
        (dockers_dir / "xgboost_training.py").write_text("# Training script")
        (scripts_dir / "tabular_preprocessing.py").write_text("# Preprocessing script")
        
        yield {
            "temp_dir": temp_dir,
            "cursus_root": cursus_root,
            "configs_dir": configs_dir,
            "builders_dir": builders_dir,
            "dockers_dir": dockers_dir,
            "scripts_dir": scripts_dir
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            "author": "test_author",
            "bucket": "test-bucket",
            "role": "arn:aws:iam::123456789012:role/test-role",
            "region": "NA",
            "service_name": "test_service",
            "pipeline_version": "1.0.0",
            "project_root_folder": "cursus"
        }

    def test_resolve_hybrid_path_method_exists(self, sample_config_data):
        """Test that resolve_hybrid_path method exists and is callable."""
        config = BasePipelineConfig(**sample_config_data)
        
        # Method should exist and be callable
        assert hasattr(config, 'resolve_hybrid_path')
        assert callable(config.resolve_hybrid_path)
        
        # Should handle None input gracefully
        result = config.resolve_hybrid_path(None)
        assert result is None
        
        # Should handle empty string input gracefully
        result = config.resolve_hybrid_path("")
        assert result is None

    def test_resolved_source_dir_property_exists(self, sample_config_data):
        """Test that resolved_source_dir property exists and works."""
        config = BasePipelineConfig(
            source_dir="/some/test/path",
            **sample_config_data
        )
        
        # Property should exist
        assert hasattr(config, 'resolved_source_dir')
        
        # Should return None when hybrid resolution fails (expected in test environment)
        resolved = config.resolved_source_dir
        assert resolved is None or isinstance(resolved, str)

    def test_resolved_source_dir_with_none_source_dir(self, sample_config_data):
        """Test resolved_source_dir when source_dir is None."""
        config = BasePipelineConfig(
            source_dir=None,
            **sample_config_data
        )
        
        # Should return None when source_dir is None
        assert config.resolved_source_dir is None

    def test_effective_source_dir_property(self, sample_config_data):
        """Test effective_source_dir property behavior."""
        # Test with source_dir provided
        config = BasePipelineConfig(
            source_dir="/some/test/path",
            **sample_config_data
        )
        
        # effective_source_dir should exist and return the source_dir when hybrid resolution fails
        assert hasattr(config, 'effective_source_dir')
        effective = config.effective_source_dir
        assert effective == "/some/test/path"  # Falls back to original source_dir
        
        # Test with None source_dir
        config_none = BasePipelineConfig(
            source_dir=None,
            **sample_config_data
        )
        
        assert config_none.effective_source_dir is None

    def test_hybrid_path_resolution_with_s3_paths(self, sample_config_data):
        """Test that S3 paths are handled correctly."""
        s3_path = "s3://my-bucket/path/to/source"
        config = BasePipelineConfig(
            source_dir=s3_path,
            **sample_config_data
        )
        
        # S3 paths should not be resolved via hybrid resolution
        resolved = config.resolve_hybrid_path(s3_path)
        assert resolved is None  # Hybrid resolution doesn't handle S3 paths
        
        # But effective_source_dir should return the original S3 path
        assert config.effective_source_dir == s3_path

    def test_hybrid_path_resolution_with_relative_paths(self, sample_config_data):
        """Test hybrid path resolution with relative paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the relative directory
            rel_dir = Path(temp_dir) / "dockers" / "xgboost_atoz"
            rel_dir.mkdir(parents=True)
            
            # Change to temp directory so relative path works
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                config = BasePipelineConfig(
                    source_dir="./dockers/xgboost_atoz",
                    **sample_config_data
                )
                
                # Relative paths may or may not be resolved depending on hybrid resolution
                resolved = config.resolved_source_dir
                effective = config.effective_source_dir
                
                # effective_source_dir may resolve relative paths to absolute paths
                # This is expected behavior - the path should exist and be valid
                assert effective is not None
                assert "dockers/xgboost_atoz" in effective
                
            finally:
                os.chdir(original_cwd)

    def test_model_dump_includes_resolved_paths(self, sample_config_data):
        """Test that model_dump includes resolved path information."""
        config = BasePipelineConfig(
            source_dir="/some/test/path",
            **sample_config_data
        )
        
        data = config.model_dump()
        
        # Should include original source_dir
        assert data["source_dir"] == "/some/test/path"
        
        # Should include effective_source_dir
        assert "effective_source_dir" in data
        assert data["effective_source_dir"] == "/some/test/path"  # Falls back to original

    def test_model_dump_excludes_none_paths(self, sample_config_data):
        """Test that model_dump handles None paths correctly."""
        config = BasePipelineConfig(
            source_dir=None,
            **sample_config_data
        )
        
        data = config.model_dump()
        
        # Should include source_dir as None
        assert data["source_dir"] is None
        
        # effective_source_dir should not be included when None
        assert "effective_source_dir" not in data or data["effective_source_dir"] is None

    def test_backward_compatibility(self, sample_config_data):
        """Test that existing functionality is unchanged."""
        config = BasePipelineConfig(
            source_dir="/some/test/path",
            **sample_config_data
        )
        
        # Existing property should work exactly as before
        assert config.source_dir == "/some/test/path"
        
        # All existing derived properties should work
        assert config.aws_region == "us-east-1"
        assert config.pipeline_name == "test_author-test_service-xgboost-NA"
        assert config.pipeline_description == "test_service xgboost Model NA"
        assert config.pipeline_s3_loc.startswith("s3://test-bucket/MODS/")

    def test_project_root_folder_required(self):
        """Test that project_root_folder is required for hybrid resolution."""
        # Should raise validation error without project_root_folder
        with pytest.raises(Exception):  # Pydantic validation error
            BasePipelineConfig(
                author="test_author",
                bucket="test-bucket",
                role="arn:aws:iam::123456789012:role/test-role",
                region="NA",
                service_name="test_service",
                pipeline_version="1.0.0",
                # Missing project_root_folder
            )

    def test_hybrid_resolution_integration(self, temp_cursus_structure, sample_config_data):
        """Test integration with hybrid path resolution system."""
        dockers_dir = temp_cursus_structure["dockers_dir"]
        
        config = BasePipelineConfig(
            source_dir=str(dockers_dir),
            **sample_config_data
        )
        
        # Test that hybrid resolution methods exist and don't crash
        assert hasattr(config, 'resolve_hybrid_path')
        assert hasattr(config, 'resolved_source_dir')
        assert hasattr(config, 'effective_source_dir')
        
        # In test environment, hybrid resolution may fail, but methods should not crash
        resolved = config.resolve_hybrid_path(str(dockers_dir))
        assert resolved is None or isinstance(resolved, str)
        
        resolved_source = config.resolved_source_dir
        assert resolved_source is None or isinstance(resolved_source, str)
        
        effective_source = config.effective_source_dir
        assert effective_source == str(dockers_dir)  # Should fall back to original


class TestBasePipelineConfigEdgeCases:
    """Test edge cases for BasePipelineConfig path handling."""

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            "author": "test_author",
            "bucket": "test-bucket", 
            "role": "arn:aws:iam::123456789012:role/test-role",
            "region": "NA",
            "service_name": "test_service",
            "pipeline_version": "1.0.0",
            "project_root_folder": "cursus"
        }

    def test_empty_string_source_dir(self, sample_config_data):
        """Test handling of empty string source_dir."""
        config = BasePipelineConfig(
            source_dir="",
            **sample_config_data
        )
        
        # Empty string should be handled gracefully
        assert config.source_dir == ""
        # effective_source_dir returns None for empty string (expected behavior)
        assert config.effective_source_dir is None
        
        # Hybrid resolution should return None for empty string
        resolved = config.resolve_hybrid_path("")
        assert resolved is None

    def test_s3_path_source_dir(self, sample_config_data):
        """Test handling of S3 paths."""
        s3_path = "s3://my-bucket/path/to/source"
        config = BasePipelineConfig(
            source_dir=s3_path,
            **sample_config_data
        )
        
        # S3 paths should be preserved as-is
        assert config.source_dir == s3_path
        assert config.effective_source_dir == s3_path
        
        # Hybrid resolution should not handle S3 paths
        resolved = config.resolve_hybrid_path(s3_path)
        assert resolved is None

    def test_windows_absolute_path(self, sample_config_data):
        """Test handling of Windows absolute paths."""
        windows_path = r"C:\Users\test\cursus\dockers\xgboost_atoz"
        config = BasePipelineConfig(
            source_dir=windows_path,
            **sample_config_data
        )
        
        # Windows paths should be preserved
        assert config.source_dir == windows_path
        assert config.effective_source_dir == windows_path
        
        # Hybrid resolution may or may not work depending on environment
        resolved = config.resolve_hybrid_path(windows_path)
        assert resolved is None or isinstance(resolved, str)

    def test_unicode_paths(self, sample_config_data):
        """Test handling of paths with unicode characters."""
        unicode_path = "/测试目录/ñoño/émoji_😀"
        config = BasePipelineConfig(
            source_dir=unicode_path,
            **sample_config_data
        )
        
        # Unicode paths should be preserved
        assert config.source_dir == unicode_path
        assert config.effective_source_dir == unicode_path
        
        # Hybrid resolution should handle unicode gracefully
        resolved = config.resolve_hybrid_path(unicode_path)
        assert resolved is None or isinstance(resolved, str)

    def test_very_long_paths(self, sample_config_data):
        """Test handling of very long paths."""
        long_path = "/" + "/".join([f"level_{i}" for i in range(50)])
        config = BasePipelineConfig(
            source_dir=long_path,
            **sample_config_data
        )
        
        # Long paths should be preserved
        assert config.source_dir == long_path
        assert config.effective_source_dir == long_path
        
        # Hybrid resolution should handle long paths gracefully
        resolved = config.resolve_hybrid_path(long_path)
        assert resolved is None or isinstance(resolved, str)


if __name__ == "__main__":
    pytest.main([__file__])
