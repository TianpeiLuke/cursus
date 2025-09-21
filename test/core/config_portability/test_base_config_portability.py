"""
Unit tests for BasePipelineConfig portable path functionality.

Tests the automatic path conversion and portability features added to
BasePipelineConfig as part of Phase 1 implementation.
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


class TestBasePipelineConfigPortability:
    """Test portable path functionality in BasePipelineConfig."""

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
            "pipeline_version": "1.0.0"
        }

    def test_portable_source_dir_conversion(self, temp_cursus_structure, sample_config_data):
        """Test automatic conversion of absolute to relative paths."""
        cursus_root = temp_cursus_structure["cursus_root"]
        dockers_dir = temp_cursus_structure["dockers_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        builders_dir = temp_cursus_structure["builders_dir"]
        
        # Mock inspect.getfile to return our test config location
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = BasePipelineConfig(
                source_dir=str(dockers_dir),
                **sample_config_data
            )
            
            # Should convert to relative path automatically
            portable_path = config.portable_source_dir
            
            # The path conversion should work and produce a relative path
            # Since dockers_dir is not a subdirectory of builders_dir, it will use common parent fallback
            assert portable_path is not None
            assert portable_path.startswith("../")
            assert portable_path.endswith("dockers/xgboost_atoz")

    def test_portable_path_with_relative_input(self, sample_config_data):
        """Test that relative paths are kept as-is."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the relative directory to avoid validation errors
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
                
                # Should keep relative path unchanged
                assert config.portable_source_dir == "./dockers/xgboost_atoz"
            finally:
                os.chdir(original_cwd)

    def test_portable_path_with_none_input(self, sample_config_data):
        """Test that None source_dir returns None portable path."""
        config = BasePipelineConfig(
            source_dir=None,
            **sample_config_data
        )
        
        # Should return None for None input
        assert config.portable_source_dir is None

    def test_portable_path_fallback_mechanism(self, temp_cursus_structure, sample_config_data):
        """Test fallback when direct conversion fails."""
        cursus_root = temp_cursus_structure["cursus_root"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        # Create a path that doesn't share structure with builders directory
        different_root = temp_cursus_structure["temp_dir"] / "different_root" / "dockers" / "xgboost_atoz"
        different_root.mkdir(parents=True)
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = BasePipelineConfig(
                source_dir=str(different_root),
                **sample_config_data
            )
            
            # Should use common parent fallback
            portable_path = config.portable_source_dir
            assert portable_path.startswith("../")
            assert portable_path.endswith("dockers/xgboost_atoz")

    def test_portable_path_final_fallback(self, sample_config_data):
        """Test final fallback returns original path when all conversions fail."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory to avoid validation errors
            test_path = Path(temp_dir) / "completely" / "different" / "path"
            test_path.mkdir(parents=True)
            
            with patch('inspect.getfile') as mock_getfile:
                # Mock to raise an exception
                mock_getfile.side_effect = Exception("Mock error")
                
                config = BasePipelineConfig(
                    source_dir=str(test_path),
                    **sample_config_data
                )
                
                # Should return original path as final fallback
                assert config.portable_source_dir == str(test_path)

    def test_serialization_includes_portable_paths(self, temp_cursus_structure, sample_config_data):
        """Test that serialization includes both original and portable paths."""
        cursus_root = temp_cursus_structure["cursus_root"]
        dockers_dir = temp_cursus_structure["dockers_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = BasePipelineConfig(
                source_dir=str(dockers_dir),
                **sample_config_data
            )
            
            data = config.model_dump()
            
            # Should include both original and portable paths
            assert data["source_dir"] == str(dockers_dir)
            # The portable path should be a relative path using common parent fallback
            portable_path = data["portable_source_dir"]
            assert portable_path.startswith("../")
            assert portable_path.endswith("dockers/xgboost_atoz")

    def test_serialization_excludes_none_portable_paths(self, sample_config_data):
        """Test that serialization excludes None portable paths."""
        config = BasePipelineConfig(
            source_dir=None,
            **sample_config_data
        )
        
        data = config.model_dump()
        
        # Should not include portable_source_dir when it's None
        assert "portable_source_dir" not in data
        assert data["source_dir"] is None

    def test_backward_compatibility(self, temp_cursus_structure, sample_config_data):
        """Test that existing functionality is unchanged."""
        dockers_dir = temp_cursus_structure["dockers_dir"]
        
        config = BasePipelineConfig(
            source_dir=str(dockers_dir),
            **sample_config_data
        )
        
        # Existing property should work exactly as before
        assert config.source_dir == str(dockers_dir)
        
        # All existing derived properties should work
        assert config.aws_region == "us-east-1"
        assert config.pipeline_name == "test_author-test_service-xgboost-NA"
        assert config.pipeline_description == "test_service xgboost Model NA"
        assert config.pipeline_s3_loc.startswith("s3://test-bucket/MODS/")

    def test_path_conversion_caching(self, temp_cursus_structure, sample_config_data):
        """Test that path conversion results are cached for performance."""
        cursus_root = temp_cursus_structure["cursus_root"]
        dockers_dir = temp_cursus_structure["dockers_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = BasePipelineConfig(
                source_dir=str(dockers_dir),
                **sample_config_data
            )
            
            # First access should trigger conversion
            first_result = config.portable_source_dir
            
            # Second access should use cached result
            second_result = config.portable_source_dir
            
            # Results should be identical
            assert first_result == second_result
            # Should be a relative path using common parent fallback
            assert first_result.startswith("../")
            assert first_result.endswith("dockers/xgboost_atoz")
            
            # Verify caching by checking private attribute
            assert config._portable_source_dir == first_result

    def test_find_common_parent_helper(self, temp_cursus_structure, sample_config_data):
        """Test the _find_common_parent helper method."""
        config = BasePipelineConfig(**sample_config_data)
        
        # Test with paths that have common parent
        path1 = Path("/home/user/cursus/src/cursus/steps/configs")
        path2 = Path("/home/user/cursus/dockers/xgboost_atoz")
        
        common_parent = config._find_common_parent(path1, path2)
        assert common_parent == Path("/home/user/cursus")
        
        # Test with paths that have no common parent
        path1 = Path("/home/user/cursus")
        path2 = Path("/different/root/path")
        
        common_parent = config._find_common_parent(path1, path2)
        assert common_parent == Path("/")

    def test_find_common_parent_with_exception(self, sample_config_data):
        """Test _find_common_parent handles exceptions gracefully."""
        config = BasePipelineConfig(**sample_config_data)
        
        # Test with invalid paths that might cause exceptions
        with patch('pathlib.Path.parts', side_effect=Exception("Mock error")):
            path1 = Path("/some/path")
            path2 = Path("/other/path")
            
            common_parent = config._find_common_parent(path1, path2)
            assert common_parent is None

    def test_convert_via_common_parent_method(self, temp_cursus_structure, sample_config_data):
        """Test the _convert_via_common_parent fallback method."""
        cursus_root = temp_cursus_structure["cursus_root"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = BasePipelineConfig(**sample_config_data)
            
            # Test conversion via common parent
            target_path = str(cursus_root / "dockers" / "xgboost_atoz")
            result = config._convert_via_common_parent(target_path)
            
            # Should create relative path using common parent
            assert result.startswith("../")
            assert result.endswith("dockers/xgboost_atoz")

    def test_convert_via_common_parent_with_exception(self, sample_config_data):
        """Test _convert_via_common_parent handles exceptions gracefully."""
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.side_effect = Exception("Mock error")
            
            config = BasePipelineConfig(**sample_config_data)
            
            original_path = "/some/path"
            result = config._convert_via_common_parent(original_path)
            
            # Should return original path when conversion fails
            assert result == original_path


class TestBasePipelineConfigPortabilityEdgeCases:
    """Test edge cases and error conditions for portable path functionality."""

    @pytest.fixture
    def sample_config_data(self):
        """Sample configuration data for testing."""
        return {
            "author": "test_author",
            "bucket": "test-bucket", 
            "role": "arn:aws:iam::123456789012:role/test-role",
            "region": "NA",
            "service_name": "test_service",
            "pipeline_version": "1.0.0"
        }

    def test_empty_string_source_dir(self, sample_config_data):
        """Test handling of empty string source_dir."""
        config = BasePipelineConfig(
            source_dir="",
            **sample_config_data
        )
        
        # Empty string should be returned as-is
        assert config.portable_source_dir == ""

    def test_s3_path_source_dir(self, sample_config_data):
        """Test handling of S3 paths (should be returned as-is)."""
        s3_path = "s3://my-bucket/path/to/source"
        config = BasePipelineConfig(
            source_dir=s3_path,
            **sample_config_data
        )
        
        # S3 paths are not absolute filesystem paths, should be returned as-is
        assert config.portable_source_dir == s3_path

    def test_windows_absolute_path(self, sample_config_data):
        """Test handling of Windows absolute paths."""
        if os.name == 'nt':  # Only test on Windows
            windows_path = r"C:\Users\test\cursus\dockers\xgboost_atoz"
            
            with patch('inspect.getfile') as mock_getfile:
                mock_getfile.return_value = r"C:\Users\test\cursus\src\cursus\steps\configs\test_config.py"
                
                config = BasePipelineConfig(
                    source_dir=windows_path,
                    **sample_config_data
                )
                
                # Should handle Windows paths correctly
                portable_path = config.portable_source_dir
                assert ".." in portable_path
                assert "dockers" in portable_path
                assert "xgboost_atoz" in portable_path

    def test_symlink_handling(self, sample_config_data):
        """Test handling of symbolic links in paths."""
        # This test would require creating actual symlinks
        # For now, we'll test that the conversion doesn't break with symlinks
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a regular directory
            source_dir = temp_path / "source"
            source_dir.mkdir()
            
            # Create a symlink (if supported by the system)
            try:
                symlink_dir = temp_path / "symlink_source"
                symlink_dir.symlink_to(source_dir)
                
                with patch('inspect.getfile') as mock_getfile:
                    mock_getfile.return_value = str(temp_path / "config.py")
                    
                    config = BasePipelineConfig(
                        source_dir=str(symlink_dir),
                        **sample_config_data
                    )
                    
                    # Should handle symlinks without crashing
                    portable_path = config.portable_source_dir
                    assert portable_path is not None
                    
            except (OSError, NotImplementedError):
                # Symlinks not supported on this system, skip test
                pytest.skip("Symlinks not supported on this system")

    def test_very_deep_path_structure(self, sample_config_data):
        """Test handling of very deep directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a very deep structure
            deep_path = temp_path
            for i in range(20):  # Create 20 levels deep
                deep_path = deep_path / f"level_{i}"
            deep_path.mkdir(parents=True)
            
            config_path = temp_path / "config.py"
            
            with patch('inspect.getfile') as mock_getfile:
                mock_getfile.return_value = str(config_path)
                
                config = BasePipelineConfig(
                    source_dir=str(deep_path),
                    **sample_config_data
                )
                
                # Should handle deep paths without issues
                portable_path = config.portable_source_dir
                assert portable_path is not None
                # For very deep paths, it might use the common parent approach
                # which could result in either "../" or the full relative path
                assert len(portable_path) > 0

    def test_unicode_paths(self, sample_config_data):
        """Test handling of paths with unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory with unicode characters
            unicode_dir = temp_path / "测试目录" / "ñoño" / "émoji_😀"
            unicode_dir.mkdir(parents=True)
            
            config_path = temp_path / "config.py"
            
            with patch('inspect.getfile') as mock_getfile:
                mock_getfile.return_value = str(config_path)
                
                config = BasePipelineConfig(
                    source_dir=str(unicode_dir),
                    **sample_config_data
                )
                
                # Should handle unicode paths correctly
                portable_path = config.portable_source_dir
                assert portable_path is not None
                assert "测试目录" in portable_path
                assert "ñoño" in portable_path
                assert "émoji_😀" in portable_path


if __name__ == "__main__":
    pytest.main([__file__])
