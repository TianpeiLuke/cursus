"""
Unit tests for ProcessingStepConfigBase portable path functionality.

Tests the processing-specific portable path features added to
ProcessingStepConfigBase as part of Phase 1 implementation.
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

from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestProcessingStepConfigPortability:
    """Test portable path functionality in ProcessingStepConfigBase."""

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

    def test_portable_processing_source_dir_conversion(self, temp_cursus_structure, sample_config_data):
        """Test processing-specific portable path conversion."""
        cursus_root = temp_cursus_structure["cursus_root"]
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        # Mock inspect.getfile to return our test config location
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                processing_source_dir=str(scripts_dir),
                **sample_config_data
            )
            
            # Should convert to relative path automatically
            portable_path = config.portable_processing_source_dir
            assert portable_path is not None
            assert portable_path.startswith("../")
            assert portable_path.endswith("dockers/xgboost_atoz/scripts")

    def test_portable_processing_source_dir_with_none(self, sample_config_data):
        """Test that None processing_source_dir returns None portable path."""
        config = ProcessingStepConfigBase(
            processing_source_dir=None,
            **sample_config_data
        )
        
        # Should return None for None input
        assert config.portable_processing_source_dir is None

    def test_portable_effective_source_dir_processing_priority(self, temp_cursus_structure, sample_config_data):
        """Test that portable_effective_source_dir prioritizes processing_source_dir."""
        cursus_root = temp_cursus_structure["cursus_root"]
        dockers_dir = temp_cursus_structure["dockers_dir"]
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                source_dir=str(dockers_dir),
                processing_source_dir=str(scripts_dir),
                **sample_config_data
            )
            
            # Should prioritize processing_source_dir over source_dir
            portable_path = config.portable_effective_source_dir
            assert portable_path.startswith("../")
            assert portable_path.endswith("dockers/xgboost_atoz/scripts")

    def test_portable_effective_source_dir_fallback_to_base(self, temp_cursus_structure, sample_config_data):
        """Test that portable_effective_source_dir falls back to base source_dir."""
        cursus_root = temp_cursus_structure["cursus_root"]
        dockers_dir = temp_cursus_structure["dockers_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                source_dir=str(dockers_dir),
                processing_source_dir=None,
                **sample_config_data
            )
            
            # Should fall back to base source_dir
            portable_path = config.portable_effective_source_dir
            assert portable_path.startswith("../")
            assert portable_path.endswith("dockers/xgboost_atoz")

    def test_portable_script_path_generation(self, temp_cursus_structure, sample_config_data):
        """Test portable script path generation."""
        cursus_root = temp_cursus_structure["cursus_root"]
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                processing_source_dir=str(scripts_dir),
                processing_entry_point="tabular_preprocessing.py",
                **sample_config_data
            )
            
            portable_script = config.get_portable_script_path()
            assert portable_script is not None
            assert portable_script.startswith("../")
            assert portable_script.endswith("dockers/xgboost_atoz/scripts/tabular_preprocessing.py")

    def test_portable_script_path_with_no_entry_point(self, temp_cursus_structure, sample_config_data):
        """Test portable script path when no entry point is provided."""
        scripts_dir = temp_cursus_structure["scripts_dir"]
        
        config = ProcessingStepConfigBase(
            processing_source_dir=str(scripts_dir),
            processing_entry_point=None,
            **sample_config_data
        )
        
        # Should return None when no entry point
        assert config.get_portable_script_path() is None

    def test_portable_script_path_with_default(self, temp_cursus_structure, sample_config_data):
        """Test portable script path with default path parameter."""
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                processing_source_dir=str(scripts_dir),
                processing_entry_point=None,
                **sample_config_data
            )
            
            default_path = str(scripts_dir / "default_script.py")
            portable_script = config.get_portable_script_path(default_path)
            
            # Should convert the default path
            assert portable_script is not None
            assert portable_script.startswith("../")
            assert portable_script.endswith("dockers/xgboost_atoz/scripts/default_script.py")

    def test_serialization_includes_portable_paths(self, temp_cursus_structure, sample_config_data):
        """Test that serialization includes processing-specific portable paths."""
        cursus_root = temp_cursus_structure["cursus_root"]
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                processing_source_dir=str(scripts_dir),
                processing_entry_point="tabular_preprocessing.py",
                **sample_config_data
            )
            
            data = config.model_dump()
            
            # Should include both original and portable paths
            assert data["processing_source_dir"] == str(scripts_dir)
            portable_processing_path = data["portable_processing_source_dir"]
            assert portable_processing_path.startswith("../")
            assert portable_processing_path.endswith("dockers/xgboost_atoz/scripts")
            
            portable_script_path = data["portable_script_path"]
            assert portable_script_path.startswith("../")
            assert portable_script_path.endswith("dockers/xgboost_atoz/scripts/tabular_preprocessing.py")

    def test_serialization_excludes_none_portable_paths(self, sample_config_data):
        """Test that serialization excludes None portable paths."""
        config = ProcessingStepConfigBase(
            processing_source_dir=None,
            processing_entry_point=None,
            **sample_config_data
        )
        
        data = config.model_dump()
        
        # Should not include portable paths when they're None
        assert "portable_processing_source_dir" not in data
        assert "portable_script_path" not in data

    def test_backward_compatibility(self, temp_cursus_structure, sample_config_data):
        """Test that existing functionality is unchanged."""
        scripts_dir = temp_cursus_structure["scripts_dir"]
        
        config = ProcessingStepConfigBase(
            processing_source_dir=str(scripts_dir),
            processing_entry_point="tabular_preprocessing.py",
            **sample_config_data
        )
        
        # Existing properties should work exactly as before
        assert config.processing_source_dir == str(scripts_dir)
        assert config.processing_entry_point == "tabular_preprocessing.py"
        assert config.effective_source_dir == str(scripts_dir)
        assert config.script_path == str(scripts_dir / "tabular_preprocessing.py")
        
        # All base class properties should work
        assert config.aws_region == "us-east-1"
        assert config.pipeline_name == "test_author-test_service-xgboost-NA"

    def test_portable_path_caching(self, temp_cursus_structure, sample_config_data):
        """Test that portable path conversion results are cached for performance."""
        cursus_root = temp_cursus_structure["cursus_root"]
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                processing_source_dir=str(scripts_dir),
                processing_entry_point="tabular_preprocessing.py",
                **sample_config_data
            )
            
            # First access should trigger conversion
            first_result = config.portable_processing_source_dir
            
            # Second access should use cached result
            second_result = config.portable_processing_source_dir
            
            # Results should be identical
            assert first_result == second_result
            assert first_result.startswith("../")
            assert first_result.endswith("dockers/xgboost_atoz/scripts")
            
            # Verify caching by checking private attribute
            assert config._portable_processing_source_dir == first_result

    def test_script_path_caching(self, temp_cursus_structure, sample_config_data):
        """Test that portable script path conversion results are cached."""
        cursus_root = temp_cursus_structure["cursus_root"]
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                processing_source_dir=str(scripts_dir),
                processing_entry_point="tabular_preprocessing.py",
                **sample_config_data
            )
            
            # First access should trigger conversion
            first_result = config.get_portable_script_path()
            
            # Second access should use cached result
            second_result = config.get_portable_script_path()
            
            # Results should be identical
            assert first_result == second_result
            assert first_result.startswith("../")
            assert first_result.endswith("dockers/xgboost_atoz/scripts/tabular_preprocessing.py")
            
            # Verify caching by checking private attribute
            assert config._portable_script_path == first_result

    def test_s3_processing_source_dir_handling(self, sample_config_data):
        """Test handling of S3 processing source directories."""
        s3_path = "s3://my-bucket/processing/scripts"
        config = ProcessingStepConfigBase(
            processing_source_dir=s3_path,
            **sample_config_data
        )
        
        # S3 paths should be returned as-is
        assert config.portable_processing_source_dir == s3_path

    def test_s3_script_path_handling(self, sample_config_data):
        """Test handling of S3 script paths."""
        s3_path = "s3://my-bucket/processing/scripts"
        config = ProcessingStepConfigBase(
            processing_source_dir=s3_path,
            processing_entry_point="preprocessing.py",
            **sample_config_data
        )
        
        # S3 script paths should be returned as-is
        portable_script = config.get_portable_script_path()
        assert portable_script == "s3://my-bucket/processing/scripts/preprocessing.py"

    def test_mixed_base_and_processing_paths(self, temp_cursus_structure, sample_config_data):
        """Test configuration with both base and processing-specific paths."""
        cursus_root = temp_cursus_structure["cursus_root"]
        dockers_dir = temp_cursus_structure["dockers_dir"]
        scripts_dir = temp_cursus_structure["scripts_dir"]
        configs_dir = temp_cursus_structure["configs_dir"]
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                source_dir=str(dockers_dir),
                processing_source_dir=str(scripts_dir),
                processing_entry_point="tabular_preprocessing.py",
                **sample_config_data
            )
            
            data = config.model_dump()
            
            # Should include both base and processing portable paths
            base_portable = data["portable_source_dir"]
            assert base_portable.startswith("../")
            assert base_portable.endswith("dockers/xgboost_atoz")
            
            processing_portable = data["portable_processing_source_dir"]
            assert processing_portable.startswith("../")
            assert processing_portable.endswith("dockers/xgboost_atoz/scripts")
            
            script_portable = data["portable_script_path"]
            assert script_portable.startswith("../")
            assert script_portable.endswith("dockers/xgboost_atoz/scripts/tabular_preprocessing.py")

    def test_processing_path_fallback_mechanisms(self, temp_cursus_structure, sample_config_data):
        """Test fallback mechanisms for processing path conversion."""
        configs_dir = temp_cursus_structure["configs_dir"]
        
        # Create a path that doesn't share structure with builders directory
        different_root = temp_cursus_structure["temp_dir"] / "different_root" / "scripts"
        different_root.mkdir(parents=True)
        
        with patch('inspect.getfile') as mock_getfile:
            mock_getfile.return_value = str(configs_dir / "test_config.py")
            
            config = ProcessingStepConfigBase(
                processing_source_dir=str(different_root),
                **sample_config_data
            )
            
            # Should use common parent fallback
            portable_path = config.portable_processing_source_dir
            assert portable_path.startswith("../")
            assert portable_path.endswith("scripts")

    def test_processing_path_final_fallback(self, sample_config_data):
        """Test final fallback for processing path conversion."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory to avoid validation errors
            test_path = Path(temp_dir) / "completely" / "different" / "processing" / "path"
            test_path.mkdir(parents=True)
            
            with patch('inspect.getfile') as mock_getfile:
                # Mock to raise an exception
                mock_getfile.side_effect = Exception("Mock error")
                
                config = ProcessingStepConfigBase(
                    processing_source_dir=str(test_path),
                    **sample_config_data
                )
                
                # Should return original path as final fallback
                assert config.portable_processing_source_dir == str(test_path)


class TestProcessingStepConfigPortabilityEdgeCases:
    """Test edge cases and error conditions for processing portable path functionality."""

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

    def test_empty_processing_entry_point(self, sample_config_data):
        """Test handling of empty processing entry point."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scripts_dir = Path(temp_dir) / "scripts"
            scripts_dir.mkdir()
            
            # Empty entry point is not allowed by validation, so test with None instead
            config = ProcessingStepConfigBase(
                processing_source_dir=str(scripts_dir),
                processing_entry_point=None,
                **sample_config_data
            )
            
            # None entry point should result in None script path
            assert config.get_portable_script_path() is None

    def test_relative_processing_source_dir(self, sample_config_data):
        """Test handling of relative processing source directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the relative directory to avoid validation errors
            rel_dir = Path(temp_dir) / "scripts"
            rel_dir.mkdir()
            
            # Change to temp directory so relative path works
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                config = ProcessingStepConfigBase(
                    processing_source_dir="./scripts",
                    **sample_config_data
                )
                
                # Relative paths should be kept as-is
                assert config.portable_processing_source_dir == "./scripts"
            finally:
                os.chdir(original_cwd)

    def test_processing_path_with_special_characters(self, sample_config_data):
        """Test handling of paths with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory with special characters
            special_dir = temp_path / "scripts with spaces" / "special-chars_123"
            special_dir.mkdir(parents=True)
            
            config_path = temp_path / "config.py"
            
            with patch('inspect.getfile') as mock_getfile:
                mock_getfile.return_value = str(config_path)
                
                config = ProcessingStepConfigBase(
                    processing_source_dir=str(special_dir),
                    **sample_config_data
                )
                
                # Should handle special characters correctly
                portable_path = config.portable_processing_source_dir
                assert portable_path is not None
                assert "scripts with spaces" in portable_path
                assert "special-chars_123" in portable_path

    def test_very_long_processing_entry_point(self, sample_config_data):
        """Test handling of very long entry point names."""
        with tempfile.TemporaryDirectory() as temp_dir:
            scripts_dir = Path(temp_dir) / "scripts"
            scripts_dir.mkdir()
            
            # Create a very long filename
            long_filename = "very_long_processing_script_name_" + "x" * 200 + ".py"
            long_script = scripts_dir / long_filename
            long_script.write_text("# Long script")
            
            config_path = Path(temp_dir) / "config.py"
            
            with patch('inspect.getfile') as mock_getfile:
                mock_getfile.return_value = str(config_path)
                
                config = ProcessingStepConfigBase(
                    processing_source_dir=str(scripts_dir),
                    processing_entry_point=long_filename,
                    **sample_config_data
                )
                
                # Should handle long filenames correctly
                portable_script = config.get_portable_script_path()
                assert portable_script is not None
                assert long_filename in portable_script

    def test_processing_config_inheritance(self, sample_config_data):
        """Test that processing config properly inherits base portable functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create both base and processing directories
            base_dir = temp_path / "base"
            base_dir.mkdir()
            
            processing_dir = temp_path / "processing"
            processing_dir.mkdir()
            
            config_path = temp_path / "config.py"
            
            with patch('inspect.getfile') as mock_getfile:
                mock_getfile.return_value = str(config_path)
                
                config = ProcessingStepConfigBase(
                    source_dir=str(base_dir),
                    processing_source_dir=str(processing_dir),
                    **sample_config_data
                )
                
                # Should have both base and processing portable paths
                assert config.portable_source_dir is not None
                assert config.portable_processing_source_dir is not None
                assert config.portable_source_dir != config.portable_processing_source_dir

    def test_processing_config_serialization_completeness(self, sample_config_data):
        """Test that processing config serialization includes all portable paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            base_dir = temp_path / "base"
            base_dir.mkdir()
            
            processing_dir = temp_path / "processing"
            processing_dir.mkdir()
            
            script_file = processing_dir / "test_script.py"
            script_file.write_text("# Test script")
            
            config_path = temp_path / "config.py"
            
            with patch('inspect.getfile') as mock_getfile:
                mock_getfile.return_value = str(config_path)
                
                config = ProcessingStepConfigBase(
                    source_dir=str(base_dir),
                    processing_source_dir=str(processing_dir),
                    processing_entry_point="test_script.py",
                    **sample_config_data
                )
                
                data = config.model_dump()
                
                # Should include all portable paths
                portable_fields = [k for k in data.keys() if k.startswith("portable_")]
                assert "portable_source_dir" in portable_fields
                assert "portable_processing_source_dir" in portable_fields
                assert "portable_script_path" in portable_fields
                
                # All portable paths should be non-None
                for field in portable_fields:
                    assert data[field] is not None


if __name__ == "__main__":
    pytest.main([__file__])
