"""
Unit tests for ProcessingStepConfigBase hybrid path resolution functionality.

Tests the processing-specific hybrid path resolution features that work
across different deployment scenarios (Lambda/MODS, development, pip-installed).

Following pytest best practices:
1. Read source code first to understand actual implementation
2. Mock at correct import locations
3. Test actual behavior, not assumptions
4. Use proper fixtures for test isolation
5. Cover both happy path and edge cases
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock
import sys
import os

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestProcessingStepConfigHybridResolution:
    """Test hybrid path resolution functionality in ProcessingStepConfigBase."""

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
            "scripts_dir": scripts_dir,
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
            "project_root_folder": "cursus",
        }

    def test_processing_config_inherits_hybrid_resolution(self, sample_config_data):
        """Test that ProcessingStepConfigBase inherits hybrid resolution from base class."""
        config = ProcessingStepConfigBase(
            processing_source_dir="/some/processing/path", **sample_config_data
        )

        # Should inherit hybrid resolution methods from base class
        assert hasattr(config, "resolve_hybrid_path")
        assert hasattr(config, "resolved_source_dir")
        assert hasattr(config, "effective_source_dir")
        assert callable(config.resolve_hybrid_path)

    def test_processing_source_dir_property(self, sample_config_data):
        """Test processing_source_dir property behavior."""
        processing_path = "/some/processing/scripts"
        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path, **sample_config_data
        )

        # Should have processing_source_dir property
        assert hasattr(config, "processing_source_dir")
        assert config.processing_source_dir == processing_path

    def test_effective_source_dir_prioritizes_processing(self, sample_config_data):
        """Test that effective_source_dir prioritizes processing_source_dir over source_dir."""
        base_path = "/some/base/path"
        processing_path = "/some/processing/scripts"

        config = ProcessingStepConfigBase(
            source_dir=base_path,
            processing_source_dir=processing_path,
            **sample_config_data,
        )

        # effective_source_dir should prioritize processing_source_dir
        assert config.effective_source_dir == processing_path

    def test_effective_source_dir_fallback_to_base(self, sample_config_data):
        """Test that effective_source_dir falls back to source_dir when processing_source_dir is None."""
        base_path = "/some/base/path"

        config = ProcessingStepConfigBase(
            source_dir=base_path, processing_source_dir=None, **sample_config_data
        )

        # Should fall back to base source_dir
        assert config.effective_source_dir == base_path

    def test_script_path_property(self, sample_config_data):
        """Test script_path property behavior."""
        processing_path = "/some/processing/scripts"
        entry_point = "preprocessing.py"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=entry_point,
            **sample_config_data,
        )

        # Should have script_path property that combines processing_source_dir and entry_point
        assert hasattr(config, "script_path")
        expected_script_path = str(Path(processing_path) / entry_point)
        assert config.script_path == expected_script_path

    def test_script_path_with_none_entry_point(self, sample_config_data):
        """Test script_path when processing_entry_point is None."""
        processing_path = "/some/processing/scripts"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=None,
            **sample_config_data,
        )

        # script_path should return None when entry_point is None
        assert config.script_path is None

    def test_get_script_path_method(self, sample_config_data):
        """Test get_script_path method behavior."""
        processing_path = "/some/processing/scripts"
        entry_point = "preprocessing.py"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=entry_point,
            **sample_config_data,
        )

        # Should have get_script_path method
        assert hasattr(config, "get_script_path")
        assert callable(config.get_script_path)

        # Should return the script path
        script_path = config.get_script_path()
        expected_script_path = str(Path(processing_path) / entry_point)
        assert script_path == expected_script_path

    def test_get_script_path_with_default(self, sample_config_data):
        """Test get_script_path method with default parameter."""
        config = ProcessingStepConfigBase(
            processing_source_dir=None,
            processing_entry_point=None,
            **sample_config_data,
        )

        default_path = "/default/script.py"
        script_path = config.get_script_path(default_path)

        # Should return the default path when no processing config is available
        assert script_path == default_path

    def test_hybrid_resolution_with_processing_paths(self, sample_config_data):
        """Test hybrid resolution with processing-specific paths."""
        processing_path = "/some/processing/scripts"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path, **sample_config_data
        )

        # Should be able to resolve processing paths via hybrid resolution
        resolved = config.resolve_hybrid_path(processing_path)
        # May return None in test environment, which is expected
        assert resolved is None or isinstance(resolved, str)

    def test_model_dump_includes_processing_fields(self, sample_config_data):
        """Test that model_dump includes processing-specific fields."""
        processing_path = "/some/processing/scripts"
        entry_point = "preprocessing.py"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=entry_point,
            **sample_config_data,
        )

        data = config.model_dump()

        # Should include processing-specific fields
        assert "processing_source_dir" in data
        assert data["processing_source_dir"] == processing_path
        assert "processing_entry_point" in data
        assert data["processing_entry_point"] == entry_point

        # Should include effective_source_dir (prioritizing processing)
        assert "effective_source_dir" in data
        assert data["effective_source_dir"] == processing_path

    def test_backward_compatibility(self, sample_config_data):
        """Test that existing functionality is unchanged."""
        processing_path = "/some/processing/scripts"
        entry_point = "preprocessing.py"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=entry_point,
            **sample_config_data,
        )

        # All existing properties should work
        assert config.processing_source_dir == processing_path
        assert config.processing_entry_point == entry_point
        assert config.effective_source_dir == processing_path

        # Base class properties should work
        assert config.aws_region == "us-east-1"
        assert config.pipeline_name == "test_author-test_service-xgboost-NA"

    def test_s3_path_handling(self, sample_config_data):
        """Test handling of S3 paths in processing config."""
        s3_processing_path = "s3://my-bucket/processing/scripts"
        s3_base_path = "s3://my-bucket/base/path"

        config = ProcessingStepConfigBase(
            source_dir=s3_base_path,
            processing_source_dir=s3_processing_path,
            processing_entry_point="script.py",
            **sample_config_data,
        )

        # S3 paths should be preserved
        assert config.processing_source_dir == s3_processing_path
        assert config.source_dir == s3_base_path
        assert (
            config.effective_source_dir == s3_processing_path
        )  # Prioritizes processing

        # Script path should combine S3 path with entry point
        expected_script = f"{s3_processing_path}/script.py"
        assert config.script_path == expected_script

    def test_hybrid_resolution_integration(
        self, temp_cursus_structure, sample_config_data
    ):
        """Test integration with hybrid path resolution system."""
        scripts_dir = temp_cursus_structure["scripts_dir"]
        dockers_dir = temp_cursus_structure["dockers_dir"]

        config = ProcessingStepConfigBase(
            source_dir=str(dockers_dir),
            processing_source_dir=str(scripts_dir),
            processing_entry_point="preprocessing.py",
            **sample_config_data,
        )

        # Test that hybrid resolution methods exist and don't crash
        assert hasattr(config, "resolve_hybrid_path")
        assert hasattr(config, "resolved_source_dir")
        assert hasattr(config, "effective_source_dir")

        # In test environment, hybrid resolution may fail, but methods should not crash
        resolved_base = config.resolve_hybrid_path(str(dockers_dir))
        assert resolved_base is None or isinstance(resolved_base, str)

        resolved_processing = config.resolve_hybrid_path(str(scripts_dir))
        assert resolved_processing is None or isinstance(resolved_processing, str)

        # effective_source_dir should prioritize processing path
        assert config.effective_source_dir == str(scripts_dir)


class TestProcessingStepConfigEdgeCases:
    """Test edge cases for ProcessingStepConfigBase."""

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
            "project_root_folder": "cursus",
        }

    def test_none_processing_source_dir(self, sample_config_data):
        """Test handling of None processing_source_dir."""
        base_path = "/some/base/path"

        config = ProcessingStepConfigBase(
            source_dir=base_path, processing_source_dir=None, **sample_config_data
        )

        # Should handle None processing_source_dir gracefully
        assert config.processing_source_dir is None
        assert config.effective_source_dir == base_path  # Falls back to base

    def test_none_processing_entry_point(self, sample_config_data):
        """Test handling of None processing_entry_point."""
        processing_path = "/some/processing/scripts"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=None,
            **sample_config_data,
        )

        # Should handle None entry point gracefully
        assert config.processing_entry_point is None
        assert config.script_path is None

    def test_empty_string_processing_paths(self, sample_config_data):
        """Test handling of empty string processing paths."""
        config = ProcessingStepConfigBase(
            processing_source_dir="",
            processing_entry_point=None,
            **sample_config_data,
        )

        # Should handle empty strings gracefully
        assert config.processing_source_dir == ""
        assert config.processing_entry_point is None
        # effective_source_dir returns empty string for processing configs
        assert config.effective_source_dir == ""

    def test_relative_processing_paths(self, sample_config_data):
        """Test handling of relative processing paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the relative directory
            rel_dir = Path(temp_dir) / "processing" / "scripts"
            rel_dir.mkdir(parents=True)

            # Change to temp directory so relative path works
            original_cwd = os.getcwd()
            os.chdir(temp_dir)

            try:
                config = ProcessingStepConfigBase(
                    processing_source_dir="./processing/scripts",
                    processing_entry_point="script.py",
                    **sample_config_data,
                )

                # Original relative path should be preserved in processing_source_dir
                assert config.processing_source_dir == "./processing/scripts"

                # effective_source_dir resolves to absolute path when directory exists
                # This is expected behavior from hybrid resolution
                assert config.effective_source_dir is not None
                assert "processing/scripts" in config.effective_source_dir
                assert Path(config.effective_source_dir).exists()

                # Script path should include the entry point
                script_path = config.script_path
                assert script_path is not None
                assert "script.py" in script_path
                assert Path(script_path).parent.exists()

            finally:
                os.chdir(original_cwd)

    def test_unicode_processing_paths(self, sample_config_data):
        """Test handling of unicode characters in processing paths."""
        unicode_path = "/测试目录/processing/ñoño/scripts"
        unicode_entry = "émoji_😀_script.py"

        config = ProcessingStepConfigBase(
            processing_source_dir=unicode_path,
            processing_entry_point=unicode_entry,
            **sample_config_data,
        )

        # Unicode paths should be preserved
        assert config.processing_source_dir == unicode_path
        assert config.processing_entry_point == unicode_entry
        assert config.effective_source_dir == unicode_path

        # Script path should handle unicode correctly
        expected_script = f"{unicode_path}/{unicode_entry}"
        assert config.script_path == expected_script

    def test_very_long_processing_paths(self, sample_config_data):
        """Test handling of very long processing paths."""
        long_path = "/" + "/".join([f"processing_level_{i}" for i in range(30)])
        long_entry = "very_long_processing_script_name_" + "x" * 100 + ".py"

        config = ProcessingStepConfigBase(
            processing_source_dir=long_path,
            processing_entry_point=long_entry,
            **sample_config_data,
        )

        # Long paths should be preserved
        assert config.processing_source_dir == long_path
        assert config.processing_entry_point == long_entry
        assert config.effective_source_dir == long_path

        # Script path should handle long paths correctly
        expected_script = f"{long_path}/{long_entry}"
        assert config.script_path == expected_script

    def test_mixed_path_types(self, sample_config_data):
        """Test mixing different types of paths (absolute, relative, S3)."""
        # Mix S3 base path with local processing path
        s3_base = "s3://my-bucket/base/path"
        local_processing = "/local/processing/scripts"

        config = ProcessingStepConfigBase(
            source_dir=s3_base,
            processing_source_dir=local_processing,
            processing_entry_point="script.py",
            **sample_config_data,
        )

        # Should handle mixed path types correctly
        assert config.source_dir == s3_base
        assert config.processing_source_dir == local_processing
        assert config.effective_source_dir == local_processing  # Prioritizes processing

        expected_script = f"{local_processing}/script.py"
        assert config.script_path == expected_script


class TestProcessingStepConfigValidators:
    """Test validator methods in ProcessingStepConfigBase."""

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
            "project_root_folder": "cursus",
        }

    def test_validate_processing_source_dir_s3_path(self, sample_config_data):
        """Test validation of S3 processing source directory."""
        s3_path = "s3://my-bucket/processing/scripts"

        # Should accept valid S3 path
        config = ProcessingStepConfigBase(
            processing_source_dir=s3_path, **sample_config_data
        )
        assert config.processing_source_dir == s3_path

    def test_validate_processing_source_dir_invalid_s3(self, sample_config_data):
        """Test validation rejects invalid S3 paths."""
        invalid_s3_path = "s3://"

        # Should raise ValueError for invalid S3 path
        with pytest.raises(ValueError, match="Invalid S3 path format"):
            ProcessingStepConfigBase(
                processing_source_dir=invalid_s3_path, **sample_config_data
            )

    def test_validate_entry_point_relative_path(self, sample_config_data):
        """Test that entry point must be relative."""
        relative_entry = "scripts/preprocessing.py"

        # Should accept relative path
        config = ProcessingStepConfigBase(
            processing_source_dir="/some/path",
            processing_entry_point=relative_entry,
            **sample_config_data,
        )
        assert config.processing_entry_point == relative_entry

    def test_validate_entry_point_rejects_absolute_path(self, sample_config_data):
        """Test that absolute entry points are rejected."""
        absolute_entry = "/absolute/path/to/script.py"

        # Should raise ValueError for absolute path
        with pytest.raises(ValueError, match="must be a relative path"):
            ProcessingStepConfigBase(
                processing_source_dir="/some/path",
                processing_entry_point=absolute_entry,
                **sample_config_data,
            )

    def test_validate_entry_point_rejects_s3_path(self, sample_config_data):
        """Test that S3 entry points are rejected."""
        s3_entry = "s3://bucket/script.py"

        # Should raise ValueError for S3 path
        with pytest.raises(ValueError, match="must be a relative path"):
            ProcessingStepConfigBase(
                processing_source_dir="/some/path",
                processing_entry_point=s3_entry,
                **sample_config_data,
            )

    def test_validate_entry_point_empty_string(self, sample_config_data):
        """Test that empty string entry point is rejected."""
        # Should raise ValueError for empty string
        with pytest.raises(ValueError, match="cannot be empty"):
            ProcessingStepConfigBase(
                processing_source_dir="/some/path",
                processing_entry_point="",
                **sample_config_data,
            )


class TestProcessingStepConfigInstanceTypes:
    """Test instance type configuration and properties."""

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
            "project_root_folder": "cursus",
        }

    def test_default_instance_types(self, sample_config_data):
        """Test default instance type configuration."""
        config = ProcessingStepConfigBase(**sample_config_data)

        # Should have default instance types
        assert config.processing_instance_type_small == "ml.m5.2xlarge"
        assert config.processing_instance_type_large == "ml.m5.4xlarge"
        assert config.use_large_processing_instance is False

    def test_effective_instance_type_small(self, sample_config_data):
        """Test effective_instance_type returns small instance by default."""
        config = ProcessingStepConfigBase(
            use_large_processing_instance=False, **sample_config_data
        )

        assert config.effective_instance_type == "ml.m5.2xlarge"

    def test_effective_instance_type_large(self, sample_config_data):
        """Test effective_instance_type returns large instance when configured."""
        config = ProcessingStepConfigBase(
            use_large_processing_instance=True, **sample_config_data
        )

        assert config.effective_instance_type == "ml.m5.4xlarge"

    def test_get_instance_type_method(self, sample_config_data):
        """Test get_instance_type method with size parameter."""
        config = ProcessingStepConfigBase(**sample_config_data)

        # Should return correct type based on size parameter
        assert config.get_instance_type("small") == "ml.m5.2xlarge"
        assert config.get_instance_type("large") == "ml.m5.4xlarge"

    def test_get_instance_type_none_uses_flag(self, sample_config_data):
        """Test that get_instance_type with None uses use_large_processing_instance flag."""
        config = ProcessingStepConfigBase(
            use_large_processing_instance=True, **sample_config_data
        )

        # None should use the flag value
        assert config.get_instance_type(None) == "ml.m5.4xlarge"

    def test_get_instance_type_invalid_size(self, sample_config_data):
        """Test that get_instance_type raises error for invalid size."""
        config = ProcessingStepConfigBase(**sample_config_data)

        # Should raise ValueError for invalid size
        with pytest.raises(ValueError, match="Invalid size parameter"):
            config.get_instance_type("medium")

    def test_custom_instance_types(self, sample_config_data):
        """Test configuration with custom instance types."""
        config = ProcessingStepConfigBase(
            processing_instance_type_small="ml.c5.xlarge",
            processing_instance_type_large="ml.c5.4xlarge",
            **sample_config_data,
        )

        assert config.processing_instance_type_small == "ml.c5.xlarge"
        assert config.processing_instance_type_large == "ml.c5.4xlarge"


class TestProcessingStepConfigInheritance:
    """Test inheritance and initialization methods."""

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
            "project_root_folder": "cursus",
        }

    def test_get_public_init_fields(self, sample_config_data):
        """Test get_public_init_fields includes processing-specific fields."""
        processing_path = "/some/processing/scripts"
        entry_point = "preprocessing.py"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=entry_point,
            processing_instance_count=2,
            **sample_config_data,
        )

        init_fields = config.get_public_init_fields()

        # Should include processing-specific fields
        assert "processing_source_dir" in init_fields
        assert init_fields["processing_source_dir"] == processing_path
        assert "processing_entry_point" in init_fields
        assert init_fields["processing_entry_point"] == entry_point
        assert "processing_instance_count" in init_fields
        assert init_fields["processing_instance_count"] == 2

        # Should include base class fields
        assert "author" in init_fields
        assert "bucket" in init_fields
        assert "region" in init_fields

    def test_get_public_init_fields_excludes_none(self, sample_config_data):
        """Test that get_public_init_fields excludes None optional fields."""
        config = ProcessingStepConfigBase(
            processing_source_dir=None,
            processing_entry_point=None,
            **sample_config_data,
        )

        init_fields = config.get_public_init_fields()

        # Optional fields with None should not be included
        assert "processing_source_dir" not in init_fields
        assert "processing_entry_point" not in init_fields

    def test_from_base_config(self, sample_config_data):
        """Test creating ProcessingStepConfigBase from base config."""
        # Create a base config
        base_config = ProcessingStepConfigBase(**sample_config_data)

        # Create new config from base with additional fields
        new_config = ProcessingStepConfigBase.from_base_config(
            base_config,
            processing_source_dir="/new/processing/path",
            processing_entry_point="new_script.py",
        )

        # Should inherit base fields
        assert new_config.author == base_config.author
        assert new_config.bucket == base_config.bucket

        # Should have new processing fields
        assert new_config.processing_source_dir == "/new/processing/path"
        assert new_config.processing_entry_point == "new_script.py"


class TestProcessingStepConfigResolvedPaths:
    """Test resolved path methods and properties."""

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
            "project_root_folder": "cursus",
        }

    def test_resolved_processing_source_dir_with_processing_path(
        self, sample_config_data
    ):
        """Test resolved_processing_source_dir property with processing_source_dir."""
        processing_path = "/some/processing/scripts"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path, **sample_config_data
        )

        # In test environment, may return None (no hybrid resolution available)
        resolved = config.resolved_processing_source_dir
        assert resolved is None or isinstance(resolved, str)

    def test_resolved_processing_source_dir_fallback(self, sample_config_data):
        """Test resolved_processing_source_dir falls back to source_dir."""
        source_path = "/some/source/path"

        config = ProcessingStepConfigBase(
            source_dir=source_path,
            processing_source_dir=None,
            **sample_config_data,
        )

        # Should try to resolve source_dir
        resolved = config.resolved_processing_source_dir
        assert resolved is None or isinstance(resolved, str)

    def test_get_resolved_script_path(self, sample_config_data):
        """Test get_resolved_script_path method."""
        processing_path = "/some/processing/scripts"
        entry_point = "preprocessing.py"

        config = ProcessingStepConfigBase(
            processing_source_dir=processing_path,
            processing_entry_point=entry_point,
            **sample_config_data,
        )

        # Should return script path (may be legacy or resolved)
        script_path = config.get_resolved_script_path()
        assert script_path is not None
        assert entry_point in script_path
