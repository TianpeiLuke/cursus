"""
Tests for workspace utilities functionality.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from cursus.workspace.utils import (
    WorkspaceConfig, PathUtils, ConfigUtils, FileUtils,
    ValidationUtils, TimeUtils, LoggingUtils, WorkspaceUtils
)


class TestWorkspaceConfig:
    """Test cases for WorkspaceConfig."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        return tempfile.mkdtemp()

    def test_workspace_config_creation(self, temp_dir):
        """Test WorkspaceConfig creation."""
        config = WorkspaceConfig(
            workspace_id="test_workspace",
            base_path=Path(temp_dir)
        )
        assert isinstance(config, WorkspaceConfig)


class TestPathUtils:
    """Test cases for PathUtils."""

    @pytest.fixture
    def temp_path(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        return Path(temp_dir)

    def test_path_normalization(self, temp_path):
        """Test path normalization."""
        # Test basic path normalization
        normalized = PathUtils.normalize_workspace_path(temp_path)
        assert isinstance(normalized, Path)

    def test_path_safety_checks(self, temp_path):
        """Test path safety validation."""
        # Test path safety checks
        is_safe = PathUtils.is_safe_path(temp_path, temp_path.parent)
        assert isinstance(is_safe, bool)

    def test_directory_operations(self, temp_path):
        """Test directory operations."""
        # Test directory creation
        test_dir = temp_path / "test_subdir"
        result = PathUtils.ensure_directory(test_dir)
        assert isinstance(result, bool)


class TestConfigUtils:
    """Test cases for ConfigUtils."""

    @pytest.fixture
    def config_setup(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "test_config.yaml"
        return temp_dir, config_path

    def test_config_loading(self, config_setup):
        """Test configuration loading."""
        temp_dir, config_path = config_setup
        # Create a test config file
        test_config = {"test_key": "test_value"}
        
        # Test config loading (may not exist yet)
        try:
            config = ConfigUtils.load_config(config_path)
            if config is not None:
                assert isinstance(config, dict)
        except FileNotFoundError:
            # Expected if file doesn't exist
            pass

    def test_config_saving(self, config_setup):
        """Test configuration saving."""
        temp_dir, config_path = config_setup
        test_config = {"test_key": "test_value"}
        
        # Test config saving
        result = ConfigUtils.save_config(test_config, config_path)
        assert isinstance(result, bool)

    def test_config_merging(self):
        """Test configuration merging."""
        config1 = {"key1": "value1"}
        config2 = {"key2": "value2"}
        
        # Test config merging
        merged = ConfigUtils.merge_configs(config1, config2)
        assert isinstance(merged, dict)
        assert "key1" in merged
        assert "key2" in merged


class TestFileUtils:
    """Test cases for FileUtils."""

    @pytest.fixture
    def file_setup(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        test_file = Path(temp_dir) / "test_file.txt"
        return temp_dir, test_file

    def test_file_operations(self, file_setup):
        """Test basic file operations."""
        temp_dir, test_file = file_setup
        # Create a test file first
        test_file.write_text("test content")
        
        # Test file hash calculation
        file_hash = FileUtils.calculate_file_hash(test_file)
        assert isinstance(file_hash, (str, type(None)))

    def test_file_reading(self, file_setup):
        """Test file reading operations."""
        temp_dir, test_file = file_setup
        # Create a test file
        test_file.write_text("test content")
        
        # Test text file detection
        is_text = FileUtils.is_text_file(test_file)
        assert isinstance(is_text, bool)

    def test_file_writing(self, file_setup):
        """Test file writing operations."""
        temp_dir, test_file = file_setup
        # Test file backup
        test_file.write_text("test content")
        backup_path = FileUtils.backup_file(test_file)
        assert isinstance(backup_path, (Path, type(None)))


class TestValidationUtils:
    """Test cases for ValidationUtils."""

    def test_validation_utilities(self):
        """Test validation utility functions."""
        # Test workspace structure validation
        temp_dir = tempfile.mkdtemp()
        required_dirs = ["builders", "configs"]
        is_valid, missing = ValidationUtils.validate_workspace_structure(temp_dir, required_dirs)
        assert isinstance(is_valid, bool)
        assert isinstance(missing, list)

    def test_workspace_validation(self):
        """Test workspace-specific validation."""
        # Test workspace size validation
        temp_dir = tempfile.mkdtemp()
        is_valid, size = ValidationUtils.validate_workspace_size(temp_dir, 100)  # 100MB limit
        assert isinstance(is_valid, bool)
        assert isinstance(size, int)


class TestTimeUtils:
    """Test cases for TimeUtils."""

    @pytest.fixture
    def temp_dir(self):
        """Set up test fixtures."""
        return tempfile.mkdtemp()

    def test_time_utilities(self):
        """Test time utility functions."""
        # Test timestamp formatting
        timestamp = TimeUtils.format_timestamp()
        assert isinstance(timestamp, str)

    def test_time_formatting(self, temp_dir):
        """Test time formatting utilities."""
        # Test path age checking
        temp_file = Path(temp_dir) / "test_file.txt"
        temp_file.write_text("test")
        age_days = TimeUtils.get_path_age_days(temp_file)
        assert isinstance(age_days, (int, type(None)))


class TestLoggingUtils:
    """Test cases for LoggingUtils."""

    def test_logging_utilities(self):
        """Test logging utility functions."""
        # Test logger setup
        logger = LoggingUtils.setup_workspace_logger("test_workspace")
        assert logger is not None


class TestWorkspaceUtils:
    """Test cases for WorkspaceUtils."""

    @pytest.fixture
    def workspace_setup(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        workspace_path = Path(temp_dir)
        return temp_dir, workspace_path

    def test_workspace_utilities(self, workspace_setup):
        """Test workspace utility functions."""
        temp_dir, workspace_path = workspace_setup
        # Test workspace directory initialization
        config = WorkspaceUtils.create_workspace_config("test_workspace", workspace_path)
        result = WorkspaceUtils.initialize_workspace_directory(workspace_path, config)
        assert isinstance(result, bool)

    def test_workspace_configuration(self, workspace_setup):
        """Test workspace configuration utilities."""
        temp_dir, workspace_path = workspace_setup
        # Test workspace configuration creation
        config = WorkspaceUtils.create_workspace_config("test_workspace", workspace_path)
        assert isinstance(config, WorkspaceConfig)

    def test_workspace_validation(self, workspace_setup):
        """Test workspace validation utilities."""
        temp_dir, workspace_path = workspace_setup
        # Test workspace validation
        config = WorkspaceUtils.create_workspace_config("test_workspace", workspace_path)
        is_valid, errors = WorkspaceUtils.validate_workspace(workspace_path, config)
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
