"""
Tests for workspace utilities functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.cursus.workspace.utils import (
    WorkspaceConfig, PathUtils, ConfigUtils, FileUtils,
    ValidationUtils, TimeUtils, LoggingUtils, WorkspaceUtils
)


class TestWorkspaceConfig(unittest.TestCase):
    """Test cases for WorkspaceConfig."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_workspace_config_creation(self):
        """Test WorkspaceConfig creation."""
        from pathlib import Path
        config = WorkspaceConfig(
            workspace_id="test_workspace",
            base_path=Path(self.temp_dir)
        )
        self.assertIsInstance(config, WorkspaceConfig)


class TestPathUtils(unittest.TestCase):
    """Test cases for PathUtils."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)

    def test_path_normalization(self):
        """Test path normalization."""
        # Test basic path normalization
        normalized = PathUtils.normalize_workspace_path(self.test_path)
        self.assertIsInstance(normalized, Path)

    def test_path_safety_checks(self):
        """Test path safety validation."""
        # Test path safety checks
        is_safe = PathUtils.is_safe_path(self.test_path, self.test_path.parent)
        self.assertIsInstance(is_safe, bool)

    def test_directory_operations(self):
        """Test directory operations."""
        # Test directory creation
        test_dir = self.test_path / "test_subdir"
        result = PathUtils.ensure_directory(test_dir)
        self.assertIsInstance(result, bool)


class TestConfigUtils(unittest.TestCase):
    """Test cases for ConfigUtils."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"

    def test_config_loading(self):
        """Test configuration loading."""
        # Create a test config file
        test_config = {"test_key": "test_value"}
        
        # Test config loading (may not exist yet)
        try:
            config = ConfigUtils.load_config(self.config_path)
            if config is not None:
                self.assertIsInstance(config, dict)
        except FileNotFoundError:
            # Expected if file doesn't exist
            pass

    def test_config_saving(self):
        """Test configuration saving."""
        test_config = {"test_key": "test_value"}
        
        # Test config saving
        result = ConfigUtils.save_config(test_config, self.config_path)
        self.assertIsInstance(result, bool)

    def test_config_merging(self):
        """Test configuration merging."""
        config1 = {"key1": "value1"}
        config2 = {"key2": "value2"}
        
        # Test config merging
        merged = ConfigUtils.merge_configs(config1, config2)
        self.assertIsInstance(merged, dict)
        self.assertIn("key1", merged)
        self.assertIn("key2", merged)


class TestFileUtils(unittest.TestCase):
    """Test cases for FileUtils."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = Path(self.temp_dir) / "test_file.txt"

    def test_file_operations(self):
        """Test basic file operations."""
        # Create a test file first
        self.test_file.write_text("test content")
        
        # Test file hash calculation
        file_hash = FileUtils.calculate_file_hash(self.test_file)
        self.assertIsInstance(file_hash, (str, type(None)))

    def test_file_reading(self):
        """Test file reading operations."""
        # Create a test file
        self.test_file.write_text("test content")
        
        # Test text file detection
        is_text = FileUtils.is_text_file(self.test_file)
        self.assertIsInstance(is_text, bool)

    def test_file_writing(self):
        """Test file writing operations."""
        # Test file backup
        self.test_file.write_text("test content")
        backup_path = FileUtils.backup_file(self.test_file)
        self.assertIsInstance(backup_path, (Path, type(None)))


class TestValidationUtils(unittest.TestCase):
    """Test cases for ValidationUtils."""

    def test_validation_utilities(self):
        """Test validation utility functions."""
        # Test workspace structure validation
        temp_dir = tempfile.mkdtemp()
        required_dirs = ["builders", "configs"]
        is_valid, missing = ValidationUtils.validate_workspace_structure(temp_dir, required_dirs)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(missing, list)

    def test_workspace_validation(self):
        """Test workspace-specific validation."""
        # Test workspace size validation
        temp_dir = tempfile.mkdtemp()
        is_valid, size = ValidationUtils.validate_workspace_size(temp_dir, 100)  # 100MB limit
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(size, int)


class TestTimeUtils(unittest.TestCase):
    """Test cases for TimeUtils."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def test_time_utilities(self):
        """Test time utility functions."""
        # Test timestamp formatting
        timestamp = TimeUtils.format_timestamp()
        self.assertIsInstance(timestamp, str)

    def test_time_formatting(self):
        """Test time formatting utilities."""
        # Test path age checking
        temp_file = Path(self.temp_dir) / "test_file.txt"
        temp_file.write_text("test")
        age_days = TimeUtils.get_path_age_days(temp_file)
        self.assertIsInstance(age_days, (int, type(None)))


class TestLoggingUtils(unittest.TestCase):
    """Test cases for LoggingUtils."""

    def test_logging_utilities(self):
        """Test logging utility functions."""
        # Test logger setup
        logger = LoggingUtils.setup_workspace_logger("test_workspace")
        self.assertIsNotNone(logger)


class TestWorkspaceUtils(unittest.TestCase):
    """Test cases for WorkspaceUtils."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)

    def test_workspace_utilities(self):
        """Test workspace utility functions."""
        # Test workspace directory initialization
        config = WorkspaceUtils.create_workspace_config("test_workspace", self.workspace_path)
        result = WorkspaceUtils.initialize_workspace_directory(self.workspace_path, config)
        self.assertIsInstance(result, bool)

    def test_workspace_configuration(self):
        """Test workspace configuration utilities."""
        # Test workspace configuration creation
        config = WorkspaceUtils.create_workspace_config("test_workspace", self.workspace_path)
        self.assertIsInstance(config, WorkspaceConfig)

    def test_workspace_validation(self):
        """Test workspace validation utilities."""
        # Test workspace validation
        config = WorkspaceUtils.create_workspace_config("test_workspace", self.workspace_path)
        is_valid, errors = WorkspaceUtils.validate_workspace(self.workspace_path, config)
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(errors, list)


if __name__ == '__main__':
    unittest.main()
