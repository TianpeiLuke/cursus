"""
Unit tests for WorkspaceLifecycleManager.

This module provides comprehensive unit testing for the WorkspaceLifecycleManager
and its workspace lifecycle management capabilities.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.cursus.core.workspace.lifecycle import WorkspaceLifecycleManager


class TestWorkspaceLifecycleManager(unittest.TestCase):
    """Test suite for WorkspaceLifecycleManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = str(Path(self.temp_dir) / "test_workspace")
        Path(self.temp_workspace).mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.temp_workspace
        self.mock_workspace_manager.config_file = None
        self.mock_workspace_manager.auto_discover = True
    
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_lifecycle_manager_initialization(self):
        """Test WorkspaceLifecycleManager initialization."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        self.assertIs(lifecycle_manager.workspace_manager, self.mock_workspace_manager)
        self.assertEqual(lifecycle_manager.workspace_root, self.mock_workspace_manager.workspace_root)
        self.assertEqual(lifecycle_manager.templates_cache, {})
        self.assertEqual(lifecycle_manager.workspace_configs, {})
    
    def test_create_workspace_success(self):
        """Test successful workspace creation."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Mock template setup and validation config
        with patch.object(lifecycle_manager, 'setup_workspace_templates', return_value=True) as mock_setup_templates, \
             patch.object(lifecycle_manager, 'setup_validation_config', return_value=True) as mock_setup_validation, \
             patch('pathlib.Path.mkdir') as mock_mkdir:
            
            result = lifecycle_manager.create_workspace("developer_1", "standard")
            
            # Verify result
            self.assertTrue(result.success)
            self.assertEqual(result.developer_id, "developer_1")
            self.assertEqual(result.workspace_type, "standard")
            self.assertIn("developer_1", result.workspace_path)
            
            # Verify methods were called
            mock_setup_templates.assert_called_once()
            mock_setup_validation.assert_called_once()
    
    def test_create_workspace_existing_workspace(self):
        """Test creating workspace when it already exists."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create the workspace directory first
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        with patch('pathlib.Path.exists', return_value=True):
            result = lifecycle_manager.create_workspace("developer_1", "standard")
            
            # Should still succeed but indicate it already existed
            self.assertTrue(result.success)
            self.assertTrue(result.already_existed)
    
    def test_create_workspace_failure(self):
        """Test workspace creation failure."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Mock mkdir to raise an exception
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            result = lifecycle_manager.create_workspace("developer_1", "standard")
            
            # Verify failure
            self.assertFalse(result.success)
            self.assertIn("Permission denied", result.error)
    
    def test_setup_workspace_templates_standard(self):
        """Test setting up standard workspace templates."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        # Mock file operations
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('builtins.open', mock_open_multiple_files()) as mock_open:
            
            result = lifecycle_manager.setup_workspace_templates(str(workspace_path), "standard")
            
            self.assertTrue(result)
            # Verify directories were created
            self.assertGreaterEqual(mock_mkdir.call_count, 4)  # src, builders, configs, scripts, etc.
    
    def test_setup_workspace_templates_ml(self):
        """Test setting up ML workspace templates."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        # Mock file operations
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('builtins.open', mock_open_multiple_files()) as mock_open:
            
            result = lifecycle_manager.setup_workspace_templates(str(workspace_path), "ml")
            
            self.assertTrue(result)
            # ML template should create additional directories
            self.assertGreaterEqual(mock_mkdir.call_count, 6)  # Additional ML-specific directories
    
    def test_setup_validation_config(self):
        """Test setting up validation configuration."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        # Mock file operations
        with patch('builtins.open', mock_open_multiple_files()) as mock_open, \
             patch('yaml.dump') as mock_yaml_dump:
            
            result = lifecycle_manager.setup_validation_config(str(workspace_path))
            
            self.assertTrue(result)
            # Verify YAML config was written
            mock_yaml_dump.assert_called()
    
    def test_configure_workspace(self):
        """Test workspace configuration."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        mock_config = Mock()
        mock_config.developer_id = "developer_1"
        mock_config.workspace_type = "standard"
        mock_config.custom_settings = {"setting1": "value1"}
        
        with patch('builtins.open', mock_open_multiple_files()) as mock_open, \
             patch('yaml.dump') as mock_yaml_dump:
            
            result = lifecycle_manager.configure_workspace("workspace_1", mock_config)
            
            self.assertTrue(result.success)
            self.assertEqual(result.workspace_id, "workspace_1")
            # Verify config was cached
            self.assertIn("workspace_1", lifecycle_manager.workspace_configs)
    
    def test_archive_workspace(self):
        """Test workspace archiving."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create workspace directory
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "test_file.py").touch()
        
        with patch('shutil.make_archive', return_value="/archives/developer_1.zip") as mock_archive:
            result = lifecycle_manager.archive_workspace("developer_1")
            
            self.assertTrue(result.success)
            self.assertEqual(result.workspace_id, "developer_1")
            self.assertEqual(result.archive_path, "/archives/developer_1.zip")
            mock_archive.assert_called_once()
    
    def test_cleanup_workspace(self):
        """Test workspace cleanup."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create workspace directory with files
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "test_file.py").touch()
        (workspace_path / "subdir").mkdir(exist_ok=True)
        (workspace_path / "subdir" / "nested_file.py").touch()
        
        result = lifecycle_manager.cleanup_workspace("developer_1")
        
        self.assertTrue(result.success)
        self.assertEqual(result.workspace_id, "developer_1")
        self.assertGreater(result.files_removed, 0)
        self.assertFalse(workspace_path.exists())
    
    def test_delete_workspace_with_archive(self):
        """Test workspace deletion with archiving."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create workspace directory
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "important_file.py").touch()
        
        with patch.object(lifecycle_manager, 'archive_workspace') as mock_archive, \
             patch.object(lifecycle_manager, 'cleanup_workspace') as mock_cleanup:
            
            mock_archive.return_value = Mock(success=True, archive_path="/archives/developer_1.zip")
            mock_cleanup.return_value = Mock(success=True, files_removed=5)
            
            result = lifecycle_manager.delete_workspace("developer_1", archive=True)
            
            self.assertTrue(result.success)
            self.assertEqual(result.workspace_id, "developer_1")
            self.assertTrue(result.archived)
            mock_archive.assert_called_once_with("developer_1")
            mock_cleanup.assert_called_once_with("developer_1")
    
    def test_delete_workspace_without_archive(self):
        """Test workspace deletion without archiving."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        with patch.object(lifecycle_manager, 'cleanup_workspace') as mock_cleanup:
            mock_cleanup.return_value = Mock(success=True, files_removed=3)
            
            result = lifecycle_manager.delete_workspace("developer_1", archive=False)
            
            self.assertTrue(result.success)
            self.assertEqual(result.workspace_id, "developer_1")
            self.assertFalse(result.archived)
            mock_cleanup.assert_called_once_with("developer_1")
    
    def test_get_workspace_info(self):
        """Test getting workspace information."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create workspace with some structure
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "src").mkdir(exist_ok=True)
        (workspace_path / "src" / "test.py").touch()
        (workspace_path / "config.yaml").touch()
        
        info = lifecycle_manager.get_workspace_info("developer_1")
        
        self.assertEqual(info.workspace_id, "developer_1")
        self.assertTrue(info.exists)
        self.assertGreater(info.file_count, 0)
        self.assertIn("src", info.directories)
    
    def test_list_workspaces(self):
        """Test listing all workspaces."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create multiple workspaces
        for dev_id in ["developer_1", "developer_2", "developer_3"]:
            workspace_path = Path(self.temp_workspace) / dev_id
            workspace_path.mkdir(exist_ok=True)
            (workspace_path / "config.yaml").touch()
        
        workspaces = lifecycle_manager.list_workspaces()
        
        self.assertEqual(len(workspaces), 3)
        workspace_ids = [ws.workspace_id for ws in workspaces]
        self.assertIn("developer_1", workspace_ids)
        self.assertIn("developer_2", workspace_ids)
        self.assertIn("developer_3", workspace_ids)
    
    def test_validate_workspace_structure(self):
        """Test workspace structure validation."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create properly structured workspace
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "src").mkdir(exist_ok=True)
        (workspace_path / "src" / "cursus_dev").mkdir(exist_ok=True)
        (workspace_path / "src" / "cursus_dev" / "steps").mkdir(exist_ok=True)
        (workspace_path / "config.yaml").touch()
        
        result = lifecycle_manager.validate_workspace_structure("developer_1")
        
        self.assertTrue(result.valid)
        self.assertEqual(len(result.errors), 0)
        self.assertGreater(len(result.required_directories), 0)
    
    def test_validate_workspace_structure_invalid(self):
        """Test workspace structure validation with invalid structure."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create workspace with missing required directories
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        # Missing src directory and config file
        
        result = lifecycle_manager.validate_workspace_structure("developer_1")
        
        self.assertFalse(result.valid)
        self.assertGreater(len(result.errors), 0)
        self.assertTrue(any("src" in error for error in result.errors))
    
    def test_get_summary(self):
        """Test getting lifecycle manager summary."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create some workspaces
        for dev_id in ["developer_1", "developer_2"]:
            workspace_path = Path(self.temp_workspace) / dev_id
            workspace_path.mkdir(exist_ok=True)
        
        summary = lifecycle_manager.get_summary()
        
        self.assertIn('total_workspaces', summary)
        self.assertIn('workspace_root', summary)
        self.assertIn('templates_cached', summary)
        self.assertEqual(summary['total_workspaces'], 2)
        self.assertEqual(summary['workspace_root'], self.temp_workspace)
    
    def test_validate_health(self):
        """Test lifecycle manager health validation."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        health = lifecycle_manager.validate_health()
        
        self.assertIn('healthy', health)
        self.assertIn('workspace_root_accessible', health)
        self.assertIn('template_system_functional', health)
        self.assertTrue(health['healthy'])
    
    def test_error_handling_invalid_workspace_type(self):
        """Test error handling for invalid workspace type."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        result = lifecycle_manager.create_workspace("developer_1", "invalid_type")
        
        # Should handle gracefully and use default template
        self.assertTrue(result.success)
        self.assertEqual(result.workspace_type, "invalid_type")  # Records what was requested
    
    def test_concurrent_workspace_creation(self):
        """Test handling concurrent workspace creation attempts."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Simulate concurrent creation by having mkdir succeed for first call, fail for second
        with patch('pathlib.Path.mkdir', side_effect=[None, FileExistsError("Directory exists")]):
            result1 = lifecycle_manager.create_workspace("developer_1", "standard")
            result2 = lifecycle_manager.create_workspace("developer_1", "standard")
            
            # Both should succeed, second should indicate it already existed
            self.assertTrue(result1.success)
            self.assertTrue(result2.success)


def mock_open_multiple_files():
    """Helper function to mock multiple file operations."""
    mock_files = {}
    
    def mock_open_func(filename, mode='r', *args, **kwargs):
        if filename not in mock_files:
            mock_files[filename] = MagicMock()
        return mock_files[filename]
    
    return mock_open_func


if __name__ == "__main__":
    unittest.main()
