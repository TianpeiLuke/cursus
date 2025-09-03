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
from datetime import datetime, timedelta

from src.cursus.workspace.core.lifecycle import WorkspaceLifecycleManager
from src.cursus.workspace.core.manager import WorkspaceContext


class TestWorkspaceLifecycleManager(unittest.TestCase):
    """Test suite for WorkspaceLifecycleManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = Path(self.temp_dir) / "test_workspace"
        self.temp_workspace.mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.temp_workspace
        self.mock_workspace_manager.active_workspaces = {}
    
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_lifecycle_manager_initialization(self):
        """Test WorkspaceLifecycleManager initialization."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        self.assertIs(lifecycle_manager.workspace_manager, self.mock_workspace_manager)
        self.assertIsInstance(lifecycle_manager.templates, dict)
    
    def test_create_workspace_success(self):
        """Test successful workspace creation."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.touch') as mock_touch, \
             patch('pathlib.Path.exists', return_value=False):
            
            result = lifecycle_manager.create_workspace("developer_1", "developer")
            
            # Verify result
            self.assertIsInstance(result, WorkspaceContext)
            self.assertEqual(result.developer_id, "developer_1")
            self.assertEqual(result.workspace_type, "developer")
            self.assertIn("developer_1", result.workspace_path)
    
    def test_create_workspace_existing_workspace(self):
        """Test creating workspace when it already exists."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create the workspace directory first
        workspace_path = self.temp_workspace / "developers" / "developer_1"
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        with self.assertRaises(ValueError) as context:
            lifecycle_manager.create_workspace("developer_1", "developer")
        
        self.assertIn("already exists", str(context.exception))
    
    def test_create_workspace_failure(self):
        """Test workspace creation failure."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Mock mkdir to raise an exception
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            with self.assertRaises(PermissionError):
                lifecycle_manager.create_workspace("developer_1", "developer")
    
    def test_create_workspace_structure_developer(self):
        """Test creating developer workspace structure."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        workspace_path = self.temp_workspace / "developer_1"
        
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.touch') as mock_touch:
            
            lifecycle_manager._create_workspace_structure(
                workspace_path, "developer", None
            )
            
            # Verify directories were created
            self.assertGreater(mock_mkdir.call_count, 0)
            self.assertGreater(mock_touch.call_count, 0)
    
    def test_create_workspace_structure_shared(self):
        """Test creating shared workspace structure."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        workspace_path = self.temp_workspace / "shared"
        
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.touch') as mock_touch:
            
            lifecycle_manager._create_workspace_structure(
                workspace_path, "shared", None
            )
            
            # Verify directories were created
            self.assertGreater(mock_mkdir.call_count, 0)
    
    def test_apply_template(self):
        """Test applying workspace template."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        # Create a mock template
        template = Mock()
        template.template_name = "test_template"
        template.template_path = Path("nonexistent")
        template.metadata = {"structure": {"test_dir": {}}}
        
        with patch.object(lifecycle_manager, '_create_structure_from_metadata') as mock_create:
            lifecycle_manager._apply_template(workspace_path, template)
            mock_create.assert_called_once()
    
    def test_configure_workspace(self):
        """Test workspace configuration."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create a workspace context first
        workspace_context = WorkspaceContext(
            workspace_id="workspace_1",
            workspace_path=str(self.temp_workspace / "workspace_1"),
            developer_id="developer_1",
            workspace_type="developer"
        )
        self.mock_workspace_manager.active_workspaces["workspace_1"] = workspace_context
        
        config = {"setting1": "value1", "setting2": "value2"}
        
        with patch('builtins.open', MagicMock()) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            result = lifecycle_manager.configure_workspace("workspace_1", config)
            
            self.assertIsInstance(result, WorkspaceContext)
            self.assertEqual(result.workspace_id, "workspace_1")
            # Verify config was applied to metadata
            self.assertEqual(result.metadata["setting1"], "value1")
    
    def test_archive_workspace(self):
        """Test workspace archiving."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create a workspace context
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "test_file.py").touch()
        
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(workspace_path),
            developer_id="developer_1",
            workspace_type="developer"
        )
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        with patch('shutil.copytree') as mock_copytree, \
             patch('builtins.open', MagicMock()) as mock_open, \
             patch('json.dump') as mock_json_dump:
            
            result = lifecycle_manager.archive_workspace("developer_1")
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result["workspace_id"], "developer_1")
            self.assertTrue(result["success"])
            self.assertIn("archive_path", result)
    
    def test_delete_workspace(self):
        """Test workspace deletion."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create workspace directory with files
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "test_file.py").touch()
        
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(workspace_path),
            developer_id="developer_1",
            workspace_type="developer"
        )
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        with patch('shutil.rmtree') as mock_rmtree, \
             patch.object(lifecycle_manager, '_workspace_has_data', return_value=True), \
             patch.object(lifecycle_manager, '_archive_workspace_data', return_value=Path("/archive")):
            
            result = lifecycle_manager.delete_workspace("developer_1")
            
            self.assertTrue(result)
            mock_rmtree.assert_called_once()
    
    def test_restore_workspace(self):
        """Test workspace restoration from archive."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        archive_path = self.temp_workspace / "archive" / "developer_1_backup"
        archive_path.mkdir(parents=True, exist_ok=True)
        
        # Create archive metadata
        metadata_file = archive_path / "archive_metadata.json"
        metadata = {
            "workspace_id": "developer_1",
            "developer_id": "developer_1",
            "workspace_type": "developer"
        }
        
        with patch('builtins.open', MagicMock()) as mock_open, \
             patch('json.load', return_value=metadata), \
             patch('shutil.copytree') as mock_copytree:
            
            result = lifecycle_manager.restore_workspace("developer_1", str(archive_path))
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result["workspace_id"], "developer_1")
            self.assertTrue(result["success"])
            mock_copytree.assert_called_once()
    
    def test_cleanup_inactive_workspaces(self):
        """Test cleanup of inactive workspaces."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create old workspace context
        old_time = datetime.now() - timedelta(days=35)
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(self.temp_workspace / "developer_1"),
            developer_id="developer_1",
            workspace_type="developer"
        )
        workspace_context.last_accessed = old_time
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        with patch.object(lifecycle_manager, '_workspace_has_data', return_value=False), \
             patch.object(lifecycle_manager, 'delete_workspace', return_value=True):
            
            result = lifecycle_manager.cleanup_inactive_workspaces(timedelta(days=30))
            
            self.assertIsInstance(result, dict)
            self.assertIn("cleaned_up", result)
            self.assertIn("total_processed", result)
            self.assertEqual(result["total_processed"], 1)
    
    def test_get_available_templates(self):
        """Test getting available workspace templates."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        templates = lifecycle_manager.get_available_templates()
        
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
        # Should have default templates
        template_names = [t["name"] for t in templates]
        self.assertIn("basic", template_names)
    
    def test_get_statistics(self):
        """Test getting lifecycle manager statistics."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Add some workspace contexts
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(self.temp_workspace / "developer_1"),
            developer_id="developer_1",
            workspace_type="developer",
            status="active"
        )
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        stats = lifecycle_manager.get_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("available_templates", stats)
        self.assertIn("workspace_operations", stats)
        self.assertGreater(stats["available_templates"], 0)
    
    def test_workspace_has_data(self):
        """Test checking if workspace has user data."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        # Create workspace with user data
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps" / "builders"
        cursus_dev_dir.mkdir(parents=True, exist_ok=True)
        (cursus_dev_dir / "custom_builder.py").touch()
        
        has_data = lifecycle_manager._workspace_has_data(workspace_path)
        self.assertTrue(has_data)
        
        # Test workspace without user data
        empty_workspace = self.temp_workspace / "empty_workspace"
        empty_workspace.mkdir(exist_ok=True)
        
        has_data = lifecycle_manager._workspace_has_data(empty_workspace)
        self.assertFalse(has_data)
    
    def test_create_structure_from_metadata(self):
        """Test creating workspace structure from template metadata."""
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        workspace_path = self.temp_workspace / "test_workspace"
        workspace_path.mkdir(exist_ok=True)
        
        metadata = {
            "structure": {
                "src": {
                    "test_file.py": "# Test content"
                },
                "config": {}
            }
        }
        
        lifecycle_manager._create_structure_from_metadata(workspace_path, metadata)
        
        # Verify structure was created
        self.assertTrue((workspace_path / "src").exists())
        self.assertTrue((workspace_path / "src" / "test_file.py").exists())
        self.assertTrue((workspace_path / "config").exists())
    
    def test_error_handling_no_workspace_root(self):
        """Test error handling when no workspace root is configured."""
        self.mock_workspace_manager.workspace_root = None
        lifecycle_manager = WorkspaceLifecycleManager(self.mock_workspace_manager)
        
        with self.assertRaises(ValueError) as context:
            lifecycle_manager.create_workspace("developer_1", "developer")
        
        self.assertIn("No workspace root configured", str(context.exception))


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
