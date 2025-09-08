"""
Unit tests for WorkspaceIsolationManager.

This module provides comprehensive unit testing for the WorkspaceIsolationManager
and its workspace isolation capabilities.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cursus.workspace.core.isolation import WorkspaceIsolationManager, IsolationViolation
from cursus.workspace.core.manager import WorkspaceContext

class TestWorkspaceIsolationManager(unittest.TestCase):
    """Test suite for WorkspaceIsolationManager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_workspace = Path(self.temp_dir) / "test_workspace"
        self.temp_workspace.mkdir(parents=True, exist_ok=True)
        
        # Create mock workspace manager
        self.mock_workspace_manager = Mock()
        self.mock_workspace_manager.workspace_root = self.temp_workspace
        self.mock_workspace_manager.active_workspaces = {}
        self.mock_workspace_manager.config = None
    
    def tearDown(self):
        """Clean up after each test method."""
        if hasattr(self, 'temp_dir') and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_isolation_manager_initialization(self):
        """Test WorkspaceIsolationManager initialization."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        self.assertIs(isolation_manager.workspace_manager, self.mock_workspace_manager)
        self.assertIsInstance(isolation_manager.isolation_violations, list)
        self.assertEqual(len(isolation_manager.isolation_violations), 0)
    
    def test_validate_workspace_boundaries_valid(self):
        """Test workspace boundary validation with valid workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create valid workspace structure
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "src").mkdir(exist_ok=True)
        (workspace_path / "src" / "cursus_dev").mkdir(exist_ok=True)
        
        # Create workspace context
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(workspace_path),
            developer_id="developer_1",
            workspace_type="developer"
        )
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        result = isolation_manager.validate_workspace_boundaries("developer_1")
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["valid"])
        self.assertEqual(result["workspace_id"], "developer_1")
        self.assertEqual(len(result["violations"]), 0)
    
    def test_validate_workspace_boundaries_workspace_not_found(self):
        """Test workspace boundary validation with workspace not found."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Try to validate non-existent workspace
        result = isolation_manager.validate_workspace_boundaries("nonexistent_workspace")
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["valid"])
        self.assertGreater(len(result["violations"]), 0)
        self.assertIn("Workspace not found", result["violations"][0])
    
    def test_enforce_path_isolation_valid(self):
        """Test path isolation enforcement with valid access."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        workspace_path = str(self.temp_workspace / "developer_1")
        access_path = str(self.temp_workspace / "developer_1" / "src" / "test.py")
        
        result = isolation_manager.enforce_path_isolation(workspace_path, access_path)
        
        self.assertTrue(result)
    
    def test_enforce_path_isolation_invalid(self):
        """Test path isolation enforcement with invalid access."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        workspace_path = str(self.temp_workspace / "developer_1")
        access_path = str(self.temp_workspace / "developer_2" / "src" / "test.py")
        
        result = isolation_manager.enforce_path_isolation(workspace_path, access_path)
        
        self.assertFalse(result)
    
    def test_detect_isolation_violations_none(self):
        """Test isolation violation detection with clean workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create clean workspace
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "src").mkdir(exist_ok=True)
        (workspace_path / "src" / "test.py").write_text("# Clean code")
        
        # Create workspace context
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(workspace_path),
            developer_id="developer_1",
            workspace_type="developer"
        )
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        violations = isolation_manager.detect_isolation_violations("developer_1")
        
        self.assertIsInstance(violations, list)
        self.assertEqual(len(violations), 0)
    
    def test_detect_isolation_violations_workspace_not_found(self):
        """Test isolation violation detection with workspace not found."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        violations = isolation_manager.detect_isolation_violations("nonexistent_workspace")
        
        self.assertIsInstance(violations, list)
        self.assertGreater(len(violations), 0)
        self.assertEqual(violations[0].violation_type, "workspace_not_found")
        self.assertEqual(violations[0].workspace_id, "nonexistent_workspace")
    
    def test_manage_namespace_isolation(self):
        """Test namespace isolation management."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Test regular workspace
        namespaced_name = isolation_manager.manage_namespace_isolation("developer_1", "test_component")
        self.assertEqual(namespaced_name, "developer_1:test_component")
        
        # Test shared workspace
        shared_name = isolation_manager.manage_namespace_isolation("shared", "shared_component")
        self.assertEqual(shared_name, "shared_component")
    
    def test_validate_workspace_structure_valid(self):
        """Test workspace structure validation with valid structure."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create valid workspace structure
        developers_dir = self.temp_workspace / "developers"
        developers_dir.mkdir(exist_ok=True)
        
        dev_workspace = developers_dir / "developer_1"
        dev_workspace.mkdir(exist_ok=True)
        cursus_dev_dir = dev_workspace / "src" / "cursus_dev" / "steps"
        cursus_dev_dir.mkdir(parents=True, exist_ok=True)
        (cursus_dev_dir / "builders").mkdir(exist_ok=True)
        
        is_valid, issues = isolation_manager.validate_workspace_structure(self.temp_workspace)
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
    
    def test_validate_workspace_structure_invalid(self):
        """Test workspace structure validation with invalid structure."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create invalid workspace structure (missing required directories)
        empty_workspace = self.temp_workspace / "empty"
        empty_workspace.mkdir(exist_ok=True)
        
        is_valid, issues = isolation_manager.validate_workspace_structure(empty_workspace)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(issues), 0)
    
    def test_get_workspace_health_valid(self):
        """Test getting workspace health for valid workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create workspace context
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(workspace_path),
            developer_id="developer_1",
            workspace_type="developer"
        )
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        health = isolation_manager.get_workspace_health("developer_1")
        
        self.assertIsInstance(health, dict)
        self.assertEqual(health["workspace_id"], "developer_1")
        self.assertIn("healthy", health)
        self.assertIn("health_score", health)
        self.assertIn("last_checked", health)
    
    def test_get_workspace_health_not_found(self):
        """Test getting workspace health for non-existent workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        health = isolation_manager.get_workspace_health("nonexistent_workspace")
        
        self.assertIsInstance(health, dict)
        self.assertEqual(health["workspace_id"], "nonexistent_workspace")
        self.assertFalse(health["healthy"])
        self.assertEqual(health["health_score"], 0)
        self.assertIn("Workspace not found", health["issues"])
    
    def test_get_validation_summary(self):
        """Test getting validation summary."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Add some test violations
        violation = IsolationViolation(
            violation_type="test_violation",
            workspace_id="developer_1",
            description="Test violation",
            severity="medium"
        )
        isolation_manager.isolation_violations.append(violation)
        
        summary = isolation_manager.get_validation_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("total_violations", summary)
        self.assertIn("violation_types", summary)
        self.assertIn("severity_distribution", summary)
        self.assertEqual(summary["total_violations"], 1)
    
    def test_get_statistics(self):
        """Test getting isolation manager statistics."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Add workspace context
        workspace_context = WorkspaceContext(
            workspace_id="developer_1",
            workspace_path=str(self.temp_workspace / "developer_1"),
            developer_id="developer_1",
            workspace_type="developer"
        )
        self.mock_workspace_manager.active_workspaces["developer_1"] = workspace_context
        
        with patch.object(isolation_manager, 'get_workspace_health', return_value={"healthy": True}):
            stats = isolation_manager.get_statistics()
            
            self.assertIsInstance(stats, dict)
            self.assertIn("isolation_checks", stats)
            self.assertIn("violation_summary", stats)
    
    def test_is_safe_import(self):
        """Test safe import checking."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Test safe imports
        self.assertTrue(isolation_manager._is_safe_import("os"))
        self.assertTrue(isolation_manager._is_safe_import("sys"))
        self.assertTrue(isolation_manager._is_safe_import("pathlib"))
        self.assertTrue(isolation_manager._is_safe_import("numpy"))
        self.assertTrue(isolation_manager._is_safe_import("boto3"))
        
        # Test potentially unsafe imports
        self.assertFalse(isolation_manager._is_safe_import("unknown_package"))
        self.assertFalse(isolation_manager._is_safe_import("custom_module"))
    
    def test_detect_problematic_imports(self):
        """Test detection of problematic imports."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Test code with problematic imports
        code_with_problems = """
import unknown_package
from custom_module import something
import os  # This should be safe
from numpy import array  # This should be safe
"""
        
        workspace_path = self.temp_workspace / "developer_1"
        problematic = isolation_manager._detect_problematic_imports(code_with_problems, workspace_path)
        
        self.assertIsInstance(problematic, list)
        # Should detect unknown_package and custom_module but not os/numpy
        self.assertIn("unknown_package", problematic)
        self.assertIn("custom_module", problematic)
    
    def test_detect_global_variable_usage(self):
        """Test detection of global variable usage."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create workspace with global variable usage
        workspace_path = self.temp_workspace / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        test_file = workspace_path / "test.py"
        test_file.write_text("""
def test_function():
    global test_var
    test_var = "test"
""")
        
        global_vars = isolation_manager._detect_global_variable_usage(workspace_path)
        
        self.assertIsInstance(global_vars, list)
        self.assertGreater(len(global_vars), 0)
        self.assertTrue(any("test_var" in var for var in global_vars))
    
    def test_contains_absolute_paths_outside_workspace(self):
        """Test detection of absolute paths outside workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create config file with absolute path
        config_file = self.temp_workspace / "config.json"
        config_content = '{"path": "/tmp/outside_workspace", "internal": "relative/path"}'
        config_file.write_text(config_content)
        
        # This should return False since /tmp doesn't exist in our test
        result = isolation_manager._contains_absolute_paths_outside_workspace(config_file, self.temp_workspace)
        self.assertFalse(result)
    
    def test_isolation_violation_creation(self):
        """Test IsolationViolation creation and conversion."""
        violation = IsolationViolation(
            violation_type="test_violation",
            workspace_id="developer_1",
            description="Test violation description",
            severity="high",
            details={"key": "value"}
        )
        
        self.assertEqual(violation.violation_type, "test_violation")
        self.assertEqual(violation.workspace_id, "developer_1")
        self.assertEqual(violation.severity, "high")
        self.assertEqual(violation.details["key"], "value")
        
        # Test dictionary conversion
        violation_dict = violation.to_dict()
        self.assertIsInstance(violation_dict, dict)
        self.assertEqual(violation_dict["violation_type"], "test_violation")
        self.assertEqual(violation_dict["workspace_id"], "developer_1")
        self.assertIn("detected_at", violation_dict)

if __name__ == "__main__":
    unittest.main()
