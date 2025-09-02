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

from src.cursus.core.workspace.isolation import WorkspaceIsolationManager


class TestWorkspaceIsolationManager(unittest.TestCase):
    """Test suite for WorkspaceIsolationManager."""
    
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
    
    def test_isolation_manager_initialization(self):
        """Test WorkspaceIsolationManager initialization."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        self.assertIs(isolation_manager.workspace_manager, self.mock_workspace_manager)
        self.assertEqual(isolation_manager.workspace_root, self.mock_workspace_manager.workspace_root)
        self.assertEqual(isolation_manager.isolation_rules, {})
        self.assertEqual(isolation_manager.violation_cache, {})
    
    def test_validate_workspace_boundaries_valid(self):
        """Test workspace boundary validation with valid workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create valid workspace structure
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "src").mkdir(exist_ok=True)
        (workspace_path / "src" / "cursus_dev").mkdir(exist_ok=True)
        
        result = isolation_manager.validate_workspace_boundaries(str(workspace_path))
        
        self.assertTrue(result.valid)
        self.assertEqual(result.workspace_path, str(workspace_path))
        self.assertEqual(len(result.violations), 0)
    
    def test_validate_workspace_boundaries_invalid_path(self):
        """Test workspace boundary validation with path outside workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Try to validate path outside workspace root
        invalid_path = "/tmp/outside_workspace"
        
        result = isolation_manager.validate_workspace_boundaries(invalid_path)
        
        self.assertFalse(result.valid)
        self.assertGreater(len(result.violations), 0)
        self.assertTrue(any("outside workspace root" in v.description for v in result.violations))
    
    def test_enforce_path_isolation_valid(self):
        """Test path isolation enforcement with valid access."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        workspace_path = str(Path(self.temp_workspace) / "developer_1")
        access_path = str(Path(workspace_path) / "src" / "test.py")
        
        result = isolation_manager.enforce_path_isolation(workspace_path, access_path)
        
        self.assertTrue(result)
    
    def test_enforce_path_isolation_invalid(self):
        """Test path isolation enforcement with invalid access."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        workspace_path = str(Path(self.temp_workspace) / "developer_1")
        access_path = str(Path(self.temp_workspace) / "developer_2" / "src" / "test.py")
        
        result = isolation_manager.enforce_path_isolation(workspace_path, access_path)
        
        self.assertFalse(result)
    
    def test_detect_isolation_violations_none(self):
        """Test isolation violation detection with clean workspace."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create clean workspace
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "src").mkdir(exist_ok=True)
        (workspace_path / "src" / "test.py").write_text("# Clean code")
        
        violations = isolation_manager.detect_isolation_violations(str(workspace_path))
        
        self.assertEqual(len(violations), 0)
    
    def test_detect_isolation_violations_found(self):
        """Test isolation violation detection with violations."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create workspace with violations
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "src").mkdir(exist_ok=True)
        
        # Create file with absolute path reference to another workspace
        violation_code = f"import sys\nsys.path.append('{self.temp_workspace}/developer_2/src')"
        (workspace_path / "src" / "violation.py").write_text(violation_code)
        
        violations = isolation_manager.detect_isolation_violations(str(workspace_path))
        
        self.assertGreater(len(violations), 0)
        self.assertTrue(any("cross-workspace reference" in v.description for v in violations))
    
    def test_create_isolated_environment(self):
        """Test creating isolated environment."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        env = isolation_manager.create_isolated_environment("developer_1")
        
        self.assertEqual(env.workspace_id, "developer_1")
        self.assertTrue(env.isolated)
        self.assertIsNotNone(env.workspace_path)
        self.assertIsNotNone(env.python_path)
        self.assertGreater(len(env.environment_variables), 0)
    
    def test_validate_step_definition_valid(self):
        """Test step definition validation with valid definition."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        mock_step = Mock()
        mock_step.step_name = "test_step"
        mock_step.developer_id = "developer_1"
        mock_step.workspace_root = self.temp_workspace
        mock_step.dependencies = ["valid_dependency"]
        
        result = isolation_manager.validate_step_definition(mock_step)
        
        self.assertTrue(result['valid'])
        self.assertEqual(len(result['errors']), 0)
    
    def test_validate_step_definition_invalid(self):
        """Test step definition validation with invalid definition."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        mock_step = Mock()
        mock_step.step_name = "../invalid_step"  # Invalid name with path traversal
        mock_step.developer_id = "developer_1"
        mock_step.workspace_root = "/invalid/root"  # Outside workspace
        mock_step.dependencies = []
        
        result = isolation_manager.validate_step_definition(mock_step)
        
        self.assertFalse(result['valid'])
        self.assertGreater(len(result['errors']), 0)
    
    def test_validate_pipeline_isolation_valid(self):
        """Test pipeline isolation validation with valid pipeline."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        mock_pipeline = Mock()
        mock_pipeline.pipeline_name = "test_pipeline"
        mock_pipeline.workspace_root = self.temp_workspace
        mock_pipeline.steps = []
        
        # Mock step validation to return valid
        with patch.object(isolation_manager, 'validate_step_definition', return_value={'valid': True, 'errors': []}):
            result = isolation_manager.validate_pipeline_isolation(mock_pipeline)
            
            self.assertTrue(result['valid'])
            self.assertEqual(len(result['errors']), 0)
    
    def test_validate_pipeline_isolation_invalid(self):
        """Test pipeline isolation validation with invalid pipeline."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        mock_pipeline = Mock()
        mock_pipeline.pipeline_name = "test_pipeline"
        mock_pipeline.workspace_root = "/invalid/root"
        mock_pipeline.steps = [Mock()]
        
        # Mock step validation to return invalid
        with patch.object(isolation_manager, 'validate_step_definition', return_value={'valid': False, 'errors': ['Invalid step']}):
            result = isolation_manager.validate_pipeline_isolation(mock_pipeline)
            
            self.assertFalse(result['valid'])
            self.assertGreater(len(result['errors']), 0)
    
    def test_setup_isolation_rules(self):
        """Test setting up isolation rules."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        rules = {
            "allowed_imports": ["os", "sys", "pathlib"],
            "forbidden_paths": ["/tmp", "/var"],
            "max_file_size": 1024 * 1024  # 1MB
        }
        
        isolation_manager.setup_isolation_rules("developer_1", rules)
        
        self.assertIn("developer_1", isolation_manager.isolation_rules)
        self.assertEqual(isolation_manager.isolation_rules["developer_1"], rules)
    
    def test_validate_import_isolation(self):
        """Test import isolation validation."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Set up rules
        rules = {"allowed_imports": ["os", "sys"], "forbidden_imports": ["subprocess"]}
        isolation_manager.setup_isolation_rules("developer_1", rules)
        
        # Test allowed import
        result_allowed = isolation_manager.validate_import_isolation("developer_1", "import os")
        self.assertTrue(result_allowed.valid)
        
        # Test forbidden import
        result_forbidden = isolation_manager.validate_import_isolation("developer_1", "import subprocess")
        self.assertFalse(result_forbidden.valid)
    
    def test_validate_file_access_isolation(self):
        """Test file access isolation validation."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        workspace_path = str(Path(self.temp_workspace) / "developer_1")
        
        # Test access within workspace
        internal_file = str(Path(workspace_path) / "src" / "test.py")
        result_internal = isolation_manager.validate_file_access_isolation("developer_1", internal_file)
        self.assertTrue(result_internal.valid)
        
        # Test access outside workspace
        external_file = "/tmp/external_file.py"
        result_external = isolation_manager.validate_file_access_isolation("developer_1", external_file)
        self.assertFalse(result_external.valid)
    
    def test_get_isolation_report(self):
        """Test getting isolation report."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create workspace with some violations
        workspace_path = Path(self.temp_workspace) / "developer_1"
        workspace_path.mkdir(exist_ok=True)
        
        with patch.object(isolation_manager, 'detect_isolation_violations', return_value=[]):
            report = isolation_manager.get_isolation_report("developer_1")
            
            self.assertEqual(report.workspace_id, "developer_1")
            self.assertGreaterEqual(report.isolation_score, 0)
            self.assertEqual(report.violations_count, 0)
    
    def test_cleanup_isolation_cache(self):
        """Test cleaning up isolation cache."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Add some cache entries
        isolation_manager.violation_cache["developer_1"] = ["violation1", "violation2"]
        isolation_manager.violation_cache["developer_2"] = ["violation3"]
        
        # Cleanup specific workspace
        isolation_manager.cleanup_isolation_cache("developer_1")
        
        self.assertNotIn("developer_1", isolation_manager.violation_cache)
        self.assertIn("developer_2", isolation_manager.violation_cache)
        
        # Cleanup all
        isolation_manager.cleanup_isolation_cache()
        
        self.assertEqual(len(isolation_manager.violation_cache), 0)
    
    def test_get_summary(self):
        """Test getting isolation manager summary."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Add some test data
        isolation_manager.isolation_rules["developer_1"] = {"rule1": "value1"}
        isolation_manager.violation_cache["developer_1"] = ["violation1"]
        
        summary = isolation_manager.get_summary()
        
        self.assertIn('total_workspaces_monitored', summary)
        self.assertIn('isolation_rules_count', summary)
        self.assertIn('cached_violations', summary)
        self.assertIn('workspace_root', summary)
        self.assertEqual(summary['isolation_rules_count'], 1)
        self.assertEqual(summary['cached_violations'], 1)
    
    def test_validate_health(self):
        """Test isolation manager health validation."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        health = isolation_manager.validate_health()
        
        self.assertIn('healthy', health)
        self.assertIn('isolation_system_functional', health)
        self.assertIn('workspace_root_accessible', health)
        self.assertIn('cache_system_operational', health)
        self.assertTrue(health['healthy'])
    
    def test_error_handling_invalid_workspace_path(self):
        """Test error handling for invalid workspace path."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Test with non-existent path
        result = isolation_manager.validate_workspace_boundaries("/non/existent/path")
        
        self.assertFalse(result.valid)
        self.assertGreater(len(result.violations), 0)
        self.assertTrue(any("does not exist" in v.description for v in result.violations))
    
    def test_concurrent_isolation_validation(self):
        """Test concurrent isolation validation."""
        isolation_manager = WorkspaceIsolationManager(self.mock_workspace_manager)
        
        # Create multiple workspaces
        workspaces = []
        for i in range(3):
            workspace_path = Path(self.temp_workspace) / f"developer_{i}"
            workspace_path.mkdir(exist_ok=True)
            workspaces.append(str(workspace_path))
        
        # Validate all workspaces
        results = []
        for workspace in workspaces:
            result = isolation_manager.validate_workspace_boundaries(workspace)
            results.append(result)
        
        # All should be valid
        self.assertTrue(all(r.valid for r in results))
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
