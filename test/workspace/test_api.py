"""
Tests for workspace API functionality.
"""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from src.cursus.workspace.api import WorkspaceAPI, WorkspaceSetupResult, ValidationReport


class TestWorkspaceAPI(unittest.TestCase):
    """Test cases for WorkspaceAPI."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.workspace_path = Path(self.temp_dir)
        self.api = WorkspaceAPI(base_path=self.workspace_path)

    def test_workspace_api_initialization(self):
        """Test that WorkspaceAPI initializes correctly."""
        self.assertIsInstance(self.api, WorkspaceAPI)
        self.assertEqual(self.api.base_path, self.workspace_path)

    def test_setup_developer_workspace(self):
        """Test developer workspace setup."""
        # Test basic workspace setup
        result = self.api.setup_developer_workspace(
            developer_id="test_developer",
            template="basic"
        )
        self.assertIsInstance(result, WorkspaceSetupResult)
        self.assertIsInstance(result.success, bool)
        self.assertEqual(result.developer_id, "test_developer")

    def test_validate_workspace(self):
        """Test workspace validation."""
        # Test workspace validation
        report = self.api.validate_workspace(self.workspace_path)
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(hasattr(report, 'status'))

    def test_list_workspaces(self):
        """Test workspace listing."""
        # Test workspace listing
        workspaces = self.api.list_workspaces()
        self.assertIsInstance(workspaces, list)

    def test_promote_workspace_artifacts(self):
        """Test workspace artifact promotion."""
        # Test artifact promotion
        result = self.api.promote_workspace_artifacts(
            workspace_path=self.workspace_path,
            target_environment="staging"
        )
        self.assertIsNotNone(result)

    def test_get_system_health(self):
        """Test system health reporting."""
        # Test system health
        health = self.api.get_system_health()
        self.assertIsInstance(health, dict)

    def test_cleanup_workspaces(self):
        """Test workspace cleanup."""
        # Test workspace cleanup
        cleanup_result = self.api.cleanup_workspaces(
            inactive_days=30,
            dry_run=True
        )
        self.assertIsNotNone(cleanup_result)


if __name__ == '__main__':
    unittest.main()
