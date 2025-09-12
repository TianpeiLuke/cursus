"""
Tests for workspace API functionality.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile

from cursus.workspace.api import WorkspaceAPI, WorkspaceSetupResult, ValidationReport


class TestWorkspaceAPI:
    """Test cases for WorkspaceAPI."""

    @pytest.fixture
    def temp_workspace(self):
        """Set up test fixtures."""
        temp_dir = tempfile.mkdtemp()
        workspace_path = Path(temp_dir)
        yield workspace_path
        # Cleanup is handled automatically by tempfile

    @pytest.fixture
    def api(self, temp_workspace):
        """Create WorkspaceAPI instance."""
        return WorkspaceAPI(base_path=temp_workspace)

    def test_workspace_api_initialization(self, api, temp_workspace):
        """Test that WorkspaceAPI initializes correctly."""
        assert isinstance(api, WorkspaceAPI)
        assert api.base_path == temp_workspace

    def test_setup_developer_workspace(self, api):
        """Test developer workspace setup."""
        # Test basic workspace setup
        result = api.setup_developer_workspace(
            developer_id="test_developer", template="basic"
        )
        assert isinstance(result, WorkspaceSetupResult)
        assert isinstance(result.success, bool)
        assert result.developer_id == "test_developer"

    def test_validate_workspace(self, api, temp_workspace):
        """Test workspace validation."""
        # Test workspace validation
        report = api.validate_workspace(temp_workspace)
        assert isinstance(report, ValidationReport)
        assert hasattr(report, "status")

    def test_list_workspaces(self, api):
        """Test workspace listing."""
        # Test workspace listing
        workspaces = api.list_workspaces()
        assert isinstance(workspaces, list)

    def test_promote_workspace_artifacts(self, api, temp_workspace):
        """Test workspace artifact promotion."""
        # Test artifact promotion
        result = api.promote_workspace_artifacts(
            workspace_path=temp_workspace, target_environment="staging"
        )
        assert result is not None

    def test_get_system_health(self, api):
        """Test system health reporting."""
        # Test system health
        from cursus.workspace.api import HealthReport

        health = api.get_system_health()
        assert isinstance(health, HealthReport)
        assert hasattr(health, "overall_status")
        assert hasattr(health, "workspace_reports")

    def test_cleanup_workspaces(self, api):
        """Test workspace cleanup."""
        # Test workspace cleanup
        cleanup_result = api.cleanup_workspaces(inactive_days=30, dry_run=True)
        assert cleanup_result is not None
