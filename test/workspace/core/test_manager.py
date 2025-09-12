"""
Unit tests for WorkspaceManager.

This module provides comprehensive unit testing for the consolidated
WorkspaceManager and its integration with specialized managers.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from cursus.workspace.core.manager import WorkspaceManager
from cursus.workspace.core.lifecycle import WorkspaceLifecycleManager
from cursus.workspace.core.isolation import WorkspaceIsolationManager
from cursus.workspace.core.discovery import WorkspaceDiscoveryManager
from cursus.workspace.core.integration import WorkspaceIntegrationManager


class TestWorkspaceManager:
    """Test suite for WorkspaceManager."""

    @pytest.fixture
    def temp_workspace(self):
        """Set up test fixtures before each test method."""
        temp_dir = tempfile.mkdtemp()
        temp_workspace = str(Path(temp_dir) / "test_workspace")
        Path(temp_workspace).mkdir(parents=True, exist_ok=True)

        yield temp_dir, temp_workspace

        # Clean up after each test method
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

    def test_manager_initialization_default(self):
        """Test WorkspaceManager initialization with default parameters."""
        manager = WorkspaceManager()

        # Verify basic initialization - workspace_root can be None initially
        assert manager.config_file is None

        # Verify specialized managers are created
        assert isinstance(manager.lifecycle_manager, WorkspaceLifecycleManager)
        assert isinstance(manager.isolation_manager, WorkspaceIsolationManager)
        assert isinstance(manager.discovery_manager, WorkspaceDiscoveryManager)
        assert isinstance(manager.integration_manager, WorkspaceIntegrationManager)

        # Verify circular references
        assert manager.lifecycle_manager.workspace_manager is manager
        assert manager.isolation_manager.workspace_manager is manager
        assert manager.discovery_manager.workspace_manager is manager
        assert manager.integration_manager.workspace_manager is manager

    def test_manager_initialization_with_workspace_root(self, temp_workspace):
        """Test WorkspaceManager initialization with workspace root."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        assert str(manager.workspace_root) == temp_workspace_path
        assert manager.config_file is None

    def test_manager_initialization_with_config_file(self, temp_workspace):
        """Test WorkspaceManager initialization with config file."""
        temp_dir, temp_workspace_path = temp_workspace

        config_file = str(Path(temp_workspace_path) / "config.yaml")
        manager = WorkspaceManager(
            workspace_root=temp_workspace_path, config_file=config_file
        )

        assert str(manager.workspace_root) == temp_workspace_path
        assert str(manager.config_file) == config_file

    def test_manager_initialization_no_auto_discover(self, temp_workspace):
        """Test WorkspaceManager initialization without auto-discovery."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(
            workspace_root=temp_workspace_path, auto_discover=False
        )

        assert str(manager.workspace_root) == temp_workspace_path

    def test_create_workspace_delegation(self, temp_workspace):
        """Test create_workspace delegates to lifecycle manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        # Mock lifecycle manager
        mock_result = Mock()
        mock_result.workspace_id = "developer_1"
        mock_result.workspace_path = f"{temp_workspace_path}/developer_1"

        with patch.object(
            manager.lifecycle_manager, "create_workspace", return_value=mock_result
        ) as mock_create:
            result = manager.create_workspace("developer_1", "standard")

            # Verify delegation with correct parameters
            mock_create.assert_called_once_with(
                developer_id="developer_1", workspace_type="standard", template=None
            )
            assert result is mock_result

    def test_configure_workspace_delegation(self, temp_workspace):
        """Test configure_workspace delegates to lifecycle manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        # First create a workspace to configure
        mock_workspace = Mock()
        mock_workspace.workspace_id = "workspace_1"
        manager.active_workspaces["workspace_1"] = mock_workspace

        mock_config = {"setting": "value"}
        mock_result = Mock()
        mock_result.workspace_id = "workspace_1"

        with patch.object(
            manager.lifecycle_manager, "configure_workspace", return_value=mock_result
        ) as mock_configure:
            result = manager.configure_workspace("workspace_1", mock_config)

            # Verify delegation with correct parameters
            mock_configure.assert_called_once_with(
                workspace_id="workspace_1", config=mock_config
            )
            assert result is mock_result

    def test_delete_workspace_delegation(self, temp_workspace):
        """Test delete_workspace delegates to lifecycle manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        mock_result = Mock()
        mock_result.success = True

        with patch.object(
            manager.lifecycle_manager, "delete_workspace", return_value=mock_result
        ) as mock_delete:
            result = manager.delete_workspace("workspace_1")

            # Verify delegation
            mock_delete.assert_called_once_with("workspace_1")
            assert result is mock_result

    def test_discover_components_delegation(self, temp_workspace):
        """Test discover_components delegates to discovery manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        mock_result = {"components": {"builder1": "info"}}

        with patch.object(
            manager.discovery_manager, "discover_components", return_value=mock_result
        ) as mock_discover:
            result = manager.discover_components(["workspace_1", "workspace_2"])

            # Verify delegation with correct parameters
            mock_discover.assert_called_once_with(
                workspace_ids=["workspace_1", "workspace_2"], developer_id=None
            )
            assert result == mock_result

    def test_resolve_cross_workspace_dependencies_delegation(self, temp_workspace):
        """Test resolve_cross_workspace_dependencies delegates to discovery manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        mock_pipeline_def = Mock()
        mock_result = Mock()
        mock_result.dependencies = {"step1": ["step2"]}

        with patch.object(
            manager.discovery_manager,
            "resolve_cross_workspace_dependencies",
            return_value=mock_result,
        ) as mock_resolve:
            result = manager.resolve_cross_workspace_dependencies(mock_pipeline_def)

            # Verify delegation
            mock_resolve.assert_called_once_with(mock_pipeline_def)
            assert result is mock_result

    def test_stage_for_integration_delegation(self, temp_workspace):
        """Test stage_for_integration delegates to integration manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        mock_result = {"success": True, "staging_path": "/staging/component_1"}

        with patch.object(
            manager.integration_manager,
            "stage_for_integration",
            return_value=mock_result,
        ) as mock_stage:
            result = manager.stage_for_integration("component_1", "workspace_1")

            # Verify delegation with correct parameters
            mock_stage.assert_called_once_with(
                component_id="component_1",
                source_workspace="workspace_1",
                target_stage="integration",
            )
            assert result == mock_result

    def test_validate_integration_readiness_delegation(self, temp_workspace):
        """Test validate_integration_readiness delegates to integration manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        mock_result = {"ready": True, "issues": []}

        with patch.object(
            manager.integration_manager,
            "validate_integration_readiness",
            return_value=mock_result,
        ) as mock_validate:
            result = manager.validate_integration_readiness(
                ["component_1", "component_2"]
            )

            # Verify delegation
            mock_validate.assert_called_once_with(["component_1", "component_2"])
            assert result == mock_result

    def test_get_workspace_summary(self, temp_workspace):
        """Test get_workspace_summary aggregates information from all managers."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        # Mock the methods that actually exist
        with patch.object(
            manager.discovery_manager,
            "get_discovery_summary",
            return_value={"components": 10},
        ) as mock_discovery, patch.object(
            manager.isolation_manager,
            "get_validation_summary",
            return_value={"violations": 0},
        ) as mock_isolation, patch.object(
            manager.integration_manager,
            "get_integration_summary",
            return_value={"staged": 2},
        ) as mock_integration:

            summary = manager.get_workspace_summary()

            # Verify methods were called
            mock_discovery.assert_called_once()
            mock_isolation.assert_called_once()
            mock_integration.assert_called_once()

            # Verify summary structure
            assert "workspace_root" in summary
            assert "discovery_summary" in summary
            assert "validation_summary" in summary
            assert "integration_summary" in summary
            assert summary["workspace_root"] == str(manager.workspace_root)

    def test_get_workspace_health(self, temp_workspace):
        """Test get_workspace_health delegates to isolation manager."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        mock_health = {"healthy": True, "issues": []}

        with patch.object(
            manager.isolation_manager, "get_workspace_health", return_value=mock_health
        ) as mock_get_health:
            health = manager.get_workspace_health("workspace_1")

            # Verify delegation
            mock_get_health.assert_called_once_with("workspace_1")
            assert health == mock_health

    def test_error_handling_in_initialization(self):
        """Test error handling during manager initialization."""
        # Test with invalid workspace root and auto_discover=False to avoid discovery
        manager = WorkspaceManager(workspace_root="/invalid/path", auto_discover=False)
        assert manager is not None
        assert str(manager.workspace_root) == "/invalid/path"

    def test_specialized_manager_error_propagation(self, temp_workspace):
        """Test that errors from specialized managers are properly propagated."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        # Mock lifecycle manager to raise an error
        with patch.object(manager.lifecycle_manager, "create_workspace") as mock_create:
            mock_create.side_effect = ValueError("Invalid workspace configuration")

            # Error should be propagated
            with pytest.raises(ValueError) as exc_info:
                manager.create_workspace("invalid_workspace", "invalid_type")
            assert "Invalid workspace configuration" in str(exc_info.value)

    def test_manager_state_consistency(self, temp_workspace):
        """Test that manager state remains consistent across operations."""
        temp_dir, temp_workspace_path = temp_workspace

        manager = WorkspaceManager(workspace_root=temp_workspace_path)

        # Verify initial state
        initial_root = manager.workspace_root
        initial_config = manager.config_file

        # Perform some operations
        mock_workspace = Mock()
        mock_workspace.workspace_id = "test_workspace"
        with patch.object(
            manager.lifecycle_manager, "create_workspace", return_value=mock_workspace
        ):
            manager.create_workspace("test_workspace", "standard")

        with patch.object(
            manager.discovery_manager,
            "discover_components",
            return_value={"components": {}},
        ):
            manager.discover_components(["test_workspace"])

        # Verify state hasn't changed
        assert manager.workspace_root == initial_root
        assert manager.config_file == initial_config

        # Verify specialized managers still reference the same manager
        assert manager.lifecycle_manager.workspace_manager is manager
        assert manager.isolation_manager.workspace_manager is manager
        assert manager.discovery_manager.workspace_manager is manager
        assert manager.integration_manager.workspace_manager is manager
