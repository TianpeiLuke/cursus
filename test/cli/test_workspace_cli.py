"""
Pytest tests for the workspace CLI module.

This module tests all functionality of the simplified workspace command-line interface,
including component discovery, validation, search, and workspace management using
the unified WorkspaceAPI built on the step catalog architecture.
"""

import pytest
from unittest.mock import Mock, patch
import json
import tempfile
import shutil
from pathlib import Path
from click.testing import CliRunner

from cursus.cli.workspace_cli import (
    workspace_cli,
    init_workspace,
    discover_components,
    validate_workspace,
    workspace_info,
    search_components,
    system_status,
    refresh_catalog,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


class TestWorkspaceCliBasic:
    """Test basic CLI functionality and command structure."""

    def test_workspace_cli_group_exists(self, cli_runner):
        """Test that the workspace CLI group exists and is accessible."""
        result = cli_runner.invoke(workspace_cli, ["--help"])
        assert result.exit_code == 0
        assert "Simplified workspace management commands" in result.output

    def test_workspace_cli_commands_exist(self, cli_runner):
        """Test that all expected commands exist in the CLI group."""
        result = cli_runner.invoke(workspace_cli, ["--help"])
        assert result.exit_code == 0

        expected_commands = [
            "init",
            "discover", 
            "validate",
            "info",
            "search",
            "status",
            "refresh",
        ]

        for command in expected_commands:
            assert command in result.output


class TestInitWorkspaceCommand:
    """Test workspace initialization command."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_init_workspace_package_only(self, mock_workspace_api, cli_runner):
        """Test workspace initialization in package-only mode."""
        # Mock the WorkspaceAPI
        mock_api = Mock()
        mock_api.get_workspace_summary.return_value = {
            "total_workspaces": 0,
            "total_components": 65,
            "workspace_directories": [],
            "workspace_components": {}
        }
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(init_workspace, [])

        assert result.exit_code == 0
        assert "Initializing WorkspaceAPI in package-only mode" in result.output
        assert "✅ WorkspaceAPI initialized successfully" in result.output
        assert "Total components: 65" in result.output
        mock_workspace_api.assert_called_once_with(workspace_dirs=None)

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_init_workspace_with_dirs(self, mock_workspace_api, cli_runner):
        """Test workspace initialization with workspace directories."""
        # Mock the WorkspaceAPI
        mock_api = Mock()
        mock_api.get_workspace_summary.return_value = {
            "total_workspaces": 2,
            "total_components": 75,
            "workspace_directories": ["/tmp/workspace1", "/tmp/workspace2"],
            "workspace_components": {
                "workspace1": 10,
                "workspace2": 5
            }
        }
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            init_workspace, 
            ["--workspace-dirs", "/tmp/workspace1", "--workspace-dirs", "/tmp/workspace2"]
        )

        assert result.exit_code == 0
        assert "Initializing WorkspaceAPI with 2 directories" in result.output
        assert "Total workspaces: 2" in result.output
        assert "workspace1: 10 components" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_init_workspace_json_output(self, mock_workspace_api, cli_runner, temp_dir):
        """Test workspace initialization with JSON output."""
        # Mock the WorkspaceAPI
        mock_api = Mock()
        mock_summary = {
            "total_workspaces": 1,
            "total_components": 65,
            "workspace_directories": ["/tmp/workspace"],
        }
        mock_api.get_workspace_summary.return_value = mock_summary
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            init_workspace, 
            ["--workspace-dirs", "/tmp/workspace", "--output", "json"]
        )

        assert result.exit_code == 0
        
        # Verify JSON output - extract JSON from the output
        lines = result.output.strip().split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start >= 0:
            json_text = '\n'.join(lines[json_start:])
            output_data = json.loads(json_text)
            assert output_data["total_workspaces"] == 1
            assert output_data["total_components"] == 65
        else:
            # Fallback - check that JSON-like content is present
            assert '"total_workspaces": 1' in result.output
            assert '"total_components": 65' in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_init_workspace_error_handling(self, mock_workspace_api, cli_runner):
        """Test workspace initialization error handling."""
        mock_workspace_api.side_effect = Exception("Initialization failed")

        result = cli_runner.invoke(init_workspace, [])

        assert result.exit_code == 1
        assert "❌ Error initializing workspace" in result.output


class TestDiscoverComponentsCommand:
    """Test component discovery command."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_discover_components_basic(self, mock_workspace_api, cli_runner):
        """Test basic component discovery."""
        mock_api = Mock()
        mock_api.discover_components.return_value = [
            "TabularPreprocessing", "XGBoostTraining", "CurrencyConversion"
        ]
        mock_api.get_cross_workspace_components.return_value = {
            "core": ["TabularPreprocessing", "XGBoostTraining", "CurrencyConversion"]
        }
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(discover_components, [])

        assert result.exit_code == 0
        assert "🔍 Discovering components" in result.output
        assert "Found 3 components" in result.output
        assert "core (3 components)" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_discover_components_with_search(self, mock_workspace_api, cli_runner):
        """Test component discovery with search."""
        mock_component = Mock()
        mock_component.step_name = "XGBoostTraining"
        mock_component.workspace_id = "core"
        
        mock_api = Mock()
        mock_api.search_components.return_value = [mock_component]
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            discover_components, 
            ["--search", "xgboost", "--format", "list"]
        )

        assert result.exit_code == 0
        assert "XGBoostTraining" in result.output
        mock_api.search_components.assert_called_once_with("xgboost", workspace_id=None)

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_discover_components_json_format(self, mock_workspace_api, cli_runner):
        """Test component discovery with JSON format."""
        mock_api = Mock()
        mock_api.discover_components.return_value = ["TabularPreprocessing", "XGBoostTraining"]
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            discover_components, 
            ["--format", "json"]
        )

        assert result.exit_code == 0
        
        # Verify JSON output - extract JSON from the output
        lines = result.output.strip().split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('['):
                json_start = i
                break
        
        if json_start >= 0:
            json_text = '\n'.join(lines[json_start:])
            output_data = json.loads(json_text)
            assert "TabularPreprocessing" in output_data
            assert "XGBoostTraining" in output_data
        else:
            # Fallback - check that JSON-like content is present
            assert '"TabularPreprocessing"' in result.output
            assert '"XGBoostTraining"' in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_discover_components_with_details(self, mock_workspace_api, cli_runner):
        """Test component discovery with detailed information."""
        mock_info = Mock()
        mock_info.step_name = "TabularPreprocessing"
        mock_info.workspace_id = "core"
        mock_info.file_components = {
            "builder": Mock(path="/path/to/builder.py"),
            "config": Mock(path="/path/to/config.py"),
            "script": None
        }
        
        mock_api = Mock()
        mock_api.discover_components.return_value = ["TabularPreprocessing"]
        mock_api.get_component_info.return_value = mock_info
        mock_api.get_cross_workspace_components.return_value = {"core": ["TabularPreprocessing"]}
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            discover_components, 
            ["--show-details"]
        )

        assert result.exit_code == 0
        assert "🔧 TabularPreprocessing" in result.output
        assert "Workspace: core" in result.output
        assert "builder: /path/to/builder.py" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_discover_components_no_results(self, mock_workspace_api, cli_runner):
        """Test component discovery when no components found."""
        mock_api = Mock()
        mock_api.discover_components.return_value = []
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(discover_components, [])

        assert result.exit_code == 0
        assert "No components found" in result.output


class TestValidateWorkspaceCommand:
    """Test workspace validation command."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_validate_workspace_structure(self, mock_workspace_api, cli_runner):
        """Test workspace structure validation."""
        mock_api = Mock()
        mock_api.validate_workspace_structure.return_value = {
            "valid": True,
            "warnings": []
        }
        mock_workspace_api.return_value = mock_api

        # Create a temporary directory for the test
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            result = cli_runner.invoke(
                validate_workspace, 
                ["--workspace-dirs", temp_dir]
            )

            assert result.exit_code == 0
            assert "🔍 Validating workspace" in result.output
            assert "📁 Validating workspace structure" in result.output
            assert "✅" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_validate_workspace_components(self, mock_workspace_api, cli_runner):
        """Test workspace component validation."""
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.details = {
            "validated_components": 10,
            "total_components": 10
        }
        
        mock_api = Mock()
        mock_api.validate_workspace_components.return_value = mock_result
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            validate_workspace, 
            ["--workspace-id", "core"]
        )

        assert result.exit_code == 0
        assert "🔧 Validating components in 'core'" in result.output
        assert "✅ Component validation: Passed" in result.output
        assert "Components validated: 10" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_validate_component_quality(self, mock_workspace_api, cli_runner):
        """Test component quality validation."""
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.details = {
            "quality_score": 85,
            "component_completeness": 90,
            "missing_components": []
        }
        
        mock_api = Mock()
        mock_api.validate_component_quality.return_value = mock_result
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            validate_workspace, 
            ["--component", "TabularPreprocessing"]
        )

        assert result.exit_code == 0
        assert "🎯 Validating component quality: TabularPreprocessing" in result.output
        assert "✅ Quality validation: Passed" in result.output
        assert "Quality score: 85/100" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_validate_cross_workspace_compatibility(self, mock_workspace_api, cli_runner):
        """Test cross-workspace compatibility validation."""
        mock_result = Mock()
        mock_result.is_compatible = True
        mock_result.issues = []
        mock_result.compatibility_matrix = {
            "workspace1": {"conflicts": 0, "total_components": 10},
            "workspace2": {"conflicts": 1, "total_components": 8}
        }
        
        mock_api = Mock()
        mock_api.list_all_workspaces.return_value = ["workspace1", "workspace2"]
        mock_api.validate_cross_workspace_compatibility.return_value = mock_result
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            validate_workspace, 
            ["--compatibility"]
        )

        assert result.exit_code == 0
        assert "🤝 Checking cross-workspace compatibility" in result.output
        assert "✅ Cross-workspace compatibility: Compatible" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_validate_workspace_json_output(self, mock_workspace_api, cli_runner):
        """Test workspace validation with JSON output."""
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.details = {"validated_components": 5}
        
        mock_api = Mock()
        mock_api.validate_workspace_components.return_value = mock_result
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            validate_workspace, 
            ["--workspace-id", "core", "--format", "json"]
        )

        assert result.exit_code == 0
        
        # Verify JSON output - extract JSON from the output
        lines = result.output.strip().split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start >= 0:
            json_text = '\n'.join(lines[json_start:])
            output_data = json.loads(json_text)
            assert "components_core" in output_data
        else:
            # Fallback - check that JSON-like content is present
            assert '"components_core"' in result.output or 'components_core' in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_validate_workspace_with_report_save(self, mock_workspace_api, cli_runner, temp_dir):
        """Test workspace validation with report saving."""
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.details = {"validated_components": 5}
        
        mock_api = Mock()
        mock_api.validate_workspace_components.return_value = mock_result
        mock_workspace_api.return_value = mock_api

        report_path = Path(temp_dir) / "validation_report.json"

        result = cli_runner.invoke(
            validate_workspace, 
            ["--workspace-id", "core", "--report", str(report_path)]
        )

        assert result.exit_code == 0
        assert f"✓ Validation report saved: {report_path}" in result.output
        assert report_path.exists()


class TestWorkspaceInfoCommand:
    """Test workspace information command."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_workspace_info_summary(self, mock_workspace_api, cli_runner):
        """Test workspace information summary."""
        mock_api = Mock()
        mock_api.get_workspace_summary.return_value = {
            "total_workspaces": 1,
            "total_components": 65,
            "workspace_directories": ["/tmp/workspace"],
            "workspace_components": {"core": 65}
        }
        mock_api.get_system_status.return_value = {
            "workspace_api": {
                "success_rate": 0.95,
                "metrics": {"api_calls": 100}
            }
        }
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(workspace_info, [])

        assert result.exit_code == 0
        assert "📊 Workspace Information" in result.output
        assert "Total workspaces: 1" in result.output
        assert "Total components: 65" in result.output
        assert "API success rate: 95.0%" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_workspace_info_specific_component(self, mock_workspace_api, cli_runner):
        """Test workspace information for specific component."""
        mock_info = Mock()
        mock_info.step_name = "TabularPreprocessing"
        mock_info.workspace_id = "core"
        mock_info.file_components = {
            "builder": Mock(path="/path/to/builder.py"),
            "config": Mock(path="/path/to/config.py"),
            "script": None
        }
        
        mock_api = Mock()
        mock_api.get_component_info.return_value = mock_info
        mock_workspace_api.return_value = mock_api

        with patch("pathlib.Path.exists", return_value=True):
            result = cli_runner.invoke(
                workspace_info, 
                ["--component", "TabularPreprocessing"]
            )

        assert result.exit_code == 0
        assert "🔧 Component Information: TabularPreprocessing" in result.output
        assert "Name: TabularPreprocessing" in result.output
        assert "Workspace: core" in result.output
        assert "builder: /path/to/builder.py ✅" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_workspace_info_component_not_found(self, mock_workspace_api, cli_runner):
        """Test workspace information for non-existent component."""
        mock_api = Mock()
        mock_api.get_component_info.return_value = None
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            workspace_info, 
            ["--component", "NonExistentComponent"]
        )

        assert result.exit_code == 1
        assert "❌ Component not found: NonExistentComponent" in result.output


class TestSearchComponentsCommand:
    """Test component search command."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_search_components_basic(self, mock_workspace_api, cli_runner):
        """Test basic component search."""
        mock_component = Mock()
        mock_component.step_name = "XGBoostTraining"
        mock_component.workspace_id = "core"
        mock_component.file_components = {
            "builder": Mock(path="/path/to/builder.py"),
            "config": Mock(path="/path/to/config.py")
        }
        
        mock_api = Mock()
        mock_api.search_components.return_value = [mock_component]
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(search_components, ["xgboost"])

        assert result.exit_code == 0
        assert "🔍 Searching for: 'xgboost'" in result.output
        assert "Found 1 matching components" in result.output
        assert "🔧 XGBoostTraining" in result.output
        assert "Workspace: core" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_search_components_list_format(self, mock_workspace_api, cli_runner):
        """Test component search with list format."""
        mock_component = Mock()
        mock_component.step_name = "XGBoostTraining"
        
        mock_api = Mock()
        mock_api.search_components.return_value = [mock_component]
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            search_components, 
            ["xgboost", "--format", "list"]
        )

        assert result.exit_code == 0
        assert "XGBoostTraining" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_search_components_no_results(self, mock_workspace_api, cli_runner):
        """Test component search with no results."""
        mock_api = Mock()
        mock_api.search_components.return_value = []
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(search_components, ["nonexistent"])

        assert result.exit_code == 0
        assert "No matching components found" in result.output


class TestSystemStatusCommand:
    """Test system status command."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_system_status_basic(self, mock_workspace_api, cli_runner):
        """Test basic system status."""
        mock_api = Mock()
        mock_api.get_system_status.return_value = {
            "workspace_api": {
                "success_rate": 0.95,
                "metrics": {
                    "api_calls": 100,
                    "successful_operations": 95,
                    "failed_operations": 5
                },
                "workspace_directories": ["/tmp/workspace"]
            },
            "manager": {
                "total_components": 65,
                "total_workspaces": 1
            },
            "validator": {
                "metrics": {
                    "validations_performed": 20,
                    "components_validated": 15,
                    "compatibility_checks": 5
                }
            },
            "integrator": {
                "metrics": {
                    "promotions": 2,
                    "integrations": 3,
                    "rollbacks": 0
                }
            }
        }
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(system_status, [])

        assert result.exit_code == 0
        assert "📊 System Status" in result.output
        assert "Success rate: 95.0%" in result.output
        assert "Total components: 65" in result.output
        assert "Validations performed: 20" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_system_status_json_format(self, mock_workspace_api, cli_runner):
        """Test system status with JSON format."""
        mock_status = {
            "workspace_api": {"success_rate": 0.95},
            "manager": {"total_components": 65}
        }
        
        mock_api = Mock()
        mock_api.get_system_status.return_value = mock_status
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(
            system_status, 
            ["--format", "json"]
        )

        assert result.exit_code == 0
        
        # Verify JSON output - extract JSON from the output
        lines = result.output.strip().split('\n')
        json_start = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break
        
        if json_start >= 0:
            json_text = '\n'.join(lines[json_start:])
            output_data = json.loads(json_text)
            assert output_data["workspace_api"]["success_rate"] == 0.95
            assert output_data["manager"]["total_components"] == 65
        else:
            # Fallback - check that JSON-like content is present
            assert '"success_rate": 0.95' in result.output
            assert '"total_components": 65' in result.output


class TestRefreshCatalogCommand:
    """Test catalog refresh command."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_refresh_catalog_success(self, mock_workspace_api, cli_runner):
        """Test successful catalog refresh."""
        mock_api = Mock()
        mock_api.discover_components.side_effect = [
            ["Component1", "Component2"],  # Before refresh
            ["Component1", "Component2", "Component3"]  # After refresh
        ]
        mock_api.refresh_catalog.return_value = True
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(refresh_catalog, [])

        assert result.exit_code == 0
        assert "🔄 Refreshing step catalog" in result.output
        assert "✅ Catalog refresh successful" in result.output
        assert "Components before: 2" in result.output
        assert "Components after: 3" in result.output
        assert "✨ Discovered 1 new components" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_refresh_catalog_no_changes(self, mock_workspace_api, cli_runner):
        """Test catalog refresh with no changes."""
        mock_api = Mock()
        mock_api.discover_components.return_value = ["Component1", "Component2"]
        mock_api.refresh_catalog.return_value = True
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(refresh_catalog, [])

        assert result.exit_code == 0
        assert "📊 No changes detected" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_refresh_catalog_failure(self, mock_workspace_api, cli_runner):
        """Test catalog refresh failure."""
        mock_api = Mock()
        mock_api.discover_components.return_value = ["Component1"]
        mock_api.refresh_catalog.return_value = False
        mock_workspace_api.return_value = mock_api

        result = cli_runner.invoke(refresh_catalog, [])

        assert result.exit_code == 1
        assert "❌ Catalog refresh failed" in result.output


class TestWorkspaceCliErrorHandling:
    """Test error handling in workspace CLI commands."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_discover_components_error(self, mock_workspace_api, cli_runner):
        """Test error handling in discover components."""
        mock_workspace_api.side_effect = Exception("API initialization failed")

        result = cli_runner.invoke(discover_components, [])

        assert result.exit_code == 1
        assert "❌ Error discovering components" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_validate_workspace_error(self, mock_workspace_api, cli_runner):
        """Test error handling in validate workspace."""
        mock_workspace_api.side_effect = Exception("Validation failed")

        result = cli_runner.invoke(validate_workspace, [])

        assert result.exit_code == 1
        assert "❌ Error validating workspace" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_workspace_info_error(self, mock_workspace_api, cli_runner):
        """Test error handling in workspace info."""
        mock_workspace_api.side_effect = Exception("Info retrieval failed")

        result = cli_runner.invoke(workspace_info, [])

        assert result.exit_code == 1
        assert "❌ Error getting workspace info" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_search_components_error(self, mock_workspace_api, cli_runner):
        """Test error handling in search components."""
        mock_workspace_api.side_effect = Exception("Search failed")

        result = cli_runner.invoke(search_components, ["test"])

        assert result.exit_code == 1
        assert "❌ Error searching components" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_system_status_error(self, mock_workspace_api, cli_runner):
        """Test error handling in system status."""
        mock_workspace_api.side_effect = Exception("Status retrieval failed")

        result = cli_runner.invoke(system_status, [])

        assert result.exit_code == 1
        assert "❌ Error getting system status" in result.output

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_refresh_catalog_error(self, mock_workspace_api, cli_runner):
        """Test error handling in refresh catalog."""
        mock_workspace_api.side_effect = Exception("Refresh failed")

        result = cli_runner.invoke(refresh_catalog, [])

        assert result.exit_code == 1
        assert "❌ Error refreshing catalog" in result.output


class TestWorkspaceCliIntegration:
    """Integration tests for workspace CLI."""

    @patch("cursus.cli.workspace_cli.WorkspaceAPI")
    def test_full_workflow_integration(self, mock_workspace_api, cli_runner):
        """Test a complete workflow integration."""
        mock_api = Mock()
        
        # Mock discovery
        mock_api.discover_components.return_value = ["TabularPreprocessing", "XGBoostTraining"]
        
        # Mock component info
        mock_info = Mock()
        mock_info.step_name = "TabularPreprocessing"
        mock_info.workspace_id = "core"
        mock_info.file_components = {"builder": Mock(path="/path/to/builder.py")}
        mock_api.get_component_info.return_value = mock_info
        
        # Mock validation
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.details = {"quality_score": 85}
        mock_api.validate_component_quality.return_value = mock_result
        
        mock_workspace_api.return_value = mock_api

        # Test discovery
        result = cli_runner.invoke(discover_components, ["--format", "list"])
        assert result.exit_code == 0
        assert "TabularPreprocessing" in result.output

        # Test component info
        result = cli_runner.invoke(workspace_info, ["--component", "TabularPreprocessing"])
        assert result.exit_code == 0
        assert "TabularPreprocessing" in result.output

        # Test validation
        result = cli_runner.invoke(validate_workspace, ["--component", "TabularPreprocessing"])
        assert result.exit_code == 0
        assert "Quality score: 85/100" in result.output
