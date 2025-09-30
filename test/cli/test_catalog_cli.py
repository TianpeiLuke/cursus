"""
Comprehensive tests for catalog CLI commands.

This module tests all catalog CLI commands for step catalog management including
step discovery, searching, component management, workspace discovery, and metrics.

Tests focus on:
- CLI command functionality and argument parsing
- Integration with step catalog system
- Error handling and user feedback
- Multi-workspace discovery workflows
- Framework detection and filtering
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call, Mock
from click.testing import CliRunner

# Import the CLI functions we want to test
try:
    from cursus.cli.catalog_cli import (
        catalog_cli,
        list_steps,
        search_steps,
        show_step,
        show_components,
        list_frameworks,
        list_workspaces,
        show_metrics,
        discover_workspace,
    )
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


@pytest.fixture
def runner():
    """Create CLI runner fixture"""
    return CliRunner()


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_step_catalog():
    """Create mock StepCatalog instance"""
    mock_catalog = Mock()
    
    # Mock step info
    mock_step_info = Mock()
    mock_step_info.step_name = "TestStep"
    mock_step_info.workspace_id = "core"
    mock_step_info.registry_data = {"config_class": "TestStepConfig"}
    mock_step_info.file_components = {
        "script": Mock(path="/path/to/script.py", file_type="script", modified_time=None),
        "config": Mock(path="/path/to/config.py", file_type="config", modified_time=None)
    }
    
    # Configure mock methods
    mock_catalog.list_available_steps.return_value = ["TestStep", "XGBoostTraining", "PyTorchTraining"]
    mock_catalog.get_step_info.return_value = mock_step_info
    mock_catalog.detect_framework.return_value = "xgboost"
    mock_catalog.search_steps.return_value = []
    mock_catalog.discover_cross_workspace_components.return_value = {"core": ["component1", "component2"]}
    mock_catalog.get_metrics_report.return_value = {
        "total_queries": 10,
        "success_rate": 0.9,
        "avg_response_time_ms": 5.0,
        "index_build_time_s": 0.1,
        "total_steps_indexed": 65,
        "total_workspaces": 2,
        "last_index_build": "2023-01-01T00:00:00"
    }
    mock_catalog.get_job_type_variants.return_value = ["training", "validation"]
    
    return mock_catalog


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCatalogCLIBasics:
    """Test basic catalog CLI functionality."""

    def test_catalog_cli_help(self, runner):
        """Test catalog CLI help message."""
        result = runner.invoke(catalog_cli, ["--help"])

        assert result.exit_code == 0
        assert "Step catalog management commands" in result.output
        assert "list" in result.output
        assert "search" in result.output
        assert "show" in result.output
        assert "components" in result.output
        assert "frameworks" in result.output
        assert "workspaces" in result.output
        assert "metrics" in result.output
        assert "discover" in result.output

    def test_catalog_cli_no_args(self, runner):
        """Test catalog CLI with no arguments shows help."""
        result = runner.invoke(catalog_cli, [])

        assert result.exit_code == 0
        assert "Step catalog management commands" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestListStepsCommand:
    """Test list command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_steps_basic(self, mock_catalog_class, mock_step_catalog, runner):
        """Test basic list command."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["list"])

        assert result.exit_code == 0
        assert "📂 Available Steps (3 found):" in result.output
        assert "TestStep" in result.output
        assert "XGBoostTraining" in result.output
        assert "PyTorchTraining" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_steps_with_workspace_filter(self, mock_catalog_class, mock_step_catalog, runner):
        """Test list command with workspace filter."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.list_available_steps.return_value = ["WorkspaceStep"]

        result = runner.invoke(catalog_cli, ["list", "--workspace", "my_workspace"])

        assert result.exit_code == 0
        assert "WorkspaceStep" in result.output
        mock_step_catalog.list_available_steps.assert_called_with(workspace_id="my_workspace", job_type=None)

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_steps_with_framework_filter(self, mock_catalog_class, mock_step_catalog, runner):
        """Test list command with framework filter."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.detect_framework.side_effect = lambda x: "xgboost" if "XGBoost" in x else None

        result = runner.invoke(catalog_cli, ["list", "--framework", "xgboost"])

        assert result.exit_code == 0
        assert "XGBoostTraining" in result.output
        assert "PyTorchTraining" not in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_steps_json_format(self, mock_catalog_class, mock_step_catalog, runner):
        """Test list command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["list", "--format", "json"])

        assert result.exit_code == 0
        assert '"steps":' in result.output
        assert '"total": 3' in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_steps_with_limit(self, mock_catalog_class, mock_step_catalog, runner):
        """Test list command with limit."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["list", "--limit", "2"])

        assert result.exit_code == 0
        assert "📂 Available Steps (2 found):" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_steps_error_handling(self, mock_catalog_class, runner):
        """Test list command error handling."""
        mock_catalog_class.side_effect = Exception("Test error")

        result = runner.invoke(catalog_cli, ["list"])

        assert result.exit_code == 0
        assert "❌ Failed to list steps: Test error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestSearchStepsCommand:
    """Test search command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_search_steps_basic(self, mock_catalog_class, mock_step_catalog, runner):
        """Test basic search command."""
        mock_catalog_class.return_value = mock_step_catalog
        
        # Mock search results
        mock_result = Mock()
        mock_result.step_name = "XGBoostTraining"
        mock_result.workspace_id = "core"
        mock_result.match_score = 0.9
        mock_result.match_reason = "name_match"
        mock_result.components_available = ["script", "config"]
        
        mock_step_catalog.search_steps.return_value = [mock_result]

        result = runner.invoke(catalog_cli, ["search", "training"])

        assert result.exit_code == 0
        assert "🔍 Search Results for 'training' (1 found):" in result.output
        assert "XGBoostTraining" in result.output
        assert "score: 0.90" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_search_steps_json_format(self, mock_catalog_class, mock_step_catalog, runner):
        """Test search command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.search_steps.return_value = []

        result = runner.invoke(catalog_cli, ["search", "training", "--format", "json"])

        assert result.exit_code == 0
        assert '"query": "training"' in result.output
        assert '"results":' in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_search_steps_with_job_type(self, mock_catalog_class, mock_step_catalog, runner):
        """Test search command with job type filter."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["search", "training", "--job-type", "validation"])

        assert result.exit_code == 0
        mock_step_catalog.search_steps.assert_called_with("training", job_type="validation")


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestShowStepCommand:
    """Test show command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_step_basic(self, mock_catalog_class, mock_step_catalog, runner):
        """Test basic show command."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["show", "TestStep"])

        assert result.exit_code == 0
        assert "📋 Step: TestStep" in result.output
        assert "Workspace: core" in result.output
        assert "🔧 Available Components:" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_step_not_found(self, mock_catalog_class, mock_step_catalog, runner):
        """Test show command with non-existent step."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.get_step_info.return_value = None

        result = runner.invoke(catalog_cli, ["show", "NonExistentStep"])

        assert result.exit_code == 0
        assert "❌ Step not found: NonExistentStep" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_step_json_format(self, mock_catalog_class, mock_step_catalog, runner):
        """Test show command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["show", "TestStep", "--format", "json"])

        assert result.exit_code == 0
        assert '"step_name": "TestStep"' in result.output
        assert '"workspace_id": "core"' in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_step_with_variants(self, mock_catalog_class, mock_step_catalog, runner):
        """Test show command showing job type variants."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["show", "TestStep"])

        assert result.exit_code == 0
        assert "🔄 Job Type Variants:" in result.output
        assert "TestStep_training" in result.output
        assert "TestStep_validation" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestComponentsCommand:
    """Test components command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_components_basic(self, mock_catalog_class, mock_step_catalog, runner):
        """Test basic components command."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["components", "TestStep"])

        assert result.exit_code == 0
        assert "🔧 Components for TestStep:" in result.output
        assert "SCRIPT:" in result.output
        assert "CONFIG:" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_components_with_type_filter(self, mock_catalog_class, mock_step_catalog, runner):
        """Test components command with type filter."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["components", "TestStep", "--type", "script"])

        assert result.exit_code == 0
        assert "SCRIPT:" in result.output
        assert "CONFIG:" not in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_components_json_format(self, mock_catalog_class, mock_step_catalog, runner):
        """Test components command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["components", "TestStep", "--format", "json"])

        assert result.exit_code == 0
        assert '"step_name": "TestStep"' in result.output
        assert '"components":' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestFrameworksCommand:
    """Test frameworks command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_frameworks_basic(self, mock_catalog_class, mock_step_catalog, runner):
        """Test basic frameworks command."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.detect_framework.side_effect = lambda x: "xgboost" if "XGBoost" in x else "pytorch" if "PyTorch" in x else None

        result = runner.invoke(catalog_cli, ["frameworks"])

        assert result.exit_code == 0
        assert "🔧 Detected Frameworks" in result.output
        assert "xgboost:" in result.output
        assert "pytorch:" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_frameworks_json_format(self, mock_catalog_class, mock_step_catalog, runner):
        """Test frameworks command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.detect_framework.return_value = "xgboost"

        result = runner.invoke(catalog_cli, ["frameworks", "--format", "json"])

        assert result.exit_code == 0
        assert '"frameworks":' in result.output
        assert '"steps_by_framework":' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestWorkspacesCommand:
    """Test workspaces command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_workspaces_basic(self, mock_catalog_class, mock_step_catalog, runner):
        """Test basic workspaces command."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["workspaces"])

        assert result.exit_code == 0
        assert "🏢 Available Workspaces" in result.output
        assert "core:" in result.output
        assert "Steps:" in result.output
        assert "Components:" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_list_workspaces_json_format(self, mock_catalog_class, mock_step_catalog, runner):
        """Test workspaces command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["workspaces", "--format", "json"])

        assert result.exit_code == 0
        assert '"workspaces":' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestMetricsCommand:
    """Test metrics command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_metrics_basic(self, mock_catalog_class, mock_step_catalog, runner):
        """Test basic metrics command."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["metrics"])

        assert result.exit_code == 0
        assert "📊 Step Catalog Metrics:" in result.output
        assert "Total Queries: 10" in result.output
        assert "Success Rate: 90.0%" in result.output
        assert "Total Steps Indexed: 65" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_metrics_json_format(self, mock_catalog_class, mock_step_catalog, runner):
        """Test metrics command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog

        result = runner.invoke(catalog_cli, ["metrics", "--format", "json"])

        assert result.exit_code == 0
        assert '"total_queries": 10' in result.output
        assert '"success_rate": 0.9' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestDiscoverCommand:
    """Test discover command."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_discover_workspace_basic(self, mock_catalog_class, mock_step_catalog, runner, temp_workspace):
        """Test basic discover command."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.list_available_steps.return_value = ["DiscoveredStep"]

        result = runner.invoke(catalog_cli, ["discover", "--workspace-dir", temp_workspace])

        assert result.exit_code == 0
        assert "🔍 Discovery Results for" in result.output
        assert "DiscoveredStep" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_discover_workspace_json_format(self, mock_catalog_class, mock_step_catalog, runner, temp_workspace):
        """Test discover command with JSON format."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.list_available_steps.return_value = ["DiscoveredStep"]

        result = runner.invoke(catalog_cli, ["discover", "--workspace-dir", temp_workspace, "--format", "json"])

        assert result.exit_code == 0
        assert '"workspace_dir":' in result.output
        assert '"discovered_steps":' in result.output

    def test_discover_workspace_no_dir(self, runner):
        """Test discover command without workspace directory."""
        result = runner.invoke(catalog_cli, ["discover"])

        assert result.exit_code == 0
        assert "❌ Please specify a workspace directory" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_search_steps_error_handling(self, mock_catalog_class, runner):
        """Test search command error handling."""
        mock_catalog_class.side_effect = Exception("Search error")

        result = runner.invoke(catalog_cli, ["search", "test"])

        assert result.exit_code == 0
        assert "❌ Failed to search steps: Search error" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_show_step_error_handling(self, mock_catalog_class, runner):
        """Test show command error handling."""
        mock_catalog_class.side_effect = Exception("Show error")

        result = runner.invoke(catalog_cli, ["show", "TestStep"])

        assert result.exit_code == 0
        assert "❌ Failed to show step: Show error" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_components_error_handling(self, mock_catalog_class, runner):
        """Test components command error handling."""
        mock_catalog_class.side_effect = Exception("Components error")

        result = runner.invoke(catalog_cli, ["components", "TestStep"])

        assert result.exit_code == 0
        assert "❌ Failed to show components: Components error" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_frameworks_error_handling(self, mock_catalog_class, runner):
        """Test frameworks command error handling."""
        mock_catalog_class.side_effect = Exception("Frameworks error")

        result = runner.invoke(catalog_cli, ["frameworks"])

        assert result.exit_code == 0
        assert "❌ Failed to list frameworks: Frameworks error" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_workspaces_error_handling(self, mock_catalog_class, runner):
        """Test workspaces command error handling."""
        mock_catalog_class.side_effect = Exception("Workspaces error")

        result = runner.invoke(catalog_cli, ["workspaces"])

        assert result.exit_code == 0
        assert "❌ Failed to list workspaces: Workspaces error" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_metrics_error_handling(self, mock_catalog_class, runner):
        """Test metrics command error handling."""
        mock_catalog_class.side_effect = Exception("Metrics error")

        result = runner.invoke(catalog_cli, ["metrics"])

        assert result.exit_code == 0
        assert "❌ Failed to show metrics: Metrics error" in result.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_discover_error_handling(self, mock_catalog_class, runner, temp_workspace):
        """Test discover command error handling."""
        mock_catalog_class.side_effect = Exception("Discover error")

        result = runner.invoke(catalog_cli, ["discover", "--workspace-dir", temp_workspace])

        assert result.exit_code == 0
        assert "❌ Failed to discover workspace: Discover error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIIntegrationScenarios:
    """Test realistic CLI usage scenarios."""

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_step_discovery_workflow(self, mock_catalog_class, mock_step_catalog, runner):
        """Test complete step discovery workflow."""
        mock_catalog_class.return_value = mock_step_catalog

        # Step 1: List all steps
        result1 = runner.invoke(catalog_cli, ["list"])
        assert result1.exit_code == 0
        assert "TestStep" in result1.output

        # Step 2: Search for specific steps
        mock_result = Mock()
        mock_result.step_name = "XGBoostTraining"
        mock_result.workspace_id = "core"
        mock_result.match_score = 0.9
        mock_result.match_reason = "name_match"
        mock_result.components_available = ["script", "config"]
        mock_step_catalog.search_steps.return_value = [mock_result]

        result2 = runner.invoke(catalog_cli, ["search", "training"])
        assert result2.exit_code == 0
        assert "XGBoostTraining" in result2.output

        # Step 3: Show detailed step information
        result3 = runner.invoke(catalog_cli, ["show", "TestStep"])
        assert result3.exit_code == 0
        assert "📋 Step: TestStep" in result3.output

        # Step 4: Show step components
        result4 = runner.invoke(catalog_cli, ["components", "TestStep"])
        assert result4.exit_code == 0
        assert "🔧 Components for TestStep:" in result4.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_framework_analysis_workflow(self, mock_catalog_class, mock_step_catalog, runner):
        """Test framework analysis workflow."""
        mock_catalog_class.return_value = mock_step_catalog
        mock_step_catalog.detect_framework.side_effect = lambda x: "xgboost" if "XGBoost" in x else "pytorch" if "PyTorch" in x else None

        # Step 1: List frameworks
        result1 = runner.invoke(catalog_cli, ["frameworks"])
        assert result1.exit_code == 0
        assert "🔧 Detected Frameworks" in result1.output

        # Step 2: Filter steps by framework
        result2 = runner.invoke(catalog_cli, ["list", "--framework", "xgboost"])
        assert result2.exit_code == 0
        assert "XGBoostTraining" in result2.output

        # Step 3: Show metrics
        result3 = runner.invoke(catalog_cli, ["metrics"])
        assert result3.exit_code == 0
        assert "📊 Step Catalog Metrics:" in result3.output

    @patch("cursus.cli.catalog_cli.StepCatalog")
    def test_workspace_management_workflow(self, mock_catalog_class, mock_step_catalog, runner, temp_workspace):
        """Test workspace management workflow."""
        mock_catalog_class.return_value = mock_step_catalog

        # Step 1: List workspaces
        result1 = runner.invoke(catalog_cli, ["workspaces"])
        assert result1.exit_code == 0
        assert "🏢 Available Workspaces" in result1.output

        # Step 2: Discover new workspace
        mock_step_catalog.list_available_steps.return_value = ["NewWorkspaceStep"]
        result2 = runner.invoke(catalog_cli, ["discover", "--workspace-dir", temp_workspace])
        assert result2.exit_code == 0
        assert "🔍 Discovery Results for" in result2.output

        # Step 3: List steps from specific workspace
        result3 = runner.invoke(catalog_cli, ["list", "--workspace", "test_workspace"])
        assert result3.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
