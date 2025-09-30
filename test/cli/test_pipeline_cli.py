"""
Comprehensive tests for pipeline CLI commands.

This module tests all pipeline CLI commands for pipeline catalog management including
pipeline discovery, connections, recommendations, validation, and statistics.

Tests focus on:
- CLI command functionality and argument parsing
- Integration with pipeline catalog system
- Error handling and user feedback
- Pipeline discovery and recommendation workflows
- Connection traversal and validation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call, Mock
from click.testing import CliRunner

# Import the CLI functions we want to test
try:
    from cursus.cli.pipeline_cli import (
        pipeline_cli,
        list_pipelines,
        discover_pipelines,
        show_pipeline,
        show_connections,
        show_alternatives,
        find_path,
        recommend_pipelines,
        validate_registry,
        show_stats,
    )
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False


@pytest.fixture
def runner():
    """Create CLI runner fixture"""
    return CliRunner()


@pytest.fixture
def mock_pipeline_manager():
    """Create mock PipelineCatalogManager instance"""
    mock_manager = Mock()
    
    # Configure mock methods
    mock_manager.discover_pipelines.return_value = ["xgb_training_simple", "pytorch_e2e_standard", "dummy_e2e_basic"]
    mock_manager.get_pipeline_connections.return_value = {
        "alternatives": ["xgb_training_evaluation", "xgb_training_calibrated"],
        "dependencies": ["data_preprocessing"],
        "successors": ["model_evaluation"]
    }
    mock_manager.find_path.return_value = ["source_pipeline", "intermediate_pipeline", "target_pipeline"]
    mock_manager.get_recommendations.return_value = [
        {
            "pipeline_id": "xgb_e2e_comprehensive",
            "score": 0.95,
            "reason": "Perfect match for use case"
        },
        {
            "pipeline_id": "pytorch_e2e_standard", 
            "score": 0.8,
            "reason": "Good alternative framework"
        }
    ]
    mock_manager.validate_registry.return_value = {
        "is_valid": True,
        "total_issues": 0,
        "issues_by_severity": {},
        "issues_by_category": {},
        "all_issues": []
    }
    mock_manager.get_registry_stats.return_value = {
        "total_pipelines": 8,
        "frameworks": ["xgboost", "pytorch", "dummy"],
        "complexity_levels": ["simple", "standard", "comprehensive"]
    }
    
    return mock_manager


@pytest.fixture
def mock_catalog_info():
    """Create mock catalog info"""
    return {
        "total_pipelines": 8,
        "standard_pipelines": 5,
        "mods_pipelines": 2,
        "shared_dags": 7,
        "frameworks": ["xgboost", "pytorch", "dummy"],
        "complexity_levels": ["simple", "standard", "comprehensive"],
        "last_updated": "2023-01-01T00:00:00"
    }


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestPipelineCLIBasics:
    """Test basic pipeline CLI functionality."""

    def test_pipeline_cli_help(self, runner):
        """Test pipeline CLI help message."""
        result = runner.invoke(pipeline_cli, ["--help"])

        assert result.exit_code == 0
        assert "Pipeline catalog management commands" in result.output
        assert "list" in result.output
        assert "discover" in result.output
        assert "show" in result.output
        assert "connections" in result.output
        assert "alternatives" in result.output
        assert "path" in result.output
        assert "recommend" in result.output
        assert "validate" in result.output
        assert "stats" in result.output

    def test_pipeline_cli_no_args(self, runner):
        """Test pipeline CLI with no arguments shows help."""
        result = runner.invoke(pipeline_cli, [])

        assert result.exit_code == 0
        assert "Pipeline catalog management commands" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestListPipelinesCommand:
    """Test list command."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    @patch("cursus.pipeline_catalog.discover_all_pipelines")
    def test_list_pipelines_basic(self, mock_discover_all, mock_create_manager, mock_pipeline_manager, runner):
        """Test basic list command."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_discover_all.return_value = {
            "standard": ["xgb_training_simple", "pytorch_e2e_standard"],
            "mods": ["xgb_mods_e2e_comprehensive"]
        }

        result = runner.invoke(pipeline_cli, ["list"])

        assert result.exit_code == 0
        assert "📂 Available Pipelines (3 found):" in result.output
        assert "xgb_training_simple" in result.output
        assert "pytorch_e2e_standard" in result.output
        assert "xgb_mods_e2e_comprehensive" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_list_pipelines_with_framework_filter(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test list command with framework filter."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_pipeline_manager.discover_pipelines.return_value = ["xgb_training_simple", "xgb_e2e_comprehensive"]

        result = runner.invoke(pipeline_cli, ["list", "--framework", "xgboost"])

        assert result.exit_code == 0
        assert "xgb_training_simple" in result.output
        mock_pipeline_manager.discover_pipelines.assert_called_with(framework="xgboost")

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_list_pipelines_json_format(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test list command with JSON format."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["list", "--format", "json"])

        assert result.exit_code == 0
        assert '"pipelines":' in result.output
        assert '"total":' in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_list_pipelines_with_limit(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test list command with limit."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["list", "--limit", "2"])

        assert result.exit_code == 0
        assert "📂 Available Pipelines (2 found):" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_list_pipelines_error_handling(self, mock_create_manager, runner):
        """Test list command error handling."""
        mock_create_manager.side_effect = Exception("Test error")

        result = runner.invoke(pipeline_cli, ["list"])

        assert result.exit_code == 0
        assert "❌ Failed to list pipelines: Test error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestDiscoverPipelinesCommand:
    """Test discover command."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_discover_pipelines_by_framework(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test discover command by framework."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_pipeline_manager.discover_pipelines.return_value = ["xgb_training_simple", "xgb_e2e_comprehensive"]

        result = runner.invoke(pipeline_cli, ["discover", "--framework", "xgboost"])

        assert result.exit_code == 0
        assert "🔍 Discovery Results (2 found):" in result.output
        assert "xgb_training_simple" in result.output
        assert "xgb_e2e_comprehensive" in result.output
        mock_pipeline_manager.discover_pipelines.assert_called_with(framework="xgboost")

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_discover_pipelines_by_use_case(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test discover command by use case."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["discover", "--use-case", "model training"])

        assert result.exit_code == 0
        mock_pipeline_manager.discover_pipelines.assert_called_with(use_case="model training")

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_discover_pipelines_json_format(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test discover command with JSON format."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["discover", "--framework", "pytorch", "--format", "json"])

        assert result.exit_code == 0
        assert '"results":' in result.output
        assert '"criteria":' in result.output

    def test_discover_pipelines_no_criteria(self, runner):
        """Test discover command without criteria."""
        result = runner.invoke(pipeline_cli, ["discover"])

        assert result.exit_code == 0
        assert "❌ Please specify at least one search criterion" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestShowPipelineCommand:
    """Test show command."""

    @patch("cursus.pipeline_catalog.load_pipeline")
    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_show_pipeline_basic(self, mock_create_manager, mock_load_pipeline, mock_pipeline_manager, runner):
        """Test basic show command."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_pipeline_info = Mock()
        mock_pipeline_info.name = "xgb_training_simple"
        mock_pipeline_info.framework = "xgboost"
        mock_load_pipeline.return_value = mock_pipeline_info

        result = runner.invoke(pipeline_cli, ["show", "xgb_training_simple"])

        assert result.exit_code == 0
        assert "📋 Pipeline: xgb_training_simple" in result.output

    @patch("cursus.pipeline_catalog.load_pipeline")
    def test_show_pipeline_not_found(self, mock_load_pipeline, runner):
        """Test show command with non-existent pipeline."""
        mock_load_pipeline.side_effect = Exception("Pipeline not found")

        result = runner.invoke(pipeline_cli, ["show", "nonexistent_pipeline"])

        assert result.exit_code == 0
        assert "Pipeline information not available" in result.output

    @patch("cursus.pipeline_catalog.load_pipeline")
    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_show_pipeline_with_connections(self, mock_create_manager, mock_load_pipeline, mock_pipeline_manager, runner):
        """Test show command with connections."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_load_pipeline.return_value = Mock()

        result = runner.invoke(pipeline_cli, ["show", "xgb_training_simple", "--show-connections"])

        assert result.exit_code == 0
        assert "🔗 Connections:" in result.output
        assert "Alternatives:" in result.output

    @patch("cursus.pipeline_catalog.load_pipeline")
    def test_show_pipeline_json_format(self, mock_load_pipeline, runner):
        """Test show command with JSON format."""
        mock_load_pipeline.return_value = Mock()

        result = runner.invoke(pipeline_cli, ["show", "xgb_training_simple", "--format", "json"])

        assert result.exit_code == 0
        assert '"pipeline_id": "xgb_training_simple"' in result.output
        assert '"found":' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestConnectionsCommand:
    """Test connections command."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_show_connections_basic(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test basic connections command."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["connections", "xgb_training_simple"])

        assert result.exit_code == 0
        assert "🔗 Connections for xgb_training_simple:" in result.output
        assert "ALTERNATIVES:" in result.output
        assert "→ xgb_training_evaluation" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_show_connections_json_format(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test connections command with JSON format."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["connections", "xgb_training_simple", "--format", "json"])

        assert result.exit_code == 0
        assert '"pipeline_id": "xgb_training_simple"' in result.output
        assert '"connections":' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestAlternativesCommand:
    """Test alternatives command."""

    @patch("cursus.pipeline_catalog.get_pipeline_alternatives")
    def test_show_alternatives_basic(self, mock_get_alternatives, runner):
        """Test basic alternatives command."""
        mock_get_alternatives.return_value = ["xgb_training_evaluation", "xgb_training_calibrated"]

        result = runner.invoke(pipeline_cli, ["alternatives", "xgb_training_simple"])

        assert result.exit_code == 0
        assert "🔄 Alternatives to xgb_training_simple:" in result.output
        assert "xgb_training_evaluation" in result.output
        assert "xgb_training_calibrated" in result.output

    @patch("cursus.pipeline_catalog.get_pipeline_alternatives")
    def test_show_alternatives_json_format(self, mock_get_alternatives, runner):
        """Test alternatives command with JSON format."""
        mock_get_alternatives.return_value = ["alternative1", "alternative2"]

        result = runner.invoke(pipeline_cli, ["alternatives", "xgb_training_simple", "--format", "json"])

        assert result.exit_code == 0
        assert '"pipeline_id": "xgb_training_simple"' in result.output
        assert '"alternatives":' in result.output

    @patch("cursus.pipeline_catalog.get_pipeline_alternatives")
    def test_show_alternatives_none_found(self, mock_get_alternatives, runner):
        """Test alternatives command with no alternatives."""
        mock_get_alternatives.return_value = []

        result = runner.invoke(pipeline_cli, ["alternatives", "unique_pipeline"])

        assert result.exit_code == 0
        assert "No alternatives found." in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestPathCommand:
    """Test path command."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_find_path_success(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test successful path finding."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["path", "source_pipeline", "target_pipeline"])

        assert result.exit_code == 0
        assert "🛤️  Path from source_pipeline to target_pipeline:" in result.output
        assert "Start: source_pipeline" in result.output
        assert "End:   target_pipeline" in result.output
        assert "Path length: 3 steps" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_find_path_not_found(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test path finding with no path."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_pipeline_manager.find_path.return_value = None

        result = runner.invoke(pipeline_cli, ["path", "isolated1", "isolated2"])

        assert result.exit_code == 0
        assert "No connection path found." in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_find_path_json_format(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test path command with JSON format."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["path", "source", "target", "--format", "json"])

        assert result.exit_code == 0
        assert '"source": "source"' in result.output
        assert '"target": "target"' in result.output
        assert '"path":' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestRecommendCommand:
    """Test recommend command."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_recommend_pipelines_basic(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test basic recommend command."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["recommend", "--use-case", "model training"])

        assert result.exit_code == 0
        assert "💡 Recommendations for: model training" in result.output
        assert "xgb_e2e_comprehensive (score: 0.95)" in result.output
        assert "Perfect match for use case" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_recommend_pipelines_with_criteria(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test recommend command with additional criteria."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["recommend", "--use-case", "training", "--framework", "xgboost"])

        assert result.exit_code == 0
        mock_pipeline_manager.get_recommendations.assert_called_with("training", framework="xgboost")

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_recommend_pipelines_json_format(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test recommend command with JSON format."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["recommend", "--use-case", "training", "--format", "json"])

        assert result.exit_code == 0
        assert '"use_case": "training"' in result.output
        assert '"recommendations":' in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_recommend_pipelines_no_results(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test recommend command with no recommendations."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_pipeline_manager.get_recommendations.return_value = []

        result = runner.invoke(pipeline_cli, ["recommend", "--use-case", "unknown"])

        assert result.exit_code == 0
        assert "No recommendations found." in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestValidateCommand:
    """Test validate command."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_validate_registry_success(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test successful registry validation."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["validate"])

        assert result.exit_code == 0
        assert "🔍 Registry Validation Results:" in result.output
        assert "✅ Registry is valid" in result.output
        assert "Total issues: 0" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_validate_registry_with_issues(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test registry validation with issues."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_pipeline_manager.validate_registry.return_value = {
            "is_valid": False,
            "total_issues": 2,
            "issues_by_severity": {"error": 1, "warning": 1},
            "all_issues": [
                {"message": "Missing pipeline file", "severity": "error"},
                {"message": "Deprecated configuration", "severity": "warning"}
            ]
        }

        result = runner.invoke(pipeline_cli, ["validate"])

        assert result.exit_code == 0
        assert "❌ Registry validation failed" in result.output
        assert "Total issues: 2" in result.output
        assert "error: 1" in result.output
        assert "Missing pipeline file" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_validate_registry_json_format(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test validate command with JSON format."""
        mock_create_manager.return_value = mock_pipeline_manager

        result = runner.invoke(pipeline_cli, ["validate", "--format", "json"])

        assert result.exit_code == 0
        assert '"is_valid": true' in result.output
        assert '"total_issues": 0' in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestStatsCommand:
    """Test stats command."""

    @patch("cursus.pipeline_catalog.get_catalog_info")
    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_show_stats_basic(self, mock_create_manager, mock_get_catalog_info, mock_catalog_info, mock_pipeline_manager, runner):
        """Test basic stats command."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_get_catalog_info.return_value = mock_catalog_info

        result = runner.invoke(pipeline_cli, ["stats"])

        assert result.exit_code == 0
        assert "📊 Pipeline Catalog Statistics:" in result.output
        assert "Total Pipelines: 8" in result.output
        assert "Standard: 5" in result.output
        assert "MODS: 2" in result.output
        assert "Shared DAGs: 7" in result.output

    @patch("cursus.pipeline_catalog.get_catalog_info")
    def test_show_stats_json_format(self, mock_get_catalog_info, mock_catalog_info, runner):
        """Test stats command with JSON format."""
        mock_get_catalog_info.return_value = mock_catalog_info

        result = runner.invoke(pipeline_cli, ["stats", "--format", "json"])

        assert result.exit_code == 0
        assert '"total_pipelines": 8' in result.output
        assert '"frameworks":' in result.output

    @patch("cursus.pipeline_catalog.get_catalog_info")
    def test_show_stats_with_error(self, mock_get_catalog_info, runner):
        """Test stats command with error info."""
        mock_get_catalog_info.return_value = {
            "error": "Failed to load some statistics",
            "standard_pipelines": 5,
            "mods_pipelines": 2,
            "shared_dags": 7
        }

        result = runner.invoke(pipeline_cli, ["stats"])

        assert result.exit_code == 0
        assert "⚠️  Failed to load some statistics" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_discover_error_handling(self, mock_create_manager, runner):
        """Test discover command error handling."""
        mock_create_manager.side_effect = Exception("Discovery error")

        result = runner.invoke(pipeline_cli, ["discover", "--framework", "xgboost"])

        assert result.exit_code == 0
        assert "❌ Failed to discover pipelines: Discovery error" in result.output

    @patch("cursus.pipeline_catalog.load_pipeline")
    def test_show_error_handling(self, mock_load_pipeline, runner):
        """Test show command error handling."""
        mock_load_pipeline.side_effect = Exception("Show error")

        result = runner.invoke(pipeline_cli, ["show", "test_pipeline"])

        assert result.exit_code == 0
        assert "❌ Failed to show pipeline: Show error" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_connections_error_handling(self, mock_create_manager, runner):
        """Test connections command error handling."""
        mock_create_manager.side_effect = Exception("Connections error")

        result = runner.invoke(pipeline_cli, ["connections", "test_pipeline"])

        assert result.exit_code == 0
        assert "❌ Failed to show connections: Connections error" in result.output

    @patch("cursus.pipeline_catalog.get_pipeline_alternatives")
    def test_alternatives_error_handling(self, mock_get_alternatives, runner):
        """Test alternatives command error handling."""
        mock_get_alternatives.side_effect = Exception("Alternatives error")

        result = runner.invoke(pipeline_cli, ["alternatives", "test_pipeline"])

        assert result.exit_code == 0
        assert "❌ Failed to show alternatives: Alternatives error" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_path_error_handling(self, mock_create_manager, runner):
        """Test path command error handling."""
        mock_create_manager.side_effect = Exception("Path error")

        result = runner.invoke(pipeline_cli, ["path", "source", "target"])

        assert result.exit_code == 0
        assert "❌ Failed to find path: Path error" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_recommend_error_handling(self, mock_create_manager, runner):
        """Test recommend command error handling."""
        mock_create_manager.side_effect = Exception("Recommend error")

        result = runner.invoke(pipeline_cli, ["recommend", "--use-case", "training"])

        assert result.exit_code == 0
        assert "❌ Failed to get recommendations: Recommend error" in result.output

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_validate_error_handling(self, mock_create_manager, runner):
        """Test validate command error handling."""
        mock_create_manager.side_effect = Exception("Validate error")

        result = runner.invoke(pipeline_cli, ["validate"])

        assert result.exit_code == 0
        assert "❌ Failed to validate registry: Validate error" in result.output

    @patch("cursus.pipeline_catalog.get_catalog_info")
    def test_stats_error_handling(self, mock_get_catalog_info, runner):
        """Test stats command error handling."""
        mock_get_catalog_info.side_effect = Exception("Stats error")

        result = runner.invoke(pipeline_cli, ["stats"])

        assert result.exit_code == 0
        assert "❌ Failed to get statistics: Stats error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIIntegrationScenarios:
    """Test realistic CLI usage scenarios."""

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    @patch("cursus.pipeline_catalog.discover_all_pipelines")
    def test_pipeline_discovery_workflow(self, mock_discover_all, mock_create_manager, mock_pipeline_manager, runner):
        """Test complete pipeline discovery workflow."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_discover_all.return_value = {"standard": ["xgb_training_simple"], "mods": []}

        # Step 1: List all pipelines
        result1 = runner.invoke(pipeline_cli, ["list"])
        assert result1.exit_code == 0
        assert "xgb_training_simple" in result1.output

        # Step 2: Discover by framework
        result2 = runner.invoke(pipeline_cli, ["discover", "--framework", "xgboost"])
        assert result2.exit_code == 0

        # Step 3: Show pipeline details
        result3 = runner.invoke(pipeline_cli, ["show", "xgb_training_simple"])
        assert result3.exit_code == 0

        # Step 4: Show connections
        result4 = runner.invoke(pipeline_cli, ["connections", "xgb_training_simple"])
        assert result4.exit_code == 0

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    def test_pipeline_recommendation_workflow(self, mock_create_manager, mock_pipeline_manager, runner):
        """Test pipeline recommendation workflow."""
        mock_create_manager.return_value = mock_pipeline_manager

        # Step 1: Get recommendations
        result1 = runner.invoke(pipeline_cli, ["recommend", "--use-case", "model training"])
        assert result1.exit_code == 0
        assert "xgb_e2e_comprehensive" in result1.output

        # Step 2: Show recommended pipeline
        result2 = runner.invoke(pipeline_cli, ["show", "xgb_e2e_comprehensive"])
        assert result2.exit_code == 0

        # Step 3: Find alternatives
        result3 = runner.invoke(pipeline_cli, ["alternatives", "xgb_e2e_comprehensive"])
        assert result3.exit_code == 0

    @patch("cursus.pipeline_catalog.create_catalog_manager")
    @patch("cursus.pipeline_catalog.get_catalog_info")
    def test_pipeline_validation_workflow(self, mock_get_catalog_info, mock_create_manager, mock_catalog_info, mock_pipeline_manager, runner):
        """Test pipeline validation workflow."""
        mock_create_manager.return_value = mock_pipeline_manager
        mock_get_catalog_info.return_value = mock_catalog_info

        # Step 1: Show statistics
        result1 = runner.invoke(pipeline_cli, ["stats"])
        assert result1.exit_code == 0
        assert "Total Pipelines: 8" in result1.output

        # Step 2: Validate registry
        result2 = runner.invoke(pipeline_cli, ["validate"])
        assert result2.exit_code == 0
        assert "✅ Registry is valid" in result2.output

        # Step 3: Find path between pipelines
        result3 = runner.invoke(pipeline_cli, ["path", "source_pipeline", "target_pipeline"])
        assert result3.exit_code == 0
        assert "Path length: 3 steps" in result3.output


if __name__ == "__main__":
    pytest.main([__file__])
