"""
Comprehensive tests for registry CLI commands.

This module tests all registry CLI commands including workspace management,
step listing, registry validation, step resolution, and validation utilities.

Tests focus on:
- CLI command functionality and argument parsing
- Integration with registry and validation utilities
- Error handling and user feedback
- Workspace management workflows
- Performance metrics and validation
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, call, Mock
from click.testing import CliRunner

# Import the CLI functions we want to test
try:
    from cursus.cli.registry_cli import (
        registry_cli,
        init_workspace,
        list_steps,
        validate_registry,
        resolve_step,
        validate_step_definition,
        validation_status,
        reset_validation_metrics,
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


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestRegistryCLIBasics:
    """Test basic registry CLI functionality."""

    def test_registry_cli_help(self, runner):
        """Test registry CLI help message."""
        result = runner.invoke(registry_cli, ["--help"])

        assert result.exit_code == 0
        assert "Registry management commands" in result.output
        assert "init-workspace" in result.output
        assert "list-steps" in result.output
        assert "validate-registry" in result.output
        assert "resolve-step" in result.output
        assert "validate-step-definition" in result.output
        assert "validation-status" in result.output
        assert "reset-validation-metrics" in result.output

    def test_registry_cli_no_args(self, runner):
        """Test registry CLI with no arguments shows help."""
        result = runner.invoke(registry_cli, [])

        assert result.exit_code == 0
        assert "Registry management commands" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestInitWorkspaceCommand:
    """Test init-workspace command."""

    def test_init_workspace_minimal(self, runner, temp_workspace):
        """Test init-workspace with minimal arguments."""
        workspace_id = "test_developer"
        workspace_path = str(Path(temp_workspace) / workspace_id)

        result = runner.invoke(
            registry_cli,
            [
                "init-workspace",
                workspace_id,
                "--workspace-path",
                workspace_path,
            ],
        )

        assert result.exit_code == 0
        assert f"🚀 Initializing developer workspace: {workspace_id}" in result.output
        assert "✅ Created workspace directory structure" in result.output
        assert "✅ Created standard registry template" in result.output
        assert "✅ Created workspace documentation" in result.output
        assert "🎉 Developer workspace successfully created!" in result.output

        # Verify directory structure was created
        workspace_dir = Path(workspace_path)
        assert workspace_dir.exists()
        assert (workspace_dir / "src" / "cursus_dev" / "steps").exists()
        assert (workspace_dir / "src" / "cursus_dev" / "registry").exists()
        assert (workspace_dir / "README.md").exists()

    def test_init_workspace_with_template(self, runner, temp_workspace):
        """Test init-workspace with different templates."""
        workspace_id = "test_developer"
        workspace_path = str(Path(temp_workspace) / workspace_id)

        for template in ["minimal", "standard", "advanced"]:
            result = runner.invoke(
                registry_cli,
                [
                    "init-workspace",
                    workspace_id,
                    "--workspace-path",
                    workspace_path,
                    "--template",
                    template,
                    "--force",
                ],
            )

            assert result.exit_code == 0
            assert f"✅ Created {template} registry template" in result.output

    def test_init_workspace_invalid_id(self, runner):
        """Test init-workspace with invalid workspace ID."""
        result = runner.invoke(
            registry_cli,
            ["init-workspace", "invalid@workspace!"],
        )

        assert result.exit_code == 0
        assert "❌ Invalid workspace ID" in result.output

    def test_init_workspace_existing_without_force(self, runner, temp_workspace):
        """Test init-workspace with existing workspace without force."""
        workspace_id = "test_developer"
        workspace_path = str(Path(temp_workspace) / workspace_id)

        # Create workspace first time
        result1 = runner.invoke(
            registry_cli,
            [
                "init-workspace",
                workspace_id,
                "--workspace-path",
                workspace_path,
            ],
        )
        assert result1.exit_code == 0

        # Try to create again without force
        result2 = runner.invoke(
            registry_cli,
            [
                "init-workspace",
                workspace_id,
                "--workspace-path",
                workspace_path,
            ],
        )

        assert result2.exit_code == 0
        assert "❌ Workspace already exists" in result2.output

    def test_init_workspace_with_force(self, runner, temp_workspace):
        """Test init-workspace with force overwrite."""
        workspace_id = "test_developer"
        workspace_path = str(Path(temp_workspace) / workspace_id)

        # Create workspace first time
        result1 = runner.invoke(
            registry_cli,
            [
                "init-workspace",
                workspace_id,
                "--workspace-path",
                workspace_path,
            ],
        )
        assert result1.exit_code == 0

        # Create again with force
        result2 = runner.invoke(
            registry_cli,
            [
                "init-workspace",
                workspace_id,
                "--workspace-path",
                workspace_path,
                "--force",
            ],
        )

        assert result2.exit_code == 0
        assert "🎉 Developer workspace successfully created!" in result2.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestListStepsCommand:
    """Test list-steps command."""

    @patch("cursus.registry.get_all_step_names")
    @patch("cursus.registry.get_workspace_context")
    def test_list_steps_basic(self, mock_get_context, mock_get_steps, runner):
        """Test basic list-steps command."""
        mock_get_context.return_value = None
        mock_get_steps.return_value = ["StepA", "StepB", "StepC"]

        result = runner.invoke(registry_cli, ["list-steps"])

        assert result.exit_code == 0
        assert "📂 Available Steps (core registry) (3 total):" in result.output
        assert "- StepA" in result.output
        assert "- StepB" in result.output
        assert "- StepC" in result.output

    @patch("cursus.cli.registry_cli.get_all_step_names")
    @patch("cursus.cli.registry_cli.get_workspace_context")
    def test_list_steps_with_workspace(self, mock_get_context, mock_get_steps, runner):
        """Test list-steps with workspace context."""
        mock_get_context.return_value = "test_workspace"
        mock_get_steps.return_value = ["StepA", "StepB", "WorkspaceStep"]

        result = runner.invoke(registry_cli, ["list-steps", "--workspace", "test_workspace"])

        assert result.exit_code == 0
        assert "📂 Available Steps (workspace: test_workspace) (3 total):" in result.output
        assert "- WorkspaceStep" in result.output

    @patch("cursus.cli.registry_cli.UnifiedRegistryManager")
    def test_list_steps_with_source(self, mock_manager_class, runner):
        """Test list-steps with source information."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.list_all_steps.return_value = {
            "core": ["CoreStepA", "CoreStepB"],
            "workspace1": ["WorkspaceStepA"],
        }

        result = runner.invoke(registry_cli, ["list-steps", "--include-source"])

        assert result.exit_code == 0
        assert "📂 CORE Registry:" in result.output
        assert "- CoreStepA" in result.output
        assert "📂 WORKSPACE1 Registry:" in result.output
        assert "- WorkspaceStepA" in result.output

    @patch("cursus.cli.registry_cli.UnifiedRegistryManager")
    def test_list_steps_conflicts_only(self, mock_manager_class, runner):
        """Test list-steps with conflicts only."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        # Mock conflict definitions
        mock_def1 = Mock()
        mock_def1.workspace_id = "workspace1"
        mock_def1.registry_type = "local"
        
        mock_def2 = Mock()
        mock_def2.workspace_id = None
        mock_def2.registry_type = "core"
        
        mock_manager.get_step_conflicts.return_value = {
            "ConflictingStep": [mock_def1, mock_def2]
        }

        result = runner.invoke(registry_cli, ["list-steps", "--conflicts-only"])

        assert result.exit_code == 0
        assert "⚠️  Found 1 conflicting steps:" in result.output
        assert "📍 Step: ConflictingStep" in result.output

    @patch("cursus.cli.registry_cli.UnifiedRegistryManager")
    def test_list_steps_no_conflicts(self, mock_manager_class, runner):
        """Test list-steps with no conflicts."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.get_step_conflicts.return_value = {}

        result = runner.invoke(registry_cli, ["list-steps", "--conflicts-only"])

        assert result.exit_code == 0
        assert "✅ No step name conflicts detected" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestValidateRegistryCommand:
    """Test validate-registry command."""

    @patch("cursus.cli.registry_cli.get_all_step_names")
    @patch("cursus.cli.registry_cli.get_workspace_context")
    def test_validate_registry_basic(self, mock_get_context, mock_get_steps, runner):
        """Test basic validate-registry command."""
        mock_get_context.return_value = None
        mock_get_steps.return_value = ["StepA", "StepB", "StepC"]

        result = runner.invoke(registry_cli, ["validate-registry"])

        assert result.exit_code == 0
        assert "🔍 Validating registry..." in result.output
        assert "📁 Core registry" in result.output
        assert "✅ Found 3 steps" in result.output
        assert "✅ Registry validation completed" in result.output

    @patch("cursus.cli.registry_cli.get_all_step_names")
    @patch("cursus.cli.registry_cli.get_workspace_context")
    def test_validate_registry_with_workspace(self, mock_get_context, mock_get_steps, runner):
        """Test validate-registry with workspace."""
        mock_get_context.return_value = "test_workspace"
        mock_get_steps.return_value = ["StepA", "StepB"]

        result = runner.invoke(registry_cli, ["validate-registry", "--workspace", "test_workspace"])

        assert result.exit_code == 0
        assert "📁 Workspace: test_workspace" in result.output
        assert "✅ Found 2 steps" in result.output

    @patch("cursus.cli.registry_cli.UnifiedRegistryManager")
    @patch("cursus.cli.registry_cli.get_all_step_names")
    @patch("cursus.cli.registry_cli.get_workspace_context")
    def test_validate_registry_with_conflicts(self, mock_get_context, mock_get_steps, mock_manager_class, runner):
        """Test validate-registry with conflict checking."""
        mock_get_context.return_value = None
        mock_get_steps.return_value = ["StepA", "StepB"]
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        mock_manager.get_step_conflicts.return_value = {"ConflictStep": [Mock(), Mock()]}
        mock_manager.get_registry_status.return_value = {
            "core": {"step_count": 2}
        }

        result = runner.invoke(registry_cli, ["validate-registry", "--check-conflicts"])

        assert result.exit_code == 0
        assert "⚠️  Found 1 step name conflicts:" in result.output
        assert "ConflictStep: 2 definitions" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestResolveStepCommand:
    """Test resolve-step command."""

    @patch("cursus.cli.registry_cli.get_workspace_context")
    @patch("cursus.cli.registry_cli.UnifiedRegistryManager")
    def test_resolve_step_success(self, mock_manager_class, mock_get_context, runner):
        """Test successful step resolution."""
        mock_get_context.return_value = "test_workspace"
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        mock_result = Mock()
        mock_result.resolved = True
        mock_result.source_registry = "workspace"
        mock_result.resolution_strategy = "workspace_priority"
        mock_result.selected_definition = Mock()
        mock_result.selected_definition.config_class = "TestStepConfig"
        mock_result.selected_definition.builder_step_name = "TestStepBuilder"
        mock_result.selected_definition.framework = "xgboost"
        
        mock_manager.get_step.return_value = mock_result

        result = runner.invoke(registry_cli, ["resolve-step", "TestStep", "--workspace", "test_workspace"])

        assert result.exit_code == 0
        assert "🔍 Resolving step: TestStep" in result.output
        assert "✅ Step resolved successfully" in result.output
        assert "Source: workspace" in result.output
        assert "Strategy: workspace_priority" in result.output
        assert "Config: TestStepConfig" in result.output
        assert "Builder: TestStepBuilder" in result.output
        assert "Framework: xgboost" in result.output

    @patch("cursus.cli.registry_cli.get_workspace_context")
    @patch("cursus.cli.registry_cli.UnifiedRegistryManager")
    def test_resolve_step_failure(self, mock_manager_class, mock_get_context, runner):
        """Test failed step resolution."""
        mock_get_context.return_value = None
        
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        mock_result = Mock()
        mock_result.resolved = False
        mock_result.errors = ["Step not found in registry"]
        
        mock_manager.get_step.return_value = mock_result

        result = runner.invoke(registry_cli, ["resolve-step", "NonExistentStep"])

        assert result.exit_code == 0
        assert "❌ Step resolution failed" in result.output
        assert "❌ Step not found in registry" in result.output

    @patch("cursus.cli.registry_cli.get_config_class_name")
    @patch("cursus.cli.registry_cli.get_builder_step_name")
    @patch("cursus.cli.registry_cli.get_workspace_context")
    def test_resolve_step_fallback(self, mock_get_context, mock_get_builder, mock_get_config, runner):
        """Test step resolution fallback when hybrid registry unavailable."""
        mock_get_context.return_value = None
        mock_get_config.return_value = "TestStepConfig"
        mock_get_builder.return_value = "TestStepBuilder"

        with patch("cursus.cli.registry_cli.UnifiedRegistryManager", side_effect=ImportError):
            result = runner.invoke(registry_cli, ["resolve-step", "TestStep"])

        assert result.exit_code == 0
        assert "✅ Step found (basic resolution)" in result.output
        assert "Config: TestStepConfig" in result.output
        assert "Builder: TestStepBuilder" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestValidateStepDefinitionCommand:
    """Test validate-step-definition command."""

    def test_validate_step_definition_valid(self, runner):
        """Test validate-step-definition with valid step."""
        result = runner.invoke(
            registry_cli,
            [
                "validate-step-definition",
                "--name",
                "TestStep",
                "--config-class",
                "TestStepConfig",
                "--builder-name",
                "TestStepStepBuilder",
                "--sagemaker-type",
                "Processing",
            ],
        )

        assert result.exit_code == 0
        assert "🔍 Validating step definition: TestStep" in result.output
        assert "✅ Step definition is valid" in result.output

    def test_validate_step_definition_invalid(self, runner):
        """Test validate-step-definition with invalid step."""
        result = runner.invoke(
            registry_cli,
            [
                "validate-step-definition",
                "--name",
                "test_step",  # Invalid PascalCase
                "--config-class",
                "TestConfiguration",  # Invalid suffix
                "--builder-name",
                "TestBuilder",  # Invalid suffix
            ],
        )

        assert result.exit_code == 0
        assert "❌ Step definition has validation errors:" in result.output
        assert "must be PascalCase" in result.output

    def test_validate_step_definition_auto_correct(self, runner):
        """Test validate-step-definition with auto-correction."""
        result = runner.invoke(
            registry_cli,
            [
                "validate-step-definition",
                "--name",
                "test_step",
                "--config-class",
                "TestConfiguration",
                "--auto-correct",
            ],
        )

        assert result.exit_code == 0
        assert "🔧 Auto-correction applied:" in result.output
        assert "name: test_step → TestStep" in result.output

    def test_validate_step_definition_performance(self, runner):
        """Test validate-step-definition with performance metrics."""
        result = runner.invoke(
            registry_cli,
            ["validate-step-definition", "--name", "TestStep", "--performance"],
        )

        assert result.exit_code == 0
        assert "📊 Performance Metrics:" in result.output
        assert "Validation time:" in result.output
        assert "Cache hit rate:" in result.output

    def test_validate_step_definition_missing_name(self, runner):
        """Test validate-step-definition without required name."""
        result = runner.invoke(registry_cli, ["validate-step-definition"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "Error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestValidationStatusCommand:
    """Test validation-status command."""

    def test_validation_status_basic(self, runner):
        """Test basic validation-status command."""
        result = runner.invoke(registry_cli, ["validation-status"])

        assert result.exit_code == 0
        assert "📊 Validation System Status" in result.output
        assert "System Status:" in result.output
        assert "Implementation:" in result.output
        assert "Supported Modes:" in result.output
        assert "📈 Performance Metrics:" in result.output

    @patch("cursus.cli.registry_cli.get_validation_status")
    def test_validation_status_with_exception(self, mock_get_status, runner):
        """Test validation-status with exception."""
        mock_get_status.side_effect = Exception("Test error")

        result = runner.invoke(registry_cli, ["validation-status"])

        assert result.exit_code == 0
        assert "❌ Failed to get validation status:" in result.output
        assert "Test error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestResetValidationMetricsCommand:
    """Test reset-validation-metrics command."""

    def test_reset_validation_metrics_confirmed(self, runner):
        """Test reset-validation-metrics with confirmation."""
        result = runner.invoke(
            registry_cli, ["reset-validation-metrics"], input="y\n"
        )

        assert result.exit_code == 0
        assert "✅ Validation metrics and cache have been reset" in result.output
        assert "📊 Performance tracking restarted from zero" in result.output

    def test_reset_validation_metrics_cancelled(self, runner):
        """Test reset-validation-metrics when cancelled."""
        result = runner.invoke(
            registry_cli, ["reset-validation-metrics"], input="n\n"
        )

        assert result.exit_code == 1
        assert "Aborted" in result.output

    @patch("cursus.cli.registry_cli.reset_performance_metrics")
    def test_reset_validation_metrics_with_exception(self, mock_reset, runner):
        """Test reset-validation-metrics with exception."""
        mock_reset.side_effect = Exception("Test reset error")

        result = runner.invoke(
            registry_cli, ["reset-validation-metrics"], input="y\n"
        )

        assert result.exit_code == 0
        assert "❌ Failed to reset validation metrics:" in result.output
        assert "Test reset error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""

    @patch("cursus.cli.registry_cli.get_all_step_names")
    def test_list_steps_with_exception(self, mock_get_steps, runner):
        """Test list-steps with exception."""
        mock_get_steps.side_effect = Exception("Registry error")

        result = runner.invoke(registry_cli, ["list-steps"])

        assert result.exit_code == 0
        assert "❌ Failed to list steps:" in result.output
        assert "Registry error" in result.output

    @patch("cursus.cli.registry_cli.get_all_step_names")
    def test_validate_registry_with_exception(self, mock_get_steps, runner):
        """Test validate-registry with exception."""
        mock_get_steps.side_effect = Exception("Validation error")

        result = runner.invoke(registry_cli, ["validate-registry"])

        assert result.exit_code == 0
        assert "❌ Registry validation failed:" in result.output
        assert "Validation error" in result.output

    @patch("cursus.cli.registry_cli.get_workspace_context")
    def test_resolve_step_with_exception(self, mock_get_context, runner):
        """Test resolve-step with exception."""
        mock_get_context.side_effect = Exception("Resolution error")

        result = runner.invoke(registry_cli, ["resolve-step", "TestStep"])

        assert result.exit_code == 0
        assert "❌ Step resolution failed:" in result.output
        assert "Resolution error" in result.output


@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIIntegrationScenarios:
    """Test realistic CLI usage scenarios."""

    def test_workspace_creation_workflow(self, runner, temp_workspace):
        """Test complete workspace creation workflow."""
        workspace_id = "developer_test"
        workspace_path = str(Path(temp_workspace) / workspace_id)

        # Step 1: Create workspace
        result1 = runner.invoke(
            registry_cli,
            [
                "init-workspace",
                workspace_id,
                "--workspace-path",
                workspace_path,
                "--template",
                "standard",
            ],
        )

        assert result1.exit_code == 0
        assert "🎉 Developer workspace successfully created!" in result1.output

        # Step 2: Validate the created workspace structure
        workspace_dir = Path(workspace_path)
        assert workspace_dir.exists()
        assert (workspace_dir / "src" / "cursus_dev" / "registry" / "workspace_registry.py").exists()
        assert (workspace_dir / "README.md").exists()

        # Step 3: Check registry file content
        registry_file = workspace_dir / "src" / "cursus_dev" / "registry" / "workspace_registry.py"
        content = registry_file.read_text()
        assert f'developer_id": "{workspace_id}' in content
        assert "LOCAL_STEPS" in content
        assert "STEP_OVERRIDES" in content

    @patch("cursus.cli.registry_cli.get_all_step_names")
    @patch("cursus.cli.registry_cli.get_workspace_context")
    def test_registry_validation_workflow(self, mock_get_context, mock_get_steps, runner):
        """Test registry validation workflow."""
        mock_get_context.return_value = "test_workspace"
        mock_get_steps.return_value = ["StepA", "StepB", "CustomStep"]

        # Step 1: List steps
        result1 = runner.invoke(registry_cli, ["list-steps", "--workspace", "test_workspace"])
        assert result1.exit_code == 0
        assert "CustomStep" in result1.output

        # Step 2: Validate registry
        result2 = runner.invoke(registry_cli, ["validate-registry", "--workspace", "test_workspace"])
        assert result2.exit_code == 0
        assert "✅ Found 3 steps" in result2.output

        # Step 3: Validate a step definition
        result3 = runner.invoke(
            registry_cli,
            ["validate-step-definition", "--name", "CustomStep", "--config-class", "CustomStepConfig"],
        )
        assert result3.exit_code == 0

    def test_step_validation_workflow(self, runner):
        """Test step validation workflow."""
        # Step 1: Validate invalid step
        result1 = runner.invoke(
            registry_cli,
            [
                "validate-step-definition",
                "--name",
                "invalid_step",
                "--config-class",
                "InvalidConfiguration",
            ],
        )
        assert result1.exit_code == 0
        assert "❌ Step definition has validation errors:" in result1.output

        # Step 2: Apply auto-correction
        result2 = runner.invoke(
            registry_cli,
            [
                "validate-step-definition",
                "--name",
                "invalid_step",
                "--config-class",
                "InvalidConfiguration",
                "--auto-correct",
            ],
        )
        assert result2.exit_code == 0
        assert "🔧 Auto-correction applied:" in result2.output

        # Step 3: Check validation status
        result3 = runner.invoke(registry_cli, ["validation-status"])
        assert result3.exit_code == 0
        assert "📊 Validation System Status" in result3.output


if __name__ == "__main__":
    pytest.main([__file__])
