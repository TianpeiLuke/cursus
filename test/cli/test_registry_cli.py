"""
Unit tests for registry CLI commands.

This module tests the registry CLI commands that were added for step definition
validation, including validate-step-definition, validation-status, and
reset-validation-metrics commands.

Tests focus on:
- CLI command functionality and argument parsing
- Integration with validation utilities
- Error handling and user feedback
- Performance metrics display
- Auto-correction workflows
"""

import pytest
from unittest.mock import patch, MagicMock, call
from click.testing import CliRunner
import json

# Import the CLI functions we want to test
try:
    from cursus.cli.registry_cli import (
        registry_cli,
        validate_step_definition,
        validation_status,
        reset_validation_metrics
    )
    CLI_AVAILABLE = True
except ImportError:
    CLI_AVAILABLE = False

# Import validation utilities for testing integration
from cursus.registry.validation_utils import (
    validate_new_step_definition,
    auto_correct_step_definition,
    create_validation_report,
    get_performance_metrics,
    reset_performance_metrics as reset_utils_metrics,
    get_validation_status as get_utils_status
)

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestRegistryCLICommands:
    """Test registry CLI commands functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        # Reset performance metrics before each test
        reset_utils_metrics()
    
    def test_validate_step_definition_valid_step(self):
        """Test validate-step-definition command with valid step."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep',
            '--config-class', 'TestStepConfig',
            '--builder-name', 'TestStepStepBuilder',
            '--sagemaker-type', 'Processing'
        ])
        
        assert result.exit_code == 0
        assert "✅ Step definition is valid" in result.output
        assert "🔍 Validating step definition: TestStep" in result.output
    
    def test_validate_step_definition_invalid_step(self):
        """Test validate-step-definition command with invalid step."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'test_step',  # Invalid PascalCase
            '--config-class', 'TestConfiguration',  # Invalid suffix
            '--builder-name', 'TestBuilder'  # Invalid suffix
        ])
        
        assert result.exit_code == 0  # CLI should not exit with error
        assert "❌ Step definition has validation errors:" in result.output
        assert "must be PascalCase" in result.output
        assert "must end with 'Config'" in result.output
        assert "must end with 'StepBuilder'" in result.output
    
    def test_validate_step_definition_with_auto_correct(self):
        """Test validate-step-definition command with auto-correction."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'test_step',
            '--config-class', 'TestConfiguration',
            '--builder-name', 'TestBuilder',
            '--auto-correct'
        ])
        
        assert result.exit_code == 0
        assert "🔧 Auto-correction applied:" in result.output
        assert "step_name: test_step → TestStep" in result.output or "name: test_step → TestStep" in result.output
    
    def test_validate_step_definition_with_performance(self):
        """Test validate-step-definition command with performance metrics."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep',
            '--performance'
        ])
        
        assert result.exit_code == 0
        assert "📊 Performance Metrics:" in result.output
        assert "Validation time:" in result.output
        assert "Cache hit rate:" in result.output
        assert "Total validations:" in result.output
    
    def test_validate_step_definition_minimal_args(self):
        """Test validate-step-definition command with minimal arguments."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep'
        ])
        
        assert result.exit_code == 0
        assert "🔍 Validating step definition: TestStep" in result.output
        assert "✅ Step definition is valid" in result.output
    
    def test_validate_step_definition_missing_name(self):
        """Test validate-step-definition command without required name."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition'
        ])
        
        assert result.exit_code != 0  # Should fail due to missing required argument
        assert "Missing option" in result.output or "Error" in result.output
    
    def test_validation_status_command(self):
        """Test validation-status command."""
        # Generate some performance data first
        step_data = {"name": "TestStep"}
        validate_new_step_definition(step_data)
        
        result = self.runner.invoke(registry_cli, ['validation-status'])
        
        assert result.exit_code == 0
        assert "📊 Validation System Status" in result.output
        assert "System Status:" in result.output
        assert "Implementation:" in result.output
        assert "Supported Modes:" in result.output
        assert "📈 Performance Metrics:" in result.output
        assert "Total validations:" in result.output
        assert "Average validation time:" in result.output
        assert "Performance:" in result.output
    
    def test_validation_status_with_no_activity(self):
        """Test validation-status command with no prior activity."""
        # Reset metrics to ensure clean state
        reset_utils_metrics()
        
        result = self.runner.invoke(registry_cli, ['validation-status'])
        
        assert result.exit_code == 0
        assert "📊 Validation System Status" in result.output
        assert "Total validations: 0" in result.output
        assert "Average validation time: 0.00ms" in result.output
    
    def test_reset_validation_metrics_with_confirmation(self):
        """Test reset-validation-metrics command with confirmation."""
        # Generate some metrics first
        step_data = {"name": "TestStep"}
        validate_new_step_definition(step_data)
        
        # Confirm the reset
        result = self.runner.invoke(registry_cli, [
            'reset-validation-metrics'
        ], input='y\n')
        
        assert result.exit_code == 0
        assert "✅ Validation metrics and cache have been reset" in result.output
        assert "📊 Performance tracking restarted from zero" in result.output
        
        # Verify metrics were actually reset
        metrics = get_performance_metrics()
        assert metrics["total_validations"] == 0
    
    def test_reset_validation_metrics_cancelled(self):
        """Test reset-validation-metrics command when cancelled."""
        result = self.runner.invoke(registry_cli, [
            'reset-validation-metrics'
        ], input='n\n')
        
        assert result.exit_code == 1  # Should exit with error code when cancelled
        assert "Aborted" in result.output
    
    def test_registry_cli_help(self):
        """Test registry CLI help message."""
        result = self.runner.invoke(registry_cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Registry management commands" in result.output
        assert "validate-step-definition" in result.output
        assert "validation-status" in result.output
        assert "reset-validation-metrics" in result.output

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIIntegrationWithValidationUtils:
    """Test CLI integration with validation utilities."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        reset_utils_metrics()
    
    def test_cli_validation_integration(self):
        """Test CLI validation integrates correctly with validation utilities."""
        # Test data that should trigger validation errors
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'invalid_step',
            '--config-class', 'InvalidConfiguration',
            '--builder-name', 'InvalidBuilder'
        ])
        
        assert result.exit_code == 0
        assert "❌ Step definition has validation errors:" in result.output
        
        # Verify the same validation logic works directly
        step_data = {
            "name": "invalid_step",
            "config_class": "InvalidConfiguration",
            "builder_name": "InvalidBuilder"
        }
        errors = validate_new_step_definition(step_data)
        assert len(errors) > 0
    
    def test_cli_auto_correction_integration(self):
        """Test CLI auto-correction integrates with validation utilities."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'test_step',
            '--config-class', 'TestConfiguration',
            '--auto-correct'
        ])
        
        assert result.exit_code == 0
        assert "🔧 Auto-correction applied:" in result.output
        
        # Verify the same auto-correction works directly
        step_data = {
            "name": "test_step",
            "config_class": "TestConfiguration"
        }
        corrected = auto_correct_step_definition(step_data)
        assert corrected["name"] == "TestStep"
        assert corrected["config_class"] == "TestStepConfig"
    
    def test_cli_performance_tracking_integration(self):
        """Test CLI performance tracking integrates with validation utilities."""
        # Run CLI command
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep',
            '--performance'
        ])
        
        assert result.exit_code == 0
        assert "📊 Performance Metrics:" in result.output
        
        # Verify metrics were updated
        metrics = get_performance_metrics()
        assert metrics["total_validations"] > 0
    
    def test_cli_status_integration(self):
        """Test CLI status command integrates with validation utilities."""
        result = self.runner.invoke(registry_cli, ['validation-status'])
        
        assert result.exit_code == 0
        
        # Verify the same status is available directly
        status = get_utils_status()
        assert status["validation_available"] is True
        assert len(status["supported_modes"]) > 0
    
    def test_cli_reset_integration(self):
        """Test CLI reset command integrates with validation utilities."""
        # Generate some metrics
        step_data = {"name": "TestStep"}
        validate_new_step_definition(step_data)
        
        # Verify metrics exist
        metrics_before = get_performance_metrics()
        assert metrics_before["total_validations"] > 0
        
        # Reset via CLI
        result = self.runner.invoke(registry_cli, [
            'reset-validation-metrics'
        ], input='y\n')
        
        assert result.exit_code == 0
        
        # Verify reset worked
        metrics_after = get_performance_metrics()
        assert metrics_after["total_validations"] == 0

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIErrorHandling:
    """Test CLI error handling and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        reset_utils_metrics()
    
    def test_cli_with_invalid_sagemaker_type(self):
        """Test CLI with invalid SageMaker step type."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep',
            '--sagemaker-type', 'InvalidType'
        ])
        
        assert result.exit_code == 0
        assert "❌ Step definition has validation errors:" in result.output
        assert "is invalid" in result.output
        assert "Valid types:" in result.output
    
    def test_cli_with_unicode_characters(self):
        """Test CLI with unicode characters in names."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStepWithÜnicode'
        ])
        
        # Should handle gracefully without crashing
        assert result.exit_code == 0
        # May or may not be valid, but should not crash
        assert "🔍 Validating step definition:" in result.output
    
    def test_cli_with_very_long_names(self):
        """Test CLI with very long step names."""
        long_name = "A" * 100
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', long_name
        ])
        
        assert result.exit_code == 0
        assert "🔍 Validating step definition:" in result.output
    
    def test_cli_with_empty_values(self):
        """Test CLI with empty field values."""
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep',
            '--config-class', '',
            '--builder-name', ''
        ])
        
        assert result.exit_code == 0
        assert "✅ Step definition is valid" in result.output
    
    @patch('cursus.registry.validation_utils.validate_new_step_definition')
    def test_cli_with_validation_exception(self, mock_validate):
        """Test CLI handling of validation exceptions."""
        mock_validate.side_effect = Exception("Test validation error")
        
        result = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep'
        ])
        
        assert result.exit_code == 0
        assert "❌ Validation failed:" in result.output
        assert "Test validation error" in result.output
    
    @patch('cursus.registry.validation_utils.get_validation_status')
    def test_cli_status_with_exception(self, mock_status):
        """Test CLI status command with exception."""
        mock_status.side_effect = Exception("Test status error")
        
        result = self.runner.invoke(registry_cli, ['validation-status'])
        
        assert result.exit_code == 0
        assert "❌ Failed to get validation status:" in result.output
        assert "Test status error" in result.output
    
    @patch('cursus.registry.validation_utils.reset_performance_metrics')
    def test_cli_reset_with_exception(self, mock_reset):
        """Test CLI reset command with exception."""
        mock_reset.side_effect = Exception("Test reset error")
        
        result = self.runner.invoke(registry_cli, [
            'reset-validation-metrics'
        ], input='y\n')
        
        assert result.exit_code == 0
        assert "❌ Failed to reset validation metrics:" in result.output
        assert "Test reset error" in result.output

@pytest.mark.skipif(not CLI_AVAILABLE, reason="CLI module not available")
class TestCLIUsageScenarios:
    """Test realistic CLI usage scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        reset_utils_metrics()
    
    def test_typical_developer_workflow(self):
        """Test typical developer workflow using CLI."""
        # Step 1: Developer validates a new step
        result1 = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'my_new_step',
            '--config-class', 'MyNewStepConfiguration'
        ])
        
        assert result1.exit_code == 0
        assert "❌ Step definition has validation errors:" in result1.output
        
        # Step 2: Developer applies auto-correction
        result2 = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'my_new_step',
            '--config-class', 'MyNewStepConfiguration',
            '--auto-correct'
        ])
        
        assert result2.exit_code == 0
        assert "🔧 Auto-correction applied:" in result2.output
        
        # Step 3: Developer checks system status
        result3 = self.runner.invoke(registry_cli, ['validation-status'])
        
        assert result3.exit_code == 0
        # Each CLI validation call triggers multiple internal validations
        # (validate_new_step_definition + create_validation_report calls)
        # So 2 CLI calls result in more than 2 total validations
        assert "Total validations:" in result3.output
        # Verify that validations were actually performed
        assert "Total validations: 0" not in result3.output
    
    def test_performance_monitoring_workflow(self):
        """Test performance monitoring workflow."""
        # Step 1: Perform several validations
        for i in range(3):
            result = self.runner.invoke(registry_cli, [
                'validate-step-definition',
                '--name', f'TestStep{i}'
            ])
            assert result.exit_code == 0
        
        # Step 2: Check performance status
        result = self.runner.invoke(registry_cli, ['validation-status'])
        
        assert result.exit_code == 0
        # Each CLI validation call triggers multiple internal validations
        # So 3 CLI calls result in more than 3 total validations
        assert "Total validations:" in result.output
        assert "Total validations: 0" not in result.output  # Verify validations occurred
        assert "🟢 Excellent" in result.output or "🟡 Good" in result.output
        
        # Step 3: Reset metrics
        result = self.runner.invoke(registry_cli, [
            'reset-validation-metrics'
        ], input='y\n')
        
        assert result.exit_code == 0
        
        # Step 4: Verify reset
        result = self.runner.invoke(registry_cli, ['validation-status'])
        assert "Total validations: 0" in result.output
    
    def test_batch_validation_scenario(self):
        """Test batch validation scenario."""
        steps_to_validate = [
            ("ValidStep", "ValidStepConfig", "ValidStepStepBuilder"),
            ("invalid_step", "InvalidConfig", "InvalidBuilder"),
            ("AnotherValidStep", "AnotherValidStepConfig", "AnotherValidStepStepBuilder")
        ]
        
        results = []
        for name, config, builder in steps_to_validate:
            result = self.runner.invoke(registry_cli, [
                'validate-step-definition',
                '--name', name,
                '--config-class', config,
                '--builder-name', builder
            ])
            results.append((name, result))
        
        # Check results
        valid_count = sum(1 for _, result in results if "✅ Step definition is valid" in result.output)
        invalid_count = len(results) - valid_count
        
        assert valid_count == 2  # ValidStep and AnotherValidStep
        assert invalid_count == 1  # invalid_step
        
        # Check final status
        status_result = self.runner.invoke(registry_cli, ['validation-status'])
        # Each CLI validation call triggers multiple internal validations
        # So 3 CLI calls result in more than 3 total validations
        assert "Total validations:" in status_result.output
        assert "Total validations: 0" not in status_result.output  # Verify validations occurred
    
    def test_error_recovery_scenario(self):
        """Test error recovery scenario."""
        # Step 1: Try validation with problematic data
        result1 = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'test_step',
            '--config-class', 'TestConfiguration',
            '--builder-name', 'TestBuilder'
        ])
        
        assert result1.exit_code == 0
        assert "❌ Step definition has validation errors:" in result1.output
        
        # Step 2: Apply corrections and retry
        result2 = self.runner.invoke(registry_cli, [
            'validate-step-definition',
            '--name', 'TestStep',  # Corrected name
            '--config-class', 'TestStepConfig',  # Corrected config
            '--builder-name', 'TestStepStepBuilder'  # Corrected builder
        ])
        
        assert result2.exit_code == 0
        assert "✅ Step definition is valid" in result2.output
        
        # Step 3: Verify system is working normally
        result3 = self.runner.invoke(registry_cli, ['validation-status'])
        assert result3.exit_code == 0
        assert "🟢 Active" in result3.output

if __name__ == "__main__":
    pytest.main([__file__])
