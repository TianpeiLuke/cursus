#!/usr/bin/env python3
"""
Tests for the refactored UnifiedAlignmentTester with configuration-driven validation.

Tests the enhanced unified alignment tester including:
- Configuration-driven validation
- Step-type-aware validation
- Performance improvements through level skipping
- Integration with validation rulesets
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path

from cursus.validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
from cursus.validation.alignment.config import ValidationLevel, StepTypeCategory


class TestUnifiedAlignmentTesterRefactored:
    """Test refactored UnifiedAlignmentTester functionality."""

    @pytest.fixture
    def mock_workspace_dirs(self):
        """Mock workspace directories."""
        return ["/mock/workspace1", "/mock/workspace2"]

    @pytest.fixture
    def tester(self, mock_workspace_dirs):
        """Create UnifiedAlignmentTester instance for testing."""
        with patch('cursus.validation.alignment.unified_alignment_tester.StepCatalog'), \
             patch('cursus.validation.alignment.unified_alignment_tester.LevelValidators'):
            return UnifiedAlignmentTester(workspace_dirs=mock_workspace_dirs)

    def test_initialization_with_configuration_validation(self, mock_workspace_dirs):
        """Test that initialization validates configuration."""
        with patch('cursus.validation.alignment.unified_alignment_tester.validate_step_type_configuration') as mock_validate, \
             patch('cursus.validation.alignment.unified_alignment_tester.StepCatalog'), \
             patch('cursus.validation.alignment.unified_alignment_tester.LevelValidators'):
            
            mock_validate.return_value = []  # No configuration issues
            
            tester = UnifiedAlignmentTester(workspace_dirs=mock_workspace_dirs)
            
            # Should validate configuration on initialization
            mock_validate.assert_called_once()
            assert tester.workspace_dirs == mock_workspace_dirs

    def test_initialization_with_configuration_issues(self, mock_workspace_dirs):
        """Test initialization with configuration issues logs warnings."""
        with patch('cursus.validation.alignment.unified_alignment_tester.validate_step_type_configuration') as mock_validate, \
             patch('cursus.validation.alignment.unified_alignment_tester.StepCatalog'), \
             patch('cursus.validation.alignment.unified_alignment_tester.LevelValidators'), \
             patch('cursus.validation.alignment.unified_alignment_tester.logger') as mock_logger:
            
            mock_validate.return_value = ["Configuration issue 1", "Configuration issue 2"]
            
            tester = UnifiedAlignmentTester(workspace_dirs=mock_workspace_dirs)
            
            # Should log warning about configuration issues
            mock_logger.warning.assert_called_once()
            assert "Configuration issues found" in str(mock_logger.warning.call_args)

    @patch('cursus.validation.alignment.unified_alignment_tester.get_sagemaker_step_type')
    @patch('cursus.validation.alignment.unified_alignment_tester.get_validation_ruleset')
    @patch('cursus.validation.alignment.unified_alignment_tester.is_step_type_excluded')
    def test_run_validation_for_step_excluded_step(self, mock_is_excluded, mock_get_ruleset, mock_get_step_type, tester):
        """Test validation for excluded step type."""
        # Setup mocks
        mock_get_step_type.return_value = "Base"
        mock_is_excluded.return_value = True
        mock_ruleset = Mock()
        mock_ruleset.skip_reason = "Base configurations - no builder to validate"
        mock_get_ruleset.return_value = mock_ruleset
        
        # Run validation
        result = tester.run_validation_for_step("base_config")
        
        # Should return excluded status
        assert result["step_name"] == "base_config"
        assert result["sagemaker_step_type"] == "Base"
        assert result["status"] == "EXCLUDED"
        assert result["reason"] == "Base configurations - no builder to validate"
        
        # Should not run any validation levels
        assert "level1" not in result
        assert "level2" not in result
        assert "level3" not in result
        assert "level4" not in result

    @patch('cursus.validation.alignment.unified_alignment_tester.get_sagemaker_step_type')
    @patch('cursus.validation.alignment.unified_alignment_tester.get_validation_ruleset')
    @patch('cursus.validation.alignment.unified_alignment_tester.is_step_type_excluded')
    def test_run_validation_for_step_script_based(self, mock_is_excluded, mock_get_ruleset, mock_get_step_type, tester):
        """Test validation for script-based step type (full validation)."""
        # Setup mocks
        mock_get_step_type.return_value = "Processing"
        mock_is_excluded.return_value = False
        mock_ruleset = Mock()
        mock_ruleset.enabled_levels = {
            ValidationLevel.SCRIPT_CONTRACT,
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        }
        mock_ruleset.level_4_validator_class = "ProcessingStepBuilderValidator"
        mock_get_ruleset.return_value = mock_ruleset
        
        # Mock level validators
        tester.level_validators.run_level_1_validation = Mock(return_value={"passed": True, "issues": []})
        tester.level_validators.run_level_2_validation = Mock(return_value={"passed": True, "issues": []})
        tester.level_validators.run_level_3_validation = Mock(return_value={"passed": True, "issues": []})
        tester.level_validators.run_level_4_validation = Mock(return_value={"passed": True, "issues": []})
        
        # Run validation
        result = tester.run_validation_for_step("processing_script")
        
        # Should run all validation levels
        tester.level_validators.run_level_1_validation.assert_called_once_with("processing_script")
        tester.level_validators.run_level_2_validation.assert_called_once_with("processing_script")
        tester.level_validators.run_level_3_validation.assert_called_once_with("processing_script")
        tester.level_validators.run_level_4_validation.assert_called_once_with("processing_script", "ProcessingStepBuilderValidator")
        
        # Should return passing status
        assert result["step_name"] == "processing_script"
        assert result["sagemaker_step_type"] == "Processing"
        assert result["overall_status"] == "PASSED"
        assert "level_1" in result["validation_results"]
        assert "level_2" in result["validation_results"]
        assert "level_3" in result["validation_results"]
        assert "level_4" in result["validation_results"]

    @patch('cursus.validation.alignment.unified_alignment_tester.get_sagemaker_step_type')
    @patch('cursus.validation.alignment.unified_alignment_tester.get_validation_ruleset')
    @patch('cursus.validation.alignment.unified_alignment_tester.is_step_type_excluded')
    def test_run_validation_for_step_non_script(self, mock_is_excluded, mock_get_ruleset, mock_get_step_type, tester):
        """Test validation for non-script step type (skip levels 1-2)."""
        # Setup mocks
        mock_get_step_type.return_value = "CreateModel"
        mock_is_excluded.return_value = False
        mock_ruleset = Mock()
        mock_ruleset.enabled_levels = {
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        }
        mock_ruleset.level_4_validator_class = "CreateModelStepBuilderValidator"
        mock_get_ruleset.return_value = mock_ruleset
        
        # Mock level validators
        tester.level_validators.run_level_1_validation = Mock()
        tester.level_validators.run_level_2_validation = Mock()
        tester.level_validators.run_level_3_validation = Mock(return_value={"passed": True, "issues": []})
        tester.level_validators.run_level_4_validation = Mock(return_value={"passed": True, "issues": []})
        
        # Run validation
        result = tester.run_validation_for_step("create_model_step")
        
        # Should skip levels 1 and 2
        tester.level_validators.run_level_1_validation.assert_not_called()
        tester.level_validators.run_level_2_validation.assert_not_called()
        
        # Should run levels 3 and 4
        tester.level_validators.run_level_3_validation.assert_called_once_with("create_model_step")
        tester.level_validators.run_level_4_validation.assert_called_once_with("create_model_step", "CreateModelStepBuilderValidator")
        
        # Should return passing status with only levels 3 and 4
        assert result["step_name"] == "create_model_step"
        assert result["sagemaker_step_type"] == "CreateModel"
        assert result["overall_status"] == "PASSED"
        assert "level_1" not in result["validation_results"]
        assert "level_2" not in result["validation_results"]
        assert "level_3" in result["validation_results"]
        assert "level_4" in result["validation_results"]

    @patch('cursus.validation.alignment.unified_alignment_tester.get_sagemaker_step_type')
    @patch('cursus.validation.alignment.unified_alignment_tester.get_validation_ruleset')
    @patch('cursus.validation.alignment.unified_alignment_tester.is_step_type_excluded')
    def test_run_validation_for_step_with_failures(self, mock_is_excluded, mock_get_ruleset, mock_get_step_type, tester):
        """Test validation with failures."""
        # Setup mocks
        mock_get_step_type.return_value = "Processing"
        mock_is_excluded.return_value = False
        mock_ruleset = Mock()
        mock_ruleset.enabled_levels = {
            ValidationLevel.SCRIPT_CONTRACT,
            ValidationLevel.CONTRACT_SPEC,
            ValidationLevel.SPEC_DEPENDENCY,
            ValidationLevel.BUILDER_CONFIG
        }
        mock_ruleset.level_4_validator_class = "ProcessingStepBuilderValidator"
        mock_get_ruleset.return_value = mock_ruleset
        
        # Mock level validators with failures
        tester.level_validators.run_level_1_validation = Mock(return_value={
            "status": "ERROR", 
            "issues": [{"severity": "ERROR", "message": "Script validation failed"}]
        })
        tester.level_validators.run_level_2_validation = Mock(return_value={"status": "PASSED", "issues": []})
        tester.level_validators.run_level_3_validation = Mock(return_value={
            "status": "ERROR",
            "issues": [{"severity": "ERROR", "message": "Dependency validation failed"}]
        })
        tester.level_validators.run_level_4_validation = Mock(return_value={"status": "PASSED", "issues": []})
        
        # Run validation
        result = tester.run_validation_for_step("failing_script")
        
        # Should return failing status
        assert result["step_name"] == "failing_script"
        assert result["sagemaker_step_type"] == "Processing"
        assert result["overall_status"] == "FAILED"
        
        # Should have failure details
        assert result["validation_results"]["level_1"]["status"] == "ERROR"
        assert result["validation_results"]["level_3"]["status"] == "ERROR"
        assert len(result["validation_results"]["level_1"]["issues"]) == 1
        assert len(result["validation_results"]["level_3"]["issues"]) == 1

    def test_discover_scripts_integration(self, tester):
        """Test script discovery integration with step catalog."""
        # Mock step catalog methods that actual implementation calls
        mock_available_steps = ["script1", "script2", "builder1", "spec1"]
        
        tester.step_catalog.list_available_steps = Mock(return_value=mock_available_steps)
        
        # Mock _has_script_file to return True for all steps (actual implementation filters by script files)
        tester._has_script_file = Mock(return_value=True)
        
        # Test discovery
        result = tester.discover_scripts()
        
        # Should return discovered scripts (filtered by script files)
        expected = mock_available_steps  # All have script files in this test
        assert result == expected
        
        # Should call actual catalog method used by implementation
        tester.step_catalog.list_available_steps.assert_called_once()

    @patch('cursus.validation.alignment.unified_alignment_tester.get_sagemaker_step_type')
    @patch('cursus.validation.alignment.unified_alignment_tester.get_validation_ruleset')
    @patch('cursus.validation.alignment.unified_alignment_tester.is_step_type_excluded')
    def test_run_full_validation_with_mixed_step_types(self, mock_is_excluded, mock_get_ruleset, mock_get_step_type, tester):
        """Test full validation with mixed step types."""
        # Mock script discovery method that _discover_all_steps actually calls
        tester.step_catalog.list_available_steps = Mock(return_value=["processing_script", "create_model_step", "base_config"])
        
        # Mock step type detection
        def mock_step_type_side_effect(step_name):
            if step_name == "processing_script":
                return "Processing"
            elif step_name == "create_model_step":
                return "CreateModel"
            elif step_name == "base_config":
                return "Base"
            return "Unknown"
        
        mock_get_step_type.side_effect = mock_step_type_side_effect
        
        # Mock exclusion check
        def mock_exclusion_side_effect(step_type):
            return step_type == "Base"
        
        mock_is_excluded.side_effect = mock_exclusion_side_effect
        
        # Mock rulesets
        def mock_ruleset_side_effect(step_type):
            if step_type == "Processing":
                ruleset = Mock()
                ruleset.enabled_levels = {ValidationLevel.SCRIPT_CONTRACT, ValidationLevel.CONTRACT_SPEC, 
                                        ValidationLevel.SPEC_DEPENDENCY, ValidationLevel.BUILDER_CONFIG}
                ruleset.level_4_validator_class = "ProcessingStepBuilderValidator"
                return ruleset
            elif step_type == "CreateModel":
                ruleset = Mock()
                ruleset.enabled_levels = {ValidationLevel.SPEC_DEPENDENCY, ValidationLevel.BUILDER_CONFIG}
                ruleset.level_4_validator_class = "CreateModelStepBuilderValidator"
                return ruleset
            elif step_type == "Base":
                ruleset = Mock()
                ruleset.skip_reason = "Base configurations - no builder to validate"
                return ruleset
            return None
        
        mock_get_ruleset.side_effect = mock_ruleset_side_effect
        
        # Mock level validators
        tester.level_validators.run_level_1_validation = Mock(return_value={"passed": True, "issues": []})
        tester.level_validators.run_level_2_validation = Mock(return_value={"passed": True, "issues": []})
        tester.level_validators.run_level_3_validation = Mock(return_value={"passed": True, "issues": []})
        tester.level_validators.run_level_4_validation = Mock(return_value={"passed": True, "issues": []})
        
        # Run full validation
        results = tester.run_full_validation()
        
        # Should validate all discovered scripts
        assert len(results) == 3
        
        # Processing script should have all levels
        processing_result = next(r for r in results.values() if r["step_name"] == "processing_script")
        assert processing_result["overall_status"] == "PASSED"
        assert "level_1" in processing_result["validation_results"]
        assert "level_2" in processing_result["validation_results"]
        assert "level_3" in processing_result["validation_results"]
        assert "level_4" in processing_result["validation_results"]
        
        # CreateModel step should skip levels 1-2
        createmodel_result = next(r for r in results.values() if r["step_name"] == "create_model_step")
        assert createmodel_result["overall_status"] == "PASSED"
        assert "level_1" not in createmodel_result["validation_results"]
        assert "level_2" not in createmodel_result["validation_results"]
        assert "level_3" in createmodel_result["validation_results"]
        assert "level_4" in createmodel_result["validation_results"]
        
        # Base config should be excluded
        base_result = next(r for r in results.values() if r["step_name"] == "base_config")
        assert base_result["status"] == "EXCLUDED"
        assert "validation_results" not in base_result or len(base_result.get("validation_results", {})) == 0

    def test_get_validation_summary_with_step_types(self, tester):
        """Test validation summary includes step type breakdown."""
        # Mock the run_validation_for_all_steps method to return mock results
        mock_results = {
            "script1": {"step_name": "script1", "sagemaker_step_type": "Processing", "overall_status": "PASSED"},
            "script2": {"step_name": "script2", "sagemaker_step_type": "Processing", "overall_status": "FAILED"},
            "script3": {"step_name": "script3", "sagemaker_step_type": "CreateModel", "overall_status": "PASSED"},
            "script4": {"step_name": "script4", "sagemaker_step_type": "Base", "status": "EXCLUDED"},
        }
        
        # Mock the method that get_validation_summary calls internally
        tester.run_validation_for_all_steps = Mock(return_value=mock_results)
        
        # Get summary (no parameters - it calls run_validation_for_all_steps internally)
        summary = tester.get_validation_summary()
        
        # Should include step type breakdown
        assert "step_type_breakdown" in summary
        step_breakdown = summary["step_type_breakdown"]
        
        assert step_breakdown["Processing"]["total"] == 2
        assert step_breakdown["Processing"]["passed"] == 1
        assert step_breakdown["Processing"]["failed"] == 1
        assert step_breakdown["Processing"]["excluded"] == 0
        
        assert step_breakdown["CreateModel"]["total"] == 1
        assert step_breakdown["CreateModel"]["passed"] == 1
        assert step_breakdown["CreateModel"]["failed"] == 0
        assert step_breakdown["CreateModel"]["excluded"] == 0
        
        assert step_breakdown["Base"]["total"] == 1
        assert step_breakdown["Base"]["passed"] == 0
        assert step_breakdown["Base"]["failed"] == 0
        assert step_breakdown["Base"]["excluded"] == 1

    def test_performance_optimization_level_skipping(self, tester):
        """Test that level skipping provides performance optimization."""
        # This test verifies that the configuration-driven approach
        # actually skips validation levels as expected
        
        with patch('cursus.validation.alignment.unified_alignment_tester.get_sagemaker_step_type') as mock_get_step_type, \
             patch('cursus.validation.alignment.unified_alignment_tester.get_validation_ruleset') as mock_get_ruleset, \
             patch('cursus.validation.alignment.unified_alignment_tester.is_step_type_excluded') as mock_is_excluded:
            
            # Setup for non-script step type (should skip levels 1-2)
            mock_get_step_type.return_value = "CreateModel"
            mock_is_excluded.return_value = False
            mock_ruleset = Mock()
            mock_ruleset.enabled_levels = {ValidationLevel.SPEC_DEPENDENCY, ValidationLevel.BUILDER_CONFIG}
            mock_ruleset.level_4_validator_class = "CreateModelStepBuilderValidator"
            mock_get_ruleset.return_value = mock_ruleset
            
            # Mock level validators to track calls
            tester.level_validators.run_level_1_validation = Mock()
            tester.level_validators.run_level_2_validation = Mock()
            tester.level_validators.run_level_3_validation = Mock(return_value={"passed": True, "issues": []})
            tester.level_validators.run_level_4_validation = Mock(return_value={"passed": True, "issues": []})
            
            # Run validation
            result = tester.run_validation_for_step("create_model_step")
            
            # Verify performance optimization: levels 1 and 2 should be skipped
            tester.level_validators.run_level_1_validation.assert_not_called()
            tester.level_validators.run_level_2_validation.assert_not_called()
            
            # Only levels 3 and 4 should be called
            tester.level_validators.run_level_3_validation.assert_called_once()
            tester.level_validators.run_level_4_validation.assert_called_once()
            
            # Result should indicate optimization (these fields may not exist in actual implementation)
            # Just verify the core functionality - levels were skipped and result structure is correct
            assert result["step_name"] == "create_model_step"
            assert result["sagemaker_step_type"] == "CreateModel"
            assert result["overall_status"] == "PASSED"
            assert "level_3" in result["validation_results"]
            assert "level_4" in result["validation_results"]


class TestUnifiedAlignmentTesterBackwardCompatibility:
    """Test backward compatibility of refactored UnifiedAlignmentTester."""

    @pytest.fixture
    def tester(self):
        """Create UnifiedAlignmentTester instance for backward compatibility testing."""
        with patch('cursus.validation.alignment.unified_alignment_tester.StepCatalog'), \
             patch('cursus.validation.alignment.unified_alignment_tester.LevelValidators'):
            return UnifiedAlignmentTester(workspace_dirs=["/mock/workspace"])

    def test_validate_specific_script_backward_compatibility(self, tester):
        """Test that validate_specific_script method still works (backward compatibility)."""
        with patch.object(tester, 'run_validation_for_step') as mock_run_validation:
            mock_run_validation.return_value = {"step_name": "test_script", "overall_status": "PASSING"}
            
            # Should still support old method name
            result = tester.validate_specific_script("test_script")
            
            # Should call new method internally
            mock_run_validation.assert_called_once_with("test_script")
            assert result["step_name"] == "test_script"

    def test_old_api_parameters_still_work(self):
        """Test that old API parameters are still accepted."""
        with patch('cursus.validation.alignment.unified_alignment_tester.StepCatalog'), \
             patch('cursus.validation.alignment.unified_alignment_tester.LevelValidators'):
            
            # Old API should still work
            tester = UnifiedAlignmentTester(
                workspace_dirs=["/mock/workspace"],
                # These old parameters should be ignored gracefully
                level3_validation_mode="relaxed",  # Old parameter
                enable_scoring=True,  # Old parameter
            )
            
            # Should initialize successfully
            assert tester.workspace_dirs == ["/mock/workspace"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
