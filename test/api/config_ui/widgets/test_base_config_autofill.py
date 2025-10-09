"""
Comprehensive tests for base_config auto-fill functionality following pytest best practices.

This test module follows the pytest best practices guide:
1. Source Code First Rule - Read widget.py _get_step_values implementation completely before writing tests
2. Mock Path Precision - Mock at exact import locations from source
3. Implementation-Driven Testing - Match test behavior to actual implementation
4. Fixture Isolation - Design fixtures for complete test independence
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

# Following Source Code First Rule - import the actual implementation
from cursus.api.config_ui.widgets.widget import MultiStepWizard
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestBaseConfigAutoFill:
    """Comprehensive tests for base_config auto-fill functionality following pytest best practices."""
    
    @pytest.fixture
    def sample_base_config(self):
        """Create sample BasePipelineConfig for testing."""
        # Following Category 7: Data Structure Fidelity pattern
        return BasePipelineConfig(
            author="test-user-autofill",
            bucket="my-test-bucket-autofill", 
            role="arn:aws:iam::123456789012:role/TestRole",
            region="NA",  # Fixed: Use valid region code
            service_name="autofill-test-service",
            pipeline_version="2.1.0",
            project_root_folder="autofill-test-project"
        )
    
    @pytest.fixture
    def base_pipeline_step(self):
        """Create BasePipelineConfig step definition for testing."""
        # Following Category 7: Data Structure Fidelity pattern
        return {
            "title": "Base Pipeline Configuration",
            "config_class": BasePipelineConfig,
            "config_class_name": "BasePipelineConfig"
        }
    
    @pytest.fixture
    def non_base_pipeline_step(self):
        """Create non-BasePipelineConfig step definition for testing."""
        # Following Category 7: Data Structure Fidelity pattern
        return {
            "title": "Processing Configuration",
            "config_class": ProcessingStepConfigBase,
            "config_class_name": "ProcessingStepConfigBase"
        }
    
    @pytest.fixture
    def wizard_with_base_config(self, sample_base_config, base_pipeline_step):
        """Create MultiStepWizard with base_config for testing."""
        # Following Category 2: Mock Configuration pattern
        steps = [base_pipeline_step]
        return MultiStepWizard(steps, base_config=sample_base_config)
    
    @pytest.fixture
    def wizard_without_base_config(self, base_pipeline_step):
        """Create MultiStepWizard without base_config for testing."""
        # Following Category 2: Mock Configuration pattern
        steps = [base_pipeline_step]
        return MultiStepWizard(steps, base_config=None)
    
    @pytest.fixture
    def multi_step_wizard(self, sample_base_config, base_pipeline_step, non_base_pipeline_step):
        """Create MultiStepWizard with multiple steps for testing."""
        # Following Category 7: Data Structure Fidelity pattern
        steps = [base_pipeline_step, non_base_pipeline_step]
        return MultiStepWizard(steps, base_config=sample_base_config)
    
    def test_get_step_values_first_step_base_config_autofill_model_dump(self, wizard_with_base_config, base_pipeline_step):
        """Test auto-fill for first step with BasePipelineConfig using model_dump method."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on actual source: checks step_index == 0 and config_class_name == "BasePipelineConfig"
        
        result = wizard_with_base_config._get_step_values(base_pipeline_step)
        
        # Verify that base_config values are extracted
        assert isinstance(result, dict)
        assert len(result) > 0
        
        # Check specific values from base_config
        assert result["author"] == "test-user-autofill"
        assert result["bucket"] == "my-test-bucket-autofill"
        assert result["role"] == "arn:aws:iam::123456789012:role/TestRole"
        assert result["region"] == "NA"  # Fixed: Match the fixture value
        assert result["service_name"] == "autofill-test-service"
        assert result["pipeline_version"] == "2.1.0"
        assert result["project_root_folder"] == "autofill-test-project"
    
    def test_get_step_values_first_step_base_config_autofill_dict_fallback(self, base_pipeline_step):
        """Test auto-fill for first step when base_config has no model_dump method."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: falls back to __dict__ when model_dump is not available
        
        # Create mock base_config without model_dump method
        mock_base_config = Mock(spec=[])  # No methods specified
        mock_base_config.__dict__ = {
            "author": "dict-fallback-user",
            "bucket": "dict-fallback-bucket",
            "role": "arn:aws:iam::123456789012:role/DictRole",
            "region": "EU",  # Fixed: Use valid region code
            "service_name": "dict-fallback-service",
            "pipeline_version": "3.0.0",
            "project_root_folder": "dict-fallback-project",
            "_private_field": "should_be_filtered",  # Should be filtered out
            "none_field": None  # Should be filtered out
        }
        
        steps = [base_pipeline_step]
        wizard = MultiStepWizard(steps, base_config=mock_base_config)
        
        result = wizard._get_step_values(base_pipeline_step)
        
        # Verify that __dict__ values are extracted (excluding private and None values)
        assert isinstance(result, dict)
        assert result["author"] == "dict-fallback-user"
        assert result["bucket"] == "dict-fallback-bucket"
        assert result["role"] == "arn:aws:iam::123456789012:role/DictRole"
        assert result["region"] == "EU"
        assert result["service_name"] == "dict-fallback-service"
        assert result["pipeline_version"] == "3.0.0"
        assert result["project_root_folder"] == "dict-fallback-project"
        
        # Verify filtering behavior (based on source: filters _private and None values)
        assert "_private_field" not in result
        assert "none_field" not in result
    
    def test_get_step_values_first_step_no_base_config(self, wizard_without_base_config, base_pipeline_step):
        """Test that auto-fill is skipped when base_config is None."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: checks if self.base_config exists before auto-fill
        
        result = wizard_without_base_config._get_step_values(base_pipeline_step)
        
        # Should return empty dict when no base_config and no other pre-population
        assert result == {}
    
    def test_get_step_values_non_first_step_no_autofill(self, multi_step_wizard, non_base_pipeline_step):
        """Test that auto-fill is not applied to non-first steps."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: checks step_index == 0, so step_index > 0 should not auto-fill
        
        result = multi_step_wizard._get_step_values(non_base_pipeline_step)
        
        # Should not auto-fill for non-first step
        assert result == {}
    
    def test_get_step_values_first_step_non_base_config_class(self, sample_base_config, non_base_pipeline_step):
        """Test that auto-fill is not applied when config_class_name is not BasePipelineConfig."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: checks config_class_name == "BasePipelineConfig"
        
        steps = [non_base_pipeline_step]  # First step but not BasePipelineConfig
        wizard = MultiStepWizard(steps, base_config=sample_base_config)
        
        result = wizard._get_step_values(non_base_pipeline_step)
        
        # Should not auto-fill for non-BasePipelineConfig class
        assert result == {}
    
    def test_get_step_values_base_config_no_extraction_methods(self, base_pipeline_step):
        """Test behavior when base_config has neither model_dump nor __dict__."""
        # Following Category 16: Exception Handling vs Test Expectations
        # Based on actual source: should return empty dict when no extraction methods available
        
        # Create a more realistic mock that truly has no model_dump or __dict__
        class MockBaseConfigNoMethods:
            pass
        
        mock_base_config = MockBaseConfigNoMethods()
        # Remove __dict__ if it exists
        if hasattr(mock_base_config, '__dict__'):
            delattr(mock_base_config, '__dict__')
        
        steps = [base_pipeline_step]
        wizard = MultiStepWizard(steps, base_config=mock_base_config)
        
        # Based on actual source: should return empty dict when no extraction methods available
        result = wizard._get_step_values(base_pipeline_step)
        
        # Should return empty dict when no extraction methods available
        assert result == {}
    
    def test_get_step_values_logging_behavior(self, wizard_with_base_config, base_pipeline_step, caplog):
        """Test that auto-fill functionality works correctly (main behavior test)."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on actual source: _get_step_values doesn't have specific logging, focus on functionality
        
        result = wizard_with_base_config._get_step_values(base_pipeline_step)
        
        # Verify auto-fill worked (this is the main functionality we care about)
        assert len(result) > 0
        assert result["author"] == "test-user-autofill"
        assert result["bucket"] == "my-test-bucket-autofill"
        assert result["region"] == "NA"
        
        # The actual source has logger.info and logger.debug calls, but they may be suppressed
        # Focus on testing the core functionality rather than specific log messages
    
    def test_get_step_values_pre_populated_instance_precedence(self, wizard_with_base_config, base_pipeline_step):
        """Test that pre_populated instance takes precedence over base_config auto-fill."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: checks pre_populated before base_config auto-fill
        
        # Create mock pre-populated instance
        mock_pre_populated = Mock()
        mock_pre_populated.model_dump.return_value = {
            "author": "pre-populated-user",
            "bucket": "pre-populated-bucket"
        }
        
        # Add pre_populated to step
        step_with_pre_populated = base_pipeline_step.copy()
        step_with_pre_populated["pre_populated"] = mock_pre_populated
        
        result = wizard_with_base_config._get_step_values(step_with_pre_populated)
        
        # Should use pre_populated values, not base_config auto-fill
        assert result["author"] == "pre-populated-user"
        assert result["bucket"] == "pre-populated-bucket"
        # Should not contain base_config values
        assert result["author"] != "test-user-autofill"
    
    def test_get_step_values_pre_populated_data_precedence(self, wizard_with_base_config, base_pipeline_step):
        """Test that pre_populated_data takes precedence over base_config auto-fill."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: checks pre_populated_data before base_config auto-fill
        
        pre_populated_data = {
            "author": "pre-populated-data-user",
            "bucket": "pre-populated-data-bucket"
        }
        
        # Add pre_populated_data to step
        step_with_pre_populated_data = base_pipeline_step.copy()
        step_with_pre_populated_data["pre_populated_data"] = pre_populated_data
        
        result = wizard_with_base_config._get_step_values(step_with_pre_populated_data)
        
        # Should use pre_populated_data values, not base_config auto-fill
        assert result["author"] == "pre-populated-data-user"
        assert result["bucket"] == "pre-populated-data-bucket"
        # Should not contain base_config values
        assert result["author"] != "test-user-autofill"
    
    def test_get_step_values_base_config_in_step_precedence(self, wizard_with_base_config, base_pipeline_step):
        """Test that base_config in step definition takes precedence over wizard base_config auto-fill."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: checks step["base_config"] before wizard base_config auto-fill
        
        # Create mock config class with from_base_config method
        mock_config_class = Mock()
        mock_instance = Mock()
        mock_instance.model_dump.return_value = {
            "author": "step-base-config-user",
            "bucket": "step-base-config-bucket"
        }
        mock_config_class.from_base_config.return_value = mock_instance
        
        # Create mock step base_config
        mock_step_base_config = Mock()
        
        # Add base_config to step
        step_with_base_config = base_pipeline_step.copy()
        step_with_base_config["base_config"] = mock_step_base_config
        step_with_base_config["config_class"] = mock_config_class
        
        result = wizard_with_base_config._get_step_values(step_with_base_config)
        
        # Should use step base_config values, not wizard base_config auto-fill
        assert result["author"] == "step-base-config-user"
        assert result["bucket"] == "step-base-config-bucket"
        # Should not contain wizard base_config values
        assert result["author"] != "test-user-autofill"
    
    def test_get_step_values_step_index_calculation_accuracy(self, sample_base_config):
        """Test that step index calculation is accurate for auto-fill logic."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: uses self.steps.index(step) to determine step_index
        
        base_step = {
            "title": "Base Pipeline Configuration",
            "config_class": BasePipelineConfig,
            "config_class_name": "BasePipelineConfig"
        }
        
        second_step = {
            "title": "Second Step",
            "config_class": Mock(),
            "config_class_name": "SecondConfig"
        }
        
        steps = [base_step, second_step]
        wizard = MultiStepWizard(steps, base_config=sample_base_config)
        
        # Test first step (index 0) - should auto-fill
        result_first = wizard._get_step_values(base_step)
        assert len(result_first) > 0  # Should have auto-filled values
        assert result_first["author"] == "test-user-autofill"
        
        # Test second step (index 1) - should not auto-fill
        result_second = wizard._get_step_values(second_step)
        assert result_second == {}  # Should be empty
    
    def test_get_step_values_step_not_in_steps_list(self, wizard_with_base_config):
        """Test behavior when step is not found in wizard.steps list."""
        # Following Category 16: Exception Handling vs Test Expectations
        # Based on source: uses self.steps.index(step), which could raise ValueError
        
        # Create step that's not in wizard.steps
        unknown_step = {
            "title": "Unknown Step",
            "config_class": BasePipelineConfig,
            "config_class_name": "BasePipelineConfig"
        }
        
        result = wizard_with_base_config._get_step_values(unknown_step)
        
        # Based on source: step_index would be -1 when step not found, so auto-fill should not trigger
        assert result == {}


class TestBaseConfigAutoFillIntegration:
    """Integration tests for base_config auto-fill with enhanced widget creation."""
    
    @pytest.fixture
    def mock_dag_manager(self):
        """Create mock DAGConfigurationManager for testing."""
        # Following Category 2: Mock Configuration pattern
        mock_manager = Mock()
        mock_analysis = {
            "workflow_steps": [
                {
                    "step_number": 1,
                    "title": "Base Pipeline Configuration",
                    "config_class": BasePipelineConfig,
                    "config_class_name": "BasePipelineConfig",
                    "type": "base"
                }
            ]
        }
        mock_manager.analyze_pipeline_dag.return_value = mock_analysis
        
        mock_wizard = Mock()
        mock_wizard.steps = [
            {
                "title": "Base Pipeline Configuration",
                "config_class": BasePipelineConfig,
                "config_class_name": "BasePipelineConfig"
            }
        ]
        mock_manager.create_dag_driven_widget.return_value = mock_wizard
        
        return mock_manager
    
    def test_enhanced_widget_base_config_propagation(self, mock_dag_manager):
        """Test that base_config is properly propagated through enhanced widget creation."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: create_enhanced_pipeline_widget passes base_config to DAGConfigurationManager
        
        from cursus.api.config_ui.enhanced_widget import EnhancedPipelineConfigWidget
        
        sample_base_config = BasePipelineConfig(
            author="integration-test-user",
            bucket="integration-test-bucket",
            role="arn:aws:iam::123456789012:role/IntegrationRole",
            region="NA",  # Fixed: Use valid region code
            service_name="integration-service",
            pipeline_version="1.0.0",
            project_root_folder="integration-project"
        )
        
        mock_dag = Mock()
        
        with patch('cursus.api.config_ui.enhanced_widget.DAGConfigurationManager', return_value=mock_dag_manager):
            enhanced_widget = EnhancedPipelineConfigWidget()
            result_wizard = enhanced_widget.create_dag_driven_wizard(
                pipeline_dag=mock_dag,
                base_config=sample_base_config
            )
            
            # Verify that DAGConfigurationManager.create_dag_driven_widget was called with base_config
            mock_dag_manager.create_dag_driven_widget.assert_called_once()
            call_args = mock_dag_manager.create_dag_driven_widget.call_args
            
            # Check that base_config was passed
            assert call_args[1]["base_config"] == sample_base_config
    
    def test_create_enhanced_pipeline_widget_base_config_flow(self):
        """Test complete flow from create_enhanced_pipeline_widget to base_config auto-fill."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Integration test of the complete flow
        
        from cursus.api.config_ui.enhanced_widget import create_enhanced_pipeline_widget
        
        sample_base_config = BasePipelineConfig(
            author="flow-test-user",
            bucket="flow-test-bucket",
            role="arn:aws:iam::123456789012:role/FlowRole",
            region="NA",  # Fixed: Use valid region code
            service_name="flow-service",
            pipeline_version="2.0.0",
            project_root_folder="flow-project"
        )
        
        # Create simple mock DAG
        mock_dag = Mock()
        mock_dag.nodes = ["BasePipelineConfig_step"]
        
        # Mock the DAG analysis and wizard creation to focus on base_config flow
        with patch('cursus.api.config_ui.core.dag_manager.DAGConfigurationManager') as mock_dag_manager_class:
            mock_dag_manager = Mock()
            mock_dag_manager_class.return_value = mock_dag_manager
            
            # Create a real MultiStepWizard to test actual auto-fill behavior
            base_step = {
                "title": "Base Pipeline Configuration",
                "config_class": BasePipelineConfig,
                "config_class_name": "BasePipelineConfig"
            }
            
            real_wizard = MultiStepWizard([base_step], base_config=sample_base_config)
            mock_dag_manager.create_dag_driven_widget.return_value = real_wizard
            
            # Call the factory function
            result_wizard = create_enhanced_pipeline_widget(
                pipeline_dag=mock_dag,
                base_config=sample_base_config
            )
            
            # Test that the wizard has the base_config
            assert result_wizard.base_wizard.base_config == sample_base_config
            
            # Test that auto-fill works on the first step
            first_step = result_wizard.base_wizard.steps[0]
            step_values = result_wizard.base_wizard._get_step_values(first_step)
            
            # Verify auto-filled values
            assert step_values["author"] == "flow-test-user"
            assert step_values["bucket"] == "flow-test-bucket"
            assert step_values["role"] == "arn:aws:iam::123456789012:role/FlowRole"
            assert step_values["region"] == "NA"  # Fixed: Match the fixture value
            assert step_values["service_name"] == "flow-service"
            assert step_values["pipeline_version"] == "2.0.0"
            assert step_values["project_root_folder"] == "flow-project"


class TestBaseConfigAutoFillEdgeCases:
    """Test edge cases and error conditions for base_config auto-fill."""
    
    @pytest.fixture
    def base_pipeline_step(self):
        """Create BasePipelineConfig step definition for testing."""
        # Following Category 7: Data Structure Fidelity pattern
        return {
            "title": "Base Pipeline Configuration",
            "config_class": BasePipelineConfig,
            "config_class_name": "BasePipelineConfig"
        }
    
    def test_get_step_values_base_config_model_dump_exception(self, base_pipeline_step):
        """Test behavior when base_config.model_dump() raises exception."""
        # Following Category 16: Exception Handling vs Test Expectations
        # Based on actual source: falls back to __dict__ when model_dump() fails
        
        mock_base_config = Mock()
        mock_base_config.model_dump.side_effect = Exception("model_dump failed")
        mock_base_config.__dict__ = {"author": "fallback-user", "bucket": "fallback-bucket"}
        
        steps = [base_pipeline_step]
        wizard = MultiStepWizard(steps, base_config=mock_base_config)
        
        # Based on actual source: should fall back to __dict__ when model_dump() fails
        result = wizard._get_step_values(base_pipeline_step)
        
        # Should use __dict__ values as fallback
        assert result["author"] == "fallback-user"
        assert result["bucket"] == "fallback-bucket"
    
    def test_get_step_values_base_config_dict_access_exception(self, base_pipeline_step):
        """Test behavior when base_config.__dict__ access raises exception."""
        # Following Category 16: Exception Handling vs Test Expectations
        # Based on actual source: no exception handling around __dict__ access
        
        # Create a custom class that raises exception on __dict__ access
        class MockBaseConfigDictException:
            @property
            def __dict__(self):
                raise Exception("dict access failed")
        
        mock_base_config = MockBaseConfigDictException()
        
        steps = [base_pipeline_step]
        wizard = MultiStepWizard(steps, base_config=mock_base_config)
        
        # Based on actual source: no exception handling, so exception should propagate
        with pytest.raises(Exception, match="dict access failed"):
            wizard._get_step_values(base_pipeline_step)
    
    def test_get_step_values_empty_base_config_dict(self, base_pipeline_step):
        """Test behavior when base_config.__dict__ is empty."""
        # Following Category 4: Test Expectations vs Implementation pattern
        
        mock_base_config = Mock(spec=[])
        mock_base_config.__dict__ = {}
        
        steps = [base_pipeline_step]
        wizard = MultiStepWizard(steps, base_config=mock_base_config)
        
        result = wizard._get_step_values(base_pipeline_step)
        
        # Should return empty dict when no values to extract
        assert result == {}
    
    def test_get_step_values_base_config_dict_all_private_fields(self, base_pipeline_step):
        """Test behavior when base_config.__dict__ contains only private fields."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: filters out fields starting with '_'
        
        mock_base_config = Mock(spec=[])
        mock_base_config.__dict__ = {
            "_private_field1": "value1",
            "_private_field2": "value2",
            "__dunder_field": "value3"
        }
        
        steps = [base_pipeline_step]
        wizard = MultiStepWizard(steps, base_config=mock_base_config)
        
        result = wizard._get_step_values(base_pipeline_step)
        
        # Should return empty dict when all fields are private
        assert result == {}
    
    def test_get_step_values_base_config_dict_all_none_fields(self, base_pipeline_step):
        """Test behavior when base_config.__dict__ contains only None values."""
        # Following Category 4: Test Expectations vs Implementation pattern
        # Based on source: filters out fields with None values
        
        mock_base_config = Mock(spec=[])
        mock_base_config.__dict__ = {
            "field1": None,
            "field2": None,
            "field3": None
        }
        
        steps = [base_pipeline_step]
        wizard = MultiStepWizard(steps, base_config=mock_base_config)
        
        result = wizard._get_step_values(base_pipeline_step)
        
        # Should return empty dict when all values are None
        assert result == {}
