"""
Test Navigation State Persistence Fix

Tests the fix for the Next → Previous → Next navigation issue
where user data could be lost during navigation.
"""

import pytest
from unittest.mock import Mock, patch

# Import the classes we're testing - use proper imports without path manipulation
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from cursus.api.config_ui.widgets.widget import MultiStepWizard


class TestNavigationStatePersistence:
    """Test suite for navigation state persistence fix."""
    
    @pytest.fixture
    def test_base_config(self):
        """Create test base configuration."""
        return BasePipelineConfig(
            author="john",
            bucket="test-bucket",
            role="arn:aws:iam::123456789012:role/TestRole",
            region="NA",
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
    
    @pytest.fixture
    def test_steps(self):
        """Create test steps for wizard."""
        return [
            {
                "step_number": 1,
                "title": "Base Configuration",
                "config_class": BasePipelineConfig,
                "config_class_name": "BasePipelineConfig",
                "type": "base",
                "required": True
            },
            {
                "step_number": 2,
                "title": "Processing Configuration",
                "config_class": ProcessingStepConfigBase,
                "config_class_name": "ProcessingStepConfigBase",
                "type": "processing",
                "required": True
            }
        ]
    
    def test_previous_button_saves_current_step(self, test_steps, test_base_config):
        """Test that Previous button saves current step before navigation."""
        wizard = MultiStepWizard(
            steps=test_steps,
            base_config=test_base_config,
            enable_inheritance=True
        )
        
        # Start on step 1 (index 0)
        wizard.current_step = 1  # Move to step 2
        
        # Mock the step widget with form data
        mock_widget = Mock()
        mock_widget.widgets = {
            "author": Mock(value="jane"),  # Modified value
            "bucket": Mock(value="modified-bucket"),
            "role": Mock(value="arn:aws:iam::123456789012:role/ModifiedRole"),
            "region": Mock(value="EU"),
            "service_name": Mock(value="modified-service"),
            "pipeline_version": Mock(value="2.0.0"),
            "project_root_folder": Mock(value="modified-project")
        }
        mock_widget.fields = [
            {"name": field, "type": "text"}
            for field in mock_widget.widgets.keys()
        ]
        
        wizard.step_widgets[1] = mock_widget
        
        # Mock the _save_current_step method to track if it's called
        original_save = wizard._save_current_step
        save_called = []
        
        def mock_save():
            save_called.append(True)
            return original_save()
        
        wizard._save_current_step = mock_save
        
        # Mock the _update_navigation_and_step method
        wizard._update_navigation_and_step = Mock()
        
        # Simulate Previous button click
        mock_button = Mock()
        wizard._on_prev_clicked(mock_button)
        
        # Verify that save was called before navigation
        assert len(save_called) == 1, "Previous button should call _save_current_step"
        
        # Verify navigation occurred (current_step decremented)
        assert wizard.current_step == 0, "Should have moved to previous step"
        
        # Verify update method was called
        wizard._update_navigation_and_step.assert_called_once()
    
    def test_next_previous_next_preserves_data(self, test_steps, test_base_config):
        """Test the complete Next → Previous → Next scenario preserves data."""
        wizard = MultiStepWizard(
            steps=test_steps,
            base_config=test_base_config,
            enable_inheritance=True
        )
        
        # Step 1: Start on step 0, fill data, click Next
        wizard.current_step = 0
        
        # Mock step 0 widget with initial data
        mock_widget_0 = Mock()
        mock_widget_0.widgets = {
            "author": Mock(value="john"),
            "bucket": Mock(value="initial-bucket"),
            "role": Mock(value="arn:aws:iam::123456789012:role/InitialRole"),
            "region": Mock(value="NA"),
            "service_name": Mock(value="initial-service"),
            "pipeline_version": Mock(value="1.0.0"),
            "project_root_folder": Mock(value="initial-project")
        }
        mock_widget_0.fields = [
            {"name": field, "type": "text"}
            for field in mock_widget_0.widgets.keys()
        ]
        wizard.step_widgets[0] = mock_widget_0
        
        # Mock navigation update
        wizard._update_navigation_and_step = Mock()
        
        # Simulate Next button click (Step 0 → Step 1)
        mock_button = Mock()
        wizard._on_next_clicked(mock_button)
        
        # Verify we moved to step 1
        assert wizard.current_step == 1
        
        # Verify step 0 data was saved
        assert "Base Configuration" in wizard.completed_configs
        saved_config_0 = wizard.completed_configs["Base Configuration"]
        assert saved_config_0.author == "john"
        assert saved_config_0.bucket == "initial-bucket"
        
        # Step 2: On step 1, fill different data, click Previous
        mock_widget_1 = Mock()
        mock_widget_1.widgets = {
            "author": Mock(value="jane"),  # Different from step 0
            "bucket": Mock(value="step1-bucket"),
            "role": Mock(value="arn:aws:iam::123456789012:role/Step1Role"),
            "region": Mock(value="FE"),
            "service_name": Mock(value="step1-service"),
            "pipeline_version": Mock(value="1.1.0"),
            "project_root_folder": Mock(value="step1-project")
        }
        mock_widget_1.fields = [
            {"name": field, "type": "text"}
            for field in mock_widget_1.widgets.keys()
        ]
        wizard.step_widgets[1] = mock_widget_1
        
        # Simulate Previous button click (Step 1 → Step 0)
        wizard._on_prev_clicked(mock_button)
        
        # Verify we moved back to step 0
        assert wizard.current_step == 0
        
        # Verify step 1 data was saved before going back
        assert "Processing Configuration" in wizard.completed_configs
        saved_config_1 = wizard.completed_configs["Processing Configuration"]
        assert saved_config_1.author == "jane"
        assert saved_config_1.bucket == "step1-bucket"
        
        # Step 3: On step 0, modify data again, click Next
        # Modify the step 0 widget to have new values
        mock_widget_0.widgets["author"].value = "modified-john"
        mock_widget_0.widgets["bucket"].value = "final-bucket"
        
        # Simulate Next button click (Step 0 → Step 1)
        wizard._on_next_clicked(mock_button)
        
        # Verify we moved to step 1
        assert wizard.current_step == 1
        
        # Verify step 0 data was updated (not overwritten, but properly saved)
        updated_config_0 = wizard.completed_configs["Base Configuration"]
        assert updated_config_0.author == "modified-john"  # New value
        assert updated_config_0.bucket == "final-bucket"   # New value
        
        # Verify step 1 data is still preserved from previous save
        preserved_config_1 = wizard.completed_configs["Processing Configuration"]
        assert preserved_config_1.author == "jane"        # Preserved from step 2
        assert preserved_config_1.bucket == "step1-bucket" # Preserved from step 2
        
        # Final verification: Both configs have their latest values
        assert updated_config_0.author == "modified-john"
        assert preserved_config_1.author == "jane"
        assert updated_config_0 != preserved_config_1  # They are different instances
    
    def test_validation_error_prevents_navigation(self, test_steps, test_base_config):
        """Test that validation errors prevent navigation."""
        wizard = MultiStepWizard(
            steps=test_steps,
            base_config=test_base_config,
            enable_inheritance=True
        )
        
        wizard.current_step = 1
        
        # Mock _save_current_step to return False (validation error)
        wizard._save_current_step = Mock(return_value=False)
        
        # Mock _show_validation_error to track if it's called
        wizard._show_validation_error = Mock()
        
        # Mock _update_navigation_and_step
        wizard._update_navigation_and_step = Mock()
        
        # Simulate Previous button click with validation error
        mock_button = Mock()
        wizard._on_prev_clicked(mock_button)
        
        # Verify save was attempted
        wizard._save_current_step.assert_called_once()
        
        # Verify validation error was shown
        wizard._show_validation_error.assert_called_once_with(
            "Please fix validation errors before navigating"
        )
        
        # Verify navigation did NOT occur
        assert wizard.current_step == 1  # Should still be on step 1
        wizard._update_navigation_and_step.assert_not_called()
    
    def test_validation_error_display_styling(self, test_steps, test_base_config):
        """Test that validation error display has proper styling."""
        wizard = MultiStepWizard(
            steps=test_steps,
            base_config=test_base_config,
            enable_inheritance=True
        )
        
        # Mock the output widget
        wizard.output = Mock()
        wizard.output.__enter__ = Mock(return_value=wizard.output)
        wizard.output.__exit__ = Mock(return_value=None)
        
        # Mock display function
        with patch('cursus.api.config_ui.widgets.widget.display') as mock_display:
            # Call the validation error method
            wizard._show_validation_error("Test validation message")
            
            # Verify display was called
            mock_display.assert_called_once()
            
            # Get the HTML widget that was displayed
            displayed_widget = mock_display.call_args[0][0]
            html_content = displayed_widget.value
            
            # Verify the error styling is present
            assert "background: #fef2f2" in html_content
            assert "border: 1px solid #fecaca" in html_content
            assert "color: #dc2626" in html_content
            assert "⚠️ Validation Error:" in html_content
            assert "Test validation message" in html_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
