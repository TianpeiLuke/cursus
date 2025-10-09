"""
Test suite for robust widget rendering system.

Tests to ensure:
1. No duplicate displays
2. No missing rendering
3. Proper state management
4. VS Code compatibility features
"""

import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch, call
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Javascript

# Import the classes we're testing
from cursus.api.config_ui.widgets.widget import UniversalConfigWidget, MultiStepWizard
from cursus.core.base.config_base import BasePipelineConfig
from cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase


class TestRobustRendering:
    """Test suite for robust widget rendering system."""
    
    @pytest.fixture
    def mock_base_config(self):
        """Create a mock base configuration with all required fields from config_base."""
        return BasePipelineConfig(
            author="test-user",
            bucket="test-bucket", 
            role="arn:aws:iam::123456789012:role/TestRole",
            region="NA",  # Must be NA, EU, or FE according to config_base
            service_name="test-service",
            pipeline_version="1.0.0",
            project_root_folder="test-project"
        )
    
    @pytest.fixture
    def mock_form_data(self):
        """Create mock form data for UniversalConfigWidget."""
        return {
            "config_class": BasePipelineConfig,
            "config_class_name": "BasePipelineConfig",
            "fields": [
                {"name": "author", "type": "text", "required": True, "tier": "essential"},
                {"name": "bucket", "type": "text", "required": True, "tier": "essential"},
                {"name": "region", "type": "text", "required": False, "tier": "system"}
            ],
            "values": {
                "author": "test-user",
                "bucket": "test-bucket",
                "region": "us-east-1"
            }
        }
    
    @pytest.fixture
    def mock_steps(self, mock_base_config):
        """Create mock steps for MultiStepWizard."""
        return [
            {
                "title": "Base Configuration",
                "config_class": BasePipelineConfig,
                "config_class_name": "BasePipelineConfig",
                "description": "Configure base pipeline settings"
            },
            {
                "title": "Processing Configuration", 
                "config_class": ProcessingStepConfigBase,
                "config_class_name": "ProcessingStepConfigBase",
                "description": "Configure processing settings"
            }
        ]


class TestUniversalConfigWidgetRendering(TestRobustRendering):
    """Test UniversalConfigWidget rendering robustness."""
    
    def test_widget_initialization_state(self, mock_form_data):
        """Test that widget initializes with correct state flags."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Check initial state - these are the actual attributes from source code
        assert widget._is_rendered == False
        assert widget._is_displayed == False
        assert widget.output is not None
        assert isinstance(widget.output, widgets.Output)
        assert widget.widgets == {}
        assert widget.config_instance is None
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_render_idempotent(self, mock_display, mock_form_data):
        """Test that render() is idempotent - can be called multiple times safely."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Call render multiple times
        widget.render()
        widget.render()
        widget.render()
        
        # Should only render once
        assert widget._is_rendered == True
        
        # Display should be called only once per render call (within the output context)
        # The exact number depends on internal structure, but should be consistent
        first_call_count = mock_display.call_count
        
        # Call render again
        widget.render()
        
        # Should not increase display calls
        assert mock_display.call_count == first_call_count
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_display_returns_output_widget(self, mock_display, mock_form_data):
        """Test that display() returns the output widget without displaying it."""
        widget = UniversalConfigWidget(mock_form_data)
        
        result = widget.display()
        
        # Should return the output widget
        assert result == widget.output
        assert widget._is_rendered == True
        # Should not mark as displayed (that's for show() method)
        assert widget._is_displayed == False
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_show_prevents_duplicate_display(self, mock_display, mock_form_data):
        """Test that show() prevents duplicate displays."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Call show multiple times
        widget.show()
        widget.show()
        widget.show()
        
        # Should only display once
        assert widget._is_displayed == True
        
        # The display call should happen only once for the widget itself
        # (internal render calls don't count as external displays)
        display_calls = [call for call in mock_display.call_args_list 
                        if call[0] and call[0][0] == widget.output]
        assert len(display_calls) == 1
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_render_before_display(self, mock_display, mock_form_data):
        """Test that display() always calls render() first."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Mock the render method to track calls
        with patch.object(widget, 'render', wraps=widget.render) as mock_render:
            widget.display()
            
            # Should call render
            mock_render.assert_called_once()
            assert widget._is_rendered == True


class TestMultiStepWizardRendering(TestRobustRendering):
    """Test MultiStepWizard rendering robustness."""
    
    def test_wizard_initialization(self, mock_steps, mock_base_config):
        """Test that wizard initializes correctly."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        assert len(wizard.steps) == len(mock_steps)
        assert wizard.current_step == 0
        assert wizard.output is not None
        assert wizard.navigation_output is not None
        assert isinstance(wizard.output, widgets.Output)
        assert isinstance(wizard.navigation_output, widgets.Output)
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_single_container_display(self, mock_display, mock_steps, mock_base_config):
        """Test that display() creates a single container and displays it once."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        wizard.display()
        
        # Should create and display a single VBox container
        display_calls = mock_display.call_args_list
        
        # Find the call that displays the main wizard container
        container_calls = [call for call in display_calls 
                          if call[0] and isinstance(call[0][0], widgets.VBox)]
        
        # Should have at least one VBox container display
        assert len(container_calls) >= 1
        
        # The main container should contain both navigation and content outputs
        main_container = container_calls[-1][0][0]  # Last VBox should be main container
        assert wizard.navigation_output in main_container.children
        assert wizard.output in main_container.children
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    @patch('cursus.api.config_ui.widgets.widget.clear_output')
    def test_navigation_and_content_separation(self, mock_clear, mock_display, mock_steps, mock_base_config):
        """Test that navigation and content are properly separated."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        wizard.display()
        
        # Should clear both outputs
        assert mock_clear.call_count >= 2  # At least navigation and content
        
        # Should populate both outputs
        assert mock_display.call_count > 0
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_vscode_compatibility_enhancement(self, mock_display, mock_steps, mock_base_config):
        """Test that VS Code compatibility enhancements are applied."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        # Mock the VS Code enhancement method
        with patch.object(wizard, '_ensure_vscode_widget_display') as mock_enhance:
            wizard.display()
            
            # Should call VS Code enhancement
            mock_enhance.assert_called_once()
            
            # Should be called with the main container
            call_args = mock_enhance.call_args[0]
            assert len(call_args) == 1
            assert isinstance(call_args[0], widgets.VBox)
    
    def test_vscode_widget_model_initialization(self, mock_steps, mock_base_config):
        """Test VS Code widget model initialization."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        # Create a mock widget without model_id
        mock_widget = MagicMock()
        mock_widget._model_id = None
        mock_widget._gen_model_id.return_value = "test-model-id"
        mock_widget.children = []
        
        # Test the enhancement method
        wizard._ensure_vscode_widget_display(mock_widget)
        
        # Should generate model ID
        mock_widget._gen_model_id.assert_called_once()
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    @patch('IPython.display.Javascript')
    def test_javascript_enhancement_injection(self, mock_js, mock_display, mock_steps, mock_base_config):
        """Test that JavaScript enhancement is injected for VS Code compatibility."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        # Create a simple widget to test
        test_widget = widgets.VBox([])
        
        wizard._ensure_vscode_widget_display(test_widget)
        
        # Should inject JavaScript enhancement
        mock_js.assert_called_once()
        
        # Check that the JavaScript contains VS Code compatibility code
        js_code = mock_js.call_args[0][0]
        assert "VS Code Jupyter Widget Display Enhancement" in js_code
        assert "widget compatibility" in js_code.lower()


class TestDisplayStateManagement(TestRobustRendering):
    """Test display state management system."""
    
    def test_state_transitions(self, mock_form_data):
        """Test proper state transitions during widget lifecycle."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Initial state
        assert widget._is_rendered == False
        assert widget._is_displayed == False
        
        # After render
        with patch('cursus.api.config_ui.widgets.widget.display'):
            widget.render()
            assert widget._is_rendered == True
            assert widget._is_displayed == False
        
        # After show
        with patch('cursus.api.config_ui.widgets.widget.display'):
            widget.show()
            assert widget._is_rendered == True
            assert widget._is_displayed == True
    
    def test_display_method_safety(self, mock_form_data):
        """Test that display() method is safe and doesn't change state inappropriately."""
        widget = UniversalConfigWidget(mock_form_data)
        
        with patch('cursus.api.config_ui.widgets.widget.display'):
            # Call display multiple times
            result1 = widget.display()
            result2 = widget.display()
            result3 = widget.display()
            
            # Should always return the same output widget
            assert result1 == result2 == result3 == widget.output
            
            # Should render but not mark as displayed
            assert widget._is_rendered == True
            assert widget._is_displayed == False


class TestErrorHandling(TestRobustRendering):
    """Test error handling in robust rendering system."""
    
    def test_vscode_enhancement_error_handling(self, mock_steps, mock_base_config):
        """Test that VS Code enhancement errors are handled gracefully."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        # Create a widget that will cause an error
        mock_widget = MagicMock()
        mock_widget._gen_model_id.side_effect = Exception("Test error")
        
        # Should not raise exception
        try:
            wizard._ensure_vscode_widget_display(mock_widget)
        except Exception as e:
            pytest.fail(f"VS Code enhancement should handle errors gracefully, but raised: {e}")
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_render_error_recovery(self, mock_display, mock_form_data):
        """Test that render errors don't break the widget permanently."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Make display raise an error on first call
        mock_display.side_effect = [Exception("Test error"), None]
        
        # First render should fail but not break the widget
        with pytest.raises(Exception):
            widget.render()
        
        # State should not be marked as rendered due to error
        assert widget._is_rendered == False
        
        # Second render should work (error is gone)
        mock_display.side_effect = None
        widget.render()
        assert widget._is_rendered == True


class TestIntegrationScenarios(TestRobustRendering):
    """Test integration scenarios that could cause rendering issues."""
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_rapid_display_calls(self, mock_display, mock_steps, mock_base_config):
        """Test rapid successive display calls don't cause issues."""
        wizard = MultiStepWizard(mock_steps, mock_base_config)
        
        # Rapidly call display multiple times
        for _ in range(10):
            wizard.display()
        
        # Should handle gracefully without errors
        assert mock_display.call_count > 0
    
    @patch('cursus.api.config_ui.widgets.widget.display')
    def test_mixed_display_and_show_calls(self, mock_display, mock_form_data):
        """Test mixing display() and show() calls."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Mix different display methods
        widget.display()
        widget.show()
        widget.display()
        widget.show()
        
        # Should handle gracefully
        assert widget._is_rendered == True
        assert widget._is_displayed == True
    
    def test_widget_cleanup_and_recreation(self, mock_form_data):
        """Test that widgets can be properly cleaned up and recreated."""
        # Create and use widget
        widget1 = UniversalConfigWidget(mock_form_data)
        with patch('cursus.api.config_ui.widgets.widget.display'):
            widget1.render()
        
        # Create new widget with same data
        widget2 = UniversalConfigWidget(mock_form_data)
        with patch('cursus.api.config_ui.widgets.widget.display'):
            widget2.render()
        
        # Should be independent
        assert widget1.output != widget2.output
        assert widget1._is_rendered == True
        assert widget2._is_rendered == True


class TestPerformanceAndMemory(TestRobustRendering):
    """Test performance and memory aspects of robust rendering."""
    
    def test_no_memory_leaks_in_repeated_renders(self, mock_form_data):
        """Test that repeated renders don't create memory leaks."""
        widget = UniversalConfigWidget(mock_form_data)
        
        # Store initial widget count (should be 0)
        initial_widgets = len(widget.widgets)
        assert initial_widgets == 0
        
        with patch('cursus.api.config_ui.widgets.widget.display'):
            # First render should create widgets
            widget.render()
            first_render_count = len(widget.widgets)
            
            # Subsequent renders should not increase widget count
            for _ in range(4):
                widget.render()
        
        # Widget count should remain the same after first render
        assert len(widget.widgets) == first_render_count
    
    def test_efficient_state_checking(self, mock_form_data):
        """Test that state checking is efficient."""
        widget = UniversalConfigWidget(mock_form_data)
        
        with patch('cursus.api.config_ui.widgets.widget.display'):
            # First render should do work
            widget.render()
            
            # Subsequent renders should return early
            with patch.object(widget, '_create_field_section') as mock_create:
                widget.render()
                widget.render()
                
                # Should not create sections again
                mock_create.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
