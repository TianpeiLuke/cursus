"""
Test suite for CradleNativeWidget - SageMaker Native Cradle Configuration Widget

This test suite follows pytest best practices and covers:
1. Widget initialization and state management
2. 4-step wizard navigation and UX replication
3. Configuration data collection and processing
4. Embedded mode integration with completion callbacks
5. Error handling and edge cases
6. Service integration (ValidationService, ConfigBuilderService)

Following pytest best practices:
- Source code first analysis completed
- Implementation-driven test design
- Mock path precision based on actual imports
- Comprehensive error prevention strategies
"""

import pytest
import logging
from unittest.mock import Mock, MagicMock, patch, call
from typing import Dict, Any, Optional, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# Import the class under test - CRITICAL: Use exact import path from source
from src.cursus.api.config_ui.widgets.cradle_native_widget import CradleNativeWidget

# Import dependencies for mocking - based on source code analysis
try:
    from src.cursus.steps.configs.config_cradle_data_loading_step import (
        CradleDataLoadingConfig,
        DataSourcesSpecificationConfig,
        DataSourceConfig,
        MdsDataSourceConfig,
        EdxDataSourceConfig,
        AndesDataSourceConfig,
        TransformSpecificationConfig,
        JobSplitOptionsConfig,
        OutputSpecificationConfig,
        CradleJobSpecificationConfig
    )
except ImportError:
    # Handle import errors gracefully for test environment
    CradleDataLoadingConfig = Mock
    DataSourcesSpecificationConfig = Mock
    DataSourceConfig = Mock
    MdsDataSourceConfig = Mock
    EdxDataSourceConfig = Mock
    AndesDataSourceConfig = Mock
    TransformSpecificationConfig = Mock
    JobSplitOptionsConfig = Mock
    OutputSpecificationConfig = Mock
    CradleJobSpecificationConfig = Mock

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestCradleNativeWidget:
    """Test suite for CradleNativeWidget following pytest best practices."""
    
    @pytest.fixture(autouse=True)
    def reset_global_state(self):
        """Reset any global state before each test to ensure isolation."""
        # Following Category 17: Global State Management best practices
        yield
        # Cleanup after test if needed
    
    @pytest.fixture
    def base_config(self):
        """Create realistic base configuration matching source code expectations."""
        # Based on source code analysis: base_config is Dict[str, Any] with specific keys
        # CRITICAL: region must match dropdown options ['NA', 'EU', 'FE'] from source code
        return {
            'author': 'test-user',
            'bucket': 'test-bucket',
            'role': 'arn:aws:iam::123456789012:role/test-role',
            'region': 'NA',  # FIXED: Must be one of ['NA', 'EU', 'FE'] from source code
            'service_name': 'test-service',
            'pipeline_version': '1.0.0',
            'project_root_folder': 'test-project'
        }
    
    @pytest.fixture
    def mock_validation_service(self):
        """Create mock ValidationService matching source code usage."""
        mock_service = Mock()
        mock_service.build_final_config.return_value = Mock(spec=CradleDataLoadingConfig)
        return mock_service
    
    @pytest.fixture
    def mock_config_builder_service(self):
        """Create mock ConfigBuilderService matching source code usage."""
        mock_service = Mock()
        return mock_service
    
    @pytest.fixture
    def completion_callback(self):
        """Create mock completion callback for embedded mode testing."""
        return Mock()


class TestCradleNativeWidgetInitialization(TestCradleNativeWidget):
    """Test CradleNativeWidget initialization following source code analysis."""
    
    def test_initialization_default_parameters(self):
        """Test widget initialization with default parameters."""
        # Based on source code: __init__(base_config=None, embedded_mode=False, completion_callback=None)
        widget = CradleNativeWidget()
        
        # Verify initialization state from source code analysis
        assert widget.base_config == {}  # Source: self.base_config = base_config or {}
        assert widget.embedded_mode == False  # Source: default parameter
        assert widget.completion_callback is None  # Source: default parameter
        assert widget.current_step == 1  # Source: self.current_step = 1
        assert widget.total_steps == 5  # Source: self.total_steps = 5 (4 config + 1 completion)
        assert widget.steps_data == {}  # Source: self.steps_data = {}
        assert widget.completed_config is None  # Source: self.completed_config = None
        assert widget.step_widgets == {}  # Source: self.step_widgets = {}
        
        # Verify UI components are initialized but not created yet
        assert widget.main_container is None  # Source: self.main_container = None
        assert widget.wizard_content is None  # Source: self.wizard_content = None
        assert widget.navigation_container is None  # Source: self.navigation_container = None
    
    def test_initialization_with_base_config(self, base_config):
        """Test widget initialization with base configuration."""
        widget = CradleNativeWidget(base_config=base_config)
        
        # Should store the provided base_config
        assert widget.base_config == base_config
        assert widget.base_config['author'] == 'test-user'
        assert widget.base_config['bucket'] == 'test-bucket'
    
    def test_initialization_embedded_mode(self, base_config, completion_callback):
        """Test widget initialization in embedded mode."""
        widget = CradleNativeWidget(
            base_config=base_config,
            embedded_mode=True,
            completion_callback=completion_callback
        )
        
        # Verify embedded mode configuration
        assert widget.embedded_mode == True
        assert widget.completion_callback == completion_callback
        assert widget.base_config == base_config
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.ValidationService')
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.ConfigBuilderService')
    def test_service_initialization_success(self, mock_config_builder, mock_validation):
        """Test successful service initialization."""
        # Mock services to succeed
        mock_validation.return_value = Mock()
        mock_config_builder.return_value = Mock()
        
        widget = CradleNativeWidget()
        
        # Services should be initialized
        assert widget.validation_service is not None
        assert widget.config_builder is not None
        mock_validation.assert_called_once()
        mock_config_builder.assert_called_once()
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.ValidationService')
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.ConfigBuilderService')
    def test_service_initialization_failure(self, mock_config_builder, mock_validation):
        """Test graceful handling of service initialization failure."""
        # Mock services to raise exceptions
        mock_validation.side_effect = Exception("Service unavailable")
        mock_config_builder.side_effect = Exception("Service unavailable")
        
        # Should not raise exception during initialization
        widget = CradleNativeWidget()
        
        # Services should be None when initialization fails
        assert widget.validation_service is None
        assert widget.config_builder is None


class TestCradleNativeWidgetDisplay(TestCradleNativeWidget):
    """Test CradleNativeWidget display functionality."""
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_display_creates_wizard_structure(self, mock_display):
        """Test that display() creates the complete wizard structure."""
        widget = CradleNativeWidget()
        
        widget.display()
        
        # Should create main container components
        assert widget.main_container is not None
        assert widget.progress_container is not None
        assert widget.wizard_content is not None
        assert widget.navigation_container is not None
        
        # FIXED: Implementation calls display multiple times internally (HTML elements, widgets, etc.)
        # The final call should be with the main container - check that it was called with main container
        main_container_calls = [call for call in mock_display.call_args_list 
                               if call[0] and call[0][0] == widget.main_container]
        assert len(main_container_calls) == 1, "Should display main container exactly once"
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_display_initializes_step_1(self, mock_display):
        """Test that display() initializes with step 1."""
        widget = CradleNativeWidget()
        
        # Mock the step display method
        with patch.object(widget, '_show_step') as mock_show_step:
            widget.display()
            
            # Should show step 1
            mock_show_step.assert_called_once_with(1)
    
    def test_create_wizard_structure_components(self):
        """Test that _create_wizard_structure creates all required components."""
        widget = CradleNativeWidget()
        
        widget._create_wizard_structure()
        
        # Verify all components are created
        assert isinstance(widget.progress_container, widgets.HTML)
        assert isinstance(widget.wizard_content, widgets.Output)
        assert isinstance(widget.navigation_container, widgets.HTML)
        assert isinstance(widget.main_container, widgets.VBox)
        
        # Verify main container structure
        assert len(widget.main_container.children) == 5  # style, header, progress, content, navigation
        assert widget.progress_container in widget.main_container.children
        assert widget.wizard_content in widget.main_container.children
        assert widget.navigation_container in widget.main_container.children


class TestCradleNativeWidgetStepNavigation(TestCradleNativeWidget):
    """Test 4-step wizard navigation functionality."""
    
    def test_update_progress_indicator_step_states(self):
        """Test progress indicator updates correctly for different steps."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        # Test each step state
        for step in range(1, 6):  # Steps 1-5
            widget.current_step = step
            widget._update_progress_indicator()
            
            # Progress HTML should be updated
            assert widget.progress_container.value != ""
            assert "flex" in widget.progress_container.value  # Contains CSS flex layout
            
            # Should contain step indicators
            for step_num in range(1, 5):  # Only 4 steps in progress indicator
                assert str(step_num) in widget.progress_container.value
    
    def test_update_navigation_buttons(self):
        """Test navigation buttons update correctly for different steps."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        # Test step 1 - no back button
        widget.current_step = 1
        widget._update_navigation()
        
        # Should hide back button on first step
        assert "display: none;" in widget.navigation_container.value
        assert "Next" in widget.navigation_container.value
        
        # Test middle step - both buttons
        widget.current_step = 3
        widget._update_navigation()
        
        # Should show back button
        assert "display: block;" in widget.navigation_container.value or "Back" in widget.navigation_container.value
        assert "Next" in widget.navigation_container.value
        
        # Test final step - finish button
        widget.current_step = 5
        widget._update_navigation()
        
        # Should show finish button instead of next
        assert "Finish" in widget.navigation_container.value
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.clear_output')
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_show_step_calls_correct_step_method(self, mock_display, mock_clear):
        """Test that _show_step calls the correct step method."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        # Mock all step methods
        with patch.object(widget, '_show_step1_data_sources') as mock_step1, \
             patch.object(widget, '_show_step2_transform') as mock_step2, \
             patch.object(widget, '_show_step3_output') as mock_step3, \
             patch.object(widget, '_show_step4_cradle_job') as mock_step4, \
             patch.object(widget, '_show_step5_completion') as mock_step5:
            
            # Test each step
            widget._show_step(1)
            mock_step1.assert_called_once()
            
            widget._show_step(2)
            mock_step2.assert_called_once()
            
            widget._show_step(3)
            mock_step3.assert_called_once()
            
            widget._show_step(4)
            mock_step4.assert_called_once()
            
            widget._show_step(5)
            mock_step5.assert_called_once()
    
    def test_next_step_navigation(self):
        """Test next_step() method navigation logic."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        # Mock _save_current_step_data and _show_step
        with patch.object(widget, '_save_current_step_data') as mock_save, \
             patch.object(widget, '_show_step') as mock_show:
            
            # Test navigation from step 1 to 2
            widget.current_step = 1
            widget.next_step()
            
            mock_save.assert_called_once()
            mock_show.assert_called_once_with(2)
            
            # Test navigation from step 4 to 5
            mock_save.reset_mock()
            mock_show.reset_mock()
            widget.current_step = 4
            widget.next_step()
            
            mock_save.assert_called_once()
            mock_show.assert_called_once_with(5)
            
            # Test no navigation beyond final step - FIXED: Based on source code analysis
            # Source code: if self.current_step < self.total_steps: (only saves and navigates if condition met)
            mock_save.reset_mock()
            mock_show.reset_mock()
            widget.current_step = 5
            widget.next_step()
            
            # Should not save or navigate beyond step 5 (source code shows condition check first)
            mock_save.assert_not_called()  # FIXED: No save when at boundary
            mock_show.assert_not_called()  # No navigation beyond final step
    
    def test_previous_step_navigation(self):
        """Test previous_step() method navigation logic."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        # Mock _save_current_step_data and _show_step
        with patch.object(widget, '_save_current_step_data') as mock_save, \
             patch.object(widget, '_show_step') as mock_show:
            
            # Test navigation from step 2 to 1
            widget.current_step = 2
            widget.previous_step()
            
            mock_save.assert_called_once()
            mock_show.assert_called_once_with(1)
            
            # Test no navigation before first step - FIXED: Based on source code analysis
            # Source code: if self.current_step > 1: (only saves and navigates if condition met)
            mock_save.reset_mock()
            mock_show.reset_mock()
            widget.current_step = 1
            widget.previous_step()
            
            # Should not save or navigate before step 1 (source code shows condition check first)
            mock_save.assert_not_called()  # FIXED: No save when at boundary
            mock_show.assert_not_called()  # But doesn't show previous step


class TestCradleNativeWidgetStepContent(TestCradleNativeWidget):
    """Test individual step content creation and widget management."""
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_show_step1_data_sources_creates_widgets(self, mock_display):
        """Test that step 1 creates all required form widgets."""
        widget = CradleNativeWidget()
        
        widget._show_step1_data_sources()
        
        # Should create step 1 widgets
        assert 1 in widget.step_widgets
        step1_widgets = widget.step_widgets[1]
        
        # Verify all required widgets are created (based on source code analysis)
        required_widgets = [
            'author', 'bucket', 'role', 'region', 'service_name', 
            'pipeline_version', 'project_root_folder', 'start_date', 'end_date',
            'source_name', 'source_type', 'mds_service', 'mds_region',
            'add_field', 'add_button'
        ]
        
        for widget_name in required_widgets:
            assert widget_name in step1_widgets, f"Missing widget: {widget_name}"
            assert isinstance(step1_widgets[widget_name], widgets.Widget)
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_show_step1_base_config_prepopulation(self, mock_display, base_config):
        """Test that step 1 pre-populates widgets with base_config values."""
        widget = CradleNativeWidget(base_config=base_config)
        
        widget._show_step1_data_sources()
        
        # Verify widgets are pre-populated with base_config values
        step1_widgets = widget.step_widgets[1]
        assert step1_widgets['author'].value == base_config['author']
        assert step1_widgets['bucket'].value == base_config['bucket']
        assert step1_widgets['role'].value == base_config['role']
        assert step1_widgets['service_name'].value == base_config['service_name']
        assert step1_widgets['pipeline_version'].value == base_config['pipeline_version']
        assert step1_widgets['project_root_folder'].value == base_config['project_root_folder']
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_show_step2_transform_creates_widgets(self, mock_display):
        """Test that step 2 creates transform configuration widgets."""
        widget = CradleNativeWidget()
        
        widget._show_step2_transform()
        
        # Should create step 2 widgets
        assert 2 in widget.step_widgets
        step2_widgets = widget.step_widgets[2]
        
        # Verify transform widgets are created
        required_widgets = ['transform_sql', 'enable_splitting', 'days_per_split', 'merge_sql']
        for widget_name in required_widgets:
            assert widget_name in step2_widgets, f"Missing widget: {widget_name}"
            assert isinstance(step2_widgets[widget_name], widgets.Widget)
        
        # Verify default SQL is populated
        assert "SELECT" in step2_widgets['transform_sql'].value
        assert "FROM mds_source mds" in step2_widgets['transform_sql'].value
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_show_step3_output_creates_widgets(self, mock_display):
        """Test that step 3 creates output configuration widgets."""
        widget = CradleNativeWidget()
        
        widget._show_step3_output()
        
        # Should create step 3 widgets
        assert 3 in widget.step_widgets
        step3_widgets = widget.step_widgets[3]
        
        # Verify output widgets are created
        required_widgets = ['output_format', 'save_mode', 'file_count', 'keep_dots', 'include_header']
        for widget_name in required_widgets:
            assert widget_name in step3_widgets, f"Missing widget: {widget_name}"
            assert isinstance(step3_widgets[widget_name], widgets.Widget)
        
        # Verify default values
        assert step3_widgets['output_format'].value == 'PARQUET'
        assert step3_widgets['save_mode'].value == 'ERRORIFEXISTS'
        assert step3_widgets['include_header'].value == True
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_show_step4_cradle_job_creates_widgets(self, mock_display):
        """Test that step 4 creates cradle job configuration widgets."""
        widget = CradleNativeWidget()
        
        widget._show_step4_cradle_job()
        
        # Should create step 4 widgets
        assert 4 in widget.step_widgets
        step4_widgets = widget.step_widgets[4]
        
        # Verify cradle job widgets are created
        required_widgets = ['cradle_account', 'cluster_type', 'retry_count', 'extra_args']
        for widget_name in required_widgets:
            assert widget_name in step4_widgets, f"Missing widget: {widget_name}"
            assert isinstance(step4_widgets[widget_name], widgets.Widget)
        
        # Verify default values
        assert step4_widgets['cradle_account'].value == 'Buyer-Abuse-RnD-Dev'
        assert step4_widgets['cluster_type'].value == 'STANDARD'
        assert step4_widgets['retry_count'].value == 1
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_show_step5_completion_creates_widgets(self, mock_display):
        """Test that step 5 creates completion widgets."""
        widget = CradleNativeWidget()
        
        widget._show_step5_completion()
        
        # Should create step 5 widgets
        assert 5 in widget.step_widgets
        step5_widgets = widget.step_widgets[5]
        
        # Verify completion widgets are created
        assert 'job_type' in step5_widgets
        assert isinstance(step5_widgets['job_type'], widgets.RadioButtons)
        
        # Verify default job type
        assert step5_widgets['job_type'].value == 'training'
        assert 'training' in step5_widgets['job_type'].options
        assert 'validation' in step5_widgets['job_type'].options
        assert 'testing' in step5_widgets['job_type'].options
        assert 'calibration' in step5_widgets['job_type'].options


class TestCradleNativeWidgetDataCollection(TestCradleNativeWidget):
    """Test data collection and configuration creation functionality."""
    
    def test_save_current_step_data(self):
        """Test that _save_current_step_data collects widget values correctly."""
        widget = CradleNativeWidget()
        
        # Create mock widgets for step 1
        mock_widgets = {
            'author': Mock(value='test-author'),
            'bucket': Mock(value='test-bucket'),
            'region': Mock(value='us-west-2')
        }
        widget.step_widgets[1] = mock_widgets
        widget.current_step = 1
        
        widget._save_current_step_data()
        
        # Should save widget values to steps_data
        assert 1 in widget.steps_data
        step_data = widget.steps_data[1]
        assert step_data['author'] == 'test-author'
        assert step_data['bucket'] == 'test-bucket'
        assert step_data['region'] == 'us-west-2'
    
    def test_save_current_step_data_no_widgets(self):
        """Test _save_current_step_data handles missing widgets gracefully."""
        widget = CradleNativeWidget()
        widget.current_step = 1
        
        # Should not raise exception when no widgets exist
        widget._save_current_step_data()
        
        # Should not create entry in steps_data
        assert 1 not in widget.steps_data
    
    def test_create_final_config_with_validation_service(self, mock_validation_service):
        """Test _create_final_config uses ValidationService when available."""
        widget = CradleNativeWidget()
        widget.validation_service = mock_validation_service
        
        # Mock steps data
        widget.steps_data = {
            1: {'author': 'test-user', 'bucket': 'test-bucket'},
            2: {'transform_sql': 'SELECT * FROM test'},
            3: {'output_format': 'PARQUET'},
            4: {'cradle_account': 'test-account'},
            5: {'job_type': 'training'}
        }
        
        widget._create_final_config()
        
        # Should call validation service
        mock_validation_service.build_final_config.assert_called_once()
        
        # Should set completed_config
        assert widget.completed_config is not None
    
    def test_create_final_config_without_validation_service(self):
        """Test _create_final_config fallback when ValidationService unavailable."""
        widget = CradleNativeWidget()
        widget.validation_service = None
        
        # Mock steps data
        widget.steps_data = {
            1: {'author': 'test-user', 'bucket': 'test-bucket'},
            5: {'job_type': 'training'}
        }
        
        # Mock CradleDataLoadingConfig creation
        with patch('src.cursus.api.config_ui.widgets.cradle_native_widget.CradleDataLoadingConfig') as mock_config:
            mock_config.return_value = Mock()
            
            widget._create_final_config()
            
            # Should create config directly
            mock_config.assert_called_once()
            assert widget.completed_config is not None
    
    def test_create_final_config_handles_exceptions(self):
        """Test _create_final_config handles exceptions gracefully."""
        widget = CradleNativeWidget()
        widget.validation_service = Mock()
        widget.validation_service.build_final_config.side_effect = Exception("Config creation failed")
        
        # Mock steps data
        widget.steps_data = {1: {'author': 'test-user'}}
        
        # Should not raise exception
        widget._create_final_config()
        
        # Should set completed_config to None on error
        assert widget.completed_config is None
    
    def test_get_config_returns_completed_config(self):
        """Test get_config() returns the completed configuration."""
        widget = CradleNativeWidget()
        
        # Test when no config is completed
        assert widget.get_config() is None
        
        # Test when config is completed
        mock_config = Mock(spec=CradleDataLoadingConfig)
        widget.completed_config = mock_config
        
        assert widget.get_config() == mock_config


class TestCradleNativeWidgetEmbeddedMode(TestCradleNativeWidget):
    """Test embedded mode functionality and completion callbacks."""
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.clear_output')
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_finish_wizard_embedded_mode_calls_callback(self, mock_display, mock_clear, completion_callback):
        """Test that finish_wizard calls completion callback in embedded mode."""
        widget = CradleNativeWidget(embedded_mode=True, completion_callback=completion_callback)
        widget._create_wizard_structure()
        
        # Mock successful config creation
        mock_config = Mock(spec=CradleDataLoadingConfig)
        with patch.object(widget, '_create_final_config') as mock_create:
            widget.completed_config = mock_config
            
            widget.finish_wizard()
            
            # Should call completion callback with config
            completion_callback.assert_called_once_with(mock_config)
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.clear_output')
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_finish_wizard_standalone_mode_no_callback(self, mock_display, mock_clear):
        """Test that finish_wizard doesn't call callback in standalone mode."""
        widget = CradleNativeWidget(embedded_mode=False)
        widget._create_wizard_structure()
        
        # Mock successful config creation
        mock_config = Mock(spec=CradleDataLoadingConfig)
        widget.completed_config = mock_config
        
        # Should not raise exception (no callback to call)
        widget.finish_wizard()
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.clear_output')
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_finish_wizard_no_config_no_callback(self, mock_display, mock_clear, completion_callback):
        """Test that finish_wizard doesn't call callback when config creation fails."""
        widget = CradleNativeWidget(embedded_mode=True, completion_callback=completion_callback)
        widget._create_wizard_structure()
        
        # Mock failed config creation
        widget.completed_config = None
        
        widget.finish_wizard()
        
        # Should not call completion callback when no config
        completion_callback.assert_not_called()
    
    def test_set_navigation_callback_method_exists(self):
        """Test that set_navigation_callback method exists for embedded mode."""
        widget = CradleNativeWidget()
        
        # Method should exist (even if implementation is placeholder)
        assert hasattr(widget, 'set_navigation_callback')
        assert callable(widget.set_navigation_callback)
        
        # Should not raise exception when called
        mock_callback = Mock()
        widget.set_navigation_callback(mock_callback)


class TestCradleNativeWidgetErrorHandling(TestCradleNativeWidget):
    """Test error handling and edge cases."""
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.clear_output')
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_cancel_wizard_displays_cancellation_message(self, mock_display, mock_clear):
        """Test that cancel_wizard displays appropriate cancellation message."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        widget.cancel_wizard()
        
        # Should clear output and display cancellation message
        mock_clear.assert_called()
        mock_display.assert_called()
    
    def test_widget_handles_missing_step_widgets(self):
        """Test that widget handles missing step widgets gracefully."""
        widget = CradleNativeWidget()
        
        # CRITICAL FIX: Create wizard structure before testing navigation
        # This prevents AttributeError: 'CradleNativeWidget' object has no attribute 'progress_container'
        widget._create_wizard_structure()
        
        # Test navigation with no step widgets
        widget.current_step = 1
        
        # Should not raise exception
        widget.next_step()
        widget.previous_step()
    
    def test_widget_handles_invalid_step_numbers(self):
        """Test that widget handles invalid step numbers gracefully."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        # Mock step methods to track calls
        with patch.object(widget, '_show_step1_data_sources') as mock_step1, \
             patch.object(widget, '_show_step2_transform') as mock_step2, \
             patch.object(widget, '_show_step3_output') as mock_step3, \
             patch.object(widget, '_show_step4_cradle_job') as mock_step4, \
             patch.object(widget, '_show_step5_completion') as mock_step5:
            
            # Test invalid step numbers
            widget._show_step(0)  # Below range
            widget._show_step(6)  # Above range
            widget._show_step(-1)  # Negative
            
            # Should not call any step methods for invalid numbers
            mock_step1.assert_not_called()
            mock_step2.assert_not_called()
            mock_step3.assert_not_called()
            mock_step4.assert_not_called()
            mock_step5.assert_not_called()


class TestCradleNativeWidgetIntegration(TestCradleNativeWidget):
    """Test integration scenarios and end-to-end workflows."""
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.display')
    def test_complete_wizard_workflow_simulation(self, mock_display, base_config, completion_callback):
        """Test complete wizard workflow from start to finish."""
        widget = CradleNativeWidget(
            base_config=base_config,
            embedded_mode=True,
            completion_callback=completion_callback
        )
        
        # Step 1: Display widget
        widget.display()
        assert widget.main_container is not None
        assert widget.current_step == 1
        
        # Step 2: Navigate through all steps
        with patch.object(widget, '_save_current_step_data') as mock_save:
            # Navigate to each step
            for step in range(2, 6):
                widget.next_step()
                assert widget.current_step == step
                
            # Should save data at each step
            assert mock_save.call_count == 4  # Called for steps 1-4
        
        # Step 3: Complete wizard
        mock_config = Mock(spec=CradleDataLoadingConfig)
        with patch.object(widget, '_create_final_config') as mock_create:
            widget.completed_config = mock_config
            widget.finish_wizard()
            
            # Should create config and call callback
            mock_create.assert_called_once()
            completion_callback.assert_called_once_with(mock_config)
    
    def test_widget_state_consistency_during_navigation(self):
        """Test that widget state remains consistent during navigation."""
        widget = CradleNativeWidget()
        widget._create_wizard_structure()
        
        # Navigate through steps and verify state consistency
        for step in range(1, 6):
            widget.current_step = step
            widget._update_progress_indicator()
            widget._update_navigation()
            
            # State should be consistent
            assert widget.current_step == step
            assert widget.total_steps == 5
            assert widget.progress_container.value != ""
            assert widget.navigation_container.value != ""
    
    @patch('src.cursus.api.config_ui.widgets.cradle_native_widget.logger')
    def test_logging_integration(self, mock_logger):
        """Test that widget logs important events correctly."""
        widget = CradleNativeWidget(embedded_mode=True)
        
        # Should log initialization
        mock_logger.info.assert_called_with("CradleNativeWidget initialized, embedded_mode=True")
        
        # Test error logging during config creation
        widget.validation_service = Mock()
        widget.validation_service.build_final_config.side_effect = Exception("Test error")
        widget.steps_data = {1: {'author': 'test'}}
        
        widget._create_final_config()
        
        # Should log error
        mock_logger.error.assert_called_with("Error creating final config: Test error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
