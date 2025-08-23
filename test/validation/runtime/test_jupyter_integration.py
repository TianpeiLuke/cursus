"""
Tests for Jupyter Integration Components

This module provides comprehensive tests for the Jupyter notebook integration
components including NotebookInterface, VisualizationReporter, InteractiveDebugger,
NotebookTemplateManager, and AdvancedNotebookFeatures.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
from datetime import datetime
import tempfile
import shutil

# Import components to test
from src.cursus.validation.runtime.jupyter import (
    NotebookInterface,
    NotebookSession,
    VisualizationReporter,
    InteractiveDebugger,
    NotebookTemplateManager,
    AdvancedNotebookFeatures
)

from src.cursus.validation.runtime.jupyter.visualization import (
    VisualizationConfig,
    TestResultMetrics
)

from src.cursus.validation.runtime.jupyter.debugger import (
    DebugSession,
    BreakpointManager
)

from src.cursus.validation.runtime.jupyter.templates import (
    NotebookTemplate
)

from src.cursus.validation.runtime.jupyter.advanced import (
    CollaborationManager,
    AutomatedReportGenerator,
    PerformanceMonitor
)


class TestNotebookInterface:
    """Test cases for NotebookInterface"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.notebook_interface = NotebookInterface()
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_notebook_interface_initialization(self):
        """Test NotebookInterface initialization"""
        assert self.notebook_interface is not None
        assert hasattr(self.notebook_interface, 'session')
        assert hasattr(self.notebook_interface, 'executor')
    
    def test_notebook_session_creation(self):
        """Test NotebookSession model"""
        session = NotebookSession(
            session_id="test_session",
            pipeline_name="test_pipeline",
            workspace_path=self.temp_dir
        )
        
        assert session.session_id == "test_session"
        assert session.pipeline_name == "test_pipeline"
        assert session.workspace_path == self.temp_dir
        assert isinstance(session.created_at, datetime)
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.JUPYTER_AVAILABLE', True)
    def test_display_welcome_with_jupyter(self):
        """Test display_welcome when Jupyter is available"""
        # This would test the actual widget creation if Jupyter was available
        # For now, we test that the method exists and can be called
        assert hasattr(self.notebook_interface, 'display_welcome')
    
    def test_load_pipeline_config(self):
        """Test loading pipeline configuration"""
        # Create a mock pipeline config
        config_path = self.temp_dir / "pipeline.yaml"
        config_content = """
        pipeline:
          name: test_pipeline
          steps:
            - name: step1
              type: processing
        """
        config_path.write_text(config_content)
        
        # Test loading (would need actual implementation)
        assert hasattr(self.notebook_interface, 'load_pipeline')


class TestVisualizationReporter:
    """Test cases for VisualizationReporter"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.config = VisualizationConfig()
        self.reporter = VisualizationReporter(self.config)
    
    def test_visualization_config(self):
        """Test VisualizationConfig model"""
        assert self.config.theme == "plotly_white"
        assert self.config.figure_width == 1000
        assert self.config.figure_height == 600
        assert len(self.config.color_palette) == 5
    
    def test_test_result_metrics(self):
        """Test TestResultMetrics model"""
        metrics = TestResultMetrics(
            step_name="test_step",
            execution_time=1.5,
            success=True,
            timestamp=datetime.now(),
            test_type="synthetic"
        )
        
        assert metrics.step_name == "test_step"
        assert metrics.execution_time == 1.5
        assert metrics.success is True
        assert metrics.test_type == "synthetic"
    
    def test_add_test_result(self):
        """Test adding test results"""
        metrics = TestResultMetrics(
            step_name="test_step",
            execution_time=1.5,
            success=True,
            timestamp=datetime.now(),
            test_type="synthetic"
        )
        
        self.reporter.add_test_result(metrics)
        assert len(self.reporter.test_results) == 1
        assert self.reporter.test_results[0] == metrics
    
    def test_add_performance_data(self):
        """Test adding performance data"""
        performance_metrics = {
            'cpu_usage': 45.2,
            'memory_usage': 1024.5,
            'throughput': 150.0
        }
        
        self.reporter.add_performance_data("test_step", performance_metrics)
        assert "test_step" in self.reporter.performance_data
        assert self.reporter.performance_data["test_step"] == performance_metrics
    
    def test_display_summary(self):
        """Test display summary functionality"""
        # Add some test data
        metrics1 = TestResultMetrics(
            step_name="step1",
            execution_time=1.5,
            success=True,
            timestamp=datetime.now(),
            test_type="synthetic"
        )
        
        metrics2 = TestResultMetrics(
            step_name="step2",
            execution_time=2.0,
            success=False,
            timestamp=datetime.now(),
            test_type="real"
        )
        
        self.reporter.add_test_result(metrics1)
        self.reporter.add_test_result(metrics2)
        
        # Test that summary can be displayed (captures print output)
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        self.reporter.display_summary()
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        assert "Total Tests: 2" in output
        assert "Successful Tests: 1" in output
        assert "Success Rate: 50.0%" in output


class TestInteractiveDebugger:
    """Test cases for InteractiveDebugger"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.debugger = InteractiveDebugger()
        self.breakpoint_manager = BreakpointManager()
    
    def test_debug_session_creation(self):
        """Test DebugSession model"""
        session = DebugSession(
            session_id="debug_test",
            pipeline_name="test_pipeline"
        )
        
        assert session.session_id == "debug_test"
        assert session.pipeline_name == "test_pipeline"
        assert isinstance(session.created_at, datetime)
        assert len(session.breakpoints) == 0
    
    def test_breakpoint_manager(self):
        """Test BreakpointManager functionality"""
        # Add a breakpoint
        self.breakpoint_manager.add_breakpoint("step1", "x > 10")
        
        assert "step1" in self.breakpoint_manager.active_breakpoints
        assert len(self.breakpoint_manager.breakpoints) == 1
        
        # Test breakpoint condition
        context = {"x": 15}
        assert self.breakpoint_manager.should_break("step1", context) is True
        
        context = {"x": 5}
        assert self.breakpoint_manager.should_break("step1", context) is False
    
    def test_start_debug_session(self):
        """Test starting a debug session"""
        session_id = self.debugger.start_debug_session("test_pipeline")
        
        assert session_id is not None
        assert self.debugger.session is not None
        assert self.debugger.session.pipeline_name == "test_pipeline"
    
    def test_set_breakpoint(self):
        """Test setting breakpoints"""
        self.debugger.set_breakpoint("step1", "condition")
        
        breakpoints = self.debugger.breakpoint_manager.list_breakpoints()
        assert len(breakpoints) == 1
        assert breakpoints[0]['step_name'] == "step1"
        assert breakpoints[0]['condition'] == "condition"
    
    def test_error_analysis(self):
        """Test error analysis functionality"""
        error = ValueError("Test error")
        context = {"var1": "value1", "var2": 42}
        
        error_info = self.debugger.analyze_error(error, context)
        
        assert error_info['error_type'] == 'ValueError'
        assert error_info['error_message'] == 'Test error'
        assert 'traceback' in error_info
        assert error_info['context_variables'] == ['var1', 'var2']
        assert len(error_info['suggestions']) > 0


class TestNotebookTemplateManager:
    """Test cases for NotebookTemplateManager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.template_manager = NotebookTemplateManager(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_notebook_template_model(self):
        """Test NotebookTemplate model"""
        template = NotebookTemplate(
            name="test_template",
            description="Test template",
            category="testing",
            cell_templates=[
                {"cell_type": "markdown", "source": "# Test"},
                {"cell_type": "code", "source": "print('hello')"}
            ]
        )
        
        assert template.name == "test_template"
        assert template.description == "Test template"
        assert template.category == "testing"
        assert len(template.cell_templates) == 2
    
    def test_template_registration(self):
        """Test template registration"""
        template = NotebookTemplate(
            name="custom_template",
            description="Custom template",
            category="custom"
        )
        
        self.template_manager.register_template(template)
        
        assert "custom_template" in self.template_manager.templates
        retrieved = self.template_manager.get_template("custom_template")
        assert retrieved == template
    
    def test_list_templates(self):
        """Test listing templates"""
        templates = self.template_manager.list_templates()
        
        # Should have built-in templates
        assert len(templates) > 0
        
        # Check template structure
        for template_info in templates:
            assert 'name' in template_info
            assert 'description' in template_info
            assert 'category' in template_info
            assert 'created_at' in template_info
    
    def test_create_custom_template(self):
        """Test creating custom templates"""
        cell_definitions = [
            {"cell_type": "markdown", "source": "# Custom Template"},
            {"cell_type": "code", "source": "# Custom code"}
        ]
        
        template = self.template_manager.create_custom_template(
            name="custom_test",
            description="Custom test template",
            category="test",
            cell_definitions=cell_definitions
        )
        
        assert template.name == "custom_test"
        assert len(template.cell_templates) == 2
        assert "custom_test" in self.template_manager.templates
    
    def test_template_categories(self):
        """Test template categorization"""
        categories = self.template_manager.get_template_categories()
        
        assert isinstance(categories, list)
        assert len(categories) > 0
        
        # Test getting templates by category
        for category in categories:
            templates = self.template_manager.get_templates_by_category(category)
            assert isinstance(templates, list)
    
    def test_save_and_load_template(self):
        """Test saving and loading templates"""
        # Create a custom template
        template = NotebookTemplate(
            name="save_test",
            description="Save test template",
            category="test"
        )
        
        self.template_manager.register_template(template)
        
        # Save to file
        file_path = self.temp_dir / "save_test.json"
        self.template_manager.save_template_to_file("save_test", file_path)
        
        assert file_path.exists()
        
        # Create new manager and load template
        new_manager = NotebookTemplateManager(self.temp_dir)
        loaded_template = new_manager.load_template_from_file(file_path)
        
        assert loaded_template is not None
        assert loaded_template.name == "save_test"


class TestAdvancedNotebookFeatures:
    """Test cases for AdvancedNotebookFeatures"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.advanced_features = AdvancedNotebookFeatures()
        self.collaboration_manager = CollaborationManager()
        self.report_generator = AutomatedReportGenerator()
        self.performance_monitor = PerformanceMonitor()
    
    def test_collaboration_manager(self):
        """Test CollaborationManager functionality"""
        # Create a session
        session_id = self.collaboration_manager.create_session("test_pipeline", "user1")
        
        assert session_id in self.collaboration_manager.active_sessions
        session = self.collaboration_manager.active_sessions[session_id]
        assert session.pipeline_name == "test_pipeline"
        assert session.user_id == "user1"
        
        # Add bookmark
        self.collaboration_manager.add_bookmark(session_id, "bookmark1", 5, "Test bookmark")
        assert len(session.bookmarks) == 1
        assert session.bookmarks[0]['name'] == "bookmark1"
        
        # Add annotation
        self.collaboration_manager.add_annotation(session_id, 3, "Test annotation", "note")
        assert len(session.annotations) == 1
        assert session.annotations[0]['annotation'] == "Test annotation"
    
    def test_automated_report_generator(self):
        """Test AutomatedReportGenerator functionality"""
        # Test report templates
        templates = self.report_generator.report_templates
        assert 'executive_summary' in templates
        assert 'technical_report' in templates
        assert 'data_quality_report' in templates
        
        # Test report generation
        execution_data = {
            'pipeline_name': 'Test Pipeline',
            'total_steps': 5,
            'total_execution_time': 120.5,
            'execution_date': '2023-01-01',
            'total_tests': 10,
            'successful_tests': 8,
            'metrics': {
                'avg_response_time': 2.3,
                'throughput': 150.2
            }
        }
        
        report = self.report_generator.generate_report('executive_summary', execution_data)
        
        assert isinstance(report, str)
        assert 'Executive Summary' in report
        assert 'Test Pipeline' in report
        assert 'Success Rate: 80.0%' in report
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor functionality"""
        # Get current metrics
        metrics = self.performance_monitor.get_current_metrics()
        
        assert isinstance(metrics, dict)
        assert 'CPU Usage' in metrics
        assert 'Memory Usage' in metrics
        
        # Add alert
        self.performance_monitor.add_alert('memory', 'High memory usage', 'warning')
        
        assert len(self.performance_monitor.alerts) == 1
        alert = self.performance_monitor.alerts[0]
        assert alert['type'] == 'memory'
        assert alert['severity'] == 'warning'
    
    def test_auto_save_functionality(self):
        """Test auto-save functionality"""
        # Test enabling auto-save
        self.advanced_features.enable_auto_save(600)
        assert self.advanced_features.auto_save_enabled is True
        assert self.advanced_features.auto_save_interval == 600
        
        # Test disabling auto-save
        self.advanced_features.disable_auto_save()
        assert self.advanced_features.auto_save_enabled is False


class TestJupyterIntegration:
    """Integration tests for Jupyter components"""
    
    def test_module_imports(self):
        """Test that all components can be imported"""
        from src.cursus.validation.runtime.jupyter import (
            NotebookInterface,
            NotebookSession,
            VisualizationReporter,
            InteractiveDebugger,
            NotebookTemplateManager,
            AdvancedNotebookFeatures
        )
        
        # Test that classes can be instantiated
        notebook = NotebookInterface()
        reporter = VisualizationReporter()
        debugger = InteractiveDebugger()
        template_manager = NotebookTemplateManager()
        advanced = AdvancedNotebookFeatures()
        
        assert notebook is not None
        assert reporter is not None
        assert debugger is not None
        assert template_manager is not None
        assert advanced is not None
    
    def test_component_integration(self):
        """Test integration between components"""
        # Create components
        notebook = NotebookInterface()
        reporter = VisualizationReporter()
        debugger = InteractiveDebugger()
        
        # Test that they can work together
        # (This would be more comprehensive with actual Jupyter environment)
        assert hasattr(notebook, 'session')
        assert hasattr(reporter, 'test_results')
        assert hasattr(debugger, 'breakpoint_manager')
    
    @patch('src.cursus.validation.runtime.jupyter.notebook_interface.JUPYTER_AVAILABLE', False)
    def test_graceful_fallback_without_jupyter(self):
        """Test graceful fallback when Jupyter is not available"""
        # Components should still be importable and instantiable
        notebook = NotebookInterface()
        reporter = VisualizationReporter()
        debugger = InteractiveDebugger()
        
        # They should handle the lack of Jupyter gracefully
        assert notebook is not None
        assert reporter is not None
        assert debugger is not None


if __name__ == '__main__':
    pytest.main([__file__])
