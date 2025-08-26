"""
Unit tests for advanced.py module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime, timedelta

import pytest

# Import the module under test
from src.cursus.validation.runtime.jupyter.advanced import (
    AdvancedNotebookFeatures,
    NotebookSession,
    CollaborationManager,
    AutomatedReportGenerator,
    PerformanceMonitor,
    AdvancedWidgetFactory,
    JUPYTER_AVAILABLE
)


class TestNotebookSession(unittest.TestCase):
    """Test cases for NotebookSession model"""
    
    def test_notebook_session_creation(self):
        """Test NotebookSession creation with required fields"""
        workspace_path = Path("/tmp/workspace")
        session = NotebookSession(
            session_id="session_123",
            pipeline_name="test_pipeline",
            workspace_path=workspace_path
        )
        
        self.assertEqual(session.session_id, "session_123")
        self.assertEqual(session.pipeline_name, "test_pipeline")
        self.assertEqual(session.workspace_path, workspace_path)
        self.assertIsNone(session.user_id)
        self.assertIsInstance(session.created_at, datetime)
        self.assertIsInstance(session.last_activity, datetime)
        self.assertEqual(session.session_data, {})
        self.assertEqual(session.bookmarks, [])
        self.assertEqual(session.annotations, [])
    
    def test_notebook_session_with_optional_fields(self):
        """Test NotebookSession creation with optional fields"""
        workspace_path = Path("/tmp/workspace")
        session_data = {"key": "value"}
        bookmarks = [{"name": "bookmark1", "cell_index": 0}]
        annotations = [{"cell_index": 1, "annotation": "note"}]
        
        session = NotebookSession(
            session_id="session_123",
            user_id="user_456",
            pipeline_name="test_pipeline",
            workspace_path=workspace_path,
            session_data=session_data,
            bookmarks=bookmarks,
            annotations=annotations
        )
        
        self.assertEqual(session.user_id, "user_456")
        self.assertEqual(session.session_data, session_data)
        self.assertEqual(session.bookmarks, bookmarks)
        self.assertEqual(session.annotations, annotations)


class TestCollaborationManager(unittest.TestCase):
    """Test cases for CollaborationManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collab_manager = CollaborationManager()
    
    def test_initialization(self):
        """Test CollaborationManager initialization"""
        self.assertIsInstance(self.collab_manager.active_sessions, dict)
        self.assertIsInstance(self.collab_manager.shared_workspaces, dict)
        self.assertEqual(len(self.collab_manager.active_sessions), 0)
        self.assertEqual(len(self.collab_manager.shared_workspaces), 0)
    
    def test_create_session(self):
        """Test creating a new session"""
        session_id = self.collab_manager.create_session("test_pipeline")
        
        self.assertIn(session_id, self.collab_manager.active_sessions)
        self.assertTrue(session_id.startswith("session_"))
        
        session = self.collab_manager.active_sessions[session_id]
        self.assertEqual(session.pipeline_name, "test_pipeline")
        self.assertIsNone(session.user_id)
    
    def test_create_session_with_user_id(self):
        """Test creating a session with user ID"""
        session_id = self.collab_manager.create_session("test_pipeline", "user_123")
        
        session = self.collab_manager.active_sessions[session_id]
        self.assertEqual(session.user_id, "user_123")
    
    def test_add_bookmark(self):
        """Test adding a bookmark to a session"""
        session_id = self.collab_manager.create_session("test_pipeline", "user_123")
        
        self.collab_manager.add_bookmark(session_id, "test_bookmark", 5, "Test description")
        
        session = self.collab_manager.active_sessions[session_id]
        self.assertEqual(len(session.bookmarks), 1)
        
        bookmark = session.bookmarks[0]
        self.assertEqual(bookmark["name"], "test_bookmark")
        self.assertEqual(bookmark["cell_index"], 5)
        self.assertEqual(bookmark["description"], "Test description")
        self.assertEqual(bookmark["user_id"], "user_123")
        self.assertIn("created_at", bookmark)
    
    def test_add_bookmark_nonexistent_session(self):
        """Test adding bookmark to non-existent session"""
        # Should not raise an exception
        self.collab_manager.add_bookmark("nonexistent", "bookmark", 0, "desc")
        
        # No sessions should be created
        self.assertEqual(len(self.collab_manager.active_sessions), 0)
    
    def test_add_annotation(self):
        """Test adding an annotation to a session"""
        session_id = self.collab_manager.create_session("test_pipeline", "user_123")
        
        self.collab_manager.add_annotation(session_id, 3, "This is a note", "note")
        
        session = self.collab_manager.active_sessions[session_id]
        self.assertEqual(len(session.annotations), 1)
        
        annotation = session.annotations[0]
        self.assertEqual(annotation["cell_index"], 3)
        self.assertEqual(annotation["annotation"], "This is a note")
        self.assertEqual(annotation["type"], "note")
        self.assertEqual(annotation["user_id"], "user_123")
        self.assertIn("created_at", annotation)
    
    def test_add_annotation_nonexistent_session(self):
        """Test adding annotation to non-existent session"""
        # Should not raise an exception
        self.collab_manager.add_annotation("nonexistent", 0, "annotation", "note")
        
        # No sessions should be created
        self.assertEqual(len(self.collab_manager.active_sessions), 0)


class TestAutomatedReportGenerator(unittest.TestCase):
    """Test cases for AutomatedReportGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.report_generator = AutomatedReportGenerator()
    
    def test_initialization(self):
        """Test AutomatedReportGenerator initialization"""
        self.assertIsInstance(self.report_generator.report_templates, dict)
        self.assertIsInstance(self.report_generator.execution_results, list)
        
        # Should have default templates
        self.assertIn("executive_summary", self.report_generator.report_templates)
        self.assertIn("technical_report", self.report_generator.report_templates)
        self.assertIn("data_quality_report", self.report_generator.report_templates)
    
    def test_generate_report_executive_summary(self):
        """Test generating executive summary report"""
        execution_data = {
            "pipeline_name": "test_pipeline",
            "total_steps": 5,
            "total_execution_time": 120.5,
            "execution_date": "2023-01-01",
            "total_tests": 10,
            "successful_tests": 8,
            "metrics": {
                "avg_response_time": 2.3,
                "throughput": 150.2,
                "error_rate": 0.02
            }
        }
        
        report = self.report_generator.generate_report("executive_summary", execution_data)
        
        self.assertIsInstance(report, str)
        self.assertIn("Executive Summary", report)
        self.assertIn("test_pipeline", report)
        self.assertIn("5", report)  # total_steps
        self.assertIn("120.50", report)  # execution_time
        self.assertIn("80.0%", report)  # success rate
    
    def test_generate_report_technical_report(self):
        """Test generating technical report"""
        execution_data = {
            "pipeline_name": "test_pipeline",
            "total_steps": 3,
            "performance_metrics": {
                "avg_execution_time": 45.2,
                "memory_usage": 512.0,
                "throughput": 100.5
            },
            "errors": [
                {"type": "ValueError", "message": "Invalid input", "step": "step1"},
                {"type": "KeyError", "message": "Missing key", "step": "step2"}
            ],
            "data_quality": {
                "completeness": 0.95,
                "validity": 0.98,
                "consistency": 0.92
            }
        }
        
        report = self.report_generator.generate_report("technical_report", execution_data)
        
        self.assertIsInstance(report, str)
        self.assertIn("Technical Analysis Report", report)
        self.assertIn("45.20s", report)  # avg_execution_time
        self.assertIn("512.00 MB", report)  # memory_usage
        self.assertIn("2", report)  # total errors
        self.assertIn("ValueError", report)
        self.assertIn("95.0%", report)  # completeness
    
    def test_generate_report_data_quality(self):
        """Test generating data quality report"""
        execution_data = {
            "data_quality": {
                "completeness": 0.88,
                "validity": 0.94,
                "consistency": 0.96
            }
        }

        report = self.report_generator.generate_report("data_quality_report", execution_data)

        self.assertIsInstance(report, str)
        self.assertIn("Data Quality Assessment", report)
        # The template is basic and doesn't include actual percentages yet
        self.assertIn("Data Completeness", report)
        self.assertIn("Data Validity", report)
        self.assertIn("Schema Compliance", report)
    
    def test_generate_report_nonexistent_template(self):
        """Test generating report with non-existent template"""
        with self.assertRaises(ValueError) as context:
            self.report_generator.generate_report("nonexistent_template", {})
        
        self.assertIn("not found", str(context.exception))
    
    def test_generate_section_pipeline_overview(self):
        """Test generating pipeline overview section"""
        data = {
            "pipeline_name": "test_pipeline",
            "total_steps": 5,
            "total_execution_time": 120.5,
            "execution_date": "2023-01-01"
        }
        
        section = self.report_generator._generate_section("pipeline_overview", data)
        
        self.assertIn("Pipeline Overview", section)
        self.assertIn("test_pipeline", section)
        self.assertIn("5", section)
        self.assertIn("120.50", section)
        self.assertIn("2023-01-01", section)
    
    def test_generate_section_key_metrics(self):
        """Test generating key metrics section"""
        data = {
            "metrics": {
                "avg_response_time": 2.3,
                "throughput": 150.2,
                "error_rate": 0.02
            }
        }
        
        section = self.report_generator._generate_section("key_metrics", data)
        
        self.assertIn("Key Metrics", section)
        self.assertIn("Avg Response Time", section)
        self.assertIn("2.3", section)
        self.assertIn("Throughput", section)
        self.assertIn("150.2", section)
        self.assertIn("Error Rate", section)
        self.assertIn("0.02", section)
    
    def test_generate_section_success_rate(self):
        """Test generating success rate section"""
        data = {
            "total_tests": 10,
            "successful_tests": 8
        }
        
        section = self.report_generator._generate_section("success_rate", data)
        
        self.assertIn("Success Rate", section)
        self.assertIn("10", section)  # total_tests
        self.assertIn("8", section)   # successful_tests
        self.assertIn("80.0%", section)  # success rate
    
    def test_generate_section_error_analysis_no_errors(self):
        """Test generating error analysis section with no errors"""
        data = {"errors": []}
        
        section = self.report_generator._generate_section("error_analysis", data)
        
        self.assertIn("Error Analysis", section)
        self.assertIn("No errors detected", section)
    
    def test_generate_section_error_analysis_with_errors(self):
        """Test generating error analysis section with errors"""
        data = {
            "errors": [
                {"type": "ValueError", "message": "Invalid input", "step": "step1"},
                {"type": "KeyError", "message": "Missing key", "step": "step2"},
                {"type": "TypeError", "message": "Type mismatch", "step": "step3"}
            ]
        }
        
        section = self.report_generator._generate_section("error_analysis", data)
        
        self.assertIn("Error Analysis", section)
        self.assertIn("3", section)  # total errors
        self.assertIn("ValueError", section)
        self.assertIn("Invalid input", section)
        self.assertIn("step1", section)


class TestPerformanceMonitor(unittest.TestCase):
    """Test cases for PerformanceMonitor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = PerformanceMonitor()
    
    def test_initialization(self):
        """Test PerformanceMonitor initialization"""
        self.assertIsInstance(self.monitor.metrics, dict)
        self.assertIsInstance(self.monitor.alerts, list)
    
    def test_get_current_metrics(self):
        """Test getting current performance metrics"""
        metrics = self.monitor.get_current_metrics()
        
        self.assertIsInstance(metrics, dict)
        expected_keys = [
            'CPU Usage', 'Memory Usage', 'Disk I/O', 
            'Network I/O', 'Active Processes', 'Uptime'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
    
    def test_add_alert(self):
        """Test adding a performance alert"""
        self.monitor.add_alert("memory", "High memory usage detected", "warning")
        
        self.assertEqual(len(self.monitor.alerts), 1)
        
        alert = self.monitor.alerts[0]
        self.assertEqual(alert["type"], "memory")
        self.assertEqual(alert["message"], "High memory usage detected")
        self.assertEqual(alert["severity"], "warning")
        self.assertIn("timestamp", alert)
    
    def test_add_alert_default_severity(self):
        """Test adding alert with default severity"""
        self.monitor.add_alert("cpu", "CPU spike detected")
        
        alert = self.monitor.alerts[0]
        self.assertEqual(alert["severity"], "info")


class TestAdvancedWidgetFactory(unittest.TestCase):
    """Test cases for AdvancedWidgetFactory class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = AdvancedWidgetFactory()
    
    def test_initialization(self):
        """Test AdvancedWidgetFactory initialization"""
        self.assertIsNotNone(self.factory)
    
    def test_create_progress_tracker_without_jupyter(self):
        """Test creating progress tracker when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.JUPYTER_AVAILABLE', False):
            factory = AdvancedWidgetFactory()
            result = factory.create_progress_tracker(5)
            self.assertIsNone(result)
    
    def test_create_interactive_filter_without_jupyter(self):
        """Test creating interactive filter when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.JUPYTER_AVAILABLE', False):
            factory = AdvancedWidgetFactory()
            result = factory.create_interactive_filter(["col1", "col2"])
            self.assertIsNone(result)


@unittest.skipIf(not JUPYTER_AVAILABLE, "Jupyter dependencies not available")
class TestAdvancedWidgetFactoryWithJupyter(unittest.TestCase):
    """Test cases that require Jupyter dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.factory = AdvancedWidgetFactory()
    
    @patch('src.cursus.validation.runtime.jupyter.advanced.widgets')
    def test_create_progress_tracker(self, mock_widgets):
        """Test creating progress tracker widget"""
        # Mock widgets
        mock_progress = Mock()
        mock_label = Mock()
        mock_html = Mock()
        mock_vbox = Mock()
        
        mock_widgets.IntProgress.return_value = mock_progress
        mock_widgets.Label.return_value = mock_label
        mock_widgets.HTML.return_value = mock_html
        mock_widgets.VBox.return_value = mock_vbox
        
        result = self.factory.create_progress_tracker(10)
        
        self.assertIsNotNone(result)
        mock_widgets.IntProgress.assert_called_once()
        mock_widgets.Label.assert_called_once()
        mock_widgets.HTML.assert_called_once()
        mock_widgets.VBox.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.advanced.widgets')
    def test_create_interactive_filter(self, mock_widgets):
        """Test creating interactive filter widget"""
        # Mock widgets
        mock_select = Mock()
        mock_text = Mock()
        mock_button = Mock()
        mock_vbox = Mock()
        
        mock_widgets.SelectMultiple.return_value = mock_select
        mock_widgets.Text.return_value = mock_text
        mock_widgets.Button.return_value = mock_button
        mock_widgets.VBox.return_value = mock_vbox
        
        columns = ["col1", "col2", "col3"]
        result = self.factory.create_interactive_filter(columns)
        
        self.assertIsNotNone(result)
        mock_widgets.SelectMultiple.assert_called_once()
        mock_widgets.Text.assert_called_once()
        mock_widgets.Button.assert_called_once()
        mock_widgets.VBox.assert_called_once()


class TestAdvancedNotebookFeatures(unittest.TestCase):
    """Test cases for AdvancedNotebookFeatures class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.features = AdvancedNotebookFeatures()
    
    def test_initialization(self):
        """Test AdvancedNotebookFeatures initialization"""
        self.assertIsInstance(self.features.collaboration_manager, CollaborationManager)
        self.assertIsInstance(self.features.report_generator, AutomatedReportGenerator)
        self.assertIsInstance(self.features.performance_monitor, PerformanceMonitor)
        self.assertIsInstance(self.features.widget_factory, AdvancedWidgetFactory)
        
        self.assertIsNone(self.features.current_session)
        self.assertTrue(self.features.auto_save_enabled)
        self.assertEqual(self.features.auto_save_interval, 300)
    
    def test_initialization_without_jupyter(self):
        """Test initialization when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.JUPYTER_AVAILABLE', False):
            features = AdvancedNotebookFeatures()
            self.assertIsNotNone(features)
    
    def test_enable_auto_save(self):
        """Test enabling auto-save"""
        with patch('builtins.print') as mock_print:
            self.features.enable_auto_save(600)
        
        self.assertTrue(self.features.auto_save_enabled)
        self.assertEqual(self.features.auto_save_interval, 600)
        mock_print.assert_called_with("Auto-save enabled with 600s interval")
    
    def test_disable_auto_save(self):
        """Test disabling auto-save"""
        with patch('builtins.print') as mock_print:
            self.features.disable_auto_save()
        
        self.assertFalse(self.features.auto_save_enabled)
        mock_print.assert_called_with("Auto-save disabled")
    
    def test_export_session_data(self):
        """Test exporting session data"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a session
            session_id = self.features.collaboration_manager.create_session("test_pipeline")
            self.features.collaboration_manager.add_bookmark(session_id, "bookmark1", 0, "desc")
            self.features.collaboration_manager.add_annotation(session_id, 1, "annotation1", "note")
            
            output_path = Path(temp_dir) / "session_data.json"
            
            with patch('builtins.print') as mock_print:
                self.features.export_session_data(session_id, output_path)
            
            self.assertTrue(output_path.exists())
            
            # Verify exported data
            with open(output_path) as f:
                data = json.load(f)
            
            self.assertIn("session_info", data)
            self.assertIn("bookmarks", data)
            self.assertIn("annotations", data)
            self.assertIn("exported_at", data)
            
            self.assertEqual(len(data["bookmarks"]), 1)
            self.assertEqual(len(data["annotations"]), 1)
            
            mock_print.assert_called_with(f"Session data exported to: {output_path}")
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_export_session_data_nonexistent_session(self):
        """Test exporting data for non-existent session"""
        temp_dir = tempfile.mkdtemp()
        try:
            output_path = Path(temp_dir) / "session_data.json"
            
            with patch('builtins.print') as mock_print:
                self.features.export_session_data("nonexistent", output_path)
            
            self.assertFalse(output_path.exists())
            mock_print.assert_called_with("Session 'nonexistent' not found")
            
        finally:
            shutil.rmtree(temp_dir)


@unittest.skipIf(not JUPYTER_AVAILABLE, "Jupyter dependencies not available")
class TestAdvancedNotebookFeaturesWithJupyter(unittest.TestCase):
    """Test cases that require Jupyter dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.features = AdvancedNotebookFeatures()
    
    @patch('src.cursus.validation.runtime.jupyter.advanced.widgets')
    def test_create_advanced_dashboard(self, mock_widgets):
        """Test creating advanced dashboard"""
        # Mock widgets
        mock_tab = Mock()
        mock_vbox = Mock()
        mock_hbox = Mock()
        mock_button = Mock()
        mock_text = Mock()
        mock_dropdown = Mock()
        mock_output = Mock()
        mock_label = Mock()
        mock_checkbox = Mock()
        mock_slider = Mock()
        
        mock_widgets.Tab.return_value = mock_tab
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HBox.return_value = mock_hbox
        mock_widgets.Button.return_value = mock_button
        mock_widgets.Text.return_value = mock_text
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Output.return_value = mock_output
        mock_widgets.Label.return_value = mock_label
        mock_widgets.Checkbox.return_value = mock_checkbox
        mock_widgets.IntSlider.return_value = mock_slider
        
        result = self.features.create_advanced_dashboard()
        
        self.assertIsNotNone(result)
        mock_widgets.Tab.assert_called_once()
        # Should create multiple tabs
        self.assertGreater(mock_widgets.VBox.call_count, 0)
    
    def test_create_session_management_tab(self):
        """Test creating session management tab"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.widgets') as mock_widgets:
            # Mock all required widgets
            mock_widgets.HTML.return_value = Mock()
            mock_widgets.Label.return_value = Mock()
            mock_widgets.Text.return_value = Mock()
            mock_widgets.Button.return_value = Mock()
            mock_widgets.Output.return_value = Mock()
            mock_widgets.Checkbox.return_value = Mock()
            mock_widgets.IntSlider.return_value = Mock()
            mock_widgets.VBox.return_value = Mock()
            mock_widgets.HBox.return_value = Mock()
            
            result = self.features._create_session_management_tab()
            
            self.assertIsNotNone(result)
            mock_widgets.VBox.assert_called()
    
    def test_create_performance_monitoring_tab(self):
        """Test creating performance monitoring tab"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.widgets') as mock_widgets:
            # Mock all required widgets
            mock_widgets.HTML.return_value = Mock()
            mock_widgets.Button.return_value = Mock()
            mock_widgets.Output.return_value = Mock()
            mock_widgets.VBox.return_value = Mock()
            
            result = self.features._create_performance_monitoring_tab()
            
            self.assertIsNotNone(result)
            mock_widgets.VBox.assert_called()
    
    def test_create_report_generation_tab(self):
        """Test creating report generation tab"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.widgets') as mock_widgets:
            # Mock all required widgets
            mock_widgets.HTML.return_value = Mock()
            mock_widgets.Dropdown.return_value = Mock()
            mock_widgets.Button.return_value = Mock()
            mock_widgets.Output.return_value = Mock()
            mock_widgets.VBox.return_value = Mock()
            mock_widgets.HBox.return_value = Mock()
            
            result = self.features._create_report_generation_tab()
            
            self.assertIsNotNone(result)
            mock_widgets.VBox.assert_called()
    
    def test_create_collaboration_tab(self):
        """Test creating collaboration tab"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.widgets') as mock_widgets:
            # Mock all required widgets
            mock_widgets.HTML.return_value = Mock()
            mock_widgets.Text.return_value = Mock()
            mock_widgets.IntText.return_value = Mock()
            mock_widgets.Textarea.return_value = Mock()
            mock_widgets.Dropdown.return_value = Mock()
            mock_widgets.Button.return_value = Mock()
            mock_widgets.Output.return_value = Mock()
            mock_widgets.VBox.return_value = Mock()
            
            result = self.features._create_collaboration_tab()
            
            self.assertIsNotNone(result)
            mock_widgets.VBox.assert_called()
    
    def test_create_advanced_visualization_tab(self):
        """Test creating advanced visualization tab"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.widgets') as mock_widgets:
            # Mock all required widgets
            mock_widgets.HTML.return_value = Mock()
            mock_widgets.Dropdown.return_value = Mock()
            mock_widgets.Button.return_value = Mock()
            mock_widgets.Output.return_value = Mock()
            mock_widgets.VBox.return_value = Mock()
            mock_widgets.HBox.return_value = Mock()
            
            result = self.features._create_advanced_visualization_tab()
            
            self.assertIsNotNone(result)
            mock_widgets.VBox.assert_called()


class TestAdvancedNotebookFeaturesEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.features = AdvancedNotebookFeatures()
    
    def test_create_advanced_dashboard_without_jupyter(self):
        """Test creating dashboard when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.advanced.JUPYTER_AVAILABLE', False):
            features = AdvancedNotebookFeatures()
            result = features.create_advanced_dashboard()
            self.assertIsNone(result)
    
    def test_visualization_methods_without_data(self):
        """Test visualization methods with no data"""
        with patch('builtins.print') as mock_print:
            self.features._create_interactive_timeline(Mock())
            self.features._create_3d_performance_plot(Mock())
            self.features._create_network_diagram(Mock())
            self.features._create_heatmap_analysis(Mock())
        
        # Should print placeholder messages
        self.assertGreater(mock_print.call_count, 0)
    
    def test_export_session_data_creates_directory(self):
        """Test that exporting session data creates necessary directories"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a session
            session_id = self.features.collaboration_manager.create_session("test_pipeline")
            
            # Use nested path that doesn't exist
            nested_path = Path(temp_dir) / "nested" / "dir" / "session.json"
            
            self.features.export_session_data(session_id, nested_path)
            
            self.assertTrue(nested_path.exists())
            self.assertTrue(nested_path.parent.exists())
            
        finally:
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main()
