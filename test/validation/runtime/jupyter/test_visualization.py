"""
Unit tests for visualization.py module
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime, timedelta

import pytest
import pandas as pd

# Import the module under test
from src.cursus.validation.runtime.jupyter.visualization import (
    VisualizationReporter,
    VisualizationConfig,
    TestResultMetrics,
    JUPYTER_AVAILABLE
)


class TestVisualizationConfig(unittest.TestCase):
    """Test cases for VisualizationConfig model"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = VisualizationConfig()
        
        self.assertEqual(config.theme, "plotly_white")
        self.assertEqual(config.figure_width, 1000)
        self.assertEqual(config.figure_height, 600)
        self.assertTrue(config.show_grid)
        self.assertTrue(config.interactive)
        self.assertIsInstance(config.color_palette, list)
        self.assertGreater(len(config.color_palette), 0)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        custom_colors = ["#ff0000", "#00ff00", "#0000ff"]
        config = VisualizationConfig(
            theme="plotly_dark",
            color_palette=custom_colors,
            figure_width=800,
            figure_height=400,
            show_grid=False,
            interactive=False
        )
        
        self.assertEqual(config.theme, "plotly_dark")
        self.assertEqual(config.color_palette, custom_colors)
        self.assertEqual(config.figure_width, 800)
        self.assertEqual(config.figure_height, 400)
        self.assertFalse(config.show_grid)
        self.assertFalse(config.interactive)


class TestTestResultMetrics(unittest.TestCase):
    """Test cases for TestResultMetrics model"""
    
    def test_test_result_metrics_creation(self):
        """Test TestResultMetrics creation with required fields"""
        timestamp = datetime.now()
        metrics = TestResultMetrics(
            step_name="test_step",
            execution_time=1.5,
            success=True,
            timestamp=timestamp,
            test_type="synthetic"
        )
        
        self.assertEqual(metrics.step_name, "test_step")
        self.assertEqual(metrics.execution_time, 1.5)
        self.assertTrue(metrics.success)
        self.assertEqual(metrics.timestamp, timestamp)
        self.assertEqual(metrics.test_type, "synthetic")
        self.assertIsNone(metrics.memory_usage)
        self.assertIsNone(metrics.error_message)
        self.assertIsNone(metrics.data_size)
    
    def test_test_result_metrics_with_optional_fields(self):
        """Test TestResultMetrics creation with optional fields"""
        timestamp = datetime.now()
        metrics = TestResultMetrics(
            step_name="test_step",
            execution_time=2.0,
            memory_usage=512.0,
            success=False,
            error_message="Test failed",
            data_size=1000,
            timestamp=timestamp,
            test_type="real"
        )
        
        self.assertEqual(metrics.memory_usage, 512.0)
        self.assertFalse(metrics.success)
        self.assertEqual(metrics.error_message, "Test failed")
        self.assertEqual(metrics.data_size, 1000)
        self.assertEqual(metrics.test_type, "real")


class TestVisualizationReporter(unittest.TestCase):
    """Test cases for VisualizationReporter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = VisualizationConfig()
        self.reporter = VisualizationReporter(self.config)
        
        # Create sample test results
        self.sample_results = [
            TestResultMetrics(
                step_name="step1",
                execution_time=1.5,
                memory_usage=100.0,
                success=True,
                timestamp=datetime.now(),
                test_type="synthetic",
                data_size=1000
            ),
            TestResultMetrics(
                step_name="step2",
                execution_time=2.0,
                memory_usage=150.0,
                success=False,
                error_message="Test failed",
                timestamp=datetime.now() + timedelta(seconds=30),
                test_type="synthetic",
                data_size=2000
            ),
            TestResultMetrics(
                step_name="step1",
                execution_time=1.8,
                memory_usage=120.0,
                success=True,
                timestamp=datetime.now() + timedelta(seconds=60),
                test_type="real",
                data_size=1500
            )
        ]
    
    def test_initialization_without_jupyter(self):
        """Test initialization when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.visualization.JUPYTER_AVAILABLE', False):
            reporter = VisualizationReporter()
            self.assertIsNotNone(reporter)
    
    def test_initialization_with_config(self):
        """Test initialization with custom config"""
        custom_config = VisualizationConfig(theme="plotly_dark")
        reporter = VisualizationReporter(custom_config)
        
        self.assertEqual(reporter.config.theme, "plotly_dark")
        self.assertIsInstance(reporter.test_results, list)
        self.assertIsInstance(reporter.performance_data, dict)
        self.assertIsInstance(reporter.data_quality_metrics, dict)
    
    def test_add_test_result(self):
        """Test adding test results"""
        result = self.sample_results[0]
        self.reporter.add_test_result(result)
        
        self.assertEqual(len(self.reporter.test_results), 1)
        self.assertEqual(self.reporter.test_results[0], result)
    
    def test_add_performance_data(self):
        """Test adding performance data"""
        metrics = {"avg_time": 1.5, "max_memory": 200}
        self.reporter.add_performance_data("test_step", metrics)
        
        self.assertIn("test_step", self.reporter.performance_data)
        self.assertEqual(self.reporter.performance_data["test_step"], metrics)
    
    def test_add_data_quality_metrics(self):
        """Test adding data quality metrics"""
        metrics = {"completeness": 0.95, "validity": 0.98}
        self.reporter.add_data_quality_metrics("test_step", metrics)
        
        self.assertIn("test_step", self.reporter.data_quality_metrics)
        self.assertEqual(self.reporter.data_quality_metrics["test_step"], metrics)
    
    def test_display_summary(self):
        """Test display summary functionality"""
        # Add test results
        for result in self.sample_results:
            self.reporter.add_test_result(result)
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            self.reporter.display_summary()
        
        # Verify summary information was printed
        mock_print.assert_called()
        print_calls = [call[0][0] for call in mock_print.call_args_list]
        summary_text = ' '.join(print_calls)
        
        self.assertIn("Total Tests: 3", summary_text)
        self.assertIn("Successful Tests: 2", summary_text)
        self.assertIn("Success Rate: 66.7%", summary_text)
        self.assertIn("Synthetic Tests: 2", summary_text)
        self.assertIn("Real Data Tests: 1", summary_text)
    
    def test_display_summary_no_results(self):
        """Test display summary with no results"""
        with patch('builtins.print') as mock_print:
            self.reporter.display_summary()
        
        mock_print.assert_called_with("No test results available")


@unittest.skipIf(not JUPYTER_AVAILABLE, "Jupyter dependencies not available")
class TestVisualizationReporterWithJupyter(unittest.TestCase):
    """Test cases that require Jupyter dependencies"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = VisualizationConfig()
        self.reporter = VisualizationReporter(self.config)
        
        # Add sample test results
        self.sample_results = [
            TestResultMetrics(
                step_name="step1",
                execution_time=1.5,
                memory_usage=100.0,
                success=True,
                timestamp=datetime.now(),
                test_type="synthetic",
                data_size=1000
            ),
            TestResultMetrics(
                step_name="step2",
                execution_time=2.0,
                memory_usage=150.0,
                success=True,
                timestamp=datetime.now() + timedelta(seconds=30),
                test_type="synthetic",
                data_size=2000
            )
        ]
        
        for result in self.sample_results:
            self.reporter.add_test_result(result)
    
    @patch('src.cursus.validation.runtime.jupyter.visualization.go')
    @patch('src.cursus.validation.runtime.jupyter.visualization.pd')
    def test_create_execution_timeline(self, mock_pd, mock_go):
        """Test execution timeline creation"""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_df.iterrows.return_value = [
            (0, {
                'step_name': 'step1',
                'start_time': datetime.now(),
                'end_time': datetime.now() + timedelta(seconds=1.5),
                'execution_time': 1.5,
                'success': True,
                'test_type': 'synthetic',
                'error_message': 'Success'
            })
        ]
        mock_pd.DataFrame.return_value = mock_df
        
        # Mock plotly Figure
        mock_fig = Mock()
        mock_go.Figure.return_value = mock_fig
        mock_go.Scatter.return_value = Mock()
        
        result = self.reporter.create_execution_timeline()
        
        self.assertIsNotNone(result)
        mock_go.Figure.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.visualization.make_subplots')
    @patch('src.cursus.validation.runtime.jupyter.visualization.go')
    @patch('src.cursus.validation.runtime.jupyter.visualization.pd')
    def test_create_performance_dashboard(self, mock_pd, mock_go, mock_subplots):
        """Test performance dashboard creation"""
        # Mock pandas DataFrame with proper groupby behavior
        mock_df = Mock()
        mock_grouped = Mock()
        mock_mean_result = Mock()
        mock_reset_result = Mock()
        
        # Configure the mock to support both groupby patterns
        mock_df.groupby.return_value = mock_grouped
        
        # Create a mock for column selection after groupby
        mock_column_selector = Mock()
        mock_column_selector.mean.return_value.reset_index.return_value = mock_reset_result
        mock_grouped.__getitem__ = Mock(return_value=mock_column_selector)
        
        # Also support direct mean on grouped data
        mock_grouped.mean.return_value.reset_index.return_value = mock_reset_result
        
        # Create a proper mock for DataFrame indexing that supports subscripting and filtering
        mock_series = Mock()
        mock_series.empty = False
        
        # Create a mock for comparison results that also supports & operator
        mock_comparison_result = Mock()
        mock_comparison_result.__and__ = Mock(return_value=Mock())
        
        mock_series.__gt__ = Mock(return_value=mock_comparison_result)  # Support > comparison
        mock_series.__and__ = Mock(return_value=Mock())  # Support & operator
        
        # Create a mock for filtered DataFrame
        mock_filtered_df = Mock()
        mock_filtered_df.__getitem__ = Mock(return_value=mock_series)
        
        mock_df.__getitem__ = Mock(return_value=mock_series)
        mock_df.__getitem__.side_effect = lambda key: mock_filtered_df if isinstance(key, Mock) else mock_series
        
        # Make the reset_result mock support column access and assignment
        mock_reset_result.__getitem__ = Mock(return_value=['step1', 'step2'])
        mock_reset_result.__setitem__ = Mock()
        
        mock_pd.DataFrame.return_value = mock_df
        
        # Mock subplots
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        
        # Mock plotly components
        mock_go.Bar.return_value = Mock()
        mock_go.Histogram.return_value = Mock()
        mock_go.Scatter.return_value = Mock()
        
        result = self.reporter.create_performance_dashboard()
        
        self.assertIsNotNone(result)
        mock_subplots.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.visualization.make_subplots')
    @patch('src.cursus.validation.runtime.jupyter.visualization.go')
    def test_create_data_quality_report(self, mock_go, mock_subplots):
        """Test data quality report creation"""
        # Add sample data quality metrics
        self.reporter.add_data_quality_metrics("step1", {
            "completeness": 0.95,
            "validity": 0.98,
            "schema_compliance": 0.92,
            "overall_score": 0.95
        })
        self.reporter.add_data_quality_metrics("step2", {
            "completeness": 0.88,
            "validity": 0.94,
            "schema_compliance": 0.96,
            "overall_score": 0.93
        })
        
        # Mock subplots
        mock_fig = Mock()
        mock_subplots.return_value = mock_fig
        
        # Mock plotly components
        mock_go.Bar.return_value = Mock()
        mock_go.Scatter.return_value = Mock()
        
        result = self.reporter.create_data_quality_report()
        
        self.assertIsNotNone(result)
        mock_subplots.assert_called_once()
        mock_fig.add_trace.assert_called()
        mock_fig.update_layout.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.visualization.px')
    @patch('src.cursus.validation.runtime.jupyter.visualization.pd')
    def test_create_comparison_chart(self, mock_pd, mock_px):
        """Test comparison chart creation"""
        # Mock pandas DataFrame
        mock_df = Mock()
        mock_pd.DataFrame.return_value = mock_df
        
        # Mock plotly express
        mock_fig = Mock()
        mock_px.box.return_value = mock_fig
        
        result = self.reporter.create_comparison_chart("execution_time", "test_type")
        
        self.assertIsNotNone(result)
        mock_px.box.assert_called_once()
        mock_fig.update_layout.assert_called_once()
    
    @patch('src.cursus.validation.runtime.jupyter.visualization.widgets')
    @patch('src.cursus.validation.runtime.jupyter.visualization.display')
    def test_create_interactive_dashboard(self, mock_display, mock_widgets):
        """Test interactive dashboard creation"""
        # Mock widgets
        mock_output = Mock()
        mock_dropdown = Mock()
        mock_button = Mock()
        mock_vbox = Mock()
        mock_hbox = Mock()
        
        mock_widgets.Output.return_value = mock_output
        mock_widgets.Dropdown.return_value = mock_dropdown
        mock_widgets.Button.return_value = mock_button
        mock_widgets.VBox.return_value = mock_vbox
        mock_widgets.HBox.return_value = mock_hbox
        
        # Make mock_output support context manager protocol
        mock_output.__enter__ = Mock(return_value=mock_output)
        mock_output.__exit__ = Mock(return_value=None)

        result = self.reporter.create_interactive_dashboard()
        
        self.assertIsNotNone(result)
        mock_widgets.Output.assert_called()
        mock_widgets.Dropdown.assert_called()
        mock_widgets.Button.assert_called()
        mock_widgets.VBox.assert_called()
        mock_widgets.HBox.assert_called()
    
    def test_export_report_html(self):
        """Test HTML report export"""
        temp_dir = tempfile.mkdtemp()
        try:
            output_path = Path(temp_dir) / "test_report.html"
            
            with patch.object(self.reporter, '_generate_html_report', return_value="<html>Test Report</html>"):
                self.reporter.export_report(output_path, "html")
            
            self.assertTrue(output_path.exists())
            content = output_path.read_text()
            self.assertEqual(content, "<html>Test Report</html>")
            
        finally:
            shutil.rmtree(temp_dir)
    
    def test_export_report_json(self):
        """Test JSON report export"""
        temp_dir = tempfile.mkdtemp()
        try:
            output_path = Path(temp_dir) / "test_report.json"
            
            self.reporter.export_report(output_path, "json")
            
            self.assertTrue(output_path.exists())
            with open(output_path) as f:
                data = json.load(f)
            
            self.assertIn("test_results", data)
            self.assertIn("performance_data", data)
            self.assertIn("data_quality_metrics", data)
            self.assertIn("generated_at", data)
            
        finally:
            shutil.rmtree(temp_dir)
    
    @patch('src.cursus.validation.runtime.jupyter.visualization.datetime')
    def test_generate_html_report(self, mock_datetime):
        """Test HTML report generation"""
        mock_datetime.now.return_value.strftime.return_value = "2023-01-01 12:00:00"
        
        with patch.object(self.reporter, 'create_execution_timeline', return_value=Mock(to_html=Mock(return_value="<div>Timeline</div>"))):
            with patch.object(self.reporter, 'create_performance_dashboard', return_value=Mock(to_html=Mock(return_value="<div>Dashboard</div>"))):
                with patch.object(self.reporter, 'create_data_quality_report', return_value=Mock(to_html=Mock(return_value="<div>Quality</div>"))):
                    result = self.reporter._generate_html_report()
        
        self.assertIn("<html>", result)
        self.assertIn("Pipeline Runtime Testing Report", result)
        self.assertIn("2023-01-01 12:00:00", result)
        self.assertIn("<div>Timeline</div>", result)
        self.assertIn("<div>Dashboard</div>", result)
        self.assertIn("<div>Quality</div>", result)


class TestVisualizationReporterEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.reporter = VisualizationReporter()
    
    def test_create_execution_timeline_no_results(self):
        """Test timeline creation with no results"""
        result = self.reporter.create_execution_timeline()
        
        if JUPYTER_AVAILABLE:
            self.assertIsNone(result)
        else:
            self.assertIsNone(result)
    
    def test_create_performance_dashboard_no_results(self):
        """Test dashboard creation with no results"""
        result = self.reporter.create_performance_dashboard()
        
        if JUPYTER_AVAILABLE:
            self.assertIsNone(result)
        else:
            self.assertIsNone(result)
    
    def test_create_data_quality_report_no_metrics(self):
        """Test quality report creation with no metrics"""
        result = self.reporter.create_data_quality_report()
        
        if JUPYTER_AVAILABLE:
            self.assertIsNone(result)
        else:
            self.assertIsNone(result)
    
    def test_export_report_without_jupyter(self):
        """Test report export when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.visualization.JUPYTER_AVAILABLE', False):
            reporter = VisualizationReporter()
            
            with patch('builtins.print') as mock_print:
                reporter.export_report(Path("test.html"), "html")
            
            mock_print.assert_called_with("Cannot export report: Jupyter dependencies not available")
    
    def test_setup_plotly_theme_without_jupyter(self):
        """Test Plotly theme setup when Jupyter is not available"""
        with patch('src.cursus.validation.runtime.jupyter.visualization.JUPYTER_AVAILABLE', False):
            reporter = VisualizationReporter()
            # Should not raise an exception
            reporter._setup_plotly_theme()


if __name__ == '__main__':
    unittest.main()
