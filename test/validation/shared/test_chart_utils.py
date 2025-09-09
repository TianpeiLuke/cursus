"""
Unit tests for chart_utils.py

Tests shared chart generation utilities including:
- Quality color mapping and rating functions
- Score bar chart generation
- Comparison chart creation
- Trend chart functionality
- Quality distribution charts
- Chart configuration and styling
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Any

from cursus.validation.shared.chart_utils import (
    get_quality_color,
    get_quality_rating,
    create_score_bar_chart,
    create_comparison_chart,
    create_trend_chart,
    create_quality_distribution_chart,
    DEFAULT_CHART_CONFIG,
    QUALITY_THRESHOLDS
)


class TestQualityColorMapping(unittest.TestCase):
    """Test quality color mapping functionality."""
    
    def test_get_quality_color_excellent(self):
        """Test color mapping for excellent scores."""
        color = get_quality_color(95.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["excellent"])
        
        color = get_quality_color(90.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["excellent"])
    
    def test_get_quality_color_good(self):
        """Test color mapping for good scores."""
        color = get_quality_color(85.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["good"])
        
        color = get_quality_color(80.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["good"])
    
    def test_get_quality_color_satisfactory(self):
        """Test color mapping for satisfactory scores."""
        color = get_quality_color(75.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["satisfactory"])
        
        color = get_quality_color(70.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["satisfactory"])
    
    def test_get_quality_color_needs_work(self):
        """Test color mapping for needs work scores."""
        color = get_quality_color(65.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["needs_work"])
        
        color = get_quality_color(60.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["needs_work"])
    
    def test_get_quality_color_poor(self):
        """Test color mapping for poor scores."""
        color = get_quality_color(50.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["poor"])
        
        color = get_quality_color(0.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["poor"])
    
    def test_get_quality_color_custom_config(self):
        """Test color mapping with custom configuration."""
        custom_config = {
            "colors": {
                "excellent": "#custom_green",
                "good": "#custom_light_green",
                "satisfactory": "#custom_orange",
                "needs_work": "#custom_salmon",
                "poor": "#custom_red"
            }
        }
        
        color = get_quality_color(95.0, custom_config)
        self.assertEqual(color, "#custom_green")
        
        color = get_quality_color(50.0, custom_config)
        self.assertEqual(color, "#custom_red")
    
    def test_get_quality_color_edge_cases(self):
        """Test color mapping for edge cases."""
        # Test negative scores
        color = get_quality_color(-10.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["poor"])
        
        # Test scores above 100
        color = get_quality_color(110.0)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["excellent"])


class TestQualityRating(unittest.TestCase):
    """Test quality rating functionality."""
    
    def test_get_quality_rating_excellent(self):
        """Test rating for excellent scores."""
        rating = get_quality_rating(95.0)
        self.assertEqual(rating, "Excellent")
        
        rating = get_quality_rating(90.0)
        self.assertEqual(rating, "Excellent")
    
    def test_get_quality_rating_good(self):
        """Test rating for good scores."""
        rating = get_quality_rating(85.0)
        self.assertEqual(rating, "Good")
        
        rating = get_quality_rating(80.0)
        self.assertEqual(rating, "Good")
    
    def test_get_quality_rating_satisfactory(self):
        """Test rating for satisfactory scores."""
        rating = get_quality_rating(75.0)
        self.assertEqual(rating, "Satisfactory")
        
        rating = get_quality_rating(70.0)
        self.assertEqual(rating, "Satisfactory")
    
    def test_get_quality_rating_needs_work(self):
        """Test rating for needs work scores."""
        rating = get_quality_rating(65.0)
        self.assertEqual(rating, "Needs Work")
        
        rating = get_quality_rating(60.0)
        self.assertEqual(rating, "Needs Work")
    
    def test_get_quality_rating_poor(self):
        """Test rating for poor scores."""
        rating = get_quality_rating(50.0)
        self.assertEqual(rating, "Poor")
        
        rating = get_quality_rating(0.0)
        self.assertEqual(rating, "Poor")


class TestScoreBarChart(unittest.TestCase):
    """Test score bar chart creation functionality."""
    
    def test_create_score_bar_chart_no_matplotlib(self):
        """Test chart creation when matplotlib is not available."""
        with patch('cursus.validation.shared.chart_utils.plt', side_effect=ImportError):
            result = create_score_bar_chart(
                levels=["Level1", "Level2"],
                scores=[80.0, 90.0],
                title="Test Chart"
            )
            self.assertIsNone(result)
    
    @patch('cursus.validation.shared.chart_utils.plt')
    def test_create_score_bar_chart_basic(self, mock_plt):
        """Test basic score bar chart creation."""
        # Mock matplotlib components
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_plt.bar.return_value = [Mock(get_height=Mock(return_value=80.0), 
                                         get_x=Mock(return_value=0), 
                                         get_width=Mock(return_value=1))]
        
        result = create_score_bar_chart(
            levels=["Level1", "Level2"],
            scores=[80.0, 90.0],
            title="Test Chart"
        )
        
        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        mock_plt.bar.assert_called_once()
        mock_plt.title.assert_called_with("Test Chart")
        mock_plt.ylabel.assert_called_with("Score (%)")
        mock_plt.ylim.assert_called_with(0, 105)
        mock_plt.show.assert_called_once()
        
        self.assertIsNone(result)  # No output path specified
    
    @patch('cursus.validation.shared.chart_utils.plt')
    @patch('cursus.validation.shared.chart_utils.Path')
    def test_create_score_bar_chart_with_output(self, mock_path, mock_plt):
        """Test score bar chart creation with output file."""
        # Mock path operations
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.mkdir = Mock()
        
        # Mock matplotlib components
        mock_plt.bar.return_value = [Mock(get_height=Mock(return_value=80.0), 
                                         get_x=Mock(return_value=0), 
                                         get_width=Mock(return_value=1))]
        
        output_path = "/tmp/test_chart.png"
        result = create_score_bar_chart(
            levels=["Level1"],
            scores=[80.0],
            title="Test Chart",
            output_path=output_path
        )
        
        # Verify file operations
        mock_path.assert_called_with(output_path)
        mock_path_instance.parent.mkdir.assert_called_with(parents=True, exist_ok=True)
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
        
        self.assertEqual(result, output_path)
    
    @patch('cursus.validation.shared.chart_utils.plt')
    def test_create_score_bar_chart_with_overall_score(self, mock_plt):
        """Test score bar chart with overall score line."""
        mock_plt.bar.return_value = [Mock(get_height=Mock(return_value=80.0), 
                                         get_x=Mock(return_value=0), 
                                         get_width=Mock(return_value=1))]
        
        create_score_bar_chart(
            levels=["Level1", "Level2"],
            scores=[80.0, 90.0],
            title="Test Chart",
            overall_score=85.0,
            overall_rating="Good"
        )
        
        # Verify overall score line is added
        mock_plt.axhline.assert_called_with(y=85.0, color='blue', linestyle='-', alpha=0.7)
        mock_plt.text.assert_called()
    
    @patch('cursus.validation.shared.chart_utils.plt')
    def test_create_score_bar_chart_many_levels(self, mock_plt):
        """Test score bar chart with many levels (should rotate labels)."""
        mock_plt.bar.return_value = [Mock(get_height=Mock(return_value=80.0), 
                                         get_x=Mock(return_value=i), 
                                         get_width=Mock(return_value=1)) for i in range(5)]
        
        create_score_bar_chart(
            levels=["Level1", "Level2", "Level3", "Level4", "Level5"],
            scores=[80.0, 90.0, 70.0, 85.0, 95.0],
            title="Test Chart"
        )
        
        # Verify x-axis labels are rotated for many levels
        mock_plt.xticks.assert_called_with(rotation=45, ha='right')
    
    @patch('cursus.validation.shared.chart_utils.plt')
    def test_create_score_bar_chart_exception_handling(self, mock_plt):
        """Test chart creation with exception handling."""
        mock_plt.bar.side_effect = Exception("Test error")
        
        with patch('builtins.print') as mock_print:
            result = create_score_bar_chart(
                levels=["Level1"],
                scores=[80.0],
                title="Test Chart"
            )
            
            self.assertIsNone(result)
            mock_print.assert_called_with("Could not generate chart: Test error")


class TestComparisonChart(unittest.TestCase):
    """Test comparison chart creation functionality."""
    
    @patch('cursus.validation.shared.chart_utils.plt')
    @patch('cursus.validation.shared.chart_utils.np')
    def test_create_comparison_chart_basic(self, mock_np, mock_plt):
        """Test basic comparison chart creation."""
        # Mock numpy
        mock_np.arange.return_value = [0, 1, 2]
        
        # Mock matplotlib components
        mock_plt.bar.return_value = [Mock(get_height=Mock(return_value=80.0), 
                                         get_x=Mock(return_value=0), 
                                         get_width=Mock(return_value=0.4))]
        
        series_data = {
            "Series1": [80.0, 90.0, 70.0],
            "Series2": [85.0, 88.0, 75.0]
        }
        
        result = create_comparison_chart(
            categories=["Cat1", "Cat2", "Cat3"],
            series_data=series_data,
            title="Comparison Chart"
        )
        
        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        self.assertEqual(mock_plt.bar.call_count, 2)  # Two series
        mock_plt.title.assert_called_with("Comparison Chart")
        mock_plt.ylabel.assert_called_with("Score (%)")
        mock_plt.xlabel.assert_called_with("Categories")
        mock_plt.legend.assert_called_once()
        
        self.assertIsNone(result)  # No output path specified
    
    def test_create_comparison_chart_no_matplotlib(self):
        """Test comparison chart creation when matplotlib is not available."""
        with patch('cursus.validation.shared.chart_utils.plt', side_effect=ImportError):
            result = create_comparison_chart(
                categories=["Cat1"],
                series_data={"Series1": [80.0]},
                title="Test Chart"
            )
            self.assertIsNone(result)


class TestTrendChart(unittest.TestCase):
    """Test trend chart creation functionality."""
    
    @patch('cursus.validation.shared.chart_utils.plt')
    def test_create_trend_chart_basic(self, mock_plt):
        """Test basic trend chart creation."""
        result = create_trend_chart(
            x_values=[1, 2, 3, 4],
            y_values=[70.0, 80.0, 85.0, 90.0],
            title="Trend Chart",
            x_label="Time",
            y_label="Performance"
        )
        
        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        mock_plt.scatter.assert_called_once()
        mock_plt.title.assert_called_with("Trend Chart")
        mock_plt.xlabel.assert_called_with("Time")
        mock_plt.ylabel.assert_called_with("Performance")
        mock_plt.ylim.assert_called_with(0, 105)
        
        self.assertIsNone(result)  # No output path specified
    
    def test_create_trend_chart_no_matplotlib(self):
        """Test trend chart creation when matplotlib is not available."""
        with patch('cursus.validation.shared.chart_utils.plt', side_effect=ImportError):
            result = create_trend_chart(
                x_values=[1, 2, 3],
                y_values=[70.0, 80.0, 90.0],
                title="Test Chart"
            )
            self.assertIsNone(result)


class TestQualityDistributionChart(unittest.TestCase):
    """Test quality distribution chart creation functionality."""
    
    @patch('cursus.validation.shared.chart_utils.plt')
    @patch('cursus.validation.shared.chart_utils.np')
    def test_create_quality_distribution_chart_basic(self, mock_np, mock_plt):
        """Test basic quality distribution chart creation."""
        # Mock numpy
        mock_np.mean.return_value = 75.0
        
        # Mock matplotlib components
        mock_patches = [Mock() for _ in range(5)]
        mock_plt.hist.return_value = ([2, 3, 4, 2, 1], [0, 60, 70, 80, 90, 100], mock_patches)
        
        scores = [65.0, 75.0, 85.0, 95.0, 55.0, 78.0, 82.0, 88.0, 92.0, 45.0, 67.0, 73.0]
        
        result = create_quality_distribution_chart(
            scores=scores,
            title="Quality Distribution"
        )
        
        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()
        mock_plt.axvline.assert_called_with(75.0, color='red', linestyle='--', alpha=0.8)
        mock_plt.title.assert_called_with("Quality Distribution")
        mock_plt.xlabel.assert_called_with("Score Range")
        mock_plt.ylabel.assert_called_with("Count")
        
        # Verify patches are colored
        for patch in mock_patches:
            patch.set_facecolor.assert_called_once()
        
        self.assertIsNone(result)  # No output path specified
    
    def test_create_quality_distribution_chart_no_matplotlib(self):
        """Test quality distribution chart when matplotlib is not available."""
        with patch('cursus.validation.shared.chart_utils.plt', side_effect=ImportError):
            result = create_quality_distribution_chart(
                scores=[70.0, 80.0, 90.0],
                title="Test Chart"
            )
            self.assertIsNone(result)


class TestChartConfiguration(unittest.TestCase):
    """Test chart configuration functionality."""
    
    def test_default_chart_config_structure(self):
        """Test that default chart config has expected structure."""
        self.assertIn("figure_size", DEFAULT_CHART_CONFIG)
        self.assertIn("colors", DEFAULT_CHART_CONFIG)
        self.assertIn("grid_style", DEFAULT_CHART_CONFIG)
        self.assertIn("dpi", DEFAULT_CHART_CONFIG)
        self.assertIn("bbox_inches", DEFAULT_CHART_CONFIG)
        
        # Test colors
        colors = DEFAULT_CHART_CONFIG["colors"]
        expected_colors = ["excellent", "good", "satisfactory", "needs_work", "poor"]
        for color_key in expected_colors:
            self.assertIn(color_key, colors)
            self.assertIsInstance(colors[color_key], str)
            self.assertTrue(colors[color_key].startswith("#"))
    
    def test_quality_thresholds_structure(self):
        """Test that quality thresholds are properly structured."""
        self.assertIsInstance(QUALITY_THRESHOLDS, dict)
        self.assertEqual(len(QUALITY_THRESHOLDS), 5)
        
        # Test threshold values
        expected_thresholds = [90, 80, 70, 60, 0]
        for threshold in expected_thresholds:
            self.assertIn(threshold, QUALITY_THRESHOLDS)
        
        # Test threshold mappings
        self.assertEqual(QUALITY_THRESHOLDS[90], "excellent")
        self.assertEqual(QUALITY_THRESHOLDS[80], "good")
        self.assertEqual(QUALITY_THRESHOLDS[70], "satisfactory")
        self.assertEqual(QUALITY_THRESHOLDS[60], "needs_work")
        self.assertEqual(QUALITY_THRESHOLDS[0], "poor")


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Test empty levels and scores
        with patch('cursus.validation.shared.chart_utils.plt') as mock_plt:
            mock_plt.bar.return_value = []
            
            result = create_score_bar_chart(
                levels=[],
                scores=[],
                title="Empty Chart"
            )
            
            # Should still attempt to create chart
            mock_plt.figure.assert_called_once()
    
    def test_mismatched_data_lengths(self):
        """Test handling of mismatched data lengths."""
        with patch('cursus.validation.shared.chart_utils.plt') as mock_plt:
            # This should be handled by matplotlib, but we test our code doesn't crash
            create_score_bar_chart(
                levels=["Level1", "Level2"],
                scores=[80.0],  # Mismatched length
                title="Mismatched Chart"
            )
            
            mock_plt.figure.assert_called_once()
    
    def test_invalid_scores(self):
        """Test handling of invalid score values."""
        # Test with None values
        color = get_quality_color(None)
        self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["poor"])
        
        # Test with string values (should handle gracefully)
        try:
            color = get_quality_color("invalid")
            # If it doesn't crash, it should return poor color
            self.assertEqual(color, DEFAULT_CHART_CONFIG["colors"]["poor"])
        except (TypeError, ValueError):
            # It's also acceptable to raise an exception
            pass
    
    def test_file_path_edge_cases(self):
        """Test file path edge cases."""
        with patch('cursus.validation.shared.chart_utils.plt') as mock_plt:
            with patch('cursus.validation.shared.chart_utils.Path') as mock_path:
                # Test with None output path
                result = create_score_bar_chart(
                    levels=["Level1"],
                    scores=[80.0],
                    title="Test Chart",
                    output_path=None
                )
                
                self.assertIsNone(result)
                mock_path.assert_not_called()
                mock_plt.show.assert_called_once()
    
    def test_custom_config_partial(self):
        """Test with partial custom configuration."""
        partial_config = {
            "colors": {
                "excellent": "#custom_green"
                # Missing other colors
            }
        }
        
        # Should handle missing colors gracefully
        color = get_quality_color(95.0, partial_config)
        self.assertEqual(color, "#custom_green")
        
        # For missing color, should fall back to default or handle gracefully
        color = get_quality_color(50.0, partial_config)
        # Should either use default poor color or custom poor color if available
        self.assertIn(color, [DEFAULT_CHART_CONFIG["colors"]["poor"], partial_config.get("colors", {}).get("poor", DEFAULT_CHART_CONFIG["colors"]["poor"])])


if __name__ == '__main__':
    unittest.main()
