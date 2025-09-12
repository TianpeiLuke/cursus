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

import pytest
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
    QUALITY_THRESHOLDS,
)


class TestQualityColorMapping:
    """Test quality color mapping functionality."""

    def test_get_quality_color_excellent(self):
        """Test color mapping for excellent scores."""
        color = get_quality_color(95.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["excellent"]

        color = get_quality_color(90.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["excellent"]

    def test_get_quality_color_good(self):
        """Test color mapping for good scores."""
        color = get_quality_color(85.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["good"]

        color = get_quality_color(80.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["good"]

    def test_get_quality_color_satisfactory(self):
        """Test color mapping for satisfactory scores."""
        color = get_quality_color(75.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["satisfactory"]

        color = get_quality_color(70.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["satisfactory"]

    def test_get_quality_color_needs_work(self):
        """Test color mapping for needs work scores."""
        color = get_quality_color(65.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["needs_work"]

        color = get_quality_color(60.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["needs_work"]

    def test_get_quality_color_poor(self):
        """Test color mapping for poor scores."""
        color = get_quality_color(50.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["poor"]

        color = get_quality_color(0.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["poor"]

    def test_get_quality_color_custom_config(self):
        """Test color mapping with custom configuration."""
        custom_config = {
            "colors": {
                "excellent": "#custom_green",
                "good": "#custom_light_green",
                "satisfactory": "#custom_orange",
                "needs_work": "#custom_salmon",
                "poor": "#custom_red",
            }
        }

        color = get_quality_color(95.0, custom_config)
        assert color == "#custom_green"

        color = get_quality_color(50.0, custom_config)
        assert color == "#custom_red"

    def test_get_quality_color_edge_cases(self):
        """Test color mapping for edge cases."""
        # Test negative scores
        color = get_quality_color(-10.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["poor"]

        # Test scores above 100
        color = get_quality_color(110.0)
        assert color == DEFAULT_CHART_CONFIG["colors"]["excellent"]


class TestQualityRating:
    """Test quality rating functionality."""

    def test_get_quality_rating_excellent(self):
        """Test rating for excellent scores."""
        rating = get_quality_rating(95.0)
        assert rating == "Excellent"

        rating = get_quality_rating(90.0)
        assert rating == "Excellent"

    def test_get_quality_rating_good(self):
        """Test rating for good scores."""
        rating = get_quality_rating(85.0)
        assert rating == "Good"

        rating = get_quality_rating(80.0)
        assert rating == "Good"

    def test_get_quality_rating_satisfactory(self):
        """Test rating for satisfactory scores."""
        rating = get_quality_rating(75.0)
        assert rating == "Satisfactory"

        rating = get_quality_rating(70.0)
        assert rating == "Satisfactory"

    def test_get_quality_rating_needs_work(self):
        """Test rating for needs work scores."""
        rating = get_quality_rating(65.0)
        assert rating == "Needs Work"

        rating = get_quality_rating(60.0)
        assert rating == "Needs Work"

    def test_get_quality_rating_poor(self):
        """Test rating for poor scores."""
        rating = get_quality_rating(50.0)
        assert rating == "Poor"

        rating = get_quality_rating(0.0)
        assert rating == "Poor"


class TestScoreBarChart:
    """Test score bar chart creation functionality."""

    def test_create_score_bar_chart_no_matplotlib(self):
        """Test chart creation when matplotlib is not available."""
        with patch("cursus.validation.shared.chart_utils.plt", None):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'matplotlib'"),
            ):
                result = create_score_bar_chart(
                    levels=["Level1", "Level2"], scores=[80.0, 90.0], title="Test Chart"
                )
                assert result is None

    @patch("matplotlib.pyplot")
    def test_create_score_bar_chart_basic(self, mock_plt):
        """Test basic score bar chart creation."""
        # Mock matplotlib components
        mock_figure = Mock()
        mock_plt.figure.return_value = mock_figure
        mock_plt.bar.return_value = [
            Mock(
                get_height=Mock(return_value=80.0),
                get_x=Mock(return_value=0),
                get_width=Mock(return_value=1),
            )
        ]

        result = create_score_bar_chart(
            levels=["Level1", "Level2"], scores=[80.0, 90.0], title="Test Chart"
        )

        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        mock_plt.bar.assert_called_once()
        mock_plt.title.assert_called_with("Test Chart")
        mock_plt.ylabel.assert_called_with("Score (%)")
        mock_plt.ylim.assert_called_with(0, 105)
        mock_plt.show.assert_called_once()

        assert result is None  # No output path specified

    @patch("matplotlib.pyplot")
    @patch("cursus.validation.shared.chart_utils.Path")
    def test_create_score_bar_chart_with_output(self, mock_path, mock_plt):
        """Test score bar chart creation with output file."""
        # Mock path operations
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.mkdir = Mock()

        # Mock matplotlib components
        mock_plt.bar.return_value = [
            Mock(
                get_height=Mock(return_value=80.0),
                get_x=Mock(return_value=0),
                get_width=Mock(return_value=1),
            )
        ]

        output_path = "/tmp/test_chart.png"
        result = create_score_bar_chart(
            levels=["Level1"],
            scores=[80.0],
            title="Test Chart",
            output_path=output_path,
        )

        # Verify file operations
        mock_path.assert_called_with(output_path)
        mock_path_instance.parent.mkdir.assert_called_with(parents=True, exist_ok=True)
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()

        assert result == output_path

    @patch("matplotlib.pyplot")
    def test_create_score_bar_chart_with_overall_score(self, mock_plt):
        """Test score bar chart with overall score line."""
        mock_plt.bar.return_value = [
            Mock(
                get_height=Mock(return_value=80.0),
                get_x=Mock(return_value=0),
                get_width=Mock(return_value=1),
            )
        ]

        create_score_bar_chart(
            levels=["Level1", "Level2"],
            scores=[80.0, 90.0],
            title="Test Chart",
            overall_score=85.0,
            overall_rating="Good",
        )

        # Verify overall score line is added
        mock_plt.axhline.assert_called_with(
            y=85.0, color="blue", linestyle="-", alpha=0.7
        )
        mock_plt.text.assert_called()

    @patch("matplotlib.pyplot")
    def test_create_score_bar_chart_many_levels(self, mock_plt):
        """Test score bar chart with many levels (should rotate labels)."""
        mock_plt.bar.return_value = [
            Mock(
                get_height=Mock(return_value=80.0),
                get_x=Mock(return_value=i),
                get_width=Mock(return_value=1),
            )
            for i in range(5)
        ]

        create_score_bar_chart(
            levels=["Level1", "Level2", "Level3", "Level4", "Level5"],
            scores=[80.0, 90.0, 70.0, 85.0, 95.0],
            title="Test Chart",
        )

        # Verify x-axis labels are rotated for many levels
        mock_plt.xticks.assert_called_with(rotation=45, ha="right")

    @patch("matplotlib.pyplot")
    def test_create_score_bar_chart_exception_handling(self, mock_plt):
        """Test chart creation with exception handling."""
        mock_plt.bar.side_effect = Exception("Test error")

        with patch("builtins.print") as mock_print:
            result = create_score_bar_chart(
                levels=["Level1"], scores=[80.0], title="Test Chart"
            )

            assert result is None
            mock_print.assert_called_with("Could not generate chart: Test error")


class TestComparisonChart:
    """Test comparison chart creation functionality."""

    @patch("matplotlib.pyplot")
    @patch("cursus.validation.shared.chart_utils.np")
    def test_create_comparison_chart_basic(self, mock_np, mock_plt):
        """Test basic comparison chart creation."""
        # Mock numpy
        mock_np.arange.return_value = [0, 1, 2]

        # Mock matplotlib components
        mock_plt.bar.return_value = [
            Mock(
                get_height=Mock(return_value=80.0),
                get_x=Mock(return_value=0),
                get_width=Mock(return_value=0.4),
            )
        ]

        series_data = {"Series1": [80.0, 90.0, 70.0], "Series2": [85.0, 88.0, 75.0]}

        result = create_comparison_chart(
            categories=["Cat1", "Cat2", "Cat3"],
            series_data=series_data,
            title="Comparison Chart",
        )

        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        assert mock_plt.bar.call_count == 2  # Two series
        mock_plt.title.assert_called_with("Comparison Chart")
        mock_plt.ylabel.assert_called_with("Score (%)")
        mock_plt.xlabel.assert_called_with("Categories")
        mock_plt.legend.assert_called_once()

        assert result is None  # No output path specified

    def test_create_comparison_chart_no_matplotlib(self):
        """Test comparison chart creation when matplotlib is not available."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'matplotlib'"),
        ):
            result = create_comparison_chart(
                categories=["Cat1"], series_data={"Series1": [80.0]}, title="Test Chart"
            )
            assert result is None


class TestTrendChart:
    """Test trend chart creation functionality."""

    @patch("matplotlib.pyplot")
    def test_create_trend_chart_basic(self, mock_plt):
        """Test basic trend chart creation."""
        result = create_trend_chart(
            x_values=[1, 2, 3, 4],
            y_values=[70.0, 80.0, 85.0, 90.0],
            title="Trend Chart",
            x_label="Time",
            y_label="Performance",
        )

        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        mock_plt.plot.assert_called_once()
        mock_plt.scatter.assert_called_once()
        mock_plt.title.assert_called_with("Trend Chart")
        mock_plt.xlabel.assert_called_with("Time")
        mock_plt.ylabel.assert_called_with("Performance")
        mock_plt.ylim.assert_called_with(0, 105)

        assert result is None  # No output path specified

    def test_create_trend_chart_no_matplotlib(self):
        """Test trend chart creation when matplotlib is not available."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'matplotlib'"),
        ):
            result = create_trend_chart(
                x_values=[1, 2, 3], y_values=[70.0, 80.0, 90.0], title="Test Chart"
            )
            assert result is None


class TestQualityDistributionChart:
    """Test quality distribution chart creation functionality."""

    @patch("matplotlib.pyplot")
    @patch("cursus.validation.shared.chart_utils.np")
    def test_create_quality_distribution_chart_basic(self, mock_np, mock_plt):
        """Test basic quality distribution chart creation."""
        # Mock numpy
        mock_np.mean.return_value = 75.0

        # Mock matplotlib components
        mock_patches = [Mock() for _ in range(5)]
        mock_plt.hist.return_value = (
            [2, 3, 4, 2, 1],
            [0, 60, 70, 80, 90, 100],
            mock_patches,
        )

        scores = [
            65.0,
            75.0,
            85.0,
            95.0,
            55.0,
            78.0,
            82.0,
            88.0,
            92.0,
            45.0,
            67.0,
            73.0,
        ]

        result = create_quality_distribution_chart(
            scores=scores, title="Quality Distribution"
        )

        # Verify matplotlib calls
        mock_plt.figure.assert_called_once()
        mock_plt.hist.assert_called_once()
        mock_plt.axvline.assert_called_with(
            75.0, color="red", linestyle="--", alpha=0.8
        )
        mock_plt.title.assert_called_with("Quality Distribution")
        mock_plt.xlabel.assert_called_with("Score Range")
        mock_plt.ylabel.assert_called_with("Count")

        # Verify patches are colored
        for patch in mock_patches:
            patch.set_facecolor.assert_called_once()

        assert result is None  # No output path specified

    def test_create_quality_distribution_chart_no_matplotlib(self):
        """Test quality distribution chart when matplotlib is not available."""
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named 'matplotlib'"),
        ):
            result = create_quality_distribution_chart(
                scores=[70.0, 80.0, 90.0], title="Test Chart"
            )
            assert result is None


class TestChartConfiguration:
    """Test chart configuration functionality."""

    def test_default_chart_config_structure(self):
        """Test that default chart config has expected structure."""
        assert "figure_size" in DEFAULT_CHART_CONFIG
        assert "colors" in DEFAULT_CHART_CONFIG
        assert "grid_style" in DEFAULT_CHART_CONFIG
        assert "dpi" in DEFAULT_CHART_CONFIG
        assert "bbox_inches" in DEFAULT_CHART_CONFIG

        # Test colors
        colors = DEFAULT_CHART_CONFIG["colors"]
        expected_colors = ["excellent", "good", "satisfactory", "needs_work", "poor"]
        for color_key in expected_colors:
            assert color_key in colors
            assert isinstance(colors[color_key], str)
            assert colors[color_key].startswith("#")

    def test_quality_thresholds_structure(self):
        """Test that quality thresholds are properly structured."""
        assert isinstance(QUALITY_THRESHOLDS, dict)
        assert len(QUALITY_THRESHOLDS) == 5

        # Test threshold values
        expected_thresholds = [90, 80, 70, 60, 0]
        for threshold in expected_thresholds:
            assert threshold in QUALITY_THRESHOLDS

        # Test threshold mappings
        assert QUALITY_THRESHOLDS[90] == "excellent"
        assert QUALITY_THRESHOLDS[80] == "good"
        assert QUALITY_THRESHOLDS[70] == "satisfactory"
        assert QUALITY_THRESHOLDS[60] == "needs_work"
        assert QUALITY_THRESHOLDS[0] == "poor"


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        # Test empty levels and scores
        with patch("matplotlib.pyplot") as mock_plt:
            mock_plt.bar.return_value = []

            result = create_score_bar_chart(levels=[], scores=[], title="Empty Chart")

            # Should still attempt to create chart
            mock_plt.figure.assert_called_once()

    def test_mismatched_data_lengths(self):
        """Test handling of mismatched data lengths."""
        with patch("matplotlib.pyplot") as mock_plt:
            # This should be handled by matplotlib, but we test our code doesn't crash
            create_score_bar_chart(
                levels=["Level1", "Level2"],
                scores=[80.0],  # Mismatched length
                title="Mismatched Chart",
            )

            mock_plt.figure.assert_called_once()

    def test_invalid_scores(self):
        """Test handling of invalid score values."""
        # Test with None values
        color = get_quality_color(None)
        assert color == DEFAULT_CHART_CONFIG["colors"]["poor"]

        # Test with string values (should handle gracefully)
        try:
            color = get_quality_color("invalid")
            # If it doesn't crash, it should return poor color
            assert color == DEFAULT_CHART_CONFIG["colors"]["poor"]
        except (TypeError, ValueError):
            # It's also acceptable to raise an exception
            pass

    def test_file_path_edge_cases(self):
        """Test file path edge cases."""
        with patch("matplotlib.pyplot") as mock_plt:
            with patch("cursus.validation.shared.chart_utils.Path") as mock_path:
                # Test with None output path
                result = create_score_bar_chart(
                    levels=["Level1"],
                    scores=[80.0],
                    title="Test Chart",
                    output_path=None,
                )

                assert result is None
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
        assert color == "#custom_green"

        # For missing color, should fall back to default or handle gracefully
        color = get_quality_color(50.0, partial_config)
        # Should either use default poor color or custom poor color if available
        assert color in [
            DEFAULT_CHART_CONFIG["colors"]["poor"],
            partial_config.get("colors", {}).get(
                "poor", DEFAULT_CHART_CONFIG["colors"]["poor"]
            ),
        ]


if __name__ == "__main__":
    pytest.main([__file__])
