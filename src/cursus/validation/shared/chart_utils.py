"""
Shared chart generation utilities for validation systems.

This module provides common chart generation functionality that can be used
by both alignment validation and step builder testing systems.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# Common chart styling configuration
DEFAULT_CHART_CONFIG = {
    "figure_size": (10, 6),
    "colors": {
        "excellent": "#28a745",  # Green
        "good": "#90ee90",  # Light green
        "satisfactory": "#ffa500",  # Orange
        "needs_work": "#fa8072",  # Salmon
        "poor": "#dc3545",  # Red
    },
    "grid_style": {"axis": "y", "linestyle": "--", "alpha": 0.7},
    "dpi": 300,
    "bbox_inches": "tight",
}

# Quality thresholds for color mapping
QUALITY_THRESHOLDS = {
    90: "excellent",
    80: "good",
    70: "satisfactory",
    60: "needs_work",
    0: "poor",
}


def get_quality_color(score: float, config: Dict[str, Any] = None) -> str:
    """
    Get color for a score based on quality thresholds.

    Args:
        score: Score value (0-100)
        config: Chart configuration (uses default if None)

    Returns:
        Color string for the score
    """
    if config is None:
        config = DEFAULT_CHART_CONFIG

    colors = config.get("colors", DEFAULT_CHART_CONFIG["colors"])

    # Handle None scores
    if score is None:
        return colors.get("poor", DEFAULT_CHART_CONFIG["colors"]["poor"])

    for threshold, quality in sorted(QUALITY_THRESHOLDS.items(), reverse=True):
        if score >= threshold:
            return colors.get(
                quality, colors.get("poor", DEFAULT_CHART_CONFIG["colors"]["poor"])
            )

    return colors.get("poor", DEFAULT_CHART_CONFIG["colors"]["poor"])


def get_quality_rating(score: float) -> str:
    """
    Get quality rating text for a score.

    Args:
        score: Score value (0-100)

    Returns:
        Quality rating string
    """
    for threshold, quality in sorted(QUALITY_THRESHOLDS.items(), reverse=True):
        if score >= threshold:
            return quality.replace("_", " ").title()

    return "Poor"


def create_score_bar_chart(
    levels: List[str],
    scores: List[float],
    title: str,
    overall_score: float = None,
    overall_rating: str = None,
    output_path: str = None,
    config: Dict[str, Any] = None,
) -> Optional[str]:
    """
    Create a bar chart for scores across different levels.

    Args:
        levels: List of level names
        scores: List of scores corresponding to levels
        title: Chart title
        overall_score: Overall score to show as horizontal line
        overall_rating: Overall rating text
        output_path: Path to save the chart (if None, uses default naming)
        config: Chart configuration (uses default if None)

    Returns:
        Path to saved chart or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return None

    if config is None:
        config = DEFAULT_CHART_CONFIG

    try:
        # Create colors for each score
        colors = [get_quality_color(score, config) for score in scores]

        # Create the figure
        plt.figure(figsize=config.get("figure_size", (10, 6)))

        # Create bar chart
        bars = plt.bar(levels, scores, color=colors)

        # Add overall score line if provided
        if overall_score is not None:
            plt.axhline(y=overall_score, color="blue", linestyle="-", alpha=0.7)

            # Add overall score text
            text = f"Overall: {overall_score:.1f}"
            if overall_rating:
                text += f" ({overall_rating})"

            plt.text(len(levels) - 0.5, overall_score + 2, text, color="blue")

        # Add score labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        # Set chart properties
        plt.title(title)
        plt.ylabel("Score (%)")
        plt.ylim(0, 105)

        # Apply grid styling
        grid_config = config.get("grid_style", DEFAULT_CHART_CONFIG["grid_style"])
        plt.grid(**grid_config)

        # Rotate x-axis labels for better readability if many levels
        if len(levels) > 3:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Save figure
        if output_path:
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            save_config = {
                "dpi": config.get("dpi", 300),
                "bbox_inches": config.get("bbox_inches", "tight"),
            }
            plt.savefig(output_path, **save_config)
            plt.close()

            return output_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Could not generate chart: {str(e)}")
        return None


def create_comparison_chart(
    categories: List[str],
    series_data: Dict[str, List[float]],
    title: str,
    output_path: str = None,
    config: Dict[str, Any] = None,
) -> Optional[str]:
    """
    Create a comparison chart with multiple data series.

    Args:
        categories: List of category names (x-axis)
        series_data: Dictionary mapping series names to score lists
        title: Chart title
        output_path: Path to save the chart
        config: Chart configuration

    Returns:
        Path to saved chart or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return None

    if config is None:
        config = DEFAULT_CHART_CONFIG

    try:
        # Create the figure
        plt.figure(figsize=config.get("figure_size", (12, 6)))

        # Set up bar positions
        x = np.arange(len(categories))
        width = 0.8 / len(series_data)  # Width of bars

        # Create bars for each series
        for i, (series_name, scores) in enumerate(series_data.items()):
            offset = (i - len(series_data) / 2 + 0.5) * width
            colors = [get_quality_color(score, config) for score in scores]

            bars = plt.bar(x + offset, scores, width, label=series_name, color=colors)

            # Add score labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Set chart properties
        plt.title(title)
        plt.ylabel("Score (%)")
        plt.xlabel("Categories")
        plt.xticks(x, categories, rotation=45, ha="right")
        plt.ylim(0, 105)
        plt.legend()

        # Apply grid styling
        grid_config = config.get("grid_style", DEFAULT_CHART_CONFIG["grid_style"])
        plt.grid(**grid_config)

        plt.tight_layout()

        # Save figure
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            save_config = {
                "dpi": config.get("dpi", 300),
                "bbox_inches": config.get("bbox_inches", "tight"),
            }
            plt.savefig(output_path, **save_config)
            plt.close()

            return output_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Could not generate comparison chart: {str(e)}")
        return None


def create_trend_chart(
    x_values: List[Any],
    y_values: List[float],
    title: str,
    x_label: str = "X",
    y_label: str = "Score (%)",
    output_path: str = None,
    config: Dict[str, Any] = None,
) -> Optional[str]:
    """
    Create a trend line chart.

    Args:
        x_values: List of x-axis values
        y_values: List of y-axis values (scores)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        output_path: Path to save the chart
        config: Chart configuration

    Returns:
        Path to saved chart or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return None

    if config is None:
        config = DEFAULT_CHART_CONFIG

    try:
        # Create the figure
        plt.figure(figsize=config.get("figure_size", (10, 6)))

        # Create line plot
        plt.plot(x_values, y_values, marker="o", linewidth=2, markersize=6)

        # Color points based on quality
        colors = [get_quality_color(score, config) for score in y_values]
        plt.scatter(x_values, y_values, c=colors, s=50, zorder=5)

        # Set chart properties
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(0, 105)

        # Apply grid styling
        grid_config = config.get("grid_style", DEFAULT_CHART_CONFIG["grid_style"])
        plt.grid(**grid_config)

        plt.tight_layout()

        # Save figure
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            save_config = {
                "dpi": config.get("dpi", 300),
                "bbox_inches": config.get("bbox_inches", "tight"),
            }
            plt.savefig(output_path, **save_config)
            plt.close()

            return output_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Could not generate trend chart: {str(e)}")
        return None


def create_quality_distribution_chart(
    scores: List[float],
    title: str,
    output_path: str = None,
    config: Dict[str, Any] = None,
) -> Optional[str]:
    """
    Create a histogram showing the distribution of quality scores.

    Args:
        scores: List of scores to analyze
        title: Chart title
        output_path: Path to save the chart
        config: Chart configuration

    Returns:
        Path to saved chart or None if matplotlib unavailable
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return None

    if config is None:
        config = DEFAULT_CHART_CONFIG

    try:
        # Create the figure
        plt.figure(figsize=config.get("figure_size", (10, 6)))

        # Create histogram with quality-based coloring
        bins = [0, 60, 70, 80, 90, 100]
        colors = [
            config["colors"]["poor"],
            config["colors"]["needs_work"],
            config["colors"]["satisfactory"],
            config["colors"]["good"],
            config["colors"]["excellent"],
        ]

        n, bins, patches = plt.hist(scores, bins=bins, edgecolor="black", alpha=0.7)

        # Color the bars
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)

        # Add statistics
        mean_score = np.mean(scores)
        plt.axvline(mean_score, color="red", linestyle="--", alpha=0.8)
        plt.text(
            mean_score + 2,
            max(n) * 0.8,
            f"Mean: {mean_score:.1f}",
            color="red",
            fontweight="bold",
        )

        # Set chart properties
        plt.title(title)
        plt.xlabel("Score Range")
        plt.ylabel("Count")

        # Add quality labels
        quality_labels = [
            "Poor\n(0-59)",
            "Needs Work\n(60-69)",
            "Satisfactory\n(70-79)",
            "Good\n(80-89)",
            "Excellent\n(90-100)",
        ]
        plt.xticks([30, 65, 75, 85, 95], quality_labels, rotation=45, ha="right")

        plt.tight_layout()

        # Save figure
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            save_config = {
                "dpi": config.get("dpi", 300),
                "bbox_inches": config.get("bbox_inches", "tight"),
            }
            plt.savefig(output_path, **save_config)
            plt.close()

            return output_path
        else:
            plt.show()
            return None

    except Exception as e:
        print(f"Could not generate quality distribution chart: {str(e)}")
        return None
