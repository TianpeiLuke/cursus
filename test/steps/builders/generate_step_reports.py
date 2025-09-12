"""
Comprehensive report generator for all step builder types.

This script generates detailed reports and score charts for Training, Transform, 
CreateModel, and Processing step builders, creating subfolder structures with 
canonical step builder names.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import importlib

from cursus.validation.builders.universal_test import UniversalStepBuilderTest
from cursus.validation.builders.registry_discovery import (
    get_training_steps_from_registry,
    get_transform_steps_from_registry,
    get_createmodel_steps_from_registry,
    get_processing_steps_from_registry,
    load_builder_class,
)

# Import step-type-specific test frameworks
try:
    from cursus.validation.builders.variants.processing_test import (
        ProcessingStepBuilderTest,
    )
except ImportError:
    ProcessingStepBuilderTest = None
    print(
        "Warning: ProcessingStepBuilderTest not available, using UniversalStepBuilderTest"
    )


class StepBuilderReportGenerator:
    """
    Generates comprehensive reports and score charts for step builders.

    Creates subfolder structure following the pattern:
    test/steps/builders/{step_type_lowercase}/{canonical_step_name}/scoring_reports/
    """

    def __init__(self, base_path: str = None):
        """Initialize the report generator."""
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        self.reports_base = self.base_path / "reports"

        # Ensure base reports directory exists
        self.reports_base.mkdir(exist_ok=True)

        # Step type configurations
        self.step_types = {
            "training": {
                "get_steps_func": get_training_steps_from_registry,
                "display_name": "Training",
                "color": "#FF6B6B",  # Red
            },
            "transform": {
                "get_steps_func": get_transform_steps_from_registry,
                "display_name": "Transform",
                "color": "#4ECDC4",  # Teal
            },
            "createmodel": {
                "get_steps_func": get_createmodel_steps_from_registry,
                "display_name": "CreateModel",
                "color": "#45B7D1",  # Blue
            },
            "processing": {
                "get_steps_func": get_processing_steps_from_registry,
                "display_name": "Processing",
                "color": "#96CEB4",  # Green
            },
        }

    def get_canonical_step_name(self, step_name: str) -> str:
        """
        Get canonical step name by loading the builder class and getting its name.

        Args:
            step_name: Registry step name (e.g., "PyTorchTraining")

        Returns:
            Canonical builder class name (e.g., "PyTorchTrainingStepBuilder")
        """
        try:
            builder_class = load_builder_class(step_name)
            if builder_class:
                return builder_class.__name__
            else:
                # Fallback to step name + "StepBuilder"
                return f"{step_name}StepBuilder"
        except Exception as e:
            print(f"Warning: Could not get canonical name for {step_name}: {e}")
            return f"{step_name}StepBuilder"

    def create_step_subfolder(self, step_type: str, step_name: str) -> Path:
        """
        Create subfolder structure for a specific step.

        Args:
            step_type: Type of step (training, transform, etc.)
            step_name: Registry step name

        Returns:
            Path to the scoring_reports directory
        """
        canonical_name = self.get_canonical_step_name(step_name)

        # Create directory structure: {step_type}/{canonical_name}/scoring_reports/
        step_dir = self.base_path / step_type.lower() / canonical_name
        scoring_dir = step_dir / "scoring_reports"

        # Create directories
        step_dir.mkdir(parents=True, exist_ok=True)
        scoring_dir.mkdir(exist_ok=True)

        # Create README for the step directory
        readme_path = step_dir / "README.md"
        if not readme_path.exists():
            self._create_step_readme(readme_path, step_name, canonical_name, step_type)

        return scoring_dir

    def _create_step_readme(
        self, readme_path: Path, step_name: str, canonical_name: str, step_type: str
    ):
        """Create README file for a step directory."""
        content = f"""# {canonical_name} Test Reports

This directory contains test reports and scoring charts for the {step_name} step builder.

## Directory Structure

- `scoring_reports/` - Contains detailed test scoring reports and charts
  - `{canonical_name}_score_report.json` - Detailed test results in JSON format
  - `{canonical_name}_score_chart.png` - Visual score chart

## Step Information

- **Registry Name**: {step_name}
- **Builder Class**: {canonical_name}
- **Step Type**: {step_type.title()}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

The scoring reports are generated automatically by running:

```bash
python test/steps/builders/generate_step_reports.py
```

Or for specific step types:

```bash
python test/steps/builders/generate_step_reports.py --step-type {step_type}
```
"""
        readme_path.write_text(content)

    def run_step_tests(self, step_name: str, step_type: str) -> Dict[str, Any]:
        """
        Run comprehensive tests for a step builder.

        Args:
            step_name: Registry step name
            step_type: Type of step

        Returns:
            Dictionary containing test results and metadata
        """
        try:
            builder_class = load_builder_class(step_name)
            if not builder_class:
                return {
                    "error": f"Could not load builder class for {step_name}",
                    "step_name": step_name,
                    "step_type": step_type,
                    "timestamp": datetime.now().isoformat(),
                }

            # Choose appropriate test framework based on step type
            if step_type == "processing" and ProcessingStepBuilderTest is not None:
                # Use processing-specific test framework with Pattern B auto-pass logic
                tester = ProcessingStepBuilderTest(
                    builder_class, enable_scoring=True, enable_structured_reporting=True
                )
                results = tester.run_processing_validation()
            else:
                # Use universal test framework for other step types
                tester = UniversalStepBuilderTest(
                    builder_class,
                    verbose=False,
                    enable_scoring=True,
                    enable_structured_reporting=True,
                )
                results = tester.run_all_tests()

            # Add metadata
            results.update(
                {
                    "step_name": step_name,
                    "step_type": step_type,
                    "builder_class_name": builder_class.__name__,
                    "timestamp": datetime.now().isoformat(),
                    "generator_version": "1.0.0",
                }
            )

            return results

        except Exception as e:
            return {
                "error": f"Error testing {step_name}: {str(e)}",
                "step_name": step_name,
                "step_type": step_type,
                "timestamp": datetime.now().isoformat(),
            }

    def generate_score_chart(self, results: Dict[str, Any], output_path: Path):
        """
        Generate a visual score chart for test results.

        Args:
            results: Test results dictionary
            output_path: Path to save the chart
        """
        try:
            # Extract test results
            test_results = results.get("test_results", {})
            if not test_results:
                # Fallback for older format
                test_results = {
                    k: v
                    for k, v in results.items()
                    if isinstance(v, dict) and "passed" in v
                }

            if not test_results:
                print(f"Warning: No test results found for chart generation")
                return

            # Calculate scores
            total_tests = len(test_results)
            passed_tests = sum(
                1 for result in test_results.values() if result.get("passed", False)
            )
            pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Pie chart for pass/fail ratio
            labels = ["Passed", "Failed"]
            sizes = [passed_tests, total_tests - passed_tests]
            colors = ["#2ECC71", "#E74C3C"]  # Green for pass, red for fail

            ax1.pie(
                sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90
            )
            ax1.set_title(
                f'Test Results Overview\n{results.get("step_name", "Unknown")}'
            )

            # Bar chart for individual test results
            test_names = list(test_results.keys())
            test_scores = [
                1 if result.get("passed", False) else 0
                for result in test_results.values()
            ]

            # Truncate long test names for display
            display_names = [
                name.replace("test_", "").replace("_", " ").title()[:20]
                for name in test_names
            ]

            bars = ax2.bar(
                range(len(test_names)),
                test_scores,
                color=["#2ECC71" if score == 1 else "#E74C3C" for score in test_scores],
            )

            ax2.set_xlabel("Tests")
            ax2.set_ylabel("Pass (1) / Fail (0)")
            ax2.set_title(f"Individual Test Results\nPass Rate: {pass_rate:.1f}%")
            ax2.set_xticks(range(len(test_names)))
            ax2.set_xticklabels(display_names, rotation=45, ha="right")
            ax2.set_ylim(0, 1.2)

            # Add value labels on bars
            for i, (bar, score) in enumerate(zip(bars, test_scores)):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.05,
                    "✓" if score == 1 else "✗",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold",
                )

            # Add metadata
            step_name = results.get("step_name", "Unknown")
            builder_class = results.get("builder_class_name", "Unknown")
            timestamp = results.get("timestamp", "Unknown")

            fig.suptitle(
                f"{builder_class} Test Results", fontsize=16, fontweight="bold"
            )
            fig.text(0.02, 0.02, f"Generated: {timestamp}", fontsize=8, alpha=0.7)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"✅ Generated score chart: {output_path}")

        except Exception as e:
            print(f"❌ Error generating score chart: {e}")

    def generate_reports_for_step_type(self, step_type: str) -> Dict[str, Any]:
        """
        Generate reports for all steps of a given type.

        Args:
            step_type: Type of step (training, transform, createmodel, processing)

        Returns:
            Summary of generated reports
        """
        if step_type not in self.step_types:
            raise ValueError(f"Unknown step type: {step_type}")

        config = self.step_types[step_type]
        get_steps_func = config["get_steps_func"]

        print(f"\n{'='*60}")
        print(f"Generating {config['display_name']} Step Builder Reports")
        print(f"{'='*60}")

        # Get all steps for this type
        steps = get_steps_func()
        if not steps:
            print(f"No {step_type} steps found in registry")
            return {"step_type": step_type, "steps_processed": 0, "errors": []}

        print(f"Found {len(steps)} {step_type} steps: {', '.join(steps)}")

        summary = {
            "step_type": step_type,
            "steps_processed": 0,
            "successful": [],
            "errors": [],
            "timestamp": datetime.now().isoformat(),
        }

        for step_name in steps:
            print(f"\nProcessing {step_name}...")

            try:
                # Create subfolder structure
                scoring_dir = self.create_step_subfolder(step_type, step_name)
                canonical_name = self.get_canonical_step_name(step_name)

                # Run tests
                results = self.run_step_tests(step_name, step_type)

                if "error" in results:
                    print(f"❌ Error testing {step_name}: {results['error']}")
                    summary["errors"].append(
                        {"step_name": step_name, "error": results["error"]}
                    )
                    continue

                # Generate JSON report (always overwrite)
                json_path = scoring_dir / f"{canonical_name}_score_report.json"
                with open(json_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"✅ Generated JSON report: {json_path}")

                # Generate score chart (always overwrite)
                chart_path = scoring_dir / f"{canonical_name}_score_chart.png"
                self.generate_score_chart(results, chart_path)

                summary["successful"].append(
                    {
                        "step_name": step_name,
                        "canonical_name": canonical_name,
                        "json_report": str(json_path),
                        "score_chart": str(chart_path),
                    }
                )
                summary["steps_processed"] += 1

            except Exception as e:
                error_msg = f"Error processing {step_name}: {str(e)}"
                print(f"❌ {error_msg}")
                summary["errors"].append({"step_name": step_name, "error": error_msg})

        return summary

    def generate_all_reports(self) -> Dict[str, Any]:
        """
        Generate reports for all step types.

        Returns:
            Overall summary of all generated reports
        """
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE STEP BUILDER REPORTS")
        print(f"{'='*80}")
        print(f"Base directory: {self.base_path}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        overall_summary = {
            "timestamp": datetime.now().isoformat(),
            "base_path": str(self.base_path),
            "step_types": {},
            "total_steps_processed": 0,
            "total_successful": 0,
            "total_errors": 0,
        }

        for step_type in self.step_types.keys():
            try:
                summary = self.generate_reports_for_step_type(step_type)
                overall_summary["step_types"][step_type] = summary
                overall_summary["total_steps_processed"] += summary["steps_processed"]
                overall_summary["total_successful"] += len(summary["successful"])
                overall_summary["total_errors"] += len(summary["errors"])

            except Exception as e:
                error_msg = f"Error processing {step_type} step type: {str(e)}"
                print(f"❌ {error_msg}")
                overall_summary["step_types"][step_type] = {
                    "error": error_msg,
                    "steps_processed": 0,
                }
                overall_summary["total_errors"] += 1

        # Generate overall summary report
        summary_path = self.reports_base / "overall_summary.json"
        with open(summary_path, "w") as f:
            json.dump(overall_summary, f, indent=2)

        print(f"\n{'='*80}")
        print("REPORT GENERATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total steps processed: {overall_summary['total_steps_processed']}")
        print(f"Successful: {overall_summary['total_successful']}")
        print(f"Errors: {overall_summary['total_errors']}")
        print(f"Overall summary saved: {summary_path}")
        print(f"{'='*80}")

        return overall_summary


def main():
    """Main function to run report generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate step builder test reports")
    parser.add_argument(
        "--step-type",
        choices=["training", "transform", "createmodel", "processing", "all"],
        default="all",
        help="Step type to generate reports for (default: all)",
    )
    parser.add_argument(
        "--base-path",
        help="Base path for report generation (default: current directory)",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = StepBuilderReportGenerator(args.base_path)

    try:
        if args.step_type == "all":
            generator.generate_all_reports()
        else:
            generator.generate_reports_for_step_type(args.step_type)

    except Exception as e:
        print(f"❌ Error during report generation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
