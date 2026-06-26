#!/usr/bin/env python3
"""
Command-line interface for the Unified Alignment Tester.

This CLI provides comprehensive alignment validation across all four levels:
1. Script ↔ Contract Alignment
2. Contract ↔ Specification Alignment
3. Specification ↔ Dependencies Alignment
4. Builder ↔ Configuration Alignment

The CLI supports both individual script validation and batch validation of all scripts.
Updated to work with the refactored validation system and step catalog integration.
"""

import sys
import json
import click
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from ..validation.alignment.unified_alignment_tester import UnifiedAlignmentTester
# Note: AlignmentScorer removed in refactored system - scoring integrated into UnifiedAlignmentTester


def print_validation_summary(
    results: Dict[str, Any], verbose: bool = False, show_scoring: bool = False
) -> None:
    """Print validation results in a formatted way."""
    script_name = results.get("step_name", results.get("script_name", "Unknown"))
    status = results.get("overall_status", "UNKNOWN")

    # Status emoji and color - fix status value mapping
    if status == "PASSED":
        status_emoji = "✅"
        status_color = "green"
    elif status == "FAILED":
        status_emoji = "❌"
        status_color = "red"
    elif status == "EXCLUDED":
        status_emoji = "⚠️"
        status_color = "yellow"
    else:
        status_emoji = "⚠️"
        status_color = "yellow"

    click.echo(f"\n{status_emoji} {script_name}: ", nl=False)
    click.secho(status, fg=status_color, bold=True)

    # Show scoring information if requested
    if show_scoring and "scoring" in results:
        try:
            scoring_info = results["scoring"]
            overall_score = scoring_info.get("overall_score", 0.0)
            quality_rating = scoring_info.get("quality_rating", "Unknown")
            level_scores = scoring_info.get("level_scores", {})

            # Color-code the quality rating
            rating_colors = {
                "Excellent": "green",
                "Good": "green",
                "Satisfactory": "yellow",
                "Needs Work": "yellow",
                "Poor": "red",
            }
            rating_color = rating_colors.get(quality_rating, "white")

            click.echo(f"📊 Overall Score: ", nl=False)
            click.secho(
                f"{overall_score:.1f}/100", fg=rating_color, bold=True, nl=False
            )
            click.echo(f" (", nl=False)
            click.secho(quality_rating, fg=rating_color, bold=True, nl=False)
            click.echo(")")

            if verbose:
                click.echo("📈 Level Scores:")
                level_names = {
                    "level1_script_contract": "Level 1 (Script ↔ Contract)",
                    "level2_contract_spec": "Level 2 (Contract ↔ Specification)",
                    "level3_spec_dependencies": "Level 3 (Specification ↔ Dependencies)",
                    "level4_builder_config": "Level 4 (Builder ↔ Configuration)",
                }

                for level_key, score in level_scores.items():
                    level_name = level_names.get(level_key, level_key)
                    click.echo(f"  • {level_name}: {score:.1f}/100")

        except Exception as e:
            if verbose:
                click.echo(f"⚠️  Could not display scoring: {e}")

    # Print level-by-level results - fix to match actual implementation structure
    level_names = [
        "Script ↔ Contract",
        "Contract ↔ Specification",
        "Specification ↔ Dependencies",
        "Builder ↔ Configuration",
    ]

    # Get validation results from the actual structure
    validation_results = results.get("validation_results", {})

    for level_num, level_name in enumerate(level_names, 1):
        level_key = f"level_{level_num}"  # Implementation uses level_1, level_2, etc.
        level_result = validation_results.get(level_key, {})

        # Check level status - implementation uses "status" field
        level_status = level_result.get("status", "UNKNOWN")
        level_passed = level_status not in ["ERROR", "FAILED"]

        # Get issues/errors from level result
        level_issues = []
        if level_result.get("error"):
            level_issues.append(
                {
                    "severity": "ERROR",
                    "message": level_result.get("error"),
                    "recommendation": "Check the validation logs for more details",
                }
            )

        level_emoji = "✅" if level_passed else "❌"
        issues_text = f" ({len(level_issues)} issues)" if level_issues else ""

        click.echo(f"  {level_emoji} Level {level_num} ({level_name}): ", nl=False)
        display_status = "PASS" if level_passed else "FAIL"
        level_color = "green" if level_passed else "red"
        click.secho(f"{display_status}{issues_text}", fg=level_color)

        # Show issues if verbose or if there are critical/error issues
        if verbose or any(
            issue.get("severity") in ["CRITICAL", "ERROR"] for issue in level_issues
        ):
            for issue in level_issues:
                severity = issue.get("severity", "INFO")
                message = issue.get("message", "No message")

                severity_colors = {
                    "CRITICAL": "red",
                    "ERROR": "red",
                    "WARNING": "yellow",
                    "INFO": "blue",
                }

                severity_color = severity_colors.get(severity, "white")
                click.echo(f"    • ", nl=False)
                click.secho(f"[{severity}]", fg=severity_color, nl=False)
                click.echo(f" {message}")

                # Show recommendation if available
                recommendation = issue.get("recommendation")
                if recommendation and verbose:
                    click.echo(f"      💡 {recommendation}")


def _make_json_serializable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serializable representation.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable representation
    """
    # Handle None
    if obj is None:
        return None

    # Handle basic JSON types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            # Ensure keys are strings
            str_key = str(key)
            result[str_key] = _make_json_serializable(value)
        return result

    # Handle sets - convert to sorted lists
    if isinstance(obj, set):
        return sorted([_make_json_serializable(item) for item in obj])

    # Handle Path objects
    if hasattr(obj, "__fspath__"):  # Path-like objects
        return str(obj)

    # Handle datetime objects
    if hasattr(obj, "isoformat"):
        return obj.isoformat()

    # Handle Enum objects
    if hasattr(obj, "value"):
        return obj.value

    # Handle type objects
    if isinstance(obj, type):
        return str(obj.__name__)

    # For everything else, try string conversion
    try:
        str_value = str(obj)
        # Avoid storing string representations of complex objects
        if "<" in str_value and ">" in str_value and "object at" in str_value:
            return f"<{type(obj).__name__}>"
        return str_value
    except Exception:
        return f"<{type(obj).__name__}>"


def save_report(
    script_name: str, results: Dict[str, Any], output_dir: Path, format: str = "json"
) -> None:
    """Save validation results to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "json":
        output_file = output_dir / f"{script_name}_alignment_report.json"
        try:
            # Clean the results to ensure JSON serializability
            cleaned_results = _make_json_serializable(results)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(cleaned_results, f, indent=2, default=str)
            click.echo(f"📄 JSON report saved: {output_file}")
        except Exception as e:
            click.echo(f"⚠️  Warning: Could not save JSON report for {script_name}: {e}")
            # Try to save a simplified version
            try:
                simplified_results = {
                    "script_name": script_name,
                    "overall_status": results.get("overall_status", "ERROR"),
                    "error": f"JSON serialization failed: {str(e)}",
                    "metadata": results.get("metadata", {}),
                }
                cleaned_simplified = _make_json_serializable(simplified_results)
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(cleaned_simplified, f, indent=2, default=str)
                click.echo(f"📄 Simplified JSON report saved: {output_file}")
            except Exception as e2:
                click.echo(
                    f"❌ Failed to save even simplified JSON report for {script_name}: {e2}"
                )


@click.group()
@click.pass_context
def alignment(ctx):
    """
    Unified Alignment Tester for Cursus Scripts.

    Validates alignment across all four levels:
    1. Script ↔ Contract Alignment
    2. Contract ↔ Specification Alignment
    3. Specification ↔ Dependencies Alignment
    4. Builder ↔ Configuration Alignment

    Updated to work with the refactored validation system and step catalog integration.
    """
    ctx.ensure_object(dict)


@alignment.command()
@click.argument("script_name")
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to include in validation",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for the JSON report",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--show-scoring", is_flag=True, help="Show alignment scoring information")
@click.pass_context
def validate(
    ctx,
    script_name,
    workspace_dirs,
    output_dir,
    verbose,
    show_scoring,
):
    """
    Validate alignment for a specific script.

    SCRIPT_NAME: Name of the script to validate (without .py extension)

    Example:
        cursus alignment validate currency_conversion --verbose
        cursus alignment validate dummy_training --output-dir ./reports
    """
    if verbose:
        click.echo(f"🔍 Validating script: {script_name}")
        if workspace_dirs:
            click.echo(f"📁 Workspace directories: {list(workspace_dirs)}")

    try:
        # Initialize the unified alignment tester with workspace support
        workspace_dir_list = list(workspace_dirs) if workspace_dirs else []
        tester = UnifiedAlignmentTester(workspace_dirs=workspace_dir_list)

        # Run validation
        results = tester.validate_specific_script(script_name)

        # Add metadata
        results["metadata"] = {
            "script_name": script_name,
            "validation_timestamp": datetime.now().isoformat(),
            "validator_version": "2.0.0",
            "workspace_dirs": workspace_dir_list or [],
        }

        # Print results
        print_validation_summary(results, verbose, show_scoring)

        # Save JSON report if output directory specified
        if output_dir:
            save_report(script_name, results, output_dir, "json")

        # Return appropriate exit code - fix status value check
        status = results.get("overall_status", "UNKNOWN")
        if status == "PASSED":
            click.echo(f"\n✅ {script_name} passed all alignment validation checks!")
            sys.exit(0)
        elif status == "EXCLUDED":
            click.echo(f"\n⚠️  {script_name} was excluded from validation.")
            sys.exit(0)  # Excluded is not a failure
        else:
            click.echo(
                f"\n❌ {script_name} failed alignment validation. Please review the issues above."
            )
            ctx.exit(1)

    except Exception as e:
        click.echo(f"❌ Error validating {script_name}: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        ctx.exit(1)


@alignment.command()
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to include in validation",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory for JSON reports",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--show-scoring", is_flag=True, help="Show alignment scoring information")
@click.option(
    "--continue-on-error",
    is_flag=True,
    help="Continue validation even if individual scripts fail",
)
@click.pass_context
def validate_all(
    ctx,
    workspace_dirs,
    output_dir,
    verbose,
    show_scoring,
    continue_on_error,
):
    """
    Validate alignment for all scripts discovered by the step catalog.

    Discovers all Python scripts using the step catalog (workspace-aware) and runs
    comprehensive alignment validation for each one, generating detailed reports.

    Example:
        cursus alignment validate-all --output-dir ./reports --verbose
    """
    try:
        click.echo("🚀 Starting Comprehensive Script Alignment Validation")
        if verbose:
            if workspace_dirs:
                click.echo(f"📁 Workspace directories: {list(workspace_dirs)}")
            if output_dir:
                click.echo(f"📁 Output directory: {output_dir}")

        # Initialize the unified alignment tester with workspace support
        workspace_dir_list = list(workspace_dirs) if workspace_dirs else []
        tester = UnifiedAlignmentTester(workspace_dirs=workspace_dir_list)

        # Discover ALL steps for comprehensive validation (not just scripts)
        # The validation system will intelligently skip levels based on step type
        all_steps = tester.step_catalog.list_available_steps()

        # Get breakdown for user information by checking which steps have script files
        scripts_with_files = []
        steps_without_scripts = []

        for step_name in all_steps:
            if tester._has_script_file(step_name):
                scripts_with_files.append(step_name)
            else:
                steps_without_scripts.append(step_name)

        click.echo(
            f"\n📋 Discovered {len(all_steps)} total steps for comprehensive validation:"
        )
        click.echo(
            f"  • {len(scripts_with_files)} steps with scripts (full 4-level validation)"
        )
        click.echo(
            f"  • {len(steps_without_scripts)} steps without scripts (intelligent level skipping)"
        )
        click.echo(f"\nAll steps: {', '.join(all_steps)}")

        # Use all steps for validation, not just scripts
        scripts = all_steps

        # Validation results summary
        validation_summary = {
            "total_steps": len(scripts),
            "passed_steps": 0,
            "failed_steps": 0,
            "error_steps": 0,
            "validation_timestamp": datetime.now().isoformat(),
            "step_results": {},
        }

        # Validate each script
        for script_name in scripts:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"🔍 VALIDATING SCRIPT: {script_name}")
            click.echo(f"{'=' * 60}")

            try:
                results = tester.validate_specific_script(script_name)

                # Add metadata
                results["metadata"] = {
                    "script_name": script_name,
                    "validation_timestamp": datetime.now().isoformat(),
                    "validator_version": "2.0.0",
                    "workspace_dirs": workspace_dir_list or [],
                }

                # Print results
                print_validation_summary(results, verbose, show_scoring)

                # Save JSON report if output directory specified
                if output_dir:
                    save_report(script_name, results, output_dir, "json")

                # Update summary - fix status value checks
                status = results.get("overall_status", "UNKNOWN")
                validation_summary["step_results"][script_name] = {
                    "status": status,
                    "timestamp": results.get("metadata", {}).get(
                        "validation_timestamp"
                    ),
                }

                if status == "PASSED":
                    validation_summary["passed_steps"] += 1
                elif status == "FAILED":
                    validation_summary["failed_steps"] += 1
                elif status == "EXCLUDED":
                    # Excluded steps don't count as errors
                    validation_summary["passed_steps"] += 1
                else:
                    validation_summary["error_steps"] += 1

            except Exception as e:
                click.echo(f"❌ Failed to validate {script_name}: {e}")
                validation_summary["error_steps"] += 1
                validation_summary["step_results"][script_name] = {
                    "status": "ERROR",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }

                if not continue_on_error:
                    click.echo(
                        "Stopping validation due to error. Use --continue-on-error to continue."
                    )
                    ctx.exit(1)

        # Save overall summary
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_file = output_dir / "validation_summary.json"
            try:
                cleaned_summary = _make_json_serializable(validation_summary)
                with open(summary_file, "w", encoding="utf-8") as f:
                    json.dump(cleaned_summary, f, indent=2, default=str)
                click.echo(f"\n📊 Validation summary saved: {summary_file}")
            except Exception as e:
                click.echo(f"⚠️  Warning: Could not save validation summary: {e}")

        # Print final summary
        click.echo(f"\n{'=' * 80}")
        click.echo("🎯 FINAL VALIDATION SUMMARY")
        click.echo(f"{'=' * 80}")

        total = validation_summary["total_steps"]
        passed = validation_summary["passed_steps"]
        failed = validation_summary["failed_steps"]
        errors = validation_summary["error_steps"]

        click.echo(f"📊 Total Steps: {total}")
        if total > 0:
            click.secho(
                f"✅ Passed: {passed} ({passed / total * 100:.1f}%)", fg="green"
            )
            click.secho(f"❌ Failed: {failed} ({failed / total * 100:.1f}%)", fg="red")
            click.secho(
                f"⚠️  Errors: {errors} ({errors / total * 100:.1f}%)", fg="yellow"
            )
        else:
            click.secho(f"✅ Passed: {passed} (0.0%)", fg="green")
            click.secho(f"❌ Failed: {failed} (0.0%)", fg="red")
            click.secho(f"⚠️  Errors: {errors} (0.0%)", fg="yellow")

        if output_dir:
            click.echo(f"\n📁 Reports saved in: {output_dir}")

        # Return appropriate exit code
        if total == 0:
            # Special case: no steps found should be success
            click.echo(f"\n🎉 All {passed} steps passed alignment validation!")
            return  # Exit successfully without raising exception
        elif failed > 0 or errors > 0:
            click.echo(f"\n⚠️  {failed + errors} step(s) failed validation.")
            ctx.exit(1)
        else:
            click.echo(f"\n🎉 All {passed} steps passed alignment validation!")
            return  # Exit successfully without raising exception

    except Exception as e:
        click.echo(f"❌ Fatal error during validation: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        ctx.exit(1)


@alignment.command()
@click.argument("script_name")
@click.argument("level", type=click.IntRange(1, 4))
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to include in validation",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.pass_context
def validate_level(
    ctx,
    script_name,
    level,
    workspace_dirs,
    verbose,
):
    """
    Validate alignment for a specific script at a specific level.

    SCRIPT_NAME: Name of the script to validate (without .py extension)
    LEVEL: Validation level (1=Script↔Contract, 2=Contract↔Spec, 3=Spec↔Deps, 4=Builder↔Config)

    Example:
        cursus alignment validate-level currency_conversion 1 --verbose
        cursus alignment validate-level dummy_training 3
    """
    level_names = {
        1: "Script ↔ Contract",
        2: "Contract ↔ Specification",
        3: "Specification ↔ Dependencies",
        4: "Builder ↔ Configuration",
    }

    try:
        click.echo(
            f"🔍 Validating {script_name} at Level {level} ({level_names[level]})"
        )

        if verbose:
            if workspace_dirs:
                click.echo(f"📁 Workspace directories: {list(workspace_dirs)}")

        # Initialize the unified alignment tester with workspace support
        workspace_dir_list = list(workspace_dirs) if workspace_dirs else []
        tester = UnifiedAlignmentTester(workspace_dirs=workspace_dir_list)

        # Run validation for specific level
        results = tester.validate_specific_script(script_name)

        # Extract level-specific results - fix to match implementation structure
        validation_results = results.get("validation_results", {})
        level_key = f"level_{level}"  # Implementation uses level_1, level_2, etc.
        level_result = validation_results.get(level_key, {})

        # Check level status from implementation structure
        level_status = level_result.get("status", "UNKNOWN")
        level_passed = level_status not in ["ERROR", "FAILED"]

        # Get issues from level result
        level_issues = []
        if level_result.get("error"):
            level_issues.append(
                {
                    "severity": "ERROR",
                    "message": level_result.get("error"),
                    "recommendation": "Check the validation logs for more details",
                }
            )

        # Print level-specific results
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Level {level} ({level_names[level]}) Results")
        click.echo(f"{'=' * 60}")

        status_emoji = "✅" if level_passed else "❌"
        status_text = "PASSED" if level_passed else "FAILED"
        status_color = "green" if level_passed else "red"

        click.echo(f"{status_emoji} Status: ", nl=False)
        click.secho(status_text, fg=status_color, bold=True)

        if level_issues:
            click.echo(f"\n📋 Issues ({len(level_issues)}):")
            for issue in level_issues:
                severity = issue.get("severity", "INFO")
                message = issue.get("message", "No message")

                severity_colors = {
                    "CRITICAL": "red",
                    "ERROR": "red",
                    "WARNING": "yellow",
                    "INFO": "blue",
                }

                severity_color = severity_colors.get(severity, "white")
                click.echo(f"  • ", nl=False)
                click.secho(f"[{severity}]", fg=severity_color, nl=False)
                click.echo(f" {message}")

                # Show recommendation if available
                recommendation = issue.get("recommendation")
                if recommendation and verbose:
                    click.echo(f"    💡 {recommendation}")
        else:
            click.echo("\n✅ No issues found!")

        # Return appropriate exit code
        if level_passed:
            click.echo(f"\n✅ {script_name} passed Level {level} validation!")
            sys.exit(0)
        else:
            click.echo(f"\n❌ {script_name} failed Level {level} validation.")
            ctx.exit(1)

    except Exception as e:
        click.echo(f"❌ Error validating {script_name} at Level {level}: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        ctx.exit(1)


@alignment.command()
@click.option(
    "--workspace-dirs",
    multiple=True,
    type=click.Path(exists=True),
    help="Workspace directories to include in validation",
)
@click.pass_context
def list_scripts(ctx, workspace_dirs):
    """
    List all available scripts that can be validated.

    Discovers all Python scripts using the step catalog (workspace-aware).

    Example:
        cursus alignment list-scripts
        cursus alignment list-scripts --workspace-dirs /path/to/workspace
    """
    try:
        click.echo("📋 Available Scripts for Alignment Validation:")
        click.echo("=" * 50)

        # Initialize the unified alignment tester with workspace support
        workspace_dir_list = list(workspace_dirs) if workspace_dirs else []
        tester = UnifiedAlignmentTester(workspace_dirs=workspace_dir_list)

        # Discover all scripts using step catalog (workspace-aware)
        scripts = tester.discover_scripts()

        if scripts:
            for script in scripts:
                click.echo(f"  • {script}")

            click.echo(f"\nTotal: {len(scripts)} scripts found")
            if workspace_dirs:
                click.echo(f"Workspace directories: {list(workspace_dirs)}")

            click.echo(f"\nUsage examples:")
            click.echo(
                f"  cursus alignment validate {scripts[0]} --verbose --show-scoring"
            )
            click.echo(f"  cursus alignment validate-all --output-dir ./reports")
            click.echo(f"  cursus alignment validate-level {scripts[0]} 1")
        else:
            click.echo("  No scripts found.")
            if workspace_dirs:
                click.echo(
                    f"  Searched in workspace directories: {list(workspace_dirs)}"
                )
            else:
                click.echo("  Searched in package-only mode.")

    except Exception as e:
        click.echo(f"❌ Error listing scripts: {e}", err=True)
        ctx.exit(1)


def main():
    """Main entry point for alignment CLI."""
    alignment()


if __name__ == "__main__":
    main()
