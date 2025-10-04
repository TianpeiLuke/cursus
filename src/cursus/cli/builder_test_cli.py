#!/usr/bin/env python3
"""
Command-line interface for the Universal Step Builder Test System.

This CLI provides easy access to run different levels of tests and variants
for step builder validation according to the UniversalStepBuilderTestBase architecture.
Enhanced with scoring, registry discovery, and export capabilities.

Updated to work with the refactored validation system and step catalog integration.
"""

import click
import sys
import importlib
import inspect
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Type

from ..validation.builders.universal_test import UniversalStepBuilderTest
from ..validation.builders.core.integration_tests import IntegrationTests
from ..validation.builders.reporting.scoring import StepBuilderScorer
from ..step_catalog import StepCatalog
from ..registry import STEP_NAMES


def print_test_results(
    results: Dict[str, Any], verbose: bool = False, show_scoring: bool = False
) -> None:
    """Print test results in a formatted way with optional scoring display."""
    if not results:
        click.echo("❌ No test results found!")
        return

    # Handle both legacy format (raw results) and new format (with scoring/reporting)
    if "test_results" in results:
        # New format with scoring/reporting
        test_results = results["test_results"]
        scoring_data = results.get("scoring")
        structured_report = results.get("structured_report")
    else:
        # Legacy format (raw results)
        test_results = results
        scoring_data = None
        structured_report = None

    # Calculate summary statistics
    total_tests = len(test_results)
    passed_tests = sum(
        1 for result in test_results.values() if result.get("passed", False)
    )
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    # Print summary header with optional scoring
    if scoring_data and show_scoring:
        overall_score = scoring_data.get("overall", {}).get("score", 0.0)
        overall_rating = scoring_data.get("overall", {}).get("rating", "Unknown")
        click.echo(
            f"\n📊 Test Results Summary: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)"
        )
        click.echo(f"🏆 Quality Score: {overall_score:.1f}/100 - {overall_rating}")
        click.echo("=" * 70)
    else:
        click.echo(
            f"\n📊 Test Results Summary: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)"
        )
        click.echo("=" * 60)

    # Group results by test level/type
    level_groups = {
        "Level 1 (Interface)": [],
        "Level 2 (Specification)": [],
        "Level 3 (Step Creation)": [],
        "Level 4 (Integration)": [],
        "Step Type Specific": [],
        "Other": [],
    }

    for test_name, result in test_results.items():
        if any(
            interface_test in test_name
            for interface_test in [
                "inheritance",
                "naming_conventions",
                "required_methods",
                "registry_integration",
                "documentation_standards",
                "type_hints",
                "error_handling",
                "method_return_types",
                "configuration_validation",
                "generic_step_creation",
                "generic_configuration",
            ]
        ):
            level_groups["Level 1 (Interface)"].append((test_name, result))
        elif any(
            spec_test in test_name
            for spec_test in [
                "specification_usage",
                "contract_alignment",
                "environment_variable_handling",
                "job_arguments",
                "environment_variables_processing",
                "property_files_configuration",
            ]
        ):
            level_groups["Level 2 (Specification)"].append((test_name, result))
        elif any(
            creation_test in test_name
            for creation_test in [
                "step_instantiation",
                "step_configuration_validity",
                "step_dependencies_attachment",
                "step_name_generation",
                "input_path_mapping",
                "output_path_mapping",
                "property_path_validity",
                "processing_inputs_outputs",
                "processing_code_handling",
            ]
        ):
            level_groups["Level 3 (Step Creation)"].append((test_name, result))
        elif any(
            integration_test in test_name
            for integration_test in [
                "dependency_resolution",
                "step_creation",
                "step_name",
                "generic_dependency_handling",
                "processing_step_dependencies",
            ]
        ):
            level_groups["Level 4 (Integration)"].append((test_name, result))
        elif any(
            step_type_test in test_name
            for step_type_test in [
                "step_type",
                "processing",
                "training",
                "transform",
                "create_model",
                "register_model",
                "processor_creation",
                "estimator_methods",
                "transformer_methods",
            ]
        ):
            level_groups["Step Type Specific"].append((test_name, result))
        else:
            level_groups["Other"].append((test_name, result))

    # Print results by group with optional level scoring
    for group_name, group_tests in level_groups.items():
        if not group_tests:
            continue

        group_passed = sum(
            1 for _, result in group_tests if result.get("passed", False)
        )
        group_total = len(group_tests)
        group_rate = (group_passed / group_total) * 100 if group_total > 0 else 0

        # Add level score if available
        level_score_text = ""
        if scoring_data and show_scoring:
            level_key = (
                group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            )
            level_mapping = {
                "level_1_interface": "level1_interface",
                "level_2_specification": "level2_specification",
                "level_3_step_creation": "level3_step_creation",
                "level_4_integration": "level4_integration",
            }
            mapped_key = level_mapping.get(level_key)
            if mapped_key and mapped_key in scoring_data.get("levels", {}):
                level_score = scoring_data["levels"][mapped_key].get("score", 0)
                level_score_text = f" - Score: {level_score:.1f}/100"

        click.echo(
            f"\n📁 {group_name}: {group_passed}/{group_total} passed ({group_rate:.1f}%){level_score_text}"
        )

        for test_name, result in group_tests:
            status = "✅" if result.get("passed", False) else "❌"
            click.echo(f"  {status} {test_name}")

            if not result.get("passed", False) and result.get("error"):
                click.echo(f"    💬 {result['error']}")

            if verbose and result.get("details"):
                click.echo(f"    📋 Details: {result['details']}")

    click.echo("\n" + "=" * (70 if show_scoring else 60))


def print_enhanced_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """Print enhanced results with scoring and structured reporting."""
    if "test_results" not in results:
        print_test_results(results, verbose)
        return

    test_results = results["test_results"]
    scoring_data = results.get("scoring")
    structured_report = results.get("structured_report")

    # Print test results with scoring
    print_test_results(results, verbose, show_scoring=True)

    # Print additional scoring details if available
    if scoring_data and verbose:
        click.echo("\n📈 Detailed Scoring Breakdown:")
        click.echo("-" * 50)

        levels = scoring_data.get("levels", {})
        for level_name, level_data in levels.items():
            display_name = (
                level_name.replace("level", "Level ").replace("_", " ").title()
            )
            score = level_data.get("score", 0)
            passed = level_data.get("passed", 0)
            total = level_data.get("total", 0)
            click.echo(f"  {display_name}: {score:.1f}/100 ({passed}/{total} tests)")

        # Show failed tests summary
        failed_tests = scoring_data.get("failed_tests", [])
        if failed_tests:
            click.echo(f"\n❌ Failed Tests Summary ({len(failed_tests)}):")
            for test in failed_tests[:5]:  # Show first 5 failed tests
                click.echo(f"  • {test['name']}: {test['error']}")
            if len(failed_tests) > 5:
                click.echo(f"  ... and {len(failed_tests) - 5} more failed tests")


def import_builder_class(class_path: str) -> Type:
    """Import a builder class from a module path."""
    try:
        # Split module path and class name
        if "." in class_path:
            module_path, class_name = class_path.rsplit(".", 1)
        else:
            # Assume it's just a class name in the current package
            module_path = "..steps.builders"
            class_name = class_path

        # Handle src. prefix - remove it for installed package
        if module_path.startswith("src."):
            module_path = module_path[4:]  # Remove 'src.' prefix
        
        # Convert absolute cursus imports to relative imports when within the package
        if module_path.startswith("cursus."):
            module_path = "." + module_path[6:]  # Convert cursus.* to .*

        # Import the module
        if module_path.startswith(".."):
            # For relative imports, we need to specify the package
            module = importlib.import_module(module_path, package=__package__)
        else:
            # For absolute imports
            module = importlib.import_module(module_path)

        # Get the class
        builder_class = getattr(module, class_name)

        return builder_class

    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(
            f"Could not find class {class_name} in module {module_path}: {e}"
        )


def export_results_to_json(results: Dict[str, Any], output_path: str) -> None:
    """Export test results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    click.echo(f"✅ Results exported to: {output_path}")


def generate_score_chart(
    results: Dict[str, Any], builder_name: str, output_dir: str
) -> Optional[str]:
    """Generate score visualization chart."""
    if "test_results" not in results:
        # Create scorer from raw results
        scorer = StepBuilderScorer(results)
    else:
        # Create scorer from test_results
        scorer = StepBuilderScorer(results["test_results"])

    return scorer.generate_chart(builder_name, output_dir)


@click.group(name="builder-test")
def builder_test():
    """Universal Step Builder Test System.

    Run different levels of tests and variants for step builder validation
    with enhanced scoring, registry discovery, and export capabilities.
    """
    pass


@builder_test.command("test-all")
@click.argument("builder_class")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output including test details and logs"
)
@click.option(
    "--scoring",
    is_flag=True,
    help="Enable quality scoring and enhanced reporting"
)
@click.option(
    "--export-json",
    type=click.Path(),
    help="Export test results to JSON file at specified path"
)
@click.option(
    "--export-chart",
    is_flag=True,
    help="Generate score visualization chart (requires matplotlib)"
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="test_reports",
    help="Output directory for exports (default: test_reports)"
)
def test_all(builder_class: str, verbose: bool, scoring: bool, export_json: str, export_chart: bool, output_dir: str):
    """Run all tests (universal test suite)."""
    try:
        click.echo(f"🔍 Importing builder class: {builder_class}")
        builder_cls = import_builder_class(builder_class)
        click.echo(f"✅ Successfully imported: {builder_cls.__name__}")

        click.echo(f"\n🚀 Running all tests for {builder_cls.__name__}...")
        
        # Run tests with scoring if requested
        tester = UniversalStepBuilderTest(
            builder_class=builder_cls,
            verbose=verbose,
            enable_scoring=scoring,
            enable_structured_reporting=bool(export_json or export_chart),
            use_step_catalog_discovery=True,
        )
        results = tester.run_all_tests()

        # Print results with appropriate formatting
        if scoring and "test_results" in results:
            print_enhanced_results(results, verbose)
        else:
            print_test_results(results, verbose, show_scoring=scoring)

        # Handle exports
        if export_json:
            export_results_to_json(results, export_json)

        if export_chart:
            chart_path = generate_score_chart(results, builder_cls.__name__, output_dir)
            if chart_path:
                click.echo(f"📊 Score chart generated: {chart_path}")
            else:
                click.echo("⚠️  Could not generate score chart (matplotlib may not be available)")

        # Determine exit code
        if "test_results" in results:
            test_results = results["test_results"]
        else:
            test_results = results

        failed_tests = sum(
            1 for result in test_results.values() if not result.get("passed", False)
        )
        if failed_tests > 0:
            click.echo(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix the issues.")
            sys.exit(1)
        else:
            click.echo(f"\n🎉 All tests passed successfully!")

    except Exception as e:
        click.echo(f"❌ Error during test execution: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@builder_test.command("test-level")
@click.argument("level", type=click.IntRange(1, 4))
@click.argument("builder_class")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output including test details and logs"
)
def test_level(level: int, builder_class: str, verbose: bool):
    """Run tests for a specific level."""
    try:
        click.echo(f"🔍 Importing builder class: {builder_class}")
        builder_cls = import_builder_class(builder_class)
        click.echo(f"✅ Successfully imported: {builder_cls.__name__}")

        level_names = {
            1: "Interface",
            2: "Specification", 
            3: "Step Creation",
            4: "Integration",
        }
        level_name = level_names[level]
        
        click.echo(f"\n🚀 Running Level {level} ({level_name}) tests for {builder_cls.__name__}...")

        # Use the refactored UniversalStepBuilderTest for all levels
        click.echo(f"⚠️  Note: Using refactored UniversalStepBuilderTest (Level {level} tests are now integrated)")
        
        tester = UniversalStepBuilderTest.from_builder_class(
            builder_class=builder_cls,
            verbose=verbose
        )
        results = tester.run_all_tests()

        print_test_results(results, verbose)

        # Determine exit code
        failed_tests = sum(
            1 for result in results.values() if not result.get("passed", False)
        )
        if failed_tests > 0:
            click.echo(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix the issues.")
            sys.exit(1)
        else:
            click.echo(f"\n🎉 All tests passed successfully!")

    except Exception as e:
        click.echo(f"❌ Error during test execution: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@builder_test.command("test-variant")
@click.argument("variant", type=click.Choice(["processing"]))
@click.argument("builder_class")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output including test details and logs"
)
def test_variant(variant: str, builder_class: str, verbose: bool):
    """Run tests for a specific variant."""
    try:
        click.echo(f"🔍 Importing builder class: {builder_class}")
        builder_cls = import_builder_class(builder_class)
        click.echo(f"✅ Successfully imported: {builder_cls.__name__}")

        click.echo(f"\n🚀 Running {variant.title()} variant tests for {builder_cls.__name__}...")

        # Use the refactored UniversalStepBuilderTest for all variants
        click.echo(f"⚠️  Note: Using refactored UniversalStepBuilderTest ({variant} variant tests are now integrated)")
        
        tester = UniversalStepBuilderTest.from_builder_class(
            builder_class=builder_cls,
            verbose=verbose
        )
        results = tester.run_all_tests()

        print_test_results(results, verbose)

        # Determine exit code
        failed_tests = sum(
            1 for result in results.values() if not result.get("passed", False)
        )
        if failed_tests > 0:
            click.echo(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix the issues.")
            sys.exit(1)
        else:
            click.echo(f"\n🎉 All tests passed successfully!")

    except Exception as e:
        click.echo(f"❌ Error during test execution: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@builder_test.command("test-by-type")
@click.argument("sagemaker_type", type=click.Choice(["Training", "Transform", "Processing", "CreateModel", "RegisterModel"]))
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output including test details and logs"
)
@click.option(
    "--scoring",
    is_flag=True,
    help="Enable quality scoring and enhanced reporting"
)
@click.option(
    "--export-json",
    type=click.Path(),
    help="Export test results to JSON file at specified path"
)
def test_by_type(sagemaker_type: str, verbose: bool, scoring: bool, export_json: str):
    """Test all builders for a specific SageMaker step type."""
    try:
        click.echo(f"🔍 Testing all builders for SageMaker step type: {sagemaker_type}")
        
        results = UniversalStepBuilderTest.test_all_builders_by_type(
            sagemaker_step_type=sagemaker_type,
            verbose=verbose,
            enable_scoring=scoring,
        )

        if "error" in results:
            click.echo(f"❌ Error: {results['error']}")
            sys.exit(1)

        click.echo(f"\n📊 Batch Test Results for {sagemaker_type} Steps")
        click.echo("=" * 60)

        total_builders = len(results)
        successful_builders = sum(1 for r in results.values() if "error" not in r)

        click.echo(f"Tested {successful_builders}/{total_builders} builders successfully")

        for step_name, result in results.items():
            if "error" in result:
                click.echo(f"❌ {step_name}: {result['error']}")
            else:
                if scoring and "scoring" in result:
                    score = result["scoring"].get("overall", {}).get("score", 0)
                    rating = result["scoring"].get("overall", {}).get("rating", "Unknown")
                    click.echo(f"✅ {step_name}: Score {score:.1f}/100 ({rating})")
                else:
                    test_results = result.get("test_results", result)
                    total_tests = len(test_results)
                    passed_tests = sum(
                        1 for r in test_results.values() if r.get("passed", False)
                    )
                    click.echo(f"✅ {step_name}: {passed_tests}/{total_tests} tests passed")

        # Export results if requested
        if export_json:
            export_results_to_json(results, export_json)

    except Exception as e:
        click.echo(f"❌ Error during test execution: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@builder_test.command("registry-report")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output including errors"
)
def registry_report(verbose: bool):
    """Generate step catalog discovery report."""
    try:
        click.echo("🔍 Generating step catalog discovery report...")
        
        catalog = StepCatalog(workspace_dirs=None)
        all_steps = catalog.list_available_steps()
        
        # Generate report using step catalog
        step_type_counts = {}
        available_count = 0
        errors = []
        
        for step_name in all_steps:
            try:
                # Try to get step type
                from ..registry import get_sagemaker_step_type
                step_type = get_sagemaker_step_type(step_name)
                if step_type:
                    step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1
                
                # Try to load builder class
                builder_class = catalog.load_builder_class(step_name)
                if builder_class:
                    available_count += 1
                else:
                    errors.append({"step_name": step_name, "error": "Builder class not found"})
                    
            except Exception as e:
                errors.append({"step_name": step_name, "error": str(e)})

        click.echo(f"\n📊 Step Catalog Discovery Report")
        click.echo("=" * 50)
        click.echo(f"Total steps in catalog: {len(all_steps)}")
        click.echo(f"Available SageMaker step types: {', '.join(step_type_counts.keys())}")

        click.echo(f"\nStep counts by type:")
        for step_type, count in step_type_counts.items():
            click.echo(f"  • {step_type}: {count} steps")

        click.echo(f"\nAvailability summary:")
        click.echo(f"  ✅ Available: {available_count}")
        click.echo(f"  ❌ Unavailable: {len(errors)}")

        if errors and verbose:
            click.echo(f"\nErrors:")
            for error in errors[:10]:  # Show first 10 errors
                click.echo(f"  • {error['step_name']}: {error['error']}")
            if len(errors) > 10:
                click.echo(f"  ... and {len(errors) - 10} more errors")

    except Exception as e:
        click.echo(f"❌ Error generating step catalog report: {e}", err=True)
        sys.exit(1)


@builder_test.command("validate-builder")
@click.argument("step_name")
def validate_builder(step_name: str):
    """Validate that a step builder is available and can be loaded."""
    try:
        click.echo(f"🔍 Validating builder availability for: {step_name}")
        
        catalog = StepCatalog(workspace_dirs=None)
        all_steps = catalog.list_available_steps()
        
        # Check if step exists in catalog
        in_catalog = step_name in all_steps
        
        # Try to load builder class
        builder_class = None
        loadable = False
        error = None
        
        try:
            builder_class = catalog.load_builder_class(step_name)
            loadable = builder_class is not None
        except Exception as e:
            error = str(e)

        click.echo(f"\n📊 Builder Validation Results")
        click.echo("=" * 40)
        click.echo(f"Step name: {step_name}")
        click.echo(f"In step catalog: {'✅' if in_catalog else '❌'}")
        click.echo(f"Builder class found: {'✅' if builder_class else '❌'}")
        click.echo(f"Loadable: {'✅' if loadable else '❌'}")

        if error:
            click.echo(f"Error: {error}")
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Error validating builder: {e}", err=True)
        sys.exit(1)


@builder_test.command("list-builders")
def list_builders():
    """List available step builder classes."""
    try:
        click.echo("📋 Available Step Builder Classes:")
        click.echo("=" * 50)
        
        # List available builders
        import os
        import inspect
        import importlib
        import ast
        from pathlib import Path

        available_builders = []
        builders_with_missing_deps = []

        try:
            # Get the builders directory path
            current_dir = Path(__file__).parent.parent
            builders_dir = current_dir / "steps" / "builders"

            if not builders_dir.exists():
                # Fallback: try to find it in the installed package
                try:
                    from ..steps import builders
                    builders_dir = Path(builders.__file__).parent
                except ImportError:
                    click.echo("Error: Could not locate builders directory")
                    sys.exit(1)

            # Scan for Python files in the builders directory
            for file_path in builders_dir.glob("builder_*.py"):
                if file_path.name == "__init__.py":
                    continue

                module_name = file_path.stem  # filename without extension

                try:
                    # Import the module using relative import
                    relative_module_path = f"..steps.builders.{module_name}"
                    module = importlib.import_module(
                        relative_module_path, package=__package__
                    )

                    # Find classes that end with "StepBuilder"
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if (
                            name.endswith("StepBuilder")
                            and obj.__module__.endswith(f".steps.builders.{module_name}")
                            and name != "StepBuilder"
                        ):
                            full_path = f"..steps.builders.{module_name}.{name}"
                            available_builders.append(full_path)

                except ImportError as e:
                    # If import fails, try to parse the file to extract class names
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        # Parse the AST to find class definitions
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if (
                                isinstance(node, ast.ClassDef)
                                and node.name.endswith("StepBuilder")
                                and node.name != "StepBuilder"
                            ):
                                full_path = f"..steps.builders.{module_name}.{node.name}"
                                builders_with_missing_deps.append(full_path)

                    except Exception:
                        # If AST parsing also fails, skip this file
                        continue

                except Exception as e:
                    # Log other errors for debugging but continue with other modules
                    continue

        except Exception as e:
            click.echo(f"Error scanning builders directory: {str(e)}")
            sys.exit(1)

        # Combine available builders and those with missing dependencies
        all_builders = available_builders + builders_with_missing_deps

        # Sort the list for consistent output
        all_builders.sort()

        if all_builders:
            for builder in all_builders:
                click.echo(f"  • {builder}")
            click.echo(f"\nTotal: {len(all_builders)} builder classes found")
        else:
            click.echo("No builder classes found")

    except Exception as e:
        click.echo(f"❌ Error listing builders: {e}", err=True)
        sys.exit(1)


# PHASE 2 ENHANCEMENT: Dynamic Discovery Commands
@builder_test.command("test-all-discovered")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--scoring", is_flag=True, help="Enable quality scoring")
@click.option("--export-json", type=click.Path(), help="Export results to JSON file")
@click.option("--step-type", help="Filter by SageMaker step type")
def test_all_discovered(verbose: bool, scoring: bool, export_json: str, step_type: str):
    """Test all builders discovered via step catalog."""
    try:
        click.echo("🔍 Discovering builders via step catalog...")
        
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)
        
        if step_type:
            builders = catalog.get_builders_by_step_type(step_type)
            click.echo(f"Found {len(builders)} {step_type} builders")
        else:
            builders = catalog.get_all_builders()
            click.echo(f"Found {len(builders)} total builders")
        
        if not builders:
            click.echo("❌ No builders found")
            return
        
        click.echo(f"\n🧪 Testing {len(builders)} builders...")
        
        # Test all builders
        results = {}
        for i, (step_name, builder_class) in enumerate(builders.items(), 1):
            click.echo(f"\n[{i}/{len(builders)}] Testing {step_name}...")
            
            try:
                tester = UniversalStepBuilderTest(
                    builder_class=builder_class,
                    step_name=step_name,
                    verbose=verbose,
                    enable_scoring=scoring,
                    enable_structured_reporting=True,
                    use_step_catalog_discovery=True
                )
                
                test_results = tester.run_all_tests()
                results[step_name] = test_results
                
                # Quick status report
                if 'test_results' in test_results:
                    raw_results = test_results['test_results']
                    total_tests = len(raw_results)
                    passed_tests = sum(1 for r in raw_results.values() if r.get('passed', False))
                    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
                    
                    status_icon = "✅" if pass_rate >= 80 else "⚠️" if pass_rate >= 60 else "❌"
                    click.echo(f"  {status_icon} {step_name}: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
                    
                    if scoring and 'scoring' in test_results:
                        score = test_results['scoring'].get('overall', {}).get('score', 0)
                        rating = test_results['scoring'].get('overall', {}).get('rating', 'Unknown')
                        click.echo(f"  📊 Quality Score: {score:.1f}/100 ({rating})")
                
            except Exception as e:
                click.echo(f"  ❌ {step_name}: Failed with error: {e}")
                results[step_name] = {'error': str(e)}
        
        # Generate comprehensive report
        total_builders = len(results)
        successful_tests = sum(1 for r in results.values() if 'error' not in r)
        success_rate = (successful_tests / total_builders * 100) if total_builders > 0 else 0
        
        click.echo(f"\n📊 OVERALL SUMMARY:")
        click.echo(f"   Builders Tested: {total_builders}")
        click.echo(f"   Successful Tests: {successful_tests} ({success_rate:.1f}%)")
        
        # Export or auto-save results
        if export_json:
            export_results_to_json(results, export_json)
        else:
            # Auto-save results
            from ..validation.builders.results_storage import BuilderTestResultsStorage
            output_path = BuilderTestResultsStorage.save_test_results(results, "all_builders", step_type)
            click.echo(f"📁 Results automatically saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@builder_test.command("test-single")
@click.argument("canonical_name")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
@click.option("--scoring", is_flag=True, help="Enable quality scoring")
@click.option("--export-json", type=click.Path(), help="Export results to JSON file")
def test_single(canonical_name: str, verbose: bool, scoring: bool, export_json: str):
    """Test single builder by canonical name."""
    try:
        click.echo(f"🔍 Looking for builder: {canonical_name}")
        
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)
        builder_class = catalog.load_builder_class(canonical_name)
        
        if not builder_class:
            click.echo(f"❌ No builder found for: {canonical_name}")
            # Show available builders
            all_builders = catalog.get_all_builders()
            available = sorted(all_builders.keys())
            click.echo(f"Available builders: {', '.join(available[:10])}")
            if len(available) > 10:
                click.echo(f"... and {len(available) - 10} more")
            sys.exit(1)
        
        click.echo(f"✅ Found builder: {builder_class.__name__}")
        click.echo(f"\n🧪 Testing {canonical_name}...")
        
        # Run tests
        tester = UniversalStepBuilderTest(
            builder_class=builder_class,
            step_name=canonical_name,
            verbose=verbose,
            enable_scoring=scoring,
            enable_structured_reporting=True,
            use_step_catalog_discovery=True
        )
        
        results = tester.run_all_tests()
        
        # Print results
        if scoring and "test_results" in results:
            print_enhanced_results(results, verbose)
        else:
            print_test_results(results, verbose, show_scoring=scoring)
        
        # Export or auto-save results
        export_data = {canonical_name: results}
        if export_json:
            export_results_to_json(export_data, export_json)
        else:
            from ..validation.builders.results_storage import BuilderTestResultsStorage
            output_path = BuilderTestResultsStorage.save_test_results(export_data, "single_builder", canonical_name)
            click.echo(f"📁 Results automatically saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@builder_test.command("list-discovered")
@click.option("--step-type", help="Filter by SageMaker step type")
def list_discovered(step_type: str):
    """List builders discovered via step catalog."""
    try:
        click.echo("📋 Builders discovered via step catalog:")
        click.echo("=" * 50)
        
        from ..step_catalog import StepCatalog
        catalog = StepCatalog(workspace_dirs=None)
        
        if step_type:
            builders = catalog.get_builders_by_step_type(step_type)
            click.echo(f"Filtered by step type: {step_type}")
        else:
            builders = catalog.get_all_builders()
        
        if builders:
            for step_name, builder_class in sorted(builders.items()):
                click.echo(f"  • {step_name} → {builder_class.__name__}")
            click.echo(f"\nTotal: {len(builders)} builders")
        else:
            click.echo("No builders found")
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for builder test CLI."""
    builder_test()


if __name__ == "__main__":
    main()
