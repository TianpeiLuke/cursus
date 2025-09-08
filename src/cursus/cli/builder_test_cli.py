#!/usr/bin/env python3
"""
Command-line interface for the Universal Step Builder Test System.

This CLI provides easy access to run different levels of tests and variants
for step builder validation according to the UniversalStepBuilderTestBase architecture.
Enhanced with scoring, registry discovery, and export capabilities.
"""

import argparse
import sys
import importlib
import inspect
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Type

from ..validation.builders.universal_test import UniversalStepBuilderTest
from ..validation.builders.interface_tests import InterfaceTests
from ..validation.builders.specification_tests import SpecificationTests
from ..validation.builders.step_creation_tests import StepCreationTests
from ..validation.builders.integration_tests import IntegrationTests
from ..validation.builders.variants.processing_test import ProcessingStepBuilderTest
from ..validation.builders.registry_discovery import RegistryStepDiscovery
from ..validation.builders.scoring import StepBuilderScorer


def print_test_results(results: Dict[str, Any], verbose: bool = False, 
                      show_scoring: bool = False) -> None:
    """Print test results in a formatted way with optional scoring display."""
    if not results:
        print("❌ No test results found!")
        return
    
    # Handle both legacy format (raw results) and new format (with scoring/reporting)
    if 'test_results' in results:
        # New format with scoring/reporting
        test_results = results['test_results']
        scoring_data = results.get('scoring')
        structured_report = results.get('structured_report')
    else:
        # Legacy format (raw results)
        test_results = results
        scoring_data = None
        structured_report = None
    
    # Calculate summary statistics
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result.get("passed", False))
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    # Print summary header with optional scoring
    if scoring_data and show_scoring:
        overall_score = scoring_data.get('overall', {}).get('score', 0.0)
        overall_rating = scoring_data.get('overall', {}).get('rating', 'Unknown')
        print(f"\n📊 Test Results Summary: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        print(f"🏆 Quality Score: {overall_score:.1f}/100 - {overall_rating}")
        print("=" * 70)
    else:
        print(f"\n📊 Test Results Summary: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
        print("=" * 60)
    
    # Group results by test level/type
    level_groups = {
        "Level 1 (Interface)": [],
        "Level 2 (Specification)": [],
        "Level 3 (Step Creation)": [],
        "Level 4 (Integration)": [],
        "Step Type Specific": [],
        "Other": []
    }
    
    for test_name, result in test_results.items():
        if any(interface_test in test_name for interface_test in [
            "inheritance", "naming_conventions", "required_methods", "registry_integration",
            "documentation_standards", "type_hints", "error_handling", "method_return_types",
            "configuration_validation", "generic_step_creation", "generic_configuration"
        ]):
            level_groups["Level 1 (Interface)"].append((test_name, result))
        elif any(spec_test in test_name for spec_test in [
            "specification_usage", "contract_alignment", "environment_variable_handling", 
            "job_arguments", "environment_variables_processing", "property_files_configuration"
        ]):
            level_groups["Level 2 (Specification)"].append((test_name, result))
        elif any(creation_test in test_name for creation_test in [
            "step_instantiation", "step_configuration_validity", "step_dependencies_attachment",
            "step_name_generation", "input_path_mapping", "output_path_mapping", 
            "property_path_validity", "processing_inputs_outputs", "processing_code_handling"
        ]):
            level_groups["Level 3 (Step Creation)"].append((test_name, result))
        elif any(integration_test in test_name for integration_test in [
            "dependency_resolution", "step_creation", "step_name", "generic_dependency_handling",
            "processing_step_dependencies"
        ]):
            level_groups["Level 4 (Integration)"].append((test_name, result))
        elif any(step_type_test in test_name for step_type_test in [
            "step_type", "processing", "training", "transform", "create_model", "register_model",
            "processor_creation", "estimator_methods", "transformer_methods"
        ]):
            level_groups["Step Type Specific"].append((test_name, result))
        else:
            level_groups["Other"].append((test_name, result))
    
    # Print results by group with optional level scoring
    for group_name, group_tests in level_groups.items():
        if not group_tests:
            continue
            
        group_passed = sum(1 for _, result in group_tests if result.get("passed", False))
        group_total = len(group_tests)
        group_rate = (group_passed / group_total) * 100 if group_total > 0 else 0
        
        # Add level score if available
        level_score_text = ""
        if scoring_data and show_scoring:
            level_key = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            level_mapping = {
                "level_1_interface": "level1_interface",
                "level_2_specification": "level2_specification", 
                "level_3_step_creation": "level3_step_creation",
                "level_4_integration": "level4_integration"
            }
            mapped_key = level_mapping.get(level_key)
            if mapped_key and mapped_key in scoring_data.get('levels', {}):
                level_score = scoring_data['levels'][mapped_key].get('score', 0)
                level_score_text = f" - Score: {level_score:.1f}/100"
        
        print(f"\n📁 {group_name}: {group_passed}/{group_total} passed ({group_rate:.1f}%){level_score_text}")
        
        for test_name, result in group_tests:
            status = "✅" if result.get("passed", False) else "❌"
            print(f"  {status} {test_name}")
            
            if not result.get("passed", False) and result.get("error"):
                print(f"    💬 {result['error']}")
            
            if verbose and result.get("details"):
                print(f"    📋 Details: {result['details']}")
    
    print("\n" + "=" * (70 if show_scoring else 60))


def print_enhanced_results(results: Dict[str, Any], verbose: bool = False) -> None:
    """Print enhanced results with scoring and structured reporting."""
    if 'test_results' not in results:
        print_test_results(results, verbose)
        return
    
    test_results = results['test_results']
    scoring_data = results.get('scoring')
    structured_report = results.get('structured_report')
    
    # Print test results with scoring
    print_test_results(results, verbose, show_scoring=True)
    
    # Print additional scoring details if available
    if scoring_data and verbose:
        print("\n📈 Detailed Scoring Breakdown:")
        print("-" * 50)
        
        levels = scoring_data.get('levels', {})
        for level_name, level_data in levels.items():
            display_name = level_name.replace("level", "Level ").replace("_", " ").title()
            score = level_data.get('score', 0)
            passed = level_data.get('passed', 0)
            total = level_data.get('total', 0)
            print(f"  {display_name}: {score:.1f}/100 ({passed}/{total} tests)")
        
        # Show failed tests summary
        failed_tests = scoring_data.get('failed_tests', [])
        if failed_tests:
            print(f"\n❌ Failed Tests Summary ({len(failed_tests)}):")
            for test in failed_tests[:5]:  # Show first 5 failed tests
                print(f"  • {test['name']}: {test['error']}")
            if len(failed_tests) > 5:
                print(f"  ... and {len(failed_tests) - 5} more failed tests")


def run_all_tests_with_scoring(
    builder_class: Type,
    verbose: bool = False,
    enable_structured_reporting: bool = False
) -> Dict[str, Any]:
    """Run all tests with scoring enabled."""
    tester = UniversalStepBuilderTest(
        builder_class=builder_class, 
        verbose=verbose,
        enable_scoring=True,
        enable_structured_reporting=enable_structured_reporting
    )
    return tester.run_all_tests()


def run_registry_discovery_report() -> Dict[str, Any]:
    """Generate and display registry discovery report."""
    return RegistryStepDiscovery.generate_discovery_report()


def run_test_by_sagemaker_type(
    sagemaker_step_type: str,
    verbose: bool = False,
    enable_scoring: bool = True
) -> Dict[str, Any]:
    """Test all builders for a specific SageMaker step type."""
    return UniversalStepBuilderTest.test_all_builders_by_type(
        sagemaker_step_type=sagemaker_step_type,
        verbose=verbose,
        enable_scoring=enable_scoring
    )


def validate_builder_availability(step_name: str) -> Dict[str, Any]:
    """Validate that a step builder is available and can be loaded."""
    return RegistryStepDiscovery.validate_step_builder_availability(step_name)


def export_results_to_json(results: Dict[str, Any], output_path: str) -> None:
    """Export test results to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results exported to: {output_path}")


def generate_score_chart(results: Dict[str, Any], builder_name: str, output_dir: str) -> Optional[str]:
    """Generate score visualization chart."""
    if 'test_results' not in results:
        # Create scorer from raw results
        scorer = StepBuilderScorer(results)
    else:
        # Create scorer from test_results
        scorer = StepBuilderScorer(results['test_results'])
    
    return scorer.generate_chart(builder_name, output_dir)


def import_builder_class(class_path: str) -> Type:
    """Import a builder class from a module path."""
    try:
        # Split module path and class name
        if '.' in class_path:
            module_path, class_name = class_path.rsplit('.', 1)
        else:
            # Assume it's just a class name in the current package
            module_path = "cursus.steps.builders"
            class_name = class_path
        
        # Handle src. prefix - remove it for installed package
        if module_path.startswith('src.'):
            module_path = module_path[4:]  # Remove 'src.' prefix
        
        # Import the module
        module = importlib.import_module(module_path)
        
        # Get the class
        builder_class = getattr(module, class_name)
        
        return builder_class
        
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"Could not find class {class_name} in module {module_path}: {e}")


def run_level_tests(
    builder_class: Type,
    level: int,
    verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Run tests for a specific level."""
    test_classes = {
        1: InterfaceTests,
        2: SpecificationTests,
        3: StepCreationTests,
        4: IntegrationTests
    }
    
    if level not in test_classes:
        raise ValueError(f"Invalid test level: {level}. Must be 1, 2, 3, or 4.")
    
    test_class = test_classes[level]
    tester = test_class(builder_class=builder_class, verbose=verbose)
    
    return tester.run_all_tests()


def run_variant_tests(
    builder_class: Type,
    variant: str,
    verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Run tests for a specific variant."""
    variant_classes = {
        "processing": ProcessingStepBuilderTest,
        # Add more variants as they become available
        # "training": TrainingStepBuilderTest,
        # "transform": TransformStepBuilderTest,
    }
    
    if variant not in variant_classes:
        available_variants = ", ".join(variant_classes.keys())
        raise ValueError(f"Invalid variant: {variant}. Available variants: {available_variants}")
    
    variant_class = variant_classes[variant]
    tester = variant_class(builder_class=builder_class, verbose=verbose)
    
    return tester.run_all_tests()


def run_all_tests(
    builder_class: Type,
    verbose: bool = False,
    enable_scoring: bool = False
) -> Dict[str, Any]:
    """Run all tests (universal test suite)."""
    tester = UniversalStepBuilderTest(
        builder_class=builder_class, 
        verbose=verbose,
        enable_scoring=enable_scoring
    )
    return tester.run_all_tests()


def list_available_builders() -> List[str]:
    """List available step builder classes by scanning the builders directory."""
    import os
    import inspect
    import importlib
    import ast
    from pathlib import Path
    
    available_builders = []
    builders_with_missing_deps = []
    
    try:
        # Get the builders directory path
        # First try to find it relative to this module
        current_dir = Path(__file__).parent.parent
        builders_dir = current_dir / "steps" / "builders"
        
        if not builders_dir.exists():
            # Fallback: try to find it in the installed package
            try:
                import cursus.steps.builders
                builders_dir = Path(cursus.steps.builders.__file__).parent
            except ImportError:
                return ["Error: Could not locate builders directory"]
        
        # Scan for Python files in the builders directory
        for file_path in builders_dir.glob("builder_*.py"):
            if file_path.name == "__init__.py":
                continue
                
            module_name = file_path.stem  # filename without extension
            
            try:
                # Import the module using relative import
                relative_module_path = f"..steps.builders.{module_name}"
                module = importlib.import_module(relative_module_path, package=__package__)
                
                # Find classes that end with "StepBuilder"
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (name.endswith("StepBuilder") and 
                        obj.__module__ == f"cursus.steps.builders.{module_name}" and  # Ensure it's defined in this module
                        name != "StepBuilder"):  # Exclude base classes
                        
                        full_path = f"cursus.steps.builders.{module_name}.{name}"
                        available_builders.append(full_path)
                        
            except ImportError as e:
                # If import fails, try to parse the file to extract class names
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse the AST to find class definitions
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef) and node.name.endswith("StepBuilder") and node.name != "StepBuilder":
                            full_path = f"cursus.steps.builders.{module_name}.{node.name}"
                            builders_with_missing_deps.append(full_path)
                            
                except Exception:
                    # If AST parsing also fails, skip this file
                    continue
                    
            except Exception as e:
                # Log other errors for debugging but continue with other modules
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Could not process {module_name}: {e}")
                continue
    
    except Exception as e:
        return [f"Error scanning builders directory: {str(e)}"]
    
    # Combine available builders and those with missing dependencies
    all_builders = available_builders + builders_with_missing_deps
    
    # Sort the list for consistent output
    all_builders.sort()
    
    return all_builders


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Universal Step Builder Tests with enhanced scoring and registry features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests for a builder with scoring
  python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --scoring
  
  # Run tests and export results to JSON
  python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --export-json results.json
  
  # Test all builders of a specific SageMaker step type
  python -m cursus.cli.builder_test_cli test-by-type Training --verbose
  
  # Generate registry discovery report
  python -m cursus.cli.builder_test_cli registry-report
  
  # Validate builder availability
  python -m cursus.cli.builder_test_cli validate-builder XGBoostTraining
  
  # Run Level 1 tests with scoring
  python -m cursus.cli.builder_test_cli level 1 src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder --scoring
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including test details and logs"
    )
    
    parser.add_argument(
        "--scoring",
        action="store_true",
        help="Enable quality scoring and enhanced reporting"
    )
    
    parser.add_argument(
        "--export-json",
        type=str,
        metavar="PATH",
        help="Export test results to JSON file at specified path"
    )
    
    parser.add_argument(
        "--export-chart",
        action="store_true",
        help="Generate score visualization chart (requires matplotlib)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_reports",
        help="Output directory for exports (default: test_reports)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # All tests command
    all_parser = subparsers.add_parser(
        "all",
        help="Run all tests (universal test suite)"
    )
    all_parser.add_argument(
        "builder_class",
        help="Full path to the step builder class (e.g., src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder)"
    )
    
    # Level tests command
    level_parser = subparsers.add_parser(
        "level",
        help="Run tests for a specific level"
    )
    level_parser.add_argument(
        "level_number",
        type=int,
        choices=[1, 2, 3, 4],
        help="Test level to run (1=Interface, 2=Specification, 3=Step Creation, 4=Integration)"
    )
    level_parser.add_argument(
        "builder_class",
        help="Full path to the step builder class"
    )
    
    # Variant tests command
    variant_parser = subparsers.add_parser(
        "variant",
        help="Run tests for a specific variant"
    )
    variant_parser.add_argument(
        "variant_name",
        choices=["processing"],  # Add more as they become available
        help="Test variant to run"
    )
    variant_parser.add_argument(
        "builder_class",
        help="Full path to the step builder class"
    )
    
    # Test by SageMaker type command
    type_parser = subparsers.add_parser(
        "test-by-type",
        help="Test all builders for a specific SageMaker step type"
    )
    type_parser.add_argument(
        "sagemaker_type",
        choices=["Training", "Transform", "Processing", "CreateModel", "RegisterModel"],
        help="SageMaker step type to test"
    )
    
    # Registry discovery report command
    registry_parser = subparsers.add_parser(
        "registry-report",
        help="Generate registry discovery report"
    )
    
    # Validate builder command
    validate_parser = subparsers.add_parser(
        "validate-builder",
        help="Validate that a step builder is available and can be loaded"
    )
    validate_parser.add_argument(
        "step_name",
        help="Step name from registry to validate"
    )
    
    # List builders command
    list_parser = subparsers.add_parser(
        "list-builders",
        help="List available step builder classes"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        # Handle registry and validation commands that don't require builder class
        if args.command == "list-builders":
            print("📋 Available Step Builder Classes:")
            print("=" * 50)
            for builder in list_available_builders():
                print(f"  • {builder}")
            print("\nNote: This is a basic list. You can test any builder class by providing its full import path.")
            return 0
        
        elif args.command == "registry-report":
            print("🔍 Generating registry discovery report...")
            report = run_registry_discovery_report()
            
            print(f"\n📊 Registry Discovery Report")
            print("=" * 50)
            print(f"Total steps in registry: {report['total_steps']}")
            print(f"Available SageMaker step types: {', '.join(report['sagemaker_step_types'])}")
            
            print(f"\nStep counts by type:")
            for step_type, count in report['step_type_counts'].items():
                print(f"  • {step_type}: {count} steps")
            
            availability = report['availability_summary']
            print(f"\nAvailability summary:")
            print(f"  ✅ Available: {availability['available']}")
            print(f"  ❌ Unavailable: {availability['unavailable']}")
            
            if availability['errors'] and args.verbose:
                print(f"\nErrors:")
                for error in availability['errors'][:10]:  # Show first 10 errors
                    print(f"  • {error['step_name']}: {error['error']}")
                if len(availability['errors']) > 10:
                    print(f"  ... and {len(availability['errors']) - 10} more errors")
            
            return 0
        
        elif args.command == "validate-builder":
            print(f"🔍 Validating builder availability for: {args.step_name}")
            validation = validate_builder_availability(args.step_name)
            
            print(f"\n📊 Builder Validation Results")
            print("=" * 40)
            print(f"Step name: {validation['step_name']}")
            print(f"In registry: {'✅' if validation['in_registry'] else '❌'}")
            print(f"Module exists: {'✅' if validation['module_exists'] else '❌'}")
            print(f"Class exists: {'✅' if validation['class_exists'] else '❌'}")
            print(f"Loadable: {'✅' if validation['loadable'] else '❌'}")
            
            if validation['error']:
                print(f"Error: {validation['error']}")
                return 1
            
            return 0
        
        elif args.command == "test-by-type":
            print(f"🔍 Testing all builders for SageMaker step type: {args.sagemaker_type}")
            results = run_test_by_sagemaker_type(
                args.sagemaker_type, 
                verbose=args.verbose,
                enable_scoring=args.scoring
            )
            
            if 'error' in results:
                print(f"❌ Error: {results['error']}")
                return 1
            
            print(f"\n📊 Batch Test Results for {args.sagemaker_type} Steps")
            print("=" * 60)
            
            total_builders = len(results)
            successful_builders = sum(1 for r in results.values() if 'error' not in r)
            
            print(f"Tested {successful_builders}/{total_builders} builders successfully")
            
            for step_name, result in results.items():
                if 'error' in result:
                    print(f"❌ {step_name}: {result['error']}")
                else:
                    if args.scoring and 'scoring' in result:
                        score = result['scoring'].get('overall', {}).get('score', 0)
                        rating = result['scoring'].get('overall', {}).get('rating', 'Unknown')
                        print(f"✅ {step_name}: Score {score:.1f}/100 ({rating})")
                    else:
                        test_results = result.get('test_results', result)
                        total_tests = len(test_results)
                        passed_tests = sum(1 for r in test_results.values() if r.get('passed', False))
                        print(f"✅ {step_name}: {passed_tests}/{total_tests} tests passed")
            
            # Export results if requested
            if args.export_json:
                export_results_to_json(results, args.export_json)
            
            return 0
        
        # Commands that require builder class
        builder_class = None
        if hasattr(args, 'builder_class'):
            print(f"🔍 Importing builder class: {args.builder_class}")
            builder_class = import_builder_class(args.builder_class)
            print(f"✅ Successfully imported: {builder_class.__name__}")
        
        # Run the appropriate tests
        results = None
        if args.command == "all":
            print(f"\n🚀 Running all tests for {builder_class.__name__}...")
            if args.scoring:
                results = run_all_tests_with_scoring(
                    builder_class, 
                    args.verbose,
                    enable_structured_reporting=bool(args.export_json or args.export_chart)
                )
            else:
                results = run_all_tests(builder_class, args.verbose, enable_scoring=False)
                
        elif args.command == "level":
            level_names = {1: "Interface", 2: "Specification", 3: "Step Creation", 4: "Integration"}
            level_name = level_names[args.level_number]
            print(f"\n🚀 Running Level {args.level_number} ({level_name}) tests for {builder_class.__name__}...")
            results = run_level_tests(builder_class, args.level_number, args.verbose)
            
        elif args.command == "variant":
            print(f"\n🚀 Running {args.variant_name.title()} variant tests for {builder_class.__name__}...")
            results = run_variant_tests(builder_class, args.variant_name, args.verbose)
        
        # Print results with appropriate formatting
        if results:
            if args.scoring and 'test_results' in results:
                print_enhanced_results(results, args.verbose)
            else:
                print_test_results(results, args.verbose, show_scoring=args.scoring)
            
            # Handle exports
            if args.export_json:
                export_results_to_json(results, args.export_json)
            
            if args.export_chart:
                builder_name = builder_class.__name__ if builder_class else "UnknownBuilder"
                chart_path = generate_score_chart(results, builder_name, args.output_dir)
                if chart_path:
                    print(f"📊 Score chart generated: {chart_path}")
                else:
                    print("⚠️  Could not generate score chart (matplotlib may not be available)")
            
            # Determine exit code
            if 'test_results' in results:
                test_results = results['test_results']
            else:
                test_results = results
                
            failed_tests = sum(1 for result in test_results.values() if not result.get("passed", False))
            if failed_tests > 0:
                print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix the issues.")
                return 1
            else:
                print(f"\n🎉 All tests passed successfully!")
                return 0
        
        return 0
            
    except Exception as e:
        print(f"❌ Error during test execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
