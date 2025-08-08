#!/usr/bin/env python3
"""
Command-line interface for the Universal Step Builder Test System.

This CLI provides easy access to run different levels of tests and variants
for step builder validation according to the UniversalStepBuilderTestBase architecture.
"""

import argparse
import sys
import importlib
from pathlib import Path
from typing import List, Optional, Dict, Any, Type

from ..validation.builders.universal_test import UniversalStepBuilderTest
from ..validation.builders.interface_tests import InterfaceTests
from ..validation.builders.specification_tests import SpecificationTests
from ..validation.builders.path_mapping_tests import PathMappingTests
from ..validation.builders.integration_tests import IntegrationTests
from ..validation.builders.variants.processing_test import ProcessingStepBuilderTest


def print_test_results(results: Dict[str, Dict[str, Any]], verbose: bool = False) -> None:
    """Print test results in a formatted way."""
    if not results:
        print("❌ No test results found!")
        return
    
    # Calculate summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result.get("passed", False))
    pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    # Print summary header
    print(f"\n📊 Test Results Summary: {passed_tests}/{total_tests} tests passed ({pass_rate:.1f}%)")
    print("=" * 60)
    
    # Group results by test level/type
    level_groups = {
        "Level 1 (Interface)": [],
        "Level 2 (Specification)": [],
        "Level 3 (Path Mapping)": [],
        "Level 4 (Integration)": [],
        "Step Type Specific": [],
        "Other": []
    }
    
    for test_name, result in results.items():
        if any(interface_test in test_name for interface_test in [
            "inheritance", "naming_conventions", "required_methods", "registry_integration",
            "documentation_standards", "type_hints", "error_handling", "method_return_types",
            "configuration_validation"
        ]):
            level_groups["Level 1 (Interface)"].append((test_name, result))
        elif any(spec_test in test_name for spec_test in [
            "specification_usage", "contract_alignment", "environment_variable_handling", "job_arguments"
        ]):
            level_groups["Level 2 (Specification)"].append((test_name, result))
        elif any(path_test in test_name for path_test in [
            "input_path_mapping", "output_path_mapping", "property_path_validity"
        ]):
            level_groups["Level 3 (Path Mapping)"].append((test_name, result))
        elif any(integration_test in test_name for integration_test in [
            "dependency_resolution", "step_creation", "step_name"
        ]):
            level_groups["Level 4 (Integration)"].append((test_name, result))
        elif any(step_type_test in test_name for step_type_test in [
            "step_type", "processing", "training", "transform", "create_model", "register_model",
            "processor_creation", "estimator_methods", "transformer_methods"
        ]):
            level_groups["Step Type Specific"].append((test_name, result))
        else:
            level_groups["Other"].append((test_name, result))
    
    # Print results by group
    for group_name, group_tests in level_groups.items():
        if not group_tests:
            continue
            
        group_passed = sum(1 for _, result in group_tests if result.get("passed", False))
        group_total = len(group_tests)
        group_rate = (group_passed / group_total) * 100 if group_total > 0 else 0
        
        print(f"\n📁 {group_name}: {group_passed}/{group_total} passed ({group_rate:.1f}%)")
        
        for test_name, result in group_tests:
            status = "✅" if result.get("passed", False) else "❌"
            print(f"  {status} {test_name}")
            
            if not result.get("passed", False) and result.get("error"):
                print(f"    💬 {result['error']}")
            
            if verbose and result.get("details"):
                print(f"    📋 Details: {result['details']}")
    
    print("\n" + "=" * 60)


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
        3: PathMappingTests,
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
    verbose: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Run all tests (universal test suite)."""
    tester = UniversalStepBuilderTest(builder_class=builder_class, verbose=verbose)
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
                # Import the module
                module_path = f"cursus.steps.builders.{module_name}"
                module = importlib.import_module(module_path)
                
                # Find classes that end with "StepBuilder"
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (name.endswith("StepBuilder") and 
                        obj.__module__ == module_path and  # Ensure it's defined in this module
                        name != "StepBuilder"):  # Exclude base classes
                        
                        full_path = f"src.cursus.steps.builders.{module_name}.{name}"
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
                            full_path = f"src.cursus.steps.builders.{module_name}.{node.name}"
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
        description="Run Universal Step Builder Tests at different levels and variants",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests for a builder
  python -m cursus.cli.builder_test_cli all src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
  
  # Run Level 1 (Interface) tests only
  python -m cursus.cli.builder_test_cli level 1 src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
  
  # Run Level 2 (Specification) tests only
  python -m cursus.cli.builder_test_cli level 2 src.cursus.steps.builders.builder_tabular_preprocessing_step.TabularPreprocessingStepBuilder
  
  # Run Level 3 (Path Mapping) tests only
  python -m cursus.cli.builder_test_cli level 3 src.cursus.steps.builders.builder_model_eval_step.ModelEvalStepBuilder
  
  # Run Level 4 (Integration) tests only
  python -m cursus.cli.builder_test_cli level 4 src.cursus.steps.builders.builder_training_step_xgboost.XGBoostTrainingStepBuilder
  
  # Run Processing variant tests
  python -m cursus.cli.builder_test_cli variant processing src.cursus.steps.builders.builder_tabular_preprocessing_step.TabularPreprocessingStepBuilder
  
  # List available builders
  python -m cursus.cli.builder_test_cli list-builders
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output including test details and logs"
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
        help="Test level to run (1=Interface, 2=Specification, 3=Path Mapping, 4=Integration)"
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
        if args.command == "list-builders":
            print("📋 Available Step Builder Classes:")
            print("=" * 50)
            for builder in list_available_builders():
                print(f"  • {builder}")
            print("\nNote: This is a basic list. You can test any builder class by providing its full import path.")
            return 0
        
        # Import the builder class
        print(f"🔍 Importing builder class: {args.builder_class}")
        builder_class = import_builder_class(args.builder_class)
        print(f"✅ Successfully imported: {builder_class.__name__}")
        
        # Run the appropriate tests
        if args.command == "all":
            print(f"\n🚀 Running all tests for {builder_class.__name__}...")
            results = run_all_tests(builder_class, args.verbose)
        elif args.command == "level":
            level_names = {1: "Interface", 2: "Specification", 3: "Path Mapping", 4: "Integration"}
            level_name = level_names[args.level_number]
            print(f"\n🚀 Running Level {args.level_number} ({level_name}) tests for {builder_class.__name__}...")
            results = run_level_tests(builder_class, args.level_number, args.verbose)
        elif args.command == "variant":
            print(f"\n🚀 Running {args.variant_name.title()} variant tests for {builder_class.__name__}...")
            results = run_variant_tests(builder_class, args.variant_name, args.verbose)
        else:
            parser.print_help()
            return 1
        
        # Print results
        print_test_results(results, args.verbose)
        
        # Return appropriate exit code
        failed_tests = sum(1 for result in results.values() if not result.get("passed", False))
        if failed_tests > 0:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review and fix the issues.")
            return 1
        else:
            print(f"\n🎉 All tests passed successfully!")
            return 0
            
    except Exception as e:
        print(f"❌ Error during test execution: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
