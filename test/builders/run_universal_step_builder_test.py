#!/usr/bin/env python
"""
Universal Step Builder Test Runner

This script demonstrates how to use the universal step builder test with
multiple step builders to evaluate their compliance with architectural requirements.

Usage:
    python run_universal_step_builder_test.py [--verbose]

Options:
    --verbose          Print detailed test results
    --builder BUILDER  Test a specific builder class
    --spec SPEC        Use a specific specification (module.SPEC_NAME format)
    --contract CONTRACT Use a specific contract (module.CONTRACT_NAME format)
    --step-name NAME   Use a specific step name
    --score            Generate quality scores and reports for builders
    --output-dir DIR   Directory to save reports (default: test_reports)
    --no-chart         Disable chart generation for score reports
"""

import argparse
import sys
import importlib
from typing import List, Dict, Any, Type, Optional
from pathlib import Path

# Add project root to path to ensure imports work correctly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from test.pipeline_steps.universal_step_builder_test import UniversalStepBuilderTest
from test.pipeline_steps.universal_step_builder_test.scoring import score_builder_results
from src.pipeline_steps.builder_step_base import StepBuilderBase


def get_available_step_builders() -> Dict[str, Type[StepBuilderBase]]:
    """
    Get all available step builder classes from src/pipeline_steps.
    
    Returns:
        Dictionary mapping builder names to builder classes
    """
    from src.pipeline_steps import builder_step_base
    
    # Get pipeline steps directory
    pipeline_steps_dir = Path(builder_step_base.__file__).parent
    
    # Find all builder modules
    builder_modules = [
        f.stem for f in pipeline_steps_dir.glob("builder_*.py") 
        if f.is_file() and f.stem != "builder_step_base"
    ]
    
    # Import modules and get builder classes
    builders = {}
    for module_name in builder_modules:
        try:
            module = importlib.import_module(f"src.pipeline_steps.{module_name}")
            
            # Find builder class in module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, StepBuilderBase) and 
                    attr is not StepBuilderBase):
                    builders[attr.__name__] = attr
        except ImportError:
            print(f"Could not import {module_name}")
            continue
    
    return builders


def run_tests_on_builders(
    builder_classes: Dict[str, Type[StepBuilderBase]], 
    verbose: bool = False,
    config: Optional[Any] = None,
    spec: Optional[Any] = None,
    contract: Optional[Any] = None,
    step_name: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run universal tests on all provided builder classes.
    
    Args:
        builder_classes: Dictionary mapping builder names to builder classes
        verbose: Whether to print verbose output
        config: Optional config to use (will create mock if not provided)
        spec: Optional step specification (will extract from builder if not provided)
        contract: Optional script contract (will extract from builder if not provided)
        step_name: Optional step name (will extract from class name if not provided)
        
    Returns:
        Dictionary mapping builder names to test results
    """
    all_results = {}
    
    print(f"\nTesting {len(builder_classes)} step builders...\n")
    
    for builder_name, builder_class in builder_classes.items():
        print(f"Testing {builder_name}...")
        
        try:
            # Create tester with provided components
            tester = UniversalStepBuilderTest(
                builder_class, 
                config=config,
                spec=spec,
                contract=contract,
                step_name=step_name,
                verbose=verbose
            )
            
            # Run all tests
            results = tester.run_all_tests()
            
            # Count passed tests
            passed = sum(1 for test in results.values() if test["passed"])
            total = len(results)
            
            print(f"✅ {builder_name}: {passed}/{total} tests passed\n")
            
            # Store results
            all_results[builder_name] = results
            
        except Exception as e:
            print(f"❌ {builder_name}: Failed to run tests: {str(e)}\n")
            all_results[builder_name] = {"error": str(e)}
    
    return all_results


def print_summary(all_results: Dict[str, Dict[str, Any]], generate_scores: bool = False, output_dir: str = "test_reports", generate_charts: bool = True) -> None:
    """
    Print a summary of test results for all builders.
    
    Args:
        all_results: Dictionary mapping builder names to test results
        generate_scores: Whether to generate quality scores
        output_dir: Directory to save reports in
        generate_charts: Whether to generate charts
    """
    print("\n" + "=" * 80)
    print("UNIVERSAL STEP BUILDER TEST SUMMARY")
    print("=" * 80)
    
    # Count builders
    total_builders = len(all_results)
    successful_builders = sum(1 for results in all_results.values() if "error" not in results)
    
    print(f"\nTested {total_builders} step builders: {successful_builders} completed, {total_builders - successful_builders} failed to run")
    
    # Calculate compliance stats for each test
    if successful_builders > 0:
        test_stats = {}
        for builder_name, results in all_results.items():
            if "error" in results:
                continue
                
            for test_name, test_result in results.items():
                if test_name not in test_stats:
                    test_stats[test_name] = {"passed": 0, "total": 0}
                
                test_stats[test_name]["total"] += 1
                if test_result["passed"]:
                    test_stats[test_name]["passed"] += 1
        
        # Print test compliance rates
        print("\nTest Compliance Rates:")
        for test_name, stats in sorted(test_stats.items(), key=lambda x: x[0]):
            compliance_pct = (stats["passed"] / stats["total"]) * 100
            print(f"{test_name:30s}: {stats['passed']}/{stats['total']} builders ({compliance_pct:.1f}%)")
    
    # Print fully compliant builders
    fully_compliant = [
        builder_name for builder_name, results in all_results.items() 
        if "error" not in results and all(test["passed"] for test in results.values())
    ]
    
    if fully_compliant:
        print(f"\n✅ {len(fully_compliant)} Fully Compliant Builders:")
        for builder in sorted(fully_compliant):
            print(f"  • {builder}")
    
    # Print partially compliant builders
    partially_compliant = [
        builder_name for builder_name, results in all_results.items() 
        if "error" not in results and any(test["passed"] for test in results.values()) and not all(test["passed"] for test in results.values())
    ]
    
    if partially_compliant:
        print(f"\n⚠️ {len(partially_compliant)} Partially Compliant Builders:")
        for builder in sorted(partially_compliant):
            results = all_results[builder]
            passed = sum(1 for test in results.values() if test["passed"])
            total = len(results)
            print(f"  • {builder}: {passed}/{total} tests passed")
    
    # Print non-compliant builders
    non_compliant = [
        builder_name for builder_name, results in all_results.items() 
        if "error" not in results and not any(test["passed"] for test in results.values())
    ]
    
    if non_compliant:
        print(f"\n❌ {len(non_compliant)} Non-Compliant Builders:")
        for builder in sorted(non_compliant):
            print(f"  • {builder}")
    
    print("\n" + "=" * 80)
    
    # Generate quality scores if requested
    if generate_scores:
        print("\nGenerating Quality Score Reports...")
        
        for builder_name, results in all_results.items():
            if "error" in results:
                print(f"⚠️ Skipping {builder_name} (failed to run tests)")
                continue
                
            try:
                # Generate quality score report
                score_builder_results(
                    results=results,
                    builder_name=builder_name,
                    save_report=True,
                    output_dir=output_dir,
                    generate_chart=generate_charts
                )
            except Exception as e:
                print(f"⚠️ Error generating score report for {builder_name}: {str(e)}")
        
        print(f"Score reports saved to {output_dir}/")


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run universal step builder tests")
    parser.add_argument("--verbose", action="store_true", help="Print detailed test results")
    parser.add_argument("--builder", help="Test a specific builder class")
    parser.add_argument("--spec", help="Use a specific specification (module.SPEC_NAME format)")
    parser.add_argument("--contract", help="Use a specific contract (module.CONTRACT_NAME format)")
    parser.add_argument("--step-name", help="Use a specific step name")
    parser.add_argument("--score", action="store_true", help="Generate quality scores and reports")
    parser.add_argument("--output-dir", default="test_reports", help="Directory to save reports")
    parser.add_argument("--no-chart", action="store_true", help="Disable chart generation")
    args = parser.parse_args()
    
    # Get available builders
    builders = get_available_step_builders()
    
    # Filter builders if requested
    if args.builder:
        if args.builder in builders:
            builders = {args.builder: builders[args.builder]}
        else:
            print(f"Builder {args.builder} not found. Available builders:")
            for name in sorted(builders.keys()):
                print(f"  • {name}")
            return
    
    # Print available builders
    print(f"Found {len(builders)} step builders:")
    for name in sorted(builders.keys()):
        print(f"  • {name}")
    
    # Check for explicit components
    config = None
    spec = None
    contract = None
    step_name = None
    
    # Import specification if provided
    if args.spec:
        try:
            module_name, var_name = args.spec.rsplit('.', 1)
            module = importlib.import_module(module_name)
            spec = getattr(module, var_name)
            print(f"Using specification: {args.spec}")
        except (ValueError, ImportError, AttributeError) as e:
            print(f"Error importing specification {args.spec}: {e}")
            return
    
    # Import contract if provided
    if args.contract:
        try:
            module_name, var_name = args.contract.rsplit('.', 1)
            module = importlib.import_module(module_name)
            contract = getattr(module, var_name)
            print(f"Using contract: {args.contract}")
        except (ValueError, ImportError, AttributeError) as e:
            print(f"Error importing contract {args.contract}: {e}")
            return
    
    # Use provided step name if available
    if args.step_name:
        step_name = args.step_name
        print(f"Using step name: {step_name}")
    
    # Run tests with specified components
    all_results = run_tests_on_builders(
        builders, 
        verbose=args.verbose,
        spec=spec,
        contract=contract,
        step_name=step_name
    )
    
    # Print summary and generate scores if requested
    print_summary(
        all_results, 
        generate_scores=args.score,
        output_dir=args.output_dir,
        generate_charts=not args.no_chart
    )


if __name__ == "__main__":
    main()
