"""
Example usage of the enhanced universal step builder tester system.
"""

from typing import Type
from ...core.base.builder_base import StepBuilderBase
from .test_factory import UniversalStepBuilderTestFactory


def test_step_builder(builder_class: Type[StepBuilderBase], verbose: bool = True):
    """
    Test a step builder using the appropriate variant.
    
    Args:
        builder_class: The step builder class to test
        verbose: Whether to print verbose output
        
    Returns:
        Test results dictionary
    """
    print(f"\n=== Testing {builder_class.__name__} ===")
    
    # Create appropriate tester using factory
    tester = UniversalStepBuilderTestFactory.create_tester(
        builder_class, 
        verbose=verbose
    )
    
    # Print detected step information
    if verbose:
        print(f"Step Type: {tester.step_info.get('sagemaker_step_type', 'Unknown')}")
        print(f"Framework: {tester.step_info.get('framework', 'Unknown')}")
        print(f"Test Pattern: {tester.step_info.get('test_pattern', 'standard')}")
        print(f"Tester Variant: {tester.__class__.__name__}")
        print()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Print summary
    passed = sum(1 for result in results.values() if result["passed"])
    total = len(results)
    
    print(f"\n=== SUMMARY ===")
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
        for test_name, result in results.items():
            if not result["passed"]:
                print(f"  - {test_name}: {result.get('error', 'Unknown error')}")
    
    return results


def demonstrate_factory_system():
    """Demonstrate the factory system capabilities."""
    print("=== Universal Step Builder Tester Factory System ===\n")
    
    # Initialize factory
    UniversalStepBuilderTestFactory._initialize_variants()
    
    # Show available variants
    variants = UniversalStepBuilderTestFactory.get_available_variants()
    print(f"Available Variants: {list(variants.keys())}")
    
    # Show supported step types
    supported_types = [step_type for step_type in variants.keys()]
    print(f"Supported Step Types: {supported_types}")
    
    print("\nFactory System Features:")
    print("- Automatic step type detection")
    print("- Step type-specific mock creation")
    print("- Framework-aware testing")
    print("- Extensible variant system")
    print("- Comprehensive validation patterns")
    
    return variants


if __name__ == "__main__":
    # Demonstrate the system
    demonstrate_factory_system()
    
    # Example usage (would need actual builder classes):
    # from ...steps.builders.xgboost_training_step_builder import XGBoostTrainingStepBuilder
    # test_step_builder(XGBoostTrainingStepBuilder)
