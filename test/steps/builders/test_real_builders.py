#!/usr/bin/env python3
"""
Example script demonstrating how to test real step builders using the enhanced universal tester.

This script shows how to use the enhanced testing system to validate actual step builders
from the codebase, demonstrating the improvements made based on the Next Steps action items.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import cursus modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cursus.validation.builders.test_factory import TestFactory


def test_tabular_preprocessing_builder():
    """Test the TabularPreprocessingStepBuilder with enhanced system."""
    print("=" * 80)
    print("Testing TabularPreprocessingStepBuilder")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        
        # Create test factory and run tests
        factory = TestFactory()
        results = factory.test_builder(TabularPreprocessingStepBuilder, verbose=True)
        
        # Print results
        print(f"\nTest Results for {TabularPreprocessingStepBuilder.__name__}:")
        print("-" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('passed', False))
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            print(f"{status} {test_name}")
            if not result.get('passed', False) and 'error' in result:
                print(f"    Error: {result['error']}")
        
        print(f"\nSummary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        return results
        
    except ImportError as e:
        print(f"Could not import TabularPreprocessingStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing TabularPreprocessingStepBuilder: {e}")
        return None


def test_xgboost_training_builder():
    """Test the XGBoostTrainingStepBuilder with enhanced system."""
    print("\n" + "=" * 80)
    print("Testing XGBoostTrainingStepBuilder")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_training_step_xgboost import XGBoostTrainingStepBuilder
        
        # Create test factory and run tests
        factory = TestFactory()
        results = factory.test_builder(XGBoostTrainingStepBuilder, verbose=True)
        
        # Print results
        print(f"\nTest Results for {XGBoostTrainingStepBuilder.__name__}:")
        print("-" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('passed', False))
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            print(f"{status} {test_name}")
            if not result.get('passed', False) and 'error' in result:
                print(f"    Error: {result['error']}")
        
        print(f"\nSummary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        return results
        
    except ImportError as e:
        print(f"Could not import XGBoostTrainingStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing XGBoostTrainingStepBuilder: {e}")
        return None


def test_model_eval_builder():
    """Test the ModelEvalStepBuilder with enhanced system."""
    print("\n" + "=" * 80)
    print("Testing ModelEvalStepBuilder")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_model_eval_step_xgboost import ModelEvalStepBuilder
        
        # Create test factory and run tests
        factory = TestFactory()
        results = factory.test_builder(ModelEvalStepBuilder, verbose=True)
        
        # Print results
        print(f"\nTest Results for {ModelEvalStepBuilder.__name__}:")
        print("-" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('passed', False))
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            print(f"{status} {test_name}")
            if not result.get('passed', False) and 'error' in result:
                print(f"    Error: {result['error']}")
        
        print(f"\nSummary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        return results
        
    except ImportError as e:
        print(f"Could not import ModelEvalStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing ModelEvalStepBuilder: {e}")
        return None


def test_pytorch_training_builder():
    """Test the PyTorchTrainingStepBuilder with enhanced system."""
    print("\n" + "=" * 80)
    print("Testing PyTorchTrainingStepBuilder")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_training_step_pytorch import PyTorchTrainingStepBuilder
        
        # Create test factory and run tests
        factory = TestFactory()
        results = factory.test_builder(PyTorchTrainingStepBuilder, verbose=True)
        
        # Print results
        print(f"\nTest Results for {PyTorchTrainingStepBuilder.__name__}:")
        print("-" * 60)
        
        total_tests = len(results)
        passed_tests = sum(1 for result in results.values() if result.get('passed', False))
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            print(f"{status} {test_name}")
            if not result.get('passed', False) and 'error' in result:
                print(f"    Error: {result['error']}")
        
        print(f"\nSummary: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        return results
        
    except ImportError as e:
        print(f"Could not import PyTorchTrainingStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing PyTorchTrainingStepBuilder: {e}")
        return None


def main():
    """Run tests on multiple real step builders."""
    print("Enhanced Universal Step Builder Tester - Real Builder Testing")
    print("=" * 80)
    print("This script demonstrates the improvements made based on the Next Steps action items:")
    print("1. ✅ Verified full implementation of path mapping tests")
    print("2. ✅ Enhanced mock factory integration with realistic step builder patterns")
    print("3. ✅ Improved step type-specific validation")
    print("4. ✅ Added comprehensive property path validation")
    print("=" * 80)
    
    # Test different types of step builders
    all_results = {}
    
    # Test Processing step builder
    tabular_results = test_tabular_preprocessing_builder()
    if tabular_results:
        all_results['TabularPreprocessingStepBuilder'] = tabular_results
    
    # Test Training step builders
    xgboost_results = test_xgboost_training_builder()
    if xgboost_results:
        all_results['XGBoostTrainingStepBuilder'] = xgboost_results
    
    pytorch_results = test_pytorch_training_builder()
    if pytorch_results:
        all_results['PyTorchTrainingStepBuilder'] = pytorch_results
    
    # Test Processing step builder (model evaluation)
    model_eval_results = test_model_eval_builder()
    if model_eval_results:
        all_results['ModelEvalStepBuilder'] = model_eval_results
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    if all_results:
        for builder_name, results in all_results.items():
            total_tests = len(results)
            passed_tests = sum(1 for result in results.values() if result.get('passed', False))
            success_rate = passed_tests/total_tests*100 if total_tests > 0 else 0
            
            status_icon = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            print(f"{status_icon} {builder_name}: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        # Calculate overall statistics
        total_all_tests = sum(len(results) for results in all_results.values())
        passed_all_tests = sum(
            sum(1 for result in results.values() if result.get('passed', False))
            for results in all_results.values()
        )
        overall_success_rate = passed_all_tests/total_all_tests*100 if total_all_tests > 0 else 0
        
        print("-" * 80)
        print(f"🎯 OVERALL: {passed_all_tests}/{total_all_tests} tests passed ({overall_success_rate:.1f}%)")
        
        if overall_success_rate >= 80:
            print("🎉 Excellent! The enhanced universal tester is working well.")
        elif overall_success_rate >= 60:
            print("👍 Good progress! Some areas may need attention.")
        else:
            print("🔧 Needs improvement. Check the failed tests for issues.")
    else:
        print("❌ No builders could be tested. Check import paths and dependencies.")
    
    print("\n" + "=" * 80)
    print("Enhanced Universal Step Builder Tester - Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
