#!/usr/bin/env python3
"""
Example script demonstrating how to test real step builders using the enhanced universal tester.

This script shows how to use the enhanced testing system with integrated scoring and reporting
capabilities to validate actual step builders from the codebase.
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import cursus modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import directly from the module to avoid any caching issues
from cursus.validation.builders.universal_test import UniversalStepBuilderTest

def test_tabular_preprocessing_builder():
    """Test the TabularPreprocessingStepBuilder with enhanced system."""
    print("=" * 80)
    print("Testing TabularPreprocessingStepBuilder with Enhanced Universal Tester")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        
        # Create enhanced universal tester with scoring and verbose output
        tester = UniversalStepBuilderTest(
            TabularPreprocessingStepBuilder, 
            verbose=True,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run all tests with enhanced features
        results = tester.run_all_tests()
        
        # Extract test results and scoring
        test_results = results.get('test_results', {})
        scoring = results.get('scoring', {})
        
        # Print additional scoring information
        if scoring:
            overall_score = scoring.get('overall', {}).get('score', 0)
            overall_rating = scoring.get('overall', {}).get('rating', 'Unknown')
            print(f"\n📊 Quality Score: {overall_score:.1f}/100 - {overall_rating}")
        
        return test_results
        
    except ImportError as e:
        print(f"Could not import TabularPreprocessingStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing TabularPreprocessingStepBuilder: {e}")
        return None

def test_xgboost_training_builder():
    """Test the XGBoostTrainingStepBuilder with enhanced system."""
    print("\n" + "=" * 80)
    print("Testing XGBoostTrainingStepBuilder with Enhanced Universal Tester")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_xgboost_training_step import XGBoostTrainingStepBuilder
        
        # Create enhanced universal tester with scoring and verbose output
        tester = UniversalStepBuilderTest(
            XGBoostTrainingStepBuilder, 
            verbose=True,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run all tests with enhanced features
        results = tester.run_all_tests()
        
        # Extract test results and scoring
        test_results = results.get('test_results', {})
        scoring = results.get('scoring', {})
        
        # Print additional scoring information
        if scoring:
            overall_score = scoring.get('overall', {}).get('score', 0)
            overall_rating = scoring.get('overall', {}).get('rating', 'Unknown')
            print(f"\n📊 Quality Score: {overall_score:.1f}/100 - {overall_rating}")
        
        return test_results
        
    except ImportError as e:
        print(f"Could not import XGBoostTrainingStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing XGBoostTrainingStepBuilder: {e}")
        return None

def test_model_eval_builder():
    """Test the ModelEvalStepBuilder with enhanced system."""
    print("\n" + "=" * 80)
    print("Testing XGBoostModelEvalStepBuilder with Enhanced Universal Tester")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_xgboost_model_eval_step import XGBoostModelEvalStepBuilder
        
        # Create enhanced universal tester with scoring and verbose output
        tester = UniversalStepBuilderTest(
            XGBoostModelEvalStepBuilder, 
            verbose=True,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run all tests with enhanced features
        results = tester.run_all_tests()
        
        # Extract test results and scoring
        test_results = results.get('test_results', {})
        scoring = results.get('scoring', {})
        
        # Print additional scoring information
        if scoring:
            overall_score = scoring.get('overall', {}).get('score', 0)
            overall_rating = scoring.get('overall', {}).get('rating', 'Unknown')
            print(f"\n📊 Quality Score: {overall_score:.1f}/100 - {overall_rating}")
        
        return test_results
        
    except ImportError as e:
        print(f"Could not import XGBoostModelEvalStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing XGBoostModelEvalStepBuilder: {e}")
        return None

def test_pytorch_training_builder():
    """Test the PyTorchTrainingStepBuilder with enhanced system."""
    print("\n" + "=" * 80)
    print("Testing PyTorchTrainingStepBuilder with Enhanced Universal Tester")
    print("=" * 80)
    
    try:
        # Import the builder
        from cursus.steps.builders.builder_pytorch_training_step import PyTorchTrainingStepBuilder
        
        # Create enhanced universal tester with scoring and verbose output
        tester = UniversalStepBuilderTest(
            PyTorchTrainingStepBuilder, 
            verbose=True,
            enable_scoring=True,
            enable_structured_reporting=True
        )
        
        # Run all tests with enhanced features
        results = tester.run_all_tests()
        
        # Extract test results and scoring
        test_results = results.get('test_results', {})
        scoring = results.get('scoring', {})
        
        # Print additional scoring information
        if scoring:
            overall_score = scoring.get('overall', {}).get('score', 0)
            overall_rating = scoring.get('overall', {}).get('rating', 'Unknown')
            print(f"\n📊 Quality Score: {overall_score:.1f}/100 - {overall_rating}")
        
        return test_results
        
    except ImportError as e:
        print(f"Could not import PyTorchTrainingStepBuilder: {e}")
        return None
    except Exception as e:
        print(f"Error testing PyTorchTrainingStepBuilder: {e}")
        return None

def main():
    """Run tests on multiple real step builders with enhanced scoring and reporting."""
    print("Enhanced Universal Step Builder Tester - Real Builder Testing")
    print("=" * 80)
    print("This script demonstrates the enhanced testing system with integrated capabilities:")
    print("1. ✅ Comprehensive scoring system with weighted test levels")
    print("2. ✅ Quality ratings and quantitative assessment (0-100 scores)")
    print("3. ✅ Rich console output with scoring breakdown")
    print("4. ✅ Structured reporting for detailed analysis")
    print("5. ✅ Backward compatibility with existing test patterns")
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
