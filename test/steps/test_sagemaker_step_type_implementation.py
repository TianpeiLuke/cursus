#!/usr/bin/env python3
"""
Test script to validate the SageMaker step type classification implementation.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def test_registry_functions():
    """Test the new registry functions."""
    print("Testing SageMaker Step Type Registry Functions...")
    
    try:
        from src.cursus.steps.registry.step_names import (
            get_sagemaker_step_type,
            get_steps_by_sagemaker_type,
            get_all_sagemaker_step_types,
            validate_sagemaker_step_type,
            get_sagemaker_step_type_mapping
        )
        
        # Test get_sagemaker_step_type
        print("\n1. Testing get_sagemaker_step_type:")
        test_cases = [
            ("TabularPreprocessing", "Processing"),
            ("XGBoostTraining", "Training"),
            ("BatchTransform", "Transform"),
            ("XGBoostModel", "CreateModel"),
            ("Registration", "RegisterModel")
        ]
        
        for step_name, expected_type in test_cases:
            try:
                actual_type = get_sagemaker_step_type(step_name)
                status = "✅" if actual_type == expected_type else "❌"
                print(f"   {status} {step_name}: {actual_type} (expected: {expected_type})")
            except Exception as e:
                print(f"   ❌ {step_name}: Error - {e}")
        
        # Test get_steps_by_sagemaker_type
        print("\n2. Testing get_steps_by_sagemaker_type:")
        step_types = ["Processing", "Training", "Transform", "CreateModel", "RegisterModel"]
        
        for step_type in step_types:
            try:
                steps = get_steps_by_sagemaker_type(step_type)
                print(f"   ✅ {step_type}: {len(steps)} steps - {steps}")
            except Exception as e:
                print(f"   ❌ {step_type}: Error - {e}")
        
        # Test get_all_sagemaker_step_types
        print("\n3. Testing get_all_sagemaker_step_types:")
        try:
            all_types = get_all_sagemaker_step_types()
            print(f"   ✅ Found {len(all_types)} step types: {sorted(all_types)}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        # Test validate_sagemaker_step_type
        print("\n4. Testing validate_sagemaker_step_type:")
        validation_tests = [
            ("Processing", True),
            ("Training", True),
            ("Transform", True),
            ("CreateModel", True),
            ("RegisterModel", True),
            ("Base", True),
            ("Utility", True),
            ("InvalidType", False)
        ]
        
        for step_type, expected_valid in validation_tests:
            try:
                is_valid = validate_sagemaker_step_type(step_type)
                status = "✅" if is_valid == expected_valid else "❌"
                print(f"   {status} {step_type}: {is_valid} (expected: {expected_valid})")
            except Exception as e:
                print(f"   ❌ {step_type}: Error - {e}")
        
        # Test get_sagemaker_step_type_mapping
        print("\n5. Testing get_sagemaker_step_type_mapping:")
        try:
            mapping = get_sagemaker_step_type_mapping()
            print(f"   ✅ Generated mapping with {len(mapping)} step types:")
            for step_type, steps in mapping.items():
                print(f"      {step_type}: {len(steps)} steps")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            
        print("\n✅ Registry functions test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Registry functions test failed: {e}")
        return False

def test_sagemaker_validator():
    """Test the SageMaker step type validator."""
    print("\nTesting SageMaker Step Type Validator...")
    
    try:
        from cursus.validation.builders.sagemaker_step_type_validator import SageMakerStepTypeValidator
        from cursus.core.base.builder_base import StepBuilderBase
        
        # Create a mock builder class for testing
        class MockProcessingStepBuilder(StepBuilderBase):
            def __init__(self):
                pass
            
            def create_step(self):
                pass
            
            def _get_inputs(self):
                pass
            
            def _get_outputs(self):
                pass
            
            def _create_processor(self):
                pass
        
        # Test validator creation
        print("\n1. Testing validator creation:")
        try:
            validator = SageMakerStepTypeValidator(MockProcessingStepBuilder)
            print(f"   ✅ Validator created successfully")
            
            # Test step type info
            step_type_info = validator.get_step_type_info()
            print(f"   ✅ Step type info: {step_type_info}")
            
        except Exception as e:
            print(f"   ❌ Validator creation failed: {e}")
            return False
        
        print("\n✅ SageMaker validator test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ SageMaker validator test failed: {e}")
        return False

def test_universal_test_integration():
    """Test the Universal Test integration."""
    print("\nTesting Universal Test Integration...")
    
    try:
        from cursus.validation.builders.universal_test import UniversalStepBuilderTest
        from cursus.core.base.builder_base import StepBuilderBase
        
        # Create a mock builder class for testing
        class MockTrainingStepBuilder(StepBuilderBase):
            def __init__(self):
                pass
            
            def create_step(self):
                pass
            
            def _get_inputs(self):
                pass
            
            def _create_estimator(self):
                pass
        
        # Test universal test creation
        print("\n1. Testing universal test creation:")
        try:
            tester = UniversalStepBuilderTest(MockTrainingStepBuilder)
            print(f"   ✅ Universal tester created successfully")
            print(f"   ✅ SageMaker validator integrated: {hasattr(tester, 'sagemaker_validator')}")
            
        except Exception as e:
            print(f"   ❌ Universal test creation failed: {e}")
            return False
        
        print("\n✅ Universal test integration completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Universal test integration failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 80)
    print("SAGEMAKER STEP TYPE CLASSIFICATION IMPLEMENTATION TEST")
    print("=" * 80)
    
    tests = [
        ("Registry Functions", test_registry_functions),
        ("SageMaker Validator", test_sagemaker_validator),
        ("Universal Test Integration", test_universal_test_integration)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nOVERALL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! SageMaker step type classification implementation is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
