"""
Comprehensive unit test suite for the step type enhancement system.

This module runs all tests for the step type enhancement system components.
"""

import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import all test modules
from .test_framework_patterns import TestFrameworkPatterns
from .test_step_type_enhancement_router import TestStepTypeEnhancementRouter
from .step_type_enhancers.test_base_enhancer import (
    TestBaseStepEnhancer, 
    TestBaseStepEnhancerEdgeCases
)
from .step_type_enhancers.test_training_enhancer import TestTrainingStepEnhancer

class TestStepTypeEnhancementSystemComprehensive(unittest.TestCase):
    """Comprehensive integration tests for the step type enhancement system."""

    def setUp(self):
        """Set up test fixtures for integration tests."""
        pass

    def test_system_integration_smoke_test(self):
        """Smoke test to ensure all components can be imported and instantiated."""
        # Test that all main components can be imported
        try:
            from cursus.validation.alignment.framework_patterns import detect_framework_from_script_content
            from cursus.validation.alignment.step_type_enhancement_router import StepTypeEnhancementRouter
            from cursus.validation.alignment.step_type_enhancers.base_enhancer import BaseStepEnhancer
            from cursus.validation.alignment.step_type_enhancers.training_enhancer import TrainingStepEnhancer
            
            # Test instantiation
            router = StepTypeEnhancementRouter()
            training_enhancer = TrainingStepEnhancer()
            
            # Basic functionality test
            self.assertIsNotNone(router)
            self.assertIsNotNone(training_enhancer)
            self.assertEqual(training_enhancer.step_type, "Training")
            
        except ImportError as e:
            self.fail(f"Failed to import step type enhancement system components: {e}")

    def test_framework_detection_integration(self):
        """Test framework detection integration across components."""
        from cursus.validation.alignment.framework_patterns import detect_framework_from_script_content
        
        # Test XGBoost detection
        xgboost_content = """
import xgboost as xgb
model = xgb.train(params, dtrain)
"""
        framework = detect_framework_from_script_content(xgboost_content)
        self.assertEqual(framework, 'xgboost')
        
        # Test PyTorch detection
        pytorch_content = """
import torch
import torch.nn as nn
model = nn.Linear(10, 1)
"""
        framework = detect_framework_from_script_content(pytorch_content)
        self.assertEqual(framework, 'pytorch')

    def test_router_enhancer_integration(self):
        """Test integration between router and enhancers."""
        from cursus.validation.alignment.step_type_enhancement_router import StepTypeEnhancementRouter
        from cursus.validation.alignment.core_models import ValidationResult
        
        router = StepTypeEnhancementRouter()
        
        # Test that router has all expected enhancer classes defined
        expected_step_types = ["Processing", "Training", "CreateModel", "Transform", "RegisterModel", "Utility", "Base"]
        for step_type in expected_step_types:
            self.assertIn(step_type, router._enhancer_classes)
            self.assertIsNotNone(router._enhancer_classes[step_type])

    def test_step_type_requirements_completeness(self):
        """Test that all step types have complete requirements."""
        from cursus.validation.alignment.step_type_enhancement_router import StepTypeEnhancementRouter
        
        router = StepTypeEnhancementRouter()
        
        for step_type in router.enhancers.keys():
            requirements = router.get_step_type_requirements(step_type)
            
            # Each step type should have requirements (except Base which might be minimal)
            if step_type != "Base":
                self.assertIsInstance(requirements, dict)
                if requirements:  # If requirements exist, they should be complete
                    expected_keys = ["input_types", "output_types", "required_methods", "required_patterns"]
                    for key in expected_keys:
                        self.assertIn(key, requirements, f"Missing {key} in {step_type} requirements")

def create_test_suite():
    """Create a comprehensive test suite for the step type enhancement system."""
    suite = unittest.TestSuite()
    
    # Add framework patterns tests
    suite.addTest(unittest.makeSuite(TestFrameworkPatterns))
    
    # Add router tests
    suite.addTest(unittest.makeSuite(TestStepTypeEnhancementRouter))
    
    # Add base enhancer tests
    suite.addTest(unittest.makeSuite(TestBaseStepEnhancer))
    suite.addTest(unittest.makeSuite(TestBaseStepEnhancerEdgeCases))
    
    # Add training enhancer tests
    suite.addTest(unittest.makeSuite(TestTrainingStepEnhancer))
    
    # Add integration tests
    suite.addTest(unittest.makeSuite(TestStepTypeEnhancementSystemComprehensive))
    
    return suite

def run_all_tests():
    """Run all step type enhancement system tests."""
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("STEP TYPE ENHANCEMENT SYSTEM TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split('\\n')[0] if 'AssertionError: ' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('\\n')[-2] if traceback else 'Unknown error'}")
    
    print(f"{'='*60}")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run comprehensive test suite
    success = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
