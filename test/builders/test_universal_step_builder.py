"""
Universal Step Builder Test - Legacy Entry Point

This module provides backward compatibility with the original 
test_universal_step_builder.py interface by importing from the 
refactored universal_step_builder_test package.
"""

import unittest
from typing import Dict, Any, Optional, Union, Type

from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_deps.base_specifications import StepSpecification
from src.pipeline_script_contracts.base_script_contract import ScriptContract
from src.pipeline_steps.config_base import BaseModel as ConfigBase

# Import the refactored implementation
from .universal_step_builder_test import UniversalStepBuilderTest
from .universal_step_builder_test.base_test import StepName


# Re-export the TestUniversalStepBuilder class for backward compatibility
class TestUniversalStepBuilder(unittest.TestCase):
    """
    Test cases for the UniversalStepBuilderTest class itself.
    
    These tests verify that the universal test suite works correctly
    by applying it to known step builders.
    """
    
    def test_with_xgboost_training_builder(self):
        """Test UniversalStepBuilderTest with XGBoostTrainingStepBuilder."""
        try:
            # Import the builder class
            from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
            
            # Create tester
            tester = UniversalStepBuilderTest(XGBoostTrainingStepBuilder)
            
            # Run all tests
            results = tester.run_all_tests()
            
            # Check that key tests passed
            self.assertTrue(results["test_inheritance"]["passed"])
            self.assertTrue(results["test_required_methods"]["passed"])
        except ImportError:
            self.skipTest("XGBoostTrainingStepBuilder not available")
    
    def test_with_tabular_preprocessing_builder(self):
        """Test UniversalStepBuilderTest with TabularPreprocessingStepBuilder."""
        try:
            # Import the builder class
            from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
            
            # Create tester
            tester = UniversalStepBuilderTest(TabularPreprocessingStepBuilder)
            
            # Run all tests
            results = tester.run_all_tests()
            
            # Check that key tests passed
            self.assertTrue(results["test_inheritance"]["passed"])
            self.assertTrue(results["test_required_methods"]["passed"])
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder not available")

    def test_with_explicit_components(self):
        """Test UniversalStepBuilderTest with explicitly provided components."""
        try:
            # Import the builder class
            from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
            from src.pipeline_step_specs.preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
            
            # Create a custom configuration
            from types import SimpleNamespace
            config = SimpleNamespace()
            config.region = 'NA'
            config.pipeline_name = 'test-pipeline'
            config.job_type = 'training'
            
            # Create tester with explicit components
            tester = UniversalStepBuilderTest(
                TabularPreprocessingStepBuilder,
                config=config,
                spec=PREPROCESSING_TRAINING_SPEC,
                step_name='CustomPreprocessingStep'
            )
            
            # Run all tests
            results = tester.run_all_tests()
            
            # Check that key tests passed
            self.assertTrue(results["test_inheritance"]["passed"])
        except ImportError:
            self.skipTest("TabularPreprocessingStepBuilder or PREPROCESSING_TRAINING_SPEC not available")


if __name__ == '__main__':
    unittest.main()
