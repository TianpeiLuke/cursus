"""
Comprehensive unit test suite for the step type enhancement system.

This module runs all tests for the step type enhancement system components.
"""

import pytest
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

class TestStepTypeEnhancementSystemComprehensive:
    """Comprehensive integration tests for the step type enhancement system."""

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
            assert router is not None
            assert training_enhancer is not None
            assert training_enhancer.step_type == "Training"
            
        except ImportError as e:
            pytest.fail(f"Failed to import step type enhancement system components: {e}")

    def test_framework_detection_integration(self):
        """Test framework detection integration across components."""
        from cursus.validation.alignment.framework_patterns import detect_framework_from_script_content
        
        # Test XGBoost detection
        xgboost_content = """
import xgboost as xgb
model = xgb.train(params, dtrain)
"""
        framework = detect_framework_from_script_content(xgboost_content)
        assert framework == 'xgboost'
        
        # Test PyTorch detection
        pytorch_content = """
import torch
import torch.nn as nn
model = nn.Linear(10, 1)
"""
        framework = detect_framework_from_script_content(pytorch_content)
        assert framework == 'pytorch'

    def test_router_enhancer_integration(self):
        """Test integration between router and enhancers."""
        from cursus.validation.alignment.step_type_enhancement_router import StepTypeEnhancementRouter
        from cursus.validation.alignment.core_models import ValidationResult
        
        router = StepTypeEnhancementRouter()
        
        # Test that router has all expected enhancer classes defined
        expected_step_types = ["Processing", "Training", "CreateModel", "Transform", "RegisterModel", "Utility", "Base"]
        for step_type in expected_step_types:
            assert step_type in router._enhancer_classes
            assert router._enhancer_classes[step_type] is not None

    def test_step_type_requirements_completeness(self):
        """Test that all step types have complete requirements."""
        from cursus.validation.alignment.step_type_enhancement_router import StepTypeEnhancementRouter
        
        router = StepTypeEnhancementRouter()
        
        for step_type in router.enhancers.keys():
            requirements = router.get_step_type_requirements(step_type)
            
            # Each step type should have requirements (except Base which might be minimal)
            if step_type != "Base":
                assert isinstance(requirements, dict)
                if requirements:  # If requirements exist, they should be complete
                    expected_keys = ["input_types", "output_types", "required_methods", "required_patterns"]
                    for key in expected_keys:
                        assert key in requirements, f"Missing {key} in {step_type} requirements"

if __name__ == '__main__':
    # Run tests using pytest
    pytest.main([__file__, '-v'])
