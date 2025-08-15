"""
Transform Step Interface Tests (Level 1).

This module provides Level 1 interface validation tests specifically for Transform step builders.
These tests focus on Transform-specific interface requirements, method signatures, and
basic functionality validation for batch inference and model transformation workflows.
"""

from typing import Dict, Any, List, Optional, Type
import inspect

from ..interface_tests import InterfaceTests
from ....core.base.builder_base import StepBuilderBase


class TransformInterfaceTests(InterfaceTests):
    """
    Level 1 interface tests specifically for Transform step builders.
    
    Extends the base InterfaceTests with Transform-specific interface validation
    including transformer creation, batch processing configuration, and model integration.
    """
    
    def __init__(self, builder_class: Type[StepBuilderBase], **kwargs):
        """Initialize Transform interface tests."""
        super().__init__(builder_class, **kwargs)
        self.step_type = "Transform"
    
    def level1_test_transformer_creation_method(self) -> Dict[str, Any]:
        """
        Test that the builder has transformer creation methods.
        
        Transform builders should have methods to create Transformer instances
        for batch inference operations.
        """
        try:
            transformer_methods = ['_create_transformer', '_get_transformer']
            found_methods = []
            
            for method_name in transformer_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(method_name)
            
            if not found_methods:
                return {
                    "passed": False,
                    "error": f"No transformer creation methods found. Expected one of: {transformer_methods}",
                    "details": {
                        "expected_methods": transformer_methods,
                        "found_methods": found_methods,
                        "available_methods": [m for m in dir(self.builder_class) if not m.startswith('__')]
                    }
                }
            
            # Validate method signatures
            method_details = {}
            for method_name in found_methods:
                method = getattr(self.builder_class, method_name)
                sig = inspect.signature(method)
                method_details[method_name] = {
                    "signature": str(sig),
                    "parameters": list(sig.parameters.keys())
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_methods": found_methods,
                    "method_details": method_details,
                    "validation": "Transform builder has transformer creation methods"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing transformer creation methods: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level1_test_transform_input_preparation_methods(self) -> Dict[str, Any]:
        """
        Test that the builder has transform input preparation methods.
        
        Transform builders should have methods to prepare TransformInput
        configurations for batch processing.
        """
        try:
            input_methods = [
                '_prepare_transform_input', '_get_transform_input', '_create_transform_input',
                '_configure_transform_input', '_setup_transform_input'
            ]
            found_methods = []
            
            for method_name in input_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(method_name)
            
            # Also check for generic input methods that might handle transform inputs
            generic_input_methods = ['_get_inputs', '_prepare_inputs']
            for method_name in generic_input_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(f"{method_name} (generic)")
            
            if not found_methods:
                return {
                    "passed": False,
                    "error": f"No transform input preparation methods found. Expected one of: {input_methods}",
                    "details": {
                        "expected_methods": input_methods,
                        "found_methods": found_methods,
                        "note": "Transform builders should have methods to prepare TransformInput configurations"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_methods": found_methods,
                    "validation": "Transform builder has input preparation methods"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing transform input preparation methods: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level1_test_batch_processing_configuration_methods(self) -> Dict[str, Any]:
        """
        Test that the builder has batch processing configuration methods.
        
        Transform builders should have methods to configure batch processing
        parameters like batch size, concurrent transforms, and payload limits.
        """
        try:
            batch_config_methods = [
                '_configure_batch_processing', '_setup_batch_config', '_get_batch_config',
                '_configure_transform_job', '_setup_transform_config'
            ]
            found_methods = []
            
            for method_name in batch_config_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(method_name)
            
            # Check for batch-related attributes or properties
            batch_attributes = []
            for attr_name in dir(self.builder_class):
                if any(keyword in attr_name.lower() for keyword in ['batch', 'concurrent', 'payload', 'transform']):
                    if not attr_name.startswith('__') and not callable(getattr(self.builder_class, attr_name, None)):
                        batch_attributes.append(attr_name)
            
            # This is informational - not all builders need explicit batch config methods
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_methods": found_methods,
                    "batch_attributes": batch_attributes,
                    "validation": "Transform builder batch processing configuration checked",
                    "note": "Batch configuration methods are recommended but not required"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing batch processing configuration methods: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level1_test_model_integration_methods(self) -> Dict[str, Any]:
        """
        Test that the builder has model integration methods.
        
        Transform builders should have methods to integrate with trained models
        from previous steps (training or model creation steps).
        """
        try:
            model_integration_methods = [
                'integrate_with_model_step', 'set_model_name', 'configure_model_source',
                '_setup_model_integration', '_configure_model_dependency'
            ]
            found_methods = []
            
            for method_name in model_integration_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(method_name)
            
            # Check for model-related attributes
            model_attributes = []
            for attr_name in dir(self.builder_class):
                if any(keyword in attr_name.lower() for keyword in ['model', 'transformer']):
                    if not attr_name.startswith('__'):
                        model_attributes.append(attr_name)
            
            if not found_methods and not model_attributes:
                return {
                    "passed": False,
                    "error": "No model integration methods or attributes found",
                    "details": {
                        "expected_methods": model_integration_methods,
                        "found_methods": found_methods,
                        "model_attributes": model_attributes,
                        "note": "Transform builders should have ways to integrate with trained models"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_methods": found_methods,
                    "model_attributes": model_attributes,
                    "validation": "Transform builder has model integration capabilities"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing model integration methods: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level1_test_output_configuration_methods(self) -> Dict[str, Any]:
        """
        Test that the builder has output configuration methods.
        
        Transform builders should have methods to configure transform outputs
        including output paths, formats, and assembly options.
        """
        try:
            output_methods = [
                '_configure_transform_output', '_setup_output_config', '_get_transform_output',
                '_prepare_output_configuration', '_setup_output_path'
            ]
            found_methods = []
            
            for method_name in output_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(method_name)
            
            # Also check for generic output methods
            generic_output_methods = ['_get_outputs', '_prepare_outputs']
            for method_name in generic_output_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(f"{method_name} (generic)")
            
            # Check for output-related attributes
            output_attributes = []
            for attr_name in dir(self.builder_class):
                if any(keyword in attr_name.lower() for keyword in ['output', 'result', 'prediction']):
                    if not attr_name.startswith('__') and not callable(getattr(self.builder_class, attr_name, None)):
                        output_attributes.append(attr_name)
            
            if not found_methods and not output_attributes:
                return {
                    "passed": False,
                    "error": "No output configuration methods or attributes found",
                    "details": {
                        "expected_methods": output_methods,
                        "found_methods": found_methods,
                        "output_attributes": output_attributes,
                        "note": "Transform builders should have methods to configure outputs"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_methods": found_methods,
                    "output_attributes": output_attributes,
                    "validation": "Transform builder has output configuration capabilities"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing output configuration methods: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level1_test_framework_specific_methods(self) -> Dict[str, Any]:
        """
        Test for framework-specific methods in Transform builders.
        
        Different ML frameworks (XGBoost, PyTorch, SKLearn) may require
        specific transform configurations and inference patterns.
        """
        try:
            framework_patterns = {
                'xgboost': ['xgb', 'xgboost', 'dmatrix'],
                'pytorch': ['torch', 'pytorch', 'tensor'],
                'sklearn': ['sklearn', 'scikit', 'estimator'],
                'tensorflow': ['tensorflow', 'tf', 'keras']
            }
            
            detected_frameworks = []
            framework_methods = []
            
            # Check class name and methods for framework indicators
            class_name = self.builder_class.__name__.lower()
            methods = [method for method in dir(self.builder_class) if not method.startswith('__')]
            
            for framework, patterns in framework_patterns.items():
                if any(pattern in class_name for pattern in patterns):
                    detected_frameworks.append(framework)
                
                framework_specific_methods = [
                    method for method in methods 
                    if any(pattern in method.lower() for pattern in patterns)
                ]
                if framework_specific_methods:
                    framework_methods.extend(framework_specific_methods)
                    if framework not in detected_frameworks:
                        detected_frameworks.append(framework)
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "detected_frameworks": detected_frameworks,
                    "framework_methods": framework_methods,
                    "validation": "Framework-specific method detection completed",
                    "note": "Framework-specific methods are optional but recommended for specialized transforms"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing framework-specific methods: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level1_test_step_creation_pattern_compliance(self) -> Dict[str, Any]:
        """
        Test that the Transform builder follows step creation patterns.
        
        Transform builders should create TransformStep instances with proper
        transformer configuration and input/output handling.
        """
        try:
            # Check for step creation method
            if not hasattr(self.builder_class, 'create_step'):
                return {
                    "passed": False,
                    "error": "Missing create_step method",
                    "details": {
                        "note": "Transform builders must implement create_step method"
                    }
                }
            
            create_step_method = getattr(self.builder_class, 'create_step')
            sig = inspect.signature(create_step_method)
            
            # Check method signature
            method_details = {
                "signature": str(sig),
                "parameters": list(sig.parameters.keys())
            }
            
            # Check for Transform-specific step creation patterns
            transform_patterns = []
            
            # Look for transformer-related method calls in the class
            for method_name in dir(self.builder_class):
                if 'transform' in method_name.lower() and not method_name.startswith('__'):
                    transform_patterns.append(method_name)
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "create_step_method": method_details,
                    "transform_patterns": transform_patterns,
                    "validation": "Transform step creation pattern compliance checked"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing step creation pattern compliance: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all Transform-specific Level 1 interface tests.
        
        Returns:
            Dictionary mapping test names to their results
        """
        tests = {
            # Base interface tests
            "test_inheritance": self.test_inheritance,
            "test_required_methods": self.test_required_methods,
            "test_error_handling": self.test_error_handling,
            
            # Transform-specific interface tests
            "level1_test_transformer_creation_method": self.level1_test_transformer_creation_method,
            "level1_test_transform_input_preparation_methods": self.level1_test_transform_input_preparation_methods,
            "level1_test_batch_processing_configuration_methods": self.level1_test_batch_processing_configuration_methods,
            "level1_test_model_integration_methods": self.level1_test_model_integration_methods,
            "level1_test_output_configuration_methods": self.level1_test_output_configuration_methods,
            "level1_test_framework_specific_methods": self.level1_test_framework_specific_methods,
            "level1_test_step_creation_pattern_compliance": self.level1_test_step_creation_pattern_compliance,
        }
        
        results = {}
        for test_name, test_method in tests.items():
            try:
                if self.verbose:
                    print(f"Running {test_name}...")
                results[test_name] = test_method()
            except Exception as e:
                results[test_name] = {
                    "passed": False,
                    "error": f"Test execution failed: {str(e)}",
                    "details": {"exception": str(e)}
                }
        
        return results


# Convenience function for quick Transform interface validation
def validate_transform_interface(builder_class: Type[StepBuilderBase], 
                                verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for Transform step builder interfaces.
    
    Args:
        builder_class: The Transform step builder class to validate
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing test results
    """
    tester = TransformInterfaceTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
