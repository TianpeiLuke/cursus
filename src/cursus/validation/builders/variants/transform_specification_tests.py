"""
Transform Step Specification Tests (Level 2).

This module provides Level 2 specification validation tests specifically for Transform step builders.
These tests focus on Transform-specific specification compliance, contract alignment, and
batch processing configuration validation for model inference workflows.
"""

from typing import Dict, Any, List, Optional, Type
import os

from ..specification_tests import SpecificationTests
from ....core.base.builder_base import StepBuilderBase


class TransformSpecificationTests(SpecificationTests):
    """
    Level 2 specification tests specifically for Transform step builders.
    
    Extends the base SpecificationTests with Transform-specific specification validation
    including batch processing configuration, model integration, and transform job specifications.
    """
    
    def __init__(self, builder_class: Type[StepBuilderBase], **kwargs):
        """Initialize Transform specification tests."""
        super().__init__(builder_class, **kwargs)
        self.step_type = "Transform"
    
    def level2_test_batch_processing_specification_compliance(self) -> Dict[str, Any]:
        """
        Test that the builder follows batch processing specification patterns.
        
        Transform builders should properly configure batch processing parameters
        including batch size, concurrent transforms, and payload limits.
        """
        try:
            # Check for batch processing configuration attributes
            batch_config_attributes = [
                'batch_size', 'max_concurrent_transforms', 'max_payload',
                'batch_strategy', 'instance_count', 'instance_type'
            ]
            
            found_attributes = []
            for attr_name in batch_config_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_attributes.append(attr_name)
            
            # Check for batch processing methods
            batch_methods = [
                '_configure_batch_processing', '_setup_batch_config',
                '_get_batch_strategy', '_configure_transform_job'
            ]
            
            found_methods = []
            for method_name in batch_methods:
                if hasattr(self.builder_class, method_name):
                    found_methods.append(method_name)
            
            # Check specification alignment
            spec_compliance = {
                "batch_configuration": len(found_attributes) > 0 or len(found_methods) > 0,
                "found_attributes": found_attributes,
                "found_methods": found_methods
            }
            
            if not spec_compliance["batch_configuration"]:
                return {
                    "passed": False,
                    "error": "No batch processing configuration found",
                    "details": {
                        "expected_attributes": batch_config_attributes,
                        "expected_methods": batch_methods,
                        "found_attributes": found_attributes,
                        "found_methods": found_methods,
                        "note": "Transform builders should configure batch processing parameters"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "spec_compliance": spec_compliance,
                    "validation": "Transform batch processing specification compliance verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing batch processing specification compliance: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level2_test_model_integration_specification(self) -> Dict[str, Any]:
        """
        Test that the builder follows model integration specification patterns.
        
        Transform builders should properly integrate with trained models from
        previous steps and configure model dependencies correctly.
        """
        try:
            # Check for model integration attributes
            model_attributes = [
                'model_name', 'model_data', 'model_source', 'model_step',
                'transformer', 'model_uri', 'model_package_name'
            ]
            
            found_model_attributes = []
            for attr_name in model_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_model_attributes.append(attr_name)
            
            # Check for model integration methods
            model_methods = [
                'integrate_with_model_step', 'set_model_name', 'configure_model_source',
                '_setup_model_integration', '_configure_model_dependency'
            ]
            
            found_model_methods = []
            for method_name in model_methods:
                if hasattr(self.builder_class, method_name):
                    found_model_methods.append(method_name)
            
            # Check for dependency handling
            dependency_methods = [
                'add_dependency', 'set_dependencies', '_configure_dependencies'
            ]
            
            found_dependency_methods = []
            for method_name in dependency_methods:
                if hasattr(self.builder_class, method_name):
                    found_dependency_methods.append(method_name)
            
            model_integration_score = (
                len(found_model_attributes) + 
                len(found_model_methods) + 
                len(found_dependency_methods)
            )
            
            if model_integration_score == 0:
                return {
                    "passed": False,
                    "error": "No model integration configuration found",
                    "details": {
                        "expected_attributes": model_attributes,
                        "expected_methods": model_methods + dependency_methods,
                        "found_model_attributes": found_model_attributes,
                        "found_model_methods": found_model_methods,
                        "found_dependency_methods": found_dependency_methods,
                        "note": "Transform builders should integrate with trained models"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_model_attributes": found_model_attributes,
                    "found_model_methods": found_model_methods,
                    "found_dependency_methods": found_dependency_methods,
                    "integration_score": model_integration_score,
                    "validation": "Transform model integration specification verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing model integration specification: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level2_test_transform_input_specification_compliance(self) -> Dict[str, Any]:
        """
        Test that the builder follows transform input specification patterns.
        
        Transform builders should properly configure TransformInput with
        appropriate data sources, content types, and split strategies.
        """
        try:
            # Check for input configuration attributes
            input_attributes = [
                'input_data', 'data_source', 'content_type', 'split_type',
                'data_type', 'input_path', 'input_config'
            ]
            
            found_input_attributes = []
            for attr_name in input_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_input_attributes.append(attr_name)
            
            # Check for input preparation methods
            input_methods = [
                '_prepare_transform_input', '_get_transform_input', '_create_transform_input',
                '_configure_input_data', '_setup_input_config'
            ]
            
            found_input_methods = []
            for method_name in input_methods:
                if hasattr(self.builder_class, method_name):
                    found_input_methods.append(method_name)
            
            # Check for content type handling
            content_type_patterns = ['csv', 'json', 'parquet', 'text', 'application']
            content_type_handling = []
            
            # Look for content type references in class attributes or methods
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for pattern in content_type_patterns:
                            if pattern in attr_value.lower():
                                content_type_handling.append(f"{attr_name}: {pattern}")
            
            input_spec_score = len(found_input_attributes) + len(found_input_methods)
            
            if input_spec_score == 0:
                return {
                    "passed": False,
                    "error": "No transform input specification configuration found",
                    "details": {
                        "expected_attributes": input_attributes,
                        "expected_methods": input_methods,
                        "found_input_attributes": found_input_attributes,
                        "found_input_methods": found_input_methods,
                        "note": "Transform builders should configure input data specifications"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_input_attributes": found_input_attributes,
                    "found_input_methods": found_input_methods,
                    "content_type_handling": content_type_handling,
                    "input_spec_score": input_spec_score,
                    "validation": "Transform input specification compliance verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing transform input specification compliance: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level2_test_transform_output_specification_compliance(self) -> Dict[str, Any]:
        """
        Test that the builder follows transform output specification patterns.
        
        Transform builders should properly configure transform outputs with
        appropriate output paths, formats, and assembly options.
        """
        try:
            # Check for output configuration attributes
            output_attributes = [
                'output_path', 'output_config', 'accept_type', 'assemble_with',
                'output_format', 'prediction_output', 'result_path'
            ]
            
            found_output_attributes = []
            for attr_name in output_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_output_attributes.append(attr_name)
            
            # Check for output configuration methods
            output_methods = [
                '_configure_transform_output', '_setup_output_config', '_get_transform_output',
                '_prepare_output_configuration', '_setup_output_path'
            ]
            
            found_output_methods = []
            for method_name in output_methods:
                if hasattr(self.builder_class, method_name):
                    found_output_methods.append(method_name)
            
            # Check for output format handling
            output_formats = ['csv', 'json', 'parquet', 'text']
            output_format_handling = []
            
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for format_type in output_formats:
                            if format_type in attr_value.lower():
                                output_format_handling.append(f"{attr_name}: {format_type}")
            
            output_spec_score = len(found_output_attributes) + len(found_output_methods)
            
            if output_spec_score == 0:
                return {
                    "passed": False,
                    "error": "No transform output specification configuration found",
                    "details": {
                        "expected_attributes": output_attributes,
                        "expected_methods": output_methods,
                        "found_output_attributes": found_output_attributes,
                        "found_output_methods": found_output_methods,
                        "note": "Transform builders should configure output specifications"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_output_attributes": found_output_attributes,
                    "found_output_methods": found_output_methods,
                    "output_format_handling": output_format_handling,
                    "output_spec_score": output_spec_score,
                    "validation": "Transform output specification compliance verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing transform output specification compliance: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level2_test_framework_specific_specifications(self) -> Dict[str, Any]:
        """
        Test that the builder follows framework-specific specification patterns.
        
        Different ML frameworks may have specific requirements for transform
        configurations, inference patterns, and resource specifications.
        """
        try:
            framework_specs = {
                'xgboost': {
                    'attributes': ['dmatrix', 'xgb_model', 'booster'],
                    'methods': ['_create_xgb_transformer', '_configure_xgboost'],
                    'patterns': ['xgb', 'xgboost', 'dmatrix']
                },
                'pytorch': {
                    'attributes': ['torch_model', 'device', 'tensor'],
                    'methods': ['_create_pytorch_transformer', '_configure_pytorch'],
                    'patterns': ['torch', 'pytorch', 'tensor', 'cuda']
                },
                'sklearn': {
                    'attributes': ['sklearn_model', 'estimator', 'predictor'],
                    'methods': ['_create_sklearn_transformer', '_configure_sklearn'],
                    'patterns': ['sklearn', 'scikit', 'estimator']
                },
                'tensorflow': {
                    'attributes': ['tf_model', 'keras_model', 'session'],
                    'methods': ['_create_tf_transformer', '_configure_tensorflow'],
                    'patterns': ['tensorflow', 'tf', 'keras']
                }
            }
            
            detected_frameworks = []
            framework_compliance = {}
            
            class_name = self.builder_class.__name__.lower()
            class_methods = [method for method in dir(self.builder_class) if not method.startswith('__')]
            
            for framework, spec in framework_specs.items():
                # Check if framework is indicated in class name
                framework_in_name = any(pattern in class_name for pattern in spec['patterns'])
                
                # Check for framework-specific attributes
                found_attributes = []
                for attr in spec['attributes']:
                    if hasattr(self.builder_class, attr):
                        found_attributes.append(attr)
                
                # Check for framework-specific methods
                found_methods = []
                for method in spec['methods']:
                    if hasattr(self.builder_class, method):
                        found_methods.append(method)
                
                # Check for framework patterns in method names
                pattern_methods = []
                for method in class_methods:
                    if any(pattern in method.lower() for pattern in spec['patterns']):
                        pattern_methods.append(method)
                
                framework_score = (
                    (1 if framework_in_name else 0) +
                    len(found_attributes) +
                    len(found_methods) +
                    len(pattern_methods)
                )
                
                if framework_score > 0:
                    detected_frameworks.append(framework)
                    framework_compliance[framework] = {
                        "framework_in_name": framework_in_name,
                        "found_attributes": found_attributes,
                        "found_methods": found_methods,
                        "pattern_methods": pattern_methods,
                        "compliance_score": framework_score
                    }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "detected_frameworks": detected_frameworks,
                    "framework_compliance": framework_compliance,
                    "validation": "Framework-specific specification compliance checked",
                    "note": "Framework-specific specifications are optional but recommended for specialized transforms"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing framework-specific specifications: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level2_test_environment_variable_patterns(self) -> Dict[str, Any]:
        """
        Test that the builder follows transform-specific environment variable patterns.
        
        Transform builders should properly handle environment variables for
        batch processing, model loading, and inference configuration.
        """
        try:
            # Transform-specific environment variables
            transform_env_vars = [
                'SM_MODEL_DIR', 'SM_INPUT_DATA_CONFIG', 'SM_OUTPUT_DATA_DIR',
                'BATCH_SIZE', 'MAX_PAYLOAD', 'MAX_CONCURRENT_TRANSFORMS',
                'TRANSFORM_INSTANCE_TYPE', 'TRANSFORM_INSTANCE_COUNT'
            ]
            
            # Check for environment variable usage
            env_var_usage = []
            env_var_methods = []
            
            # Look for os.environ usage in methods
            for method_name in dir(self.builder_class):
                if not method_name.startswith('__') and callable(getattr(self.builder_class, method_name)):
                    method = getattr(self.builder_class, method_name)
                    # This is a basic check - in practice, you'd need to inspect method source
                    if hasattr(method, '__code__'):
                        # Check if method likely uses environment variables
                        if any(var.lower() in method_name.lower() for var in ['env', 'environ', 'config']):
                            env_var_methods.append(method_name)
            
            # Check for environment variable attributes
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for env_var in transform_env_vars:
                            if env_var in attr_value:
                                env_var_usage.append(f"{attr_name}: {env_var}")
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "expected_env_vars": transform_env_vars,
                    "env_var_usage": env_var_usage,
                    "env_var_methods": env_var_methods,
                    "validation": "Transform environment variable patterns checked",
                    "note": "Environment variable usage is recommended for flexible configuration"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing environment variable patterns: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level2_test_resource_specification_compliance(self) -> Dict[str, Any]:
        """
        Test that the builder follows resource specification patterns.
        
        Transform builders should properly configure compute resources
        including instance types, counts, and resource limits.
        """
        try:
            # Check for resource configuration attributes
            resource_attributes = [
                'instance_type', 'instance_count', 'max_concurrent_transforms',
                'max_payload', 'volume_size', 'volume_kms_key'
            ]
            
            found_resource_attributes = []
            for attr_name in resource_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_resource_attributes.append(attr_name)
            
            # Check for resource configuration methods
            resource_methods = [
                '_configure_resources', '_setup_instance_config', '_get_resource_config',
                '_configure_compute_resources', '_setup_transform_resources'
            ]
            
            found_resource_methods = []
            for method_name in resource_methods:
                if hasattr(self.builder_class, method_name):
                    found_resource_methods.append(method_name)
            
            # Check for instance type patterns
            instance_type_patterns = ['ml.', 'instance']
            instance_type_references = []
            
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for pattern in instance_type_patterns:
                            if pattern in attr_value:
                                instance_type_references.append(f"{attr_name}: {attr_value}")
            
            resource_spec_score = (
                len(found_resource_attributes) + 
                len(found_resource_methods) + 
                len(instance_type_references)
            )
            
            if resource_spec_score == 0:
                return {
                    "passed": False,
                    "error": "No resource specification configuration found",
                    "details": {
                        "expected_attributes": resource_attributes,
                        "expected_methods": resource_methods,
                        "found_resource_attributes": found_resource_attributes,
                        "found_resource_methods": found_resource_methods,
                        "note": "Transform builders should configure compute resources"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_resource_attributes": found_resource_attributes,
                    "found_resource_methods": found_resource_methods,
                    "instance_type_references": instance_type_references,
                    "resource_spec_score": resource_spec_score,
                    "validation": "Transform resource specification compliance verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing resource specification compliance: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all Transform-specific Level 2 specification tests.
        
        Returns:
            Dictionary mapping test names to their results
        """
        tests = {
            # Base specification tests
            "test_specification_usage": self.test_specification_usage,
            "test_contract_alignment": self.test_contract_alignment,
            "test_environment_variable_handling": self.test_environment_variable_handling,
            
            # Transform-specific specification tests
            "level2_test_batch_processing_specification_compliance": self.level2_test_batch_processing_specification_compliance,
            "level2_test_model_integration_specification": self.level2_test_model_integration_specification,
            "level2_test_transform_input_specification_compliance": self.level2_test_transform_input_specification_compliance,
            "level2_test_transform_output_specification_compliance": self.level2_test_transform_output_specification_compliance,
            "level2_test_framework_specific_specifications": self.level2_test_framework_specific_specifications,
            "level2_test_environment_variable_patterns": self.level2_test_environment_variable_patterns,
            "level2_test_resource_specification_compliance": self.level2_test_resource_specification_compliance,
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


# Convenience function for quick Transform specification validation
def validate_transform_specification(builder_class: Type[StepBuilderBase], 
                                    verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for Transform step builder specifications.
    
    Args:
        builder_class: The Transform step builder class to validate
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing test results
    """
    tester = TransformSpecificationTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
