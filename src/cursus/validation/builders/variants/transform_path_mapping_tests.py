"""
Transform Step Path Mapping Tests (Level 3).

This module provides Level 3 path mapping validation tests specifically for Transform step builders.
These tests focus on Transform-specific path mapping validation including TransformInput creation,
output path configuration, and model artifact path handling for batch inference workflows.
"""

from typing import Dict, Any, List, Optional, Type
import re

from ..path_mapping_tests import PathMappingTests
from ....core.base.builder_base import StepBuilderBase


class TransformPathMappingTests(PathMappingTests):
    """
    Level 3 path mapping tests specifically for Transform step builders.
    
    Extends the base PathMappingTests with Transform-specific path mapping validation
    including transform input/output handling, model path integration, and batch processing paths.
    """
    
    def __init__(self, builder_class: Type[StepBuilderBase], **kwargs):
        """Initialize Transform path mapping tests."""
        super().__init__(builder_class, **kwargs)
        self.step_type = "Transform"
    
    def level3_test_transform_input_object_creation(self) -> Dict[str, Any]:
        """
        Test that the builder properly creates TransformInput objects.
        
        Transform builders should create TransformInput objects with proper
        data sources, content types, and split strategies for batch processing.
        """
        try:
            # Check for TransformInput creation methods
            transform_input_methods = [
                '_prepare_transform_input', '_get_transform_input', '_create_transform_input',
                '_configure_transform_input', '_setup_transform_input'
            ]
            
            found_methods = []
            method_signatures = {}
            
            for method_name in transform_input_methods:
                if hasattr(self.builder_class, method_name):
                    method = getattr(self.builder_class, method_name)
                    if callable(method):
                        found_methods.append(method_name)
                        import inspect
                        sig = inspect.signature(method)
                        method_signatures[method_name] = {
                            "signature": str(sig),
                            "parameters": list(sig.parameters.keys())
                        }
            
            # Check for TransformInput-related attributes
            transform_input_attributes = [
                'input_data', 'data_source', 'content_type', 'split_type',
                'data_type', 'transform_input', 'input_config'
            ]
            
            found_attributes = []
            for attr_name in transform_input_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_attributes.append(attr_name)
            
            # Check for S3 path patterns in attributes
            s3_patterns = []
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str) and 's3://' in attr_value:
                        s3_patterns.append(f"{attr_name}: {attr_value}")
            
            if not found_methods and not found_attributes:
                return {
                    "passed": False,
                    "error": "No TransformInput creation methods or attributes found",
                    "details": {
                        "expected_methods": transform_input_methods,
                        "expected_attributes": transform_input_attributes,
                        "found_methods": found_methods,
                        "found_attributes": found_attributes,
                        "note": "Transform builders should create TransformInput objects"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_methods": found_methods,
                    "method_signatures": method_signatures,
                    "found_attributes": found_attributes,
                    "s3_patterns": s3_patterns,
                    "validation": "Transform input object creation capabilities verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing TransformInput object creation: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level3_test_transform_output_path_configuration(self) -> Dict[str, Any]:
        """
        Test that the builder properly configures transform output paths.
        
        Transform builders should configure output paths for batch inference
        results with proper S3 destinations and format specifications.
        """
        try:
            # Check for output path configuration attributes
            output_path_attributes = [
                'output_path', 'output_config', 'prediction_output', 'result_path',
                'transform_output', 'output_destination', 'output_s3_uri'
            ]
            
            found_output_attributes = []
            output_path_values = {}
            
            for attr_name in output_path_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_output_attributes.append(attr_name)
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if attr_value is not None:
                        output_path_values[attr_name] = str(attr_value)
            
            # Check for output configuration methods
            output_methods = [
                '_configure_transform_output', '_setup_output_config', '_get_transform_output',
                '_prepare_output_configuration', '_setup_output_path'
            ]
            
            found_output_methods = []
            for method_name in output_methods:
                if hasattr(self.builder_class, method_name):
                    found_output_methods.append(method_name)
            
            # Validate S3 path patterns in output configurations
            s3_output_patterns = []
            valid_s3_paths = []
            
            for attr_name, attr_value in output_path_values.items():
                if 's3://' in attr_value:
                    s3_output_patterns.append(f"{attr_name}: {attr_value}")
                    # Basic S3 path validation
                    if re.match(r's3://[a-z0-9.-]+/.*', attr_value):
                        valid_s3_paths.append(attr_value)
            
            # Check for output format specifications
            output_formats = ['csv', 'json', 'parquet', 'text']
            format_specifications = []
            
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for format_type in output_formats:
                            if format_type in attr_value.lower():
                                format_specifications.append(f"{attr_name}: {format_type}")
            
            output_config_score = (
                len(found_output_attributes) + 
                len(found_output_methods) + 
                len(s3_output_patterns)
            )
            
            if output_config_score == 0:
                return {
                    "passed": False,
                    "error": "No transform output path configuration found",
                    "details": {
                        "expected_attributes": output_path_attributes,
                        "expected_methods": output_methods,
                        "found_output_attributes": found_output_attributes,
                        "found_output_methods": found_output_methods,
                        "note": "Transform builders should configure output paths"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_output_attributes": found_output_attributes,
                    "found_output_methods": found_output_methods,
                    "output_path_values": output_path_values,
                    "s3_output_patterns": s3_output_patterns,
                    "valid_s3_paths": valid_s3_paths,
                    "format_specifications": format_specifications,
                    "output_config_score": output_config_score,
                    "validation": "Transform output path configuration verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing transform output path configuration: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level3_test_model_artifact_path_handling(self) -> Dict[str, Any]:
        """
        Test that the builder properly handles model artifact paths.
        
        Transform builders should properly reference model artifacts from
        training or model creation steps for batch inference.
        """
        try:
            # Check for model path attributes
            model_path_attributes = [
                'model_data', 'model_uri', 'model_artifact_path', 'model_s3_path',
                'model_source', 'trained_model_path', 'model_package_path'
            ]
            
            found_model_attributes = []
            model_path_values = {}
            
            for attr_name in model_path_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_model_attributes.append(attr_name)
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if attr_value is not None:
                        model_path_values[attr_name] = str(attr_value)
            
            # Check for model integration methods
            model_integration_methods = [
                'integrate_with_model_step', 'set_model_data', 'configure_model_source',
                '_setup_model_path', '_configure_model_artifact'
            ]
            
            found_model_methods = []
            for method_name in model_integration_methods:
                if hasattr(self.builder_class, method_name):
                    found_model_methods.append(method_name)
            
            # Validate model path patterns
            model_s3_patterns = []
            property_path_patterns = []
            
            for attr_name, attr_value in model_path_values.items():
                if 's3://' in attr_value:
                    model_s3_patterns.append(f"{attr_name}: {attr_value}")
                
                # Check for SageMaker property path patterns
                if '.properties.' in attr_value or 'ModelArtifacts' in attr_value:
                    property_path_patterns.append(f"{attr_name}: {attr_value}")
            
            # Check for model name references
            model_name_attributes = []
            for attr_name in dir(self.builder_class):
                if 'model_name' in attr_name.lower() and not attr_name.startswith('__'):
                    model_name_attributes.append(attr_name)
            
            model_path_score = (
                len(found_model_attributes) + 
                len(found_model_methods) + 
                len(model_s3_patterns) + 
                len(property_path_patterns)
            )
            
            if model_path_score == 0:
                return {
                    "passed": False,
                    "error": "No model artifact path handling found",
                    "details": {
                        "expected_attributes": model_path_attributes,
                        "expected_methods": model_integration_methods,
                        "found_model_attributes": found_model_attributes,
                        "found_model_methods": found_model_methods,
                        "note": "Transform builders should handle model artifact paths"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_model_attributes": found_model_attributes,
                    "found_model_methods": found_model_methods,
                    "model_path_values": model_path_values,
                    "model_s3_patterns": model_s3_patterns,
                    "property_path_patterns": property_path_patterns,
                    "model_name_attributes": model_name_attributes,
                    "model_path_score": model_path_score,
                    "validation": "Model artifact path handling verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing model artifact path handling: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level3_test_batch_processing_path_patterns(self) -> Dict[str, Any]:
        """
        Test that the builder follows batch processing path patterns.
        
        Transform builders should properly handle batch processing paths
        including input data splitting, output assembly, and temporary paths.
        """
        try:
            # Check for batch processing path attributes
            batch_path_attributes = [
                'batch_input_path', 'batch_output_path', 'temp_path', 'working_dir',
                'split_input_path', 'assembled_output_path', 'batch_data_path'
            ]
            
            found_batch_attributes = []
            for attr_name in batch_path_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_batch_attributes.append(attr_name)
            
            # Check for batch processing methods
            batch_methods = [
                '_configure_batch_paths', '_setup_batch_processing', '_get_batch_config',
                '_configure_split_strategy', '_setup_assembly_config'
            ]
            
            found_batch_methods = []
            for method_name in batch_methods:
                if hasattr(self.builder_class, method_name):
                    found_batch_methods.append(method_name)
            
            # Check for split and assembly configuration
            split_assembly_config = []
            
            # Look for split type and assembly configuration
            split_patterns = ['Line', 'RecordIO', 'TFRecord', 'None']
            assembly_patterns = ['Line', 'None']
            
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for pattern in split_patterns:
                            if pattern in attr_value:
                                split_assembly_config.append(f"{attr_name}: split={pattern}")
                        for pattern in assembly_patterns:
                            if pattern in attr_value and 'assemble' in attr_name.lower():
                                split_assembly_config.append(f"{attr_name}: assemble={pattern}")
            
            # Check for batch strategy configuration
            batch_strategies = ['MultiRecord', 'SingleRecord']
            batch_strategy_config = []
            
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for strategy in batch_strategies:
                            if strategy in attr_value:
                                batch_strategy_config.append(f"{attr_name}: {strategy}")
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_batch_attributes": found_batch_attributes,
                    "found_batch_methods": found_batch_methods,
                    "split_assembly_config": split_assembly_config,
                    "batch_strategy_config": batch_strategy_config,
                    "validation": "Batch processing path patterns checked",
                    "note": "Batch processing path configuration is recommended for optimal performance"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing batch processing path patterns: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level3_test_dependency_input_extraction(self) -> Dict[str, Any]:
        """
        Test that the builder properly extracts inputs from dependencies.
        
        Transform builders should extract input data paths from upstream
        processing steps and model artifacts from training/model steps.
        """
        try:
            # Check for dependency extraction methods
            dependency_methods = [
                'extract_inputs_from_dependencies', '_get_dependency_outputs',
                '_extract_model_from_dependencies', '_get_upstream_outputs'
            ]
            
            found_dependency_methods = []
            for method_name in dependency_methods:
                if hasattr(self.builder_class, method_name):
                    found_dependency_methods.append(method_name)
            
            # Check for dependency handling attributes
            dependency_attributes = [
                'dependencies', 'upstream_steps', 'model_step', 'data_step',
                'input_dependencies', 'model_dependencies'
            ]
            
            found_dependency_attributes = []
            for attr_name in dependency_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_dependency_attributes.append(attr_name)
            
            # Check for property path extraction patterns
            property_path_patterns = []
            
            # Look for SageMaker property path usage
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        if '.properties.' in attr_value:
                            property_path_patterns.append(f"{attr_name}: {attr_value}")
            
            # Check for step reference patterns
            step_reference_patterns = []
            step_reference_keywords = ['step', 'Step', 'properties', 'Properties']
            
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    if any(keyword in attr_name for keyword in step_reference_keywords):
                        step_reference_patterns.append(attr_name)
            
            dependency_score = (
                len(found_dependency_methods) + 
                len(found_dependency_attributes) + 
                len(property_path_patterns)
            )
            
            if dependency_score == 0:
                return {
                    "passed": False,
                    "error": "No dependency input extraction capabilities found",
                    "details": {
                        "expected_methods": dependency_methods,
                        "expected_attributes": dependency_attributes,
                        "found_dependency_methods": found_dependency_methods,
                        "found_dependency_attributes": found_dependency_attributes,
                        "note": "Transform builders should extract inputs from dependencies"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_dependency_methods": found_dependency_methods,
                    "found_dependency_attributes": found_dependency_attributes,
                    "property_path_patterns": property_path_patterns,
                    "step_reference_patterns": step_reference_patterns,
                    "dependency_score": dependency_score,
                    "validation": "Dependency input extraction capabilities verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing dependency input extraction: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level3_test_content_type_and_format_handling(self) -> Dict[str, Any]:
        """
        Test that the builder properly handles content types and formats.
        
        Transform builders should properly configure content types for input
        data and accept types for output data in batch processing.
        """
        try:
            # Check for content type attributes
            content_type_attributes = [
                'content_type', 'input_content_type', 'accept_type', 'output_format',
                'data_format', 'input_format', 'output_content_type'
            ]
            
            found_content_attributes = []
            content_type_values = {}
            
            for attr_name in content_type_attributes:
                if hasattr(self.builder_class, attr_name):
                    found_content_attributes.append(attr_name)
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if attr_value is not None:
                        content_type_values[attr_name] = str(attr_value)
            
            # Check for supported content types
            supported_content_types = [
                'text/csv', 'application/json', 'application/x-parquet',
                'text/plain', 'application/jsonlines'
            ]
            
            content_type_matches = []
            for attr_name, attr_value in content_type_values.items():
                for content_type in supported_content_types:
                    if content_type in attr_value:
                        content_type_matches.append(f"{attr_name}: {content_type}")
            
            # Check for format handling methods
            format_methods = [
                '_configure_content_type', '_setup_data_format', '_get_input_format',
                '_configure_output_format', '_setup_content_handling'
            ]
            
            found_format_methods = []
            for method_name in format_methods:
                if hasattr(self.builder_class, method_name):
                    found_format_methods.append(method_name)
            
            # Check for format validation patterns
            format_patterns = ['csv', 'json', 'parquet', 'jsonlines', 'text']
            format_references = []
            
            for attr_name in dir(self.builder_class):
                if not attr_name.startswith('__'):
                    attr_value = getattr(self.builder_class, attr_name, None)
                    if isinstance(attr_value, str):
                        for pattern in format_patterns:
                            if pattern in attr_value.lower():
                                format_references.append(f"{attr_name}: {pattern}")
            
            content_handling_score = (
                len(found_content_attributes) + 
                len(found_format_methods) + 
                len(content_type_matches)
            )
            
            if content_handling_score == 0:
                return {
                    "passed": False,
                    "error": "No content type and format handling found",
                    "details": {
                        "expected_attributes": content_type_attributes,
                        "expected_methods": format_methods,
                        "found_content_attributes": found_content_attributes,
                        "found_format_methods": found_format_methods,
                        "note": "Transform builders should handle content types and formats"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "found_content_attributes": found_content_attributes,
                    "found_format_methods": found_format_methods,
                    "content_type_values": content_type_values,
                    "content_type_matches": content_type_matches,
                    "format_references": format_references,
                    "content_handling_score": content_handling_score,
                    "validation": "Content type and format handling verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing content type and format handling: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all Transform-specific Level 3 path mapping tests.
        
        Returns:
            Dictionary mapping test names to their results
        """
        tests = {
            # Base path mapping tests
            "test_input_path_mapping": self.test_input_path_mapping,
            "test_output_path_mapping": self.test_output_path_mapping,
            "test_property_path_validity": self.test_property_path_validity,
            
            # Transform-specific path mapping tests
            "level3_test_transform_input_object_creation": self.level3_test_transform_input_object_creation,
            "level3_test_transform_output_path_configuration": self.level3_test_transform_output_path_configuration,
            "level3_test_model_artifact_path_handling": self.level3_test_model_artifact_path_handling,
            "level3_test_batch_processing_path_patterns": self.level3_test_batch_processing_path_patterns,
            "level3_test_dependency_input_extraction": self.level3_test_dependency_input_extraction,
            "level3_test_content_type_and_format_handling": self.level3_test_content_type_and_format_handling,
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


# Convenience function for quick Transform path mapping validation
def validate_transform_path_mapping(builder_class: Type[StepBuilderBase], 
                                   verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for Transform step builder path mapping.
    
    Args:
        builder_class: The Transform step builder class to validate
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing test results
    """
    tester = TransformPathMappingTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
