"""
Transform Step Integration Tests (Level 4).

This module provides Level 4 integration validation tests specifically for Transform step builders.
These tests focus on Transform-specific integration validation including complete TransformStep creation,
model integration workflows, and end-to-end batch processing validation.
"""

from typing import Dict, Any, List, Optional, Type
from unittest.mock import Mock, MagicMock

from ..integration_tests import IntegrationTests
from ....core.base.builder_base import StepBuilderBase


class TransformIntegrationTests(IntegrationTests):
    """
    Level 4 integration tests specifically for Transform step builders.
    
    Extends the base IntegrationTests with Transform-specific integration validation
    including complete step creation, model integration, and batch processing workflows.
    """
    
    def __init__(self, builder_class: Type[StepBuilderBase], **kwargs):
        """Initialize Transform integration tests."""
        super().__init__(builder_class, **kwargs)
        self.step_type = "Transform"
    
    def level4_test_complete_transform_step_creation(self) -> Dict[str, Any]:
        """
        Test that the builder can create a complete TransformStep.
        
        Transform builders should be able to create a complete TransformStep
        with proper transformer configuration, input/output handling, and dependencies.
        """
        try:
            # Create a mock builder instance
            builder_instance = self._create_mock_builder_instance()
            
            # Mock dependencies for transform step
            mock_dependencies = self._create_mock_transform_dependencies()
            
            # Set up builder with dependencies
            if hasattr(builder_instance, 'dependencies'):
                builder_instance.dependencies = mock_dependencies
            elif hasattr(builder_instance, 'set_dependencies'):
                builder_instance.set_dependencies(mock_dependencies)
            
            # Test step creation
            step_creation_result = self._test_step_creation(builder_instance)
            
            if not step_creation_result["success"]:
                return {
                    "passed": False,
                    "error": f"Transform step creation failed: {step_creation_result['error']}",
                    "details": {
                        "step_creation_result": step_creation_result,
                        "mock_dependencies": [str(dep) for dep in mock_dependencies],
                        "note": "Transform builders should create complete TransformStep instances"
                    }
                }
            
            # Validate the created step
            created_step = step_creation_result["step"]
            step_validation = self._validate_transform_step(created_step)
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "step_creation_result": step_creation_result,
                    "step_validation": step_validation,
                    "validation": "Complete Transform step creation verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing complete Transform step creation: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level4_test_model_integration_workflow(self) -> Dict[str, Any]:
        """
        Test that the builder properly integrates with model steps.
        
        Transform builders should integrate with training or model creation
        steps to access trained models for batch inference.
        """
        try:
            # Create a mock builder instance
            builder_instance = self._create_mock_builder_instance()
            
            # Create mock model step
            mock_model_step = self._create_mock_model_step()
            
            # Test model integration
            integration_methods = [
                'integrate_with_model_step', 'set_model_name', 'configure_model_source'
            ]
            
            integration_results = {}
            successful_integrations = 0
            
            for method_name in integration_methods:
                if hasattr(builder_instance, method_name):
                    try:
                        method = getattr(builder_instance, method_name)
                        if method_name == 'integrate_with_model_step':
                            result = method(mock_model_step)
                        elif method_name == 'set_model_name':
                            result = method("test-model")
                        elif method_name == 'configure_model_source':
                            result = method("s3://bucket/model.tar.gz")
                        else:
                            result = method()
                        
                        integration_results[method_name] = {
                            "success": True,
                            "result": str(result) if result is not None else "None"
                        }
                        successful_integrations += 1
                        
                    except Exception as e:
                        integration_results[method_name] = {
                            "success": False,
                            "error": str(e)
                        }
            
            # Check for model-related attributes after integration
            model_attributes = self._check_model_attributes(builder_instance)
            
            # Test dependency handling
            dependency_handling = self._test_dependency_handling(builder_instance, [mock_model_step])
            
            if successful_integrations == 0 and not model_attributes["found_attributes"]:
                return {
                    "passed": False,
                    "error": "No model integration capabilities found",
                    "details": {
                        "integration_methods": integration_methods,
                        "integration_results": integration_results,
                        "model_attributes": model_attributes,
                        "note": "Transform builders should integrate with model steps"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "integration_results": integration_results,
                    "successful_integrations": successful_integrations,
                    "model_attributes": model_attributes,
                    "dependency_handling": dependency_handling,
                    "validation": "Model integration workflow verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing model integration workflow: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level4_test_batch_processing_configuration_integration(self) -> Dict[str, Any]:
        """
        Test that the builder properly integrates batch processing configuration.
        
        Transform builders should configure batch processing parameters
        and integrate them into the TransformStep creation.
        """
        try:
            # Create a mock builder instance
            builder_instance = self._create_mock_builder_instance()
            
            # Test batch configuration methods
            batch_config_methods = [
                '_configure_batch_processing', '_setup_batch_config', '_get_batch_config'
            ]
            
            batch_config_results = {}
            for method_name in batch_config_methods:
                if hasattr(builder_instance, method_name):
                    try:
                        method = getattr(builder_instance, method_name)
                        result = method()
                        batch_config_results[method_name] = {
                            "success": True,
                            "result": str(result) if result is not None else "None"
                        }
                    except Exception as e:
                        batch_config_results[method_name] = {
                            "success": False,
                            "error": str(e)
                        }
            
            # Check batch processing attributes
            batch_attributes = [
                'batch_size', 'max_concurrent_transforms', 'max_payload',
                'batch_strategy', 'instance_count', 'instance_type'
            ]
            
            found_batch_attributes = {}
            for attr_name in batch_attributes:
                if hasattr(builder_instance, attr_name):
                    attr_value = getattr(builder_instance, attr_name)
                    found_batch_attributes[attr_name] = str(attr_value)
            
            # Test transformer creation with batch configuration
            transformer_creation = self._test_transformer_creation(builder_instance)
            
            # Test step creation with batch configuration
            step_creation = self._test_step_creation_with_batch_config(builder_instance)
            
            batch_integration_score = (
                len(batch_config_results) + 
                len(found_batch_attributes) + 
                (1 if transformer_creation["success"] else 0) +
                (1 if step_creation["success"] else 0)
            )
            
            if batch_integration_score == 0:
                return {
                    "passed": False,
                    "error": "No batch processing configuration integration found",
                    "details": {
                        "batch_config_methods": batch_config_methods,
                        "batch_config_results": batch_config_results,
                        "found_batch_attributes": found_batch_attributes,
                        "note": "Transform builders should integrate batch processing configuration"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "batch_config_results": batch_config_results,
                    "found_batch_attributes": found_batch_attributes,
                    "transformer_creation": transformer_creation,
                    "step_creation": step_creation,
                    "batch_integration_score": batch_integration_score,
                    "validation": "Batch processing configuration integration verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing batch processing configuration integration: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level4_test_framework_specific_transform_workflow(self) -> Dict[str, Any]:
        """
        Test framework-specific transform workflows.
        
        Different ML frameworks may have specific requirements for
        transform step creation and batch inference configuration.
        """
        try:
            # Detect framework from builder class
            framework = self._detect_framework()
            
            # Create a mock builder instance
            builder_instance = self._create_mock_builder_instance()
            
            # Framework-specific workflow tests
            framework_workflows = {
                'xgboost': self._test_xgboost_transform_workflow,
                'pytorch': self._test_pytorch_transform_workflow,
                'sklearn': self._test_sklearn_transform_workflow,
                'tensorflow': self._test_tensorflow_transform_workflow
            }
            
            workflow_results = {}
            
            if framework and framework in framework_workflows:
                # Test framework-specific workflow
                workflow_test = framework_workflows[framework]
                workflow_results[framework] = workflow_test(builder_instance)
            else:
                # Test generic workflow
                workflow_results['generic'] = self._test_generic_transform_workflow(builder_instance)
            
            # Test framework-specific configurations
            framework_config = self._test_framework_specific_config(builder_instance, framework)
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "detected_framework": framework,
                    "workflow_results": workflow_results,
                    "framework_config": framework_config,
                    "validation": "Framework-specific transform workflow tested",
                    "note": "Framework-specific workflows are optional but recommended"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing framework-specific transform workflow: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level4_test_input_output_integration_workflow(self) -> Dict[str, Any]:
        """
        Test that the builder properly integrates input/output workflows.
        
        Transform builders should properly handle input data from dependencies
        and configure output destinations for batch inference results.
        """
        try:
            # Create a mock builder instance
            builder_instance = self._create_mock_builder_instance()
            
            # Create mock dependencies with outputs
            mock_dependencies = self._create_mock_data_dependencies()
            
            # Test input extraction from dependencies
            input_extraction = self._test_input_extraction(builder_instance, mock_dependencies)
            
            # Test transform input creation
            transform_input_creation = self._test_transform_input_creation(builder_instance)
            
            # Test output configuration
            output_configuration = self._test_output_configuration(builder_instance)
            
            # Test complete input/output workflow
            io_workflow = self._test_complete_io_workflow(builder_instance, mock_dependencies)
            
            io_integration_score = (
                (1 if input_extraction["success"] else 0) +
                (1 if transform_input_creation["success"] else 0) +
                (1 if output_configuration["success"] else 0) +
                (1 if io_workflow["success"] else 0)
            )
            
            if io_integration_score == 0:
                return {
                    "passed": False,
                    "error": "No input/output integration workflow found",
                    "details": {
                        "input_extraction": input_extraction,
                        "transform_input_creation": transform_input_creation,
                        "output_configuration": output_configuration,
                        "io_workflow": io_workflow,
                        "note": "Transform builders should integrate input/output workflows"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "input_extraction": input_extraction,
                    "transform_input_creation": transform_input_creation,
                    "output_configuration": output_configuration,
                    "io_workflow": io_workflow,
                    "io_integration_score": io_integration_score,
                    "validation": "Input/output integration workflow verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing input/output integration workflow: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    def level4_test_end_to_end_transform_pipeline_integration(self) -> Dict[str, Any]:
        """
        Test end-to-end transform pipeline integration.
        
        Transform builders should integrate properly into complete ML pipelines
        with upstream data processing and downstream result processing.
        """
        try:
            # Create a mock builder instance
            builder_instance = self._create_mock_builder_instance()
            
            # Create complete pipeline dependencies
            pipeline_dependencies = self._create_mock_pipeline_dependencies()
            
            # Test pipeline integration
            pipeline_integration = self._test_pipeline_integration(builder_instance, pipeline_dependencies)
            
            # Test step name generation
            step_name_generation = self._test_step_name_generation(builder_instance)
            
            # Test dependency resolution
            dependency_resolution = self._test_dependency_resolution(builder_instance, pipeline_dependencies)
            
            # Test complete pipeline step creation
            pipeline_step_creation = self._test_pipeline_step_creation(builder_instance, pipeline_dependencies)
            
            # Test pipeline validation
            pipeline_validation = self._test_pipeline_validation(builder_instance, pipeline_dependencies)
            
            pipeline_score = (
                (1 if pipeline_integration["success"] else 0) +
                (1 if step_name_generation["success"] else 0) +
                (1 if dependency_resolution["success"] else 0) +
                (1 if pipeline_step_creation["success"] else 0) +
                (1 if pipeline_validation["success"] else 0)
            )
            
            if pipeline_score < 2:  # At least 2 components should work
                return {
                    "passed": False,
                    "error": "Insufficient end-to-end pipeline integration",
                    "details": {
                        "pipeline_integration": pipeline_integration,
                        "step_name_generation": step_name_generation,
                        "dependency_resolution": dependency_resolution,
                        "pipeline_step_creation": pipeline_step_creation,
                        "pipeline_validation": pipeline_validation,
                        "pipeline_score": pipeline_score,
                        "note": "Transform builders should integrate into complete ML pipelines"
                    }
                }
            
            return {
                "passed": True,
                "error": None,
                "details": {
                    "pipeline_integration": pipeline_integration,
                    "step_name_generation": step_name_generation,
                    "dependency_resolution": dependency_resolution,
                    "pipeline_step_creation": pipeline_step_creation,
                    "pipeline_validation": pipeline_validation,
                    "pipeline_score": pipeline_score,
                    "validation": "End-to-end transform pipeline integration verified"
                }
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": f"Error testing end-to-end transform pipeline integration: {str(e)}",
                "details": {"exception": str(e)}
            }
    
    # Helper methods for Transform-specific testing
    
    def _create_mock_transform_dependencies(self) -> List[Mock]:
        """Create mock dependencies for transform testing."""
        # Mock training step
        mock_training_step = Mock()
        mock_training_step.name = "training-step"
        mock_training_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        # Mock data processing step
        mock_data_step = Mock()
        mock_data_step.name = "data-processing-step"
        mock_data_step.properties.ProcessingOutputConfig.Outputs = {
            'inference_data': Mock(S3Output=Mock(S3Uri="s3://bucket/inference-data/"))
        }
        
        return [mock_training_step, mock_data_step]
    
    def _create_mock_model_step(self) -> Mock:
        """Create a mock model step for integration testing."""
        mock_model_step = Mock()
        mock_model_step.name = "model-step"
        mock_model_step.properties.ModelName = "test-model"
        mock_model_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        return mock_model_step
    
    def _create_mock_data_dependencies(self) -> List[Mock]:
        """Create mock data dependencies for input/output testing."""
        mock_preprocessing = Mock()
        mock_preprocessing.name = "preprocessing-step"
        mock_preprocessing.properties.ProcessingOutputConfig.Outputs = {
            'processed_data': Mock(S3Output=Mock(S3Uri="s3://bucket/processed-data/"))
        }
        return [mock_preprocessing]
    
    def _create_mock_pipeline_dependencies(self) -> List[Mock]:
        """Create complete mock pipeline dependencies."""
        dependencies = []
        dependencies.extend(self._create_mock_transform_dependencies())
        dependencies.extend(self._create_mock_data_dependencies())
        return dependencies
    
    def _validate_transform_step(self, step) -> Dict[str, Any]:
        """Validate a created TransformStep."""
        validation_result = {
            "is_transform_step": False,
            "has_transformer": False,
            "has_inputs": False,
            "step_name": None,
            "step_type": None
        }
        
        try:
            # Check if it's a TransformStep-like object
            if hasattr(step, 'name') and hasattr(step, 'transformer'):
                validation_result["is_transform_step"] = True
                validation_result["step_name"] = getattr(step, 'name', None)
                validation_result["step_type"] = type(step).__name__
            
            # Check for transformer
            if hasattr(step, 'transformer'):
                validation_result["has_transformer"] = True
            
            # Check for inputs
            if hasattr(step, 'inputs') or hasattr(step, 'input'):
                validation_result["has_inputs"] = True
                
        except Exception as e:
            validation_result["validation_error"] = str(e)
        
        return validation_result
    
    def _test_transformer_creation(self, builder_instance) -> Dict[str, Any]:
        """Test transformer creation."""
        transformer_methods = ['_create_transformer', '_get_transformer']
        
        for method_name in transformer_methods:
            if hasattr(builder_instance, method_name):
                try:
                    method = getattr(builder_instance, method_name)
                    result = method()
                    return {
                        "success": True,
                        "method": method_name,
                        "result": str(result) if result is not None else "None"
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "method": method_name,
                        "error": str(e)
                    }
        
        return {"success": False, "error": "No transformer creation methods found"}
    
    def _test_step_creation_with_batch_config(self, builder_instance) -> Dict[str, Any]:
        """Test step creation with batch configuration."""
        try:
            # Set some batch configuration attributes if they exist
            batch_attrs = {
                'batch_size': 1000,
                'max_concurrent_transforms': 1,
                'max_payload': 6,
                'batch_strategy': 'MultiRecord'
            }
            
            for attr_name, attr_value in batch_attrs.items():
                if hasattr(builder_instance, attr_name):
                    setattr(builder_instance, attr_name, attr_value)
            
            # Try to create step
            if hasattr(builder_instance, 'create_step'):
                step = builder_instance.create_step()
                return {
                    "success": True,
                    "step_created": True,
                    "step_type": type(step).__name__
                }
            
            return {"success": False, "error": "No create_step method"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_transform_input_creation(self, builder_instance) -> Dict[str, Any]:
        """Test transform input creation."""
        input_methods = [
            '_prepare_transform_input', '_get_transform_input', '_create_transform_input'
        ]
        
        for method_name in input_methods:
            if hasattr(builder_instance, method_name):
                try:
                    method = getattr(builder_instance, method_name)
                    result = method()
                    return {
                        "success": True,
                        "method": method_name,
                        "result": str(result) if result is not None else "None"
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "method": method_name,
                        "error": str(e)
                    }
        
        return {"success": False, "error": "No transform input creation methods found"}
    
    def _test_output_configuration(self, builder_instance) -> Dict[str, Any]:
        """Test output configuration."""
        output_methods = [
            '_configure_transform_output', '_setup_output_config', '_get_transform_output'
        ]
        
        for method_name in output_methods:
            if hasattr(builder_instance, method_name):
                try:
                    method = getattr(builder_instance, method_name)
                    result = method()
                    return {
                        "success": True,
                        "method": method_name,
                        "result": str(result) if result is not None else "None"
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "method": method_name,
                        "error": str(e)
                    }
        
        return {"success": False, "error": "No output configuration methods found"}
    
    def _test_complete_io_workflow(self, builder_instance, dependencies) -> Dict[str, Any]:
        """Test complete input/output workflow."""
        try:
            # Set dependencies
            if hasattr(builder_instance, 'dependencies'):
                builder_instance.dependencies = dependencies
            
            # Try to create step with I/O workflow
            if hasattr(builder_instance, 'create_step'):
                step = builder_instance.create_step()
                return {
                    "success": True,
                    "workflow_completed": True,
                    "step_type": type(step).__name__
                }
            
            return {"success": False, "error": "No create_step method for I/O workflow"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _detect_framework(self) -> Optional[str]:
        """Detect ML framework from builder class name."""
        class_name = self.builder_class.__name__.lower()
        
        if 'xgboost' in class_name or 'xgb' in class_name:
            return 'xgboost'
        elif 'pytorch' in class_name or 'torch' in class_name:
            return 'pytorch'
        elif 'sklearn' in class_name or 'scikit' in class_name:
            return 'sklearn'
        elif 'tensorflow' in class_name or 'tf' in class_name:
            return 'tensorflow'
        
        return None
    
    def _test_xgboost_transform_workflow(self, builder_instance) -> Dict[str, Any]:
        """Test XGBoost-specific transform workflow."""
        try:
            # Check for XGBoost-specific attributes/methods
            xgboost_indicators = ['xgb', 'xgboost', 'dmatrix', 'booster']
            found_indicators = []
            
            for indicator in xgboost_indicators:
                for attr_name in dir(builder_instance):
                    if indicator in attr_name.lower():
                        found_indicators.append(attr_name)
            
            return {
                "success": len(found_indicators) > 0,
                "found_indicators": found_indicators,
                "framework": "xgboost"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_pytorch_transform_workflow(self, builder_instance) -> Dict[str, Any]:
        """Test PyTorch-specific transform workflow."""
        try:
            # Check for PyTorch-specific attributes/methods
            pytorch_indicators = ['torch', 'pytorch', 'tensor', 'cuda', 'device']
            found_indicators = []
            
            for indicator in pytorch_indicators:
                for attr_name in dir(builder_instance):
                    if indicator in attr_name.lower():
                        found_indicators.append(attr_name)
            
            return {
                "success": len(found_indicators) > 0,
                "found_indicators": found_indicators,
                "framework": "pytorch"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_sklearn_transform_workflow(self, builder_instance) -> Dict[str, Any]:
        """Test SKLearn-specific transform workflow."""
        try:
            # Check for SKLearn-specific attributes/methods
            sklearn_indicators = ['sklearn', 'scikit', 'estimator', 'predictor']
            found_indicators = []
            
            for indicator in sklearn_indicators:
                for attr_name in dir(builder_instance):
                    if indicator in attr_name.lower():
                        found_indicators.append(attr_name)
            
            return {
                "success": len(found_indicators) > 0,
                "found_indicators": found_indicators,
                "framework": "sklearn"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_tensorflow_transform_workflow(self, builder_instance) -> Dict[str, Any]:
        """Test TensorFlow-specific transform workflow."""
        try:
            # Check for TensorFlow-specific attributes/methods
            tf_indicators = ['tensorflow', 'tf', 'keras', 'session']
            found_indicators = []
            
            for indicator in tf_indicators:
                for attr_name in dir(builder_instance):
                    if indicator in attr_name.lower():
                        found_indicators.append(attr_name)
            
            return {
                "success": len(found_indicators) > 0,
                "found_indicators": found_indicators,
                "framework": "tensorflow"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_generic_transform_workflow(self, builder_instance) -> Dict[str, Any]:
        """Test generic transform workflow."""
        try:
            # Check for generic transform capabilities
            generic_methods = ['create_step', '_create_transformer', '_prepare_transform_input']
            found_methods = []
            
            for method_name in generic_methods:
                if hasattr(builder_instance, method_name):
                    found_methods.append(method_name)
            
            return {
                "success": len(found_methods) > 0,
                "found_methods": found_methods,
                "framework": "generic"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_framework_specific_config(self, builder_instance, framework: Optional[str]) -> Dict[str, Any]:
        """Test framework-specific configuration."""
        try:
            config_attributes = []
            
            if framework:
                # Look for framework-specific configuration
                for attr_name in dir(builder_instance):
                    if framework in attr_name.lower() and not attr_name.startswith('__'):
                        config_attributes.append(attr_name)
            
            return {
                "framework": framework,
                "config_attributes": config_attributes,
                "has_framework_config": len(config_attributes) > 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all Transform-specific Level 4 integration tests.
        
        Returns:
            Dictionary mapping test names to their results
        """
        tests = {
            # Base integration tests
            "test_dependency_resolution": self.test_dependency_resolution,
            "test_step_creation": self.test_step_creation,
            "test_step_name": self.test_step_name,
            
            # Transform-specific integration tests
            "level4_test_complete_transform_step_creation": self.level4_test_complete_transform_step_creation,
            "level4_test_model_integration_workflow": self.level4_test_model_integration_workflow,
            "level4_test_batch_processing_configuration_integration": self.level4_test_batch_processing_configuration_integration,
            "level4_test_framework_specific_transform_workflow": self.level4_test_framework_specific_transform_workflow,
            "level4_test_input_output_integration_workflow": self.level4_test_input_output_integration_workflow,
            "level4_test_end_to_end_transform_pipeline_integration": self.level4_test_end_to_end_transform_pipeline_integration,
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


# Convenience function for quick Transform integration validation
def validate_transform_integration(builder_class: Type[StepBuilderBase], 
                                  verbose: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Quick validation function for Transform step builder integration.
    
    Args:
        builder_class: The Transform step builder class to validate
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary containing test results
    """
    tester = TransformIntegrationTests(builder_class, verbose=verbose)
    return tester.run_all_tests()
