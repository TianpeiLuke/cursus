"""
CreateModel Step Builder Validator

This module provides validation for CreateModel step builders following the
priority hierarchy system with universal and CreateModel-specific rules.
"""

from typing import Dict, Any, List, Type
import logging

from .step_type_specific_validator import StepTypeSpecificValidator

logger = logging.getLogger(__name__)


class CreateModelStepBuilderValidator(StepTypeSpecificValidator):
    """
    Validator for CreateModel step builders following priority hierarchy.
    
    CreateModel steps require:
    - Universal methods: validate_configuration, _get_inputs, create_step
    - CreateModel-specific methods: _create_model
    - CreateModel-specific _get_outputs: returns None (SageMaker handles automatically)
    """
    
    def _validate_step_type_specifics(self, step_name: str, builder_class: Type, step_type: str) -> List[Dict[str, Any]]:
        """
        Validate CreateModel-specific requirements.
        
        Args:
            step_name: Name of the step to validate
            builder_class: The builder class to validate
            step_type: The SageMaker step type (should be "CreateModel")
            
        Returns:
            List of CreateModel-specific validation issues
        """
        logger.debug(f"Validating CreateModel-specific requirements for {step_name}")
        
        issues = []
        
        # Validate _create_model method (required for CreateModel steps)
        model_issues = self._validate_create_model_method(builder_class)
        issues.extend(model_issues)
        
        # Validate _get_outputs method for CreateModel steps
        output_issues = self._validate_createmodel_outputs(builder_class)
        issues.extend(output_issues)
        
        # Validate model artifact handling and configuration
        model_config_issues = self._validate_model_configuration(builder_class)
        issues.extend(model_config_issues)
        
        # Validate optional _get_image_uri method
        image_uri_issues = self._validate_image_uri_method(builder_class)
        issues.extend(image_uri_issues)
        
        logger.debug(f"CreateModel-specific validation completed for {step_name}: {len(issues)} issues found")
        return issues
    
    def _validate_create_model_method(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _create_model method implementation.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _create_model method
        """
        issues = []
        
        if not hasattr(builder_class, "_create_model"):
            issues.append({
                "level": "ERROR",
                "message": "Missing required CreateModel method: _create_model",
                "method_name": "_create_model",
                "rule_type": "step_specific",
                "details": {
                    "purpose": "Create SageMaker Model instance for CreateModel step",
                    "expected_signature": "_create_model(self, model_data: str) -> Model",
                    "return_type": "sagemaker.model.Model or subclass"
                }
            })
        else:
            # Method exists, validate signature if possible
            try:
                import inspect
                method = getattr(builder_class, "_create_model")
                signature = inspect.signature(method)
                
                # Basic parameter validation
                params = list(signature.parameters.keys())
                expected_params = ["self", "model_data"]
                
                if len(params) < len(expected_params):
                    issues.append({
                        "level": "WARNING",
                        "message": "_create_model method may have incorrect signature",
                        "method_name": "_create_model",
                        "rule_type": "step_specific",
                        "details": {
                            "actual_params": params,
                            "expected_params": expected_params,
                            "signature_check": "basic_parameter_count",
                            "usage": "model_data typically comes from training step output"
                        }
                    })
                
                # Check if model_data parameter exists
                if "model_data" not in params:
                    issues.append({
                        "level": "WARNING",
                        "message": "_create_model should accept model_data parameter",
                        "method_name": "_create_model",
                        "rule_type": "step_specific",
                        "details": {
                            "missing_param": "model_data",
                            "purpose": "Receive model artifacts from training step",
                            "typical_usage": "model_data from TrainingStep output"
                        }
                    })
            
            except Exception as e:
                logger.debug(f"Could not validate _create_model signature: {str(e)}")
        
        return issues
    
    def _validate_createmodel_outputs(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _get_outputs method for CreateModel steps.
        
        CreateModel steps should return None from _get_outputs (SageMaker handles automatically).
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _get_outputs method
        """
        issues = []
        
        if hasattr(builder_class, "_get_outputs"):
            # Check if method is properly implemented for CreateModel
            try:
                import inspect
                method = getattr(builder_class, "_get_outputs")
                
                # Check if method is overridden (CreateModel steps may override to return None)
                if self._is_method_overridden(builder_class, "_get_outputs"):
                    # Additional validation could check return type annotation if available
                    signature = inspect.signature(method)
                    if signature.return_annotation != inspect.Signature.empty:
                        return_annotation = str(signature.return_annotation)
                        if "None" not in return_annotation and "NoneType" not in return_annotation:
                            issues.append({
                                "level": "INFO",
                                "message": "_get_outputs return type annotation may not match CreateModel requirements",
                                "method_name": "_get_outputs",
                                "rule_type": "step_specific",
                                "details": {
                                    "actual_annotation": return_annotation,
                                    "expected_annotation": "None",
                                    "usage": "CreateModel steps don't produce explicit outputs - SageMaker handles model registration"
                                }
                            })
                else:
                    # Using inherited implementation is fine for CreateModel
                    logger.debug("CreateModel step using inherited _get_outputs implementation")
            
            except Exception as e:
                logger.debug(f"Could not validate _get_outputs for CreateModel: {str(e)}")
        
        return issues
    
    def _validate_model_configuration(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate CreateModel-specific configuration patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for model configuration
        """
        issues = []
        
        # Check if class handles model artifacts properly
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for model artifact patterns
            model_patterns = ["model_data", "image_uri", "container_def", "primary_container"]
            found_patterns = [pattern for pattern in model_patterns if pattern in source]
            
            if not found_patterns:
                issues.append({
                    "level": "WARNING",
                    "message": "No model artifact patterns found in CreateModel step builder",
                    "method_name": "model_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "missing_patterns": model_patterns,
                        "purpose": "CreateModel steps typically configure model artifacts and container definitions",
                        "recommendation": "Consider adding model_data or image_uri configuration"
                    }
                })
            
            # Check for role configuration
            has_role_config = "role" in source or "execution_role" in source
            if not has_role_config:
                issues.append({
                    "level": "WARNING",
                    "message": "No role configuration found in CreateModel step builder",
                    "method_name": "role_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "missing_config": "role or execution_role",
                        "purpose": "CreateModel steps require IAM role for model execution",
                        "typical_usage": "role=sagemaker.get_execution_role()"
                    }
                })
            
            # Check for environment variables configuration
            has_env_vars = "environment" in source or "env" in source
            if not has_env_vars:
                issues.append({
                    "level": "INFO",
                    "message": "No environment variables configuration found",
                    "method_name": "environment_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "missing_config": "environment variables",
                        "purpose": "Models often benefit from environment variable configuration",
                        "recommendation": "Consider adding environment variables if needed"
                    }
                })
        
        except Exception as e:
            logger.debug(f"Could not validate model configuration: {str(e)}")
        
        return issues
    
    def _validate_image_uri_method(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate optional _get_image_uri method for CreateModel steps.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _get_image_uri method
        """
        issues = []
        
        if hasattr(builder_class, "_get_image_uri"):
            # Check if method is overridden (optional for CreateModel steps)
            if self._is_method_overridden(builder_class, "_get_image_uri"):
                try:
                    import inspect
                    method = getattr(builder_class, "_get_image_uri")
                    signature = inspect.signature(method)
                    
                    # Validate signature for CreateModel-specific image URI generation
                    params = list(signature.parameters.keys())
                    if "self" not in params:
                        issues.append({
                            "level": "WARNING",
                            "message": "_get_image_uri override should include 'self' parameter",
                            "method_name": "_get_image_uri",
                            "rule_type": "step_specific",
                            "details": {
                                "actual_params": params,
                                "expected_pattern": "self parameter required",
                                "purpose": "Generate container image URI for model"
                            }
                        })
                    
                    # Check return type annotation if available
                    if signature.return_annotation != inspect.Signature.empty:
                        return_annotation = str(signature.return_annotation)
                        if "str" not in return_annotation:
                            issues.append({
                                "level": "INFO",
                                "message": "_get_image_uri should return string URI",
                                "method_name": "_get_image_uri",
                                "rule_type": "step_specific",
                                "details": {
                                    "actual_annotation": return_annotation,
                                    "expected_annotation": "str",
                                    "purpose": "Return container image URI for model deployment"
                                }
                            })
                
                except Exception as e:
                    logger.debug(f"Could not validate _get_image_uri override: {str(e)}")
        
        return issues
    
    def validate_createmodel_step_patterns(self, step_name: str) -> Dict[str, Any]:
        """
        Validate common CreateModel step implementation patterns.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing pattern validation results
        """
        logger.info(f"Validating CreateModel step patterns for {step_name}")
        
        try:
            builder_class = self._get_builder_class(step_name)
            if not builder_class:
                return {
                    "status": "ERROR",
                    "error": f"Could not find builder class for step: {step_name}"
                }
            
            pattern_issues = []
            
            # Check for common CreateModel patterns
            model_creation_issues = self._validate_model_creation_pattern(builder_class)
            pattern_issues.extend(model_creation_issues)
            
            # Check for container configuration patterns
            container_config_issues = self._validate_container_configuration(builder_class)
            pattern_issues.extend(container_config_issues)
            
            return {
                "status": "COMPLETED",
                "step_name": step_name,
                "pattern_validation": "createmodel_specific",
                "issues": pattern_issues,
                "total_issues": len(pattern_issues)
            }
        
        except Exception as e:
            logger.error(f"CreateModel pattern validation failed for {step_name}: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _validate_model_creation_pattern(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate model creation patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for model creation patterns
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for common model types
            model_types = [
                "XGBoostModel", "PyTorchModel", "TensorFlowModel", "SKLearnModel", 
                "Model", "FrameworkModel", "HuggingFaceModel"
            ]
            
            found_model_types = [mtype for mtype in model_types if mtype in source]
            
            if not found_model_types:
                issues.append({
                    "level": "INFO",
                    "message": "No common model types found in CreateModel step builder",
                    "method_name": "model_creation_pattern",
                    "rule_type": "step_specific",
                    "details": {
                        "common_types": model_types,
                        "purpose": "CreateModel steps typically use specific model types",
                        "recommendation": "Consider using framework-specific model classes"
                    }
                })
            else:
                logger.debug(f"Found model types: {found_model_types}")
                
                # Additional validation for specific model types
                if "XGBoostModel" in found_model_types:
                    xgboost_issues = self._validate_xgboost_model_patterns(source)
                    issues.extend(xgboost_issues)
                
                if "PyTorchModel" in found_model_types:
                    pytorch_issues = self._validate_pytorch_model_patterns(source)
                    issues.extend(pytorch_issues)
        
        except Exception as e:
            logger.debug(f"Could not validate model creation pattern: {str(e)}")
        
        return issues
    
    def _validate_container_configuration(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate container configuration patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for container configuration
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for container configuration patterns
            container_patterns = ["image_uri", "model_data", "environment", "container_def"]
            found_patterns = [pattern for pattern in container_patterns if pattern in source]
            
            if len(found_patterns) < 2:  # Expect at least image_uri and model_data
                issues.append({
                    "level": "INFO",
                    "message": "CreateModel step may be missing common container configuration",
                    "method_name": "container_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "found_patterns": found_patterns,
                        "expected_patterns": container_patterns,
                        "purpose": "CreateModel steps typically configure container image and model artifacts",
                        "recommendation": "Consider adding image_uri and model_data configuration"
                    }
                })
        
        except Exception as e:
            logger.debug(f"Could not validate container configuration: {str(e)}")
        
        return issues
    
    def _validate_xgboost_model_patterns(self, source: str) -> List[Dict[str, Any]]:
        """
        Validate XGBoost model-specific patterns.
        
        Args:
            source: Source code of the builder class
            
        Returns:
            List of validation issues for XGBoost model patterns
        """
        issues = []
        
        # Check for XGBoost model-specific patterns
        xgboost_patterns = ["framework_version", "py_version", "model_data"]
        found_patterns = [pattern for pattern in xgboost_patterns if pattern in source]
        
        if len(found_patterns) < 2:  # Expect at least framework_version and model_data
            issues.append({
                "level": "INFO",
                "message": "XGBoost model may be missing common configuration patterns",
                "method_name": "xgboost_model_configuration",
                "rule_type": "step_specific",
                "details": {
                    "found_patterns": found_patterns,
                    "expected_patterns": xgboost_patterns,
                    "recommendation": "XGBoost models typically specify framework_version and model_data"
                }
            })
        
        return issues
    
    def _validate_pytorch_model_patterns(self, source: str) -> List[Dict[str, Any]]:
        """
        Validate PyTorch model-specific patterns.
        
        Args:
            source: Source code of the builder class
            
        Returns:
            List of validation issues for PyTorch model patterns
        """
        issues = []
        
        # Check for PyTorch model-specific patterns
        pytorch_patterns = ["framework_version", "py_version", "model_data", "source_dir", "entry_point"]
        found_patterns = [pattern for pattern in pytorch_patterns if pattern in source]
        
        if len(found_patterns) < 3:  # Expect framework_version, py_version, and model_data
            issues.append({
                "level": "INFO",
                "message": "PyTorch model may be missing common configuration patterns",
                "method_name": "pytorch_model_configuration",
                "rule_type": "step_specific",
                "details": {
                    "found_patterns": found_patterns,
                    "expected_patterns": pytorch_patterns,
                    "recommendation": "PyTorch models typically specify framework_version, py_version, and model_data"
                }
            })
        
        return issues
