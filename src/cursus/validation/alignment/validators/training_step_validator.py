"""
Training Step Builder Validator

This module provides validation for Training step builders following the
priority hierarchy system with universal and Training-specific rules.
"""

from typing import Dict, Any, List, Type
import logging

from .step_type_specific_validator import StepTypeSpecificValidator

logger = logging.getLogger(__name__)


class TrainingStepBuilderValidator(StepTypeSpecificValidator):
    """
    Validator for Training step builders following priority hierarchy.
    
    Training steps require:
    - Universal methods: validate_configuration, _get_inputs, create_step
    - Training-specific methods: _create_estimator
    - Training-specific _get_outputs: returns str (used by _create_estimator)
    """
    
    def _validate_step_type_specifics(self, step_name: str, builder_class: Type, step_type: str) -> List[Dict[str, Any]]:
        """
        Validate Training-specific requirements.
        
        Args:
            step_name: Name of the step to validate
            builder_class: The builder class to validate
            step_type: The SageMaker step type (should be "Training")
            
        Returns:
            List of Training-specific validation issues
        """
        logger.debug(f"Validating Training-specific requirements for {step_name}")
        
        issues = []
        
        # Validate _create_estimator method (required for Training steps)
        estimator_issues = self._validate_create_estimator_method(builder_class)
        issues.extend(estimator_issues)
        
        # Validate _get_outputs method for Training steps
        output_issues = self._validate_training_outputs(builder_class)
        issues.extend(output_issues)
        
        # Validate TrainingInput handling and hyperparameter configuration
        training_config_issues = self._validate_training_configuration(builder_class)
        issues.extend(training_config_issues)
        
        # Validate estimator type patterns
        estimator_type_issues = self._validate_estimator_type_patterns(builder_class)
        issues.extend(estimator_type_issues)
        
        logger.debug(f"Training-specific validation completed for {step_name}: {len(issues)} issues found")
        return issues
    
    def _validate_create_estimator_method(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _create_estimator method implementation.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _create_estimator method
        """
        issues = []
        
        if not hasattr(builder_class, "_create_estimator"):
            issues.append({
                "level": "ERROR",
                "message": "Missing required Training method: _create_estimator",
                "method_name": "_create_estimator",
                "rule_type": "step_specific",
                "details": {
                    "purpose": "Create SageMaker Estimator instance for Training step",
                    "expected_signature": "_create_estimator(self, output_path: str) -> Estimator",
                    "return_type": "sagemaker.estimator.Estimator or subclass"
                }
            })
        else:
            # Method exists, validate signature if possible
            try:
                import inspect
                method = getattr(builder_class, "_create_estimator")
                signature = inspect.signature(method)
                
                # Basic parameter validation
                params = list(signature.parameters.keys())
                expected_params = ["self", "output_path"]
                
                if len(params) < len(expected_params):
                    issues.append({
                        "level": "WARNING",
                        "message": "_create_estimator method may have incorrect signature",
                        "method_name": "_create_estimator",
                        "rule_type": "step_specific",
                        "details": {
                            "actual_params": params,
                            "expected_params": expected_params,
                            "signature_check": "basic_parameter_count",
                            "usage": "output_path typically comes from _get_outputs() result"
                        }
                    })
                
                # Check if output_path parameter exists
                if "output_path" not in params:
                    issues.append({
                        "level": "WARNING",
                        "message": "_create_estimator should accept output_path parameter",
                        "method_name": "_create_estimator",
                        "rule_type": "step_specific",
                        "details": {
                            "missing_param": "output_path",
                            "purpose": "Specify where training artifacts should be stored",
                            "typical_usage": "output_path = self._get_outputs()"
                        }
                    })
            
            except Exception as e:
                logger.debug(f"Could not validate _create_estimator signature: {str(e)}")
        
        return issues
    
    def _validate_training_outputs(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _get_outputs method for Training steps.
        
        Training steps should return str from _get_outputs (used by _create_estimator).
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _get_outputs method
        """
        issues = []
        
        if hasattr(builder_class, "_get_outputs"):
            # Check if method is properly implemented for Training
            try:
                import inspect
                method = getattr(builder_class, "_get_outputs")
                
                # Check if method is overridden (Training steps typically override this)
                if not self._is_method_overridden(builder_class, "_get_outputs"):
                    issues.append({
                        "level": "WARNING",
                        "message": "Training steps typically override _get_outputs to return str",
                        "method_name": "_get_outputs",
                        "rule_type": "step_specific",
                        "details": {
                            "expected_return_type": "str",
                            "purpose": "Define output path for training artifacts",
                            "usage": "Passed to _create_estimator(output_path=...)"
                        }
                    })
                
                # Additional validation could check return type annotation if available
                signature = inspect.signature(method)
                if signature.return_annotation != inspect.Signature.empty:
                    return_annotation = str(signature.return_annotation)
                    if "str" not in return_annotation and "String" not in return_annotation:
                        issues.append({
                            "level": "INFO",
                            "message": "_get_outputs return type annotation may not match Training requirements",
                            "method_name": "_get_outputs",
                            "rule_type": "step_specific",
                            "details": {
                                "actual_annotation": return_annotation,
                                "expected_annotation": "str",
                                "usage": "Training steps use string output path"
                            }
                        })
            
            except Exception as e:
                logger.debug(f"Could not validate _get_outputs for Training: {str(e)}")
        
        return issues
    
    def _validate_training_configuration(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate Training-specific configuration patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for training configuration
        """
        issues = []
        
        # Check if class imports TrainingInput
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for TrainingInput usage
            has_training_input = "TrainingInput" in source
            
            if not has_training_input:
                issues.append({
                    "level": "WARNING",
                    "message": "Training step builder should use TrainingInput for input handling",
                    "method_name": "class_imports",
                    "rule_type": "step_specific",
                    "details": {
                        "missing_import": "TrainingInput",
                        "purpose": "Define input sources for training job",
                        "typical_usage": "from sagemaker.inputs import TrainingInput"
                    }
                })
            
            # Check for hyperparameter configuration patterns
            hyperparameter_patterns = ["hyperparameters", "HyperParameters", "set_hyperparameters"]
            found_hyperparameter_patterns = [pattern for pattern in hyperparameter_patterns if pattern in source]
            
            if not found_hyperparameter_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No hyperparameter configuration patterns found",
                    "method_name": "hyperparameter_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": hyperparameter_patterns,
                        "purpose": "Training steps typically configure hyperparameters",
                        "recommendation": "Consider adding hyperparameter configuration"
                    }
                })
        
        except Exception as e:
            logger.debug(f"Could not validate training configuration: {str(e)}")
        
        return issues
    
    def _validate_estimator_type_patterns(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate estimator type patterns for Training steps.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for estimator type patterns
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for common estimator types
            estimator_types = [
                "XGBoost", "PyTorch", "TensorFlow", "SKLearn", "Estimator",
                "Framework", "Algorithm", "HuggingFace"
            ]
            
            found_estimator_types = [etype for etype in estimator_types if etype in source]
            
            if not found_estimator_types:
                issues.append({
                    "level": "INFO",
                    "message": "No common estimator types found in Training step builder",
                    "method_name": "estimator_type_pattern",
                    "rule_type": "step_specific",
                    "details": {
                        "common_types": estimator_types,
                        "purpose": "Training steps typically use specific estimator types",
                        "recommendation": "Consider using framework-specific estimators"
                    }
                })
            else:
                logger.debug(f"Found estimator types: {found_estimator_types}")
                
                # Additional validation for specific estimator types
                if "XGBoost" in found_estimator_types:
                    xgboost_issues = self._validate_xgboost_patterns(source)
                    issues.extend(xgboost_issues)
                
                if "PyTorch" in found_estimator_types:
                    pytorch_issues = self._validate_pytorch_patterns(source)
                    issues.extend(pytorch_issues)
        
        except Exception as e:
            logger.debug(f"Could not validate estimator type patterns: {str(e)}")
        
        return issues
    
    def _validate_xgboost_patterns(self, source: str) -> List[Dict[str, Any]]:
        """
        Validate XGBoost-specific patterns.
        
        Args:
            source: Source code of the builder class
            
        Returns:
            List of validation issues for XGBoost patterns
        """
        issues = []
        
        # Check for XGBoost-specific patterns
        xgboost_patterns = ["framework_version", "py_version", "entry_point"]
        found_patterns = [pattern for pattern in xgboost_patterns if pattern in source]
        
        if len(found_patterns) < 2:  # Expect at least framework_version and py_version
            issues.append({
                "level": "INFO",
                "message": "XGBoost estimator may be missing common configuration patterns",
                "method_name": "xgboost_configuration",
                "rule_type": "step_specific",
                "details": {
                    "found_patterns": found_patterns,
                    "expected_patterns": xgboost_patterns,
                    "recommendation": "XGBoost estimators typically specify framework_version and py_version"
                }
            })
        
        return issues
    
    def _validate_pytorch_patterns(self, source: str) -> List[Dict[str, Any]]:
        """
        Validate PyTorch-specific patterns.
        
        Args:
            source: Source code of the builder class
            
        Returns:
            List of validation issues for PyTorch patterns
        """
        issues = []
        
        # Check for PyTorch-specific patterns
        pytorch_patterns = ["framework_version", "py_version", "source_dir", "entry_point"]
        found_patterns = [pattern for pattern in pytorch_patterns if pattern in source]
        
        if len(found_patterns) < 3:  # Expect framework_version, py_version, and source_dir/entry_point
            issues.append({
                "level": "INFO",
                "message": "PyTorch estimator may be missing common configuration patterns",
                "method_name": "pytorch_configuration",
                "rule_type": "step_specific",
                "details": {
                    "found_patterns": found_patterns,
                    "expected_patterns": pytorch_patterns,
                    "recommendation": "PyTorch estimators typically specify framework_version, py_version, and source_dir"
                }
            })
        
        return issues
    
    def validate_training_step_patterns(self, step_name: str) -> Dict[str, Any]:
        """
        Validate common Training step implementation patterns.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing pattern validation results
        """
        logger.info(f"Validating Training step patterns for {step_name}")
        
        try:
            builder_class = self._get_builder_class(step_name)
            if not builder_class:
                return {
                    "status": "ERROR",
                    "error": f"Could not find builder class for step: {step_name}"
                }
            
            pattern_issues = []
            
            # Check for common Training patterns
            estimator_creation_issues = self._validate_estimator_creation_pattern(builder_class)
            pattern_issues.extend(estimator_creation_issues)
            
            # Check for training job configuration patterns
            job_config_issues = self._validate_training_job_configuration(builder_class)
            pattern_issues.extend(job_config_issues)
            
            return {
                "status": "COMPLETED",
                "step_name": step_name,
                "pattern_validation": "training_specific",
                "issues": pattern_issues,
                "total_issues": len(pattern_issues)
            }
        
        except Exception as e:
            logger.error(f"Training pattern validation failed for {step_name}: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _validate_estimator_creation_pattern(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate estimator creation patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for estimator creation patterns
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for estimator instantiation patterns
            instantiation_patterns = ["role=", "instance_type=", "instance_count="]
            found_patterns = [pattern for pattern in instantiation_patterns if pattern in source]
            
            if len(found_patterns) < 2:
                issues.append({
                    "level": "INFO",
                    "message": "Training step may be missing common estimator configuration",
                    "method_name": "estimator_creation_pattern",
                    "rule_type": "step_specific",
                    "details": {
                        "found_patterns": found_patterns,
                        "expected_patterns": instantiation_patterns,
                        "purpose": "Training estimators typically specify role and instance configuration",
                        "recommendation": "Consider adding role and instance_type configuration"
                    }
                })
        
        except Exception as e:
            logger.debug(f"Could not validate estimator creation pattern: {str(e)}")
        
        return issues
    
    def _validate_training_job_configuration(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate training job configuration patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for training job configuration
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for training job configuration patterns
            job_config_patterns = ["max_run", "use_spot_instances", "checkpoint_s3_uri", "volume_size"]
            found_patterns = [pattern for pattern in job_config_patterns if pattern in source]
            
            if not found_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No training job configuration patterns found",
                    "method_name": "training_job_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": job_config_patterns,
                        "purpose": "Training jobs often benefit from configuration optimization",
                        "recommendation": "Consider adding max_run or spot instance configuration"
                    }
                })
            else:
                logger.debug(f"Found training job config patterns: {found_patterns}")
        
        except Exception as e:
            logger.debug(f"Could not validate training job configuration: {str(e)}")
        
        return issues
