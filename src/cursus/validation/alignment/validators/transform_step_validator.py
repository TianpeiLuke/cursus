"""
Transform Step Builder Validator

This module provides validation for Transform step builders following the
priority hierarchy system with universal and Transform-specific rules.
"""

from typing import Dict, Any, List, Type
import logging

from .step_type_specific_validator import StepTypeSpecificValidator

logger = logging.getLogger(__name__)


class TransformStepBuilderValidator(StepTypeSpecificValidator):
    """
    Validator for Transform step builders following priority hierarchy.
    
    Transform steps require:
    - Universal methods: validate_configuration, _get_inputs, create_step
    - Transform-specific methods: _create_transformer
    - Transform-specific _get_outputs: returns str (used by _create_transformer)
    """
    
    def _validate_step_type_specifics(self, step_name: str, builder_class: Type, step_type: str) -> List[Dict[str, Any]]:
        """
        Validate Transform-specific requirements.
        
        Args:
            step_name: Name of the step to validate
            builder_class: The builder class to validate
            step_type: The SageMaker step type (should be "Transform")
            
        Returns:
            List of Transform-specific validation issues
        """
        logger.debug(f"Validating Transform-specific requirements for {step_name}")
        
        issues = []
        
        # Validate _create_transformer method (required for Transform steps)
        transformer_issues = self._validate_create_transformer_method(builder_class)
        issues.extend(transformer_issues)
        
        # Validate _get_outputs method for Transform steps
        output_issues = self._validate_transform_outputs(builder_class)
        issues.extend(output_issues)
        
        # Validate TransformInput handling and configuration
        transform_config_issues = self._validate_transform_configuration(builder_class)
        issues.extend(transform_config_issues)
        
        # Validate transformer type patterns
        transformer_type_issues = self._validate_transformer_type_patterns(builder_class)
        issues.extend(transformer_type_issues)
        
        logger.debug(f"Transform-specific validation completed for {step_name}: {len(issues)} issues found")
        return issues
    
    def _validate_create_transformer_method(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _create_transformer method implementation.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _create_transformer method
        """
        issues = []
        
        if not hasattr(builder_class, "_create_transformer"):
            issues.append({
                "level": "ERROR",
                "message": "Missing required Transform method: _create_transformer",
                "method_name": "_create_transformer",
                "rule_type": "step_specific",
                "details": {
                    "purpose": "Create SageMaker Transformer instance for Transform step",
                    "expected_signature": "_create_transformer(self, model_name, output_path=None) -> Transformer",
                    "return_type": "sagemaker.transformer.Transformer"
                }
            })
        else:
            # Method exists, validate signature if possible
            try:
                import inspect
                method = getattr(builder_class, "_create_transformer")
                signature = inspect.signature(method)
                
                # Basic parameter validation
                params = list(signature.parameters.keys())
                expected_params = ["self", "model_name"]  # output_path is optional
                
                if len(params) < len(expected_params):
                    issues.append({
                        "level": "WARNING",
                        "message": "_create_transformer method may have incorrect signature",
                        "method_name": "_create_transformer",
                        "rule_type": "step_specific",
                        "details": {
                            "actual_params": params,
                            "expected_params": expected_params + ["output_path (optional)"],
                            "signature_check": "basic_parameter_count",
                            "usage": "model_name is required, output_path is optional"
                        }
                    })
                
                # Check if model_name parameter exists
                if "model_name" not in params:
                    issues.append({
                        "level": "WARNING",
                        "message": "_create_transformer should accept model_name parameter",
                        "method_name": "_create_transformer",
                        "rule_type": "step_specific",
                        "details": {
                            "missing_param": "model_name",
                            "purpose": "Specify which model to use for transformation",
                            "typical_usage": "model_name from CreateModel step output"
                        }
                    })
            
            except Exception as e:
                logger.debug(f"Could not validate _create_transformer signature: {str(e)}")
        
        return issues
    
    def _validate_transform_outputs(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _get_outputs method for Transform steps.
        
        Transform steps should return str from _get_outputs (used by _create_transformer).
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _get_outputs method
        """
        issues = []
        
        if hasattr(builder_class, "_get_outputs"):
            # Check if method is properly implemented for Transform
            try:
                import inspect
                method = getattr(builder_class, "_get_outputs")
                
                # Check if method is overridden (Transform steps typically override this)
                if not self._is_method_overridden(builder_class, "_get_outputs"):
                    issues.append({
                        "level": "WARNING",
                        "message": "Transform steps typically override _get_outputs to return str",
                        "method_name": "_get_outputs",
                        "rule_type": "step_specific",
                        "details": {
                            "expected_return_type": "str",
                            "purpose": "Define output path for transform results",
                            "usage": "Passed to _create_transformer(output_path=...)"
                        }
                    })
                
                # Additional validation could check return type annotation if available
                signature = inspect.signature(method)
                if signature.return_annotation != inspect.Signature.empty:
                    return_annotation = str(signature.return_annotation)
                    if "str" not in return_annotation and "String" not in return_annotation:
                        issues.append({
                            "level": "INFO",
                            "message": "_get_outputs return type annotation may not match Transform requirements",
                            "method_name": "_get_outputs",
                            "rule_type": "step_specific",
                            "details": {
                                "actual_annotation": return_annotation,
                                "expected_annotation": "str",
                                "usage": "Transform steps use string output path"
                            }
                        })
            
            except Exception as e:
                logger.debug(f"Could not validate _get_outputs for Transform: {str(e)}")
        
        return issues
    
    def _validate_transform_configuration(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate Transform-specific configuration patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for transform configuration
        """
        issues = []
        
        # Check if class imports TransformInput
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for TransformInput usage
            has_transform_input = "TransformInput" in source
            
            if not has_transform_input:
                issues.append({
                    "level": "WARNING",
                    "message": "Transform step builder should use TransformInput for input handling",
                    "method_name": "class_imports",
                    "rule_type": "step_specific",
                    "details": {
                        "missing_import": "TransformInput",
                        "purpose": "Define input sources for transform job",
                        "typical_usage": "from sagemaker.inputs import TransformInput"
                    }
                })
            
            # Check for batch transform configuration patterns
            batch_patterns = ["instance_type", "instance_count", "max_concurrent_transforms", "max_payload"]
            found_batch_patterns = [pattern for pattern in batch_patterns if pattern in source]
            
            if not found_batch_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No batch transform configuration patterns found",
                    "method_name": "batch_transform_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": batch_patterns,
                        "purpose": "Transform steps typically configure batch transform parameters",
                        "recommendation": "Consider adding instance_type and instance_count configuration"
                    }
                })
            
            # Check for data format configuration
            data_format_patterns = ["content_type", "split_type", "compression_type", "accept"]
            found_format_patterns = [pattern for pattern in data_format_patterns if pattern in source]
            
            if not found_format_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No data format configuration patterns found",
                    "method_name": "data_format_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": data_format_patterns,
                        "purpose": "Transform steps often need to specify data format parameters",
                        "recommendation": "Consider adding content_type or accept configuration"
                    }
                })
        
        except Exception as e:
            logger.debug(f"Could not validate transform configuration: {str(e)}")
        
        return issues
    
    def _validate_transformer_type_patterns(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate transformer type patterns for Transform steps.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for transformer type patterns
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for common transformer creation patterns
            transformer_patterns = [
                "Transformer", "transformer", "model.transformer", "create_transformer"
            ]
            
            found_transformer_patterns = [pattern for pattern in transformer_patterns if pattern in source]
            
            if not found_transformer_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No transformer creation patterns found in Transform step builder",
                    "method_name": "transformer_creation_pattern",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": transformer_patterns,
                        "purpose": "Transform steps typically create Transformer instances",
                        "recommendation": "Consider using model.transformer() or Transformer class"
                    }
                })
            else:
                logger.debug(f"Found transformer patterns: {found_transformer_patterns}")
                
                # Additional validation for model-based transformers
                if "model.transformer" in found_transformer_patterns:
                    model_transformer_issues = self._validate_model_transformer_patterns(source)
                    issues.extend(model_transformer_issues)
        
        except Exception as e:
            logger.debug(f"Could not validate transformer type patterns: {str(e)}")
        
        return issues
    
    def _validate_model_transformer_patterns(self, source: str) -> List[Dict[str, Any]]:
        """
        Validate model-based transformer patterns.
        
        Args:
            source: Source code of the builder class
            
        Returns:
            List of validation issues for model-based transformer patterns
        """
        issues = []
        
        # Check for model-based transformer configuration
        model_transformer_patterns = ["instance_type", "instance_count", "output_path"]
        found_patterns = [pattern for pattern in model_transformer_patterns if pattern in source]
        
        if len(found_patterns) < 2:  # Expect at least instance_type and output_path
            issues.append({
                "level": "INFO",
                "message": "Model-based transformer may be missing common configuration",
                "method_name": "model_transformer_configuration",
                "rule_type": "step_specific",
                "details": {
                    "found_patterns": found_patterns,
                    "expected_patterns": model_transformer_patterns,
                    "recommendation": "Model transformers typically specify instance_type and output_path"
                }
            })
        
        return issues
    
    def validate_transform_step_patterns(self, step_name: str) -> Dict[str, Any]:
        """
        Validate common Transform step implementation patterns.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing pattern validation results
        """
        logger.info(f"Validating Transform step patterns for {step_name}")
        
        try:
            builder_class = self._get_builder_class(step_name)
            if not builder_class:
                return {
                    "status": "ERROR",
                    "error": f"Could not find builder class for step: {step_name}"
                }
            
            pattern_issues = []
            
            # Check for common Transform patterns
            transformer_creation_issues = self._validate_transformer_creation_pattern(builder_class)
            pattern_issues.extend(transformer_creation_issues)
            
            # Check for batch transform job configuration patterns
            batch_config_issues = self._validate_batch_transform_configuration(builder_class)
            pattern_issues.extend(batch_config_issues)
            
            return {
                "status": "COMPLETED",
                "step_name": step_name,
                "pattern_validation": "transform_specific",
                "issues": pattern_issues,
                "total_issues": len(pattern_issues)
            }
        
        except Exception as e:
            logger.error(f"Transform pattern validation failed for {step_name}: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _validate_transformer_creation_pattern(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate transformer creation patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for transformer creation patterns
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for transformer instantiation patterns
            instantiation_patterns = ["instance_type=", "instance_count=", "output_path="]
            found_patterns = [pattern for pattern in instantiation_patterns if pattern in source]
            
            if len(found_patterns) < 2:
                issues.append({
                    "level": "INFO",
                    "message": "Transform step may be missing common transformer configuration",
                    "method_name": "transformer_creation_pattern",
                    "rule_type": "step_specific",
                    "details": {
                        "found_patterns": found_patterns,
                        "expected_patterns": instantiation_patterns,
                        "purpose": "Transform steps typically specify instance and output configuration",
                        "recommendation": "Consider adding instance_type and output_path configuration"
                    }
                })
        
        except Exception as e:
            logger.debug(f"Could not validate transformer creation pattern: {str(e)}")
        
        return issues
    
    def _validate_batch_transform_configuration(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate batch transform configuration patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for batch transform configuration
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for batch transform optimization patterns
            optimization_patterns = ["max_concurrent_transforms", "max_payload", "strategy"]
            found_patterns = [pattern for pattern in optimization_patterns if pattern in source]
            
            if not found_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No batch transform optimization patterns found",
                    "method_name": "batch_transform_optimization",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": optimization_patterns,
                        "purpose": "Batch transform jobs often benefit from optimization configuration",
                        "recommendation": "Consider adding max_concurrent_transforms or max_payload configuration"
                    }
                })
            else:
                logger.debug(f"Found batch transform optimization patterns: {found_patterns}")
            
            # Check for data handling patterns
            data_handling_patterns = ["assemble_with", "split_type", "compression_type"]
            found_data_patterns = [pattern for pattern in data_handling_patterns if pattern in source]
            
            if not found_data_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No data handling patterns found",
                    "method_name": "data_handling_configuration",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": data_handling_patterns,
                        "purpose": "Transform steps often need to specify data handling parameters",
                        "recommendation": "Consider adding assemble_with or split_type configuration"
                    }
                })
            else:
                logger.debug(f"Found data handling patterns: {found_data_patterns}")
        
        except Exception as e:
            logger.debug(f"Could not validate batch transform configuration: {str(e)}")
        
        return issues
