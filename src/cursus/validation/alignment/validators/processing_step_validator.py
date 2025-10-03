"""
Processing Step Builder Validator

This module provides validation for Processing step builders following the
priority hierarchy system with universal and Processing-specific rules.
"""

from typing import Dict, Any, List, Type
import logging

from .step_type_specific_validator import StepTypeSpecificValidator

logger = logging.getLogger(__name__)


class ProcessingStepBuilderValidator(StepTypeSpecificValidator):
    """
    Validator for Processing step builders following priority hierarchy.
    
    Processing steps require:
    - Universal methods: validate_configuration, _get_inputs, create_step
    - Processing-specific methods: _create_processor
    - Processing-specific _get_outputs: returns List[ProcessingOutput]
    """
    
    def _validate_step_type_specifics(self, step_name: str, builder_class: Type, step_type: str) -> List[Dict[str, Any]]:
        """
        Validate Processing-specific requirements.
        
        Args:
            step_name: Name of the step to validate
            builder_class: The builder class to validate
            step_type: The SageMaker step type (should be "Processing")
            
        Returns:
            List of Processing-specific validation issues
        """
        logger.debug(f"Validating Processing-specific requirements for {step_name}")
        
        issues = []
        
        # Validate _create_processor method (required for Processing steps)
        processor_issues = self._validate_create_processor_method(builder_class)
        issues.extend(processor_issues)
        
        # Validate _get_outputs method for Processing steps
        output_issues = self._validate_processing_outputs(builder_class)
        issues.extend(output_issues)
        
        # Validate ProcessingInput/ProcessingOutput handling
        input_output_issues = self._validate_processing_input_output_handling(builder_class)
        issues.extend(input_output_issues)
        
        # Validate optional _get_job_arguments override
        job_args_issues = self._validate_job_arguments_override(builder_class)
        issues.extend(job_args_issues)
        
        logger.debug(f"Processing-specific validation completed for {step_name}: {len(issues)} issues found")
        return issues
    
    def _validate_create_processor_method(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _create_processor method implementation.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _create_processor method
        """
        issues = []
        
        if not hasattr(builder_class, "_create_processor"):
            issues.append({
                "level": "ERROR",
                "message": "Missing required Processing method: _create_processor",
                "method_name": "_create_processor",
                "rule_type": "step_specific",
                "details": {
                    "purpose": "Create SageMaker Processor instance for Processing step",
                    "expected_signature": "_create_processor(self) -> Processor",
                    "return_type": "sagemaker.processing.Processor or subclass"
                }
            })
        else:
            # Method exists, validate signature if possible
            try:
                import inspect
                method = getattr(builder_class, "_create_processor")
                signature = inspect.signature(method)
                
                # Basic parameter validation - should only have 'self'
                params = list(signature.parameters.keys())
                expected_params = ["self"]
                
                if len(params) > len(expected_params):
                    issues.append({
                        "level": "WARNING",
                        "message": "_create_processor method should only take 'self' parameter",
                        "method_name": "_create_processor",
                        "rule_type": "step_specific",
                        "details": {
                            "actual_params": params,
                            "expected_params": expected_params,
                            "signature_check": "parameter_count_validation"
                        }
                    })
            
            except Exception as e:
                logger.debug(f"Could not validate _create_processor signature: {str(e)}")
        
        return issues
    
    def _validate_processing_outputs(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate _get_outputs method for Processing steps.
        
        Processing steps should return List[ProcessingOutput] from _get_outputs.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _get_outputs method
        """
        issues = []
        
        if hasattr(builder_class, "_get_outputs"):
            # Check if method is properly implemented for Processing
            try:
                import inspect
                method = getattr(builder_class, "_get_outputs")
                
                # Check if method is overridden (Processing steps typically override this)
                if not self._is_method_overridden(builder_class, "_get_outputs"):
                    issues.append({
                        "level": "WARNING",
                        "message": "Processing steps typically override _get_outputs to return List[ProcessingOutput]",
                        "method_name": "_get_outputs",
                        "rule_type": "step_specific",
                        "details": {
                            "expected_return_type": "List[ProcessingOutput]",
                            "purpose": "Define output destinations for processing results",
                            "usage": "Used directly by ProcessingStep constructor"
                        }
                    })
                
                # Additional validation could check return type annotation if available
                signature = inspect.signature(method)
                if signature.return_annotation != inspect.Signature.empty:
                    return_annotation = str(signature.return_annotation)
                    if "List" not in return_annotation and "ProcessingOutput" not in return_annotation:
                        issues.append({
                            "level": "INFO",
                            "message": "_get_outputs return type annotation may not match Processing requirements",
                            "method_name": "_get_outputs",
                            "rule_type": "step_specific",
                            "details": {
                                "actual_annotation": return_annotation,
                                "expected_annotation": "List[ProcessingOutput]"
                            }
                        })
            
            except Exception as e:
                logger.debug(f"Could not validate _get_outputs for Processing: {str(e)}")
        
        return issues
    
    def _validate_processing_input_output_handling(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate ProcessingInput/ProcessingOutput handling patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for input/output handling
        """
        issues = []
        
        # Check if class imports ProcessingInput/ProcessingOutput
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for ProcessingInput/ProcessingOutput imports or usage
            has_processing_input = "ProcessingInput" in source
            has_processing_output = "ProcessingOutput" in source
            
            if not has_processing_input:
                issues.append({
                    "level": "WARNING",
                    "message": "Processing step builder should use ProcessingInput for input handling",
                    "method_name": "class_imports",
                    "rule_type": "step_specific",
                    "details": {
                        "missing_import": "ProcessingInput",
                        "purpose": "Define input sources for processing job",
                        "typical_usage": "from sagemaker.processing import ProcessingInput"
                    }
                })
            
            if not has_processing_output:
                issues.append({
                    "level": "WARNING",
                    "message": "Processing step builder should use ProcessingOutput for output handling",
                    "method_name": "class_imports",
                    "rule_type": "step_specific",
                    "details": {
                        "missing_import": "ProcessingOutput",
                        "purpose": "Define output destinations for processing results",
                        "typical_usage": "from sagemaker.processing import ProcessingOutput"
                    }
                })
        
        except Exception as e:
            logger.debug(f"Could not validate ProcessingInput/Output usage: {str(e)}")
        
        return issues
    
    def _validate_job_arguments_override(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate optional _get_job_arguments override for Processing steps.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for _get_job_arguments override
        """
        issues = []
        
        if hasattr(builder_class, "_get_job_arguments"):
            # Check if method is overridden (optional for Processing steps)
            if self._is_method_overridden(builder_class, "_get_job_arguments"):
                try:
                    import inspect
                    method = getattr(builder_class, "_get_job_arguments")
                    signature = inspect.signature(method)
                    
                    # Validate signature for Processing-specific job arguments
                    params = list(signature.parameters.keys())
                    if "self" not in params:
                        issues.append({
                            "level": "WARNING",
                            "message": "_get_job_arguments override should include 'self' parameter",
                            "method_name": "_get_job_arguments",
                            "rule_type": "step_specific",
                            "details": {
                                "actual_params": params,
                                "expected_pattern": "self parameter required",
                                "purpose": "Provide command-line arguments for processing script"
                            }
                        })
                
                except Exception as e:
                    logger.debug(f"Could not validate _get_job_arguments override: {str(e)}")
        
        return issues
    
    def validate_processing_step_patterns(self, step_name: str) -> Dict[str, Any]:
        """
        Validate common Processing step implementation patterns.
        
        Args:
            step_name: Name of the step to validate
            
        Returns:
            Dictionary containing pattern validation results
        """
        logger.info(f"Validating Processing step patterns for {step_name}")
        
        try:
            builder_class = self._get_builder_class(step_name)
            if not builder_class:
                return {
                    "status": "ERROR",
                    "error": f"Could not find builder class for step: {step_name}"
                }
            
            pattern_issues = []
            
            # Check for common Processing patterns
            processor_creation_issues = self._validate_processor_creation_pattern(builder_class)
            pattern_issues.extend(processor_creation_issues)
            
            # Check for script execution patterns
            script_execution_issues = self._validate_script_execution_pattern(builder_class)
            pattern_issues.extend(script_execution_issues)
            
            return {
                "status": "COMPLETED",
                "step_name": step_name,
                "pattern_validation": "processing_specific",
                "issues": pattern_issues,
                "total_issues": len(pattern_issues)
            }
        
        except Exception as e:
            logger.error(f"Processing pattern validation failed for {step_name}: {str(e)}")
            return {
                "status": "ERROR",
                "error": str(e)
            }
    
    def _validate_processor_creation_pattern(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate processor creation patterns.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for processor creation patterns
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for common processor types
            processor_types = [
                "ScriptProcessor", "FrameworkProcessor", "PySparkProcessor", 
                "SparkJarProcessor", "Processor"
            ]
            
            found_processor_types = [ptype for ptype in processor_types if ptype in source]
            
            if not found_processor_types:
                issues.append({
                    "level": "INFO",
                    "message": "No common processor types found in Processing step builder",
                    "method_name": "processor_creation_pattern",
                    "rule_type": "step_specific",
                    "details": {
                        "common_types": processor_types,
                        "purpose": "Processing steps typically use specific processor types",
                        "recommendation": "Consider using ScriptProcessor or FrameworkProcessor"
                    }
                })
            else:
                logger.debug(f"Found processor types: {found_processor_types}")
        
        except Exception as e:
            logger.debug(f"Could not validate processor creation pattern: {str(e)}")
        
        return issues
    
    def _validate_script_execution_pattern(self, builder_class: Type) -> List[Dict[str, Any]]:
        """
        Validate script execution patterns for Processing steps.
        
        Args:
            builder_class: The builder class to validate
            
        Returns:
            List of validation issues for script execution patterns
        """
        issues = []
        
        try:
            import inspect
            source = inspect.getsource(builder_class)
            
            # Check for script-related patterns
            script_patterns = ["source_dir", "entry_point", "code", "command"]
            found_patterns = [pattern for pattern in script_patterns if pattern in source]
            
            if not found_patterns:
                issues.append({
                    "level": "INFO",
                    "message": "No script execution patterns found in Processing step builder",
                    "method_name": "script_execution_pattern",
                    "rule_type": "step_specific",
                    "details": {
                        "common_patterns": script_patterns,
                        "purpose": "Processing steps typically specify script execution details",
                        "recommendation": "Consider specifying source_dir and entry_point"
                    }
                })
            else:
                logger.debug(f"Found script patterns: {found_patterns}")
        
        except Exception as e:
            logger.debug(f"Could not validate script execution pattern: {str(e)}")
        
        return issues
