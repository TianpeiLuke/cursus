"""
Step Type Enhancement Router

Central routing system that directs validation enhancement to appropriate step type enhancers.
Provides step type-specific validation requirements and coordinates enhancement across all SageMaker step types.
"""

from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from .alignment_utils import detect_step_type_from_registry


class StepTypeEnhancementRouter:
    """
    Routes validation enhancement to appropriate step type enhancer.
    
    Central coordinator for step type-aware validation that:
    - Detects step type from script name
    - Routes to appropriate step type enhancer
    - Provides step type requirements and patterns
    - Manages enhancer lifecycle and caching
    """
    
    def __init__(self):
        """Initialize the router with all step type enhancers."""
        # Lazy loading to avoid circular imports
        self._enhancers = {}
        self._enhancer_classes = {
            "Processing": "ProcessingStepEnhancer",
            "Training": "TrainingStepEnhancer", 
            "CreateModel": "CreateModelStepEnhancer",
            "Transform": "TransformStepEnhancer",
            "RegisterModel": "RegisterModelStepEnhancer",
            "Utility": "UtilityStepEnhancer",
            "Base": "BaseStepEnhancer"
        }
    
    @property
    def enhancers(self):
        """Public access to enhancers dictionary."""
        return self._enhancers
    
    def enhance_validation(self, script_name: str, existing_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route validation enhancement to appropriate step type enhancer.
        
        Args:
            script_name: Name of the script being validated
            existing_results: Existing validation results to enhance
            
        Returns:
            Enhanced validation results with step type-specific issues
        """
        step_type = detect_step_type_from_registry(script_name)
        
        if not step_type:
            # Default to Processing if step type cannot be determined
            step_type = "Processing"
        
        enhancer = self._get_enhancer(step_type)
        
        if enhancer:
            try:
                return enhancer.enhance_validation(existing_results, script_name)
            except Exception as e:
                # If enhancement fails, return original results with warning
                if isinstance(existing_results, dict):
                    existing_results.setdefault('issues', []).append({
                        'severity': 'WARNING',
                        'category': 'step_type_enhancement_error',
                        'message': f'Step type enhancement failed for {step_type}: {str(e)}',
                        'details': {'script': script_name, 'step_type': step_type, 'error': str(e)},
                        'recommendation': 'Check step type enhancer implementation'
                    })
                return existing_results
        
        return existing_results
    
    def _get_enhancer(self, step_type: str):
        """Get or create enhancer for the specified step type."""
        if step_type not in self._enhancers:
            enhancer_class_name = self._enhancer_classes.get(step_type)
            if enhancer_class_name:
                try:
                    # Import the enhancer class dynamically using relative import
                    import importlib
                    relative_module_name = f".step_type_enhancers.{step_type.lower()}_enhancer"
                    module = importlib.import_module(relative_module_name, package=__package__)
                    enhancer_class = getattr(module, enhancer_class_name)
                    self._enhancers[step_type] = enhancer_class()
                except (ImportError, AttributeError) as e:
                    # If enhancer cannot be loaded, use base enhancer
                    try:
                        from .step_type_enhancers.base_enhancer import BaseStepEnhancer
                        self._enhancers[step_type] = BaseStepEnhancer(step_type)
                    except ImportError:
                        return None
        
        return self._enhancers.get(step_type)
    
    def get_step_type_requirements(self, step_type: str) -> Dict[str, Any]:
        """
        Get validation requirements for each step type.
        
        Args:
            step_type: The SageMaker step type
            
        Returns:
            Dictionary containing step type requirements and patterns
        """
        requirements = {
            "Processing": {
                "input_types": ["ProcessingInput"],
                "output_types": ["ProcessingOutput"],
                "required_methods": ["_create_processor"],
                "required_patterns": ["data_transformation", "environment_variables"],
                "common_frameworks": ["pandas", "sklearn", "numpy"],
                "typical_paths": ["/opt/ml/processing/input", "/opt/ml/processing/output"],
                "validation_focus": ["data_processing", "file_operations", "environment_variables"]
            },
            "Training": {
                "input_types": ["TrainingInput"],
                "output_types": ["model_artifacts"],
                "required_methods": ["_create_estimator", "_prepare_hyperparameters_file"],
                "required_patterns": ["training_loop", "model_saving", "hyperparameter_loading"],
                "common_frameworks": ["xgboost", "pytorch", "sklearn", "tensorflow"],
                "typical_paths": ["/opt/ml/input/data/train", "/opt/ml/model", "/opt/ml/input/data/config"],
                "validation_focus": ["training_patterns", "model_persistence", "hyperparameter_handling"]
            },
            "CreateModel": {
                "input_types": ["model_artifacts"],
                "output_types": ["model_endpoint"],
                "required_methods": ["_create_model"],
                "required_patterns": ["model_loading", "inference_code"],
                "common_frameworks": ["xgboost", "pytorch", "sklearn", "tensorflow"],
                "typical_paths": ["/opt/ml/model"],
                "validation_focus": ["model_loading", "inference_functions", "container_configuration"]
            },
            "Transform": {
                "input_types": ["TransformInput"],
                "output_types": ["transform_results"],
                "required_methods": ["_create_transformer"],
                "required_patterns": ["batch_processing", "model_inference"],
                "common_frameworks": ["xgboost", "pytorch", "sklearn"],
                "typical_paths": ["/opt/ml/transform"],
                "validation_focus": ["batch_processing", "model_inference", "transform_configuration"]
            },
            "RegisterModel": {
                "input_types": ["model_artifacts"],
                "output_types": ["registered_model"],
                "required_methods": ["_create_model_package"],
                "required_patterns": ["model_metadata", "approval_workflow"],
                "common_frameworks": ["sagemaker"],
                "typical_paths": [],
                "validation_focus": ["model_metadata", "approval_workflow", "model_package_creation"]
            },
            "Utility": {
                "input_types": ["various"],
                "output_types": ["prepared_files"],
                "required_methods": ["_prepare_files"],
                "required_patterns": ["file_preparation"],
                "common_frameworks": ["boto3", "json"],
                "typical_paths": ["/opt/ml/processing/input", "/opt/ml/processing/output"],
                "validation_focus": ["file_preparation", "parameter_generation", "special_case_handling"]
            },
            "Base": {
                "input_types": ["base_inputs"],
                "output_types": ["base_outputs"],
                "required_methods": ["create_step"],
                "required_patterns": ["foundation_patterns"],
                "common_frameworks": ["sagemaker"],
                "typical_paths": [],
                "validation_focus": ["foundation_patterns", "step_creation", "basic_validation"]
            }
        }
        
        return requirements.get(step_type, {})
    
    def get_supported_step_types(self) -> list[str]:
        """Get list of all supported step types."""
        return list(self._enhancer_classes.keys())
    
    def enhance_validation_results(self, existing_results: Dict[str, Any], script_name: str) -> Dict[str, Any]:
        """
        Enhance validation results with step type-specific validation.
        
        Args:
            existing_results: Existing validation results to enhance
            script_name: Name of the script being validated
            
        Returns:
            Enhanced validation results with step type-specific issues
        """
        return self.enhance_validation(script_name, existing_results)
    
    def get_step_type_statistics(self) -> Dict[str, Any]:
        """Get statistics about step type usage and validation patterns."""
        return {
            "supported_step_types": len(self._enhancer_classes),
            "loaded_enhancers": len(self._enhancers),
            "step_type_mapping": dict(self._enhancer_classes),
            "requirements_available": [step_type for step_type in self._enhancer_classes.keys()]
        }
