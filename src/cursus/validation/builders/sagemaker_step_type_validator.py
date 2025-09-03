"""
SageMaker Step Type Validator for step builders.
Validates step builders based on their SageMaker step type requirements.
"""

from typing import Dict, Any, List, Type, Optional
import logging
import inspect
from abc import ABC, abstractmethod

from ...core.base.builder_base import StepBuilderBase
from ...registry.step_names import get_sagemaker_step_type, validate_sagemaker_step_type
from .base_test import ValidationViolation, ValidationLevel

logger = logging.getLogger(__name__)


class SageMakerStepTypeValidator:
    """Validates step builders based on their SageMaker step type."""
    
    def __init__(self, builder_class: Type[StepBuilderBase]):
        """
        Initialize validator with a step builder class.
        
        Args:
            builder_class: The step builder class to validate
        """
        self.builder_class = builder_class
        self.step_name = self._detect_step_name()
        self.sagemaker_step_type = get_sagemaker_step_type(self.step_name) if self.step_name else None
        
    def _detect_step_name(self) -> Optional[str]:
        """Detect step name from builder class name."""
        class_name = self.builder_class.__name__
        
        # Remove common suffixes to get base name
        suffixes = ["StepBuilder", "Builder", "Step"]
        for suffix in suffixes:
            if class_name.endswith(suffix):
                class_name = class_name[:-len(suffix)]
                break
        
        # Try to find matching step name in registry
        from ...registry.step_names import STEP_NAMES
        for step_name, info in STEP_NAMES.items():
            if info["builder_step_name"].replace("StepBuilder", "").replace("Builder", "") == class_name:
                return step_name
                
        return None
    
    def validate_step_type_compliance(self) -> List[ValidationViolation]:
        """Validate compliance with SageMaker step type requirements."""
        violations = []
        
        if not self.step_name:
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="step_type_detection",
                message=f"Could not detect step name for builder class: {self.builder_class.__name__}",
                details="Builder class name should match registry pattern"
            ))
            return violations
            
        if not self.sagemaker_step_type:
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="step_type_classification",
                message=f"No SageMaker step type found for step: {self.step_name}",
                details="Step should be classified in the registry"
            ))
            return violations
            
        # Validate step type exists
        if not validate_sagemaker_step_type(self.sagemaker_step_type):
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="step_type_validation",
                message=f"Invalid SageMaker step type: {self.sagemaker_step_type}",
                details="Step type should be one of: Processing, Training, Transform, CreateModel, RegisterModel, Base, Utility"
            ))
            return violations
        
        # Run step-type-specific validation
        if self.sagemaker_step_type == "Processing":
            violations.extend(self._validate_processing_step())
        elif self.sagemaker_step_type == "Training":
            violations.extend(self._validate_training_step())
        elif self.sagemaker_step_type == "Transform":
            violations.extend(self._validate_transform_step())
        elif self.sagemaker_step_type == "CreateModel":
            violations.extend(self._validate_create_model_step())
        elif self.sagemaker_step_type == "RegisterModel":
            violations.extend(self._validate_register_model_step())
        elif self.sagemaker_step_type in ["Base", "Utility"]:
            # Special cases - no specific validation needed
            pass
            
        return violations
    
    def _validate_processing_step(self) -> List[ValidationViolation]:
        """Validate Processing step requirements."""
        violations = []
        
        # Check that create_step method exists and returns ProcessingStep
        if hasattr(self.builder_class, 'create_step'):
            create_step_method = getattr(self.builder_class, 'create_step')
            if hasattr(create_step_method, '__annotations__'):
                return_annotation = create_step_method.__annotations__.get('return')
                if return_annotation:
                    # Check if return type suggests ProcessingStep
                    return_type_str = str(return_annotation)
                    if 'ProcessingStep' not in return_type_str:
                        violations.append(ValidationViolation(
                            level=ValidationLevel.WARNING,
                            category="processing_step_return_type",
                            message="Processing step builder should return ProcessingStep",
                            details=f"Current return annotation: {return_type_str}"
                        ))
        
        # Check for processor creation methods
        expected_methods = ['_create_processor', '_get_processor']
        found_methods = []
        for method_name in expected_methods:
            if hasattr(self.builder_class, method_name):
                found_methods.append(method_name)
                
        if not found_methods:
            violations.append(ValidationViolation(
                level=ValidationLevel.WARNING,
                category="processing_step_methods",
                message="Processing step should have processor creation methods",
                details=f"Expected one of: {expected_methods}"
            ))
        
        # Check for input/output handling methods
        if not hasattr(self.builder_class, '_get_inputs'):
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="processing_step_inputs",
                message="Processing step must implement _get_inputs method",
                details="Method should return List[ProcessingInput]"
            ))
            
        if not hasattr(self.builder_class, '_get_outputs'):
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="processing_step_outputs",
                message="Processing step must implement _get_outputs method",
                details="Method should return List[ProcessingOutput]"
            ))
        
        return violations
    
    def _validate_training_step(self) -> List[ValidationViolation]:
        """Validate Training step requirements."""
        violations = []
        
        # Check that create_step method returns TrainingStep
        if hasattr(self.builder_class, 'create_step'):
            create_step_method = getattr(self.builder_class, 'create_step')
            if hasattr(create_step_method, '__annotations__'):
                return_annotation = create_step_method.__annotations__.get('return')
                if return_annotation:
                    return_type_str = str(return_annotation)
                    if 'TrainingStep' not in return_type_str:
                        violations.append(ValidationViolation(
                            level=ValidationLevel.WARNING,
                            category="training_step_return_type",
                            message="Training step builder should return TrainingStep",
                            details=f"Current return annotation: {return_type_str}"
                        ))
        
        # Check for estimator creation methods
        expected_methods = ['_create_estimator', '_get_estimator']
        found_methods = []
        for method_name in expected_methods:
            if hasattr(self.builder_class, method_name):
                found_methods.append(method_name)
                
        if not found_methods:
            violations.append(ValidationViolation(
                level=ValidationLevel.WARNING,
                category="training_step_methods",
                message="Training step should have estimator creation methods",
                details=f"Expected one of: {expected_methods}"
            ))
        
        # Check for input handling (should return Dict[str, TrainingInput])
        if not hasattr(self.builder_class, '_get_inputs'):
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="training_step_inputs",
                message="Training step must implement _get_inputs method",
                details="Method should return Dict[str, TrainingInput]"
            ))
        
        # Check for hyperparameter handling
        hyperparameter_methods = ['_prepare_hyperparameters_file', '_get_hyperparameters']
        found_hyperparameter_methods = []
        for method_name in hyperparameter_methods:
            if hasattr(self.builder_class, method_name):
                found_hyperparameter_methods.append(method_name)
                
        if not found_hyperparameter_methods:
            violations.append(ValidationViolation(
                level=ValidationLevel.INFO,
                category="training_step_hyperparameters",
                message="Training step may benefit from hyperparameter handling methods",
                details=f"Consider implementing one of: {hyperparameter_methods}"
            ))
        
        return violations
    
    def _validate_transform_step(self) -> List[ValidationViolation]:
        """Validate Transform step requirements."""
        violations = []
        
        # Check that create_step method returns TransformStep
        if hasattr(self.builder_class, 'create_step'):
            create_step_method = getattr(self.builder_class, 'create_step')
            if hasattr(create_step_method, '__annotations__'):
                return_annotation = create_step_method.__annotations__.get('return')
                if return_annotation:
                    return_type_str = str(return_annotation)
                    if 'TransformStep' not in return_type_str:
                        violations.append(ValidationViolation(
                            level=ValidationLevel.WARNING,
                            category="transform_step_return_type",
                            message="Transform step builder should return TransformStep",
                            details=f"Current return annotation: {return_type_str}"
                        ))
        
        # Check for transformer creation methods
        expected_methods = ['_create_transformer', '_get_transformer']
        found_methods = []
        for method_name in expected_methods:
            if hasattr(self.builder_class, method_name):
                found_methods.append(method_name)
                
        if not found_methods:
            violations.append(ValidationViolation(
                level=ValidationLevel.WARNING,
                category="transform_step_methods",
                message="Transform step should have transformer creation methods",
                details=f"Expected one of: {expected_methods}"
            ))
        
        # Check for input handling (should return TransformInput)
        if not hasattr(self.builder_class, '_get_inputs'):
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="transform_step_inputs",
                message="Transform step must implement _get_inputs method",
                details="Method should return TransformInput"
            ))
        
        return violations
    
    def _validate_create_model_step(self) -> List[ValidationViolation]:
        """Validate CreateModel step requirements."""
        violations = []
        
        # Check that create_step method returns CreateModelStep
        if hasattr(self.builder_class, 'create_step'):
            create_step_method = getattr(self.builder_class, 'create_step')
            if hasattr(create_step_method, '__annotations__'):
                return_annotation = create_step_method.__annotations__.get('return')
                if return_annotation:
                    return_type_str = str(return_annotation)
                    if 'CreateModelStep' not in return_type_str:
                        violations.append(ValidationViolation(
                            level=ValidationLevel.WARNING,
                            category="create_model_step_return_type",
                            message="CreateModel step builder should return CreateModelStep",
                            details=f"Current return annotation: {return_type_str}"
                        ))
        
        # Check for model creation methods
        expected_methods = ['_create_model', '_get_model']
        found_methods = []
        for method_name in expected_methods:
            if hasattr(self.builder_class, method_name):
                found_methods.append(method_name)
                
        if not found_methods:
            violations.append(ValidationViolation(
                level=ValidationLevel.WARNING,
                category="create_model_step_methods",
                message="CreateModel step should have model creation methods",
                details=f"Expected one of: {expected_methods}"
            ))
        
        # Check for input handling (should handle model_data)
        if not hasattr(self.builder_class, '_get_inputs'):
            violations.append(ValidationViolation(
                level=ValidationLevel.ERROR,
                category="create_model_step_inputs",
                message="CreateModel step must implement _get_inputs method",
                details="Method should handle model_data input"
            ))
        
        return violations
    
    def _validate_register_model_step(self) -> List[ValidationViolation]:
        """Validate RegisterModel step requirements."""
        violations = []
        
        # Check that create_step method returns RegisterModel
        if hasattr(self.builder_class, 'create_step'):
            create_step_method = getattr(self.builder_class, 'create_step')
            if hasattr(create_step_method, '__annotations__'):
                return_annotation = create_step_method.__annotations__.get('return')
                if return_annotation:
                    return_type_str = str(return_annotation)
                    if 'RegisterModel' not in return_type_str:
                        violations.append(ValidationViolation(
                            level=ValidationLevel.WARNING,
                            category="register_model_step_return_type",
                            message="RegisterModel step builder should return RegisterModel",
                            details=f"Current return annotation: {return_type_str}"
                        ))
        
        # Check for model registration methods
        expected_methods = ['_create_model_package', '_get_model_package_args']
        found_methods = []
        for method_name in expected_methods:
            if hasattr(self.builder_class, method_name):
                found_methods.append(method_name)
                
        if not found_methods:
            violations.append(ValidationViolation(
                level=ValidationLevel.INFO,
                category="register_model_step_methods",
                message="RegisterModel step may benefit from model package methods",
                details=f"Consider implementing one of: {expected_methods}"
            ))
        
        return violations
    
    def get_step_type_info(self) -> Dict[str, Any]:
        """Get information about the step type classification."""
        return {
            "builder_class": self.builder_class.__name__,
            "detected_step_name": self.step_name,
            "sagemaker_step_type": self.sagemaker_step_type,
            "is_valid_step_type": validate_sagemaker_step_type(self.sagemaker_step_type) if self.sagemaker_step_type else False
        }
