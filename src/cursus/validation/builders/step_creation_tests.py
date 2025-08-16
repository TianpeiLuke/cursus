"""
Level 3 Step Creation Tests for step builders.

These tests focus on core step builder functionality:
- Step instantiation validation
- Step type compliance checking
- Step configuration validity
- Step name generation
- Step dependencies attachment
"""

from typing import Dict, Any, List, Union, Optional
from .base_test import UniversalStepBuilderTestBase


class StepCreationTests(UniversalStepBuilderTestBase):
    """
    Level 3 tests focusing on step creation validation.
    
    These tests validate that a step builder correctly creates valid SageMaker steps
    with proper configuration and compliance with expected step types.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return step type-specific test methods for step creation tests."""
        step_type = self.step_info.get('sagemaker_step_type', 'Unknown')
        
        if step_type == "Processing":
            return ['test_processing_step_creation']
        elif step_type == "Training":
            return ['test_training_step_creation']
        elif step_type == "Transform":
            return ['test_transform_step_creation']
        elif step_type == "CreateModel":
            return ['test_create_model_step_creation']
        else:
            return []  # Generic tests only
    
    def _configure_step_type_mocks(self) -> None:
        """Configure step type-specific mock objects for step creation tests."""
        # Step creation tests work with any valid configuration
        # Mock factory handles step-type specific configuration creation
        pass
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate step type-specific requirements for step creation tests."""
        return {
            "step_creation_tests_completed": True,
            "core_functionality_validated": True
        }
    
    def test_step_instantiation(self) -> None:
        """Test that builder creates a valid step instance."""
        try:
            # Create builder instance with mock config
            builder = self._create_builder_instance()
            
            # Test step creation
            step = builder.create_step()
            
            # Validate step instance
            self._assert(
                step is not None,
                "Builder should create a step instance"
            )
            
            # Validate step has basic attributes
            self._assert(
                hasattr(step, 'name'),
                "Step must have a 'name' attribute"
            )
            
            # Log successful step creation
            step_type = type(step).__name__
            self._log(f"Successfully created step instance of type: {step_type}")
            
        except Exception as e:
            self._assert(
                False,
                f"Step instantiation failed: {str(e)}"
            )
    
    def test_step_type_compliance(self) -> None:
        """Test that created step matches expected SageMaker step type."""
        try:
            # Create builder instance with mock config
            builder = self._create_builder_instance()
            
            # Get expected step type from registry
            expected_step_type = self.step_info.get('sagemaker_step_type', 'Unknown')
            
            if expected_step_type == 'Unknown':
                self._log("Skipping step type compliance test - unknown step type")
                return
            
            # Create step
            step = builder.create_step()
            
            # Get actual step type
            actual_step_type = type(step).__name__
            
            # Map expected step type to actual class name
            expected_class_name = self._get_expected_step_class_name(expected_step_type)
            
            # Validate step type compliance
            self._assert(
                actual_step_type == expected_class_name,
                f"Expected step type {expected_class_name}, got {actual_step_type}"
            )
            
            self._log(f"Step type compliance validated: {actual_step_type}")
            
        except Exception as e:
            self._assert(
                False,
                f"Step type compliance test failed: {str(e)}"
            )
    
    def test_step_configuration_validity(self) -> None:
        """Test that step is configured with valid parameters."""
        try:
            # Create builder instance with mock config
            builder = self._create_builder_instance()
            
            # Create step
            step = builder.create_step()
            
            # Validate step has required attributes
            required_attrs = ['name']
            for attr in required_attrs:
                self._assert(
                    hasattr(step, attr),
                    f"Step missing required attribute: {attr}"
                )
            
            # Validate step name is not empty
            self._assert(
                step.name and len(step.name.strip()) > 0,
                "Step name must not be empty"
            )
            
            # Step type-specific configuration validation
            self._validate_step_type_specific_configuration(step)
            
            self._log(f"Step configuration validated for step: {step.name}")
            
        except Exception as e:
            self._assert(
                False,
                f"Step configuration validity test failed: {str(e)}"
            )
    
    def test_step_name_generation(self) -> None:
        """Test that step names are generated correctly."""
        try:
            # Create builder instance with mock config
            builder = self._create_builder_instance()
            
            # Create step
            step = builder.create_step()
            
            # Validate step name format
            step_name = step.name
            
            # Basic name validation
            self._assert(
                isinstance(step_name, str),
                "Step name must be a string"
            )
            
            self._assert(
                len(step_name) > 0,
                "Step name must not be empty"
            )
            
            # Validate name doesn't contain invalid characters
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            for char in invalid_chars:
                self._assert(
                    char not in step_name,
                    f"Step name contains invalid character: {char}"
                )
            
            # Log step name
            self._log(f"Step name generated: {step_name}")
            
        except Exception as e:
            self._assert(
                False,
                f"Step name generation test failed: {str(e)}"
            )
    
    def test_step_dependencies_attachment(self) -> None:
        """Test that step dependencies are properly handled."""
        try:
            # Create builder instance with mock config
            builder = self._create_builder_instance()
            
            # Create step
            step = builder.create_step()
            
            # Check if step has dependency-related attributes
            # This varies by step type, so we do basic validation
            
            # For steps that support dependencies, check they're handled properly
            if hasattr(step, 'depends_on'):
                depends_on = step.depends_on
                if depends_on is not None:
                    self._assert(
                        isinstance(depends_on, (list, tuple)),
                        "Step dependencies must be a list or tuple"
                    )
            
            # Log dependency status
            has_dependencies = hasattr(step, 'depends_on') and step.depends_on
            self._log(f"Step dependency handling validated. Has dependencies: {has_dependencies}")
            
        except Exception as e:
            self._assert(
                False,
                f"Step dependencies attachment test failed: {str(e)}"
            )
    
    # Step type-specific creation tests
    
    def test_processing_step_creation(self) -> None:
        """Test Processing step-specific creation requirements."""
        try:
            builder = self._create_builder_instance()
            step = builder.create_step()
            
            # Validate ProcessingStep specific attributes
            self._assert(
                hasattr(step, 'processor'),
                "ProcessingStep must have a processor attribute"
            )
            
            # Validate processor configuration
            processor = step.processor
            if processor:
                self._assert(
                    hasattr(processor, 'role'),
                    "Processor must have a role attribute"
                )
                
                self._assert(
                    hasattr(processor, 'instance_type'),
                    "Processor must have an instance_type attribute"
                )
            
            self._log("Processing step creation validated")
            
        except Exception as e:
            self._assert(
                False,
                f"Processing step creation test failed: {str(e)}"
            )
    
    def test_training_step_creation(self) -> None:
        """Test Training step-specific creation requirements."""
        try:
            builder = self._create_builder_instance()
            step = builder.create_step()
            
            # Validate TrainingStep specific attributes
            self._assert(
                hasattr(step, 'estimator'),
                "TrainingStep must have an estimator attribute"
            )
            
            # Validate estimator configuration
            estimator = step.estimator
            if estimator:
                self._assert(
                    hasattr(estimator, 'role'),
                    "Estimator must have a role attribute"
                )
                
                self._assert(
                    hasattr(estimator, 'instance_type'),
                    "Estimator must have an instance_type attribute"
                )
            
            self._log("Training step creation validated")
            
        except Exception as e:
            self._assert(
                False,
                f"Training step creation test failed: {str(e)}"
            )
    
    def test_transform_step_creation(self) -> None:
        """Test Transform step-specific creation requirements."""
        try:
            builder = self._create_builder_instance()
            step = builder.create_step()
            
            # Validate TransformStep specific attributes
            self._assert(
                hasattr(step, 'transformer'),
                "TransformStep must have a transformer attribute"
            )
            
            # Validate transformer configuration
            transformer = step.transformer
            if transformer:
                self._assert(
                    hasattr(transformer, 'model_name') or hasattr(transformer, 'model_data'),
                    "Transformer must have model_name or model_data attribute"
                )
            
            self._log("Transform step creation validated")
            
        except Exception as e:
            self._assert(
                False,
                f"Transform step creation test failed: {str(e)}"
            )
    
    def test_create_model_step_creation(self) -> None:
        """Test CreateModel step-specific creation requirements."""
        try:
            builder = self._create_builder_instance()
            step = builder.create_step()
            
            # Validate CreateModelStep specific attributes
            self._assert(
                hasattr(step, 'model'),
                "CreateModelStep must have a model attribute"
            )
            
            # Validate model configuration
            model = step.model
            if model:
                self._assert(
                    hasattr(model, 'name'),
                    "Model must have a name attribute"
                )
            
            self._log("CreateModel step creation validated")
            
        except Exception as e:
            self._assert(
                False,
                f"CreateModel step creation test failed: {str(e)}"
            )
    
    # Helper methods
    
    def _get_expected_step_class_name(self, step_type: str) -> str:
        """Map step type to expected class name."""
        step_type_mapping = {
            "Processing": "ProcessingStep",
            "Training": "TrainingStep", 
            "Transform": "TransformStep",
            "CreateModel": "CreateModelStep",
            "Tuning": "TuningStep",
            "Lambda": "LambdaStep",
            "Callback": "CallbackStep",
            "Condition": "ConditionStep",
            "Fail": "FailStep",
            "EMR": "EMRStep",
            "AutoML": "AutoMLStep",
            "NotebookJob": "NotebookJobStep",
            "MimsModelRegistrationProcessing": "MimsModelRegistrationProcessingStep",
            "CradleDataLoading": "CradleDataLoadingStep"
        }
        
        return step_type_mapping.get(step_type, f"{step_type}Step")
    
    def _validate_step_type_specific_configuration(self, step) -> None:
        """Validate step type-specific configuration."""
        step_type = type(step).__name__
        
        if "ProcessingStep" in step_type:
            self._validate_processing_step_config(step)
        elif "TrainingStep" in step_type:
            self._validate_training_step_config(step)
        elif "TransformStep" in step_type:
            self._validate_transform_step_config(step)
        elif "CreateModelStep" in step_type:
            self._validate_create_model_step_config(step)
        else:
            # Generic validation for other step types
            self._log(f"Generic configuration validation for step type: {step_type}")
    
    def _validate_processing_step_config(self, step) -> None:
        """Validate ProcessingStep configuration."""
        if hasattr(step, 'inputs') and step.inputs:
            self._assert(
                isinstance(step.inputs, list),
                "ProcessingStep inputs must be a list"
            )
        
        if hasattr(step, 'outputs') and step.outputs:
            self._assert(
                isinstance(step.outputs, list),
                "ProcessingStep outputs must be a list"
            )
    
    def _validate_training_step_config(self, step) -> None:
        """Validate TrainingStep configuration."""
        if hasattr(step, 'inputs') and step.inputs:
            self._assert(
                isinstance(step.inputs, dict),
                "TrainingStep inputs must be a dictionary"
            )
    
    def _validate_transform_step_config(self, step) -> None:
        """Validate TransformStep configuration."""
        if hasattr(step, 'transform_inputs') and step.transform_inputs:
            self._assert(
                isinstance(step.transform_inputs, list),
                "TransformStep inputs must be a list"
            )
    
    def _validate_create_model_step_config(self, step) -> None:
        """Validate CreateModelStep configuration."""
        if hasattr(step, 'model') and step.model:
            model = step.model
            if hasattr(model, 'primary_container'):
                self._assert(
                    model.primary_container is not None,
                    "CreateModelStep model must have primary_container"
                )
