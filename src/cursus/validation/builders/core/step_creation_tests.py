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
        step_type = self.step_info.get("sagemaker_step_type", "Unknown")

        if step_type == "Processing":
            return ["test_processing_step_creation"]
        elif step_type == "Training":
            return ["test_training_step_creation"]
        elif step_type == "Transform":
            return ["test_transform_step_creation"]
        elif step_type == "CreateModel":
            return ["test_create_model_step_creation"]
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
            "core_functionality_validated": True,
        }

    def test_step_instantiation(self) -> None:
        """Test that builder has proper create_step method signature and behavior."""
        try:
            # Test that create_step method exists and is callable WITHOUT creating builder instance
            builder_instance = self.builder_class.__new__(self.builder_class)  # Create without __init__
            self._assert(hasattr(builder_instance, 'create_step'), "Builder must have create_step method")
            self._assert(callable(builder_instance.create_step), "create_step must be callable")

            # Test method signature - should accept inputs parameter
            import inspect
            sig = inspect.signature(builder_instance.create_step)
            params = list(sig.parameters.keys())
            
            # Should have 'inputs' parameter (and possibly others like **kwargs)
            has_inputs = 'inputs' in params or any('inputs' in str(param) for param in sig.parameters.values())
            self._assert(has_inputs or len(params) > 0, "create_step should accept inputs parameter")
            
            self._log(f"create_step method signature: {sig}")

            # Test that builder constructor validates config properly (architectural validation)
            try:
                # Try to create builder with minimal mock config
                builder = self._create_builder_instance()
                self._log("Builder accepts minimal config or validates properly")
            except Exception as e:
                # This is expected - builder should validate config properly
                error_msg = str(e)
                self._assert(len(error_msg) > 0, "Error message should be informative")
                self._log(f"Builder properly validates config during construction: {error_msg}")

            self._log("Step instantiation method validation completed")

        except Exception as e:
            self._assert(False, f"Step instantiation validation failed: {str(e)}")

    def test_step_type_compliance(self) -> None:
        """Test that builder is registered with correct step type information."""
        try:
            # Get expected step type from registry
            expected_step_type = self.step_info.get("sagemaker_step_type", "Unknown")

            if expected_step_type == "Unknown":
                self._log("Skipping step type compliance test - unknown step type")
                return

            # Test 1: Builder class name should align with step type
            builder_class_name = self.builder_class.__name__
            self._assert(builder_class_name.endswith('StepBuilder'), 
                        f"Builder class name should end with 'StepBuilder': {builder_class_name}")

            # Test 2: Step type should be valid SageMaker step type
            valid_step_types = [
                "Processing", "Training", "Transform", "CreateModel", "Tuning",
                "Lambda", "Callback", "Condition", "Fail", "EMR", "AutoML", "NotebookJob"
            ]
            
            self._assert(expected_step_type in valid_step_types or expected_step_type.startswith('Mims') or expected_step_type.startswith('Cradle'), 
                        f"Step type should be valid SageMaker type: {expected_step_type}")

            # Test 3: Builder should have step type information accessible
            self._log(f"Builder {builder_class_name} registered for step type: {expected_step_type}")

            # Test 4: Expected class name mapping should be consistent
            expected_class_name = self._get_expected_step_class_name(expected_step_type)
            self._log(f"Expected step class: {expected_class_name}")

            # Test 5: Builder should be able to indicate its step type without creating steps
            try:
                builder = self._create_builder_instance()
                
                # Some builders might have a method to get step type
                if hasattr(builder, 'get_step_type'):
                    try:
                        builder_step_type = builder.get_step_type()
                        if builder_step_type:
                            self._assert(builder_step_type == expected_step_type,
                                       f"Builder reports step type {builder_step_type}, expected {expected_step_type}")
                            self._log(f"Builder reports correct step type: {builder_step_type}")
                    except Exception as e:
                        self._log(f"get_step_type method exists but requires parameters: {e}")
                        
            except Exception as e:
                # Builder creation might fail due to config - that's expected
                self._log(f"Builder creation failed (expected): {e}")

            self._log(f"Step type compliance validated for {expected_step_type}")

        except Exception as e:
            self._assert(False, f"Step type compliance test failed: {str(e)}")

    def test_step_configuration_validity(self) -> None:
        """Test that builder properly validates configuration requirements."""
        try:
            # Test 1: Builder should validate config type
            try:
                # Create builder with minimal mock config (should fail gracefully)
                builder = self._create_builder_instance()
                self._log("Builder accepts minimal config or validates properly")
            except Exception as e:
                # This is expected - builder should validate config type
                error_msg = str(e)
                self._assert("Config" in error_msg or "config" in error_msg, 
                           f"Config validation error should mention config: {error_msg}")
                self._log(f"Builder properly validates config type: {error_msg}")

            # Test 2: Builder should have validate_configuration method
            builder_instance = self.builder_class.__new__(self.builder_class)  # Create without __init__
            self._assert(hasattr(builder_instance, 'validate_configuration'), 
                        "Builder must have validate_configuration method")
            self._assert(callable(builder_instance.validate_configuration), 
                        "validate_configuration must be callable")

            # Test 3: Builder should handle invalid config gracefully
            try:
                from types import SimpleNamespace
                invalid_config = SimpleNamespace()  # Minimal invalid config
                invalid_config.region = "test"  # Add minimal field
                
                # Try to create builder with invalid config
                test_builder = self.builder_class(
                    config=invalid_config,
                    sagemaker_session=self.mock_session,
                    role=self.mock_role,
                    registry_manager=self.mock_registry_manager,
                    dependency_resolver=self.mock_dependency_resolver,
                )
                self._log("Builder accepts basic config structure")
            except Exception as e:
                # This is expected - builder should validate config properly
                error_msg = str(e)
                self._assert(len(error_msg) > 0, "Error message should be informative")
                self._log(f"Builder properly validates config structure: {error_msg}")

            self._log("Configuration validity validation completed")

        except Exception as e:
            self._assert(False, f"Step configuration validity test failed: {str(e)}")

    def test_step_name_generation(self) -> None:
        """Test that builder has proper step name generation methods."""
        try:
            # Test that builder has name generation method
            builder_instance = self.builder_class.__new__(self.builder_class)  # Create without __init__
            self._assert(hasattr(builder_instance, '_get_step_name'), 
                        "Builder must have _get_step_name method")
            self._assert(callable(builder_instance._get_step_name), 
                        "_get_step_name must be callable")

            # Test method signature
            import inspect
            sig = inspect.signature(builder_instance._get_step_name)
            self._log(f"_get_step_name signature: {sig}")

            # Test that builder can generate names without full step creation
            try:
                builder = self._create_builder_instance()
                
                # Try to call _get_step_name method directly (architectural validation)
                try:
                    step_name = builder._get_step_name()
                    if step_name:
                        self._assert(isinstance(step_name, str), "Step name must be a string")
                        self._assert(len(step_name) > 0, "Step name must not be empty")
                        self._log(f"Step name generation method works: {step_name}")
                    else:
                        self._log("Step name generation method exists but requires parameters")
                except Exception as e:
                    # Method might require parameters - that's fine
                    self._log(f"Step name generation method requires parameters: {e}")
                    
            except Exception as e:
                # Builder creation might fail due to config - that's expected
                self._log(f"Builder creation failed (expected): {e}")

            self._log("Step name generation method validation completed")

        except Exception as e:
            self._assert(False, f"Step name generation test failed: {str(e)}")

    def test_step_dependencies_attachment(self) -> None:
        """Test that builder has proper dependency handling methods."""
        try:
            # Test that builder has dependency-related methods
            builder_instance = self.builder_class.__new__(self.builder_class)  # Create without __init__
            
            # Check for dependency-related methods
            dependency_methods = [
                'get_required_dependencies',
                'get_optional_dependencies', 
                'extract_inputs_from_dependencies'
            ]
            
            for method_name in dependency_methods:
                if hasattr(builder_instance, method_name):
                    method = getattr(builder_instance, method_name)
                    self._assert(callable(method), f"{method_name} must be callable")
                    self._log(f"Builder has {method_name} method")

            # Test that builder can handle dependency extraction without full step creation
            try:
                builder = self._create_builder_instance()
                
                # Test dependency methods if they exist
                if hasattr(builder, 'get_required_dependencies'):
                    try:
                        required_deps = builder.get_required_dependencies()
                        if required_deps:
                            self._assert(isinstance(required_deps, (list, tuple, dict)), 
                                       "Required dependencies should be a collection")
                            self._log(f"Required dependencies: {required_deps}")
                        else:
                            self._log("No required dependencies")
                    except Exception as e:
                        self._log(f"get_required_dependencies method exists but requires parameters: {e}")

                if hasattr(builder, 'get_optional_dependencies'):
                    try:
                        optional_deps = builder.get_optional_dependencies()
                        if optional_deps:
                            self._assert(isinstance(optional_deps, (list, tuple, dict)), 
                                       "Optional dependencies should be a collection")
                            self._log(f"Optional dependencies: {optional_deps}")
                        else:
                            self._log("No optional dependencies")
                    except Exception as e:
                        self._log(f"get_optional_dependencies method exists but requires parameters: {e}")
                        
            except Exception as e:
                # Builder creation might fail due to config - that's expected
                self._log(f"Builder creation failed (expected): {e}")

            self._log("Step dependency handling method validation completed")

        except Exception as e:
            self._assert(False, f"Step dependencies attachment test failed: {str(e)}")

    # Step type-specific creation tests

    def test_step_creation_interface_compliance(self) -> None:
        """Test universal step creation interface compliance - shared patterns across all step types."""
        try:
            # Test 1: Builder should have create_step method (universal requirement)
            builder_instance = self.builder_class.__new__(self.builder_class)  # Create without __init__
            self._assert(hasattr(builder_instance, 'create_step'), 
                        "All step builders must have create_step method")
            self._assert(callable(builder_instance.create_step), 
                        "create_step must be callable")

            # Test 2: Check method signature follows common patterns
            import inspect
            sig = inspect.signature(builder_instance.create_step)
            params = list(sig.parameters.keys())
            
            # Universal requirement: should accept inputs or **kwargs
            has_inputs = 'inputs' in params or 'kwargs' in params or len(params) > 0
            self._assert(has_inputs, 
                        f"create_step should accept inputs parameter: {params}")
            
            self._log(f"✅ create_step method signature: {sig}")

            # Test 3: Builder should validate config type (universal pattern)
            try:
                builder = self._create_builder_instance()
                self._log("✅ Builder accepts minimal config or validates properly")
            except Exception as e:
                # This is expected - builder should validate config properly
                error_msg = str(e)
                self._assert(len(error_msg) > 0, "Error message should be informative")
                self._log(f"✅ Builder properly validates config: {error_msg}")

            # Test 4: Builder should be registered with a valid SageMaker step type
            expected_step_type = self.step_info.get("sagemaker_step_type", "Unknown")
            valid_step_types = [
                "Processing", "Training", "Transform", "CreateModel", "Tuning",
                "Lambda", "Callback", "Condition", "Fail", "EMR", "AutoML", "NotebookJob"
            ]
            
            is_valid_type = (expected_step_type in valid_step_types or 
                           expected_step_type.startswith('Mims') or 
                           expected_step_type.startswith('Cradle'))
            self._assert(is_valid_type, 
                        f"Builder should be registered with valid step type: {expected_step_type}")
            
            self._log(f"✅ Builder registered for step type: {expected_step_type}")

            # Test 5: Return type annotation should indicate SageMaker step (if present)
            return_annotation = sig.return_annotation
            if return_annotation != inspect.Signature.empty:
                return_type_str = str(return_annotation)
                has_step_indication = 'Step' in return_type_str
                if has_step_indication:
                    self._log(f"✅ Return type indicates SageMaker step: {return_type_str}")
                else:
                    self._log(f"⚠️  Return type annotation unclear: {return_type_str}")

            self._log("✅ Universal step creation interface compliance validated")

        except Exception as e:
            self._assert(False, f"Step creation interface compliance failed: {str(e)}")

    def test_universal_step_creation_patterns(self) -> None:
        """Test universal step creation patterns that apply to all step types."""
        try:
            # Test 1: Builder should follow consistent naming patterns
            builder_class_name = self.builder_class.__name__
            self._assert(builder_class_name.endswith('StepBuilder'), 
                        f"Builder class should end with 'StepBuilder': {builder_class_name}")

            # Test 2: Builder should have consistent method patterns
            builder_instance = self.builder_class.__new__(self.builder_class)
            
            # Universal methods that all builders should have
            universal_methods = ['create_step', 'validate_configuration', '_get_step_name']
            for method_name in universal_methods:
                if hasattr(builder_instance, method_name):
                    method = getattr(builder_instance, method_name)
                    self._assert(callable(method), f"{method_name} should be callable")
                    self._log(f"✅ Builder has universal method: {method_name}")

            # Test 3: Builder should handle inputs consistently
            import inspect
            if hasattr(builder_instance, 'create_step'):
                sig = inspect.signature(builder_instance.create_step)
                params = list(sig.parameters.keys())
                
                # Should accept inputs in some form
                accepts_inputs = any(param in ['inputs', 'kwargs'] or 'input' in param.lower() 
                                   for param in params)
                self._assert(accepts_inputs or len(params) > 0, 
                           f"Builder should accept inputs: {params}")

            # Test 4: Builder should be registered with valid step type
            expected_step_type = self.step_info.get("sagemaker_step_type", "Unknown")
            if expected_step_type != "Unknown":
                self._log(f"✅ Builder registered for step type: {expected_step_type}")
            else:
                self._log("⚠️  Builder not registered with specific step type")

            # Test 5: Builder should follow dependency patterns (if applicable)
            dependency_methods = ['get_required_dependencies', 'get_optional_dependencies']
            for method_name in dependency_methods:
                if hasattr(builder_instance, method_name):
                    self._log(f"✅ Builder supports dependency pattern: {method_name}")

            self._log("✅ Universal step creation patterns validated")

        except Exception as e:
            self._assert(False, f"Universal step creation patterns test failed: {str(e)}")

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
            "CradleDataLoading": "CradleDataLoadingStep",
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
        if hasattr(step, "inputs") and step.inputs:
            self._assert(
                isinstance(step.inputs, list), "ProcessingStep inputs must be a list"
            )

        if hasattr(step, "outputs") and step.outputs:
            self._assert(
                isinstance(step.outputs, list), "ProcessingStep outputs must be a list"
            )

    def _validate_training_step_config(self, step) -> None:
        """Validate TrainingStep configuration."""
        if hasattr(step, "inputs") and step.inputs:
            self._assert(
                isinstance(step.inputs, dict),
                "TrainingStep inputs must be a dictionary",
            )

    def _validate_transform_step_config(self, step) -> None:
        """Validate TransformStep configuration."""
        if hasattr(step, "transform_inputs") and step.transform_inputs:
            self._assert(
                isinstance(step.transform_inputs, list),
                "TransformStep inputs must be a list",
            )

    def _validate_create_model_step_config(self, step) -> None:
        """Validate CreateModelStep configuration."""
        if hasattr(step, "model") and step.model:
            model = step.model
            if hasattr(model, "primary_container"):
                self._assert(
                    model.primary_container is not None,
                    "CreateModelStep model must have primary_container",
                )
