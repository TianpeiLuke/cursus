"""
Level 4 Integration Tests for step builders.

These tests focus on the integration aspects:
- Dependency resolution
- Step creation with proper attributes
"""

from .base_test import UniversalStepBuilderTestBase


class IntegrationTests(UniversalStepBuilderTestBase):
    """
    Level 4 tests focusing on system integration.
    
    These tests validate that a step builder correctly integrates with
    the wider system, including dependency resolution and step creation.
    These tests require the deepest understanding of the system architecture.
    """
    
    def test_dependency_resolution(self) -> None:
        """Test that the builder resolves dependencies correctly."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Create mock dependencies
        dependencies = self._create_mock_dependencies()
        
        # Test extraction of inputs from dependencies
        try:
            extracted_inputs = builder.extract_inputs_from_dependencies(dependencies)
            
            self._assert(
                isinstance(extracted_inputs, dict),
                f"extract_inputs_from_dependencies() must return a dictionary"
            )
            
            # Verify extracted inputs include required dependencies
            try:
                required_deps = builder.get_required_dependencies()
                for dep_name in required_deps:
                    self._assert(
                        dep_name in extracted_inputs,
                        f"Extracted inputs must include required dependency {dep_name}"
                    )
            except Exception as e:
                self._log(f"Could not get required dependencies: {str(e)}")
        except Exception as e:
            self._assert(
                False,
                f"Dependency resolution failed: {str(e)}"
            )
    
    def test_step_creation(self) -> None:
        """Test that the builder creates a valid step."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Create mock dependencies
        dependencies = self._create_mock_dependencies()
        
        # Test step creation
        try:
            step = builder.create_step(
                dependencies=dependencies,
                enable_caching=True
            )
            
            # Verify step has required attributes
            self._assert(
                step is not None,
                "Step must be created successfully"
            )
            
            # Verify step has spec attached
            self._assert(
                hasattr(step, '_spec'),
                "Step must have _spec attribute"
            )
            
            # Verify step has name
            self._assert(
                hasattr(step, 'name'),
                "Step must have name attribute"
            )
            
            # Verify step is a SageMaker Step
            try:
                from sagemaker.workflow.steps import Step as SageMakerStep
                self._assert(
                    isinstance(step, SageMakerStep),
                    "Step must be a SageMaker Step instance"
                )
                
                # Verify step has correct name derived from step name registry
                if self._provided_step_name:
                    expected_name = self._provided_step_name
                    self._assert(
                        step.name == expected_name,
                        f"Step name must match provided name: {expected_name}"
                    )
                elif hasattr(builder, '_get_step_name'):
                    expected_name = builder._get_step_name()
                    self._assert(
                        step.name == expected_name,
                        f"Step name must match expected name from registry: {expected_name}"
                    )
            except ImportError:
                self._log("SageMaker Step class not available, skipping type check")
        except Exception as e:
            self._assert(
                False,
                f"Step creation failed: {str(e)}"
            )
            
    def test_step_name(self) -> None:
        """Test that the builder correctly generates step name."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Test the _get_step_name method
        self._assert(
            hasattr(builder, '_get_step_name'),
            "Builder must have _get_step_name method"
        )
        
        # Get step name
        step_name = builder._get_step_name()
        
        # Check that step name is non-empty
        self._assert(
            step_name and isinstance(step_name, str) and len(step_name) > 0,
            f"Step name must be a non-empty string, got {step_name}"
        )
        
        # Check that step name follows the convention (extracted from class name)
        class_name = builder.__class__.__name__
        if class_name.endswith("StepBuilder"):
            canonical_name = class_name[:-11]  # Remove "StepBuilder" suffix
            
            # Check if job_type is included when available
            if hasattr(builder.config, 'job_type') and builder.config.job_type:
                expected_name = f"{canonical_name}-{builder.config.job_type.capitalize()}"
                self._assert(
                    step_name == expected_name,
                    f"Step name should be '{expected_name}' for class '{class_name}' with job_type '{builder.config.job_type}', got '{step_name}'"
                )
            else:
                self._assert(
                    step_name == canonical_name,
                    f"Step name should be '{canonical_name}' for class '{class_name}', got '{step_name}'"
                )
                
        # Test that the step name is used in create_step
        try:
            # Create mock dependencies
            dependencies = self._create_mock_dependencies()
            
            # Create a step
            step = builder.create_step(
                dependencies=dependencies,
                enable_caching=True
            )
            
            # Verify step has the correct name
            self._assert(
                step.name == step_name,
                f"Step name mismatch: step.name='{step.name}', expected='{step_name}'"
            )
        except Exception as e:
            self._assert(
                False,
                f"Step creation failed during name test: {str(e)}"
            )
