"""
Level 3 Path Mapping Tests for step builders.

These tests focus on path mapping and property path validation:
- Input path mapping correctness
- Output path mapping correctness
- Property path validity and resolution
"""

from typing import Dict, Any, List, Union
from .base_test import UniversalStepBuilderTestBase


class PathMappingTests(UniversalStepBuilderTestBase):
    """
    Level 3 tests focusing on path mapping validation.
    
    These tests validate that a step builder correctly maps input and output
    paths and handles property path resolution.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return step type-specific test methods for path mapping tests."""
        return []  # Path mapping tests are generic
    
    def _configure_step_type_mocks(self) -> None:
        """Configure step type-specific mock objects for path mapping tests."""
        pass  # Generic path mapping tests
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate step type-specific requirements for path mapping tests."""
        return {
            "path_mapping_tests_completed": True,
            "step_type_agnostic": True
        }
    
    def test_input_path_mapping(self) -> None:
        """Test that the builder correctly maps specification dependencies to script contract paths."""
        try:
            # Create instance with mock config
            builder = self._create_builder_instance()
            
            if not builder.spec or not builder.contract:
                self._log("Skipping input path mapping test - no spec or contract available")
                return
            
            # Create sample inputs dictionary based on specification dependencies
            inputs = {}
            for dep_name, dep_spec in builder.spec.dependencies.items():
                logical_name = dep_spec.logical_name
                inputs[logical_name] = f"s3://test-bucket/test/{logical_name}"
            
            # Get inputs from the builder
            step_inputs = builder._get_inputs(inputs)
            
            # Validate input structure based on step type
            step_type = self.step_info.get('sagemaker_step_type', 'Unknown')
            
            if step_type == "Processing":
                self._validate_processing_inputs(step_inputs, builder, inputs)
            elif step_type == "Training":
                self._validate_training_inputs(step_inputs, builder, inputs)
            elif step_type == "Transform":
                self._validate_transform_inputs(step_inputs, builder, inputs)
            else:
                self._log(f"Generic input validation for step type: {step_type}")
                self._assert(
                    step_inputs is not None,
                    "Step inputs must not be None"
                )
                
        except Exception as e:
            self._assert(
                False,
                f"Input path mapping test failed: {str(e)}"
            )
    
    def test_output_path_mapping(self) -> None:
        """Test that the builder correctly maps specification outputs to script contract paths."""
        try:
            # Create instance with mock config
            builder = self._create_builder_instance()
            
            if not builder.spec or not builder.contract:
                self._log("Skipping output path mapping test - no spec or contract available")
                return
            
            # Create sample outputs dictionary based on specification outputs
            outputs = {}
            for out_name, out_spec in builder.spec.outputs.items():
                logical_name = out_spec.logical_name
                outputs[logical_name] = f"s3://test-bucket/test/{logical_name}"
            
            # Get outputs from the builder
            step_outputs = builder._get_outputs(outputs)
            
            # Validate output structure based on step type
            step_type = self.step_info.get('sagemaker_step_type', 'Unknown')
            
            if step_type == "Processing":
                self._validate_processing_outputs(step_outputs, builder, outputs)
            elif step_type == "Training":
                self._validate_training_outputs(step_outputs, builder, outputs)
            elif step_type == "Transform":
                self._validate_transform_outputs(step_outputs, builder, outputs)
            else:
                self._log(f"Generic output validation for step type: {step_type}")
                self._assert(
                    step_outputs is not None,
                    "Step outputs must not be None"
                )
                
        except Exception as e:
            self._assert(
                False,
                f"Output path mapping test failed: {str(e)}"
            )
    
    def test_property_path_validity(self) -> None:
        """Test that the builder uses valid property paths."""
        try:
            # Create instance with mock config
            builder = self._create_builder_instance()
            
            if not builder.spec:
                self._log("Skipping property path validity test - no spec available")
                return
            
            # Check each output specification for valid property paths
            if hasattr(builder.spec, 'outputs'):
                for output_name, output_spec in builder.spec.outputs.items():
                    # Check that property path exists
                    self._assert(
                        hasattr(output_spec, 'property_path') and output_spec.property_path,
                        f"Output {output_name} must have a property_path"
                    )
                    
                    # Validate property path format
                    property_path = output_spec.property_path
                    self._validate_property_path_format(property_path, output_name)
                    
                    # Test property path resolution if possible
                    self._test_property_path_resolution(property_path, output_name, builder)
            else:
                self._log("No outputs found in specification")
                
        except Exception as e:
            self._assert(
                False,
                f"Property path validity test failed: {str(e)}"
            )
    
    def _validate_processing_inputs(self, step_inputs: List, builder, inputs: Dict[str, Any]) -> None:
        """Validate ProcessingInput objects."""
        self._assert(
            isinstance(step_inputs, list),
            "Processing step inputs must be a list"
        )
        
        for proc_input in step_inputs:
            # Check that this is a valid input object (ProcessingInput)
            self._assert(
                hasattr(proc_input, "source") or hasattr(proc_input, "s3_data"),
                f"Processing input must have source or s3_data attribute"
            )
            
            # Check that the input has an input_name attribute
            self._assert(
                hasattr(proc_input, "input_name"),
                f"Processing input must have input_name attribute"
            )
            
            # If it has a destination attribute, check that it matches a path in the contract
            if hasattr(proc_input, "destination"):
                destination = proc_input.destination
                self._assert(
                    any(path == destination for path in builder.contract.expected_input_paths.values()),
                    f"Input destination {destination} must match a path in the contract"
                )
    
    def _validate_training_inputs(self, step_inputs: Dict, builder, inputs: Dict[str, Any]) -> None:
        """Validate TrainingInput objects."""
        self._assert(
            isinstance(step_inputs, dict),
            "Training step inputs must be a dictionary"
        )
        
        for channel_name, training_input in step_inputs.items():
            # Check that this is a valid TrainingInput object
            self._assert(
                hasattr(training_input, "s3_data"),
                f"Training input for channel {channel_name} must have s3_data attribute"
            )
            
            # Validate channel name format
            self._assert(
                isinstance(channel_name, str) and len(channel_name) > 0,
                f"Channel name must be a non-empty string, got: {channel_name}"
            )
    
    def _validate_transform_inputs(self, step_inputs: List, builder, inputs: Dict[str, Any]) -> None:
        """Validate TransformInput objects."""
        self._assert(
            isinstance(step_inputs, list),
            "Transform step inputs must be a list"
        )
        
        for transform_input in step_inputs:
            # Check that this is a valid input object
            self._assert(
                hasattr(transform_input, "data_source") or hasattr(transform_input, "s3_data_source"),
                f"Transform input must have data_source or s3_data_source attribute"
            )
    
    def _validate_processing_outputs(self, step_outputs: List, builder, outputs: Dict[str, Any]) -> None:
        """Validate ProcessingOutput objects."""
        self._assert(
            isinstance(step_outputs, list),
            "Processing step outputs must be a list"
        )
        
        for proc_output in step_outputs:
            # Check that this is a valid output object
            self._assert(
                hasattr(proc_output, "source"),
                f"Processing output must have source attribute"
            )
            
            # Check that the output has an output_name attribute
            self._assert(
                hasattr(proc_output, "output_name"),
                f"Processing output must have output_name attribute"
            )
            
            # Check that the source attribute matches a path in the contract
            source = proc_output.source
            self._assert(
                any(path == source for path in builder.contract.expected_output_paths.values()),
                f"Output source {source} must match a path in the contract"
            )
            
            # Check that the destination attribute is set correctly
            self._assert(
                hasattr(proc_output, "destination"),
                f"Processing output must have destination attribute"
            )
    
    def _validate_training_outputs(self, step_outputs: str, builder, outputs: Dict[str, Any]) -> None:
        """Validate Training step output path."""
        self._assert(
            isinstance(step_outputs, str),
            "Training step outputs must be a string (output path)"
        )
        
        # Validate S3 URI format
        self._assert(
            step_outputs.startswith("s3://") or isinstance(step_outputs, dict),
            f"Training output path must be S3 URI or pipeline reference, got: {step_outputs}"
        )
    
    def _validate_transform_outputs(self, step_outputs: str, builder, outputs: Dict[str, Any]) -> None:
        """Validate Transform step output path."""
        self._assert(
            isinstance(step_outputs, str),
            "Transform step outputs must be a string (output path)"
        )
        
        # Validate S3 URI format
        self._assert(
            step_outputs.startswith("s3://") or isinstance(step_outputs, dict),
            f"Transform output path must be S3 URI or pipeline reference, got: {step_outputs}"
        )
    
    def _validate_property_path_format(self, property_path: str, output_name: str) -> None:
        """Validate property path format."""
        self._assert(
            isinstance(property_path, str) and len(property_path) > 0,
            f"Property path for output {output_name} must be a non-empty string"
        )
        
        # Basic format validation - should contain step reference patterns
        valid_patterns = [
            "Steps.",  # SageMaker step reference
            "Properties.",  # Property reference
            "ModelArtifacts",  # Model artifacts reference
            "ProcessingOutputConfig",  # Processing output reference
        ]
        
        has_valid_pattern = any(pattern in property_path for pattern in valid_patterns)
        if not has_valid_pattern:
            self._log(f"Warning: Property path '{property_path}' may not follow expected patterns")
    
    def _test_property_path_resolution(self, property_path: str, output_name: str, builder) -> None:
        """Test property path resolution if possible."""
        try:
            # Try to parse property path components
            if "Steps." in property_path:
                # Extract step name and property
                parts = property_path.split(".")
                if len(parts) >= 3:  # e.g., Steps.StepName.Property
                    step_name = parts[1]
                    property_name = ".".join(parts[2:])
                    
                    self._assert(
                        len(step_name) > 0,
                        f"Step name in property path must not be empty: {property_path}"
                    )
                    
                    self._assert(
                        len(property_name) > 0,
                        f"Property name in property path must not be empty: {property_path}"
                    )
                    
                    self._log(f"Property path '{property_path}' references step '{step_name}' property '{property_name}'")
                else:
                    self._log(f"Warning: Property path '{property_path}' has unexpected format")
            else:
                self._log(f"Property path '{property_path}' uses non-standard format")
                
        except Exception as e:
            self._log(f"Could not parse property path '{property_path}': {str(e)}")
