"""
Level 3 Path Mapping Tests for step builders.

These tests focus on the correct mapping of paths between specifications and contracts:
- Input path mapping from specification dependencies to contract paths
- Output path mapping from specification outputs to contract paths
- Property path validation for outputs
"""

from .base_test import UniversalStepBuilderTestBase


class PathMappingTests(UniversalStepBuilderTestBase):
    """
    Level 3 tests focusing on path mapping between specifications and contracts.
    
    These tests validate that a step builder correctly maps input and output paths
    between specifications and script contracts, and that property paths are valid.
    These tests require deeper understanding of the system architecture.
    """
    
    def test_input_path_mapping(self) -> None:
        """Test that the builder correctly maps specification dependencies to script contract paths."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Skip if no contract
        if not hasattr(builder, 'contract') or builder.contract is None:
            self._log("Contract is None, skipping input path mapping test")
            return
        
        # Skip if no specification
        if not hasattr(builder, 'spec') or builder.spec is None:
            self._log("Specification is None, skipping input path mapping test")
            return
            
        # Create sample inputs dictionary
        inputs = {}
        if hasattr(builder.spec, 'dependencies'):
            for dep_name, dep_spec in builder.spec.dependencies.items():
                logical_name = dep_spec.logical_name
                inputs[logical_name] = f"s3://bucket/test/{logical_name}"
        
        # Skip if no inputs
        if not inputs:
            self._log("No inputs defined in specification, skipping input path mapping test")
            return
            
        # Get inputs from the builder
        try:
            processing_inputs = builder._get_inputs(inputs)
            
            # Check that result is not None
            self._assert(
                processing_inputs is not None,
                "_get_inputs() must not return None"
            )
            
            # Skip further tests if result is empty
            if not processing_inputs:
                self._log("_get_inputs() returned empty list or dict, skipping further tests")
                return
                
            # Handle different return types (list or dict)
            if isinstance(processing_inputs, dict):
                items = list(processing_inputs.values())
            elif isinstance(processing_inputs, list):
                items = processing_inputs
            else:
                self._assert(
                    False,
                    f"_get_inputs() must return a list or dict, got {type(processing_inputs)}"
                )
                return
            
            # Check that each input has the correct structure
            for proc_input in items:
                # Check that this is a valid input object (ProcessingInput, TrainingInput, etc.)
                self._assert(
                    hasattr(proc_input, "source") or hasattr(proc_input, "s3_data"),
                    f"Processing input must have source or s3_data attribute"
                )
                
                # Check that the input has an input_name or channel_name attribute
                has_name = (hasattr(proc_input, "input_name") or
                           hasattr(proc_input, "channel_name"))
                self._assert(
                    has_name,
                    f"Processing input must have input_name or channel_name attribute"
                )
                
                # If it has a destination attribute, check that it matches a path in the contract
                if hasattr(proc_input, "destination"):
                    destination = proc_input.destination
                    self._assert(
                        any(path == destination for path in builder.contract.expected_input_paths.values()),
                        f"Input destination {destination} must match a path in the contract"
                    )
        except Exception as e:
            self._assert(
                False,
                f"Error getting inputs: {str(e)}"
            )
    
    def test_output_path_mapping(self) -> None:
        """Test that the builder correctly maps specification outputs to script contract paths."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Skip if no contract
        if not hasattr(builder, 'contract') or builder.contract is None:
            self._log("Contract is None, skipping output path mapping test")
            return
        
        # Skip if no specification
        if not hasattr(builder, 'spec') or builder.spec is None:
            self._log("Specification is None, skipping output path mapping test")
            return
            
        # Create sample outputs dictionary
        outputs = {}
        if hasattr(builder.spec, 'outputs'):
            for out_name, out_spec in builder.spec.outputs.items():
                logical_name = out_spec.logical_name
                outputs[logical_name] = f"s3://bucket/test/{logical_name}"
        
        # Skip if no outputs
        if not outputs and not hasattr(builder.spec, 'outputs'):
            self._log("No outputs defined in specification, skipping output path mapping test")
            return
            
        # Get outputs from the builder
        try:
            processing_outputs = builder._get_outputs(outputs)
            
            # Handle different return types
            if isinstance(processing_outputs, str):
                # For training steps, _get_outputs may return a string path
                self._log("_get_outputs() returned a string path, skipping further tests")
                return
            
            # Check that result is not None
            self._assert(
                processing_outputs is not None,
                "_get_outputs() must not return None"
            )
            
            # Skip further tests if result is empty
            if not processing_outputs:
                self._log("_get_outputs() returned empty list or dict, skipping further tests")
                return
                
            # Handle different return types (list or dict)
            if isinstance(processing_outputs, dict):
                items = list(processing_outputs.values())
            elif isinstance(processing_outputs, list):
                items = processing_outputs
            else:
                self._assert(
                    False,
                    f"_get_outputs() must return a list, dict, or string, got {type(processing_outputs)}"
                )
                return
            
            # Check that each output has the correct structure
            for proc_output in items:
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
        except Exception as e:
            self._assert(
                False,
                f"Error getting outputs: {str(e)}"
            )
    
    def test_property_path_validity(self) -> None:
        """Test that output specification property paths are valid."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Skip if no specification
        if not hasattr(builder, 'spec') or builder.spec is None:
            self._log("Specification is None, skipping property path validity test")
            return
            
        # Skip if no outputs in specification
        if not hasattr(builder.spec, 'outputs') or not builder.spec.outputs:
            self._log("No outputs in specification, skipping property path validity test")
            return
            
        # Try to import PropertyReference
        try:
            from src.pipeline_deps.property_reference import PropertyReference
        except ImportError:
            self._log("PropertyReference not available, skipping property path validity test")
            return
            
        # Check each output specification
        for output_name, output_spec in builder.spec.outputs.items():
            # Check that property path exists
            self._assert(
                hasattr(output_spec, 'property_path') and output_spec.property_path,
                f"Output {output_name} must have a property_path"
            )
            
            # Create dummy property reference
            try:
                prop_ref = PropertyReference(
                    step_name="TestStep",
                    output_spec=output_spec
                )
                
                # Attempt to parse property path
                path_parts = prop_ref._parse_property_path(output_spec.property_path)
                self._assert(
                    isinstance(path_parts, list) and len(path_parts) > 0,
                    f"Property path '{output_spec.property_path}' must be parseable"
                )
                
                # Check that the path can be converted to SageMaker property
                sagemaker_prop = prop_ref.to_sagemaker_property()
                self._assert(
                    isinstance(sagemaker_prop, dict) and "Get" in sagemaker_prop,
                    f"Property path must be convertible to SageMaker property"
                )
            except Exception as e:
                self._assert(
                    False,
                    f"Error processing property path '{output_spec.property_path}': {str(e)}"
                )
