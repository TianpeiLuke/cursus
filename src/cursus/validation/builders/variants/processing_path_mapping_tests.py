"""
Level 3 Processing-Specific Path Mapping Tests for step builders.

These tests focus on Processing step path mapping and property path validation:
- ProcessingInput/ProcessingOutput object creation
- Container path mapping from contracts
- S3 path validation and normalization
- Special input handling patterns (local paths, file uploads)
- Property path validity for Processing outputs
"""

import os
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..path_mapping_tests import PathMappingTests


class ProcessingPathMappingTests(PathMappingTests):
    """
    Level 3 Processing-specific path mapping tests.
    
    These tests validate that Processing step builders correctly map paths
    between specifications, contracts, and SageMaker ProcessingInput/ProcessingOutput objects.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return Processing-specific path mapping test methods."""
        return [
            "test_processing_input_object_creation",
            "test_processing_output_object_creation",
            "test_container_path_mapping",
            "test_s3_path_validation",
            "test_local_path_override_handling",
            "test_file_upload_path_handling",
            "test_processing_property_paths",
            "test_input_output_path_consistency",
            "test_optional_input_handling"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing-specific mock objects for path mapping tests."""
        # Mock ProcessingInput and ProcessingOutput classes
        self.mock_processing_input_class = Mock()
        self.mock_processing_output_class = Mock()
        
        # Mock ProcessingInput instances
        self.mock_processing_input = Mock()
        self.mock_processing_input.input_name = "test_input"
        self.mock_processing_input.source = "s3://bucket/input/data"
        self.mock_processing_input.destination = "/opt/ml/processing/input/data"
        
        # Mock ProcessingOutput instances
        self.mock_processing_output = Mock()
        self.mock_processing_output.output_name = "test_output"
        self.mock_processing_output.source = "/opt/ml/processing/output/data"
        self.mock_processing_output.destination = "s3://bucket/output/data"
        
        # Mock specification with Processing-specific structure
        self.mock_processing_spec = Mock()
        self.mock_processing_spec.dependencies = {
            "input_data": Mock(logical_name="input_data", required=True),
            "metadata": Mock(logical_name="metadata", required=False),
            "inference_scripts": Mock(logical_name="inference_scripts_input", required=True)
        }
        self.mock_processing_spec.outputs = {
            "processed_data": Mock(
                logical_name="processed_data",
                property_path="Steps.ProcessingStep.ProcessingOutputConfig.Outputs['processed_data']"
            ),
            "statistics": Mock(
                logical_name="statistics", 
                property_path="Steps.ProcessingStep.ProcessingOutputConfig.Outputs['statistics']"
            )
        }
        
        # Mock contract with Processing container paths
        self.mock_contract = Mock()
        self.mock_contract.expected_input_paths = {
            "input_data": "/opt/ml/processing/input/data",
            "metadata": "/opt/ml/processing/input/metadata",
            "inference_scripts_input": "/opt/ml/processing/input/code"
        }
        self.mock_contract.expected_output_paths = {
            "processed_data": "/opt/ml/processing/output/data",
            "statistics": "/opt/ml/processing/output/stats"
        }
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate Processing-specific requirements for path mapping tests."""
        return {
            "path_mapping_tests_completed": True,
            "processing_specific_validations": True,
            "container_path_mapping_validated": True,
            "s3_path_validation_completed": True
        }
    
    def test_processing_input_object_creation(self) -> None:
        """Test that Processing builders create valid ProcessingInput objects."""
        self._log("Testing ProcessingInput object creation")
        
        if hasattr(self.builder_class, '_get_inputs'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            # Mock specification and contract
            builder.spec = self.mock_processing_spec
            builder.contract = self.mock_contract
            
            inputs = {
                "input_data": "s3://bucket/input/data",
                "metadata": "s3://bucket/input/metadata"
            }
            
            try:
                processing_inputs = builder._get_inputs(inputs)
                
                # Validate ProcessingInput objects
                self._assert(isinstance(processing_inputs, list), "Should return list of ProcessingInput objects")
                
                for proc_input in processing_inputs:
                    # Check ProcessingInput structure
                    self._assert(hasattr(proc_input, 'input_name'), "ProcessingInput should have input_name")
                    self._assert(hasattr(proc_input, 'source'), "ProcessingInput should have source")
                    self._assert(hasattr(proc_input, 'destination'), "ProcessingInput should have destination")
                    
                    # Validate input_name is string
                    self._assert(isinstance(proc_input.input_name, str), "input_name should be string")
                    
                    # Validate source is S3 URI or local path
                    source = proc_input.source
                    self._assert(
                        isinstance(source, str) and (source.startswith("s3://") or os.path.isabs(source)),
                        f"Source should be S3 URI or absolute path, got: {source}"
                    )
                    
                    # Validate destination is container path
                    destination = proc_input.destination
                    self._assert(
                        isinstance(destination, str) and destination.startswith("/opt/ml/"),
                        f"Destination should be container path, got: {destination}"
                    )
                
                self._assert(True, "ProcessingInput object creation validated")
                
            except Exception as e:
                self._log(f"ProcessingInput creation test failed: {e}")
                self._assert(False, f"ProcessingInput creation test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Processing builders should have _get_inputs method")
    
    def test_processing_output_object_creation(self) -> None:
        """Test that Processing builders create valid ProcessingOutput objects."""
        self._log("Testing ProcessingOutput object creation")
        
        if hasattr(self.builder_class, '_get_outputs'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            # Mock specification and contract
            builder.spec = self.mock_processing_spec
            builder.contract = self.mock_contract
            
            outputs = {
                "processed_data": "s3://bucket/output/data",
                "statistics": "s3://bucket/output/stats"
            }
            
            try:
                processing_outputs = builder._get_outputs(outputs)
                
                # Validate ProcessingOutput objects
                self._assert(isinstance(processing_outputs, list), "Should return list of ProcessingOutput objects")
                
                for proc_output in processing_outputs:
                    # Check ProcessingOutput structure
                    self._assert(hasattr(proc_output, 'output_name'), "ProcessingOutput should have output_name")
                    self._assert(hasattr(proc_output, 'source'), "ProcessingOutput should have source")
                    self._assert(hasattr(proc_output, 'destination'), "ProcessingOutput should have destination")
                    
                    # Validate output_name is string
                    self._assert(isinstance(proc_output.output_name, str), "output_name should be string")
                    
                    # Validate source is container path
                    source = proc_output.source
                    self._assert(
                        isinstance(source, str) and source.startswith("/opt/ml/"),
                        f"Source should be container path, got: {source}"
                    )
                    
                    # Validate destination is S3 URI
                    destination = proc_output.destination
                    self._assert(
                        isinstance(destination, str) and destination.startswith("s3://"),
                        f"Destination should be S3 URI, got: {destination}"
                    )
                
                self._assert(True, "ProcessingOutput object creation validated")
                
            except Exception as e:
                self._log(f"ProcessingOutput creation test failed: {e}")
                self._assert(False, f"ProcessingOutput creation test failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(False, "Processing builders should have _get_outputs method")
    
    def test_container_path_mapping(self) -> None:
        """Test that Processing builders correctly map logical names to container paths."""
        self._log("Testing container path mapping from contracts")
        
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock specification and contract
        builder.spec = self.mock_processing_spec
        builder.contract = self.mock_contract
        
        try:
            # Test input path mapping
            if hasattr(builder, '_get_inputs'):
                inputs = {"input_data": "s3://bucket/input/data"}
                processing_inputs = builder._get_inputs(inputs)
                
                if processing_inputs:
                    # Find the input_data ProcessingInput
                    input_data_proc = next(
                        (pi for pi in processing_inputs if pi.input_name == "input_data"), 
                        None
                    )
                    if input_data_proc:
                        expected_path = self.mock_contract.expected_input_paths["input_data"]
                        self._assert(
                            input_data_proc.destination == expected_path,
                            f"Input destination should match contract path: {expected_path}"
                        )
            
            # Test output path mapping
            if hasattr(builder, '_get_outputs'):
                outputs = {"processed_data": "s3://bucket/output/data"}
                processing_outputs = builder._get_outputs(outputs)
                
                if processing_outputs:
                    # Find the processed_data ProcessingOutput
                    output_data_proc = next(
                        (po for po in processing_outputs if po.output_name == "processed_data"), 
                        None
                    )
                    if output_data_proc:
                        expected_path = self.mock_contract.expected_output_paths["processed_data"]
                        self._assert(
                            output_data_proc.source == expected_path,
                            f"Output source should match contract path: {expected_path}"
                        )
            
            self._assert(True, "Container path mapping validated")
            
        except Exception as e:
            self._log(f"Container path mapping test failed: {e}")
            self._assert(False, f"Container path mapping test failed: {e}")
    
    def test_s3_path_validation(self) -> None:
        """Test that Processing builders validate and normalize S3 paths."""
        self._log("Testing S3 path validation and normalization")
        
        if hasattr(self.builder_class, '_normalize_s3_uri') or hasattr(self.builder_class, '_validate_s3_uri'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            # Test S3 URI normalization
            test_uris = [
                "s3://bucket/path/to/data",
                "s3://bucket/path/to/data/",  # Trailing slash
                "s3://bucket//path//to//data",  # Double slashes
            ]
            
            for uri in test_uris:
                try:
                    if hasattr(builder, '_normalize_s3_uri'):
                        normalized = builder._normalize_s3_uri(uri)
                        self._assert(
                            normalized.startswith("s3://"),
                            f"Normalized URI should start with s3://, got: {normalized}"
                        )
                    
                    if hasattr(builder, '_validate_s3_uri'):
                        is_valid = builder._validate_s3_uri(uri)
                        self._assert(is_valid, f"URI should be valid: {uri}")
                        
                except Exception as e:
                    self._log(f"S3 path validation failed for {uri}: {e}")
                    self._assert(False, f"S3 path validation failed for {uri}: {e}")
            
            # Test invalid URIs
            invalid_uris = [
                "http://bucket/path",  # Wrong protocol
                "/local/path",  # Local path
                "bucket/path",  # Missing protocol
            ]
            
            for uri in invalid_uris:
                try:
                    if hasattr(builder, '_validate_s3_uri'):
                        is_valid = builder._validate_s3_uri(uri)
                        self._assert(not is_valid, f"URI should be invalid: {uri}")
                        
                except Exception as e:
                    # Expected for invalid URIs
                    self._log(f"Expected validation failure for invalid URI {uri}: {e}")
            
            self._assert(True, "S3 path validation completed")
        else:
            self._log("No S3 path validation methods found - skipping")
            self._assert(True, "S3 path validation methods not required")
    
    def test_local_path_override_handling(self) -> None:
        """Test that Processing builders handle local path overrides (Package step pattern)."""
        self._log("Testing local path override handling")
        
        # This test is specific to builders that support local path overrides
        if hasattr(self.builder_class, '_get_inputs'):
            config = Mock()
            config.source_dir = "inference"  # Package step pattern
            builder = self.builder_class(config=config)
            
            # Mock specification and contract
            builder.spec = self.mock_processing_spec
            builder.contract = self.mock_contract
            
            inputs = {
                "input_data": "s3://bucket/input/data",
                "inference_scripts_input": "s3://bucket/scripts"  # This might be overridden
            }
            
            try:
                processing_inputs = builder._get_inputs(inputs)
                
                # Check if local path override is applied
                if processing_inputs:
                    inference_input = next(
                        (pi for pi in processing_inputs if pi.input_name == "inference_scripts_input"), 
                        None
                    )
                    
                    if inference_input:
                        # Local path override should use local directory
                        if hasattr(config, 'source_dir') and config.source_dir:
                            self._assert(
                                not inference_input.source.startswith("s3://"),
                                "Local path override should not use S3 URI"
                            )
                            self._log(f"Local path override detected: {inference_input.source}")
                        else:
                            self._log("No local path override configuration found")
                
                self._assert(True, "Local path override handling validated")
                
            except Exception as e:
                self._log(f"Local path override test failed: {e}")
                self._assert(False, f"Local path override test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Processing builders should have _get_inputs method")
    
    def test_file_upload_path_handling(self) -> None:
        """Test that Processing builders handle file upload patterns (DummyTraining step pattern)."""
        self._log("Testing file upload path handling")
        
        # This test is specific to builders that upload files to S3
        if hasattr(self.builder_class, '_upload_model_to_s3') or hasattr(self.builder_class, '_prepare_hyperparameters_file'):
            config = Mock()
            config.pretrained_model_path = "/local/model.tar.gz"
            config.pipeline_s3_loc = "s3://bucket/pipeline"
            builder = self.builder_class(config=config)
            builder.session = Mock()
            
            try:
                # Test model upload
                if hasattr(builder, '_upload_model_to_s3'):
                    with patch('cursus.steps.builders.S3Uploader') as mock_uploader:
                        mock_uploader.upload.return_value = None
                        
                        s3_uri = builder._upload_model_to_s3()
                        
                        self._assert(
                            s3_uri.startswith("s3://"),
                            f"Uploaded model URI should be S3 URI, got: {s3_uri}"
                        )
                        
                        # Verify upload was called
                        mock_uploader.upload.assert_called_once()
                
                # Test hyperparameters file preparation
                if hasattr(builder, '_prepare_hyperparameters_file'):
                    config.hyperparameters = Mock()
                    config.hyperparameters.model_dump.return_value = {"param1": "value1"}
                    
                    with patch('cursus.steps.builders.S3Uploader') as mock_uploader:
                        mock_uploader.upload.return_value = None
                        
                        s3_uri = builder._prepare_hyperparameters_file()
                        
                        self._assert(
                            s3_uri.startswith("s3://"),
                            f"Uploaded hyperparameters URI should be S3 URI, got: {s3_uri}"
                        )
                
                self._assert(True, "File upload path handling validated")
                
            except Exception as e:
                self._log(f"File upload path handling test failed: {e}")
                self._assert(False, f"File upload path handling test failed: {e}")
        else:
            self._log("No file upload methods found - skipping")
            self._assert(True, "File upload methods not required for this builder")
    
    def test_processing_property_paths(self) -> None:
        """Test that Processing builders use valid property paths for outputs."""
        self._log("Testing Processing property path validation")
        
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock specification with property paths
        builder.spec = self.mock_processing_spec
        
        try:
            # Validate property paths in specification
            if hasattr(builder.spec, 'outputs'):
                for output_name, output_spec in builder.spec.outputs.items():
                    if hasattr(output_spec, 'property_path'):
                        property_path = output_spec.property_path
                        
                        # Validate Processing-specific property path patterns
                        self._assert(
                            "ProcessingOutputConfig" in property_path,
                            f"Processing property path should contain 'ProcessingOutputConfig': {property_path}"
                        )
                        
                        self._assert(
                            "Outputs[" in property_path,
                            f"Processing property path should reference Outputs: {property_path}"
                        )
                        
                        # Validate step reference
                        self._assert(
                            property_path.startswith("Steps."),
                            f"Property path should start with 'Steps.': {property_path}"
                        )
                        
                        self._log(f"Valid Processing property path: {property_path}")
            
            self._assert(True, "Processing property paths validated")
            
        except Exception as e:
            self._log(f"Processing property path test failed: {e}")
            self._assert(False, f"Processing property path test failed: {e}")
    
    def test_input_output_path_consistency(self) -> None:
        """Test that input and output paths are consistent with specification and contract."""
        self._log("Testing input/output path consistency")
        
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock specification and contract
        builder.spec = self.mock_processing_spec
        builder.contract = self.mock_contract
        
        try:
            # Test input consistency
            if hasattr(builder, '_get_inputs'):
                inputs = {"input_data": "s3://bucket/input/data"}
                processing_inputs = builder._get_inputs(inputs)
                
                # Check that all required inputs are present
                required_inputs = [
                    dep.logical_name for dep in builder.spec.dependencies.values() 
                    if dep.required
                ]
                
                if processing_inputs and required_inputs:
                    input_names = [pi.input_name for pi in processing_inputs]
                    for required_input in required_inputs:
                        if required_input in inputs:  # Only check if input was provided
                            self._assert(
                                required_input in input_names,
                                f"Required input {required_input} should be in ProcessingInputs"
                            )
            
            # Test output consistency
            if hasattr(builder, '_get_outputs'):
                outputs = {"processed_data": "s3://bucket/output/data"}
                processing_outputs = builder._get_outputs(outputs)
                
                # Check that all specified outputs are present
                if processing_outputs and hasattr(builder.spec, 'outputs'):
                    output_names = [po.output_name for po in processing_outputs]
                    for spec_output in builder.spec.outputs.keys():
                        if spec_output in outputs:  # Only check if output was provided
                            self._assert(
                                spec_output in output_names,
                                f"Specified output {spec_output} should be in ProcessingOutputs"
                            )
            
            self._assert(True, "Input/output path consistency validated")
            
        except Exception as e:
            self._log(f"Path consistency test failed: {e}")
            self._assert(False, f"Path consistency test failed: {e}")
    
    def test_optional_input_handling(self) -> None:
        """Test that Processing builders correctly handle optional inputs."""
        self._log("Testing optional input handling")
        
        if hasattr(self.builder_class, '_get_inputs'):
            config = Mock()
            builder = self.builder_class(config=config)
            
            # Mock specification and contract
            builder.spec = self.mock_processing_spec
            builder.contract = self.mock_contract
            
            # Provide only required inputs, skip optional ones
            inputs = {
                "input_data": "s3://bucket/input/data"
                # Skip "metadata" which is optional
            }
            
            try:
                processing_inputs = builder._get_inputs(inputs)
                
                # Should not fail with missing optional inputs
                self._assert(isinstance(processing_inputs, list), "Should handle optional inputs gracefully")
                
                # Check that optional inputs are not included when not provided
                if processing_inputs:
                    input_names = [pi.input_name for pi in processing_inputs]
                    self._assert(
                        "metadata" not in input_names,
                        "Optional input should not be included when not provided"
                    )
                    
                    # Required inputs should still be present
                    self._assert(
                        "input_data" in input_names,
                        "Required input should be present"
                    )
                
                self._assert(True, "Optional input handling validated")
                
            except Exception as e:
                self._log(f"Optional input handling test failed: {e}")
                self._assert(False, f"Optional input handling test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Processing builders should have _get_inputs method")
