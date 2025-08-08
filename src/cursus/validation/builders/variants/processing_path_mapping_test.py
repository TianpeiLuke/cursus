"""
Level 3 Processing Path Mapping Tests - Enhanced Processing-Specific Path Mapping Validation.

These tests focus on Processing step-specific path mapping and validation:
- ProcessingInput/ProcessingOutput object creation and validation
- Container path mapping from script contracts
- Special input handling patterns (local path override, file upload)
- S3 path normalization and validation
"""

from typing import Dict, List, Any
from ..path_mapping_tests import PathMappingTests


class ProcessingPathMappingTests(PathMappingTests):
    """
    Enhanced Level 3 path mapping tests for Processing step builders.
    
    Extends the generic path mapping tests with Processing-specific validations
    based on the patterns identified in Processing Step Builder Patterns analysis.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Processing-specific path mapping test methods."""
        return [
            "test_processing_input_creation",
            "test_processing_output_creation",
            "test_container_path_mapping",
            "test_special_input_handling",
            "test_s3_path_normalization",
            "test_file_upload_patterns",
            "test_local_path_override_patterns",
            "test_dependency_input_extraction"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing step-specific mock objects."""
        # Get Processing-specific mocks from factory
        self.step_type_mocks = self.mock_factory.create_step_type_mocks()
        
        # Log Processing step info if verbose
        if self.verbose:
            framework = self.step_info.get('framework', 'Unknown')
            special_patterns = self.step_info.get('special_input_patterns', [])
            self._log(f"Processing path mapping tests - Framework: {framework}, Special patterns: {special_patterns}")
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate Processing step-specific requirements."""
        validation_results = {
            "is_processing_step": self.step_info.get("sagemaker_step_type") == "Processing",
            "has_specification": self.step_info.get("has_specification", False),
            "has_contract": self.step_info.get("has_contract", False),
            "supports_special_patterns": len(self.step_info.get("special_input_patterns", [])) > 0
        }
        
        return validation_results
    
    # Processing-specific path mapping tests
    
    def test_processing_input_creation(self) -> None:
        """Test ProcessingInput object creation and validation."""
        self._log("Testing ProcessingInput creation...")
        
        try:
            builder = self._create_builder_instance()
            
            if not builder.spec or not builder.contract:
                self._log("No specification or contract available, skipping test")
                return
            
            # Create test inputs based on specification
            test_inputs = {}
            for dep_name, dep_spec in builder.spec.dependencies.items():
                logical_name = dep_spec.logical_name
                test_inputs[logical_name] = f"s3://test-bucket/input/{logical_name}"
            
            # Get ProcessingInput objects
            processing_inputs = builder._get_inputs(test_inputs)
            
            self._assert(
                isinstance(processing_inputs, list),
                "Processing inputs must be a list"
            )
            
            # Validate each ProcessingInput object
            for proc_input in processing_inputs:
                self._validate_processing_input_object(proc_input, builder)
                
        except Exception as e:
            self._assert(False, f"ProcessingInput creation test failed: {str(e)}")
    
    def test_processing_output_creation(self) -> None:
        """Test ProcessingOutput object creation and validation."""
        self._log("Testing ProcessingOutput creation...")
        
        try:
            builder = self._create_builder_instance()
            
            if not builder.spec or not builder.contract:
                self._log("No specification or contract available, skipping test")
                return
            
            # Create test outputs based on specification
            test_outputs = {}
            for out_name, out_spec in builder.spec.outputs.items():
                logical_name = out_spec.logical_name
                test_outputs[logical_name] = f"s3://test-bucket/output/{logical_name}"
            
            # Get ProcessingOutput objects
            processing_outputs = builder._get_outputs(test_outputs)
            
            self._assert(
                isinstance(processing_outputs, list),
                "Processing outputs must be a list"
            )
            
            # Validate each ProcessingOutput object
            for proc_output in processing_outputs:
                self._validate_processing_output_object(proc_output, builder)
                
        except Exception as e:
            self._assert(False, f"ProcessingOutput creation test failed: {str(e)}")
    
    def test_container_path_mapping(self) -> None:
        """Test container path mapping from script contracts."""
        self._log("Testing container path mapping...")
        
        try:
            builder = self._create_builder_instance()
            
            if not builder.contract:
                self._log("No contract available, skipping test")
                return
            
            contract = builder.contract
            
            # Test input path mapping
            if hasattr(contract, 'expected_input_paths'):
                input_paths = contract.expected_input_paths
                self._validate_container_paths(input_paths, "input")
                
                # Test that ProcessingInputs use these paths
                if builder.spec:
                    test_inputs = {}
                    for dep_name, dep_spec in builder.spec.dependencies.items():
                        logical_name = dep_spec.logical_name
                        test_inputs[logical_name] = f"s3://test-bucket/{logical_name}"
                    
                    processing_inputs = builder._get_inputs(test_inputs)
                    self._validate_input_path_usage(processing_inputs, input_paths)
            
            # Test output path mapping
            if hasattr(contract, 'expected_output_paths'):
                output_paths = contract.expected_output_paths
                self._validate_container_paths(output_paths, "output")
                
                # Test that ProcessingOutputs use these paths
                if builder.spec:
                    test_outputs = {}
                    for out_name, out_spec in builder.spec.outputs.items():
                        logical_name = out_spec.logical_name
                        test_outputs[logical_name] = f"s3://test-bucket/{logical_name}"
                    
                    processing_outputs = builder._get_outputs(test_outputs)
                    self._validate_output_path_usage(processing_outputs, output_paths)
                    
        except Exception as e:
            self._assert(False, f"Container path mapping test failed: {str(e)}")
    
    def test_special_input_handling(self) -> None:
        """Test special input handling patterns."""
        self._log("Testing special input handling...")
        
        special_patterns = self.step_info.get('special_input_patterns', [])
        if not special_patterns:
            self._log("No special input patterns detected, skipping test")
            return
        
        try:
            builder = self._create_builder_instance()
            
            for pattern in special_patterns:
                if pattern == 'local_path_override':
                    self._test_local_path_override(builder)
                elif pattern == 'file_upload':
                    self._test_file_upload_pattern(builder)
                elif pattern == 's3_path_handling':
                    self._test_s3_path_handling(builder)
                else:
                    self._log(f"Unknown special pattern: {pattern}")
                    
        except Exception as e:
            self._assert(False, f"Special input handling test failed: {str(e)}")
    
    def test_s3_path_normalization(self) -> None:
        """Test S3 path normalization and validation."""
        self._log("Testing S3 path normalization...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test S3 path normalization methods if available
            if hasattr(builder, '_normalize_s3_uri'):
                # Test various S3 URI formats
                test_uris = [
                    "s3://bucket/path/file.txt",
                    "s3://bucket/path/",
                    "s3://bucket-name/deep/nested/path/file.json"
                ]
                
                for uri in test_uris:
                    normalized = builder._normalize_s3_uri(uri)
                    self._assert(
                        isinstance(normalized, str),
                        f"Normalized S3 URI must be string: {normalized}"
                    )
                    self._log(f"Normalized {uri} -> {normalized}")
            
            # Test S3 path validation methods if available
            if hasattr(builder, '_validate_s3_uri'):
                valid_uris = [
                    "s3://valid-bucket/path/file.txt",
                    "s3://another-bucket/data/"
                ]
                
                for uri in valid_uris:
                    is_valid = builder._validate_s3_uri(uri)
                    self._assert(
                        isinstance(is_valid, bool),
                        f"S3 URI validation must return boolean: {is_valid}"
                    )
                    
        except Exception as e:
            self._assert(False, f"S3 path normalization test failed: {str(e)}")
    
    def test_file_upload_patterns(self) -> None:
        """Test file upload patterns (DummyTraining step pattern)."""
        self._log("Testing file upload patterns...")
        
        builder_name = self.builder_class.__name__
        if "DummyTraining" not in builder_name:
            self._log("Not a DummyTraining step, skipping file upload pattern test")
            return
        
        try:
            builder = self._create_builder_instance()
            
            # Test file upload methods if available
            if hasattr(builder, '_upload_model_to_s3'):
                self._log("Found model upload method")
                # Note: We can't actually test upload without real files
                # But we can validate the method exists and has proper signature
            
            if hasattr(builder, '_prepare_hyperparameters_file'):
                self._log("Found hyperparameters preparation method")
                # Note: We can't actually test without real config
                # But we can validate the method exists
            
            # Test that inputs handle file uploads
            test_inputs = {}
            processing_inputs = builder._get_inputs(test_inputs)
            
            # DummyTraining should create inputs even without explicit inputs
            # (it uploads files automatically)
            self._assert(
                isinstance(processing_inputs, list),
                "DummyTraining should create ProcessingInputs for uploaded files"
            )
            
        except Exception as e:
            self._assert(False, f"File upload patterns test failed: {str(e)}")
    
    def test_local_path_override_patterns(self) -> None:
        """Test local path override patterns (Package step pattern)."""
        self._log("Testing local path override patterns...")
        
        builder_name = self.builder_class.__name__
        if "Package" not in builder_name:
            self._log("Not a Package step, skipping local path override pattern test")
            return
        
        try:
            builder = self._create_builder_instance()
            
            # Test local path override behavior
            # Package step should override dependency inputs with local paths
            test_inputs = {
                "inference_scripts_input": "s3://bucket/dependency-provided-path"
            }
            
            processing_inputs = builder._get_inputs(test_inputs)
            
            # Should have ProcessingInput for inference scripts
            inference_input = None
            for proc_input in processing_inputs:
                if hasattr(proc_input, 'input_name') and proc_input.input_name == "inference_scripts_input":
                    inference_input = proc_input
                    break
            
            if inference_input:
                # Should use local path, not the dependency-provided S3 path
                source = getattr(inference_input, 'source', None)
                if source:
                    self._assert(
                        not source.startswith('s3://bucket/dependency-provided-path'),
                        "Package step should override dependency path with local path"
                    )
                    self._log(f"Local path override successful: {source}")
            
        except Exception as e:
            self._assert(False, f"Local path override patterns test failed: {str(e)}")
    
    def test_dependency_input_extraction(self) -> None:
        """Test dependency input extraction patterns."""
        self._log("Testing dependency input extraction...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test extract_inputs_from_dependencies method if available
            if hasattr(builder, 'extract_inputs_from_dependencies'):
                # Create mock dependencies
                mock_dependencies = []
                
                # Test with empty dependencies
                extracted = builder.extract_inputs_from_dependencies(mock_dependencies)
                self._assert(
                    isinstance(extracted, dict),
                    "Extracted inputs must be a dictionary"
                )
                
                self._log(f"Extracted inputs from empty dependencies: {extracted}")
            else:
                self._log("No dependency extraction method found")
            
            # Test that _get_inputs can handle both explicit inputs and dependency inputs
            test_inputs = {"explicit_input": "s3://bucket/explicit"}
            processing_inputs = builder._get_inputs(test_inputs)
            
            self._assert(
                isinstance(processing_inputs, list),
                "Processing inputs must be a list"
            )
            
        except Exception as e:
            self._assert(False, f"Dependency input extraction test failed: {str(e)}")
    
    # Helper methods for validation
    
    def _validate_processing_input_object(self, proc_input, builder) -> None:
        """Validate a ProcessingInput object structure."""
        # Check required attributes
        required_attrs = ['input_name']
        for attr in required_attrs:
            self._assert(
                hasattr(proc_input, attr),
                f"ProcessingInput must have {attr} attribute"
            )
        
        # Check source or s3_data
        has_source = hasattr(proc_input, 'source') or hasattr(proc_input, 's3_data')
        self._assert(
            has_source,
            "ProcessingInput must have source or s3_data attribute"
        )
        
        # Check destination if present
        if hasattr(proc_input, 'destination'):
            destination = proc_input.destination
            self._assert(
                isinstance(destination, str),
                f"ProcessingInput destination must be string: {destination}"
            )
            
            # Container paths should be absolute
            if destination.startswith('/'):
                self._log(f"Valid container destination path: {destination}")
            else:
                self._log(f"Warning: Destination path may not be absolute: {destination}")
        
        # Log input details
        if self.verbose:
            input_name = getattr(proc_input, 'input_name', 'unknown')
            source = getattr(proc_input, 'source', getattr(proc_input, 's3_data', 'unknown'))
            destination = getattr(proc_input, 'destination', 'unknown')
            self._log(f"ProcessingInput: {input_name} | {source} -> {destination}")
    
    def _validate_processing_output_object(self, proc_output, builder) -> None:
        """Validate a ProcessingOutput object structure."""
        # Check required attributes
        required_attrs = ['output_name', 'source']
        for attr in required_attrs:
            self._assert(
                hasattr(proc_output, attr),
                f"ProcessingOutput must have {attr} attribute"
            )
        
        # Check source path (container path)
        source = proc_output.source
        self._assert(
            isinstance(source, str),
            f"ProcessingOutput source must be string: {source}"
        )
        
        # Container paths should be absolute
        if source.startswith('/'):
            self._log(f"Valid container source path: {source}")
        else:
            self._log(f"Warning: Source path may not be absolute: {source}")
        
        # Check destination if present
        if hasattr(proc_output, 'destination'):
            destination = proc_output.destination
            self._assert(
                isinstance(destination, str),
                f"ProcessingOutput destination must be string: {destination}"
            )
        
        # Log output details
        if self.verbose:
            output_name = getattr(proc_output, 'output_name', 'unknown')
            destination = getattr(proc_output, 'destination', 'unknown')
            self._log(f"ProcessingOutput: {output_name} | {source} -> {destination}")
    
    def _validate_container_paths(self, paths: Dict[str, str], path_type: str) -> None:
        """Validate container path dictionary."""
        self._assert(
            isinstance(paths, dict),
            f"Container {path_type} paths must be a dictionary"
        )
        
        for logical_name, container_path in paths.items():
            self._assert(
                isinstance(logical_name, str) and len(logical_name) > 0,
                f"Logical name must be non-empty string: {logical_name}"
            )
            
            self._assert(
                isinstance(container_path, str) and container_path.startswith('/'),
                f"Container path for {logical_name} must be absolute path: {container_path}"
            )
            
            if self.verbose:
                self._log(f"Container {path_type} path: {logical_name} -> {container_path}")
    
    def _validate_input_path_usage(self, processing_inputs: List, expected_paths: Dict[str, str]) -> None:
        """Validate that ProcessingInputs use expected container paths."""
        for proc_input in processing_inputs:
            if hasattr(proc_input, 'input_name') and hasattr(proc_input, 'destination'):
                input_name = proc_input.input_name
                destination = proc_input.destination
                
                # Check if this input uses an expected path
                if input_name in expected_paths:
                    expected_path = expected_paths[input_name]
                    
                    # Destination should match or be within the expected path
                    if destination == expected_path or destination.startswith(expected_path):
                        self._log(f"Input path mapping correct: {input_name} -> {destination}")
                    else:
                        self._log(f"Warning: Input path mismatch for {input_name}: expected {expected_path}, got {destination}")
    
    def _validate_output_path_usage(self, processing_outputs: List, expected_paths: Dict[str, str]) -> None:
        """Validate that ProcessingOutputs use expected container paths."""
        for proc_output in processing_outputs:
            if hasattr(proc_output, 'output_name') and hasattr(proc_output, 'source'):
                output_name = proc_output.output_name
                source = proc_output.source
                
                # Check if this output uses an expected path
                if output_name in expected_paths:
                    expected_path = expected_paths[output_name]
                    
                    # Source should match or be within the expected path
                    if source == expected_path or source.startswith(expected_path):
                        self._log(f"Output path mapping correct: {output_name} -> {source}")
                    else:
                        self._log(f"Warning: Output path mismatch for {output_name}: expected {expected_path}, got {source}")
    
    # Helper methods for special pattern testing
    
    def _test_local_path_override(self, builder) -> None:
        """Test local path override pattern."""
        self._log("Testing local path override pattern...")
        
        # This pattern is specific to Package step
        builder_name = self.builder_class.__name__
        if "Package" in builder_name:
            self._log("Package step detected - testing local path override")
            # Detailed testing is done in test_local_path_override_patterns
        else:
            self._log("Local path override pattern not applicable to this step")
    
    def _test_file_upload_pattern(self, builder) -> None:
        """Test file upload pattern."""
        self._log("Testing file upload pattern...")
        
        # This pattern is specific to DummyTraining step
        builder_name = self.builder_class.__name__
        if "DummyTraining" in builder_name:
            self._log("DummyTraining step detected - testing file upload pattern")
            # Detailed testing is done in test_file_upload_patterns
        else:
            self._log("File upload pattern not applicable to this step")
    
    def _test_s3_path_handling(self, builder) -> None:
        """Test S3 path handling pattern."""
        self._log("Testing S3 path handling pattern...")
        
        # Check for S3 path utility methods
        s3_methods = ['_normalize_s3_uri', '_validate_s3_uri']
        found_methods = []
        
        for method in s3_methods:
            if hasattr(builder, method):
                found_methods.append(method)
        
        if found_methods:
            self._log(f"Found S3 path methods: {found_methods}")
        else:
            self._log("No S3 path utility methods found")
