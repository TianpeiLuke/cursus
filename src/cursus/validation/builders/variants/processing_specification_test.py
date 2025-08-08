"""
Level 2 Processing Specification Tests - Enhanced Processing-Specific Specification Validation.

These tests focus on Processing step-specific specification and contract compliance:
- Job type-based specification loading patterns
- Environment variable construction patterns
- Job arguments validation patterns
- Specification-driven input/output handling
"""

from typing import Dict, List, Any
import json
from ..specification_tests import SpecificationTests


class ProcessingSpecificationTests(SpecificationTests):
    """
    Enhanced Level 2 specification tests for Processing step builders.
    
    Extends the generic specification tests with Processing-specific validations
    based on the patterns identified in Processing Step Builder Patterns analysis.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Processing-specific specification test methods."""
        return [
            "test_job_type_specification_loading",
            "test_environment_variable_patterns",
            "test_job_arguments_patterns",
            "test_specification_driven_inputs",
            "test_specification_driven_outputs",
            "test_contract_path_mapping",
            "test_multi_job_type_support",
            "test_framework_specific_specifications"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing step-specific mock objects."""
        # Get Processing-specific mocks from factory
        self.step_type_mocks = self.mock_factory.create_step_type_mocks()
        
        # Log Processing step info if verbose
        if self.verbose:
            framework = self.step_info.get('framework', 'Unknown')
            job_types = self.step_info.get('supported_job_types', [])
            self._log(f"Processing specification tests - Framework: {framework}, Job types: {job_types}")
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate Processing step-specific requirements."""
        validation_results = {
            "is_processing_step": self.step_info.get("sagemaker_step_type") == "Processing",
            "has_specification": self.step_info.get("has_specification", False),
            "has_contract": self.step_info.get("has_contract", False),
            "supports_job_types": len(self.step_info.get("supported_job_types", [])) > 0
        }
        
        return validation_results
    
    # Processing-specific specification tests
    
    def test_job_type_specification_loading(self) -> None:
        """Test job type-based specification loading patterns."""
        self._log("Testing job type specification loading...")
        
        try:
            builder = self._create_builder_instance()
            
            # Check if builder supports job types
            config = builder.config
            if not hasattr(config, 'job_type'):
                self._log("Builder does not support job types, skipping test")
                return
            
            job_type = config.job_type
            self._log(f"Testing job type: {job_type}")
            
            # Validate specification loading based on job type
            if hasattr(builder, 'spec') and builder.spec:
                spec = builder.spec
                
                # Check that specification is appropriate for job type
                self._assert(
                    hasattr(spec, 'dependencies'),
                    f"Specification for job type '{job_type}' must have dependencies"
                )
                
                self._assert(
                    hasattr(spec, 'outputs'),
                    f"Specification for job type '{job_type}' must have outputs"
                )
                
                # Log specification details
                if self.verbose:
                    dep_count = len(spec.dependencies) if spec.dependencies else 0
                    out_count = len(spec.outputs) if spec.outputs else 0
                    self._log(f"Specification loaded: {dep_count} dependencies, {out_count} outputs")
            else:
                self._log("No specification loaded - may use dynamic loading")
                
        except Exception as e:
            self._assert(False, f"Job type specification loading test failed: {str(e)}")
    
    def test_environment_variable_patterns(self) -> None:
        """Test Processing-specific environment variable construction patterns."""
        self._log("Testing environment variable patterns...")
        
        try:
            builder = self._create_builder_instance()
            env_vars = builder._get_environment_variables()
            
            # Test common Processing environment variable patterns
            self._test_basic_env_vars(env_vars, builder)
            self._test_step_specific_env_vars(env_vars, builder)
            self._test_json_serialized_env_vars(env_vars, builder)
            self._test_list_env_vars(env_vars, builder)
            
        except Exception as e:
            self._assert(False, f"Environment variable patterns test failed: {str(e)}")
    
    def test_job_arguments_patterns(self) -> None:
        """Test Processing-specific job arguments patterns."""
        self._log("Testing job arguments patterns...")
        
        try:
            builder = self._create_builder_instance()
            job_args = builder._get_job_arguments()
            
            if job_args is None:
                self._log("No job arguments pattern (uses environment variables only)")
                return
            
            self._assert(
                isinstance(job_args, list),
                "Job arguments must be a list"
            )
            
            # Test common job argument patterns
            self._test_job_type_arguments(job_args, builder)
            self._test_configuration_arguments(job_args, builder)
            self._test_optional_arguments(job_args, builder)
            
        except Exception as e:
            self._assert(False, f"Job arguments patterns test failed: {str(e)}")
    
    def test_specification_driven_inputs(self) -> None:
        """Test specification-driven input handling."""
        self._log("Testing specification-driven inputs...")
        
        try:
            builder = self._create_builder_instance()
            
            if not builder.spec:
                self._log("No specification available, skipping test")
                return
            
            # Create test inputs based on specification
            test_inputs = {}
            for dep_name, dep_spec in builder.spec.dependencies.items():
                logical_name = dep_spec.logical_name
                test_inputs[logical_name] = f"s3://test-bucket/{logical_name}"
            
            # Test input processing
            processing_inputs = builder._get_inputs(test_inputs)
            
            self._assert(
                isinstance(processing_inputs, list),
                "Processing inputs must be a list"
            )
            
            # Validate each ProcessingInput
            for proc_input in processing_inputs:
                self._validate_processing_input(proc_input, builder)
                
        except Exception as e:
            self._assert(False, f"Specification-driven inputs test failed: {str(e)}")
    
    def test_specification_driven_outputs(self) -> None:
        """Test specification-driven output handling."""
        self._log("Testing specification-driven outputs...")
        
        try:
            builder = self._create_builder_instance()
            
            if not builder.spec:
                self._log("No specification available, skipping test")
                return
            
            # Create test outputs based on specification
            test_outputs = {}
            for out_name, out_spec in builder.spec.outputs.items():
                logical_name = out_spec.logical_name
                test_outputs[logical_name] = f"s3://test-bucket/{logical_name}"
            
            # Test output processing
            processing_outputs = builder._get_outputs(test_outputs)
            
            self._assert(
                isinstance(processing_outputs, list),
                "Processing outputs must be a list"
            )
            
            # Validate each ProcessingOutput
            for proc_output in processing_outputs:
                self._validate_processing_output(proc_output, builder)
                
        except Exception as e:
            self._assert(False, f"Specification-driven outputs test failed: {str(e)}")
    
    def test_contract_path_mapping(self) -> None:
        """Test contract-based path mapping."""
        self._log("Testing contract path mapping...")
        
        try:
            builder = self._create_builder_instance()
            
            if not builder.contract:
                self._log("No contract available, skipping test")
                return
            
            contract = builder.contract
            
            # Test input path mapping
            if hasattr(contract, 'expected_input_paths'):
                input_paths = contract.expected_input_paths
                self._assert(
                    isinstance(input_paths, dict),
                    "Contract input paths must be a dictionary"
                )
                
                for logical_name, container_path in input_paths.items():
                    self._assert(
                        isinstance(container_path, str) and container_path.startswith('/'),
                        f"Container path for {logical_name} must be absolute path: {container_path}"
                    )
            
            # Test output path mapping
            if hasattr(contract, 'expected_output_paths'):
                output_paths = contract.expected_output_paths
                self._assert(
                    isinstance(output_paths, dict),
                    "Contract output paths must be a dictionary"
                )
                
                for logical_name, container_path in output_paths.items():
                    self._assert(
                        isinstance(container_path, str) and container_path.startswith('/'),
                        f"Container path for {logical_name} must be absolute path: {container_path}"
                    )
                    
        except Exception as e:
            self._assert(False, f"Contract path mapping test failed: {str(e)}")
    
    def test_multi_job_type_support(self) -> None:
        """Test multi-job-type support patterns."""
        self._log("Testing multi-job-type support...")
        
        supported_job_types = self.step_info.get('supported_job_types', [])
        if len(supported_job_types) <= 1:
            self._log("Single job type or no job types detected, skipping test")
            return
        
        try:
            # Test that builder can handle different job types
            for job_type in supported_job_types:
                self._log(f"Testing job type: {job_type}")
                
                # Create config with specific job type
                config = self._create_test_config()
                if hasattr(config, 'job_type'):
                    config.job_type = job_type
                
                # Create builder with job type
                builder = self.builder_class(
                    config=config,
                    sagemaker_session=self.mock_session,
                    role=self.mock_role,
                    registry_manager=self.mock_registry_manager,
                    dependency_resolver=self.mock_dependency_resolver
                )
                
                # Validate that specification is loaded for this job type
                if builder.spec:
                    self._log(f"Specification loaded for job type: {job_type}")
                else:
                    self._log(f"No specification for job type: {job_type}")
                    
        except Exception as e:
            self._assert(False, f"Multi-job-type support test failed: {str(e)}")
    
    def test_framework_specific_specifications(self) -> None:
        """Test framework-specific specification requirements."""
        self._log("Testing framework-specific specifications...")
        
        framework = self.step_info.get('framework')
        if not framework:
            self._log("No framework detected, skipping test")
            return
        
        try:
            builder = self._create_builder_instance()
            
            if framework == 'sklearn':
                self._test_sklearn_specifications(builder)
            elif framework == 'xgboost':
                self._test_xgboost_specifications(builder)
            else:
                self._log(f"Unknown framework: {framework}, performing generic validation")
                
        except Exception as e:
            self._assert(False, f"Framework-specific specifications test failed: {str(e)}")
    
    # Helper methods for environment variable testing
    
    def _test_basic_env_vars(self, env_vars: Dict[str, str], builder) -> None:
        """Test basic environment variables."""
        # Check for common SageMaker environment variables
        common_vars = ['SAGEMAKER_PROGRAM', 'SAGEMAKER_SUBMIT_DIRECTORY']
        for var in common_vars:
            if var not in env_vars:
                self._log(f"Note: Common env var '{var}' not found")
    
    def _test_step_specific_env_vars(self, env_vars: Dict[str, str], builder) -> None:
        """Test step-specific environment variables."""
        builder_name = self.builder_class.__name__
        
        if "TabularPreprocessing" in builder_name:
            # TabularPreprocessing-specific env vars
            expected_vars = ['LABEL_FIELD']
            for var in expected_vars:
                if var in env_vars:
                    self._log(f"Found TabularPreprocessing env var: {var}")
        elif "CurrencyConversion" in builder_name:
            # CurrencyConversion-specific env vars
            expected_vars = ['CURRENCY_CONVERSION_DICT', 'CATEGORICAL_COLUMNS']
            for var in expected_vars:
                if var in env_vars:
                    self._log(f"Found CurrencyConversion env var: {var}")
    
    def _test_json_serialized_env_vars(self, env_vars: Dict[str, str], builder) -> None:
        """Test JSON-serialized environment variables."""
        json_vars = []
        for key, value in env_vars.items():
            if self._is_json_string(value):
                json_vars.append(key)
        
        if json_vars:
            self._log(f"Found JSON-serialized env vars: {json_vars}")
            
            # Validate JSON format
            for var in json_vars:
                try:
                    json.loads(env_vars[var])
                except json.JSONDecodeError:
                    self._assert(False, f"Invalid JSON in env var {var}: {env_vars[var]}")
    
    def _test_list_env_vars(self, env_vars: Dict[str, str], builder) -> None:
        """Test comma-separated list environment variables."""
        list_vars = []
        for key, value in env_vars.items():
            if ',' in value and not self._is_json_string(value):
                list_vars.append(key)
        
        if list_vars:
            self._log(f"Found list env vars: {list_vars}")
    
    def _is_json_string(self, value: str) -> bool:
        """Check if a string is valid JSON."""
        try:
            json.loads(value)
            return True
        except (json.JSONDecodeError, TypeError):
            return False
    
    # Helper methods for job arguments testing
    
    def _test_job_type_arguments(self, job_args: List[str], builder) -> None:
        """Test job type arguments."""
        config = builder.config
        if hasattr(config, 'job_type'):
            job_type = config.job_type
            
            # Check if job type is passed as argument
            job_type_patterns = ['--job_type', '--job-type']
            for pattern in job_type_patterns:
                if pattern in job_args:
                    idx = job_args.index(pattern)
                    if idx + 1 < len(job_args):
                        arg_value = job_args[idx + 1]
                        self._log(f"Found job type argument: {pattern} {arg_value}")
    
    def _test_configuration_arguments(self, job_args: List[str], builder) -> None:
        """Test configuration-based arguments."""
        # Look for common configuration arguments
        config_patterns = ['--mode', '--marketplace-id-col', '--enable-conversion']
        for pattern in config_patterns:
            if pattern in job_args:
                self._log(f"Found configuration argument: {pattern}")
    
    def _test_optional_arguments(self, job_args: List[str], builder) -> None:
        """Test optional arguments."""
        # Check argument pairing (each flag should have a value)
        for i in range(0, len(job_args), 2):
            if i + 1 < len(job_args):
                flag = job_args[i]
                value = job_args[i + 1]
                
                if flag.startswith('--'):
                    self._assert(
                        not value.startswith('--'),
                        f"Argument flag {flag} should have a value, not another flag: {value}"
                    )
    
    # Helper methods for input/output validation
    
    def _validate_processing_input(self, proc_input, builder) -> None:
        """Validate a ProcessingInput object."""
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
        
        # Check destination
        if hasattr(proc_input, 'destination'):
            destination = proc_input.destination
            self._assert(
                isinstance(destination, str) and destination.startswith('/'),
                f"ProcessingInput destination must be absolute path: {destination}"
            )
    
    def _validate_processing_output(self, proc_output, builder) -> None:
        """Validate a ProcessingOutput object."""
        # Check required attributes
        required_attrs = ['output_name', 'source']
        for attr in required_attrs:
            self._assert(
                hasattr(proc_output, attr),
                f"ProcessingOutput must have {attr} attribute"
            )
        
        # Check source path
        source = proc_output.source
        self._assert(
            isinstance(source, str) and source.startswith('/'),
            f"ProcessingOutput source must be absolute path: {source}"
        )
        
        # Check destination
        if hasattr(proc_output, 'destination'):
            destination = proc_output.destination
            self._assert(
                isinstance(destination, str),
                f"ProcessingOutput destination must be string: {destination}"
            )
    
    # Helper methods for framework-specific testing
    
    def _test_sklearn_specifications(self, builder) -> None:
        """Test SKLearn-specific specifications."""
        self._log("Testing SKLearn-specific specifications...")
        
        # SKLearn processors typically use standard specifications
        if builder.spec:
            self._log("SKLearn specification loaded successfully")
        else:
            self._log("No specification for SKLearn processor")
    
    def _test_xgboost_specifications(self, builder) -> None:
        """Test XGBoost-specific specifications."""
        self._log("Testing XGBoost-specific specifications...")
        
        # XGBoost processors may have framework-specific requirements
        if builder.spec:
            self._log("XGBoost specification loaded successfully")
            
            # Check for XGBoost-specific dependencies
            if hasattr(builder.spec, 'dependencies'):
                deps = builder.spec.dependencies
                xgboost_deps = ['model_input', 'eval_data_input']
                for dep in xgboost_deps:
                    if dep in deps:
                        self._log(f"Found XGBoost dependency: {dep}")
        else:
            self._log("No specification for XGBoost processor")
