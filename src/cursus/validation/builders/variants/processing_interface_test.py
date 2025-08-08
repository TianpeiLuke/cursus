"""
Level 1 Processing Interface Tests - Enhanced Processing-Specific Interface Validation.

These tests focus on Processing step-specific interface requirements:
- Processor creation method validation
- Processing-specific configuration attributes
- Framework-specific method signatures
- Processing step creation patterns (Pattern A vs Pattern B)
"""

from typing import Dict, List, Any
from ..interface_tests import InterfaceTests


class ProcessingInterfaceTests(InterfaceTests):
    """
    Enhanced Level 1 interface tests for Processing step builders.
    
    Extends the generic interface tests with Processing-specific validations
    based on the patterns identified in Processing Step Builder Patterns analysis.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Processing-specific interface test methods."""
        return [
            "test_processor_creation_method",
            "test_processing_configuration_attributes",
            "test_framework_specific_methods",
            "test_step_creation_pattern_compliance",
            "test_processing_input_output_methods",
            "test_environment_variables_method",
            "test_job_arguments_method"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing step-specific mock objects."""
        # Get Processing-specific mocks from factory
        self.step_type_mocks = self.mock_factory.create_step_type_mocks()
        
        # Log Processing step info if verbose
        if self.verbose:
            framework = self.step_info.get('framework', 'Unknown')
            pattern = self.step_info.get('step_creation_pattern', 'Unknown')
            self._log(f"Processing step detected - Framework: {framework}, Pattern: {pattern}")
            
        # Set up Processing-specific mock attributes
        self.mock_processor = self.step_type_mocks.get('processor_class')
        self.mock_processing_input = self.step_type_mocks.get('processing_input')
        self.mock_processing_output = self.step_type_mocks.get('processing_output')
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate Processing step-specific requirements."""
        validation_results = {
            "is_processing_step": self.step_info.get("sagemaker_step_type") == "Processing",
            "processor_framework_detected": self.step_info.get("framework") is not None,
            "step_creation_pattern_detected": self.step_info.get("step_creation_pattern") is not None,
            "processing_mocks_created": len(self.step_type_mocks) > 0
        }
        
        return validation_results
    
    # Processing-specific interface tests
    
    def test_processor_creation_method(self) -> None:
        """Test that the builder implements _create_processor method correctly."""
        self._log("Testing processor creation method...")
        
        # Check method exists
        self._assert(
            hasattr(self.builder_class, '_create_processor'),
            "Processing step builder must implement _create_processor() method"
        )
        
        method = getattr(self.builder_class, '_create_processor')
        self._assert(
            callable(method),
            "_create_processor must be callable"
        )
        
        # Check method is not abstract
        self._assert(
            not getattr(method, '__isabstractmethod__', False),
            "_create_processor() must be implemented, not abstract"
        )
        
        # Test method can be called
        try:
            builder = self._create_builder_instance()
            processor = builder._create_processor()
            
            # Validate processor type based on framework
            framework = self.step_info.get('framework')
            if framework == 'sklearn':
                # Should create SKLearnProcessor
                self._assert(
                    'SKLearn' in str(type(processor)),
                    f"SKLearn framework should create SKLearnProcessor, got {type(processor)}"
                )
            elif framework == 'xgboost':
                # Should create XGBoostProcessor
                self._assert(
                    'XGBoost' in str(type(processor)),
                    f"XGBoost framework should create XGBoostProcessor, got {type(processor)}"
                )
            
        except Exception as e:
            self._assert(False, f"_create_processor() method failed: {str(e)}")
    
    def test_processing_configuration_attributes(self) -> None:
        """Test Processing-specific configuration attributes."""
        self._log("Testing processing configuration attributes...")
        
        try:
            builder = self._create_builder_instance()
            config = builder.config
            
            # Standard processing configuration attributes
            required_attrs = [
                'processing_instance_count',
                'processing_volume_size',
                'processing_instance_type_large',
                'processing_instance_type_small',
                'processing_framework_version',
                'use_large_processing_instance'
            ]
            
            for attr in required_attrs:
                self._assert(
                    hasattr(config, attr),
                    f"Processing config must have attribute: {attr}"
                )
            
            # Framework-specific attributes
            framework = self.step_info.get('framework')
            if framework == 'xgboost':
                self._assert(
                    hasattr(config, 'py_version'),
                    "XGBoost processing config must have py_version attribute"
                )
            
            # Script-related attributes
            script_attrs = ['processing_entry_point', 'source_dir']
            for attr in script_attrs:
                if not hasattr(config, attr):
                    self._log(f"Warning: Processing config missing script attribute: {attr}")
                    
        except Exception as e:
            self._assert(False, f"Processing configuration attributes test failed: {str(e)}")
    
    def test_framework_specific_methods(self) -> None:
        """Test framework-specific method implementations."""
        self._log("Testing framework-specific methods...")
        
        framework = self.step_info.get('framework')
        if not framework:
            self._log("No framework detected, skipping framework-specific tests")
            return
        
        try:
            builder = self._create_builder_instance()
            
            if framework == 'sklearn':
                # SKLearn-specific validations
                self._test_sklearn_specific_methods(builder)
            elif framework == 'xgboost':
                # XGBoost-specific validations
                self._test_xgboost_specific_methods(builder)
            else:
                self._log(f"Unknown framework: {framework}, performing generic validation")
                
        except Exception as e:
            self._assert(False, f"Framework-specific methods test failed: {str(e)}")
    
    def test_step_creation_pattern_compliance(self) -> None:
        """Test compliance with step creation patterns (Pattern A vs Pattern B)."""
        self._log("Testing step creation pattern compliance...")
        
        pattern = self.step_info.get('step_creation_pattern')
        framework = self.step_info.get('framework')
        
        if not pattern:
            self._log("No step creation pattern detected, inferring from framework")
            if framework == 'sklearn':
                pattern = 'pattern_a'
            elif framework == 'xgboost':
                pattern = 'pattern_b'
        
        try:
            builder = self._create_builder_instance()
            
            if pattern == 'pattern_a':
                # Pattern A: Direct ProcessingStep creation
                self._test_pattern_a_compliance(builder)
            elif pattern == 'pattern_b':
                # Pattern B: processor.run() + step_args
                self._test_pattern_b_compliance(builder)
            else:
                self._log(f"Unknown pattern: {pattern}, performing generic validation")
                
        except Exception as e:
            self._assert(False, f"Step creation pattern compliance test failed: {str(e)}")
    
    def test_processing_input_output_methods(self) -> None:
        """Test Processing-specific input/output method signatures."""
        self._log("Testing processing input/output methods...")
        
        # Test _get_inputs method signature
        method = getattr(self.builder_class, '_get_inputs', None)
        self._assert(method is not None, "Processing builder must implement _get_inputs")
        
        # Test _get_outputs method signature
        method = getattr(self.builder_class, '_get_outputs', None)
        self._assert(method is not None, "Processing builder must implement _get_outputs")
        
        try:
            builder = self._create_builder_instance()
            
            # Test methods can handle empty inputs/outputs
            empty_inputs = builder._get_inputs({})
            self._assert(
                isinstance(empty_inputs, list),
                "_get_inputs must return a list for Processing steps"
            )
            
            empty_outputs = builder._get_outputs({})
            self._assert(
                isinstance(empty_outputs, list),
                "_get_outputs must return a list for Processing steps"
            )
            
        except Exception as e:
            self._assert(False, f"Processing input/output methods test failed: {str(e)}")
    
    def test_environment_variables_method(self) -> None:
        """Test environment variables method for Processing steps."""
        self._log("Testing environment variables method...")
        
        try:
            builder = self._create_builder_instance()
            env_vars = builder._get_environment_variables()
            
            self._assert(
                isinstance(env_vars, dict),
                "_get_environment_variables must return a dictionary"
            )
            
            # Check for common Processing environment variables
            common_env_vars = ['SAGEMAKER_PROGRAM', 'SAGEMAKER_SUBMIT_DIRECTORY']
            for env_var in common_env_vars:
                if env_var not in env_vars:
                    self._log(f"Note: Common env var '{env_var}' not found")
            
            # Validate all values are strings
            for key, value in env_vars.items():
                self._assert(
                    isinstance(key, str) and isinstance(value, str),
                    f"Environment variable {key} must have string key and value"
                )
                
        except Exception as e:
            self._assert(False, f"Environment variables method test failed: {str(e)}")
    
    def test_job_arguments_method(self) -> None:
        """Test job arguments method for Processing steps."""
        self._log("Testing job arguments method...")
        
        try:
            builder = self._create_builder_instance()
            job_args = builder._get_job_arguments()
            
            # Job arguments can be None or list
            if job_args is not None:
                self._assert(
                    isinstance(job_args, list),
                    "_get_job_arguments must return a list or None"
                )
                
                # All arguments must be strings
                for arg in job_args:
                    self._assert(
                        isinstance(arg, str),
                        f"Job argument must be string, got {type(arg)}: {arg}"
                    )
            else:
                self._log("Job arguments method returned None (acceptable)")
                
        except Exception as e:
            self._assert(False, f"Job arguments method test failed: {str(e)}")
    
    # Helper methods for framework-specific testing
    
    def _test_sklearn_specific_methods(self, builder) -> None:
        """Test SKLearn-specific method implementations."""
        self._log("Testing SKLearn-specific methods...")
        
        # SKLearn processors typically use standard configuration
        config = builder.config
        self._assert(
            hasattr(config, 'processing_framework_version'),
            "SKLearn config must have processing_framework_version"
        )
    
    def _test_xgboost_specific_methods(self, builder) -> None:
        """Test XGBoost-specific method implementations."""
        self._log("Testing XGBoost-specific methods...")
        
        # XGBoost processors require additional configuration
        config = builder.config
        self._assert(
            hasattr(config, 'framework_version'),
            "XGBoost config must have framework_version"
        )
        self._assert(
            hasattr(config, 'py_version'),
            "XGBoost config must have py_version"
        )
    
    def _test_pattern_a_compliance(self, builder) -> None:
        """Test Pattern A (Direct ProcessingStep creation) compliance."""
        self._log("Testing Pattern A compliance...")
        
        # Pattern A should use direct ProcessingStep instantiation
        # This is validated by checking the create_step method behavior
        self._assert(
            hasattr(builder, 'create_step'),
            "Pattern A builders must have create_step method"
        )
        
        # Pattern A typically uses 'code' parameter for script
        config = builder.config
        if hasattr(config, 'get_script_path'):
            script_path = config.get_script_path()
            self._assert(
                script_path is not None,
                "Pattern A should have script path available"
            )
    
    def _test_pattern_b_compliance(self, builder) -> None:
        """Test Pattern B (processor.run() + step_args) compliance."""
        self._log("Testing Pattern B compliance...")
        
        # Pattern B should use processor.run() to create step_args
        # This is validated by checking configuration for source_dir support
        config = builder.config
        
        # Pattern B typically supports source_dir for package upload
        source_dir_attrs = ['processing_source_dir', 'source_dir']
        has_source_dir = any(hasattr(config, attr) for attr in source_dir_attrs)
        
        if not has_source_dir:
            self._log("Warning: Pattern B should support source_dir for package upload")
        
        # Pattern B typically has entry point configuration
        entry_point_attrs = ['processing_entry_point', 'entry_point']
        has_entry_point = any(hasattr(config, attr) for attr in entry_point_attrs)
        
        self._assert(
            has_entry_point,
            "Pattern B should have entry point configuration"
        )
