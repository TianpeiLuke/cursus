"""
Level 1 Interface Tests for Processing step builders.

These tests focus on Processing-specific interface requirements:
- Processor creation methods (_create_processor)
- Framework-specific method signatures (SKLearn vs XGBoost)
- Step creation pattern compliance (Pattern A vs Pattern B)
- Processing-specific configuration attributes
- Environment variables and job arguments methods
"""

from ..interface_tests import InterfaceTests


class ProcessingInterfaceTests(InterfaceTests):
    """
    Level 1 Processing-specific interface tests.
    
    Extends the base InterfaceTests with Processing step-specific validations
    based on processing_step_builder_patterns.md analysis.
    """
    
    def get_step_type_specific_tests(self) -> list:
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
    
    def test_processor_creation_method(self):
        """Test that Processing builder implements _create_processor method."""
        self._assert(
            hasattr(self.builder_class, '_create_processor'),
            "Processing builder must implement _create_processor method"
        )
        
        # Test processor creation
        try:
            builder = self._create_builder_instance()
            processor = builder._create_processor()
            
            self._assert(
                processor is not None,
                "Processor creation should return a valid processor object"
            )
            
            # Check processor type based on framework
            framework = self.step_info.get('framework', '').lower()
            processor_class_name = processor.__class__.__name__
            
            if 'xgboost' in framework:
                self._assert(
                    'XGBoost' in processor_class_name,
                    f"XGBoost framework should create XGBoostProcessor, got {processor_class_name}"
                )
            elif 'sklearn' in framework or framework == 'sklearn':
                self._assert(
                    'SKLearn' in processor_class_name,
                    f"SKLearn framework should create SKLearnProcessor, got {processor_class_name}"
                )
            else:
                # Generic processor validation
                self._assert(
                    'Processor' in processor_class_name,
                    f"Should create a Processor object, got {processor_class_name}"
                )
                
        except Exception as e:
            self._assert(False, f"Processor creation failed: {str(e)}")
    
    def test_processing_configuration_attributes(self):
        """Test Processing-specific configuration attributes."""
        builder = self._create_builder_instance()
        
        # Check required processing configuration attributes
        required_attrs = [
            'processing_instance_count', 
            'processing_volume_size',
            'processing_instance_type_large', 
            'processing_instance_type_small',
            'processing_framework_version', 
            'use_large_processing_instance'
        ]
        
        for attr in required_attrs:
            if hasattr(builder.config, attr):
                self._log(f"✓ Processing config has {attr}")
            else:
                self._log(f"Warning: Processing config missing {attr}")
    
    def test_framework_specific_methods(self):
        """Test framework-specific method implementations."""
        builder = self._create_builder_instance()
        framework = self.step_info.get('framework', '').lower()
        
        # All Processing builders should have these methods
        required_methods = ['_create_processor', '_get_inputs', '_get_outputs']
        
        for method in required_methods:
            self._assert(
                hasattr(builder, method) and callable(getattr(builder, method)),
                f"Processing builder should have {method} method"
            )
        
        # Framework-specific method checks
        if 'xgboost' in framework:
            # XGBoost processors may have additional methods
            self._log("XGBoost framework detected - checking XGBoost-specific patterns")
        elif 'sklearn' in framework:
            # SKLearn processors follow standard patterns
            self._log("SKLearn framework detected - checking SKLearn-specific patterns")
    
    def test_step_creation_pattern_compliance(self):
        """Test step creation pattern compliance based on framework."""
        builder = self._create_builder_instance()
        
        # Check that builder has create_step method
        self._assert(
            hasattr(builder, 'create_step') and callable(builder.create_step),
            "Processing builder must have create_step method"
        )
        
        # Check step creation pattern based on framework
        framework = self.step_info.get('framework', '').lower()
        pattern = self.step_info.get('step_creation_pattern', 'Pattern A')
        
        if 'xgboost' in framework:
            # XGBoost steps should use Pattern B (processor.run + step_args)
            self._log("XGBoost framework should use Pattern B (processor.run + step_args)")
            if pattern != 'Pattern B':
                self._log("Warning: XGBoost step may not be using recommended Pattern B")
        else:
            # SKLearn steps should use Pattern A (direct ProcessingStep creation)
            self._log("SKLearn framework should use Pattern A (direct ProcessingStep creation)")
            if pattern != 'Pattern A':
                self._log("Warning: SKLearn step may not be using recommended Pattern A")
    
    def test_processing_input_output_methods(self):
        """Test Processing-specific input/output methods."""
        builder = self._create_builder_instance()
        
        # Check input/output methods
        self._assert(
            hasattr(builder, '_get_inputs') and callable(builder._get_inputs),
            "Processing builder must have _get_inputs method"
        )
        
        self._assert(
            hasattr(builder, '_get_outputs') and callable(builder._get_outputs),
            "Processing builder must have _get_outputs method"
        )
        
        # Test method signatures for Processing-specific requirements
        import inspect
        
        # Check _get_inputs signature
        try:
            inputs_sig = inspect.signature(builder._get_inputs)
            inputs_params = list(inputs_sig.parameters.keys())
            self._assert(
                'inputs' in inputs_params,
                "_get_inputs method should have 'inputs' parameter"
            )
        except Exception as e:
            self._log(f"Could not inspect _get_inputs signature: {str(e)}")
        
        # Check _get_outputs signature  
        try:
            outputs_sig = inspect.signature(builder._get_outputs)
            outputs_params = list(outputs_sig.parameters.keys())
            self._assert(
                'outputs' in outputs_params,
                "_get_outputs method should have 'outputs' parameter"
            )
        except Exception as e:
            self._log(f"Could not inspect _get_outputs signature: {str(e)}")
    
    def test_environment_variables_method(self):
        """Test Processing-specific environment variables method."""
        builder = self._create_builder_instance()
        
        # Check environment variables method
        self._assert(
            hasattr(builder, '_get_environment_variables') and callable(builder._get_environment_variables),
            "Processing builder must have _get_environment_variables method"
        )
        
        try:
            env_vars = builder._get_environment_variables()
            self._assert(
                isinstance(env_vars, dict),
                "Environment variables should be returned as a dictionary"
            )
            
            # Check for common Processing environment variables
            common_env_vars = ['LABEL_FIELD', 'JOB_TYPE']
            for var in common_env_vars:
                if var in env_vars:
                    self._log(f"✓ Found common Processing env var: {var}")
                    
        except Exception as e:
            self._assert(False, f"Environment variables method failed: {str(e)}")
    
    def test_job_arguments_method(self):
        """Test Processing-specific job arguments method."""
        builder = self._create_builder_instance()
        
        # Check job arguments method
        self._assert(
            hasattr(builder, '_get_job_arguments') and callable(builder._get_job_arguments),
            "Processing builder must have _get_job_arguments method"
        )
        
        try:
            job_args = builder._get_job_arguments()
            self._assert(
                job_args is None or isinstance(job_args, list),
                "Job arguments should be None or a list"
            )
            
            if job_args is not None:
                # Check job arguments patterns
                for arg in job_args:
                    self._assert(
                        isinstance(arg, str),
                        f"Job argument should be string, got {type(arg)}"
                    )
                
                # Check for common Processing job argument patterns
                if hasattr(builder.config, 'job_type'):
                    job_type_found = any('job' in arg.lower() for arg in job_args)
                    if job_type_found:
                        self._log("✓ Found job_type in job arguments")
                    else:
                        self._log("Info: No job_type argument found (may use environment variables)")
            else:
                self._log("Processing step uses no job arguments (environment variables only)")
                
        except Exception as e:
            self._assert(False, f"Job arguments method failed: {str(e)}")
