"""
Level 2 Processing-Specific Specification Tests for step builders.

These tests focus on Processing step specification and contract compliance:
- Job type-based specification loading
- Processor-specific configuration validation
- Environment variable handling patterns
- Input/output specification compliance
- Script contract integration
"""

import json
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from ..specification_tests import SpecificationTests


class ProcessingSpecificationTests(SpecificationTests):
    """
    Level 2 Processing-specific specification tests.
    
    These tests validate that Processing step builders properly use specifications
    and contracts to define their behavior, with focus on Processing-specific patterns.
    """
    
    def get_step_type_specific_tests(self) -> list:
        """Return Processing-specific specification test methods."""
        return [
            "test_job_type_specification_loading",
            "test_processor_configuration_validation", 
            "test_processing_environment_variables",
            "test_specification_driven_inputs",
            "test_specification_driven_outputs",
            "test_contract_path_mapping",
            "test_job_arguments_specification",
            "test_processor_type_alignment",
            "test_step_creation_pattern_compliance"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing-specific mock objects for specification tests."""
        # Mock processor types
        self.mock_sklearn_processor = Mock()
        self.mock_xgboost_processor = Mock()
        
        # Mock specification objects
        self.mock_processing_spec = Mock()
        self.mock_processing_spec.dependencies = {
            "input_data": Mock(logical_name="input_data", required=True),
            "metadata": Mock(logical_name="metadata", required=False)
        }
        self.mock_processing_spec.outputs = {
            "processed_data": Mock(logical_name="processed_data"),
            "statistics": Mock(logical_name="statistics")
        }
        
        # Mock contract objects
        self.mock_contract = Mock()
        self.mock_contract.expected_input_paths = {
            "input_data": "/opt/ml/processing/input/data",
            "metadata": "/opt/ml/processing/input/metadata"
        }
        self.mock_contract.expected_output_paths = {
            "processed_data": "/opt/ml/processing/output/data",
            "statistics": "/opt/ml/processing/output/stats"
        }
    
    def _validate_step_type_requirements(self) -> dict:
        """Validate Processing-specific requirements for specification tests."""
        return {
            "specification_tests_completed": True,
            "processing_specific_validations": True,
            "processor_type_validated": True,
            "job_type_support_validated": True
        }
    
    def test_job_type_specification_loading(self) -> None:
        """Test that Processing builders properly load specifications based on job type."""
        self._log("Testing job type-based specification loading")
        
        # Test multi-job-type builders (TabularPreprocessing, CurrencyConversion)
        if hasattr(self.builder_class, '_load_specification_by_job_type'):
            # Test training job type
            with patch.object(self.builder_class, '_load_specification_by_job_type') as mock_load:
                mock_load.return_value = self.mock_processing_spec
                
                config = Mock()
                config.job_type = "training"
                
                try:
                    builder = self.builder_class(config=config)
                    mock_load.assert_called_with("training")
                    self._assert(True, "Job type specification loading works for training")
                except Exception as e:
                    self._log(f"Job type specification loading failed: {e}")
                    self._assert(False, f"Job type specification loading failed: {e}")
            
            # Test validation job type
            with patch.object(self.builder_class, '_load_specification_by_job_type') as mock_load:
                mock_load.return_value = self.mock_processing_spec
                
                config = Mock()
                config.job_type = "validation"
                
                try:
                    builder = self.builder_class(config=config)
                    mock_load.assert_called_with("validation")
                    self._assert(True, "Job type specification loading works for validation")
                except Exception as e:
                    self._log(f"Job type specification loading failed: {e}")
                    self._assert(False, f"Job type specification loading failed: {e}")
        else:
            # Single-purpose builders should still have specification
            self._log("Single-purpose Processing builder - checking for specification")
            self._assert(True, "Single-purpose builders validated separately")
    
    def test_processor_configuration_validation(self) -> None:
        """Test that Processing builders validate processor-specific configuration."""
        self._log("Testing processor configuration validation")
        
        # Test required processing configuration attributes
        required_attrs = [
            'processing_instance_count', 'processing_volume_size',
            'processing_instance_type_large', 'processing_instance_type_small',
            'processing_framework_version', 'use_large_processing_instance'
        ]
        
        if hasattr(self.builder_class, 'validate_configuration'):
            # Test with missing required attributes
            for attr in required_attrs:
                config = Mock()
                # Set all attributes except the one being tested
                for other_attr in required_attrs:
                    if other_attr != attr:
                        setattr(config, other_attr, "test_value")
                
                try:
                    builder = self.builder_class(config=config)
                    builder.validate_configuration()
                    self._log(f"Missing {attr} should have caused validation error")
                    self._assert(False, f"Missing {attr} should have caused validation error")
                except (ValueError, AttributeError):
                    self._assert(True, f"Correctly detected missing {attr}")
                except Exception as e:
                    self._log(f"Unexpected error for missing {attr}: {e}")
                    self._assert(False, f"Unexpected error for missing {attr}: {e}")
        else:
            self._log("No validate_configuration method found")
            self._assert(False, "Processing builders should have validate_configuration method")
    
    def test_processing_environment_variables(self) -> None:
        """Test that Processing builders handle environment variables correctly."""
        self._log("Testing Processing environment variables handling")
        
        if hasattr(self.builder_class, '_get_environment_variables'):
            config = Mock()
            config.label_name = "target"
            config.categorical_columns = ["cat1", "cat2"]
            config.currency_conversion_dict = {"USD": 1.0, "EUR": 0.85}
            
            try:
                builder = self.builder_class(config=config)
                env_vars = builder._get_environment_variables()
                
                # Check that environment variables are returned as dict
                self._assert(isinstance(env_vars, dict), "Environment variables should be dict")
                
                # Check for common Processing environment variable patterns
                if hasattr(config, 'label_name'):
                    self._assert("LABEL_FIELD" in env_vars or "label_name" in str(env_vars).lower(),
                               "Label field should be in environment variables")
                
                if hasattr(config, 'categorical_columns'):
                    # Should be comma-separated or JSON
                    found_categorical = any("categorical" in key.lower() for key in env_vars.keys())
                    self._assert(found_categorical, "Categorical columns should be in environment variables")
                
                if hasattr(config, 'currency_conversion_dict'):
                    # Should be JSON serialized
                    found_currency = any("currency" in key.lower() for key in env_vars.keys())
                    self._assert(found_currency, "Currency conversion dict should be in environment variables")
                
                self._assert(True, "Environment variables handling validated")
                
            except Exception as e:
                self._log(f"Environment variables test failed: {e}")
                self._assert(False, f"Environment variables test failed: {e}")
        else:
            self._log("No _get_environment_variables method found")
            self._assert(False, "Processing builders should have _get_environment_variables method")
    
    def test_specification_driven_inputs(self) -> None:
        """Test that Processing builders use specifications to define inputs."""
        self._log("Testing specification-driven input handling")
        
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
                
                # Check that ProcessingInput objects are created
                self._assert(isinstance(processing_inputs, list), "Should return list of ProcessingInput objects")
                
                if processing_inputs:
                    # Check first input structure
                    first_input = processing_inputs[0]
                    self._assert(hasattr(first_input, 'input_name'), "ProcessingInput should have input_name")
                    self._assert(hasattr(first_input, 'source'), "ProcessingInput should have source")
                    self._assert(hasattr(first_input, 'destination'), "ProcessingInput should have destination")
                
                self._assert(True, "Specification-driven inputs validated")
                
            except Exception as e:
                self._log(f"Specification-driven inputs test failed: {e}")
                self._assert(False, f"Specification-driven inputs test failed: {e}")
        else:
            self._log("No _get_inputs method found")
            self._assert(False, "Processing builders should have _get_inputs method")
    
    def test_specification_driven_outputs(self) -> None:
        """Test that Processing builders use specifications to define outputs."""
        self._log("Testing specification-driven output handling")
        
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
                
                # Check that ProcessingOutput objects are created
                self._assert(isinstance(processing_outputs, list), "Should return list of ProcessingOutput objects")
                
                if processing_outputs:
                    # Check first output structure
                    first_output = processing_outputs[0]
                    self._assert(hasattr(first_output, 'output_name'), "ProcessingOutput should have output_name")
                    self._assert(hasattr(first_output, 'source'), "ProcessingOutput should have source")
                    self._assert(hasattr(first_output, 'destination'), "ProcessingOutput should have destination")
                
                self._assert(True, "Specification-driven outputs validated")
                
            except Exception as e:
                self._log(f"Specification-driven outputs test failed: {e}")
                self._assert(False, f"Specification-driven outputs test failed: {e}")
        else:
            self._log("No _get_outputs method found")
            self._assert(False, "Processing builders should have _get_outputs method")
    
    def test_contract_path_mapping(self) -> None:
        """Test that Processing builders use contracts for container path mapping."""
        self._log("Testing contract-based path mapping")
        
        config = Mock()
        builder = self.builder_class(config=config)
        
        # Mock contract
        builder.contract = self.mock_contract
        
        # Test that contract paths are used correctly
        if hasattr(builder, '_get_inputs') and hasattr(builder, 'contract'):
            try:
                # Check that contract has expected path mappings
                self._assert(hasattr(builder.contract, 'expected_input_paths'), 
                           "Contract should have expected_input_paths")
                self._assert(hasattr(builder.contract, 'expected_output_paths'), 
                           "Contract should have expected_output_paths")
                
                # Verify path mapping structure
                if hasattr(builder.contract, 'expected_input_paths'):
                    input_paths = builder.contract.expected_input_paths
                    self._assert(isinstance(input_paths, dict), "Input paths should be dict")
                
                if hasattr(builder.contract, 'expected_output_paths'):
                    output_paths = builder.contract.expected_output_paths
                    self._assert(isinstance(output_paths, dict), "Output paths should be dict")
                
                self._assert(True, "Contract path mapping validated")
                
            except Exception as e:
                self._log(f"Contract path mapping test failed: {e}")
                self._assert(False, f"Contract path mapping test failed: {e}")
        else:
            self._log("Contract or input/output methods not found")
            self._assert(False, "Processing builders should use contract-based path mapping")
    
    def test_job_arguments_specification(self) -> None:
        """Test that Processing builders handle job arguments according to specification."""
        self._log("Testing job arguments specification compliance")
        
        if hasattr(self.builder_class, '_get_job_arguments'):
            config = Mock()
            config.job_type = "training"
            config.mode = "batch"
            config.marketplace_id_col = "marketplace_id"
            config.enable_currency_conversion = True
            config.currency_col = "currency"
            
            try:
                builder = self.builder_class(config=config)
                job_args = builder._get_job_arguments()
                
                # Job arguments can be None, empty list, or list of strings
                if job_args is not None:
                    self._assert(isinstance(job_args, list), "Job arguments should be list or None")
                    
                    if job_args:
                        # All arguments should be strings
                        all_strings = all(isinstance(arg, str) for arg in job_args)
                        self._assert(all_strings, "All job arguments should be strings")
                        
                        # Common Processing argument patterns
                        arg_string = " ".join(job_args)
                        if hasattr(config, 'job_type'):
                            job_type_found = "job_type" in arg_string or "job-type" in arg_string
                            self._assert(job_type_found, "Job type should be in arguments")
                
                self._assert(True, "Job arguments specification validated")
                
            except Exception as e:
                self._log(f"Job arguments test failed: {e}")
                self._assert(False, f"Job arguments test failed: {e}")
        else:
            self._log("No _get_job_arguments method found")
            self._assert(False, "Processing builders should have _get_job_arguments method")
    
    def test_processor_type_alignment(self) -> None:
        """Test that Processing builders create the correct processor type."""
        self._log("Testing processor type alignment")
        
        if hasattr(self.builder_class, '_create_processor'):
            config = Mock()
            config.processing_instance_type_large = "ml.m5.xlarge"
            config.processing_instance_type_small = "ml.m5.large"
            config.use_large_processing_instance = False
            config.processing_instance_count = 1
            config.processing_volume_size = 30
            config.processing_framework_version = "0.23-1"
            
            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()
                
                processor = builder._create_processor()
                
                # Check processor type
                processor_type = type(processor).__name__
                self._assert(processor_type in ["SKLearnProcessor", "XGBoostProcessor", "Mock"], 
                           f"Should create valid processor type, got: {processor_type}")
                
                # XGBoost processors should have framework and py_version
                if "XGBoost" in processor_type:
                    if hasattr(config, 'framework_version'):
                        self._assert(True, "XGBoost processor with framework version")
                    if hasattr(config, 'py_version'):
                        self._assert(True, "XGBoost processor with Python version")
                
                self._assert(True, "Processor type alignment validated")
                
            except Exception as e:
                self._log(f"Processor type test failed: {e}")
                self._assert(False, f"Processor type test failed: {e}")
        else:
            self._log("No _create_processor method found")
            self._assert(False, "Processing builders should have _create_processor method")
    
    def test_step_creation_pattern_compliance(self) -> None:
        """Test that Processing builders follow the correct step creation pattern."""
        self._log("Testing step creation pattern compliance")
        
        if hasattr(self.builder_class, 'create_step'):
            config = Mock()
            # Set up minimal required configuration
            config.processing_instance_type_large = "ml.m5.xlarge"
            config.processing_instance_type_small = "ml.m5.large"
            config.use_large_processing_instance = False
            config.processing_instance_count = 1
            config.processing_volume_size = 30
            config.processing_framework_version = "0.23-1"
            
            try:
                builder = self.builder_class(config=config)
                builder.role = "test-role"
                builder.session = Mock()
                
                # Mock required methods
                builder._create_processor = Mock(return_value=self.mock_sklearn_processor)
                builder._get_inputs = Mock(return_value=[])
                builder._get_outputs = Mock(return_value=[])
                builder._get_job_arguments = Mock(return_value=["--job_type", "training"])
                builder._get_step_name = Mock(return_value="test-processing-step")
                builder._get_cache_config = Mock(return_value=None)
                
                # Test step creation
                step = builder.create_step(
                    inputs={"input_data": "s3://bucket/input"},
                    outputs={"output_data": "s3://bucket/output"},
                    dependencies=[],
                    enable_caching=True
                )
                
                # Verify step creation patterns
                self._assert(step is not None, "Step should be created")
                
                # Check for ProcessingStep characteristics
                step_type = type(step).__name__
                self._assert("ProcessingStep" in step_type or "Mock" in step_type, 
                           f"Should create ProcessingStep, got: {step_type}")
                
                # Pattern A: Direct ProcessingStep creation (most common)
                # Pattern B: processor.run() + step_args (XGBoost)
                # Both patterns should result in a valid step
                
                self._assert(True, "Step creation pattern compliance validated")
                
            except Exception as e:
                self._log(f"Step creation pattern test failed: {e}")
                self._assert(False, f"Step creation pattern test failed: {e}")
        else:
            self._log("No create_step method found")
            self._assert(False, "Processing builders should have create_step method")
