"""
Processing step builder test variant.
"""

from typing import Dict, List, Any
from ..base_test import UniversalStepBuilderTestBase


class ProcessingStepBuilderTest(UniversalStepBuilderTestBase):
    """
    Specialized test variant for Processing step builders.
    
    This class provides comprehensive validation for Processing steps based on
    the patterns identified in the Processing Step Builder Patterns design document.
    """
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Processing step-specific test methods."""
        return [
            "test_processor_creation",
            "test_processing_inputs_outputs",
            "test_processing_job_arguments",
            "test_environment_variables_processing",
            "test_property_files_configuration",
            "test_processing_code_handling"
        ]
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing step-specific mock objects."""
        # Get Processing-specific mocks from factory
        self.step_type_mocks = self.mock_factory.create_step_type_mocks()
        
        # Log Processing step info if verbose
        if self.verbose:
            self._log(f"Processing step detected - Framework: {self.step_info.get('framework', 'Unknown')}")
            self._log(f"Test pattern: {self.step_info.get('test_pattern', 'standard')}")
            
        # Set up Processing-specific mock attributes
        self.mock_processor = self.step_type_mocks.get('processor_class')
        self.mock_processing_input = self.step_type_mocks.get('processing_input')
        self.mock_processing_output = self.step_type_mocks.get('processing_output')
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate Processing step-specific requirements."""
        validation_results = {
            "is_processing_step": self.step_info.get("sagemaker_step_type") == "Processing",
            "processor_framework_detected": self.step_info.get("framework") is not None,
            "processing_mocks_created": len(self.step_type_mocks) > 0,
            "expected_processing_dependencies": len(self._get_expected_dependencies()) > 0
        }
        
        return validation_results
    
    # Processing step-specific test methods
    
    def test_processor_creation(self):
        """Validate processor creation patterns."""
        self._log("Testing processor creation...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test that builder can create processor
            self._assert(hasattr(builder, 'build'), "Builder should have build method")
            
            # Test processor configuration based on framework
            framework = self.step_info.get('framework')
            if framework:
                self._log(f"Testing {framework} processor creation")
                
                # Validate framework-specific processor attributes
                config = builder.config
                if framework == "sklearn":
                    self._assert(
                        hasattr(config, 'framework_version'),
                        "SKLearn processor should have framework_version"
                    )
                elif framework == "xgboost":
                    self._assert(
                        hasattr(config, 'framework_version'),
                        "XGBoost processor should have framework_version"
                    )
            
        except Exception as e:
            self._assert(False, f"Processor creation test failed: {str(e)}")
    
    def test_processing_inputs_outputs(self):
        """Test ProcessingInput and ProcessingOutput handling."""
        self._log("Testing processing inputs and outputs...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test that builder handles processing inputs/outputs
            # This would typically involve checking the build method's behavior
            # For now, we validate the configuration supports it
            
            config = builder.config
            self._assert(
                hasattr(config, 'processing_entry_point'),
                "Processing config should have entry point"
            )
            
            # Test expected dependencies (inputs)
            expected_deps = self._get_expected_dependencies()
            self._assert(len(expected_deps) > 0, "Processing step should have input dependencies")
            
            # Log expected inputs/outputs
            if self.verbose:
                self._log(f"Expected processing inputs: {expected_deps}")
            
        except Exception as e:
            self._assert(False, f"Processing inputs/outputs test failed: {str(e)}")
    
    def test_processing_job_arguments(self):
        """Test processing job arguments construction."""
        self._log("Testing processing job arguments...")
        
        try:
            builder = self._create_builder_instance()
            config = builder.config
            
            # Test processing-specific configuration
            self._assert(
                hasattr(config, 'processing_instance_type'),
                "Processing config should have instance type"
            )
            self._assert(
                hasattr(config, 'processing_instance_count'),
                "Processing config should have instance count"
            )
            self._assert(
                hasattr(config, 'processing_volume_size'),
                "Processing config should have volume size"
            )
            
            # Test source directory configuration
            self._assert(
                hasattr(config, 'source_dir'),
                "Processing config should have source directory"
            )
            
        except Exception as e:
            self._assert(False, f"Processing job arguments test failed: {str(e)}")
    
    def test_environment_variables_processing(self):
        """Test environment variable setup for processing."""
        self._log("Testing processing environment variables...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test that builder can handle environment variables
            # This is typically configured in the processing job
            
            # For tabular preprocessing, test specific environment variables
            builder_name = self.builder_class.__name__
            if "TabularPreprocessing" in builder_name:
                self._log("Testing tabular preprocessing environment variables")
                # Tabular preprocessing typically uses specific env vars
                # This would be validated in the actual build process
            
            # Test basic environment variable support
            config = builder.config
            if hasattr(config, 'get_script_path'):
                script_path = config.get_script_path()
                self._assert(script_path is not None, "Script path should be available")
            
        except Exception as e:
            self._assert(False, f"Environment variables test failed: {str(e)}")
    
    def test_property_files_configuration(self):
        """Test property files configuration for processing."""
        self._log("Testing property files configuration...")
        
        try:
            builder = self._create_builder_instance()
            
            # Test property file handling
            # Processing steps often use property files for configuration
            
            config = builder.config
            
            # Test that configuration supports property files
            if hasattr(config, 'get_script_contract'):
                contract = config.get_script_contract()
                # Contract may be None, which is acceptable
                self._log(f"Script contract: {contract}")
            
            # Test pipeline configuration
            self._assert(
                hasattr(config, 'pipeline_s3_loc'),
                "Config should have pipeline S3 location"
            )
            
        except Exception as e:
            self._assert(False, f"Property files configuration test failed: {str(e)}")
    
    def test_processing_code_handling(self):
        """Test processing code and script handling."""
        self._log("Testing processing code handling...")
        
        try:
            builder = self._create_builder_instance()
            config = builder.config
            
            # Test script configuration
            self._assert(
                hasattr(config, 'processing_entry_point'),
                "Processing config should have entry point"
            )
            
            # Test source directory
            self._assert(
                hasattr(config, 'source_dir'),
                "Processing config should have source directory"
            )
            
            # Test image URI configuration
            if hasattr(config, 'get_image_uri'):
                image_uri = config.get_image_uri()
                self._assert(image_uri is not None, "Image URI should be available")
                self._log(f"Processing image URI: {image_uri}")
            
            # Test framework-specific code handling
            framework = self.step_info.get('framework')
            if framework:
                self._log(f"Testing {framework} code handling")
                
                if framework in ["sklearn", "xgboost"]:
                    self._assert(
                        hasattr(config, 'py_version'),
                        f"{framework} processor should have Python version"
                    )
            
        except Exception as e:
            self._assert(False, f"Processing code handling test failed: {str(e)}")
    
    def test_processing_step_dependencies(self):
        """Test Processing step dependency handling."""
        self._log("Testing processing step dependencies...")
        
        try:
            builder = self._create_builder_instance()
            expected_deps = self._get_expected_dependencies()
            
            # Test dependency resolution
            self._assert(len(expected_deps) > 0, "Processing step should have dependencies")
            
            # Test specific dependency patterns
            builder_name = self.builder_class.__name__
            if "TabularPreprocessing" in builder_name:
                self._assert(
                    "DATA" in expected_deps,
                    "Tabular preprocessing should depend on DATA"
                )
            elif "ModelEval" in builder_name:
                expected_model_deps = ["model_input", "eval_data_input"]
                for dep in expected_model_deps:
                    if dep in expected_deps:
                        self._log(f"Found expected model evaluation dependency: {dep}")
            
            # Log all dependencies
            if self.verbose:
                self._log(f"Processing dependencies: {expected_deps}")
            
        except Exception as e:
            self._assert(False, f"Processing dependencies test failed: {str(e)}")
