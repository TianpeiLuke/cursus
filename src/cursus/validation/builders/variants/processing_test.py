"""
Enhanced Processing step builder test variant with 4-level hierarchy.

This module provides comprehensive validation for Processing steps using the enhanced
4-level test hierarchy based on the patterns identified in Processing Step Builder
Patterns analysis.
"""

from typing import Dict, List, Any
from ..base_test import UniversalStepBuilderTestBase
from .processing_interface_test import ProcessingInterfaceTests
from .processing_specification_test import ProcessingSpecificationTests
from .processing_path_mapping_test import ProcessingPathMappingTests
from .processing_integration_test import ProcessingIntegrationTests


class ProcessingStepBuilderTest(UniversalStepBuilderTestBase):
    """
    Enhanced Processing step builder test with 4-level hierarchy.
    
    This class orchestrates the 4-level testing approach for Processing steps:
    - Level 1: Interface Tests (ProcessingInterfaceTests)
    - Level 2: Specification Tests (ProcessingSpecificationTests)  
    - Level 3: Path Mapping Tests (ProcessingPathMappingTests)
    - Level 4: Integration Tests (ProcessingIntegrationTests)
    
    Based on the patterns identified in Processing Step Builder Patterns analysis.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with 4-level test hierarchy."""
        # Initialize 4-level test hierarchy first
        self._init_test_levels_early(*args, **kwargs)
        
        # Then call parent init
        super().__init__(*args, **kwargs)
    
    def _init_test_levels_early(self, *args, **kwargs) -> None:
        """Initialize the 4-level test hierarchy early (before parent init)."""
        # Extract parameters that will be available after parent init
        builder_class = kwargs.get('builder_class')
        verbose = kwargs.get('verbose', False)
        
        # Create placeholder objects that will be properly initialized later
        self.interface_tests = None
        self.specification_tests = None
        self.path_mapping_tests = None
        self.integration_tests = None
    
    def _init_test_levels(self) -> None:
        """Initialize the 4-level test hierarchy."""
        # Level 1: Interface Tests
        self.interface_tests = ProcessingInterfaceTests(
            builder_class=self.builder_class,
            mock_factory=self.mock_factory,
            step_info=self.step_info,
            verbose=self.verbose
        )
        
        # Level 2: Specification Tests
        self.specification_tests = ProcessingSpecificationTests(
            builder_class=self.builder_class,
            mock_factory=self.mock_factory,
            step_info=self.step_info,
            verbose=self.verbose
        )
        
        # Level 3: Path Mapping Tests
        self.path_mapping_tests = ProcessingPathMappingTests(
            builder_class=self.builder_class,
            mock_factory=self.mock_factory,
            step_info=self.step_info,
            verbose=self.verbose
        )
        
        # Level 4: Integration Tests
        self.integration_tests = ProcessingIntegrationTests(
            builder_class=self.builder_class,
            mock_factory=self.mock_factory,
            step_info=self.step_info,
            verbose=self.verbose
        )
    
    def get_step_type_specific_tests(self) -> List[str]:
        """Return Processing step-specific test methods from all levels."""
        all_tests = []
        
        # Collect tests from all 4 levels
        all_tests.extend([f"level1_{test}" for test in self.interface_tests.get_step_type_specific_tests()])
        all_tests.extend([f"level2_{test}" for test in self.specification_tests.get_step_type_specific_tests()])
        all_tests.extend([f"level3_{test}" for test in self.path_mapping_tests.get_step_type_specific_tests()])
        all_tests.extend([f"level4_{test}" for test in self.integration_tests.get_step_type_specific_tests()])
        
        return all_tests
    
    def _configure_step_type_mocks(self) -> None:
        """Configure Processing step-specific mock objects."""
        # Get Processing-specific mocks from factory
        self.step_type_mocks = self.mock_factory.create_step_type_mocks()
        
        # Log Processing step info if verbose
        if self.verbose:
            framework = self.step_info.get('framework', 'Unknown')
            pattern = self.step_info.get('step_creation_pattern', 'Unknown')
            job_types = self.step_info.get('supported_job_types', [])
            special_patterns = self.step_info.get('special_input_patterns', [])
            
            self._log(f"Processing step detected:")
            self._log(f"  Framework: {framework}")
            self._log(f"  Step creation pattern: {pattern}")
            self._log(f"  Supported job types: {job_types}")
            self._log(f"  Special input patterns: {special_patterns}")
            
        # Set up Processing-specific mock attributes
        self.mock_processor = self.step_type_mocks.get('processor_class')
        self.mock_processing_input = self.step_type_mocks.get('processing_input')
        self.mock_processing_output = self.step_type_mocks.get('processing_output')
        
        # Initialize test levels if not already done
        if self.interface_tests is None:
            self._init_test_levels()
        
        # Configure mocks for all test levels
        if self.interface_tests:
            self.interface_tests._configure_step_type_mocks()
        if self.specification_tests:
            self.specification_tests._configure_step_type_mocks()
        if self.path_mapping_tests:
            self.path_mapping_tests._configure_step_type_mocks()
        if self.integration_tests:
            self.integration_tests._configure_step_type_mocks()
    
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate Processing step-specific requirements across all levels."""
        validation_results = {
            "is_processing_step": self.step_info.get("sagemaker_step_type") == "Processing",
            "processor_framework_detected": self.step_info.get("framework") is not None,
            "step_creation_pattern_detected": self.step_info.get("step_creation_pattern") is not None,
            "processing_mocks_created": len(self.step_type_mocks) > 0,
            "expected_processing_dependencies": len(self._get_expected_dependencies()) > 0
        }
        
        # Collect validation results from all levels
        validation_results.update({
            "level1_requirements": self.interface_tests._validate_step_type_requirements(),
            "level2_requirements": self.specification_tests._validate_step_type_requirements(),
            "level3_requirements": self.path_mapping_tests._validate_step_type_requirements(),
            "level4_requirements": self.integration_tests._validate_step_type_requirements()
        })
        
        return validation_results
    
    # Level 1: Interface Tests (delegated methods)
    
    def level1_test_processor_creation_method(self):
        """Level 1: Test processor creation method."""
        return self.interface_tests.test_processor_creation_method()
    
    def level1_test_processing_configuration_attributes(self):
        """Level 1: Test processing configuration attributes."""
        return self.interface_tests.test_processing_configuration_attributes()
    
    def level1_test_framework_specific_methods(self):
        """Level 1: Test framework-specific methods."""
        return self.interface_tests.test_framework_specific_methods()
    
    def level1_test_step_creation_pattern_compliance(self):
        """Level 1: Test step creation pattern compliance."""
        return self.interface_tests.test_step_creation_pattern_compliance()
    
    def level1_test_processing_input_output_methods(self):
        """Level 1: Test processing input/output methods."""
        return self.interface_tests.test_processing_input_output_methods()
    
    def level1_test_environment_variables_method(self):
        """Level 1: Test environment variables method."""
        return self.interface_tests.test_environment_variables_method()
    
    def level1_test_job_arguments_method(self):
        """Level 1: Test job arguments method."""
        return self.interface_tests.test_job_arguments_method()
    
    # Level 2: Specification Tests (delegated methods)
    
    def level2_test_job_type_specification_loading(self):
        """Level 2: Test job type specification loading."""
        return self.specification_tests.test_job_type_specification_loading()
    
    def level2_test_environment_variable_patterns(self):
        """Level 2: Test environment variable patterns."""
        return self.specification_tests.test_environment_variable_patterns()
    
    def level2_test_job_arguments_patterns(self):
        """Level 2: Test job arguments patterns."""
        return self.specification_tests.test_job_arguments_patterns()
    
    def level2_test_specification_driven_inputs(self):
        """Level 2: Test specification-driven inputs."""
        return self.specification_tests.test_specification_driven_inputs()
    
    def level2_test_specification_driven_outputs(self):
        """Level 2: Test specification-driven outputs."""
        return self.specification_tests.test_specification_driven_outputs()
    
    def level2_test_contract_path_mapping(self):
        """Level 2: Test contract path mapping."""
        return self.specification_tests.test_contract_path_mapping()
    
    def level2_test_multi_job_type_support(self):
        """Level 2: Test multi-job-type support."""
        return self.specification_tests.test_multi_job_type_support()
    
    def level2_test_framework_specific_specifications(self):
        """Level 2: Test framework-specific specifications."""
        return self.specification_tests.test_framework_specific_specifications()
    
    # Level 3: Path Mapping Tests (delegated methods)
    
    def level3_test_processing_input_creation(self):
        """Level 3: Test ProcessingInput creation."""
        return self.path_mapping_tests.test_processing_input_creation()
    
    def level3_test_processing_output_creation(self):
        """Level 3: Test ProcessingOutput creation."""
        return self.path_mapping_tests.test_processing_output_creation()
    
    def level3_test_container_path_mapping(self):
        """Level 3: Test container path mapping."""
        return self.path_mapping_tests.test_container_path_mapping()
    
    def level3_test_special_input_handling(self):
        """Level 3: Test special input handling."""
        return self.path_mapping_tests.test_special_input_handling()
    
    def level3_test_s3_path_normalization(self):
        """Level 3: Test S3 path normalization."""
        return self.path_mapping_tests.test_s3_path_normalization()
    
    def level3_test_file_upload_patterns(self):
        """Level 3: Test file upload patterns."""
        return self.path_mapping_tests.test_file_upload_patterns()
    
    def level3_test_local_path_override_patterns(self):
        """Level 3: Test local path override patterns."""
        return self.path_mapping_tests.test_local_path_override_patterns()
    
    def level3_test_dependency_input_extraction(self):
        """Level 3: Test dependency input extraction."""
        return self.path_mapping_tests.test_dependency_input_extraction()
    
    # Level 4: Integration Tests (delegated methods)
    
    def level4_test_step_creation_pattern_execution(self):
        """Level 4: Test step creation pattern execution."""
        return self.integration_tests.test_step_creation_pattern_execution()
    
    def level4_test_framework_specific_step_creation(self):
        """Level 4: Test framework-specific step creation."""
        return self.integration_tests.test_framework_specific_step_creation()
    
    def level4_test_processing_dependency_resolution(self):
        """Level 4: Test processing dependency resolution."""
        return self.integration_tests.test_processing_dependency_resolution()
    
    def level4_test_step_name_generation(self):
        """Level 4: Test step name generation."""
        return self.integration_tests.test_step_name_generation()
    
    def level4_test_cache_configuration(self):
        """Level 4: Test cache configuration."""
        return self.integration_tests.test_cache_configuration()
    
    def level4_test_step_dependencies_handling(self):
        """Level 4: Test step dependencies handling."""
        return self.integration_tests.test_step_dependencies_handling()
    
    def level4_test_end_to_end_step_creation(self):
        """Level 4: Test end-to-end step creation."""
        return self.integration_tests.test_end_to_end_step_creation()
    
    def level4_test_specification_attachment(self):
        """Level 4: Test specification attachment."""
        return self.integration_tests.test_specification_attachment()
    
    # Legacy compatibility methods (for backward compatibility)
    
    def test_processor_creation(self):
        """Legacy: Validate processor creation patterns."""
        return self.level1_test_processor_creation_method()
    
    def test_processing_inputs_outputs(self):
        """Legacy: Test ProcessingInput and ProcessingOutput handling."""
        return self.level3_test_processing_input_creation()
    
    def test_processing_job_arguments(self):
        """Legacy: Test processing job arguments construction."""
        return self.level2_test_job_arguments_patterns()
    
    def test_environment_variables_processing(self):
        """Legacy: Test environment variable setup for processing."""
        return self.level2_test_environment_variable_patterns()
    
    def test_property_files_configuration(self):
        """Legacy: Test property files configuration for processing."""
        return self.level2_test_contract_path_mapping()
    
    def test_processing_code_handling(self):
        """Legacy: Test processing code and script handling."""
        return self.level1_test_step_creation_pattern_compliance()
    
    def test_processing_step_dependencies(self):
        """Legacy: Test Processing step dependency handling."""
        return self.level4_test_processing_dependency_resolution()
