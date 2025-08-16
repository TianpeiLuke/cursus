"""
Processing Step Builder Validation Test Suite.

This module provides comprehensive validation for Processing step builders using
a modular 4-level testing approach:

Level 1: Interface Tests - Basic interface and inheritance validation
Level 2: Specification Tests - Specification and contract compliance
Level 3: Path Mapping Tests - Input/output path mapping validation  
Level 4: Integration Tests - End-to-end step creation and system integration

The tests are designed to validate Processing-specific patterns including:
- SKLearnProcessor vs XGBoostProcessor usage
- Pattern A (direct ProcessingStep) vs Pattern B (processor.run + step_args)
- Job type-based specification loading
- Container path mapping from contracts
- Environment variable handling
- Special input patterns (local paths, file uploads)
"""

from typing import Dict, Any, List, Optional
from unittest.mock import Mock

from .processing_interface_tests import ProcessingInterfaceTests
from .processing_specification_tests import ProcessingSpecificationTests
from .processing_step_creation_tests import ProcessingStepCreationTests
from .processing_integration_tests import ProcessingIntegrationTests
from ..universal_test import UniversalStepBuilderTest


class ProcessingStepBuilderTest(UniversalStepBuilderTest):
    """
    Comprehensive Processing step builder validation test suite.
    
    This class orchestrates all 4 levels of Processing-specific validation tests
    using the modular test structure. It extends the UniversalStepBuilderTest
    to provide Processing-specific testing capabilities.
    """
    
    def __init__(self, builder_class, step_info: Optional[Dict[str, Any]] = None, 
                 enable_scoring: bool = False, enable_structured_reporting: bool = False):
        """
        Initialize Processing step builder test suite.
        
        Args:
            builder_class: The Processing step builder class to test
            step_info: Optional step information dictionary
            enable_scoring: Whether to enable scoring functionality
            enable_structured_reporting: Whether to enable structured reporting
        """
        # Set Processing-specific step info
        if step_info is None:
            step_info = {
                'sagemaker_step_type': 'Processing',
                'step_category': 'processing',
                'processor_types': ['SKLearnProcessor', 'XGBoostProcessor'],
                'creation_patterns': ['Pattern A (Direct ProcessingStep)', 'Pattern B (processor.run + step_args)'],
                'common_job_types': ['training', 'validation', 'testing', 'calibration'],
                'special_features': ['local_path_override', 'file_upload', 's3_path_validation']
            }
        
        # Store step_info for processing-specific use
        self.step_info = step_info
        
        # Initialize parent class without step_info parameter
        super().__init__(
            builder_class=builder_class,
            enable_scoring=enable_scoring,
            enable_structured_reporting=enable_structured_reporting
        )
        
        # Initialize Processing-specific test levels
        self._initialize_processing_test_levels()
    
    def _initialize_processing_test_levels(self) -> None:
        """Initialize Processing-specific test level instances."""
        # Level 1: Processing Interface Tests
        self.level1_tester = ProcessingInterfaceTests(
            builder_class=self.builder_class,
            step_info=self.step_info
        )
        
        # Level 2: Processing Specification Tests
        self.level2_tester = ProcessingSpecificationTests(
            builder_class=self.builder_class,
            step_info=self.step_info
        )
        
        # Level 3: Processing Step Creation Tests
        self.level3_tester = ProcessingStepCreationTests(
            builder_class=self.builder_class,
            step_info=self.step_info
        )
        
        # Level 4: Processing Integration Tests
        self.level4_tester = ProcessingIntegrationTests(
            builder_class=self.builder_class,
            step_info=self.step_info
        )
        
        # Override the base test levels with Processing-specific ones
        self.test_levels = {
            1: self.level1_tester,
            2: self.level2_tester,
            3: self.level3_tester,
            4: self.level4_tester
        }
    
    def get_processing_specific_info(self) -> Dict[str, Any]:
        """Get Processing-specific information for reporting."""
        return {
            'step_type': 'Processing',
            'supported_processors': ['SKLearnProcessor', 'XGBoostProcessor'],
            'creation_patterns': {
                'pattern_a': 'Direct ProcessingStep creation (SKLearnProcessor)',
                'pattern_b': 'processor.run() + step_args (XGBoostProcessor)'
            },
            'job_type_support': 'Multi-job-type (training/validation/testing/calibration)',
            'special_features': {
                'local_path_override': 'Package step pattern for inference scripts',
                'file_upload': 'DummyTraining step pattern for model/config upload',
                's3_path_validation': 'S3 URI normalization and validation',
                'environment_variables': 'JSON serialization for complex configurations'
            },
            'container_paths': {
                'input_base': '/opt/ml/processing/input',
                'output_base': '/opt/ml/processing/output',
                'code_base': '/opt/ml/processing/input/code'
            }
        }
    
    def run_processing_validation(self, levels: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run Processing-specific validation tests.
        
        Args:
            levels: Optional list of test levels to run (1-4). If None, runs all levels.
            
        Returns:
            Dictionary containing test results and Processing-specific information
        """
        if levels is None:
            levels = [1, 2, 3, 4]
        
        # Run the standard validation
        results = self.run_all_tests()
        
        # Add Processing-specific information
        if isinstance(results, dict):
            results['processing_info'] = self.get_processing_specific_info()
            results['test_suite'] = 'ProcessingStepBuilderTest'
            
            # Add level-specific summaries
            if 'test_results' in results:
                test_results = results['test_results']
                
                # Level 1 summary
                if 1 in levels and 'level_1' in test_results:
                    level1_results = test_results['level_1']
                    level1_results['summary'] = 'Processing interface and inheritance validation'
                    level1_results['focus'] = 'Processor creation methods, framework-specific attributes, step creation patterns'
                
                # Level 2 summary
                if 2 in levels and 'level_2' in test_results:
                    level2_results = test_results['level_2']
                    level2_results['summary'] = 'Processing specification and contract compliance'
                    level2_results['focus'] = 'Job type specification loading, environment variables, processor configuration'
                
                # Level 3 summary
                if 3 in levels and 'level_3' in test_results:
                    level3_results = test_results['level_3']
                    level3_results['summary'] = 'Processing path mapping and property path validation'
                    level3_results['focus'] = 'ProcessingInput/Output creation, container paths, S3 validation, special patterns'
                
                # Level 4 summary
                if 4 in levels and 'level_4' in test_results:
                    level4_results = test_results['level_4']
                    level4_results['summary'] = 'Processing integration and end-to-end workflow'
                    level4_results['focus'] = 'Complete step creation, Pattern A/B validation, dependency resolution'
        
        return results
    
    def validate_processor_type(self, expected_processor: str = None) -> Dict[str, Any]:
        """
        Validate the processor type used by the Processing builder.
        
        Args:
            expected_processor: Expected processor type ('SKLearnProcessor' or 'XGBoostProcessor')
            
        Returns:
            Validation results for processor type
        """
        results = {
            'processor_validation': True,
            'processor_type': None,
            'creation_pattern': None,
            'validation_details': []
        }
        
        try:
            # Create a test instance to check processor type
            config = Mock()
            config.processing_framework_version = "0.23-1"
            config.processing_instance_type_large = "ml.m5.xlarge"
            config.processing_instance_type_small = "ml.m5.large"
            config.use_large_processing_instance = False
            config.processing_instance_count = 1
            config.processing_volume_size = 30
            
            builder = self.builder_class(config=config)
            builder.role = "test-role"
            builder.session = Mock()
            
            if hasattr(builder, '_create_processor'):
                processor = builder._create_processor()
                processor_type = type(processor).__name__
                
                results['processor_type'] = processor_type
                results['validation_details'].append(f"Detected processor type: {processor_type}")
                
                # Determine creation pattern
                if processor_type == 'SKLearnProcessor':
                    results['creation_pattern'] = 'Pattern A (Direct ProcessingStep)'
                elif processor_type == 'XGBoostProcessor':
                    results['creation_pattern'] = 'Pattern B (processor.run + step_args)'
                else:
                    results['creation_pattern'] = 'Unknown pattern'
                
                # Validate against expected processor if provided
                if expected_processor:
                    if processor_type == expected_processor:
                        results['validation_details'].append(f"Processor type matches expected: {expected_processor}")
                    else:
                        results['processor_validation'] = False
                        results['validation_details'].append(f"Processor type mismatch. Expected: {expected_processor}, Got: {processor_type}")
                
            else:
                results['processor_validation'] = False
                results['validation_details'].append("No _create_processor method found")
                
        except Exception as e:
            results['processor_validation'] = False
            results['validation_details'].append(f"Processor validation failed: {str(e)}")
        
        return results
    
    def validate_job_type_support(self, job_types: List[str] = None) -> Dict[str, Any]:
        """
        Validate job type support for multi-job-type Processing builders.
        
        Args:
            job_types: List of job types to test. Defaults to common Processing job types.
            
        Returns:
            Validation results for job type support
        """
        if job_types is None:
            job_types = ['training', 'validation', 'testing', 'calibration']
        
        results = {
            'job_type_support': True,
            'supported_job_types': [],
            'unsupported_job_types': [],
            'validation_details': []
        }
        
        for job_type in job_types:
            try:
                config = Mock()
                config.job_type = job_type
                
                # Try to create builder with this job type
                builder = self.builder_class(config=config)
                
                results['supported_job_types'].append(job_type)
                results['validation_details'].append(f"Job type '{job_type}' supported")
                
            except Exception as e:
                results['unsupported_job_types'].append(job_type)
                results['validation_details'].append(f"Job type '{job_type}' not supported: {str(e)}")
        
        # Overall support validation
        if results['unsupported_job_types']:
            results['job_type_support'] = len(results['supported_job_types']) > 0
        
        return results
    
    def generate_processing_report(self, include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive Processing step builder validation report.
        
        Args:
            include_recommendations: Whether to include improvement recommendations
            
        Returns:
            Comprehensive validation report
        """
        # Run full validation
        validation_results = self.run_processing_validation()
        
        # Add processor type validation
        processor_validation = self.validate_processor_type()
        
        # Add job type support validation
        job_type_validation = self.validate_job_type_support()
        
        # Compile comprehensive report
        report = {
            'builder_class': self.builder_class.__name__,
            'test_suite': 'ProcessingStepBuilderTest',
            'validation_timestamp': self._get_timestamp(),
            'validation_results': validation_results,
            'processor_validation': processor_validation,
            'job_type_validation': job_type_validation,
            'processing_info': self.get_processing_specific_info()
        }
        
        # Add recommendations if requested
        if include_recommendations:
            report['recommendations'] = self._generate_processing_recommendations(
                validation_results, processor_validation, job_type_validation
            )
        
        return report
    
    def _generate_processing_recommendations(self, validation_results: Dict[str, Any], 
                                           processor_validation: Dict[str, Any],
                                           job_type_validation: Dict[str, Any]) -> List[str]:
        """Generate Processing-specific improvement recommendations."""
        recommendations = []
        
        # Processor-specific recommendations
        if not processor_validation.get('processor_validation', True):
            recommendations.append("Fix processor type validation issues")
        
        processor_type = processor_validation.get('processor_type')
        if processor_type == 'SKLearnProcessor':
            recommendations.append("Consider implementing Pattern A validation for SKLearnProcessor")
        elif processor_type == 'XGBoostProcessor':
            recommendations.append("Consider implementing Pattern B validation for XGBoostProcessor")
        
        # Job type recommendations
        if not job_type_validation.get('job_type_support', True):
            unsupported = job_type_validation.get('unsupported_job_types', [])
            if unsupported:
                recommendations.append(f"Consider adding support for job types: {', '.join(unsupported)}")
        
        # Level-specific recommendations
        if isinstance(validation_results, dict) and 'test_results' in validation_results:
            test_results = validation_results['test_results']
            
            for level, level_results in test_results.items():
                if isinstance(level_results, dict) and level_results.get('failed_tests'):
                    failed_tests = level_results['failed_tests']
                    if failed_tests:
                        recommendations.append(f"Address {len(failed_tests)} failed tests in {level}")
        
        # Processing-specific feature recommendations
        recommendations.extend([
            "Ensure proper S3 path validation and normalization",
            "Implement comprehensive environment variable handling",
            "Validate container path mapping from contracts",
            "Test both Pattern A and Pattern B step creation if applicable"
        ])
        
        return recommendations
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for reporting."""
        from datetime import datetime
        return datetime.now().isoformat()


# Convenience function for quick Processing builder validation
def validate_processing_builder(builder_class, enable_scoring: bool = False, 
                              enable_structured_reporting: bool = False) -> Dict[str, Any]:
    """
    Convenience function to validate a Processing step builder.
    
    Args:
        builder_class: The Processing step builder class to validate
        enable_scoring: Whether to enable scoring functionality
        enable_structured_reporting: Whether to enable structured reporting
        
    Returns:
        Comprehensive validation results
    """
    tester = ProcessingStepBuilderTest(
        builder_class=builder_class,
        enable_scoring=enable_scoring,
        enable_structured_reporting=enable_structured_reporting
    )
    
    return tester.generate_processing_report()


# Convenience function for processor type validation
def validate_processor_type(builder_class, expected_processor: str = None) -> Dict[str, Any]:
    """
    Convenience function to validate processor type for a Processing builder.
    
    Args:
        builder_class: The Processing step builder class to validate
        expected_processor: Expected processor type ('SKLearnProcessor' or 'XGBoostProcessor')
        
    Returns:
        Processor type validation results
    """
    tester = ProcessingStepBuilderTest(builder_class=builder_class)
    return tester.validate_processor_type(expected_processor)


# Convenience function for job type support validation
def validate_job_type_support(builder_class, job_types: List[str] = None) -> Dict[str, Any]:
    """
    Convenience function to validate job type support for a Processing builder.
    
    Args:
        builder_class: The Processing step builder class to validate
        job_types: List of job types to test
        
    Returns:
        Job type support validation results
    """
    tester = ProcessingStepBuilderTest(builder_class=builder_class)
    return tester.validate_job_type_support(job_types)
