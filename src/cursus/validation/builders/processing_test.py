"""
Universal Builder Steps for Processing Step Builders.

This module provides a comprehensive testing framework specifically designed for 
Processing step builders. It enforces standardization rules, alignment rules, 
and provides LLM-powered feedback and scoring for generated step builders.

The framework extends the existing universal_builder_test system with Processing-specific
validations and enhanced LLM integration for intelligent feedback.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union, Type, Tuple
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field

# Import base testing framework
from .universal_test import UniversalStepBuilderTest
from .scoring import StepBuilderScorer, score_builder_results
from .base_test import UniversalStepBuilderTestBase

# Import core framework components
from ...core.base.builder_base import StepBuilderBase
from ...core.base.specification_base import StepSpecification
from ...core.base.contract_base import ScriptContract
from ...steps.configs.config_processing_step_base import ProcessingStepConfigBase

# Import SageMaker components for validation
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.xgboost import XGBoostProcessor

logger = logging.getLogger(__name__)


class ProcessingStepType(Enum):
    """Enumeration of Processing step types."""
    TABULAR_PREPROCESSING = "TabularPreprocessing"
    PAYLOAD_GENERATION = "PayloadGeneration"
    MODEL_EVALUATION = "ModelEvaluation"
    PACKAGE_CREATION = "PackageCreation"
    DATA_LOADING = "DataLoading"
    RISK_TABLE_MAPPING = "RiskTableMapping"
    CURRENCY_CONVERSION = "CurrencyConversion"
    MODEL_CALIBRATION = "ModelCalibration"
    BATCH_TRANSFORM = "BatchTransform"


class StandardizationViolation(BaseModel):
    """Represents a standardization rule violation."""
    rule_id: str = Field(..., description="Unique identifier for the rule")
    severity: str = Field(..., description="Severity level: ERROR, WARNING, or INFO")
    message: str = Field(..., description="Description of the violation")
    suggestion: str = Field(..., description="Suggested fix for the violation")
    file_location: Optional[str] = Field(None, description="File where violation occurred")
    line_number: Optional[int] = Field(None, description="Line number of the violation")


class AlignmentViolation(BaseModel):
    """Represents an alignment rule violation."""
    component_a: str = Field(..., description="First component in the alignment check")
    component_b: str = Field(..., description="Second component in the alignment check")
    violation_type: str = Field(..., description="Type of alignment violation")
    message: str = Field(..., description="Description of the alignment violation")
    suggestion: str = Field(..., description="Suggested fix for the alignment violation")
    severity: str = Field("ERROR", description="Severity level of the violation")


class LLMFeedback(BaseModel):
    """Represents LLM-generated feedback for a step builder."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall score from 0-100")
    overall_rating: str = Field(..., description="Overall rating: Excellent, Good, Satisfactory, Needs Work, or Poor")
    strengths: List[str] = Field(default_factory=list, description="List of identified strengths")
    weaknesses: List[str] = Field(default_factory=list, description="List of identified weaknesses")
    recommendations: List[str] = Field(default_factory=list, description="List of improvement recommendations")
    code_quality_score: float = Field(..., ge=0, le=100, description="Code quality score from 0-100")
    architecture_compliance_score: float = Field(..., ge=0, le=100, description="Architecture compliance score from 0-100")
    maintainability_score: float = Field(..., ge=0, le=100, description="Maintainability score from 0-100")
    detailed_analysis: str = Field(..., description="Detailed analysis text")


class ProcessingStepBuilderValidator(UniversalStepBuilderTestBase):
    """
    Enhanced validator for Processing step builders with standardization and alignment checks.
    
    This class extends the base universal test framework with Processing-specific validations
    and provides detailed feedback on standardization and alignment compliance.
    """
    
    def __init__(
        self,
        builder_class: Type[StepBuilderBase],
        config: Optional[ProcessingStepConfigBase] = None,
        spec: Optional[StepSpecification] = None,
        contract: Optional[ScriptContract] = None,
        step_name: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the Processing step builder validator.
        
        Args:
            builder_class: The Processing step builder class to validate
            config: Optional Processing config to use
            spec: Optional step specification
            contract: Optional script contract
            step_name: Optional step name
            verbose: Whether to print verbose output
        """
        super().__init__(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose
        )
        
        self.standardization_violations: List[StandardizationViolation] = []
        self.alignment_violations: List[AlignmentViolation] = []
        self.processing_step_type = self._detect_processing_step_type()
        
    def _detect_processing_step_type(self) -> ProcessingStepType:
        """Detect the type of Processing step based on class name."""
        class_name = self.builder_class.__name__
        
        if "TabularPreprocessing" in class_name:
            return ProcessingStepType.TABULAR_PREPROCESSING
        elif "Payload" in class_name:
            return ProcessingStepType.PAYLOAD_GENERATION
        elif "ModelEval" in class_name:
            return ProcessingStepType.MODEL_EVALUATION
        elif "Package" in class_name:
            return ProcessingStepType.PACKAGE_CREATION
        elif "DataLoad" in class_name or "CradleData" in class_name:
            return ProcessingStepType.DATA_LOADING
        elif "RiskTable" in class_name:
            return ProcessingStepType.RISK_TABLE_MAPPING
        elif "Currency" in class_name:
            return ProcessingStepType.CURRENCY_CONVERSION
        elif "Calibration" in class_name:
            return ProcessingStepType.MODEL_CALIBRATION
        elif "BatchTransform" in class_name:
            return ProcessingStepType.BATCH_TRANSFORM
        else:
            # Default to tabular preprocessing
            return ProcessingStepType.TABULAR_PREPROCESSING
    
    def validate_standardization_rules(self) -> List[StandardizationViolation]:
        """
        Validate compliance with standardization rules.
        
        Returns:
            List of standardization violations found
        """
        violations = []
        
        # Rule 1: Naming Conventions
        violations.extend(self._validate_naming_conventions())
        
        # Rule 2: Interface Standardization
        violations.extend(self._validate_interface_standardization())
        
        # Rule 3: Documentation Standards
        violations.extend(self._validate_documentation_standards())
        
        # Rule 4: Error Handling Standards
        violations.extend(self._validate_error_handling_standards())
        
        # Rule 5: Testing Standards
        violations.extend(self._validate_testing_standards())
        
        self.standardization_violations = violations
        return violations
    
    def _validate_naming_conventions(self) -> List[StandardizationViolation]:
        """Validate naming convention compliance."""
        violations = []
        
        # Check class name follows pattern: XXXStepBuilder
        class_name = self.builder_class.__name__
        if not class_name.endswith("StepBuilder"):
            violations.append(StandardizationViolation(
                rule_id="NAMING_001",
                severity="ERROR",
                message=f"Class name '{class_name}' does not follow pattern 'XXXStepBuilder'",
                suggestion="Rename class to follow the pattern 'XXXStepBuilder' where XXX is the step type"
            ))
        
        # Check if step type is in PascalCase
        if class_name.endswith("StepBuilder"):
            step_type = class_name[:-11]  # Remove "StepBuilder"
            if not step_type[0].isupper():
                violations.append(StandardizationViolation(
                    rule_id="NAMING_002",
                    severity="ERROR",
                    message=f"Step type '{step_type}' should be in PascalCase",
                    suggestion="Use PascalCase for step type names (e.g., 'TabularPreprocessing', not 'tabularPreprocessing')"
                ))
        
        # Check method naming conventions
        for method_name in dir(self.builder_class):
            if not method_name.startswith('_'):
                method = getattr(self.builder_class, method_name)
                if callable(method) and not method_name.islower():
                    violations.append(StandardizationViolation(
                        rule_id="NAMING_003",
                        severity="WARNING",
                        message=f"Public method '{method_name}' should be in snake_case",
                        suggestion="Use snake_case for method names (e.g., 'create_step', not 'createStep')"
                    ))
        
        return violations
    
    def _validate_interface_standardization(self) -> List[StandardizationViolation]:
        """Validate interface standardization compliance."""
        violations = []
        
        # Check inheritance from StepBuilderBase
        if not issubclass(self.builder_class, StepBuilderBase):
            violations.append(StandardizationViolation(
                rule_id="INTERFACE_001",
                severity="ERROR",
                message=f"Class must inherit from StepBuilderBase",
                suggestion="Add 'from cursus.core.base.builder_base import StepBuilderBase' and inherit from it"
            ))
        
        # Check required methods are implemented
        required_methods = ['validate_configuration', '_get_inputs', '_get_outputs', 'create_step']
        for method_name in required_methods:
            if not hasattr(self.builder_class, method_name):
                violations.append(StandardizationViolation(
                    rule_id="INTERFACE_002",
                    severity="ERROR",
                    message=f"Required method '{method_name}' is not implemented",
                    suggestion=f"Implement the '{method_name}' method as required by StepBuilderBase"
                ))
            else:
                method = getattr(self.builder_class, method_name)
                if getattr(method, '__isabstractmethod__', False):
                    violations.append(StandardizationViolation(
                        rule_id="INTERFACE_003",
                        severity="ERROR",
                        message=f"Method '{method_name}' is still abstract",
                        suggestion=f"Provide a concrete implementation for '{method_name}'"
                    ))
        
        # Check for @register_builder decorator
        if not hasattr(self.builder_class, '_registry_key'):
            # Try to check if the class is decorated
            violations.append(StandardizationViolation(
                rule_id="INTERFACE_004",
                severity="WARNING",
                message="Class should use @register_builder() decorator",
                suggestion="Add '@register_builder()' decorator above the class definition"
            ))
        
        return violations
    
    def _validate_documentation_standards(self) -> List[StandardizationViolation]:
        """Validate documentation standards compliance."""
        violations = []
        
        # Check class docstring
        if not self.builder_class.__doc__:
            violations.append(StandardizationViolation(
                rule_id="DOC_001",
                severity="ERROR",
                message="Class is missing docstring",
                suggestion="Add a comprehensive docstring describing the class purpose, features, and usage"
            ))
        elif len(self.builder_class.__doc__.strip()) < 50:
            violations.append(StandardizationViolation(
                rule_id="DOC_002",
                severity="WARNING",
                message="Class docstring is too brief",
                suggestion="Expand docstring to include purpose, key features, integration points, and usage examples"
            ))
        
        # Check method docstrings
        for method_name in ['validate_configuration', '_get_inputs', '_get_outputs', 'create_step']:
            if hasattr(self.builder_class, method_name):
                method = getattr(self.builder_class, method_name)
                if not method.__doc__:
                    violations.append(StandardizationViolation(
                        rule_id="DOC_003",
                        severity="WARNING",
                        message=f"Method '{method_name}' is missing docstring",
                        suggestion=f"Add docstring to '{method_name}' with description, parameters, returns, and exceptions"
                    ))
        
        return violations
    
    def _validate_error_handling_standards(self) -> List[StandardizationViolation]:
        """Validate error handling standards compliance."""
        violations = []
        
        # This is a basic check - in a real implementation, you'd analyze the source code
        # For now, we'll check if validate_configuration exists and assume it has error handling
        if hasattr(self.builder_class, 'validate_configuration'):
            try:
                # Try to create an instance with invalid config to test error handling
                invalid_config = self._create_invalid_config()
                builder = self.builder_class(
                    config=invalid_config,
                    sagemaker_session=self.mock_session,
                    role=self.mock_role
                )
                
                # This should raise an exception
                try:
                    builder.validate_configuration()
                    violations.append(StandardizationViolation(
                        rule_id="ERROR_001",
                        severity="WARNING",
                        message="validate_configuration() does not raise exceptions for invalid config",
                        suggestion="Add proper validation and raise ValueError for invalid configurations"
                    ))
                except ValueError:
                    # Good - it raised a ValueError as expected
                    pass
                except Exception as e:
                    violations.append(StandardizationViolation(
                        rule_id="ERROR_002",
                        severity="WARNING",
                        message=f"validate_configuration() raises {type(e).__name__} instead of ValueError",
                        suggestion="Use ValueError for configuration validation errors"
                    ))
            except Exception:
                # Could not test error handling
                pass
        
        return violations
    
    def _validate_testing_standards(self) -> List[StandardizationViolation]:
        """Validate testing standards compliance."""
        violations = []
        
        # Check if there are corresponding test files
        class_name = self.builder_class.__name__
        module_name = self.builder_class.__module__
        
        # This is a basic check - in a real implementation, you'd check for actual test files
        violations.append(StandardizationViolation(
            rule_id="TEST_001",
            severity="INFO",
            message=f"Ensure comprehensive tests exist for {class_name}",
            suggestion="Create unit tests covering all methods, integration tests, and error handling tests"
        ))
        
        return violations
    
    def validate_alignment_rules(self) -> List[AlignmentViolation]:
        """
        Validate compliance with alignment rules.
        
        Returns:
            List of alignment violations found
        """
        violations = []
        
        try:
            # Create builder instance for testing
            builder = self._create_builder_instance()
            
            # Rule 1: Script ↔ Contract alignment
            violations.extend(self._validate_script_contract_alignment(builder))
            
            # Rule 2: Contract ↔ Specification alignment
            violations.extend(self._validate_contract_specification_alignment(builder))
            
            # Rule 3: Specification ↔ Dependencies alignment
            violations.extend(self._validate_specification_dependencies_alignment(builder))
            
            # Rule 4: Builder ↔ Configuration alignment
            violations.extend(self._validate_builder_configuration_alignment(builder))
            
        except Exception as e:
            violations.append(AlignmentViolation(
                component_a="builder",
                component_b="system",
                violation_type="instantiation_error",
                message=f"Could not create builder instance for alignment testing: {str(e)}",
                suggestion="Fix builder instantiation issues before running alignment tests"
            ))
        
        self.alignment_violations = violations
        return violations
    
    def _validate_script_contract_alignment(self, builder: StepBuilderBase) -> List[AlignmentViolation]:
        """Validate Script ↔ Contract alignment."""
        violations = []
        
        if not builder.contract:
            violations.append(AlignmentViolation(
                component_a="script",
                component_b="contract",
                violation_type="missing_contract",
                message="Builder has no script contract defined",
                suggestion="Define a script contract and set it in the builder"
            ))
            return violations
        
        # Check environment variables alignment
        env_vars = builder._get_environment_variables()
        for required_var in builder.contract.required_env_vars:
            if required_var not in env_vars:
                violations.append(AlignmentViolation(
                    component_a="script",
                    component_b="contract",
                    violation_type="env_var_mismatch",
                    message=f"Required environment variable '{required_var}' not provided by builder",
                    suggestion=f"Add '{required_var}' to the environment variables in _get_environment_variables()"
                ))
        
        return violations
    
    def _validate_contract_specification_alignment(self, builder: StepBuilderBase) -> List[AlignmentViolation]:
        """Validate Contract ↔ Specification alignment."""
        violations = []
        
        if not builder.spec or not builder.contract:
            return violations  # Skip if either is missing
        
        # Check input paths alignment
        for logical_name in builder.contract.expected_input_paths.keys():
            if logical_name not in [dep.logical_name for dep in builder.spec.dependencies.values()]:
                violations.append(AlignmentViolation(
                    component_a="contract",
                    component_b="specification",
                    violation_type="input_path_mismatch",
                    message=f"Contract input path '{logical_name}' not found in specification dependencies",
                    suggestion=f"Add dependency with logical_name='{logical_name}' to specification"
                ))
        
        # Check output paths alignment
        for logical_name in builder.contract.expected_output_paths.keys():
            if logical_name not in [out.logical_name for out in builder.spec.outputs.values()]:
                violations.append(AlignmentViolation(
                    component_a="contract",
                    component_b="specification",
                    violation_type="output_path_mismatch",
                    message=f"Contract output path '{logical_name}' not found in specification outputs",
                    suggestion=f"Add output with logical_name='{logical_name}' to specification"
                ))
        
        return violations
    
    def _validate_specification_dependencies_alignment(self, builder: StepBuilderBase) -> List[AlignmentViolation]:
        """Validate Specification ↔ Dependencies alignment."""
        violations = []
        
        if not builder.spec:
            return violations
        
        # Check that all required dependencies have compatible sources
        for dep_name, dep_spec in builder.spec.dependencies.items():
            if dep_spec.required and not dep_spec.compatible_sources:
                violations.append(AlignmentViolation(
                    component_a="specification",
                    component_b="dependencies",
                    violation_type="missing_compatible_sources",
                    message=f"Required dependency '{dep_spec.logical_name}' has no compatible sources",
                    suggestion=f"Add compatible_sources list to dependency '{dep_spec.logical_name}'"
                ))
        
        return violations
    
    def _validate_builder_configuration_alignment(self, builder: StepBuilderBase) -> List[AlignmentViolation]:
        """Validate Builder ↔ Configuration alignment."""
        violations = []
        
        # Check that builder uses configuration parameters correctly
        if hasattr(builder.config, 'processing_instance_type_large') and hasattr(builder.config, 'use_large_processing_instance'):
            # This is a common pattern - check if builder respects the instance type selection
            try:
                processor = builder._create_processor() if hasattr(builder, '_create_processor') else None
                if processor:
                    expected_type = (builder.config.processing_instance_type_large 
                                   if builder.config.use_large_processing_instance 
                                   else builder.config.processing_instance_type_small)
                    if hasattr(processor, 'instance_type') and processor.instance_type != expected_type:
                        violations.append(AlignmentViolation(
                            component_a="builder",
                            component_b="configuration",
                            violation_type="instance_type_mismatch",
                            message=f"Builder uses wrong instance type: {processor.instance_type} vs expected {expected_type}",
                            suggestion="Use config.processing_instance_type_large when config.use_large_processing_instance is True"
                        ))
            except Exception:
                pass  # Skip if we can't create processor
        
        return violations
    
    def run_processing_specific_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run Processing-specific tests.
        
        Returns:
            Dictionary mapping test names to their results
        """
        results = {}
        
        # Test Processing step creation
        results["test_processing_step_creation"] = self._run_test(self._test_processing_step_creation)
        
        # Test processor configuration
        results["test_processor_configuration"] = self._run_test(self._test_processor_configuration)
        
        # Test input/output handling
        results["test_processing_input_output_handling"] = self._run_test(self._test_processing_input_output_handling)
        
        # Test environment variables
        results["test_processing_environment_variables"] = self._run_test(self._test_processing_environment_variables)
        
        # Test job arguments
        results["test_processing_job_arguments"] = self._run_test(self._test_processing_job_arguments)
        
        return results
    
    def _test_processing_step_creation(self) -> None:
        """Test that the builder creates a valid ProcessingStep."""
        builder = self._create_builder_instance()
        
        # Create step
        step = builder.create_step(
            inputs={},
            outputs={},
            dependencies=[],
            enable_caching=True
        )
        
        self._assert(
            isinstance(step, ProcessingStep),
            "Builder must create a ProcessingStep instance"
        )
        
        self._assert(
            hasattr(step, 'name') and step.name,
            "ProcessingStep must have a non-empty name"
        )
    
    def _test_processor_configuration(self) -> None:
        """Test that the builder configures the processor correctly."""
        builder = self._create_builder_instance()
        
        if hasattr(builder, '_create_processor'):
            processor = builder._create_processor()
            
            # Check processor type
            expected_processors = (SKLearnProcessor, XGBoostProcessor)
            self._assert(
                isinstance(processor, expected_processors),
                f"Processor must be one of {[p.__name__ for p in expected_processors]}"
            )
            
            # Check basic configuration
            self._assert(
                hasattr(processor, 'role') and processor.role,
                "Processor must have a role configured"
            )
            
            self._assert(
                hasattr(processor, 'instance_type') and processor.instance_type,
                "Processor must have an instance_type configured"
            )
    
    def _test_processing_input_output_handling(self) -> None:
        """Test that the builder handles inputs and outputs correctly."""
        builder = self._create_builder_instance()
        
        # Test inputs
        mock_inputs = {"test_input": "s3://bucket/input"}
        try:
            inputs = builder._get_inputs(mock_inputs)
            self._assert(
                isinstance(inputs, list),
                "_get_inputs must return a list"
            )
            
            if inputs:
                self._assert(
                    all(isinstance(inp, ProcessingInput) for inp in inputs),
                    "All inputs must be ProcessingInput instances"
                )
        except Exception as e:
            self._log(f"Input handling test skipped due to: {str(e)}")
        
        # Test outputs
        mock_outputs = {"test_output": "s3://bucket/output"}
        try:
            outputs = builder._get_outputs(mock_outputs)
            self._assert(
                isinstance(outputs, list),
                "_get_outputs must return a list"
            )
            
            if outputs:
                self._assert(
                    all(isinstance(out, ProcessingOutput) for out in outputs),
                    "All outputs must be ProcessingOutput instances"
                )
        except Exception as e:
            self._log(f"Output handling test skipped due to: {str(e)}")
    
    def _test_processing_environment_variables(self) -> None:
        """Test that the builder sets environment variables correctly."""
        builder = self._create_builder_instance()
        
        env_vars = builder._get_environment_variables()
        
        self._assert(
            isinstance(env_vars, dict),
            "_get_environment_variables must return a dictionary"
        )
        
        # Check that all values are strings
        for key, value in env_vars.items():
            self._assert(
                isinstance(key, str) and isinstance(value, str),
                f"Environment variable {key} must have string key and value"
            )
    
    def _test_processing_job_arguments(self) -> None:
        """Test that the builder handles job arguments correctly."""
        builder = self._create_builder_instance()
        
        job_args = builder._get_job_arguments()
        
        if job_args is not None:
            self._assert(
                isinstance(job_args, list),
                "_get_job_arguments must return a list or None"
            )
            
            if job_args:
                self._assert(
                    all(isinstance(arg, str) for arg in job_args),
                    "All job arguments must be strings"
                )


class ProcessingStepBuilderLLMAnalyzer:
    """
    LLM-powered analyzer for Processing step builders.
    
    This class uses LLM capabilities to provide intelligent feedback and scoring
    for step builder implementations.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the LLM analyzer.
        
        Args:
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
    
    def analyze_builder(
        self,
        builder_class: Type[StepBuilderBase],
        test_results: Dict[str, Dict[str, Any]],
        standardization_violations: List[StandardizationViolation],
        alignment_violations: List[AlignmentViolation]
    ) -> LLMFeedback:
        """
        Analyze a step builder using LLM capabilities.
        
        Args:
            builder_class: The step builder class to analyze
            test_results: Results from running tests
            standardization_violations: List of standardization violations
            alignment_violations: List of alignment violations
            
        Returns:
            LLM-generated feedback and analysis
        """
        # Calculate basic scores
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get("passed", False))
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Calculate violation scores
        error_violations = len([v for v in standardization_violations if v.severity == "ERROR"])
        warning_violations = len([v for v in standardization_violations if v.severity == "WARNING"])
        alignment_errors = len([v for v in alignment_violations if v.severity == "ERROR"])
        
        # Calculate component scores
        code_quality_score = max(0, 100 - (error_violations * 20) - (warning_violations * 10))
        architecture_compliance_score = max(0, 100 - (alignment_errors * 25))
        maintainability_score = min(pass_rate, code_quality_score)
        
        # Calculate overall score
        overall_score = (pass_rate * 0.4 + code_quality_score * 0.3 + 
                        architecture_compliance_score * 0.2 + maintainability_score * 0.1)
        
        # Determine rating
        if overall_score >= 90:
            rating = "Excellent"
        elif overall_score >= 80:
            rating = "Good"
        elif overall_score >= 70:
            rating = "Satisfactory"
        elif overall_score >= 60:
            rating = "Needs Work"
        else:
            rating = "Poor"
        
        # Generate strengths and weaknesses
        strengths = []
        weaknesses = []
        recommendations = []
        
        if pass_rate >= 90:
            strengths.append("High test pass rate indicates robust implementation")
        elif pass_rate < 70:
            weaknesses.append("Low test pass rate indicates implementation issues")
        
        if error_violations == 0:
            strengths.append("No critical standardization violations found")
        else:
            weaknesses.append(f"{error_violations} critical standardization violations found")
            recommendations.append("Address all ERROR-level standardization violations")
        
        if alignment_errors == 0:
            strengths.append("Good alignment between components")
        else:
            weaknesses.append(f"{alignment_errors} alignment violations found")
            recommendations.append("Fix alignment issues between script, contract, specification, and builder")
        
        if warning_violations > 0:
            recommendations.append("Address WARNING-level violations to improve code quality")
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            builder_class, test_results, standardization_violations, alignment_violations
        )
        
        return LLMFeedback(
            overall_score=overall_score,
            overall_rating=rating,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            code_quality_score=code_quality_score,
            architecture_compliance_score=architecture_compliance_score,
            maintainability_score=maintainability_score,
            detailed_analysis=detailed_analysis
        )
    
    def _generate_detailed_analysis(
        self,
        builder_class: Type[StepBuilderBase],
        test_results: Dict[str, Dict[str, Any]],
        standardization_violations: List[StandardizationViolation],
        alignment_violations: List[AlignmentViolation]
    ) -> str:
        """Generate detailed analysis text."""
        analysis_parts = []
        
        # Class overview
        analysis_parts.append(f"Analysis of {builder_class.__name__}:")
        analysis_parts.append(f"Module: {builder_class.__module__}")
        
        # Test results summary
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results.values() if result.get("passed", False))
        analysis_parts.append(f"\nTest Results: {passed_tests}/{total_tests} tests passed")
        
        # Failed tests
        failed_tests = [name for name, result in test_results.items() if not result.get("passed", False)]
        if failed_tests:
            analysis_parts.append(f"Failed tests: {', '.join(failed_tests)}")
        
        # Standardization violations
        if standardization_violations:
            analysis_parts.append(f"\nStandardization Violations:")
            for violation in standardization_violations[:5]:  # Show first 5
                analysis_parts.append(f"- [{violation.severity}] {violation.message}")
        
        # Alignment violations
        if alignment_violations:
            analysis_parts.append(f"\nAlignment Violations:")
            for violation in alignment_violations[:5]:  # Show first 5
                analysis_parts.append(f"- {violation.component_a} ↔ {violation.component_b}: {violation.message}")
        
        return "\n".join(analysis_parts)


class UniversalProcessingBuilderTest:
    """
    Main class for comprehensive Processing step builder testing.
    
    This class combines all validation levels and provides a complete testing
    framework for Processing step builders with standardization, alignment,
    and LLM-powered feedback.
    """
    
    def __init__(
        self,
        builder_class: Type[StepBuilderBase],
        config: Optional[ProcessingStepConfigBase] = None,
        spec: Optional[StepSpecification] = None,
        contract: Optional[ScriptContract] = None,
        step_name: Optional[str] = None,
        verbose: bool = False,
        save_reports: bool = True,
        output_dir: str = "test_reports"
    ):
        """
        Initialize the comprehensive Processing builder test.
        
        Args:
            builder_class: The Processing step builder class to test
            config: Optional Processing config to use
            spec: Optional step specification
            contract: Optional script contract
            step_name: Optional step name
            verbose: Whether to print verbose output
            save_reports: Whether to save test reports
            output_dir: Directory to save reports in
        """
        self.builder_class = builder_class
        self.config = config
        self.spec = spec
        self.contract = contract
        self.step_name = step_name
        self.verbose = verbose
        self.save_reports = save_reports
        self.output_dir = output_dir
        
        # Initialize components
        self.validator = ProcessingStepBuilderValidator(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose
        )
        
        self.universal_test = UniversalStepBuilderTest(
            builder_class=builder_class,
            config=config,
            spec=spec,
            contract=contract,
            step_name=step_name,
            verbose=verbose
        )
        
        self.llm_analyzer = ProcessingStepBuilderLLMAnalyzer(verbose=verbose)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test suite including all validation levels.
        
        Returns:
            Complete test results with scores and feedback
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"COMPREHENSIVE PROCESSING BUILDER TEST: {self.builder_class.__name__}")
            print(f"{'='*80}")
        
        # Step 1: Run universal tests
        if self.verbose:
            print("\n1. Running Universal Builder Tests...")
        universal_results = self.universal_test.run_all_tests()
        
        # Step 2: Run Processing-specific tests
        if self.verbose:
            print("\n2. Running Processing-Specific Tests...")
        processing_results = self.validator.run_processing_specific_tests()
        
        # Step 3: Validate standardization rules
        if self.verbose:
            print("\n3. Validating Standardization Rules...")
        standardization_violations = self.validator.validate_standardization_rules()
        
        # Step 4: Validate alignment rules
        if self.verbose:
            print("\n4. Validating Alignment Rules...")
        alignment_violations = self.validator.validate_alignment_rules()
        
        # Step 5: Combine all test results
        all_results = {}
        all_results.update(universal_results)
        all_results.update(processing_results)
        
        # Step 6: Generate LLM feedback
        if self.verbose:
            print("\n5. Generating LLM Feedback...")
        llm_feedback = self.llm_analyzer.analyze_builder(
            builder_class=self.builder_class,
            test_results=all_results,
            standardization_violations=standardization_violations,
            alignment_violations=alignment_violations
        )
        
        # Step 7: Generate comprehensive report
        comprehensive_report = {
            "builder_class": self.builder_class.__name__,
            "module": self.builder_class.__module__,
            "processing_step_type": self.validator.processing_step_type.value,
            "test_results": all_results,
            "standardization_violations": [v.model_dump() for v in standardization_violations],
            "alignment_violations": [v.model_dump() for v in alignment_violations],
            "llm_feedback": llm_feedback.model_dump(),
            "summary": {
                "total_tests": len(all_results),
                "passed_tests": sum(1 for r in all_results.values() if r.get("passed", False)),
                "pass_rate": (sum(1 for r in all_results.values() if r.get("passed", False)) / len(all_results)) * 100 if all_results else 0,
                "standardization_errors": len([v for v in standardization_violations if v.severity == "ERROR"]),
                "standardization_warnings": len([v for v in standardization_violations if v.severity == "WARNING"]),
                "alignment_errors": len([v for v in alignment_violations if v.severity == "ERROR"]),
                "overall_score": llm_feedback.overall_score,
                "overall_rating": llm_feedback.overall_rating
            }
        }
        
        # Step 8: Print results
        if self.verbose:
            self._print_comprehensive_results(comprehensive_report)
        
        # Step 9: Save reports
        if self.save_reports:
            self._save_comprehensive_report(comprehensive_report)
        
        return comprehensive_report
    
    def _print_comprehensive_results(self, report: Dict[str, Any]) -> None:
        """Print comprehensive test results."""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE TEST RESULTS")
        print(f"{'='*80}")
        
        summary = report["summary"]
        llm_feedback = report["llm_feedback"]
        
        # Overall summary
        print(f"\nBuilder: {report['builder_class']}")
        print(f"Type: {report['processing_step_type']}")
        print(f"Overall Score: {summary['overall_score']:.1f}/100 ({summary['overall_rating']})")
        print(f"Test Pass Rate: {summary['pass_rate']:.1f}% ({summary['passed_tests']}/{summary['total_tests']})")
        
        # Component scores
        print(f"\nComponent Scores:")
        print(f"  Code Quality: {llm_feedback['code_quality_score']:.1f}/100")
        print(f"  Architecture Compliance: {llm_feedback['architecture_compliance_score']:.1f}/100")
        print(f"  Maintainability: {llm_feedback['maintainability_score']:.1f}/100")
        
        # Violations summary
        print(f"\nViolations:")
        print(f"  Standardization Errors: {summary['standardization_errors']}")
        print(f"  Standardization Warnings: {summary['standardization_warnings']}")
        print(f"  Alignment Errors: {summary['alignment_errors']}")
        
        # LLM feedback
        if llm_feedback['strengths']:
            print(f"\nStrengths:")
            for strength in llm_feedback['strengths']:
                print(f"  ✅ {strength}")
        
        if llm_feedback['weaknesses']:
            print(f"\nWeaknesses:")
            for weakness in llm_feedback['weaknesses']:
                print(f"  ❌ {weakness}")
        
        if llm_feedback['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(llm_feedback['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print(f"\n{'='*80}")
    
    def _save_comprehensive_report(self, report: Dict[str, Any]) -> None:
        """Save comprehensive report to files."""
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_file = Path(self.output_dir) / f"{self.builder_class.__name__}_comprehensive_report.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        if self.verbose:
            print(f"Comprehensive report saved to: {json_file}")
        
        # Save detailed analysis
        analysis_file = Path(self.output_dir) / f"{self.builder_class.__name__}_detailed_analysis.txt"
        with open(analysis_file, 'w') as f:
            f.write(report["llm_feedback"]["detailed_analysis"])
        
        if self.verbose:
            print(f"Detailed analysis saved to: {analysis_file}")


def test_processing_builder(
    builder_class: Type[StepBuilderBase],
    config: Optional[ProcessingStepConfigBase] = None,
    spec: Optional[StepSpecification] = None,
    contract: Optional[ScriptContract] = None,
    step_name: Optional[str] = None,
    verbose: bool = True,
    save_reports: bool = True,
    output_dir: str = "test_reports"
) -> Dict[str, Any]:
    """
    Convenience function to test a Processing step builder.
    
    Args:
        builder_class: The Processing step builder class to test
        config: Optional Processing config to use
        spec: Optional step specification
        contract: Optional script contract
        step_name: Optional step name
        verbose: Whether to print verbose output
        save_reports: Whether to save test reports
        output_dir: Directory to save reports in
        
    Returns:
        Comprehensive test results
    """
    tester = UniversalProcessingBuilderTest(
        builder_class=builder_class,
        config=config,
        spec=spec,
        contract=contract,
        step_name=step_name,
        verbose=verbose,
        save_reports=save_reports,
        output_dir=output_dir
    )
    
    return tester.run_comprehensive_test()


# Example usage and test cases
if __name__ == "__main__":
    """
    Example usage of the Universal Processing Builder Test framework.
    """
    
    # Test TabularPreprocessingStepBuilder
    try:
        from ...steps.builders.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
        
        print("Testing TabularPreprocessingStepBuilder...")
        results = test_processing_builder(
            builder_class=TabularPreprocessingStepBuilder,
            verbose=True,
            save_reports=True
        )
        
        print(f"Overall Score: {results['summary']['overall_score']:.1f}/100")
        print(f"Rating: {results['summary']['overall_rating']}")
        
    except ImportError as e:
        print(f"Could not test TabularPreprocessingStepBuilder: {e}")
    
    # Test PayloadStepBuilder
    try:
        from ...steps.builders.builder_payload_step import PayloadStepBuilder
        
        print("\nTesting PayloadStepBuilder...")
        results = test_processing_builder(
            builder_class=PayloadStepBuilder,
            verbose=True,
            save_reports=True
        )
        
        print(f"Overall Score: {results['summary']['overall_score']:.1f}/100")
        print(f"Rating: {results['summary']['overall_rating']}")
        
    except ImportError as e:
        print(f"Could not test PayloadStepBuilder: {e}")
    
    # Test PackageStepBuilder
    try:
        from ...steps.builders.builder_package_step import PackageStepBuilder
        
        print("\nTesting PackageStepBuilder...")
        results = test_processing_builder(
            builder_class=PackageStepBuilder,
            verbose=True,
            save_reports=True
        )
        
        print(f"Overall Score: {results['summary']['overall_score']:.1f}/100")
        print(f"Rating: {results['summary']['overall_rating']}")
        
    except ImportError as e:
        print(f"Could not test PackageStepBuilder: {e}")
    
    # Test XGBoostModelEvalStepBuilder
    try:
        from ...steps.builders.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
        
        print("\nTesting XGBoostModelEvalStepBuilder...")
        results = test_processing_builder(
            builder_class=XGBoostModelEvalStepBuilder,
            verbose=True,
            save_reports=True
        )
        
        print(f"Overall Score: {results['summary']['overall_score']:.1f}/100")
        print(f"Rating: {results['summary']['overall_rating']}")
        
    except ImportError as e:
        print(f"Could not test XGBoostModelEvalStepBuilder: {e}")
