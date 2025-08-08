"""
Base class for universal step builder tests.
"""

import contextlib
from abc import ABC, abstractmethod
from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import Dict, List, Any, Optional, Union, Type, Callable
from enum import Enum
from pydantic import BaseModel
from sagemaker.workflow.steps import Step

# Import new components
from .step_info_detector import StepInfoDetector
from .mock_factory import StepTypeMockFactory


class ValidationLevel(Enum):
    """Validation violation severity levels."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class ValidationViolation(BaseModel):
    """Represents a validation violation."""
    level: ValidationLevel
    category: str
    message: str
    details: str = ""

# Import base classes for type hints
from ...core.base.builder_base import StepBuilderBase
from ...core.base.specification_base import StepSpecification
from ...core.base.contract_base import ScriptContract
from ...core.base.config_base import BaseModel as ConfigBase

# Step name is string type from the registry
from ...steps.registry.step_names import STEP_NAMES
StepName = str  # Step names are stored as string keys in STEP_NAMES dictionary


class UniversalStepBuilderTestBase(ABC):
    """
    Base class for universal step builder tests.
    
    This class provides common setup and utility methods for testing step builders.
    Specific test suites inherit from this class to add their test methods.
    """
    
    def __init__(
        self, 
        builder_class: Type[StepBuilderBase],
        config: Optional[ConfigBase] = None,
        spec: Optional[StepSpecification] = None,
        contract: Optional[ScriptContract] = None,
        step_name: Optional[Union[str, StepName]] = None,
        verbose: bool = False,
        test_reporter: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize with explicit components.
        
        Args:
            builder_class: The step builder class to test
            config: Optional config to use (will create mock if not provided)
            spec: Optional step specification (will extract from builder if not provided)
            contract: Optional script contract (will extract from builder if not provided)
            step_name: Optional step name (will extract from class name if not provided)
            verbose: Whether to print verbose output
            test_reporter: Optional function to report test results
            **kwargs: Additional arguments for subclasses
        """
        self.builder_class = builder_class
        self._provided_config = config
        self._provided_spec = spec
        self._provided_contract = contract
        self._provided_step_name = step_name
        self.verbose = verbose
        self.test_reporter = test_reporter or (lambda *args, **kwargs: None)
        
        # Detect step information using new detector
        self.step_info_detector = StepInfoDetector(builder_class)
        self.step_info = self.step_info_detector.detect_step_info()
        
        # Create mock factory based on step info
        self.mock_factory = StepTypeMockFactory(self.step_info)
        
        # Setup test environment
        self._setup_test_environment()
        
        # Configure step type-specific mocks
        self._configure_step_type_mocks()
    
    @abstractmethod
    def get_step_type_specific_tests(self) -> List[str]:
        """Return step type-specific test methods."""
        pass
    
    @abstractmethod
    def _configure_step_type_mocks(self) -> None:
        """Configure step type-specific mock objects."""
        pass
    
    @abstractmethod
    def _validate_step_type_requirements(self) -> Dict[str, Any]:
        """Validate step type-specific requirements."""
        pass
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all tests in this test suite.
        
        Returns:
            Dictionary mapping test names to their results
        """
        # Get all methods that start with "test_"
        test_methods = [
            getattr(self, name) for name in dir(self) 
            if name.startswith('test_') and callable(getattr(self, name))
        ]
        
        results = {}
        for test_method in test_methods:
            results[test_method.__name__] = self._run_test(test_method)
            
        # Report overall results
        self._report_overall_results(results)
        
        return results
    
    def _setup_test_environment(self) -> None:
        """Set up mock objects and test fixtures."""
        # Mock SageMaker session
        self.mock_session = MagicMock()
        self.mock_session.boto_session.client.return_value = MagicMock()
        
        # Mock IAM role
        self.mock_role = 'arn:aws:iam::123456789012:role/MockRole'
        
        # Create mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Configure dependency resolver for successful resolution
        self.mock_dependency_resolver.resolve_step_dependencies.return_value = {
            dep: MagicMock() for dep in self._get_expected_dependencies()
        }
        
        # Mock boto3 client
        self.mock_boto3_client = MagicMock()
        
        # Track assertions for reporting
        self.assertions = []
    
    def _create_builder_instance(self) -> StepBuilderBase:
        """Create a builder instance with mock configuration."""
        # Use provided config or create mock configuration
        config = self._provided_config if self._provided_config else self._create_mock_config()
        
        # Create builder instance
        builder = self.builder_class(
            config=config,
            sagemaker_session=self.mock_session,
            role=self.mock_role,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # If specification was provided, set it on the builder
        if self._provided_spec:
            builder.spec = self._provided_spec
            
        # If contract was provided, set it on the builder
        if self._provided_contract:
            builder.contract = self._provided_contract
            
        # If step name was provided, override the builder's _get_step_name method
        if self._provided_step_name:
            builder._get_step_name = lambda *args, **kwargs: self._provided_step_name
        
        return builder
    
    def _create_mock_config(self) -> SimpleNamespace:
        """Create a mock configuration for the builder using the factory."""
        # Use the mock factory to create step type-specific config
        return self.mock_factory.create_mock_config()
    
    def _create_invalid_config(self) -> SimpleNamespace:
        """Create an invalid configuration for testing error handling."""
        # Create a minimal config without required attributes
        mock_config = SimpleNamespace()
        mock_config.region = 'NA'  # Include only the region
        
        return mock_config
    
    def _create_mock_dependencies(self) -> List[Step]:
        """Create mock dependencies for the builder."""
        # Create a list of mock steps
        dependencies = []
        
        # Get expected dependencies
        expected_deps = self._get_expected_dependencies()
        
        # Create a mock step for each expected dependency
        for i, dep_name in enumerate(expected_deps):
            # Create mock step
            step = MagicMock()
            step.name = f"Mock{dep_name.capitalize()}Step"
            
            # Add properties attribute with outputs
            step.properties = MagicMock()
            
            # Add ProcessingOutputConfig for processing steps
            if "Processing" in step.name:
                step.properties.ProcessingOutputConfig = MagicMock()
                step.properties.ProcessingOutputConfig.Outputs = {
                    dep_name: MagicMock(
                        S3Output=MagicMock(
                            S3Uri=f"s3://bucket/prefix/{dep_name}"
                        )
                    )
                }
            
            # Add ModelArtifacts for training steps
            if "Training" in step.name:
                step.properties.ModelArtifacts = MagicMock(
                    S3ModelArtifacts=f"s3://bucket/prefix/{dep_name}"
                )
            
            # Add _spec attribute
            step._spec = MagicMock()
            step._spec.outputs = {
                dep_name: MagicMock(
                    logical_name=dep_name,
                    property_path=f"properties.Outputs['{dep_name}'].S3Uri"
                )
            }
            
            dependencies.append(step)
        
        return dependencies
    
    def _get_expected_dependencies(self) -> List[str]:
        """Get the list of expected dependency names for the builder."""
        # Use the mock factory to get expected dependencies
        return self.mock_factory.get_expected_dependencies()
    
    @contextlib.contextmanager
    def _assert_raises(self, expected_exception):
        """Context manager to assert that an exception is raised."""
        try:
            yield
            self._assert(False, f"Expected {expected_exception.__name__} to be raised")
        except expected_exception:
            pass
        except Exception as e:
            self._assert(False, f"Expected {expected_exception.__name__} but got {type(e).__name__}")
    
    def _assert(self, condition: bool, message: str) -> None:
        """Assert that a condition is true."""
        # Add assertion to list
        self.assertions.append((condition, message))
        
        # Log message if verbose
        if self.verbose and not condition:
            print(f"❌ FAILED: {message}")
        elif self.verbose and condition:
            print(f"✅ PASSED: {message}")
    
    def _log(self, message: str) -> None:
        """Log a message if verbose."""
        if self.verbose:
            print(f"ℹ️ INFO: {message}")
    
    def _run_test(self, test_method: Callable) -> Dict[str, Any]:
        """Run a single test method and capture results."""
        # Reset assertions
        self.assertions = []
        
        # Run test
        try:
            # Log test start
            self._log(f"Running {test_method.__name__}...")
            
            # Run test method
            test_method()
            
            # Check if any assertions failed
            failed = [msg for cond, msg in self.assertions if not cond]
            
            # Return result
            if failed:
                return {
                    "passed": False,
                    "name": test_method.__name__,
                    "error": "\n".join(failed)
                }
            else:
                return {
                    "passed": True,
                    "name": test_method.__name__,
                    "assertions": len(self.assertions)
                }
        except Exception as e:
            # Return error result
            return {
                "passed": False,
                "name": test_method.__name__,
                "error": str(e),
                "exception": e
            }
    
    def _report_overall_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Report overall test results."""
        # Count passed tests
        passed = sum(1 for result in results.values() if result["passed"])
        total = len(results)
        
        # Log overall result
        if self.verbose:
            print(f"\n=== TEST RESULTS FOR {self.builder_class.__name__} ===")
            print(f"PASSED: {passed}/{total} tests")
            
            # Log details for each test
            for test_name, result in results.items():
                if result["passed"]:
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED: {result['error']}")
            
            print("=" * 40)
