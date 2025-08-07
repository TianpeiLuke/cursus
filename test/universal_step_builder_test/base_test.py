"""
Base class for universal step builder tests.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock
from typing import Dict, List, Any, Optional, Union, Type, Callable
from sagemaker.workflow.steps import Step

# Import base classes for type hints
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_deps.base_specifications import StepSpecification
from src.pipeline_script_contracts.base_script_contract import ScriptContract
from src.pipeline_steps.config_base import BaseModel as ConfigBase  # Using Pydantic BaseModel that ConfigBase extends

# Step name is string type from the registry
from src.pipeline_registry.step_names import STEP_NAMES
StepName = str  # Step names are stored as string keys in STEP_NAMES dictionary


class UniversalStepBuilderTestBase:
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
        test_reporter: Optional[Callable] = None
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
        """
        self.builder_class = builder_class
        self._provided_config = config
        self._provided_spec = spec
        self._provided_contract = contract
        self._provided_step_name = step_name
        self.verbose = verbose
        self.test_reporter = test_reporter or (lambda *args, **kwargs: None)
        self._setup_test_environment()
    
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
        """Create a mock configuration for the builder."""
        # Basic config with required attributes
        mock_config = SimpleNamespace()
        mock_config.region = 'NA'
        mock_config.pipeline_name = 'test-pipeline'
        mock_config.pipeline_s3_loc = 's3://bucket/prefix'
        
        # Add hyperparameters if needed
        mock_hp = SimpleNamespace()
        mock_hp.model_dump = lambda: {'param': 'value'}
        mock_config.hyperparameters = mock_hp
        
        # Add common methods
        mock_config.get_image_uri = lambda: 'mock-image-uri'
        mock_config.get_script_path = lambda: 'mock_script.py'
        mock_config.get_script_contract = lambda: None
        
        # Add builder-specific attributes
        self._add_builder_specific_config(mock_config)
        
        return mock_config
    
    def _create_invalid_config(self) -> SimpleNamespace:
        """Create an invalid configuration for testing error handling."""
        # Create a minimal config without required attributes
        mock_config = SimpleNamespace()
        mock_config.region = 'NA'  # Include only the region
        
        return mock_config
    
    def _add_builder_specific_config(self, mock_config: SimpleNamespace) -> None:
        """Add builder-specific configuration attributes."""
        # Get builder class name
        builder_name = self.builder_class.__name__
        
        if "XGBoostTraining" in builder_name:
            self._add_xgboost_training_config(mock_config)
        elif "TabularPreprocessing" in builder_name:
            self._add_tabular_preprocessing_config(mock_config)
        elif "ModelEval" in builder_name:
            self._add_model_eval_config(mock_config)
        # Add more builder-specific configurations as needed
        else:
            self._add_generic_config(mock_config)
    
    def _add_xgboost_training_config(self, mock_config: SimpleNamespace) -> None:
        """Add XGBoost training-specific configuration attributes."""
        mock_config.training_instance_type = 'ml.m5.xlarge'
        mock_config.training_instance_count = 1
        mock_config.training_volume_size = 30
        mock_config.training_entry_point = 'train_xgb.py'
        mock_config.source_dir = 'src/pipeline_scripts'
        mock_config.framework_version = '1.7-1'
        mock_config.py_version = 'py3'
    
    def _add_tabular_preprocessing_config(self, mock_config: SimpleNamespace) -> None:
        """Add tabular preprocessing-specific configuration attributes."""
        mock_config.processing_instance_type = 'ml.m5.large'
        mock_config.processing_instance_count = 1
        mock_config.processing_volume_size = 30
        mock_config.processing_entry_point = 'tabular_preprocess.py'
        mock_config.source_dir = 'src/pipeline_scripts'
    
    def _add_model_eval_config(self, mock_config: SimpleNamespace) -> None:
        """Add model evaluation-specific configuration attributes."""
        mock_config.processing_instance_type = 'ml.m5.large'
        mock_config.processing_instance_count = 1
        mock_config.processing_volume_size = 30
        mock_config.processing_entry_point = 'model_evaluation_xgb.py'
        mock_config.source_dir = 'src/pipeline_scripts'
        mock_config.id_field = 'id'
        mock_config.label_field = 'label'
    
    def _add_generic_config(self, mock_config: SimpleNamespace) -> None:
        """Add generic configuration attributes for unknown builder types."""
        mock_config.instance_type = 'ml.m5.large'
        mock_config.instance_count = 1
        mock_config.volume_size = 30
        mock_config.entry_point = 'generic_script.py'
        mock_config.source_dir = 'src/pipeline_scripts'
    
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
        # Try to create a builder instance to get required dependencies
        try:
            # Create a temporary builder instance
            temp_builder = self._create_builder_instance()
            
            # Get required dependencies
            if hasattr(temp_builder, 'get_required_dependencies'):
                return temp_builder.get_required_dependencies()
        except Exception:
            pass
        
        # Fallback: guess dependency names based on builder class name
        builder_name = self.builder_class.__name__
        
        if "XGBoostTraining" in builder_name:
            return ["input_path"]
        elif "TabularPreprocessing" in builder_name:
            return ["DATA"]
        elif "ModelEval" in builder_name:
            return ["model_input", "eval_data_input"]
        
        # Default
        return ["input"]
    
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
