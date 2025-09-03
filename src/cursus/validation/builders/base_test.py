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
from ...registry.step_names import STEP_NAMES
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
        # Fix: Configure mock session with proper region string
        self.mock_session.boto_region_name = 'us-east-1'
        
        # CRITICAL FIX: Configure all S3-related methods to return strings, not MagicMock
        self.mock_session.default_bucket.return_value = 'test-bucket'
        self.mock_session.default_bucket_prefix = 'test-prefix'
        
        # Fix: Provide proper SageMaker config structure instead of MagicMock
        self.mock_session.sagemaker_config = {
            'SchemaVersion': '1.0',
            'SageMaker': {
                'PythonSDK': {
                    'Modules': {
                        'Session': {
                            'DefaultS3Bucket': 'test-bucket',
                            'DefaultS3ObjectKeyPrefix': 'test-prefix'
                        }
                    }
                }
            }
        }
        
        # Fix: Add proper settings mock with string values instead of MagicMock
        mock_settings = MagicMock()
        mock_settings.local_download_dir = '/tmp/test_downloads'  # Provide actual string path
        mock_settings.default_bucket = 'test-bucket'
        mock_settings.s3_kms_key = None
        self.mock_session.settings = mock_settings
        
        # CRITICAL FIX: Mock S3 upload methods to prevent actual S3 operations
        self.mock_session.upload_data.return_value = 's3://test-bucket/uploaded-code/model.tar.gz'
        
        # CRITICAL FIX: Mock boto3 S3 client methods that are called during model creation
        mock_s3_client = MagicMock()
        mock_s3_client.head_object.return_value = {'ContentLength': 1024}
        mock_s3_client.download_file.return_value = None  # Mock successful download
        self.mock_session.boto_session.client.return_value = mock_s3_client
        
        # CRITICAL FIX: Mock file operations to prevent actual file system operations
        import tempfile
        import os
        
        # Create a temporary directory that actually exists
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock model tar.gz file
        mock_model_file = os.path.join(self.temp_dir, 'model.tar.gz')
        with open(mock_model_file, 'wb') as f:
            f.write(b'mock model data')
        
        # ENHANCED FIX: Mock the download and tar extraction process
        def mock_download_file(bucket, key, local_path):
            import shutil
            import tarfile
            import os
            
            # Copy our mock model file to the requested location
            shutil.copy2(mock_model_file, local_path)
            
            # If the local_path is a tar file, create a proper tar structure
            if local_path.endswith('.tar.gz') or 'tar_file' in local_path:
                # Create a temporary directory for tar contents
                tar_extract_dir = os.path.dirname(local_path)
                model_dir = os.path.join(tar_extract_dir, 'model')
                os.makedirs(model_dir, exist_ok=True)
                
                # Create mock model files that SageMaker expects
                mock_model_py = os.path.join(model_dir, 'model.py')
                with open(mock_model_py, 'w') as f:
                    f.write('# Mock model file\nprint("Mock model loaded")\n')
                
                mock_requirements = os.path.join(model_dir, 'requirements.txt')
                with open(mock_requirements, 'w') as f:
                    f.write('torch>=1.0.0\n')
                
                # Create a proper tar.gz file with the model contents
                with tarfile.open(local_path, 'w:gz') as tar:
                    tar.add(model_dir, arcname='.')
        
        mock_s3_client.download_file.side_effect = mock_download_file
        
        # CRITICAL FIX: Mock tarfile operations to handle model extraction
        import tarfile
        original_tarfile_open = tarfile.open
        
        def mock_tarfile_open(name, mode='r', **kwargs):
            # If it's our mock tar file, return a mock that can extract properly
            if name and (name.endswith('.tar.gz') or 'tar_file' in str(name)):
                mock_tar = MagicMock()
                
                # Mock extractall to create expected files - handle filter parameter for newer Python versions
                def mock_extractall(path=None, members=None, numeric_owner=False, filter=None):
                    if path:
                        os.makedirs(path, exist_ok=True)
                        # Create expected model files
                        model_py = os.path.join(path, 'model.py')
                        with open(model_py, 'w') as f:
                            f.write('# Mock model file\nprint("Mock model loaded")\n')
                        
                        requirements_txt = os.path.join(path, 'requirements.txt')
                        with open(requirements_txt, 'w') as f:
                            f.write('torch>=1.0.0\n')
                
                mock_tar.extractall = mock_extractall
                mock_tar.close = MagicMock()
                mock_tar.__enter__ = lambda self: self
                mock_tar.__exit__ = lambda self, *args: None
                
                return mock_tar
            else:
                # For other files, use the original tarfile.open
                return original_tarfile_open(name, mode, **kwargs)
        
        # Patch tarfile.open globally during tests
        import builtins
        if not hasattr(builtins, '_original_tarfile_open'):
            builtins._original_tarfile_open = tarfile.open
            tarfile.open = mock_tarfile_open
        
        # Fix: Configure processor.run() to return proper step arguments
        # This is needed for Pattern B ProcessingStep creation (XGBoost, etc.)
        self._setup_processor_run_mock()
        
        # Mock IAM role
        self.mock_role = 'arn:aws:iam::123456789012:role/MockRole'
        
        # Create mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Configure dependency resolver for successful resolution
        mock_dependencies = self._create_mock_dependencies()
        dependency_dict = {}
        expected_deps = self._get_expected_dependencies()
        
        # Map expected dependency names to mock steps
        for i, dep_name in enumerate(expected_deps):
            if i < len(mock_dependencies):
                dependency_dict[dep_name] = mock_dependencies[i]
            else:
                # Fallback to simple mock if we don't have enough mock dependencies
                dependency_dict[dep_name] = MagicMock()
        
        # Configure multiple methods that builders might use to check for dependencies
        self.mock_dependency_resolver.resolve_step_dependencies.return_value = dependency_dict
        self.mock_dependency_resolver.get_resolved_dependencies.return_value = dependency_dict
        self.mock_dependency_resolver.has_dependency.side_effect = lambda dep_name: dep_name in dependency_dict
        self.mock_dependency_resolver.get_dependency.side_effect = lambda dep_name: dependency_dict.get(dep_name)
        
        # Also configure the registry manager to provide dependencies
        self.mock_registry_manager.get_step_dependencies.return_value = dependency_dict
        self.mock_registry_manager.resolve_dependencies.return_value = dependency_dict
        
        # Mock boto3 client
        self.mock_boto3_client = MagicMock()
        
        # Track assertions for reporting
        self.assertions = []
    
    def _setup_processor_run_mock(self) -> None:
        """
        Configure processor.run() mock to return proper step arguments.
        
        This fixes the issue where processor.run() returns None, causing
        ProcessingStep creation to fail with "either step_args or processor 
        need to be given, but not both" error.
        """
        # Import here to avoid circular imports
        from unittest.mock import patch, MagicMock
        
        # Import the _StepArguments class that SageMaker expects
        try:
            from sagemaker.workflow.pipeline_context import _StepArguments
        except ImportError:
            # Fallback if the import path changes
            from sagemaker.workflow.utilities import _StepArguments
        
        # Create a proper _StepArguments object that ProcessingStep expects
        mock_step_args_dict = {
            'ProcessingJobName': 'test-processing-job',
            'ProcessingResources': {
                'ClusterConfig': {
                    'InstanceType': 'ml.m5.large',
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 30
                }
            },
            'RoleArn': 'arn:aws:iam::123456789012:role/MockRole',
            'ProcessingInputs': [],
            'ProcessingOutputConfig': {
                'Outputs': []
            },
            'AppSpecification': {
                'ImageUri': 'test-image-uri',
                'ContainerEntrypoint': ['python3'],
                'ContainerArguments': []
            },
            'Environment': {},
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400
            }
        }
        
        # Create a _StepArguments object with the proper caller_name
        mock_step_args = _StepArguments(
            caller_name='run',  # This is what SageMaker expects for processor.run()
            **mock_step_args_dict
        )
        
        # Patch all processor classes to return proper _StepArguments from run()
        processor_classes = [
            'sagemaker.processing.Processor',
            'sagemaker.xgboost.XGBoostProcessor',
            'sagemaker.sklearn.SKLearnProcessor',
            'sagemaker.pytorch.PyTorchProcessor',
            'sagemaker.tensorflow.TensorFlowProcessor'
        ]
        
        # Store patches for cleanup if needed
        self._processor_patches = []
        
        for processor_class_path in processor_classes:
            try:
                # Patch the run method to return mock _StepArguments
                patcher = patch(f'{processor_class_path}.run', return_value=mock_step_args)
                patcher.start()
                self._processor_patches.append(patcher)
            except (ImportError, AttributeError):
                # Skip if the processor class doesn't exist
                continue
    
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
    
    def _get_required_dependencies_from_spec(self, builder: StepBuilderBase) -> List[str]:
        """Extract required dependency logical names from builder specification."""
        required_deps = []
        
        if hasattr(builder, 'spec') and builder.spec and hasattr(builder.spec, 'dependencies'):
            for _, dependency_spec in builder.spec.dependencies.items():
                if dependency_spec.required:
                    required_deps.append(dependency_spec.logical_name)
        
        return required_deps
    
    def _create_mock_inputs_for_builder(self, builder: StepBuilderBase) -> Dict[str, str]:
        """Create mock inputs dictionary for a builder based on its required dependencies."""
        mock_inputs = {}
        
        # Get required dependencies from specification
        required_deps = self._get_required_dependencies_from_spec(builder)
        
        # Generate mock S3 URIs for each required dependency
        for logical_name in required_deps:
            mock_inputs[logical_name] = self._generate_mock_s3_uri(logical_name)
        
        # For CreateModel steps, also check if we need to add dependencies from the mock factory
        if hasattr(self.step_info, 'get') and self.step_info.get('sagemaker_step_type') == 'CreateModel':
            expected_deps = self._get_expected_dependencies()
            for dep_name in expected_deps:
                if dep_name not in mock_inputs:
                    mock_inputs[dep_name] = self._generate_mock_s3_uri(dep_name)
        
        return mock_inputs
    
    def _generate_mock_s3_uri(self, logical_name: str) -> str:
        """Generate a mock S3 URI for a given logical dependency name."""
        # Create appropriate S3 URI based on dependency type
        if logical_name.lower() in ['data', 'input_data']:
            uri = f"s3://test-bucket/processing-data/{logical_name}"
        elif logical_name.lower() in ['input_path']:
            uri = f"s3://test-bucket/training-data/{logical_name}"
        elif logical_name.lower() in ['model_input', 'model_artifacts', 'model_data']:
            uri = f"s3://test-bucket/model-artifacts/{logical_name}"
        elif logical_name.lower() in ['hyperparameters_s3_uri']:
            uri = f"s3://test-bucket/hyperparameters/{logical_name}/hyperparameters.json"
        else:
            uri = f"s3://test-bucket/generic/{logical_name}"
        
        # Ensure we return an actual string, not a MagicMock
        return str(uri)
    
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
        """Create mock dependencies for the builder with enhanced property mapping."""
        # Create a list of mock steps
        dependencies = []
        
        # Get expected dependencies
        expected_deps = self._get_expected_dependencies()
        
        # Create a mock step for each expected dependency
        for i, dep_name in enumerate(expected_deps):
            # Create mock step
            step = MagicMock()
            step.name = f"Mock{dep_name.capitalize()}Step"
            
            # Add properties attribute with comprehensive outputs
            step.properties = MagicMock()
            
            # Configure properties based on dependency type
            if dep_name.lower() in ['data', 'input_data']:
                # Processing step output for data dependencies
                step.properties.ProcessingOutputConfig = MagicMock()
                step.properties.ProcessingOutputConfig.Outputs = [
                    MagicMock(
                        OutputName=dep_name,
                        S3Output=MagicMock(
                            S3Uri=f"s3://test-bucket/processing/{dep_name}",
                            LocalPath=f"/opt/ml/processing/output/{dep_name}"
                        )
                    )
                ]
                # Also provide direct access pattern
                setattr(step.properties.ProcessingOutputConfig, dep_name, 
                       MagicMock(S3Uri=f"s3://test-bucket/processing/{dep_name}"))
                
            elif dep_name.lower() in ['input_path']:
                # Training step input path - could come from processing or training step
                step.properties.ProcessingOutputConfig = MagicMock()
                step.properties.ProcessingOutputConfig.Outputs = [
                    MagicMock(
                        OutputName='training_data',
                        S3Output=MagicMock(
                            S3Uri=f"s3://test-bucket/training-data/{dep_name}",
                            LocalPath="/opt/ml/processing/output/training_data"
                        )
                    )
                ]
                # Also add ModelArtifacts in case it's from a training step
                step.properties.ModelArtifacts = MagicMock()
                step.properties.ModelArtifacts.S3ModelArtifacts = f"s3://test-bucket/model-artifacts/{dep_name}"
                
            elif dep_name.lower() in ['model_input', 'model_artifacts', 'model_data']:
                # Model artifacts from training steps - CRITICAL FIX FOR CREATEMODEL
                mock_s3_uri = f"s3://test-bucket/model-artifacts/{dep_name}"
                
                # Create a custom mock that always returns the string value
                class StringMock:
                    def __init__(self, value):
                        self.value = str(value)
                    
                    def __str__(self):
                        return self.value
                    
                    def __repr__(self):
                        return self.value
                    
                    def __getattr__(self, name):
                        return self.value
                    
                    def __getitem__(self, key):
                        return self.value
                
                # Set up the ModelArtifacts structure with string mock
                step.properties.ModelArtifacts = MagicMock()
                step.properties.ModelArtifacts.S3ModelArtifacts = StringMock(mock_s3_uri)
                
                # Also configure the MagicMock to return the string directly
                step.properties.ModelArtifacts.configure_mock(S3ModelArtifacts=mock_s3_uri)
                
                # Also provide processing output format for model evaluation steps
                step.properties.ProcessingOutputConfig = MagicMock()
                step.properties.ProcessingOutputConfig.Outputs = [
                    MagicMock(
                        OutputName=dep_name,
                        S3Output=MagicMock(
                            S3Uri=f"s3://test-bucket/model/{dep_name}",
                            LocalPath=f"/opt/ml/processing/output/{dep_name}"
                        )
                    )
                ]
                
            else:
                # Generic dependency - provide both processing and model artifact patterns
                step.properties.ProcessingOutputConfig = MagicMock()
                step.properties.ProcessingOutputConfig.Outputs = [
                    MagicMock(
                        OutputName=dep_name,
                        S3Output=MagicMock(
                            S3Uri=f"s3://test-bucket/generic/{dep_name}",
                            LocalPath=f"/opt/ml/processing/output/{dep_name}"
                        )
                    )
                ]
                step.properties.ModelArtifacts = MagicMock()
                step.properties.ModelArtifacts.S3ModelArtifacts = f"s3://test-bucket/generic/{dep_name}"
            
            # Add comprehensive _spec attribute for dependency resolution
            step._spec = MagicMock()
            step._spec.outputs = {
                dep_name: MagicMock(
                    logical_name=dep_name,
                    property_path=self._get_property_path_for_dependency(dep_name)
                )
            }
            
            # Add step type information
            step.step_type = self._infer_step_type_from_dependency(dep_name)
            
            dependencies.append(step)
        
        return dependencies
    
    def _get_expected_dependencies(self) -> List[str]:
        """Get the list of expected dependency names for the builder."""
        # Use the mock factory to get expected dependencies
        return self.mock_factory.get_expected_dependencies()
    
    def _get_property_path_for_dependency(self, dep_name: str) -> str:
        """Get the appropriate property path for a dependency based on its type."""
        if dep_name.lower() in ['data', 'input_data', 'model_input']:
            return f"ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
        elif dep_name.lower() in ['input_path']:
            return f"ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
        elif dep_name.lower() in ['model_artifacts']:
            return f"ModelArtifacts.S3ModelArtifacts"
        else:
            return f"ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
    
    def _infer_step_type_from_dependency(self, dep_name: str) -> str:
        """Infer the step type that would produce this dependency."""
        if dep_name.lower() in ['data', 'input_data']:
            return 'Processing'
        elif dep_name.lower() in ['input_path']:
            return 'Processing'  # Training data usually comes from processing
        elif dep_name.lower() in ['model_input', 'model_artifacts']:
            return 'Training'  # Model artifacts come from training
        else:
            return 'Processing'  # Default to processing
    
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
