from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import tempfile
import json
import shutil
import boto3
from botocore.exceptions import ClientError

from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sagemaker.s3 import S3Uploader
from sagemaker.workflow.functions import Join

from ..configs.config_training_step_xgboost import XGBoostTrainingConfig
from ...core.base.builder_base import StepBuilderBase
from .s3_utils import S3PathHandler
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver
from ..registry.builder_registry import register_builder

# Import XGBoost training specification
try:
    from ..specs.xgboost_training_spec import XGBOOST_TRAINING_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    XGBOOST_TRAINING_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


@register_builder()
class XGBoostTrainingStepBuilder(StepBuilderBase):
    """
    Builder for an XGBoost Training Step.
    This class is responsible for configuring and creating a SageMaker TrainingStep
    that trains an XGBoost model.
    """

    def __init__(
        self,
        config: XGBoostTrainingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None
    ):
        """
        Initializes the builder with a specific configuration for the training step.

        Args:
            config: A XGBoostTrainingConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Training Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
                         
        Raises:
            ValueError: If specification is not available or config is invalid
        """
        if not isinstance(config, XGBoostTrainingConfig):
            raise ValueError(
                "XGBoostTrainingStepBuilder requires a XGBoostTrainingConfig instance."
            )
            
        # Load XGBoost training specification
        if not SPEC_AVAILABLE or XGBOOST_TRAINING_SPEC is None:
            raise ValueError("XGBoost training specification not available")
            
        self.log_info("Using XGBoost training specification")
        
        super().__init__(
            config=config,
            spec=XGBOOST_TRAINING_SPEC,  # Add specification
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: XGBoostTrainingConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating XGBoostTrainingConfig...")
        
        # Validate required attributes
        required_attrs = [
            'training_instance_type',
            'training_instance_count',
            'training_volume_size',
            'training_entry_point',
            'source_dir',
            'framework_version',
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"XGBoostTrainingConfig missing required attribute: {attr}")
        
        # Input/output validation is now handled by specifications
        self.log_info("Configuration validation relies on step specifications")
            
        self.log_info("XGBoostTrainingConfig validation succeeded.")

    def _create_estimator(self, output_path=None) -> XGBoost:
        """
        Creates and configures the XGBoost estimator for the SageMaker Training Job.
        This defines the execution environment for the training script, including the instance
        type, framework version, and environment variables.
        
        Args:
            output_path: Optional override for model output path. If provided, this will be used
                         instead of self.config.output_path.

        Returns:
            An instance of sagemaker.xgboost.XGBoost.
        """
        # Note: We don't pass hyperparameters directly here because they are passed
        # through the "config" input channel instead
        
        return XGBoost(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            base_job_name=self._generate_job_name(),  # Use standardized method with auto-detection
            sagemaker_session=self.session,
            output_path=output_path,  # Use provided output_path directly
            environment=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the training job.
        These variables are used to control the behavior of the training script
        without needing to pass them as hyperparameters.

        Returns:
            A dictionary of environment variables.
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()
        
        # Add environment variables from config if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)
            
        self.log_info("Training environment variables: %s", env_vars)
        return env_vars
        
    def _create_data_channels_from_source(self, base_path):
        """
        Create train, validation, and test channel inputs from a base path.
        
        Args:
            base_path: Base S3 path containing train/val/test subdirectories
            
        Returns:
            Dictionary of channel name to TrainingInput
        """
        from sagemaker.workflow.functions import Join
        
        # Base path is used directly - property references are handled by PipelineAssembler
        
        channels = {
            "train": TrainingInput(s3_data=Join(on='/', values=[base_path, "train/"])),
            "val": TrainingInput(s3_data=Join(on='/', values=[base_path, "val/"])),
            "test": TrainingInput(s3_data=Join(on='/', values=[base_path, "test/"]))
        }
        
        return channels

    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """
        Get inputs for the step using specification and contract.
        
        This method creates TrainingInput objects for each dependency defined in the specification.
        Special handling is implemented for hyperparameters_s3_uri to always use internally generated ones.
        
        Args:
            inputs: Input data sources keyed by logical name
            
        Returns:
            Dictionary of TrainingInput objects keyed by channel name
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for input mapping")
            
        training_inputs = {}
        matched_inputs = set()  # Track which inputs we've handled
        
        # SPECIAL CASE: Always generate hyperparameters internally first
        hyperparameters_key = "hyperparameters_s3_uri"
        
        # Generate hyperparameters file regardless of whether inputs contains it
        internal_hyperparameters_s3_uri = self._prepare_hyperparameters_file()
        self.log_info("[TRAINING INPUT OVERRIDE] Generated hyperparameters internally at: %s", internal_hyperparameters_s3_uri)
        self.log_info("[TRAINING INPUT OVERRIDE] This will be used regardless of any dependency-provided values")
        
        # Get container path from contract for the hyperparameters
        hyperparams_container_path = None
        if hyperparameters_key in self.contract.expected_input_paths:
            hyperparams_container_path = self.contract.expected_input_paths[hyperparameters_key]
            
            # Extract the channel name from the container path
            # For '/opt/ml/input/data/config/hyperparameters.json', the channel name would be 'config'
            parts = hyperparams_container_path.split('/')
            if len(parts) > 4 and parts[1] == "opt" and parts[2] == "ml" and parts[3] == "input" and parts[4] == "data":
                channel_name = parts[5]  # This would be 'config'
                # Property references are now handled by PipelineAssembler
                training_inputs[channel_name] = TrainingInput(s3_data=internal_hyperparameters_s3_uri)
                self.log_info("Created %s channel from internally generated hyperparameters: %s", 
                             channel_name, internal_hyperparameters_s3_uri)
        else:
            # Fallback to 'config' if not in contract
            training_inputs["config"] = TrainingInput(s3_data=internal_hyperparameters_s3_uri)
            self.log_info("Created config channel from internally generated hyperparameters: %s", 
                         internal_hyperparameters_s3_uri)
        
        matched_inputs.add(hyperparameters_key)
        
        # Create a copy of the inputs dictionary
        working_inputs = inputs.copy()
        
        # Remove our special case from the inputs dictionary
        if hyperparameters_key in working_inputs:
            external_path = working_inputs[hyperparameters_key]
            self.log_info("[TRAINING INPUT OVERRIDE] Ignoring dependency-provided hyperparameters: %s", external_path)
            self.log_info("[TRAINING INPUT OVERRIDE] Using internal hyperparameters instead: %s", internal_hyperparameters_s3_uri)
            del working_inputs[hyperparameters_key]
        
        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name
            
            # Skip inputs we've already handled
            if logical_name in matched_inputs:
                continue
                
            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in working_inputs:
                continue
                
            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in working_inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")
            
            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
                
                # SPECIAL HANDLING FOR input_path
                # For '/opt/ml/input/data', we need to create train/val/test channels
                if logical_name == "input_path":
                    base_path = working_inputs[logical_name]
                    
                    # Create separate channels for each data split using helper method
                    data_channels = self._create_data_channels_from_source(base_path)
                    training_inputs.update(data_channels)
                    # Safe logging that handles Pipeline variables using the standard pattern
                    self.log_info("Created data channels from %s: %s", logical_name, base_path)
                else:
                    # For other inputs, extract the channel name from the container path
                    parts = container_path.split('/')
                    if len(parts) > 4 and parts[1] == "opt" and parts[2] == "ml" and parts[3] == "input" and parts[4] == "data":
                        if len(parts) > 5:
                            channel_name = parts[5]  # Extract channel name from path
                            # Input data is used directly - property references are handled by PipelineAssembler
                            training_inputs[channel_name] = TrainingInput(s3_data=working_inputs[logical_name])
                            # Safe logging that handles Pipeline variables using the standard pattern
                            self.log_info("Created %s channel from %s: %s", channel_name, logical_name, working_inputs[logical_name])
                        else:
                            # If no specific channel in path, use logical name as channel
                            # Input data is used directly - property references are handled by PipelineAssembler
                            training_inputs[logical_name] = TrainingInput(s3_data=working_inputs[logical_name])
                            # Safe logging that handles Pipeline variables using the standard pattern
                            self.log_info("Created %s channel from %s: %s", logical_name, logical_name, working_inputs[logical_name])
            else:
                raise ValueError(f"No container path found for input: {logical_name}")
                
        return training_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        Get outputs for the step using specification and contract.
        
        For training steps, this returns the output path where model artifacts and evaluation results will be stored.
        SageMaker uses this single output_path parameter for both:
        - model.tar.gz (from /opt/ml/model/)
        - output.tar.gz (from /opt/ml/output/data/)
        
        Args:
            outputs: Output destinations keyed by logical name
            
        Returns:
            Output path for model artifacts and evaluation results
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for output mapping")
            
        # First, check if any output path is explicitly provided in the outputs dictionary
        primary_output_path = None
        
        # Check if model_output or evaluation_output are in the outputs dictionary
        output_logical_names = [spec.logical_name for _, spec in self.spec.outputs.items()]
        
        for logical_name in output_logical_names:
            if logical_name in outputs:
                primary_output_path = outputs[logical_name]
                self.log_info(f"Using provided output path from '{logical_name}': {primary_output_path}")
                break
                
        # If no output path was provided, generate a default one
        if primary_output_path is None:
            # Generate a clean path that will be used as the base for all outputs
            primary_output_path = f"{self.config.pipeline_s3_loc}/xgboost_training/"
            self.log_info(f"Using generated base output path: {primary_output_path}")
        
        # Remove trailing slash if present for consistency with S3 path handling
        if primary_output_path.endswith('/'):
            primary_output_path = primary_output_path[:-1]
        
        # Get base job name for logging purposes
        base_job_name = self._generate_job_name()
        
        # Log how SageMaker will structure outputs under this path
        self.log_info(f"SageMaker will organize outputs using base job name: {base_job_name}")
        self.log_info(f"Full job name will be: {base_job_name}-[timestamp]")
        self.log_info(f"Output path structure will be: {primary_output_path}/{base_job_name}-[timestamp]/")
        self.log_info(f"  - Model artifacts will be in: {primary_output_path}/{base_job_name}-[timestamp]/output/model.tar.gz")
        self.log_info(f"  - Evaluation results will be in: {primary_output_path}/{base_job_name}-[timestamp]/output/output.tar.gz")
        
        return primary_output_path

        
    def _normalize_s3_uri(self, uri: str, description: str = "S3 URI") -> str:
        """
        Normalizes an S3 URI to ensure it has no trailing slashes and is properly formatted.
        Uses S3PathHandler for consistent path handling.
        
        Args:
            uri: The S3 URI to normalize
            description: Description for logging purposes
            
        Returns:
            Normalized S3 URI
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
        
        # Handle Pipeline step references with Get key - return as is
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info("Found Pipeline step reference during normalization: %s", uri)
            return uri
        
        return S3PathHandler.normalize(uri, description)
        
    def _get_s3_directory_path(self, uri: str, filename: str = None) -> str:
        """
        Gets the directory part of an S3 URI, handling special cases correctly.
        Uses S3PathHandler for consistent path handling.
        
        Args:
            uri: The S3 URI which may or may not contain a filename
            filename: Optional filename to check for at the end of the URI
            
        Returns:
            The directory part of the URI without trailing slash
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
            
        # Handle Pipeline step references with Get key - return as is
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info("Found Pipeline step reference in directory path: %s", uri)
            return uri
            
        return S3PathHandler.ensure_directory(uri, filename)

    
    def _prepare_hyperparameters_file(self) -> str:
        """
        Serializes the hyperparameters to JSON, uploads it to S3, and
        returns that full S3 URI. This eliminates the need for a separate
        HyperparameterPrepStep in the pipeline.
        """
        # Start with model_dump() to get proper JSON types
        hyperparams_dict = self.config.hyperparameters.model_dump()
        
        # Add derived properties manually
        hyperparams_dict["is_binary"] = self.config.hyperparameters.is_binary
        hyperparams_dict["num_classes"] = self.config.hyperparameters.num_classes
        hyperparams_dict["input_tab_dim"] = self.config.hyperparameters.input_tab_dim
        hyperparams_dict["objective"] = self.config.hyperparameters.objective
        hyperparams_dict["eval_metric"] = self.config.hyperparameters.eval_metric
        
        # Ensure class_weights matches num_classes
        if "class_weights" not in hyperparams_dict or len(hyperparams_dict["class_weights"]) != hyperparams_dict["num_classes"]:
            hyperparams_dict["class_weights"] = [1.0] * hyperparams_dict["num_classes"]
        
        local_dir = Path(tempfile.mkdtemp())
        local_file = local_dir / "hyperparameters.json"
        
        try:
            # Write JSON locally
            with open(local_file, "w") as f:
                json.dump(hyperparams_dict, indent=2, fp=f)
            self.log_info("Created hyperparameters JSON file at %s", local_file)

            # Construct S3 URI for the config directory
            prefix = self.config.hyperparameters_s3_uri if hasattr(self.config, 'hyperparameters_s3_uri') else None
            if not prefix:
                # Fallback path construction
                bucket = self.config.bucket if hasattr(self.config, 'bucket') else "sandboxdependency-abuse-secureaisandboxteamshare-1l77v9am252um"
                pipeline_name = self.config.pipeline_name if hasattr(self.config, 'pipeline_name') else "xgboost-model"
                current_date = getattr(self.config, 'current_date', "2025-06-02")
                prefix = f"s3://{bucket}/{pipeline_name}/training_config/{current_date}" # No trailing slash
            
            # Use our helper methods for consistent path handling
            config_dir = self._normalize_s3_uri(prefix, "hyperparameters prefix")
            self.log_info("Normalized hyperparameters prefix: %s", config_dir)
            
            # Check if hyperparameters.json is already in the path
            if S3PathHandler.get_name(config_dir) == "hyperparameters.json":
                # Use path as is if it already includes the filename
                target_s3_uri = config_dir
                self.log_info("Using existing hyperparameters path: %s", target_s3_uri)
            else:
                # Otherwise append the filename using S3PathHandler.join for proper path handling
                target_s3_uri = S3PathHandler.join(config_dir, "hyperparameters.json")
                self.log_info("Constructed hyperparameters path: %s", target_s3_uri)
                
            self.log_info("Using hyperparameters S3 target URI: %s", target_s3_uri)

            # Check if file exists and handle appropriately
            s3_parts = target_s3_uri.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
            
            s3_client = self.session.boto_session.client('s3')
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                self.log_info("Found existing hyperparameters file at %s", target_s3_uri)
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.log_info("No existing hyperparameters file found at %s", target_s3_uri)
                else:
                    self.log_warning("Error checking existing file: %s", str(e))

            # Upload the file
            self.log_info("Uploading hyperparameters from %s to %s", local_file, target_s3_uri)
            S3Uploader.upload(str(local_file), target_s3_uri, sagemaker_session=self.session)
            
            self.log_info("Hyperparameters successfully uploaded to %s", target_s3_uri)
            return target_s3_uri
        
        finally:
            # Clean up temporary files
            shutil.rmtree(local_dir)
            
    def _validate_s3_uri(self, uri: str, description: str = "data") -> bool:
        """
        Validates that a string is a properly formatted S3 URI.
        Uses S3PathHandler for consistent path validation.
        
        Args:
            uri: The URI to validate
            description: Description of what the URI is for (used in error messages)
            
        Returns:
            True if valid, False otherwise
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            # For PipelineVariables, we trust they'll resolve to valid URIs at execution time
            return True
            
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            # For Get expressions, we also trust they'll resolve properly at execution time
            self.log_info("Found Pipeline step reference: %s", uri)
            return True
        
        if not isinstance(uri, str):
            self.log_warning("Invalid %s URI: type %s", description, type(uri).__name__)
            return False
        
        # Use S3PathHandler for validation
        valid = S3PathHandler.is_valid(uri)
        if not valid:
            self.log_warning("Invalid %s URI format: %s", description, uri)
        
        return valid
    
    def create_step(self, **kwargs) -> TrainingStep:
        """
        Creates a SageMaker TrainingStep for the pipeline.
        
        This method creates the XGBoost estimator, sets up training inputs from the input data,
        uploads hyperparameters, and creates the SageMaker TrainingStep.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Dictionary mapping input channel names to their S3 locations
                - input_path: Direct parameter for training data input path (for backward compatibility)
                - output_path: Direct parameter for model output path (for backward compatibility)
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.
                
        Returns:
            A configured TrainingStep instance.
        """
        # Extract common parameters
        inputs_raw = kwargs.get('inputs', {})
        input_path = kwargs.get('input_path')
        output_path = kwargs.get('output_path')
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        self.log_info("Creating XGBoost TrainingStep...")
        
        # Get the step name using standardized automatic step type detection
        step_name = self._get_step_name()
        
        # Handle inputs
        inputs = {}
        
        # If dependencies are provided, extract inputs from them using the resolver
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
                
        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)
        
        # Add direct parameters if provided
        if input_path is not None:
            inputs["input_path"] = input_path
            
        # Get training inputs using specification-driven method
        # Note: _get_inputs now handles generating hyperparameters internally
        training_inputs = self._get_inputs(inputs)
        
        # Make sure we have the inputs we need
        if len(training_inputs) == 0:
            raise ValueError("No training inputs available. Provide input_path or ensure dependencies supply necessary outputs.")
        
        self.log_info("Final training inputs: %s", list(training_inputs.keys()))
        
        # Get output path using specification-driven method
        output_path = self._get_outputs({})
        
        # Create estimator
        estimator = self._create_estimator(output_path)
        
        # Create the training step
        try:
            training_step = TrainingStep(
                name=step_name,
                estimator=estimator,
                inputs=training_inputs,
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching)
            )
            
            # Attach specification to the step for future reference
            setattr(training_step, '_spec', self.spec)
            
            # Log successful creation
            self.log_info("Created TrainingStep with name: %s", training_step.name)
            
            return training_step
            
        except Exception as e:
            self.log_warning("Error creating XGBoost TrainingStep: %s", str(e))
            raise ValueError(f"Failed to create XGBoostTrainingStep: {str(e)}") from e
