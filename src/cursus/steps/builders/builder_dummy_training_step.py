"""
MIMS DummyTraining Step Builder.

This module defines the builder that creates SageMaker processing steps
for the DummyTraining component, which processes a pretrained model with
hyperparameters to make it available for downstream packaging and payload steps.
"""

import logging
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.functions import Join
from sagemaker.s3 import S3Uploader
from botocore.exceptions import ClientError

from ..configs.config_dummy_training_step import DummyTrainingConfig
from ...core.base.builder_base import StepBuilderBase
from .s3_utils import S3PathHandler
from ..specs.dummy_training_spec import DUMMY_TRAINING_SPEC
from ...registry.builder_registry import register_builder

logger = logging.getLogger(__name__)


@register_builder()
class DummyTrainingStepBuilder(StepBuilderBase):
    """Builder for DummyTraining processing step that handles pretrained model processing with hyperparameters."""

    def __init__(
        self,
        config: DummyTrainingConfig,
        sagemaker_session=None,
        role=None,
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None,
    ):
        """Initialize the DummyTraining step builder.

        Args:
            config: Configuration for the DummyTraining step
            sagemaker_session: SageMaker session to use
            role: IAM role for SageMaker execution
            notebook_root: Root directory for notebook execution
            registry_manager: Registry manager for dependency injection
            dependency_resolver: Dependency resolver for dependency injection

        Raises:
            ValueError: If config is not a DummyTrainingConfig instance
        """
        if not isinstance(config, DummyTrainingConfig):
            raise ValueError(
                "DummyTrainingStepBuilder requires a DummyTrainingConfig instance."
            )

        super().__init__(
            config=config,
            spec=DUMMY_TRAINING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: DummyTrainingConfig = config

    def validate_configuration(self):
        """
        Validate the provided configuration.

        Raises:
            ValueError: If pretrained_model_path is not provided
            FileNotFoundError: If the pretrained model file doesn't exist
        """
        self.log_info("Validating DummyTrainingConfig...")

        # Check for required local file
        if not self.config.pretrained_model_path:
            raise ValueError("pretrained_model_path is required in DummyTrainingConfig")

        # Check if file exists (if path is concrete and not a variable)
        if not hasattr(self.config.pretrained_model_path, "expr"):
            model_path = Path(self.config.pretrained_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Pretrained model not found at {model_path}")

            # Additional validation: check file extension
            if not model_path.suffix == ".tar.gz" and not str(model_path).endswith(
                ".tar.gz"
            ):
                self.log_warning(
                    f"Model file {model_path} does not have .tar.gz extension"
                )

        # Check for hyperparameters
        if (
            not hasattr(self.config, "hyperparameters")
            or not self.config.hyperparameters
        ):
            raise ValueError(
                "Model hyperparameters are required in DummyTrainingConfig"
            )

        self.log_info("DummyTrainingConfig validation succeeded.")




    def _upload_model_to_s3(self) -> str:
        """
        Upload the pretrained model to S3.

        Returns:
            S3 URI where the model was uploaded

        Raises:
            Exception: If upload fails
        """
        self.log_info(
            f"Uploading pretrained model from {self.config.pretrained_model_path}"
        )

        # Construct target S3 URI using base output path and Join for parameter compatibility
        from sagemaker.workflow.functions import Join
        base_output_path = self._get_base_output_path()
        target_s3_uri = Join(on="/", values=[base_output_path, "dummy_training", "input", "model.tar.gz"])

        try:
            # Upload the file - S3Uploader handles Join objects properly
            S3Uploader.upload(
                self.config.pretrained_model_path,
                target_s3_uri,
                sagemaker_session=self.session,
            )

            self.log_info(f"Uploaded model to {target_s3_uri}")
            return target_s3_uri
        except Exception as e:
            self.log_error(f"Failed to upload model to S3: {e}")
            import traceback

            self.log_error(traceback.format_exc())
            raise

    def _prepare_hyperparameters_file(self) -> str:
        """
        Serializes the hyperparameters to JSON, uploads it to S3, and
        returns that full S3 URI.

        This method is optimized for AWS Lambda execution with proper error handling,
        resource management, and disk space considerations.

        Returns:
            S3 URI or Join object that will resolve to the hyperparameters file location

        Raises:
            ValueError: If hyperparameters serialization fails
            Exception: If S3 upload fails after retries
        """
        # Serialize hyperparameters with error handling
        try:
            hyperparams_dict = self.config.hyperparameters.model_dump()
        except Exception as e:
            raise ValueError(f"Failed to serialize hyperparameters: {e}") from e

        # Use Lambda-friendly temporary directory with unique naming
        import uuid
        unique_id = str(uuid.uuid4())[:8]  # Short unique identifier
        local_dir = Path(tempfile.gettempdir()) / f"hyperparams_{unique_id}"
        local_file = local_dir / "hyperparameters.json"

        try:
            # Create directory and ensure it exists
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Write JSON locally with error handling
            try:
                with open(local_file, "w", encoding='utf-8') as f:
                    json.dump(hyperparams_dict, f, indent=2, ensure_ascii=False)
                self.log_info("Created hyperparameters JSON file at %s (size: %d bytes)", 
                             local_file, local_file.stat().st_size)
            except (IOError, OSError) as e:
                raise ValueError(f"Failed to write hyperparameters file: {e}") from e

            # Construct target S3 URI using base output path and Join for parameter compatibility
            # Always use the unified system path for consistency and portability
            from sagemaker.workflow.functions import Join
            base_output_path = self._get_base_output_path()
            target_s3_uri = Join(on="/", values=[base_output_path, "training_config", "hyperparameters.json"])
            self.log_info("Using unified system hyperparameters path: %s", target_s3_uri)

            # Upload the file with enhanced error handling for Lambda environment
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.log_info("Uploading hyperparameters from %s to %s (attempt %d/%d)", 
                                 local_file, target_s3_uri, attempt + 1, max_retries)
                    
                    S3Uploader.upload(
                        str(local_file), 
                        target_s3_uri, 
                        sagemaker_session=self.session
                    )
                    
                    self.log_info("Hyperparameters successfully uploaded to %s", target_s3_uri)
                    return target_s3_uri
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        self.log_error("Failed to upload hyperparameters after %d attempts: %s", max_retries, e)
                        raise Exception(f"S3 upload failed after {max_retries} attempts: {e}") from e
                    else:
                        self.log_warning("Upload attempt %d failed, retrying: %s", attempt + 1, e)
                        import time
                        time.sleep(2 ** attempt)  # Exponential backoff

        except Exception as e:
            self.log_error("Error in _prepare_hyperparameters_file: %s", e)
            raise
        finally:
            # Robust cleanup - ensure temporary files are removed even if upload fails
            try:
                if local_dir.exists():
                    shutil.rmtree(local_dir, ignore_errors=True)
                    self.log_debug("Cleaned up temporary directory: %s", local_dir)
            except Exception as cleanup_error:
                self.log_warning("Failed to clean up temporary directory %s: %s", local_dir, cleanup_error)
                # Don't raise cleanup errors - they shouldn't fail the main operation

    def _get_processor(self):
        """
        Get the processor for the step.

        Returns:
            SKLearnProcessor: Configured processor for running the step
        """
        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=self.config.get_instance_type(),
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Create environment variables for the processing job.

        Returns:
            Dict[str, str]: Environment variables for the processing job
        """
        # Get base environment variables from contract
        env_vars = super()._get_environment_variables()

        # Add any specific environment variables needed for DummyTraining
        # For example, we could add model paths or other configuration settings

        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the processor using the specification and contract.

        Args:
            inputs: Dictionary of input sources keyed by logical name

        Returns:
            List of ProcessingInput objects for the processor

        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for input mapping")

        processing_inputs = []

        # Use either the uploaded model or one provided through dependencies
        model_s3_uri = inputs.get("pretrained_model_path")
        if not model_s3_uri:
            # Upload the local model file if no S3 path is provided
            model_s3_uri = self._upload_model_to_s3()

        # Handle PipelineVariable objects
        if hasattr(model_s3_uri, "expr"):
            self.log_info(
                f"Processing PipelineVariable for model_s3_uri: {model_s3_uri.expr}"
            )

        # Get container path from contract for model
        model_container_path = self.contract.expected_input_paths.get(
            "pretrained_model_path"
        )
        if not model_container_path:
            raise ValueError(
                "Script contract missing required input path: pretrained_model_path"
            )

        # Add model input
        processing_inputs.append(
            ProcessingInput(
                source=model_s3_uri,
                destination=os.path.dirname(model_container_path),
                input_name="model",
            )
        )

        # Handle hyperparameters - either use the provided one or generate a new one
        hyperparams_s3_uri = inputs.get("hyperparameters_s3_uri")
        if not hyperparams_s3_uri:
            # Generate hyperparameters JSON and upload to S3
            hyperparams_s3_uri = self._prepare_hyperparameters_file()

        # Handle PipelineVariable objects
        if hasattr(hyperparams_s3_uri, "expr"):
            self.log_info(
                f"Processing PipelineVariable for hyperparams_s3_uri: {hyperparams_s3_uri.expr}"
            )

        # Get container path from contract for hyperparameters
        hyperparams_container_path = self.contract.expected_input_paths.get(
            "hyperparameters_s3_uri"
        )
        if not hyperparams_container_path:
            raise ValueError(
                "Script contract missing required input path: hyperparameters_s3_uri"
            )

        # Add hyperparameters input
        processing_inputs.append(
            ProcessingInput(
                source=hyperparams_s3_uri,
                destination=os.path.dirname(hyperparams_container_path),
                input_name="config",
            )
        )

        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Get outputs for the processor using the specification and contract.

        Args:
            outputs: Dictionary of output destinations keyed by logical name

        Returns:
            List of ProcessingOutput objects for the processor

        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")

        if not self.contract:
            raise ValueError("Script contract is required for output mapping")

        # Use the pipeline S3 location to construct output path using base output path and Join for parameter compatibility
        from sagemaker.workflow.functions import Join
        base_output_path = self._get_base_output_path()
        default_output_path = Join(on="/", values=[base_output_path, "dummy_training", "output"])
        output_path = outputs.get("model_input", default_output_path)

        # Handle PipelineVariable objects in output_path
        if hasattr(output_path, "expr"):
            self.log_info(
                f"Processing PipelineVariable for output_path: {output_path.expr}"
            )

        # Get source path from contract
        source_path = self.contract.expected_output_paths.get("model_input")
        if not source_path:
            raise ValueError(
                "Script contract missing required output path: model_input"
            )

        return [
            ProcessingOutput(
                output_name="model_input",  # Using consistent name matching specification
                source=source_path,
                destination=output_path,
            )
        ]

    def _get_job_arguments(self) -> Optional[List[str]]:
        """
        Returns None as job arguments since the dummy training script now uses
        standard paths defined directly in the script.

        Returns:
            None since no arguments are needed
        """
        self.log_info("No command-line arguments needed for dummy training script")
        return None

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Create the processing step.

        Args:
            **kwargs: Additional keyword arguments for step creation including:
                     - inputs: Dictionary of input sources keyed by logical name
                     - outputs: Dictionary of output destinations keyed by logical name
                     - dependencies: List of steps this step depends on
                     - enable_caching: Whether to enable caching for this step

        Returns:
            ProcessingStep: The configured processing step

        Raises:
            ValueError: If inputs cannot be extracted
            Exception: If step creation fails
        """
        try:
            # Extract parameters
            inputs_raw = kwargs.get("inputs", {})
            outputs = kwargs.get("outputs", {})
            dependencies = kwargs.get("dependencies", [])
            enable_caching = kwargs.get("enable_caching", True)

            # Handle inputs
            inputs = {}

            # If dependencies are provided, extract inputs from them
            if dependencies:
                try:
                    extracted_inputs = self.extract_inputs_from_dependencies(
                        dependencies
                    )
                    inputs.update(extracted_inputs)
                except Exception as e:
                    self.log_warning(
                        "Failed to extract inputs from dependencies: %s", e
                    )

            # Add explicitly provided inputs (overriding any extracted ones)
            inputs.update(inputs_raw)

            # Create processor and get inputs/outputs
            processor = self._get_processor()
            processing_inputs = self._get_inputs(inputs)
            processing_outputs = self._get_outputs(outputs)

            # Get step name using standardized method with auto-detection
            step_name = self._get_step_name()

            # Get job arguments from contract
            script_args = self._get_job_arguments()

            # Create the step using direct ProcessingStep instantiation
            step = ProcessingStep(
                name=step_name,
                processor=processor,
                inputs=processing_inputs,
                outputs=processing_outputs,
                code=self.config.get_script_path(),
                job_arguments=script_args,
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching),
            )

            # Store specification in step for future reference
            setattr(step, "_spec", self.spec)

            return step

        except Exception as e:
            self.log_error(f"Error creating DummyTraining step: {e}")
            import traceback

            self.log_error(traceback.format_exc())
            raise ValueError(f"Failed to create DummyTraining step: {str(e)}") from e
