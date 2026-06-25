"""
Data Uploading Step Builder.

SDK delegation builder that wraps DataUploadingStep from the SAIS SDK.
Uploads data from S3 to Andes. This is a SINK node (no outputs).
"""

from typing import Dict, Any, Optional, List
from sagemaker.workflow.steps import Step
import logging

from secure_ai_sandbox_workflow_python_sdk.data_uploading.data_uploading_step import (
    DataUploadingStep,
)

from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver
from ..configs.config_data_uploading_step import DataUploadingConfig

logger = logging.getLogger(__name__)


class DataUploadingStepBuilder(StepBuilderBase):
    """
    Builder for DataUploading step via SDK delegation.

    Wraps DataUploadingStep from secure_ai_sandbox_workflow_python_sdk.
    Input S3 location is resolved from the upstream step's output via the
    assembler (Properties pipeline variable), with an optional config override
    for manual/debug execution.
    """

    def __init__(
        self,
        config: DataUploadingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None,
    ):
        if not isinstance(config, DataUploadingConfig):
            raise ValueError(
                "DataUploadingStepBuilder requires a DataUploadingConfig instance."
            )

        from ..interfaces import load_step_interface

        contract, spec = load_step_interface("DataUploading")

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: DataUploadingConfig = config
        self.contract = contract
        self._resolved_input_s3 = None

    def validate_configuration(self) -> None:
        """Validate config fields against API constraints."""
        self.log_info("Validating DataUploadingConfig...")

        if not self.config.provider_name:
            raise ValueError("provider_name is required")
        if not self.config.table_name:
            raise ValueError("table_name is required")
        if self.config.table_version < 1:
            raise ValueError("table_version must be >= 1")
        if not self.config.cradle_account:
            raise ValueError("cradle_account is required")
        if not self.config.schema_sdl:
            raise ValueError("schema_sdl is required")

        self.log_info("DataUploadingConfig validation succeeded.")

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Extract resolved input_data from assembler-provided inputs.

        The assembler resolves upstream output S3 via PropertyReference and passes
        it as a SageMaker Properties object (pipeline variable). We store it for
        use in create_step() where it's passed to DataUploadingStep(input_s3_location=...).
        The SDK internally converts it to --input-path job argument via
        DataUploadProcessor.get_processing_job_arguments() — builder does NOT handle
        job arguments directly.

        Falls back to config.input_s3_location for manual/debug execution.

        Returns empty list (no ProcessingInput objects — SDK handles input delivery).
        """
        dep_logical_name = "input_data"

        if dep_logical_name in inputs:
            self._resolved_input_s3 = inputs[dep_logical_name]
            self.log_info("Resolved input_data from upstream DAG (pipeline variable)")
        elif self.config.input_s3_location:
            self._resolved_input_s3 = self.config.input_s3_location
            self.log_info(
                "Using config.input_s3_location override: %s",
                self.config.input_s3_location,
            )
        else:
            raise ValueError(
                f"Required input '{dep_logical_name}' not provided and "
                "config.input_s3_location is not set. "
                "Ensure upstream step with compatible output is connected in DAG, "
                "or provide input_s3_location in config for manual execution."
            )

        return []

    def _get_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sink node — no outputs."""
        return {}

    def create_step(self, **kwargs) -> Step:
        """
        Create DataUploadingStep with resolved input S3 location.

        The input_s3_location is either:
        - A SageMaker Properties object (pipeline variable) from the assembler
        - A string S3 URI from config (manual/debug mode)

        In pipeline mode, SageMaker serializes the Properties object as
        {"Get": "Steps.Upstream...S3Uri"} and resolves at execution time.
        """
        inputs_raw = kwargs.get("inputs", {})
        dependencies = kwargs.get("dependencies", [])

        self._get_inputs(inputs_raw)

        step_name = self._get_step_name()
        self.log_info("Creating DataUploadingStep: %s", step_name)

        try:
            step = DataUploadingStep(
                step_name=step_name,
                role=self.role,
                sagemaker_session=self.session,
                input_s3_location=self._resolved_input_s3,
            )

            if dependencies:
                step.add_depends_on(dependencies)

            if self.spec:
                setattr(step, "_spec", self.spec)
            if self.contract:
                setattr(step, "_contract", self.contract)

            self.log_info("Created DataUploadingStep: %s", step.name)
            return step

        except Exception as e:
            self.log_error("Error creating DataUploadingStep: %s", e)
            raise ValueError(f"Failed to create DataUploadingStep: {e}") from e
