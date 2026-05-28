"""
EdxUploading Step Builder.

Standard ProcessingStep builder that uploads S3 data to EDX via EdxDataLoader.
Uses ScriptProcessor with SAIS Docker image. SINK node (no outputs).
"""

from typing import Dict, Optional, Any, List, TYPE_CHECKING
import json
import logging

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ScriptProcessor

if TYPE_CHECKING:
    from ...core.deps.registry_manager import RegistryManager
    from ...core.deps.dependency_resolver import UnifiedDependencyResolver

from ..configs.config_edx_uploading_step import EdxUploadingConfig
from ..specs.edx_uploading_spec import EDX_UPLOADING_SPEC
from ...core.base.builder_base import StepBuilderBase

try:
    from ..contracts.edx_uploading_contract import EDX_UPLOADING_CONTRACT

    CONTRACT_AVAILABLE = True
except ImportError:
    EDX_UPLOADING_CONTRACT = None
    CONTRACT_AVAILABLE = False

from mods_workflow_core.utils.constants import (
    KMS_ENCRYPTION_KEY_PARAM,
    PROCESSING_JOB_SHARED_NETWORK_CONFIG,
)

logger = logging.getLogger(__name__)


class EdxUploadingStepBuilder(StepBuilderBase):
    """
    Builder for EdxUploading ProcessingStep.

    Creates a ScriptProcessor with the SAIS Docker image and mounts
    upstream S3 data as ProcessingInput. The script uploads files to EDX.
    """

    def __init__(
        self,
        config: EdxUploadingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None,
    ):
        if not isinstance(config, EdxUploadingConfig):
            raise ValueError(
                "EdxUploadingStepBuilder requires an EdxUploadingConfig instance."
            )

        super().__init__(
            config=config,
            spec=EDX_UPLOADING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: EdxUploadingConfig = config
        self.contract = EDX_UPLOADING_CONTRACT if CONTRACT_AVAILABLE else None

    def validate_configuration(self) -> None:
        """Validate required configuration fields."""
        self.log_info("Validating EdxUploadingConfig...")

        if not self.config.edx_provider:
            raise ValueError("edx_provider is required")
        if not self.config.edx_subject:
            raise ValueError("edx_subject is required")
        if not self.config.edx_dataset:
            raise ValueError("edx_dataset is required")
        if not self.config.edx_manifest_key:
            raise ValueError("edx_manifest_key is required")

        self.log_info("EdxUploadingConfig validation succeeded.")

    def _get_environment_variables(self) -> Dict[str, str]:
        """Build environment variables for the processing container."""
        env_vars = {
            "AWS_STS_REGIONAL_ENDPOINTS": "regional",
            "AWS_DEFAULT_REGION": self.config.aws_region or "us-east-1",
            "EDX_DATASET_ARN": self.config.edx_arn_base,
            "EDX_MANIFEST_KEY": self.config.edx_manifest_key,
        }

        if self.config.edx_manifest_key_parts:
            env_vars["EDX_MANIFEST_KEY_PARTS"] = json.dumps(
                self.config.edx_manifest_key_parts
            )

        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Mount upstream S3 data as ProcessingInput."""
        dep_logical_name = "input_data"

        if dep_logical_name in inputs:
            source = inputs[dep_logical_name]
            self.log_info("Resolved input_data from upstream DAG")
        elif (
            hasattr(self.config, "input_s3_location") and self.config.input_s3_location
        ):
            source = self.config.input_s3_location
            self.log_info("Using config input_s3_location override: %s", source)
        else:
            raise ValueError(
                f"Required input '{dep_logical_name}' not provided and "
                "no input_s3_location override in config."
            )

        return [
            ProcessingInput(
                source=source,
                destination="/opt/ml/processing/input/data",
                input_name="input_data",
            )
        ]

    def _get_outputs(self, outputs: Dict[str, Any]) -> List:
        """SINK node — no outputs."""
        return []

    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the EdxUploading ProcessingStep."""
        inputs_raw = kwargs.get("inputs", {})
        dependencies = kwargs.get("dependencies", [])

        self.validate_configuration()

        step_name = self._get_step_name()
        self.log_info("Creating EdxUploading ProcessingStep: %s", step_name)

        instance_type = (
            self.config.processing_instance_type_large
            if self.config.use_large_processing_instance
            else self.config.processing_instance_type_small
        )

        processor = ScriptProcessor(
            image_uri=f"{self.role.split(':')[4]}.dkr.ecr.{self.config.aws_region or 'us-east-1'}.amazonaws.com/sais_python_lib_docker_image",
            role=self.role,
            instance_count=self.config.processing_instance_count,
            instance_type=instance_type,
            volume_size_in_gb=self.config.processing_volume_size,
            command=["python3"],
            sagemaker_session=self.session,
            volume_kms_key=KMS_ENCRYPTION_KEY_PARAM,
            network_config=PROCESSING_JOB_SHARED_NETWORK_CONFIG,
            env=self._get_environment_variables(),
        )

        processing_inputs = self._get_inputs(inputs_raw)

        script_path = self._resolve_script_path()

        step = ProcessingStep(
            name=step_name,
            processor=processor,
            code=script_path,
            inputs=processing_inputs,
            outputs=[],
        )

        if dependencies:
            step.add_depends_on(dependencies)

        self.log_info("Created EdxUploading ProcessingStep: %s", step_name)
        return step

    def _resolve_script_path(self) -> str:
        """Resolve the path to edx_upload.py."""
        if self.config.processing_source_dir:
            from pathlib import Path

            source_dir = Path(self.config.processing_source_dir)
            script = source_dir / self.config.processing_entry_point
            if script.exists():
                return str(script)

        if (
            hasattr(self.config, "effective_source_dir")
            and self.config.effective_source_dir
        ):
            from pathlib import Path

            return str(
                Path(self.config.effective_source_dir)
                / self.config.processing_entry_point
            )

        from pathlib import Path

        return str(
            Path(__file__).parent.parent
            / "scripts"
            / self.config.processing_entry_point
        )
