"""
Redshift Data Loading Step Builder — SDK Delegation.

Imports RedshiftDataLoadingStep from SAIS SDK directly (same pattern as
CradleDataLoading and DataUploading). The SDK step internally creates its
own processor, ProcessingOutput, env vars, image URI, and instance config.

SOURCE node — no upstream data inputs. Config delivered via execution document.
"""

import logging
from typing import Dict, Any, List, Optional

from sagemaker.workflow.steps import Step

from secure_ai_sandbox_workflow_python_sdk.redshift_data_loading.redshift_data_loading_step import (
    RedshiftDataLoadingStep,
)

from ...core.base.builder_base import StepBuilderBase
from ...core.deps.registry_manager import RegistryManager
from ...core.deps.dependency_resolver import UnifiedDependencyResolver
from ..configs.config_redshift_data_loading_step import RedshiftDataLoadingConfig

logger = logging.getLogger(__name__)


class RedshiftDataLoadingStepBuilder(StepBuilderBase):
    """
    SDK delegation builder for RedshiftDataLoading.

    Wraps RedshiftDataLoadingStep from SAIS SDK.
    SOURCE node — no upstream inputs. Config delivered via execution document.
    """

    def __init__(
        self,
        config: RedshiftDataLoadingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None,
    ):
        if not isinstance(config, RedshiftDataLoadingConfig):
            raise ValueError(
                "RedshiftDataLoadingStepBuilder requires a RedshiftDataLoadingConfig instance."
            )

        from ..interfaces import load_step_interface

        _contract, spec = load_step_interface("RedshiftDataLoading")

        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: RedshiftDataLoadingConfig = config

    def validate_configuration(self) -> None:
        """Validate required config fields."""
        self.log_info("Validating RedshiftDataLoadingConfig...")

        required_attrs = [
            "cluster_id",
            "db_name",
            "role_arn",
            "cluster_endpoint",
            "query",
        ]
        for attr in required_attrs:
            if not getattr(self.config, attr, None):
                raise ValueError(
                    f"RedshiftDataLoadingConfig missing required attribute: {attr}"
                )

        if self.config.output_data_source_type == "EDX" and not self.config.edx_arn:
            raise ValueError("edx_arn is required when output_data_source_type='EDX'")

        self.log_info("RedshiftDataLoadingConfig validation succeeded.")

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[Any]:
        """SOURCE node — no upstream data inputs."""
        return []

    def _get_outputs(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """SDK step creates ProcessingOutput internally — return empty."""
        if self.contract and hasattr(self.contract, "expected_output_paths"):
            for (
                logical_name,
                container_path,
            ) in self.contract.expected_output_paths.items():
                self.log_info(
                    "Contract defines output path for '%s': %s",
                    logical_name,
                    container_path,
                )
        return {}

    def create_step(self, **kwargs) -> Step:
        """
        Create RedshiftDataLoadingStep via SDK delegation.

        The SDK step internally handles: processor creation, Docker image URI,
        instance type/volume, ProcessingOutput with Join-based S3 destination,
        env vars, cache config, and max runtime.
        """
        dependencies = kwargs.get("dependencies", [])
        enable_caching = kwargs.get("enable_caching", True)

        step_name = self._get_step_name()
        self.log_info("Creating RedshiftDataLoadingStep: %s", step_name)

        step = RedshiftDataLoadingStep(
            step_name=step_name,
            role=self.role,
            sagemaker_session=self.session,
        )

        if dependencies:
            step.add_depends_on(dependencies)

        if not enable_caching and hasattr(step, "cache_config"):
            step.cache_config.enable_caching = False

        if self.spec:
            setattr(step, "_spec", self.spec)
        if self.contract:
            setattr(step, "_contract", self.contract)

        self.log_info("Created RedshiftDataLoadingStep: %s", step.name)
        return step
