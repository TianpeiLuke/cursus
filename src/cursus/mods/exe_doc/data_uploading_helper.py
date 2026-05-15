"""
Data Uploading Helper for execution document generation.

This module provides the DataUploadingHelper class that extracts execution
document configurations from DataUploading step configurations.
"""

import logging
from typing import Dict, Any

from .base import ExecutionDocumentHelper, ExecutionDocumentGenerationError

try:
    from ...steps.configs.config_data_uploading_step import DataUploadingConfig

    DATA_UPLOADING_CONFIG_AVAILABLE = True
except ImportError:
    DATA_UPLOADING_CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataUploadingHelper(ExecutionDocumentHelper):
    """
    Helper for extracting execution document configurations from DataUploading steps.

    Unlike CradleDataLoadingHelper (which builds Coral model objects and serializes),
    this helper is simple: DataUploadingConfig.job_config already produces the exact
    dict expected by the execution document / CreateDataUploadJobRequest.
    """

    def __init__(self):
        """Initialize the DataUploading helper."""
        self.logger = logging.getLogger(__name__)

    def can_handle_step(self, step_name: str, config) -> bool:
        """
        Check if this helper can handle the given step configuration.

        Args:
            step_name: Name of the step
            config: Step configuration object

        Returns:
            True if this helper can handle the configuration, False otherwise
        """
        if DATA_UPLOADING_CONFIG_AVAILABLE:
            try:
                if isinstance(config, DataUploadingConfig):
                    return True
            except Exception:
                pass

        config_type_name = type(config).__name__.lower()
        return (
            "datauploading" in config_type_name or "data_uploading" in config_type_name
        )

    def get_execution_step_name(self, step_name: str, config) -> str:
        """
        Get execution document step name.

        DataUploading steps don't typically have job_type variants,
        so the step name is used as-is or with minor normalization.

        Args:
            step_name: Original step name from DAG
            config: Configuration object

        Returns:
            Execution document step name
        """
        if hasattr(config, "job_type") and config.job_type:
            job_type = config.job_type
            suffix = f"_{job_type.lower()}"
            if step_name.endswith(suffix):
                base_name = step_name[: -len(suffix)]
            else:
                base_name = step_name
            return f"{base_name}-{job_type.capitalize()}"

        return step_name

    def extract_step_config(self, step_name: str, config) -> Dict[str, Any]:
        """
        Extract execution document configuration from DataUploading step config.

        This is straightforward: DataUploadingConfig.job_config already assembles
        the exact dict needed for the execution document (matches the sample_config
        format expected by DataUploadProcessor's JSON Schema validation).

        Args:
            step_name: Name of the step
            config: DataUploading configuration object

        Returns:
            Dictionary containing the execution document configuration

        Raises:
            ExecutionDocumentGenerationError: If configuration extraction fails
        """
        try:
            self.logger.info(
                f"Extracting DataUploading execution document config for step: {step_name}"
            )

            if hasattr(config, "job_config"):
                result = config.job_config
            else:
                result = self._build_config_dict(config)

            self.logger.info(
                f"Successfully extracted DataUploading config for step: {step_name}"
            )
            return result

        except Exception as e:
            self.logger.error(
                f"Failed to extract DataUploading config for step {step_name}: {e}"
            )
            raise ExecutionDocumentGenerationError(
                f"DataUploading configuration extraction failed for step {step_name}: {e}"
            ) from e

    def _build_config_dict(self, config) -> Dict[str, Any]:
        """
        Fallback: manually build the config dict if job_config property not available.

        Args:
            config: Configuration object with individual fields

        Returns:
            Dictionary matching CreateDataUploadJobRequest schema
        """
        return {
            "schema": getattr(config, "schema_sdl", ""),
            "inputFormat": getattr(config, "input_format", "CSV"),
            "cradleAccount": getattr(config, "cradle_account", ""),
            "hasHeader": getattr(config, "has_header", True),
            "clusterType": getattr(config, "cluster_type", "MEDIUM"),
            "dataSink": {
                "dataSinkType": "ANDES",
                "andesDataSinkProperties": {
                    "providerName": getattr(config, "provider_name", ""),
                    "tableName": getattr(config, "table_name", ""),
                    "tableVersion": getattr(config, "table_version", 1),
                },
            },
        }
