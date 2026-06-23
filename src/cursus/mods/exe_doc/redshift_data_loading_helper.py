"""
Redshift Data Loading Helper for execution document generation.

Converts RedshiftDataLoadingConfig Pydantic fields into the 3-specification
JSON structure (clusterSpecification, querySpecification, outputSpecification)
expected by the SAIS SDK script at /opt/ml/processing/config/config.
"""

import logging
from typing import Dict, Any

from .base import ExecutionDocumentHelper, ExecutionDocumentGenerationError

try:
    from ...steps.configs.config_redshift_data_loading_step import (
        RedshiftDataLoadingConfig,
    )

    REDSHIFT_CONFIG_AVAILABLE = True
except ImportError:
    REDSHIFT_CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedshiftDataLoadingHelper(ExecutionDocumentHelper):
    """
    Helper for extracting execution document configurations from RedshiftDataLoading steps.

    Converts flat Cursus Pydantic fields into the 3-specification JSON structure
    (clusterSpecification, querySpecification, outputSpecification) expected by
    the SAIS SDK Redshift script.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def can_handle_step(self, step_name: str, config) -> bool:
        """Check if this helper can handle the given step configuration."""
        if REDSHIFT_CONFIG_AVAILABLE:
            try:
                if isinstance(config, RedshiftDataLoadingConfig):
                    return True
            except Exception:
                pass

        config_type_name = type(config).__name__.lower()
        return "redshift" in config_type_name

    def get_execution_step_name(self, step_name: str, config) -> str:
        """
        Get execution document step name.

        Handles job_type suffix: RedshiftDataLoading_mo_orders → RedshiftDataLoading-MoOrders
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
        Extract 3-specification JSON from RedshiftDataLoading config.

        Uses config.step_config_json if available (computed property),
        otherwise builds from individual fields.

        Returns:
            Dict matching the SAIS SDK's 3-spec structure:
            {clusterSpecification, querySpecification, outputSpecification}
        """
        try:
            self.logger.info(
                f"Extracting RedshiftDataLoading execution document for: {step_name}"
            )

            if hasattr(config, "step_config_json"):
                result = config.step_config_json
            else:
                result = self._build_config_dict(config)

            self.logger.info(f"Successfully extracted config for: {step_name}")
            return result

        except Exception as e:
            self.logger.error(
                f"Failed to extract RedshiftDataLoading config for step {step_name}: {e}"
            )
            raise ExecutionDocumentGenerationError(
                f"RedshiftDataLoading config extraction failed for {step_name}: {e}"
            ) from e

    def _build_config_dict(self, config) -> Dict[str, Any]:
        """Fallback: build 3-spec dict from individual config fields."""
        result: Dict[str, Any] = {
            "clusterSpecification": {
                "clusterId": getattr(config, "cluster_id", ""),
                "dbName": getattr(config, "db_name", ""),
                "roleArn": getattr(config, "role_arn", ""),
                "clusterEndpoint": getattr(config, "cluster_endpoint", ""),
                "port": getattr(config, "port", 5439),
            },
            "querySpecification": {
                "query": getattr(config, "query", ""),
                "connectorType": getattr(
                    config, "connector_type", "redshift_connector"
                ),
                "isUsingAndes3": getattr(config, "is_using_andes3", True),
            },
            "outputSpecification": {},
        }

        output_type = getattr(config, "output_data_source_type", "S3")
        if output_type == "EDX":
            result["outputSpecification"] = {
                "dataSourceType": "EDX",
                "edxArn": getattr(config, "edx_arn", ""),
                "dropHeader": getattr(config, "drop_header", False),
            }
        elif getattr(config, "drop_header", False):
            result["outputSpecification"] = {"dropHeader": True}

        return result

    def get_step_type(self) -> list:
        """Return STEP_TYPE for execution document."""
        return ["WORKFLOW_INPUT", "RedshiftDataLoadingStep"]
