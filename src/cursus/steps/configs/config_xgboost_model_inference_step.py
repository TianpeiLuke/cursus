"""
Model Inference Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the XGBoost model inference step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, model_validator, field_validator
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.xgboost_model_inference_contract import XGBOOST_MODEL_INFERENCE_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class XGBoostModelInferenceConfig(ProcessingStepConfigBase):
    """
    Configuration for XGBoost model inference step with self-contained derivation logic.

    This class defines the configuration parameters for the XGBoost model inference step,
    which generates predictions from trained models without computing evaluation metrics.
    This is designed for pure inference workflows where predictions are needed for
    downstream processing (e.g., model calibration, batch scoring).

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    id_name: str = Field(
        ...,
        description="Name of the ID field in the dataset (required for inference).",
    )

    label_name: str = Field(
        ...,
        description="Name of the label field in the dataset (required for inference).",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="xgboost_model_inference.py",
        description="Entry point script for model inference.",
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'test').",
    )

    output_format: str = Field(
        default="csv",
        description="Output format for inference results (csv, parquet, json).",
    )

    json_orient: str = Field(
        default="records",
        description="JSON orientation when output_format is 'json' (records, index, values, split, table).",
    )

    # XGBoost specific fields
    xgboost_framework_version: str = Field(
        default="1.7-1", description="XGBoost framework version for processing"
    )

    # For inference jobs, we typically use smaller instances than evaluation
    use_large_processing_instance: bool = Field(
        default=False, description="Whether to use large instance type for processing (inference typically needs less resources)"
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # Currently no derived fields specific to model inference
    # beyond what's inherited from the ProcessingStepConfigBase class

    # Field validators

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format is supported."""
        valid_formats = {"csv", "parquet", "json"}
        if v.lower() not in valid_formats:
            raise ValueError(
                f"output_format must be one of {valid_formats}, got '{v}'"
            )
        return v.lower()

    @field_validator("json_orient")
    @classmethod
    def validate_json_orient(cls, v: str) -> str:
        """Validate JSON orientation is supported."""
        valid_orients = {"records", "index", "values", "split", "table"}
        if v.lower() not in valid_orients:
            raise ValueError(
                f"json_orient must be one of {valid_orients}, got '{v}'"
            )
        return v.lower()

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "XGBoostModelInferenceConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_inference_config(self) -> "XGBoostModelInferenceConfig":
        """Additional validation specific to inference configuration"""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("inference step requires a processing_entry_point")

        valid_job_types = {"training", "calibration", "validation", "testing"}
        if self.job_type not in valid_job_types:
            raise ValueError(
                f"job_type must be one of {valid_job_types}, got '{self.job_type}'"
            )

        # Validate required fields from script contract
        if not self.id_name:
            raise ValueError(
                "id_name must be provided (required by model inference contract)"
            )

        if not self.label_name:
            raise ValueError(
                "label_name must be provided (required by model inference contract)"
            )

        logger.debug(f"ID field '{self.id_name}' and label field '{self.label_name}' will be used for inference")

        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the model inference script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add model inference specific environment variables
        env_vars.update(
            {
                "ID_FIELD": self.id_name,
                "LABEL_FIELD": self.label_name,
                "JOB_TYPE": self.job_type,
                "OUTPUT_FORMAT": self.output_format,
                "JSON_ORIENT": self.json_orient,
            }
        )

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The model inference script contract
        """
        return XGBOOST_MODEL_INFERENCE_CONTRACT

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include inference-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and inference-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add model inference specific fields
        inference_fields = {
            # Tier 1 - Essential User Inputs
            "id_name": self.id_name,
            "label_name": self.label_name,
            # Tier 2 - System Inputs with Defaults
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "output_format": self.output_format,
            "json_orient": self.json_orient,
            "xgboost_framework_version": self.xgboost_framework_version,
            "use_large_processing_instance": self.use_large_processing_instance,
        }

        # Combine base fields and inference fields (inference fields take precedence if overlap)
        init_fields = {**base_fields, **inference_fields}

        return init_fields
