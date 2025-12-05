"""
Multi-Task Model Inference Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the LightGBMMT multi-task model inference step
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
from ..contracts.lightgbmmt_model_inference_contract import (
    LIGHTGBMMT_MODEL_INFERENCE_CONTRACT,
)

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class LightGBMMTModelInferenceConfig(ProcessingStepConfigBase):
    """
    Configuration for LightGBMMT multi-task model inference step with self-contained derivation logic.

    This class defines the configuration parameters for the LightGBMMT multi-task model inference step,
    which generates per-task predictions from trained multi-task models without computing evaluation metrics.
    This is designed for pure inference workflows where predictions are needed for downstream processing
    (e.g., model calibration, batch scoring, unlabeled data scoring).

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

    task_label_names: List[str] = Field(
        ...,
        description="List of task names for multi-task inference. Must contain at least 2 tasks. Used to name prediction columns (task_name_prob). Corresponding label columns in input data are OPTIONAL for inference.",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="lightgbmmt_model_inference.py",
        description="Entry point script for multi-task model inference.",
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to perform inference on (e.g., 'training', 'calibration', 'validation', 'test').",
    )

    output_format: str = Field(
        default="csv",
        description="Output format for inference results (csv, tsv, parquet, json).",
    )

    json_orient: str = Field(
        default="records",
        description="JSON orientation when output_format is 'json' (records, index, values, split).",
    )

    # LightGBM specific fields
    framework_version: str = Field(
        default="2.1.2",
        description="PyTorch framework version for processing (LightGBM installed via pip)",
    )

    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    # For inference jobs, we typically use smaller instances than evaluation
    use_large_processing_instance: bool = Field(
        default=False,
        description="Whether to use large instance type for processing (inference typically needs less resources)",
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
        valid_formats = {"csv", "tsv", "parquet", "json"}
        if v.lower() not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}, got '{v}'")
        return v.lower()

    @field_validator("json_orient")
    @classmethod
    def validate_json_orient(cls, v: str) -> str:
        """Validate JSON orientation is supported."""
        valid_orients = {"records", "index", "values", "split"}
        if v.lower() not in valid_orients:
            raise ValueError(f"json_orient must be one of {valid_orients}, got '{v}'")
        return v.lower()

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "LightGBMMTModelInferenceConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_inference_config(self) -> "LightGBMMTModelInferenceConfig":
        """Additional validation specific to multi-task inference configuration"""
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
                "id_name must be provided (required by multi-task model inference contract)"
            )

        if not self.task_label_names or len(self.task_label_names) == 0:
            raise ValueError(
                "task_label_names must be a non-empty list (required by multi-task model inference contract)"
            )

        # Validate minimum number of tasks
        if len(self.task_label_names) < 2:
            raise ValueError(
                f"task_label_names must contain at least 2 tasks for multi-task inference, got {len(self.task_label_names)}"
            )

        # Validate no duplicate task names
        if len(self.task_label_names) != len(set(self.task_label_names)):
            duplicates = [
                name
                for name in self.task_label_names
                if self.task_label_names.count(name) > 1
            ]
            raise ValueError(
                f"task_label_names contains duplicate task names: {set(duplicates)}"
            )

        logger.debug(
            f"ID field '{self.id_name}' and {len(self.task_label_names)} task names "
            f"{self.task_label_names} will be used for multi-task inference"
        )

        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the multi-task model inference script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add USE_SECURE_PYPI (inherited from base config)
        env_vars["USE_SECURE_PYPI"] = str(self.use_secure_pypi).lower()

        # Add multi-task model inference specific environment variables
        env_vars.update(
            {
                "ID_FIELD": self.id_name,
                "TASK_LABEL_NAMES": ",".join(
                    self.task_label_names
                ),  # Comma-separated list
                "OUTPUT_FORMAT": self.output_format,
                "JSON_ORIENT": self.json_orient,
            }
        )

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The multi-task model inference script contract
        """
        return LIGHTGBMMT_MODEL_INFERENCE_CONTRACT

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include multi-task inference-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and inference-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add multi-task model inference specific fields
        inference_fields = {
            # Tier 1 - Essential User Inputs
            "id_name": self.id_name,
            "task_label_names": self.task_label_names,
            # Tier 2 - System Inputs with Defaults
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "output_format": self.output_format,
            "json_orient": self.json_orient,
            "framework_version": self.framework_version,
            "py_version": self.py_version,
            "use_large_processing_instance": self.use_large_processing_instance,
        }

        # Combine base fields and inference fields (inference fields take precedence if overlap)
        init_fields = {**base_fields, **inference_fields}

        return init_fields
