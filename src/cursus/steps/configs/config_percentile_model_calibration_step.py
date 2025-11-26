"""
Percentile Model Calibration Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the PercentileModelCalibration step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from typing import Optional, List, Union, Any, Dict
import json
from pathlib import Path
from pydantic import Field, model_validator, PrivateAttr

from .config_processing_step_base import ProcessingStepConfigBase
from ...core.base.hyperparameters_base import ModelHyperparameters


class PercentileModelCalibrationConfig(ProcessingStepConfigBase):
    """
    Configuration for PercentileModelCalibration step with self-contained derivation logic.

    This class defines the configuration parameters for the PercentileModelCalibration step,
    which creates percentile mapping from model scores using ROC curve analysis for consistent
    risk interpretation. The step converts raw model scores to percentile values that represent
    the relative risk ranking of predictions.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    # At least one of score_field or score_fields must be provided

    score_field: Optional[str] = Field(
        default=None,
        description="Name of the score column to calibrate (single-task mode). "
        "Use this for backward compatibility or when calibrating a single score field.",
    )

    score_fields: Optional[List[str]] = Field(
        default=None,
        description="List of score column names to calibrate (multi-task mode). "
        "Use this when calibrating multiple score fields independently. "
        "Example: ['task_0_prob', 'task_1_prob', 'task_2_prob']. "
        "If both score_field and score_fields are provided, score_fields takes precedence.",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Calibration parameters with defaults
    n_bins: int = Field(
        default=1000,
        gt=0,
        description="Number of bins for ROC curve analysis (default: 1000)",
    )

    accuracy: float = Field(
        default=0.001,
        gt=0,
        le=1,
        description="Accuracy threshold for percentile mapping (default: 0.001)",
    )

    # Job type parameter for variant handling
    job_type: str = Field(
        default="calibration",
        description="Which data split to use for calibration (e.g., 'training', 'calibration', 'validation', 'test').",
    )

    # Processing parameters - set defaults specific to percentile calibration
    processing_entry_point: str = Field(
        default="percentile_model_calibration.py",
        description="Script entry point filename",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # For now, there are no derived fields specific to percentile model calibration beyond
    # what's inherited from the ProcessingStepConfigBase class

    model_config = ProcessingStepConfigBase.model_config

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "PercentileModelCalibrationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_config(self) -> "PercentileModelCalibrationConfig":
        """Validate configuration and ensure defaults are set.

        Returns:
            Self: The validated configuration object

        Raises:
            ValueError: If any validation fails
        """
        # Basic validation - inherited from base class

        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        # Validate input/output paths in contract
        required_input_paths = ["evaluation_data"]
        for path_name in required_input_paths:
            if path_name not in contract.expected_input_paths:
                raise ValueError(
                    f"Script contract missing required input path: {path_name}"
                )

        required_output_paths = [
            "calibration_output",
            "metrics_output",
            "calibrated_data",
        ]
        for path_name in required_output_paths:
            if path_name not in contract.expected_output_paths:
                raise ValueError(
                    f"Script contract missing required output path: {path_name}"
                )

        # Validate job_type
        valid_job_types = {"training", "calibration", "validation", "testing"}
        if self.job_type not in valid_job_types:
            raise ValueError(
                f"job_type must be one of {valid_job_types}, got '{self.job_type}'"
            )

        # Validate n_bins range
        if self.n_bins > 10000:
            raise ValueError(
                f"n_bins ({self.n_bins}) should not exceed 10000 for performance reasons"
            )

        # Validate accuracy range
        if self.accuracy >= 1.0:
            raise ValueError(f"accuracy ({self.accuracy}) must be less than 1.0")

        # Validate that at least one of score_field or score_fields is provided
        if not self.score_field and not self.score_fields:
            raise ValueError(
                "At least one of 'score_field' (single-task) or 'score_fields' (multi-task) must be provided"
            )

        # Validate score_fields if provided
        if self.score_fields:
            if not isinstance(self.score_fields, list):
                raise ValueError("score_fields must be a list of strings")
            if len(self.score_fields) == 0:
                raise ValueError("score_fields cannot be an empty list")
            if not all(isinstance(field, str) for field in self.score_fields):
                raise ValueError("All elements in score_fields must be strings")

        return self

    def get_script_contract(self):
        """Return the script contract for this step.

        Returns:
            ScriptContract: The contract for this step's script.
        """
        from ..contracts.percentile_model_calibration_contract import (
            PERCENTILE_MODEL_CALIBRATION_CONTRACT,
        )

        return PERCENTILE_MODEL_CALIBRATION_CONTRACT

    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processing script.

        Returns:
            dict: Dictionary of environment variables to be passed to the processing script.
        """
        env = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add percentile calibration-specific environment variables
        env_updates = {
            "N_BINS": str(self.n_bins),
            "ACCURACY": str(self.accuracy),
        }

        # Handle score field(s) - priority to score_fields (multi-task)
        if self.score_fields:
            # Multi-task mode: pass as comma-separated string
            env_updates["SCORE_FIELDS"] = ",".join(self.score_fields)
            # Also pass empty SCORE_FIELD to avoid confusion
            env_updates["SCORE_FIELD"] = ""
        elif self.score_field:
            # Single-task mode: pass score_field
            env_updates["SCORE_FIELD"] = self.score_field
            env_updates["SCORE_FIELDS"] = ""

        env.update(env_updates)
        return env

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include percentile calibration-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and percentile calibration-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add percentile calibration-specific fields
        calibration_fields = {
            # Tier 1 - Essential User Inputs
            "score_field": self.score_field,
            "score_fields": self.score_fields,
            # Tier 2 - System Inputs with Defaults
            "n_bins": self.n_bins,
            "accuracy": self.accuracy,
            "job_type": self.job_type,
        }

        # Combine base fields and calibration fields (calibration fields take precedence if overlap)
        init_fields = {**base_fields, **calibration_fields}

        return init_fields
