"""
Model Evaluation Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the XGBoost model evaluation step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, model_validator, PrivateAttr
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.xgboost_model_eval_contract import XGBOOST_MODEL_EVAL_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class XGBoostModelEvalConfig(ProcessingStepConfigBase):
    """
    Configuration for XGBoost model evaluation step with self-contained derivation logic.

    This class defines the configuration parameters for the XGBoost model evaluation step,
    which calculates evaluation metrics for trained models. This is crucial for
    measuring model performance and comparing different models or configurations.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    id_name: str = Field(
        ...,
        description="Name of the ID field in the dataset (required for evaluation).",
    )

    label_name: str = Field(
        ...,
        description="Name of the label field in the dataset (required for evaluation).",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="xgboost_model_eval.py",
        description="Entry point script for model evaluation.",
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'test').",
    )

    eval_metric_choices: List[str] = Field(
        default_factory=lambda: ["auc", "average_precision", "f1_score"],
        description="List of evaluation metrics to compute",
    )

    # XGBoost specific fields
    xgboost_framework_version: str = Field(
        default="1.5-1", description="XGBoost framework version for processing"
    )

    # For most processing jobs, we want to use a larger instance
    use_large_processing_instance: bool = Field(
        default=True, description="Whether to use large instance type for processing"
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # Currently no derived fields specific to model evaluation
    # beyond what's inherited from the ProcessingStepConfigBase class

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "XGBoostModelEvalConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_eval_config(self) -> "XGBoostModelEvalConfig":
        """Additional validation specific to evaluation configuration"""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("evaluation step requires a processing_entry_point")

        valid_job_types = {"training", "calibration", "validation", "testing"}
        if self.job_type not in valid_job_types:
            raise ValueError(
                f"job_type must be one of {valid_job_types}, got '{self.job_type}'"
            )

        # Validate required fields from script contract
        if not self.id_name:
            raise ValueError(
                "id_name must be provided (required by model evaluation contract)"
            )

        if not self.label_name:
            raise ValueError(
                "label_name must be provided (required by model evaluation contract)"
            )

        logger.debug(f"ID field '{self.id_name}' and label field '{self.label_name}' will be used for evaluation")

        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the model evaluation script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add model evaluation specific environment variables
        env_vars.update(
            {
                "ID_FIELD": self.id_name,
                "LABEL_FIELD": self.label_name,
                "JOB_TYPE": self.job_type,
            }
        )

        # Add eval metric choices
        if self.eval_metric_choices:
            env_vars["EVAL_METRIC_CHOICES"] = ",".join(self.eval_metric_choices)

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The model evaluation script contract
        """
        return XGBOOST_MODEL_EVAL_CONTRACT

    # Removed get_script_path override - now inherits modernized version from ProcessingStepConfigBase
    # which includes hybrid resolution and comprehensive fallbacks
    # The special case logic for returning only entry point name was deemed unnecessary
    # as the builder can extract the filename from the full path if needed

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include evaluation-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and evaluation-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add model evaluation specific fields
        eval_fields = {
            # Tier 1 - Essential User Inputs
            "id_name": self.id_name,
            "label_name": self.label_name,
            # Tier 2 - System Inputs with Defaults
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "xgboost_framework_version": self.xgboost_framework_version,
            "use_large_processing_instance": self.use_large_processing_instance,
        }

        # Add eval_metric_choices if set to non-default value
        default_metrics = ["auc", "average_precision", "f1_score"]
        if self.eval_metric_choices != default_metrics:
            eval_fields["eval_metric_choices"] = self.eval_metric_choices

        # Combine base fields and evaluation fields (evaluation fields take precedence if overlap)
        init_fields = {**base_fields, **eval_fields}

        return init_fields
