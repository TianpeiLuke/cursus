"""
PyTorch Model Evaluation Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for the PyTorch model evaluation step
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
from ..contracts.pytorch_model_eval_contract import PYTORCH_MODEL_EVAL_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class PyTorchModelEvalConfig(ProcessingStepConfigBase):
    """
    Configuration for PyTorch model evaluation step with self-contained derivation logic.

    This class defines the configuration parameters for the PyTorch model evaluation step,
    which calculates evaluation metrics for trained PyTorch models. This is crucial for
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
        default="pytorch_model_eval.py",
        description="Entry point script for PyTorch model evaluation.",
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'test').",
    )

    eval_metric_choices: List[str] = Field(
        default_factory=lambda: ["auroc", "average_precision", "f1_score"],
        description="List of evaluation metrics to compute",
    )

    # PyTorch specific fields
    framework_version: str = Field(
        default="2.1.2", description="PyTorch framework version for processing"
    )

    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    # For most processing jobs, we want to use a larger instance
    use_large_processing_instance: bool = Field(
        default=True, description="Whether to use large instance type for processing"
    )

    # Model comparison configuration (Tier 2 - Optional with defaults)
    comparison_mode: bool = Field(
        default=False,
        description="Enable model comparison functionality to compare with previous model scores",
    )

    previous_score_field: str = Field(
        default="",
        description="Name of the column containing previous model scores for comparison (required when comparison_mode=True)",
    )

    comparison_metrics: str = Field(
        default="all",
        description="Comparison metrics to compute: 'all' for comprehensive metrics, 'basic' for essential metrics only",
    )

    statistical_tests: bool = Field(
        default=True,
        description="Enable statistical significance tests (McNemar's test, paired t-test, Wilcoxon test)",
    )

    comparison_plots: bool = Field(
        default=True,
        description="Enable comparison visualizations (side-by-side ROC/PR curves, scatter plots, distributions)",
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # Currently no derived fields specific to model evaluation
    # beyond what's inherited from the ProcessingStepConfigBase class

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "PyTorchModelEvalConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for now

        return self

    @model_validator(mode="after")
    def validate_eval_config(self) -> "PyTorchModelEvalConfig":
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
                "id_name must be provided (required by PyTorch model evaluation contract)"
            )

        if not self.label_name:
            raise ValueError(
                "label_name must be provided (required by PyTorch model evaluation contract)"
            )

        # Validate comparison mode configuration
        if self.comparison_mode:
            if not self.previous_score_field or self.previous_score_field.strip() == "":
                raise ValueError(
                    "previous_score_field must be provided when comparison_mode is True"
                )

            # Validate comparison_metrics value
            valid_comparison_metrics = {"all", "basic"}
            if self.comparison_metrics not in valid_comparison_metrics:
                raise ValueError(
                    f"comparison_metrics must be one of {valid_comparison_metrics}, got '{self.comparison_metrics}'"
                )

            logger.info(
                f"Comparison mode enabled with previous score field: '{self.previous_score_field}'"
            )
        else:
            logger.debug(
                "Comparison mode disabled - standard evaluation will be performed"
            )

        logger.debug(
            f"ID field '{self.id_name}' and label field '{self.label_name}' will be used for evaluation"
        )

        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the PyTorch model evaluation script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add PyTorch model evaluation specific environment variables
        env_vars.update(
            {
                "ID_FIELD": self.id_name,
                "LABEL_FIELD": self.label_name,
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
            }
        )

        # Add eval metric choices
        if self.eval_metric_choices:
            env_vars["EVAL_METRIC_CHOICES"] = ",".join(self.eval_metric_choices)

        # Add comparison mode environment variables
        env_vars.update(
            {
                "COMPARISON_MODE": str(self.comparison_mode).lower(),
                "PREVIOUS_SCORE_FIELD": self.previous_score_field,
                "COMPARISON_METRICS": self.comparison_metrics,
                "STATISTICAL_TESTS": str(self.statistical_tests).lower(),
                "COMPARISON_PLOTS": str(self.comparison_plots).lower(),
            }
        )

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The PyTorch model evaluation script contract
        """
        return PYTORCH_MODEL_EVAL_CONTRACT

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

        # Add PyTorch model evaluation specific fields
        eval_fields = {
            # Tier 1 - Essential User Inputs
            "id_name": self.id_name,
            "label_name": self.label_name,
            # Tier 2 - System Inputs with Defaults
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "framework_version": self.framework_version,
            "py_version": self.py_version,
            "use_large_processing_instance": self.use_large_processing_instance,
            # Tier 2 - Comparison mode fields (only include if non-default)
            "comparison_mode": self.comparison_mode,
            "previous_score_field": self.previous_score_field,
            "comparison_metrics": self.comparison_metrics,
            "statistical_tests": self.statistical_tests,
            "comparison_plots": self.comparison_plots,
        }

        # Add eval_metric_choices if set to non-default value
        default_metrics = ["auroc", "average_precision", "f1_score"]
        if self.eval_metric_choices != default_metrics:
            eval_fields["eval_metric_choices"] = self.eval_metric_choices

        # Combine base fields and evaluation fields (evaluation fields take precedence if overlap)
        init_fields = {**base_fields, **eval_fields}

        return init_fields
