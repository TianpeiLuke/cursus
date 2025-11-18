"""
Active Sample Selection Step Configuration

This module implements the configuration class for the Active Sample Selection step
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import Field, field_validator, model_validator
from typing import Literal, Dict, Any, Optional, TYPE_CHECKING
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.active_sample_selection_contract import (
    ACTIVE_SAMPLE_SELECTION_CONTRACT,
)

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class ActiveSampleSelectionConfig(ProcessingStepConfigBase):
    """
    Configuration for Active Sample Selection step.

    Supports both Semi-Supervised Learning (SSL) and Active Learning workflows
    with Pydantic validation to prevent strategy misuse.

    Three-Tier Configuration:
    - Tier 1: Essential User Inputs (none - all have defaults)
    - Tier 2: System Fields with Defaults (all selection parameters)
    - Tier 3: Derived Fields (inherited from ProcessingStepConfigBase)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    use_case: Literal["ssl", "active_learning", "auto"] = Field(
        ...,
        description=(
            "Use case for validation: ssl, active_learning, or auto (required). "
            "When auto, no validation. When specified, validates strategy compatibility."
        ),
    )

    id_field: str = Field(
        ...,
        description=(
            "ID column name in predictions data (required). "
            "Essential for tracking samples in iterative SSL/AL workflows and merging with training data."
        ),
    )

    label_field: str = Field(
        ..., description="Label column name in predictions data (required)"
    )

    score_field: Optional[str] = Field(
        ...,
        description=(
            "Score column name for single score column (required). "
            "For multiclass with prob_class_* columns, set to empty string '' or None. "
            "For binary classification with prob_class_1, set to 'prob_class_1'. "
            "For custom score column like 'confidence_score', set to 'confidence_score'."
        ),
    )

    # ===== System Fields with Defaults (Tier 2) =====

    # Core selection parameters
    selection_strategy: Literal[
        "confidence_threshold", "top_k_per_class", "uncertainty", "diversity", "badge"
    ] = Field(
        default="confidence_threshold",
        description=(
            "Selection strategy. "
            "SSL: confidence_threshold, top_k_per_class. "
            "Active Learning: uncertainty, diversity, badge."
        ),
    )

    # Data field configuration
    output_format: Literal["csv", "parquet"] = Field(
        default="csv", description="Output format for selected samples"
    )

    # SSL-specific parameters
    confidence_threshold: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="For SSL: minimum confidence threshold (0.5-1.0)",
    )

    k_per_class: int = Field(
        default=100, ge=1, description="For SSL: top-k samples per class"
    )

    max_samples: int = Field(
        default=0, ge=0, description="For SSL: max samples to select (0=no limit)"
    )

    # Active Learning-specific parameters
    uncertainty_mode: Literal["margin", "entropy", "least_confidence"] = Field(
        default="margin", description="For Active Learning: uncertainty sampling mode"
    )

    batch_size: int = Field(
        default=32, ge=1, description="For Active Learning: number of samples to select"
    )

    metric: Literal["euclidean", "cosine"] = Field(
        default="euclidean",
        description="For Active Learning diversity/BADGE: distance metric",
    )

    random_seed: int = Field(
        default=42, ge=0, description="Random seed for reproducibility"
    )

    # Score field prefix for multiclass
    score_field_prefix: str = Field(
        default="prob_class_",
        description="Prefix for multiple score columns in multiclass",
    )

    # Processing configuration
    processing_entry_point: str = Field(
        default="active_sample_selection.py",
        description="Entry point script for active sample selection",
    )

    processing_framework_version: str = Field(
        default="1.2-1", description="SKLearn framework version for processing"
    )

    job_type: str = Field(
        default="ssl_selection",
        description="Type of selection job (e.g., 'ssl_selection', 'active_learning_selection')",
    )

    # For active sampling, typically use smaller instances
    use_large_processing_instance: bool = Field(
        default=False, description="Whether to use large instance type for processing"
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Validators =====

    @field_validator("selection_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate selection strategy is one of the allowed values."""
        allowed = {
            "confidence_threshold",
            "top_k_per_class",
            "uncertainty",
            "diversity",
            "badge",
        }
        if v not in allowed:
            raise ValueError(f"selection_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("use_case")
    @classmethod
    def validate_use_case(cls, v: str) -> str:
        """Validate use case is one of the allowed values."""
        allowed = {"ssl", "active_learning", "auto"}
        if v not in allowed:
            raise ValueError(f"use_case must be one of {allowed}, got '{v}'")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        allowed = {"csv", "parquet"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_strategy_use_case_compatibility(self) -> "ActiveSampleSelectionConfig":
        """
        ⚠️ CRITICAL: Validate strategy is compatible with use case.

        This cross-field validation prevents:
        - Using uncertainty strategies for SSL (creates noisy pseudo-labels)
        - Using confidence strategies for Active Learning (wastes human effort)

        Raises:
            ValueError: If strategy is incompatible with use_case
        """
        # Define strategy categories
        SSL_STRATEGIES = {"confidence_threshold", "top_k_per_class"}
        ACTIVE_LEARNING_STRATEGIES = {"uncertainty", "diversity", "badge"}

        # Skip validation if use_case is "auto"
        if self.use_case == "auto":
            logger.debug(
                f"use_case='auto', skipping validation for strategy '{self.selection_strategy}'"
            )
            return self

        # Validate SSL use case
        if self.use_case == "ssl":
            if self.selection_strategy not in SSL_STRATEGIES:
                raise ValueError(
                    f"❌ Strategy '{self.selection_strategy}' is NOT valid for SSL! "
                    f"SSL requires confidence-based strategies: {SSL_STRATEGIES}. "
                    f"Strategy '{self.selection_strategy}' selects UNCERTAIN samples, "
                    f"which would create noisy pseudo-labels and degrade model performance. "
                    f"Use 'confidence_threshold' or 'top_k_per_class' instead."
                )
            logger.info(
                f"✓ Strategy '{self.selection_strategy}' validated for SSL use case"
            )

        # Validate Active Learning use case
        elif self.use_case == "active_learning":
            if self.selection_strategy not in ACTIVE_LEARNING_STRATEGIES:
                raise ValueError(
                    f"⚠️ Strategy '{self.selection_strategy}' is NOT recommended for Active Learning! "
                    f"Active Learning typically uses: {ACTIVE_LEARNING_STRATEGIES}. "
                    f"Strategy '{self.selection_strategy}' selects CONFIDENT samples, "
                    f"which wastes human labeling effort on easy samples. "
                    f"Use 'uncertainty', 'diversity', or 'badge' instead."
                )
            logger.info(
                f"✓ Strategy '{self.selection_strategy}' validated for Active Learning use case"
            )

        return self

    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "ActiveSampleSelectionConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Add any active-sampling-specific initialization here

        return self

    # ===== Helper Methods =====

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the processing job.

        Returns:
            Dictionary of environment variables
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add core environment variables
        env_vars.update(
            {
                "SELECTION_STRATEGY": self.selection_strategy,
                "USE_CASE": self.use_case,
                "ID_FIELD": self.id_field,
                "LABEL_FIELD": self.label_field,
                "OUTPUT_FORMAT": self.output_format,
                "RANDOM_SEED": str(self.random_seed),
            }
        )

        # Add score field configuration
        if self.score_field:
            env_vars["SCORE_FIELD"] = self.score_field
        env_vars["SCORE_FIELD_PREFIX"] = self.score_field_prefix

        # Add SSL-specific variables
        if self.selection_strategy in {"confidence_threshold", "top_k_per_class"}:
            env_vars.update(
                {
                    "CONFIDENCE_THRESHOLD": str(self.confidence_threshold),
                    "MAX_SAMPLES": str(self.max_samples),
                    "K_PER_CLASS": str(self.k_per_class),
                }
            )

        # Add Active Learning-specific variables
        if self.selection_strategy in {"uncertainty", "diversity", "badge"}:
            env_vars.update(
                {
                    "UNCERTAINTY_MODE": self.uncertainty_mode,
                    "BATCH_SIZE": str(self.batch_size),
                    "METRIC": self.metric,
                }
            )

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The active sample selection script contract
        """
        return ACTIVE_SAMPLE_SELECTION_CONTRACT

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include selection-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add active sample selection specific fields
        selection_fields = {
            # Core selection parameters
            "selection_strategy": self.selection_strategy,
            "use_case": self.use_case,
            "id_field": self.id_field,
            "label_field": self.label_field,
            "output_format": self.output_format,
            "random_seed": self.random_seed,
            # SSL-specific parameters
            "confidence_threshold": self.confidence_threshold,
            "k_per_class": self.k_per_class,
            "max_samples": self.max_samples,
            # Active Learning-specific parameters
            "uncertainty_mode": self.uncertainty_mode,
            "batch_size": self.batch_size,
            "metric": self.metric,
            # Score field configuration
            "score_field": self.score_field,
            "score_field_prefix": self.score_field_prefix,
            # Processing configuration
            "processing_entry_point": self.processing_entry_point,
            "processing_framework_version": self.processing_framework_version,
            "job_type": self.job_type,
            "use_large_processing_instance": self.use_large_processing_instance,
        }

        # Combine base fields and selection fields (selection fields take precedence if overlap)
        init_fields = {**base_fields, **selection_fields}

        return init_fields
