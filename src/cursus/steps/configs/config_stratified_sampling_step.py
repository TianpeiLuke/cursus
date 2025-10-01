"""
Stratified Sampling Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for stratified sampling, using a self-contained design where each field
is properly categorized according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import contract
from ..contracts.stratified_sampling_contract import STRATIFIED_SAMPLING_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class StratifiedSamplingConfig(ProcessingStepConfigBase):
    """
    Configuration for the Stratified Sampling step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    strata_column: str = Field(
        description="Column name to stratify by (e.g., target variable, confounding variable)."
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="stratified_sampling.py",
        description="Relative path (within processing_source_dir) to the stratified sampling script.",
    )

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    sampling_strategy: str = Field(
        default="balanced",
        description="Sampling strategy: 'balanced' (class imbalance), 'proportional_min' (causal analysis), 'optimal' (variance optimization)",
    )

    target_sample_size: int = Field(
        default=1000,
        ge=1,
        description="Total desired sample size per split",
    )

    min_samples_per_stratum: int = Field(
        default=10,
        ge=1,
        description="Minimum samples per stratum for statistical power",
    )

    variance_column: Optional[str] = Field(
        default=None,
        description="Column for variance calculation (needed for optimal strategy)",
    )

    random_state: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====
    # (No derived properties needed - base class handles script path)

    # ===== Validators =====

    @field_validator("strata_column")
    @classmethod
    def validate_strata_column(cls, v: str) -> str:
        """
        Ensure strata_column is a non-empty string.
        """
        if not v or not v.strip():
            raise ValueError("strata_column must be a non-empty string")
        return v

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """
        Ensure processing_entry_point is a non‐empty relative path.
        """
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non‐empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError(
                "processing_entry_point must be a relative path within source directory"
            )
        return v

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """
        Ensure job_type is one of the allowed values.
        """
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("sampling_strategy")
    @classmethod
    def validate_sampling_strategy(cls, v: str) -> str:
        """
        Ensure sampling_strategy is one of the allowed values.
        """
        allowed = {"balanced", "proportional_min", "optimal"}
        if v not in allowed:
            raise ValueError(f"sampling_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("variance_column")
    @classmethod
    def validate_variance_column(cls, v: Optional[str]) -> Optional[str]:
        """
        Ensure variance_column is a non-empty string if provided.
        """
        if v is not None and (not v or not v.strip()):
            raise ValueError("variance_column must be a non-empty string if provided")
        return v

    # Cross-field validation
    @model_validator(mode="after")
    def validate_strategy_requirements(self) -> "StratifiedSamplingConfig":
        """
        Validate that required fields are provided for specific strategies.
        """
        if self.sampling_strategy == "optimal" and self.variance_column is None:
            logger.warning(
                "optimal sampling strategy works best with variance_column specified. "
                "Using default variance if variance_column is not provided."
            )
        return self

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "StratifiedSamplingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The stratified sampling script contract
        """
        return STRATIFIED_SAMPLING_CONTRACT

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include stratified sampling specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add stratified sampling specific fields
        sampling_fields = {
            "strata_column": self.strata_column,
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "sampling_strategy": self.sampling_strategy,
            "target_sample_size": self.target_sample_size,
            "min_samples_per_stratum": self.min_samples_per_stratum,
            "random_state": self.random_state,
        }

        # Only include variance_column if it's set
        if self.variance_column is not None:
            sampling_fields["variance_column"] = self.variance_column

        # Combine fields (sampling fields take precedence if overlap)
        init_fields = {**base_fields, **sampling_fields}

        return init_fields
