"""
Pseudo Label Merge Step Configuration

This module implements the configuration class for the Pseudo Label Merge step
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
from ..contracts.pseudo_label_merge_contract import PSEUDO_LABEL_MERGE_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class PseudoLabelMergeConfig(ProcessingStepConfigBase):
    """
    Configuration for Pseudo Label Merge step.

    Intelligently merges labeled base data with pseudo-labeled or augmented samples
    for Semi-Supervised Learning (SSL) and Active Learning workflows.

    Three-Tier Configuration:
    - Tier 1: Essential User Inputs (label_field)
    - Tier 2: System Fields with Defaults (merge parameters, split ratios, etc.)
    - Tier 3: Derived Fields (inherited from ProcessingStepConfigBase)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    label_field: str = Field(
        ...,
        description=(
            "Label column name in both base and augmentation data (required). "
            "This field must exist in both datasets for successful merge. "
            "The augmentation data's pseudo_label column will be converted to this field name."
        ),
    )

    id_field: str = Field(
        ...,
        description=(
            "ID column name for schema validation and tracking (required). "
            "Used to ensure consistency across merge operations. "
            "Should be unique identifier for each sample. "
            "Common names: 'id', 'sample_id', 'record_id', 'customer_id'. "
            "Essential for tracking samples in iterative SSL/AL workflows."
        ),
    )

    pseudo_label_column: str = Field(
        ...,
        description=(
            "Column name for pseudo-labels in augmentation data (required). "
            "This column will be converted to label_field during merge. "
            "Common names: 'pseudo_label', 'prediction', 'predicted_label', 'model_prediction'. "
            "Must match the actual column name in your augmentation data."
        ),
    )

    # ===== System Fields with Defaults (Tier 2) =====

    # Core merge parameters
    add_provenance: bool = Field(
        default=True,
        description=(
            "Add data_source column to track sample origin. "
            "Values: 'original' (base data) or 'pseudo_labeled' (augmentation data). "
            "Recommended for SSL/AL workflows to track pseudo-label quality."
        ),
    )

    output_format: Literal["csv", "tsv", "parquet"] = Field(
        default="csv",
        description=(
            "Output format for merged data. "
            "CSV: default, widely compatible. "
            "TSV: tab-separated for large text fields. "
            "Parquet: recommended for large datasets (better compression/performance)."
        ),
    )

    # Split ratio configuration
    use_auto_split_ratios: bool = Field(
        default=True,
        description=(
            "Auto-infer split ratios from base data proportions (RECOMMENDED). "
            "When True, calculates actual base data ratios (e.g., 10K/2K/2K → 71.4%/14.3%/14.3%) "
            "and applies same distribution to augmentation data. "
            "When False, uses manual train_ratio and test_val_ratio. "
            "Auto-inference ensures augmentation follows base data characteristics."
        ),
    )

    train_ratio: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=0.9,
        description=(
            "Manual train split ratio (0.1-0.9). Only used when use_auto_split_ratios=False. "
            "Example: 0.7 means 70% train, 30% for test+val. "
            "Recommended: Use auto-inference (default) instead of manual ratios."
        ),
    )

    test_val_ratio: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=0.9,
        description=(
            "Test vs val ratio of holdout set (0.1-0.9). Only used when use_auto_split_ratios=False. "
            "Example: 0.5 means equal test/val split from holdout. "
            "Recommended: Use auto-inference (default) instead of manual ratios."
        ),
    )

    # Data handling parameters
    preserve_confidence: bool = Field(
        default=True,
        description=(
            "Keep confidence/probability scores from augmentation data. "
            "When True, preserves columns like 'confidence', 'score', 'prob_*'. "
            "When False, removes these columns to reduce dataset size. "
            "Recommended True for SSL quality analysis and debugging."
        ),
    )

    # Split behavior parameters
    stratify: bool = Field(
        default=True,
        description=(
            "Use stratified splits to maintain class balance. "
            "When True, ensures augmentation distribution matches label proportions. "
            "Recommended True for imbalanced datasets. "
            "Set False for regression or when class balance not critical."
        ),
    )

    random_seed: int = Field(
        default=42,
        ge=0,
        description=(
            "Random seed for reproducibility in split operations. "
            "Ensures consistent augmentation distribution across runs. "
            "Critical for experiment reproducibility in SSL/AL workflows."
        ),
    )

    # Processing configuration
    processing_entry_point: str = Field(
        default="pseudo_label_merge.py",
        description="Entry point script for pseudo label merge",
    )

    processing_framework_version: str = Field(
        default="1.2-1", description="SKLearn framework version for processing"
    )

    job_type: str = Field(
        default="training",
        description=(
            "Type of merge job. "
            "training: Uses split-aware merge with train/test/val distribution. "
            "validation/testing/calibration: Uses simple concatenation merge."
        ),
    )

    # For merge operations, typically use smaller instances
    use_large_processing_instance: bool = Field(
        default=False,
        description="Whether to use large instance type for processing. False recommended for most merge operations.",
    )

    model_config = ProcessingStepConfigBase.model_config

    # ===== Validators =====

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format."""
        allowed = {"csv", "tsv", "parquet"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got '{v}'")
        return v

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate job type is one of the allowed values."""
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("label_field", "id_field", "pseudo_label_column")
    @classmethod
    def validate_field_names(cls, v: str) -> str:
        """Validate field names are non-empty and don't contain special characters."""
        if not v or not v.strip():
            raise ValueError("Field name cannot be empty")

        # Check for problematic characters
        if any(char in v for char in [" ", "\t", "\n", "\r", ",", ";"]):
            raise ValueError(
                f"Field name '{v}' contains invalid characters. "
                f"Avoid spaces, tabs, newlines, commas, and semicolons."
            )

        return v.strip()

    @model_validator(mode="after")
    def validate_manual_ratios(self) -> "PseudoLabelMergeConfig":
        """
        Validate manual split ratios when auto-inference is disabled.

        Ensures train_ratio and test_val_ratio are provided when needed.
        """
        if not self.use_auto_split_ratios:
            # For training jobs with manual ratios, both must be provided
            if self.job_type == "training":
                if self.train_ratio is None:
                    raise ValueError(
                        "train_ratio is required when use_auto_split_ratios=False "
                        "and job_type='training'. Either enable auto-inference "
                        "(recommended) or provide train_ratio."
                    )

                if self.test_val_ratio is None:
                    raise ValueError(
                        "test_val_ratio is required when use_auto_split_ratios=False "
                        "and job_type='training'. Either enable auto-inference "
                        "(recommended) or provide test_val_ratio."
                    )

                # Validate ratio values
                if not (0.1 <= self.train_ratio <= 0.9):
                    raise ValueError(
                        f"train_ratio must be between 0.1 and 0.9, got {self.train_ratio}"
                    )

                if not (0.1 <= self.test_val_ratio <= 0.9):
                    raise ValueError(
                        f"test_val_ratio must be between 0.1 and 0.9, got {self.test_val_ratio}"
                    )

                logger.info(
                    f"Using manual split ratios: train={self.train_ratio}, "
                    f"test_val={self.test_val_ratio}"
                )
            else:
                # Non-training jobs don't use split ratios
                if self.train_ratio is not None or self.test_val_ratio is not None:
                    logger.warning(
                        f"train_ratio and test_val_ratio are ignored for "
                        f"job_type='{self.job_type}' (non-training jobs use simple merge)"
                    )
        else:
            # Auto-inference enabled
            if self.train_ratio is not None or self.test_val_ratio is not None:
                logger.warning(
                    "train_ratio and test_val_ratio are ignored when "
                    "use_auto_split_ratios=True (auto-inference is enabled)"
                )

        return self

    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "PseudoLabelMergeConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Add any pseudo-label-merge-specific initialization here
        # (Currently none needed)

        return self

    # ===== Helper Methods =====

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the processing job.

        Returns:
            Dictionary of environment variables matching script contract
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add core required environment variables
        env_vars.update(
            {
                "LABEL_FIELD": self.label_field,
                "ID_FIELD": self.id_field,
                "PSEUDO_LABEL_COLUMN": self.pseudo_label_column,
            }
        )

        # Add optional environment variables with string conversion
        env_vars.update(
            {
                "ADD_PROVENANCE": str(self.add_provenance).lower(),
                "OUTPUT_FORMAT": self.output_format,
                "USE_AUTO_SPLIT_RATIOS": str(self.use_auto_split_ratios).lower(),
                "PRESERVE_CONFIDENCE": str(self.preserve_confidence).lower(),
                "STRATIFY": str(self.stratify).lower(),
                "RANDOM_SEED": str(self.random_seed),
            }
        )

        # Add manual split ratios if provided and auto-inference disabled
        if not self.use_auto_split_ratios:
            if self.train_ratio is not None:
                env_vars["TRAIN_RATIO"] = str(self.train_ratio)
            if self.test_val_ratio is not None:
                env_vars["TEST_VAL_RATIO"] = str(self.test_val_ratio)

        return env_vars

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The pseudo label merge script contract
        """
        return PSEUDO_LABEL_MERGE_CONTRACT

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include merge-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add pseudo label merge specific fields
        merge_fields = {
            # Core merge parameters
            "label_field": self.label_field,
            "add_provenance": self.add_provenance,
            "output_format": self.output_format,
            # Split ratio configuration
            "use_auto_split_ratios": self.use_auto_split_ratios,
            "train_ratio": self.train_ratio,
            "test_val_ratio": self.test_val_ratio,
            # Schema alignment parameters
            "pseudo_label_column": self.pseudo_label_column,
            "id_field": self.id_field,
            "preserve_confidence": self.preserve_confidence,
            # Split behavior parameters
            "stratify": self.stratify,
            "random_seed": self.random_seed,
            # Processing configuration
            "processing_entry_point": self.processing_entry_point,
            "processing_framework_version": self.processing_framework_version,
            "job_type": self.job_type,
            "use_large_processing_instance": self.use_large_processing_instance,
        }

        # Combine base fields and merge fields (merge fields take precedence if overlap)
        init_fields = {**base_fields, **merge_fields}

        return init_fields
