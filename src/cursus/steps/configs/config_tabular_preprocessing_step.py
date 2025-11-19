"""
Tabular Preprocessing Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for tabular data preprocessing, using a self-contained design where each field
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
from ..contracts.tabular_preprocessing_contract import TABULAR_PREPROCESSING_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class TabularPreprocessingConfig(ProcessingStepConfigBase):
    """
    Configuration for the Tabular Preprocessing step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    # Tabular preprocessing use job_type to determine if it need label_name as required input;
    # so job_type is essential input here.

    job_type: str = Field(
        description="One of ['training','validation','testing','calibration']",
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    label_name: Optional[str] = Field(
        default=None,
        description="Label field name for the target variable. Optional for calibration job types.",
    )

    processing_entry_point: str = Field(
        default="tabular_preprocessing.py",
        description="Relative path (within processing_source_dir) to the tabular preprocessing script.",
    )

    train_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of data to allocate to the training set (only used if job_type=='training').",
    )

    test_val_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of the holdout to allocate to the test set vs. validation (only if job_type=='training').",
    )

    output_format: str = Field(
        default="CSV",
        description="Output format for processed data ('CSV', 'TSV', or 'Parquet'). Default: CSV",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    _full_script_path: Optional[str] = PrivateAttr(default=None)
    _preprocessing_environment_variables: Optional[Dict[str, str]] = PrivateAttr(
        default=None
    )

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====

    @property
    def full_script_path(self) -> Optional[str]:
        """
        Get full path to the preprocessing script.

        Returns:
            Full path to the script
        """
        if self._full_script_path is None:
            # Get effective source directory
            source_dir = self.effective_source_dir
            if source_dir is None:
                return None

            # Combine with entry point
            if source_dir.startswith("s3://"):
                self._full_script_path = (
                    f"{source_dir.rstrip('/')}/{self.processing_entry_point}"
                )
            else:
                self._full_script_path = str(
                    Path(source_dir) / self.processing_entry_point
                )

        return self._full_script_path

    @property
    def preprocessing_environment_variables(self) -> Dict[str, str]:
        """
        Get preprocessing-specific environment variables.

        Returns:
            Dictionary mapping environment variable names to values
        """
        if self._preprocessing_environment_variables is None:
            env_vars = {}

            # Add label field
            if self.label_name:
                env_vars["LABEL_FIELD"] = self.label_name

            # Add split ratios
            env_vars["TRAIN_RATIO"] = str(self.train_ratio)
            env_vars["TEST_VAL_RATIO"] = str(self.test_val_ratio)

            # Add output format
            env_vars["OUTPUT_FORMAT"] = self.output_format

            self._preprocessing_environment_variables = env_vars

        return self._preprocessing_environment_variables

    # ===== Validators =====

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
    def validate_data_type(cls, v: str) -> str:
        """
        Ensure job_type is one of the allowed values.
        """
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("train_ratio", "test_val_ratio")
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        """
        Ensure the ratio is strictly between 0 and 1 (not including 0 or 1).
        """
        if not (0.0 < v < 1.0):
            raise ValueError(f"Split ratio must be strictly between 0 and 1, got {v}")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """
        Ensure output_format is one of the allowed values (case-insensitive).
        """
        allowed = {"CSV", "TSV", "Parquet"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}, got '{v}'")
        return v

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "TabularPreprocessingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize full script path if possible
        source_dir = self.effective_source_dir
        if source_dir is not None:
            if source_dir.startswith("s3://"):
                self._full_script_path = (
                    f"{source_dir.rstrip('/')}/{self.processing_entry_point}"
                )
            else:
                self._full_script_path = str(
                    Path(source_dir) / self.processing_entry_point
                )

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The tabular preprocessing script contract
        """
        return TABULAR_PREPROCESSING_CONTRACT

    # Removed get_script_path override - now inherits modernized version from ProcessingStepConfigBase
    # which includes hybrid resolution and comprehensive fallbacks

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include tabular preprocessing specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add tabular preprocessing specific fields
        preprocessing_fields = {
            "label_name": self.label_name,
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "train_ratio": self.train_ratio,
            "test_val_ratio": self.test_val_ratio,
        }

        # Combine fields (preprocessing fields take precedence if overlap)
        init_fields = {**base_fields, **preprocessing_fields}

        return init_fields

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)

        # Add derived properties
        if self.full_script_path:
            data["full_script_path"] = self.full_script_path

        return data
