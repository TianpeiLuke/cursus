"""
TSA Data Preprocessing Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for TSA (Temporal Self-Attention) data preprocessing, using a self-contained
design where each field is properly categorized according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from .config_processing_step_base import ProcessingStepConfigBase

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class TSAPreprocessingConfig(ProcessingStepConfigBase):
    """
    Configuration for the TSA Preprocessing step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    job_type: str = Field(
        default="training",
        description="Processing job type. One of ['training', 'validation', 'testing', 'calibration']",
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="tsa_preprocessing.py",
        description="Relative path (within processing_source_dir) to the TSA preprocessing script.",
    )

    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    framework_version: str = Field(
        default="2.1.0",
        description="PyTorch framework version for processing",
    )

    tag: str = Field(
        default="is_abusive_mdr",
        description="Primary label field name (e.g., is_abusive_mdr for fraud detection).",
    )

    tag2: str = Field(
        default="is_flr",
        description="Secondary label field name (e.g., is_flr for false legitimacy rate) for dual-task learning.",
    )

    target_positive_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Target positive rate for downsampling in training mode. Only applies to training jobs.",
    )

    time_window_train: int = Field(
        default=240,
        ge=1,
        description="Time window in days for filtering training data.",
    )

    time_window_calib: int = Field(
        default=150,
        ge=1,
        description="Time window in days for filtering calibration data.",
    )

    time_window_vali: int = Field(
        default=90,
        ge=1,
        description="Time window in days for filtering validation/testing data.",
    )

    amount_field: str = Field(
        default="concamt",
        description="Column name for concession/transaction amount used to evaluate model performance (e.g., 'concamt', 'totalAmount'). Used for dollar_recall metric calculations in model evaluation.",
    )

    preprocessor_path: Optional[str] = Field(
        default=None,
        description="Optional override for preprocessor file path. If provided, this takes precedence over the automatically constructed path from preprocessor input. Use format: '/opt/ml/processing/input/code/preprocessor.pkl'",
    )

    # Hyperparameters from TSA model (aligned with hyperparameters_tsa.py)
    seq_len: int = Field(
        default=51,
        ge=1,
        description="Fixed sequence length for TSA model padding. Must match the value used during model training.",
    )

    data_version: str = Field(
        default="v0",
        description="Data version string used in output filenames (e.g., 'v0', 'v1'). Allows tracking different data preprocessing versions.",
    )

    seed: int = Field(
        default=0,
        ge=0,
        description="Random seed for reproducibility in downsampling operations during training.",
    )

    enable_tsa_streaming: bool = Field(
        default=False,
        description="Enable true streaming mode with memory-mapped arrays for fixed memory usage regardless of dataset size. When enabled, processes data in batches using direct write to avoid loading full dataset into memory. Recommended for large datasets to prevent OOM errors.",
    )

    tsa_streaming_batch_size: int = Field(
        default=10,
        ge=1,
        description="Number of files to process per batch in streaming mode. Only used when enable_tsa_streaming=True. Larger batches process faster but use more memory. Recommended: 10-20 files per batch.",
    )

    validation_split_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Fraction of downsampled training data to hold out as validation. "
        "Ensures train and validation sets share the same positive rate after downsampling. "
        "Only applies when job_type='training'. Set to 0.0 to disable.",
    )

    time_delta_cap: int = Field(
        default=20736000,
        ge=0,
        description="Maximum time delta in seconds for temporal attention capping. "
        "Default: 20,736,000 (240 days). Must match the value used in inference scripts "
        "(tsa_inference_*_calibrated.py). Passed as TIME_DELTA_CAP env var to preprocessing.",
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
        Get preprocessing-specific environment variables based on configuration.

        These are static environment variables derived from config fields like
        target_positive_rate, time windows, and tag. Dynamic environment variables
        that depend on step inputs (like PREPROCESSOR_PATH) are added by the builder
        at runtime when the step is created.

        Returns:
            Dictionary mapping environment variable names to values
        """
        if self._preprocessing_environment_variables is None:
            env_vars = {}

            # Add region identifier (convert to lowercase for artifact file naming)
            # Base config uses "NA"/"EU"/"FE", but artifacts use "na"/"eu"/"fe"
            env_vars["REGION"] = self.region

            # Add label tags for dual-task learning
            env_vars["TAG"] = self.tag
            env_vars["TAG2"] = self.tag2

            # Add target positive rate
            env_vars["TARGET_POSITIVE_RATE"] = str(self.target_positive_rate)

            # Add time windows
            env_vars["TIME_WINDOW_TRAIN"] = str(self.time_window_train)
            env_vars["TIME_WINDOW_CALIB"] = str(self.time_window_calib)
            env_vars["TIME_WINDOW_VALI"] = str(self.time_window_vali)

            # Add amount field for dollar_recall calculations
            env_vars["AMOUNT_FIELD"] = self.amount_field

            # Add hyperparameters from TSA model (must match training hyperparameters)
            env_vars["SEQ_LEN"] = str(self.seq_len)
            env_vars["DATA_VERSION"] = self.data_version
            env_vars["SEED"] = str(self.seed)

            # Add validation split ratio (only effective for training job_type)
            env_vars["VALIDATION_SPLIT_RATIO"] = str(self.validation_split_ratio)

            # Add streaming mode parameters
            if self.enable_tsa_streaming:
                env_vars["ENABLE_TSA_STREAMING"] = "true"
                env_vars["TSA_STREAMING_BATCH_SIZE"] = str(
                    self.tsa_streaming_batch_size
                )
            else:
                env_vars["ENABLE_TSA_STREAMING"] = "false"

            # Add time delta cap for temporal attention
            env_vars["TIME_DELTA_CAP"] = str(self.time_delta_cap)

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
    def validate_job_type(cls, v: str) -> str:
        """
        Ensure job_type is one of the allowed values.
        """
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("target_positive_rate")
    @classmethod
    def validate_target_positive_rate(cls, v: float) -> float:
        """
        Ensure target_positive_rate is between 0 and 1 (inclusive).
        """
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"target_positive_rate must be between 0 and 1, got {v}")
        return v

    @field_validator("time_window_train", "time_window_calib", "time_window_vali")
    @classmethod
    def validate_time_window(cls, v: int) -> int:
        """
        Ensure time window is a positive integer.
        """
        if v < 1:
            raise ValueError(f"Time window must be at least 1 day, got {v}")
        return v

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "TSAPreprocessingConfig":
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
    # get_script_contract() / get_script_path() are inherited from BasePipelineConfig, which loads
    # the contract from the unified .step.yaml interface via the step catalog (Design B standard —
    # matches ModelCalibration / XGBoostModelEval / TabularPreprocessing, which define no override).

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include TSA preprocessing specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add TSA preprocessing specific fields
        preprocessing_fields = {
            "job_type": self.job_type,
            "processing_entry_point": self.processing_entry_point,
            "py_version": self.py_version,
            "framework_version": self.framework_version,
            "tag": self.tag,
            "tag2": self.tag2,
            "target_positive_rate": self.target_positive_rate,
            "time_window_train": self.time_window_train,
            "time_window_calib": self.time_window_calib,
            "time_window_vali": self.time_window_vali,
            "amount_field": self.amount_field,
            "preprocessor_path": self.preprocessor_path,
            "seq_len": self.seq_len,
            "data_version": self.data_version,
            "seed": self.seed,
            "enable_tsa_streaming": self.enable_tsa_streaming,
            "tsa_streaming_batch_size": self.tsa_streaming_batch_size,
            "validation_split_ratio": self.validation_split_ratio,
            "time_delta_cap": self.time_delta_cap,
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
