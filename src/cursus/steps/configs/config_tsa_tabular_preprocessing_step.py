"""
TSA Tabular Preprocessing Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for TSA (Travel & Spend Analytics) tabular data preprocessing.

Three-tier field design:
  Tier 1: Essential User Inputs  — required fields provided by the caller.
  Tier 2: System Fields          — reasonable defaults that can be overridden.
  Tier 3: Derived Fields         — private, exposed via read-only properties.

Key divergences from the generic TabularPreprocessingConfig:
  • TSA-domain env-var fields: tsa_label_field, tsa_id_fields, tsa_date_field.
  • Preprocessor output path field: preprocessor_output_path.
  • preprocessing_environment_variables includes TSA_* keys.
  • Entry point defaults to tsa_tabular_preprocessing.py.
  • get_job_arguments() emits --label-field, --id-fields, --date-field,
    --preprocessor-output-path in addition to --job_type.
"""

from pydantic import Field, field_validator, model_validator, PrivateAttr
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TSATabularPreprocessingConfig(ProcessingStepConfigBase):
    """
    Configuration for the TSA Tabular Preprocessing step.

    Inherits from ProcessingStepConfigBase and extends it with:
    - TSA-domain environment variable fields (label, ID columns, date column).
    - Preprocessor artifact output path.
    - TSA-specific job arguments passed via argparse to the script.
    """

    # =========================================================================
    # Tier 1: Essential User Inputs
    # =========================================================================

    job_type: str = Field(
        description="Lowercase alphanumeric slice name (e.g. 'training','validation','testing','calibration')",
    )

    # =========================================================================
    # Tier 2: System Fields with Defaults
    # =========================================================================

    # --- TSA domain fields ---------------------------------------------------

    tsa_label_field: Optional[str] = Field(
        default=None,
        description=(
            "Name of the target/label column in the TSA dataset. "
            "Passed as --label-field to the script and exported as TSA_LABEL_FIELD. "
            "Optional for calibration jobs."
        ),
    )

    tsa_id_fields: Optional[str] = Field(
        default=None,
        description=(
            "Comma-separated list of ID columns to exclude from feature engineering "
            "(e.g. 'account_id,transaction_id'). "
            "Passed as --id-fields and exported as TSA_ID_FIELDS."
        ),
    )

    tsa_date_field: Optional[str] = Field(
        default=None,
        description=(
            "Name of the primary date/timestamp column used for temporal feature extraction "
            "(e.g. 'transaction_date'). "
            "Passed as --date-field and exported as TSA_DATE_FIELD."
        ),
    )

    preprocessor_output_path: str = Field(
        default="/opt/ml/processing/output/preprocessor",
        description=(
            "Container path where the fitted sklearn preprocessor pipeline "
            "(preprocessor.pkl) will be written. "
            "Passed as --preprocessor-output-path to the script."
        ),
    )

    # --- Generic preprocessing parameters ------------------------------------

    processing_entry_point: str = Field(
        default="tsa_tabular_preprocessing.py",
        description=(
            "Relative path (within processing_source_dir) to the TSA preprocessing script."
        ),
    )

    train_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of data allocated to the training set "
            "(used only when job_type=='training')."
        ),
    )

    test_val_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the holdout set allocated to the test split vs. validation "
            "(used only when job_type=='training')."
        ),
    )

    output_format: str = Field(
        default="CSV",
        description="Output format for processed data. One of CSV / TSV / Parquet.",
    )

    max_workers: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum parallel workers for shard reading. "
            "0=auto (uses cpu_count), 1=sequential (lowest memory)."
        ),
    )

    batch_size: int = Field(
        default=5,
        ge=2,
        le=10,
        description="DataFrame concatenation batch size. Range 2–10.",
    )

    optimize_memory: bool = Field(
        default=False,
        description=(
            "Enable dtype optimisation to reduce memory usage. "
            "Downcasts numeric types and converts low-cardinality columns to category."
        ),
    )

    streaming_batch_size: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of shards to process per batch in streaming mode. "
            "0=disabled (loads all shards at once)."
        ),
    )

    enable_true_streaming: bool = Field(
        default=False,
        description=(
            "Enable fully parallel streaming mode with 1:1 shard mapping. "
            "8–10× faster than batch mode with fixed memory usage."
        ),
    )

    # =========================================================================
    # Tier 3: Derived / Private Fields
    # =========================================================================

    _full_script_path: Optional[str] = PrivateAttr(default=None)
    _tsa_environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # =========================================================================
    # Properties for Derived Fields
    # =========================================================================

    @property
    def full_script_path(self) -> Optional[str]:
        """Full resolved path to the TSA preprocessing entry-point script."""
        if self._full_script_path is None:
            source_dir = self.effective_source_dir
            if source_dir is None:
                return None
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
    def tsa_environment_variables(self) -> Dict[str, str]:
        """
        Returns the full set of environment variables required by the TSA
        preprocessing script, including both generic and TSA-specific knobs.
        """
        if self._tsa_environment_variables is None:
            env_vars: Dict[str, str] = {}

            # --- TSA-specific vars ---
            if self.tsa_label_field:
                env_vars["TSA_LABEL_FIELD"] = self.tsa_label_field
            if self.tsa_id_fields:
                env_vars["TSA_ID_FIELDS"] = self.tsa_id_fields
            if self.tsa_date_field:
                env_vars["TSA_DATE_FIELD"] = self.tsa_date_field

            # --- Generic preprocessing vars ---
            env_vars["OUTPUT_FORMAT"] = self.output_format
            env_vars["TRAIN_RATIO"] = str(self.train_ratio)
            env_vars["TEST_VAL_RATIO"] = str(self.test_val_ratio)
            env_vars["MAX_WORKERS"] = str(self.max_workers)
            env_vars["BATCH_SIZE"] = str(self.batch_size)
            env_vars["OPTIMIZE_MEMORY"] = "true" if self.optimize_memory else "false"
            env_vars["STREAMING_BATCH_SIZE"] = str(self.streaming_batch_size)
            env_vars["ENABLE_TRUE_STREAMING"] = (
                "true" if self.enable_true_streaming else "false"
            )

            self._tsa_environment_variables = env_vars

        return self._tsa_environment_variables

    # Keep legacy alias used by ProcessingStepHandler env-var injection
    @property
    def preprocessing_environment_variables(self) -> Dict[str, str]:
        """Alias for tsa_environment_variables (ProcessingStepHandler compatibility)."""
        return self.tsa_environment_variables

    # =========================================================================
    # Validators
    # =========================================================================

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """Entry point must be a non-empty relative path."""
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non-empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError(
                "processing_entry_point must be a relative path within source directory"
            )
        return v

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """job_type must be lowercase alphanumeric (with underscores)."""
        if not v.replace("_", "").isalnum() or v != v.lower():
            raise ValueError(
                f"job_type must be lowercase alphanumeric (with underscores), got '{v}'"
            )
        return v

    @field_validator("train_ratio", "test_val_ratio")
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        """Split ratios must be strictly between 0 and 1."""
        if not (0.0 < v < 1.0):
            raise ValueError(f"Split ratio must be strictly between 0 and 1, got {v}")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """
        output_format is case-sensitive per the field_validator in TabularPreprocessingConfig.
        Allowed: CSV / TSV / Parquet (exact case).
        """
        allowed = {"CSV", "TSV", "Parquet"}
        match = next((a for a in allowed if a.lower() == v.lower()), None)
        if match is None:
            raise ValueError(
                f"output_format must be one of {sorted(allowed)} (case-insensitive input, "
                f"stored as canonical case), got '{v}'"
            )
        return match

    # =========================================================================
    # Model Validator (derived-field initialisation)
    # =========================================================================

    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "TSATabularPreprocessingConfig":
        """Initialise derived fields once after Pydantic validation completes."""
        super().initialize_derived_fields()

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

    # =========================================================================
    # Overrides
    # =========================================================================

    def get_public_init_fields(self) -> Dict[str, Any]:
        """Return all fields needed to construct child/sibling config instances."""
        base_fields = super().get_public_init_fields()
        tsa_fields = {
            "job_type": self.job_type,
            "tsa_label_field": self.tsa_label_field,
            "tsa_id_fields": self.tsa_id_fields,
            "tsa_date_field": self.tsa_date_field,
            "preprocessor_output_path": self.preprocessor_output_path,
            "processing_entry_point": self.processing_entry_point,
            "train_ratio": self.train_ratio,
            "test_val_ratio": self.test_val_ratio,
            "output_format": self.output_format,
        }
        return {**base_fields, **tsa_fields}

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Include derived properties in serialisation."""
        data = super().model_dump(**kwargs)
        if self.full_script_path:
            data["full_script_path"] = self.full_script_path
        return data

    def get_job_arguments(self) -> Optional[List[str]]:
        """
        Returns the CLI arguments passed to the SageMaker Processing container.

        Includes --job_type (standard) plus TSA-specific flags:
        --label-field, --id-fields, --date-field, --preprocessor-output-path.
        """
        args: List[str] = self._job_type_arg()  # ["--job_type", "<value>"]

        if self.tsa_label_field:
            args += ["--label-field", self.tsa_label_field]
        if self.tsa_id_fields:
            args += ["--id-fields", self.tsa_id_fields]
        if self.tsa_date_field:
            args += ["--date-field", self.tsa_date_field]

        # Always pass preprocessor output path so the script knows where to write
        args += ["--preprocessor-output-path", self.preprocessor_output_path]

        return args
