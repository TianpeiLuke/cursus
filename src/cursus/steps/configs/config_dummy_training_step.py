"""
Configuration for DummyTraining step with flexible input modes.

This module defines the configuration class for the DummyTraining step,
which is an INTERNAL node that can accept optional inputs from previous steps
or fall back to reading from the source directory.
"""

from pydantic import Field, model_validator, field_validator
from typing import TYPE_CHECKING, Optional, Union
from pathlib import Path

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.dummy_training_contract import DUMMY_TRAINING_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract


class DummyTrainingConfig(ProcessingStepConfigBase):
    """
    Configuration for DummyTraining step with flexible input modes.

    This configuration follows the Three-Tier Config Design pattern:

    **Tier 1 (Essential Fields)**: Required user inputs
    - `pretrained_model_path`: Path to model artifacts (S3 URI, local path, or None)
    - Inherited required fields from BasePipelineConfig and ProcessingStepConfigBase

    **Tier 2 (System Fields)**: Fields with defaults that can be overridden
    - `processing_entry_point`: Entry point script (default: "dummy_training.py")
    - Instance types, volume sizes, etc. (inherited from base)

    **Tier 3 (Derived Fields)**: Calculated from Tier 1 and Tier 2
    - Inherited derived fields like `effective_source_dir`, `script_path`

    ## Model Artifacts Input (Tier 1 Essential Field)

    The `pretrained_model_path` field accepts three types of values:

    1. **None (default)** - SOURCE Fallback Assumption:
       - Assumes model.tar.gz is located at `source_dir/models/model.tar.gz`
       - Default behavior for backward compatibility
       - Example: `pretrained_model_path=None` or omit the field

    2. **S3 URI** - Explicit S3 Path:
       - Full S3 path to model directory or file
       - Examples:
         - `s3://my-bucket/models/` (directory)
         - `s3://my-bucket/models/model.tar.gz` (file)

    3. **Local Path** - Explicit Local Directory:
       - Relative or absolute path to model directory
       - Examples:
         - `./models/` (relative)
         - `/absolute/path/to/models/` (absolute)
         - `source_dir/models/` (relative to source)

    **Priority Resolution**: Config field → Dependency injection → SOURCE fallback

    ## Hyperparameters Resolution

    Hyperparameters follow dependency injection pattern:
    - From `hyperparameters_s3_uri` channel (if provided via dependency injection)
    - Falls back to multiple SOURCE locations:
      - `/opt/ml/code/hyperparams/hyperparameters.json`
      - `source_dir/hyperparams/hyperparameters.json`
      - `source_dir/hyperparameters.json`

    ## Use Cases

    - **SOURCE Fallback**: `pretrained_model_path=None` (default, uses source_dir/models/)
    - **Explicit S3**: `pretrained_model_path="s3://bucket/path/to/models/"`
    - **Explicit Local**: `pretrained_model_path="path/to/models/"` or `"./models/"`
    - **Absolute Path**: `pretrained_model_path="/absolute/path/to/models/"`

    ## Expected Source Directory Structure (when pretrained_model_path=None)

    ```
    source_dir/
    ├── dummy_training.py          # Main processing script
    ├── models/                    # Model directory
    │   └── model.tar.gz          # Pre-trained model artifacts
    └── hyperparams/              # Hyperparameters directory (optional)
        └── hyperparameters.json  # Hyperparameters file
    ```

    ## Example Configs

    ```python
    # SOURCE fallback (None assumption)
    config = DummyTrainingConfig(
        pretrained_model_path=None,  # or omit entirely
        # ... other required fields
    )

    # Explicit S3 path
    config = DummyTrainingConfig(
        pretrained_model_path="s3://my-bucket/models/",
        # ... other required fields
    )

    # Explicit local path
    config = DummyTrainingConfig(
        pretrained_model_path="./models/",
        # ... other required fields
    )
    ```
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    pretrained_model_path: Optional[Union[str, Path]] = Field(
        ...,
        description="Path to pretrained model.tar.gz file or directory containing it. "
        "Supports S3 URI (s3://...), local directory path, or None. "
        "If None, assumes model.tar.gz is located at source_dir/models/model.tar.gz",
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="dummy_training.py",
        description="Entry point script for dummy training step.",
    )

    # ===== Derived Fields (Tier 3) =====
    # Inherited from ProcessingStepConfigBase:
    # - effective_source_dir (from processing_source_dir or source_dir)
    # - effective_instance_type (based on use_large_processing_instance)
    # - script_path (combining effective_source_dir and processing_entry_point)

    model_config = ProcessingStepConfigBase.model_config

    # ===== Validators =====

    @field_validator("pretrained_model_path")
    @classmethod
    def validate_pretrained_model_path(
        cls, v: Optional[Union[str, Path]]
    ) -> Optional[str]:
        """
        Validate pretrained_model_path is None, S3 URI, or local directory path.

        Converts Path objects to strings for consistent handling.

        Args:
            v: Path value to validate (str, Path, or None)

        Returns:
            Validated path as string or None

        Raises:
            ValueError: If path format is invalid
        """
        # None is valid - means use SOURCE fallback (source_dir/models/model.tar.gz)
        if v is None:
            return v

        # Convert Path to string for consistent handling
        if isinstance(v, Path):
            v = str(v)

        # Empty string not allowed
        if not v:
            raise ValueError(
                "pretrained_model_path cannot be empty string. Use None to indicate SOURCE fallback"
            )

        # Check if S3 path
        if v.startswith("s3://"):
            if not v.replace("s3://", "").strip("/"):
                raise ValueError(f"Invalid S3 path format: {v}")
            return v

        # Local path (absolute or relative)
        # Don't validate existence - that's a runtime check in builders
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "DummyTrainingConfig":
        """
        Validate configuration for INTERNAL node with optional inputs.

        For INTERNAL nodes with optional dependencies, we validate:
        - Entry point is specified
        - Script contract is valid
        - Output paths are correctly defined

        File existence is checked at runtime, not configuration time.

        Returns:
            Self with validated configuration
        """
        # Basic validation - entry point is required
        if not self.processing_entry_point:
            raise ValueError("DummyTraining step requires a processing_entry_point")

        # Validate script contract
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        # For INTERNAL nodes with optional inputs, contract can have input paths
        # but they should be optional (not enforced at config time)

        # Ensure we have the required output path (updated logical name)
        if "model_output" not in contract.expected_output_paths:
            raise ValueError(
                "Script contract missing required output path: model_output"
            )

        return self

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The DummyTraining script contract
        """
        return DUMMY_TRAINING_CONTRACT
