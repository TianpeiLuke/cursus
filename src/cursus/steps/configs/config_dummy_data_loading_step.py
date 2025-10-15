"""
Dummy Data Loading Step Configuration

This module implements the configuration class for the Dummy Data Loading step,
which processes user-provided data instead of calling internal Cradle services.
"""

from pydantic import Field, model_validator
from typing import TYPE_CHECKING, Optional, Dict, Any, Union
from pathlib import Path

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.dummy_data_loading_contract import DUMMY_DATA_LOADING_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract


class DummyDataLoadingConfig(ProcessingStepConfigBase):
    """
    Configuration for a dummy data loading step.

    This configuration follows the three-tier field categorization:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that users can override
    3. Tier 3: Derived Fields - fields calculated from other fields, stored in private attributes
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    data_source: Union[str, Path] = Field(
        ...,
        description="Local directory path or S3 URI where the input data is stored. "
                   "Examples: '/path/to/local/data' or 's3://bucket/path/to/data'"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="dummy_data_loading.py", 
        description="Entry point script for dummy data loading."
    )

    # Data processing options
    max_file_size_mb: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Maximum file size in MB to process (safety limit)"
    )

    supported_formats: list[str] = Field(
        default=["csv", "parquet", "json", "jsonl"],
        description="List of supported data formats for processing"
    )

    # Update to Pydantic V2 style model_config
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",  # Allow extra fields for type-aware serialization
    }

    @model_validator(mode="after")
    def validate_config(self) -> "DummyDataLoadingConfig":
        """
        Validate configuration and ensure defaults are set.

        This validator ensures that:
        1. Data source is provided and properly formatted
        2. Entry point is provided
        3. Script contract is available and valid
        4. Required input/output paths are defined in the script contract
        """
        # Validate data source
        if not self.data_source:
            raise ValueError("data_source is required for dummy data loading step")

        # Convert Path to string for consistency
        if isinstance(self.data_source, Path):
            self.data_source = str(self.data_source)

        # Validate data source format
        data_source_str = str(self.data_source)
        if data_source_str.startswith("s3://"):
            # Validate S3 URI format
            if not data_source_str.replace("s3://", "").strip("/"):
                raise ValueError(f"Invalid S3 URI format: {data_source_str}")
        else:
            # For local paths, we don't validate existence at config time for portability
            # Validation will happen at execution time in the step builder
            pass

        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("dummy data loading step requires a processing_entry_point")


        # Validate supported formats
        valid_formats = {"csv", "parquet", "json", "jsonl", "pq"}
        invalid_formats = set(self.supported_formats) - valid_formats
        if invalid_formats:
            raise ValueError(f"Unsupported data formats: {invalid_formats}. "
                           f"Valid formats: {valid_formats}")

        return self

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The dummy data loading script contract
        """
        return DUMMY_DATA_LOADING_CONTRACT

    def is_s3_source(self) -> bool:
        """
        Check if the data source is an S3 URI.

        Returns:
            bool: True if data source is S3, False if local path
        """
        return str(self.data_source).startswith("s3://")

    def get_data_source_uri(self) -> str:
        """
        Get the data source as a string URI.

        Returns:
            str: Data source URI (S3 or local path)
        """
        return str(self.data_source)

    def get_supported_extensions(self) -> set[str]:
        """
        Get supported file extensions based on configured formats.

        Returns:
            set[str]: Set of supported file extensions (with dots)
        """
        extension_map = {
            "csv": {".csv"},
            "parquet": {".parquet", ".pq"},
            "json": {".json"},
            "jsonl": {".jsonl"}
        }
        
        extensions = set()
        for format_name in self.supported_formats:
            extensions.update(extension_map.get(format_name, set()))
        
        return extensions

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        
        # Add derived properties to output
        data["is_s3_source"] = self.is_s3_source()
        data["data_source_uri"] = self.get_data_source_uri()
        data["supported_extensions"] = list(self.get_supported_extensions())
        
        return data
