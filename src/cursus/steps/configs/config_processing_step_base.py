"""
Processing Step Base Configuration with Self-Contained Derivation Logic

This module implements the base configuration class for SageMaker Processing steps
using a self-contained design where derived fields are private with read-only properties.
"""

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    ValidationInfo,
    PrivateAttr,
)
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from ...core.base.config_base import BasePipelineConfig


class ProcessingStepConfigBase(BasePipelineConfig):
    """Base configuration for SageMaker Processing Steps with self-contained derivation logic."""

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Processing instance settings
    processing_instance_count: int = Field(
        default=1, ge=1, le=10, description="Instance count for processing jobs"
    )

    processing_volume_size: int = Field(
        default=500, ge=10, le=1000, description="Volume size for processing jobs in GB"
    )

    processing_instance_type_large: str = Field(
        default="ml.m5.4xlarge", description="Large instance type for processing step."
    )

    processing_instance_type_small: str = Field(
        default="ml.m5.2xlarge", description="Small instance type for processing step."
    )

    use_large_processing_instance: bool = Field(
        default=False,
        description="Set to True to use large instance type, False for small instance type.",
    )

    # Script and directory settings
    processing_source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for processing scripts. Falls back to base source_dir if not provided.",
    )

    processing_entry_point: Optional[str] = Field(
        default=None,
        description="Entry point script for processing, must be relative to source directory. Can be overridden by derived classes.",
    )

    processing_script_arguments: Optional[List[str]] = Field(
        default=None, description="Optional arguments for the processing script."
    )

    # Framework version
    processing_framework_version: str = Field(
        default="1.2-1",  # Using 1.2-1 (Python 3.8) as default
        description="Version of the scikit-learn framework to use in SageMaker Processing. Format: '<sklearn-version>-<build-number>'",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _effective_source_dir: Optional[str] = PrivateAttr(default=None)
    _effective_instance_type: Optional[str] = PrivateAttr(default=None)
    _script_path: Optional[str] = PrivateAttr(default=None)

    model_config = BasePipelineConfig.model_config

    # Public read-only properties for derived fields

    @property
    def effective_source_dir(self) -> Optional[str]:
        """
        Get effective source directory with hybrid resolution.
        
        Resolution Priority:
        1. Hybrid resolution of processing_source_dir
        2. Hybrid resolution of source_dir
        3. Legacy values (processing_source_dir, source_dir)
        """
        if self._effective_source_dir is None:
            # Strategy 1: Hybrid resolution of processing_source_dir
            if self.processing_source_dir:
                resolved = self.resolve_hybrid_path(self.processing_source_dir)
                if resolved and Path(resolved).exists():
                    self._effective_source_dir = resolved
                    return self._effective_source_dir
            
            # Strategy 2: Hybrid resolution of source_dir
            if self.source_dir:
                resolved = self.resolve_hybrid_path(self.source_dir)
                if resolved and Path(resolved).exists():
                    self._effective_source_dir = resolved
                    return self._effective_source_dir
            
            # Strategy 3: Legacy fallback (current behavior)
            if self.processing_source_dir is not None:
                self._effective_source_dir = self.processing_source_dir
            else:
                self._effective_source_dir = self.source_dir
        
        return self._effective_source_dir

    @property
    def effective_instance_type(self) -> str:
        """Get the appropriate instance type based on the use_large_processing_instance flag."""
        if self._effective_instance_type is None:
            self._effective_instance_type = (
                self.processing_instance_type_large
                if self.use_large_processing_instance
                else self.processing_instance_type_small
            )
        return self._effective_instance_type

    @property
    def script_path(self) -> Optional[str]:
        """
        Get script path with hybrid resolution.
        
        Uses modernized effective_source_dir which already includes hybrid resolution.
        """
        if self.processing_entry_point is None:
            return None

        if self._script_path is None:
            # Use modernized effective_source_dir (which includes hybrid resolution)
            effective_source = self.effective_source_dir
            if effective_source is None:
                return None

            # Construct full script path
            if effective_source.startswith("s3://"):
                self._script_path = (
                    f"{effective_source.rstrip('/')}/{self.processing_entry_point}"
                )
            else:
                self._script_path = str(
                    Path(effective_source) / self.processing_entry_point
                )

        return self._script_path

    @property
    def resolved_processing_source_dir(self) -> Optional[str]:
        """Get resolved processing source directory using hybrid resolution."""
        if self.processing_source_dir:
            return self.resolve_hybrid_path(self.processing_source_dir)
        elif self.source_dir:
            return self.resolve_hybrid_path(self.source_dir)
        return None

    def get_resolved_script_path(self) -> Optional[str]:
        """Get resolved script path for step builders using hybrid resolution."""
        if not self.processing_entry_point:
            return None
        
        # Try hybrid resolution first
        resolved_source_dir = self.resolved_processing_source_dir
        if resolved_source_dir:
            return str(Path(resolved_source_dir) / self.processing_entry_point)
        
        # Fallback to legacy script_path property
        return self.script_path

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["effective_source_dir"] = self.effective_source_dir
        data["effective_instance_type"] = self.effective_instance_type
        if self.script_path:
            data["script_path"] = self.script_path
        
        return data

    # Validators

    @field_validator("processing_source_dir")
    @classmethod
    def validate_processing_source_dir(cls, v: Optional[str]) -> Optional[str]:
        """Validate processing source directory format (S3 paths only)."""
        if v is not None:
            if v.startswith("s3://"):
                if not v.replace("s3://", "").strip("/"):
                    raise ValueError(f"Invalid S3 path format: {v}")
            # Removed local path existence validation to improve configuration portability
            # Path validation should happen at execution time in builders, not at config creation time
        return v

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_is_relative(cls, v: Optional[str]) -> Optional[str]:
        """Validate entry point is a relative path if provided."""
        if v is not None:
            if not v:
                raise ValueError("processing_entry_point if provided cannot be empty.")
            if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
                raise ValueError(
                    f"processing_entry_point ('{v}') must be a relative path within source directory."
                )
        return v

    @field_validator("processing_framework_version")
    @classmethod
    def validate_framework_version(cls, v: str) -> str:
        """
        Validate processing framework version matches SageMaker SKLearn versions.
        Reference: https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html
        """
        # Define versions by Python compatibility
        py37_versions = [
            "0.20.0-1",
            "0.23-1",  # Supports scikit-learn 0.23.2
            "0.23-2",  # Supports scikit-learn 0.23.2
        ]

        py38_versions = [
            "0.23-3",  # Supports scikit-learn 0.23.2
            "0.23-4",  # Supports scikit-learn 0.23.2
            "0.24-0",  # Supports scikit-learn 0.24.x
            "0.24-1",  # Supports scikit-learn 0.24.x
            "1.0-1",  # Supports scikit-learn 1.0.2
            "1.2-1",  # Supports scikit-learn 1.2.2
        ]

        py39_versions = [
            "1.3-1",  # Supports scikit-learn 1.3.x
            "1.4-1",  # Supports scikit-learn 1.4.x
            "1.5-1",  # Supports scikit-learn 1.5.x
            "2.0-1",  # Supports scikit-learn 2.0.x
        ]

        # Combined list of all valid versions
        valid_versions = py37_versions + py38_versions + py39_versions

        # Check if version is valid
        if v not in valid_versions:
            # Prepare a more informative error message
            py_compatibility = "\nPython version compatibility:\n"
            py_compatibility += (
                "- Python 3.7 (NOT RECOMMENDED): " + ", ".join(py37_versions) + "\n"
            )
            py_compatibility += "- Python 3.8: " + ", ".join(py38_versions) + "\n"
            py_compatibility += (
                "- Python 3.9 and newer: " + ", ".join(py39_versions) + "\n"
            )

            raise ValueError(
                f"Invalid processing framework version: {v}.\n"
                f"Must be one of the valid SageMaker SKLearn processing container versions.\n"
                f"{py_compatibility}"
                f"\nRecommendation: Use 1.4-1 or newer for best compatibility."
            )

        # Add warning for Python 3.7 versions
        if v in py37_versions:
            logger.warning(
                f"Warning: Framework version {v} uses Python 3.7, which is no longer supported. "
                f"Consider upgrading to a version that supports Python 3.8 or 3.9."
            )

        return v

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "ProcessingStepConfigBase":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize processing-specific derived fields
        # Priority 1: Processing source dir (fallback)
        if self.processing_source_dir is not None:
            self._effective_source_dir = self.processing_source_dir
        # Priority 2: Base source dir (final fallback)
        else:
            self._effective_source_dir = self.source_dir

        self._effective_instance_type = (
            self.processing_instance_type_large
            if self.use_large_processing_instance
            else self.processing_instance_type_small
        )

        # Initialize script path if entry point is provided
        if (
            self.processing_entry_point is not None
            and self._effective_source_dir is not None
        ):
            if self._effective_source_dir.startswith("s3://"):
                self._script_path = f"{self._effective_source_dir.rstrip('/')}/{self.processing_entry_point}"
            else:
                self._script_path = str(
                    Path(self._effective_source_dir) / self.processing_entry_point
                )

        return self

    @model_validator(mode="after")
    def validate_entry_point_paths(self) -> "ProcessingStepConfigBase":
        """Validate entry point configuration requirements (without file existence checks)."""
        if self.processing_entry_point is None:
            logger.debug(
                "No processing_entry_point provided in base config. Skipping path validation."
            )
            return self

        effective_source_dir = self.effective_source_dir

        if not effective_source_dir:
            if not self.processing_entry_point.startswith("s3://"):
                raise ValueError(
                    "Either processing_source_dir or source_dir must be defined "
                    "to locate local processing_entry_point."
                )
        elif effective_source_dir.startswith("s3://"):
            logger.debug(
                f"Processing source directory ('{effective_source_dir}') is S3. "
                f"Assuming processing_entry_point '{self.processing_entry_point}' exists within it."
            )
        else:
            # Removed file existence validation to improve configuration portability
            # File validation should happen at execution time in builders, not at config creation time
            logger.debug(
                f"Processing entry point configured: '{self.processing_entry_point}' "
                f"in source directory '{effective_source_dir}'"
            )

        return self

    # Legacy compatibility methods

    def get_effective_source_dir(self) -> Optional[str]:
        """Get the effective source directory (legacy compatibility)."""
        return self.effective_source_dir

    def get_instance_type(self, size: Optional[str] = None) -> str:
        """
        Get the appropriate instance type based on size parameter or configuration.

        Args:
            size (Optional[str]): Override 'small' or 'large'. If None, uses use_large_processing_instance.

        Returns:
            str: The corresponding instance type
        """
        if size is None:
            return self.effective_instance_type

        if size.lower() == "large":
            return self.processing_instance_type_large
        elif size.lower() == "small":
            return self.processing_instance_type_small
        else:
            raise ValueError(
                f"Invalid size parameter: {size}. Must be 'small' or 'large'"
            )

    def get_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """
        Get script path with hybrid resolution and comprehensive fallbacks.
        
        Resolution Priority:
        1. Modernized script_path property (includes hybrid resolution)
        2. Direct hybrid resolution of entry_point
        3. Legacy get_resolved_script_path() method
        4. Default path fallback
        
        Args:
            default_path: Default path to use if all resolution methods fail
            
        Returns:
            Optional[str]: Resolved script path or default_path if not found
        """
        
        # Strategy 1: Use modernized script_path property (includes hybrid resolution)
        path = self.script_path
        if path and Path(path).exists():
            return path
        
        # Strategy 2: Direct hybrid resolution of entry_point
        if self.processing_entry_point:
            # Try with processing_source_dir first
            if self.processing_source_dir:
                relative_path = f"{self.processing_source_dir}/{self.processing_entry_point}"
            elif self.source_dir:
                relative_path = f"{self.source_dir}/{self.processing_entry_point}"
            else:
                relative_path = self.processing_entry_point
            
            resolved = self.resolve_hybrid_path(relative_path)
            if resolved and Path(resolved).exists():
                return resolved
        
        # Strategy 3: Legacy get_resolved_script_path() method
        try:
            resolved_path = self.get_resolved_script_path()
            if resolved_path:
                return resolved_path
        except Exception:
            pass
        
        # Strategy 4: Default fallback
        return default_path

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include processing-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and processing-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (BasePipelineConfig)
        base_fields = super().get_public_init_fields()

        # Add processing-specific fields (Tier 2 - System Inputs with Defaults)
        processing_fields = {
            "processing_instance_count": self.processing_instance_count,
            "processing_volume_size": self.processing_volume_size,
            "processing_instance_type_large": self.processing_instance_type_large,
            "processing_instance_type_small": self.processing_instance_type_small,
            "use_large_processing_instance": self.use_large_processing_instance,
            "processing_framework_version": self.processing_framework_version,
        }

        # Only include optional fields if they're set
        if self.processing_source_dir is not None:
            processing_fields["processing_source_dir"] = self.processing_source_dir

        if self.processing_entry_point is not None:
            processing_fields["processing_entry_point"] = self.processing_entry_point

        if self.processing_script_arguments is not None:
            processing_fields["processing_script_arguments"] = (
                self.processing_script_arguments
            )

        # Combine base fields and processing fields (processing fields take precedence if overlap)
        init_fields = {**base_fields, **processing_fields}

        return init_fields
