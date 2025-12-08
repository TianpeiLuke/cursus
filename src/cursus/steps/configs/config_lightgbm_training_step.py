"""
LightGBM Training Step Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker LightGBM Training steps
using a self-contained design where derived fields are private with read-only properties.
Fields are organized into three tiers:
1. Tier 1: Essential User Inputs - fields that users must explicitly provide
2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
"""

from pydantic import BaseModel, Field, model_validator, field_validator, PrivateAttr
from typing import List, Optional, Dict, Any, ClassVar
from pathlib import Path
import json
from datetime import datetime

from ...core.base.config_base import BasePipelineConfig


class LightGBMTrainingConfig(BasePipelineConfig):
    """
    Configuration specific to the SageMaker LightGBM Training Step.
    This version is streamlined to work with specification-driven architecture.
    Input/output paths are now provided via step specifications and dependencies.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    training_entry_point: str = Field(
        description="Entry point script for LightGBM training."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Instance configuration
    training_instance_type: str = Field(
        default="ml.m5.4xlarge", description="Instance type for LightGBM training job."
    )

    training_instance_count: int = Field(
        default=1, ge=1, description="Number of instances for LightGBM training job."
    )

    training_volume_size: int = Field(
        default=30, ge=1, description="Volume size (GB) for training instances."
    )

    # Framework versions for SageMaker PyTorch container (used for LightGBM)
    framework_version: str = Field(
        default="2.1.2",
        description="SageMaker PyTorch framework version for LightGBM container.",
    )

    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    ca_repository_arn: str = Field(
        default="arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi",
        description="CodeArtifact repository ARN for secure PyPI access. Only used when use_secure_pypi=True.",
    )

    # Override model_class to match hyperparameters
    model_class: str = Field(
        default="lightgbm", description="Model class identifier, set to LightGBM."
    )

    # Hyperparameters handling configuration
    skip_hyperparameters_s3_uri: bool = Field(
        default=True,
        description="Whether to skip hyperparameters_s3_uri channel during _get_inputs. "
        "If True (default), hyperparameters are loaded from script folder. "
        "If False, hyperparameters_s3_uri channel is created as TrainingInput.",
    )

    # Environment variables for preprocessing artifact control
    use_secure_pypi: bool = Field(
        default=True,
        description="Controls PyPI source for package installation. "
        "If True (default), uses secure CodeArtifact PyPI. "
        "If False, uses public PyPI.",
    )

    use_precomputed_imputation: bool = Field(
        default=False,
        description="Controls whether to use pre-computed imputation artifacts. "
        "If True, expects input data to be already imputed and loads impute_dict.pkl from model_artifacts_input, skipping inline computation. "
        "If False (default), computes imputation inline and transforms data.",
    )

    use_precomputed_risk_tables: bool = Field(
        default=False,
        description="Controls whether to use pre-computed risk table artifacts. "
        "If True, expects input data to be already risk-mapped and loads risk_table_map.pkl from model_artifacts_input, skipping inline computation. "
        "If False (default), computes risk tables inline and transforms data.",
    )

    use_precomputed_features: bool = Field(
        default=False,
        description="Controls whether to use pre-computed feature selection. "
        "If True, expects input data to be already feature-selected and loads selected_features.json from model_artifacts_input, skipping inline computation. "
        "If False (default), uses all features without selection.",
    )

    use_native_categorical: bool = Field(
        default=True,
        description="Controls categorical feature handling mode for LightGBM. "
        "If True (default), uses LightGBM native categorical features with DictionaryEncodingProcessor for integer encoding. "
        "If False, uses risk table mapping (XGBoost-style) for categorical features. "
        "Can be overridden at runtime via USE_NATIVE_CATEGORICAL environment variable.",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    _hyperparameter_file: Optional[str] = PrivateAttr(default=None)

    model_config = BasePipelineConfig.model_config

    # Public read-only properties for derived fields

    @property
    def hyperparameter_file(self) -> str:
        """Get hyperparameter file path."""
        if self._hyperparameter_file is None:
            self._hyperparameter_file = f"{self.pipeline_s3_loc}/hyperparameters/{self.region}_hyperparameters.json"
        return self._hyperparameter_file

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        # Add derived properties to output
        data["hyperparameter_file"] = self.hyperparameter_file
        return data

    # Initialize derived fields at creation time to avoid potential validation loops
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "LightGBMTrainingConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize training-specific derived fields
        self._hyperparameter_file = (
            f"{self.pipeline_s3_loc}/hyperparameters/{self.region}_hyperparameters.json"
        )

        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the LightGBM training script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add training-specific environment variables
        env_vars.update(
            {
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
                "USE_PRECOMPUTED_IMPUTATION": str(
                    self.use_precomputed_imputation
                ).lower(),
                "USE_PRECOMPUTED_RISK_TABLES": str(
                    self.use_precomputed_risk_tables
                ).lower(),
                "USE_PRECOMPUTED_FEATURES": str(self.use_precomputed_features).lower(),
                "USE_NATIVE_CATEGORICAL": str(self.use_native_categorical).lower(),
                "REGION": self.region,
            }
        )

        return env_vars

    @field_validator("training_instance_type")
    @classmethod
    def _validate_sagemaker_lightgbm_instance_type(cls, v: str) -> str:
        """Validate instance types suitable for LightGBM built-in algorithm."""
        # LightGBM works well on CPU instances - built-in algorithm is CPU-optimized
        valid_cpu_instances = [
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c5.large",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
            "ml.r5.large",  # Memory-optimized for large datasets
            "ml.r5.xlarge",
            "ml.r5.2xlarge",
            "ml.r5.4xlarge",
            "ml.r5.8xlarge",
            "ml.r5.12xlarge",
            "ml.r5.16xlarge",
            "ml.r5.24xlarge",
        ]
        if v not in valid_cpu_instances:
            raise ValueError(
                f"Invalid training instance type for LightGBM: {v}. "
                f"LightGBM built-in algorithm is CPU-optimized. "
                f"Must be one of: {', '.join(valid_cpu_instances)}"
            )
        return v

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include LightGBM training-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and LightGBM training-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (BasePipelineConfig)
        base_fields = super().get_public_init_fields()

        # Add LightGBM training-specific fields (Tier 1 and Tier 2)
        training_fields = {
            "training_entry_point": self.training_entry_point,
            "training_instance_type": self.training_instance_type,
            "training_instance_count": self.training_instance_count,
            "training_volume_size": self.training_volume_size,
            "framework_version": self.framework_version,
            "py_version": self.py_version,
            "ca_repository_arn": self.ca_repository_arn,
            "model_class": self.model_class,
            "skip_hyperparameters_s3_uri": self.skip_hyperparameters_s3_uri,
            "use_secure_pypi": self.use_secure_pypi,
            "use_precomputed_imputation": self.use_precomputed_imputation,
            "use_precomputed_risk_tables": self.use_precomputed_risk_tables,
            "use_precomputed_features": self.use_precomputed_features,
            "use_native_categorical": self.use_native_categorical,
        }

        # Combine base fields and training fields (training fields take precedence if overlap)
        init_fields = {**base_fields, **training_fields}

        return init_fields
