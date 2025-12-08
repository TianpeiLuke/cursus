from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from ...core.base.hyperparameters_base import ModelHyperparameters
from ...core.base.config_base import BasePipelineConfig


class PyTorchTrainingConfig(BasePipelineConfig):
    """
    Configuration specific to the SageMaker PyTorch Training Step.
    This version is streamlined to work with specification-driven architecture.
    Input/output paths are now provided via step specifications and dependencies.
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    training_entry_point: str = Field(
        description="Entry point script for Pytorch training."
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Instance configuration
    training_instance_type: str = Field(
        default="ml.g5.12xlarge", description="Instance type for training job."
    )
    training_instance_count: int = Field(
        default=1, ge=1, description="Number of instances for training job."
    )
    training_volume_size: int = Field(
        default=30, ge=1, description="Volume size (GB) for training instances."
    )

    # Framework versions for SageMaker PyTorch container
    framework_version: str = Field(
        default="2.1.2", description="SageMaker PyTorch framework version."
    )
    py_version: str = Field(
        default="py310",
        description="Python version for the SageMaker PyTorch container.",
    )

    ca_repository_arn: str = Field(
        default="arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi",
        description="CodeArtifact repository ARN for secure PyPI access. Only used when use_secure_pypi=True.",
    )

    # Hyperparameters handling configuration
    skip_hyperparameters_s3_uri: bool = Field(
        default=True,
        description="Whether to skip hyperparameters_s3_uri channel during _get_inputs. "
        "If True (default), hyperparameters are loaded from script folder. "
        "If False, hyperparameters_s3_uri channel is created as TrainingInput.",
    )

    # Hyperparameters object (optional for backward compatibility)
    hyperparameters: Optional[ModelHyperparameters] = Field(
        None,
        description="Model hyperparameters (optional when using external JSON files)",
    )

    # Pre-computed artifact flags
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

    model_config = BasePipelineConfig.model_config

    @field_validator("training_instance_type")
    @classmethod
    def _validate_sagemaker_training_instance_type(cls, v: str) -> str:
        valid_instances = [
            "ml.m5.4xlarge",
            "ml.m5.8xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.g4dn.16xlarge",
            "ml.g5.12xlarge",
            "ml.g5.16xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
            "ml.p4d.24xlarge",
        ]
        if v not in valid_instances:
            raise ValueError(
                f"Invalid training instance type: {v}. "
                f"Must be one of: {', '.join(valid_instances)}"
            )
        return v

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the PyTorch training script.

        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        # Get base environment variables from parent class if available
        env_vars = (
            super().get_environment_variables()
            if hasattr(super(), "get_environment_variables")
            else {}
        )

        # Add PyTorch training specific environment variables
        env_vars.update(
            {
                "REGION": self.region,
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
                "USE_PRECOMPUTED_IMPUTATION": str(
                    self.use_precomputed_imputation
                ).lower(),
                "USE_PRECOMPUTED_RISK_TABLES": str(
                    self.use_precomputed_risk_tables
                ).lower(),
                "USE_PRECOMPUTED_FEATURES": str(self.use_precomputed_features).lower(),
            }
        )

        return env_vars

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include PyTorch training-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        Includes both base fields (from parent) and PyTorch training-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (BasePipelineConfig)
        base_fields = super().get_public_init_fields()

        # Add PyTorch training-specific fields (Tier 1 and Tier 2)
        training_fields = {
            "training_entry_point": self.training_entry_point,
            "training_instance_type": self.training_instance_type,
            "training_instance_count": self.training_instance_count,
            "training_volume_size": self.training_volume_size,
            "framework_version": self.framework_version,
            "py_version": self.py_version,
            "ca_repository_arn": self.ca_repository_arn,
            "skip_hyperparameters_s3_uri": self.skip_hyperparameters_s3_uri,
            "use_precomputed_imputation": self.use_precomputed_imputation,
            "use_precomputed_risk_tables": self.use_precomputed_risk_tables,
            "use_precomputed_features": self.use_precomputed_features,
        }

        # Add hyperparameters if present (use model_dump for Pydantic models)
        if self.hyperparameters is not None:
            training_fields["hyperparameters"] = self.hyperparameters.model_dump()

        # Combine base fields and training fields (training fields take precedence if overlap)
        init_fields = {**base_fields, **training_fields}

        return init_fields
