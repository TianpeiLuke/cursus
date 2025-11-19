"""
Feature Selection Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for feature selection, using a self-contained design where each field
is properly categorized according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import contract
from ..contracts.feature_selection_contract import FEATURE_SELECTION_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class FeatureSelectionConfig(ProcessingStepConfigBase):
    """
    Configuration for the Feature Selection step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    label_field: str = Field(
        description="Target column name for feature selection (e.g., 'target', 'label', 'y')."
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="feature_selection.py",
        description="Relative path (within processing_source_dir) to the feature selection script.",
    )

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    # Feature selection method configuration
    feature_selection_methods: str = Field(
        default="variance,correlation,mutual_info,rfe",
        description="Comma-separated list of feature selection methods to apply",
    )

    n_features_to_select: int = Field(
        default=10,
        ge=1,
        description="Number of features to select in final ensemble",
    )

    correlation_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Threshold for removing highly correlated features",
    )

    variance_threshold: float = Field(
        default=0.01,
        ge=0.0,
        description="Threshold for removing low-variance features",
    )

    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    combination_strategy: str = Field(
        default="voting",
        description="Strategy for combining method results: 'voting', 'ranking', 'scoring'",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _method_list: Optional[List[str]] = PrivateAttr(default=None)

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====

    @property
    def environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the feature selection script.

        Returns:
            Dictionary of environment variables
        """
        if self._environment_variables is None:
            env_vars = {
                "LABEL_FIELD": self.label_field,
                "FEATURE_SELECTION_METHODS": self.feature_selection_methods,
                "N_FEATURES_TO_SELECT": str(self.n_features_to_select),
                "CORRELATION_THRESHOLD": str(self.correlation_threshold),
                "VARIANCE_THRESHOLD": str(self.variance_threshold),
                "RANDOM_STATE": str(self.random_state),
                "COMBINATION_STRATEGY": self.combination_strategy,
                "USE_SECURE_PYPI": str(self.use_secure_pypi).lower(),
            }

            self._environment_variables = env_vars

        return self._environment_variables

    @property
    def method_list(self) -> List[str]:
        """
        Get list of feature selection methods from comma-separated string.

        Returns:
            List of method names
        """
        if self._method_list is None:
            methods = [
                method.strip() for method in self.feature_selection_methods.split(",")
            ]
            # Filter out empty strings
            self._method_list = [method for method in methods if method]

        return self._method_list

    # ===== Validators =====

    @field_validator("label_field")
    @classmethod
    def validate_label_field(cls, v: str) -> str:
        """
        Ensure label_field is a non-empty string.
        """
        if not v or not v.strip():
            raise ValueError("label_field must be a non-empty string")
        return v.strip()

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

    @field_validator("feature_selection_methods")
    @classmethod
    def validate_feature_selection_methods(cls, v: str) -> str:
        """
        Validate feature selection methods string.
        """
        if not v or not v.strip():
            raise ValueError("feature_selection_methods must be a non-empty string")

        # Parse methods and validate each one
        methods = [method.strip() for method in v.split(",")]
        valid_methods = {
            "variance",
            "correlation",
            "mutual_info",
            "chi2",
            "f_test",
            "rfe",
            "importance",
            "lasso",
            "permutation",
        }

        for method in methods:
            if method and method not in valid_methods:
                raise ValueError(
                    f"Invalid feature selection method '{method}'. "
                    f"Valid methods are: {', '.join(sorted(valid_methods))}"
                )

        # Filter out empty methods and rejoin
        valid_methods_list = [method for method in methods if method]
        if not valid_methods_list:
            raise ValueError(
                "At least one valid feature selection method must be specified"
            )

        return ",".join(valid_methods_list)

    @field_validator("combination_strategy")
    @classmethod
    def validate_combination_strategy(cls, v: str) -> str:
        """
        Ensure combination_strategy is one of the allowed values.
        """
        allowed = {"voting", "ranking", "scoring"}
        if v not in allowed:
            raise ValueError(
                f"combination_strategy must be one of {allowed}, got '{v}'"
            )
        return v

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "FeatureSelectionConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize method list
        _ = self.method_list

        # Initialize environment variables
        _ = self.environment_variables

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The feature selection script contract
        """
        return FEATURE_SELECTION_CONTRACT

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include feature selection specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add feature selection specific fields
        feature_selection_fields = {
            "label_field": self.label_field,
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "feature_selection_methods": self.feature_selection_methods,
            "n_features_to_select": self.n_features_to_select,
            "correlation_threshold": self.correlation_threshold,
            "variance_threshold": self.variance_threshold,
            "random_state": self.random_state,
            "combination_strategy": self.combination_strategy,
        }

        # Combine fields (feature selection fields take precedence if overlap)
        init_fields = {**base_fields, **feature_selection_fields}

        return init_fields

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)

        # Add derived properties
        data["environment_variables"] = self.environment_variables
        data["method_list"] = self.method_list

        return data
