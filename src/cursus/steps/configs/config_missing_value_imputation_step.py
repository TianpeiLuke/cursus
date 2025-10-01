"""
Missing Value Imputation Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for missing value imputation, using a self-contained design where each field
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
from ..contracts.missing_value_imputation_contract import MISSING_VALUE_IMPUTATION_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class MissingValueImputationConfig(ProcessingStepConfigBase):
    """
    Configuration for the Missing Value Imputation step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    label_field: str = Field(
        description="Target column name to exclude from imputation (e.g., 'target', 'label', 'y')."
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="missing_value_imputation.py",
        description="Relative path (within processing_source_dir) to the missing value imputation script.",
    )

    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    # Imputation strategy defaults
    default_numerical_strategy: str = Field(
        default="mean",
        description="Default imputation strategy for numerical columns: 'mean', 'median', 'constant'",
    )

    default_categorical_strategy: str = Field(
        default="mode",
        description="Default imputation strategy for categorical columns: 'mode', 'constant'",
    )

    default_text_strategy: str = Field(
        default="mode",
        description="Default imputation strategy for text/string columns: 'mode', 'constant', 'empty'",
    )

    # Constant fill values
    numerical_constant_value: float = Field(
        default=0.0,
        description="Constant value for numerical imputation when using 'constant' strategy",
    )

    categorical_constant_value: str = Field(
        default="Unknown",
        description="Constant value for categorical imputation when using 'constant' strategy",
    )

    text_constant_value: str = Field(
        default="Unknown",
        description="Constant value for text imputation when using 'constant' strategy",
    )

    # Advanced configuration options
    categorical_preserve_dtype: bool = Field(
        default=True,
        description="Whether to preserve pandas categorical dtype during imputation",
    )

    auto_detect_categorical: bool = Field(
        default=True,
        description="Enable automatic categorical vs text detection based on unique value ratios",
    )

    categorical_unique_ratio_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Threshold for categorical detection (unique values / total values)",
    )

    validate_fill_values: bool = Field(
        default=True,
        description="Enable pandas NA value validation to avoid problematic fill values",
    )

    exclude_columns: Optional[List[str]] = Field(
        default=None,
        description="List of column names to exclude from imputation (in addition to label_field)",
    )

    column_strategies: Optional[Dict[str, str]] = Field(
        default=None,
        description="Column-specific imputation strategies (column_name -> strategy)",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _effective_exclude_columns: Optional[List[str]] = PrivateAttr(default=None)

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====

    @property
    def environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the imputation script.

        Returns:
            Dictionary of environment variables
        """
        if self._environment_variables is None:
            env_vars = {
                "LABEL_FIELD": self.label_field,
                "DEFAULT_NUMERICAL_STRATEGY": self.default_numerical_strategy,
                "DEFAULT_CATEGORICAL_STRATEGY": self.default_categorical_strategy,
                "DEFAULT_TEXT_STRATEGY": self.default_text_strategy,
                "NUMERICAL_CONSTANT_VALUE": str(self.numerical_constant_value),
                "CATEGORICAL_CONSTANT_VALUE": self.categorical_constant_value,
                "TEXT_CONSTANT_VALUE": self.text_constant_value,
                "CATEGORICAL_PRESERVE_DTYPE": str(self.categorical_preserve_dtype).lower(),
                "AUTO_DETECT_CATEGORICAL": str(self.auto_detect_categorical).lower(),
                "CATEGORICAL_UNIQUE_RATIO_THRESHOLD": str(self.categorical_unique_ratio_threshold),
                "VALIDATE_FILL_VALUES": str(self.validate_fill_values).lower(),
            }

            # Add exclude columns if specified
            if self.effective_exclude_columns:
                env_vars["EXCLUDE_COLUMNS"] = ",".join(self.effective_exclude_columns)
            else:
                env_vars["EXCLUDE_COLUMNS"] = ""

            # Add column-specific strategies if specified
            if self.column_strategies:
                for column, strategy in self.column_strategies.items():
                    env_var_name = f"COLUMN_STRATEGY_{column.upper()}"
                    env_vars[env_var_name] = strategy

            self._environment_variables = env_vars

        return self._environment_variables

    @property
    def effective_exclude_columns(self) -> List[str]:
        """
        Get effective list of columns to exclude from imputation.
        Combines label_field with user-specified exclude_columns.

        Returns:
            List of column names to exclude
        """
        if self._effective_exclude_columns is None:
            exclude_list = []
            
            # Always exclude label field
            if self.label_field:
                exclude_list.append(self.label_field)
            
            # Add user-specified exclude columns
            if self.exclude_columns:
                exclude_list.extend(self.exclude_columns)
            
            # Remove duplicates while preserving order
            seen = set()
            self._effective_exclude_columns = []
            for col in exclude_list:
                if col not in seen:
                    seen.add(col)
                    self._effective_exclude_columns.append(col)

        return self._effective_exclude_columns

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

    @field_validator("default_numerical_strategy")
    @classmethod
    def validate_numerical_strategy(cls, v: str) -> str:
        """
        Ensure default_numerical_strategy is one of the allowed values.
        """
        allowed = {"mean", "median", "constant"}
        if v not in allowed:
            raise ValueError(f"default_numerical_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("default_categorical_strategy")
    @classmethod
    def validate_categorical_strategy(cls, v: str) -> str:
        """
        Ensure default_categorical_strategy is one of the allowed values.
        """
        allowed = {"mode", "constant"}
        if v not in allowed:
            raise ValueError(f"default_categorical_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("default_text_strategy")
    @classmethod
    def validate_text_strategy(cls, v: str) -> str:
        """
        Ensure default_text_strategy is one of the allowed values.
        """
        allowed = {"mode", "constant", "empty"}
        if v not in allowed:
            raise ValueError(f"default_text_strategy must be one of {allowed}, got '{v}'")
        return v

    @field_validator("categorical_constant_value", "text_constant_value")
    @classmethod
    def validate_text_fill_values(cls, v: str) -> str:
        """
        Validate that text fill values are pandas-safe.
        """
        # Common pandas NA values to avoid
        pandas_na_values = {
            "N/A", "NA", "NULL", "NaN", "nan", "NAN",
            "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", 
            "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>",
            "null", "Null", "none", "None", "NONE"
        }
        
        if v in pandas_na_values:
            logger.warning(
                f"Fill value '{v}' may be interpreted as NA by pandas. "
                f"Consider using 'Unknown', 'Missing', or 'MISSING_VALUE' instead."
            )
        
        return v

    @field_validator("exclude_columns")
    @classmethod
    def validate_exclude_columns(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """
        Ensure exclude_columns contains non-empty strings if provided.
        """
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("exclude_columns must be a list of strings")
            
            validated_columns = []
            for col in v:
                if not isinstance(col, str) or not col.strip():
                    raise ValueError("All exclude_columns must be non-empty strings")
                validated_columns.append(col.strip())
            
            return validated_columns
        
        return v

    @field_validator("column_strategies")
    @classmethod
    def validate_column_strategies(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """
        Validate column-specific strategies.
        """
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("column_strategies must be a dictionary")
            
            valid_strategies = {"mean", "median", "constant", "mode", "empty"}
            validated_strategies = {}
            
            for column, strategy in v.items():
                if not isinstance(column, str) or not column.strip():
                    raise ValueError("All column names in column_strategies must be non-empty strings")
                
                if not isinstance(strategy, str) or strategy not in valid_strategies:
                    raise ValueError(
                        f"Strategy '{strategy}' for column '{column}' must be one of {valid_strategies}"
                    )
                
                validated_strategies[column.strip()] = strategy
            
            return validated_strategies
        
        return v

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "MissingValueImputationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize effective exclude columns
        _ = self.effective_exclude_columns

        # Initialize environment variables
        _ = self.environment_variables

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The missing value imputation script contract
        """
        return MISSING_VALUE_IMPUTATION_CONTRACT

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include missing value imputation specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add missing value imputation specific fields
        imputation_fields = {
            "label_field": self.label_field,
            "processing_entry_point": self.processing_entry_point,
            "job_type": self.job_type,
            "default_numerical_strategy": self.default_numerical_strategy,
            "default_categorical_strategy": self.default_categorical_strategy,
            "default_text_strategy": self.default_text_strategy,
            "numerical_constant_value": self.numerical_constant_value,
            "categorical_constant_value": self.categorical_constant_value,
            "text_constant_value": self.text_constant_value,
            "categorical_preserve_dtype": self.categorical_preserve_dtype,
            "auto_detect_categorical": self.auto_detect_categorical,
            "categorical_unique_ratio_threshold": self.categorical_unique_ratio_threshold,
            "validate_fill_values": self.validate_fill_values,
        }

        # Only include optional fields if they're set
        if self.exclude_columns is not None:
            imputation_fields["exclude_columns"] = self.exclude_columns

        if self.column_strategies is not None:
            imputation_fields["column_strategies"] = self.column_strategies

        # Combine fields (imputation fields take precedence if overlap)
        init_fields = {**base_fields, **imputation_fields}

        return init_fields

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)

        # Add derived properties
        data["environment_variables"] = self.environment_variables
        data["effective_exclude_columns"] = self.effective_exclude_columns

        return data
