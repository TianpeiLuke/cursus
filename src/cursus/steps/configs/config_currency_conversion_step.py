"""
Currency Conversion Configuration with Self-Contained Derivation Logic

This module implements the configuration class for SageMaker Processing steps
for currency conversion, using a self-contained design where each field
is properly categorized according to the three-tier design:
1. Essential User Inputs (Tier 1) - Required fields that must be provided by users
2. System Fields (Tier 2) - Fields with reasonable defaults that can be overridden
3. Derived Fields (Tier 3) - Fields calculated from other fields, private with read-only properties
"""

from pydantic import BaseModel, Field, field_validator, model_validator, PrivateAttr
from typing import Dict, Optional, Any, List, TYPE_CHECKING
from pathlib import Path
import json
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import contract
from ..contracts.currency_conversion_contract import CURRENCY_CONVERSION_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class CurrencyConversionMappingConfig(BaseModel):
    """
    Single currency conversion mapping entry.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    marketplace_id: str = Field(description="Marketplace ID")
    currency_code: str = Field(description="Currency code (e.g., 'USD', 'EUR')")
    conversion_rate: float = Field(description="Conversion rate to default currency")

    # ===== System Inputs with Defaults (Tier 2) =====
    # None currently for this class

    # ===== Derived Fields (Tier 3) =====
    # None currently for this class

    # Validators
    @field_validator("conversion_rate")
    @classmethod
    def validate_rate(cls, v: float) -> float:
        """Ensure conversion rate is positive."""
        if v <= 0:
            raise ValueError("conversion_rate must be positive")
        return v


class CurrencyConversionDictConfig(BaseModel):
    """
    Currency conversion dictionary with mappings.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    mappings: List[CurrencyConversionMappingConfig] = Field(
        description="List of marketplace to currency mappings"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # None currently for this class

    # ===== Derived Fields (Tier 3) =====
    # None currently for this class

    # Validators
    @field_validator("mappings")
    @classmethod
    def validate_mappings(cls, v: List[CurrencyConversionMappingConfig]):
        """Validate mappings list."""
        if not v:
            raise ValueError("mappings list cannot be empty")
        # Check that at least one rate is 1.0 (the default currency)
        if not any(m.conversion_rate == 1.0 for m in v):
            raise ValueError("At least one mapping must have conversion_rate of 1.0")
        return v


class CurrencyConversionConfig(ProcessingStepConfigBase):
    """
    Configuration for the Currency Conversion step with three-tier field categorization.
    Inherits from ProcessingStepConfigBase.

    Fields are categorized into:
    - Tier 1: Essential User Inputs - Required from users
    - Tier 2: System Fields - Default values that can be overridden
    - Tier 3: Derived Fields - Private with read-only property access

    This configuration follows the specification-driven approach where inputs and outputs
    are defined by step specifications and script contracts, not by hardcoded dictionaries.
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    currency_conversion_vars: List[str] = Field(
        description="List of monetary columns to convert"
    )

    currency_conversion_dict: CurrencyConversionDictConfig = Field(
        description="Currency conversion mappings"
    )

    # ===== System Fields with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="currency_conversion.py",
        description="Entry point script for currency conversion",
    )

    currency_code_field: Optional[str] = Field(
        default=None, description="Name of column containing currency codes directly"
    )

    marketplace_id_field: Optional[str] = Field(
        default=None, description="Name of column containing marketplace IDs"
    )

    default_currency: str = Field(default="USD", description="Default currency code")

    n_workers: int = Field(default=50, ge=1, description="Number of parallel workers")

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields
    # They are private with public read-only property access

    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({"arbitrary_types_allowed": True, "validate_assignment": True})

    # ===== Properties for Derived Fields =====

    @property
    def environment_variables(self) -> Dict[str, str]:
        """
        Generate environment variables for the currency conversion script.

        Returns:
            Dictionary of environment variables
        """
        if self._environment_variables is None:
            env_vars = {
                "CURRENCY_CONVERSION_VARS": json.dumps(self.currency_conversion_vars),
                "CURRENCY_CONVERSION_DICT": json.dumps(
                    self.currency_conversion_dict.model_dump()
                ),
                "CURRENCY_CODE_FIELD": self.currency_code_field or "",
                "MARKETPLACE_ID_FIELD": self.marketplace_id_field or "",
                "DEFAULT_CURRENCY": self.default_currency,
                "N_WORKERS": str(self.n_workers),
            }
            self._environment_variables = env_vars

        return self._environment_variables

    # ===== Validators =====

    @field_validator("currency_conversion_vars")
    @classmethod
    def validate_vars(cls, v: List[str]) -> List[str]:
        """
        Validate currency conversion variables list.
        """
        if not v:
            raise ValueError("currency_conversion_vars cannot be empty")
        if len(v) != len(set(v)):
            dup = [x for x in v if v.count(x) > 1]
            raise ValueError(f"Duplicate vars in currency_conversion_vars: {dup}")
        return v

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

    @model_validator(mode="after")
    def validate_currency_fields(self) -> "CurrencyConversionConfig":
        """
        Ensure at least one of currency_code_field or marketplace_id_field is not None.
        """
        if self.currency_code_field is None and self.marketplace_id_field is None:
            raise ValueError(
                "At least one of 'currency_code_field' or 'marketplace_id_field' must be provided (cannot be None)"
            )
        return self

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "CurrencyConversionConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize environment variables
        _ = self.environment_variables

        return self

    # ===== Script Contract =====

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The currency conversion script contract
        """
        return CURRENCY_CONVERSION_CONTRACT

    # ===== Overrides for Inheritance =====

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include currency conversion specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add currency conversion specific fields
        currency_conversion_fields = {
            "currency_conversion_vars": self.currency_conversion_vars,
            "currency_conversion_dict": self.currency_conversion_dict,
            "processing_entry_point": self.processing_entry_point,
            "currency_code_field": self.currency_code_field,
            "marketplace_id_field": self.marketplace_id_field,
            "default_currency": self.default_currency,
            "n_workers": self.n_workers,
        }

        # Combine fields (currency conversion fields take precedence if overlap)
        init_fields = {**base_fields, **currency_conversion_fields}

        return init_fields

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        # Get base fields first
        data = super().model_dump(**kwargs)

        # Add derived properties
        data["environment_variables"] = self.environment_variables

        return data
