from pydantic import Field, field_validator, model_validator
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
import json
import logging

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.currency_conversion_contract import CURRENCY_CONVERSION_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract

logger = logging.getLogger(__name__)


class CurrencyConversionConfig(ProcessingStepConfigBase):
    """
    Configuration for currency conversion processing step.

    This configuration follows the specification-driven approach where inputs and outputs
    are defined by step specifications and script contracts, not by hardcoded dictionaries.
    """

    # Job type configuration
    job_type: str = Field(
        default="training",
        description="One of ['training','validation','testing','calibration']",
    )

    # Processing entry point
    processing_entry_point: str = Field(
        default="currency_conversion.py",
        description="Entry point script for currency conversion.",
    )

    # Instance sizing
    use_large_processing_instance: bool = Field(
        default=False, description="Whether to use large instance type."
    )

    # Currency conversion mode
    mode: str = Field(
        default="per_split", description="One of ['per_split','split_after_conversion']"
    )

    # Split ratios (used when mode is split_after_conversion)
    train_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Train fraction when split_after_conversion",
    )
    test_val_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Test vs val split within holdout"
    )
    label_field: str = Field(
        ..., description="Label column name for stratified splitting"
    )

    # Currency conversion parameters
    marketplace_id_col: str = Field(..., description="Column with marketplace IDs")
    currency_col: Optional[str] = Field(
        default=None,
        description="Optional column with currency codes; else infer from marketplace_info",
    )
    currency_conversion_var_list: List[str] = Field(
        default_factory=list, description="Which numeric columns to convert"
    )
    currency_conversion_dict: Dict[str, float] = Field(
        ..., description="Map currency code → conversion rate"
    )
    marketplace_info: Dict[str, Dict[str, str]] = Field(
        ..., description="Map marketplace ID → {'currency_code':...}"
    )
    enable_currency_conversion: bool = Field(
        default=True, description="Turn off conversion if False"
    )
    default_currency: str = Field(default="USD", description="Fallback currency code")
    skip_invalid_currencies: bool = Field(
        default=False, description="If True, fill invalid codes with default_currency"
    )

    model_config = ProcessingStepConfigBase.model_config.copy()
    model_config.update({
        'arbitrary_types_allowed': True,
        'validate_assignment': True
    })

    @field_validator("job_type")
    @classmethod
    def _validate_job_type(cls, v: str) -> str:
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got {v!r}")
        return v

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        allowed = {"per_split", "split_after_conversion"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator("currency_conversion_dict")
    @classmethod
    def _validate_dict(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("currency_conversion_dict cannot be empty")
        if 1.0 not in v.values():
            raise ValueError("currency_conversion_dict must include a rate of 1.0")
        for k, rate in v.items():
            if rate <= 0:
                raise ValueError(f"Rate for {k} must be positive; got {rate}")
        return v

    @field_validator("currency_conversion_var_list")
    @classmethod
    def _validate_vars(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            dup = [x for x in v if v.count(x) > 1]
            raise ValueError(f"Duplicate vars in currency_conversion_var_list: {dup}")
        return v

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """Ensure processing_entry_point is a non‐empty relative path."""
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non‐empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError(
                "processing_entry_point must be a relative path within source directory"
            )
        return v

    @model_validator(mode="after")
    def validate_config(self) -> "CurrencyConversionConfig":
        """Validate currency conversion configuration."""
        if self.enable_currency_conversion:
            if not self.marketplace_id_col:
                raise ValueError("marketplace_id_col required when conversion enabled")
            if not self.currency_conversion_var_list:
                raise ValueError("currency_conversion_var_list cannot be empty")
            if not self.marketplace_info:
                raise ValueError("marketplace_info must be provided")
            if self.mode == "split_after_conversion":
                if not self.label_field:
                    raise ValueError("label_field required for split_after_conversion")

        # Validate that required environment variables from the contract have values
        contract = self.get_script_contract()
        if contract and contract.required_env_vars:
            for env_var in contract.required_env_vars:
                if (
                    env_var == "CURRENCY_CONVERSION_VARS"
                    and not self.currency_conversion_var_list
                ):
                    raise ValueError(
                        "currency_conversion_var_list is required by the script contract"
                    )
                elif (
                    env_var == "CURRENCY_CONVERSION_DICT"
                    and not self.currency_conversion_dict
                ):
                    raise ValueError(
                        "currency_conversion_dict is required by the script contract"
                    )
                elif env_var == "MARKETPLACE_INFO" and not self.marketplace_info:
                    raise ValueError(
                        "marketplace_info is required by the script contract"
                    )
                elif env_var == "LABEL_FIELD" and not self.label_field:
                    raise ValueError("label_field is required by the script contract")

        return self

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The currency conversion script contract
        """
        return CURRENCY_CONVERSION_CONTRACT

    # Removed get_script_path override - now inherits modernized version from ProcessingStepConfigBase
    # which includes hybrid resolution and comprehensive fallbacks
