from pydantic import Field, model_validator, PrivateAttr
from typing import TYPE_CHECKING, Optional, Dict, Any
from pathlib import Path

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..contracts.package_contract import PACKAGE_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract


class PackageConfig(ProcessingStepConfigBase):
    """
    Configuration for a model packaging step.

    This configuration follows the three-tier field categorization:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that users can override
    3. Tier 3: Derived Fields - fields calculated from other fields, stored in private attributes
    """

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    processing_entry_point: str = Field(
        default="package.py", description="Entry point script for packaging."
    )

    # Update to Pydantic V2 style model_config
    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",  # Allow extra fields like __model_type__ and __model_module__ for type-aware serialization
    }

    @model_validator(mode="after")
    def validate_config(self) -> "PackageConfig":
        """
        Validate configuration and ensure defaults are set.

        This validator ensures that:
        1. Entry point is provided
        2. Script contract is available and valid
        3. Required input paths are defined in the script contract
        """
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("packaging step requires a processing_entry_point")

        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")

        if "model_input" not in contract.expected_input_paths:
            raise ValueError("Script contract missing required input path: model_input")

        if "inference_scripts_input" not in contract.expected_input_paths:
            raise ValueError(
                "Script contract missing required input path: inference_scripts_input"
            )

        return self

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The package script contract
        """
        return PACKAGE_CONTRACT

    # Removed get_script_path override - now inherits modernized version from ProcessingStepConfigBase
    # which includes hybrid resolution and comprehensive fallbacks
    # The contract fallback logic was deemed unnecessary since processing_entry_point has a default value
