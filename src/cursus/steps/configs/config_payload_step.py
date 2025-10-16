from pydantic import BaseModel, Field, model_validator, field_validator, PrivateAttr, ConfigDict, field_serializer
from typing import Optional, Dict, List, Any, Union, TYPE_CHECKING, ClassVar
from pathlib import Path
from datetime import datetime
from enum import Enum

import json
import logging

logger = logging.getLogger(__name__)

from .config_processing_step_base import ProcessingStepConfigBase
from .config_registration_step import VariableType

# Import the script contract
from ..contracts.payload_contract import PAYLOAD_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ...core.base.contract_base import ScriptContract


class PayloadConfig(ProcessingStepConfigBase):
    """
    Configuration for payload generation and testing.

    This configuration follows the three-tier field categorization:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that users can override
    3. Tier 3: Derived Fields - fields calculated from other fields, stored in private attributes
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide
    
    # NOTE: Variable lists removed - script gets all variable information from hyperparameters.json
    # The script extracts field information from hyperparameters using:
    # - full_field_list, tab_field_list, cat_field_list from hyperparameters
    # - Creates variable list dynamically using create_model_variable_list()
    # Config-based variable lists were redundant and unused.

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Entry point script
    processing_entry_point: str = Field(
        default="payload.py", description="Entry point script for payload generation"
    )

    # Content and response types
    source_model_inference_content_types: List[str] = Field(
        default=["text/csv"],
        description="Content type for model inference input. Must be exactly ['text/csv'] or ['application/json']",
    )

    source_model_inference_response_types: List[str] = Field(
        default=["application/json"],
        description="Response type for model inference output. Must be exactly ['text/csv'] or ['application/json']",
    )

    # Default values for payload generation
    default_numeric_value: float = Field(
        default=0.0, description="Default value for numeric fields"
    )

    default_text_value: str = Field(
        default="DEFAULT_TEXT", description="Default value for text fields"
    )

    # Special field values dictionary
    special_field_values: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary of special TEXT fields and their template values",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes

    # Valid types for validation
    _VALID_TYPES: ClassVar[List[str]] = ["NUMERIC", "TEXT"]

    # Update to Pydantic V2 style model_config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,  # Changed from True to False to prevent recursion
        extra="allow",  # Changed from 'forbid' to 'allow' to accept metadata fields during deserialization
    )

    # Custom serializer for Path fields (Pydantic V2 approach)
    @field_serializer('processing_source_dir', 'source_dir', when_used='json')
    def serialize_path_fields(self, value: Optional[Union[str, Path]]) -> Optional[str]:
        """Serialize Path objects to strings"""
        if value is None:
            return None
        return str(value)

    # Removed sample_payload_s3_key property - S3 path construction should happen in builders/scripts

    # Removed validators for variable lists since those fields were removed
    # Script gets all variable information from hyperparameters.json

    # Model validators

    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "PayloadConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for PayloadConfig
        return self

    # Removed validate_special_fields validator since it referenced the removed variable list fields
    # Special field validation is no longer needed since the script gets variables from hyperparameters
    # and special_field_values is optional with proper defaults

    # Methods for payload generation and paths


    # Removed ensure_payload_path() and get_full_payload_path() methods
    # These are redundant and not portable - S3 path construction should happen in builders/scripts

    # Removed get_field_default_value() method - this is processing logic that belongs in the script
    # Config should only provide the default values, not compute them

    # Removed payload generation and processing methods - these belong in the script, not config
    # Config should only handle user input and configuration, not actual processing logic

    # Script and contract handling

    def get_script_contract(self) -> "ScriptContract":
        """
        Get script contract for this configuration.

        Returns:
            The payload script contract
        """
        return PAYLOAD_CONTRACT

    # Removed get_script_path() method - using inherited implementation from base config
    # The base config's implementation handles script path resolution properly

    # Removed redundant input/output variable helper methods:
    # - get_normalized_input_variables() - duplicates script logic
    # - get_input_variables_as_dict() - duplicates script logic
    # These methods duplicate processing logic that belongs in the script, not config.
    # Config should only provide the raw data, not process it.

    # Removed model_dump() method - it was redundant, just calling super().model_dump()
    # Using inherited implementation from base config
