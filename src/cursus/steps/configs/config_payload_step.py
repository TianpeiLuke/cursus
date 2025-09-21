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

    # Model registration fields
    model_owner: str = Field(description="Team ID of model owner")

    model_domain: str = Field(description="Domain for model registration")

    model_objective: str = Field(description="Objective of model registration")

    # Variable lists for input and output
    source_model_inference_output_variable_list: Dict[str, str] = Field(
        description="Dictionary of output variables and their types (NUMERIC or TEXT)"
    )

    source_model_inference_input_variable_list: Union[
        Dict[str, str], List[List[str]]
    ] = Field(
        description="Input variables and their types. Can be either:\n"
        "1. Dictionary: {'var1': 'NUMERIC', 'var2': 'TEXT'}\n"
        "2. List of pairs: [['var1', 'NUMERIC'], ['var2', 'TEXT']]"
    )

    # Performance metrics
    expected_tps: int = Field(ge=1, description="Expected transactions per second")

    max_latency_in_millisecond: int = Field(
        ge=100, le=10000, description="Maximum acceptable latency in milliseconds"
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Model framework settings
    framework: str = Field(
        default="xgboost", description="ML framework used for the model"
    )

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

    # Performance thresholds
    max_acceptable_error_rate: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Maximum acceptable error rate (0-1)"
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

    # Validators for inputs

    @field_validator("source_model_inference_input_variable_list")
    @classmethod
    def validate_input_variable_list(
        cls, v: Union[Dict[str, str], List[List[str]]]
    ) -> Union[Dict[str, str], List[List[str]]]:
        """
        Validate input variable list format with string types.

        Args:
            v: Either a dictionary of variable names to types,
               or a list of [variable_name, variable_type] pairs

        Returns:
            Original value if valid, without modification
        """
        if not v:  # If empty
            raise ValueError("Input variable list cannot be empty")

        # Handle dictionary format
        if isinstance(v, dict):
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Key must be string, got {type(key)} for key: {key}"
                    )

                # Check if string value is valid
                if not isinstance(value, str):
                    raise ValueError(f"Value must be string, got {type(value)}")

                if value.upper() not in cls._VALID_TYPES:
                    raise ValueError(f"Value must be 'NUMERIC' or 'TEXT', got: {value}")
            return v

        # Handle list format
        elif isinstance(v, list):
            for item in v:
                if not isinstance(item, list) or len(item) != 2:
                    raise ValueError(
                        "Each item must be a list of [variable_name, variable_type]"
                    )

                var_name, var_type = item
                if not isinstance(var_name, str):
                    raise ValueError(
                        f"Variable name must be string, got {type(var_name)}"
                    )

                if not isinstance(var_type, str):
                    raise ValueError(f"Type must be string, got {type(var_type)}")

                if var_type.upper() not in cls._VALID_TYPES:
                    raise ValueError(
                        f"Type must be 'NUMERIC' or 'TEXT', got: {var_type}"
                    )
            return v

        else:
            raise ValueError("Must be either a dictionary or a list of pairs")

    @field_validator("source_model_inference_output_variable_list")
    @classmethod
    def validate_output_variable_list(cls, v: Dict[str, str]) -> Dict[str, str]:
        """
        Validate output variable dictionary format with string types.

        Args:
            v: Dictionary mapping variable names to types

        Returns:
            Original dictionary if valid, without modification
        """
        if not v:  # If empty dictionary
            raise ValueError("Output variable list cannot be empty")

        for key, value in v.items():
            # Validate key is a string
            if not isinstance(key, str):
                raise ValueError(f"Key must be string, got {type(key)} for key: {key}")

            # Check if string value is valid
            if not isinstance(value, str):
                raise ValueError(f"Value must be string, got {type(value)}")

            if value.upper() not in cls._VALID_TYPES:
                raise ValueError(f"Value must be 'NUMERIC' or 'TEXT', got: {value}")

        return v

    # Model validators

    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "PayloadConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # No additional derived fields to initialize for PayloadConfig
        return self

    @model_validator(mode="after")
    def validate_special_fields(self) -> "PayloadConfig":
        """Validate special fields configuration if provided"""
        if not self.special_field_values:
            return self

        # Check if all special fields are in input variable list
        invalid_fields = []
        input_vars = self.source_model_inference_input_variable_list

        for field_name in self.special_field_values:
            if isinstance(input_vars, dict):
                if field_name not in input_vars:
                    invalid_fields.append(field_name)
                else:
                    field_type = input_vars[field_name]
                    if field_type.upper() != "TEXT":
                        raise ValueError(
                            f"Special field '{field_name}' must be of type TEXT, "
                            f"got {field_type}"
                        )
            else:  # List format
                field_found = False
                for var_name, var_type in input_vars:
                    if var_name == field_name:
                        field_found = True
                        if var_type.upper() != "TEXT":
                            raise ValueError(
                                f"Special field '{field_name}' must be of type TEXT, "
                                f"got {var_type}"
                            )
                        break
                if not field_found:
                    invalid_fields.append(field_name)

        if invalid_fields:
            raise ValueError(
                f"Special fields not found in input variable list: {invalid_fields}"
            )

        # No model_copy - just return self directly
        return self

    # Methods for payload generation and paths

    def get_effective_source_dir(self) -> Optional[str]:
        """Get the effective source directory"""
        return self.processing_source_dir or self.source_dir

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

    # Input/output variable helpers

    def get_normalized_input_variables(self) -> List[List[str]]:
        """
        Get input variables normalized to list format with string types.
        Compatible with format from create_model_variable_list.

        Returns:
            List of [name, type] pairs with string types
        """
        input_vars = self.source_model_inference_input_variable_list
        result = []

        if isinstance(input_vars, dict):
            # Convert dict to list format
            for name, var_type in input_vars.items():
                type_str = str(var_type).upper()
                result.append([name, type_str])
        else:
            # Already list format, just standardize types
            for name, var_type in input_vars:
                type_str = str(var_type).upper()
                result.append([name, type_str])

        return result

    def get_input_variables_as_dict(self) -> Dict[str, str]:
        """
        Get input variables as a dictionary mapping names to string types.
        Compatible with the second return value from create_model_variable_json.

        Returns:
            Dictionary mapping variable names to string types
        """
        input_vars = self.source_model_inference_input_variable_list
        result = {}

        # Already in dict format
        if isinstance(input_vars, dict):
            for name, var_type in input_vars.items():
                result[name] = str(var_type).upper()

        # Convert list format to dict format
        else:
            for name, var_type in input_vars:
                result[name] = str(var_type).upper()

        return result

    # Removed model_dump() method - it was redundant, just calling super().model_dump()
    # Using inherited implementation from base config
