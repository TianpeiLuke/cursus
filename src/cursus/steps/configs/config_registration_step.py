from pydantic import BaseModel, Field, model_validator, field_validator, PrivateAttr, ConfigDict, field_serializer
from typing import Union, Optional, Dict, List, Any, ClassVar, Tuple
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging

from ...core.base.config_base import BasePipelineConfig

logger = logging.getLogger(__name__)


class VariableType(str, Enum):
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"

    @classmethod
    def _missing_(cls, value: str) -> Optional["VariableType"]:
        """Handle string values"""
        try:
            return cls(value.upper())
        except ValueError:
            return None

    def __str__(self) -> str:
        """String representation"""
        return self.value


def create_inference_variable_list(
    numeric_fields: List[str] = None,
    text_fields: List[str] = None,
    output_format: str = "dict",
) -> Union[Dict[str, Union[VariableType, str]], List[List[str]]]:
    """
    Create an inference variable list for model input variables using separate lists for numeric and text fields.
    This is a helper function that can be used standalone or within RegistrationConfig.

    Args:
        numeric_fields: List of field names that should be treated as NUMERIC
        text_fields: List of field names that should be treated as TEXT
        output_format: Format for storing variable list - either 'dict' or 'list'

    Returns:
        A dictionary mapping variable names to their types, or a list of [name, type] pairs,
        depending on the output_format parameter
    """
    # Initialize with empty lists if not provided
    numeric_fields = numeric_fields or []
    text_fields = text_fields or []

    # Validate inputs are lists of strings
    for field_name in numeric_fields:
        if not isinstance(field_name, str):
            raise ValueError(
                f"Field name must be string, got {type(field_name)} for: {field_name}"
            )

    for field_name in text_fields:
        if not isinstance(field_name, str):
            raise ValueError(
                f"Field name must be string, got {type(field_name)} for: {field_name}"
            )

    # Check for duplicates between numeric and text fields
    common_fields = set(numeric_fields) & set(text_fields)
    if common_fields:
        raise ValueError(f"Fields cannot be both numeric and text: {common_fields}")

    # Validate output format
    if output_format not in ["dict", "list"]:
        raise ValueError(
            f"Output format must be 'dict' or 'list', got: {output_format}"
        )

    # Create the variable list in the requested format
    if output_format == "dict":
        # Dictionary format - map field names to their types
        result = {}

        # Add numeric fields
        for field_name in numeric_fields:
            result[field_name] = VariableType.NUMERIC

        # Add text fields
        for field_name in text_fields:
            result[field_name] = VariableType.TEXT

    else:  # 'list' format
        # List format - create list of [name, type] pairs
        result = []

        # Add numeric fields
        for field_name in numeric_fields:
            result.append([field_name, VariableType.NUMERIC.value])

        # Add text fields
        for field_name in text_fields:
            result.append([field_name, VariableType.TEXT.value])

    return result


class RegistrationConfig(BasePipelineConfig):
    """
    Configuration for model registration step, following the three-tier categorization:

    Tier 1: Essential User Inputs - fields that users must explicitly provide
    Tier 2: System Inputs - fields with reasonable defaults that users can override
    Tier 3: Derived Fields - private fields with read-only property access
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    model_owner: str = Field(description="Team ID of model owner")

    model_domain: str = Field(description="Domain for model registration")

    model_objective: str = Field(description="Objective of model registration")

    framework: str = Field(description="ML framework used for the model")

    inference_entry_point: str = Field(description="Entry point script for inference")

    source_model_inference_input_variable_list: Union[
        Dict[str, Union[VariableType, str]], List[List[str]]
    ] = Field(
        default_factory=dict,
        description="Input variables and their types. Can be either:\n"
        "1. Dictionary: {'var1': 'NUMERIC', 'var2': 'TEXT'}\n"
        "2. List of pairs: [['var1', 'NUMERIC'], ['var2', 'TEXT']]",
    )

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    inference_instance_type: str = Field(
        default="ml.m5.large",
        description="Instance type for inference endpoint/transform job",
    )

    source_model_inference_content_types: List[str] = Field(
        default=["text/csv"],
        description="Content type for model inference input. Must be exactly ['text/csv'] or ['application/json']",
    )

    source_model_inference_response_types: List[str] = Field(
        default=["application/json"],
        description="Response type for model inference output. Must be exactly ['text/csv'] or ['application/json']",
    )

    source_model_inference_output_variable_list: Dict[str, VariableType] = Field(
        default={"legacy-score": VariableType.NUMERIC},
        description="Dictionary of output variables and their types (NUMERIC or TEXT)",
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields, stored in private attributes
    # with public read-only properties for access

    # Removed _source_model_inference_input_variable_list as it's now a field
    _variable_schema: Optional[Dict[str, Dict[str, List[Dict[str, str]]]]] = (
        PrivateAttr(default=None)
    )

    # Update to Pydantic V2 style model_config
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="allow",  # Accept metadata fields during deserialization
    )

    # Custom serializer for VariableType fields (Pydantic V2 approach)
    @field_serializer('source_model_inference_output_variable_list')
    def serialize_output_variable_list(self, value: Dict[str, VariableType]) -> Dict[str, str]:
        """Serialize VariableType enum values to strings"""
        return {k: v.value if isinstance(v, VariableType) else v for k, v in value.items()}

    @field_serializer('source_model_inference_input_variable_list')
    def serialize_input_variable_list(self, value: Union[Dict[str, Union[VariableType, str]], List[List[str]]]) -> Union[Dict[str, str], List[List[str]]]:
        """Serialize VariableType enum values to strings in input variable list"""
        if isinstance(value, dict):
            return {k: v.value if isinstance(v, VariableType) else v for k, v in value.items()}
        return value  # List format already uses string values

    # ===== Property Accessors for Derived Fields =====
    # (No property accessor needed for source_model_inference_input_variable_list since it's now a field)

    # ===== Validators =====

    @field_validator("inference_instance_type")
    @classmethod
    def validate_inference_instance_type(cls, v: str) -> str:
        """Validate the inference instance type"""
        if not v.startswith("ml."):
            raise ValueError(
                f"Invalid inference instance type: {v}. Must start with 'ml.'"
            )
        return v

    @field_validator("framework")
    @classmethod
    def validate_framework(cls, v: str) -> str:
        """Validate the ML framework"""
        valid_frameworks = ["xgboost", "sklearn", "pytorch", "tensorflow"]
        if v.lower() not in valid_frameworks:
            raise ValueError(f"Framework must be one of {valid_frameworks}")
        return v.lower()

    # ===== Model Validation =====

    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "RegistrationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()
        return self

    @model_validator(mode="after")
    def validate_registration_configs(self) -> "RegistrationConfig":
        """Validate registration-specific configurations (without file existence checks)"""
        # Removed file existence validation to improve configuration portability
        # File validation should happen at execution time in builders, not at config creation time
        
        # Only validate that source_dir is provided if inference_entry_point is a relative path
        if self.inference_entry_point and not self.inference_entry_point.startswith("s3://"):
            if not self.source_dir:
                raise ValueError(
                    "source_dir must be provided when inference_entry_point is a relative path"
                )

        return self

    @field_validator(
        "source_model_inference_content_types", "source_model_inference_response_types"
    )
    @classmethod
    def validate_content_types(cls, v: List[str]) -> List[str]:
        """Validate content and response types"""
        valid_types = [["text/csv"], ["application/json"]]
        if v not in valid_types:
            raise ValueError(f"Content/Response types must be one of {valid_types}")
        return v

    @field_validator("source_model_inference_input_variable_list")
    @classmethod
    def validate_input_variable_list(
        cls, v: Union[Dict[str, Union[VariableType, str]], List[List[str]]]
    ) -> Union[Dict[str, Union[VariableType, str]], List[List[str]]]:
        """
        Validate input variable list format.

        Args:
            v: Either a dictionary of variable names to types,
               or a list of [variable_name, variable_type] pairs

        Returns:
            Validated input variable list
        """
        if v is None:
            return {}  # Return empty dict as default

        # Handle dictionary format
        if isinstance(v, dict):
            result = {}
            for key, value in v.items():
                if not isinstance(key, str):
                    raise ValueError(
                        f"Key must be string, got {type(key)} for key: {key}"
                    )

                # Convert string values to VariableType enum
                if isinstance(value, str):
                    try:
                        value = VariableType(value.upper())
                    except ValueError:
                        raise ValueError(
                            f"Value must be 'NUMERIC' or 'TEXT', got: {value}"
                        )
                elif isinstance(value, VariableType):
                    # Keep VariableType as is
                    pass
                else:
                    raise ValueError(
                        f"Value must be string or VariableType, got: {type(value)}"
                    )

                result[key] = value
            return result

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

                if var_type.upper() not in ["NUMERIC", "TEXT"]:
                    raise ValueError(
                        f"Type must be 'NUMERIC' or 'TEXT', got: {var_type}"
                    )

            return v

        else:
            raise ValueError("Must be either a dictionary or a list of pairs")

    @field_validator("source_model_inference_output_variable_list")
    @classmethod
    def validate_output_variable_list(
        cls, v: Dict[str, Union[VariableType, str]]
    ) -> Dict[str, str]:
        """Validate variable lists and convert to string values"""
        if not v:  # If empty dictionary
            return v

        result = {}
        for key, value in v.items():
            # Validate key is a string
            if not isinstance(key, str):
                raise ValueError(f"Key must be string, got {type(key)} for key: {key}")

            # Convert VariableType to string or validate string value
            if isinstance(value, VariableType):
                result[key] = value.value
            elif isinstance(value, str) and value in [vt.value for vt in VariableType]:
                result[key] = value
            else:
                raise ValueError(
                    f"Value must be either 'NUMERIC' or 'TEXT', got: {value}"
                )

        return result

    # ===== Property Accessors for Derived Method Results =====

    @property
    def variable_schema(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """Generate variable schema for model registration"""
        if self._variable_schema is None:
            schema = {"input": {"variables": []}, "output": {"variables": []}}

            # Handle input variables in either format
            input_vars = self.source_model_inference_input_variable_list
            if isinstance(input_vars, dict):
                # Dictionary format
                for var_name, var_type in input_vars.items():
                    schema["input"]["variables"].append(
                        {
                            "name": var_name,
                            "type": (
                                var_type
                                if isinstance(var_type, str)
                                else var_type.value
                            ),
                        }
                    )
            elif isinstance(input_vars, list):
                # List format
                for var_name, var_type in input_vars:
                    schema["input"]["variables"].append(
                        {"name": var_name, "type": var_type}
                    )

            # Add output variables
            for (
                name,
                var_type,
            ) in self.source_model_inference_output_variable_list.items():
                schema["output"]["variables"].append(
                    {
                        "name": name,
                        "type": (
                            var_type if isinstance(var_type, str) else var_type.value
                        ),
                    }
                )

            self._variable_schema = schema

        return self._variable_schema

    # ===== Legacy Methods =====

    def get_variable_schema(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """Legacy method that forwards to the property"""
        return self.variable_schema

    # Removed get_script_path override - now inherits modernized version from BasePipelineConfig
    # which includes hybrid resolution and comprehensive fallbacks
    # Note: This config uses inference_entry_point instead of processing_entry_point,
    # but the modernized base method can handle this through the comprehensive fallback strategy

    # ===== Serialization =====

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)

        # Add derived properties to output
        data["variable_schema"] = self.variable_schema

        # Process variable lists for proper serialization
        if "source_model_inference_output_variable_list" in data:
            data["source_model_inference_output_variable_list"] = {
                k: v.value if isinstance(v, VariableType) else v
                for k, v in data["source_model_inference_output_variable_list"].items()
            }

        if "source_model_inference_input_variable_list" in data:
            input_vars = data["source_model_inference_input_variable_list"]
            if isinstance(input_vars, dict):
                data["source_model_inference_input_variable_list"] = {
                    k: v.value if isinstance(v, VariableType) else v
                    for k, v in input_vars.items()
                }

        return data

    # ===== Methods for working with input variable lists =====

    def set_source_model_inference_input_variable_list(
        self,
        numeric_fields: List[str] = None,
        text_fields: List[str] = None,
        output_format: str = "dict",
    ) -> None:
        """
        Set the input variable list for model inference using separate lists for numeric and text fields.

        Args:
            numeric_fields: List of field names that should be treated as NUMERIC
            text_fields: List of field names that should be treated as TEXT
            output_format: Format for storing variable list - either 'dict' or 'list'
        """
        # Use the standalone function to create the variable list
        result = create_inference_variable_list(
            numeric_fields=numeric_fields,
            text_fields=text_fields,
            output_format=output_format,
        )

        # Set the input variable list (now a regular field, not a private attribute)
        self.source_model_inference_input_variable_list = result

        # Invalidate cached schema so it will be regenerated
        self._variable_schema = None
