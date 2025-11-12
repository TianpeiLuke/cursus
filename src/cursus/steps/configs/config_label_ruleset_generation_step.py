"""
Label Ruleset Generation Step Configuration

This module implements the configuration class for the Label Ruleset Generation step
using the three-tier design pattern for optimal user experience and maintainability.
"""

from pydantic import BaseModel, Field, PrivateAttr, model_validator, field_validator
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum
import json
import logging
import uuid

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class ComparisonOperator(str, Enum):
    """
    Supported comparison operators for rule conditions.

    Categories:
    - Comparison: equals, not_equals, gt, gte, lt, lte
    - Collection: in_collection, not_in_collection
    - String: contains, not_contains, starts_with, ends_with, regex_match
    - Null: is_null, is_not_null
    """

    # Comparison operators
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GT = ">"
    GTE = ">="
    LT = "<"
    LTE = "<="

    # Collection operators
    IN = "in"
    NOT_IN = "not_in"

    # String operators
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX_MATCH = "regex_match"

    # Null operators
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"


class RuleCondition(BaseModel):
    """
    Single condition in a rule.

    Supports nested logical operators (all_of, any_of, none_of) and
    leaf conditions with field comparisons using validated operators.

    All fields are optional (Tier 2) but mutually exclusive validation ensures
    conditions are either leaf (field/operator/value) or logical (all_of/any_of/none_of).
    """

    # ===== Tier 2: Optional Condition Types (Mutually Exclusive) =====

    # Leaf condition fields (used together for field comparisons)
    field: Optional[str] = Field(
        default=None, description="Field name for leaf condition"
    )

    operator: Optional[ComparisonOperator] = Field(
        default=None, description="Comparison operator from ComparisonOperator enum"
    )

    value: Optional[Any] = Field(
        default=None, description="Expected value for comparison"
    )

    # Logical operators (for nested conditions)
    all_of: Optional[List["RuleCondition"]] = Field(
        default=None, description="All conditions must be true (AND logic)"
    )

    any_of: Optional[List["RuleCondition"]] = Field(
        default=None, description="At least one condition must be true (OR logic)"
    )

    none_of: Optional[List["RuleCondition"]] = Field(
        default=None, description="All conditions must be false (NOT logic)"
    )

    @model_validator(mode="after")
    def validate_condition_structure(self) -> "RuleCondition":
        """Validate that condition is either a leaf or a logical operator."""
        # Count how many condition types are set
        is_leaf = self.field is not None or self.operator is not None
        has_logical = (
            self.all_of is not None
            or self.any_of is not None
            or self.none_of is not None
        )

        if is_leaf and has_logical:
            raise ValueError(
                "Condition cannot be both a leaf (field/operator) and logical (all_of/any_of/none_of)"
            )

        if not is_leaf and not has_logical:
            raise ValueError(
                "Condition must be either a leaf (field/operator) or logical (all_of/any_of/none_of)"
            )

        # Validate leaf condition completeness
        if is_leaf:
            if self.field is None or self.operator is None:
                raise ValueError("Leaf condition must have both 'field' and 'operator'")

            # Null operators don't need value
            if self.operator not in [
                ComparisonOperator.IS_NULL,
                ComparisonOperator.IS_NOT_NULL,
            ]:
                if self.value is None:
                    raise ValueError(
                        f"Condition with operator '{self.operator}' must have 'value'"
                    )

        return self

    def to_script_format(self) -> Dict[str, Any]:
        """Convert to format expected by script."""
        result = {}

        # Leaf condition
        if self.field is not None:
            result["field"] = self.field
            result["operator"] = (
                self.operator.value
                if isinstance(self.operator, Enum)
                else self.operator
            )
            if self.value is not None:
                result["value"] = self.value

        # Logical operators
        if self.all_of is not None:
            result["all_of"] = [c.to_script_format() for c in self.all_of]
        if self.any_of is not None:
            result["any_of"] = [c.to_script_format() for c in self.any_of]
        if self.none_of is not None:
            result["none_of"] = [c.to_script_format() for c in self.none_of]

        return result

    model_config = {"extra": "forbid", "validate_assignment": True}


class LabelConfig(BaseModel):
    """
    Pydantic model for label configuration with multi-label support.

    Supports three modes via output_label_type:
    - 'binary': Single binary column
    - 'multiclass': Single multiclass column
    - 'multilabel': Multiple columns (new)

    Follows three-tier design:
    - Tier 1: Required user inputs
    - Tier 2: Optional user inputs with defaults
    """

    # ===== Tier 1: Required User Inputs =====

    # Unified output field (works for all modes)
    output_label_name: Union[str, List[str]] = Field(
        ...,
        description=(
            "Output label column name(s). "
            "String for single column (binary/multiclass), "
            "List[str] for multiple columns (multilabel)"
        ),
    )

    # Extended to support multilabel
    output_label_type: str = Field(
        ...,
        description="Type of classification: 'binary', 'multiclass', or 'multilabel'",
    )

    # Flexible: Global (List) or Per-Column (Dict)
    label_values: Union[
        List[Union[int, str]],  # Global: same for all columns
        Dict[str, List[Union[int, str]]],  # Per-column: different per column
    ] = Field(
        ...,
        description=(
            "Valid label values. "
            "List for global (all columns same), "
            "Dict[column_name -> values] for per-column"
        ),
    )

    # Flexible: Global (Dict) or Per-Column (Dict[Dict])
    label_mapping: Union[
        Dict[str, str],  # Global: same for all columns
        Dict[str, Dict[str, str]],  # Per-column: different per column
    ] = Field(
        ...,
        description=(
            "Label to human-readable mapping. "
            "Dict for global (all columns same), "
            "Dict[column_name -> mapping] for per-column"
        ),
    )

    # Flexible: Global (int/str) or Per-Column (Dict)
    default_label: Union[
        int,
        str,  # Global: same for all columns
        Dict[str, Union[int, str]],  # Per-column: different per column
    ] = Field(
        ...,
        description=(
            "Default label when no rules match. "
            "int/str for global (all columns same), "
            "Dict[column_name -> value] for per-column"
        ),
    )

    # ===== Tier 2: Optional User Inputs with Defaults =====

    evaluation_mode: str = Field(
        default="priority",
        description="Rule evaluation mode: 'priority' (first match wins) or 'confidence' (highest confidence wins)",
    )

    sparse_representation: bool = Field(
        default=True,
        description="Use NaN for non-matching categories in multilabel mode",
    )

    @field_validator("output_label_type")
    @classmethod
    def validate_label_type(cls, v: str) -> str:
        """Validate label_type is valid."""
        if v not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(
                "output_label_type must be 'binary', 'multiclass', or 'multilabel'"
            )
        return v

    @field_validator("evaluation_mode")
    @classmethod
    def validate_evaluation_mode(cls, v: str) -> str:
        """Validate evaluation mode."""
        if v not in ["priority", "confidence"]:
            raise ValueError("evaluation_mode must be 'priority' or 'confidence'")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "LabelConfig":
        """Validate fields match output_label_type."""
        is_list = isinstance(self.output_label_name, list)

        if self.output_label_type in ["binary", "multiclass"]:
            # Single-label: normalize to string
            if is_list:
                if len(self.output_label_name) != 1:
                    raise ValueError(
                        f"{self.output_label_type} requires single column name"
                    )
                # Normalize single-element list to string
                self.output_label_name = self.output_label_name[0]

            # Validate label_values and label_mapping are global format
            if isinstance(self.label_values, dict):
                raise ValueError("Single-label mode requires list for label_values")
            if isinstance(self.label_mapping, dict) and any(
                isinstance(v, dict) for v in self.label_mapping.values()
            ):
                raise ValueError(
                    "Single-label mode requires simple dict for label_mapping"
                )

        elif self.output_label_type == "multilabel":
            # Multilabel: must be list with at least 2 columns
            if not is_list:
                raise ValueError("multilabel requires list of column names")
            if len(self.output_label_name) < 2:
                raise ValueError("multilabel requires at least 2 columns")

            # Check for duplicates
            if len(self.output_label_name) != len(set(self.output_label_name)):
                raise ValueError("Duplicate column names in multilabel")

            # Validate per-column structures if used
            valid_columns = set(self.output_label_name)

            if isinstance(self.label_values, dict):
                label_values_cols = set(self.label_values.keys())
                missing = valid_columns - label_values_cols
                extra = label_values_cols - valid_columns

                if missing:
                    raise ValueError(f"label_values missing columns: {missing}")
                if extra:
                    raise ValueError(
                        f"label_values has invalid columns not in output_label_name: {extra}"
                    )

            if isinstance(self.label_mapping, dict) and all(
                isinstance(v, dict) for v in self.label_mapping.values()
            ):
                # Per-column mapping
                label_mapping_cols = set(self.label_mapping.keys())
                missing = valid_columns - label_mapping_cols
                extra = label_mapping_cols - valid_columns

                if missing:
                    raise ValueError(f"label_mapping missing columns: {missing}")
                if extra:
                    raise ValueError(
                        f"label_mapping has invalid columns not in output_label_name: {extra}"
                    )

            if isinstance(self.default_label, dict):
                # Per-column default_label
                default_label_cols = set(self.default_label.keys())
                missing = valid_columns - default_label_cols
                extra = default_label_cols - valid_columns

                if missing:
                    raise ValueError(f"default_label missing columns: {missing}")
                if extra:
                    raise ValueError(
                        f"default_label has invalid columns not in output_label_name: {extra}"
                    )

        return self

    @model_validator(mode="after")
    def validate_default_label(self) -> "LabelConfig":
        """Validate default_label is in label_values."""
        # Handle per-column default_label
        if isinstance(self.default_label, dict):
            for col, default_val in self.default_label.items():
                if isinstance(self.label_values, dict):
                    col_values = [str(v) for v in self.label_values.get(col, [])]
                    if str(default_val) not in col_values:
                        raise ValueError(
                            f"default_label[{col}] '{default_val}' must be in label_values[{col}]"
                        )
                else:
                    label_values_str = [str(v) for v in self.label_values]
                    if str(default_val) not in label_values_str:
                        raise ValueError(
                            f"default_label[{col}] '{default_val}' must be in label_values"
                        )
        else:
            # Global default_label
            if isinstance(self.label_values, list):
                label_values_str = [str(v) for v in self.label_values]
            else:
                # Collect all values from per-column
                label_values_str = []
                for col_values in self.label_values.values():
                    label_values_str.extend([str(v) for v in col_values])

            if str(self.default_label) not in label_values_str:
                raise ValueError(
                    f"default_label '{self.default_label}' must be in label_values"
                )

        return self

    @model_validator(mode="after")
    def validate_binary_constraints(self) -> "LabelConfig":
        """Validate binary classification uses [0, 1] values."""
        if self.output_label_type == "binary":
            if isinstance(self.label_values, list) and set(self.label_values) != {
                0,
                1,
            }:
                logger.warning(
                    f"Binary classification should use label_values [0, 1], got {self.label_values}"
                )
        return self

    def to_script_format(self) -> Dict[str, Any]:
        """Convert to format expected by script."""
        return {
            "output_label_name": self.output_label_name,
            "output_label_type": self.output_label_type,
            "label_values": self.label_values,
            "label_mapping": self.label_mapping,
            "default_label": self.default_label,
            "evaluation_mode": self.evaluation_mode,
            "sparse_representation": self.sparse_representation,
        }

    model_config = {"extra": "forbid", "validate_assignment": True}


class FieldConfig(BaseModel):
    """
    Pydantic model for field configuration.

    Defines the schema of fields that can be referenced in rules.

    Follows three-tier design:
    - Tier 1: Required user inputs
    - Tier 2: Optional user inputs with defaults
    """

    # ===== Tier 1: Required User Inputs =====

    required_fields: List[str] = Field(
        ...,
        min_length=1,
        description="Array of required field names that must exist in data",
    )

    field_types: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping field names to types: 'string', 'int', 'float', 'bool'",
    )

    # ===== Tier 2: Optional User Inputs with Defaults =====

    optional_fields: List[str] = Field(
        default_factory=list,
        description="Array of optional field names that may exist in data",
    )

    @field_validator("field_types")
    @classmethod
    def validate_field_types(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate field types are valid."""
        valid_types = {"string", "int", "float", "bool"}
        for field, field_type in v.items():
            if field_type not in valid_types:
                raise ValueError(
                    f"Invalid type '{field_type}' for field '{field}'. Must be one of {valid_types}"
                )
        return v

    @model_validator(mode="after")
    def validate_all_fields_have_types(self) -> "FieldConfig":
        """Validate all declared fields have types."""
        all_fields = set(self.required_fields) | set(self.optional_fields)
        typed_fields = set(self.field_types.keys())

        missing_types = all_fields - typed_fields
        if missing_types:
            raise ValueError(f"Fields missing type definitions: {missing_types}")

        extra_types = typed_fields - all_fields
        if extra_types:
            logger.warning(
                f"field_types contains fields not in required/optional: {extra_types}"
            )

        return self

    def to_script_format(self) -> Dict[str, Any]:
        """Convert to format expected by script."""
        return {
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "field_types": self.field_types,
        }

    model_config = {"extra": "forbid", "validate_assignment": True}


class RuleDefinition(BaseModel):
    """
    Pydantic model for a single rule definition.

    Defines a classification rule with conditions and output label.
    The rule_id is auto-generated and should not be provided by users.

    Follows three-tier design:
    - Tier 1: Required user inputs
    - Tier 2: Optional user inputs with defaults
    - Tier 3: Derived fields (private, auto-generated)
    """

    # ===== Tier 1: Required User Inputs =====

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable rule name",
    )

    priority: int = Field(
        ...,
        ge=1,
        description="Priority for evaluation (lower = higher priority, 1 = highest)",
    )

    conditions: RuleCondition = Field(
        ...,
        description="Nested condition expression using RuleCondition with validated operators",
    )

    output_label: Union[int, str, Dict[str, Union[int, str]]] = Field(
        ...,
        description=(
            "Output label value(s). "
            "int/str for single-label mode, "
            "Dict[column_name -> value] for multilabel mode"
        ),
    )

    # ===== Tier 2: Optional User Inputs with Defaults =====

    enabled: bool = Field(
        default=True,
        description="Whether rule is active",
    )

    description: str = Field(
        default="",
        description="Description of what this rule identifies",
    )

    # ===== Tier 3: Derived Fields (Private, Auto-Generated) =====

    _rule_id: str = PrivateAttr(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")

    @property
    def rule_id(self) -> str:
        """Get auto-generated unique rule identifier."""
        return self._rule_id

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is not empty."""
        if not v.strip():
            raise ValueError("rule name cannot be empty or whitespace")
        return v.strip()

    @model_validator(mode="after")
    def validate_output_label(self) -> "RuleDefinition":
        """Validate output_label format."""
        # Validate multi-label dict is not empty
        if isinstance(self.output_label, dict) and len(self.output_label) == 0:
            raise ValueError("output_label dict cannot be empty for multilabel mode")

        return self

    def to_script_format(self) -> Dict[str, Any]:
        """Convert to format expected by script."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "conditions": self.conditions.to_script_format(),
            "output_label": self.output_label,
            "description": self.description,
        }

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include auto-generated rule_id."""
        data = super().model_dump(**kwargs)
        data["rule_id"] = self.rule_id
        return data

    model_config = {"extra": "forbid", "validate_assignment": True}


class RulesetDefinitionList(BaseModel):
    """
    Pydantic model for a list of rule definitions with validation.

    Ensures rule IDs are unique and provides utility methods.
    """

    rules: List[RuleDefinition] = Field(
        ...,
        min_length=1,
        description="List of rule definitions (at least one required)",
    )

    @field_validator("rules")
    @classmethod
    def validate_unique_rule_ids(cls, v: List[RuleDefinition]) -> List[RuleDefinition]:
        """Validate all rule IDs are unique."""
        if not v:
            raise ValueError("At least one rule definition is required")

        rule_ids = set()
        for i, rule in enumerate(v):
            if rule.rule_id in rule_ids:
                raise ValueError(f"Duplicate rule_id: '{rule.rule_id}' at index {i}")
            rule_ids.add(rule.rule_id)

        return v

    def to_script_format(self) -> List[Dict[str, Any]]:
        """Convert all rules to format expected by script."""
        return [rule.to_script_format() for rule in self.rules]

    def to_json(self, **kwargs) -> str:
        """Convert to JSON string in script format."""
        return json.dumps(self.to_script_format(), **kwargs)

    def get_rule_ids(self) -> List[str]:
        """Get list of all rule IDs."""
        return [rule.rule_id for rule in self.rules]

    def get_rule_by_id(self, rule_id: str) -> Optional[RuleDefinition]:
        """Get rule by ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def sort_by_priority(self) -> "RulesetDefinitionList":
        """Return new RulesetDefinitionList sorted by priority."""
        sorted_rules = sorted(self.rules, key=lambda x: x.priority)
        return RulesetDefinitionList(rules=sorted_rules)

    model_config = {"extra": "forbid", "validate_assignment": True}


class LabelRulesetGenerationConfig(ProcessingStepConfigBase):
    """
    Configuration for Label Ruleset Generation step using three-tier design.

    This step validates and optimizes user-defined classification rules for
    transparent, maintainable rule-based label mapping in ML training pipelines.

    Tier 1: Essential user inputs (required)
    Tier 2: System inputs with defaults (optional)
    Tier 3: Derived fields (private with property access)
    """

    # ===== Tier 1: Essential User Inputs (Required) =====

    # Label configuration (required)
    label_config: LabelConfig = Field(
        ...,
        description="Label configuration defining output label structure and valid values",
    )

    # Rule definitions (required)
    rule_definitions: RulesetDefinitionList = Field(
        ...,
        description="List of rule definitions for classification",
    )

    # ===== Tier 2: System Inputs with Defaults (Optional) =====

    # Configuration path - defaults to standard 'ruleset_configs' subdirectory
    ruleset_configs_path: str = Field(
        default="ruleset_configs",
        description="Subdirectory name or relative path under the processing source directory for ruleset configuration files (label_config.json, ruleset.json). Must be a relative path, not absolute. Note: field_config.json is auto-generated by the script from rules.",
    )

    # Validation settings
    enable_field_validation: bool = Field(
        default=True,
        description="Enable field schema validation (validates field references against declared schema)",
    )

    enable_label_validation: bool = Field(
        default=True,
        description="Enable label value validation (ensures output labels match configuration)",
    )

    enable_logic_validation: bool = Field(
        default=True,
        description="Enable rule logic validation (checks for tautologies, contradictions, unreachable rules)",
    )

    enable_rule_optimization: bool = Field(
        default=True,
        description="Enable rule priority optimization (reorders rules by complexity for efficient execution)",
    )

    # Processing step overrides
    processing_entry_point: str = Field(
        default="label_ruleset_generation.py",
        description="Entry point script for ruleset generation",
    )

    # ===== Tier 3: Derived Fields (Private with Property Access) =====

    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _resolved_ruleset_configs_path: Optional[str] = PrivateAttr(default=None)

    @property
    def environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processing step."""
        if self._environment_variables is None:
            self._environment_variables = {
                "ENABLE_FIELD_VALIDATION": str(self.enable_field_validation).lower(),
                "ENABLE_LABEL_VALIDATION": str(self.enable_label_validation).lower(),
                "ENABLE_LOGIC_VALIDATION": str(self.enable_logic_validation).lower(),
                "ENABLE_RULE_OPTIMIZATION": str(self.enable_rule_optimization).lower(),
            }

        return self._environment_variables

    @property
    def resolved_ruleset_configs_path(self) -> Optional[str]:
        """
        Get resolved absolute path for ruleset configurations.

        Uses effective_source_dir from base class for consistency.

        Returns:
            Absolute path to ruleset configs directory, or None if not configured

        Raises:
            ValueError: If ruleset_configs_path is set but source directory cannot be resolved
        """
        if self.ruleset_configs_path is None:
            return None

        if self._resolved_ruleset_configs_path is None:
            # Use effective_source_dir from base class
            resolved_source_dir = self.effective_source_dir
            if resolved_source_dir is None:
                raise ValueError(
                    "Cannot resolve ruleset_configs_path: no processing source directory configured. "
                    "Set either processing_source_dir or source_dir in configuration."
                )

            # Construct full path: resolved_source_dir / 'ruleset_configs'
            self._resolved_ruleset_configs_path = str(
                Path(resolved_source_dir) / self.ruleset_configs_path
            )

        return self._resolved_ruleset_configs_path

    def generate_ruleset_config_bundle(self) -> None:
        """
        Generate complete ruleset configuration bundle.

        Creates JSON files for non-None configurations in the configured ruleset_configs_path:
        - label_config.json (if label_config is not None)
        - field_config.json (if field_config is not None)
        - ruleset.json (if rule_definitions is not None)

        Only generates files for configurations that are provided.

        Raises:
            ValueError: If ruleset_configs_path is not configured
        """
        output_dir = Path(self.resolved_ruleset_configs_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # Generate label_config.json (if provided)
        if self.label_config is not None:
            label_config_file = output_dir / "label_config.json"
            with open(label_config_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.label_config.to_script_format(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info(f"Generated label config: {label_config_file}")
            generated_files.append("label_config.json")

        # Generate ruleset.json (if provided)
        # Note: field_config.json is NOT generated here - the script will infer fields from rules
        if self.rule_definitions is not None:
            ruleset_file = output_dir / "ruleset.json"
            with open(ruleset_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.rule_definitions.to_script_format(),
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            logger.info(f"Generated ruleset: {ruleset_file}")
            generated_files.append("ruleset.json")

        if generated_files:
            logger.info(f"Generated ruleset configuration bundle in: {output_dir}")
            logger.info(
                f"Bundle contains {len(generated_files)} JSON configuration files: {', '.join(generated_files)}"
            )
        else:
            logger.warning(f"No configuration files generated - all configs are None")

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)

        # Add derived properties to output
        data["environment_variables"] = self.environment_variables

        # Add resolved path properties if configured
        if self.ruleset_configs_path is not None:
            data["resolved_ruleset_configs_path"] = self.resolved_ruleset_configs_path

        return data

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "LabelRulesetGenerationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Note: field_config inference is now handled by the script, not config
        # The script will infer fields from rules during generation

        # Initialize ruleset-specific derived fields
        _ = self.environment_variables

        # Auto-generate ruleset config bundle after all configurations are ready
        try:
            self.generate_ruleset_config_bundle()
            logger.info(
                f"Auto-generated ruleset configuration bundle at: {self.resolved_ruleset_configs_path}"
            )
        except Exception as e:
            # Log warning but don't fail initialization
            logger.warning(f"Failed to auto-generate ruleset config bundle: {e}")
            logger.info(
                "You can manually call generate_ruleset_config_bundle() after providing missing settings"
            )

        return self

    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..contracts.label_ruleset_generation_contract import (
            LABEL_RULESET_GENERATION_CONTRACT,
        )

        return LABEL_RULESET_GENERATION_CONTRACT

    def get_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """
        Get script path for the label ruleset generation step.

        Args:
            default_path: Default script path to use if not found via other methods

        Returns:
            Script path resolved from processing_entry_point and source directories
        """
        # Use the parent class implementation which handles hybrid resolution
        return super().get_script_path(default_path)

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include ruleset-specific fields.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add ruleset-specific fields
        ruleset_fields = {
            "ruleset_configs_path": self.ruleset_configs_path,
            "enable_field_validation": self.enable_field_validation,
            "enable_label_validation": self.enable_label_validation,
            "enable_logic_validation": self.enable_logic_validation,
            "enable_rule_optimization": self.enable_rule_optimization,
        }

        # Combine base fields and ruleset fields
        init_fields = {**base_fields, **ruleset_fields}

        return init_fields


def load_rules_from_json(json_data: str) -> RulesetDefinitionList:
    """
    Load rules from JSON string with validation.

    Args:
        json_data: JSON string containing rule definitions

    Returns:
        Validated RulesetDefinitionList

    Raises:
        ValueError: If JSON is invalid or rules don't validate
        pydantic.ValidationError: If rule data doesn't match schema
    """
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if isinstance(data, list):
        # List of rule dictionaries
        return RulesetDefinitionList(rules=[RuleDefinition(**rule) for rule in data])
    elif isinstance(data, dict):
        # Single rule dictionary
        return RulesetDefinitionList(rules=[RuleDefinition(**data)])
    else:
        raise ValueError(
            "JSON data must be a list of rules or a single rule dictionary"
        )


def load_rules_from_dict(data: Any) -> RulesetDefinitionList:
    """
    Load rules from dictionary/list data with validation.

    Args:
        data: Dictionary or list containing rule definitions

    Returns:
        Validated RulesetDefinitionList

    Raises:
        pydantic.ValidationError: If rule data doesn't match schema
    """
    if isinstance(data, list):
        # List of rule dictionaries
        return RulesetDefinitionList(rules=[RuleDefinition(**rule) for rule in data])
    elif isinstance(data, dict):
        # Single rule dictionary
        return RulesetDefinitionList(rules=[RuleDefinition(**data)])
    else:
        raise ValueError("Data must be a list of rules or a single rule dictionary")
