"""
Bedrock Prompt Template Generation Step Configuration

This module implements the configuration class for the Bedrock Prompt Template Generation step
using the three-tier design pattern for optimal user experience and maintainability.
"""

from pydantic import BaseModel, Field, PrivateAttr, model_validator, field_validator
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class CategoryDefinition(BaseModel):
    """
    Pydantic model for a single category definition used in classification tasks.

    Based on the expected format from bedrock_prompt_template_generation.py script:
    - Required fields: name, description, conditions, key_indicators
    - Optional fields: exceptions, examples, priority
    - Additional fields: aliases, validation_rules (for future extensibility)
    """

    # Required fields (validated by script)
    name: str = Field(
        ...,
        min_length=1,
        description="The category name (must be unique and non-empty)",
    )

    description: str = Field(
        ...,
        min_length=1,
        description="Clear description of what this category represents",
    )

    conditions: List[str] = Field(
        ...,
        min_length=1,
        description="List of conditions that must be met for this category (at least one required)",
    )

    key_indicators: List[str] = Field(
        ...,
        min_length=1,
        description="List of key indicators or signals for this category (at least one required)",
    )

    # Optional fields (used by script if present)
    exceptions: List[str] = Field(
        default_factory=list,
        description="List of exceptions - things that should NOT be classified as this category",
    )

    examples: Optional[List[str]] = Field(
        default=None,
        description="Optional list of example texts that belong to this category",
    )

    priority: int = Field(
        default=1,
        ge=1,
        description="Priority for category ordering (lower numbers = higher priority, used for sorting)",
    )

    # Additional optional fields for extensibility
    aliases: Optional[List[str]] = Field(
        default=None, description="Optional list of alternative names for this category"
    )

    validation_rules: Optional[List[str]] = Field(
        default=None,
        description="Optional list of additional validation rules for this category",
    )

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """Validate that name is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Category name cannot be empty or whitespace")
        return v.strip()

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v: str) -> str:
        """Validate that description is not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Category description cannot be empty or whitespace")
        return v.strip()

    @field_validator("conditions")
    @classmethod
    def conditions_must_not_be_empty(cls, v: List[str]) -> List[str]:
        """Validate that conditions list contains non-empty strings."""
        if not v:
            raise ValueError("At least one condition is required")

        cleaned_conditions = []
        for i, condition in enumerate(v):
            if not condition or not condition.strip():
                raise ValueError(f"Condition {i} cannot be empty or whitespace")
            cleaned_conditions.append(condition.strip())

        return cleaned_conditions

    @field_validator("key_indicators")
    @classmethod
    def key_indicators_must_not_be_empty(cls, v: List[str]) -> List[str]:
        """Validate that key_indicators list contains non-empty strings."""
        if not v:
            raise ValueError("At least one key indicator is required")

        cleaned_indicators = []
        for i, indicator in enumerate(v):
            if not indicator or not indicator.strip():
                raise ValueError(f"Key indicator {i} cannot be empty or whitespace")
            cleaned_indicators.append(indicator.strip())

        return cleaned_indicators

    @field_validator("exceptions")
    @classmethod
    def exceptions_must_not_be_empty_strings(cls, v: List[str]) -> List[str]:
        """Validate that exceptions list doesn't contain empty strings."""
        if not v:
            return v

        cleaned_exceptions = []
        for i, exception in enumerate(v):
            if not exception or not exception.strip():
                raise ValueError(f"Exception {i} cannot be empty or whitespace")
            cleaned_exceptions.append(exception.strip())

        return cleaned_exceptions

    @field_validator("examples")
    @classmethod
    def examples_must_not_be_empty_strings(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        """Validate that examples list doesn't contain empty strings."""
        if not v:
            return v

        cleaned_examples = []
        for i, example in enumerate(v):
            if not example or not example.strip():
                raise ValueError(f"Example {i} cannot be empty or whitespace")
            cleaned_examples.append(example.strip())

        return cleaned_examples

    def to_script_format(self) -> Dict[str, Any]:
        """
        Convert to the format expected by the bedrock_prompt_template_generation.py script.

        Returns:
            Dictionary in the format expected by the script
        """
        result = {
            "name": self.name,
            "description": self.description,
            "conditions": self.conditions,
            "key_indicators": self.key_indicators,
            "exceptions": self.exceptions,
            "priority": self.priority,
        }

        # Only include examples if they exist (script checks for None/empty)
        if self.examples:
            result["examples"] = self.examples

        # Include optional fields if they exist
        if self.aliases:
            result["aliases"] = self.aliases

        if self.validation_rules:
            result["validation_rules"] = self.validation_rules

        return result

    model_config = {"extra": "forbid", "validate_assignment": True}


class CategoryDefinitionList(BaseModel):
    """
    Pydantic model for a list of category definitions with validation.

    Ensures that category names are unique and provides utility methods
    for working with the category list.
    """

    categories: List[CategoryDefinition] = Field(
        ...,
        min_length=1,
        description="List of category definitions (at least one required)",
    )

    @field_validator("categories")
    @classmethod
    def categories_must_have_unique_names(
        cls, v: List[CategoryDefinition]
    ) -> List[CategoryDefinition]:
        """Validate that all category names are unique."""
        if not v:
            raise ValueError("At least one category definition is required")

        names = set()
        for i, category in enumerate(v):
            if category.name in names:
                raise ValueError(
                    f'Duplicate category name: "{category.name}" at index {i}'
                )
            names.add(category.name)

        return v

    def to_script_format(self) -> List[Dict[str, Any]]:
        """
        Convert all categories to the format expected by the script.

        Returns:
            List of dictionaries in the format expected by the script
        """
        return [category.to_script_format() for category in self.categories]

    def to_json(self, **kwargs) -> str:
        """
        Convert to JSON string in script format.

        Args:
            **kwargs: Additional arguments passed to json.dumps

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_script_format(), **kwargs)

    def get_category_names(self) -> List[str]:
        """Get list of all category names."""
        return [category.name for category in self.categories]

    def get_category_by_name(self, name: str) -> Optional[CategoryDefinition]:
        """
        Get category by name.

        Args:
            name: Category name to search for

        Returns:
            CategoryDefinition if found, None otherwise
        """
        for category in self.categories:
            if category.name == name:
                return category
        return None

    def sort_by_priority(self) -> "CategoryDefinitionList":
        """
        Return a new CategoryDefinitionList sorted by priority.

        Returns:
            New CategoryDefinitionList with categories sorted by priority (ascending)
        """
        sorted_categories = sorted(self.categories, key=lambda x: x.priority)
        return CategoryDefinitionList(categories=sorted_categories)

    model_config = {"extra": "forbid", "validate_assignment": True}


class SystemPromptConfig(BaseModel):
    """
    Configuration for system prompt generation with comprehensive defaults.

    This model defines how the AI's role, expertise, and behavioral guidelines
    are structured in the system prompt component of the template.
    """

    role_definition: str = Field(
        default="expert analyst",
        description="The AI's primary role (e.g., 'expert analyst', 'data scientist', 'classification specialist')",
    )

    expertise_areas: List[str] = Field(
        default=["data analysis", "classification", "pattern recognition"],
        description="List of expertise domains the AI should demonstrate knowledge in",
    )

    responsibilities: List[str] = Field(
        default=[
            "analyze data accurately",
            "classify content systematically",
            "provide clear reasoning",
        ],
        description="List of primary tasks and responsibilities the AI should perform",
    )

    behavioral_guidelines: List[str] = Field(
        default=["be precise", "be objective", "be thorough", "be consistent"],
        description="List of behavioral instructions that guide the AI's approach",
    )

    tone: str = Field(
        default="professional",
        description="Communication tone (e.g., 'professional', 'casual', 'technical', 'formal')",
    )

    model_config = {
        "extra": "allow"
    }  # Allow additional fields for future extensibility


class OutputFormatConfig(BaseModel):
    """
    Configuration for output format generation with comprehensive defaults.

    This model defines the structure and validation requirements for the
    expected output format in the generated prompt template.

    Supports both structured_json and structured_text formats with full customization.
    """

    format_type: str = Field(
        default="structured_json",
        description="Type of output format ('structured_json', 'structured_text', 'hybrid')",
    )

    required_fields: List[str] = Field(
        default=["category", "confidence", "key_evidence", "reasoning"],
        description="List of required fields in the output format",
    )

    field_descriptions: Dict[str, str] = Field(
        default_factory=lambda: {
            "category": "The classified category name (must be exactly one of the defined categories)",
            "confidence": "Confidence score between 0.0 and 1.0 indicating certainty of classification",
            "key_evidence": "Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision.",
            "reasoning": "Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected",
        },
        description="Dictionary mapping field names to their descriptions",
    )

    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""Optional JSON schema for validation schema generation. When provided, this schema
        will be used directly for generating the validation schema instead of deriving it from field_descriptions.
        
        This allows proper specification of complex types like nested objects and arrays:
        
        Example:
        {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": []},  # enum will be populated from categories
                "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "key_evidence": {
                    "type": "object",
                    "properties": {
                        "message_evidence": {"type": "array", "items": {"type": "string"}},
                        "shipping_evidence": {"type": "array", "items": {"type": "string"}},
                        "timeline_evidence": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["message_evidence", "shipping_evidence", "timeline_evidence"]
                }
            },
            "required": ["category", "confidence_score", "key_evidence"]
        }
        
        If not provided, a simple schema will be generated from field_descriptions with all fields as strings.
        """,
    )

    validation_requirements: List[str] = Field(
        default_factory=lambda: [
            "category must match one of the predefined category names exactly",
            "confidence must be a number between 0.0 and 1.0",
            "key_evidence must align with category conditions and avoid category exceptions",
            "key_evidence must reference specific content from the input data",
            "reasoning must explain the logical connection between evidence and category selection",
        ],
        description="List of validation requirements for the output format",
    )

    evidence_validation_rules: List[str] = Field(
        default_factory=lambda: [
            "Evidence MUST align with at least one condition for the selected category",
            "Evidence MUST NOT match any exceptions listed for the selected category",
            "Evidence should reference specific content from the input data",
            "Multiple pieces of supporting evidence strengthen the classification",
        ],
        description="List of specific rules for validating evidence fields",
    )

    # ===== NEW: Structured Text Format Configuration =====

    header_text: Optional[str] = Field(
        default=None,
        description="Custom header text for output format section (used in structured_text format)",
    )

    structured_text_sections: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="""Configuration for structured text output format sections. Each section defines:
        - number: Section number (e.g., 1, 2, 3)
        - header: Section header text (e.g., 'Category', 'Confidence Score')
        - format: 'single_value' or 'subsections'
        - placeholder: Placeholder text for the value
        - subsections: List of subsection configurations (for format='subsections')
        - item_prefix: Prefix for list items (e.g., '[sep] ')
        - indent: Indentation for subsections (e.g., '   ')
        """,
    )

    formatting_rules: Optional[List[str]] = Field(
        default=None,
        description="List of formatting rules to include in output format specification",
    )

    example_output: Optional[Any] = Field(
        default=None,
        description="Example output to show in the template (string or list of lines)",
    )

    model_config = {
        "extra": "allow"
    }  # Allow additional fields for future extensibility


class InstructionConfig(BaseModel):
    """
    Configuration for instruction generation with comprehensive defaults.

    This model defines which instruction components should be included
    in the generated prompt template to guide the AI's analysis process.

    Supports both basic boolean flags and detailed classification guidelines.
    """

    include_analysis_steps: bool = Field(
        default=True, description="Include numbered step-by-step analysis instructions"
    )

    include_decision_criteria: bool = Field(
        default=True, description="Include decision-making criteria section"
    )

    include_reasoning_requirements: bool = Field(
        default=True, description="Include reasoning requirements and expectations"
    )

    step_by_step_format: bool = Field(
        default=True,
        description="Use numbered step format for analysis instructions (False = bullet points)",
    )

    include_evidence_validation: bool = Field(
        default=True, description="Include evidence validation rules and requirements"
    )

    # ===== NEW: Detailed Classification Guidelines =====

    classification_guidelines: Optional[Dict[str, Any]] = Field(
        default=None,
        description="""Detailed classification guidelines with hierarchical structure. Expected format:
        {
          "sections": [
            {
              "title": "## Main Section Title",
              "subsections": [
                {
                  "title": "### Subsection Title",
                  "content": ["Line 1", "Line 2", "..."]
                }
              ]
            }
          ]
        }
        
        This allows for comprehensive, structured classification guidance that can be
        customized for specific task requirements without modifying code.
        """,
    )

    model_config = {
        "extra": "allow"
    }  # Allow additional fields for future extensibility


def create_system_prompt_config_from_json(json_str: str) -> SystemPromptConfig:
    """
    Create SystemPromptConfig from JSON string with robust fallback to "{}".

    Args:
        json_str: JSON string configuration (can be empty or invalid)

    Returns:
        SystemPromptConfig instance with defaults applied
    """
    try:
        if not json_str or json_str.strip() == "{}":
            return SystemPromptConfig()

        config_dict = json.loads(json_str)
        return SystemPromptConfig(**config_dict)

    except Exception as e:
        logger.warning(
            f"Failed to parse system_prompt_config JSON: {e}. Using defaults."
        )
        return SystemPromptConfig()


def create_output_format_config_from_json(json_str: str) -> OutputFormatConfig:
    """
    Create OutputFormatConfig from JSON string with robust fallback to "{}".

    Args:
        json_str: JSON string configuration (can be empty or invalid)

    Returns:
        OutputFormatConfig instance with defaults applied
    """
    try:
        if not json_str or json_str.strip() == "{}":
            return OutputFormatConfig()

        config_dict = json.loads(json_str)
        return OutputFormatConfig(**config_dict)

    except Exception as e:
        logger.warning(
            f"Failed to parse output_format_config JSON: {e}. Using defaults."
        )
        return OutputFormatConfig()


def create_instruction_config_from_json(json_str: str) -> InstructionConfig:
    """
    Create InstructionConfig from JSON string with robust fallback to "{}".

    Args:
        json_str: JSON string configuration (can be empty or invalid)

    Returns:
        InstructionConfig instance with defaults applied
    """
    try:
        if not json_str or json_str.strip() == "{}":
            return InstructionConfig()

        config_dict = json.loads(json_str)
        return InstructionConfig(**config_dict)

    except Exception as e:
        logger.warning(f"Failed to parse instruction_config JSON: {e}. Using defaults.")
        return InstructionConfig()


class BedrockPromptTemplateGenerationConfig(ProcessingStepConfigBase):
    """
    Configuration for Bedrock Prompt Template Generation step using three-tier design.

    This step generates structured prompt templates for classification tasks using the
    5-component architecture pattern optimized for LLM performance.

    Tier 1: Essential user inputs (required)
    Tier 2: System inputs with defaults (optional)
    Tier 3: Derived fields (private with property access)
    """

    # ===== Tier 1: Essential User Inputs (Required) =====
    # These fields must be provided by users with no defaults

    # Input configuration - users must specify what input fields their template expects
    input_placeholders: List[str] = Field(
        ...,
        description="List of input field names to include in the template (e.g., ['input_data', 'context', 'metadata'])",
    )

    # ===== Tier 2: System Inputs with Defaults (Optional) =====
    # These fields have sensible defaults but can be overridden

    # Configuration path - defaults to standard 'prompt_configs' subdirectory
    prompt_configs_path: str = Field(
        default="prompt_configs",
        description="Subdirectory name or relative path under the processing source directory for prompt configuration files (system_prompt.json, output_format.json, instruction.json, category_definitions.json). Must be a relative path, not absolute. Examples: 'prompt_configs', 'docker/prompt_configs', 'config/prompts'",
    )

    # These fields have sensible defaults but can be overridden

    # Template generation settings
    template_task_type: str = Field(
        default="classification",
        description="Type of task for template generation (classification, sentiment_analysis, content_moderation)",
    )

    template_style: str = Field(
        default="structured",
        description="Style of template generation (structured, conversational, technical)",
    )

    validation_level: str = Field(
        default="standard",
        description="Level of template validation (basic, standard, comprehensive)",
    )

    # Output configuration
    output_format_type: str = Field(
        default="structured_json",
        description="Type of output format (structured_json, formatted_text, hybrid)",
    )

    required_output_fields: List[str] = Field(
        default=["category", "confidence", "key_evidence", "reasoning"],
        description="List of required fields in the output format",
    )

    # Template features
    include_examples: bool = Field(
        default=True, description="Include examples in the generated template"
    )

    generate_validation_schema: bool = Field(
        default=True, description="Generate JSON validation schema for downstream use"
    )

    template_version: str = Field(
        default="1.0", description="Version identifier for the generated template"
    )

    # Typed configuration fields (primary interface)
    system_prompt_settings: Optional[SystemPromptConfig] = Field(
        default=None,
        description="System prompt configuration with comprehensive defaults",
    )

    output_format_settings: Optional[OutputFormatConfig] = Field(
        default=None,
        description="Output format configuration with comprehensive defaults",
    )

    instruction_settings: Optional[InstructionConfig] = Field(
        default=None,
        description="Instruction configuration with comprehensive defaults",
    )

    # Category definitions (can be provided directly or via file path)
    category_definitions: Optional[List[CategoryDefinition]] = Field(
        default=None, description="List of category definitions for template generation"
    )

    # Processing step overrides
    processing_entry_point: str = Field(
        default="bedrock_prompt_template_generation.py",
        description="Entry point script for prompt template generation",
    )

    # ===== Tier 3: Derived Fields (Private with Property Access) =====
    # These fields are calculated from other fields

    _effective_system_prompt_config: Optional[Dict[str, Any]] = PrivateAttr(
        default=None
    )
    _effective_output_format_config: Optional[Dict[str, Any]] = PrivateAttr(
        default=None
    )
    _effective_instruction_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _effective_categories: Optional[CategoryDefinitionList] = PrivateAttr(default=None)
    _template_metadata: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _resolved_prompt_configs_path: Optional[str] = PrivateAttr(default=None)

    # Public properties for derived fields

    @property
    def effective_system_prompt_config(self) -> Dict[str, Any]:
        """Get system prompt configuration from typed settings or defaults."""
        if self._effective_system_prompt_config is None:
            if self.system_prompt_settings is not None:
                # Use provided typed Pydantic model
                self._effective_system_prompt_config = (
                    self.system_prompt_settings.model_dump()
                )
                logger.debug("Using provided system_prompt_settings")
            else:
                # Use comprehensive defaults
                self._effective_system_prompt_config = SystemPromptConfig().model_dump()
                logger.debug("Using default system_prompt_settings")

        return self._effective_system_prompt_config

    @property
    def effective_output_format_config(self) -> Dict[str, Any]:
        """Get output format configuration from typed settings or defaults."""
        if self._effective_output_format_config is None:
            if self.output_format_settings is not None:
                # Use provided typed Pydantic model
                self._effective_output_format_config = (
                    self.output_format_settings.model_dump()
                )
                logger.debug("Using provided output_format_settings")
            else:
                # Use comprehensive defaults with integration from other fields
                default_config = OutputFormatConfig(
                    format_type=self.output_format_type,
                    required_fields=self.required_output_fields,
                )
                self._effective_output_format_config = default_config.model_dump()
                logger.debug(
                    "Using default output_format_settings with field integration"
                )

        return self._effective_output_format_config

    @property
    def effective_instruction_config(self) -> Dict[str, Any]:
        """Get instruction configuration from typed settings or defaults."""
        if self._effective_instruction_config is None:
            if self.instruction_settings is not None:
                # Use provided typed Pydantic model
                self._effective_instruction_config = (
                    self.instruction_settings.model_dump()
                )
                logger.debug("Using provided instruction_settings")
            else:
                # Use comprehensive defaults
                self._effective_instruction_config = InstructionConfig().model_dump()
                logger.debug("Using default instruction_settings")

        return self._effective_instruction_config

    @property
    def effective_categories(self) -> Optional[CategoryDefinitionList]:
        """Get effective category definitions from direct definitions."""
        if self._effective_categories is None:
            if self.category_definitions:
                # Use directly provided category definitions
                self._effective_categories = CategoryDefinitionList(
                    categories=self.category_definitions
                )
                logger.debug("Using provided category_definitions")
            else:
                self._effective_categories = None

        return self._effective_categories

    @property
    def template_metadata(self) -> Dict[str, Any]:
        """Get template generation metadata."""
        if self._template_metadata is None:
            self._template_metadata = {
                "template_version": self.template_version,
                "task_type": self.template_task_type,
                "template_style": self.template_style,
                "validation_level": self.validation_level,
                "output_format": self.output_format_type,
                "includes_examples": self.include_examples,
                "input_placeholders": self.input_placeholders,
                "required_output_fields": self.required_output_fields,
                "generate_validation_schema": self.generate_validation_schema,
            }

        return self._template_metadata

    @property
    def environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processing step (file-based approach)."""
        if self._environment_variables is None:
            # File-based approach: large configs are provided as JSON files, not environment variables
            # Only include environment variables that are NOT available in JSON config files
            self._environment_variables = {
                "TEMPLATE_TASK_TYPE": self.template_task_type,
                "TEMPLATE_STYLE": self.template_style,
                "VALIDATION_LEVEL": self.validation_level,
                "INPUT_PLACEHOLDERS": json.dumps(self.input_placeholders),
                "INCLUDE_EXAMPLES": str(self.include_examples).lower(),
                "GENERATE_VALIDATION_SCHEMA": str(
                    self.generate_validation_schema
                ).lower(),
                "TEMPLATE_VERSION": self.template_version,
            }
            # Note: OUTPUT_FORMAT_TYPE and REQUIRED_OUTPUT_FIELDS are now available in output_format.json

        return self._environment_variables

    @property
    def resolved_prompt_configs_path(self) -> Optional[str]:
        """
        Get resolved absolute path for prompt configurations with hybrid resolution.

        Uses effective_source_dir from base class for consistency.

        Returns:
            Absolute path to prompt configs directory, or None if not configured

        Raises:
            ValueError: If prompt_configs_path is set but source directory cannot be resolved
        """
        if self.prompt_configs_path is None:
            return None

        if self._resolved_prompt_configs_path is None:
            # Use effective_source_dir from base class (includes hybrid resolution)
            resolved_source_dir = self.effective_source_dir
            if resolved_source_dir is None:
                raise ValueError(
                    "Cannot resolve prompt_configs_path: no processing source directory configured. "
                    "Set either processing_source_dir or source_dir in configuration."
                )

            # Construct full path: resolved_source_dir / 'prompt_configs'
            self._resolved_prompt_configs_path = str(
                Path(resolved_source_dir) / self.prompt_configs_path
            )

        return self._resolved_prompt_configs_path

    def generate_prompt_config_bundle(self) -> None:
        """
        Generate complete prompt configuration bundle for the refactored file-based approach.

        Creates JSON files needed by the script in the configured prompt_configs_path:
        - system_prompt.json (only if system_prompt_settings is provided)
        - output_format.json (only if output_format_settings is provided)
        - instruction.json (only if instruction_settings is provided)
        - category_definitions.json (only if category_definitions is provided)

        All configuration files are optional - the script will use defaults when files are missing.

        Raises:
            ValueError: If prompt_configs_path is not configured
        """
        # prompt_configs_path is now required (Tier 1), so no need to check for None

        output_dir = Path(self.resolved_prompt_configs_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = []

        # Generate system_prompt.json only if system_prompt_settings is provided
        if self.system_prompt_settings is not None:
            system_prompt_file = output_dir / "system_prompt.json"
            with open(system_prompt_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.effective_system_prompt_config, f, indent=2, ensure_ascii=False
                )
            logger.info(f"Generated system prompt config: {system_prompt_file}")
            generated_files.append("system_prompt.json")
        else:
            logger.info(
                "Skipping system_prompt.json generation (system_prompt_settings is None - will use defaults)"
            )

        # Generate output_format.json only if output_format_settings is provided
        if self.output_format_settings is not None:
            output_format_file = output_dir / "output_format.json"
            with open(output_format_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.effective_output_format_config, f, indent=2, ensure_ascii=False
                )
            logger.info(f"Generated output format config: {output_format_file}")
            generated_files.append("output_format.json")
        else:
            logger.info(
                "Skipping output_format.json generation (output_format_settings is None - will use defaults)"
            )

        # Generate instruction.json only if instruction_settings is provided
        if self.instruction_settings is not None:
            instruction_file = output_dir / "instruction.json"
            with open(instruction_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.effective_instruction_config, f, indent=2, ensure_ascii=False
                )
            logger.info(f"Generated instruction config: {instruction_file}")
            generated_files.append("instruction.json")
        else:
            logger.info(
                "Skipping instruction.json generation (instruction_settings is None - will use defaults)"
            )

        # Generate category_definitions.json only if category definitions are available
        categories = self.effective_categories
        if categories:
            category_definitions_file = output_dir / "category_definitions.json"
            with open(category_definitions_file, "w", encoding="utf-8") as f:
                json.dump(
                    categories.to_script_format(), f, indent=2, ensure_ascii=False
                )
            logger.info(f"Generated category definitions: {category_definitions_file}")
            generated_files.append("category_definitions.json")
        else:
            logger.info(
                "Skipping category_definitions.json generation (no category definitions available)"
            )

        logger.info(f"Generated prompt configuration bundle in: {output_dir}")
        logger.info(
            f"Bundle contains {len(generated_files)} JSON configuration files: {', '.join(generated_files)}"
        )

    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)

        # Add derived properties to output
        data["effective_system_prompt_config"] = self.effective_system_prompt_config
        data["effective_output_format_config"] = self.effective_output_format_config
        data["effective_instruction_config"] = self.effective_instruction_config
        data["effective_categories"] = self.effective_categories
        data["template_metadata"] = self.template_metadata
        data["environment_variables"] = self.environment_variables

        # Add resolved path properties if they're configured
        if self.prompt_configs_path is not None:
            data["resolved_prompt_configs_path"] = self.resolved_prompt_configs_path

        return data

    # Initialize derived fields at creation time
    @model_validator(mode="after")
    def initialize_derived_fields(self) -> "BedrockPromptTemplateGenerationConfig":
        """Initialize all derived fields once after validation."""
        # Call parent validator first
        super().initialize_derived_fields()

        # Initialize template-specific derived fields
        _ = self.effective_system_prompt_config
        _ = self.effective_output_format_config
        _ = self.effective_instruction_config
        _ = self.effective_categories
        _ = self.template_metadata
        _ = self.environment_variables

        # Auto-generate prompt config bundle after all configurations are ready
        # generate_prompt_config_bundle() handles each JSON file independently
        try:
            self.generate_prompt_config_bundle()
            logger.info(
                f"Auto-generated prompt configuration bundle at: {self.resolved_prompt_configs_path}"
            )
        except Exception as e:
            # Log warning but don't fail initialization - user can manually call generate_prompt_config_bundle()
            logger.warning(f"Failed to auto-generate prompt config bundle: {e}")
            logger.info(
                "You can manually call generate_prompt_config_bundle() after providing missing settings"
            )

        return self

    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..contracts.bedrock_prompt_template_generation_contract import (
            BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT,
        )

        return BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT

    def get_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """
        Get script path for the Bedrock prompt template generation step.

        Args:
            default_path: Default script path to use if not found via other methods

        Returns:
            Script path resolved from processing_entry_point and source directories
        """
        # Use the parent class implementation which handles hybrid resolution
        return super().get_script_path(default_path)

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include template-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.

        Returns:
            Dict[str, Any]: Dictionary of field names to values for child initialization
        """
        # Get fields from parent class (ProcessingStepConfigBase)
        base_fields = super().get_public_init_fields()

        # Add template-specific fields (Tier 2 - System Inputs with Defaults)
        template_fields = {
            "template_task_type": self.template_task_type,
            "template_style": self.template_style,
            "validation_level": self.validation_level,
            "input_placeholders": self.input_placeholders,
            "output_format_type": self.output_format_type,
            "required_output_fields": self.required_output_fields,
            "include_examples": self.include_examples,
            "generate_validation_schema": self.generate_validation_schema,
            "template_version": self.template_version,
            # Include effective (resolved) configuration values for inheritance
            "_effective_system_prompt_config": self.effective_system_prompt_config,
            "_effective_output_format_config": self.effective_output_format_config,
            "_effective_instruction_config": self.effective_instruction_config,
        }

        # Combine base fields and template fields (template fields take precedence if overlap)
        init_fields = {**base_fields, **template_fields}

        return init_fields


def load_categories_from_json(json_data: str) -> CategoryDefinitionList:
    """
    Load categories from JSON string with validation.

    Args:
        json_data: JSON string containing category definitions

    Returns:
        Validated CategoryDefinitionList

    Raises:
        ValueError: If JSON is invalid or categories don't validate
        pydantic.ValidationError: If category data doesn't match schema
    """
    try:
        data = json.loads(json_data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if isinstance(data, list):
        # List of category dictionaries
        return CategoryDefinitionList(
            categories=[CategoryDefinition(**cat) for cat in data]
        )
    elif isinstance(data, dict):
        # Single category dictionary
        return CategoryDefinitionList(categories=[CategoryDefinition(**data)])
    else:
        raise ValueError(
            "JSON data must be a list of categories or a single category dictionary"
        )


def load_categories_from_dict(data: Any) -> CategoryDefinitionList:
    """
    Load categories from dictionary/list data with validation.

    Args:
        data: Dictionary or list containing category definitions

    Returns:
        Validated CategoryDefinitionList

    Raises:
        pydantic.ValidationError: If category data doesn't match schema
    """
    if isinstance(data, list):
        # List of category dictionaries
        return CategoryDefinitionList(
            categories=[CategoryDefinition(**cat) for cat in data]
        )
    elif isinstance(data, dict):
        # Single category dictionary
        return CategoryDefinitionList(categories=[CategoryDefinition(**data)])
    else:
        raise ValueError(
            "Data must be a list of categories or a single category dictionary"
        )
