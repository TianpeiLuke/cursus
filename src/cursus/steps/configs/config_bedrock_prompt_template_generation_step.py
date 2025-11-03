"""
Bedrock Prompt Template Generation Step Configuration

This module implements the configuration class for the Bedrock Prompt Template Generation step
using the three-tier design pattern for optimal user experience and maintainability.
"""

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


class SystemPromptConfig(BaseModel):
    """
    Configuration for system prompt generation with comprehensive defaults.
    
    This model defines how the AI's role, expertise, and behavioral guidelines
    are structured in the system prompt component of the template.
    """
    
    role_definition: str = Field(
        default="expert analyst",
        description="The AI's primary role (e.g., 'expert analyst', 'data scientist', 'classification specialist')"
    )
    
    expertise_areas: List[str] = Field(
        default=["data analysis", "classification", "pattern recognition"],
        description="List of expertise domains the AI should demonstrate knowledge in"
    )
    
    responsibilities: List[str] = Field(
        default=["analyze data accurately", "classify content systematically", "provide clear reasoning"],
        description="List of primary tasks and responsibilities the AI should perform"
    )
    
    behavioral_guidelines: List[str] = Field(
        default=["be precise", "be objective", "be thorough", "be consistent"],
        description="List of behavioral instructions that guide the AI's approach"
    )
    
    tone: str = Field(
        default="professional",
        description="Communication tone (e.g., 'professional', 'casual', 'technical', 'formal')"
    )
    
    include_expertise_statement: bool = Field(
        default=True,
        description="Whether to include expertise areas in the system prompt"
    )
    
    include_task_context: bool = Field(
        default=True,
        description="Whether to include task context and responsibilities in the system prompt"
    )
    
    model_config = {"extra": "allow"}  # Allow additional fields for future extensibility


class OutputFormatConfig(BaseModel):
    """
    Configuration for output format generation with comprehensive defaults.
    
    This model defines the structure and validation requirements for the
    expected output format in the generated prompt template.
    """
    
    format_type: str = Field(
        default="structured_json",
        description="Type of output format ('structured_json', 'formatted_text', 'hybrid')"
    )
    
    required_fields: List[str] = Field(
        default=["category", "confidence", "key_evidence", "reasoning"],
        description="List of required fields in the output format"
    )
    
    field_descriptions: Dict[str, str] = Field(
        default_factory=lambda: {
            'category': 'The classified category name (must be exactly one of the defined categories)',
            'confidence': 'Confidence score between 0.0 and 1.0 indicating certainty of classification',
            'key_evidence': 'Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision.',
            'reasoning': 'Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected'
        },
        description="Dictionary mapping field names to their descriptions"
    )
    
    validation_requirements: List[str] = Field(
        default_factory=lambda: [
            'category must match one of the predefined category names exactly',
            'confidence must be a number between 0.0 and 1.0',
            'key_evidence must align with category conditions and avoid category exceptions',
            'key_evidence must reference specific content from the input data',
            'reasoning must explain the logical connection between evidence and category selection'
        ],
        description="List of validation requirements for the output format"
    )
    
    include_field_constraints: bool = Field(
        default=True,
        description="Whether to include field constraints in the output format specification"
    )
    
    include_formatting_rules: bool = Field(
        default=True,
        description="Whether to include formatting rules in the output format specification"
    )
    
    evidence_validation_rules: List[str] = Field(
        default_factory=lambda: [
            'Evidence MUST align with at least one condition for the selected category',
            'Evidence MUST NOT match any exceptions listed for the selected category',
            'Evidence should reference specific content from the input data',
            'Multiple pieces of supporting evidence strengthen the classification'
        ],
        description="List of specific rules for validating evidence fields"
    )
    
    model_config = {"extra": "allow"}  # Allow additional fields for future extensibility


class InstructionConfig(BaseModel):
    """
    Configuration for instruction generation with comprehensive defaults.
    
    This model defines which instruction components should be included
    in the generated prompt template to guide the AI's analysis process.
    """
    
    include_analysis_steps: bool = Field(
        default=True,
        description="Include numbered step-by-step analysis instructions"
    )
    
    include_decision_criteria: bool = Field(
        default=True,
        description="Include decision-making criteria section"
    )
    
    include_edge_case_handling: bool = Field(
        default=True,
        description="Include edge case handling guidance"
    )
    
    include_confidence_guidance: bool = Field(
        default=True,
        description="Include confidence scoring guidance"
    )
    
    include_reasoning_requirements: bool = Field(
        default=True,
        description="Include reasoning requirements and expectations"
    )
    
    step_by_step_format: bool = Field(
        default=True,
        description="Use numbered step format for analysis instructions"
    )
    
    include_evidence_validation: bool = Field(
        default=True,
        description="Include evidence validation rules and requirements"
    )
    
    model_config = {"extra": "allow"}  # Allow additional fields for future extensibility


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
        logger.warning(f"Failed to parse system_prompt_config JSON: {e}. Using defaults.")
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
        logger.warning(f"Failed to parse output_format_config JSON: {e}. Using defaults.")
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

    # No essential fields beyond base class requirements
    # The step can work with just category definitions from upstream steps

    # ===== Tier 2: System Inputs with Defaults (Optional) =====
    # These fields have sensible defaults but can be overridden

    # Template generation settings
    template_task_type: str = Field(
        default="classification",
        description="Type of task for template generation (classification, sentiment_analysis, content_moderation)"
    )

    template_style: str = Field(
        default="structured",
        description="Style of template generation (structured, conversational, technical)"
    )

    validation_level: str = Field(
        default="standard",
        description="Level of template validation (basic, standard, comprehensive)"
    )

    # Input configuration
    input_placeholders: List[str] = Field(
        default=["input_data"],
        description="List of input field names to include in the template"
    )

    # Output configuration
    output_format_type: str = Field(
        default="structured_json",
        description="Type of output format (structured_json, formatted_text, hybrid)"
    )

    required_output_fields: List[str] = Field(
        default=["category", "confidence", "key_evidence", "reasoning"],
        description="List of required fields in the output format"
    )

    # Template features
    include_examples: bool = Field(
        default=True,
        description="Include examples in the generated template"
    )

    generate_validation_schema: bool = Field(
        default=True,
        description="Generate JSON validation schema for downstream use"
    )

    template_version: str = Field(
        default="1.0",
        description="Version identifier for the generated template"
    )

    # Typed configuration fields (primary interface)
    system_prompt_settings: Optional[SystemPromptConfig] = Field(
        default=None,
        description="System prompt configuration with comprehensive defaults"
    )

    output_format_settings: Optional[OutputFormatConfig] = Field(
        default=None,
        description="Output format configuration with comprehensive defaults"
    )

    instruction_settings: Optional[InstructionConfig] = Field(
        default=None,
        description="Instruction configuration with comprehensive defaults"
    )

    # Input file paths (relative to processing source directory)
    category_definitions_path: str = Field(
        default=None,
        description="Path to category definitions directory/file, relative to processing source directory"
    )


    # Processing step overrides
    processing_entry_point: str = Field(
        default="bedrock_prompt_template_generation.py",
        description="Entry point script for prompt template generation"
    )

    # ===== Tier 3: Derived Fields (Private with Property Access) =====
    # These fields are calculated from other fields

    _effective_system_prompt_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _effective_output_format_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _effective_instruction_config: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _template_metadata: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    _environment_variables: Optional[Dict[str, str]] = PrivateAttr(default=None)
    _resolved_category_definitions_path: Optional[str] = PrivateAttr(default=None)

    # Public properties for derived fields

    @property
    def effective_system_prompt_config(self) -> Dict[str, Any]:
        """Get system prompt configuration from typed settings or defaults."""
        if self._effective_system_prompt_config is None:
            if self.system_prompt_settings is not None:
                # Use provided typed Pydantic model
                self._effective_system_prompt_config = self.system_prompt_settings.model_dump()
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
                self._effective_output_format_config = self.output_format_settings.model_dump()
                logger.debug("Using provided output_format_settings")
            else:
                # Use comprehensive defaults with integration from other fields
                default_config = OutputFormatConfig(
                    format_type=self.output_format_type,
                    required_fields=self.required_output_fields
                )
                self._effective_output_format_config = default_config.model_dump()
                logger.debug("Using default output_format_settings with field integration")
        
        return self._effective_output_format_config

    @property
    def effective_instruction_config(self) -> Dict[str, Any]:
        """Get instruction configuration from typed settings or defaults."""
        if self._effective_instruction_config is None:
            if self.instruction_settings is not None:
                # Use provided typed Pydantic model
                self._effective_instruction_config = self.instruction_settings.model_dump()
                logger.debug("Using provided instruction_settings")
            else:
                # Use comprehensive defaults
                self._effective_instruction_config = InstructionConfig().model_dump()
                logger.debug("Using default instruction_settings")
        
        return self._effective_instruction_config

    @property
    def template_metadata(self) -> Dict[str, Any]:
        """Get template generation metadata."""
        if self._template_metadata is None:
            self._template_metadata = {
                'template_version': self.template_version,
                'task_type': self.template_task_type,
                'template_style': self.template_style,
                'validation_level': self.validation_level,
                'output_format': self.output_format_type,
                'includes_examples': self.include_examples,
                'input_placeholders': self.input_placeholders,
                'required_output_fields': self.required_output_fields,
                'generate_validation_schema': self.generate_validation_schema
            }
        
        return self._template_metadata

    @property
    def environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processing step with typed config integration."""
        if self._environment_variables is None:
            # Use effective configurations (which handle typed settings > JSON strings > defaults)
            self._environment_variables = {
                'TEMPLATE_TASK_TYPE': self.template_task_type,
                'TEMPLATE_STYLE': self.template_style,
                'VALIDATION_LEVEL': self.validation_level,
                'SYSTEM_PROMPT_CONFIG': json.dumps(self.effective_system_prompt_config),
                'OUTPUT_FORMAT_CONFIG': json.dumps(self.effective_output_format_config),
                'INSTRUCTION_CONFIG': json.dumps(self.effective_instruction_config),
                'INPUT_PLACEHOLDERS': json.dumps(self.input_placeholders),
                'OUTPUT_FORMAT_TYPE': self.output_format_type,
                'REQUIRED_OUTPUT_FIELDS': json.dumps(self.required_output_fields),
                'INCLUDE_EXAMPLES': str(self.include_examples).lower(),
                'GENERATE_VALIDATION_SCHEMA': str(self.generate_validation_schema).lower(),
                'TEMPLATE_VERSION': self.template_version
            }
        
        return self._environment_variables

    @property
    def resolved_category_definitions_path(self) -> Optional[str]:
        """
        Get resolved absolute path for category definitions with hybrid resolution.
        
        Returns:
            Absolute path to category definitions file/directory, or None if not configured
            
        Raises:
            ValueError: If category_definitions_path is set but source directory cannot be resolved
        """
        if self.category_definitions_path is None:
            return None
        
        if self._resolved_category_definitions_path is None:
            effective_source = self.effective_source_dir
            if effective_source is None:
                raise ValueError(
                    "Cannot resolve category_definitions_path: no processing source directory configured. "
                    "Set either processing_source_dir or source_dir in configuration."
                )
            
            # Construct full path following same pattern as script_path
            if effective_source.startswith("s3://"):
                self._resolved_category_definitions_path = f"{effective_source.rstrip('/')}/{self.category_definitions_path}"
            else:
                self._resolved_category_definitions_path = str(Path(effective_source) / self.category_definitions_path)
        
        return self._resolved_category_definitions_path


    # Custom model_dump method to include derived properties
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include derived properties."""
        data = super().model_dump(**kwargs)
        
        # Add derived properties to output
        data["effective_system_prompt_config"] = self.effective_system_prompt_config
        data["effective_output_format_config"] = self.effective_output_format_config
        data["effective_instruction_config"] = self.effective_instruction_config
        data["template_metadata"] = self.template_metadata
        data["environment_variables"] = self.environment_variables
        
        # Add resolved path properties if they're configured
        if self.category_definitions_path is not None:
            data["resolved_category_definitions_path"] = self.resolved_category_definitions_path
        
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
        _ = self.template_metadata
        _ = self.environment_variables
        
        return self

    def get_script_contract(self):
        """Return the script contract for this step."""
        from ..contracts.bedrock_prompt_template_generation_contract import BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT
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
