"""
Bedrock Prompt Template Generation Step Configuration

This module implements the configuration class for the Bedrock Prompt Template Generation step
using the three-tier design pattern for optimal user experience and maintainability.
"""

from pydantic import Field, PrivateAttr, model_validator
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import logging

from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)


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

    # Advanced configuration (JSON strings for complex configurations)
    system_prompt_config: str = Field(
        default="{}",
        description="JSON configuration for system prompt customization"
    )

    output_format_config: str = Field(
        default="{}",
        description="JSON configuration for output format customization"
    )

    instruction_config: str = Field(
        default="{}",
        description="JSON configuration for instruction customization"
    )

    # Input file paths (relative to processing source directory)
    category_definitions_path: str = Field(
        default=None,
        description="Path to category definitions directory/file, relative to processing source directory"
    )

    output_schema_template_path: Optional[str] = Field(
        default=None,
        description="Path to output schema template file, relative to processing source directory (optional)"
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
    _resolved_output_schema_template_path: Optional[str] = PrivateAttr(default=None)

    # Public properties for derived fields

    @property
    def effective_system_prompt_config(self) -> Dict[str, Any]:
        """Get parsed and validated system prompt configuration."""
        if self._effective_system_prompt_config is None:
            try:
                config = json.loads(self.system_prompt_config) if self.system_prompt_config else {}
                
                # Apply defaults for missing keys
                defaults = {
                    'role_definition': 'expert analyst',
                    'expertise_areas': ['data analysis', 'classification', 'pattern recognition'],
                    'responsibilities': ['analyze data accurately', 'classify content systematically', 'provide clear reasoning'],
                    'behavioral_guidelines': ['be precise', 'be objective', 'be thorough', 'be consistent'],
                    'tone': 'professional',
                    'include_expertise_statement': True,
                    'include_task_context': True
                }
                
                self._effective_system_prompt_config = {**defaults, **config}
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid system_prompt_config JSON: {e}. Using defaults.")
                self._effective_system_prompt_config = {
                    'role_definition': 'expert analyst',
                    'expertise_areas': ['data analysis', 'classification', 'pattern recognition'],
                    'responsibilities': ['analyze data accurately', 'classify content systematically', 'provide clear reasoning'],
                    'behavioral_guidelines': ['be precise', 'be objective', 'be thorough', 'be consistent']
                }
        
        return self._effective_system_prompt_config

    @property
    def effective_output_format_config(self) -> Dict[str, Any]:
        """Get parsed and validated output format configuration."""
        if self._effective_output_format_config is None:
            try:
                config = json.loads(self.output_format_config) if self.output_format_config else {}
                
                # Apply defaults for missing keys
                defaults = {
                    'format_type': self.output_format_type,
                    'required_fields': self.required_output_fields,
                    'field_descriptions': {
                        'category': 'The classified category name (must be exactly one of the defined categories)',
                        'confidence': 'Confidence score between 0.0 and 1.0 indicating certainty of classification',
                        'key_evidence': 'Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision.',
                        'reasoning': 'Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected'
                    },
                    'validation_requirements': [
                        'category must match one of the predefined category names exactly',
                        'confidence must be a number between 0.0 and 1.0',
                        'key_evidence must align with category conditions and avoid category exceptions',
                        'key_evidence must reference specific content from the input data',
                        'reasoning must explain the logical connection between evidence and category selection'
                    ],
                    'include_field_constraints': True,
                    'include_formatting_rules': True,
                    'evidence_validation_rules': [
                        'Evidence MUST align with at least one condition for the selected category',
                        'Evidence MUST NOT match any exceptions listed for the selected category',
                        'Evidence should reference specific content from the input data',
                        'Multiple pieces of supporting evidence strengthen the classification'
                    ]
                }
                
                self._effective_output_format_config = {**defaults, **config}
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid output_format_config JSON: {e}. Using defaults.")
                self._effective_output_format_config = {
                    'format_type': self.output_format_type,
                    'required_fields': self.required_output_fields,
                    'field_descriptions': {
                        'category': 'The classified category name',
                        'confidence': 'Confidence score between 0.0 and 1.0',
                        'key_evidence': 'Specific evidence supporting the classification',
                        'reasoning': 'Clear explanation of the decision-making process'
                    }
                }
        
        return self._effective_output_format_config

    @property
    def effective_instruction_config(self) -> Dict[str, Any]:
        """Get parsed and validated instruction configuration."""
        if self._effective_instruction_config is None:
            try:
                config = json.loads(self.instruction_config) if self.instruction_config else {}
                
                # Apply defaults for missing keys
                defaults = {
                    'include_analysis_steps': True,
                    'include_decision_criteria': True,
                    'include_edge_case_handling': True,
                    'include_confidence_guidance': True,
                    'include_reasoning_requirements': True,
                    'step_by_step_format': True,
                    'include_evidence_validation': True
                }
                
                self._effective_instruction_config = {**defaults, **config}
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid instruction_config JSON: {e}. Using defaults.")
                self._effective_instruction_config = {
                    'include_analysis_steps': True,
                    'include_decision_criteria': True,
                    'include_evidence_validation': True
                }
        
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
        """Get environment variables for the processing step."""
        if self._environment_variables is None:
            self._environment_variables = {
                'TEMPLATE_TASK_TYPE': self.template_task_type,
                'TEMPLATE_STYLE': self.template_style,
                'VALIDATION_LEVEL': self.validation_level,
                'SYSTEM_PROMPT_CONFIG': self.system_prompt_config,
                'OUTPUT_FORMAT_CONFIG': self.output_format_config,
                'INSTRUCTION_CONFIG': self.instruction_config,
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

    @property
    def resolved_output_schema_template_path(self) -> Optional[str]:
        """
        Get resolved absolute path for output schema template with hybrid resolution.
        
        Returns:
            Absolute path to output schema template file, or None if not configured
            
        Raises:
            ValueError: If output_schema_template_path is set but source directory cannot be resolved
        """
        if self.output_schema_template_path is None:
            return None
        
        if self._resolved_output_schema_template_path is None:
            effective_source = self.effective_source_dir
            if effective_source is None:
                raise ValueError(
                    "Cannot resolve output_schema_template_path: no processing source directory configured. "
                    "Set either processing_source_dir or source_dir in configuration."
                )
            
            # Construct full path following same pattern as script_path
            if effective_source.startswith("s3://"):
                self._resolved_output_schema_template_path = f"{effective_source.rstrip('/')}/{self.output_schema_template_path}"
            else:
                self._resolved_output_schema_template_path = str(Path(effective_source) / self.output_schema_template_path)
        
        return self._resolved_output_schema_template_path

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
        if self.output_schema_template_path is not None:
            data["resolved_output_schema_template_path"] = self.resolved_output_schema_template_path
        
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
            "system_prompt_config": self.system_prompt_config,
            "output_format_config": self.output_format_config,
            "instruction_config": self.instruction_config,
        }
        
        # Combine base fields and template fields (template fields take precedence if overlap)
        init_fields = {**base_fields, **template_fields}
        
        return init_fields
