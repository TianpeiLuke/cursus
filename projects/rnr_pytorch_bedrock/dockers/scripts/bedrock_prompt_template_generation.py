"""
Bedrock Prompt Template Generation Script

Generates structured prompt templates for categorization and classification tasks
following the 5-component architecture pattern for optimal LLM performance.
"""

import os
import json
import argparse
import pandas as pd
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants
CONTAINER_PATHS = {
    "INPUT_PROMPT_CONFIGS_DIR": "/opt/ml/processing/input/prompt_configs",
    "OUTPUT_TEMPLATES_DIR": "/opt/ml/processing/output/templates",
    "OUTPUT_METADATA_DIR": "/opt/ml/processing/output/metadata",
    "OUTPUT_SCHEMA_DIR": "/opt/ml/processing/output/schema",
}

# Default system prompt configuration
DEFAULT_SYSTEM_PROMPT_CONFIG = {
    "role_definition": "expert analyst",
    "expertise_areas": ["data analysis", "classification", "pattern recognition"],
    "responsibilities": [
        "analyze data accurately",
        "classify content systematically",
        "provide clear reasoning",
    ],
    "behavioral_guidelines": [
        "be precise",
        "be objective",
        "be thorough",
        "be consistent",
    ],
    "tone": "professional",
}

# Default output format configuration
DEFAULT_OUTPUT_FORMAT_CONFIG = {
    "format_type": "structured_json",
    "required_fields": ["category", "confidence", "key_evidence", "reasoning"],
    "field_descriptions": {
        "category": "The classified category name (must be exactly one of the defined categories)",
        "confidence": "Confidence score between 0.0 and 1.0 indicating certainty of classification",
        "key_evidence": "Specific evidence from input data that aligns with the selected category conditions and does NOT match any category exceptions. Reference exact content that supports the classification decision.",
        "reasoning": "Clear explanation of the decision-making process, showing how the evidence supports the selected category while considering why other categories were rejected",
    },
    "validation_requirements": [
        "category must match one of the predefined category names exactly",
        "confidence must be a number between 0.0 and 1.0",
        "key_evidence must align with category conditions and avoid category exceptions",
        "key_evidence must reference specific content from the input data",
        "reasoning must explain the logical connection between evidence and category selection",
    ],
    "evidence_validation_rules": [
        "Evidence MUST align with at least one condition for the selected category",
        "Evidence MUST NOT match any exceptions listed for the selected category",
        "Evidence should reference specific content from the input data",
        "Multiple pieces of supporting evidence strengthen the classification",
    ],
}

# Default instruction configuration
DEFAULT_INSTRUCTION_CONFIG = {
    "include_analysis_steps": True,
    "include_decision_criteria": True,
    "include_reasoning_requirements": True,
    "step_by_step_format": True,
    "include_evidence_validation": True,
}


class PlaceholderResolver:
    """
    Resolves placeholders marked with ${} syntax from various data sources.
    Tracks placeholder resolution and validates completion.

    Connects category definitions to output format through schema enrichment.
    """

    def __init__(
        self, categories: List[Dict[str, Any]], schema: Optional[Dict[str, Any]] = None
    ):
        self.categories = categories
        self.schema = schema
        self.placeholder_registry = {}  # Track all placeholders
        self.resolution_status = {}  # Track resolution success/failure

    def resolve_placeholder(
        self, placeholder: str, field_name: str, source_hint: Optional[str] = None
    ) -> str:
        """
        Resolve a placeholder marked with ${} syntax.

        Args:
            placeholder: Placeholder text (e.g., "${category_enum}")
            field_name: Field this placeholder is for (e.g., "category")
            source_hint: Optional hint about data source

        Returns:
            Resolved placeholder text
        """
        # Check if this is a dynamic placeholder
        if not placeholder or not placeholder.startswith("${"):
            return placeholder  # Literal text, no resolution needed

        # Extract placeholder name
        placeholder_name = placeholder.strip("${}")

        # Register this placeholder
        self.placeholder_registry[placeholder_name] = {
            "field_name": field_name,
            "source_hint": source_hint,
            "original": placeholder,
        }

        # Try to resolve
        try:
            resolved = self._resolve_by_strategy(
                placeholder_name, field_name, source_hint
            )
            self.resolution_status[placeholder_name] = {
                "status": "success",
                "result": resolved,
            }
            logger.info(
                f"Resolved placeholder ${{{placeholder_name}}} → {resolved[:50]}..."
            )
            return resolved
        except Exception as e:
            self.resolution_status[placeholder_name] = {
                "status": "failed",
                "error": str(e),
            }
            logger.warning(f"Failed to resolve ${{{placeholder_name}}}: {e}")
            # Fallback to descriptive placeholder
            return f"[{field_name.upper()}_UNRESOLVED]"

    def _resolve_by_strategy(
        self, placeholder_name: str, field_name: str, source_hint: Optional[str]
    ) -> str:
        """Resolve placeholder using appropriate strategy."""

        # Strategy 1: Explicit source hint
        if source_hint == "schema_enum":
            return self._resolve_from_schema_enum(field_name)
        elif source_hint == "schema_range":
            return self._resolve_from_schema_range(field_name)
        elif source_hint == "categories":
            return self._resolve_from_categories()

        # Strategy 2: Infer from placeholder name
        if "enum" in placeholder_name or "category" in placeholder_name:
            return self._resolve_from_schema_enum(field_name)
        elif "range" in placeholder_name or "numeric" in placeholder_name:
            return self._resolve_from_schema_range(field_name)

        # Strategy 3: Try schema lookup by field name
        return self._resolve_from_schema_generic(field_name)

    def _resolve_from_schema_enum(self, field_name: str) -> str:
        """Resolve from schema enum values."""
        if not self.schema:
            raise ValueError(f"No schema available for {field_name}")

        properties = self.schema.get("properties", {})
        if field_name not in properties:
            raise ValueError(f"Field {field_name} not in schema")

        field_schema = properties[field_name]
        if "enum" not in field_schema:
            raise ValueError(f"Field {field_name} has no enum in schema")

        enum_values = field_schema["enum"]
        if len(enum_values) <= 5:
            return f"One of: {', '.join(enum_values)}"
        else:
            first_few = enum_values[:3]
            return f"One of: {', '.join(first_few)}, ... (see full list above)"

    def _resolve_from_schema_range(self, field_name: str) -> str:
        """Resolve from schema numeric range."""
        if not self.schema:
            raise ValueError(f"No schema available for {field_name}")

        properties = self.schema.get("properties", {})
        if field_name not in properties:
            raise ValueError(f"Field {field_name} not in schema")

        field_schema = properties[field_name]
        field_type = field_schema.get("type")

        if field_type not in ["number", "integer"]:
            raise ValueError(f"Field {field_name} is not numeric")

        min_val = field_schema.get("minimum")
        max_val = field_schema.get("maximum")

        if min_val is None or max_val is None:
            raise ValueError(f"Field {field_name} missing min/max")

        if field_type == "number":
            return f"Number between {min_val} and {max_val} (e.g., 0.85)"
        else:
            return f"Integer between {min_val} and {max_val}"

    def _resolve_from_categories(self) -> str:
        """Resolve directly from category list."""
        if not self.categories:
            raise ValueError("No categories available")

        category_names = [cat["name"] for cat in self.categories]
        if len(category_names) <= 5:
            return f"One of: {', '.join(category_names)}"
        else:
            first_few = category_names[:3]
            return f"One of: {', '.join(first_few)}, ... (see full list above)"

    def _resolve_from_schema_generic(self, field_name: str) -> str:
        """Try generic schema-based resolution."""
        if not self.schema:
            raise ValueError(f"No schema available for {field_name}")

        properties = self.schema.get("properties", {})
        if field_name not in properties:
            raise ValueError(f"Field {field_name} not in schema")

        field_schema = properties[field_name]
        field_type = field_schema.get("type", "string")

        # Try enum first
        if "enum" in field_schema:
            return self._resolve_from_schema_enum(field_name)

        # Try numeric range
        if field_type in ["number", "integer"]:
            return self._resolve_from_schema_range(field_name)

        # Default description
        description = field_schema.get("description", f"The {field_name} value")
        return f"[{description}]"

    def validate_all_resolved(self) -> Dict[str, Any]:
        """
        Validate that all registered placeholders were successfully resolved.

        Returns:
            Validation report with any failures
        """
        report = {
            "total_placeholders": len(self.placeholder_registry),
            "successful": 0,
            "failed": 0,
            "failures": [],
        }

        for name, status in self.resolution_status.items():
            if status["status"] == "success":
                report["successful"] += 1
            else:
                report["failed"] += 1
                report["failures"].append(
                    {
                        "placeholder": name,
                        "field": self.placeholder_registry[name]["field_name"],
                        "error": status["error"],
                    }
                )

        report["all_resolved"] = report["failed"] == 0
        return report


class PromptTemplateGenerator:
    """
    Generates structured prompt templates for classification tasks using
    the 5-component architecture pattern.
    """

    def __init__(
        self, config: Dict[str, Any], schema_template: Optional[Dict[str, Any]] = None
    ):
        self.config = config
        self.categories = self._load_categories()

        # Enrich schema with category enum before creating placeholder resolver
        self.schema_template = self._enrich_schema_with_categories(schema_template)

        # Create placeholder resolver with enriched schema
        self.placeholder_resolver = PlaceholderResolver(
            self.categories, self.schema_template
        )

    def _load_categories(self) -> List[Dict[str, Any]]:
        """Load and validate category definitions from config."""
        categories = json.loads(self.config.get("category_definitions", "[]"))

        if not categories:
            raise ValueError("No category definitions provided")

        # Validate each category
        for i, category in enumerate(categories):
            required_fields = ["name", "description", "conditions", "key_indicators"]
            for field in required_fields:
                if field not in category or not category[field]:
                    raise ValueError(f"Category {i}: missing required field '{field}'")

        # Sort by priority if available
        categories.sort(key=lambda x: x.get("priority", 999))

        return categories

    def _enrich_schema_with_categories(
        self, schema: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich schema with category enum values from category definitions.
        This creates the connection between category definitions and output format.

        Args:
            schema: Original schema template

        Returns:
            Enriched schema with category enum populated
        """
        if not schema or not self.categories:
            return schema

        # Make a copy to avoid mutating the original
        enriched_schema = schema.copy() if schema else {}

        # Update category field enum if it exists
        if (
            "properties" in enriched_schema
            and "category" in enriched_schema["properties"]
        ):
            category_names = [cat["name"] for cat in self.categories]
            enriched_schema["properties"]["category"]["enum"] = category_names
            logger.info(
                f"Enriched schema with {len(category_names)} category enum values"
            )

        return enriched_schema

    def _header_to_field_name(self, header: str) -> str:
        """
        Normalize section header to match validation schema field names.

        Examples:
            "Category" → "category"
            "Confidence Score" → "confidence"
            "Key Evidence" → "key_evidence"
        """
        # Common mappings
        field_mappings = {
            "confidence score": "confidence",
            "key evidence": "key_evidence",
            # Add more as patterns emerge
        }

        normalized = header.lower()
        if normalized in field_mappings:
            return field_mappings[normalized]

        # Default: convert to snake_case
        return normalized.replace(" ", "_")

    def generate_template(self) -> Dict[str, Any]:
        """Generate complete prompt template with 5-component structure."""
        template = {
            "system_prompt": self._generate_system_prompt(),
            "user_prompt_template": self._generate_user_prompt_template(),
            "metadata": self._generate_template_metadata(),
        }

        # Validate all placeholders were resolved
        placeholder_validation = self.placeholder_resolver.validate_all_resolved()

        if not placeholder_validation["all_resolved"]:
            logger.warning(
                f"Some placeholders failed to resolve: {placeholder_validation['failures']}"
            )
        else:
            logger.info(
                f"All {placeholder_validation['successful']} placeholders resolved successfully"
            )

        # Include placeholder validation in metadata
        template["metadata"]["placeholder_validation"] = placeholder_validation

        return template

    def _generate_system_prompt(self) -> str:
        """Generate system prompt with role assignment and expertise definition."""
        # Use system prompt config loaded from JSON file
        system_config = self.config.get(
            "system_prompt_config", DEFAULT_SYSTEM_PROMPT_CONFIG
        )

        role_definition = system_config.get("role_definition")
        expertise_areas = system_config.get("expertise_areas")
        responsibilities = system_config.get("responsibilities")
        behavioral_guidelines = system_config.get("behavioral_guidelines")
        tone = system_config.get("tone", "professional")

        system_prompt_parts = []

        # Tone adjustments - modify language based on tone setting
        tone_adjustments = self._get_tone_adjustments(tone)

        # Role assignment with tone-appropriate language
        system_prompt_parts.append(
            f"{tone_adjustments['opener']} {role_definition} with extensive knowledge in {', '.join(expertise_areas)}."
        )

        # Responsibilities with tone-appropriate connector
        if responsibilities:
            system_prompt_parts.append(
                f"{tone_adjustments['task_connector']} {', '.join(responsibilities)}."
            )

        # Behavioral guidelines with tone-appropriate adverb
        if behavioral_guidelines:
            guidelines_text = ", ".join(behavioral_guidelines)
            system_prompt_parts.append(
                f"{tone_adjustments['guideline_adverb']} {guidelines_text} in your analysis."
            )

        return " ".join(system_prompt_parts)

    def _get_tone_adjustments(self, tone: str) -> Dict[str, str]:
        """
        Get tone-appropriate language adjustments.

        Args:
            tone: Desired tone (professional, casual, technical, formal)

        Returns:
            Dictionary of tone-adjusted phrases
        """
        tone_map = {
            "professional": {
                "opener": "You are an",
                "task_connector": "Your task is to",
                "guideline_adverb": "Always",
            },
            "casual": {
                "opener": "Hey! You're a",
                "task_connector": "Your job is to",
                "guideline_adverb": "Make sure to",
            },
            "technical": {
                "opener": "System role: You are a",
                "task_connector": "Core functions include:",
                "guideline_adverb": "Operational guidelines require:",
            },
            "formal": {
                "opener": "You shall function as an",
                "task_connector": "Your responsibilities encompass:",
                "guideline_adverb": "You must consistently",
            },
        }

        return tone_map.get(tone.lower(), tone_map["professional"])

    def _generate_user_prompt_template(self) -> str:
        """Generate user prompt template with all 5 components."""
        components = []

        # Component 1: System prompt (already handled separately)

        # Component 2: Category definitions
        components.append(self._generate_category_definitions_section())

        # Component 3: Input placeholders
        components.append(self._generate_input_placeholders_section())

        # Component 4: Instructions and rules
        components.append(self._generate_instructions_section())

        # Component 5: Output format schema
        components.append(self._generate_output_format_section())

        return "\n\n".join(components)

    def _generate_category_definitions_section(self) -> str:
        """Generate category definitions with conditions and exceptions."""
        section_parts = ["Categories and their criteria:"]

        for i, category in enumerate(self.categories, 1):
            category_parts = [f"\n{i}. {category['name']}"]

            # Description
            if category.get("description"):
                category_parts.append(f"    - {category['description']}")

            # Key elements/indicators
            if category.get("key_indicators"):
                category_parts.append("    - Key elements:")
                for indicator in category["key_indicators"]:
                    category_parts.append(f"        * {indicator}")

            # Conditions
            if category.get("conditions"):
                category_parts.append("    - Conditions:")
                for condition in category["conditions"]:
                    category_parts.append(f"        * {condition}")

            # Exceptions
            if category.get("exceptions"):
                category_parts.append("    - Must NOT include:")
                for exception in category["exceptions"]:
                    category_parts.append(f"        * {exception}")

            # Examples if available
            if (
                category.get("examples")
                and self.config.get("INCLUDE_EXAMPLES", "true").lower() == "true"
            ):
                category_parts.append("    - Examples:")
                for example in category["examples"]:
                    category_parts.append(f"        * {example}")

            section_parts.append("\n".join(category_parts))

        return "\n".join(section_parts)

    def _generate_input_placeholders_section(self) -> str:
        """Generate input placeholders section."""
        placeholders = json.loads(
            self.config.get("INPUT_PLACEHOLDERS", '["input_data"]')
        )

        section_parts = ["Analysis Instructions:", ""]
        section_parts.append("Please analyze:")

        for placeholder in placeholders:
            section_parts.append(f"{placeholder.title()}: {{{placeholder}}}")

        return "\n".join(section_parts)

    def _generate_instructions_section(self) -> str:
        """
        Generate instructions and rules section.
        Supports both basic boolean flags and detailed classification guidelines.
        """
        # Use instruction config loaded from JSON file
        instruction_config = self.config.get(
            "instruction_config", DEFAULT_INSTRUCTION_CONFIG
        )

        instructions = ["Provide your analysis in the following structured format:", ""]

        # Analysis steps with format control
        if instruction_config.get("include_analysis_steps", True):
            use_step_by_step = instruction_config.get("step_by_step_format", True)
            analysis_steps = [
                "Carefully review all provided data",
                "Identify key patterns and indicators",
                "Match against category criteria",
                "Select the most appropriate category",
                "Validate evidence against conditions and exceptions",
                "Provide confidence assessment and reasoning",
            ]

            if use_step_by_step:
                # Numbered format
                instructions.extend(
                    [f"{i + 1}. {step}" for i, step in enumerate(analysis_steps)]
                )
            else:
                # Bullet point format
                instructions.extend([f"- {step}" for step in analysis_steps])
            instructions.append("")

        # Decision criteria section
        if instruction_config.get("include_decision_criteria", True):
            instructions.extend(
                [
                    "Decision Criteria:",
                    "- Base decisions on explicit evidence in the data",
                    "- Consider all category conditions and exceptions",
                    "- Choose the category with the strongest evidence match",
                    "- Provide clear reasoning for your classification",
                    "",
                ]
            )

        # Reasoning requirements section (NEW)
        if instruction_config.get("include_reasoning_requirements", True):
            instructions.extend(
                [
                    "Reasoning Requirements:",
                    "- Explain WHY the evidence supports the selected category",
                    "- Address HOW the evidence aligns with category conditions",
                    "- Clarify WHAT makes this category the best match",
                    "- Describe WHY other categories were ruled out (if applicable)",
                    "",
                ]
            )

        # Evidence validation section
        if instruction_config.get("include_evidence_validation", True):
            instructions.extend(
                [
                    "Key Evidence Validation:",
                    "- Evidence MUST align with at least one condition for the selected category",
                    "- Evidence MUST NOT match any exceptions listed for the selected category",
                    "- Evidence should reference specific content from the input data",
                    "- Multiple pieces of supporting evidence strengthen the classification",
                    "",
                ]
            )

        # Detailed classification guidelines (from config structure)
        classification_guidelines = instruction_config.get("classification_guidelines")
        if classification_guidelines:
            guidelines_text = self._generate_classification_guidelines(
                classification_guidelines
            )
            if guidelines_text:
                instructions.extend([guidelines_text, ""])

        return "\n".join(instructions)

    def _generate_classification_guidelines(self, guidelines: Dict[str, Any]) -> str:
        """
        Generate detailed classification guidelines from config structure.

        Args:
            guidelines: Dictionary containing sections with hierarchical structure

        Returns:
            Formatted guideline text
        """
        guideline_parts = []

        sections = guidelines.get("sections", [])
        for section in sections:
            # Add main section title
            section_title = section.get("title", "")
            if section_title:
                guideline_parts.append(section_title)
                guideline_parts.append("")

            # Add subsections
            subsections = section.get("subsections", [])
            for subsection in subsections:
                # Add subsection title
                subsection_title = subsection.get("title", "")
                if subsection_title:
                    guideline_parts.append(subsection_title)
                    guideline_parts.append("")

                # Add subsection content
                content = subsection.get("content", [])
                if content:
                    guideline_parts.extend(content)
                    guideline_parts.append("")

        return "\n".join(guideline_parts)

    def _generate_output_format_section(self) -> str:
        """Generate output format schema section based on format_type."""
        # Use output format config loaded from JSON file
        output_config = self.config.get(
            "output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG
        )
        format_type = output_config.get("format_type", "structured_json")

        if format_type == "structured_text":
            return self._generate_structured_text_output_format_from_config()
        else:
            # Default to JSON schema-based generation
            return self._generate_custom_output_format_from_schema()

    def _generate_structured_text_output_format_from_config(self) -> str:
        """
        Generate structured text output format from configuration.
        Fully driven by output_format.json configuration - no hard-coding.
        """
        output_config = self.config.get(
            "output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG
        )

        format_parts = ["## Required Output Format", ""]

        # Use header text from config (like structured_text does)
        header_text = output_config.get(
            "header_text",
            "**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**",
        )
        # Ensure header_text is not None
        if header_text:
            format_parts.append(header_text)
            format_parts.append("")

        # Generate example structure from config
        structured_text_sections = output_config.get("structured_text_sections", [])

        if structured_text_sections:
            format_parts.append("```")
            for section in structured_text_sections:
                section_lines = self._generate_section_from_config(section)
                format_parts.extend(section_lines)
            format_parts.append("```")
            format_parts.append("")

        # Add field descriptions if provided
        field_descriptions = output_config.get("field_descriptions", {})
        if field_descriptions:
            format_parts.append("**Field Descriptions:**")
            for field, description in field_descriptions.items():
                format_parts.append(f"- **{field}**: {description}")
            format_parts.append("")

        # Add formatting rules if provided
        formatting_rules = output_config.get("formatting_rules", [])
        if formatting_rules:
            format_parts.append("**Formatting Rules:**")
            for rule in formatting_rules:
                format_parts.append(f"- {rule}")
            format_parts.append("")

        # Add validation requirements if provided
        validation_requirements = output_config.get("validation_requirements", [])
        if validation_requirements:
            format_parts.append("**Validation Requirements:**")
            for req in validation_requirements:
                format_parts.append(f"- {req}")
            format_parts.append("")

        # Add evidence validation rules if provided
        evidence_validation_rules = output_config.get("evidence_validation_rules", [])
        if evidence_validation_rules:
            format_parts.append("**Evidence Validation:**")
            for rule in evidence_validation_rules:
                format_parts.append(f"- {rule}")
            format_parts.append("")

        # Add example output if provided in config
        example_output = output_config.get("example_output")
        if example_output:
            format_parts.append("**Example Output:**")
            format_parts.append("")
            format_parts.append("```")
            if isinstance(example_output, str):
                format_parts.append(example_output)
            elif isinstance(example_output, list):
                format_parts.extend(example_output)
            format_parts.append("```")

        return "\n".join(format_parts)

    def _generate_section_from_config(self, section: Dict[str, Any]) -> List[str]:
        """
        Generate a section's text from its configuration.
        Supports flexible section formats defined in config.
        Uses PlaceholderResolver to dynamically fill placeholders from categories/schema.
        """
        lines = []

        number = section.get("number", "")
        header = section.get("header", "")
        section_format = section.get("format", "single_value")
        placeholder = section.get(
            "placeholder", f"[{header.upper().replace(' ', '_')}]"
        )
        placeholder_source = section.get("placeholder_source")

        # Normalize header to field name for schema lookup
        field_name = self._header_to_field_name(header)

        # Resolve placeholder using PlaceholderResolver
        resolved_placeholder = self.placeholder_resolver.resolve_placeholder(
            placeholder, field_name, placeholder_source
        )

        # Generate section header with resolved placeholder
        if number:
            lines.append(f"{number}. {header}: {resolved_placeholder}")
        else:
            lines.append(f"{header}: {resolved_placeholder}")
        lines.append("")

        # Handle subsections if present
        if section_format == "subsections":
            subsections = section.get("subsections", [])
            item_prefix = section.get("item_prefix", "[sep] ")
            indent = section.get("indent", "   ")

            for subsection in subsections:
                # Subsection can be a string or dict
                if isinstance(subsection, str):
                    subsection_header = subsection
                    subsection_items = section.get(
                        "subsection_example_items",
                        [f"{item_prefix}[Item 1]", f"{item_prefix}[Item 2]"],
                    )
                else:
                    subsection_header = subsection.get("name", "")
                    subsection_items = subsection.get(
                        "example_items", [f"{item_prefix}[Item 1]"]
                    )

                lines.append(f"{indent}* {subsection_header}:")
                for item in subsection_items:
                    lines.append(f"{indent}  {item}")
            lines.append("")

        return lines

    def _generate_custom_output_format_from_schema(self) -> str:
        """Generate output format section from custom JSON schema template with full output_config support."""
        schema = self.schema_template
        output_config = self.config.get(
            "output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG
        )

        format_parts = ["## Required Output Format", ""]

        # Use header text from config (like structured_text does)
        header_text = output_config.get(
            "header_text",
            "**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**",
        )
        # Ensure header_text is not None
        if header_text:
            format_parts.append(header_text)
            format_parts.append("")

        # Check if example_output is provided as dict
        example_output = output_config.get("example_output")
        use_real_example = isinstance(example_output, dict)

        if use_real_example:
            # Use the provided example directly (without markdown wrappers to prevent Claude from mimicking them)
            format_parts.extend(
                [
                    json.dumps(example_output, indent=2, ensure_ascii=False),
                    "",
                ]
            )
            logger.info("Using provided example_output dict for JSON format")
        else:
            # Generate placeholder structure from schema (without markdown wrappers to prevent Claude from mimicking them)
            format_parts.extend(["{"])

            # Extract properties from schema
            properties = schema.get("properties", {})
            required_fields = schema.get("required", list(properties.keys()))

            # Generate JSON structure from schema
            for i, field in enumerate(required_fields):
                field_schema = properties.get(field, {})
                field_type = field_schema.get("type", "string")
                description = field_schema.get("description", f"The {field} value")

                # Generate example value based on type
                if field_type == "string":
                    if "enum" in field_schema:
                        example_value = f"One of: {', '.join(field_schema['enum'])}"
                    else:
                        example_value = description
                elif field_type == "number":
                    min_val = field_schema.get("minimum", 0)
                    max_val = field_schema.get("maximum", 1)
                    example_value = f"Number between {min_val} and {max_val}"
                elif field_type == "array":
                    example_value = "Array of values"
                elif field_type == "boolean":
                    example_value = "true or false"
                else:
                    example_value = description

                comma = "," if i < len(required_fields) - 1 else ""
                format_parts.append(f'    "{field}": "{example_value}"{comma}')

            format_parts.extend(["}", ""])
            logger.info(
                "Generated placeholder JSON structure from schema (no example_output provided)"
            )

        format_parts.append("Field Descriptions:")

        # Get field descriptions from config (prefer) or schema (fallback)
        field_descriptions = output_config.get("field_descriptions", {})
        properties = schema.get("properties", {})
        required_fields = schema.get("required", list(properties.keys()))

        # Add detailed field descriptions
        for field in required_fields:
            field_schema = properties.get(field, {})

            # Prefer config description, fallback to schema
            if field in field_descriptions:
                description = field_descriptions[field]
            else:
                description = field_schema.get("description", f"The {field} value")

            field_type = field_schema.get("type", "string")

            # Add type and constraint information
            constraints = []
            if field_type == "number":
                if "minimum" in field_schema:
                    constraints.append(f"minimum: {field_schema['minimum']}")
                if "maximum" in field_schema:
                    constraints.append(f"maximum: {field_schema['maximum']}")
            elif field_type == "string" and "enum" in field_schema:
                constraints.append(f"must be one of: {', '.join(field_schema['enum'])}")

            constraint_text = f" ({', '.join(constraints)})" if constraints else ""
            format_parts.append(
                f"- **{field}** ({field_type}): {description}{constraint_text}"
            )

        format_parts.append("")

        # Add category-specific validation if category field exists
        if "category" in required_fields and properties.get("category", {}).get("enum"):
            category_names = properties["category"]["enum"]
            format_parts.extend(
                [
                    "**Category Validation:**",
                    f"- The category field must exactly match one of: {', '.join(category_names)}",
                    "- Category names are case-sensitive and must match exactly",
                    "",
                ]
            )

        # Add formatting rules from config (like structured_text does)
        formatting_rules = output_config.get("formatting_rules", [])
        if formatting_rules:
            format_parts.append("**Formatting Rules:**")
            for rule in formatting_rules:
                format_parts.append(f"- {rule}")
            format_parts.append("")

        # Add validation requirements from config (like structured_text does)
        validation_requirements = output_config.get("validation_requirements", [])
        if validation_requirements:
            format_parts.append("**Validation Requirements:**")
            for req in validation_requirements:
                format_parts.append(f"- {req}")
            format_parts.append("")

        # Add evidence validation rules from config (like structured_text does)
        evidence_validation_rules = output_config.get("evidence_validation_rules", [])
        if evidence_validation_rules:
            format_parts.append("**Evidence Validation:**")
            for rule in evidence_validation_rules:
                format_parts.append(f"- {rule}")
            format_parts.append("")

        format_parts.extend(
            [
                "Do not include any text before or after the JSON object. Only return valid JSON.",
            ]
        )

        return "\n".join(format_parts)

    def _generate_template_metadata(self) -> Dict[str, Any]:
        """Generate metadata about the template."""
        return {
            "template_version": self.config.get("TEMPLATE_VERSION", "1.0"),
            "generation_timestamp": datetime.now().isoformat(),
            "task_type": self.config.get("TEMPLATE_TASK_TYPE", "classification"),
            "template_style": self.config.get("TEMPLATE_STYLE", "structured"),
            "category_count": len(self.categories),
            "category_names": [cat["name"] for cat in self.categories],
            "output_format": self.config.get("output_format_config", {}).get(
                "format_type", "structured_json"
            ),
            "validation_level": self.config.get("VALIDATION_LEVEL", "standard"),
            "includes_examples": self.config.get("INCLUDE_EXAMPLES", "true").lower()
            == "true",
            "generator_config": {
                "system_prompt_config": self.config.get("system_prompt_config", {}),
                "output_format_config": self.config.get("output_format_config", {}),
                "instruction_config": self.config.get("instruction_config", {}),
            },
        }


class TemplateValidator:
    """Validates generated prompt templates for quality and completeness."""

    def __init__(self, validation_level: str = "standard"):
        self.validation_level = validation_level

    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template and return validation results."""
        validation_results = {
            "is_valid": True,
            "quality_score": 0.0,
            "validation_details": [],
            "recommendations": [],
        }

        # Validate system prompt
        system_validation = self._validate_system_prompt(
            template.get("system_prompt", "")
        )
        validation_results["validation_details"].append(system_validation)

        # Validate user prompt template
        user_validation = self._validate_user_prompt_template(
            template.get("user_prompt_template", "")
        )
        validation_results["validation_details"].append(user_validation)

        # Validate metadata
        metadata_validation = self._validate_metadata(template.get("metadata", {}))
        validation_results["validation_details"].append(metadata_validation)

        # Calculate overall quality score
        scores = [v["score"] for v in validation_results["validation_details"]]
        validation_results["quality_score"] = (
            sum(scores) / len(scores) if scores else 0.0
        )

        # Determine overall validity
        validation_results["is_valid"] = all(
            v["is_valid"] for v in validation_results["validation_details"]
        )

        # Generate recommendations
        validation_results["recommendations"] = self._generate_recommendations(
            validation_results["validation_details"]
        )

        return validation_results

    def _validate_system_prompt(self, system_prompt: str) -> Dict[str, Any]:
        """Validate system prompt component."""
        result = {
            "component": "system_prompt",
            "is_valid": True,
            "score": 0.0,
            "issues": [],
        }

        if not system_prompt or not system_prompt.strip():
            result["is_valid"] = False
            result["issues"].append("System prompt is empty")
            result["score"] = 0.0
            return result

        score = 0.0

        # Check for role definition
        if any(
            word in system_prompt.lower()
            for word in ["you are", "expert", "analyst", "specialist"]
        ):
            score += 0.3
        else:
            result["issues"].append("Missing clear role definition")

        # Check for expertise areas
        if any(
            word in system_prompt.lower()
            for word in ["knowledge", "experience", "expertise"]
        ):
            score += 0.2
        else:
            result["issues"].append("Missing expertise statement")

        # Check for task context
        if any(
            word in system_prompt.lower()
            for word in ["task", "analyze", "classify", "categorize"]
        ):
            score += 0.3
        else:
            result["issues"].append("Missing task context")

        # Check for behavioral guidelines
        if any(
            word in system_prompt.lower()
            for word in ["precise", "objective", "thorough", "accurate"]
        ):
            score += 0.2
        else:
            result["issues"].append("Missing behavioral guidelines")

        result["score"] = score
        if score < 0.7:
            result["is_valid"] = False

        return result

    def _validate_user_prompt_template(self, user_prompt: str) -> Dict[str, Any]:
        """Validate user prompt template component."""
        result = {
            "component": "user_prompt_template",
            "is_valid": True,
            "score": 0.0,
            "issues": [],
        }

        if not user_prompt or not user_prompt.strip():
            result["is_valid"] = False
            result["issues"].append("User prompt template is empty")
            result["score"] = 0.0
            return result

        score = 0.0

        # Check for category definitions
        if "categories" in user_prompt.lower() and "criteria" in user_prompt.lower():
            score += 0.25
        else:
            result["issues"].append("Missing category definitions section")

        # Check for input placeholders
        if "{" in user_prompt and "}" in user_prompt:
            score += 0.25
        else:
            result["issues"].append("Missing input placeholders")

        # Check for instructions
        if any(
            word in user_prompt.lower()
            for word in ["analyze", "instructions", "provide", "format"]
        ):
            score += 0.25
        else:
            result["issues"].append("Missing analysis instructions")

        # Check for output format
        if any(
            word in user_prompt.lower()
            for word in ["json", "format", "structure", "output"]
        ):
            score += 0.25
        else:
            result["issues"].append("Missing output format specification")

        result["score"] = score
        if score < 0.7:
            result["is_valid"] = False

        return result

    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template metadata."""
        result = {"component": "metadata", "is_valid": True, "score": 1.0, "issues": []}

        required_fields = [
            "template_version",
            "generation_timestamp",
            "task_type",
            "category_count",
        ]
        missing_fields = [field for field in required_fields if field not in metadata]

        if missing_fields:
            result["issues"].append(
                f"Missing metadata fields: {', '.join(missing_fields)}"
            )
            result["score"] = max(0.0, 1.0 - (len(missing_fields) * 0.2))
            if len(missing_fields) > 2:
                result["is_valid"] = False

        return result

    def _generate_recommendations(
        self, validation_details: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        for detail in validation_details:
            if detail["score"] < 0.8:
                component = detail["component"]
                recommendations.append(
                    f"Improve {component}: {'; '.join(detail['issues'])}"
                )

        return recommendations


def _generate_processing_config(config: Dict[str, str]) -> Dict[str, Any]:
    """Generate processing configuration metadata (non-redundant)."""
    return {
        "format_type": config.get("output_format_config", {}).get(
            "format_type", "structured_json"
        ),
        "response_model_name": f"{config.get('TEMPLATE_TASK_TYPE', 'classification').title()}Response",
        "validation_level": config.get("VALIDATION_LEVEL", "standard"),
    }


def load_config_from_json_file(
    config_path: str,
    config_name: str,
    default_config: Dict[str, Any],
    log: Callable[[str], None],
) -> Dict[str, Any]:
    """Load configuration from JSON file with fallback to defaults."""
    config_file = Path(config_path) / f"{config_name}.json"

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                log(f"Loaded {config_name} config from {config_file}")
                return {**default_config, **config}  # Merge with defaults
        except Exception as e:
            log(
                f"Failed to load {config_name} config from {config_file}: {e}. Using defaults."
            )
            return default_config
    else:
        log(f"{config_name} config file not found at {config_file}. Using defaults.")
        return default_config


def load_category_definitions(
    prompt_configs_path: str, log: Callable[[str], None]
) -> List[Dict[str, Any]]:
    """Load category definitions from prompt configs directory."""
    config_dir = Path(prompt_configs_path)

    if not config_dir.exists():
        log(f"Prompt configs directory not found: {prompt_configs_path}")
        return []

    # Load category_definitions.json
    categories_file = config_dir / "category_definitions.json"
    if categories_file.exists():
        try:
            with open(categories_file, "r", encoding="utf-8") as f:
                categories = json.load(f)
                log(f"Loaded category definitions from {categories_file}")
                return categories if isinstance(categories, list) else [categories]
        except Exception as e:
            log(f"Failed to load category definitions from {categories_file}: {e}")
            return []
    else:
        log(f"Category definitions file not found: {categories_file}")
        return []


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for prompt template generation, refactored for testability.

    Args:
        input_paths: Dictionary of input paths with logical names
        output_paths: Dictionary of output paths with logical names
        environ_vars: Dictionary of environment variables
        job_args: Command line arguments
        logger: Optional logger object (defaults to print if None)

    Returns:
        Dictionary containing generation results and statistics
    """
    # Use print function if no logger is provided
    log = logger or print

    try:
        # Load configurations from JSON files in prompt_configs directory
        prompt_configs_path = input_paths.get("prompt_configs")
        if not prompt_configs_path:
            raise ValueError("No prompt_configs input path provided")

        # Load category definitions
        categories = load_category_definitions(prompt_configs_path, log)
        if not categories:
            raise ValueError("No category definitions found in prompt configs")

        # Load configuration files from prompt_configs directory
        system_prompt_config = load_config_from_json_file(
            prompt_configs_path, "system_prompt", DEFAULT_SYSTEM_PROMPT_CONFIG, log
        )

        output_format_config = load_config_from_json_file(
            prompt_configs_path, "output_format", DEFAULT_OUTPUT_FORMAT_CONFIG, log
        )

        instruction_config = load_config_from_json_file(
            prompt_configs_path, "instruction", DEFAULT_INSTRUCTION_CONFIG, log
        )

        # Generate schema template from output format config
        schema_template = None

        # Priority 1: Check for json_schema field in output_format_config
        if (
            output_format_config
            and "json_schema" in output_format_config
            and output_format_config["json_schema"]
        ):
            schema_template = output_format_config["json_schema"]
            log(
                "Using json_schema from OutputFormatConfig for validation schema generation"
            )
        # Priority 2: Check if output_format_config itself is a JSON schema (backward compatibility)
        elif output_format_config and "type" in output_format_config:
            # Output format config contains a JSON schema
            schema_template = output_format_config
            log(
                "Using JSON schema from output_format.json for format generation (legacy format)"
            )
        else:
            # Generate default schema template
            required_fields = output_format_config.get(
                "required_fields",
                ["category", "confidence", "key_evidence", "reasoning"],
            )
            field_descriptions = output_format_config.get("field_descriptions", {})

            schema_template = {
                "type": "object",
                "properties": {},
                "required": required_fields,
                "additionalProperties": False,
            }

            # Generate properties from config
            for field in required_fields:
                if field == "confidence":
                    schema_template["properties"][field] = {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": field_descriptions.get(
                            field, "Confidence score between 0.0 and 1.0"
                        ),
                    }
                elif field == "category":
                    schema_template["properties"][field] = {
                        "type": "string",
                        "enum": [cat["name"] for cat in categories],
                        "description": field_descriptions.get(
                            field, "The classified category name"
                        ),
                    }
                else:
                    schema_template["properties"][field] = {
                        "type": "string",
                        "description": field_descriptions.get(
                            field, f"The {field} value"
                        ),
                    }

            log("Generated default output schema template from output_format.json")

        # Update category enum in schema if it has a category field
        if (
            "properties" in schema_template
            and "category" in schema_template["properties"]
            and schema_template["properties"]["category"].get("type") == "string"
        ):
            schema_template["properties"]["category"]["enum"] = [
                cat["name"] for cat in categories
            ]

        # Build configuration from environment variables and loaded JSON configs
        # Use JSON config values where available, fall back to environment variables
        config = {
            "TEMPLATE_TASK_TYPE": environ_vars.get(
                "TEMPLATE_TASK_TYPE", "classification"
            ),
            "TEMPLATE_STYLE": environ_vars.get("TEMPLATE_STYLE", "structured"),
            "VALIDATION_LEVEL": environ_vars.get("VALIDATION_LEVEL", "standard"),
            "category_definitions": json.dumps(categories),
            "system_prompt_config": system_prompt_config,
            "output_format_config": output_format_config,
            "instruction_config": instruction_config,
            "INPUT_PLACEHOLDERS": environ_vars.get(
                "INPUT_PLACEHOLDERS", '["input_data"]'
            ),
            # Values now come from JSON config files, no longer needed as separate config keys
            "INCLUDE_EXAMPLES": environ_vars.get("INCLUDE_EXAMPLES", "true"),
            "GENERATE_VALIDATION_SCHEMA": environ_vars.get(
                "GENERATE_VALIDATION_SCHEMA", "true"
            ),
            "TEMPLATE_VERSION": environ_vars.get("TEMPLATE_VERSION", "1.0"),
        }

        # Initialize template generator with schema template (default or custom)
        generator = PromptTemplateGenerator(config, schema_template)

        # Generate template
        log("Generating prompt template...")
        template = generator.generate_template()

        # Validate template
        validator = TemplateValidator(config["VALIDATION_LEVEL"])
        validation_results = validator.validate_template(template)

        log(
            f"Template validation completed. Quality score: {validation_results['quality_score']:.2f}"
        )

        # Create output directories
        templates_path = Path(output_paths["prompt_templates"])
        metadata_path = Path(output_paths["template_metadata"])
        schema_path = Path(output_paths["validation_schema"])

        templates_path.mkdir(parents=True, exist_ok=True)
        metadata_path.mkdir(parents=True, exist_ok=True)
        schema_path.mkdir(parents=True, exist_ok=True)

        # Save generated template
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save prompts.json (main template file)
        prompts_file = templates_path / "prompts.json"
        template_output = {
            "system_prompt": template["system_prompt"],
            "user_prompt_template": template["user_prompt_template"],
            "input_placeholders": json.loads(
                config.get("INPUT_PLACEHOLDERS", '["input_data"]')
            ),
        }

        with open(prompts_file, "w", encoding="utf-8") as f:
            json.dump(template_output, f, indent=2, ensure_ascii=False)

        log(f"Saved prompt template to: {prompts_file}")

        # Save template metadata
        metadata_file = metadata_path / f"template_metadata_{timestamp}.json"
        metadata_output = {
            **template["metadata"],
            "validation_results": validation_results,
            "generation_config": {
                "task_type": config["TEMPLATE_TASK_TYPE"],
                "template_style": config["TEMPLATE_STYLE"],
                "validation_level": config["VALIDATION_LEVEL"],
                "category_count": len(categories),
            },
        }

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata_output, f, indent=2, ensure_ascii=False, default=str)

        log(f"Saved template metadata to: {metadata_file}")

        # Generate and save validation schema if requested
        if config["GENERATE_VALIDATION_SCHEMA"].lower() == "true":
            schema_file = schema_path / f"validation_schema_{timestamp}.json"

            # Use custom schema template if available, otherwise generate default schema
            if schema_template:
                # Use the custom schema template directly
                validation_schema = schema_template.copy()

                # Update category enum if it exists in the schema
                if (
                    "properties" in validation_schema
                    and "category" in validation_schema["properties"]
                    and validation_schema["properties"]["category"].get("type")
                    == "string"
                ):
                    validation_schema["properties"]["category"]["enum"] = [
                        cat["name"] for cat in categories
                    ]

                log("Using custom schema template for validation schema generation")
            else:
                # Generate default JSON schema for output validation
                required_fields = output_format_config.get(
                    "required_fields",
                    ["category", "confidence", "key_evidence", "reasoning"],
                )
                validation_schema = {
                    "type": "object",
                    "properties": {},
                    "required": required_fields,
                    "additionalProperties": False,
                }

                # Add field definitions
                field_descriptions = config.get("output_format_config", {}).get(
                    "field_descriptions", {}
                )
                for field in required_fields:
                    if field == "confidence":
                        validation_schema["properties"][field] = {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": field_descriptions.get(
                                field, "Confidence score between 0.0 and 1.0"
                            ),
                        }
                    elif field == "category":
                        validation_schema["properties"][field] = {
                            "type": "string",
                            "enum": [cat["name"] for cat in categories],
                            "description": field_descriptions.get(
                                field, "The classified category name"
                            ),
                        }
                    else:
                        validation_schema["properties"][field] = {
                            "type": "string",
                            "description": field_descriptions.get(
                                field, f"The {field} value"
                            ),
                        }

                log("Generated default validation schema")

            # Enhance validation schema with processing metadata for Bedrock Processing step integration
            enhanced_validation_schema = {
                "title": "Bedrock Response Validation Schema",
                "description": "Schema for validating Bedrock LLM responses with processing metadata",
                **validation_schema,
                # Processing metadata for Bedrock Processing step
                "processing_config": _generate_processing_config(config),
                # Template integration metadata
                "template_metadata": {
                    "template_version": config.get("TEMPLATE_VERSION", "1.0"),
                    "generation_timestamp": timestamp,
                    "category_count": len(categories),
                    "category_names": [cat["name"] for cat in categories],
                    "output_format_source": "output_format.json",
                    "task_type": config.get("TEMPLATE_TASK_TYPE", "classification"),
                    "template_style": config.get("TEMPLATE_STYLE", "structured"),
                },
            }

            with open(schema_file, "w", encoding="utf-8") as f:
                json.dump(enhanced_validation_schema, f, indent=2, ensure_ascii=False)

            log(
                f"Saved enhanced validation schema with processing metadata to: {schema_file}"
            )

        # Prepare results summary
        results = {
            "success": True,
            "template_generated": True,
            "validation_passed": validation_results["is_valid"],
            "quality_score": validation_results["quality_score"],
            "category_count": len(categories),
            "template_version": config["TEMPLATE_VERSION"],
            "output_files": {
                "prompts": str(prompts_file),
                "metadata": str(metadata_file),
                "schema": str(schema_file)
                if config["GENERATE_VALIDATION_SCHEMA"].lower() == "true"
                else None,
            },
            "validation_details": validation_results,
            "generation_timestamp": datetime.now().isoformat(),
        }

        log(f"Template generation completed successfully")
        log(f"Quality score: {validation_results['quality_score']:.2f}")
        log(f"Categories processed: {len(categories)}")

        return results

    except Exception as e:
        log(f"Template generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # Argument parser
        parser = argparse.ArgumentParser(
            description="Bedrock prompt template generation script"
        )
        parser.add_argument(
            "--include-examples",
            action="store_true",
            help="Include examples in template",
        )
        parser.add_argument(
            "--generate-validation-schema",
            action="store_true",
            help="Generate validation schema",
        )
        parser.add_argument(
            "--template-version", default="1.0", help="Template version identifier"
        )

        args = parser.parse_args()

        # Set up path dictionaries
        input_paths = {"prompt_configs": CONTAINER_PATHS["INPUT_PROMPT_CONFIGS_DIR"]}

        output_paths = {
            "prompt_templates": CONTAINER_PATHS["OUTPUT_TEMPLATES_DIR"],
            "template_metadata": CONTAINER_PATHS["OUTPUT_METADATA_DIR"],
            "validation_schema": CONTAINER_PATHS["OUTPUT_SCHEMA_DIR"],
        }

        # Environment variables dictionary (streamlined - no large JSON configs)
        environ_vars = {
            "TEMPLATE_TASK_TYPE": os.environ.get(
                "TEMPLATE_TASK_TYPE", "classification"
            ),
            "TEMPLATE_STYLE": os.environ.get("TEMPLATE_STYLE", "structured"),
            "VALIDATION_LEVEL": os.environ.get("VALIDATION_LEVEL", "standard"),
            "INPUT_PLACEHOLDERS": os.environ.get(
                "INPUT_PLACEHOLDERS", '["input_data"]'
            ),
            "INCLUDE_EXAMPLES": os.environ.get(
                "INCLUDE_EXAMPLES", str(args.include_examples).lower()
            ),
            "GENERATE_VALIDATION_SCHEMA": os.environ.get(
                "GENERATE_VALIDATION_SCHEMA",
                str(args.generate_validation_schema).lower(),
            ),
            "TEMPLATE_VERSION": os.environ.get(
                "TEMPLATE_VERSION", args.template_version
            ),
        }

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Log key parameters
        logger.info(f"Starting prompt template generation with parameters:")
        logger.info(f"  Task Type: {environ_vars['TEMPLATE_TASK_TYPE']}")
        logger.info(f"  Template Style: {environ_vars['TEMPLATE_STYLE']}")
        logger.info(f"  Validation Level: {environ_vars['VALIDATION_LEVEL']}")
        logger.info(f"  Include Examples: {environ_vars['INCLUDE_EXAMPLES']}")
        logger.info(f"  Generate Schema: {environ_vars['GENERATE_VALIDATION_SCHEMA']}")
        logger.info(f"  Template Version: {environ_vars['TEMPLATE_VERSION']}")

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary
        logger.info(
            f"Prompt template generation completed successfully. Results: {result}"
        )
        sys.exit(0)

    except Exception as e:
        logging.error(f"Error in prompt template generation script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
