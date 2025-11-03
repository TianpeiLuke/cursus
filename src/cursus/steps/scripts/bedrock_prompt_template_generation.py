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
from dataclasses import asdict
from jinja2 import Template, Environment, BaseLoader
import jsonschema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Container path constants
CONTAINER_PATHS = {
    "INPUT_CATEGORIES_DIR": "/opt/ml/processing/input/categories",
    "OUTPUT_TEMPLATES_DIR": "/opt/ml/processing/output/templates",
    "OUTPUT_METADATA_DIR": "/opt/ml/processing/output/metadata",
    "OUTPUT_SCHEMA_DIR": "/opt/ml/processing/output/schema"
}

# Default system prompt configuration
DEFAULT_SYSTEM_PROMPT_CONFIG = {
    'role_definition': 'expert analyst',
    'expertise_areas': ['data analysis', 'classification', 'pattern recognition'],
    'responsibilities': ['analyze data accurately', 'classify content systematically', 'provide clear reasoning'],
    'behavioral_guidelines': ['be precise', 'be objective', 'be thorough', 'be consistent'],
    'tone': 'professional',
    'include_expertise_statement': True,
    'include_task_context': True
}

# Default output format configuration
DEFAULT_OUTPUT_FORMAT_CONFIG = {
    'format_type': 'structured_json',
    'required_fields': ['category', 'confidence', 'key_evidence', 'reasoning'],
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

# Default instruction configuration
DEFAULT_INSTRUCTION_CONFIG = {
    'include_analysis_steps': True,
    'include_decision_criteria': True,
    'include_edge_case_handling': True,
    'include_confidence_guidance': True,
    'include_reasoning_requirements': True,
    'step_by_step_format': True,
    'include_evidence_validation': True
}



class PromptTemplateGenerator:
    """
    Generates structured prompt templates for classification tasks using
    the 5-component architecture pattern.
    """
    
    def __init__(self, config: Dict[str, Any], schema_template: Optional[Dict[str, Any]] = None):
        self.config = config
        self.categories = self._load_categories()
        self.schema_template = schema_template
        self.template_env = Environment(loader=BaseLoader())
        
    def _load_categories(self) -> List[Dict[str, Any]]:
        """Load and validate category definitions from config."""
        categories = json.loads(self.config.get('category_definitions', '[]'))
        
        if not categories:
            raise ValueError("No category definitions provided")
        
        # Validate each category
        for i, category in enumerate(categories):
            required_fields = ['name', 'description', 'conditions', 'key_indicators']
            for field in required_fields:
                if field not in category or not category[field]:
                    raise ValueError(f"Category {i}: missing required field '{field}'")
        
        # Sort by priority if available
        categories.sort(key=lambda x: x.get('priority', 999))
        
        return categories
    
    def generate_template(self) -> Dict[str, Any]:
        """Generate complete prompt template with 5-component structure."""
        template = {
            'system_prompt': self._generate_system_prompt(),
            'user_prompt_template': self._generate_user_prompt_template(),
            'metadata': self._generate_template_metadata()
        }
        
        return template
    
    def _generate_system_prompt(self) -> str:
        """Generate system prompt with role assignment and expertise definition."""
        # Load user config and merge with comprehensive defaults
        user_config = json.loads(self.config.get('SYSTEM_PROMPT_CONFIG', '{}'))
        system_config = {**DEFAULT_SYSTEM_PROMPT_CONFIG, **user_config}
        
        role_definition = system_config.get('role_definition')
        expertise_areas = system_config.get('expertise_areas')
        responsibilities = system_config.get('responsibilities')
        behavioral_guidelines = system_config.get('behavioral_guidelines')
        
        system_prompt_parts = []
        
        # Role assignment
        system_prompt_parts.append(f"You are an {role_definition} with extensive knowledge in {', '.join(expertise_areas)}.")
        
        # Responsibilities
        if responsibilities:
            system_prompt_parts.append(f"Your task is to {', '.join(responsibilities)}.")
        
        # Behavioral guidelines
        if behavioral_guidelines:
            guidelines_text = ', '.join(behavioral_guidelines)
            system_prompt_parts.append(f"Always {guidelines_text} in your analysis.")
        
        return ' '.join(system_prompt_parts)
    
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
        
        return '\n\n'.join(components)
    
    def _generate_category_definitions_section(self) -> str:
        """Generate category definitions with conditions and exceptions."""
        section_parts = ["Categories and their criteria:"]
        
        for i, category in enumerate(self.categories, 1):
            category_parts = [f"\n{i}. {category['name']}"]
            
            # Description
            if category.get('description'):
                category_parts.append(f"    - {category['description']}")
            
            # Key elements/indicators
            if category.get('key_indicators'):
                category_parts.append("    - Key elements:")
                for indicator in category['key_indicators']:
                    category_parts.append(f"        * {indicator}")
            
            # Conditions
            if category.get('conditions'):
                category_parts.append("    - Conditions:")
                for condition in category['conditions']:
                    category_parts.append(f"        * {condition}")
            
            # Exceptions
            if category.get('exceptions'):
                category_parts.append("    - Must NOT include:")
                for exception in category['exceptions']:
                    category_parts.append(f"        * {exception}")
            
            # Examples if available
            if category.get('examples') and self.config.get('INCLUDE_EXAMPLES', 'true').lower() == 'true':
                category_parts.append("    - Examples:")
                for example in category['examples']:
                    category_parts.append(f"        * {example}")
            
            section_parts.append('\n'.join(category_parts))
        
        return '\n'.join(section_parts)
    
    def _generate_input_placeholders_section(self) -> str:
        """Generate input placeholders section."""
        placeholders = json.loads(self.config.get('INPUT_PLACEHOLDERS', '["input_data"]'))
        
        section_parts = ["Analysis Instructions:", ""]
        section_parts.append("Please analyze:")
        
        for placeholder in placeholders:
            section_parts.append(f"{placeholder.title()}: {{{placeholder}}}")
        
        return '\n'.join(section_parts)
    
    def _generate_instructions_section(self) -> str:
        """Generate instructions and rules section."""
        # Load user config and merge with comprehensive defaults
        user_config = json.loads(self.config.get('INSTRUCTION_CONFIG', '{}'))
        instruction_config = {**DEFAULT_INSTRUCTION_CONFIG, **user_config}
        
        instructions = [
            "Provide your analysis in the following structured format:",
            ""
        ]
        
        if instruction_config.get('include_analysis_steps', True):
            instructions.extend([
                "1. Carefully review all provided data",
                "2. Identify key patterns and indicators",
                "3. Match against category criteria",
                "4. Select the most appropriate category",
                "5. Validate evidence against conditions and exceptions",
                "6. Provide confidence assessment and reasoning",
                ""
            ])
        
        if instruction_config.get('include_decision_criteria', True):
            instructions.extend([
                "Decision Criteria:",
                "- Base decisions on explicit evidence in the data",
                "- Consider all category conditions and exceptions",
                "- Choose the category with the strongest evidence match",
                "- Provide clear reasoning for your classification",
                ""
            ])
        
        if instruction_config.get('include_evidence_validation', True):
            instructions.extend([
                "Key Evidence Validation:",
                "- Evidence MUST align with at least one condition for the selected category",
                "- Evidence MUST NOT match any exceptions listed for the selected category",
                "- Evidence should reference specific content from the input data",
                "- Multiple pieces of supporting evidence strengthen the classification",
                ""
            ])
        
        return '\n'.join(instructions)
    
    def _generate_output_format_section(self) -> str:
        """Generate output format schema section using schema template (default or custom)."""
        # Always use schema-based generation (either default or custom schema)
        return self._generate_custom_output_format_from_schema()
    
    def _generate_custom_output_format_from_schema(self) -> str:
        """Generate output format section from custom JSON schema template."""
        schema = self.schema_template
        
        format_parts = [
            "## Required Output Format",
            "",
            "**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**",
            "",
            "```json",
            "{"
        ]
        
        # Extract properties from schema
        properties = schema.get('properties', {})
        required_fields = schema.get('required', list(properties.keys()))
        
        # Generate JSON structure from schema
        for i, field in enumerate(required_fields):
            field_schema = properties.get(field, {})
            field_type = field_schema.get('type', 'string')
            description = field_schema.get('description', f"The {field} value")
            
            # Generate example value based on type
            if field_type == 'string':
                if 'enum' in field_schema:
                    example_value = f"One of: {', '.join(field_schema['enum'])}"
                else:
                    example_value = description
            elif field_type == 'number':
                min_val = field_schema.get('minimum', 0)
                max_val = field_schema.get('maximum', 1)
                example_value = f"Number between {min_val} and {max_val}"
            elif field_type == 'array':
                example_value = "Array of values"
            elif field_type == 'boolean':
                example_value = "true or false"
            else:
                example_value = description
            
            comma = "," if i < len(required_fields) - 1 else ""
            format_parts.append(f'    "{field}": "{example_value}"{comma}')
        
        format_parts.extend([
            "}",
            "```",
            "",
            "Field Descriptions:"
        ])
        
        # Add detailed field descriptions
        for field in required_fields:
            field_schema = properties.get(field, {})
            description = field_schema.get('description', f"The {field} value")
            field_type = field_schema.get('type', 'string')
            
            # Add type and constraint information
            constraints = []
            if field_type == 'number':
                if 'minimum' in field_schema:
                    constraints.append(f"minimum: {field_schema['minimum']}")
                if 'maximum' in field_schema:
                    constraints.append(f"maximum: {field_schema['maximum']}")
            elif field_type == 'string' and 'enum' in field_schema:
                constraints.append(f"must be one of: {', '.join(field_schema['enum'])}")
            
            constraint_text = f" ({', '.join(constraints)})" if constraints else ""
            format_parts.append(f"- **{field}** ({field_type}): {description}{constraint_text}")
        
        # Add category-specific validation if category field exists
        if 'category' in required_fields and properties.get('category', {}).get('enum'):
            category_names = properties['category']['enum']
            format_parts.extend([
                "",
                "**Category Validation:**",
                f"- The category field must exactly match one of: {', '.join(category_names)}",
                "- Category names are case-sensitive and must match exactly"
            ])
        
        format_parts.extend([
            "",
            "Do not include any text before or after the JSON object. Only return valid JSON."
        ])
        
        return '\n'.join(format_parts)
    
    def _generate_template_metadata(self) -> Dict[str, Any]:
        """Generate metadata about the template."""
        return {
            'template_version': self.config.get('TEMPLATE_VERSION', '1.0'),
            'generation_timestamp': datetime.now().isoformat(),
            'task_type': self.config.get('TEMPLATE_TASK_TYPE', 'classification'),
            'template_style': self.config.get('TEMPLATE_STYLE', 'structured'),
            'category_count': len(self.categories),
            'category_names': [cat['name'] for cat in self.categories],
            'output_format': self.config.get('OUTPUT_FORMAT_TYPE', 'structured_json'),
            'validation_level': self.config.get('VALIDATION_LEVEL', 'standard'),
            'includes_examples': self.config.get('INCLUDE_EXAMPLES', 'true').lower() == 'true',
            'generator_config': {
                'system_prompt_config': json.loads(self.config.get('SYSTEM_PROMPT_CONFIG', '{}')),
                'output_format_config': json.loads(self.config.get('OUTPUT_FORMAT_CONFIG', '{}'))
            }
        }


class TemplateValidator:
    """Validates generated prompt templates for quality and completeness."""
    
    def __init__(self, validation_level: str = "standard"):
        self.validation_level = validation_level
    
    def validate_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template and return validation results."""
        validation_results = {
            'is_valid': True,
            'quality_score': 0.0,
            'validation_details': [],
            'recommendations': []
        }
        
        # Validate system prompt
        system_validation = self._validate_system_prompt(template.get('system_prompt', ''))
        validation_results['validation_details'].append(system_validation)
        
        # Validate user prompt template
        user_validation = self._validate_user_prompt_template(template.get('user_prompt_template', ''))
        validation_results['validation_details'].append(user_validation)
        
        # Validate metadata
        metadata_validation = self._validate_metadata(template.get('metadata', {}))
        validation_results['validation_details'].append(metadata_validation)
        
        # Calculate overall quality score
        scores = [v['score'] for v in validation_results['validation_details']]
        validation_results['quality_score'] = sum(scores) / len(scores) if scores else 0.0
        
        # Determine overall validity
        validation_results['is_valid'] = all(v['is_valid'] for v in validation_results['validation_details'])
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results['validation_details'])
        
        return validation_results
    
    def _validate_system_prompt(self, system_prompt: str) -> Dict[str, Any]:
        """Validate system prompt component."""
        result = {
            'component': 'system_prompt',
            'is_valid': True,
            'score': 0.0,
            'issues': []
        }
        
        if not system_prompt or not system_prompt.strip():
            result['is_valid'] = False
            result['issues'].append("System prompt is empty")
            result['score'] = 0.0
            return result
        
        score = 0.0
        
        # Check for role definition
        if any(word in system_prompt.lower() for word in ['you are', 'expert', 'analyst', 'specialist']):
            score += 0.3
        else:
            result['issues'].append("Missing clear role definition")
        
        # Check for expertise areas
        if any(word in system_prompt.lower() for word in ['knowledge', 'experience', 'expertise']):
            score += 0.2
        else:
            result['issues'].append("Missing expertise statement")
        
        # Check for task context
        if any(word in system_prompt.lower() for word in ['task', 'analyze', 'classify', 'categorize']):
            score += 0.3
        else:
            result['issues'].append("Missing task context")
        
        # Check for behavioral guidelines
        if any(word in system_prompt.lower() for word in ['precise', 'objective', 'thorough', 'accurate']):
            score += 0.2
        else:
            result['issues'].append("Missing behavioral guidelines")
        
        result['score'] = score
        if score < 0.7:
            result['is_valid'] = False
        
        return result
    
    def _validate_user_prompt_template(self, user_prompt: str) -> Dict[str, Any]:
        """Validate user prompt template component."""
        result = {
            'component': 'user_prompt_template',
            'is_valid': True,
            'score': 0.0,
            'issues': []
        }
        
        if not user_prompt or not user_prompt.strip():
            result['is_valid'] = False
            result['issues'].append("User prompt template is empty")
            result['score'] = 0.0
            return result
        
        score = 0.0
        
        # Check for category definitions
        if 'categories' in user_prompt.lower() and 'criteria' in user_prompt.lower():
            score += 0.25
        else:
            result['issues'].append("Missing category definitions section")
        
        # Check for input placeholders
        if '{' in user_prompt and '}' in user_prompt:
            score += 0.25
        else:
            result['issues'].append("Missing input placeholders")
        
        # Check for instructions
        if any(word in user_prompt.lower() for word in ['analyze', 'instructions', 'provide', 'format']):
            score += 0.25
        else:
            result['issues'].append("Missing analysis instructions")
        
        # Check for output format
        if any(word in user_prompt.lower() for word in ['json', 'format', 'structure', 'output']):
            score += 0.25
        else:
            result['issues'].append("Missing output format specification")
        
        result['score'] = score
        if score < 0.7:
            result['is_valid'] = False
        
        return result
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template metadata."""
        result = {
            'component': 'metadata',
            'is_valid': True,
            'score': 1.0,
            'issues': []
        }
        
        required_fields = ['template_version', 'generation_timestamp', 'task_type', 'category_count']
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if missing_fields:
            result['issues'].append(f"Missing metadata fields: {', '.join(missing_fields)}")
            result['score'] = max(0.0, 1.0 - (len(missing_fields) * 0.2))
            if len(missing_fields) > 2:
                result['is_valid'] = False
        
        return result
    
    def _generate_recommendations(self, validation_details: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for detail in validation_details:
            if detail['score'] < 0.8:
                component = detail['component']
                recommendations.append(f"Improve {component}: {'; '.join(detail['issues'])}")
        
        return recommendations


def _generate_processing_config(config: Dict[str, str]) -> Dict[str, Any]:
    """Generate processing configuration metadata (non-redundant)."""
    return {
        "format_type": config.get('OUTPUT_FORMAT_TYPE', 'structured_json'),
        "response_model_name": f"{config.get('TEMPLATE_TASK_TYPE', 'classification').title()}Response",
        "validation_level": config.get('VALIDATION_LEVEL', 'standard')
    }


def load_category_definitions(categories_path: str, log: Callable[[str], None]) -> List[Dict[str, Any]]:
    """Load category definitions from input files (JSON/CSV)."""
    categories_dir = Path(categories_path)
    
    if not categories_dir.exists():
        log(f"Categories directory not found: {categories_path}")
        return []
    
    categories = []
    
    # Look for JSON files first
    json_files = list(categories_dir.glob("*.json"))
    if json_files:
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_categories = json.load(f)
                    if isinstance(file_categories, list):
                        categories.extend(file_categories)
                    else:
                        categories.append(file_categories)
                log(f"Loaded categories from {json_file}")
            except Exception as e:
                log(f"Failed to load categories from {json_file}: {e}")
    
    # Look for CSV files if no JSON found
    if not categories and list(categories_dir.glob("*.csv")):
        csv_files = list(categories_dir.glob("*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    category = {
                        'name': row.get('name', ''),
                        'description': row.get('description', ''),
                        'conditions': row.get('conditions', '').split(';') if row.get('conditions') else [],
                        'exceptions': row.get('exceptions', '').split(';') if row.get('exceptions') else [],
                        'key_indicators': row.get('key_indicators', '').split(';') if row.get('key_indicators') else [],
                        'priority': int(row.get('priority', 1))
                    }
                    categories.append(category)
                log(f"Loaded categories from {csv_file}")
            except Exception as e:
                log(f"Failed to load categories from {csv_file}: {e}")
    
    return categories


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
        # Load category definitions from input files
        categories = []
        if 'category_definitions' in input_paths:
            categories = load_category_definitions(input_paths['category_definitions'], log)
        
        if not categories:
            raise ValueError("No category definitions found in input files")
        
        # Load output schema template from OUTPUT_FORMAT_CONFIG or generate default
        schema_template = None
        
        # Try to load JSON schema from OUTPUT_FORMAT_CONFIG
        output_format_config = json.loads(environ_vars.get('OUTPUT_FORMAT_CONFIG', '{}'))
        if output_format_config and 'type' in output_format_config:
            # OUTPUT_FORMAT_CONFIG contains a JSON schema
            schema_template = output_format_config
            log("Using JSON schema from OUTPUT_FORMAT_CONFIG for format generation")
        
        # Generate default schema template if no custom schema is provided
        if not schema_template:
            # Generate default schema from DEFAULT_OUTPUT_FORMAT_CONFIG
            default_config = DEFAULT_OUTPUT_FORMAT_CONFIG
            required_fields = default_config['required_fields']
            field_descriptions = default_config['field_descriptions']
            
            schema_template = {
                "type": "object",
                "properties": {},
                "required": required_fields,
                "additionalProperties": False
            }
            
            # Generate properties from default config
            for field in required_fields:
                if field == 'confidence':
                    schema_template['properties'][field] = {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": field_descriptions.get(field, "Confidence score between 0.0 and 1.0")
                    }
                elif field == 'category':
                    schema_template['properties'][field] = {
                        "type": "string",
                        "enum": [cat['name'] for cat in categories],
                        "description": field_descriptions.get(field, "The classified category name")
                    }
                else:
                    schema_template['properties'][field] = {
                        "type": "string",
                        "description": field_descriptions.get(field, f"The {field} value")
                    }
            
            log("Generated default output schema template from DEFAULT_OUTPUT_FORMAT_CONFIG")
        else:
            # Update category enum in custom schema if it has a category field
            if ('properties' in schema_template and 
                'category' in schema_template['properties'] and
                schema_template['properties']['category'].get('type') == 'string'):
                schema_template['properties']['category']['enum'] = [cat['name'] for cat in categories]
            log("Using custom output schema template for format generation")
        
        # Build configuration from environment variables and loaded data
        config = {
            'TEMPLATE_TASK_TYPE': environ_vars.get('TEMPLATE_TASK_TYPE', 'classification'),
            'TEMPLATE_STYLE': environ_vars.get('TEMPLATE_STYLE', 'structured'),
            'VALIDATION_LEVEL': environ_vars.get('VALIDATION_LEVEL', 'standard'),
            'category_definitions': json.dumps(categories),
            'SYSTEM_PROMPT_CONFIG': environ_vars.get('SYSTEM_PROMPT_CONFIG', '{}'),
            'OUTPUT_FORMAT_CONFIG': environ_vars.get('OUTPUT_FORMAT_CONFIG', '{}'),
            'INSTRUCTION_CONFIG': environ_vars.get('INSTRUCTION_CONFIG', '{}'),
            'INPUT_PLACEHOLDERS': environ_vars.get('INPUT_PLACEHOLDERS', '["input_data"]'),
            'OUTPUT_FORMAT_TYPE': environ_vars.get('OUTPUT_FORMAT_TYPE', 'structured_json'),
            'REQUIRED_OUTPUT_FIELDS': environ_vars.get('REQUIRED_OUTPUT_FIELDS', '["category", "confidence", "key_evidence", "reasoning"]'),
            'INCLUDE_EXAMPLES': environ_vars.get('INCLUDE_EXAMPLES', 'true'),
            'GENERATE_VALIDATION_SCHEMA': environ_vars.get('GENERATE_VALIDATION_SCHEMA', 'true'),
            'TEMPLATE_VERSION': environ_vars.get('TEMPLATE_VERSION', '1.0')
        }
        
        # Initialize template generator with schema template (default or custom)
        generator = PromptTemplateGenerator(config, schema_template)
        
        # Generate template
        log("Generating prompt template...")
        template = generator.generate_template()
        
        # Validate template
        validator = TemplateValidator(config['VALIDATION_LEVEL'])
        validation_results = validator.validate_template(template)
        
        log(f"Template validation completed. Quality score: {validation_results['quality_score']:.2f}")
        
        # Create output directories
        templates_path = Path(output_paths['prompt_templates'])
        metadata_path = Path(output_paths['template_metadata'])
        schema_path = Path(output_paths['validation_schema'])
        
        templates_path.mkdir(parents=True, exist_ok=True)
        metadata_path.mkdir(parents=True, exist_ok=True)
        schema_path.mkdir(parents=True, exist_ok=True)
        
        # Save generated template
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save prompts.json (main template file)
        prompts_file = templates_path / "prompts.json"
        template_output = {
            'system_prompt': template['system_prompt'],
            'user_prompt_template': template['user_prompt_template'],
            'input_placeholders': json.loads(config.get('INPUT_PLACEHOLDERS', '["input_data"]'))
        }
        
        with open(prompts_file, 'w', encoding='utf-8') as f:
            json.dump(template_output, f, indent=2, ensure_ascii=False)
        
        log(f"Saved prompt template to: {prompts_file}")
        
        # Save template metadata
        metadata_file = metadata_path / f"template_metadata_{timestamp}.json"
        metadata_output = {
            **template['metadata'],
            'validation_results': validation_results,
            'generation_config': {
                'task_type': config['TEMPLATE_TASK_TYPE'],
                'template_style': config['TEMPLATE_STYLE'],
                'validation_level': config['VALIDATION_LEVEL'],
                'category_count': len(categories)
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_output, f, indent=2, ensure_ascii=False, default=str)
        
        log(f"Saved template metadata to: {metadata_file}")
        
        # Generate and save validation schema if requested
        if config['GENERATE_VALIDATION_SCHEMA'].lower() == 'true':
            schema_file = schema_path / f"validation_schema_{timestamp}.json"
            
            # Use custom schema template if available, otherwise generate default schema
            if schema_template:
                # Use the custom schema template directly
                validation_schema = schema_template.copy()
                
                # Update category enum if it exists in the schema
                if ('properties' in validation_schema and 
                    'category' in validation_schema['properties'] and
                    validation_schema['properties']['category'].get('type') == 'string'):
                    validation_schema['properties']['category']['enum'] = [cat['name'] for cat in categories]
                
                log("Using custom schema template for validation schema generation")
            else:
                # Generate default JSON schema for output validation
                required_fields = json.loads(config['REQUIRED_OUTPUT_FIELDS'])
                validation_schema = {
                    "type": "object",
                    "properties": {},
                    "required": required_fields,
                    "additionalProperties": False
                }
                
                # Add field definitions
                field_descriptions = json.loads(config.get('OUTPUT_FORMAT_CONFIG', '{}')).get('field_descriptions', {})
                for field in required_fields:
                    if field == 'confidence':
                        validation_schema['properties'][field] = {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": field_descriptions.get(field, "Confidence score between 0.0 and 1.0")
                        }
                    elif field == 'category':
                        validation_schema['properties'][field] = {
                            "type": "string",
                            "enum": [cat['name'] for cat in categories],
                            "description": field_descriptions.get(field, "The classified category name")
                        }
                    else:
                        validation_schema['properties'][field] = {
                            "type": "string",
                            "description": field_descriptions.get(field, f"The {field} value")
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
                    "template_version": config.get('TEMPLATE_VERSION', '1.0'),
                    "generation_timestamp": timestamp,
                    "category_count": len(categories),
                    "category_names": [cat['name'] for cat in categories],
                    "output_format_source": "OUTPUT_FORMAT_CONFIG" if json.loads(environ_vars.get('OUTPUT_FORMAT_CONFIG', '{}')) else "DEFAULT_CONFIG",
                    "task_type": config.get('TEMPLATE_TASK_TYPE', 'classification'),
                    "template_style": config.get('TEMPLATE_STYLE', 'structured')
                }
            }
            
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_validation_schema, f, indent=2, ensure_ascii=False)
            
            log(f"Saved enhanced validation schema with processing metadata to: {schema_file}")
        
        # Prepare results summary
        results = {
            'success': True,
            'template_generated': True,
            'validation_passed': validation_results['is_valid'],
            'quality_score': validation_results['quality_score'],
            'category_count': len(categories),
            'template_version': config['TEMPLATE_VERSION'],
            'output_files': {
                'prompts': str(prompts_file),
                'metadata': str(metadata_file),
                'schema': str(schema_file) if config['GENERATE_VALIDATION_SCHEMA'].lower() == 'true' else None
            },
            'validation_details': validation_results,
            'generation_timestamp': datetime.now().isoformat()
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
        parser = argparse.ArgumentParser(description="Bedrock prompt template generation script")
        parser.add_argument("--include-examples", action="store_true", help="Include examples in template")
        parser.add_argument("--generate-validation-schema", action="store_true", help="Generate validation schema")
        parser.add_argument("--template-version", default="1.0", help="Template version identifier")
        
        args = parser.parse_args()

        # Set up path dictionaries
        input_paths = {
            "category_definitions": CONTAINER_PATHS["INPUT_CATEGORIES_DIR"]
        }

        output_paths = {
            "prompt_templates": CONTAINER_PATHS["OUTPUT_TEMPLATES_DIR"],
            "template_metadata": CONTAINER_PATHS["OUTPUT_METADATA_DIR"],
            "validation_schema": CONTAINER_PATHS["OUTPUT_SCHEMA_DIR"]
        }

        # Environment variables dictionary
        environ_vars = {
            "TEMPLATE_TASK_TYPE": os.environ.get("TEMPLATE_TASK_TYPE", "classification"),
            "TEMPLATE_STYLE": os.environ.get("TEMPLATE_STYLE", "structured"),
            "VALIDATION_LEVEL": os.environ.get("VALIDATION_LEVEL", "standard"),
            "SYSTEM_PROMPT_CONFIG": os.environ.get("SYSTEM_PROMPT_CONFIG", "{}"),
            "OUTPUT_FORMAT_CONFIG": os.environ.get("OUTPUT_FORMAT_CONFIG", "{}"),
            "INSTRUCTION_CONFIG": os.environ.get("INSTRUCTION_CONFIG", "{}"),
            "INPUT_PLACEHOLDERS": os.environ.get("INPUT_PLACEHOLDERS", '["input_data"]'),
            "OUTPUT_FORMAT_TYPE": os.environ.get("OUTPUT_FORMAT_TYPE", "structured_json"),
            "REQUIRED_OUTPUT_FIELDS": os.environ.get("REQUIRED_OUTPUT_FIELDS", '["category", "confidence", "key_evidence", "reasoning"]'),
            "INCLUDE_EXAMPLES": os.environ.get("INCLUDE_EXAMPLES", str(args.include_examples).lower()),
            "GENERATE_VALIDATION_SCHEMA": os.environ.get("GENERATE_VALIDATION_SCHEMA", str(args.generate_validation_schema).lower()),
            "TEMPLATE_VERSION": os.environ.get("TEMPLATE_VERSION", args.template_version)
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
        logger.info(f"  Output Format: {environ_vars['OUTPUT_FORMAT_TYPE']}")
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
        logger.info(f"Prompt template generation completed successfully. Results: {result}")
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Error in prompt template generation script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
