"""
Bedrock Prompt Template Generation Script Contract

Defines the contract for the Bedrock prompt template generation script that creates
structured prompt templates for classification tasks using the 5-component architecture pattern.
"""

from ...core.base.contract_base import ScriptContract

BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT = ScriptContract(
    entry_point="bedrock_prompt_template_generation.py",
    expected_input_paths={
        "category_definitions": "/opt/ml/processing/input/categories",
    },
    expected_output_paths={
        "prompt_templates": "/opt/ml/processing/output/templates",
        "template_metadata": "/opt/ml/processing/output/metadata",
        "validation_schema": "/opt/ml/processing/output/schema",
    },
    expected_arguments={
        "include-examples": "boolean flag to include examples in template",
        "generate-validation-schema": "boolean flag to generate validation schema",
        "template-version": "template version identifier",
    },
    required_env_vars=[],
    optional_env_vars={
        "TEMPLATE_TASK_TYPE": "classification",
        "TEMPLATE_STYLE": "structured",
        "VALIDATION_LEVEL": "standard",
        "SYSTEM_PROMPT_CONFIG": "{}",
        "OUTPUT_FORMAT_CONFIG": "{}",
        "INSTRUCTION_CONFIG": "{}",
        "INPUT_PLACEHOLDERS": '["input_data"]',
        "OUTPUT_FORMAT_TYPE": "structured_json",
        "REQUIRED_OUTPUT_FIELDS": '["category", "confidence", "key_evidence", "reasoning"]',
        "INCLUDE_EXAMPLES": "true",
        "GENERATE_VALIDATION_SCHEMA": "true",
        "TEMPLATE_VERSION": "1.0",
    },
    framework_requirements={
        "pandas": ">=1.2.0",
        "jinja2": ">=3.0.0",
        "jsonschema": ">=4.0.0",
        "pathlib": ">=1.0.0",
    },
    description="""
    Bedrock prompt template generation script that creates structured prompt templates
    for classification tasks using the 5-component architecture pattern optimized for LLM performance.
    
    The script generates comprehensive prompt templates with:
    1. System prompt with role assignment and expertise definition
    2. Category definitions with conditions, exceptions, and key indicators
    3. Input placeholders for dynamic content injection
    4. Step-by-step analysis instructions and decision criteria
    5. Structured output format specification with validation rules
    
    Input Structure:
    - /opt/ml/processing/input/categories: Category definition files (required)
      - Supports JSON (.json) and CSV (.csv) formats
      - JSON: Single category object or array of category objects
      - CSV: Comma-separated with semicolon-separated array fields
      - Required fields: name, description, conditions, key_indicators
      - Optional fields: exceptions, examples, priority, validation_rules, aliases
    
    Output Structure:
    - /opt/ml/processing/output/templates: Generated prompt templates
      - /opt/ml/processing/output/templates/prompts.json: Main template file
        * system_prompt: Role definition and behavioral guidelines
        * user_prompt_template: Complete 5-component template with placeholders
    - /opt/ml/processing/output/metadata: Template metadata and validation results
      - /opt/ml/processing/output/metadata/template_metadata_{timestamp}.json
        * Generation configuration and validation results
        * Quality metrics and component scores
        * Category statistics and template information
    - /opt/ml/processing/output/schema: Validation schemas for downstream use
      - /opt/ml/processing/output/schema/validation_schema_{timestamp}.json
        * JSON schema for validating Bedrock responses
        * Category enum constraints and field type validation
        * Evidence validation rules and requirements
    
    Contract aligned with script implementation:
    - Inputs: category_definitions (required only)
    - Outputs: prompt_templates (primary), template_metadata, validation_schema
    - Arguments: include-examples, generate-validation-schema, template-version
    - Schema Configuration: OUTPUT_FORMAT_CONFIG environment variable (JSON schema format)
    
    Environment Variables (all optional with defaults):
    - TEMPLATE_TASK_TYPE: Type of classification task (default: "classification")
    - TEMPLATE_STYLE: Template style format (default: "structured")
    - VALIDATION_LEVEL: Validation strictness level (default: "standard")
    - SYSTEM_PROMPT_CONFIG: JSON config for system prompt customization
    - OUTPUT_FORMAT_CONFIG: JSON schema for output format customization (supports full JSON Schema format)
    - INSTRUCTION_CONFIG: JSON config for instruction customization
    - INPUT_PLACEHOLDERS: JSON array of input field names (default: ["input_data"])
    - OUTPUT_FORMAT_TYPE: Output format type (default: "structured_json")
    - REQUIRED_OUTPUT_FIELDS: JSON array of required output fields
    - INCLUDE_EXAMPLES: Include examples in template (default: "true")
    - GENERATE_VALIDATION_SCHEMA: Generate validation schema (default: "true")
    - TEMPLATE_VERSION: Template version identifier (default: "1.0")
    
    Category Definition Format:
    JSON Single Category:
    {
      "name": "Positive",
      "description": "Positive sentiment or favorable opinion",
      "conditions": ["Contains positive language", "Expresses satisfaction"],
      "exceptions": ["Sarcastic statements", "Backhanded compliments"],
      "key_indicators": ["good", "excellent", "satisfied", "happy"],
      "examples": ["This is great!", "Love this product"],
      "priority": 1,
      "validation_rules": ["Must contain positive indicator"],
      "aliases": ["positive_sentiment", "favorable"]
    }
    
    JSON Multiple Categories:
    [
      {"name": "Positive", "description": "...", "conditions": [...], ...},
      {"name": "Negative", "description": "...", "conditions": [...], ...},
      {"name": "Neutral", "description": "...", "conditions": [...], ...}
    ]
    
    CSV Format:
    name,description,conditions,exceptions,key_indicators,priority,examples,validation_rules,aliases
    Positive,"Positive sentiment","Contains positive;Expresses satisfaction","Sarcastic;Backhanded","good;excellent",1,"Great!;Love it","Must contain positive","positive_sentiment"
    
    Template Generation Features:
    - 5-Component Architecture: System prompt, category definitions, input placeholders, instructions, output format
    - Intelligent Defaults: Comprehensive default configurations for all components
    - Evidence Validation: Key evidence must align with conditions and avoid exceptions
    - Quality Scoring: Template validation with component-specific quality metrics
    - Multiple Input Fields: Support for complex input structures via INPUT_PLACEHOLDERS
    - Flexible Output: Structured JSON with 4-field format (category, confidence, key_evidence, reasoning)
    
    Generated Template Usage:
    1. Load template from prompts.json
    2. Format user_prompt_template with actual data using placeholder substitution
    3. Use system_prompt and formatted user_prompt with Bedrock API
    4. Validate Bedrock responses using generated validation schema
    
    Quality Assurance:
    - Template validation with quality scoring (0.0-1.0)
    - Component-specific validation (system prompt, user template, metadata)
    - Production readiness threshold (minimum 0.7 quality score)
    - Comprehensive error handling and recovery mechanisms
    
    Integration Ready:
    - Direct compatibility with Bedrock processing steps
    - Standard SageMaker container paths
    - Comprehensive metadata for monitoring and debugging
    - Validation schemas for downstream quality control
    """,
)
