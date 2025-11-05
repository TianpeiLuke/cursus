"""
Bedrock Prompt Template Generation Script Contract

Defines the contract for the Bedrock prompt template generation script that creates
structured prompt templates for classification tasks using the 5-component architecture pattern.
"""

from ...core.base.contract_base import ScriptContract

BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT = ScriptContract(
    entry_point="bedrock_prompt_template_generation.py",
    expected_input_paths={
        "prompt_configs": "/opt/ml/processing/input/prompt_configs",
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
        "INPUT_PLACEHOLDERS": '["input_data"]',
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
    - /opt/ml/processing/input/prompt_configs: Prompt configuration directory (required)
      - system_prompt.json: System prompt configuration (optional, uses defaults if missing)
      - output_format.json: Output format configuration (optional, uses defaults if missing)
      - instruction.json: Instruction configuration (optional, uses defaults if missing)
      - category_definitions.json: Category definitions (required)
        * Array of category objects with required fields: name, description, conditions, key_indicators
        * Optional fields: exceptions, examples, priority, validation_rules, aliases
    
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
    - Inputs: prompt_configs (required directory with JSON config files)
    - Outputs: prompt_templates (primary), template_metadata, validation_schema
    - Arguments: include-examples, generate-validation-schema, template-version
    - Configuration: File-based approach using JSON files instead of environment variables
    
    Environment Variables (streamlined - all optional with defaults):
    - TEMPLATE_TASK_TYPE: Type of classification task (default: "classification")
    - TEMPLATE_STYLE: Template style format (default: "structured")
    - VALIDATION_LEVEL: Validation strictness level (default: "standard")
    - INPUT_PLACEHOLDERS: JSON array of input field names (default: ["input_data"])
    - INCLUDE_EXAMPLES: Include examples in template (default: "true")
    - GENERATE_VALIDATION_SCHEMA: Generate validation schema (default: "true")
    - TEMPLATE_VERSION: Template version identifier (default: "1.0")
    
    Configuration Files (in prompt_configs directory):
    - system_prompt.json: System prompt configuration (role, expertise, behavioral guidelines)
    - output_format.json: Output format configuration (field descriptions, validation rules)
    - instruction.json: Instruction configuration (analysis steps, decision criteria)
    - category_definitions.json: Category definitions (required - name, description, conditions, key_indicators)
    
    Configuration File Formats:
    
    category_definitions.json (required):
    [
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
    ]
    
    system_prompt.json (optional):
    {
      "role_definition": "expert analyst",
      "expertise_areas": ["data analysis", "classification", "pattern recognition"],
      "responsibilities": ["analyze data accurately", "classify content systematically"],
      "behavioral_guidelines": ["be precise", "be objective", "be thorough"],
      "tone": "professional",
      "include_expertise_statement": true,
      "include_task_context": true
    }
    
    output_format.json (optional):
    {
      "format_type": "structured_json",
      "required_fields": ["category", "confidence", "key_evidence", "reasoning"],
      "field_descriptions": {
        "category": "The classified category name",
        "confidence": "Confidence score between 0.0 and 1.0",
        "key_evidence": "Specific evidence from input data",
        "reasoning": "Clear explanation of the decision-making process"
      },
      "validation_requirements": ["category must match predefined names exactly"],
      "evidence_validation_rules": ["Evidence must align with category conditions"]
    }
    
    instruction.json (optional):
    {
      "include_analysis_steps": true,
      "include_decision_criteria": true,
      "include_evidence_validation": true,
      "step_by_step_format": true
    }
    
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
