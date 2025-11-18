---
tags:
  - code
  - processing_script
  - bedrock
  - llm_processing
  - prompt_generation
  - template_generation
keywords:
  - bedrock prompt template generation
  - 5-component architecture
  - classification prompts
  - LLM templates
  - validation schema
  - placeholder resolution
  - template quality scoring
topics:
  - LLM prompt engineering
  - template generation
  - classification systems
  - bedrock integration
language: python
date of note: 2025-11-18
---

# Bedrock Prompt Template Generation Script Documentation

## Overview

The Bedrock Prompt Template Generation script (`bedrock_prompt_template_generation.py`) is a sophisticated template generation engine that creates structured, high-quality prompt templates for classification and categorization tasks using the **5-component architecture pattern** optimized for Large Language Model (LLM) performance. This script represents the first stage of a two-step template-driven LLM processing pipeline, generating reusable prompt templates and validation schemas that are consumed by downstream Bedrock Processing steps.

The script implements intelligent template composition with placeholder resolution, quality validation, and comprehensive schema generation. It transforms category definitions and configuration files into production-ready prompt templates that ensure consistent, high-quality LLM interactions across all classification tasks.

**Key Innovation**: Unlike traditional static prompt engineering, this script enables **configuration-driven prompt generation** where prompts are dynamically composed from modular components, allowing teams to manage classification tasks through JSON configurations rather than hard-coded prompts.

## Purpose and Major Tasks

### Primary Purpose

Generate structured, validated prompt templates for LLM-based classification tasks that ensure consistent formatting, comprehensive category definitions, and automated response validation across all Bedrock processing operations.

### Major Tasks

1. **Load Category Definitions**: Parse and validate category definitions from JSON configuration files, ensuring all required fields (name, description, conditions, key_indicators) are present and properly formatted

2. **Compose System Prompts**: Generate role-based system prompts with expertise definitions, behavioral guidelines, and tone-appropriate language based on configuration settings

3. **Build Category Definition Sections**: Create comprehensive category documentation sections with conditions, exceptions, key indicators, and examples for each classification category

4. **Resolve Dynamic Placeholders**: Intelligent placeholder resolution system that dynamically fills template placeholders from category definitions and validation schemas using multiple resolution strategies

5. **Generate Instructions and Rules**: Compose step-by-step analysis instructions, decision criteria, reasoning requirements, and evidence validation rules based on instruction configuration

6. **Create Output Format Specifications**: Generate detailed output format sections supporting both structured JSON and structured text formats with comprehensive field descriptions and validation rules

7. **Validate Template Quality**: Assess generated templates using multi-component quality scoring (system prompt, user template, metadata) with production readiness thresholds

8. **Generate Validation Schemas**: Create JSON schemas for automated response validation in downstream Bedrock Processing steps, enriched with processing metadata and category constraints

9. **Enrich Schemas with Categories**: Dynamically populate schema enum fields with category names from definitions, creating the connection between category definitions and output format validation

10. **Export Template Artifacts**: Save generated templates, metadata, and validation schemas to standardized output locations for consumption by Bedrock Processing steps

## Script Contract

### Entry Point
```python
bedrock_prompt_template_generation.py
```

### Input Paths

| Path Key | Container Path | Description |
|----------|---------------|-------------|
| `prompt_configs` | `/opt/ml/processing/input/prompt_configs` | **Required**. Directory containing JSON configuration files for template generation |

**Configuration Files in `prompt_configs` Directory:**

1. **`category_definitions.json`** (Required)
   - Array of category objects defining all classification categories
   - Required fields per category: `name`, `description`, `conditions`, `key_indicators`
   - Optional fields: `exceptions`, `examples`, `priority`, `validation_rules`, `aliases`

2. **`system_prompt.json`** (Optional)
   - System prompt configuration (role definition, expertise, behavioral guidelines)
   - Uses comprehensive defaults if not provided

3. **`output_format.json`** (Optional)
   - Output format configuration (field descriptions, validation rules, format type)
   - Uses comprehensive defaults if not provided

4. **`instruction.json`** (Optional)
   - Instruction configuration (analysis steps, decision criteria, evidence validation)
   - Uses comprehensive defaults if not provided

### Output Paths

| Path Key | Container Path | Description |
|----------|---------------|-------------|
| `prompt_templates` | `/opt/ml/processing/output/templates` | Primary output directory containing generated prompt templates |
| `template_metadata` | `/opt/ml/processing/output/metadata` | Template metadata including generation config and validation results |
| `validation_schema` | `/opt/ml/processing/output/schema` | JSON schemas for validating Bedrock responses in downstream steps |

**Generated Files:**

1. **`/opt/ml/processing/output/templates/prompts.json`**
   - Main template file consumed by Bedrock Processing steps
   - Contains: `system_prompt`, `user_prompt_template`, `input_placeholders`

2. **`/opt/ml/processing/output/metadata/template_metadata_{timestamp}.json`**
   - Template generation metadata and quality metrics
   - Contains: generation config, validation results, quality scores, category statistics

3. **`/opt/ml/processing/output/schema/validation_schema_{timestamp}.json`**
   - JSON schema for validating Bedrock responses
   - Contains: category enum constraints, field validations, processing metadata

### Environment Variables

All environment variables are **optional** with comprehensive defaults:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TEMPLATE_TASK_TYPE` | string | `"classification"` | Type of classification task the template is designed for |
| `TEMPLATE_STYLE` | string | `"structured"` | Template style format (structured, conversational, technical) |
| `VALIDATION_LEVEL` | string | `"standard"` | Validation strictness level (minimal, standard, strict) |
| `INPUT_PLACEHOLDERS` | JSON array | `["input_data"]` | Array of input field names for dynamic content injection |
| `INCLUDE_EXAMPLES` | string | `"true"` | Include category examples in template (true/false) |
| `GENERATE_VALIDATION_SCHEMA` | string | `"true"` | Generate validation schema for downstream use (true/false) |
| `TEMPLATE_VERSION` | string | `"1.0"` | Template version identifier for tracking and compatibility |

### Job Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--include-examples` | flag | No | False | Include category examples in generated template |
| `--generate-validation-schema` | flag | No | False | Generate JSON validation schema for downstream validation |
| `--template-version` | string | No | `"1.0"` | Template version identifier for tracking |

**Note**: Job arguments override corresponding environment variables when provided.

### Framework Requirements

```python
{
    "pandas": ">=1.2.0",      # Data manipulation and CSV processing
    "jinja2": ">=3.0.0",      # Template rendering (future extensibility)
    "jsonschema": ">=4.0.0",  # Schema validation and generation
    "pathlib": ">=1.0.0"      # Path manipulation
}
```

## Input Data Structure

### Configuration File: `category_definitions.json`

**Format**: JSON array of category objects

**Required Structure**:
```json
[
  {
    "name": "CategoryName",
    "description": "Clear description of what this category represents",
    "conditions": [
      "Condition 1 that must be met",
      "Condition 2 that must be met"
    ],
    "key_indicators": [
      "keyword1",
      "keyword2",
      "phrase pattern"
    ],
    "exceptions": [
      "Exception case 1 to exclude",
      "Exception case 2 to exclude"
    ],
    "examples": [
      "Example 1 of this category",
      "Example 2 of this category"
    ],
    "priority": 1,
    "validation_rules": [
      "Additional validation rule 1"
    ],
    "aliases": [
      "alternative_name_1",
      "alternative_name_2"
    ]
  }
]
```

**Field Descriptions**:
- **name** (string, required): Unique category identifier used in output validation
- **description** (string, required): Human-readable category description
- **conditions** (array, required): Conditions that must be met for classification
- **key_indicators** (array, required): Keywords, phrases, or patterns indicating this category
- **exceptions** (array, optional): Cases that should NOT be classified in this category
- **examples** (array, optional): Concrete examples for this category
- **priority** (integer, optional): Priority for ambiguous cases (lower = higher priority)
- **validation_rules** (array, optional): Additional validation requirements
- **aliases** (array, optional): Alternative names for this category

### Configuration File: `system_prompt.json`

**Format**: JSON object with system prompt configuration

**Optional Structure** (uses defaults if not provided):
```json
{
  "role_definition": "expert analyst",
  "expertise_areas": [
    "data analysis",
    "classification",
    "pattern recognition"
  ],
  "responsibilities": [
    "analyze data accurately",
    "classify content systematically",
    "provide clear reasoning"
  ],
  "behavioral_guidelines": [
    "be precise",
    "be objective",
    "be thorough",
    "be consistent"
  ],
  "tone": "professional"
}
```

**Tone Options**: `"professional"`, `"casual"`, `"technical"`, `"formal"` (affects language style)

### Configuration File: `output_format.json`

**Format**: JSON object with output format configuration

**Optional Structure** (uses defaults if not provided):
```json
{
  "format_type": "structured_json",
  "required_fields": [
    "category",
    "confidence",
    "key_evidence",
    "reasoning"
  ],
  "field_descriptions": {
    "category": "The classified category name (must match predefined categories)",
    "confidence": "Confidence score between 0.0 and 1.0",
    "key_evidence": "Specific evidence from input data that aligns with category conditions",
    "reasoning": "Clear explanation of the decision-making process"
  },
  "validation_requirements": [
    "category must match one of the predefined category names exactly",
    "confidence must be a number between 0.0 and 1.0",
    "key_evidence must align with category conditions and avoid exceptions"
  ],
  "evidence_validation_rules": [
    "Evidence MUST align with at least one condition for the selected category",
    "Evidence MUST NOT match any exceptions listed for the selected category"
  ]
}
```

**Format Types**: 
- `"structured_json"`: JSON output with schema validation
- `"structured_text"`: Structured text format with sections

### Configuration File: `instruction.json`

**Format**: JSON object with instruction configuration

**Optional Structure** (uses defaults if not provided):
```json
{
  "include_analysis_steps": true,
  "include_decision_criteria": true,
  "include_reasoning_requirements": true,
  "step_by_step_format": true,
  "include_evidence_validation": true
}
```

## Output Data Structure

### Primary Output: `prompts.json`

**Location**: `/opt/ml/processing/output/templates/prompts.json`

**Format**: JSON object with generated template components

```json
{
  "system_prompt": "You are an expert analyst with extensive knowledge in...",
  "user_prompt_template": "Categories and their criteria:\n\n1. Category1...",
  "input_placeholders": ["input_data"]
}
```

**Field Descriptions**:
- **system_prompt**: Complete system prompt with role definition and behavioral guidelines
- **user_prompt_template**: Complete 5-component user prompt with placeholders (e.g., `{input_data}`)
- **input_placeholders**: Array of placeholder names to be replaced with actual data

**Usage Pattern**:
```python
# Load template
with open("prompts.json") as f:
    template = json.load(f)

# Format with actual data
user_prompt = template["user_prompt_template"].format(input_data="...")

# Use with Bedrock
response = bedrock.invoke_model(
    system_prompt=template["system_prompt"],
    user_prompt=user_prompt
)
```

### Metadata Output: `template_metadata_{timestamp}.json`

**Location**: `/opt/ml/processing/output/metadata/template_metadata_{timestamp}.json`

**Format**: JSON object with comprehensive metadata

```json
{
  "template_version": "1.0",
  "generation_timestamp": "2025-11-18T12:00:00",
  "task_type": "classification",
  "template_style": "structured",
  "category_count": 5,
  "category_names": ["Category1", "Category2", "..."],
  "output_format": "structured_json",
  "validation_level": "standard",
  "includes_examples": true,
  "generator_config": {
    "system_prompt_config": {...},
    "output_format_config": {...},
    "instruction_config": {...}
  },
  "placeholder_validation": {
    "total_placeholders": 4,
    "successful": 4,
    "failed": 0,
    "all_resolved": true,
    "failures": []
  },
  "validation_results": {
    "is_valid": true,
    "quality_score": 0.95,
    "validation_details": [
      {
        "component": "system_prompt",
        "is_valid": true,
        "score": 1.0,
        "issues": []
      },
      {
        "component": "user_prompt_template",
        "is_valid": true,
        "score": 1.0,
        "issues": []
      }
    ],
    "recommendations": []
  }
}
```

### Validation Schema Output: `validation_schema_{timestamp}.json`

**Location**: `/opt/ml/processing/output/schema/validation_schema_{timestamp}.json`

**Format**: Enhanced JSON schema with processing metadata

```json
{
  "title": "Bedrock Response Validation Schema",
  "description": "Schema for validating Bedrock LLM responses with processing metadata",
  "type": "object",
  "properties": {
    "category": {
      "type": "string",
      "enum": ["Category1", "Category2", "..."],
      "description": "The classified category name"
    },
    "confidence": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Confidence score between 0.0 and 1.0"
    },
    "key_evidence": {
      "type": "string",
      "description": "Specific evidence from input data"
    },
    "reasoning": {
      "type": "string",
      "description": "Clear explanation of the decision-making process"
    }
  },
  "required": ["category", "confidence", "key_evidence", "reasoning"],
  "additionalProperties": false,
  "processing_config": {
    "format_type": "structured_json",
    "response_model_name": "ClassificationResponse",
    "validation_level": "standard"
  },
  "template_metadata": {
    "template_version": "1.0",
    "generation_timestamp": "20251118_120000",
    "category_count": 5,
    "category_names": ["Category1", "..."]
  }
}
```

## Key Functions and Tasks

### Component 1: Configuration Loading and Management

#### Function: `load_config_from_json_file`
**Purpose**: Load configuration from JSON file with fallback to defaults

**Algorithm**:
```
FUNCTION load_config_from_json_file(config_path, config_name, default_config, log):
    config_file = Path(config_path) / f"{config_name}.json"
    
    IF config_file.exists():
        TRY:
            config = json.load(config_file)
            log(f"Loaded {config_name} config from {config_file}")
            RETURN {**default_config, **config}  # Merge with defaults
        CATCH Exception as e:
            log(f"Failed to load: {e}. Using defaults.")
            RETURN default_config
    ELSE:
        log(f"Config file not found. Using defaults.")
        RETURN default_config
END FUNCTION
```

**Parameters**:
- `config_path` (str): Path to configuration directory
- `config_name` (str): Configuration file name (without .json extension)
- `default_config` (dict): Default configuration to use as fallback
- `log` (callable): Logging function

**Returns**: Dictionary containing merged configuration (file + defaults)

**Integration**: Called during initialization to load all configuration files

---

#### Function: `load_category_definitions`
**Purpose**: Load and validate category definitions from configuration directory

**Algorithm**:
```
FUNCTION load_category_definitions(prompt_configs_path, log):
    config_dir = Path(prompt_configs_path)
    
    IF NOT config_dir.exists():
        log("Prompt configs directory not found")
        RETURN []
    
    categories_file = config_dir / "category_definitions.json"
    
    IF categories_file.exists():
        TRY:
            categories = json.load(categories_file)
            log("Loaded category definitions")
            RETURN categories IF is_list ELSE [categories]
        CATCH Exception as e:
            log(f"Failed to load: {e}")
            RETURN []
    ELSE:
        log("Category definitions file not found")
        RETURN []
END FUNCTION
```

**Returns**: List of category definition dictionaries

**Error Handling**: Returns empty list on failure, allowing script to report meaningful error

---

### Component 2: Placeholder Resolution System

#### Class: `PlaceholderResolver`
**Purpose**: Intelligent placeholder resolution engine that dynamically fills template placeholders from category definitions and validation schemas

**Core Data Structures**:
```python
{
    "placeholder_registry": {
        "placeholder_name": {
            "field_name": "category",
            "source_hint": "schema_enum",
            "original": "${category_enum}"
        }
    },
    "resolution_status": {
        "placeholder_name": {
            "status": "success",
            "result": "One of: Category1, Category2, ..."
        }
    }
}
```

---

#### Method: `resolve_placeholder`
**Purpose**: Resolve a single placeholder using multi-strategy resolution

**Algorithm**:
```
METHOD resolve_placeholder(placeholder, field_name, source_hint):
    # Check if placeholder needs resolution
    IF NOT placeholder OR NOT placeholder.startswith("${"):
        RETURN placeholder  # Literal text, no resolution needed
    
    # Extract placeholder name
    placeholder_name = placeholder.strip("${}")
    
    # Register placeholder for tracking
    self.placeholder_registry[placeholder_name] = {
        "field_name": field_name,
        "source_hint": source_hint,
        "original": placeholder
    }
    
    # Try to resolve using strategies
    TRY:
        resolved = self._resolve_by_strategy(placeholder_name, field_name, source_hint)
        self.resolution_status[placeholder_name] = {
            "status": "success",
            "result": resolved
        }
        log(f"Resolved ${{{placeholder_name}}} → {resolved[:50]}...")
        RETURN resolved
    CATCH Exception as e:
        self.resolution_status[placeholder_name] = {
            "status": "failed",
            "error": str(e)
        }
        log(f"Failed to resolve ${{{placeholder_name}}}: {e}")
        # Fallback to descriptive placeholder
        RETURN f"[{field_name.upper()}_UNRESOLVED]"
END METHOD
```

**Resolution Strategies** (in priority order):
1. **Explicit source hint**: Use provided hint (e.g., "schema_enum", "schema_range", "categories")
2. **Infer from placeholder name**: Match patterns like "enum", "category", "range", "numeric"
3. **Schema lookup by field name**: Try to resolve from validation schema using field name

**Returns**: Resolved placeholder text or fallback placeholder

---

#### Method: `_resolve_from_schema_enum`
**Purpose**: Resolve placeholder from schema enum values

**Algorithm**:
```
METHOD _resolve_from_schema_enum(field_name):
    IF NOT self.schema:
        RAISE ValueError("No schema available")
    
    properties = self.schema.get("properties", {})
    
    IF field_name NOT IN properties:
        RAISE ValueError(f"Field {field_name} not in schema")
    
    field_schema = properties[field_name]
    
    IF "enum" NOT IN field_schema:
        RAISE ValueError(f"Field {field_name} has no enum")
    
    enum_values = field_schema["enum"]
    
    # Format based on enum length
    IF len(enum_values) <= 5:
        RETURN f"One of: {', '.join(enum_values)}"
    ELSE:
        first_few = enum_values[:3]
        RETURN f"One of: {', '.join(first_few)}, ... (see full list above)"
END METHOD
```

**Example Resolutions**:
- Input: `"${category_enum}"`, field: `"category"`, enum: `["Positive", "Negative", "Neutral"]`
- Output: `"One of: Positive, Negative, Neutral"`

---

#### Method: `validate_all_resolved`
**Purpose**: Validate that all registered placeholders were successfully resolved

**Algorithm**:
```
METHOD validate_all_resolved():
    report = {
        "total_placeholders": len(self.placeholder_registry),
        "successful": 0,
        "failed": 0,
        "failures": []
    }
    
    FOR name, status IN self.resolution_status.items():
        IF status["status"] == "success":
            report["successful"] += 1
        ELSE:
            report["failed"] += 1
            report["failures"].append({
                "placeholder": name,
                "field": self.placeholder_registry[name]["field_name"],
                "error": status["error"]
            })
    
    report["all_resolved"] = (report["failed"] == 0)
    RETURN report
END METHOD
```

**Returns**: Comprehensive validation report with failure details

**Integration**: Called after template generation to ensure all placeholders resolved

---

### Component 3: Template Generation Engine

#### Class: `PromptTemplateGenerator`
**Purpose**: Main template generator implementing 5-component architecture pattern

**Initialization**:
```python
def __init__(self, config, schema_template):
    self.config = config
    self.categories = self._load_categories()  # Load and validate
    
    # Enrich schema with category enum
    self.schema_template = self._enrich_schema_with_categories(schema_template)
    
    # Create placeholder resolver with enriched schema
    self.placeholder_resolver = PlaceholderResolver(
        self.categories, 
        self.schema_template
    )
```

---

#### Method: `_enrich_schema_with_categories`
**Purpose**: Dynamically populate schema category enum from category definitions

**Algorithm**:
```
METHOD _enrich_schema_with_categories(schema):
    IF NOT schema OR NOT self.categories:
        RETURN schema
    
    # Make a copy to avoid mutating original
    enriched_schema = schema.copy()
    
    # Update category field enum if it exists
    IF "properties" IN enriched_schema AND "category" IN enriched_schema["properties"]:
        category_names = [cat["name"] FOR cat IN self.categories]
        enriched_schema["properties"]["category"]["enum"] = category_names
        log(f"Enriched schema with {len(category_names)} category enum values")
    
    RETURN enriched_schema
END METHOD
```

**Critical Connection**: This creates the link between category definitions and output format validation

---

#### Method: `generate_template`
**Purpose**: Generate complete prompt template with all 5 components

**Algorithm**:
```
METHOD generate_template():
    template = {
        "system_prompt": self._generate_system_prompt(),
        "user_prompt_template": self._generate_user_prompt_template(),
        "metadata": self._generate_template_metadata()
    }
    
    # Validate all placeholders were resolved
    placeholder_validation = self.placeholder_resolver.validate_all_resolved()
    
    IF NOT placeholder_validation["all_resolved"]:
        log(f"Some placeholders failed: {placeholder_validation['failures']}")
    ELSE:
        log(f"All {placeholder_validation['successful']} placeholders resolved")
    
    # Include placeholder validation in metadata
    template["metadata"]["placeholder_validation"] = placeholder_validation
    
    RETURN template
END METHOD
```

**Returns**: Complete template dictionary with system prompt, user template, and metadata

---

#### Method: `_generate_system_prompt`
**Purpose**: Generate system prompt with role assignment and tone-appropriate language

**Algorithm**:
```
METHOD _generate_system_prompt():
    system_config = self.config.get("system_prompt_config", DEFAULT_SYSTEM_PROMPT_CONFIG)
    
    role_definition = system_config.get("role_definition")
    expertise_areas = system_config.get("expertise_areas")
    responsibilities = system_config.get("responsibilities")
    behavioral_guidelines = system_config.get("behavioral_guidelines")
    tone = system_config.get("tone", "professional")
    
    # Get tone-appropriate language adjustments
    tone_adjustments = self._get_tone_adjustments(tone)
    
    parts = []
    
    # Role assignment with tone-appropriate language
    parts.append(
        f"{tone_adjustments['opener']} {role_definition} with extensive knowledge in {', '.join(expertise_areas)}."
    )
    
    # Responsibilities with tone-appropriate connector
    IF responsibilities:
        parts.append(
            f"{tone_adjustments['task_connector']} {', '.join(responsibilities)}."
        )
    
    # Behavioral guidelines with tone-appropriate adverb
    IF behavioral_guidelines:
        guidelines_text = ", ".join(behavioral_guidelines)
        parts.append(
            f"{tone_adjustments['guideline_adverb']} {guidelines_text} in your analysis."
        )
    
    RETURN " ".join(parts)
END METHOD
```

**Tone Mappings**:
- **professional**: "You are an", "Your task is to", "Always"
- **casual**: "Hey! You're a", "Your job is to", "Make sure to"
- **technical**: "System role: You are a", "Core functions include:", "Operational guidelines require:"
- **formal**: "You shall function as an", "Your responsibilities encompass:", "You must consistently"

---

#### Method: `_generate_category_definitions_section`
**Purpose**: Generate comprehensive category definitions with conditions and exceptions

**Algorithm**:
```
METHOD _generate_category_definitions_section():
    section_parts = ["Categories and their criteria:"]
    
    FOR i, category IN enumerate(self.categories, 1):
        category_parts = [f"\n{i}. {category['name']}"]
        
        # Description
        IF category.get("description"):
            category_parts.append(f"    - {category['description']}")
        
        # Key elements/indicators
        IF category.get("key_indicators"):
            category_parts.append("    - Key elements:")
            FOR indicator IN category["key_indicators"]:
                category_parts.append(f"        * {indicator}")
        
        # Conditions
        IF category.get("conditions"):
            category_parts.append("    - Conditions:")
            FOR condition IN category["conditions"]:
                category_parts.append(f"        * {condition}")
        
        # Exceptions (critical for avoiding false positives)
        IF category.get("exceptions"):
            category_parts.append("    - Must NOT include:")
            FOR exception IN category["exceptions"]:
                category_parts.append(f"        * {exception}")
        
        # Examples if available and enabled
        IF category.get("examples") AND self.config.get("INCLUDE_EXAMPLES") == "true":
            category_parts.append("    - Examples:")
            FOR example IN category["examples"]:
                category_parts.append(f"        * {example}")
        
        section_parts.append("\n".join(category_parts))
    
    RETURN "\n".join(section_parts)
END METHOD
```

**Output Example**:
```
Categories and their criteria:

1. Positive
    - Positive sentiment or favorable opinion
    - Key elements:
        * good
        * excellent
        * satisfied
    - Conditions:
        * Contains positive language
        * Expresses satisfaction
    - Must NOT include:
        * Sarcastic statements
        * Backhanded compliments
```

---

#### Method: `_generate_output_format_section`
**Purpose**: Generate output format specification based on format_type

**Algorithm**:
```
METHOD _generate_output_format_section():
    output_config = self.config.get("output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG)
    format_type = output_config.get("format_type", "structured_json")
    
    IF format_type == "structured_text":
        RETURN self._generate_structured_text_output_format_from_config()
    ELSE:
        # Default to JSON schema-based generation
        RETURN self._generate_custom_output_format_from_schema()
END METHOD
```

**Supports Two Format Types**:
1. **structured_json**: JSON format with schema validation (default)
2. **structured_text**: Structured text format with sections

---

#### Method: `_generate_custom_output_format_from_schema`
**Purpose**: Generate JSON output format specification from validation schema

**Algorithm**:
```
METHOD _generate_custom_output_format_from_schema():
    schema = self.schema_template
    output_config = self.config.get("output_format_config", DEFAULT_OUTPUT_FORMAT_CONFIG)
    
    format_parts = ["## Required Output Format", ""]
    
    # Add header text
    header_text = output_config.get("header_text", "**CRITICAL: You must respond with a valid JSON object...**")
    format_parts.append(header_text)
    format_parts.append("")
    
    # Check if example_output is provided as dict
    example_output = output_config.get("example_output")
    use_real_example = isinstance(example_output, dict)
    
    IF use_real_example:
        # Use the provided example directly
        format_parts.append(json.dumps(example_output, indent=2))
        format_parts.append("")
    ELSE:
        # Generate placeholder structure from schema
        format_parts.append("{")
        
        properties = schema.get("properties", {})
        required_fields = schema.get("required", list(properties.keys()))
        
        FOR i, field IN enumerate(required_fields):
            field_schema = properties.get(field, {})
            field_type = field_schema.get("type", "string")
            
            # Generate example value based on type
            IF field_type == "string":
                IF "enum" IN field_schema:
                    example_value = f"One of: {', '.join(field_schema['enum'])}"
                ELSE:
                    example_value = field_schema.get("description", f"The {field} value")
            ELIF field_type == "number":
                min_val = field_schema.get("minimum", 0)
                max_val = field_schema.get("maximum", 1)
                example_value = f"Number between {min_val} and {max_val}"
            ELIF field_type == "array":
                example_value = "Array of values"
            ELIF field_type == "boolean":
                example_value = "true or false"
            ELSE:
                example_value = field_schema.get("description", f"The {field} value")
            
            comma = "," IF i < len(required_fields) - 1 ELSE ""
            format_parts.append(f'    "{field}": "{example_value}"{comma}')
        
        format_parts.append("}")
        format_parts.append("")
    
    # Add field descriptions from config or schema
    format_parts.append("Field Descriptions:")
    field_descriptions = output_config.get("field_descriptions", {})
    
    FOR field IN required_fields:
        field_schema = properties.get(field, {})
        
        # Prefer config description, fallback to schema
        IF field IN field_descriptions:
            description = field_descriptions[field]
        ELSE:
            description = field_schema.get("description", f"The {field} value")
        
        field_type = field_schema.get("type", "string")
        
        # Add type and constraint information
        constraints = []
        IF field_type == "number":
            IF "minimum" IN field_schema:
                constraints.append(f"minimum: {field_schema['minimum']}")
            IF "maximum" IN field_schema:
                constraints.append(f"maximum: {field_schema['maximum']}")
        ELIF field_type == "string" AND "enum" IN field_schema:
            constraints.append(f"must be one of: {', '.join(field_schema['enum'])}")
        
        constraint_text = f" ({', '.join(constraints)})" IF constraints ELSE ""
        format_parts.append(f"- **{field}** ({field_type}): {description}{constraint_text}")
    
    # Add category-specific validation
    # Add validation requirements from config
    # Add evidence validation rules
    # Add closing instruction
    
    RETURN "\n".join(format_parts)
END METHOD
```

**Key Features**:
- Supports both example-based and schema-based format generation
- Dynamically generates field descriptions with type constraints
- Adds category validation, formatting rules, and evidence validation
- Compatible with BedrockProcessing downstream validation

---

### Component 4: Template Validation Engine

#### Class: `TemplateValidator`
**Purpose**: Validate generated templates for quality and production readiness

**Validation Levels**:
- **minimal**: Basic completeness checks
- **standard**: Component-level validation with scoring (default)
- **strict**: Comprehensive validation with high quality thresholds

---

#### Method: `validate_template`
**Purpose**: Validate template and calculate quality scores

**Algorithm**:
```
METHOD validate_template(template):
    validation_results = {
        "is_valid": True,
        "quality_score": 0.0,
        "validation_details": [],
        "recommendations": []
    }
    
    # Validate system prompt
    system_validation = self._validate_system_prompt(template.get("system_prompt", ""))
    validation_results["validation_details"].append(system_validation)
    
    # Validate user prompt template
    user_validation = self._validate_user_prompt_template(template.get("user_prompt_template", ""))
    validation_results["validation_details"].append(user_validation)
    
    # Validate metadata
    metadata_validation = self._validate_metadata(template.get("metadata", {}))
    validation_results["validation_details"].append(metadata_validation)
    
    # Calculate overall quality score
    scores = [v["score"] FOR v IN validation_results["validation_details"]]
    validation_results["quality_score"] = sum(scores) / len(scores) IF scores ELSE 0.0
    
    # Determine overall validity
    validation_results["is_valid"] = all(v["is_valid"] FOR v IN validation_results["validation_details"])
    
    # Generate recommendations
    validation_results["recommendations"] = self._generate_recommendations(validation_results["validation_details"])
    
    RETURN validation_results
END METHOD
```

**Quality Scoring**:
- **1.0-0.9**: Excellent (production-ready)
- **0.9-0.8**: Good (minor improvements possible)
- **0.8-0.7**: Acceptable (needs review)
- **<0.7**: Poor (requires revision)

---

#### Method: `_validate_system_prompt`
**Purpose**: Validate system prompt component quality

**Algorithm**:
```
METHOD _validate_system_prompt(system_prompt):
    result = {
        "component": "system_prompt",
        "is_valid": True,
        "score": 0.0,
        "issues": []
    }
    
    IF NOT system_prompt OR NOT system_prompt.strip():
        result["is_valid"] = False
        result["issues"].append("System prompt is empty")
        result["score"] = 0.0
        RETURN result
    
    score = 0.0
    
    # Check for role definition (30%)
    IF any(word IN system_prompt.lower() FOR word IN ["you are", "expert", "analyst"]):
        score += 0.3
    ELSE:
        result["issues"].append("Missing clear role definition")
    
    # Check for expertise areas (20%)
    IF any(word IN system_prompt.lower() FOR word IN ["knowledge", "experience", "expertise"]):
        score += 0.2
    ELSE:
        result["issues"].append("Missing expertise statement")
    
    # Check for task context (30%)
    IF any(word IN system_prompt.lower() FOR word IN ["task", "analyze", "classify"]):
        score += 0.3
    ELSE:
        result["issues"].append("Missing task context")
    
    # Check for behavioral guidelines (20%)
    IF any(word IN system_prompt.lower() FOR word IN ["precise", "objective", "thorough"]):
        score += 0.2
    ELSE:
        result["issues"].append("Missing behavioral guidelines")
    
    result["score"] = score
    IF score < 0.7:
        result["is_valid"] = False
    
    RETURN result
END METHOD
```

**Validation Criteria**:
- Role definition (30% weight)
- Expertise statement (20% weight)
- Task context (30% weight)
- Behavioral guidelines (20% weight)

**Production Threshold**: 0.7 minimum score

---

### Component 5: Main Processing Logic

#### Function: `main`
**Purpose**: Main logic for prompt template generation, refactored for testability

**Algorithm**:
```
FUNCTION main(input_paths, output_paths, environ_vars, job_args, logger):
    TRY:
        # Load configurations from JSON files
        prompt_configs_path = input_paths.get("prompt_configs")
        
        IF NOT prompt_configs_path:
            RAISE ValueError("No prompt_configs input path provided")
        
        # Load category definitions (required)
        categories = load_category_definitions(prompt_configs_path, log)
        
        IF NOT categories:
            RAISE ValueError("No category definitions found")
        
        # Load configuration files with defaults
        system_prompt_config = load_config_from_json_file(
            prompt_configs_path, "system_prompt", DEFAULT_SYSTEM_PROMPT_CONFIG, log
        )
        
        output_format_config = load_config_from_json_file(
            prompt_configs_path, "output_format", DEFAULT_OUTPUT_FORMAT_CONFIG, log
        )
        
        instruction_config = load_config_from_json_file(
            prompt_configs_path, "instruction", DEFAULT_INSTRUCTION_CONFIG, log
        )
        
        # Generate or load schema template
        schema_template = self._determine_schema_template(
            output_format_config, categories
        )
        
        # Build configuration dictionary
        config = {
            "TEMPLATE_TASK_TYPE": environ_vars.get("TEMPLATE_TASK_TYPE", "classification"),
            "TEMPLATE_STYLE": environ_vars.get("TEMPLATE_STYLE", "structured"),
            "VALIDATION_LEVEL": environ_vars.get("VALIDATION_LEVEL", "standard"),
            "category_definitions": json.dumps(categories),
            "system_prompt_config": system_prompt_config,
            "output_format_config": output_format_config,
            "instruction_config": instruction_config,
            "INPUT_PLACEHOLDERS": environ_vars.get("INPUT_PLACEHOLDERS", '["input_data"]'),
            "INCLUDE_EXAMPLES": environ_vars.get("INCLUDE_EXAMPLES", "true"),
            "GENERATE_VALIDATION_SCHEMA": environ_vars.get("GENERATE_VALIDATION_SCHEMA", "true"),
            "TEMPLATE_VERSION": environ_vars.get("TEMPLATE_VERSION", "1.0")
        }
        
        # Initialize template generator
        generator = PromptTemplateGenerator(config, schema_template)
        
        # Generate template
        log("Generating prompt template...")
        template = generator.generate_template()
        
        # Validate template
        validator = TemplateValidator(config["VALIDATION_LEVEL"])
        validation_results = validator.validate_template(template)
        
        log(f"Quality score: {validation_results['quality_score']:.2f}")
        
        # Save outputs
        self._save_template_outputs(
            template, validation_results, config, 
            output_paths, log
        )
        
        # Prepare results summary
        results = {
            "success": True,
            "template_generated": True,
            "validation_passed": validation_results["is_valid"],
            "quality_score": validation_results["quality_score"],
            "category_count": len(categories),
            "template_version": config["TEMPLATE_VERSION"],
            "output_files": {...},
            "validation_details": validation_results,
            "generation_timestamp": datetime.now().isoformat()
        }
        
        log("Template generation completed successfully")
        RETURN results
        
    CATCH Exception as e:
        log(f"Template generation failed: {str(e)}")
        RAISE
END FUNCTION
```

**Parameters**:
- `input_paths`: Dictionary of input paths
- `output_paths`: Dictionary of output paths
- `environ_vars`: Environment variables dictionary
- `job_args`: Command line arguments
- `logger`: Optional logger function

**Returns**: Dictionary with generation results and statistics

## Algorithms and Data Structures

### Algorithm 1: Multi-Strategy Placeholder Resolution

**Purpose**: Intelligently resolve placeholders from multiple data sources

**Strategy Priority**:
1. Explicit source hint (highest priority)
2. Placeholder name pattern matching
3. Schema field lookup
4. Fallback placeholder

**Example Execution**:
```
Input: placeholder="${category_enum}", field="category", hint="schema_enum"

Step 1: Check hint → "schema_enum" found
Step 2: Look up schema["properties"]["category"]["enum"]
Step 3: Format enum values: ["Positive", "Negative"] → "One of: Positive, Negative"
Step 4: Return resolved text

Result: "One of: Positive, Negative"
```

**Benefits**:
- Flexible resolution from multiple sources
- Graceful degradation with fallbacks
- Comprehensive tracking and validation

---

### Algorithm 2: Template Quality Scoring

**Purpose**: Calculate multi-component quality scores for generated templates

**Component Weights**:
```
quality_score = (system_prompt_score + user_template_score + metadata_score) / 3
```

**Scoring Rubric**:

**System Prompt** (4 criteria):
- Role definition: 0.30
- Expertise areas: 0.20
- Task context: 0.30
- Behavioral guidelines: 0.20

**User Template** (4 criteria):
- Category definitions: 0.25
- Input placeholders: 0.25
- Instructions: 0.25
- Output format: 0.25

**Metadata** (completeness check):
- All required fields: 1.0
- Missing 1-2 fields: 0.8-0.9
- Missing 3+ fields: <0.7 (fails)

**Production Readiness**:
- ≥ 0.9: Excellent
- ≥ 0.8: Good
- ≥ 0.7: Acceptable
- < 0.7: Requires revision

---

### Data Structure 1: Category Definition

**Purpose**: Comprehensive category specification for classification

**Structure**:
```python
{
    "name": str,                    # Required: unique identifier
    "description": str,             # Required: clear description
    "conditions": List[str],        # Required: classification criteria
    "key_indicators": List[str],    # Required: keywords/patterns
    "exceptions": List[str],        # Optional: exclusion cases
    "examples": List[str],          # Optional: concrete examples
    "priority": int,                # Optional: priority ranking
    "validation_rules": List[str],  # Optional: additional rules
    "aliases": List[str]            # Optional: alternative names
}
```

**Usage Pattern**:
```python
# Load from category_definitions.json
categories = [
    {
        "name": "Positive",
        "description": "Positive sentiment",
        "conditions": ["Contains positive language"],
        "key_indicators": ["good", "great", "excellent"],
        "exceptions": ["Sarcasm"],
        "priority": 1
    }
]

# Sorted by priority (lower = higher)
categories.sort(key=lambda x: x.get("priority", 999))
```

---

### Data Structure 2: Validation Schema

**Purpose**: JSON schema for automated response validation

**Enhanced Structure**:
```python
{
    "title": str,
    "description": str,
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": [...],          # Populated from categories
            "description": str
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": str
        },
        # ... other fields
    },
    "required": List[str],
    "additionalProperties": false,
    "processing_config": {          # Processing metadata
        "format_type": str,
        "response_model_name": str,
        "validation_level": str
    },
    "template_metadata": {          # Template metadata
        "template_version": str,
        "generation_timestamp": str,
        "category_count": int,
        "category_names": List[str]
    }
}
```

**Integration**: Used by BedrockProcessing step for automatic response validation

## Performance Characteristics

### Computational Complexity

**Template Generation**:
- Time: O(n × m) where n = categories, m = average field count
- Space: O(n × k) where k = average configuration size
- Typical: 5-20 categories, <1 second generation time

**Placeholder Resolution**:
- Time: O(p × s) where p = placeholders, s = resolution strategies
- Space: O(p) for placeholder registry
- Typical: 5-15 placeholders, <100ms resolution time

**Template Validation**:
- Time: O(c) where c = validation criteria count
- Space: O(1) constant validation state
- Typical: 10-20 criteria, <50ms validation time

### Resource Usage

**Memory**:
- Configuration loading: ~100-500 KB per config file
- Template generation: ~1-5 MB working memory
- Output files: ~10-100 KB per generated file

**Disk I/O**:
- Reads: 1-4 configuration files (category definitions required, others optional)
- Writes: 2-3 output files (templates, metadata, optional schema)

**Network**: None (local processing only)

## Error Handling

### Configuration Errors

**Missing Required Files**:
```python
# category_definitions.json missing
ERROR: "No category definitions found in prompt configs"
ACTION: Raises ValueError, script exits with error code 1
```

**Invalid JSON Format**:
```python
# Malformed JSON in config file
ERROR: "Failed to load system_prompt config: JSONDecodeError"
ACTION: Falls back to default configuration, logs warning
```

**Missing Required Fields**:
```python
# Category missing required field
ERROR: "Category 0: missing required field 'conditions'"
ACTION: Raises ValueError during validation
```

### Template Generation Errors

**Placeholder Resolution Failures**:
```python
# Placeholder cannot be resolved
WARNING: "Failed to resolve ${category_enum}: Field category not in schema"
ACTION: Uses fallback placeholder "[CATEGORY_UNRESOLVED]", continues generation
```

**Schema Enrichment Failures**:
```python
# Cannot populate schema enum
WARNING: "Cannot enrich schema: No categories available"
ACTION: Uses schema without enum enrichment, logs warning
```

### Validation Errors

**Low Quality Score**:
```python
# Generated template quality < 0.7
WARNING: "Template quality score 0.65 below threshold 0.7"
ACTION: Marks as invalid, includes recommendations, still saves output
```

**Component Validation Failures**:
```python
# System prompt missing required elements
ISSUE: "Missing clear role definition"
ACTION: Reduces component score, includes in validation details
```

### Error Recovery Strategy

1. **Graceful Degradation**: Use defaults when optional configs missing
2. **Fallback Placeholders**: Use descriptive placeholders when resolution fails
3. **Comprehensive Logging**: Log all warnings and errors with context
4. **Partial Success**: Save outputs even with validation warnings
5. **Detailed Reports**: Include all errors in metadata for debugging

## Best Practices

### Configuration Management

**Category Definitions**:
```python
# Good: Clear, specific conditions
{
    "name": "HighRisk",
    "conditions": [
        "Transaction amount > $10,000",
        "From high-risk country",
        "New customer account"
    ],
    "exceptions": [
        "Verified business customer",
        "Approved by manual review"
    ]
}

# Bad: Vague, overlapping conditions
{
    "name": "Risky",
    "conditions": ["Suspicious", "Unusual"],
    "exceptions": []  # No exceptions defined
}
```

**System Prompt Configuration**:
```python
# Good: Specific expertise and clear guidelines
{
    "role_definition": "expert fraud analyst",
    "expertise_areas": [
        "transaction pattern analysis",
        "risk assessment",
        "fraud detection"
    ],
    "behavioral_guidelines": [
        "be thorough in evidence examination",
        "be conservative in risk assessment",
        "provide clear reasoning"
    ]
}

# Bad: Generic, vague descriptions
{
    "role_definition": "analyst",
    "expertise_areas": ["analysis"],
    "behavioral_guidelines": ["be good"]
}
```

### Template Design

**Input Placeholders**:
```python
# Good: Descriptive, specific names
"INPUT_PLACEHOLDERS": '["transaction_details", "customer_profile", "historical_behavior"]'

# Bad: Generic names
"INPUT_PLACEHOLDERS": '["data1", "data2", "data3"]'
```

**Output Format**:
```python
# Good: Clear field descriptions with validation rules
{
    "field_descriptions": {
        "category": "The risk category name (must match exactly: LowRisk, MediumRisk, HighRisk)",
        "confidence": "Confidence score 0.0-1.0 indicating certainty of classification",
        "key_evidence": "Specific transaction patterns and customer behaviors that support the classification"
    },
    "evidence_validation_rules": [
        "Evidence MUST reference specific transaction details",
        "Evidence MUST NOT contradict any category exceptions"
    ]
}

# Bad: Vague descriptions
{
    "field_descriptions": {
        "category": "The category",
        "confidence": "The score",
        "key_evidence": "The evidence"
    }
}
```

### Quality Assurance

**Pre-Generation Checks**:
- Validate all category definitions have required fields
- Check for duplicate category names
- Verify schema template structure
- Test configuration file loading

**Post-Generation Validation**:
- Review quality score (>= 0.8 recommended)
- Check placeholder resolution (all resolved)
- Validate output file generation
- Test template with sample data

**Integration Testing**:
- Test with BedrockProcessing step end-to-end
- Validate schema compatibility
- Check response validation accuracy
- Verify error handling paths

## Example Configurations

### Example 1: Simple Sentiment Classification

**category_definitions.json**:
```json
[
  {
    "name": "Positive",
    "description": "Positive sentiment or favorable opinion",
    "conditions": [
      "Contains positive language or expressions",
      "Expresses satisfaction or approval"
    ],
    "key_indicators": ["good", "great", "excellent", "love", "satisfied"],
    "exceptions": ["Sarcastic statements", "Backhanded compliments"],
    "examples": ["This is great!", "I love this product"],
    "priority": 1
  },
  {
    "name": "Negative",
    "description": "Negative sentiment or unfavorable opinion",
    "conditions": [
      "Contains negative language or criticism",
      "Expresses dissatisfaction or disapproval"
    ],
    "key_indicators": ["bad", "terrible", "hate", "disappointed", "poor"],
    "exceptions": ["Constructive criticism with positive intent"],
    "examples": ["This is terrible", "Very disappointed"],
    "priority": 1
  },
  {
    "name": "Neutral",
    "description": "Neutral sentiment without clear positive or negative opinion",
    "conditions": [
      "Lacks strong emotional language",
      "Presents factual information without judgment"
    ],
    "key_indicators": ["okay", "fine", "average", "acceptable"],
    "examples": ["It works as expected", "Standard quality"],
    "priority": 2
  }
]
```

**Environment Variables**:
```bash
TEMPLATE_TASK_TYPE=classification
TEMPLATE_STYLE=structured
VALIDATION_LEVEL=standard
INPUT_PLACEHOLDERS='["review_text"]'
INCLUDE_EXAMPLES=true
GENERATE_VALIDATION_SCHEMA=true
TEMPLATE_VERSION=1.0
```

### Example 2: Multi-Field Risk Assessment

**category_definitions.json**:
```json
[
  {
    "name": "LowRisk",
    "description": "Low-risk transaction with minimal fraud indicators",
    "conditions": [
      "Transaction amount < $1,000",
      "Established customer (>6 months)",
      "Domestic transaction",
      "Standard shipping address"
    ],
    "key_indicators": [
      "small_amount",
      "verified_customer",
      "domestic",
      "normal_pattern"
    ],
    "exceptions": [
      "Multiple transactions in short timeframe",
      "Unusual for customer profile"
    ],
    "priority": 3
  },
  {
    "name": "HighRisk",
    "description": "High-risk transaction requiring manual review",
    "conditions": [
      "Transaction amount > $5,000",
      "International transaction",
      "New customer (<30 days)",
      "Mismatch in customer data"
    ],
    "key_indicators": [
      "large_amount",
      "international",
      "new_account",
      "data_mismatch"
    ],
    "priority": 1
  }
]
```

**system_prompt.json**:
```json
{
  "role_definition": "expert fraud analyst",
  "expertise_areas": [
    "transaction pattern analysis",
    "risk assessment",
    "fraud detection"
  ],
  "responsibilities": [
    "analyze transaction details accurately",
    "assess fraud risk systematically",
    "provide clear risk justification"
  ],
  "behavioral_guidelines": [
    "be thorough in evidence examination",
    "be conservative in risk assessment",
    "be precise in reasoning"
  ],
  "tone": "professional"
}
```

**INPUT_PLACEHOLDERS**:
```json
'["transaction_amount", "customer_tenure", "transaction_location", "shipping_address", "payment_method"]'
```

## Integration Patterns

### Pipeline Integration

**Two-Step Template-Driven Pipeline**:

```
Step 1: BedrockPromptTemplateGeneration
├── Input: category_definitions.json, config files
├── Output: prompts.json, validation_schema.json
└── Dependencies: None

Step 2: BedrockProcessing
├── Input: prompts.json (from Step 1), validation_schema.json (from Step 1), input_data
├── Output: classified_results.csv
└── Dependencies: BedrockPromptTemplateGeneration outputs
```

**Configuration-Driven Approach**:
- Prompts defined in JSON configurations
- No hard-coded prompts in code
- Easy updates without code changes
- Version control for prompt templates

### Bedrock Processing Integration

**Template Consumption**:
```python
# Bedrock Processing loads generated template
with open("/opt/ml/processing/input/prompt_templates/prompts.json") as f:
    template = json.load(f)

# Format with actual data
user_prompt = template["user_prompt_template"].format(
    transaction_amount="$12,500",
    customer_tenure="15 days",
    transaction_location="International",
    shipping_address="Non-standard",
    payment_method="New card"
)

# Invoke Bedrock
response = bedrock_client.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "system": template["system_prompt"],
        "messages": [{
            "role": "user",
            "content": user_prompt
        }]
    })
)
```

**Validation Integration**:
```python
# Load validation schema
with open("/opt/ml/processing/input/validation_schema/validation_schema_*.json") as f:
    schema = json.load(f)

# Validate Bedrock response
import jsonschema
try:
    jsonschema.validate(instance=response_json, schema=schema)
    # Response is valid
except jsonschema.ValidationError as e:
    # Handle validation error
    logger.error(f"Response validation failed: {e.message}")
```

### Dependency Resolution

**Automatic Path Resolution**:
- BedrockProcessing step automatically detects template outputs
- Uses specification-driven dependency matching
- No manual path configuration required

**Example Specification**:
```python
# BedrockPromptTemplateGeneration spec
outputs = [
    {"name": "prompt_templates", "type": "templates"},
    {"name": "validation_schema", "type": "schema"}
]

# BedrockProcessing spec
inputs = [
    {"name": "prompt_templates", "type": "templates", "required": True},
    {"name": "validation_schema", "type": "schema", "required": False}
]

# Framework automatically matches outputs → inputs
```

## Troubleshooting

### Issue 1: Template Quality Score Too Low

**Symptom**: Quality score < 0.7, template marked as invalid

**Diagnosis**:
```python
# Check validation details in metadata
{
    "validation_results": {
        "quality_score": 0.65,
        "validation_details": [
            {
                "component": "system_prompt",
                "score": 0.5,
                "issues": ["Missing clear role definition", "Missing task context"]
            }
        ]
    }
}
```

**Solutions**:
1. Enhance system_prompt.json with clearer role definition
2. Add specific expertise areas and behavioral guidelines
3. Review tone setting (use "professional" for most cases)
4. Regenerate template and verify improved score

---

### Issue 2: Placeholder Resolution Failures

**Symptom**: Placeholders show "[FIELD_UNRESOLVED]" in generated template

**Diagnosis**:
```python
# Check placeholder validation in metadata
{
    "placeholder_validation": {
        "all_resolved": false,
        "failures": [
            {
                "placeholder": "category_enum",
                "field": "category",
                "error": "Field category not in schema"
            }
        ]
    }
}
```

**Solutions**:
1. Verify schema_template includes all required fields
2. Check category enrichment succeeded (category enum populated)
3. Verify output_format.json includes proper field definitions
4. Check placeholder names match field names in schema

---

### Issue 3: Category Enum Not Populated

**Symptom**: Schema category field has no enum values

**Diagnosis**:
```python
# Check schema properties
{
    "properties": {
        "category": {
            "type": "string",
            "enum": [],  # Empty!
            "description": "The classified category name"
        }
    }
}
```

**Solutions**:
1. Verify category_definitions.json loaded successfully
2. Check categories have valid "name" fields
3. Verify schema enrichment executed (check logs for "Enriched schema" message)
4. Ensure schema_template has category field before enrichment

---

### Issue 4: BedrockProcessing Cannot Find Template

**Symptom**: Downstream step fails with "Template file not found"

**Diagnosis**:
- Check output path configuration
- Verify file actually created (check container paths)
- Review dependency resolution in pipeline

**Solutions**:
1. Verify output path matches BedrockProcessing input path
2. Check SageMaker ProcessingStep output configuration
3. Use specification-driven dependency resolution
4. Verify template file exists: `/opt/ml/processing/output/templates/prompts.json`

---

### Issue 5: Schema Validation Fails in BedrockProcessing

**Symptom**: All Bedrock responses fail schema validation

**Diagnosis**:
```python
# Check validation errors
ValidationError: "'PositiveCategory' is not one of ['Positive', 'Negative', 'Neutral']"
```

**Solutions**:
1. Verify category names match exactly (case-sensitive)
2. Check schema enum values populated correctly
3. Ensure LLM instructions emphasize exact category name matching
4. Review category names for typos or inconsistencies

## References

### Related Scripts
- [`bedrock_processing.py`](bedrock_processing_script.md): Downstream real-time processing using generated templates
- [`bedrock_batch_processing.py`](bedrock_batch_processing_script.md): Downstream batch processing using generated templates

### Related Documentation
- **Contract**: [`src/cursus/steps/contracts/bedrock_prompt_template_generation_contract.py`](../../src/cursus/steps/contracts/bedrock_prompt_template_generation_contract.py)

### Related Design Documents
- **[Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md)**: Complete step builder patterns with contract, spec, config, and builder implementations for the 5-component architecture
- **[Bedrock Prompt Template Generation Input Formats](../1_design/bedrock_prompt_template_generation_input_formats.md)**: Comprehensive specification of all input configuration formats including category definitions, system prompts, output formats, and instruction configurations
- **[Bedrock Prompt Template Generation Output Design](../1_design/bedrock_prompt_template_generation_output_design.md)**: Detailed design of generated outputs including template structure, validation schemas, and metadata formats
- **[Bedrock Prompt Template Generation Buyer Seller Example](../1_design/bedrock_prompt_template_generation_buyer_seller_example.md)**: Complete end-to-end example demonstrating buyer-seller classification with full configuration files and expected outputs

### External Resources
- **AWS Bedrock Documentation**: [https://docs.aws.amazon.com/bedrock/](https://docs.aws.amazon.com/bedrock/) - AWS Bedrock service documentation for LLM integration
