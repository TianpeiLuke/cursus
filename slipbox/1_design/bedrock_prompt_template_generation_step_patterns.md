---
tags:
  - design
  - step_builders
  - bedrock_steps
  - prompt_generation
  - template_patterns
  - sagemaker
  - llm_processing
  - classification_tasks
keywords:
  - bedrock prompt template generation
  - classification prompt patterns
  - structured prompt templates
  - category definition templates
  - LLM prompt engineering
  - template generation automation
topics:
  - step builder patterns
  - prompt template generation
  - SageMaker LLM processing
  - bedrock integration
  - classification task automation
language: python
date of note: 2025-10-31
---

# Bedrock Prompt Template Generation Step Builder Patterns

## Overview

This document defines the design patterns for Bedrock prompt template generation step builder implementations in the cursus framework. The Bedrock prompt template generation step creates **ProcessingStep** instances that generate structured, reusable prompt templates specifically designed for categorization and classification tasks. These templates follow a standardized 5-component structure and integrate seamlessly with the existing Bedrock processing steps.

## SageMaker Step Type Classification

Bedrock prompt template generation steps create **ProcessingStep** instances using **SKLearnProcessor** with specialized template generation logic:
- **SKLearnProcessor**: Standard processing framework for template generation and validation
- **Template Structure**: 5-component prompt template architecture for classification tasks
- **Category Management**: Structured category definitions with conditions and exceptions
- **Schema Generation**: Automated output format and validation schema creation
- **Integration Ready**: Direct compatibility with existing Bedrock processing steps

## Key Differences from Standard Processing Steps

### 1. Template-Centric Processing Pattern
```python
# Standard Processing Step: Data transformation focus
processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type=config.processing_instance_type
)

# Prompt Template Generation Step: Template creation focus
class PromptTemplateGenerator:
    def __init__(self, config):
        self.task_type = config.task_type
        self.categories = config.category_definitions
        self.output_schema = config.output_schema_config
        self.template_style = config.template_style
        
    def generate_template(self):
        return {
            'system_prompt': self._generate_system_prompt(),
            'category_definitions': self._generate_category_definitions(),
            'input_placeholders': self._generate_input_placeholders(),
            'instructions': self._generate_instructions(),
            'output_format': self._generate_output_format()
        }
```

### 2. Category-Driven Configuration Pattern
```python
# User-configurable category definitions with structured metadata
class BedrockPromptTemplateGenerationStepConfig(ProcessingStepConfigBase):
    # Task Configuration
    task_type: str = Field(
        default="classification",
        description="Type of task: classification, categorization, analysis"
    )
    
    # Category Definitions - Core of the template generation
    category_definitions: List[CategoryDefinition] = Field(
        default_factory=list,
        description="List of category definitions with conditions and exceptions"
    )
    
    # Template Structure Configuration
    template_components: TemplateComponents = Field(
        default_factory=TemplateComponents,
        description="Configuration for each template component"
    )
    
    # Output Schema Configuration
    output_schema_config: OutputSchemaConfig = Field(
        default_factory=OutputSchemaConfig,
        description="Configuration for output format and validation schema"
    )

@dataclass
class CategoryDefinition:
    """Definition of a single category for classification"""
    name: str
    description: str
    conditions: List[str]
    exceptions: List[str]
    key_indicators: List[str]
    examples: Optional[List[str]] = None
    priority: int = 1
    validation_rules: Optional[List[str]] = None
```

### 3. Structured Template Generation Pattern
```python
# Standard Processing: Fixed output format
outputs = [
    ProcessingOutput(
        output_name="processed_data",
        source="/opt/ml/processing/output/data"
    )
]

# Prompt Template Generation: Structured template outputs
class TemplateComponents:
    """Configuration for template component generation"""
    
    def __init__(self):
        self.system_prompt_config = SystemPromptConfig()
        self.category_section_config = CategorySectionConfig()
        self.input_placeholder_config = InputPlaceholderConfig()
        self.instruction_config = InstructionConfig()
        self.output_format_config = OutputFormatConfig()

@dataclass
class SystemPromptConfig:
    """Configuration for system prompt generation"""
    role_definition: str = "expert analyst"
    expertise_areas: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    behavioral_guidelines: List[str] = field(default_factory=list)
    tone: str = "professional"

@dataclass
class OutputFormatConfig:
    """Configuration for output format generation"""
    format_type: str = "structured_json"  # structured_json, formatted_text, hybrid
    required_fields: List[str] = field(default_factory=list)
    field_descriptions: Dict[str, str] = field(default_factory=dict)
    validation_requirements: List[str] = field(default_factory=list)
    example_output: Optional[str] = None
```

### 4. Template Validation and Quality Assurance Pattern
```python
# Standard Processing: Basic validation
def validate_configuration(self) -> None:
    if not hasattr(self.config, 'required_attr'):
        raise ValueError("Missing required attribute")

# Prompt Template Generation: Comprehensive template validation
class TemplateValidator:
    """Validates generated prompt templates for quality and completeness"""
    
    def validate_template(self, template: Dict[str, Any]) -> ValidationResult:
        validation_results = []
        
        # Validate system prompt
        validation_results.append(self._validate_system_prompt(template['system_prompt']))
        
        # Validate category definitions
        validation_results.append(self._validate_categories(template['category_definitions']))
        
        # Validate input placeholders
        validation_results.append(self._validate_placeholders(template['input_placeholders']))
        
        # Validate instructions
        validation_results.append(self._validate_instructions(template['instructions']))
        
        # Validate output format
        validation_results.append(self._validate_output_format(template['output_format']))
        
        return ValidationResult(
            is_valid=all(r.is_valid for r in validation_results),
            validation_details=validation_results,
            quality_score=self._calculate_quality_score(validation_results),
            recommendations=self._generate_recommendations(validation_results)
        )
```

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Bedrock prompt template generation step builders follow this architecture:

```python
@register_builder()
class BedrockPromptTemplateGenerationStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load prompt template generation specification
        spec = BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate prompt template generation configuration
        
    def _create_processor(self) -> SKLearnProcessor:
        # Create SKLearnProcessor for template generation
        
    def _get_environment_variables(self) -> Dict[str, str]:
        # Build template generation environment variables
        
    def _get_template_generation_config(self) -> Dict[str, Any]:
        # Build template generation configuration
        
    def _get_inputs(self, inputs) -> List[ProcessingInput]:
        # Create ProcessingInput objects for category definitions and configs
        
    def _get_outputs(self, outputs) -> List[ProcessingOutput]:
        # Create ProcessingOutput objects for generated templates
        
    def _get_job_arguments(self) -> List[str]:
        # Build command-line arguments for template generation
        
    def create_step(self, **kwargs) -> ProcessingStep:
        # Orchestrate prompt template generation step creation
```

### 2. Template Generation Strategy Pattern

```python
def _get_template_generation_config(self) -> Dict[str, Any]:
    """Build template generation configuration based on task requirements."""
    
    config = {
        'task_type': self.config.task_type,
        'template_style': self.config.template_style,
        'categories': [asdict(cat) for cat in self.config.category_definitions],
        'output_schema': asdict(self.config.output_schema_config),
        'validation_level': self.config.validation_level
    }
    
    # Add component-specific configurations
    config['system_prompt_config'] = asdict(self.config.template_components.system_prompt_config)
    config['category_section_config'] = asdict(self.config.template_components.category_section_config)
    config['input_placeholder_config'] = asdict(self.config.template_components.input_placeholder_config)
    config['instruction_config'] = asdict(self.config.template_components.instruction_config)
    config['output_format_config'] = asdict(self.config.template_components.output_format_config)
    
    return config
```

### 3. Environment Variables Pattern for Template Generation (Simplified)

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Build template generation environment variables."""
    env_vars = super()._get_environment_variables()
    
    # Task configuration (optimal defaults)
    env_vars["TEMPLATE_TASK_TYPE"] = self.config.task_type  # Always "classification"
    env_vars["TEMPLATE_STYLE"] = self.config.template_style  # Always "structured"
    env_vars["VALIDATION_LEVEL"] = self.config.validation_level  # Always "standard"
    
    # Template component configurations as JSON (auto-configured)
    env_vars["SYSTEM_PROMPT_CONFIG"] = json.dumps(
        asdict(self.config.template_components.system_prompt_config)
    )
    env_vars["OUTPUT_FORMAT_CONFIG"] = json.dumps(
        asdict(self.config.template_components.output_format_config)
    )
    env_vars["INSTRUCTION_CONFIG"] = json.dumps(
        asdict(self.config.template_components.instruction_config)
    )
    
    # Input placeholder configuration (smart defaults)
    env_vars["INPUT_PLACEHOLDERS"] = json.dumps(self.config.input_placeholders)
    env_vars["ADDITIONAL_CONTEXT_FIELDS"] = json.dumps(self.config.additional_context_fields)
    
    # Output configuration (optimal defaults)
    env_vars["OUTPUT_FORMAT_TYPE"] = self.config.output_schema_config.format_type
    env_vars["REQUIRED_OUTPUT_FIELDS"] = json.dumps(self.config.output_schema_config.required_fields)
    
    # Quality and validation settings (optimal defaults)
    env_vars["INCLUDE_EXAMPLES"] = str(self.config.include_examples)
    env_vars["GENERATE_VALIDATION_SCHEMA"] = str(self.config.generate_validation_schema)
    env_vars["TEMPLATE_VERSION"] = self.config.template_version
    
    return env_vars
```

**Key Changes**:
- **Removed**: `CATEGORY_DEFINITIONS` env var (now loaded from input files)
- **Input-based**: Category definitions, task requirements, and output schema templates are loaded from S3 inputs
- **Environment-based**: Only configuration settings and auto-configured components
- **No Overlap**: Clear separation between file inputs and configuration environment variables

### 4. Job Arguments Pattern for Template Generation (Simplified)

```python
def _get_job_arguments(self) -> List[str]:
    """Build command-line arguments for template generation script."""
    args = []
    
    # Only include arguments that provide functionality not covered by environment variables
    # Most configuration is passed via environment variables to reduce redundancy
    
    if self.config.include_examples:
        args.append("--include-examples")
        
    if self.config.generate_validation_schema:
        args.append("--generate-validation-schema")
    
    return args
```

**Redundancy Elimination**:
- **Removed**: `--task-type` (available as `TEMPLATE_TASK_TYPE` env var)
- **Removed**: `--template-style` (available as `TEMPLATE_STYLE` env var)
- **Removed**: `--validation-level` (available as `VALIDATION_LEVEL` env var)
- **Removed**: `--output-format` (available as `OUTPUT_FORMAT_TYPE` env var)
- **Removed**: `--template-version` (available as `TEMPLATE_VERSION` env var)
- **Kept**: `--include-examples` and `--generate-validation-schema` (boolean flags for optional features)

### 5. Specification-Driven Input/Output Pattern for Template Generation

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """Create ProcessingInput objects for template generation."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_inputs = []
    
    for _, dependency_spec in self.spec.dependencies.items():
        logical_name = dependency_spec.logical_name
        
        # Skip optional inputs not provided
        if not dependency_spec.required and logical_name not in inputs:
            continue
            
        # Validate required inputs
        if dependency_spec.required and logical_name not in inputs:
            raise ValueError(f"Required input '{logical_name}' not provided")
        
        # Get container path from contract
        container_path = self.contract.expected_input_paths[logical_name]
        
        processing_inputs.append(ProcessingInput(
            input_name=logical_name,
            source=inputs[logical_name],
            destination=container_path
        ))
        
    return processing_inputs

def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """Create ProcessingOutput objects for generated templates."""
    if not self.spec or not self.contract:
        raise ValueError("Step specification and contract are required")
        
    processing_outputs = []
    
    for _, output_spec in self.spec.outputs.items():
        logical_name = output_spec.logical_name
        container_path = self.contract.expected_output_paths[logical_name]
        
        # Use provided destination or generate default
        destination = outputs.get(logical_name) or self._generate_output_path(logical_name)
        
        processing_outputs.append(ProcessingOutput(
            output_name=logical_name,
            source=container_path,
            destination=destination
        ))
        
    return processing_outputs
```

### 6. Processor Creation Pattern for Template Generation

```python
def _create_processor(self) -> SKLearnProcessor:
    """Create SKLearnProcessor for template generation."""
    instance_type = (self.config.processing_instance_type_large 
                    if self.config.use_large_processing_instance 
                    else self.config.processing_instance_type_small)
    
    return SKLearnProcessor(
        framework_version=self.config.processing_framework_version,
        role=self.role,
        instance_type=instance_type,
        instance_count=self.config.processing_instance_count,
        volume_size_in_gb=self.config.processing_volume_size,
        base_job_name=self._generate_job_name(),
        sagemaker_session=self.session,
        env=self._get_environment_variables(),
    )
```

### 7. Step Creation Pattern for Template Generation

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """Create Bedrock Prompt Template Generation ProcessingStep."""
    # Extract parameters
    inputs_raw = kwargs.get('inputs', {})
    outputs = kwargs.get('outputs', {})
    dependencies = kwargs.get('dependencies', [])
    enable_caching = kwargs.get('enable_caching', True)
    
    # Handle inputs from dependencies and explicit inputs
    inputs = {}
    if dependencies:
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        inputs.update(extracted_inputs)
    inputs.update(inputs_raw)
    
    # Create components
    processor = self._create_processor()
    proc_inputs = self._get_inputs(inputs)
    proc_outputs = self._get_outputs(outputs)
    job_args = self._get_job_arguments()
    
    # Get standardized step name
    step_name = self._get_step_name()
    
    # Create step directly (Pattern A - same as standard processing steps)
    step = ProcessingStep(
        name=step_name,
        processor=processor,
        inputs=proc_inputs,
        outputs=proc_outputs,
        code=self.config.get_script_path(),
        job_arguments=job_args,
        depends_on=dependencies,
        cache_config=self._get_cache_config(enable_caching)
    )
    
    # Attach specification for future reference
    setattr(step, '_spec', self.spec)
    
    return step
```

## Configuration Validation Patterns

### Standard Template Generation Configuration Validation
```python
def validate_configuration(self) -> None:
    """Validate prompt template generation configuration."""
    # Validate base processing configuration
    required_processing_attrs = [
        'processing_instance_count', 'processing_volume_size',
        'processing_instance_type_large', 'processing_instance_type_small',
        'processing_framework_version', 'use_large_processing_instance'
    ]
    
    for attr in required_processing_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required processing attribute: {attr}")
    
    # Validate template generation specific configuration
    required_template_attrs = [
        'task_type', 'template_style', 'category_definitions', 
        'output_schema_config', 'validation_level'
    ]
    
    for attr in required_template_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required template attribute: {attr}")
    
    # Validate category definitions
    if not self.config.category_definitions:
        raise ValueError("At least one category definition is required")
    
    for i, category in enumerate(self.config.category_definitions):
        if not category.name or not category.description:
            raise ValueError(f"Category {i}: name and description are required")
        if not category.conditions:
            raise ValueError(f"Category {i}: at least one condition is required")
    
    # Validate task type
    valid_task_types = ['classification', 'categorization', 'analysis', 'evaluation']
    if self.config.task_type not in valid_task_types:
        raise ValueError(f"Invalid task_type: {self.config.task_type}")
    
    # Validate template style
    valid_styles = ['structured', 'conversational', 'technical', 'detailed']
    if self.config.template_style not in valid_styles:
        raise ValueError(f"Invalid template_style: {self.config.template_style}")
    
    # Validate output format
    valid_formats = ['structured_json', 'formatted_text', 'hybrid']
    if self.config.output_schema_config.format_type not in valid_formats:
        raise ValueError(f"Invalid output format_type: {self.config.output_schema_config.format_type}")
    
    self.log_info("BedrockPromptTemplateGenerationStepConfig validation succeeded")
```

### Category Definition Validation Pattern
```python
def _validate_category_definitions(self, categories: List[CategoryDefinition]) -> None:
    """Validate category definitions for completeness and consistency."""
    category_names = set()
    
    for i, category in enumerate(categories):
        # Check for duplicate names
        if category.name in category_names:
            raise ValueError(f"Duplicate category name: {category.name}")
        category_names.add(category.name)
        
        # Validate required fields
        if not category.name.strip():
            raise ValueError(f"Category {i}: name cannot be empty")
        if not category.description.strip():
            raise ValueError(f"Category {i}: description cannot be empty")
        if not category.conditions:
            raise ValueError(f"Category {i}: at least one condition is required")
        if not category.key_indicators:
            raise ValueError(f"Category {i}: at least one key indicator is required")
        
        # Validate priority
        if not isinstance(category.priority, int) or category.priority < 1:
            raise ValueError(f"Category {i}: priority must be a positive integer")
        
        # Validate conditions and exceptions format
        for j, condition in enumerate(category.conditions):
            if not condition.strip():
                raise ValueError(f"Category {i}, condition {j}: cannot be empty")
        
        for j, exception in enumerate(category.exceptions):
            if not exception.strip():
                raise ValueError(f"Category {i}, exception {j}: cannot be empty")
```

## Design Components Integration

### 1. Step Specification Pattern

```python
# specs/bedrock_prompt_template_generation_spec.py
BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("BedrockPromptTemplateGeneration"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_prompt_template_generation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="category_definitions",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["ProcessingStep", "DataLoad", "ConfigPrep"],
            semantic_keywords=["categories", "definitions", "classification", "config", "schema", "taxonomy"],
            data_type="S3Uri",
            description="Category definitions with conditions, exceptions, and metadata for template generation"
        ),
        DependencySpec(
            logical_name="task_requirements",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["ProcessingStep", "ConfigPrep", "DataLoad"],
            semantic_keywords=["requirements", "task", "config", "specification", "parameters", "settings"],
            data_type="S3Uri",
            description="Optional task requirements and configuration parameters for template customization"
        ),
        DependencySpec(
            logical_name="output_schema_template",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["ProcessingStep", "SchemaPrep", "ConfigPrep"],
            semantic_keywords=["schema", "template", "format", "structure", "output", "validation"],
            data_type="S3Uri",
            description="Optional output schema template for customizing generated output format"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="prompt_templates",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['prompt_templates'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated prompt templates in JSON format ready for Bedrock processing",
            aliases=["templates", "prompts", "prompt_config", "generated_templates"]
        ),
        OutputSpec(
            logical_name="template_metadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['template_metadata'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Metadata about generated templates including validation results and quality metrics",
            aliases=["metadata", "validation_report", "template_info", "quality_metrics"]
        ),
        OutputSpec(
            logical_name="validation_schema",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['validation_schema'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated validation schema for output format validation",
            aliases=["schema", "validation", "output_schema", "format_schema"]
        )
    ]
)
```

### 2. Script Contract Pattern

```python
# contracts/bedrock_prompt_template_generation_contract.py
BEDROCK_PROMPT_TEMPLATE_GENERATION_CONTRACT = ProcessingScriptContract(
    entry_point="bedrock_prompt_template_generation.py",
    expected_input_paths={
        "category_definitions": "/opt/ml/processing/input/categories",
        "task_requirements": "/opt/ml/processing/input/requirements",
        "output_schema_template": "/opt/ml/processing/input/schema"
    },
    expected_output_paths={
        "prompt_templates": "/opt/ml/processing/output/templates",
        "template_metadata": "/opt/ml/processing/output/metadata",
        "validation_schema": "/opt/ml/processing/output/schema"
    },
    expected_arguments={
        # Removed redundant arguments that are already provided via environment variables:
        # "--task-type": Already available as TEMPLATE_TASK_TYPE env var
        # "--template-style": Already available as TEMPLATE_STYLE env var  
        # "--validation-level": Already available as VALIDATION_LEVEL env var
        # "--output-format": Already available as OUTPUT_FORMAT_TYPE env var
        # "--template-version": Already available as TEMPLATE_VERSION env var
        
        # Keep only arguments that provide additional functionality not covered by env vars
        "--include-examples": "Include examples in generated templates (flag)",
        "--generate-validation-schema": "Generate validation schema (flag)"
    },
    required_env_vars=[
        "TEMPLATE_TASK_TYPE",
        "TEMPLATE_STYLE", 
        "OUTPUT_FORMAT_TYPE",
        "VALIDATION_LEVEL"
    ],
    optional_env_vars={
        "SYSTEM_PROMPT_CONFIG": "JSON configuration for system prompt generation",
        "OUTPUT_FORMAT_CONFIG": "JSON configuration for output format generation",
        "INPUT_PLACEHOLDERS": "JSON list of input placeholder configurations",
        "ADDITIONAL_CONTEXT_FIELDS": "JSON list of additional context field names",
        "REQUIRED_OUTPUT_FIELDS": "JSON list of required output field names",
        "INCLUDE_EXAMPLES": "Whether to include examples in generated templates",
        "GENERATE_VALIDATION_SCHEMA": "Whether to generate validation schema",
        "TEMPLATE_VERSION": "Version identifier for generated templates"
    },
    framework_requirements={
        "pydantic": ">=2.0.0",
        "jinja2": ">=3.0.0",
        "jsonschema": ">=4.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "numpy": ">=1.19.0"
    },
    description="""
    Bedrock prompt template generation script that:
    1. Loads category definitions and task requirements from input files
    2. Generates structured prompt templates with 5-component architecture:
       - System prompt with role assignment and expertise definition
       - Category definitions with conditions, exceptions, and key indicators
       - Input placeholders for data and context variables
       - Instructions and rules for LLM inference guidance
       - Output format schema with field definitions and validation rules
    3. Validates generated templates for completeness and quality
    4. Outputs templates in JSON format compatible with Bedrock processing steps
    5. Generates validation schemas and metadata for template quality assurance
    
    Template Generation Features:
    - Configurable template styles (structured, conversational, technical, detailed)
    - Dynamic category definition processing with priority handling
    - Automated output schema generation with field validation
    - Template quality scoring and validation reporting
    - Integration-ready output format for seamless Bedrock processing
    
    Input Structure:
    - /opt/ml/processing/input/categories: Category definitions (JSON/CSV)
    - /opt/ml/processing/input/requirements: Task requirements (JSON)
    - /opt/ml/processing/input/schema: Output schema template (JSON)
    
    Output Structure:
    - /opt/ml/processing/output/templates: Generated prompt templates (prompts.json)
    - /opt/ml/processing/output/metadata: Template metadata and validation results
    - /opt/ml/processing/output/schema: Generated validation schemas
    
    Template Structure (5-Component Architecture):
    1. System Prompt: Role definition, expertise areas, behavioral guidelines
    2. Category Definitions: Structured category descriptions with conditions/exceptions
    3. Input Placeholders: Variable placeholders for data and context injection
    4. Instructions: Processing rules, guidelines, and inference directions
    5. Output Format: Structured schema with field definitions and validation rules
    """
)
```

### 3. Configuration Class Pattern

```python
# configs/config_bedrock_prompt_template_generation_step.py
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pydantic import Field, BaseModel

class BedrockPromptTemplateGenerationStepConfig(ProcessingStepConfigBase):
    """Configuration for Bedrock prompt template generation step."""
    
    def __init__(self):
        super().__init__()
        
        # Core Required Configuration - Only thing user needs to provide
        self.category_definitions: List[CategoryDefinition] = []
        
        # Optimal Defaults - No user configuration needed
        self._task_type: str = "classification"  # Always optimal for categorization
        self._template_style: str = "structured"  # Always optimal for automated processing
        self._validation_level: str = "standard"  # Always optimal balance
        
        # Auto-configured based on category complexity
        self.template_components: TemplateComponents = TemplateComponents()
        self.output_schema_config: OutputSchemaConfig = OutputSchemaConfig()
        
        # Smart defaults for common use cases
        self.input_placeholders: List[str] = ["dialogue", "shiptrack", "max_estimated_arrival_date"]
        self.additional_context_fields: List[str] = []
        
        # Auto-enabled features
        self.include_examples: bool = True
        self.generate_validation_schema: bool = True
        self.template_version: str = "1.0"
        
        # Quality control with optimal settings
        self.min_quality_score: float = 0.8
        self.enable_quality_checks: bool = True
        
    def _auto_configure_from_categories(self):
        """Automatically configure optimal settings based on category complexity."""
        if not self.category_definitions:
            return
            
        # Auto-detect complexity indicators
        category_count = len(self.category_definitions)
        has_examples = any(cat.examples for cat in self.category_definitions if cat.examples)
        has_exceptions = any(cat.exceptions for cat in self.category_definitions if cat.exceptions)
        avg_conditions = sum(len(cat.conditions) for cat in self.category_definitions) / category_count
        
        # Auto-configure template components based on complexity
        if category_count > 10 or has_examples or has_exceptions or avg_conditions > 3:
            # Complex categorization detected
            self.template_components.category_section_config.detailed_conditions = True
            self.template_components.category_section_config.exception_handling = True
            self.template_components.instruction_config.include_edge_case_handling = True
        else:
            # Simple classification detected
            self.template_components.category_section_config.detailed_conditions = False
            self.template_components.category_section_config.exception_handling = False
            self.template_components.instruction_config.include_edge_case_handling = False
        
        # Auto-configure output format based on category names and structure
        category_names = [cat.name for cat in self.category_definitions]
        self.output_schema_config.field_descriptions["category"] = f"Exactly one of: {', '.join(category_names)}"
    
    @property
    def task_type(self) -> str:
        """Always returns optimal task type."""
        return self._task_type
    
    @property 
    def template_style(self) -> str:
        """Always returns optimal template style."""
        return self._template_style
        
    @property
    def validation_level(self) -> str:
        """Always returns optimal validation level."""
        return self._validation_level

@dataclass
class CategoryDefinition:
    """Definition of a single category for classification tasks."""
    name: str
    description: str
    conditions: List[str]
    exceptions: List[str]
    key_indicators: List[str]
    examples: Optional[List[str]] = None
    priority: int = 1
    validation_rules: Optional[List[str]] = None
    aliases: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate category definition after initialization."""
        if not self.name or not self.name.strip():
            raise ValueError("Category name cannot be empty")
        if not self.description or not self.description.strip():
            raise ValueError("Category description cannot be empty")
        if not self.conditions:
            raise ValueError("At least one condition is required")
        if not self.key_indicators:
            raise ValueError("At least one key indicator is required")

@dataclass
class TemplateComponents:
    """Configuration for template component generation."""
    system_prompt_config: 'SystemPromptConfig' = field(default_factory=lambda: SystemPromptConfig())
    category_section_config: 'CategorySectionConfig' = field(default_factory=lambda: CategorySectionConfig())
    input_placeholder_config: 'InputPlaceholderConfig' = field(default_factory=lambda: InputPlaceholderConfig())
    instruction_config: 'InstructionConfig' = field(default_factory=lambda: InstructionConfig())
    output_format_config: 'OutputFormatConfig' = field(default_factory=lambda: OutputFormatConfig())

@dataclass
class SystemPromptConfig:
    """Configuration for system prompt generation."""
    role_definition: str = "expert analyst"
    expertise_areas: List[str] = field(default_factory=lambda: ["data analysis", "classification"])
    responsibilities: List[str] = field(default_factory=lambda: ["analyze data", "classify content", "provide insights"])
    behavioral_guidelines: List[str] = field(default_factory=lambda: ["be precise", "be objective", "be thorough"])
    tone: str = "professional"
    include_expertise_statement: bool = True
    include_task_context: bool = True

@dataclass
class CategorySectionConfig:
    """Configuration for category section generation."""
    include_priority_ordering: bool = True
    include_examples: bool = True
    include_validation_rules: bool = True
    detailed_conditions: bool = True
    exception_handling: bool = True
    cross_category_guidance: bool = True

@dataclass
class InputPlaceholderConfig:
    """Configuration for input placeholder generation."""
    placeholder_format: str = "curly_braces"  # curly_braces, angle_brackets, custom
    include_descriptions: bool = True
    include_data_types: bool = True
    include_examples: bool = False
    custom_format_template: Optional[str] = None

@dataclass
class InstructionConfig:
    """Configuration for instruction section generation."""
    include_analysis_steps: bool = True
    include_decision_criteria: bool = True
    include_edge_case_handling: bool = True
    include_confidence_guidance: bool = True
    include_reasoning_requirements: bool = True
    step_by_step_format: bool = True

@dataclass
class OutputFormatConfig:
    """Configuration for output format generation."""
    format_type: str = "structured_json"  # structured_json, formatted_text, hybrid
    required_fields: List[str] = field(default_factory=lambda: ["category", "confidence", "reasoning"])
    field_descriptions: Dict[str, str] = field(default_factory=dict)
    validation_requirements: List[str] = field(default_factory=list)
    example_output: Optional[str] = None
    include_field_constraints: bool = True
    include_formatting_rules: bool = True
    
    def __post_init__(self):
        """Set default field descriptions if not provided."""
        if not self.field_descriptions:
            self.field_descriptions = {
                "category": "The classified category name",
                "confidence": "Confidence score between 0.0 and 1.0",
                "reasoning": "Explanation of the classification decision"
            }

    def validate_template_generation_configuration(self) -> None:
        """Validate template generation configuration - simplified validation."""
        # Only validate what user actually provides
        if not self.category_definitions:
            raise ValueError("At least one category definition is required")
        
        # Validate each category definition
        category_names = set()
        for i, category in enumerate(self.category_definitions):
            if category.name in category_names:
                raise ValueError(f"Duplicate category name: {category.name}")
            category_names.add(category.name)
        
        # Auto-configure based on categories after validation
        self._auto_configure_from_categories()
        
        # All other settings use optimal defaults - no validation needed
    
    def get_template_generation_summary(self) -> Dict[str, Any]:
        """Get summary of template generation configuration."""
        return {
            'task_type': self.task_type,  # Always "classification"
            'template_style': self.template_style,  # Always "structured"
            'validation_level': self.validation_level,  # Always "standard"
            'category_count': len(self.category_definitions),
            'category_names': [cat.name for cat in self.category_definitions],
            'output_format': self.output_schema_config.format_type,  # Always "structured_json"
            'required_fields': self.output_schema_config.required_fields,
            'input_placeholders': self.input_placeholders,
            'auto_configured': True,
            'complexity_detected': self._detect_complexity(),
            'template_version': self.template_version,
            'quality_checks_enabled': self.enable_quality_checks,
            'min_quality_score': self.min_quality_score
        }
    
    def _detect_complexity(self) -> str:
        """Detect complexity level based on category definitions."""
        if not self.category_definitions:
            return "simple"
            
        category_count = len(self.category_definitions)
        has_examples = any(cat.examples for cat in self.category_definitions if cat.examples)
        has_exceptions = any(cat.exceptions for cat in self.category_definitions if cat.exceptions)
        avg_conditions = sum(len(cat.conditions) for cat in self.category_definitions) / category_count
        
        if category_count > 10 or has_examples or has_exceptions or avg_conditions > 3:
            return "complex"
        else:
            return "simple"
```

### 4. Processing Script Pattern

```python
# scripts/bedrock_prompt_template_generation.py
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


class PromptTemplateGenerator:
    """
    Generates structured prompt templates for classification tasks using
    the 5-component architecture pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categories = self._load_categories()
        self.template_env = Environment(loader=BaseLoader())
        
    def _load_categories(self) -> List[Dict[str, Any]]:
        """Load and validate category definitions."""
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
        system_config = json.loads(self.config.get('system_prompt_config', '{}'))
        
        role_definition = system_config.get('role_definition', 'expert analyst')
        expertise_areas = system_config.get('expertise_areas', ['data analysis', 'classification'])
        responsibilities = system_config.get('responsibilities', ['analyze data', 'classify content'])
        behavioral_guidelines = system_config.get('behavioral_guidelines', ['be precise', 'be objective'])
        
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
            if category.get('examples') and self.config.get('include_examples', True):
                category_parts.append("    - Examples:")
                for example in category['examples']:
                    category_parts.append(f"        * {example}")
            
            section_parts.append('\n'.join(category_parts))
        
        return '\n'.join(section_parts)
    
    def _generate_input_placeholders_section(self) -> str:
        """Generate input placeholders section."""
        placeholders = json.loads(self.config.get('input_placeholders', '["input_data"]'))
        additional_fields = json.loads(self.config.get('additional_context_fields', '[]'))
        
        section_parts = ["Analysis Instructions:", ""]
        section_parts.append("Please analyze:")
        
        for placeholder in placeholders:
            section_parts.append(f"{placeholder.title()}: {{{placeholder}}}")
        
        for field in additional_fields:
            section_parts.append(f"{field.title()}: {{{field}}}")
        
        return '\n'.join(section_parts)
    
    def _generate_instructions_section(self) -> str:
        """Generate instructions and rules section."""
        instruction_config = json.loads(self.config.get('instruction_config', '{}'))
        
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
                "5. Provide confidence assessment and reasoning",
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
        
        return '\n'.join(instructions)
    
    def _generate_output_format_section(self) -> str:
        """Generate output format schema section."""
        output_config = json.loads(self.config.get('output_format_config', '{}'))
        format_type = output_config.get('format_type', 'structured_json')
        required_fields = output_config.get('required_fields', ['category', 'confidence', 'reasoning'])
        field_descriptions = output_config.get('field_descriptions', {})
        
        if format_type == 'structured_json':
            format_parts = [
                "## Required Output Format",
                "",
                "**CRITICAL: You must respond with a valid JSON object that follows this exact structure:**",
                "",
                "```json",
                "{"
            ]
            
            for i, field in enumerate(required_fields):
                description = field_descriptions.get(field, f"The {field} value")
                comma = "," if i < len(required_fields) - 1 else ""
                format_parts.append(f'    "{field}": "{description}"{comma}')
            
            format_parts.extend([
                "}",
                "```",
                "",
                "Field Descriptions:"
            ])
            
            for field in required_fields:
                description = field_descriptions.get(field, f"The {field} value")
                format_parts.append(f"- **{field}**: {description}")
            
            format_parts.extend([
                "",
                "Do not include any text before or after the JSON object. Only return valid JSON."
            ])
            
        else:
            # Formatted text or hybrid format
            format_parts = [
                "## Required Output Format",
                "",
                "Provide your response in the following structured format:"
            ]
            
            for field in required_fields:
                description = field_descriptions.get(field, f"The {field} value")
                format_parts.append(f"**{field.title()}**: {description}")
        
        return '\n'.join(format_parts)
    
    def _generate_template_metadata(self) -> Dict[str, Any]:
        """Generate metadata about the template."""
        return {
            'template_version': self.config.get('template_version', '1.0'),
            'generation_timestamp': datetime.now().isoformat(),
            'task_type': self.config.get('task_type', 'classification'),
            'template_style': self.config.get('template_style', 'structured'),
            'category_count': len(self.categories),
            'category_names': [cat['name'] for cat in self.categories],
            'output_format': self.config.get('output_format_type', 'structured_json'),
            'validation_level': self.config.get('validation_level', 'comprehensive'),
            'includes_examples': self.config.get('include_examples', True),
            'generator_config': {
                'system_prompt_config': json.loads(self.config.get('system_prompt_config', '{}')),
                'output_format_config': json.loads(self.config.get('output_format_config', '{}'))
            }
        }


class TemplateValidator:
    """Validates generated prompt templates for quality and completeness."""
    
    def __init__(self, validation_level: str = "comprehensive"):
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


def load_category_definitions(categories_path: str, log: Callable[[str], None]) -> List[Dict[str, Any]]:
    """Load category definitions from input files."""
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
        
        # Build configuration from environment variables and loaded data
        config = {
            'task_type': environ_vars.get('TEMPLATE_TASK_TYPE', 'classification'),
            'template_style': environ_vars.get('TEMPLATE_STYLE', 'structured'),
            'validation_level': environ_vars.get('VALIDATION_LEVEL', 'comprehensive'),
            'category_definitions': json.dumps(categories),
            'system_prompt_config': environ_vars.get('SYSTEM_PROMPT_CONFIG', '{}'),
            'output_format_config': environ_vars.get('OUTPUT_FORMAT_CONFIG', '{}'),
            'instruction_config': environ_vars.get('INSTRUCTION_CONFIG', '{}'),
            'input_placeholders': environ_vars.get('INPUT_PLACEHOLDERS', '["input_data"]'),
            'additional_context_fields': environ_vars.get('ADDITIONAL_CONTEXT_FIELDS', '[]'),
            'output_format_type': environ_vars.get('OUTPUT_FORMAT_TYPE', 'structured_json'),
            'required_output_fields': environ_vars.get('REQUIRED_OUTPUT_FIELDS', '["category", "confidence", "reasoning"]'),
            'include_examples': environ_vars.get('INCLUDE_EXAMPLES', 'true').lower() == 'true',
            'generate_validation_schema': environ_vars.get('GENERATE_VALIDATION_SCHEMA', 'true').lower() == 'true',
            'template_version': environ_vars.get('TEMPLATE_VERSION', '1.0')
        }
        
        # Initialize template generator
        generator = PromptTemplateGenerator(config)
        
        # Generate template
        log("Generating prompt template...")
        template = generator.generate_template()
        
        # Validate template
        validator = TemplateValidator(config['validation_level'])
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
            'user_prompt_template': template['user_prompt_template']
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
                'task_type': config['task_type'],
                'template_style': config['template_style'],
                'validation_level': config['validation_level'],
                'category_count': len(categories)
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_output, f, indent=2, ensure_ascii=False, default=str)
        
        log(f"Saved template metadata to: {metadata_file}")
        
        # Generate and save validation schema if requested
        if config['generate_validation_schema']:
            schema_file = schema_path / f"validation_schema_{timestamp}.json"
            
            # Generate JSON schema for output validation
            required_fields = json.loads(config['required_output_fields'])
            validation_schema = {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": required_fields,
                "additionalProperties": False
            }
            
            # Add field definitions
            field_descriptions = json.loads(config.get('output_format_config', '{}')).get('field_descriptions', {})
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
            
            with open(schema_file, 'w', encoding='utf-8') as f:
                json.dump(validation_schema, f, indent=2, ensure_ascii=False)
            
            log(f"Saved validation schema to: {schema_file}")
        
        # Prepare results summary
        results = {
            'success': True,
            'template_generated': True,
            'validation_passed': validation_results['is_valid'],
            'quality_score': validation_results['quality_score'],
            'category_count': len(categories),
            'template_version': config['template_version'],
            'output_files': {
                'prompts': str(prompts_file),
                'metadata': str(metadata_file),
                'schema': str(schema_file) if config['generate_validation_schema'] else None
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
        parser.add_argument("--task-type", default="classification", help="Type of classification task")
        parser.add_argument("--template-style", default="structured", help="Style of template generation")
        parser.add_argument("--validation-level", default="comprehensive", help="Level of template validation")
        parser.add_argument("--output-format", default="structured_json", help="Output format type")
        parser.add_argument("--include-examples", action="store_true", help="Include examples in template")
        parser.add_argument("--generate-validation-schema", action="store_true", help="Generate validation schema")
        parser.add_argument("--template-version", default="1.0", help="Template version identifier")
        
        args = parser.parse_args()

        # Define standard SageMaker paths
        INPUT_CATEGORIES_DIR = "/opt/ml/processing/input/categories"
        INPUT_REQUIREMENTS_DIR = "/opt/ml/processing/input/requirements"
        INPUT_SCHEMA_DIR = "/opt/ml/processing/input/schema"
        OUTPUT_TEMPLATES_DIR = "/opt/ml/processing/output/templates"
        OUTPUT_METADATA_DIR = "/opt/ml/processing/output/metadata"
        OUTPUT_SCHEMA_DIR = "/opt/ml/processing/output/schema"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Log key parameters
        logger.info(f"Starting prompt template generation with parameters:")
        logger.info(f"  Task Type: {args.task_type}")
        logger.info(f"  Template Style: {args.template_style}")
        logger.info(f"  Validation Level: {args.validation_level}")
        logger.info(f"  Output Format: {args.output_format}")
        logger.info(f"  Include Examples: {args.include_examples}")
        logger.info(f"  Generate Schema: {args.generate_validation_schema}")
        logger.info(f"  Template Version: {args.template_version}")

        # Set up path dictionaries
        input_paths = {
            "category_definitions": INPUT_CATEGORIES_DIR,
            "task_requirements": INPUT_REQUIREMENTS_DIR,
            "output_schema_template": INPUT_SCHEMA_DIR
        }

        output_paths = {
            "prompt_templates": OUTPUT_TEMPLATES_DIR,
            "template_metadata": OUTPUT_METADATA_DIR,
            "validation_schema": OUTPUT_SCHEMA_DIR
        }

        # Environment variables dictionary
        environ_vars = {
            "TEMPLATE_TASK_TYPE": os.environ.get("TEMPLATE_TASK_TYPE", args.task_type),
            "TEMPLATE_STYLE": os.environ.get("TEMPLATE_STYLE", args.template_style),
            "VALIDATION_LEVEL": os.environ.get("VALIDATION_LEVEL", args.validation_level),
            "CATEGORY_DEFINITIONS": os.environ.get("CATEGORY_DEFINITIONS", "[]"),
            "SYSTEM_PROMPT_CONFIG": os.environ.get("SYSTEM_PROMPT_CONFIG", "{}"),
            "OUTPUT_FORMAT_CONFIG": os.environ.get("OUTPUT_FORMAT_CONFIG", "{}"),
            "INSTRUCTION_CONFIG": os.environ.get("INSTRUCTION_CONFIG", "{}"),
            "INPUT_PLACEHOLDERS": os.environ.get("INPUT_PLACEHOLDERS", '["input_data"]'),
            "ADDITIONAL_CONTEXT_FIELDS": os.environ.get("ADDITIONAL_CONTEXT_FIELDS", "[]"),
            "OUTPUT_FORMAT_TYPE": os.environ.get("OUTPUT_FORMAT_TYPE", args.output_format),
            "REQUIRED_OUTPUT_FIELDS": os.environ.get("REQUIRED_OUTPUT_FIELDS", '["category", "confidence", "reasoning"]'),
            "INCLUDE_EXAMPLES": os.environ.get("INCLUDE_EXAMPLES", str(args.include_examples).lower()),
            "GENERATE_VALIDATION_SCHEMA": os.environ.get("GENERATE_VALIDATION_SCHEMA", str(args.generate_validation_schema).lower()),
            "TEMPLATE_VERSION": os.environ.get("TEMPLATE_VERSION", args.template_version)
        }

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
```

## Key Differences Between Template Generation Step Types

### 1. By Task Complexity
- **Simple Classification**: Basic category definitions with minimal conditions
- **Complex Categorization**: Detailed category hierarchies with extensive conditions and exceptions
- **Multi-Label Classification**: Support for multiple category assignments
- **Hierarchical Classification**: Nested category structures with parent-child relationships

### 2. By Task Complexity
- **Simple Classification**: Basic category definitions with minimal conditions
- **Complex Categorization**: Detailed category hierarchies with extensive conditions and exceptions
- **Multi-Label Classification**: Support for multiple category assignments
- **Hierarchical Classification**: Nested category structures with parent-child relationships

### 3. By Output Format
- **JSON Response**: Structured JSON output with schema validation (default)
- **Formatted Text**: Human-readable text with specific formatting requirements
- **Hybrid**: Combination of structured data and natural language explanations

## Best Practices Identified

1. **Category-Driven Design**: Structure templates around well-defined category specifications
2. **5-Component Architecture**: Maintain consistent template structure across all generated templates
3. **Configurable Generation**: Support multiple template styles and output formats
4. **Quality Validation**: Implement comprehensive template validation and quality scoring
5. **Integration Ready**: Generate templates compatible with existing Bedrock processing steps
6. **Version Control**: Include versioning and metadata for template management
7. **Schema Generation**: Automatically generate validation schemas for output formats
8. **Extensible Framework**: Support custom template components and validation rules
9. **Error Handling**: Graceful handling of invalid category definitions and generation failures
10. **Documentation**: Comprehensive metadata and documentation for generated templates

## Testing Implications

Bedrock prompt template generation step builders should be tested for:

1. **Category Processing**: Correct parsing and validation of category definitions
2. **Template Structure**: Proper generation of all 5 template components
3. **Output Format Generation**: Correct schema and format specification generation
4. **Validation Logic**: Comprehensive template quality validation
5. **Configuration Handling**: Proper processing of all configuration parameters
6. **File I/O Operations**: Correct reading of input files and writing of output files
7. **Error Recovery**: Proper error handling for invalid inputs and generation failures
8. **Integration Compatibility**: Templates work correctly with Bedrock processing steps
9. **Version Management**: Proper versioning and metadata generation
10. **Quality Scoring**: Accurate quality assessment and recommendation generation

### Recommended Test Categories

#### Template Generation Tests
- Category definition parsing and validation
- System prompt generation with different configurations
- User prompt template generation with all components
- Output format schema generation
- Template metadata creation

#### Validation Tests
- Template quality scoring accuracy
- Validation rule enforcement
- Recommendation generation
- Schema validation correctness

#### Integration Tests
- End-to-end template generation workflow
- Compatibility with Bedrock processing steps
- File format and structure validation
- Configuration parameter processing

#### Error Handling Tests
- Invalid category definition handling
- Missing required configuration handling
- File I/O error recovery
- Template validation failure handling

## Implementation Examples

### Complete Template Generation Step Builder

```python
from typing import Dict, Optional, Any, List
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_bedrock_prompt_template_generation_step import BedrockPromptTemplateGenerationStepConfig
from ...core.base.builder_base import StepBuilderBase

# Import template generation specification
try:
    from ..specs.bedrock_prompt_template_generation_spec import BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC = None
    SPEC_AVAILABLE = False


class BedrockPromptTemplateGenerationStepBuilder(StepBuilderBase):
    """Builder for Bedrock Prompt Template Generation Step."""
    
    def __init__(self, config: BedrockPromptTemplateGenerationStepConfig, sagemaker_session=None, 
                 role: Optional[str] = None, registry_manager=None, 
                 dependency_resolver=None):
        if not isinstance(config, BedrockPromptTemplateGenerationStepConfig):
            raise ValueError("BedrockPromptTemplateGenerationStepBuilder requires BedrockPromptTemplateGenerationStepConfig")
            
        if not SPEC_AVAILABLE or BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC is None:
            raise ValueError("Bedrock prompt template generation specification not available")
            
        super().__init__(
            config=config,
            spec=BEDROCK_PROMPT_TEMPLATE_GENERATION_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: BedrockPromptTemplateGenerationStepConfig = config
    
    def validate_configuration(self) -> None:
        """Validate prompt template generation configuration."""
        # Validate base processing configuration
        required_processing_attrs = [
            'processing_instance_count', 'processing_volume_size',
            'processing_instance_type_large', 'processing_instance_type_small',
            'processing_framework_version', 'use_large_processing_instance'
        ]
        
        for attr in required_processing_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"Missing required processing attribute: {attr}")
        
        # Validate template generation specific configuration
        self.config.validate_template_generation_configuration()
        
        self.log_info("BedrockPromptTemplateGenerationStepConfig validation succeeded")
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create Bedrock Prompt Template Generation ProcessingStep."""
        # Extract parameters
        inputs_raw = kwargs.get('inputs', {})
        outputs = kwargs.get('outputs', {})
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Handle inputs from dependencies
        inputs = {}
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
        
        inputs.update(inputs_raw)
        
        # Create components
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()
        
        # Get standardized step name
        step_name = self._get_step_name()
        
        # Create step
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        
        # Attach specification for future reference
        setattr(step, '_spec', self.spec)
        
        self.log_info("Created Bedrock Prompt Template Generation ProcessingStep: %s", step_name)
        return step
```

## Step Registry Integration

### Registry Pattern for Template Generation Steps

Following the cursus framework registry pattern, template generation steps must be registered in `src/cursus/registry/step_names_original.py`:

```python
# Registry entries for Bedrock prompt template generation steps
"BedrockPromptTemplateGeneration": {
    "config_class": "BedrockPromptTemplateGenerationStepConfig",
    "builder_step_name": "BedrockPromptTemplateGenerationStepBuilder", 
    "spec_type": "BedrockPromptTemplateGeneration",
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Bedrock prompt template generation step for classification tasks",
},
"BedrockPromptTemplateGenerationClassification": {
    "config_class": "BedrockPromptTemplateGenerationClassificationConfig",
    "builder_step_name": "BedrockPromptTemplateGenerationClassificationStepBuilder",
    "spec_type": "BedrockPromptTemplateGenerationClassification", 
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Specialized prompt template generation for classification tasks",
},
"BedrockPromptTemplateGenerationCategorization": {
    "config_class": "BedrockPromptTemplateGenerationCategorizationConfig",
    "builder_step_name": "BedrockPromptTemplateGenerationCategorizationStepBuilder",
    "spec_type": "BedrockPromptTemplateGenerationCategorization",
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Specialized prompt template generation for categorization tasks",
},
```

**Key Registry Pattern Notes:**
- **sagemaker_step_type**: Must be `"Processing"` for all template generation steps
- **spec_type**: Use case-specific specification type (e.g., "BedrockPromptTemplateGeneration")
- **config_class**: Use case-specific configuration class name
- **builder_step_name**: Use case-specific builder class name

## Integration with Bedrock Processing Steps

### Seamless Integration Pattern

The prompt template generation step is designed to integrate seamlessly with existing Bedrock processing steps:

```python
# Example pipeline integration
def create_classification_pipeline():
    # Step 1: Generate prompt templates
    template_generation_step = BedrockPromptTemplateGenerationStepBuilder(
        config=template_config
    ).create_step(
        inputs={
            'category_definitions': category_definitions_s3_uri
        },
        outputs={
            'prompt_templates': 's3://bucket/templates/',
            'template_metadata': 's3://bucket/metadata/',
            'validation_schema': 's3://bucket/schema/'
        }
    )
    
    # Step 2: Use generated templates in Bedrock processing
    bedrock_processing_step = BedrockProcessingStepBuilder(
        config=bedrock_config
    ).create_step(
        inputs={
            'input_data': input_data_s3_uri,
            'prompt_config': template_generation_step.properties.ProcessingOutputConfig.Outputs['prompt_templates'].S3Output.S3Uri
        },
        outputs={
            'processed_data': 's3://bucket/results/',
            'analysis_summary': 's3://bucket/summary/'
        },
        dependencies=[template_generation_step]
    )
    
    return [template_generation_step, bedrock_processing_step]
```

### Output Compatibility

The template generation step outputs are fully compatible with Bedrock processing step inputs:

```python
# Template generation output format
{
    "system_prompt": "You are an expert analyst...",
    "user_prompt_template": "Categories and their criteria:\n\n1. Category1...",
    "metadata": {
        "template_version": "1.0",
        "generation_timestamp": "2025-10-31T15:00:00",
        "task_type": "classification",
        "category_count": 5
    }
}

# Bedrock processing step consumption
# The above JSON is saved as prompts.json and consumed directly by the Bedrock step
```

## Simplified User Experience

### The Problem with Overengineering
The original design included multiple configuration options (template-style, validation-level, task-type) that users don't need to distinguish between. This creates unnecessary complexity and decision fatigue.

### Simplified Design Philosophy
**One Input, Optimal Output**: Users provide only category definitions. The system automatically configures everything else optimally.

## Usage Examples

### Example 1: Simple Classification (Simplified)

```python
# Define categories - ONLY thing user needs to provide
categories = [
    CategoryDefinition(
        name="Positive",
        description="Positive sentiment or feedback",
        conditions=["Contains positive language", "Expresses satisfaction"],
        exceptions=["Sarcastic positive statements"],
        key_indicators=["good", "excellent", "satisfied", "happy"]
    ),
    CategoryDefinition(
        name="Negative", 
        description="Negative sentiment or feedback",
        conditions=["Contains negative language", "Expresses dissatisfaction"],
        exceptions=["Constructive criticism"],
        key_indicators=["bad", "terrible", "disappointed", "angry"]
    )
]

# Simplified configuration - everything else is automatic
config = BedrockPromptTemplateGenerationStepConfig()
config.category_definitions = categories
# That's it! No template_style, validation_level, task_type needed

# Create step
builder = BedrockPromptTemplateGenerationStepBuilder(config)
step = builder.create_step(
    inputs={'category_definitions': 's3://bucket/categories.json'},
    outputs={'prompt_templates': 's3://bucket/templates/'}
)

# System automatically:
# - Uses "structured" template style (optimal for automation)
# - Uses "standard" validation level (optimal balance)
# - Uses "classification" task type (optimal for categories)
# - Detects "simple" complexity and configures accordingly
```

### Example 2: Complex Categorization (Auto-Detected)

```python
# Define complex categories - system auto-detects complexity
categories = [
    CategoryDefinition(
        name="TrueDNR",
        description="Delivered Not Received - Package marked as delivered but buyer claims non-receipt",
        conditions=[
            "Package marked as delivered (EVENT_301)",
            "Buyer claims non-receipt",
            "Tracking shows delivery"
        ],
        exceptions=[
            "Buyer received wrong item",
            "Package damaged on delivery"
        ],
        key_indicators=[
            "delivered but not received",
            "tracking shows delivered",
            "missing package investigation"
        ],
        examples=[
            "Package shows delivered but I never got it",
            "Tracking says delivered yesterday but nothing here"
        ],
        priority=1
    ),
    # ... 12 more complex categories with examples and exceptions
]

# Still simple configuration - complexity auto-detected
config = BedrockPromptTemplateGenerationStepConfig()
config.category_definitions = categories
# System automatically detects:
# - 13 categories (>10) = complex
# - Has examples = complex  
# - Has exceptions = complex
# - Configures detailed_conditions=True, exception_handling=True

# Create step - same simple interface
builder = BedrockPromptTemplateGenerationStepBuilder(config)
step = builder.create_step(
    inputs={'category_definitions': 's3://bucket/categories.json'},
    outputs={'prompt_templates': 's3://bucket/templates/'}
)

# System automatically:
# - Uses optimal settings for complex categorization
# - Generates sophisticated prompt template like your example
# - Includes all 5 components with appropriate complexity
# - Validates with standard quality checks
```

### Example 3: Integration with Bedrock Processing (Simplified)

```python
# Step 1: Generate templates (simplified)
template_config = BedrockPromptTemplateGenerationStepConfig()
template_config.category_definitions = your_categories  # Only required input

template_step = BedrockPromptTemplateGenerationStepBuilder(template_config).create_step(
    inputs={'category_definitions': 's3://bucket/categories.json'},
    outputs={'prompt_templates': 's3://bucket/templates/'}
)

# Step 2: Use templates in Bedrock processing (unchanged)
bedrock_step = BedrockProcessingStepBuilder(bedrock_config).create_step(
    inputs={
        'input_data': 's3://bucket/data/',
        'prompt_config': template_step.properties.ProcessingOutputConfig.Outputs['prompt_templates'].S3Output.S3Uri
    },
    outputs={'processed_data': 's3://bucket/results/'},
    dependencies=[template_step]
)
```

## Automatic Intelligence Features

### 1. **Complexity Detection**
```python
def _detect_complexity(self) -> str:
    category_count = len(self.category_definitions)
    has_examples = any(cat.examples for cat in self.category_definitions if cat.examples)
    has_exceptions = any(cat.exceptions for cat in self.category_definitions if cat.exceptions)
    avg_conditions = sum(len(cat.conditions) for cat in self.category_definitions) / category_count
    
    if category_count > 10 or has_examples or has_exceptions or avg_conditions > 3:
        return "complex"  # Auto-enables detailed features
    else:
        return "simple"   # Auto-uses streamlined features
```

### 2. **Smart Defaults**
- **Input Placeholders**: Automatically set to `["dialogue", "shiptrack", "max_estimated_arrival_date"]` for common use cases
- **Output Format**: Always `"structured_json"` for automated processing compatibility
- **Template Style**: Always `"structured"` for clarity and consistency
- **Validation Level**: Always `"standard"` for optimal balance

### 3. **Auto-Configuration**
```python
def _auto_configure_from_categories(self):
    if self._detect_complexity() == "complex":
        self.template_components.category_section_config.detailed_conditions = True
        self.template_components.category_section_config.exception_handling = True
        self.template_components.instruction_config.include_edge_case_handling = True
    else:
        # Streamlined for simple cases
        self.template_components.category_section_config.detailed_conditions = False
        self.template_components.instruction_config.include_edge_case_handling = False
```

## Benefits of Simplified Design

1. **Reduced Cognitive Load**: Users focus on business logic (categories) not technical configuration
2. **Optimal Performance**: System chooses best settings automatically
3. **Fewer Errors**: No invalid configuration combinations
4. **Faster Adoption**: Minimal learning curve
5. **Consistent Quality**: Always uses proven optimal settings
6. **Future-Proof**: Can add intelligence without breaking user code

## Migration from Complex to Simple

### Before (Overengineered):
```python
config.task_type = "categorization"           # User had to choose
config.template_style = "detailed"           # User had to choose  
config.validation_level = "comprehensive"    # User had to choose
config.category_definitions = categories     # User provides
config.output_schema_config.format_type = "structured_json"  # User had to choose
```

### After (Simplified):
```python
config.category_definitions = categories     # User provides - ONLY required input
# Everything else is automatically optimal
```

The simplified design eliminates decision fatigue while ensuring optimal results through intelligent automation.

## Summary

This comprehensive design document establishes the patterns and architecture for Bedrock prompt template generation steps in the cursus framework. The key innovations include:

1. **5-Component Architecture**: Structured template generation following system prompt, category definitions, input placeholders, instructions, and output format components.

2. **Category-Driven Design**: Templates built around well-defined category specifications with conditions, exceptions, and key indicators.

3. **Quality Validation**: Comprehensive template validation with quality scoring and recommendations.

4. **Seamless Integration**: Direct compatibility with existing Bedrock processing steps through standardized output formats.

5. **Configurable Generation**: Support for multiple template styles, output formats, and validation levels.

6. **Extensible Framework**: Modular design supporting custom template components and validation rules.

The design enables automated generation of high-quality, structured prompt templates specifically optimized for categorization and classification tasks, while maintaining full integration with the existing cursus framework and Bedrock processing capabilities.
