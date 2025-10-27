---
tags:
  - design
  - step_builders
  - bedrock_steps
  - patterns
  - sagemaker
  - llm_processing
  - aws_bedrock
keywords:
  - bedrock processing step patterns
  - AWS Bedrock integration
  - LLM processing
  - inference profile management
  - programmable response models
  - configurable model lists
topics:
  - step builder patterns
  - bedrock processing implementation
  - SageMaker LLM processing
  - AWS Bedrock step architecture
language: python
date of note: 2025-10-26
---

# Bedrock Processing Step Builder Patterns

## Overview

This document defines the design patterns for Bedrock processing step builder implementations in the cursus framework. Bedrock processing steps create **ProcessingStep** instances that invoke AWS Bedrock models for Large Language Model (LLM) processing tasks. These steps provide configurable model management, programmable response models, and intelligent fallback strategies for production LLM workflows.

## SageMaker Step Type Classification

Bedrock processing steps create **ProcessingStep** instances using **SKLearnProcessor** with specialized Bedrock integration:
- **SKLearnProcessor**: Standard processing framework with Bedrock client integration
- **Bedrock Runtime**: AWS Bedrock service integration for LLM inference
- **Model Management**: Intelligent switching between inference profiles and on-demand models
- **Response Processing**: Structured response parsing with Pydantic model validation

## Key Differences from Standard Processing Steps

### 1. Model Management Pattern
```python
# Standard Processing Step: Fixed processor configuration
processor = SKLearnProcessor(
    framework_version="1.2-1",
    instance_type=config.processing_instance_type
)

# Bedrock Processing Step: Dynamic model selection with fallback
class BedrockModelStrategy:
    def __init__(self, config):
        self.primary_model = config.primary_model_id
        self.inference_profile_arn = config.inference_profile_arn
        self.fallback_model = config.fallback_model_id
        self.strategy = self._determine_model_strategy()
    
    def _determine_model_strategy(self):
        if self.primary_model in config.inference_profile_required_models:
            return {
                'use_inference_profile': True,
                'effective_model': self.inference_profile_arn or self._get_global_profile(),
                'fallback_model': self.fallback_model
            }
        return {
            'use_inference_profile': False,
            'effective_model': self.primary_model,
            'fallback_model': None
        }
```

### 2. Configurable Model Lists Pattern
```python
# User-configurable model compatibility lists
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    # Configurable Model Lists - User can override defaults
    inference_profile_required_models: List[str] = Field(
        default_factory=lambda: [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-4-haiku-20250101-v1:0",
            "anthropic.claude-4-sonnet-20250101-v1:0",
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        ],
        description="Models that require inference profiles (user-configurable)"
    )
    
    on_demand_compatible_models: List[str] = Field(
        default_factory=lambda: [
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0"
        ],
        description="Models compatible with on-demand throughput (user-configurable)"
    )
```

### 3. Programmable Response Models Pattern
```python
# Standard Processing: Fixed output format
outputs = [
    ProcessingOutput(
        output_name="processed_data",
        source="/opt/ml/processing/output/data"
    )
]

# Bedrock Processing: Programmable response models
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    response_model_class: Optional[str] = Field(
        default=None,
        description="Fully qualified class name for response model (e.g., 'mymodule.MyResponseModel')"
    )
    
    response_format: str = Field(
        default="json",
        description="Expected response format: 'json', 'text', or 'structured'"
    )

# Example usage with custom response model
from pydantic import BaseModel

class CustomAnalysisResponse(BaseModel):
    category: str
    confidence_score: float
    reasoning: List[str]
    evidence: Dict[str, List[str]]

# Configuration
config.response_model_class = "myproject.models.CustomAnalysisResponse"
config.response_format = "structured"
```

### 4. Prompt Template System Pattern
```python
# Standard Processing: Fixed script logic
job_arguments = ["--job_type", config.job_type]

# Bedrock Processing: Configurable prompt templates
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for the model"
    )
    
    user_prompt_template: str = Field(
        default="Analyze the following data: {input_data}",
        description="User prompt template with placeholders"
    )
    
    additional_input_columns: List[str] = Field(
        default_factory=list,
        description="Additional columns to include in prompt template"
    )

# Template usage in processing script
def format_prompt(input_data: str, additional_data: Dict[str, Any]) -> str:
    template_vars = {'input_data': input_data}
    template_vars.update(additional_data)
    return config.user_prompt_template.format(**template_vars)
```

## Common Implementation Patterns

### 1. Base Architecture Pattern

All Bedrock processing step builders follow this architecture:

```python
@register_builder()
class BedrockProcessingStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load Bedrock processing specification
        spec = BEDROCK_PROCESSING_SPEC
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate Bedrock-specific configuration
        
    def _create_processor(self) -> SKLearnProcessor:
        # Create SKLearnProcessor with Bedrock environment
        
    def _get_environment_variables(self) -> Dict[str, str]:
        # Build Bedrock-specific environment variables
        
    def _get_model_strategy_config(self) -> Dict[str, Any]:
        # Determine model usage strategy
        
    def _get_inputs(self, inputs) -> List[ProcessingInput]:
        # Create ProcessingInput objects using specification
        
    def _get_outputs(self, outputs) -> List[ProcessingOutput]:
        # Create ProcessingOutput objects for Bedrock results
        
    def _get_job_arguments(self) -> List[str]:
        # Build command-line arguments for Bedrock processing
        
    def create_step(self, **kwargs) -> ProcessingStep:
        # Orchestrate Bedrock processing step creation
```

### 2. Model Strategy Determination Pattern

```python
def _get_model_strategy_config(self) -> Dict[str, Any]:
    """Determine model usage strategy based on configuration."""
    model_id = self.config.primary_model_id
    
    strategy = {
        'primary_model': model_id,
        'use_inference_profile': False,
        'fallback_model': self.config.fallback_model_id,
        'inference_profile_arn': self.config.inference_profile_arn
    }
    
    # Check if model requires inference profile
    if model_id in self.config.inference_profile_required_models:
        strategy['use_inference_profile'] = True
        
        if self.config.inference_profile_arn:
            strategy['effective_model'] = self.config.inference_profile_arn
        else:
            # Try to use global profile ID if available
            if model_id.startswith('anthropic.claude-4') or 'claude-sonnet-4' in model_id:
                global_profile = model_id.replace('anthropic.', 'global.anthropic.')
                strategy['effective_model'] = global_profile
            else:
                strategy['effective_model'] = model_id
    else:
        strategy['effective_model'] = model_id
        
    return strategy
```

### 3. Environment Variables Pattern for Bedrock

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Build Bedrock-specific environment variables."""
    env_vars = super()._get_environment_variables()
    
    # Model configuration
    env_vars["BEDROCK_PRIMARY_MODEL_ID"] = self.config.primary_model_id
    env_vars["BEDROCK_FALLBACK_MODEL_ID"] = self.config.fallback_model_id
    
    # Inference profile configuration
    if self.config.inference_profile_arn:
        env_vars["BEDROCK_INFERENCE_PROFILE_ARN"] = self.config.inference_profile_arn
    
    # Model lists as JSON
    env_vars["BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS"] = json.dumps(
        self.config.inference_profile_required_models
    )
    env_vars["BEDROCK_ON_DEMAND_COMPATIBLE_MODELS"] = json.dumps(
        self.config.on_demand_compatible_models
    )
    
    # Prompt configuration
    if self.config.system_prompt:
        env_vars["BEDROCK_SYSTEM_PROMPT"] = self.config.system_prompt
    env_vars["BEDROCK_USER_PROMPT_TEMPLATE"] = self.config.user_prompt_template
    
    # Response configuration
    env_vars["BEDROCK_RESPONSE_FORMAT"] = self.config.response_format
    if self.config.response_model_class:
        env_vars["BEDROCK_RESPONSE_MODEL_CLASS"] = self.config.response_model_class
    
    # API configuration
    env_vars["BEDROCK_MAX_TOKENS"] = str(self.config.max_tokens)
    env_vars["BEDROCK_TEMPERATURE"] = str(self.config.temperature)
    env_vars["BEDROCK_TOP_P"] = str(self.config.top_p)
    env_vars["BEDROCK_MAX_RETRIES"] = str(self.config.max_retries)
    
    # Processing configuration
    env_vars["BEDROCK_BATCH_SIZE"] = str(self.config.batch_size)
    env_vars["BEDROCK_INPUT_DATA_COLUMN"] = self.config.input_data_column
    env_vars["BEDROCK_OUTPUT_COLUMN_PREFIX"] = self.config.output_column_prefix
    
    # Additional input columns as JSON
    if self.config.additional_input_columns:
        env_vars["BEDROCK_ADDITIONAL_INPUT_COLUMNS"] = json.dumps(
            self.config.additional_input_columns
        )
    
    return env_vars
```

### 4. Job Arguments Pattern for Bedrock

```python
def _get_job_arguments(self) -> List[str]:
    """Build command-line arguments for Bedrock processing script."""
    args = [
        "--primary-model-id", self.config.primary_model_id,
        "--batch-size", str(self.config.batch_size),
        "--max-retries", str(self.config.max_retries),
        "--input-column", self.config.input_data_column,
        "--output-prefix", self.config.output_column_prefix,
        "--response-format", self.config.response_format
    ]
    
    # Add optional arguments
    if self.config.system_prompt:
        args.extend(["--system-prompt", self.config.system_prompt])
        
    if self.config.response_model_class:
        args.extend(["--response-model-class", self.config.response_model_class])
        
    if self.config.inference_profile_arn:
        args.extend(["--inference-profile-arn", self.config.inference_profile_arn])
    
    return args
```

### 5. Specification-Driven Input/Output Pattern for Bedrock

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """Create ProcessingInput objects for Bedrock processing."""
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
    """Create ProcessingOutput objects for Bedrock results."""
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

### 6. Processor Creation Pattern for Bedrock

```python
def _create_processor(self) -> SKLearnProcessor:
    """Create SKLearnProcessor with Bedrock-specific configuration."""
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

### 7. Step Creation Pattern for Bedrock

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """Create Bedrock ProcessingStep."""
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

### Standard Bedrock Configuration Validation
```python
def validate_configuration(self) -> None:
    """Validate Bedrock processing configuration."""
    # Validate base processing configuration
    required_processing_attrs = [
        'processing_instance_count', 'processing_volume_size',
        'processing_instance_type_large', 'processing_instance_type_small',
        'processing_framework_version', 'use_large_processing_instance'
    ]
    
    for attr in required_processing_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required processing attribute: {attr}")
    
    # Validate Bedrock-specific configuration
    required_bedrock_attrs = [
        'primary_model_id', 'fallback_model_id', 'max_tokens', 
        'temperature', 'top_p', 'batch_size', 'max_retries'
    ]
    
    for attr in required_bedrock_attrs:
        if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
            raise ValueError(f"Missing required Bedrock attribute: {attr}")
    
    # Validate model lists
    if not self.config.inference_profile_required_models:
        raise ValueError("inference_profile_required_models cannot be empty")
    
    if not self.config.on_demand_compatible_models:
        raise ValueError("on_demand_compatible_models cannot be empty")
    
    # Validate response format
    valid_formats = ['json', 'text', 'structured']
    if self.config.response_format not in valid_formats:
        raise ValueError(f"Invalid response_format: {self.config.response_format}")
    
    # Validate response model class if provided
    if self.config.response_model_class:
        try:
            self._validate_response_model_class(self.config.response_model_class)
        except Exception as e:
            raise ValueError(f"Invalid response_model_class: {e}")
```

### Response Model Validation Pattern
```python
def _validate_response_model_class(self, class_path: str) -> None:
    """Validate that response model class exists and is a Pydantic model."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # Check if it's a Pydantic model
        from pydantic import BaseModel
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Response model class must inherit from pydantic.BaseModel")
            
    except ImportError as e:
        raise ValueError(f"Cannot import response model module: {e}")
    except AttributeError as e:
        raise ValueError(f"Response model class not found: {e}")
```

## Design Components Integration

### 1. Step Specification Pattern

```python
# specs/bedrock_processing_spec.py
BEDROCK_PROCESSING_SPEC = StepSpecification(
    step_type=get_spec_step_type("BedrockProcessing"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_bedrock_processing_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["ProcessingStep", "DataLoad", "TabularPreprocessing"],
            semantic_keywords=["data", "input", "text", "dataset", "processed", "analyze"],
            data_type="S3Uri",
            description="Input data for Bedrock LLM processing"
        ),
        DependencySpec(
            logical_name="prompt_config",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["ProcessingStep", "PromptPrep"],
            semantic_keywords=["config", "prompt", "template", "system", "instructions"],
            data_type="S3Uri",
            description="Prompt configuration and templates (optional)"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Data with Bedrock LLM analysis results",
            aliases=["bedrock_output", "llm_results", "analyzed_data"]
        ),
        OutputSpec(
            logical_name="analysis_summary",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['analysis_summary'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Summary of Bedrock processing results and statistics",
            aliases=["summary", "stats", "processing_report"]
        )
    ]
)
```

### 2. Script Contract Pattern

```python
# contracts/bedrock_processing_contract.py
BEDROCK_PROCESSING_CONTRACT = ProcessingScriptContract(
    entry_point="bedrock_processing.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "prompt_config": "/opt/ml/processing/input/config"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/data",
        "analysis_summary": "/opt/ml/processing/output/summary"
    },
    expected_arguments={
        "--primary-model-id": "Primary Bedrock model ID to use",
        "--batch-size": "Batch size for processing",
        "--max-retries": "Maximum retries for API calls",
        "--input-column": "Column name containing input data",
        "--output-prefix": "Prefix for output columns",
        "--response-format": "Expected response format (json/text/structured)"
    },
    required_env_vars=[
        "BEDROCK_PRIMARY_MODEL_ID",
        "BEDROCK_FALLBACK_MODEL_ID",
        "BEDROCK_USER_PROMPT_TEMPLATE",
        "BEDROCK_RESPONSE_FORMAT"
    ],
    optional_env_vars={
        "BEDROCK_INFERENCE_PROFILE_ARN": "Inference profile ARN for provisioned throughput",
        "BEDROCK_SYSTEM_PROMPT": "System prompt for the model",
        "BEDROCK_RESPONSE_MODEL_CLASS": "Pydantic model class for response validation",
        "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": "JSON list of models requiring inference profiles",
        "BEDROCK_ON_DEMAND_COMPATIBLE_MODELS": "JSON list of on-demand compatible models",
        "BEDROCK_ADDITIONAL_INPUT_COLUMNS": "JSON list of additional input columns"
    },
    framework_requirements={
        "boto3": ">=1.26.0",
        "pydantic": ">=2.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "tenacity": ">=8.0.0",
        "numpy": ">=1.19.0"
    },
    description="""
    Bedrock processing script that:
    1. Loads input data from CSV/Parquet files
    2. Configures AWS Bedrock client with model strategy (inference profile vs on-demand)
    3. Processes data in batches through Bedrock LLM models
    4. Handles intelligent fallback between inference profiles and on-demand models
    5. Parses and validates responses using configurable Pydantic models
    6. Saves processed results and analysis summary
    
    Model Management Features:
    - Automatic detection of models requiring inference profiles
    - Intelligent fallback to on-demand models when inference profiles fail
    - User-configurable model compatibility lists
    - Support for both ARN-based and global profile ID-based inference profiles
    
    Prompt System Features:
    - Configurable system and user prompts with template variables
    - Support for additional input columns in prompt templates
    - Dynamic prompt formatting based on input data structure
    
    Response Processing Features:
    - Configurable response formats (JSON, text, structured)
    - Pydantic model validation for structured responses
    - Automatic error handling and retry logic with exponential backoff
    - Comprehensive logging and monitoring
    
    Input Structure:
    - /opt/ml/processing/input/data: CSV/Parquet files with text data
    - /opt/ml/processing/input/config: Optional prompt configuration files
    
    Output Structure:
    - /opt/ml/processing/output/data: Processed data with LLM results
    - /opt/ml/processing/output/summary: Processing statistics and summary
    """
)
```

### 3. Configuration Class Pattern

```python
# configs/config_bedrock_processing_step.py
class BedrockProcessingStepConfig(ProcessingStepConfigBase):
    """Configuration for Bedrock processing step with configurable model management."""
    
    def __init__(self):
        super().__init__()
        
        # Model Configuration
        self.primary_model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
        self.inference_profile_arn: Optional[str] = None
        self.fallback_model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
        
        # Configurable Model Lists - User can override these
        self.inference_profile_required_models: List[str] = [
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "anthropic.claude-4-haiku-20250101-v1:0",
            "anthropic.claude-4-sonnet-20250101-v1:0",
            "anthropic.claude-4-opus-20250101-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-opus-4-20250514-v1:0",
            "us.anthropic.claude-opus-4-20250514-v1:0",
            "global.anthropic.claude-sonnet-4-20250514-v1:0"
        ]
        
        self.on_demand_compatible_models: List[str] = [
            "anthropic.claude-v2",
            "anthropic.claude-v2:1",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0", 
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-instant-v1"
        ]
        
        # Prompt Configuration
        self.system_prompt: Optional[str] = None
        self.user_prompt_template: str = "Analyze the following data: {input_data}"
        
        # Response Model Configuration
        self.response_model_class: Optional[str] = None
        self.response_format: str = "json"  # "json", "text", "structured"
        
        # Bedrock API Configuration
        self.max_tokens: int = 4000
        self.temperature: float = 0.1
        self.top_p: float = 0.9
        
        # Processing Configuration
        self.batch_size: int = 10
        self.max_retries: int = 3
        
        # Input/Output Configuration
        self.input_data_column: str = "input_text"
        self.additional_input_columns: List[str] = []
        self.output_column_prefix: str = "bedrock_"
    
    def validate_bedrock_configuration(self) -> None:
        """Validate Bedrock-specific configuration."""
        # Validate model configuration
        if not self.primary_model_id:
            raise ValueError("primary_model_id is required")
        
        if not self.fallback_model_id:
            raise ValueError("fallback_model_id is required")
        
        # Validate model lists
        if not self.inference_profile_required_models:
            raise ValueError("inference_profile_required_models cannot be empty")
        
        if not self.on_demand_compatible_models:
            raise ValueError("on_demand_compatible_models cannot be empty")
        
        # Validate API parameters
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        # Validate processing parameters
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        
        # Validate response format
        valid_formats = ['json', 'text', 'structured']
        if self.response_format not in valid_formats:
            raise ValueError(f"response_format must be one of: {valid_formats}")
        
        # Validate response model class if provided
        if self.response_model_class and self.response_format != 'structured':
            raise ValueError("response_model_class requires response_format='structured'")
    
    def get_model_strategy(self) -> Dict[str, Any]:
        """Get model usage strategy based on configuration."""
        strategy = {
            'primary_model': self.primary_model_id,
            'use_inference_profile': False,
            'fallback_model': self.fallback_model_id,
            'inference_profile_arn': self.inference_profile_arn
        }
        
        # Check if model requires inference profile
        if self.primary_model_id in self.inference_profile_required_models:
            strategy['use_inference_profile'] = True
            
            if self.inference_profile_arn:
                strategy['effective_model'] = self.inference_profile_arn
            else:
                # Try to use global profile ID if available
                if 'claude-4' in self.primary_model_id or 'claude-sonnet-4' in self.primary_model_id:
                    global_profile = self.primary_model_id.replace('anthropic.', 'global.anthropic.')
                    strategy['effective_model'] = global_profile
                else:
                    strategy['effective_model'] = self.primary_model_id
        else:
            strategy['effective_model'] = self.primary_model_id
            
        return strategy
```

### 4. Processing Script Pattern

```python
# scripts/bedrock_processing.py
"""
Bedrock processing script with configurable model management and programmable response models.
"""

import os
import json
import argparse
import pandas as pd
import boto3
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockProcessor:
    """Bedrock processing with intelligent model management."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bedrock_client = None
        self.response_model_class = None
        self._initialize_bedrock_client()
        self._load_response_model()
    
    def _initialize_bedrock_client(self):
        """Initialize Bedrock client."""
        self.bedrock_client = boto3.client('bedrock-runtime')
        logger.info("Initialized Bedrock client")
    
    def _load_response_model(self):
        """Load response model class if specified."""
        if self.config.get('response_model_class'):
            try:
                module_path, class_name = self.config['response_model_class'].rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                self.response_model_class = getattr(module, class_name)
                logger.info(f"Loaded response model: {self.config['response_model_class']}")
            except Exception as e:
                logger.warning(f"Failed to load response model: {e}")
    
    def _determine_model_strategy(self) -> Dict[str, Any]:
        """Determine model usage strategy."""
        model_id = self.config['primary_model_id']
        inference_profile_required = json.loads(
            self.config.get('inference_profile_required_models', '[]')
        )
        
        strategy = {
            'primary_model': model_id,
            'use_inference_profile': False,
            'fallback_model': self.config['fallback_model_id'],
            'inference_profile_arn': self.config.get('inference_profile_arn')
        }
        
        if model_id in inference_profile_required:
            strategy['use_inference_profile'] = True
            
            if strategy['inference_profile_arn']:
                strategy['effective_model'] = strategy['inference_profile_arn']
            else:
                # Try global profile ID
                if 'claude-4' in model_id or 'claude-sonnet-4' in model_id:
                    global_profile = model_id.replace('anthropic.', 'global.anthropic.')
                    strategy['effective_model'] = global_profile
                else:
                    strategy['effective_model'] = model_id
        else:
            strategy['effective_model'] = model_id
            
        return strategy
    
    def _format_prompt(self, input_data: str, additional_data: Dict[str, Any] = None) -> str:
        """Format prompt with input data and additional columns."""
        template_vars = {'input_data': input_data}
        
        if additional_data:
            template_vars.update(additional_data)
            
        return self.config['user_prompt_template'].format(**template_vars)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _invoke_bedrock(self, prompt: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke Bedrock with retry logic and fallback."""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": int(self.config['max_tokens']),
            "temperature": float(self.config['temperature']),
            "top_p": float(self.config['top_p']),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.config.get('system_prompt'):
            request_body["system"] = self.config['system_prompt']
        
        # Try primary model/profile
        try:
            response = self.bedrock_client.invoke_model(
                modelId=strategy['effective_model'],
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json"
            )
            return json.loads(response['body'].read())
            
        except ClientError as e:
            if strategy['use_inference_profile'] and 'ValidationException' in str(e):
                # Fallback to on-demand model
                logger.warning(f"Inference profile failed, falling back to: {strategy['fallback_model']}")
                response = self.bedrock_client.invoke_model(
                    modelId=strategy['fallback_model'],
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json"
                )
                return json.loads(response['body'].read())
            else:
                raise
    
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Bedrock response based on configuration."""
        if 'content' in response and len(response['content']) > 0:
            response_text = response['content'][0].get('text', '')
        else:
            raise ValueError("No content in Bedrock response")
        
        if self.config['response_format'] == "json":
            try:
                parsed_json = json.loads(response_text)
                if self.response_model_class:
                    # Validate with Pydantic model
                    validated_response = self.response_model_class(**parsed_json)
                    return validated_response.model_dump()
                return parsed_json
            except json.JSONDecodeError:
                return {"raw_response": response_text, "parse_error": "Invalid JSON"}
                
        elif self.config['response_format'] == "structured" and self.response_model_class:
            # Try to extract structured data from text
            return self._extract_structured_response(response_text)
            
        else:
            return {"response": response_text}
    
    def _extract_structured_response(self, response_text: str) -> Dict[str, Any]:
        """Extract structured response from text using response model."""
        # Implementation depends on specific response model requirements
        # This is a placeholder for custom extraction logic
        try:
            # Try JSON extraction first
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_json = json.loads(json_str)
                validated_response = self.response_model_class(**parsed_json)
                return validated_response.model_dump()
        except Exception as e:
            logger.warning(f"Failed to extract structured response: {e}")
        
        return {"raw_response": response_text, "extraction_error": "Failed to extract structured data"}
    
    def process_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of data through Bedrock."""
        strategy = self._determine_model_strategy()
        results = []
        
        input_column = self.config['input_data_column']
        additional_columns = json.loads(self.config.get('additional_input_columns', '[]'))
        output_prefix = self.config['output_column_prefix']
        
        for idx, row in input_df.iterrows():
            try:
                # Prepare input data
                input_data = row[input_column]
                additional_data = {col: row[col] for col in additional_columns if col in row}
                
                # Format prompt
                prompt = self._format_prompt(input_data, additional_data)
                
                # Invoke Bedrock
                response = self._invoke_bedrock(prompt, strategy)
                
                # Parse response
                parsed_result = self._parse_response(response)
                
                # Add to results
                result_row = row.to_dict()
                for key, value in parsed_result.items():
                    result_row[f"{output_prefix}{key}"] = value
                    
                result_row[f"{output_prefix}status"] = "success"
                results.append(result_row)
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                result_row = row.to_dict()
                result_row[f"{output_prefix}status"] = "error"
                result_row[f"{output_prefix}error"] = str(e)
                results.append(result_row)
        
        return pd.DataFrame(results)


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Bedrock processing script")
    parser.add_argument("--primary-model-id", required=True, help="Primary Bedrock model ID")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries")
    parser.add_argument("--input-column", default="input_text", help="Input data column name")
    parser.add_argument("--output-prefix", default="bedrock_", help="Output column prefix")
    parser.add_argument("--response-format", default="json", help="Response format")
    parser.add_argument("--system-prompt", help="System prompt")
    parser.add_argument("--response-model-class", help="Response model class")
    parser.add_argument("--inference-profile-arn", help="Inference profile ARN")
    
    args = parser.parse_args()
    
    # Build configuration from environment variables and arguments
    config = {
        'primary_model_id': args.primary_model_id,
        'fallback_model_id': os.environ.get('BEDROCK_FALLBACK_MODEL_ID', ''),
        'inference_profile_arn': args.inference_profile_arn or os.environ.get('BEDROCK_INFERENCE_PROFILE_ARN'),
        'inference_profile_required_models': os.environ.get('BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS', '[]'),
        'on_demand_compatible_models': os.environ.get('BEDROCK_ON_DEMAND_COMPATIBLE_MODELS', '[]'),
        'system_prompt': args.system_prompt or os.environ.get('BEDROCK_SYSTEM_PROMPT'),
        'user_prompt_template': os.environ.get('BEDROCK_USER_PROMPT_TEMPLATE', 'Analyze: {input_data}'),
        'response_format': args.response_format,
        'response_model_class': args.response_model_class or os.environ.get('BEDROCK_RESPONSE_MODEL_CLASS'),
        'max_tokens': int(os.environ.get('BEDROCK_MAX_TOKENS', '4000')),
        'temperature': float(os.environ.get('BEDROCK_TEMPERATURE', '0.1')),
        'top_p': float(os.environ.get('BEDROCK_TOP_P', '0.9')),
        'batch_size': args.batch_size,
        'max_retries': args.max_retries,
        'input_data_column': args.input_column,
        'output_column_prefix': args.output_prefix,
        'additional_input_columns': os.environ.get('BEDROCK_ADDITIONAL_INPUT_COLUMNS', '[]')
    }
    
    try:
        # Initialize processor
        processor = BedrockProcessor(config)
        
        # Load input data
        input_path = Path("/opt/ml/processing/input/data")
        output_path = Path("/opt/ml/processing/output/data")
        summary_path = Path("/opt/ml/processing/output/summary")
        
        # Create output directories
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path.mkdir(parents=True, exist_ok=True)
        
        # Process all CSV/Parquet files in input directory
        input_files = list(input_path.glob("*.csv")) + list(input_path.glob("*.parquet"))
        
        if not input_files:
            raise ValueError("No input files found in /opt/ml/processing/input/data")
        
        all_results = []
        processing_stats = {
            'total_files': len(input_files),
            'total_records': 0,
            'successful_records': 0,
            'failed_records': 0,
            'files_processed': []
        }
        
        for input_file in input_files:
            logger.info(f"Processing file: {input_file}")
            
            # Load data
            if input_file.suffix == '.csv':
                df = pd.read_csv(input_file)
            else:
                df = pd.read_parquet(input_file)
            
            # Process batch
            result_df = processor.process_batch(df)
            
            # Update statistics
            processing_stats['total_records'] += len(df)
            processing_stats['successful_records'] += len(result_df[result_df[f"{config['output_column_prefix']}status"] == "success"])
            processing_stats['failed_records'] += len(result_df[result_df[f"{config['output_column_prefix']}status"] == "error"])
            processing_stats['files_processed'].append({
                'filename': input_file.name,
                'records': len(df),
                'success_rate': len(result_df[result_df[f"{config['output_column_prefix']}status"] == "success"]) / len(df)
            })
            
            # Save results
            output_file = output_path / f"processed_{input_file.name}"
            if input_file.suffix == '.csv':
                result_df.to_csv(output_file, index=False)
            else:
                result_df.to_parquet(output_file, index=False)
            
            all_results.append(result_df)
            logger.info(f"Saved results to: {output_file}")
        
        # Combine all results if multiple files
        if len(all_results) > 1:
            combined_df = pd.concat(all_results, ignore_index=True)
            combined_output = output_path / "combined_results.parquet"
            combined_df.to_parquet(combined_output, index=False)
            logger.info(f"Saved combined results to: {combined_output}")
        
        # Save processing summary
        processing_stats['overall_success_rate'] = (
            processing_stats['successful_records'] / processing_stats['total_records']
            if processing_stats['total_records'] > 0 else 0
        )
        
        summary_file = summary_path / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(processing_stats, f, indent=2)
        
        logger.info("Processing completed successfully")
        logger.info(f"Total records: {processing_stats['total_records']}")
        logger.info(f"Success rate: {processing_stats['overall_success_rate']:.2%}")
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
```

## Key Differences Between Bedrock Step Types

### 1. By Model Management Strategy
- **Inference Profile Required**: Models requiring provisioned throughput (Claude 4, latest Claude 3.5)
- **On-Demand Compatible**: Models supporting on-demand throughput (Claude 3, older Claude 3.5)
- **Hybrid Strategy**: Intelligent fallback between inference profiles and on-demand models

### 2. By Response Processing
- **JSON Response**: Structured JSON output with optional Pydantic validation
- **Text Response**: Raw text output for simple use cases
- **Structured Response**: Custom parsing with mandatory Pydantic model validation

### 3. By Prompt Complexity
- **Simple Prompts**: Single input column with basic template
- **Complex Prompts**: Multiple input columns with advanced templating
- **Dynamic Prompts**: Runtime prompt generation based on data characteristics

### 4. By Use Case
- **Text Classification**: Categorizing text data with confidence scores
- **Content Analysis**: Extracting insights and structured information
- **Data Enrichment**: Adding LLM-generated features to existing datasets
- **Quality Assessment**: Evaluating content quality and compliance

## Best Practices Identified

1. **Configurable Model Lists**: Allow users to override default model compatibility lists
2. **Intelligent Fallback**: Implement robust fallback strategies for model failures
3. **Programmable Responses**: Support custom Pydantic models for structured outputs
4. **Template-Driven Prompts**: Use configurable prompt templates with variable substitution
5. **Batch Processing**: Process data in configurable batches for efficiency
6. **Comprehensive Logging**: Detailed logging for debugging and monitoring
7. **Error Handling**: Graceful handling of API failures and invalid responses
8. **Retry Logic**: Exponential backoff retry strategies for transient failures
9. **Resource Optimization**: Appropriate instance sizing for LLM processing workloads
10. **Cost Management**: Intelligent model selection to optimize costs

## Testing Implications

Bedrock processing step builders should be tested for:

1. **Model Strategy Determination**: Correct model selection based on configuration
2. **Inference Profile Handling**: Proper ARN and global profile ID usage
3. **Fallback Logic**: Correct fallback to on-demand models when profiles fail
4. **Environment Variable Processing**: Proper handling of all Bedrock-specific env vars
5. **Prompt Template Formatting**: Correct variable substitution in templates
6. **Response Model Validation**: Pydantic model loading and validation
7. **Batch Processing**: Correct handling of different batch sizes
8. **Error Recovery**: Proper error handling and retry logic
9. **Output Generation**: Correct ProcessingOutput creation for all formats
10. **Configuration Validation**: Comprehensive validation of all Bedrock parameters
11. **API Integration**: Mock Bedrock API calls for testing
12. **File Format Support**: Support for CSV and Parquet input/output formats

### Recommended Test Categories

#### Model Management Tests
- Model compatibility list validation
- Inference profile ARN handling
- Global profile ID generation
- Fallback model selection

#### Response Processing Tests
- JSON response parsing and validation
- Pydantic model integration
- Structured response extraction
- Error response handling

#### Prompt System Tests
- Template variable substitution
- Additional input column handling
- System prompt integration
- Dynamic prompt generation

#### Integration Tests
- End-to-end processing workflow
- Multiple file processing
- Batch size optimization
- Resource utilization

## Implementation Examples

### Complete Bedrock Processing Step Builder

```python
from typing import Dict, Optional, Any, List
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from ..configs.config_bedrock_processing_step import BedrockProcessingStepConfig
from ...core.base.builder_base import StepBuilderBase

# Import Bedrock processing specification
try:
    from ..specs.bedrock_processing_spec import BEDROCK_PROCESSING_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    BEDROCK_PROCESSING_SPEC = None
    SPEC_AVAILABLE = False


class BedrockProcessingStepBuilder(StepBuilderBase):
    """Builder for Bedrock Processing Step with configurable model management."""
    
    def __init__(self, config: BedrockProcessingStepConfig, sagemaker_session=None, 
                 role: Optional[str] = None, registry_manager=None, 
                 dependency_resolver=None):
        if not isinstance(config, BedrockProcessingStepConfig):
            raise ValueError("BedrockProcessingStepBuilder requires BedrockProcessingStepConfig")
            
        if not SPEC_AVAILABLE or BEDROCK_PROCESSING_SPEC is None:
            raise ValueError("Bedrock processing specification not available")
            
        super().__init__(
            config=config,
            spec=BEDROCK_PROCESSING_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver,
        )
        self.config: BedrockProcessingStepConfig = config
    
    def validate_configuration(self) -> None:
        """Validate Bedrock processing configuration."""
        # Validate base processing configuration
        required_processing_attrs = [
            'processing_instance_count', 'processing_volume_size',
            'processing_instance_type_large', 'processing_instance_type_small',
            'processing_framework_version', 'use_large_processing_instance'
        ]
        
        for attr in required_processing_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"Missing required processing attribute: {attr}")
        
        # Validate Bedrock-specific configuration
        self.config.validate_bedrock_configuration()
        
        self.log_info("BedrockProcessingStepConfig validation succeeded")
    
    def _get_model_strategy_config(self) -> Dict[str, Any]:
        """Get model usage strategy based on configuration."""
        return self.config.get_model_strategy()
    
    def _create_processor(self) -> SKLearnProcessor:
        """Create SKLearnProcessor with Bedrock-specific configuration."""
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
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Build Bedrock-specific environment variables."""
        env_vars = super()._get_environment_variables()
        
        # Model configuration
        env_vars["BEDROCK_PRIMARY_MODEL_ID"] = self.config.primary_model_id
        env_vars["BEDROCK_FALLBACK_MODEL_ID"] = self.config.fallback_model_id
        
        # Inference profile configuration
        if self.config.inference_profile_arn:
            env_vars["BEDROCK_INFERENCE_PROFILE_ARN"] = self.config.inference_profile_arn
        
        # Model lists as JSON
        env_vars["BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS"] = json.dumps(
            self.config.inference_profile_required_models
        )
        env_vars["BEDROCK_ON_DEMAND_COMPATIBLE_MODELS"] = json.dumps(
            self.config.on_demand_compatible_models
        )
        
        # Prompt configuration
        if self.config.system_prompt:
            env_vars["BEDROCK_SYSTEM_PROMPT"] = self.config.system_prompt
        env_vars["BEDROCK_USER_PROMPT_TEMPLATE"] = self.config.user_prompt_template
        
        # Response configuration
        env_vars["BEDROCK_RESPONSE_FORMAT"] = self.config.response_format
        if self.config.response_model_class:
            env_vars["BEDROCK_RESPONSE_MODEL_CLASS"] = self.config.response_model_class
        
        # API configuration
        env_vars["BEDROCK_MAX_TOKENS"] = str(self.config.max_tokens)
        env_vars["BEDROCK_TEMPERATURE"] = str(self.config.temperature)
        env_vars["BEDROCK_TOP_P"] = str(self.config.top_p)
        env_vars["BEDROCK_MAX_RETRIES"] = str(self.config.max_retries)
        
        # Processing configuration
        env_vars["BEDROCK_BATCH_SIZE"] = str(self.config.batch_size)
        env_vars["BEDROCK_INPUT_DATA_COLUMN"] = self.config.input_data_column
        env_vars["BEDROCK_OUTPUT_COLUMN_PREFIX"] = self.config.output_column_prefix
        
        # Additional input columns as JSON
        if self.config.additional_input_columns:
            env_vars["BEDROCK_ADDITIONAL_INPUT_COLUMNS"] = json.dumps(
                self.config.additional_input_columns
            )
        
        return env_vars
    
    def _get_job_arguments(self) -> List[str]:
        """Build command-line arguments for Bedrock processing script."""
        args = [
            "--primary-model-id", self.config.primary_model_id,
            "--batch-size", str(self.config.batch_size),
            "--max-retries", str(self.config.max_retries),
            "--input-column", self.config.input_data_column,
            "--output-prefix", self.config.output_column_prefix,
            "--response-format", self.config.response_format
        ]
        
        # Add optional arguments
        if self.config.system_prompt:
            args.extend(["--system-prompt", self.config.system_prompt])
            
        if self.config.response_model_class:
            args.extend(["--response-model-class", self.config.response_model_class])
            
        if self.config.inference_profile_arn:
            args.extend(["--inference-profile-arn", self.config.inference_profile_arn])
        
        return args
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create Bedrock ProcessingStep."""
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
        
        self.log_info("Created Bedrock ProcessingStep: %s", step_name)
        return step
```

## Step Registry Integration

### Registry Pattern for Bedrock Steps

Following the cursus framework registry pattern, Bedrock steps must be registered in `src/cursus/registry/step_names_original.py`:

```python
# Registry entries for Bedrock processing steps
"BedrockProcessing": {
    "config_class": "BedrockProcessingStepConfig",
    "builder_step_name": "BedrockProcessingStepBuilder", 
    "spec_type": "BedrockProcessing",
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "AWS Bedrock LLM processing step with configurable model management",
},
"BedrockTextClassification": {
    "config_class": "BedrockTextClassificationConfig",
    "builder_step_name": "BedrockTextClassificationStepBuilder",
    "spec_type": "BedrockTextClassification", 
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Bedrock text classification with structured response models",
},
"BedrockContentAnalysis": {
    "config_class": "BedrockContentAnalysisConfig",
    "builder_step_name": "BedrockContentAnalysisStepBuilder",
    "spec_type": "BedrockContentAnalysis",
    "sagemaker_step_type": "Processing",  # SageMaker step type for processing
    "description": "Bedrock content analysis and insight extraction",
},
```

**Key Registry Pattern Notes:**
- **sagemaker_step_type**: Must be `"Processing"` for all Bedrock processing steps
- **spec_type**: Use case-specific specification type (e.g., "BedrockProcessing")
- **config_class**: Use case-specific configuration class name
- **builder_step_name**: Use case-specific builder class name

This comprehensive pattern analysis provides the foundation for creating robust, production-ready Bedrock processing steps in the cursus framework, with intelligent model management, configurable response processing, and seamless integration with the existing cursus architecture.
