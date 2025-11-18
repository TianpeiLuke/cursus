---
tags:
  - code
  - processing_script
  - bedrock_processing
  - llm_integration
  - aws_bedrock
keywords:
  - bedrock processing
  - AWS Bedrock
  - template-driven processing
  - dynamic Pydantic models
  - inference profile management
  - concurrent processing
  - LLM response validation
  - job type handling
topics:
  - AWS Bedrock integration
  - LLM processing pipelines
  - template-driven automation
  - response validation
  - concurrent API processing
language: python
date of note: 2025-11-18
---

# Bedrock Processing Script Documentation

## Overview

The Bedrock Processing script (`bedrock_processing.py`) is a production-ready, template-driven AWS Bedrock integration that processes input data through Large Language Models (LLMs) using structured prompt templates and validation schemas. This script represents the second stage of a two-step template-driven LLM processing pipeline, consuming outputs from the Bedrock Prompt Template Generation step to provide zero-configuration, validated LLM processing.

The script's architecture emphasizes reliability, performance, and maintainability through intelligent fallback strategies, concurrent processing capabilities, and comprehensive error handling. It automatically adapts to different job types (training, validation, testing, calibration), preserves data split structures for ML workflows, and provides detailed processing statistics for monitoring and debugging.

Key distinguishing features include dynamic Pydantic model creation from JSON schemas for type-safe response validation, intelligent Unicode quote normalization to handle LLM-generated content, inference profile management with automatic fallback for enterprise-scale deployments, and flexible processing modes (sequential or concurrent) optimized for different throughput requirements. The script integrates seamlessly with SageMaker Processing infrastructure while maintaining testability through dependency injection and clean separation of concerns.

## Purpose and Major Tasks

### Primary Purpose

Process tabular input data through AWS Bedrock LLMs using template-driven prompts and validate responses using dynamically-generated Pydantic models to produce structured, validated analysis results.

### Major Tasks

1. **Template Integration and Configuration**: Load prompt templates and validation schemas from Bedrock Prompt Template Generation step outputs, configure AWS Bedrock client with inference profile management, and create dynamic Pydantic models for response validation.

2. **Input Data Processing**: Detect and load data in multiple formats (CSV, TSV, Parquet) with automatic format preservation, map DataFrame columns to template placeholders for prompt generation, and handle train/val/test split structures for training job types.

3. **Prompt Generation and Formatting**: Format user prompts by replacing template placeholders with row data, handle missing placeholders with graceful degradation, and support complex prompt templates with multiple input fields and JSON examples.

4. **AWS Bedrock API Invocation**: Execute Bedrock API calls with intelligent retry logic and exponential backoff, manage inference profiles with automatic fallback to on-demand models, support both sequential and concurrent processing modes with configurable rate limiting, and handle thread-local Bedrock clients for optimal concurrent performance.

5. **Response Parsing and Validation**: Extract JSON from LLM responses using intelligent brace counting, repair Unicode quotes and German-style quotation patterns, validate responses using dynamically-created Pydantic models, and provide structured error responses with detailed failure information.

6. **Batch Processing and Output Management**: Process data in configurable batches with intermediate result saving, preserve input data format (CSV/TSV/Parquet) in outputs, maintain train/val/test directory structure for training job types, and aggregate processing statistics across batches and splits.

7. **Concurrent Processing with Rate Limiting**: Execute multiple requests in parallel using ThreadPoolExecutor, enforce API rate limits with semaphore-based concurrency control, isolate thread failures to prevent cascade effects, and maintain thread-local state for Bedrock clients.

8. **Job Type-Aware Processing**: Detect and handle training job type with train/val/test subdirectories, process non-training job types (validation, testing, calibration) as single datasets, adapt output naming and structure based on job type, and provide fallback behavior when expected structure not found.

9. **Error Handling and Resilience**: Implement comprehensive exception handling at record, batch, and job levels, provide structured error responses with metadata for debugging, continue processing despite individual record failures, and save intermediate results for recovery and monitoring.

10. **Processing Statistics and Monitoring**: Track success rates, validation rates, and error counts at multiple granularities, generate comprehensive processing summaries with model information and template status, provide file-by-file and split-by-split statistics for analysis, and include performance metrics and throughput information.

## Script Contract

### Entry Point

```python
entry_point = "bedrock_processing.py"
```

### Input Paths

| Logical Name | Container Path | Description |
|--------------|----------------|-------------|
| `input_data` | `/opt/ml/processing/input/data` | Input data files in CSV, TSV, or Parquet format. For training job type, expects train/val/test subdirectories. For other job types, expects files directly in directory. Data columns must match template input placeholders. |
| `prompt_templates` | `/opt/ml/processing/input/templates` | Prompt templates from Bedrock Prompt Template Generation step. Must contain `prompts.json` with `system_prompt`, `user_prompt_template`, and `input_placeholders` fields. |
| `validation_schema` | `/opt/ml/processing/input/schema` | Validation schemas from Bedrock Prompt Template Generation step. Must contain `validation_schema_*.json` with JSON schema for response structure including `properties`, `required`, and `processing_config` sections. |

### Output Paths

| Logical Name | Container Path | Description |
|--------------|----------------|-------------|
| `processed_data` | `/opt/ml/processing/output/data` | Processed data with Bedrock responses. For training jobs, maintains train/val/test directory structure. Files include original data plus response fields with configurable prefix (default: `llm_`). Preserves input format (CSV/TSV/Parquet). |
| `analysis_summary` | `/opt/ml/processing/output/summary` | Processing statistics and metadata in JSON format. Includes success rates, validation rates, model information, template integration status, and file-by-file processing details with timestamp. |

### Environment Variables

#### Required Environment Variables

| Variable | Description |
|----------|-------------|
| `BEDROCK_PRIMARY_MODEL_ID` | Primary AWS Bedrock model ID to use for processing. Example: `"anthropic.claude-sonnet-4-20250514-v1:0"`. This is the only required configuration. |

#### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BEDROCK_FALLBACK_MODEL_ID` | `""` | Fallback model ID used when inference profile fails with ValidationException. Essential for production reliability when using inference profiles. Example: `"anthropic.claude-3-5-sonnet-20241022-v2:0"`. |
| `BEDROCK_INFERENCE_PROFILE_ARN` | `None` | Inference profile ARN for capacity management and cost optimization. Example: `"arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"`. |
| `BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS` | `"[]"` | JSON array of model IDs that require inference profiles. Auto-configures known models. Example: `'["anthropic.claude-sonnet-4-20250514-v1:0"]'`. |
| `AWS_DEFAULT_REGION` | `"us-east-1"` | AWS region for Bedrock API calls. All Bedrock operations execute in this region. |
| `BEDROCK_MAX_TOKENS` | `"8192"` | Maximum tokens for Bedrock response generation. Controls response length and cost. |
| `BEDROCK_TEMPERATURE` | `"1.0"` | Temperature parameter for response generation. Higher values increase randomness. Range: 0.0-1.0. |
| `BEDROCK_TOP_P` | `"0.999"` | Top-p sampling parameter for nucleus sampling. Controls diversity of responses. Range: 0.0-1.0. |
| `BEDROCK_BATCH_SIZE` | `"10"` | Number of records processed per batch. Affects memory usage and intermediate save frequency. |
| `BEDROCK_MAX_RETRIES` | `"3"` | Maximum retries for failed Bedrock API calls. Uses exponential backoff between retries. |
| `BEDROCK_OUTPUT_COLUMN_PREFIX` | `"llm_"` | Prefix for output columns containing Bedrock responses. Prevents column name conflicts with input data. |
| `BEDROCK_SKIP_ERROR_RECORDS` | `"false"` | If `"true"`, filters error records from final output. Useful for downstream processing that requires only successful results. |
| `BEDROCK_CONCURRENCY_MODE` | `"sequential"` | Processing mode: `"sequential"` (single-threaded, safer) or `"concurrent"` (multi-threaded, 3-10x faster). |
| `BEDROCK_MAX_CONCURRENT_WORKERS` | `"5"` | Number of concurrent threads for parallel processing. Only applies when `CONCURRENCY_MODE="concurrent"`. Recommended range: 3-10. |
| `BEDROCK_RATE_LIMIT_PER_SECOND` | `"10"` | Maximum API requests per second in concurrent mode. Enforces rate limiting to respect Bedrock API limits. |
| `USE_SECURE_PYPI` | `"false"` | If `"true"`, uses secure CodeArtifact PyPI for package installation. If `"false"`, uses public PyPI. |

### Job Arguments

| Argument | Type | Required | Choices | Description |
|----------|------|----------|---------|-------------|
| `--job_type` | `str` | Yes | `training`, `validation`, `testing`, `calibration` | Determines processing behavior and output structure. `training` expects train/val/test subdirectories. Others expect single dataset. |
| `--batch-size` | `int` | No | - | Batch size for processing. Overrides `BEDROCK_BATCH_SIZE` environment variable. Default: 10. |
| `--max-retries` | `int` | No | - | Maximum retries for Bedrock calls. Overrides `BEDROCK_MAX_RETRIES` environment variable. Default: 3. |

### Framework Requirements

```python
{
    "pandas": ">=1.2.0",
    "boto3": ">=1.26.0",
    "pydantic": ">=2.0.0",  # Required for dynamic model creation
    "tenacity": ">=8.0.0",  # Required for retry logic
    "pathlib": ">=1.0.0",
}
```

## Input Data Structure

### Primary Input: Data Files

**Location**: `/opt/ml/processing/input/data`

**Supported Formats**:
- CSV (`.csv`): Comma-separated values
- TSV (`.tsv`): Tab-separated values  
- Parquet (`.parquet`): Columnar binary format (recommended for performance)

**Job Type-Specific Structure**:

**Training Job Type** (`job_type="training"`):
```
/opt/ml/processing/input/data/
├── train/
│   ├── train_data.csv
│   └── train_data.parquet
├── val/
│   ├── val_data.csv
│   └── val_data.parquet
└── test/
    ├── test_data.csv
    └── test_data.parquet
```

**Non-Training Job Types** (`validation`, `testing`, `calibration`):
```
/opt/ml/processing/input/data/
├── data_file_1.csv
├── data_file_2.parquet
└── data_file_3.tsv
```

**Column Requirements**:
- Must include all columns referenced in template `input_placeholders`
- Column names must match placeholder names exactly (case-sensitive)
- Missing placeholder columns result in `[Missing: column_name]` in prompts

**Example Data Structure**:
```python
{
    "dialogue": "Customer: I need help with my order...",
    "shiptrack": "SHIP-12345",
    "max_estimated_arrival_date": "2025-12-01",
    # ... other columns referenced in template
}
```

### Secondary Input: Prompt Templates

**Location**: `/opt/ml/processing/input/templates/prompts.json`

**Required Structure**:
```json
{
  "system_prompt": "You are an expert analyst...",
  "user_prompt_template": "Analyze the following dialogue: {dialogue}\n\nShipment: {shiptrack}\nETA: {max_estimated_arrival_date}",
  "input_placeholders": ["dialogue", "shiptrack", "max_estimated_arrival_date"]
}
```

**Fields**:
- `system_prompt` (str): System-level prompt providing role and context to the LLM
- `user_prompt_template` (str): User prompt template with placeholder variables in `{variable_name}` format
- `input_placeholders` (List[str]): List of placeholder variable names that must exist as DataFrame columns

### Tertiary Input: Validation Schema

**Location**: `/opt/ml/processing/input/schema/validation_schema_*.json`

**Required Structure**:
```json
{
  "properties": {
    "category": {
      "type": "string",
      "enum": ["Category1", "Category2", "Category3"],
      "description": "The classified category"
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score between 0.0 and 1.0"
    },
    "reasoning": {
      "type": "string",
      "description": "Explanation of classification decision"
    }
  },
  "required": ["category", "confidence", "reasoning"],
  "processing_config": {
    "response_model_name": "BedrockResponse"
  }
}
```

**Schema Capabilities**:
- **Type Validation**: Supports `string`, `number`, `integer`, `boolean`, `array`, `object`
- **Enum Constraints**: Validates categorical fields against allowed values
- **Nested Objects**: Creates nested Pydantic models for complex structures
- **Required Fields**: Enforces presence of critical response fields
- **Dynamic Model Creation**: Automatically generates Pydantic models at runtime

## Output Data Structure

### Primary Output: Processed Data

**Location**: `/opt/ml/processing/output/data`

**Training Job Type Output Structure**:
```
/opt/ml/processing/output/data/
├── train/
│   └── train_processed_data.parquet
├── val/
│   └── val_processed_data.parquet
└── test/
    └── test_processed_data.parquet
```

**Non-Training Job Type Output Structure**:
```
/opt/ml/processing/output/data/
└── {job_type}_processed_data.parquet
```

**Output Column Structure**:

Original input columns are preserved, with Bedrock response fields added using configurable prefix (default: `llm_`):

```python
{
    # Original input columns
    "dialogue": "Customer: I need help...",
    "shiptrack": "SHIP-12345",
    
    # Bedrock response fields (with llm_ prefix)
    "llm_category": "Delivery Issue",
    "llm_confidence": 0.92,
    "llm_reasoning": "Customer expresses concern about delivery status...",
    
    # Processing metadata
    "llm_status": "success",  # or "error"
    "llm_error": None,  # or error message string
    "llm_parse_status": "success",  # or "json_failed", "validation_failed"
    "llm_validation_passed": True,  # or False
}
```

**Intermediate Batch Results** (saved during processing):
```
/opt/ml/processing/output/data/
├── batch_0001_results.parquet
├── batch_0002_results.parquet
└── batch_0003_results.parquet
```

### Secondary Output: Processing Summary

**Location**: `/opt/ml/processing/output/summary/processing_summary_{job_type}_{timestamp}.json`

**Summary Structure**:
```json
{
  "job_type": "training",
  "total_files": 3,
  "total_records": 1000,
  "successful_records": 985,
  "failed_records": 15,
  "validation_passed_records": 980,
  "overall_success_rate": 0.985,
  "overall_validation_rate": 0.980,
  "processing_timestamp": "2025-11-18T12:00:00",
  "effective_model_id": "anthropic.claude-sonnet-4-20250514-v1:0",
  "model_info": {
    "arn": "arn:aws:bedrock:...",
    "method": "arn"
  },
  "template_integration": {
    "system_prompt_loaded": true,
    "user_prompt_template_loaded": true,
    "validation_schema_loaded": true,
    "pydantic_model_created": true
  },
  "files_processed": [
    {
      "filename": "train_data.parquet",
      "records": 700,
      "successful": 690,
      "failed": 10,
      "validation_passed": 688,
      "success_rate": 0.9857,
      "validation_rate": 0.9828
    }
  ],
  "splits_processed": [
    {
      "split_name": "train",
      "total_files": 1,
      "total_records": 700,
      "successful_records": 690,
      "failed_records": 10,
      "validation_passed_records": 688,
      "success_rate": 0.9857,
      "validation_rate": 0.9828,
      "files_processed": [...]
    }
  ]
}
```

## Key Functions and Tasks

### Component 1: Package Installation and Configuration

#### `install_packages(packages: list, use_secure: bool = USE_SECURE_PYPI) -> None`

**Purpose**: Install required Python packages from either public PyPI or secure CodeArtifact PyPI.

**Algorithm**:
```
1. Log package installation configuration (source, count)
2. IF use_secure is True:
     Call install_packages_from_secure_pypi(packages)
   ELSE:
     Call install_packages_from_public_pypi(packages)
3. Log installation success or propagate errors
```

**Parameters**:
- `packages` (List[str]): Package specifications (e.g., `["pandas==1.5.0", "numpy"]`)
- `use_secure` (bool): If True, use CodeArtifact; if False, use public PyPI

**Returns**: None (raises exception on failure)

**Key Operations**:
- Automatically detects PyPI source from `USE_SECURE_PYPI` environment variable
- Handles authentication token retrieval for secure CodeArtifact access
- Provides detailed logging for package installation process

#### `_get_secure_pypi_access_token() -> str`

**Purpose**: Retrieve CodeArtifact authorization token for secure PyPI access using AWS STS role assumption.

**Algorithm**:
```
1. Configure AWS STS regional endpoints
2. Create STS client for us-east-1 region
3. Get current caller identity
4. Assume SecurePyPIReadRole_{account_id} role
5. Create CodeArtifact client with temporary credentials
6. Get authorization token from CodeArtifact domain "amazon"
7. Return authorization token
```

**Returns**: Authorization token string for CodeArtifact access

**Error Handling**: Raises exception with detailed error message if token retrieval fails

### Component 2: Unicode Quote Normalization and JSON Repair

#### `normalize_unicode_quotes(text: str) -> str`

**Purpose**: Normalize Unicode quotation marks to ASCII equivalents while preserving structural JSON quotes.

**Algorithm**:
```
1. FOR EACH Unicode double quote character in UNICODE_DOUBLE_QUOTES:
     Replace with ASCII single quote (')
2. FOR EACH Unicode single quote character in UNICODE_SINGLE_QUOTES:
     Replace with ASCII single quote (')
3. Return normalized text
```

**Critical Design Decision**: Maps ALL fancy quotes to ASCII apostrophe (') rather than double quote (") to avoid creating new JSON delimiters on error path.

**Supported Unicode Quotes**:
- Double quotes: `" " „ ‟` → `'`
- Single quotes: `' ' ‚ ‛` → `'`

#### `repair_json(text: str) -> str`

**Purpose**: Repair Unicode/fancy quotes in JSON responses with focused quote-only repair strategy.

**Algorithm**:
```
1. Apply regex replacement for German quote pattern:
     „text" → \"text\"
2. Normalize remaining Unicode quotes to ASCII equivalents
3. Return repaired JSON string
```

**Production Validation**: Based on analysis of 378,878 records showing 100% of parse errors due to Unicode quotes.

**Key Features**:
- ONLY handles quote-related issues (no generic comma/whitespace fixes)
- Preserves ASCII double quotes (") which are structural in JSON
- Focuses on primary failure mode (German „text" pattern)

#### `extract_json_candidate(response_text: str) -> str`

**Purpose**: Extract first complete JSON object using intelligent brace counting that handles assistant prefilling.

**Algorithm**:
```
1. Find first '{' in response text
2. IF no opening brace found:
     Return trimmed original text
3. Initialize: brace_count=0, in_string=False, escape_next=False
4. FOR EACH character from first '{' to end:
     IF escape_next is True:
       Set escape_next=False, continue
     IF character is '\':
       Set escape_next=True, continue
     IF character is '"' AND NOT escape_next:
       Toggle in_string state, continue
     IF NOT in_string:
       IF character is '{': brace_count++
       IF character is '}': 
         brace_count--
         IF brace_count == 0:
           Return substring from first '{' to current '}'
5. Return substring from first '{' to end (incomplete object)
```

**Handles**:
- Assistant prefilling (response starting with `{` from API)
- Nested JSON objects (only extracts first complete object)
- Braces inside strings (doesn't count them)
- Escape sequences in strings

### Component 3: BedrockProcessor Class

#### `__init__(self, config: Dict[str, Any])`

**Purpose**: Initialize Bedrock processor with template-driven configuration and dynamic Pydantic model creation.

**Algorithm**:
```
1. Store configuration and initialize attributes
2. Create Bedrock client with boto3
3. Configure inference profile settings:
     - Check if inference_profile_arn provided
     - Check if model in inference_profile_required_models list
     - Auto-configure known models (e.g., Claude 4)
     - Set effective_model_id for API calls
4. Load validation schema from configuration
5. Create dynamic Pydantic model from schema:
     - Extract properties and required fields
     - Convert JSON schema types to Python types
     - Create Pydantic model with create_model()
     - Handle nested objects recursively
6. Initialize rate limiting state for concurrent processing:
     - Create semaphore with max_concurrent_workers
     - Initialize thread-local storage
     - Initialize rate limit tracking
```

**Thread Safety**: Initializes thread-local storage for Bedrock clients and rate limiting state.

#### `_configure_inference_profile(self) -> None`

**Purpose**: Configure inference profile settings with intelligent fallback and auto-detection.

**Algorithm**:
```
1. Get model_id and inference_profile_arn from config
2. Parse inference_profile_required_models JSON array
3. IF inference_profile_arn provided:
     Use ARN as effective_model_id
     Set method='arn' in info
4. ELSE IF model_id in inference_profile_required_models:
     IF model is "anthropic.claude-sonnet-4-20250514-v1:0":
       Auto-configure to global profile ID
       effective_model_id = "global.anthropic.claude-sonnet-4-20250514-v1:0"
       Set method='profile_id' in info
     ELSE IF "claude-4" or "claude-sonnet-4" in model_id:
       Log warning about potential profile requirement
5. ELSE IF model_id starts with "global.":
     Already a profile ID, use as-is
     Set method='profile_id' in info
```

**Fallback Strategy**: Always configures fallback_model_id for ValidationException handling.

#### `_create_response_model_from_schema(self) -> None`

**Purpose**: Create dynamic Pydantic model from JSON validation schema for type-safe response parsing.

**Algorithm**:
```
1. Extract properties, required fields, and processing_config from schema
2. IF no properties found:
     Log warning, set response_model_class=None, return
3. FOR EACH field in properties:
     field_type = _convert_json_schema_type_to_python(field_schema)
     description = field_schema.get("description", default)
     IF field in required:
       fields[field] = (field_type, Field(..., description=description))
     ELSE:
       fields[field] = (Optional[field_type], Field(None, description=description))
4. model_name = processing_config.get("response_model_name", "BedrockResponse")
5. Create Pydantic model: response_model_class = create_model(model_name, **fields)
6. Log success with field names
7. ON ERROR:
     Log error, set response_model_class=None
```

**Supported Types**: Handles primitives, arrays, enums, and nested objects recursively.

#### `_convert_json_schema_type_to_python(self, field_schema: Dict[str, Any]) -> type`

**Purpose**: Convert JSON schema type definition to Python type for Pydantic model creation.

**Algorithm**:
```
1. field_type = field_schema.get("type", "string")
2. MATCH field_type:
     CASE "string":
       IF "enum" in field_schema:
         Return str  # Enum validation handled by schema
       ELSE:
         Return str
     CASE "number": Return float
     CASE "integer": Return int
     CASE "boolean": Return bool
     CASE "array":
       items_schema = field_schema.get("items", {})
       items_type = items_schema.get("type")
       IF items_type == "string": Return List[str]
       IF items_type == "number": Return List[float]
       IF items_type == "integer": Return List[int]
       IF items_type == "object":
         nested_model = _create_nested_model_from_schema(items_schema)
         Return List[nested_model]
       ELSE: Return list
     CASE "object":
       Return _create_nested_model_from_schema(field_schema)
     DEFAULT: Return str
```

**Type Mapping**:
| JSON Schema Type | Python Type |
|------------------|-------------|
| `string` | `str` |
| `number` | `float` |
| `integer` | `int` |
| `boolean` | `bool` |
| `array` (string items) | `List[str]` |
| `array` (number items) | `List[float]` |
| `array` (object items) | `List[NestedModel]` |
| `object` | Dynamically created Pydantic model |

#### `_create_nested_model_from_schema(self, object_schema: Dict[str, Any]) -> type`

**Purpose**: Recursively create nested Pydantic models for object-type fields in schemas.

**Algorithm**:
```
1. Extract properties and required fields from object_schema
2. IF no properties:
     Return dict (generic fallback)
3. nested_fields = {}
4. FOR EACH property in properties:
     prop_type = _convert_json_schema_type_to_python(prop_schema)
     prop_description = prop_schema.get("description", default)
     IF property in required:
       nested_fields[property] = (prop_type, Field(..., description=prop_description))
     ELSE:
       nested_fields[property] = (Optional[prop_type], Field(None, description=prop_description))
5. model_name = object_schema.get("title", f"NestedModel_{id(object_schema)}")
6. Return create_model(model_name, **nested_fields)
```

**Recursion**: Handles arbitrarily nested object structures through recursive type conversion.

#### `_format_prompt(self, row_data: Dict[str, Any]) -> str`

**Purpose**: Format user prompt template by replacing placeholders with DataFrame row data.

**Algorithm**:
```
1. placeholders = config.get("input_placeholders", [])
2. IF placeholders is empty:
     Extract placeholders via regex: r"\{(\w+)\}"
3. formatted_prompt = config["user_prompt_template"]
4. FOR EACH placeholder in placeholders:
     placeholder_pattern = "{" + placeholder + "}"
     IF placeholder in row_data:
       value = str(row_data[placeholder]) if not None else ""
       Replace placeholder_pattern with value in formatted_prompt
     ELSE:
       Log warning about missing placeholder
       Replace placeholder_pattern with "[Missing: {placeholder}]"
5. Return formatted_prompt
```

**Design Choice**: Uses string replacement rather than `.format()` to avoid interpreting curly braces in JSON examples as placeholders.

#### `@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))`
#### `_invoke_bedrock(self, prompt: str) -> Dict[str, Any]`

**Purpose**: Invoke AWS Bedrock with intelligent fallback strategy and automatic retry logic.

**Algorithm**:
```
1. IF concurrency_mode == "concurrent":
     Enforce rate limiting via _enforce_rate_limit()
     client = _get_thread_local_bedrock_client()
   ELSE:
     client = self.bedrock_client
2. Build request_body with:
     - anthropic_version="bedrock-2023-05-31"
     - max_tokens from config
     - temperature, top_p from config
     - messages=[{"role": "user", "content": prompt},
                 {"role": "assistant", "content": "{"}]  # Force JSON via prefilling
     - system prompt if configured
3. TRY:
     response = client.invoke_model(
       modelId=effective_model_id,
       body=json.dumps(request_body),
       contentType="application/json"
     )
     Return parsed response JSON
4. ON ValidationException:
     IF fallback_model configured:
       Log fallback activation
       Retry with fallback_model_id
       Return parsed response JSON
     ELSE:
       Re-raise exception
5. ON other exceptions:
     Re-raise (retry decorator handles exponential backoff)
```

**Retry Logic**: Exponential backoff with 3 attempts (multiplier=1, min=4s, max=10s wait).

**Thread Safety**: Uses thread-local clients for concurrent processing to avoid client sharing issues.

#### `_parse_response_with_pydantic(self, response: Dict[str, Any]) -> Dict[str, Any]`

**Purpose**: Parse Bedrock response using Pydantic model validation with focused quote repair strategy.

**Algorithm**:
```
1. Extract response_text from response["content"][0]["text"]
2. IF response_model_class exists:
     # STEP 0: Handle assistant prefilling
     IF response_text doesn't start with "{":
       Prepend "{" to response_text
     
     # STEP 1: Extract JSON candidate
     complete_json = extract_json_candidate(response_text)
     
     # STEP 2: Try parsing as-is
     TRY:
       validated_response = response_model_class.model_validate_json(complete_json)
       result = validated_response.model_dump()
       result["parse_status"] = "success"
       result["validation_passed"] = True
       Return result
     ON ValidationError or JSONDecodeError:
       # STEP 3: Repair and retry
       Log warning about initial parsing failure
       repaired_json = repair_json(complete_json)
       TRY:
         validated_response = response_model_class.model_validate_json(repaired_json)
         result = validated_response.model_dump()
         result["parse_status"] = "success"
         result["validation_passed"] = True
         Return result
       ON ValidationError or JSONDecodeError as second_error:
         Log both errors with JSON samples for debugging
         Raise second_error
3. ELSE (no Pydantic model):
     complete_json = extract_json_candidate(response_text)
     parsed_json = json.loads(complete_json)
     parsed_json["parse_status"] = "json_only"
     parsed_json["validation_passed"] = False
     Return parsed_json
4. ON ValidationError:
     Return structured error with validation_error details
5. ON JSONDecodeError:
     Return structured error with json_error details
```

**Two-Step Repair Strategy**: Try parse → repair quotes → retry. No fallback to raw text to ensure structured output.

#### `process_single_case(self, row_data: Dict[str, Any]) -> Dict[str, Any]`

**Purpose**: Process single case through Bedrock pipeline from prompt formatting to validated response.

**Algorithm**:
```
1. TRY:
     formatted_prompt = _format_prompt(row_data)
     response = _invoke_bedrock(formatted_prompt)
     parsed_result = _parse_response_with_pydantic(response)
     
     result = {
       **parsed_result,
       "processing_status": "success",
       "error_message": None,
       "model_info": {
         "effective_model_id": effective_model_id,
         "inference_profile_info": inference_profile_info
       }
     }
     Return result
2. ON Exception:
     Log error
     error_result = {
       "processing_status": "error",
       "error_message": str(exception),
       "raw_response": None,
       "parse_status": "error",
       "validation_passed": False,
       "model_info": {...}
     }
     IF response_model_class exists:
       Add default None values for expected fields
     Return error_result
```

**Structured Errors**: Always returns dict with consistent structure, never raises exceptions to caller.

#### `process_batch_concurrent(self, df: pd.DataFrame, batch_size: Optional[int], save_intermediate: bool) -> pd.DataFrame`

**Purpose**: Process batch using concurrent execution with ThreadPoolExecutor and rate limiting.

**Algorithm**:
```
1. batch_size = batch_size or config.get("batch_size", 10)
2. total_batches = (len(df) + batch_size - 1) // batch_size
3. Extract and validate placeholders from template
4. FOR i in range(0, len(df), batch_size):
     batch_df = df.iloc[i:i+batch_size]
     batch_num = i // batch_size + 1
     
     # Concurrent processing with ThreadPoolExecutor
     WITH ThreadPoolExecutor(max_workers=max_concurrent_workers) as executor:
       # Submit all tasks
       future_to_row = {}
       FOR idx, row in batch_df.iterrows():
         future = executor.submit(process_single_case_with_rate_limiting, row.to_dict())
         future_to_row[future] = (idx, row)
       
       # Collect results as they complete
       batch_results = []
       FOR future in as_completed(future_to_row):
         idx, original_row = future_to_row[future]
         TRY:
           result = future.result()
           # Combine original row with Bedrock results
           result_row = original_row.to_dict()
           Add Bedrock fields with prefix
           Add processing metadata
           batch_results.append(result_row)
         ON Exception:
           Log error
           Create error_row with status="error"
           batch_results.append(error_row)
     
     results.extend(batch_results)
     
     IF save_intermediate:
       Save batch_results to intermediate parquet file
5. Return DataFrame(results)
```

**Concurrency Features**:
- Thread-local Bedrock clients for thread safety
- Semaphore-based concurrency control
- Individual failure isolation (one failure doesn't stop batch)
- Rate limiting enforced per thread

#### `process_batch_sequential(self, df: pd.DataFrame, batch_size: Optional[int], save_intermediate: bool) -> pd.DataFrame`

**Purpose**: Process batch using sequential single-threaded execution for compatibility and debugging.

**Algorithm**:
```
1. batch_size = batch_size or config.get("batch_size", 10)
2. Extract and validate placeholders from template
3. FOR i in range(0, len(df), batch_size):
     batch_df = df.iloc[i:i+batch_size]
     batch_num = i // batch_size + 1
     
     batch_results = []
     FOR idx, row in batch_df.iterrows():
       row_data = row.to_dict()
       result = process_single_case(row_data)
       
       # Combine original row with Bedrock results
       result_row = row_data.copy()
       Add Bedrock fields with prefix
       Add processing metadata
       batch_results.append(result_row)
     
     results.extend(batch_results)
     
     IF save_intermediate:
       Save batch_results to intermediate parquet file
4. Return DataFrame(results)
```

**Use Cases**: Default mode for reliability, easier debugging, and compatibility.

### Component 4: File I/O and Format Preservation

#### `load_dataframe_with_format(file_path: Path) -> tuple`

**Purpose**: Load DataFrame and detect its format for format-preserving output.

**Algorithm**:
```
1. detected_format = _detect_file_format(file_path)
2. MATCH detected_format:
     CASE "csv": df = pd.read_csv(file_path)
     CASE "tsv": df = pd.read_csv(file_path, sep="\t")
     CASE "parquet": df = pd.read_parquet(file_path)
     DEFAULT: Raise RuntimeError
3. Return (df, detected_format)
```

**Returns**: Tuple of (DataFrame, format_string) for downstream format preservation.

#### `save_dataframe_with_format(df: pd.DataFrame, output_path: Path, format_str: str) -> Path`

**Purpose**: Save DataFrame in specified format, automatically adding correct file extension.

**Algorithm**:
```
1. MATCH format_str:
     CASE "csv":
       file_path = output_path.with_suffix(".csv")
       df.to_csv(file_path, index=False)
     CASE "tsv":
       file_path = output_path.with_suffix(".tsv")
       df.to_csv(file_path, sep="\t", index=False)
     CASE "parquet":
       file_path = output_path.with_suffix(".parquet")
       df.to_parquet(file_path, index=False)
     DEFAULT: Raise RuntimeError
2. Return file_path
```

**Format Preservation**: Ensures output matches input format for seamless pipeline integration.

### Component 5: Template and Schema Loading

#### `load_prompt_templates(templates_path: str, log: Callable) -> Dict[str, Any]`

**Purpose**: Load prompt templates from Bedrock Prompt Template Generation step output.

**Algorithm**:
```
1. templates_dir = Path(templates_path)
2. IF NOT templates_dir.exists():
     Raise ValueError
3. prompts_file = templates_dir / "prompts.json"
4. IF NOT prompts_file.exists():
     Raise ValueError
5. TRY:
     WITH open(prompts_file) as f:
       json_templates = json.load(f)
     
     templates = {}
     IF "system_prompt" in json_templates:
       templates["system_prompt"] = json_templates["system_prompt"]
     IF "user_prompt_template" in json_templates:
       templates["user_prompt_template"] = json_templates["user_prompt_template"]
     IF "input_placeholders" in json_templates:
       templates["input_placeholders"] = json_templates["input_placeholders"]
     ELSE:
       Log "will use regex fallback"
     
     Return templates
6. ON Exception:
     Raise ValueError with detailed error
```

**Required Fields**: `system_prompt`, `user_prompt_template`, `input_placeholders`

#### `load_validation_schema(schema_path: str, log: Callable) -> Dict[str, Any]`

**Purpose**: Load validation schema from Bedrock Prompt Template Generation step output.

**Algorithm**:
```
1. schema_dir = Path(schema_path)
2. IF NOT schema_dir.exists():
     Raise ValueError
3. schema_files = list(schema_dir.glob("validation_schema_*.json"))
4. IF NOT schema_files:
     Raise ValueError
5. schema_file = sorted(schema_files)[-1]  # Most recent
6. TRY:
     WITH open(schema_file) as f:
       schema = json.load(f)
     
     # Validate structure
     required_sections = ["properties", "required"]
     FOR section in required_sections:
       IF section NOT in schema:
         Raise ValueError
     
     Return schema
7. ON Exception:
     Raise ValueError with detailed error
```

**Schema Validation**: Ensures required sections (`properties`, `required`) are present.

### Component 6: Job Type Handling

#### `process_split_directory(split_name: str, split_input_path: Path, split_output_path: Path, processor: BedrockProcessor, config: Dict, log: Callable) -> Dict[str, Any]`

**Purpose**: Process single data split (train/val/test) while maintaining directory structure.

**Algorithm**:
```
1. Create split_output_path directory
2. Find input files in split directory (csv, parquet)
3. IF no files:
     Return empty statistics dict
4. Initialize split_stats dictionary
5. FOR EACH input_file:
     Log processing start
     df, input_format = load_dataframe_with_format(input_file)
     result_df = processor.process_batch(df, save_intermediate=False)
     
     # Update statistics
     Calculate success_count, failed_count, validation_passed_count
     Update split_stats aggregates
     Append file-level stats to files_processed list
     
     # Filter errors if configured
     IF config.get("skip_error_records"):
       Filter out error rows
     
     # Save results
     output_base = split_output_path / f"{split_name}_processed_data"
     saved_file = save_dataframe_with_format(result_df, output_base, input_format)
     Log save completion
6. Calculate split-level rates (success_rate, validation_rate)
7. Return split_stats
```

**Structure Preservation**: Maintains train/val/test directory organization for ML workflows.

### Component 7: Main Processing Logic

#### `main(input_paths: Dict, output_paths: Dict, environ_vars: Dict, job_args: argparse.Namespace, logger: Optional[Callable]) -> Dict[str, Any]`

**Purpose**: Main entry point coordinating template loading, processor initialization, and job type-aware processing.

**Algorithm**:
```
1. Extract job_type from job_args
2. Load prompt templates from input_paths["prompt_templates"]
3. Load validation schema from input_paths["validation_schema"]
4. Build config dict with:
     - Model configuration from environ_vars
     - Templates from loaded templates
     - Validation schema
     - API parameters
     - Concurrency configuration
5. Initialize BedrockProcessor(config)
6. Extract input/output paths
7. Initialize processing_stats dictionary
8. MATCH job_type:
     CASE "training":
       Look for train/val/test subdirectories
       IF subdirectories found:
         FOR EACH split in [train, val, test]:
           IF split exists:
             split_stats = process_split_directory(...)
             Aggregate statistics
             Append to splits_processed
       ELSE:
         Fall back to single dataset processing
     CASE "validation", "testing", "calibration":
       Process as single dataset:
       FOR EACH input_file:
         Load data with format detection
         result_df = processor.process_batch(df)
         Update statistics
         Filter errors if configured
         Save with job_type prefix
9. Calculate overall statistics (success_rate, validation_rate)
10. Add processing_timestamp
11. Save processing summary to JSON
12. Log completion with summary statistics
13. Return processing_stats
```

**Adaptive Processing**: Automatically detects expected structure and falls back gracefully.

## Algorithms and Data Structures

### Algorithm 1: Dynamic Pydantic Model Creation

**Purpose**: Convert JSON schema to Pydantic model at runtime for type-safe validation.

**Data Flow**:
```
JSON Schema → Field Type Conversion → Pydantic Field Creation → Dynamic Model
```

**Detailed Algorithm**:
```
FUNCTION create_dynamic_model(schema: Dict) -> Type[BaseModel]:
  properties = schema["properties"]
  required = schema["required"]
  fields = {}
  
  FOR field_name, field_schema IN properties.items():
    python_type = convert_json_type_to_python(field_schema)
    
    IF field_schema has enum constraint:
      # Pydantic validates enum automatically
      python_type = str
    
    IF field_schema.type == "object":
      # Recursive nested model creation
      nested_model = create_dynamic_model(field_schema)
      python_type = nested_model
    
    IF field_schema.type == "array" AND items.type == "object":
      # Array of nested objects
      nested_model = create_dynamic_model(field_schema["items"])
      python_type = List[nested_model]
    
    IF field_name IN required:
      fields[field_name] = (python_type, Field(..., description=desc))
    ELSE:
      fields[field_name] = (Optional[python_type], Field(None, description=desc))
  
  model_name = schema.get("response_model_name", "BedrockResponse")
  RETURN create_model(model_name, **fields)
```

**Benefits**:
- Type-safe response parsing without manual model definition
- Automatic validation of response structure
- Support for complex nested schemas
- Runtime flexibility for different classification tasks

### Algorithm 2: JSON Extraction with Brace Counting

**Purpose**: Extract first complete JSON object from LLM response handling assistant prefilling and nested structures.

**State Machine**:
```
States: {SEEKING_START, IN_OBJECT, IN_STRING, ESCAPING}
Counters: brace_count (tracks nesting level)
```

**Detailed Algorithm**:
```
FUNCTION extract_json_candidate(response_text: str) -> str:
  start = response_text.find("{")
  IF start == -1:
    RETURN response_text.strip()
  
  brace_count = 0
  in_string = False
  escape_next = False
  
  FOR i FROM start TO len(response_text):
    char = response_text[i]
    
    # Handle escape sequences
    IF escape_next:
      escape_next = False
      CONTINUE
    
    IF char == "\\":
      escape_next = True
      CONTINUE
    
    # Track string boundaries (braces in strings don't count)
    IF char == '"' AND NOT escape_next:
      in_string = NOT in_string
      CONTINUE
    
    # Count braces only outside strings
    IF NOT in_string:
      IF char == "{":
        brace_count++
      ELSE IF char == "}":
        brace_count--
        IF brace_count == 0:
          # Found complete JSON object
          RETURN response_text[start:i+1]
  
  # Incomplete object - return what we have
  RETURN response_text[start:].strip()
```

**Handles Edge Cases**:
- Assistant prefilling (response starts with `{`)
- Nested JSON objects
- Braces inside string values
- Escape sequences (`\"`, `\\`)
- Incomplete JSON objects (returns partial)

### Algorithm 3: Unicode Quote Repair

**Purpose**: Fix Unicode quotation marks in LLM responses that break JSON parsing.

**Quote Mapping Strategy**:
```
ALL Unicode Quotes → ASCII Apostrophe (')
  - Preserves JSON structure (doesn't create new delimiters)
  - Handles German pattern „text" specifically
  - Avoids creating unmatched quotes
```

**Detailed Algorithm**:
```
FUNCTION repair_json(text: str) -> str:
  # STEP 1: Fix German pattern „text" → \"text\"
  # Pattern: „ followed by content, then closing quote
  pattern = r'„([^""\u201c\u201d]*)["\u201c\u201d]'
  text = pattern.sub(r'\\"\1\\"', text)
  
  # STEP 2: Normalize remaining Unicode quotes
  FOR unicode_quote, replacement IN UNICODE_DOUBLE_QUOTES.items():
    text = text.replace(unicode_quote, replacement)  # → '
  
  FOR unicode_quote, replacement IN UNICODE_SINGLE_QUOTES.items():
    text = text.replace(unicode_quote, replacement)  # → '
  
  RETURN text
```

**Production Validation**:
- Tested on 378,878 records
- 100% of parse errors due to Unicode quotes
- Zero false positives (no valid JSON broken)

### Data Structure 1: Thread-Local Storage

**Purpose**: Maintain per-thread state for Bedrock clients in concurrent processing.

**Structure**:
```python
class BedrockProcessor:
    def __init__(self):
        self.thread_local = threading.local()
        # Each thread gets own attributes in self.thread_local
    
    def _get_thread_local_bedrock_client(self):
        if not hasattr(self.thread_local, "bedrock_client"):
            # Create client for this thread
            self.thread_local.bedrock_client = boto3.client(...)
        return self.thread_local.bedrock_client
```

**Benefits**:
- Eliminates client sharing between threads
- Prevents race conditions
- Automatic cleanup when thread completes
- No manual synchronization needed

### Data Structure 2: Processing Statistics Hierarchy

**Purpose**: Track processing metrics at multiple granularities for monitoring and debugging.

**Hierarchy**:
```
ProcessingStats (Job Level)
├── job_type
├── total_records
├── overall_success_rate
├── files_processed []
│   ├── filename
│   ├── records
│   ├── success_rate
│   └── validation_rate
└── splits_processed []  (training jobs only)
    ├── split_name
    ├── total_records
    ├── success_rate
    └── files_processed []
```

**Aggregation Strategy**:
- File-level: Individual file metrics
- Split-level: Aggregated across files in split
- Job-level: Aggregated across all splits or files

## Performance Characteristics

### Throughput Metrics

**Sequential Processing**:
- **Rate**: ~1-2 records/second (depending on model and response time)
- **Bottleneck**: Single-threaded API calls
- **Use Case**: Debugging, small datasets (<100 records)

**Concurrent Processing**:
- **Rate**: ~5-15 records/second (with 5-10 workers)
- **Speedup**: 3-10x faster than sequential
- **Bottleneck**: API rate limits and worker count
- **Use Case**: Production, large datasets (>1000 records)

### Memory Usage

**Per-Record Memory**:
- Prompt template: ~1-5 KB
- Response: ~1-10 KB
- DataFrame row: ~1-5 KB per column
- **Total per record**: ~10-50 KB

**Batch Memory**:
- Batch size 10: ~100-500 KB
- Batch size 100: ~1-5 MB
- **Recommendation**: Keep batch size ≤100 for memory efficiency

**Concurrent Memory**:
- Thread overhead: ~1-2 MB per thread
- 10 workers: ~10-20 MB overhead
- **Total with 10 workers processing batch of 100**: ~15-25 MB

### Processing Time Estimates

**Per-Record Time**:
- API call: 1-5 seconds (model dependent)
- Parsing/validation: <100ms
- Format prompt: <10ms
- **Total**: ~1-5 seconds per record

**Batch Processing Time**:
| Dataset Size | Sequential | Concurrent (10 workers) |
|--------------|------------|-------------------------|
| 100 records | ~2-10 minutes | ~20-60 seconds |
| 1,000 records | ~20-100 minutes | ~3-10 minutes |
| 10,000 records | ~3-17 hours | ~30-100 minutes |

### Rate Limiting

**Bedrock API Limits** (typical):
- Requests per second: 10-50 (model dependent)
- Tokens per minute: 100K-1M (model dependent)

**Script Rate Limiting**:
- Configurable via `BEDROCK_RATE_LIMIT_PER_SECOND`
- Enforced only in concurrent mode
- Semaphore limits total concurrent requests
- Per-thread rate limiting with time tracking

## Error Handling

### Error Categories and Responses

#### 1. Template Loading Errors

**Cause**: Missing or malformed template files

**Handling**:
```python
try:
    templates = load_prompt_templates(templates_path, log)
except ValueError as e:
    log(f"Failed to load templates: {e}")
    raise  # Fatal error - cannot proceed without templates
```

**User Action**: Verify template files exist and are valid JSON

#### 2. Schema Loading Errors

**Cause**: Missing or invalid validation schema

**Handling**:
```python
try:
    schema = load_validation_schema(schema_path, log)
except ValueError as e:
    log(f"Failed to load schema: {e}")
    raise  # Fatal error - validation not possible
```

**User Action**: Verify schema files exist with required sections

#### 3. API Invocation Errors

**Cause**: Network issues, authentication, rate limits, model unavailable

**Handling**:
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(...))
def _invoke_bedrock(prompt):
    try:
        response = client.invoke_model(modelId=effective_model_id, ...)
        return response
    except ValidationException:
        # Intelligent fallback
        if fallback_model:
            response = client.invoke_model(modelId=fallback_model, ...)
            return response
    # Retry decorator handles transient errors
```

**Recovery**: Automatic retry with exponential backoff, fallback to alternate model

#### 4. JSON Parsing Errors

**Cause**: Malformed JSON in LLM response, Unicode quotes

**Handling**:
```python
try:
    # Step 1: Try parse as-is
    result = response_model.model_validate_json(json_text)
except (ValidationError, JSONDecodeError):
    # Step 2: Repair and retry
    repaired = repair_json(json_text)
    try:
        result = response_model.model_validate_json(repaired)
    except:
        # Return structured error, continue processing
        return {
            "processing_status": "error",
            "parse_status": "json_failed",
            "raw_response": json_text[:500]
        }
```

**Recovery**: Two-step repair strategy, structured error response, processing continues

#### 5. Validation Errors

**Cause**: Response doesn't match expected schema (missing fields, wrong types)

**Handling**:
```python
try:
    validated = response_model.model_validate_json(json_text)
    return {**validated.model_dump(), "validation_passed": True}
except ValidationError as e:
    return {
        "validation_error": str(e),
        "parse_status": "validation_failed",
        "validation_passed": False,
        "raw_response": json_text[:500]
    }
```

**Response**: Structured error with validation details, record marked as failed, processing continues

#### 6. Missing Placeholder Errors

**Cause**: Template references column not in DataFrame

**Handling**:
```python
if placeholder not in row_data:
    log(f"Warning: Missing placeholder '{placeholder}'")
    formatted_prompt = formatted_prompt.replace(
        placeholder_pattern,
        f"[Missing: {placeholder}]"
    )
```

**Recovery**: Graceful degradation with placeholder substitution, warning logged, processing continues

### Error Response Structure

All errors return consistent structure:
```python
{
    "processing_status": "error",
    "error_message": "Detailed error description",
    "parse_status": "error" | "json_failed" | "validation_failed",
    "validation_passed": False,
    "raw_response": "First 500 chars of response (if available)",
    "model_info": {...},
    # Default None values for expected output fields
    "category": None,
    "confidence": None,
    "reasoning": None
}
```

### Logging Strategy

**Log Levels**:
- **INFO**: Normal processing progress, batch completion, statistics
- **WARNING**: Degraded functionality, missing placeholders, repair attempts
- **ERROR**: Failed processing, validation errors, API errors

**Log Content**:
```
INFO: Processing batch 1/10 (100 records)
INFO: Loaded templates: system_prompt=True, user_prompt_template=True
WARNING: Missing DataFrame column for placeholder 'field_name'
WARNING: Initial JSON parsing failed, attempting repair
ERROR: Error processing case: ValidationException - model not found
ERROR: JSON repair failed. Original error: Unterminated string
```

## Best Practices

### Configuration

**Model Selection**:
- Use inference profiles for production workloads (cost optimization)
- Always configure fallback model for reliability
- Test with small batches before large-scale processing

**Concurrency**:
- Start with sequential mode for debugging and validation
- Use concurrent mode for production with 5-10 workers
- Adjust rate limits based on your Bedrock API quotas
- Monitor CloudWatch metrics for throttling

**Batch Size**:
- Small datasets (<100): batch_size=10
- Medium datasets (100-1000): batch_size=50
- Large datasets (>1000): batch_size=100
- Balance memory usage vs. intermediate save frequency

### Template Design

**Placeholder Strategy**:
- Use descriptive placeholder names matching DataFrame columns
- Document placeholder requirements in template metadata
- Validate placeholder availability before large-scale processing

**Prompt Engineering**:
- Keep system prompt concise and focused on role definition
- Use clear, specific instructions in user prompt template
- Include output format examples in template for consistency
- Test prompts with diverse input samples

### Schema Design

**Field Definition**:
- Mark critical fields as required in schema
- Use enum constraints for categorical outputs
- Provide clear field descriptions for LLM guidance
- Include validation rules in field constraints

**Type Safety**:
- Use specific types (integer, number, boolean) over generic string
- Define nested objects for complex structured outputs
- Use arrays for multi-value fields
- Leverage Pydantic validation for data quality

### Error Management

**Proactive Monitoring**:
- Review processing summaries for success rates
- Investigate batches with high error rates
- Monitor validation_passed rates for schema alignment
- Track parse_status distribution for parsing issues

**Error Recovery**:
- Enable `skip_error_records` for pipelines requiring only successful results
- Save intermediate results for recovery from failures
- Use structured errors for automated error analysis
- Implement alerting on error rate thresholds

### Performance Optimization

**Concurrent Processing**:
```python
# Production configuration for high throughput
export BEDROCK_CONCURRENCY_MODE="concurrent"
export BEDROCK_MAX_CONCURRENT_WORKERS="10"
export BEDROCK_RATE_LIMIT_PER_SECOND="15"
export BEDROCK_BATCH_SIZE="100"
```

**Sequential Processing**:
```python
# Development configuration for debugging
export BEDROCK_CONCURRENCY_MODE="sequential"
export BEDROCK_BATCH_SIZE="10"
```

**Format Selection**:
- Use Parquet for large datasets (10x faster than CSV)
- Use CSV for human readability and compatibility
- TSV for tab-delimited requirements

## Example Configurations

### Example 1: Simple Classification (Sequential)

**Use Case**: Small dataset classification with single model

```bash
# Environment Configuration
export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
export BEDROCK_CONCURRENCY_MODE="sequential"
export BEDROCK_BATCH_SIZE="10"
export BEDROCK_OUTPUT_COLUMN_PREFIX="llm_"

# Template Structure (prompts.json)
{
  "system_prompt": "You are an expert customer service analyst.",
  "user_prompt_template": "Classify this customer message:\n\n{message}",
  "input_placeholders": ["message"]
}

# Schema Structure (validation_schema.json)
{
  "properties": {
    "category": {"type": "string", "enum": ["Question", "Complaint", "Praise"]},
    "confidence": {"type": "number"}
  },
  "required": ["category", "confidence"]
}

# Execution
python bedrock_processing.py --job_type validation
```

**Expected Output**:
- Single file: `validation_processed_data.parquet`
- Columns: `message`, `llm_category`, `llm_confidence`, `llm_status`
- Processing summary with success rates

### Example 2: Training Pipeline (Concurrent)

**Use Case**: Large training dataset with train/val/test splits

```bash
# Environment Configuration
export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
export BEDROCK_INFERENCE_PROFILE_ARN="arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"
export BEDROCK_FALLBACK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
export BEDROCK_CONCURRENCY_MODE="concurrent"
export BEDROCK_MAX_CONCURRENT_WORKERS="10"
export BEDROCK_RATE_LIMIT_PER_SECOND="15"
export BEDROCK_BATCH_SIZE="100"

# Complex Template (prompts.json)
{
  "system_prompt": "You are an expert analyst for delivery predictions.",
  "user_prompt_template": "Analyze:\nDialogue: {dialogue}\nShipment: {shiptrack}\nETA: {max_estimated_arrival_date}\n\nPredict delivery issue category.",
  "input_placeholders": ["dialogue", "shiptrack", "max_estimated_arrival_date"]
}

# Execution
python bedrock_processing.py --job_type training
```

**Expected Output**:
```
/opt/ml/processing/output/data/
├── train/train_processed_data.parquet
├── val/val_processed_data.parquet
└── test/test_processed_data.parquet
```

**Processing Summary**:
- Split-level statistics for train/val/test
- Overall success rate: ~98%
- Concurrent processing: 10x faster than sequential

### Example 3: Production Multi-Label Classification

**Use Case**: Production deployment with complex schema and error filtering

```bash
# Environment Configuration
export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
export BEDROCK_CONCURRENCY_MODE="concurrent"
export BEDROCK_MAX_CONCURRENT_WORKERS="8"
export BEDROCK_RATE_LIMIT_PER_SECOND="10"
export BEDROCK_SKIP_ERROR_RECORDS="true"  # Filter errors from output
export BEDROCK_OUTPUT_COLUMN_PREFIX="pred_"

# Complex Multi-Label Schema (validation_schema.json)
{
  "properties": {
    "primary_category": {"type": "string", "enum": ["Cat1", "Cat2", "Cat3"]},
    "secondary_categories": {"type": "array", "items": {"type": "string"}},
    "confidence_scores": {
      "type": "object",
      "properties": {
        "primary": {"type": "number"},
        "secondary": {"type": "number"}
      }
    },
    "reasoning": {"type": "string"}
  },
  "required": ["primary_category", "confidence_scores", "reasoning"]
}

# Execution
python bedrock_processing.py --job_type calibration
```

**Expected Output**:
- Filtered dataset with only successful predictions
- Nested object handling for `confidence_scores`
- Array handling for `secondary_categories`
- Production-ready error filtering

## Integration Patterns

### Integration 1: With Bedrock Prompt Template Generation

**Workflow**:
```
1. Bedrock Prompt Template Generation Step
   ↓ Outputs: prompts.json, validation_schema_*.json
2. Bedrock Processing Step (This Script)
   ↓ Outputs: processed_data, analysis_summary
3. Downstream Processing (Training, Evaluation)
```

**Configuration**:
```python
# Step Builder Configuration
bedrock_processing_step = BedrockProcessingStepBuilder(config)
bedrock_processing_step.create_step(
    inputs={
        "input_data": preprocessing_step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri,
        "prompt_templates": template_gen_step.properties.ProcessingOutputConfig.Outputs["prompt_templates"].S3Output.S3Uri,
        "validation_schema": template_gen_step.properties.ProcessingOutputConfig.Outputs["validation_schema"].S3Output.S3Uri
    },
    outputs={
        "processed_data": f"s3://{bucket}/bedrock_processed",
        "analysis_summary": f"s3://{bucket}/bedrock_summary"
    }
)
```

### Integration 2: With Tabular Preprocessing

**Input Structure Alignment**:
- Tabular Preprocessing outputs train/val/test splits
- Bedrock Processing preserves split structure
- PyTorch Training consumes preserved splits

**Benefits**:
- Seamless ML pipeline integration
- Maintains data split integrity
- Enables proper model validation

### Integration 3: With Model Training

**Use Case**: Label augmentation for training

**Workflow**:
```
TabularPreprocessing → BedrockProcessing → PyTorchTraining
                              ↓
                    Augmented training labels
```

**Benefits**:
- LLM-generated labels for semi-supervised learning
- Template-driven label consistency
- Validation-based quality control

## Related Scripts and Components

### Upstream Dependencies

1. **Bedrock Prompt Template Generation** (`bedrock_prompt_template_generation.py`)
   - Generates prompt templates and validation schemas
   - Required input for this script
   - Documentation: `slipbox/scripts/bedrock_prompt_template_generation_script.md`

2. **Tabular Preprocessing** (`tabular_preprocessing.py`)
   - Produces train/val/test data splits
   - Common upstream step for training pipelines
   - Documentation: TBD

### Downstream Consumers

1. **PyTorch Training Steps**
   - Consume processed data with LLM-generated labels
   - Use preserved train/val/test structure

2. **Model Evaluation Steps**
   - Analyze LLM prediction quality
   - Compare LLM vs. model predictions

### Alternative: Bedrock Batch Processing

**Comparison**:
| Feature | Bedrock Processing | Bedrock Batch Processing |
|---------|-------------------|-------------------------|
| Processing Mode | Real-time synchronous | Asynchronous batch |
| Throughput | 5-15 records/sec (concurrent) | 100+ records/sec |
| Cost | On-demand pricing | 50% discount |
| Use Case | Interactive, immediate results | Large-scale, batch jobs |
| Latency | Seconds per record | Hours for job completion |

**When to Use Bedrock Batch Processing**:
- Datasets >10,000 records
- Non-time-sensitive processing
- Cost optimization priority
- Acceptable 24-48 hour latency

**Documentation**: `slipbox/scripts/bedrock_batch_processing_script.md`

## Troubleshooting Guide

### Problem 1: High Error Rate

**Symptoms**: >10% of records failing with parse_status="json_failed"

**Diagnosis**:
```bash
# Check processing summary
cat /opt/ml/processing/output/summary/processing_summary_*.json | jq '.overall_success_rate'

# Examine failed records
cat /opt/ml/processing/output/data/*.parquet | grep '"llm_status": "error"'
```

**Solutions**:
1. Verify template placeholders match DataFrame columns
2. Check for prompt template issues (malformed JSON examples)
3. Review LLM responses in error records for patterns
4. Consider adjusting max_tokens if responses truncated

### Problem 2: ValidationException with Inference Profile

**Symptoms**: "ValidationException: Could not find model"

**Diagnosis**:
```python
# Check effective model ID
import json
summary = json.load(open('processing_summary.json'))
print(summary['effective_model_id'])
print(summary['model_info'])
```

**Solutions**:
1. Verify inference profile ARN is correct
2. Ensure fallback model is configured
3. Check model ID format (use "global." prefix for profile IDs)
4. Confirm IAM permissions for inference profile access

### Problem 3: Low Validation Rate

**Symptoms**: validation_passed_records < 90% of successful_records

**Diagnosis**:
```bash
# Analyze validation failures
cat summary.json | jq '.files_processed[] | select(.validation_rate < 0.9)'
```

**Solutions**:
1. Review schema required fields - may be too strict
2. Check if LLM consistently omits certain fields
3. Adjust prompt template to emphasize required fields
4. Consider making some fields optional in schema

### Problem 4: Slow Processing

**Symptoms**: Processing takes longer than expected

**Diagnosis**:
```bash
# Check concurrency mode
echo $BEDROCK_CONCURRENCY_MODE

# Monitor API rate limiting
# Check CloudWatch Bedrock metrics
```

**Solutions**:
1. Enable concurrent mode: `BEDROCK_CONCURRENCY_MODE="concurrent"`
2. Increase workers: `BEDROCK_MAX_CONCURRENT_WORKERS="10"`
3. Adjust rate limit: `BEDROCK_RATE_LIMIT_PER_SECOND="15"`
4. Use Parquet format instead of CSV for I/O speed
5. Increase batch size: `BEDROCK_BATCH_SIZE="100"`

### Problem 5: Out of Memory

**Symptoms**: Container crashes or OOM errors

**Diagnosis**:
```bash
# Check batch size and dataset size
echo $BEDROCK_BATCH_SIZE
ls -lh /opt/ml/processing/input/data/
```

**Solutions**:
1. Reduce batch size: `BEDROCK_BATCH_SIZE="10"`
2. Reduce concurrent workers: `BEDROCK_MAX_CONCURRENT_WORKERS="3"`
3. Request larger instance type in step configuration
4. Use Parquet format (more memory efficient than CSV)

## Version History and Changes

### Version 1.0 (Current)

**Features**:
- Template-driven processing with zero configuration
- Dynamic Pydantic model creation from JSON schemas
- Concurrent processing with rate limiting
- Job type-aware processing (training/validation/testing/calibration)
- Inference profile management with automatic fallback
- Unicode quote normalization and JSON repair
- Comprehensive error handling and recovery
- Format preservation (CSV/TSV/Parquet)
- Split structure preservation for ML workflows
- Detailed processing statistics and monitoring

**Known Limitations**:
- Maximum 100 concurrent workers (ThreadPoolExecutor limit)
- Inference profile auto-configuration only for known models
- Schema validation limited to JSON Schema Draft 7
- No support for streaming responses
- Template placeholders must be in DataFrame columns

**Future Enhancements**:
- Streaming response support for longer outputs
- Custom validation rule support beyond JSON Schema
- Multi-region fallback for resilience
- Automatic prompt optimization based on success rates
- Integration with Amazon Bedrock Guardrails

## References and Resources

### Internal Documentation

- **Contract**: [`src/cursus/steps/contracts/bedrock_processing_contract.py`](../../src/cursus/steps/contracts/bedrock_processing_contract.py)
- **Template Generation**: [`bedrock_prompt_template_generation_script.md`](bedrock_prompt_template_generation_script.md)
- **Batch Processing**: [`bedrock_batch_processing_script.md`](bedrock_batch_processing_script.md)

### External Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/): Official AWS documentation for Bedrock LLM service
- [Pydantic Documentation](https://docs.pydantic.dev/latest/): Data validation using Python type annotations
- [JSON Schema Specification](https://json-schema.org/): JSON Schema standard for validation
- [Tenacity Retry Library](https://tenacity.readthedocs.io/): Retry logic with exponential backoff
- [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html): Python concurrent execution framework

### Related Design Documents

- **[Bedrock Processing Step Builder Patterns](../1_design/bedrock_processing_step_builder_patterns.md)**: Template-driven processing architecture and step builder integration patterns
- **[Bedrock Batch Processing Step Builder Patterns](../1_design/bedrock_batch_processing_step_builder_patterns.md)**: Batch processing architecture and cost optimization strategies
- **[Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md)**: Upstream template generation design and integration patterns

---

**Document Status**: Complete
**Last Updated**: 2025-11-18
**Review Status**: Production-Ready
**Maintainer**: ML Infrastructure Team
