---
tags:
  - code
  - processing_script
  - bedrock
  - batch_inference
  - llm_processing
  - aws_integration
keywords:
  - AWS Bedrock
  - batch inference
  - LLM processing
  - cost optimization
  - template-driven processing
  - Pydantic validation
  - S3 integration
  - automatic fallback
  - JSONL conversion
topics:
  - AWS Bedrock integration
  - batch processing
  - cost-efficient LLM inference
  - enterprise ML pipelines
language: python
date of note: 2025-11-18
---

# Bedrock Batch Processing Script Documentation

## Overview

The `bedrock_batch_processing.py` script provides AWS Bedrock batch inference capabilities for cost-efficient processing of large datasets. It extends the functionality of `bedrock_processing.py` while maintaining identical input/output interfaces, enabling seamless drop-in replacement with up to 50% cost savings for large-scale LLM processing tasks.

The script integrates with the Bedrock Prompt Template Generation step to provide a complete template-driven LLM processing pipeline with automatic batch vs real-time processing selection, intelligent fallback mechanisms, and comprehensive error handling.

## Purpose and Major Tasks

### Primary Purpose
Process large datasets through AWS Bedrock LLMs using batch inference for cost optimization while maintaining full compatibility with real-time processing workflows and providing automatic fallback for reliability.

### Major Tasks
1. **Package Installation**: Dynamic package installation with secure/public PyPI support
2. **Template Integration**: Load prompt templates and validation schemas from Template Generation step
3. **Processing Mode Selection**: Intelligently choose between batch and real-time processing
4. **Batch Job Management**: Create, monitor, and retrieve AWS Bedrock batch inference jobs
5. **JSONL Conversion**: Convert DataFrames to Bedrock-compatible JSONL format
6. **S3 Operations**: Upload inputs and download batch results using cursus framework patterns
7. **Multi-Job Processing**: Split large datasets across multiple batch jobs for AWS compliance
8. **Result Reconstruction**: Convert batch results back to DataFrame format with validation
9. **Automatic Fallback**: Seamlessly fall back to real-time processing on batch failures
10. **Format Preservation**: Maintain input file formats (CSV, TSV, Parquet) in outputs

## Script Contract

### Entry Point
```
bedrock_batch_processing.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `input_data` | `/opt/ml/processing/input/data` | Input data files (CSV, TSV, Parquet) |
| `prompt_templates` | `/opt/ml/processing/input/templates` | Prompt templates from Template Generation step |
| `validation_schema` | `/opt/ml/processing/input/schema` | Validation schemas for response processing |

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `processed_data` | `/opt/ml/processing/output/data` | Processed data with Bedrock responses |
| `analysis_summary` | `/opt/ml/processing/output/summary` | Processing statistics and metadata |

### Required Environment Variables
| Variable | Description | Example |
|----------|-------------|---------|
| `BEDROCK_PRIMARY_MODEL_ID` | Primary Bedrock model ID | `"anthropic.claude-sonnet-4-20250514-v1:0"` |

### Optional Environment Variables

#### Standard Bedrock Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `BEDROCK_FALLBACK_MODEL_ID` | `""` | Fallback model for inference profile failures |
| `BEDROCK_INFERENCE_PROFILE_ARN` | `""` | Inference profile ARN for capacity management |
| `BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS` | `"[]"` | JSON array of models requiring inference profiles |
| `AWS_DEFAULT_REGION` | `"us-east-1"` | AWS region for Bedrock API |
| `BEDROCK_MAX_TOKENS` | `"32768"` | Maximum tokens for responses |
| `BEDROCK_TEMPERATURE` | `"1.0"` | Temperature for response generation |
| `BEDROCK_TOP_P` | `"0.999"` | Top-p sampling parameter |
| `BEDROCK_BATCH_SIZE` | `"10"` | Records per processing batch (real-time mode) |
| `BEDROCK_MAX_RETRIES` | `"3"` | Maximum retries for failed requests |
| `BEDROCK_OUTPUT_COLUMN_PREFIX` | `"llm_"` | Prefix for output columns |
| `BEDROCK_SKIP_ERROR_RECORDS` | `"false"` | Skip error records in output |
| `BEDROCK_MAX_CONCURRENT_WORKERS` | `"5"` | Concurrent threads (real-time mode) |
| `BEDROCK_RATE_LIMIT_PER_SECOND` | `"10"` | API requests per second limit |
| `BEDROCK_CONCURRENCY_MODE` | `"sequential"` | Processing mode (sequential/concurrent) |

#### Batch-Specific Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `BEDROCK_BATCH_MODE` | `"auto"` | Batch processing mode: auto/batch/realtime |
| `BEDROCK_BATCH_THRESHOLD` | `"1000"` | Minimum records for automatic batch processing |
| `BEDROCK_BATCH_ROLE_ARN` | `""` | IAM role ARN for batch inference jobs |
| `BEDROCK_BATCH_INPUT_S3_PATH` | `""` | S3 path for batch input data (set by step builder) |
| `BEDROCK_BATCH_OUTPUT_S3_PATH` | `""` | S3 path for batch output data (set by step builder) |
| `BEDROCK_BATCH_TIMEOUT_HOURS` | `"24"` | Maximum hours for batch job completion |
| `BEDROCK_MAX_RECORDS_PER_JOB` | `"45000"` | Maximum records per batch job (AWS limit: 50K) |
| `BEDROCK_MAX_CONCURRENT_BATCH_JOBS` | `"20"` | Maximum concurrent batch jobs (AWS limit) |

#### Input Truncation Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `BEDROCK_MAX_INPUT_FIELD_LENGTH` | `"400000"` | Maximum characters per input field |
| `BEDROCK_TRUNCATION_ENABLED` | `"true"` | Enable input field truncation |
| `BEDROCK_LOG_TRUNCATIONS` | `"true"` | Log truncation operations |

#### Package Installation
| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SECURE_PYPI` | `"false"` | Use secure CodeArtifact PyPI (true/false) |

### Job Arguments
| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--job_type` | `str` | Yes | Job type: training/validation/testing/calibration |
| `--batch-size` | `int` | No | Batch size for real-time processing (default: 10) |
| `--max-retries` | `int` | No | Maximum retries for Bedrock calls (default: 3) |

## Input Data Structure

### Expected Input Format
```
input_data/
├── train/ (for training job_type)
│   ├── data.csv (or .tsv, .parquet, .csv.gz, .parquet.gz)
│   └── ...
├── val/
│   └── data.csv
├── test/
│   └── data.csv
└── data.csv (for non-training job_types)
```

### Template Files (from Template Generation Step)
```
prompt_templates/
└── prompts.json
    {
      "system_prompt": "System prompt for Bedrock",
      "user_prompt_template": "Template with {placeholder} fields",
      "input_placeholders": ["placeholder1", "placeholder2"]
    }
```

### Validation Schema (from Template Generation Step)
```
validation_schema/
└── validation_schema_TIMESTAMP.json
    {
      "properties": {
        "field1": {"type": "string", "description": "..."},
        "field2": {"type": "number", "description": "..."}
      },
      "required": ["field1"],
      "processing_config": {
        "response_model_name": "BedrockResponse"
      }
    }
```

### Required Data Columns
- Data columns must match template `input_placeholders`
- Example: If template has `{buyer_name}` and `{product}`, DataFrame needs `buyer_name` and `product` columns

## Output Data Structure

### Processed Data Output
```
processed_data/
├── train/ (mirrors input structure for training job_type)
│   └── train_processed_data.{csv|tsv|parquet}
├── val/
│   └── val_processed_data.{csv|tsv|parquet}
├── test/
│   └── test_processed_data.{csv|tsv|parquet}
└── {job_type}_processed_data.{csv|tsv|parquet} (non-training)
```

**Output Columns**:
- All original input columns (preserved)
- `{prefix}field1`, `{prefix}field2`: Validated response fields (from schema)
- `{prefix}status`: Processing status (success/error)
- `{prefix}error`: Error message (if status=error)
- `{prefix}validation_passed`: Pydantic validation result (boolean)
- `{prefix}parse_status`: Parsing status (success/validation_failed/json_failed/error)

### Processing Summary Output
```
analysis_summary/
└── processing_summary_{job_type}_TIMESTAMP.json
    {
      "job_type": "training",
      "total_records": 50000,
      "successful_records": 49500,
      "failed_records": 500,
      "validation_passed_records": 49000,
      "overall_success_rate": 0.99,
      "overall_validation_rate": 0.98,
      "batch_processing_used": true,
      "batch_job_info": {...},
      "truncation_stats": {...},
      "splits_processed": [...]
    }
```

## Processing Modes

### 1. Auto Mode (Recommended)
**Behavior**: Automatically selects batch vs real-time based on data size and configuration

**Decision Logic**:
```python
use_batch = (
    len(df) >= BEDROCK_BATCH_THRESHOLD and
    BEDROCK_BATCH_ROLE_ARN is not None and
    S3_paths_configured
)
```

**Use Cases**:
- Production pipelines with variable data sizes
- Cost optimization without manual intervention
- Default recommended mode

### 2. Batch Mode (Forced)
**Behavior**: Always uses batch processing regardless of data size

**Requirements**:
- `BEDROCK_BATCH_ROLE_ARN` configured
- S3 paths set by step builder
- Valid IAM permissions

**Use Cases**:
- Guaranteed cost savings needed
- Large dataset processing
- Non-latency-sensitive workflows

### 3. Real-time Mode (Forced)
**Behavior**: Identical to `bedrock_processing.py` sequential processing

**Use Cases**:
- Low-latency requirements
- Small datasets (< 1000 records)
- Development and testing

## Key Functions and Tasks

### Package Installation Component

#### `install_packages(packages, use_secure)`
**Purpose**: Install required Python packages from PyPI (public or secure)

**Algorithm**:
```python
1. Check USE_SECURE_PYPI environment variable
2. If secure:
   - Retrieve CodeArtifact access token via AWS STS
   - Build secure index URL with token
   - Install via pip with --index-url
3. If public:
   - Install via standard pip
```

**Packages Installed**:
- `pydantic==2.11.2`: Dynamic response model creation
- `tenacity==8.5.0`: Retry logic with exponential backoff
- `boto3>=1.35.0`: AWS SDK for Bedrock and S3
- `botocore>=1.35.0`: AWS core library

### File I/O Component

#### `_detect_file_format(file_path)`
**Purpose**: Detect data file format from extension

**Supported Formats**:
- `.csv`: Comma-separated values
- `.tsv`: Tab-separated values
- `.parquet`: Apache Parquet binary format

**Returns**: Format string ('csv', 'tsv', or 'parquet')

#### `load_dataframe_with_format(file_path)`
**Purpose**: Load DataFrame with automatic format detection

**Algorithm**:
```python
1. Detect format from file extension
2. Use appropriate pandas reader:
   - CSV: pd.read_csv()
   - TSV: pd.read_csv(sep='\t')
   - Parquet: pd.read_parquet()
3. Return (DataFrame, format_string)
```

#### `save_dataframe_with_format(df, output_path, format_str)`
**Purpose**: Save DataFrame in specified format with proper extension

**Algorithm**:
```python
1. Add appropriate file extension
2. Use format-specific writer
3. Return saved file path
```

### JSON Processing Component

#### `normalize_unicode_quotes(text)`
**Purpose**: Convert Unicode quotation marks to ASCII equivalents

**Unicode Mappings**:
```python
# All fancy quotes → ASCII apostrophe (')
"\u201c" → "'"  # Left double quote
"\u201d" → "'"  # Right double quote
"\u2018" → "'"  # Left single quote
"\u2019" → "'"  # Right single quote
```

**Critical Design**: Maps to apostrophe (') not double quote (") to prevent creating new JSON delimiters

#### `repair_json(text)`
**Purpose**: Repair Unicode/fancy quotes in JSON responses

**Algorithm**:
```python
1. Fix German quote pattern: „text" → \"text\"
   - Pattern: „([^""\u201c\u201d]*)["\u201c\u201d]
   - Replacement: \\"\1\\"
2. Normalize remaining Unicode quotes to ASCII
3. Return repaired JSON string
```

**Production Context**: Based on analysis of 378,878 records showing 100% of parse errors due to Unicode quotes

#### `extract_json_candidate(response_text)`
**Purpose**: Extract first complete JSON object using intelligent brace counting

**Algorithm**:
```python
1. Find first opening brace '{'
2. Track brace balance while accounting for:
   - Escape sequences (\")
   - String boundaries (ignore braces inside strings)
3. Return substring when brace_count returns to 0
4. Fallback: return from first brace if no complete object
```

### BedrockProcessor Base Class

#### `__init__(config)`
**Purpose**: Initialize Bedrock processor with template-driven configuration

**Initialization Steps**:
1. Initialize Bedrock client
2. Configure inference profile settings
3. Load validation schema
4. Create dynamic Pydantic model from schema
5. Set up rate limiting and concurrency
6. Configure input truncation

#### `_create_response_model_from_schema()`
**Purpose**: Create dynamic Pydantic model from JSON schema

**Algorithm**:
```python
1. Extract properties and required fields from schema
2. For each property:
   a. Convert JSON schema type to Python type
   b. Handle nested objects recursively
   c. Create Field with description
   d. Mark as required or Optional
3. Use create_model() to generate Pydantic class
4. Store as self.response_model_class
```

**Type Conversions**:
| JSON Schema Type | Python Type | Notes |
|-----------------|-------------|-------|
| `string` | `str` | With enum support |
| `number` | `float` | Decimal values |
| `integer` | `int` | Whole numbers |
| `boolean` | `bool` | True/False |
| `array` | `List[T]` | Typed arrays |
| `object` | Nested Model | Recursive creation |

#### `_format_prompt(row_data)`
**Purpose**: Format prompt using template placeholders and DataFrame row data

**Algorithm**:
```python
1. Apply input truncation if enabled
2. Extract placeholders from template:
   - Use input_placeholders from config (preferred)
   - Fall back to regex extraction
3. For each placeholder:
   a. Replace {placeholder} with row_data[placeholder]
   b. Convert value to string
   c. Handle missing placeholders with warning
4. Return formatted prompt string
```

**Input Truncation**:
```python
if len(field_value) > MAX_LENGTH:
    truncated = field_value[:MAX_LENGTH - len(marker)] + marker
    log truncation statistics
```

#### `_invoke_bedrock(prompt)`
**Purpose**: Invoke Bedrock API with intelligent fallback strategy

**Algorithm**:
```python
1. Enforce rate limiting (concurrent mode)
2. Build request body:
   - anthropic_version: "bedrock-2023-05-31"
   - max_tokens, temperature, top_p
   - messages: [user prompt, assistant prefill "{"]
   - system prompt (if configured)
3. Try primary model/inference profile:
   a. client.invoke_model(modelId=effective_model_id)
   b. Parse JSON response
4. On ValidationException:
   a. Fall back to fallback_model_id
   b. Retry with on-demand model
5. Return response dictionary
```

**Retry Mechanism**: Using `@retry` decorator with exponential backoff (3 attempts, 4-10 second delays)

#### `_parse_response_with_pydantic(response)`
**Purpose**: Parse Bedrock response with Pydantic validation and focused quote repair

**Algorithm**:
```python
1. Extract response text from API response
2. Handle assistant prefilling (prepend "{" if missing)
3. Extract JSON candidate (smart brace counting)
4. Try parsing as-is:
   a. Use response_model_class.model_validate_json()
   b. Return validated dict on success
5. On failure, apply focused quote repair:
   a. repair_json(complete_json)
   b. Retry validation
6. On second failure:
   a. Log both errors with JSON samples
   b. Raise exception for error handling
7. Return result with parse_status metadata
```

**Parse Status Values**:
- `"success"`: Validation passed
- `"json_only"`: JSON parsed without Pydantic (no model)
- `"validation_failed"`: Pydantic validation failed
- `"json_failed"`: JSON parsing failed
- `"error"`: Processing error

### BedrockBatchProcessor Class (Extends BedrockProcessor)

#### `__init__(config)`
**Purpose**: Initialize batch processor with additional batch-specific configuration

**Additional Configuration**:
- Batch mode and threshold
- AWS Bedrock batch limits (45K records/job, 20 concurrent jobs)
- S3 bucket and prefix parsing
- Bedrock batch client initialization
- Multi-job processing support

#### `_parse_s3_path(s3_path)`
**Purpose**: Parse S3 path into bucket and prefix components

**Algorithm**:
```python
if s3_path.startswith("s3://"):
    remove "s3://" prefix
    split on "/"
    bucket = first element
    prefix = remaining elements joined with "/"
    return (bucket, prefix)
```

#### `should_use_batch_processing(df)`
**Purpose**: Determine whether to use batch or real-time processing

**Decision Logic**:
```python
if batch_mode == "realtime":
    return False
elif batch_mode == "batch":
    return True
else:  # auto mode
    return (
        len(df) >= batch_threshold and
        batch_role_arn is not None and
        input_bucket is not None and
        output_bucket is not None
    )
```

#### `convert_df_to_jsonl(df)`
**Purpose**: Convert DataFrame to Bedrock batch JSONL format

**Algorithm**:
```python
jsonl_records = []
for pos, row in enumerate(df):
    # Use parent class template formatting
    prompt = self._format_prompt(row.to_dict())
    
    # Create Bedrock batch record
    record = {
        "recordId": f"record_{pos}",
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "{"}  # Prefilling
            ],
            "system": system_prompt  # if configured
        }
    }
    jsonl_records.append(record)

return jsonl_records
```

#### `upload_jsonl_to_s3(jsonl_records)`
**Purpose**: Upload JSONL data to S3 using multipart upload for large files

**Algorithm**:
```python
1. Generate timestamped S3 key
2. Convert records to JSONL string
3. Check content size:
   a. If < 100MB: use standard put_object()
   b. If >= 100MB: use multipart upload
4. Return S3 URI
```

**Multipart Upload** (for files > 100MB):
```python
1. Initiate multipart upload
2. Split content into 100MB chunks
3. Upload each part with part number
4. Collect part ETags
5. Complete multipart upload with parts list
6. On error: abort upload to avoid orphaned parts
```

#### `create_batch_job(input_s3_uri)`
**Purpose**: Create Bedrock batch inference job

**Algorithm**:
```python
1. Generate job name: "cursus-bedrock-batch-TIMESTAMP"
2. Validate job name against AWS requirements:
   - Pattern: [a-zA-Z0-9]{1,63}(-*[a-zA-Z0-9\+\-\.]){0,63}
   - No underscores allowed
3. Generate output S3 URI from framework path
4. Call bedrock_batch_client.create_model_invocation_job():
   - jobName, roleArn, modelId
   - inputDataConfig with S3 URI
   - outputDataConfig with S3 URI
   - timeoutDurationInHours
5. Return job ARN
```

**Job Name Validation**:
- AWS Bedrock requires specific naming pattern
- Alphanumeric, hyphens, plus signs, dots allowed
- Underscores and special characters NOT allowed
- Maximum 126 characters

#### `monitor_batch_job(job_arn)`
**Purpose**: Monitor batch job until completion with exponential backoff

**Algorithm**:
```python
start_time = time.time()
check_count = 0

while True:
    # Get job status
    response = bedrock_batch_client.get_model_invocation_job(job_arn)
    status = response["status"]
    
    # Log progress
    elapsed = time.time() - start_time
    log(f"Status: {status}, Elapsed: {elapsed/60:.1f} min")
    
    # Check terminal states
    if status == "Completed":
        return response
    elif status in ["Failed", "Stopping", "Stopped"]:
        raise RuntimeError(f"Job failed: {status}")
    
    # Exponential backoff
    check_count += 1
    wait_time = min(60, 10 * (1.2 ** check_count))
    time.sleep(wait_time)
```

**Status Polling Strategy**:
- Initial wait: 10 seconds
- Exponential increase: 1.2x multiplier
- Maximum wait: 60 seconds
- Continues until terminal state

#### `download_batch_results(job_response)`
**Purpose**: Download and parse batch job results from S3

**Algorithm**:
```python
1. Extract output S3 URI from job response
2. Parse bucket and prefix from URI
3. List objects in output location
4. Filter for .jsonl.out files (Bedrock output format)
5. For each result file:
   a. Download from S3
   b. Parse JSONL (one JSON object per line)
   c. Collect all results
6. Return list of result dictionaries
```

#### `convert_batch_results_to_df(batch_results, original_df)`
**Purpose**: Convert batch results back to DataFrame format

**Algorithm**:
```python
1. Create mapping: recordId → original DataFrame index
2. For each result:
   a. Extract original row data by index
   b. Parse modelOutput using parent class method
   c. Add LLM response fields with prefix
   d. Add processing metadata (status, error)
   e. Preserve original index
3. Convert to DataFrame
4. Restore original index and sort
5. Return results DataFrame
```

**Critical**: Maintains exact compatibility with real-time processing output format

#### `_split_dataframe_for_batch(df)`
**Purpose**: Split DataFrame into chunks complying with AWS Bedrock limits

**AWS Limits**:
- Record count: 50,000 per file (conservative: 45,000)
- File size: 1GB per file (conservative: 900MB)

**Algorithm**:
```python
chunks = []
current_start = 0

while current_start < total_records:
    # Start with max record count
    current_end = min(current_start + max_records, total_records)
    chunk = df[current_start:current_end]
    
    # Check size constraint
    jsonl = convert_df_to_jsonl(chunk)
    size = estimate_jsonl_size(jsonl)
    
    # Reduce if too large
    while size > MAX_SIZE and len(chunk) > 1:
        size_ratio = MAX_SIZE / size
        new_size = int(len(chunk) * size_ratio * 0.9)  # 90% safety
        chunk = df[current_start:current_start + new_size]
        jsonl = convert_df_to_jsonl(chunk)
        size = estimate_jsonl_size(jsonl)
    
    chunks.append(chunk)
    current_start = current_end

return chunks
```

#### `_monitor_multiple_batch_jobs(job_arns)`
**Purpose**: Monitor multiple batch jobs in parallel

**Algorithm**:
```python
job_statuses = {arn: "Submitted" for arn in job_arns}
job_responses = {arn: None for arn in job_arns}

while True:
    all_completed = True
    
    # Check each job
    for job_arn in job_arns:
        if job_statuses[job_arn] in terminal_states:
            continue
        
        response = get_job_status(job_arn)
        job_statuses[job_arn] = response["status"]
        job_responses[job_arn] = response
        
        if status not in terminal_states:
            all_completed = False
    
    # Log progress
    log_job_statistics(job_statuses)
    
    # Check completion
    if all_completed:
        check_for_failures(job_statuses)
        return [job_responses[arn] for arn in job_arns]
    
    # Exponential backoff
    time.sleep(wait_time)
```

#### `_process_multi_batch_jobs(df)`
**Purpose**: Process multiple batch jobs for large datasets (> 45K records)

**Workflow**:
```python
1. Split DataFrame into compliant chunks
2. For each chunk:
   a. Convert to JSONL
   b. Upload to S3
   c. Create batch job
3. Monitor all jobs in parallel
4. For each job response:
   a. Download results
   b. Convert to DataFrame
5. Merge all results
6. Sort by original index
7. Return combined DataFrame
```

**Example**: 100K records → 3 jobs (45K, 45K, 10K) processed in parallel

### Template Integration Component

#### `load_prompt_templates(templates_path, log)`
**Purpose**: Load prompt templates from Template Generation step output

**Expected File**: `prompts.json`

**Algorithm**:
```python
1. Check templates directory exists
2. Load prompts.json file
3. Extract fields:
   - system_prompt: System prompt for Bedrock
   - user_prompt_template: Template with placeholders
   - input_placeholders: List of placeholder names
4. Validate required fields present
5. Return templates dictionary
```

#### `load_validation_schema(schema_path, log)`
**Purpose**: Load validation schema from Template Generation step output

**Expected File**: `validation_schema_*.json`

**Algorithm**:
```python
1. Find validation schema files (pattern matching)
2. Use most recent schema file
3. Load JSON schema
4. Validate required sections:
   - properties: Field definitions
   - required: Required field list
5. Return schema dictionary
```

### Main Processing Logic

#### `process_split_directory(split_name, split_input_path, split_output_path, processor, config, log)`
**Purpose**: Process a single split directory (train/val/test)

**Algorithm**:
```python
1. Create output directory
2. Find input files (CSV, TSV, Parquet, compressed)
3. For each input file:
   a. Load with format detection
   b. Process using batch processor (auto-selection)
   c. Track batch processing usage
   d. Calculate statistics
   e. Filter error records if configured
   f. Save with format preservation
4. Calculate split-level statistics
5. Return split statistics
```

#### `main(input_paths, output_paths, environ_vars, job_args, logger)`
**Purpose**: Main orchestration logic for batch processing

**Workflow**:
```python
1. Load prompt templates (required)
2. Load validation schema (required)
3. Build configuration with batch settings
4. Initialize BedrockBatchProcessor
5. Determine job type:
   a. Training: Process train/val/test splits
   b. Non-training: Process single dataset
6. For each file/split:
   a. Load data with format detection
   b. Process with automatic batch/real-time selection
   c. Track statistics and batch usage
   d. Save results with format preservation
7. Calculate overall statistics
8. Save processing summary
9. Log results and truncation stats
10. Return processing statistics
```

## Batch Processing Workflow

### End-to-End Workflow
```
1. Data Size Assessment
   ↓
2. Mode Selection (auto/batch/realtime)
   ↓
3. [If Batch] JSONL Conversion
   ↓
4. [If Batch] S3 Upload (with multipart for large files)
   ↓
5. [If Batch] Job Creation
   ↓
6. [If Batch] Job Monitoring (exponential backoff)
   ↓
7. [If Batch] Result Download
   ↓
8. [If Batch] DataFrame Reconstruction
   ↓
9. [On Failure] Automatic Fallback to Real-time
   ↓
10. Format Preservation and Output
```

### Multi-Job Processing for Large Datasets
```
For datasets > 45K records:

1. Split into chunks (45K each)
2. Create multiple batch jobs in parallel
3. Monitor all jobs concurrently
4. Download results independently
5. Merge results maintaining order
```

## Algorithms and Data Structures

### Unicode Quote Repair Algorithm
**Problem**: LLM responses contain Unicode quotes that break JSON parsing

**Solution Strategy**:
1. **Focused Repair**: Only fix quote-related issues
2. **German Quote Pattern**: Specifically handle „text" → \"text\"
3. **Normalization**: Convert remaining Unicode quotes to ASCII
4. **Preservation**: Never touch ASCII double quotes (structural JSON)

**Production Evidence**: 100% of 341 parse errors in 378K records caused by Unicode quotes

### Brace-Counting JSON Extraction
**Problem**: Extract complete JSON from responses with assistant prefilling

**Algorithm**:
```python
brace_count = 0
in_string = False
escape_next = False

for char in response[start:]:
    if escape_next:
        escape_next = False
        continue
    
    if char == "\\":
        escape_next = True
        continue
    
    if char == '"' and not escape_next:
        in_string = not in_string
        continue
    
    if not in_string:
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                return response[start:i+1]
```

**Key Features**:
- Handles escape sequences correctly
- Ignores braces inside strings
- Returns on first complete object
- Prevents false matches from nested objects

### S3 Multipart Upload Algorithm
**Problem**: Upload large JSONL files (> 100MB) to S3

**AWS Requirements**:
- Minimum part size: 5MB (except last part)
- Maximum parts: 10,000
- Maximum object size: 5TB

**Implementation**:
```python
PART_SIZE = 100MB  # Well above 5MB minimum

1. Initiate multipart upload → get upload_id
2. Split file into PART_SIZE chunks
3. For each chunk:
   a. Upload as part_number with upload_id
   b. Collect ETag from response
4. Complete upload with parts list
5. On error: Abort upload (cleanup)
```

**Error Handling**: Automatic abort to prevent orphaned parts

### Batch Job Splitting Algorithm
**Problem**: Split large datasets complying with dual AWS limits

**Constraints**:
- Record limit: 45,000 per job (conservative from 50K)
- Size limit: 900MB per job (conservative from 1GB)

**Algorithm**:
```python
while records_remaining:
    # Start with max records
    chunk = df[start:start + 45000]
    
    # Check size constraint
    jsonl_size = estimate_jsonl_size(chunk)
    
    # Iteratively reduce if too large
    while jsonl_size > 900MB and len(chunk) > 1:
        reduction_ratio = (900MB / jsonl_size) * 0.9  # 90% safety
        new_size = int(len(chunk) * reduction_ratio)
        chunk = df[start:start + new_size]
        jsonl_size = estimate_jsonl_size(chunk)
    
    chunks.append(chunk)
    start += len(chunk)
```

**Size Estimation**: Sample-based approach (first 100 records → extrapolate)

## Performance Characteristics

### Processing Mode Comparison

| Metric | Real-time | Batch | Multi-Job Batch |
|--------|-----------|-------|-----------------|
| Cost | Baseline | -50% | -50% |
| Latency | Low (minutes) | High (hours) | High (hours) |
| Max Records | Unlimited | 45,000 | Unlimited |
| Concurrent Requests | 5-10 | N/A | N/A |
| Throughput | 10-50 rec/min | 1000s/hour | 10Ks/hour |
| Best For | < 1K records | 1K-45K records | > 45K records |

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| JSONL Conversion | O(n * m) | O(n * m) | n=records, m=avg record size |
| S3 Upload | O(n) | O(1) | Streaming upload |
| Job Monitoring | O(t) | O(1) | t=job duration |
| Result Download | O(n) | O(n) | JSONL parsing |
| DataFrame Reconstruction | O(n) | O(n) | Index mapping |
| Multi-Job Split | O(n) | O(n) | Chunk creation |
| Multipart Upload | O(n/c) | O(c) | c=chunk size (100MB) |

### Cost Optimization

**Batch Processing Savings**:
- Real-time: $0.003 per 1K input tokens
- Batch: $0.0015 per 1K input tokens (50% discount)

**Example Calculation** (10K records, 1K tokens each):
```
Real-time: 10,000 * 1 * $0.003 = $30
Batch:     10,000 * 1 * $0.0015 = $15
Savings:   $15 (50%)
```

**When Batch is Cost-Effective**:
- Datasets >= 1,000 records
- Non-latency-sensitive workflows
- Batch processing overhead justified by savings

## Error Handling

### Automatic Fallback Strategy

**Trigger Conditions**:
1. Batch job creation fails
2. Job monitoring timeout exceeded
3. Result download fails
4. S3 access errors
5. Any batch-specific exception

**Fallback Workflow**:
```python
try:
    # Attempt batch processing
    result = process_batch_inference(df)
except Exception as e:
    logger.warning(f"Batch processing failed: {e}")
    logger.info("Falling back to real-time processing...")
    # Seamlessly fall back to parent class
    result = super().process_batch(df)
```

**Transparency**: User receives complete results regardless of mode

### Error Response Structure

**Real-time Processing Error**:
```python
{
    "processing_status": "error",
    "error_message": "Bedrock API error: ...",
    "llm_status": "error",
    "llm_error": "...",
    "validation_passed": False,
    # Default None for expected fields
}
```

**Batch Processing Error** (converted to same format):
```python
{
    "processing_status": "error",
    "error_message": "Batch inference failed: ...",
    "llm_status": "error",
    "llm_error": "...",
    "validation_passed": False,
}
```

## Best Practices

### For Production Deployments

1. **Use Auto Mode**: Let system decide batch vs real-time
2. **Configure IAM Role**: Set `BEDROCK_BATCH_ROLE_ARN` for batch capability
3. **Monitor Costs**: Track batch_processing_used in summaries
4. **Set Truncation Limits**: Configure `BEDROCK_MAX_INPUT_FIELD_LENGTH`
5. **Enable Logging**: Use `BEDROCK_LOG_TRUNCATIONS=true`

### For Development

1. **Use Real-time Mode**: Set `BEDROCK_BATCH_MODE=realtime` for faster iteration
2. **Small Test Batches**: Test with < 100 records
3. **Validate Templates**: Ensure templates load correctly
4. **Check Schemas**: Verify Pydantic model creation

### For Cost Optimization

1. **Batch for Large Datasets**: Always use batch for > 1K records
2. **Adjust Threshold**: Lower `BEDROCK_BATCH_THRESHOLD` if cost-sensitive
3. **Monitor Job Failures**: Track fallback rate in summaries
4. **Use Inference Profiles**: Leverage capacity reservations

### For Large Datasets (> 45K)

1. **Trust Multi-Job Processing**: System automatically splits
2. **Monitor Job Status**: Check AWS Bedrock console
3. **Configure Timeouts**: Increase `BEDROCK_BATCH_TIMEOUT_HOURS` if needed
4. **Parallel Job Monitoring**: System handles concurrency

## Example Configurations

### Production Configuration (Auto Mode)
```bash
export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
export BEDROCK_FALLBACK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
export BEDROCK_BATCH_MODE="auto"
export BEDROCK_BATCH_THRESHOLD="1000"
export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
export BEDROCK_TRUNCATION_ENABLED="true"
export BEDROCK_MAX_INPUT_FIELD_LENGTH="400000"
```

### Cost-Optimized Configuration (Force Batch)
```bash
export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
export BEDROCK_BATCH_MODE="batch"
export BEDROCK_BATCH_THRESHOLD="500"  # Lower threshold
export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
```

### Development Configuration (Real-time Only)
```bash
export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
export BEDROCK_BATCH_MODE="realtime"
export BEDROCK_MAX_TOKENS="4096"  # Lower for faster testing
```

### Large Dataset Configuration (Multi-Job)
```bash
export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
export BEDROCK_BATCH_MODE="auto"
export BEDROCK_BATCH_THRESHOLD="10000"
export BEDROCK_MAX_RECORDS_PER_JOB="45000"
export BEDROCK_MAX_CONCURRENT_BATCH_JOBS="20"
export BEDROCK_BATCH_TIMEOUT_HOURS="48"  # Extended for large jobs
export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
```

## Integration Patterns

### Upstream Integration (Template Generation)
```
BedrockPromptTemplateGeneration
   ↓ (outputs: prompts.json, validation_schema.json)
BedrockBatchProcessing
   ↓ (outputs: processed_data, summary)
```

### Downstream Integration (Training)
```
BedrockBatchProcessing
   ↓ (train/val/test splits with LLM labels)
PyTorchTraining / XGBoostTraining
   ↓ (trained model)
```

### Comparison with bedrock_processing.py

| Feature | bedrock_processing.py | bedrock_batch_processing.py |
|---------|----------------------|----------------------------|
| Real-time Processing | ✓ | ✓ (identical) |
| Batch Processing | ✗ | ✓ |
| Cost Savings | 0% | Up to 50% |
| Auto Mode Selection | ✗ | ✓ |
| Multi-Job Support | ✗ | ✓ |
| S3 Integration | ✗ | ✓ |
| Automatic Fallback | ✗ | ✓ |
| Output Compatibility | N/A | 100% |
| Interface Compatibility | N/A | 100% |

## Troubleshooting

### Batch Job Failures

**Symptom**: Batch job status = "Failed"

**Common Causes**:
1. **IAM Permissions**: Role lacks Bedrock/S3 permissions
2. **Input Format**: JSONL format errors
3. **Model Access**: Model not available in region
4. **Quota Limits**: Exceeded batch job quota

**Solution**: Check CloudWatch logs, verify IAM role, confirm model access

### S3 Upload Errors

**Symptom**: "Failed to upload JSONL to S3"

**Common Causes**:
1. **Bucket Access**: No PutObject permission
2. **Bucket Region**: Bucket in different region
3. **Large Files**: Multipart upload failure

**Solution**: Verify bucket permissions, check region match, review multipart logs

### Fallback to Real-time

**Symptom**: "Using real-time processing" despite batch configuration

**Common Causes**:
1. **Missing Role**: `BEDROCK_BATCH_ROLE_ARN` not set
2. **Below Threshold**: Dataset < `BEDROCK_BATCH_THRESHOLD`
3. **S3 Path Missing**: Framework S3 paths not configured

**Solution**: Check environment variables, verify data size, confirm step builder configuration

### Pydantic Validation Errors

**Symptom**: High validation_failed rate

**Common Causes**:
1. **Schema Mismatch**: Response doesn't match schema
2. **Missing Fields**: Required fields not in response
3. **Type Errors**: Field values wrong type

**Solution**: Review validation schema, check LLM prompts, adjust schema

## References

### Related Scripts
- [`bedrock_processing.py`](bedrock_processing_script.md): Real-time processing equivalent with identical interface
- [`bedrock_prompt_template_generation.py`](bedrock_prompt_template_generation_script.md): Upstream template generation step (TBD)
- [`pseudo_label_merge.py`](pseudo_label_merge_script.md): Merges batch-processed labels with labeled data (TBD)

### Related Documentation
- **Contract**: [`src/cursus/steps/contracts/bedrock_batch_processing_contract.py`](../../src/cursus/steps/contracts/bedrock_batch_processing_contract.py)

### Related Design Documents
- **[Bedrock Batch Processing Step Builder Patterns](../1_design/bedrock_batch_processing_step_builder_patterns.md)**: Batch processing architecture, cost optimization strategies, and automatic fallback design
- **[Bedrock Processing Step Builder Patterns](../1_design/bedrock_processing_step_builder_patterns.md)**: Real-time processing architecture and template-driven patterns (shared foundation)
- **[Bedrock Prompt Template Generation Step Patterns](../1_design/bedrock_prompt_template_generation_step_patterns.md)**: Upstream template generation design and integration patterns

### External Resources
- [AWS Bedrock Batch Inference Documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html): Official batch inference guide and best practices
- [AWS Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/): Complete API documentation for Bedrock services
- [Amazon S3 Multipart Upload](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html): Multipart upload guide for large files
- [Pydantic Documentation](https://docs.pydantic.dev/latest/): Data validation using Python type annotations
- [Tenacity Retry Library](https://tenacity.readthedocs.io/): Retry logic with exponential backoff

### Framework References
- **[Script Development Guide](../0_developer_guide/script_development_guide.md)**: Cursus framework patterns for script development
- **[Step Builder Guide](../0_developer_guide/step_builder.md)**: Step builder patterns including S3 path management
