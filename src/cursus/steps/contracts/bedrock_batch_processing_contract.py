"""
Bedrock Batch Processing Script Contract

Defines the contract for the Bedrock batch processing script that processes input data
through AWS Bedrock models using batch inference capabilities with automatic fallback
to real-time processing. Integrates with generated prompt templates and validation schemas
from the Bedrock Prompt Template Generation step.
"""

from ...core.base.contract_base import ScriptContract

BEDROCK_BATCH_PROCESSING_CONTRACT = ScriptContract(
    entry_point="bedrock_batch_processing.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "prompt_templates": "/opt/ml/processing/input/templates",
        "validation_schema": "/opt/ml/processing/input/schema",
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/data",
        "analysis_summary": "/opt/ml/processing/output/summary",
    },
    expected_arguments={
        "batch-size": "batch size for processing (default: 10)",
        "max-retries": "maximum retries for Bedrock calls (default: 3)",
    },
    required_env_vars=["BEDROCK_PRIMARY_MODEL_ID"],
    optional_env_vars={
        # Standard Bedrock configuration (inherited from bedrock_processing.py)
        "BEDROCK_FALLBACK_MODEL_ID": "",
        "BEDROCK_INFERENCE_PROFILE_ARN": "",
        "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": "[]",
        "AWS_DEFAULT_REGION": "us-east-1",
        "BEDROCK_MAX_TOKENS": "32768",
        "BEDROCK_TEMPERATURE": "1.0",
        "BEDROCK_TOP_P": "0.999",
        "BEDROCK_BATCH_SIZE": "10",
        "BEDROCK_MAX_RETRIES": "3",
        "BEDROCK_OUTPUT_COLUMN_PREFIX": "llm_",
        "BEDROCK_SKIP_ERROR_RECORDS": "false",
        "BEDROCK_MAX_CONCURRENT_WORKERS": "5",
        "BEDROCK_RATE_LIMIT_PER_SECOND": "10",
        "BEDROCK_CONCURRENCY_MODE": "sequential",
        # Batch-specific configuration
        "BEDROCK_BATCH_MODE": "auto",
        "BEDROCK_BATCH_THRESHOLD": "1000",
        "BEDROCK_BATCH_ROLE_ARN": "",
        "BEDROCK_BATCH_INPUT_S3_PATH": "",
        "BEDROCK_BATCH_OUTPUT_S3_PATH": "",
        "BEDROCK_BATCH_TIMEOUT_HOURS": "24",
        # AWS Bedrock batch limits (configurable)
        "BEDROCK_MAX_RECORDS_PER_JOB": "45000",
        "BEDROCK_MAX_CONCURRENT_BATCH_JOBS": "20",
        # Input truncation configuration
        "BEDROCK_MAX_INPUT_FIELD_LENGTH": "400000",
        "BEDROCK_TRUNCATION_ENABLED": "true",
        "BEDROCK_LOG_TRUNCATIONS": "true",
        "USE_SECURE_PYPI": "false",
    },
    framework_requirements={
        "pandas": ">=1.2.0",
        "boto3": ">=1.26.0",
        "pydantic": ">=2.0.0",
        "tenacity": ">=8.0.0",
        "pathlib": ">=1.0.0",
    },
    description="""
    Bedrock batch processing script that provides AWS Bedrock batch inference capabilities
    while maintaining identical input/output interface to bedrock_processing.py. Offers
    cost-efficient batch processing for large datasets with automatic fallback to real-time
    processing when batch processing is not suitable or fails.
    
    The script integrates seamlessly with the Bedrock Prompt Template Generation step
    to provide a complete template-driven LLM processing pipeline with:
    1. Template-driven prompt formatting using input placeholders (identical to bedrock_processing.py)
    2. Dynamic Pydantic model creation from validation schemas
    3. Intelligent batch vs real-time processing selection
    4. AWS Bedrock batch inference job management
    5. Cursus framework-compliant S3 path management
    6. Comprehensive error handling with automatic fallback
    7. Job type-aware processing with train/val/test split preservation
    
    Key Features:
    
    Batch Processing Capabilities:
    - AWS Bedrock batch inference for cost-efficient processing of large datasets
    - Automatic JSONL conversion using existing template logic
    - S3-based input/output management using cursus framework patterns
    - Batch job creation, monitoring, and result retrieval
    - Intelligent fallback to real-time processing on batch failures
    
    Processing Mode Selection:
    - Auto Mode (default): Automatically selects batch vs real-time based on data size
    - Batch Mode: Forces batch processing regardless of data size
    - Real-time Mode: Uses sequential processing identical to bedrock_processing.py
    - Configurable threshold for automatic batch processing (default: 1000 records)
    
    Cost Optimization:
    - Up to 50% cost reduction for large datasets through AWS Bedrock batch pricing
    - Intelligent processing mode selection to minimize costs
    - Batch processing for datasets >= 1000 records (configurable)
    - Real-time processing for smaller datasets to minimize latency
    
    Framework Integration:
    - Uses cursus framework patterns for S3 path management
    - Leverages _get_base_output_path() and Join() from step builders
    - Compatible with PIPELINE_EXECUTION_TEMP_DIR parameter
    - Environment variables set automatically by step builder
    
    Job Type Handling (Identical to bedrock_processing.py):
    The script adapts its behavior based on the job_type argument:
    
    Training Job Type (job_type="training"):
    - Expects train/val/test subdirectory structure from TabularPreprocessing
    - Processes each split separately while preserving directory structure
    - Automatic batch processing selection per split based on data size
    - Output maintains train/val/test organization for PyTorch training compatibility
    - Fallback: If no subdirectories found, processes as single dataset
    
    Non-Training Job Types (job_type="validation", "testing", "calibration"):
    - Expects single dataset from TabularPreprocessing
    - Processes all files in input directory with batch processing capability
    - Output includes job_type in filename for identification
    
    Input Structure (Identical to bedrock_processing.py):
    - /opt/ml/processing/input/data: Input data files (required)
      - Training job type: Expects train/, val/, test/ subdirectories with data files
      - Non-training job types: Expects data files directly in input directory
      - Supports CSV (.csv), Parquet (.parquet), and compressed versions (.csv.gz, .parquet.gz)
      - Data columns must match template input placeholders
    - /opt/ml/processing/input/templates: Prompt templates from Template Generation step (required)
      - /opt/ml/processing/input/templates/prompts.json: Main template file
        * system_prompt: System prompt for Bedrock API
        * user_prompt_template: User prompt template with placeholders
        * input_placeholders: List of placeholder field names for DataFrame column mapping
    - /opt/ml/processing/input/schema: Validation schemas from Template Generation step (required)
      - /opt/ml/processing/input/schema/validation_schema_*.json: JSON schema for response validation
        * properties: Field definitions for dynamic Pydantic model creation
        * required: Required field specifications
        * processing_config: Response processing metadata
    
    Output Structure (Identical to bedrock_processing.py):
    - /opt/ml/processing/output/data: Processed data with Bedrock responses
      - processed_{filename}_{timestamp}.parquet: Efficient binary format
      - processed_{filename}_{timestamp}.csv: Human-readable format
      - batch_{batch_num:04d}_results.parquet: Intermediate batch results (real-time mode)
      - Split-specific outputs for training job type (train/, val/, test/ subdirectories)
    - /opt/ml/processing/output/summary: Processing statistics and metadata
      - processing_summary_{job_type}_{timestamp}.json: Comprehensive processing statistics
        * Success rates, validation rates, model information
        * File-by-file processing details and performance metrics
        * Template integration status and Pydantic model creation results
        * Batch processing usage statistics and job information
    
    Environment Variables:
    
    Standard Bedrock Configuration (Inherited from bedrock_processing.py):
    - BEDROCK_PRIMARY_MODEL_ID: Primary Bedrock model ID (required)
      - Example: "anthropic.claude-sonnet-4-20250514-v1:0"
    - BEDROCK_FALLBACK_MODEL_ID: Fallback model for inference profile failures (optional)
      - Example: "anthropic.claude-3-5-sonnet-20241022-v2:0"
    - BEDROCK_INFERENCE_PROFILE_ARN: Inference profile ARN for capacity management (optional)
      - Example: "arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"
    - BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS: JSON array of models requiring inference profiles (optional)
      - Example: '["anthropic.claude-sonnet-4-20250514-v1:0"]'
    - AWS_DEFAULT_REGION: AWS region for Bedrock API (optional, default: "us-east-1")
    - BEDROCK_MAX_TOKENS: Maximum tokens for Bedrock responses (optional, default: "32768")
      - Optimized for Claude 4 (50% of 64K maximum for reliability)
    - BEDROCK_TEMPERATURE: Temperature for response generation (optional, default: "1.0")
    - BEDROCK_TOP_P: Top-p sampling parameter (optional, default: "0.999")
    - BEDROCK_BATCH_SIZE: Number of records per processing batch (optional, default: "10")
    - BEDROCK_MAX_RETRIES: Maximum retries for failed requests (optional, default: "3")
    - BEDROCK_OUTPUT_COLUMN_PREFIX: Prefix for output columns (optional, default: "llm_")
    - BEDROCK_MAX_CONCURRENT_WORKERS: Number of concurrent threads (optional, default: "5")
    - BEDROCK_RATE_LIMIT_PER_SECOND: API requests per second limit (optional, default: "10")
    - BEDROCK_CONCURRENCY_MODE: Processing mode (optional, default: "sequential")
    
    Batch-Specific Configuration:
    - BEDROCK_BATCH_MODE: Batch processing mode (optional, default: "auto")
      - "auto": Automatically selects batch vs real-time based on data size and configuration
      - "batch": Forces batch processing regardless of data size
      - "realtime": Forces real-time processing identical to bedrock_processing.py
    - BEDROCK_BATCH_THRESHOLD: Minimum records for automatic batch processing (optional, default: "1000")
      - Datasets with >= threshold records will use batch processing in auto mode
    - BEDROCK_BATCH_ROLE_ARN: IAM role ARN for batch inference jobs (required for batch processing)
      - Example: "arn:aws:iam::123456789012:role/BedrockBatchRole"
      - Must have permissions for Bedrock batch inference and S3 access
    - BEDROCK_BATCH_INPUT_S3_PATH: S3 path for batch input data (set by step builder)
      - Example: "s3://pipeline-bucket/execution-123/bedrock-batch/input"
      - Uses cursus framework patterns via _get_base_output_path() and Join()
    - BEDROCK_BATCH_OUTPUT_S3_PATH: S3 path for batch output data (set by step builder)
      - Example: "s3://pipeline-bucket/execution-123/bedrock-batch/output"
      - Uses cursus framework patterns via _get_base_output_path() and Join()
    - BEDROCK_BATCH_TIMEOUT_HOURS: Maximum hours for batch job completion (optional, default: "24")
      - Batch jobs exceeding this timeout will be terminated
    
    Batch Processing Workflow:
    1. Data Size Assessment: Evaluates dataset size against threshold
    2. Mode Selection: Chooses batch vs real-time based on configuration
    3. JSONL Conversion: Converts DataFrame to Bedrock batch format using template logic
    4. S3 Upload: Uploads JSONL data to framework-provided S3 input path
    5. Job Creation: Creates Bedrock batch inference job with specified model and role
    6. Job Monitoring: Monitors job status with exponential backoff
    7. Result Download: Downloads and parses batch results from S3
    8. DataFrame Reconstruction: Converts results back to DataFrame format
    9. Fallback Handling: Falls back to real-time processing on any batch failure
    
    Template Integration Features (Identical to bedrock_processing.py):
    - Zero-Configuration Processing: Templates provide all prompt configuration
    - Input Placeholder Mapping: Automatic DataFrame column to template placeholder mapping
    - Dynamic Pydantic Models: Runtime model creation from validation schemas
    - Validation Schema Integration: Structured response parsing with validation
    - Fallback Mechanisms: Graceful degradation when templates or schemas unavailable
    
    Processing Modes:
    - Sequential Processing: Original single-threaded mode for compatibility (real-time fallback)
    - Batch Processing: AWS Bedrock batch inference for cost-efficient large dataset processing
    - Automatic Selection: Intelligent mode selection based on data size and configuration
    
    Inference Profile Management (Identical to bedrock_processing.py):
    - Automatic Profile Detection: Auto-configures known models requiring profiles
    - Intelligent Fallback: Falls back to on-demand models on ValidationException
    - Profile ARN Support: Direct ARN specification for custom profiles
    - Global Profile IDs: Support for global inference profile identifiers
    
    Error Handling and Resilience:
    - Exponential Backoff: Retry logic with exponential backoff using tenacity
    - Structured Error Responses: Consistent error format with metadata
    - Batch Job Monitoring: Comprehensive job status tracking with timeout handling
    - Automatic Fallback: Seamless fallback to real-time processing on batch failures
    - Intermediate Saves: Batch results saved for recovery and monitoring
    
    Response Processing (Identical to bedrock_processing.py):
    - Pydantic Validation: Dynamic model creation from validation schemas
    - JSON Fallback: Graceful fallback when Pydantic validation fails
    - Parse Status Tracking: Detailed parsing and validation status for each response
    - Evidence Validation: Key evidence alignment with category conditions
    
    Performance Optimizations:
    - Batch Processing: Up to 50% cost reduction for large datasets
    - Intelligent Mode Selection: Automatic optimization based on data characteristics
    - S3 Integration: Efficient data transfer using cursus framework patterns
    - Job Monitoring: Optimized polling with exponential backoff
    - Memory Efficiency: Streaming JSONL processing for large datasets
    
    Quality Assurance:
    - Comprehensive Logging: Detailed processing logs with batch job information
    - Validation Tracking: Success rates and validation pass rates
    - Model Information: Effective model IDs and inference profile details
    - Template Integration Status: Template loading and Pydantic model creation status
    - Batch Processing Statistics: Job information and processing mode usage
    
    Usage Examples:
    
    Auto Mode (Recommended):
    ```bash
    export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
    export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
    export BEDROCK_BATCH_MODE="auto"
    export BEDROCK_BATCH_THRESHOLD="1000"
    ```
    
    Forced Batch Processing:
    ```bash
    export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
    export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
    export BEDROCK_BATCH_MODE="batch"
    ```
    
    Real-time Processing (Identical to bedrock_processing.py):
    ```bash
    export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
    export BEDROCK_BATCH_MODE="realtime"
    ```
    
    Inference Profile with Batch Processing:
    ```bash
    export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
    export BEDROCK_INFERENCE_PROFILE_ARN="arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"
    export BEDROCK_FALLBACK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
    export BEDROCK_BATCH_ROLE_ARN="arn:aws:iam::123456789012:role/BedrockBatchRole"
    ```
    
    Integration with Template Generation Step:
    1. Bedrock Prompt Template Generation step creates templates and schemas
    2. Bedrock Batch Processing step loads templates and schemas automatically
    3. Input data processed using template placeholders and validated using schemas
    4. Results include original data plus Bedrock responses with validation status
    5. Batch processing automatically selected for large datasets (>= 1000 records)
    
    Output Data Structure (Identical to bedrock_processing.py):
    - Original input columns preserved
    - Bedrock response fields with configurable prefix (default: "llm_")
    - Processing metadata: status, validation_passed, parse_status
    - Error information: error messages for failed processing
    - Model information: effective model ID and inference profile details
    - Batch processing metadata: processing mode used, job information
    
    The script is production-ready with comprehensive error handling, performance
    optimizations, cost efficiency, and integration capabilities for enterprise
    LLM processing pipelines. It provides a drop-in replacement for bedrock_processing.py
    with significant cost savings for large datasets while maintaining identical
    functionality and output formats.
    """,
)
