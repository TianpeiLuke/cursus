"""
Bedrock Processing Script Contract

Defines the contract for the Bedrock processing script that processes input data
through AWS Bedrock models using generated prompt templates and validation schemas
from the Bedrock Prompt Template Generation step.
"""

from ...core.base.contract_base import ScriptContract

BEDROCK_PROCESSING_CONTRACT = ScriptContract(
    entry_point="bedrock_processing.py",
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
        "BEDROCK_FALLBACK_MODEL_ID": "",
        "BEDROCK_INFERENCE_PROFILE_ARN": "",
        "BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS": "[]",
        "AWS_DEFAULT_REGION": "us-east-1",
        "BEDROCK_MAX_TOKENS": "8192",
        "BEDROCK_TEMPERATURE": "1.0",
        "BEDROCK_TOP_P": "0.999",
        "BEDROCK_BATCH_SIZE": "10",
        "BEDROCK_MAX_RETRIES": "3",
        "BEDROCK_OUTPUT_COLUMN_PREFIX": "llm_",
        "BEDROCK_SKIP_ERROR_RECORDS": "false",
        "BEDROCK_MAX_CONCURRENT_WORKERS": "5",
        "BEDROCK_RATE_LIMIT_PER_SECOND": "10",
        "BEDROCK_CONCURRENCY_MODE": "sequential",
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
    Bedrock processing script that processes input data through AWS Bedrock models
    using generated prompt templates and validation schemas from the Bedrock Prompt
    Template Generation step. Supports template-driven response processing with
    dynamic Pydantic model creation and both sequential and concurrent processing modes.
    
    The script integrates seamlessly with the Bedrock Prompt Template Generation step
    to provide a complete template-driven LLM processing pipeline with:
    1. Template-driven prompt formatting using input placeholders
    2. Dynamic Pydantic model creation from validation schemas
    3. Intelligent inference profile management with fallback support
    4. Configurable concurrent processing with rate limiting
    5. Comprehensive error handling and retry logic
    6. Job type-aware processing with train/val/test split preservation
    
    Job Type Handling:
    The script adapts its behavior based on the job_type argument (passed by the step builder):
    
    Training Job Type (job_type="training"):
    - Expects train/val/test subdirectory structure from TabularPreprocessing
    - Processes each split separately while preserving directory structure
    - Output maintains train/val/test organization for PyTorch training compatibility
    - Fallback: If no subdirectories found, processes as single dataset
    
    Non-Training Job Types (job_type="validation", "testing", "calibration"):
    - Expects single dataset from TabularPreprocessing
    - Processes all files in input directory
    - Output includes job_type in filename for identification
    
    Input Structure:
    - /opt/ml/processing/input/data: Input data files (required)
      - Training job type: Expects train/, val/, test/ subdirectories with data files
      - Non-training job types: Expects data files directly in input directory
      - Supports CSV (.csv) and Parquet (.parquet) formats
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
    
    Output Structure:
    - /opt/ml/processing/output/data: Processed data with Bedrock responses
      - processed_{filename}_{timestamp}.parquet: Efficient binary format
      - processed_{filename}_{timestamp}.csv: Human-readable format
      - batch_{batch_num:04d}_results.parquet: Intermediate batch results
      - combined_results_{timestamp}.parquet: Combined results (multiple input files)
    - /opt/ml/processing/output/summary: Processing statistics and metadata
      - processing_summary_{timestamp}.json: Comprehensive processing statistics
        * Success rates, validation rates, model information
        * File-by-file processing details and performance metrics
        * Template integration status and Pydantic model creation results
    
    Environment Variables:
    - BEDROCK_PRIMARY_MODEL_ID: Primary Bedrock model ID (required)
      - Example: "anthropic.claude-sonnet-4-20250514-v1:0"
    - BEDROCK_FALLBACK_MODEL_ID: Fallback model for inference profile failures (optional)
      - Example: "anthropic.claude-3-5-sonnet-20241022-v2:0"
      - Essential for production reliability when using inference profiles
    - BEDROCK_INFERENCE_PROFILE_ARN: Inference profile ARN for capacity management (optional)
      - Example: "arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"
    - BEDROCK_INFERENCE_PROFILE_REQUIRED_MODELS: JSON array of models requiring inference profiles (optional)
      - Example: '["anthropic.claude-sonnet-4-20250514-v1:0"]'
    - AWS_DEFAULT_REGION: AWS region for Bedrock API (optional, default: "us-east-1")
    - BEDROCK_MAX_TOKENS: Maximum tokens for Bedrock responses (optional, default: "8192")
    - BEDROCK_TEMPERATURE: Temperature for response generation (optional, default: "1.0")
    - BEDROCK_TOP_P: Top-p sampling parameter (optional, default: "0.999")
    - BEDROCK_BATCH_SIZE: Number of records per processing batch (optional, default: "10")
    - BEDROCK_MAX_RETRIES: Maximum retries for failed requests (optional, default: "3")
    - BEDROCK_OUTPUT_COLUMN_PREFIX: Prefix for output columns (optional, default: "llm_")
    
    Concurrency Configuration:
    - BEDROCK_CONCURRENCY_MODE: Processing mode (optional, default: "sequential")
      - "sequential": Single-threaded processing (safer, easier debugging)
      - "concurrent": Multi-threaded processing (faster, 3-10x speedup)
    - BEDROCK_MAX_CONCURRENT_WORKERS: Number of concurrent threads (optional, default: "5")
      - Recommended range: 3-10 workers depending on rate limits
    - BEDROCK_RATE_LIMIT_PER_SECOND: API requests per second limit (optional, default: "10")
      - Enforced only in concurrent mode to respect Bedrock API limits
    
    Template Integration Features:
    - Zero-Configuration Processing: Templates provide all prompt configuration
    - Input Placeholder Mapping: Automatic DataFrame column to template placeholder mapping
    - Dynamic Pydantic Models: Runtime model creation from validation schemas
    - Validation Schema Integration: Structured response parsing with validation
    - Fallback Mechanisms: Graceful degradation when templates or schemas unavailable
    
    Processing Modes:
    - Sequential Processing: Original single-threaded mode for compatibility
    - Concurrent Processing: Multi-threaded with ThreadPoolExecutor
      - Thread-local Bedrock clients for optimal performance
      - Configurable rate limiting to respect API limits
      - Semaphore-based concurrency control
      - Individual thread failure isolation
    
    Inference Profile Management:
    - Automatic Profile Detection: Auto-configures known models requiring profiles
    - Intelligent Fallback: Falls back to on-demand models on ValidationException
    - Profile ARN Support: Direct ARN specification for custom profiles
    - Global Profile IDs: Support for global inference profile identifiers
    
    Error Handling and Resilience:
    - Exponential Backoff: Retry logic with exponential backoff using tenacity
    - Structured Error Responses: Consistent error format with metadata
    - Batch Isolation: Individual record failures don't stop batch processing
    - Intermediate Saves: Batch results saved for recovery and monitoring
    
    Response Processing:
    - Pydantic Validation: Dynamic model creation from validation schemas
    - JSON Fallback: Graceful fallback when Pydantic validation fails
    - Parse Status Tracking: Detailed parsing and validation status for each response
    - Evidence Validation: Key evidence alignment with category conditions
    
    Performance Optimizations:
    - Batch Processing: Configurable batch sizes for memory efficiency
    - Concurrent Execution: ThreadPoolExecutor for parallel processing
    - Rate Limiting: Intelligent rate limiting to maximize throughput
    - Intermediate Caching: Batch-level result caching for recovery
    
    Quality Assurance:
    - Comprehensive Logging: Detailed processing logs with performance metrics
    - Validation Tracking: Success rates and validation pass rates
    - Model Information: Effective model IDs and inference profile details
    - Template Integration Status: Template loading and Pydantic model creation status
    
    Usage Examples:
    
    Sequential Processing (Default):
    ```bash
    export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
    export BEDROCK_CONCURRENCY_MODE="sequential"
    ```
    
    Concurrent Processing (High Performance):
    ```bash
    export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
    export BEDROCK_FALLBACK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
    export BEDROCK_CONCURRENCY_MODE="concurrent"
    export BEDROCK_MAX_CONCURRENT_WORKERS="10"
    export BEDROCK_RATE_LIMIT_PER_SECOND="15"
    ```
    
    Inference Profile Configuration:
    ```bash
    export BEDROCK_PRIMARY_MODEL_ID="anthropic.claude-sonnet-4-20250514-v1:0"
    export BEDROCK_INFERENCE_PROFILE_ARN="arn:aws:bedrock:us-east-1:123456789012:inference-profile/abc123"
    export BEDROCK_FALLBACK_MODEL_ID="anthropic.claude-3-5-sonnet-20241022-v2:0"
    ```
    
    Integration with Template Generation Step:
    1. Bedrock Prompt Template Generation step creates templates and schemas
    2. Bedrock Processing step loads templates and schemas automatically
    3. Input data processed using template placeholders and validated using schemas
    4. Results include original data plus Bedrock responses with validation status
    
    Output Data Structure:
    - Original input columns preserved
    - Bedrock response fields with configurable prefix (default: "llm_")
    - Processing metadata: status, validation_passed, parse_status
    - Error information: error messages for failed processing
    - Model information: effective model ID and inference profile details
    
    The script is production-ready with comprehensive error handling, performance
    optimizations, and integration capabilities for enterprise LLM processing pipelines.
    """,
)
