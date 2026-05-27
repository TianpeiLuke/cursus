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
        # Input truncation configuration
        "BEDROCK_MAX_INPUT_FIELD_LENGTH": "400000",
        "BEDROCK_TRUNCATION_ENABLED": "true",
        "BEDROCK_LOG_TRUNCATIONS": "true",
        "USE_SECURE_PYPI": "false",
        # Config-embedded template support (self-contained mode)
        "BEDROCK_USER_PROMPT_TEMPLATE": "",
        "BEDROCK_SYSTEM_PROMPT": "",
        "BEDROCK_INPUT_PLACEHOLDERS": "[]",
        "BEDROCK_VALIDATION_SCHEMA": "{}",
        # Structured output mode
        "BEDROCK_USE_STRUCTURED_OUTPUT": "false",
        # Converse API mode
        "BEDROCK_USE_CONVERSE_API": "false",
        # Adaptive rate limiting
        "BEDROCK_ADAPTIVE_RATE_LIMITING": "false",
    },
    framework_requirements={
        "pandas": ">=1.2.0",
        "boto3": ">=1.26.0",
        "pydantic": ">=2.0.0",
        "tenacity": ">=8.0.0",
        "pathlib": ">=1.0.0",
    },
    description="""
    Bedrock processing script with three invocation modes and production robustness.

    Invocation Modes:
    - invoke_model (default): Anthropic-format with assistant prefilling for JSON output
    - Structured output (tool_use): Guaranteed schema compliance, 0% parse failures
    - Converse API: Model-agnostic, supports Nova/Llama/Mistral without format changes

    Self-Contained Mode:
    Prompt templates and validation schemas can be embedded directly in config via
    environment variables (BEDROCK_USER_PROMPT_TEMPLATE, BEDROCK_VALIDATION_SCHEMA),
    eliminating the need for an upstream BedrockPromptTemplateGeneration step.
    Fallback chain: upstream step output > env var config > error.

    Robustness Features:
    - Circuit breaker: Trips after N consecutive failures, blocks for recovery period
    - Adaptive rate limiting: Auto-tunes req/s based on observed throttle rate
    - Checkpoint/resume: Saves progress per batch, resumes on restart
    - Retry with exponential backoff (tenacity)
    - Inference profile fallback to on-demand model on ValidationException

    Processing Modes:
    - Sequential: Single-threaded, safer for debugging
    - Concurrent: Multi-threaded with configurable workers and rate limiting

    Job Type Handling:
    - training: Processes train/val/test subdirectories, preserves structure
    - Other: Processes single dataset from input directory

    Input:
    - /opt/ml/processing/input/data: CSV/Parquet data (required)
    - /opt/ml/processing/input/templates: prompts.json (optional with self-contained fallback)
    - /opt/ml/processing/input/schema: validation_schema_*.json (optional with fallback)

    Output:
    - /opt/ml/processing/output/data: Original data + llm_* prefixed response columns
    - /opt/ml/processing/output/summary: processing_summary_*.json with metrics

    Key Environment Variables:
    - BEDROCK_PRIMARY_MODEL_ID (required): Model ID for inference
    - BEDROCK_USE_STRUCTURED_OUTPUT: Enable tool_use for schema-enforced output
    - BEDROCK_USE_CONVERSE_API: Enable model-agnostic Converse API
    - BEDROCK_ADAPTIVE_RATE_LIMITING: Enable auto-tuning of request rate
    - BEDROCK_CONCURRENCY_MODE: 'sequential' or 'concurrent'
    - BEDROCK_USER_PROMPT_TEMPLATE: Config-embedded prompt (self-contained mode)
    - BEDROCK_VALIDATION_SCHEMA: Config-embedded JSON schema (self-contained mode)
    """,
)
