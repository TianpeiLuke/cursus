"""
Temporal Sequence Normalization Script Contract

Defines the contract for the temporal sequence normalization script that handles
temporal sequence data loading, validation, normalization, and padding/truncation
for machine learning models.
"""

from ...core.base.contract_base import ScriptContract

TEMPORAL_SEQUENCE_NORMALIZATION_CONTRACT = ScriptContract(
    entry_point="temporal_sequence_normalization.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "SIGNATURE": "/opt/ml/processing/input/signature",
    },
    expected_output_paths={"normalized_sequences": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=[
        "SEQUENCE_LENGTH",
        "SEQUENCE_SEPARATOR",
        "TEMPORAL_FIELD",
        "SEQUENCE_GROUPING_FIELD",
        "RECORD_ID_FIELD",
    ],
    optional_env_vars={
        "MISSING_INDICATORS": '["", "My Text String", null]',
        "TIME_DELTA_MAX_SECONDS": "10000000",
        "PADDING_STRATEGY": "pre",
        "TRUNCATION_STRATEGY": "post",
        "ENABLE_MULTI_SEQUENCE": "false",
        "SECONDARY_ENTITY_FIELD": "creditCardId",
        "SEQUENCE_NAMING_PATTERN": "*_seq_by_{entity}.*",
        "ENABLE_DISTRIBUTED_PROCESSING": "false",
        "CHUNK_SIZE": "10000",
        "MAX_WORKERS": "auto",
        "VALIDATION_STRATEGY": "strict",
        "OUTPUT_FORMAT": "numpy",
        "INCLUDE_ATTENTION_MASKS": "true",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Temporal sequence normalization script that:
    1. Loads temporal sequence data from multiple formats (CSV, TSV, JSON, Parquet)
    2. Applies sequence ordering by temporal field
    3. Validates sequence data integrity with configurable strategies
    4. Handles missing values in sequences
    5. Computes time deltas for temporal relationships
    6. Normalizes sequence lengths through padding/truncation
    7. Outputs normalized sequences with optional attention masks
    
    Contract aligned with actual script implementation:
    - Inputs: 
      * DATA (required) - reads temporal sequence data from /opt/ml/processing/input/data
      * SIGNATURE (optional) - reads column signature from /opt/ml/processing/input/signature
    - Outputs: normalized_sequences (primary) - writes to /opt/ml/processing/output
    - Arguments: job-type (required) - defines processing mode for different data splits
    
    Script Implementation Details:
    - Supports multi-format data loading (CSV, TSV, JSON, Parquet) including compressed files
    - Automatic sequence field detection based on configurable naming patterns
    - Five core processing operations:
      * SequenceOrderingOperation - temporal sorting and duplicate handling
      * DataValidationOperation - configurable strict/lenient validation
      * MissingValueHandlingOperation - configurable missing value strategies
      * TimeDeltaComputationOperation - relative time delta computation with capping
      * SequencePaddingOperation - sequence length normalization with attention masks
    - Multi-sequence support for dual-entity processing (e.g., customer + credit card)
    - Configurable output formats (numpy, parquet, csv)
    - Memory-efficient processing with optional chunking for large datasets
    
    Environment Variable Details:
    - SEQUENCE_LENGTH: Target sequence length for padding/truncation (default: 51)
    - SEQUENCE_SEPARATOR: Separator for sequence values within fields (default: "~")
    - TEMPORAL_FIELD: Field name containing timestamps (default: "orderDate")
    - SEQUENCE_GROUPING_FIELD: Field name used to group records into temporal sequences (default: "customerId")
    - RECORD_ID_FIELD: Field name that uniquely identifies individual records (default: "objectId")
    - MISSING_INDICATORS: JSON array of missing value indicators
    - TIME_DELTA_MAX_SECONDS: Maximum time delta cap in seconds (default: 10000000)
    - PADDING_STRATEGY: "pre" or "post" padding strategy (default: "pre")
    - TRUNCATION_STRATEGY: "pre" or "post" truncation strategy (default: "post")
    - VALIDATION_STRATEGY: "strict" or "lenient" validation mode (default: "strict")
    - OUTPUT_FORMAT: Output format - "numpy", "parquet", or "csv" (default: "numpy")
    - INCLUDE_ATTENTION_MASKS: Generate attention masks for padded sequences (default: "true")
    
    Multi-Sequence Configuration:
    - ENABLE_MULTI_SEQUENCE: Enable dual-sequence processing (default: "false")
    - SECONDARY_ENTITY_FIELD: Secondary entity field for dual-sequence (default: "creditCardId")
    - SEQUENCE_NAMING_PATTERN: Pattern for sequence field detection (default: "*_seq_by_{entity}.*")
    
    Processing Configuration:
    - ENABLE_DISTRIBUTED_PROCESSING: Enable chunked processing (default: "false")
    - CHUNK_SIZE: Chunk size for distributed processing (default: "10000")
    - MAX_WORKERS: Number of parallel workers (default: "auto")
    
    Output Structure:
    - Normalized sequence arrays saved in specified format
    - Metadata JSON file with configuration and shape information
    - Optional attention masks for padded sequences
    - Support for both single and multi-sequence outputs
    
    TSA Integration Support:
    - Designed for fraud detection temporal sequences
    - Dual-sequence processing (customer ID + credit card ID)
    - Complex nested field name handling with dot notation
    - Time delta computation with fraud-specific capping
    - Configurable validation for production data quality requirements
    """,
)
