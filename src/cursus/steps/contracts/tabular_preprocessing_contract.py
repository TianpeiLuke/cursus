"""
Tabular Preprocessing Script Contract

Defines the contract for the tabular preprocessing script that handles data loading,
cleaning, and splitting for training/validation/testing.
"""

from ...core.base.contract_base import ScriptContract

TABULAR_PREPROCESSING_CONTRACT = ScriptContract(
    entry_point="tabular_preprocessing.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "SIGNATURE": "/opt/ml/processing/input/signature",
    },
    expected_output_paths={"processed_data": "/opt/ml/processing/output"},
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["TRAIN_RATIO", "TEST_VAL_RATIO"],
    optional_env_vars={
        "LABEL_FIELD": "",
        "OUTPUT_FORMAT": "CSV",
        "MAX_WORKERS": "0",
        "BATCH_SIZE": "5",
        "OPTIMIZE_MEMORY": "true",
        "STREAMING_BATCH_SIZE": "0",
        "ENABLE_TRUE_STREAMING": "false",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Tabular preprocessing script that:
    1. Combines data shards from input directory
    2. Loads column signature for CSV/TSV files if provided
    3. Cleans and processes label field
    4. Splits data into train/test/val for training jobs
    5. Outputs processed files in configurable format (CSV/TSV/Parquet)
    
    Contract aligned with actual script implementation:
    - Inputs: 
      * DATA (required) - reads from /opt/ml/processing/input/data
      * SIGNATURE (optional) - reads from /opt/ml/processing/input/signature
    - Outputs: processed_data (primary) - writes to /opt/ml/processing/output
    - Arguments: job_type (required) - defines processing mode (training/validation/testing)
    
    Script Implementation Details:
    - Reads data shards (CSV, JSON, Parquet) from input/data directory
    - Loads signature file containing column names for CSV/TSV files
    - Supports gzipped files and various formats
    - Uses signature column names for CSV/TSV files when available
    - Processes labels (converts categorical to numeric if needed)
    - Splits data based on job_type (training creates train/test/val splits)
    - Outputs processed files to split subdirectories under /opt/ml/processing/output
    
    Output Format Configuration:
    - OUTPUT_FORMAT environment variable controls output format
    - Valid values: "CSV" (default), "TSV", "Parquet"
    - Case-insensitive, defaults to CSV if invalid value provided
    - Format applies to all output splits (train/val/test)
    - Parquet recommended for large datasets (better compression and performance)
    
    Processing Modes:
    - ENABLE_TRUE_STREAMING (default: "false") - **PRIMARY MODE SELECTOR**
      * "false": Batch mode - loads full DataFrame into memory
        - Single consolidated output file per split
        - Uses stratified splits when labels available
        - Best for smaller datasets that fit in memory
      * "true": Fully parallel streaming mode - 1:1 shard mapping
        - Each input shard processed independently in parallel
        - Output preserves input shard numbers (part-00042 → part-00042)
        - No consolidation - direct write to final sharded format
        - 8-10× faster than batch mode
        - Fixed memory usage regardless of dataset size
        - Uses approximate stratification when labels available
        - **Recommended for large datasets and distributed training**
    
    Memory Optimization Configuration (Batch Mode Only):
    - STREAMING_BATCH_SIZE (default: "0"=disabled) - Batch mode memory control
      * Only applies when ENABLE_TRUE_STREAMING="false"
      * Enables incremental loading to avoid loading all shards at once
      * When set > 0, processes that many shards per batch
      * Recommended values: 10-20 for 80-90% memory reduction
      * Example: 100 shards × 50MB = 5GB peak → With "15": 750MB peak
    - MAX_WORKERS (default: "0"=auto) - Parallel shard reading workers
      * Controls parallelism in both batch and streaming modes
      * "0" = auto-detect (uses all CPUs, fastest)
      * "1" = sequential (lowest memory)
      * "2-4" = moderate parallelism (balanced)
    - BATCH_SIZE (default: "5") - DataFrame concatenation batch size
      * Batch mode only: controls memory during concatenation
      * Smaller values (3-5) reduce peak memory
    - OPTIMIZE_MEMORY (default: "true") - Dtype optimization
      * Both modes: downcasts numeric types, converts to category
      * 30-50% memory reduction
      * Recommended to keep enabled
    
    Recommended Configuration by Use Case:
    
    Small datasets (< 10GB):
      ENABLE_TRUE_STREAMING="false"  # Use batch mode
      STREAMING_BATCH_SIZE="0"        # Not needed
      MAX_WORKERS="0"                 # Use all CPUs
    
    Large datasets (> 10GB) or distributed training:
      ENABLE_TRUE_STREAMING="true"   # Use fully parallel mode
      MAX_WORKERS="0"                 # Use all CPUs for 8-10× speedup
      # Output ready for streaming consumption, no consolidation needed
    
    Memory-constrained batch mode:
      ENABLE_TRUE_STREAMING="false"
      STREAMING_BATCH_SIZE="15"      # Process 15 shards at a time
      MAX_WORKERS="2"                 # Moderate parallelism
    
    Signature File Format:
    - CSV format with comma-separated column names
    - Applied only to CSV/TSV files, ignored for JSON/Parquet formats
    - Backward compatible - works without signature file
    """,
)
