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
    
    Memory Optimization Configuration:
    - MAX_WORKERS (default: "0"=auto) - Controls parallel shard reading workers
      * Set to "1" for sequential processing (lowest memory, ~60% reduction)
      * Set to "2" for moderate parallelism (balanced, ~40% reduction)
      * Set to "0" or omit for automatic (uses all CPUs, highest memory)
      * Recommendation: Start with "2", use "1" if out-of-memory errors persist
    - BATCH_SIZE (default: "5") - Controls DataFrame concatenation batch size
      * Smaller values reduce peak memory during concatenation (10-20% reduction)
      * Valid range: 2-10, recommended: 3-5 for memory-constrained environments
    - OPTIMIZE_MEMORY (default: "true") - Enables dtype optimization
      * Downcasts int64→int32, float64→float32 (30-50% memory reduction)
      * Converts low-cardinality object columns to category type
      * Set to "false" to disable optimization (not recommended)
    - Combined effect: Can reduce memory usage by 70-80% with MAX_WORKERS=1
    
    Signature File Format:
    - CSV format with comma-separated column names
    - Applied only to CSV/TSV files, ignored for JSON/Parquet formats
    - Backward compatible - works without signature file
    """,
)
