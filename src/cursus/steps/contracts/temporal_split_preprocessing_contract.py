"""
Temporal Split Preprocessing Script Contract

Defines the contract for the temporal split preprocessing script that handles data loading,
temporal splitting, customer-level splitting, and main task label generation.
"""

from ...core.base.contract_base import ScriptContract

TEMPORAL_SPLIT_PREPROCESSING_CONTRACT = ScriptContract(
    entry_point="temporal_split_preprocessing.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "SIGNATURE": "/opt/ml/processing/input/signature",
    },
    expected_output_paths={
        "training_data": "/opt/ml/processing/output/training_data",
        "oot_data": "/opt/ml/processing/output/oot_data",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=["DATE_COLUMN", "GROUP_ID_COLUMN", "SPLIT_DATE"],
    optional_env_vars={
        "TRAIN_RATIO": "0.9",
        "RANDOM_SEED": "42",
        "OUTPUT_FORMAT": "CSV",
        "MAX_WORKERS": "",
        "BATCH_SIZE": "10",
        "LABEL_FIELD": "",
        "TARGETS": "",
        "MAIN_TASK_INDEX": "",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Temporal split preprocessing script with comprehensive features:
    1. Temporal cutoff (date-based split for OOT test)
    2. Customer-level random split (train/validation)
    3. Ensures no customer leakage between train and OOT
    4. Parallel processing for large datasets
    5. Signature file support
    6. Memory-efficient batch concatenation
    7. Multiple output formats (CSV, TSV, Parquet)
    8. Main task label generation from subtasks
    
    Contract aligned with actual script implementation following cursus alignment rules:
    - Arguments: Single job-type argument with CLI-style hyphens, argparse converts to Python underscores
    - Environment Variables: All configuration passed via environment variables (no CLI argument fallbacks)
    - Inputs: 
      * DATA (required) - reads from /opt/ml/processing/input/data
      * SIGNATURE (optional) - reads from /opt/ml/processing/input/signature
    - Outputs: processed_data (primary) - writes to /opt/ml/processing/output
    
    Script Implementation Details:
    - Follows cursus framework patterns (single job-type argument, environment variable configuration)
    - Reads data shards (CSV, JSON, Parquet) from input/data directory
    - Loads signature file containing column names for CSV/TSV files
    - Supports gzipped files and various formats
    - Uses signature column names for CSV/TSV files when available
    - Performs temporal split based on date column and split date
    - Splits customers randomly for train/validation (no leakage)
    - Generates main task labels from subtasks if targets provided
    - Outputs processed files to split subdirectories under /opt/ml/processing/output
    
    Temporal Split Logic:
    - Pre-split data: records before split_date
    - Post-split data: records on/after split_date (becomes OOT after filtering)
    - Customer split: randomly assign customers to train/validation
    - OOT filtering: remove training customers from post-split data to prevent leakage
    - Final splits: train, val (from pre-split), oot (from post-split, filtered)
    
    Main Task Label Generation:
    - Takes maximum value across subtasks for each sample
    - Example: if targets=['is_abuse','is_abusive_dnr','is_abusive_pda','is_abusive_rr']
      and main_task_index=0, then 'is_abuse' = max('is_abusive_dnr', 'is_abusive_pda', 'is_abusive_rr')
    - Supports both JSON format ['col1','col2'] and comma-separated format col1,col2,col3
    
    Required Environment Variables:
    - DATE_COLUMN: Name of the date column for temporal split
    - GROUP_ID_COLUMN: Name of the group ID column
    - SPLIT_DATE: Date for temporal split (YYYY-MM-DD format)
    
    Optional Environment Variables:
    - TRAIN_RATIO: Ratio of customers for training (default: 0.9)
    - RANDOM_SEED: Random seed for reproducibility (default: 42)
    - OUTPUT_FORMAT: Output format - CSV/TSV/Parquet (default: CSV)
    - MAX_WORKERS: Maximum parallel workers for processing (default: auto-detect)
    - BATCH_SIZE: Batch size for DataFrame concatenation (default: 10)
    - LABEL_FIELD: Optional label field for compatibility with standard preprocessing
    - TARGETS: Target columns for main task label generation (comma-separated or JSON format)
    - MAIN_TASK_INDEX: Index of main task in targets list (default: 0)
    
    Output Format Configuration:
    - OUTPUT_FORMAT environment variable controls output format
    - Valid values: "CSV" (default), "TSV", "Parquet"
    - Case-insensitive, defaults to CSV if invalid value provided
    - Format applies to all output splits (train/val/oot)
    - Parquet recommended for large datasets (better compression and performance)
    
    Advanced Features:
    - Parallel shard reading with configurable MAX_WORKERS
    - Memory-efficient batch concatenation with configurable BATCH_SIZE
    - Comprehensive debug output for troubleshooting
    - Compatible with standard preprocessing (optional LABEL_FIELD processing)
    - Follows cursus framework conventions for consistency
    
    Signature File Format:
    - CSV format with comma-separated column names
    - Applied only to CSV/TSV files, ignored for JSON/Parquet formats
    - Backward compatible - works without signature file
    """,
)
