"""
Dummy Data Loading Script Contract

Defines the contract for the Dummy data loading script that processes
user-provided data instead of calling internal Cradle services.
This step serves as a drop-in replacement for CradleDataLoadingStep.
"""

from ...core.base.contract_base import ScriptContract

DUMMY_DATA_LOADING_CONTRACT = ScriptContract(
    entry_point="dummy_data_loading.py",
    expected_input_paths={
        "INPUT_DATA": "/opt/ml/processing/input/data",  # Input data channel
    },
    expected_output_paths={
        "SIGNATURE": "/opt/ml/processing/output/signature",
        "METADATA": "/opt/ml/processing/output/metadata",
        "DATA": "/opt/ml/processing/output/data",  # Data output directory
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables
    ],
    optional_env_vars={
        "WRITE_DATA_SHARDS": "false",
        "SHARD_SIZE": "10000",
        "OUTPUT_FORMAT": "CSV",
        "MAX_WORKERS": "0",
        "BATCH_SIZE": "5",
        "OPTIMIZE_MEMORY": "false",
        "STREAMING_BATCH_SIZE": "0",
        "ENABLE_TRUE_STREAMING": "false",
        "METADATA_FORMAT": "JSON",
    },
    framework_requirements={"python": ">=3.7", "boto3": ">=1.26.0"},
    description="""
    Dummy data loading script that:
    1. Reads user-provided data from input data channel
    2. Processes and validates the input data
    3. Writes output signature for data schema
    4. Writes metadata file with field type information
    5. Copies/processes data to output location
    
    Input Structure:
    - /opt/ml/processing/input/data: User-provided data files (CSV, Parquet, JSON)
    - Configuration is provided via the job configuration and not through input files
    - /opt/ml/processing/config/config: Data loading configuration is provided by the step creation process
    
    Output Structure:
    - /opt/ml/processing/output/signature/signature: Schema information for the loaded data
    - /opt/ml/processing/output/metadata/metadata: Metadata about fields (type information)
    - Data is processed and made available at the specified output location
    
    Environment Variables (Optional):
    - WRITE_DATA_SHARDS: Enable enhanced data sharding mode (true/false, default: false)
    - SHARD_SIZE: Number of rows per shard file (integer, default: 10000)
    - OUTPUT_FORMAT: Output format for data shards (CSV/JSON/PARQUET, default: CSV)
    
    Memory Optimization Configuration:
    - STREAMING_BATCH_SIZE (default: "0"=disabled) - **PRIMARY MEMORY CONTROL**
      * Enables streaming batch processing to avoid loading all files at once
      * When set > 0, processes that many files per batch, freeing memory between batches
      * Set to "0" to disable (original behavior, loads all files into memory)
      * Recommended values:
        - "15-20" for moderate memory reduction (80-85% reduction)
        - "10-15" for high memory reduction (85-90% reduction)
        - "5-10" for maximum memory reduction (90-95% reduction)
      * Example: 100 files × 50MB = 5GB peak → With "15": 750MB peak (85% reduction)
      * Trade-off: Slightly slower due to multiple concatenation passes
      * **Use this first when encountering out-of-memory errors**
    - MAX_WORKERS (default: "0"=auto) - Controls parallel file reading workers
      * Set to "1" for sequential processing (lowest memory, ~60% reduction)
      * Set to "2" for moderate parallelism (balanced, ~40% reduction)
      * Set to "0" or omit for automatic (uses all CPUs, highest memory)
      * Recommendation: Start with "2", use "1" if out-of-memory errors persist
    - BATCH_SIZE (default: "5") - Controls DataFrame concatenation batch size
      * Smaller values reduce peak memory during concatenation (10-20% reduction)
      * Valid range: 2-10, recommended: 3-5 for memory-constrained environments
      * Works with STREAMING_BATCH_SIZE for fine-tuning memory usage
    - OPTIMIZE_MEMORY (default: "true") - Enables dtype optimization
      * Downcasts int64→int32, float64→float32 (30-50% memory reduction)
      * Converts low-cardinality object columns to category type
      * Set to "false" to disable optimization (not recommended)
    - Combined effect: With STREAMING_BATCH_SIZE=15 + MAX_WORKERS=2 + OPTIMIZE_MEMORY=true,
      can achieve 90-95% memory reduction in extreme cases
    
    Memory Optimization Strategy (Progressive Troubleshooting):
    Level 1 (First try if OOM errors):
      STREAMING_BATCH_SIZE="15"  # 80-85% reduction
      MAX_WORKERS="0"             # Keep auto (fastest)
      
    Level 2 (Still OOM):
      STREAMING_BATCH_SIZE="10"  # 85-90% reduction
      MAX_WORKERS="2"             # Moderate parallelism
      
    Level 3 (Maximum memory savings):
      STREAMING_BATCH_SIZE="5"   # 90-95% reduction
      MAX_WORKERS="1"             # Sequential processing
      BATCH_SIZE="3"              # Further reduce concat batches
    
    True Streaming Mode (ENABLE_TRUE_STREAMING):
    - ENABLE_TRUE_STREAMING (default: "false") - **MAXIMUM MEMORY EFFICIENCY**
      * Enables true streaming mode that NEVER loads full DataFrame into memory
      * When enabled, processes data in batches and writes shards incrementally
      * Memory usage: Fixed ~2GB per batch regardless of total data size
      * Scalability: Unlimited (tested with 100GB+ datasets)
      * **Requires WRITE_DATA_SHARDS=true** (automatically enabled if not set)
      * **Automatically sets STREAMING_BATCH_SIZE=10** if not provided
      * Trade-off: 5-10% slower for small datasets, same speed for large datasets
      * Use case: Out-of-memory errors on large datasets (30M+ rows, 800+ columns)
      * Differences from batch mode:
        - Never accumulates full DataFrame (batch mode eventually loads all data)
        - Generates metadata from first batch only (approximation for large data)
        - Always writes data as shards (cannot write single file)
      * Example configuration for 30M rows × 800 columns:
        ENABLE_TRUE_STREAMING="true"
        METADATA_FORMAT="MODS"        # Lightweight metadata (recommended)
        STREAMING_BATCH_SIZE="10"     # 10 files per batch (auto-set if omitted)
        SHARD_SIZE="100000"           # 100K rows per output shard
        WRITE_DATA_SHARDS="true"      # Required (auto-enabled)
      * Memory comparison:
        - Batch mode: 25GB peak (loads all data)
        - Streaming mode: 2GB peak (12.5× reduction, fixed regardless of data size)
    
    Metadata Format Configuration (METADATA_FORMAT):
    - METADATA_FORMAT (default: "JSON") - Controls metadata output format
      * "JSON": Detailed metadata with statistics (min/max/mean/std for numeric columns)
        - Requires full DataFrame scan in batch mode
        - In streaming mode, statistics computed from first batch only (approximation)
        - Larger file size (~10-100KB for large datasets)
        - Use when: Need detailed statistics, small datasets, batch mode
      * "MODS": Lightweight CSV format (3 columns: varname, iscategory, datatype)
        - Compatible with MODS Cradle Data Loading format
        - Can be generated from first batch only (streaming-friendly)
        - Smaller file size (~1-10KB)
        - Recommended for: True streaming mode, large datasets, memory-constrained environments
      * Both formats produce compatible outputs for downstream processing steps
      * Format structure:
        - JSON: {"version": "1.0", "data_info": {...}, "column_info": {"col1": {...}, ...}}
        - MODS: CSV with header [varname, iscategory, datatype] + data rows
    
    The script performs the following operations:
    - Reads user-provided data from the input data channel
    - Auto-detects data format (CSV, Parquet, JSON)
    - Generates schema signature based on the actual data
    - Writes metadata files with field type information
    - Processes and outputs data in configurable format:
      * Legacy mode (default): Writes placeholder file for backward compatibility
      * Enhanced mode: Writes actual data as shards in part-*.{format} naming convention
    
    Enhanced Data Sharding Mode:
    When WRITE_DATA_SHARDS=true, the script outputs data in a format compatible with
    tabular preprocessing scripts:
    - Data is written as multiple shard files (part-00000.csv, part-00001.csv, etc.)
    - Supports CSV, JSON, and Parquet output formats
    - Configurable shard size for optimal performance
    - Maintains full compatibility with downstream processing steps
    
    This script is designed to replace CradleDataLoadingStep by processing
    user-provided data instead of calling internal Cradle services.
    As an Internal Node, this step requires input data dependencies.
    """,
)
