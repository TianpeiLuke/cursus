"""
Pseudo Label Merge Script Contract

Defines the contract for the pseudo label merge script that handles intelligent merging
of labeled and pseudo-labeled/augmented data for Semi-Supervised Learning (SSL) and
Active Learning workflows.
"""

from ...core.base.contract_base import ScriptContract

PSEUDO_LABEL_MERGE_CONTRACT = ScriptContract(
    entry_point="pseudo_label_merge.py",
    expected_input_paths={
        "base_data": "/opt/ml/processing/input/base_data",
        "augmentation_data": "/opt/ml/processing/input/augmentation_data",
    },
    expected_output_paths={"merged_data": "/opt/ml/processing/output/merged_data"},
    expected_arguments={
        # job_type comes from command-line argument
    },
    required_env_vars=["LABEL_FIELD"],
    optional_env_vars={
        "ADD_PROVENANCE": "true",
        "OUTPUT_FORMAT": "csv",
        "USE_AUTO_SPLIT_RATIOS": "true",
        "TRAIN_RATIO": "",
        "TEST_VAL_RATIO": "",
        "PSEUDO_LABEL_COLUMN": "pseudo_label",
        "ID_FIELD": "id",
        "PRESERVE_CONFIDENCE": "true",
        "STRATIFY": "true",
        "RANDOM_SEED": "42",
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "scikit-learn": ">=1.0.0",
    },
    description="""
    Pseudo label merge script that intelligently combines original labeled training data
    with pseudo-labeled or augmented samples for SSL and Active Learning workflows.
    
    Key Features:
    1. Split-aware merge for training jobs (maintains train/test/val boundaries)
    2. Auto-inferred split ratios (adapts to base data proportions)
    3. Simple merge for validation/testing/calibration jobs
    4. Data format preservation (CSV/TSV/Parquet)
    5. Schema alignment and provenance tracking
    
    Contract aligned with actual script implementation:
    - Inputs:
      * base_data (required) - reads from /opt/ml/processing/input/base_data
        Contains original labeled data with optional train/test/val split structure
      * augmentation_data (required) - reads from /opt/ml/processing/input/augmentation_data
        Contains pseudo-labeled or actively selected samples
    - Outputs:
      * merged_data (primary) - writes to /opt/ml/processing/output/merged_data
        Contains merged data maintaining split structure with provenance tracking
    - Arguments:
      * job_type (required) - defines merge mode:
        - "training": Uses split-aware merge with auto-inferred or manual ratios
        - "validation": Simple concatenation merge
        - "testing": Simple concatenation merge
        - "calibration": Simple concatenation merge
    
    Script Implementation Details:
    
    Data Loading:
    - Detects and loads multiple formats: CSV, TSV, Parquet (with .gz support)
    - Handles sharded data (part-*.csv, part-*.parquet)
    - Auto-detects split structure (train/test/val subdirectories)
    - Falls back to single dataset if splits not found
    
    Split-Aware Merge (Training Jobs):
    - Detects train/test/val split structure in base data
    - Auto-infers split ratios from base data proportions (RECOMMENDED)
    - Alternative: Manual split ratios via TRAIN_RATIO and TEST_VAL_RATIO
    - Distributes augmentation data proportionally across splits
    - Uses stratified splitting to maintain class balance
    - Adds provenance tracking (data_source: "original" vs "pseudo_labeled")
    - Merges corresponding splits maintaining boundaries
    
    Simple Merge (Non-Training Jobs):
    - Used for validation/testing/calibration jobs
    - Concatenates base and augmentation data directly
    - Adds provenance tracking
    - No split structure manipulation
    
    Schema Alignment:
    - Converts pseudo_label column to label column
    - Extracts common columns between datasets
    - Aligns data types for compatibility
    - Handles type mismatches gracefully
    
    Format Preservation:
    - Maintains input format (CSV/TSV/Parquet) unless OUTPUT_FORMAT specified
    - Preserves directory structure (split subdirectories)
    - Generates merge metadata JSON with configuration and statistics
    
    Provenance Tracking:
    - Adds "data_source" column to all records
    - "original" for base labeled data
    - "pseudo_labeled" for augmentation data
    - Validates provenance in merged output
    
    Auto-Inferred Split Ratios (NEW FEATURE):
    - Calculates actual proportions from base data splits
    - Example: 10K train / 2K test / 2K val → 71.4% / 14.3% / 14.3%
    - Applies same ratios to augmentation data distribution
    - Ensures augmentation follows base data characteristics
    - Enabled by default via USE_AUTO_SPLIT_RATIOS=true
    - Zero configuration needed - adapts automatically
    
    Environment Variable Details:
    
    Required:
    - LABEL_FIELD: Name of the label column in both datasets
    
    Optional (with defaults):
    - USE_AUTO_SPLIT_RATIOS: "true" (recommended) - auto-infer from base data
    - TRAIN_RATIO: "" (None) - only used if USE_AUTO_SPLIT_RATIOS=false
    - TEST_VAL_RATIO: "" (None) - only used if USE_AUTO_SPLIT_RATIOS=false
    - OUTPUT_FORMAT: "csv" - output format (csv/tsv/parquet)
    - ADD_PROVENANCE: "true" - add data_source column
    - PSEUDO_LABEL_COLUMN: "pseudo_label" - column name in augmentation data
    - ID_FIELD: "id" - ID column name for schema validation
    - PRESERVE_CONFIDENCE: "true" - keep confidence scores from augmentation
    - STRATIFY: "true" - use stratified splits to maintain class balance
    - RANDOM_SEED: "42" - random seed for reproducibility
    
    Output Structure (Training Job with Splits):
    ```
    /opt/ml/processing/output/merged_data/
      train/
        train_processed_data.csv  (base + augmentation train portion)
      test/
        test_processed_data.csv   (base + augmentation test portion)
      val/
        val_processed_data.csv    (base + augmentation val portion)
      merge_metadata.json         (merge operation metadata)
    ```
    
    Output Structure (Non-Training Jobs):
    ```
    /opt/ml/processing/output/merged_data/
      {job_type}/
        {job_type}_processed_data.csv  (base + augmentation)
      merge_metadata.json
    ```
    
    Merge Metadata JSON:
    - job_type: Type of merge job executed
    - merge_strategy: "split_aware" or "simple"
    - base_splits: Shape and count of base data by split
    - augmentation_count: Number of augmentation samples
    - merged_splits: Shape and count of merged data by split
    - configuration: All environment variable settings used
    - output_paths: Actual file paths created
    - timestamp: ISO format timestamp of merge operation
    
    Error Handling:
    - Validates LABEL_FIELD exists in both datasets
    - Checks for common columns between datasets
    - Validates split ratios sum to 1.0 (normalizes if needed)
    - Handles type mismatches in schema alignment
    - Validates provenance column in output
    - Comprehensive logging at each step
    
    Use Cases:
    1. SSL Pretraining → Fine-tuning:
       - Merge small labeled data with pseudo-labeled data from pretraining
       - Use auto-inferred ratios to maintain data characteristics
       - Train fine-tuned model on combined dataset
    
    2. Active Learning Iteration:
       - Merge existing labeled data with actively selected samples
       - Confidence scores preserved for analysis
       - Provenance tracking for sample selection audit
    
    3. Data Augmentation:
       - Merge original data with augmented/synthesized samples
       - Stratified distribution across splits
       - Maintains train/test/val integrity
    
    Design Documentation:
    See slipbox/1_design/pseudo_label_merge_script_design.md for complete design details.
    """,
)
