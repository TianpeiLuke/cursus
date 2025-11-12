"""
Label Ruleset Execution Script Contract

Defines the contract for the label ruleset execution script that applies validated
rulesets to processed data to generate classification labels. Integrates with the
Label Ruleset Generation step and supports stacked preprocessing patterns.
"""

from ...core.base.contract_base import ScriptContract

LABEL_RULESET_EXECUTION_CONTRACT = ScriptContract(
    entry_point="label_ruleset_execution.py",
    expected_input_paths={
        "validated_ruleset": "/opt/ml/processing/input/validated_ruleset",
        "input_data": "/opt/ml/processing/input/data",
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/processed_data",
        "execution_report": "/opt/ml/processing/output/execution_report",
    },
    expected_arguments={
        # No expected arguments - job_type comes from config
    },
    required_env_vars=[],
    optional_env_vars={
        "FAIL_ON_MISSING_FIELDS": "true",
        "ENABLE_RULE_MATCH_TRACKING": "true",
        "ENABLE_PROGRESS_LOGGING": "true",
        "PREFERRED_INPUT_FORMAT": "",
    },
    framework_requirements={
        "pandas": ">=1.2.0",
        "pathlib": ">=1.0.0",
    },
    description="""
    Label ruleset execution script that applies validated rulesets to processed data
    to generate classification labels. Provides execution-time field validation,
    priority-based rule evaluation, and comprehensive execution statistics.
    
    The script integrates seamlessly with the Label Ruleset Generation step and
    supports stacked preprocessing patterns by using processed_data for both input
    and output, enabling preprocessing pipeline composition.
    
    Key Features:
    
    Execution-Time Field Validation:
    - Validates all required fields exist in actual DataFrame
    - Validates all fields used in rules exist in actual DataFrame
    - Provides data quality warnings (high null percentages > 50%)
    - Configurable fail-fast or skip-on-error behavior
    
    Priority-Based Rule Evaluation:
    - First matching rule wins (priority-based evaluation)
    - Fail-safe error handling (logs and continues on rule errors)
    - Default label fallback when no rules match
    - Comprehensive statistics tracking per rule
    
    Multi-Format Support:
    - Auto-detection: CSV (.csv, .csv.gz), TSV (.tsv, .tsv.gz), Parquet (.parquet, .pq)
    - Format preservation: Outputs in same format as input
    - Compression support: Handles gzipped CSV/TSV files
    
    Job Type Support (Aligned with Tabular Preprocessing):
    
    Training Job Type (job_type="training"):
    - Processes all three splits: train/, val/, test/
    - Each split processed independently with statistics
    - Output maintains train/val/test organization for training compatibility
    - Directory structure: train_processed_data.*, val_processed_data.*, test_processed_data.*
    
    Non-Training Job Types (job_type="validation", "testing", "calibration"):
    - Processes single split matching job_type
    - validation → validation/ directory
    - testing → testing/ directory
    - calibration → calibration/ directory
    - Directory structure: {split}_processed_data.*
    
    Input Structure:
    - /opt/ml/processing/input/validated_ruleset: Validated ruleset from Generation step (required)
      - /opt/ml/processing/input/validated_ruleset/validated_ruleset.json: Complete ruleset configuration
        * version: Ruleset version identifier
        * generated_timestamp: ISO timestamp of ruleset generation
        * label_config: Output label configuration
          - output_label_name: Name of label column to create
          - output_label_type: Type of classification (binary, multiclass)
          - label_values: List of valid label values
          - label_mapping: Human-readable label descriptions
          - default_label: Default label when no rules match
          - evaluation_mode: Rule evaluation strategy (priority)
        * field_config: Field requirements and types
          - required_fields: List of fields that must exist in data
          - field_types: Dictionary mapping field names to types (string, int, float, bool)
        * ruleset: Array of validated rules (sorted by priority)
          - rule_id: Unique rule identifier
          - name: Human-readable rule name
          - priority: Evaluation priority (lower = evaluated first)
          - enabled: Whether rule is active (true/false)
          - conditions: Nested condition expression tree
          - output_label: Label value to assign when rule matches
          - description: Rule purpose documentation
          - complexity_score: Rule complexity metric
        * metadata: Ruleset statistics and validation summary
    
    - /opt/ml/processing/input/processed_data: Processed data from preprocessing steps (required)
      - Training job type: Expects train/, val/, test/ subdirectories
        * train/train_processed_data.{csv|tsv|parquet}
        * val/val_processed_data.{csv|tsv|parquet}
        * test/test_processed_data.{csv|tsv|parquet}
      - Non-training job types: Expects single split directory
        * {job_type}/{job_type}_processed_data.{csv|tsv|parquet}
      - Data must contain all fields referenced in ruleset rules
      - Supports outputs from: TabularPreprocessing, BedrockProcessing, RiskTableMapping, etc.
    
    Output Structure:
    - /opt/ml/processing/output/processed_data: Labeled data (same format as input)
      - Training job type: Maintains train/val/test split structure
        * train/train_processed_data.{csv|tsv|parquet} (original + label column)
        * val/val_processed_data.{csv|tsv|parquet} (original + label column)
        * test/test_processed_data.{csv|tsv|parquet} (original + label column)
      - Non-training job types: Maintains split structure
        * {job_type}/{job_type}_processed_data.{csv|tsv|parquet} (original + label column)
      - Format preserved: CSV outputs CSV, Parquet outputs Parquet, etc.
      - New column added: {output_label_name} with assigned label values
    
    - /opt/ml/processing/output/execution_report: Execution statistics (JSON)
      - execution_report.json: Comprehensive execution statistics
        * ruleset_version: Version of applied ruleset
        * ruleset_timestamp: When ruleset was generated
        * execution_timestamp: When execution occurred
        * label_config: Applied label configuration
        * split_statistics: Per-split execution details
          - total_rows: Number of rows processed
          - label_distribution: Count of each label value assigned
          - execution_stats: Detailed rule matching statistics
            * total_evaluated: Total rows evaluated
            * rule_match_counts: Matches per rule
            * default_label_count: Rows using default label
            * rule_match_percentages: Match rates per rule
            * default_label_percentage: Default label usage rate
        * total_rules_evaluated: Number of active rules in ruleset
      - rule_match_statistics.json: Detailed per-split rule match information
        * Same structure as split_statistics in execution_report
        * Useful for debugging and rule optimization
    
    Environment Variables:
    
    Validation Configuration:
    - FAIL_ON_MISSING_FIELDS: Fail execution if required fields missing (optional, default: "true")
      - "true": Raises ValueError and stops execution on missing fields
      - "false": Logs warning and skips splits with missing fields
      - Enables graceful degradation in pipelines with optional preprocessing steps
    
    Execution Configuration:
    - ENABLE_RULE_MATCH_TRACKING: Track which rules match (optional, default: "true")
      - "true": Maintains detailed per-rule match statistics
      - "false": Disables tracking for performance optimization
    - ENABLE_PROGRESS_LOGGING: Log progress during processing (optional, default: "true")
      - "true": Logs detailed progress information
      - "false": Minimal logging for production environments
    
    Rule Evaluation Engine:
    
    Supported Operators:
    - Comparison: equals, not_equals, >, >=, <, <=
    - Collection: in, not_in
    - String: contains, not_contains, starts_with, ends_with, regex_match
    - Null: is_null, is_not_null
    
    Logical Operators:
    - all_of: All conditions must be true (AND)
    - any_of: At least one condition must be true (OR)
    - none_of: All conditions must be false (NOT)
    - Supports nested logical expressions with arbitrary depth
    
    Evaluation Strategy:
    - Priority-based: Rules evaluated in priority order (1, 2, 3, ...)
    - First match wins: Returns immediately when first rule matches
    - Default fallback: Uses default_label when no rules match
    - Fail-safe: Continues evaluation on individual rule errors
    
    Field Validation (Execution-Time):
    
    Validation Checks:
    - Required fields exist: All fields in field_config.required_fields present
    - Used fields exist: All fields referenced in rule conditions present
    - Data quality warnings: Alerts when fields have > 50% null values
    
    Validation Behavior:
    - Performed before rule evaluation for each split
    - Missing fields: Raises error or skips split based on FAIL_ON_MISSING_FIELDS
    - Data quality warnings: Logged but do not stop execution
    - Validation results: Included in execution report
    
    Statistics Tracking:
    
    Per-Split Statistics:
    - Total rows processed
    - Label distribution (count per label value)
    - Rule match counts (matches per rule)
    - Default label count (unmatched rows)
    - Rule match percentages (percentage per rule)
    - Default label percentage (unmatched percentage)
    
    Aggregate Statistics:
    - Total splits processed successfully
    - Overall execution time
    - Ruleset version and timestamp
    - Configuration applied
    
    Integration Patterns:
    
    Stacked Preprocessing Pattern:
    The script uses processed_data for both input and output, enabling seamless
    preprocessing pipeline composition:
    
    ```
    Tabular Preprocessing → processed_data
        ↓
    Bedrock Processing → processed_data (adds LLM outputs)
        ↓
    Ruleset Execution → processed_data (adds labels based on rules + LLM)
        ↓
    Stratified Sampling → processed_data (balances classes)
        ↓
    Training Step
    ```
    
    Pipeline Integration Example:
    ```python
    # Step 1: Tabular preprocessing
    preprocessing_step = TabularPreprocessingStepBuilder(config).create_step()
    
    # Step 2: Bedrock processing (adds LLM categorization columns)
    bedrock_step = BedrockBatchProcessingStepBuilder(config).create_step(
        inputs={'processed_data': preprocessing_step.properties...}
    )
    
    # Step 3: Generate validated ruleset (independent, runs in parallel)
    ruleset_generator_step = RulesetGeneratorStepBuilder(config).create_step()
    
    # Step 4: Execute ruleset (depends on both Bedrock and Generator)
    ruleset_executor_step = RulesetExecutorStepBuilder(config).create_step(
        inputs={
            'validated_ruleset': ruleset_generator_step.properties...,
            'processed_data': bedrock_step.properties...  # Uses LLM outputs
        },
        dependencies=[ruleset_generator_step, bedrock_step]
    )
    
    # Step 5: Training (uses labeled data)
    training_step = TrainingStepBuilder(config).create_step(
        inputs={'training_data': ruleset_executor_step.properties...}
    )
    ```
    
    Example Rule Usage:
    
    Rules can reference any fields in the processed data, including LLM outputs:
    ```json
    {
      "rule_id": "rule_001",
      "name": "High confidence TrueDNR",
      "priority": 1,
      "enabled": true,
      "conditions": {
        "all_of": [
          {"field": "llm_category", "operator": "equals", "value": "TrueDNR"},
          {"field": "llm_confidence", "operator": ">=", "value": 0.8}
        ]
      },
      "output_label": 0,
      "description": "High confidence TrueDNR from LLM indicates no reversal"
    }
    ```
    
    Error Handling and Resilience:
    
    Execution-Level Errors:
    - Missing input files: Logged with clear error messages
    - Invalid ruleset format: Raises error with details
    - Missing required fields: Configurable fail-fast or skip behavior
    
    Rule-Level Errors:
    - Type conversion errors: Logged, evaluation continues with next rule
    - Null value handling: Explicit operators for null checks
    - Invalid operators: Raises ValueError with unsupported operator name
    
    Split-Level Errors:
    - Missing split directories: Logged as warning, continues with other splits
    - Empty data files: Logged as warning, skips split
    - Field validation failures: Configurable skip or fail behavior
    
    Performance Optimizations:
    
    Rule Evaluation:
    - Priority-based early termination (first match wins)
    - Vectorized DataFrame operations where possible
    - Minimal memory footprint with row-by-row evaluation
    
    Statistics Tracking:
    - In-memory counters with O(1) updates
    - Optional detailed tracking can be disabled for performance
    - Efficient label distribution computation using value_counts()
    
    I/O Optimization:
    - Format auto-detection avoids unnecessary conversions
    - Compression support for reduced I/O time
    - Parquet format support for large datasets
    
    Quality Assurance:
    
    Comprehensive Logging:
    - Startup parameter logging
    - Per-split processing progress
    - Field validation results
    - Label distribution summaries
    - Completion statistics
    
    Execution Tracking:
    - Detailed rule match statistics
    - Default label usage tracking
    - Data quality warnings
    - Processing timestamps
    
    Output Validation:
    - Label values validated against label_config.label_values
    - Statistics consistency checks
    - Format preservation verification
    
    Usage Examples:
    
    Training Pipeline (All Splits):
    ```bash
    export FAIL_ON_MISSING_FIELDS="true"
    export ENABLE_RULE_MATCH_TRACKING="true"
    python label_ruleset_execution.py --job-type training
    ```
    
    Single Split Processing:
    ```bash
    python label_ruleset_execution.py --job-type validation
    ```
    
    Lenient Field Validation:
    ```bash
    export FAIL_ON_MISSING_FIELDS="false"
    python label_ruleset_execution.py --job-type training
    ```
    
    Output Data Structure:
    
    === Single-Label Output (Binary/Multiclass) ===
    - All original columns preserved
    - New label column added (name from ruleset configuration)
    - No modification to existing columns
    - Same row order maintained
    - Same data types preserved
    
    Example Single-Label Output:
    | txn_id | amount | payment_method | final_reversal_flag |
    |--------|--------|----------------|---------------------|
    | 1      | 1200   | CC             | 1                   |
    | 2      | 500    | DC             | 0                   |
    | 3      | 8000   | ACH            | 1                   |
    
    === Multi-Label Output (NEW) ===
    - All original columns preserved
    - Multiple label columns added (names from ruleset configuration)
    - No modification to existing columns
    - Same row order maintained
    - Sparse representation: NaN for non-matching columns (default)
    - Dense representation: default_label for all columns (optional)
    
    Example Multi-Label Output (Sparse):
    | txn_id | amount | payment_method | is_fraud_CC | is_fraud_DC | is_fraud_ACH |
    |--------|--------|----------------|-------------|-------------|--------------|
    | 1      | 1200   | CC             | 1           | NaN         | NaN          |
    | 2      | 500    | DC             | NaN         | 0           | NaN          |
    | 3      | 8000   | ACH            | NaN         | NaN         | 1            |
    | 4      | 2000   | CC             | 1           | NaN         | NaN          |
    
    Example Multi-Label Output (Dense):
    | txn_id | amount | payment_method | is_fraud_CC | is_fraud_DC | is_fraud_ACH |
    |--------|--------|----------------|-------------|-------------|--------------|
    | 1      | 1200   | CC             | 1           | 0           | 0            |
    | 2      | 500    | DC             | 0           | 0           | 0            |
    | 3      | 8000   | ACH            | 0           | 0           | 1            |
    | 4      | 2000   | CC             | 1           | 0           | 0            |
    
    Multi-Label Statistics Format:
    {
      "label_type": "multilabel",
      "total_evaluated": 1000,
      "per_column_statistics": {
        "is_fraud_CC": {
          "rule_match_counts": {"rule_cc_001": 50, "rule_multi_001": 30},
          "default_label_count": 20,
          "rule_match_percentages": {"rule_cc_001": 5.0, "rule_multi_001": 3.0},
          "default_label_percentage": 2.0
        },
        "is_fraud_DC": {
          "rule_match_counts": {"rule_dc_001": 40, "rule_multi_001": 30},
          "default_label_count": 30,
          "rule_match_percentages": {"rule_dc_001": 4.0, "rule_multi_001": 3.0},
          "default_label_percentage": 3.0
        },
        "is_fraud_ACH": {
          "rule_match_counts": {"rule_ach_001": 60, "rule_multi_001": 30},
          "default_label_count": 10,
          "rule_match_percentages": {"rule_ach_001": 6.0, "rule_multi_001": 3.0},
          "default_label_percentage": 1.0
        }
      }
    }
    
    Backward Compatibility:
    - Single-label mode remains the default
    - Existing pipelines continue to work unchanged
    - Statistics format extended, not replaced
    - Single-label output format unchanged
    
    The script is production-ready with comprehensive error handling, performance
    optimizations, and integration capabilities for enterprise ML pipelines. It
    provides transparent, rule-based label generation with full auditability and
    easy modification through the ruleset configuration.
    """,
)
