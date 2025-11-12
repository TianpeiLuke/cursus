"""
Label Ruleset Generation Script Contract

Defines the contract for the label ruleset generation script that validates and optimizes
user-defined classification rules for transparent, maintainable rule-based classification.
"""

from ...core.base.contract_base import ScriptContract

LABEL_RULESET_GENERATION_CONTRACT = ScriptContract(
    entry_point="label_ruleset_generation.py",
    expected_input_paths={
        "ruleset_configs": "/opt/ml/processing/input/ruleset_configs",
    },
    expected_output_paths={
        "validated_ruleset": "/opt/ml/processing/output/validated_ruleset",
        "validation_report": "/opt/ml/processing/output/validation_report",
    },
    expected_arguments={},
    required_env_vars=[],
    optional_env_vars={
        "ENABLE_FIELD_VALIDATION": "true",
        "ENABLE_LABEL_VALIDATION": "true",
        "ENABLE_LOGIC_VALIDATION": "true",
        "ENABLE_RULE_OPTIMIZATION": "true",
    },
    framework_requirements={
        "pathlib": ">=1.0.0",
    },
    description="""
    Label ruleset generation script that validates and optimizes user-defined classification rules
    for transparent, maintainable rule-based label mapping in ML training pipelines.
    
    The script performs two-tier validation and optimization:
    1. Label Value Validation: Ensures output labels match configuration
    2. Rule Logic Validation: Checks for tautologies, contradictions, and unreachable rules
    3. Rule Optimization: Reorders rules by complexity for efficient execution
    4. Field Inference: Automatically infers field schema from rule definitions
    
    Note: Field schema validation is no longer performed at generation time. The script automatically
    infers field configuration from the rules themselves, ensuring perfect consistency.
    
    Input Structure:
    - /opt/ml/processing/input/ruleset_configs: Ruleset configuration directory (required)
      - label_config.json: Label configuration (required)
        * output_label_name: Name of the output label column
        * output_label_type: "binary" or "multiclass"
        * label_values: Array of valid label values (e.g., [0, 1])
        * label_mapping: Dictionary mapping label values to human-readable names
        * default_label: Default label when no rules match
        * evaluation_mode: Rule evaluation mode (default: "priority")
      - ruleset.json: Rule definitions (required)
        * Array of rule objects with:
          - rule_id: Unique rule identifier
          - name: Human-readable rule name
          - priority: Priority for evaluation (lower = higher priority)
          - enabled: Whether rule is active (default: true)
          - conditions: Nested condition expression (supports all_of, any_of, none_of)
          - output_label: Label value to output when rule matches
          - description: Description of what this rule identifies
    
    Output Structure:
    - /opt/ml/processing/output/validated_ruleset: Validated and optimized ruleset
      - validated_ruleset.json: Main validated ruleset file
        * version: Ruleset version (1.0)
        * generated_timestamp: ISO timestamp
        * label_config: Label configuration (same as input)
        * field_config: Field configuration (same as input)
        * ruleset: Optimized rules (sorted by priority)
        * metadata: Validation and optimization metadata
          - total_rules: Total number of rules
          - enabled_rules: Number of enabled rules
          - disabled_rules: Number of disabled rules
          - field_usage: Dictionary of field usage counts
          - validation_summary: Validation results summary
          - optimization_metadata: Optimization settings applied
    - /opt/ml/processing/output/validation_report: Detailed validation report
      - validation_report.json: Comprehensive validation results
        * validation_status: "passed" or "failed"
        * field_validation: Field validation results
        * label_validation: Label validation results
        * logic_validation: Logic validation results
        * optimization_applied: Whether optimization was applied
        * metadata: Additional metadata from validated ruleset
    
    Contract aligned with script implementation:
    - Inputs: ruleset_configs (required directory with JSON config files)
    - Outputs: validated_ruleset (primary), validation_report (detailed diagnostics)
    - Arguments: None (all configuration via environment variables)
    - Configuration: File-based approach using JSON files
    
    Environment Variables (all optional with defaults):
    - ENABLE_FIELD_VALIDATION: Enable field schema validation (default: "true")
    - ENABLE_LABEL_VALIDATION: Enable label value validation (default: "true")
    - ENABLE_LOGIC_VALIDATION: Enable rule logic validation (default: "true")
    - ENABLE_RULE_OPTIMIZATION: Enable rule priority optimization (default: "true")
    
    Input Configuration File Formats:
    
    === Single-Label Configuration (Binary/Multiclass) ===
    
    label_config.json (required):
    {
      "output_label_name": "final_reversal_flag",
      "output_label_type": "binary",
      "label_values": [0, 1],
      "label_mapping": {
        "0": "No_Reversal",
        "1": "Reversal"
      },
      "default_label": 1,
      "evaluation_mode": "priority"
    }
    
    === Multi-Label Configuration (NEW) ===
    
    label_config.json with multiple output columns:
    {
      "output_label_name": ["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
      "output_label_type": "multilabel",
      "label_values": [0, 1],
      "label_mapping": {
        "0": "No_Fraud",
        "1": "Fraud"
      },
      "default_label": 0,
      "evaluation_mode": "priority",
      "sparse_representation": true
    }
    
    Per-Column Configuration (Advanced):
    {
      "output_label_name": ["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
      "output_label_type": "multilabel",
      "label_values": {
        "is_fraud_CC": [0, 1],
        "is_fraud_DC": [0, 1],
        "is_fraud_ACH": [0, 1, 2]
      },
      "label_mapping": {
        "is_fraud_CC": {"0": "No_Fraud", "1": "Fraud"},
        "is_fraud_DC": {"0": "No_Fraud", "1": "Fraud"},
        "is_fraud_ACH": {"0": "No_Fraud", "1": "Low_Risk_Fraud", "2": "High_Risk_Fraud"}
      },
      "default_label": {
        "is_fraud_CC": 0,
        "is_fraud_DC": 0,
        "is_fraud_ACH": 0
      },
      "evaluation_mode": "priority",
      "sparse_representation": true
    }
    
    === Single-Label Rules ===
    
    ruleset.json (required):
    [
      {
        "rule_id": "rule_001",
        "name": "High confidence TrueDNR",
        "priority": 1,
        "enabled": true,
        "conditions": {
          "all_of": [
            {
              "field": "category",
              "operator": "equals",
              "value": "TrueDNR"
            },
            {
              "field": "confidence_score",
              "operator": ">=",
              "value": 0.8
            }
          ]
        },
        "output_label": 0,
        "description": "High confidence TrueDNR cases indicate no reversal"
      }
    ]
    
    === Multi-Label Rules (NEW) ===
    
    ruleset.json with multilabel output:
    [
      {
        "rule_id": "rule_cc_001",
        "name": "High value CC transaction",
        "priority": 1,
        "enabled": true,
        "conditions": {
          "all_of": [
            {"field": "payment_method", "operator": "equals", "value": "CC"},
            {"field": "amount", "operator": ">", "value": 1000}
          ]
        },
        "output_label": {"is_fraud_CC": 1},
        "description": "High value credit card transaction flagged as fraud"
      },
      {
        "rule_id": "rule_dc_001",
        "name": "International DC transaction",
        "priority": 2,
        "enabled": true,
        "conditions": {
          "all_of": [
            {"field": "payment_method", "operator": "equals", "value": "DC"},
            {"field": "is_international", "operator": "equals", "value": true}
          ]
        },
        "output_label": {"is_fraud_DC": 1},
        "description": "International debit card transaction flagged as fraud"
      },
      {
        "rule_id": "rule_multi_001",
        "name": "Suspicious pattern across methods",
        "priority": 3,
        "enabled": true,
        "conditions": {
          "all_of": [
            {"field": "velocity_score", "operator": ">", "value": 0.9},
            {"field": "device_risk", "operator": "equals", "value": "high"}
          ]
        },
        "output_label": {
          "is_fraud_CC": 1,
          "is_fraud_DC": 1,
          "is_fraud_ACH": 1
        },
        "description": "Suspicious pattern applies to all payment methods"
      }
    ]
    
    Condition Operators Supported:
    - Comparison: equals, not_equals, >, >=, <, <=
    - Collection: in, not_in
    - String: contains, not_contains, starts_with, ends_with, regex_match
    - Null: is_null, is_not_null
    - Logical: all_of (AND), any_of (OR), none_of (NOT)
    
    Validation Features:
    
    1. Field Inference (Automatic):
       - Automatically infers field schema from rule definitions
       - Extracts all field names used in conditions
       - Infers field types from values used (string, int, float, bool)
       - Generates field_config in output (not required in input)
    
    2. Label Value Validation:
       - All output_label values must be in label_values
       - Default label must be valid
       - Binary classification enforces [0, 1] values
       - Identifies uncovered label values (warnings)
       - Detects conflicting rules (same priority, different outputs)
    
    3. Rule Logic Validation:
       - Detects tautologies (always-true conditions)
       - Detects contradictions (always-false conditions)
       - Checks operator-type compatibility
       - Identifies potentially unreachable rules
    
    Optimization Features:
    
    1. Complexity-Based Ordering:
       - Calculates complexity score for each rule
       - Reorders rules from simple to complex
       - Assigns final priorities sequentially
    
    2. Field Usage Analysis:
       - Tracks which fields are used most frequently
       - Provides field usage statistics in metadata
    
    Quality Assurance:
    - Comprehensive validation with clear error messages
    - Validation fails fast on critical errors
    - Warnings for non-critical issues
    - Detailed validation report for debugging
    
    Separation of Concerns:
    - Generation-Time: Schema validation, rule logic validation, optimization
    - Execution-Time: Data availability validation, rule execution (handled by separate step)
    
    Integration Ready:
    - Direct compatibility with RulesetExecutor step
    - Standard SageMaker container paths
    - Comprehensive metadata for monitoring
    - Clear separation from execution concerns
    
    Usage in Pipeline:
    1. RulesetGenerator validates and optimizes rules (this script)
    2. TabularPreprocessing provides processed data
    3. RulesetExecutor applies validated rules to data
    4. Training step consumes labeled data
    
    Error Handling:
    - Clear error messages for missing required files
    - Detailed validation failures in report
    - Fails fast on critical validation errors
    - Continues processing on warnings
    """,
)
