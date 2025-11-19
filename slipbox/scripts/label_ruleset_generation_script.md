---
tags:
  - code
  - processing_script
  - label_generation
  - rule_validation
  - rule_optimization
keywords:
  - label ruleset generation
  - rule validation
  - rule optimization
  - transparent classification
  - rule-based labeling
  - field inference
  - multilabel support
  - priority-based evaluation
  - rule logic validation
topics:
  - label generation
  - rule-based classification
  - transparent ML
  - validation frameworks
language: python
date of note: 2025-11-18
---

# Label Ruleset Generation Script Documentation

## Overview

The `label_ruleset_generation.py` script validates and optimizes user-defined classification rules for transparent, maintainable rule-based label mapping in ML training pipelines. It serves as the first stage in rule-based classification workflows, ensuring rule quality before execution.

The script performs comprehensive validation across multiple dimensions (label values, rule logic, type compatibility) and optimizes rule execution order based on complexity analysis. A key innovation is automatic field schema inference from rule definitions, eliminating the need for separate field configuration and ensuring perfect consistency between rules and field types.

The script supports both single-label (binary/multiclass) and multi-label classification modes, with flexible per-column configuration for advanced use cases. It integrates seamlessly with downstream execution steps through standardized output formats and comprehensive metadata generation.

## Purpose and Major Tasks

### Primary Purpose
Validate and optimize user-defined classification rules to ensure correctness, consistency, and efficient execution in rule-based label mapping workflows for ML training pipelines.

### Major Tasks
1. **Input Configuration Loading**: Load and parse JSON configuration files (label_config.json, ruleset.json)
2. **Automatic Field Inference**: Extract field schema from rule definitions without requiring separate field configuration
3. **Label Value Validation**: Verify all output labels match configured label values and detect conflicts
4. **Rule Logic Validation**: Check for tautologies, contradictions, unreachable rules, and type mismatches
5. **Coverage Analysis**: Identify uncovered label values and orphaned label columns (multilabel mode)
6. **Rule Optimization**: Reorder rules by complexity for efficient execution
7. **Field Usage Analysis**: Track and report field usage statistics across all rules
8. **Validated Ruleset Generation**: Produce optimized ruleset with comprehensive metadata
9. **Validation Report Creation**: Generate detailed validation diagnostics for debugging
10. **Quality Assurance**: Provide comprehensive error messages and warnings for rule quality issues

## Script Contract

### Entry Point
```
label_ruleset_generation.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `ruleset_configs` | `/opt/ml/processing/input/ruleset_configs` | Configuration directory containing JSON config files |

**Expected Input Files**:
- `label_config.json` (required): Label configuration with output structure and valid values
- `ruleset.json` (required): Array of rule definitions with conditions and output labels

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `validated_ruleset` | `/opt/ml/processing/output/validated_ruleset` | Validated and optimized ruleset with metadata |
| `validation_report` | `/opt/ml/processing/output/validation_report` | Detailed validation diagnostics and quality metrics |

**Output Files**:
- `validated_ruleset.json`: Complete validated ruleset with optimized priorities and metadata
- `validation_report.json`: Comprehensive validation results with warnings and statistics

### Required Environment Variables
None - all configuration via files

### Optional Environment Variables

#### Validation Controls
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_FIELD_VALIDATION` | `"true"` | Enable field schema validation (deprecated - now auto-inferred) |
| `ENABLE_LABEL_VALIDATION` | `"true"` | Enable label value validation against configured label_values |
| `ENABLE_LOGIC_VALIDATION` | `"true"` | Enable rule logic validation (tautologies, contradictions, type errors) |

#### Optimization Controls
| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_RULE_OPTIMIZATION` | `"true"` | Enable complexity-based rule reordering for execution efficiency |

### Job Arguments
None - all configuration via environment variables and input files

## Input Data Structure

### Expected Input Format
```
ruleset_configs/
├── label_config.json (required)
└── ruleset.json (required)
```

### label_config.json Structure

#### Single-Label Configuration (Binary/Multiclass)
```json
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
```

#### Multi-Label Configuration (Basic)
```json
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
```

#### Multi-Label Configuration (Per-Column)
```json
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
    "is_fraud_ACH": {"0": "No_Fraud", "1": "Low_Risk", "2": "High_Risk"}
  },
  "default_label": {
    "is_fraud_CC": 0,
    "is_fraud_DC": 0,
    "is_fraud_ACH": 0
  },
  "evaluation_mode": "priority",
  "sparse_representation": true
}
```

**Required Fields**:
- `output_label_name` (string or list): Column name(s) for output labels
- `output_label_type` (string): "binary", "multiclass", or "multilabel"
- `label_values` (list or dict): Valid label values (global or per-column)
- `label_mapping` (dict): Human-readable names for label values
- `default_label` (any or dict): Default when no rules match
- `evaluation_mode` (string): Rule evaluation strategy (typically "priority")

**Optional Fields**:
- `sparse_representation` (bool): Enable sparse encoding for multilabel (default: false)

### ruleset.json Structure

#### Single-Label Rules
```json
[
  {
    "rule_id": "rule_001",
    "name": "High confidence TrueDNR",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "all_of": [
        {"field": "category", "operator": "equals", "value": "TrueDNR"},
        {"field": "confidence_score", "operator": ">=", "value": 0.8}
      ]
    },
    "output_label": 0,
    "description": "High confidence TrueDNR cases indicate no reversal"
  }
]
```

#### Multi-Label Rules
```json
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
    "description": "High value credit card transactions flagged as fraud"
  },
  {
    "rule_id": "rule_multi_001",
    "name": "Suspicious pattern across methods",
    "priority": 2,
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
```

**Required Rule Fields**:
- `rule_id` (string): Unique rule identifier
- `name` (string): Human-readable rule name
- `priority` (int): Evaluation priority (lower = higher priority)
- `conditions` (dict): Nested condition expression
- `output_label` (any or dict): Label value(s) when rule matches

**Optional Rule Fields**:
- `enabled` (bool): Whether rule is active (default: true)
- `description` (string): Rule explanation

**Supported Condition Operators**:
- **Comparison**: `equals`, `not_equals`, `>`, `>=`, `<`, `<=`
- **Collection**: `in`, `not_in`
- **String**: `contains`, `not_contains`, `starts_with`, `ends_with`, `regex_match`
- **Null Check**: `is_null`, `is_not_null`
- **Logical**: `all_of` (AND), `any_of` (OR), `none_of` (NOT)

## Output Data Structure

### Output Directory Structure
```
validated_ruleset/
└── validated_ruleset.json

validation_report/
└── validation_report.json
```

### validated_ruleset.json Structure
```json
{
  "version": "1.0",
  "generated_timestamp": "2025-11-18T14:30:00.123456",
  "label_config": { /* same as input */ },
  "field_config": {
    "required_fields": ["category", "confidence_score", "amount"],
    "field_types": {
      "category": "string",
      "confidence_score": "float",
      "amount": "int"
    }
  },
  "ruleset": [ /* optimized rules with updated priorities */ ],
  "metadata": {
    "total_rules": 15,
    "enabled_rules": 14,
    "disabled_rules": 1,
    "field_usage": {
      "category": 8,
      "confidence_score": 5,
      "amount": 3
    },
    "validation_summary": {
      "field_validation": "passed_at_config_level",
      "label_validation": "passed",
      "logic_validation": "passed_with_warnings",
      "warnings": ["Rule 'always_true_rule' has always-true condition"]
    }
  }
}
```

**Key Output Components**:
- `version`: Ruleset format version (1.0)
- `generated_timestamp`: ISO 8601 timestamp
- `label_config`: Label configuration (preserved from input)
- `field_config`: Auto-inferred field schema with types
- `ruleset`: Optimized rules with updated priorities
- `metadata`: Comprehensive validation and optimization statistics

### validation_report.json Structure
```json
{
  "validation_status": "passed",
  "field_validation": {"passed_at_config_level": true},
  "label_validation": {
    "valid": true,
    "missing_fields": [],
    "invalid_labels": [],
    "uncovered_classes": [2],
    "conflicting_rules": [],
    "warnings": ["Label value 2 not covered by any rule"]
  },
  "logic_validation": {
    "valid": true,
    "tautologies": ["always_true_rule"],
    "contradictions": [],
    "unreachable_rules": [],
    "type_mismatches": [],
    "warnings": ["Rule 'always_true_rule' has always-true condition"]
  },
  "optimization_applied": true,
  "metadata": { /* same as validated_ruleset metadata */ }
}
```

## Key Functions and Tasks

### Configuration Loading Component

#### `main(input_paths, output_paths, environ_vars, job_args, logger)`
**Purpose**: Orchestrate the complete validation and optimization workflow

**Algorithm**:
```python
1. Load label_config.json from input directory
2. Load ruleset.json from input directory
3. Infer field_config from rules automatically
4. Assemble complete user_ruleset dictionary
5. Initialize validators (label, logic, coverage)
6. Run validation pipeline:
   a. Label validation (if enabled)
   b. Logic validation (if enabled)
   c. Coverage validation (multilabel only)
7. Check validation results - fail fast on errors
8. Optimize ruleset (if enabled)
9. Generate validated ruleset with metadata
10. Save validated_ruleset.json
11. Save validation_report.json
```

**Parameters**:
- `input_paths` (dict): Input directory paths
- `output_paths` (dict): Output file paths
- `environ_vars` (dict): Environment variable configuration
- `job_args` (Namespace): Command-line arguments (unused)
- `logger` (callable): Logging function

**Returns**: `dict` - Processing results with validated ruleset and report

### Field Inference Component

#### `infer_field_config_from_rules(rules, log)`
**Purpose**: Automatically infer complete field configuration from rule definitions

**Algorithm**:
```python
1. Initialize field_values dictionary
2. For each rule in rules:
   a. Extract conditions
   b. Call extract_fields_and_values() recursively
   c. Accumulate field names and values
3. For each field discovered:
   a. Call infer_field_type() based on values
   b. Count rule usage
4. Generate field_config:
   a. required_fields = all discovered fields
   b. field_types = inferred type dictionary
5. Log inference results
6. Return field_config dictionary
```

**Parameters**:
- `rules` (list): List of rule definitions
- `log` (callable): Logging function

**Returns**: `dict` - Complete field configuration with structure:
```python
{
    "required_fields": ["field1", "field2", ...],
    "field_types": {"field1": "string", "field2": "int", ...}
}
```

**Complexity**: O(n × m) where n = rules, m = avg fields per rule

#### `extract_fields_and_values(condition)`
**Purpose**: Recursively extract all field names and their values from nested conditions

**Algorithm**:
```python
1. Initialize field_values dictionary
2. Check condition type:
   a. If "all_of": recurse on each subcondition, merge results
   b. If "any_of": recurse on each subcondition, merge results
   c. If "none_of": recurse on each subcondition, merge results
   d. If leaf (has "field"): extract field and value
3. Return field_values dictionary
```

**Parameters**:
- `condition` (dict): Condition expression (may be nested)

**Returns**: `dict` - Mapping of field names to lists of values

#### `infer_field_type(values)`
**Purpose**: Infer field data type from observed values in conditions

**Type Priority Order**: `string` > `float` > `int` > `bool`

**Algorithm**:
```python
1. If no values, return "string" (default)
2. Collect types of all non-null values
3. Apply priority order:
   - If any string: return "string"
   - Elif any float: return "float"
   - Elif any int: return "int"
   - Elif any bool: return "bool"
4. Return "string" (fallback)
```

**Parameters**:
- `values` (list): List of values observed for a field

**Returns**: `str` - Inferred type: "string", "int", "float", or "bool"

### Label Validation Component

#### `RulesetLabelValidator.validate_labels(ruleset)`
**Purpose**: Validate all output labels match configured label values (supports multilabel)

**Algorithm**:
```python
1. Extract label_config from ruleset
2. Get label_values, label_type, default_label, output_label_name
3. Validate multilabel configuration structure:
   a. Check output_label_name is list
   b. Check for duplicate column names
   c. Validate per-column structures if used
4. Convert label_values to validation set
5. Validate default_label:
   a. For dict (per-column): validate each column's default
   b. For scalar (global): validate against global label_values
6. Validate all rules:
   a. Extract output_label
   b. For dict (multilabel):
      - Validate target columns exist
      - Validate values for each column
   c. For scalar (single-label):
      - Validate value in label_values
7. Check binary constraints (label_values should be [0, 1])
8. Identify uncovered label values (single-label only)
9. Check for conflicting rules (same priority, different outputs)
10. Return ValidationResult
```

**Parameters**:
- `ruleset` (dict): Complete ruleset with label_config and rules

**Returns**: `ValidationResult` - Validation status with detailed diagnostics

**Validation Checks**:
- Output labels exist in label_values
- Default label is valid
- Binary mode uses [0, 1] values
- No conflicting rules at same priority
- Uncovered label values (warnings)
- Multilabel structure correctness

### Logic Validation Component

#### `RulesetLogicValidator.validate_logic(ruleset)`
**Purpose**: Validate rule logic for common errors and inefficiencies

**Algorithm**:
```python
1. Extract rules and field_types from ruleset
2. For each rule:
   a. Check for tautologies (always-true conditions)
   b. Check for contradictions (always-false conditions)
   c. Validate operator-type compatibility
   d. Collect errors and warnings
3. Check for unreachable rules (shadowed by higher priority)
4. Return ValidationResult with all findings
```

**Parameters**:
- `ruleset` (dict): Complete ruleset configuration

**Returns**: `ValidationResult` - Logic validation status

**Validation Checks**:
- Tautologies (always-true conditions)
- Contradictions (always-false conditions)
- Type compatibility (numeric operators on numeric fields, etc.)
- Unreachable rules (simplified heuristic)

#### `_is_tautology(condition)`
**Purpose**: Detect always-true conditions (simplified check)

**Algorithm**:
```python
1. If condition is empty: return True
2. If operator is "is_not_null": return True (simplified)
3. Return False
```

**Note**: Simplified heuristic - full implementation would require field metadata

#### `_is_contradiction(condition)`
**Purpose**: Detect always-false conditions

**Algorithm**:
```python
1. If "all_of" in condition:
   a. Track field-value pairs
   b. For each subcondition with "equals":
      - If field seen with different value: return True (contradiction)
2. Return False
```

**Example**: `field = A AND field = B` where A ≠ B

#### `_check_type_compatibility(condition, field_types)`
**Purpose**: Validate operator compatibility with field types

**Algorithm**:
```python
1. Handle nested conditions recursively
2. For leaf conditions:
   a. Extract field, operator, value
   b. Look up field_type
   c. Check operator-type compatibility:
      - Numeric operators (>, >=, <, <=) require numeric types
      - String operators (contains, starts_with, regex_match) require string type
   d. Collect errors for mismatches
3. Return list of type compatibility errors
```

**Parameters**:
- `condition` (dict): Condition expression
- `field_types` (dict): Field type mapping

**Returns**: `list` - Type mismatch error messages

### Coverage Validation Component

#### `RuleCoverageValidator.validate(label_config, rules)`
**Purpose**: Validate rule coverage for all label columns (multilabel mode)

**Algorithm**:
```python
1. Check if multilabel mode, else return (not applicable)
2. Extract output columns from label_config
3. Collect covered columns:
   a. For each enabled rule:
      - If output_label is dict: add all keys to covered set
4. Calculate uncovered columns:
   uncovered = output_columns - covered_columns
5. Add warnings for uncovered columns
6. Return ValidationResult
```

**Parameters**:
- `label_config` (dict): Label configuration
- `rules` (list): Rule definitions

**Returns**: `ValidationResult` - Coverage validation status

### Optimization Component

#### `optimize_ruleset(ruleset, enable_complexity, enable_field_grouping, log)`
**Purpose**: Optimize ruleset using complexity-based ordering and field grouping

**Algorithm**:
```python
1. Deep copy rules to avoid mutation
2. If enable_complexity:
   a. Calculate complexity_score for each rule
   b. Sort rules by complexity (simple first)
   c. Log complexity scores
3. If enable_field_grouping:
   a. Extract used_fields for each rule
   b. Group rules with similar field usage (simplified)
4. Assign final priorities (1, 2, 3, ...)
5. Log priority changes
6. Return optimized ruleset with metadata
```

**Parameters**:
- `ruleset` (dict): Input ruleset
- `enable_complexity` (bool): Enable complexity-based ordering
- `enable_field_grouping` (bool): Enable field usage grouping
- `log` (callable): Logging function

**Returns**: `dict` - Optimized ruleset with reordered rules

**Optimization Strategies**:
1. **Complexity-Based Ordering**: Simple rules evaluated first
2. **Field Usage Grouping**: Rules using similar fields grouped together (optional)

#### `calculate_complexity(condition)`
**Purpose**: Calculate complexity score for a condition expression

**Algorithm**:
```python
1. If "all_of": return 1 + sum of subcondition complexities
2. If "any_of": return 1 + sum of subcondition complexities
3. If "none_of": return 1 + sum of subcondition complexities
4. Else (leaf condition):
   a. Base complexity = 1
   b. If regex_match: add 2
   c. If in/not_in with list: add len(list) // 10
   d. Return complexity
```

**Parameters**:
- `condition` (dict): Condition expression

**Returns**: `int` - Complexity score (higher = more complex)

**Complexity**: O(n) where n = total conditions in expression

### Analysis Component

#### `analyze_field_usage(rules)`
**Purpose**: Analyze field usage frequency across all rules

**Algorithm**:
```python
1. Initialize field_counts dictionary
2. For each rule:
   a. Extract all fields from conditions
   b. Increment count for each field
3. Sort fields by usage count (descending)
4. Return sorted dictionary
```

**Parameters**:
- `rules` (list): Rule definitions

**Returns**: `dict` - Field names mapped to usage counts

## Algorithms and Data Structures

### Field Type Inference Algorithm
**Problem**: Determine field data types without explicit schema, ensuring compatibility with rule operators

**Solution Strategy**:
1. Collect all values used for each field across all rules
2. Analyze value types using Python's type system
3. Apply priority order for mixed types (string > float > int > bool)
4. Use string as default fallback for safety

**Algorithm**:
```python
def infer_field_type(values: List[Any]) -> str:
    if not values:
        return "string"  # Safe default
    
    types_seen = set()
    for val in values:
        if val is None:
            continue
        if isinstance(val, bool):
            types_seen.add("bool")
        elif isinstance(val, int):
            types_seen.add("int")
        elif isinstance(val, float):
            types_seen.add("float")
        elif isinstance(val, str):
            types_seen.add("string")
    
    # Priority order: string > float > int > bool
    if "string" in types_seen:
        return "string"
    if "float" in types_seen:
        return "float"
    if "int" in types_seen:
        return "int"
    if "bool" in types_seen:
        return "bool"
    
    return "string"  # Fallback
```

**Complexity**: O(n) where n = number of values

**Key Features**:
- Prioritizes more general types (string can represent anything)
- Handles mixed types gracefully
- Null-safe (skips None values)
- Conservative default (string)

### Complexity-Based Rule Ordering Algorithm
**Problem**: Order rules to minimize average evaluation time by placing simpler rules first

**Solution Strategy**:
1. Calculate complexity score for each rule based on condition structure
2. Sort rules by complexity (ascending)
3. Assign new sequential priorities
4. Preserve original priorities in metadata

**Algorithm**:
```python
def calculate_complexity(condition: dict) -> int:
    # Recursive complexity calculation
    if "all_of" in condition:
        return 1 + sum(calculate_complexity(c) for c in condition["all_of"])
    elif "any_of" in condition:
        return 1 + sum(calculate_complexity(c) for c in condition["any_of"])
    elif "none_of" in condition:
        return 1 + sum(calculate_complexity(c) for c in condition["none_of"])
    else:
        # Leaf condition
        complexity = 1
        operator = condition.get("operator", "")
        value = condition.get("value")
        
        if operator == "regex_match":
            complexity += 2  # Regex is expensive
        elif operator in ("in", "not_in") and isinstance(value, list):
            complexity += len(value) // 10  # Large lists add complexity
        
        return complexity

# Sort rules by complexity
rules.sort(key=lambda r: calculate_complexity(r.get("conditions", {})))
```

**Complexity**: 
- Per-rule: O(c) where c = conditions in rule
- Total: O(n × c + n log n) where n = rules

**Key Features**:
- Recursive handling of nested conditions
- Accounts for expensive operators (regex, large lists)
- Logarithmic factor from sorting is negligible for typical rule counts

### ValidationResult Data Structure
**Purpose**: Aggregate validation results from multiple validators with detailed diagnostics

**Structure**:
```python
class ValidationResult:
    def __init__(self, valid: bool = True):
        self.valid = valid                    # Overall validation status
        self.missing_fields = []              # Required fields not in data
        self.undeclared_fields = []           # Fields used but not declared
        self.type_errors = []                 # Type validation errors
        self.invalid_labels = []              # Invalid output_label values
        self.uncovered_classes = []           # Label values without rules
        self.conflicting_rules = []           # Same priority, different outputs
        self.tautologies = []                 # Always-true conditions
        self.contradictions = []              # Always-false conditions
        self.unreachable_rules = []           # Rules shadowed by higher priority
        self.type_mismatches = []             # Operator-type incompatibilities
        self.warnings = []                    # Non-critical issues
```

**Key Features**:
- Separate tracking for errors (fail validation) vs warnings (informational)
- Detailed categorization enables targeted debugging
- Serializable to JSON via `__dict__()` method
- Supports accumulation from multiple validators

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Load Configuration | O(n) | O(n) | n = total JSON size |
| Field Inference | O(r × c) | O(f) | r = rules, c = avg conditions, f = unique fields |
| Label Validation | O(r × l) | O(l) | l = label columns (multilabel) |
| Logic Validation | O(r × c) | O(r) | Checks each condition in each rule |
| Type Compatibility | O(r × c) | O(1) | No additional storage |
| Complexity Calculation | O(r × c) | O(r) | Stores complexity per rule |
| Rule Sorting | O(r log r) | O(r) | Standard comparison sort |
| Field Usage Analysis | O(r × c) | O(f) | Tracks usage per field |

**Overall Complexity**: O(r × c + r log r) where r = rules, c = avg conditions per rule

**Typical Performance**:
- Small rulesets (< 100 rules): < 1 second
- Medium rulesets (100-1000 rules): 1-5 seconds
- Large rulesets (> 1000 rules): 5-30 seconds

### Memory Usage

**Peak Memory**: O(r × c) for storing rules and conditions in memory

**Optimization Opportunities**:
1. Streaming validation for very large rulesets
2. Parallel validation of independent rules
3. Caching complexity scores if revalidating

## Error Handling

### Error Types

#### Input Validation Errors
- **Missing Required Files**: `FileNotFoundError` when label_config.json or ruleset.json not found
  - **Handling**: Fail immediately with clear error message indicating which file is missing
  
- **JSON Parse Errors**: Invalid JSON syntax in configuration files
  - **Handling**: Propagate JSON parser exception with file context

#### Label Validation Errors
- **Invalid Output Labels**: output_label value not in label_values
  - **Handling**: Collect all invalid labels, add to ValidationResult, fail validation
  - **Example**: Rule outputs label 3 but label_values = [0, 1]

- **Invalid Default Label**: default_label not in label_values
  - **Handling**: Add to invalid_labels list, fail validation

- **Multilabel Structure Errors**: Incorrect multilabel configuration
  - **Handling**: Add to type_errors list, fail validation
  - **Examples**: output_label_name not a list, duplicate column names

#### Logic Validation Errors
- **Contradictions**: Always-false conditions (e.g., field = A AND field = B)
  - **Handling**: Add to contradictions list, fail validation

- **Type Mismatches**: Incompatible operator-type combinations
  - **Handling**: Add to type_mismatches list, fail validation
  - **Example**: Numeric operator > on string field

#### Processing Errors
- **File Write Errors**: Cannot write output files
  - **Handling**: Propagate OS error with context

### Error Response Structure
When validation fails, the script raises `RuntimeError` with message and writes detailed validation_report.json:

```json
{
  "validation_status": "failed",
  "label_validation": {
    "valid": false,
    "invalid_labels": [
      ["rule_001", 3, "not in label_values"]
    ],
    "type_errors": [],
    "warnings": []
  },
  "logic_validation": {
    "valid": false,
    "contradictions": ["rule_impossible"],
    "type_mismatches": [
      ["rule_002", "Numeric operator '>' on non-numeric field 'category' (type: string)"]
    ]
  }
}
```

## Best Practices

### For Production Deployments
1. **Enable All Validators**: Keep all validation flags enabled (field, label, logic) to catch errors early
2. **Review Validation Reports**: Always check validation_report.json for warnings even when validation passes
3. **Monitor Field Usage**: Use field_usage metadata to identify underutilized or overused fields
4. **Version Control Rulesets**: Store validated rulesets with version/timestamp for traceability
5. **Test with Sample Data**: Validate rules work correctly with execution step before production deployment

### For Development
1. **Start Simple**: Begin with single-label binary classification before moving to multilabel
2. **Iterative Refinement**: Add rules incrementally and validate after each addition
3. **Use Descriptive Names**: Provide clear rule names and descriptions for maintainability
4. **Check Uncovered Classes**: Address warnings about uncovered label values to ensure complete coverage
5. **Leverage Auto-Inference**: Let the script infer field types - no need to maintain separate field configs

### For Performance Optimization
1. **Enable Rule Optimization**: Keep ENABLE_RULE_OPTIMIZATION=true for complexity-based ordering
2. **Simplify Complex Rules**: Break down rules with very high complexity scores into simpler rules
3. **Remove Tautologies**: Address tautology warnings to eliminate unnecessary rule evaluations
4. **Review Priorities**: Ensure most commonly matched rules have lower priorities (evaluated first)
5. **Monitor Optimization Impact**: Check metadata.optimization_metadata to understand applied optimizations

## Example Configurations

### Binary Classification (Fraud Detection)
```bash
export ENABLE_FIELD_VALIDATION="true"
export ENABLE_LABEL_VALIDATION="true"
export ENABLE_LOGIC_VALIDATION="true"
export ENABLE_RULE_OPTIMIZATION="true"
```

**Input Files**:
- label_config.json: Binary classification (0=No_Fraud, 1=Fraud)
- ruleset.json: 50 rules with priority-based evaluation

**Use Case**: Credit card fraud detection with rule-based pre-screening

### Multiclass Classification (Risk Scoring)
```bash
export ENABLE_LABEL_VALIDATION="true"
export ENABLE_LOGIC_VALIDATION="true"
export ENABLE_RULE_OPTIMIZATION="true"
```

**Input Files**:
- label_config.json: Multiclass (0=Low, 1=Medium, 2=High, 3=Critical)
- ruleset.json: 100 rules with graduated risk levels

**Use Case**: Transaction risk assessment with multiple severity levels

### Multilabel Classification (Payment Methods)
```bash
export ENABLE_LABEL_VALIDATION="true"
export ENABLE_LOGIC_VALIDATION="true"
export ENABLE_RULE_OPTIMIZATION="true"
```

**Input Files**:
- label_config.json: Multilabel with 3 columns (is_fraud_CC, is_fraud_DC, is_fraud_ACH)
- ruleset.json: 75 rules with per-column and multi-column outputs

**Use Case**: Payment fraud detection across multiple payment methods simultaneously

## Integration Patterns

### Upstream Integration
```
Configuration Files (JSON)
   ↓ (label_config.json, ruleset.json)
LabelRulesetGeneration
   ↓ (validated_ruleset.json, validation_report.json)
```

**Input Sources**:
- Manual configuration files created by ML engineers
- Generated configuration from UI tools
- Exported rulesets from rule management systems

### Downstream Integration
```
LabelRulesetGeneration
   ↓ (validated_ruleset.json)
TabularPreprocessing
   ↓ (processed_data)
LabelRulesetExecution
   ↓ (labeled_data)
TrainingStep
```

**Output Consumers**:
- **LabelRulesetExecution**: Primary consumer - applies validated rules to data
- **Documentation Systems**: Use validation_report for rule quality monitoring
- **ML Pipeline Orchestrators**: Check validation_status before proceeding

### Workflow Example
1. **Rule Definition**: ML engineer defines classification rules in JSON format
2. **Validation & Optimization**: LabelRulesetGeneration validates and optimizes rules
3. **Data Preprocessing**: TabularPreprocessing prepares data with required fields
4. **Rule Execution**: LabelRulesetExecution applies validated rules to processed data
5. **Model Training**: Training step uses rule-generated labels

## Troubleshooting

### Invalid Output Labels

**Symptom**: Validation fails with "output_label not in label_values" errors

**Common Causes**:
1. **Typo in label values**: Rule outputs 2 but label_values = [0, 1]
2. **Wrong label type**: Using integers when strings expected or vice versa
3. **Outdated rules**: Rules reference old label values after config change

**Solution**:
1. Check validation_report.json for specific invalid labels
2. Verify label_values in label_config.json match rule outputs
3. Ensure type consistency (all integers or all strings)
4. Update rules to use current label_values

### Type Mismatch Errors

**Symptom**: Logic validation fails with operator-type compatibility errors

**Common Causes**:
1. **Numeric operator on string field**: Using > on categorical field
2. **String operator on numeric field**: Using contains on numeric field
3. **Incorrect field type inference**: Field inferred as wrong type

**Solution**:
1. Review type_mismatches in validation_report.json
2. Check inferred field_types in validated_ruleset field_config
3. Fix operator choices to match field types
4. Use appropriate operators: equals for strings, >, < for numbers

### Always-False Rules (Contradictions)

**Symptom**: Validation fails with contradictions detected

**Common Causes**:
1. **Field equals two different values**: category = "A" AND category = "B"
2. **Impossible range conditions**: amount > 100 AND amount < 50
3. **Copy-paste errors**: Duplicated conditions with conflicting values

**Solution**:
1. Check contradictions list in validation_report.json
2. Review rule logic for impossible conditions
3. Use OR (any_of) instead of AND (all_of) where appropriate
4. Fix or remove contradictory rules

### Uncovered Label Values

**Symptom**: Validation passes but warns about uncovered label values

**Common Causes**:
1. **Incomplete rule coverage**: No rules output specific label value
2. **Only default label used**: Relying solely on default_label
3. **Disabled rules**: Rules covering value are disabled

**Solution**:
1. Review uncovered_classes in validation_report.json
2. Add rules to cover all label values or adjust label_values
3. Enable relevant disabled rules if intentionally disabled
4. Verify default_label handles edge cases appropriately

### Multilabel Configuration Errors

**Symptom**: Multilabel validation fails with structure errors

**Common Causes**:
1. **output_label_name not a list**: Single string instead of list for multilabel
2. **Duplicate column names**: Same column name appears twice
3. **Missing per-column config**: Per-column structures incomplete

**Solution**:
1. Check type_errors in validation_report.json
2. Ensure output_label_name is list: ["col1", "col2", "col3"]
3. Verify no duplicate names in output_label_name
4. If using per-column config, ensure all columns covered

## References

### Related Scripts
- [`label_ruleset_execution.py`](label_ruleset_execution_script.md): Executes validated rulesets on processed data to generate labels (execution step)

### Related Documentation
- **Step Builder**: Step builder implementation and integration patterns
- **Config Class**: Configuration class for ruleset generation step
- **Contract**: [`src/cursus/steps/contracts/label_ruleset_generation_contract.py`](../../src/cursus/steps/contracts/label_ruleset_generation_contract.py)
- **Step Specification**: Specification defining inputs, outputs, and step behavior

### Related Design Documents
- **[Label Ruleset Generation Step Patterns](../1_design/label_ruleset_generation_step_patterns.md)**: Step builder patterns and integration architecture for ruleset generation
- **[Label Ruleset Generation Configuration Examples](../1_design/label_ruleset_generation_configuration_examples.md)**: Complete working examples for binary, multiclass, and multilabel classification
- **[Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md)**: Design for multilabel support with per-column configuration
- **[Label Ruleset Optimization Patterns](../1_design/label_ruleset_optimization_patterns.md)**: Optimization strategies for rule performance and execution efficiency
- **[Label Ruleset Execution Step Patterns](../1_design/label_ruleset_execution_step_patterns.md)**: Execution step patterns showing how validated rulesets are applied to data

### External References
- [Transparent Machine Learning](https://arxiv.org/abs/1811.10154): Research on interpretable rule-based classification
- [Rule-Based Classification Systems](https://en.wikipedia.org/wiki/Rule-based_system): Overview of rule-based system design principles
