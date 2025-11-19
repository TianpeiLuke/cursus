---
tags:
  - code
  - processing_script
  - label_execution
  - rule_evaluation
  - priority_based_evaluation
keywords:
  - label ruleset execution
  - rule evaluation
  - priority-based evaluation
  - field validation
  - multilabel execution
  - rule statistics
  - transparent classification
topics:
  - label execution
  - rule-based classification
  - execution validation
  - ML pipelines
language: python
date of note: 2025-11-18
---

# Label Ruleset Execution Script Documentation

## Overview

The `label_ruleset_execution.py` script applies validated classification rulesets to processed data to generate labels for ML training pipelines. It serves as the second stage in rule-based classification workflows, executing pre-validated rules with comprehensive error handling and statistics tracking.

The script performs execution-time field validation to ensure data compatibility, evaluates rules using priority-based first-match-wins strategy, and generates comprehensive execution statistics for monitoring and debugging. A key innovation is its support for stacked preprocessing patterns through shared processed_data directories, enabling seamless integration with upstream data transformation steps.

The script supports both single-label (binary/multiclass) and multi-label classification modes with configurable sparse/dense representation. It integrates seamlessly with LabelRulesetGeneration output and maintains full backward compatibility with existing pipelines.

## Purpose and Major Tasks

### Primary Purpose
Execute validated classification rulesets on processed data to generate labels with execution-time validation, priority-based evaluation, and comprehensive statistics tracking for transparent ML training pipelines.

### Major Tasks
1. **Validated Ruleset Loading**: Load pre-validated ruleset configuration from generation step
2. **Execution-Time Field Validation**: Verify all required fields exist in actual DataFrame before evaluation
3. **Data Quality Checks**: Identify fields with high null percentages (>50%) and warn users
4. **Priority-Based Rule Evaluation**: Apply rules in priority order with first-match-wins strategy
5. **Multilabel Support**: Handle both single-label and multi-label classification with sparse/dense modes
6. **Default Label Fallback**: Apply default label when no rules match a data row
7. **Fail-Safe Error Handling**: Continue evaluation on individual rule errors with comprehensive logging
8. **Statistics Tracking**: Track rule match counts, percentages, and label distributions per split
9. **Format Preservation**: Maintain input format (CSV/TSV/Parquet) in output files
10. **Multi-Split Processing**: Handle train/val/test splits for training jobs or single splits for other job types

## Script Contract

### Entry Point
```
label_ruleset_execution.py
```

### Input Paths
| Path | Location | Description |
|------|----------|-------------|
| `validated_ruleset` | `/opt/ml/processing/input/validated_ruleset` | Validated ruleset from generation step (JSON) |
| `input_data` | `/opt/ml/processing/input/data` | Processed data with train/val/test or single split |

**Expected Input Files**:
- `validated_ruleset/validated_ruleset.json` (required): Complete validated ruleset configuration
- `input_data/{split}/{split}_processed_data.{csv|tsv|parquet}` (required): Processed data files

### Output Paths
| Path | Location | Description |
|------|----------|-------------|
| `processed_data` | `/opt/ml/processing/output/processed_data` | Labeled data in same format as input |
| `execution_report` | `/opt/ml/processing/output/execution_report` | Execution statistics and diagnostics |

**Output Files**:
- `processed_data/{split}/{split}_processed_data.{csv|tsv|parquet}`: Original data + label column(s)
- `execution_report/execution_report.json`: Comprehensive execution statistics
- `execution_report/rule_match_statistics.json`: Detailed per-split rule match data

### Required Environment Variables
None - all configuration via files and job arguments

### Optional Environment Variables

#### Validation Controls
| Variable | Default | Description |
|----------|---------|-------------|
| `FAIL_ON_MISSING_FIELDS` | `"true"` | Fail execution if required fields missing in data |
| `ENABLE_RULE_MATCH_TRACKING` | `"true"` | Track detailed rule match statistics (can disable for performance) |
| `ENABLE_PROGRESS_LOGGING` | `"true"` | Enable detailed progress logging during execution |

#### Format Controls
| Variable | Default | Description |
|----------|---------|-------------|
| `PREFERRED_INPUT_FORMAT` | `""` | Prefer specific format when multiple exist: "csv", "tsv", or "parquet" |

### Job Arguments
| Argument | Required | Choices | Description |
|----------|----------|---------|-------------|
| `--job-type` | Yes | training, validation, testing, calibration | Determines splits to process |

**Job Type Behavior**:
- `training`: Processes train/, val/, test/ subdirectories
- `validation`: Processes validation/ subdirectory only
- `testing`: Processes testing/ subdirectory only
- `calibration`: Processes calibration/ subdirectory only

## Input Data Structure

### Expected Input Format
```
validated_ruleset/
└── validated_ruleset.json (from LabelRulesetGeneration)

input_data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}
├── val/
│   └── val_processed_data.{csv|tsv|parquet}
└── test/
    └── test_processed_data.{csv|tsv|parquet}
```

### validated_ruleset.json Structure

```json
{
  "version": "1.0",
  "generated_timestamp": "2025-11-18T14:30:00.123456",
  "label_config": {
    "output_label_name": "final_reversal_flag",
    "output_label_type": "binary",
    "label_values": [0, 1],
    "label_mapping": {
      "0": "No_Reversal",
      "1": "Reversal"
    },
    "default_label": 1,
    "evaluation_mode": "priority"
  },
  "field_config": {
    "required_fields": ["category", "confidence_score", "amount"],
    "field_types": {
      "category": "string",
      "confidence_score": "float",
      "amount": "int"
    }
  },
  "ruleset": [
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
      "description": "High confidence TrueDNR cases indicate no reversal",
      "complexity_score": 2
    }
  ],
  "metadata": {
    "total_rules": 15,
    "enabled_rules": 14,
    "disabled_rules": 1
  }
}
```

### Input Data Files Structure

**Single-Label Data** (from TabularPreprocessing or other steps):
```csv
txn_id,amount,payment_method,category,confidence_score
1,1200,CC,TrueDNR,0.85
2,500,DC,Disputed,0.65
3,8000,ACH,HighRisk,0.92
```

**Multi-Label Data** (from BedrockProcessing with LLM outputs):
```csv
txn_id,amount,payment_method,llm_category_cc,llm_category_dc,llm_category_ach
1,1200,CC,fraud,clean,clean
2,500,DC,clean,clean,clean
3,8000,ACH,clean,clean,fraud
```

## Output Data Structure

### Output Directory Structure
```
processed_data/
├── train/
│   └── train_processed_data.{csv|tsv|parquet}  # Original + label(s)
├── val/
│   └── val_processed_data.{csv|tsv|parquet}    # Original + label(s)
└── test/
    └── test_processed_data.{csv|tsv|parquet}   # Original + label(s)

execution_report/
├── execution_report.json
└── rule_match_statistics.json
```

### Single-Label Output Format

**Output Data** (original columns + new label column):
```csv
txn_id,amount,payment_method,category,confidence_score,final_reversal_flag
1,1200,CC,TrueDNR,0.85,0
2,500,DC,Disputed,0.65,1
3,8000,ACH,HighRisk,0.92,1
```

**Key Properties**:
- All original columns preserved
- New label column added (name from `output_label_name`)
- Same row order maintained
- Same data types preserved
- Format matches input (CSV→CSV, Parquet→Parquet)

### Multi-Label Output Format

**Sparse Representation** (default, NaN for non-matching):
```csv
txn_id,amount,payment_method,is_fraud_CC,is_fraud_DC,is_fraud_ACH
1,1200,CC,1,,
2,500,DC,,,0
3,8000,ACH,,,1
```

**Dense Representation** (all columns filled with default):
```csv
txn_id,amount,payment_method,is_fraud_CC,is_fraud_DC,is_fraud_ACH
1,1200,CC,1,0,0
2,500,DC,0,0,0
3,8000,ACH,0,0,1
```

### execution_report.json Structure

```json
{
  "ruleset_version": "1.0",
  "ruleset_timestamp": "2025-11-18T14:30:00.123456",
  "execution_timestamp": "2025-11-18T15:45:00.789012",
  "label_config": {
    "output_label_name": "final_reversal_flag",
    "output_label_type": "binary",
    "label_values": [0, 1],
    "default_label": 1
  },
  "split_statistics": {
    "train": {
      "total_rows": 10000,
      "label_distribution": {
        "0": 6500,
        "1": 3500
      },
      "execution_stats": {
        "total_evaluated": 10000,
        "rule_match_counts": {
          "rule_001": 6500,
          "rule_002": 2000,
          "rule_003": 1000
        },
        "default_label_count": 500,
        "rule_match_percentages": {
          "rule_001": 65.0,
          "rule_002": 20.0,
          "rule_003": 10.0
        },
        "default_label_percentage": 5.0
      }
    },
    "val": { /* similar structure */ },
    "test": { /* similar structure */ }
  },
  "total_rules_evaluated": 14
}
```

### Multi-Label Execution Statistics

```json
{
  "label_type": "multilabel",
  "total_evaluated": 1000,
  "per_column_statistics": {
    "is_fraud_CC": {
      "rule_match_counts": {
        "rule_cc_001": 50,
        "rule_multi_001": 30
      },
      "default_label_count": 20,
      "rule_match_percentages": {
        "rule_cc_001": 5.0,
        "rule_multi_001": 3.0
      },
      "default_label_percentage": 2.0
    },
    "is_fraud_DC": { /* similar structure */ },
    "is_fraud_ACH": { /* similar structure */ }
  }
}
```

## Key Functions and Tasks

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args, logger)`
**Purpose**: Orchestrate complete ruleset execution workflow with multi-split processing

**Algorithm**:
```python
1. Load validated_ruleset.json from input directory
2. Initialize RulesetFieldValidator
3. Initialize RuleEngine with validated ruleset
4. Determine splits to process based on job_type:
   - training → [train, val, test]
   - other → [job_type]
5. For each split:
   a. Find data file (CSV/TSV/Parquet with format detection)
   b. Load DataFrame with automatic format detection
   c. Validate fields exist in DataFrame
   d. If validation fails:
      - FAIL_ON_MISSING_FIELDS=true: raise error
      - FAIL_ON_MISSING_FIELDS=false: skip split, log warning
   e. Apply rules to generate labels via evaluate_batch()
   f. Compute label distribution
   g. Collect execution statistics
   h. Save labeled data in same format as input
   i. Reset engine statistics for next split
6. Generate execution_report.json with all statistics
7. Save rule_match_statistics.json for detailed analysis
8. Return processed splits dictionary
```

**Parameters**:
- `input_paths` (dict): Input directory paths
- `output_paths` (dict): Output directory paths
- `environ_vars` (dict): Environment variable configuration
- `job_args` (Namespace): Command-line arguments with job_type
- `logger` (callable): Logging function

**Returns**: `Dict[str, pd.DataFrame]` - Processed DataFrames by split name

**Complexity**: O(n × s × r) where n = total rows across splits, s = splits, r = avg active rules

### Field Validation Component

#### `RulesetFieldValidator`
**Purpose**: Validate field availability in actual data at execution time (post-generation validation)

**Key Methods**:

##### `validate_fields(ruleset, data_df)`
**Purpose**: Perform comprehensive execution-time field validation

**Algorithm**:
```python
1. Extract field_config from ruleset (required_fields, field_types)
2. Extract all enabled rules from ruleset
3. Collect all field references from rule conditions:
   a. Recursively traverse condition tree
   b. Extract field names from leaf conditions
   c. Build set of used_fields
4. Get available_fields from DataFrame columns
5. Check required fields:
   a. missing_required = required_fields - available_fields
   b. If missing_required: mark invalid, add to missing_fields
6. Check used fields:
   a. missing_used = used_fields - available_fields
   b. If missing_used: mark invalid, add to missing_fields
7. Data quality checks:
   a. For each used field in available fields:
      - Calculate null_percentage = nulls / total_rows
      - If null_percentage > 0.5: add warning
8. Return ValidationResult with status and diagnostics
```

**Parameters**:
- `ruleset` (dict): Complete validated ruleset
- `data_df` (pd.DataFrame): Actual DataFrame to validate

**Returns**: `Dict[str, Any]` - Validation results:
```python
{
    "valid": bool,           # Overall validation status
    "missing_fields": [],    # Fields required but not in data
    "warnings": []           # Data quality warnings
}
```

**Validation Checks**:
- Required fields exist in DataFrame
- Fields used in rules exist in DataFrame
- Null percentages below warning threshold

##### `_extract_fields_from_conditions(condition)`
**Purpose**: Recursively extract all field names from nested condition expressions

**Algorithm**:
```python
1. Initialize empty fields list
2. Check condition structure:
   a. If "all_of": recurse on each subcondition, extend fields
   b. If "any_of": recurse on each subcondition, extend fields
   c. If "none_of": recurse on each subcondition, extend fields
   d. If "field" (leaf): append field name to fields
3. Return collected fields list
```

**Complexity**: O(c) where c = total conditions in expression tree

### Rule Evaluation Engine Component

#### `RuleEngine`
**Purpose**: Evaluate validated rules against data rows to produce labels with multilabel support

**Initialization** `__init__(validated_ruleset)`:
```python
1. Extract label_config, field_config, ruleset, metadata
2. Filter to enabled rules only (already sorted by priority)
3. Determine label_type (binary, multiclass, multilabel)
4. Normalize output_label_name to list:
   - String → [name] for single-label
   - List → unchanged for multilabel
5. Extract sparse_representation setting (multilabel)
6. Extract default_label and evaluation_mode
7. Initialize statistics tracking dictionaries:
   - rule_match_counts: {col: {rule_id: 0}} for each column
   - default_label_counts: {col: 0} for each column
   - total_evaluated: 0
```

**Key Methods**:

##### `evaluate_row(row)`
**Purpose**: Evaluate rules against single row, dispatching to appropriate mode

**Algorithm**:
```python
1. Increment total_evaluated counter
2. Check label_type:
   a. If binary or multiclass: call _evaluate_row_single_label(row)
   b. If multilabel: call _evaluate_row_multilabel(row)
3. Return result (int/str for single-label, dict for multilabel)
```

**Returns**:
- Single-label mode: `int | str` - Label value
- Multilabel mode: `Dict[str, Any]` - Column → value mapping

##### `_evaluate_row_single_label(row)`
**Purpose**: Evaluate rules for single-label mode (backward compatible)

**Algorithm**:
```python
1. Get output column name (output_columns[0])
2. For each rule in priority order:
   a. Try to evaluate rule conditions
   b. If conditions satisfied:
      - Get rule_id and output_label
      - Increment rule_match_counts[output_col][rule_id]
      - Return output_label (FIRST MATCH WINS)
   c. If error during evaluation:
      - Log warning with rule_id and error
      - Continue to next rule (FAIL-SAFE)
3. If no rules matched:
   a. Increment default_label_counts[output_col]
   b. Return default_label
```

**Returns**: `int | str` - Assigned label value

**Complexity**: O(r × c) where r = active rules, c = avg conditions per rule (best case O(1) with first match)

##### `_evaluate_row_multilabel(row)`
**Purpose**: Evaluate rules for multilabel mode with sparse/dense representation

**Algorithm**:
```python
1. Initialize result dictionary:
   a. If sparse_representation: {col: NaN for col in output_columns}
   b. If dense: {col: default_label[col] for col in output_columns}
      - Handle per-column default_label if dict
      - Use global default_label if scalar
2. For each rule in priority order:
   a. Try to evaluate rule conditions
   b. If conditions not satisfied: continue
   c. If conditions satisfied:
      - Get output_label from rule
      - If output_label is dict (multilabel rule):
        * For each (col, value) in output_label.items():
          - Skip if col not in result
          - If column not yet set (is NaN or default):
            * Set result[col] = value
            * Increment rule_match_counts[col][rule_id]
   d. If error during evaluation:
      - Log warning with rule_id and error
      - Continue to next rule (FAIL-SAFE)
3. Fill remaining NaN columns (sparse mode only):
   a. For each col with NaN value:
      - Increment default_label_counts[col]
      - If dense mode: set to default_label
4. Return result dictionary
```

**Returns**: `Dict[str, Any]` - Column → label value mapping

**Key Features**:
- Priority-based evaluation (earlier rules win)
- Partial matching (rule can set subset of columns)
- Sparse/dense representation support
- Per-column statistics tracking

##### `evaluate_batch(df)`
**Purpose**: Evaluate rules for entire DataFrame efficiently

**Algorithm**:
```python
1. Check label_type:
   a. If binary or multiclass (single-label):
      - Apply evaluate_row to each row
      - Assign results to single output column
      - Return DataFrame with new column
   b. If multilabel:
      - Apply evaluate_row with result_type="expand"
      - Results is DataFrame with all label columns
      - Assign each result column to original DataFrame
      - Return DataFrame with new columns
2. Return modified DataFrame
```

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame

**Returns**: `pd.DataFrame` - DataFrame with label column(s) added

**Complexity**: O(n × r × c) where n = rows, r = active rules, c = avg conditions

##### `get_statistics()`
**Purpose**: Retrieve comprehensive execution statistics with multilabel support

**Algorithm**:
```python
1. Check label_type:
   a. If binary or multiclass (single-label):
      - Get output_col = output_columns[0]
      - Return dictionary with:
        * total_evaluated
        * rule_match_counts[output_col]
        * default_label_count[output_col]
        * rule_match_percentages (computed from counts)
        * default_label_percentage (computed from count)
   
   b. If multilabel:
      - Initialize stats with label_type, total_evaluated
      - For each col in output_columns:
        * Build col_stats dictionary:
          - rule_match_counts[col]
          - default_label_count[col]
          - rule_match_percentages (computed)
          - default_label_percentage (computed)
        * Add to per_column_statistics[col]
      - Return comprehensive multilabel statistics
2. Return statistics dictionary
```

**Returns**: `Dict[str, Any]` - Execution statistics

##### `_evaluate_conditions(conditions, row)`
**Purpose**: Recursively evaluate nested condition expressions

**Algorithm**:
```python
1. Check condition type:
   a. If "all_of": return all(_evaluate_conditions(c, row) for c in all_of)
   b. If "any_of": return any(_evaluate_conditions(c, row) for c in any_of)
   c. If "none_of": return not any(_evaluate_conditions(c, row) for c in none_of)
   d. Else (leaf condition): return _evaluate_leaf_condition(conditions, row)
```

**Parameters**:
- `conditions` (dict): Condition expression (may be nested)
- `row` (pd.Series): DataFrame row

**Returns**: `bool` - Whether conditions are satisfied

**Complexity**: O(c) where c = conditions in expression tree

##### `_evaluate_leaf_condition(condition, row)`
**Purpose**: Evaluate single field comparison condition

**Algorithm**:
```python
1. Extract field, operator, expected_value from condition
2. Check field exists in row:
   - If missing: return False
3. Get actual_value from row[field]
4. Handle null values:
   - If is_null(actual_value):
     * operator == "is_null": return True
     * operator == "is_not_null": return False
     * else: return False (null doesn't match comparisons)
5. Apply operator comparison:
   - Call _apply_operator(operator, actual_value, expected_value)
6. Return comparison result
```

**Parameters**:
- `condition` (dict): Single condition with field, operator, value
- `row` (pd.Series): DataFrame row

**Returns**: `bool` - Whether condition is satisfied

##### `_apply_operator(operator, actual, expected)`
**Purpose**: Apply comparison operator between actual and expected values

**Supported Operators**:

**Comparison**: `equals`, `not_equals`, `>`, `>=`, `<`, `<=`
```python
if operator == "equals": return actual == expected
if operator == ">": return float(actual) > float(expected)
```

**Collection**: `in`, `not_in`
```python
if operator == "in": return actual in expected
if operator == "not_in": return actual not in expected
```

**String**: `contains`, `not_contains`, `starts_with`, `ends_with`, `regex_match`
```python
if operator == "contains": return str(expected) in str(actual)
if operator == "regex_match": return bool(re.search(expected, str(actual)))
```

**Null**: `is_null`, `is_not_null`
```python
# Already handled in _evaluate_leaf_condition
return False or True
```

**Parameters**:
- `operator` (str): Comparison operator
- `actual` (Any): Actual value from data
- `expected` (Any): Expected value from rule

**Returns**: `bool` - Comparison result

**Raises**: `ValueError` if operator unsupported

### Format Detection and I/O Component

#### `_detect_file_format(file_path)`
**Purpose**: Detect file format based on file extension

**Algorithm**:
```python
1. Get file suffix (lowercase): .csv, .tsv, .parquet, etc.
2. Map suffix to format:
   - .csv, .csv.gz → "csv"
   - .tsv, .tsv.gz → "tsv"
   - .parquet, .pq → "parquet"
   - Unknown → "csv" (default fallback)
3. Return format string
```

**Returns**: `str` - File format: "csv", "tsv", or "parquet"

#### `_read_dataframe(file_path)`
**Purpose**: Read DataFrame from file with automatic format detection

**Algorithm**:
```python
1. Detect format: format = _detect_file_format(file_path)
2. Read based on format:
   - csv: pd.read_csv(file_path)
   - tsv: pd.read_csv(file_path, sep="\t")
   - parquet: pd.read_parquet(file_path)
3. Return DataFrame
```

**Parameters**:
- `file_path` (Path): Path to data file

**Returns**: `pd.DataFrame` - Loaded data

**Raises**: `ValueError` if format unsupported

#### `_write_dataframe(df, file_path, file_format)`
**Purpose**: Write DataFrame to file in specified format

**Algorithm**:
```python
1. Create parent directories if needed
2. Write based on format:
   - csv: df.to_csv(file_path, index=False)
   - tsv: df.to_csv(file_path, sep="\t", index=False)
   - parquet: df.to_parquet(file_path, index=False)
3. Handle errors
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to write
- `file_path` (Path): Output file path
- `file_format` (str): Format to write

**Raises**: `ValueError` if format unsupported

## Algorithms and Data Structures

### Priority-Based First-Match-Wins Algorithm
**Problem**: Efficiently evaluate rules to assign labels, minimizing evaluation time while ensuring deterministic behavior

**Solution Strategy**:
1. Sort rules by priority during validation (done in generation step)
2. Evaluate rules sequentially in priority order
3. Return immediately on first match (early termination)
4. Use default label if no matches

**Algorithm**:
```python
def evaluate_row_first_match(row, active_rules, default_label):
    for rule in active_rules:  # Already sorted by priority
        try:
            if evaluate_conditions(rule.conditions, row):
                return rule.output_label  # FIRST MATCH WINS
        except Exception as e:
            log_warning(rule.rule_id, e)
            continue  # FAIL-SAFE: continue on error
    
    return default_label  # NO MATCH: use default
```

**Complexity**:
- Best case: O(c₁) - First rule matches, c₁ = conditions in first rule
- Average case: O(m × c̄) - m rules evaluated before match, c̄ = avg conditions
- Worst case: O(r × c̄) - All rules evaluated, no match

**Key Features**:
- Deterministic: Same input always produces same output
- Efficient: Early termination on match
- Fail-safe: Continues on individual rule errors
- Optimizable: Rule ordering affects performance

### Recursive Condition Evaluation Algorithm
**Problem**: Evaluate arbitrarily nested logical expressions (AND/OR/NOT) efficiently

**Solution Strategy**:
1. Recursive descent through condition tree
2. Short-circuit evaluation for AND/OR
3. Cache-friendly leaf evaluation

**Algorithm**:
```python
def evaluate_conditions(condition, row):
    # Logical operators (recursive)
    if "all_of" in condition:
        # Short-circuit AND: stop on first False
        return all(evaluate_conditions(c, row) for c in condition["all_of"])
    
    elif "any_of" in condition:
        # Short-circuit OR: stop on first True
        return any(evaluate_conditions(c, row) for c in condition["any_of"])
    
    elif "none_of" in condition:
        # Short-circuit NOT: stop on first True, then negate
        return not any(evaluate_conditions(c, row) for c in condition["none_of"])
    
    # Leaf condition (base case)
    else:
        return evaluate_leaf_condition(condition, row)
```

**Complexity**: O(c) where c = conditions in tree

**Key Features**:
- Short-circuit evaluation (Python's all/any built-ins)
- Handles arbitrary nesting depth
- Efficient for common cases (shallow trees)
- Memory efficient (no explicit stack needed)

### Statistics Tracking Data Structure
**Problem**: Track rule match statistics efficiently during evaluation without impacting performance

**Solution Strategy**:
1. In-memory dictionaries with O(1) updates
2. Per-column tracking for multilabel support
3. Lazy percentage computation (only on get_statistics())

**Data Structure**:
```python
class RuleEngine:
    def __init__(self, validated_ruleset):
        # Per-column statistics tracking
        self.rule_match_counts = {
            col: {rule_id: 0 for rule in active_rules}
            for col in output_columns
        }
        # Nested dict: column → rule_id → count
        
        self.default_label_counts = {
            col: 0 for col in output_columns
        }
        # Dict: column → count
        
        self.total_evaluated = 0  # Global counter
    
    def _update_stats(self, col, rule_id):
        # O(1) update during evaluation
        self.rule_match_counts[col][rule_id] += 1
    
    def get_statistics(self):
        # O(r × l) where r = rules, l = label columns
        # Compute percentages from counts
        return {
            col: {
                "rule_match_percentages": {
                    rid: (count / total * 100)
                    for rid, count in self.rule_match_counts[col].items()
                }
            }
            for col in output_columns
        }
```

**Complexity**:
- Update: O(1) per rule match
- Retrieval: O(r × l) where r = rules, l = label columns
- Space: O(r × l) for storage

**Key Features**:
- Minimal evaluation overhead (simple counter increments)
- Supports multilabel with per-column tracking
- Lazy computation of derived metrics
- Memory efficient for typical rule counts (<1000 rules)

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Load Ruleset | O(r) | O(r) | r = rules in ruleset |
| Field Validation | O(f + c) | O(f) | f = fields, c = conditions |
| Single Row Evaluation | O(r × c̄) | O(1) | Early termination on match |
| Batch Evaluation | O(n × r × c̄) | O(n) | n = rows, c̄ = avg conditions |
| Statistics Collection | O(r × l) | O(r × l) | l = label columns |
| Format Detection | O(1) | O(1) | File extension check |
| Data I/O | O(n × m) | O(n × m) | n = rows, m = columns |

**Overall Complexity**: O(n × r × c̄) where n = rows, r = active rules, c̄ = avg conditions per rule

**Typical Performance**:
- Small datasets (<10K rows): < 5 seconds
- Medium datasets (10K-100K rows): 5-30 seconds  
- Large datasets (>100K rows): 30-300 seconds (depends on rule complexity)

### Memory Usage

**Peak Memory**: O(n × m + r × l) for DataFrame + statistics

- DataFrame: O(n × m) where n = rows, m = columns
- Statistics: O(r × l) where r = rules, l = label columns
- Rule engine: O(r × c̄) for active rules and conditions

**Optimization Opportunities**:
1. Chunked processing for very large datasets (>1M rows)
2. Rule complexity optimization (simpler rules evaluated first)
3. Disable statistics tracking for performance-critical scenarios
4. Parallel split processing (train/val/test independently)

## Error Handling

### Error Types

#### Execution-Level Errors
- **Missing Ruleset File**: `FileNotFoundError` when validated_ruleset.json not found
  - **Handling**: Fails immediately with clear error message indicating missing file
  
- **Invalid Ruleset Format**: JSON parse errors or missing required keys
  - **Handling**: Propagates JSON parser exception or KeyError with context

- **Missing Split Directories**: Expected split directory (train/val/test) not found
  - **Handling**: Logs warning, continues with other splits (graceful degradation)

#### Field Validation Errors
- **Missing Required Fields**: Fields in field_config.required_fields not in DataFrame
  - **Handling**: Controlled by FAIL_ON_MISSING_FIELDS environment variable
  - **true** (default): Raises ValueError with list of missing fields
  - **false**: Logs warning, skips split, continues with other splits

- **Missing Used Fields**: Fields referenced in rules not in DataFrame  
  - **Handling**: Same as missing required fields (controlled by FAIL_ON_MISSING_FIELDS)

- **High Null Percentages**: Fields with >50% null values
  - **Handling**: Logs warning but continues execution (data quality alert)

#### Rule-Level Errors
- **Type Conversion Errors**: Cannot convert value types for comparison (e.g., int > string)
  - **Handling**: Logs warning with rule_id and error, continues with next rule (fail-safe)

- **Null Value Handling**: Null values in data for field comparisons
  - **Handling**: Returns False for non-null operators, handles is_null/is_not_null explicitly

- **Unsupported Operators**: Operator not in supported list
  - **Handling**: Raises ValueError with operator name

#### I/O Errors
- **File Read Errors**: Cannot read input data file
  - **Handling**: Propagates pandas read error with file path context

- **File Write Errors**: Cannot write output files
  - **Handling**: Propagates OS error with context

- **Format Detection Errors**: Unsupported file format
  - **Handling**: Falls back to CSV format, logs warning

### Error Response Structure

When execution fails, detailed error information is logged:

```json
{
  "error_type": "FieldValidationError",
  "error_message": "Required fields missing in data",
  "missing_fields": ["category", "confidence_score"],
  "split": "train",
  "ruleset_version": "1.0",
  "total_fields_required": 5,
  "fields_available": ["txn_id", "amount", "payment_method"]
}
```

## Best Practices

### For Production Deployments
1. **Enable All Validation**: Keep FAIL_ON_MISSING_FIELDS=true to catch field mismatches early
2. **Monitor Statistics**: Review execution_report.json for rule match rates and default label usage
3. **Check Data Quality**: Address high null percentage warnings before production
4. **Version Control Rulesets**: Track ruleset versions and execution timestamps for auditability
5. **Test with Sample Data**: Validate rules work correctly on representative data before production

### For Development
1. **Start with Lenient Validation**: Use FAIL_ON_MISSING_FIELDS=false for exploratory development
2. **Review Statistics**: Use rule_match_statistics.json to identify unused or overused rules
3. **Iterate on Rules**: Use execution statistics to optimize rule priority order
4. **Test Edge Cases**: Verify default_label behavior when no rules match
5. **Check Label Distributions**: Ensure label distributions are reasonable for training

### For Performance Optimization
1. **Optimize Rule Order**: Place most frequently matched rules at lower priorities (evaluated first)
2. **Simplify Complex Rules**: Break down rules with high complexity scores
3. **Disable Tracking**: Set ENABLE_RULE_MATCH_TRACKING=false for performance-critical scenarios
4. **Use Parquet Format**: Parquet is faster for large datasets than CSV
5. **Process Splits in Parallel**: Run separate jobs for train/val/test if needed

## Example Configurations

### Basic Binary Classification
```bash
export FAIL_ON_MISSING_FIELDS="true"
export ENABLE_RULE_MATCH_TRACKING="true"
python label_ruleset_execution.py --job-type training
```

**Use Case**: Standard fraud detection with strict field validation

### Lenient Field Validation
```bash
export FAIL_ON_MISSING_FIELDS="false"
export ENABLE_RULE_MATCH_TRACKING="true"
python label_ruleset_execution.py --job-type training
```

**Use Case**: Exploratory development where some preprocessing steps may be skipped

### Performance-Optimized
```bash
export FAIL_ON_MISSING_FIELDS="true"
export ENABLE_RULE_MATCH_TRACKING="false"
export ENABLE_PROGRESS_LOGGING="false"
python label_ruleset_execution.py --job-type training
```

**Use Case**: Production deployment with large datasets where performance is critical

### Single Split Processing
```bash
export FAIL_ON_MISSING_FIELDS="true"
python label_ruleset_execution.py --job-type validation
```

**Use Case**: Labeling validation split only for model evaluation

### Multilabel Classification
```bash
export FAIL_ON_MISSING_FIELDS="true"
export ENABLE_RULE_MATCH_TRACKING="true"
python label_ruleset_execution.py --job-type training
```

**Input**: Ruleset with `output_label_type: "multilabel"` and multiple output columns

**Use Case**: Payment fraud detection across multiple payment methods simultaneously

## Integration Patterns

### Upstream Integration
```
TabularPreprocessing → processed_data
   ↓
BedrockProcessing → processed_data (adds LLM outputs)
   ↓
LabelRulesetGeneration → validated_ruleset (parallel)
   ↓
LabelRulesetExecution (depends on both)
```

**Input Sources**:
- **validated_ruleset**: From LabelRulesetGeneration step (JSON configuration)
- **processed_data**: From preprocessing steps (TabularPreprocessing, BedrockProcessing, etc.)

### Downstream Integration
```
LabelRulesetExecution → processed_data (with labels)
   ↓
StratifiedSampling → processed_data (balanced)
   ↓
TrainingStep (XGBoost/LightGBM/PyTorch)
```

**Output Consumers**:
- **TrainingStep**: Primary consumer - uses labeled data for model training
- **StratifiedSampling**: May balance label distributions before training
- **Active Learning Pipelines**: May use labels to identify uncertain samples

### Stacked Preprocessing Pattern

The script enables seamless preprocessing pipeline composition through shared `processed_data` directories:

```python
# Step 1: Tabular preprocessing
preprocessing_step = TabularPreprocessingStepBuilder(config).create_step()

# Step 2: Bedrock processing (adds LLM categorization)
bedrock_step = BedrockBatchProcessingStepBuilder(config).create_step(
    inputs={'processed_data': preprocessing_step.properties...}
)

# Step 3: Generate validated ruleset (runs in parallel)
ruleset_gen_step = LabelRulesetGenerationStepBuilder(config).create_step()

# Step 4: Execute ruleset (depends on both Bedrock and Generator)
ruleset_exec_step = LabelRulesetExecutionStepBuilder(config).create_step(
    inputs={
        'validated_ruleset': ruleset_gen_step.properties...,
        'processed_data': bedrock_step.properties...  # Uses LLM outputs
    },
    dependencies=[ruleset_gen_step, bedrock_step]
)

# Step 5: Training (uses labeled data)
training_step = XGBoostTrainingStepBuilder(config).create_step(
    inputs={'training_data': ruleset_exec_step.properties...}
)
```

### Workflow Example
1. **TabularPreprocessing**: Cleans and transforms raw data
2. **BedrockProcessing**: Adds LLM categorization columns (llm_category_cc, llm_category_dc)
3. **LabelRulesetGeneration**: Validates rules that reference LLM outputs (parallel)
4. **LabelRulesetExecution**: Applies rules to generate labels based on LLM outputs + other fields
5. **Training**: Uses labeled data to train XGBoost model

## Troubleshooting

### Missing Field Errors

**Symptom**: Execution fails with "Required fields missing in data" errors

**Common Causes**:
1. **Upstream preprocessing skipped**: Expected preprocessing step didn't run
2. **Field renamed in preprocessing**: Field names changed from ruleset expectations
3. **Wrong input directory**: pointed to wrong data source

**Solution**:
1. Check field names in validation_result.missing_fields
2. Verify upstream preprocessing steps completed successfully
3. Ensure field_config.required_fields matches actual data schema
4. Use FAIL_ON_MISSING_FIELDS=false for graceful degradation during debugging

### No Rules Matching

**Symptom**: All rows assigned default_label (default_label_percentage = 100%)

**Common Causes**:
1. **Rule conditions too strict**: No data rows satisfy any rule conditions
2. **Wrong operator types**: Using numeric operators on string fields
3. **Field value mismatches**: Expected values don't match actual data
4. **All rules disabled**: All rules have enabled=false

**Solution**:
1. Review rule_match_statistics.json to see which rules matched
2. Check rule conditions against actual data values
3. Verify field types in field_config match actual data types
4. Ensure at least some rules have enabled=true
5. Use simpler rules first to debug matching issues

### Label Distribution Imbalance

**Symptom**: Label distribution heavily skewed (e.g., 95% class 0, 5% class 1)

**Common Causes**:
1. **Rule priorities favor one class**: High-priority rules all output same label
2. **Default label dominates**: Most rows don't match any rules
3. **Data distribution issue**: Actual data is naturally imbalanced

**Solution**:
1. Review rule_match_percentages in execution_report.json
2. Reorder rules to balance matches across labels
3. Add more rules to cover underrepresented classes
4. Consider downstream stratified sampling if data is naturally imbalanced
5. Check if default_label should be different value

### Format Detection Errors

**Symptom**: Wrong file format detected or read errors

**Common Causes**:
1. **Multiple formats in directory**: Both CSV and Parquet files present
2. **Inconsistent extensions**: File extension doesn't match content
3. **Compressed files**: Gzipped files not detected properly

**Solution**:
1. Ensure only one data file per split directory
2. Use PREFERRED_INPUT_FORMAT to specify format explicitly
3. Check file extensions match actual format
4. For compressed files, use .csv.gz or .tsv.gz extensions
5. Remove unexpected files from split directories

### Multilabel Statistics Issues

**Symptom**: Per-column statistics don't sum correctly or NaN values in output

**Common Causes**:
1. **Sparse representation misunderstood**: NaN is intentional for sparse mode
2. **Per-column configuration mismatch**: Rules don't cover all columns
3. **Rule output format wrong**: Rules should output dicts for multilabel

**Solution**:
1. Check sparse_representation setting in label_config
2. Use dense mode (sparse_representation=false) if NaN problematic
3. Verify rules output dictionaries: {"is_fraud_CC": 1, "is_fraud_DC": 0}
4. Review per_column_statistics in execution_report.json
5. Ensure all columns have at least one rule or default coverage

## References

### Related Scripts
- [`label_ruleset_generation.py`](label_ruleset_generation_script.md): Validates and optimizes rulesets before execution (generation step)
- [`pseudo_label_merge.py`](pseudo_label_merge_script.md): Merges labeled data with pseudo-labeled samples for SSL pipelines
- [`active_sample_selection.py`](active_sample_selection_script.md): Selects high-value samples from labeled data for active learning

### Related Documentation
- **Step Builder**: Step builder implementation and integration patterns
- **Config Class**: Configuration class for ruleset execution step
- **Contract**: [`src/cursus/steps/contracts/label_ruleset_execution_contract.py`](../../src/cursus/steps/contracts/label_ruleset_execution_contract.py)
- **Step Specification**: Specification defining inputs, outputs, and step behavior

### Related Design Documents
- **[Label Ruleset Execution Step Patterns](../1_design/label_ruleset_execution_step_patterns.md)**: Step builder patterns and integration architecture for ruleset execution
- **[Label Ruleset Generation Step Patterns](../1_design/label_ruleset_generation_step_patterns.md)**: Generation step patterns showing how rulesets are validated before execution
- **[Label Ruleset Optimization Patterns](../1_design/label_ruleset_optimization_patterns.md)**: Optimization strategies applied during generation that affect execution performance
- **[Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md)**: Design for multilabel support in execution
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation strategy used in execution

### External References
- [Transparent Machine Learning](https://arxiv.org/abs/1811.10154): Research on interpretable rule-based classification
- [First-Match Evaluation Strategies](https://en.wikipedia.org/wiki/Rule-based_system#Inference_engines): Background on priority-based rule evaluation
- [Pandas DataFrame Operations](https://pandas.pydata.org/docs/user_guide/basics.html): DataFrame manipulation used in evaluation
