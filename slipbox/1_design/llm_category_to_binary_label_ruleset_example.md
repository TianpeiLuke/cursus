---
tags:
  - design
  - implementation
  - label_ruleset
  - configuration_examples
  - documentation
  - user_guide
keywords:
  - category to label mapping
  - binary classification  
  - rule-based labeling
  - LLM output transformation
  - reversal flag generation
topics:
  - configuration examples
  - real-world use cases
  - label ruleset patterns
language: json, python
date of note: 2025-11-09
---

# LLM Category to Binary Label Mapping: Complete Ruleset Configuration Example

This document provides a complete, working example showing how to use the **Label Ruleset Generation and Execution** system to transform LLM classification outputs (`llm_category`) into binary labels (`llm_reversal_flag`) for downstream modeling.

## Overview

This example demonstrates:
- **Binary label generation** from 13 categorical LLM outputs
- **Category-based mapping rules** with clear business logic  
- **Two-step process**: Generation (validation) → Execution (labeling)
- **Complete JSON configurations** for all components
- **Integration with Bedrock Processing** outputs

## Category to Reversal Flag Mapping

### Categories → llm_reversal_flag = 0 (No Reversal)

These categories indicate NO financial reversal is needed:

1. **TrueDNR**: Package delivered but buyer claims non-receipt (fraud indicator)
2. **PDA_Undeliverable**: Package stuck in transit (logistics issue)
3. **PDA_Early_Refund**: Refund given before delivery (abuse pattern)
4. **Returnless_Refund**: Refund without return requirement (goodwill gesture)

### Categories → llm_reversal_flag = 1 (Reversal Required)

These categories indicate financial reversal IS needed:

1. **Confirmed_Delay**: External delays confirmed (refund justified)
2. **Delivery_Attempt_Failed**: Failed delivery with return to sender
3. **Seller_Unable_To_Ship**: Seller cannot fulfill order
4. **Buyer_Received_WrongORDefective_Item**: Quality issues requiring refund
5. **BuyerCancellation**: Buyer cancels before delivery
6. **Return_NoLongerNeeded**: Post-delivery return for unwanted item
7. **Product_Information_Support**: Support queries (potential refund)
8. **Insufficient_Information**: Unable to classify (flag for review)

## Part 1: Label Configuration

Create `label_config.json` defining the output label structure:

```json
{
  "output_label_name": "llm_reversal_flag",
  "output_label_type": "binary",
  "label_values": [0, 1],
  "label_mapping": {
    "0": "No_Reversal_Required",
    "1": "Reversal_Required"
  },
  "default_label": 1,
  "evaluation_mode": "priority"
}
```

**Configuration Details:**
- `output_label_name`: Name of the new column to create
- `output_label_type`: "binary" enforces only 0/1 values
- `label_values`: Valid numerical labels (must match output_label in rules)
- `label_mapping`: Human-readable names for each label
- `default_label`: Used when no rules match (1 = conservative, flag for review)
- `evaluation_mode`: "priority" means first matching rule wins

## Part 2: Rule Definitions (Efficient Approach with IN Operator)

Create `rules.json` with category mapping rules using the **IN operator** for efficiency:

```json
[
  {
    "name": "No_Reversal_Categories",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "field": "llm_category",
      "operator": "in",
      "value": [
        "TrueDNR",
        "PDA_Undeliverable",
        "PDA_Early_Refund",
        "Returnless_Refund"
      ]
    },
    "output_label": 0,
    "description": "Categories indicating NO reversal: delivered/fraud patterns, logistics delays, goodwill refunds"
  },
  {
    "name": "Reversal_Required_Categories",
    "priority": 2,
    "enabled": true,
    "conditions": {
      "field": "llm_category",
      "operator": "in",
      "value": [
        "Confirmed_Delay",
        "Delivery_Attempt_Failed",
        "Seller_Unable_To_Ship",
        "Buyer_Received_WrongORDefective_Item",
        "BuyerCancellation",
        "Return_NoLongerNeeded",
        "Product_Information_Support",
        "Insufficient_Information"
      ]
    },
    "output_label": 1,
    "description": "Categories indicating reversal required: legitimate refunds, quality issues, cancellations, manual review cases"
  }
]
```

**Benefits of IN Operator Approach:**
- **Fewer rules**: 2 rules instead of 12 (easier to maintain)
- **Clearer grouping**: Categories are visually grouped by label
- **Better performance**: Single evaluation per label group
- **Easier updates**: Add new categories to the appropriate list

**Alternative: Individual Rules per Category**

If you prefer explicit mapping for better traceability in statistics:

```json
[
  {
    "name": "TrueDNR_No_Reversal",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "field": "llm_category",
      "operator": "equals",
      "value": "TrueDNR"
    },
    "output_label": 0,
    "description": "Package delivered but buyer claims non-receipt - fraud indicator, no reversal"
  },
  {
    "name": "PDA_Undeliverable_No_Reversal",
    "priority": 2,
    "enabled": true,
    "conditions": {
      "field": "llm_category",
      "operator": "equals",
      "value": "PDA_Undeliverable"
    },
    "output_label": 0,
    "description": "Package stuck in transit - logistics issue, no reversal yet"
  }
  // ... 10 more rules ...
]
```

**Trade-offs:**
- Individual rules: Better statistics tracking per category, but more rules
- IN operator: More efficient, fewer rules, but rule match statistics show group counts

**Rule Structure:**
- `name`: Human-readable rule identifier
- `priority`: Lower number = higher priority (evaluated first)
- `enabled`: Can disable rules without deleting them
- `conditions`: Field comparison logic (supports nested AND/OR/NOT)
- `output_label`: Must be in `label_values` from label_config
- `description`: Explains the business logic

## Part 3: Optional Enhanced Rules with Confidence Filtering

For higher accuracy, add confidence thresholds (optional):

```json
[
  {
    "name": "TrueDNR_High_Confidence_No_Reversal",
    "priority": 1,
    "enabled": true,
    "conditions": {
      "all_of": [
        {
          "field": "llm_category",
          "operator": "equals",
          "value": "TrueDNR"
        },
        {
          "field": "llm_confidence_score",
          "operator": ">=",
          "value": 0.8
        }
      ]
    },
    "output_label": 0,
    "description": "High-confidence TrueDNR - strong fraud indicator, no reversal"
  },
  {
    "name": "Low_Confidence_Manual_Review",
    "priority": 13,
    "enabled": true,
    "conditions": {
      "field": "llm_confidence_score",
      "operator": "<",
      "value": 0.5
    },
    "output_label": 1,
    "description": "Low confidence classification - flag for manual review"
  }
]
```

**Nested Conditions:**
- `all_of`: All conditions must be true (AND logic)
- `any_of`: At least one condition must be true (OR logic)
- `none_of`: All conditions must be false (NOT logic)

## Part 4: Python DAG Configuration for Ruleset Generation

Configure the ruleset generation step in your DAG using the **efficient IN operator approach**:

```python
from pathlib import Path
from src.cursus.steps.configs.config_label_ruleset_generation_step import (
    LabelRulesetGenerationConfig,
    LabelConfig,
    RuleDefinition,
    RuleCondition,
    ComparisonOperator
)

# ============================================================================
# STEP 1: Define Label Configuration
# ============================================================================

label_config = LabelConfig(
    output_label_name="llm_reversal_flag",
    output_label_type="binary",
    label_values=[0, 1],
    label_mapping={
        "0": "No_Reversal_Required",
        "1": "Reversal_Required"
    },
    default_label=1,  # Conservative: flag for review if no rules match
    evaluation_mode="priority"
)

# ============================================================================
# STEP 2: Define Mapping Rules Using IN Operator (Efficient Approach)
# ============================================================================

# Rule 1: Categories indicating NO reversal (output_label = 0)
rule_no_reversal = RuleDefinition(
    name="No_Reversal_Categories",
    priority=1,
    enabled=True,
    conditions=RuleCondition(
        field="llm_category",
        operator=ComparisonOperator.IN,
        value=[
            "TrueDNR",
            "PDA_Undeliverable",
            "PDA_Early_Refund",
            "Returnless_Refund"
        ]
    ),
    output_label=0,
    description="Categories indicating NO reversal: delivered/fraud patterns, logistics delays, goodwill refunds"
)

# Rule 2: Categories indicating reversal required (output_label = 1)
rule_reversal_required = RuleDefinition(
    name="Reversal_Required_Categories",
    priority=2,
    enabled=True,
    conditions=RuleCondition(
        field="llm_category",
        operator=ComparisonOperator.IN,
        value=[
            "Confirmed_Delay",
            "Delivery_Attempt_Failed",
            "Seller_Unable_To_Ship",
            "Buyer_Received_WrongORDefective_Item",
            "BuyerCancellation",
            "Return_NoLongerNeeded",
            "Product_Information_Support",
            "Insufficient_Information"
        ]
    ),
    output_label=1,
    description="Categories indicating reversal required: legitimate refunds, quality issues, cancellations, manual review cases"
)

# Collect all rules (just 2 rules now!)
all_rules = [rule_no_reversal, rule_reversal_required]

# ============================================================================
# STEP 3: Configure Ruleset Generation Step
# ============================================================================

step_name = "LabelRulesetGeneration"

factory.set_step_config(
    step_name,
    
    # Label configuration (Pydantic model)
    label_settings=label_config,
    
    # Rule definitions (list of Pydantic models)
    rule_definitions=all_rules,
    
    # Validation configuration
    enable_field_validation=True,
    enable_label_validation=True,
    enable_logic_validation=True,
    fail_on_validation_warning=False,
    
    # Optimization configuration
    enable_rule_optimization=True,
    optimize_by_frequency=False,
    
    # Sample data validation
    use_sample_data_validation=True,
    sample_size=100,
    
    # Output configuration
    include_metadata=True,
    include_optimization_report=True,
    
    # Processing configuration
    processing_entry_point='label_ruleset_generation.py',
    instance_type="ml.m5.xlarge",
    instance_count=1,
    max_runtime_in_seconds=3600,
    volume_size_in_gb=30
)

print(f"✅ {step_name} configured")
```

## Part 5: Python DAG Configuration for Ruleset Execution

Configure the ruleset execution step in your DAG:

```python
from src.cursus.steps.configs.config_label_ruleset_execution_step import (
    LabelRulesetExecutionConfig
)

step_name = "LabelRulesetExecution"

factory.set_step_config(
    step_name,
    
    # Execution configuration
    fail_on_missing_fields=True,  # Fail if llm_category field missing
    enable_rule_match_tracking=True,  # Track which rules match
    enable_progress_logging=True,  # Log progress during execution
    
    # Data format configuration
    preferred_input_format="parquet",  # Prefer parquet if available
    
    # Processing configuration
    processing_entry_point='label_ruleset_execution.py',
    instance_type="ml.m5.2xlarge",
    instance_count=1,
    max_runtime_in_seconds=7200,
    volume_size_in_gb=50
)

print(f"✅ {step_name} configured")
```

## Generated Ruleset Structure

When the ruleset generation step runs, it auto-generates JSON files from your Pydantic configurations and produces a validated ruleset:

### Validated Ruleset Output (validated_ruleset.json)

```json
{
  "version": "1.0",
  "generated_timestamp": "2025-11-09T19:30:00Z",
  "label_config": {
    "output_label_name": "llm_reversal_flag",
    "output_label_type": "binary",
    "label_values": [0, 1],
    "label_mapping": {
      "0": "No_Reversal_Required",
      "1": "Reversal_Required"
    },
    "default_label": 1,
    "evaluation_mode": "priority"
  },
  "field_config": {
    "required_fields": ["llm_category"],
    "optional_fields": [],
    "field_types": {
      "llm_category": "string"
    }
  },
  "ruleset": [
    {
      "rule_id": "rule_12ab34cd",
      "name": "TrueDNR_No_Reversal",
      "priority": 1,
      "enabled": true,
      "conditions": {
        "field": "llm_category",
        "operator": "equals",
        "value": "TrueDNR"
      },
      "output_label": 0,
      "description": "Package delivered but buyer claims non-receipt - fraud indicator, no reversal"
    }
    // ... 11 more rules ...
  ],
  "metadata": {
    "total_rules": 12,
    "enabled_rules": 12,
    "disabled_rules": 0,
    "field_usage": {
      "llm_category": 12
    },
    "validation_summary": {
      "field_validation": "passed",
      "label_validation": "passed",
      "logic_validation": "passed",
      "warnings": []
    }
  }
}
```

### Execution Report Output (execution_report.json)

```json
{
  "ruleset_version": "1.0",
  "ruleset_timestamp": "2025-11-09T19:30:00Z",
  "execution_timestamp": "2025-11-09T19:45:00Z",
  "label_config": {
    "output_label_name": "llm_reversal_flag",
    "output_label_type": "binary",
    "label_values": [0, 1]
  },
  "split_statistics": {
    "train": {
      "total_rows": 10000,
      "label_distribution": {
        "0": 3500,
        "1": 6500
      },
      "execution_stats": {
        "total_evaluated": 10000,
        "default_label_count": 0,
        "rule_match_counts": {
          "rule_12ab34cd": 800,
          "rule_23bc45de": 700,
          "rule_34cd56ef": 600,
          "rule_45de67fg": 500
        },
        "rule_match_percentages": {
          "rule_12ab34cd": 8.0,
          "rule_23bc45de": 7.0,
          "rule_34cd56ef": 6.0,
          "rule_45de67fg": 5.0
        },
        "default_label_percentage": 0.0
      }
    },
    "val": {
      "total_rows": 2000,
      "label_distribution": {
        "0": 700,
        "1": 1300
      }
    },
    "test": {
      "total_rows": 2000,
      "label_distribution": {
        "0": 680,
        "1": 1320
      }
    }
  },
  "total_rules_evaluated": 12
}
```

## Key Features Demonstrated

### 1. Modular Configuration
- Each component (label_config, rules) defined separately
- Pydantic models provide type safety and validation
- Auto-generates JSON files for script consumption

### 2. Two-Step Process
- **Generation**: Validates rules before execution (catch errors early)
- **Execution**: Applies validated rules efficiently (single pass)

### 3. Comprehensive Validation
- **Field validation**: Ensures llm_category exists in data
- **Label validation**: All output_label values are valid (0 or 1)
- **Logic validation**: Checks for unreachable rules, contradictions

### 4. Detailed Statistics
- **Rule match tracking**: See which categories are most common
- **Label distribution**: Understand class balance per split
- **Coverage analysis**: Identify missing categories

### 5. Format Preservation
- Maintains input format (CSV/TSV/Parquet)
- Supports train/val/test splits automatically
- Handles compressed files (.gz)

## Benefits of Enhanced Approach

✅ **Zero Hard-Coding**: All mapping logic in configuration  
✅ **Type Safety**: Pydantic models catch errors at config time  
✅ **Validation First**: Rules validated before applying to data  
✅ **Auditable**: Full execution statistics and match tracking  
✅ **Reusable**: Validated rulesets can be applied to new data  
✅ **Testable**: Validate configs independently of data  

## Next Steps

1. Define your label configuration (Part 1)
2. Define your mapping rules (Part 2)
3. Configure ruleset generation step (Part 4)
4. Configure ruleset execution step (Part 5)
5. Run the pipeline - generation will validate, execution will label

The system will auto-generate all JSON files and produce:
- `validated_ruleset.json` (validated rules ready for execution)
- `validation_report.json` (detailed validation results)
- `execution_report.json` (rule match statistics)
- `rule_match_statistics.json` (detailed match breakdown)

Your data will now have the `llm_reversal_flag` column ready for model training!
