---
tags:
  - project_planning
  - label_ruleset
  - multilabel
  - multi_task_learning
  - implementation_plan
keywords:
  - label ruleset multilabel extension
  - multi-label classification
  - category-conditional rules
  - sparse label generation
  - rule-based multilabel
topics:
  - label ruleset expansion
  - multi-task learning
  - rule-based classification
  - step enhancement
language: python
date of note: 2025-11-11
---

# Label Ruleset Multilabel Extension Implementation Plan

## Overview

This document provides a comprehensive implementation plan for expanding the Label Ruleset Generation and Execution steps to support multi-label output for multi-task learning scenarios. The extension enables category-conditional rule evaluation where rules can generate sparse multi-label columns based on categorical features (e.g., payment methods), combining the power of rule-based classification with multi-task learning architectures.

## Related Documentation

### Design Documents
- **[Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md)** - Complete design specification
- **[Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md)** - Entry point for all step documentation

### Current Implementation
- **[Label Ruleset Generation Script](../../src/cursus/steps/scripts/label_ruleset_generation.py)** - Current generation script
- **[Label Ruleset Execution Script](../../src/cursus/steps/scripts/label_ruleset_execution.py)** - Current execution script
- **[Label Ruleset Generation Contract](../../src/cursus/steps/contracts/label_ruleset_generation_contract.py)** - Current contract
- **[Label Ruleset Execution Contract](../../src/cursus/steps/contracts/label_ruleset_execution_contract.py)** - Current contract
- **[Label Ruleset Generation Config](../../src/cursus/steps/configs/config_label_ruleset_generation_step.py)** - Current configuration

### Related Multi-Task Documents
- **[LightGBM Multi-Task Training Step Design](../1_design/lightgbm_multi_task_training_step_design.md)** - Multi-task training consumer
- **[MTGBM Multi-Task Learning Design](../1_design/mtgbm_multi_task_learning_design.md)** - Comprehensive MTGBM architecture

## Motivation

### Limitations of Current System

The current Label Ruleset system supports **single-label** output only:
- One rule evaluation per row → one output label
- Cannot generate category-specific labels
- No support for sparse multi-label matrices
- Limited to single-task classification

### Value Proposition of Multilabel Extension

Combining rule-based systems with multilabel output enables:
- **Category-conditional business logic**: Different rules per payment method
- **Domain knowledge encoding**: Expert-defined fraud patterns per category
- **Sparse representation**: Efficient multi-label matrices
- **Auditability**: Track which rules fire for which categories
- **Backward Compatibility**: Seamless integration with existing single-label workflows

## Implementation Roadmap

### Phase 1: Extend Configuration Models (Foundation)
**Duration**: 2-3 days  
**Risk**: Low  
**Priority**: Critical
**Status**: ✅ COMPLETE

- [x] Update `LabelConfig` class with multi-label fields
- [x] Update `RuleDefinition` class with `output_label` Union support
- [x] Add validators for backward compatibility
- [x] Add comprehensive dict key validation (missing + extra columns)
- [x] Update JSON generation logic for multi-label configs

### Phase 2: Extend Generation Script Validators (Validation)
**Duration**: 3-4 days  
**Risk**: Medium  
**Priority**: Critical
**Status**: ✅ COMPLETE

- [x] Extend `RulesetLabelValidator` for multilabel support
- [x] Add `RuleCoverageValidator` class
- [x] Update main function to use coverage validator
- [x] No changes needed to `RulesetLogicValidator` (validates conditions, not outputs)

### Phase 3: Extend Execution Script Engine (Core Logic)
**Duration**: 4-5 days  
**Risk**: High  
**Priority**: Critical
**Status**: ✅ COMPLETE

- [x] Extend `RuleEngine.__init__` for multilabel detection
- [x] Implement `_evaluate_row_single_label` and `_evaluate_row_multilabel` methods
- [x] Update `evaluate_batch` for multi-column output
- [x] Update `get_statistics` for per-column metrics
- [x] Update main function for multilabel label distribution and statistics reset

### Phase 4: Update Contracts (Documentation)
**Duration**: 1-2 days  
**Risk**: Low  
**Priority**: High
**Status**: ✅ COMPLETE

- [x] Update generation contract documentation
- [x] Update execution contract documentation
- [x] Add multi-label examples (config and rules)
- [x] Document sparse representation format
- [x] Add output structure examples

### Phase 5: Testing & Integration (Quality Assurance)
**Duration**: 3-4 days  
**Risk**: Medium  
**Priority**: Critical

- [ ] Unit tests for multi-label validation
- [ ] Unit tests for multi-label execution
- [ ] Integration tests with multi-task training
- [ ] Backward compatibility tests
- [ ] Performance benchmarking

**Total Estimated Duration**: 13-18 days (2.5-3.5 weeks)

## Detailed Implementation Specifications

### Phase 1: Configuration Model Extensions

#### File: `src/cursus/steps/configs/config_label_ruleset_generation_step.py`

**1.1 Update `LabelConfig` Class**

**Key Design Principle**: Minimal changes with maximum backward compatibility. Use Union types to support both single-label and multilabel modes seamlessly.

```python
class LabelConfig(BaseModel):
    """
    Pydantic model for label configuration with multi-label support.
    
    Supports three modes via output_label_type:
    - 'binary': Single binary column
    - 'multiclass': Single multiclass column  
    - 'multilabel': Multiple columns (new)
    """

    # ===== Tier 1: Required User Inputs =====
    
    # Unified output field (works for all modes)
    output_label_name: Union[str, List[str]] = Field(
        ...,
        description=(
            "Output label column name(s). "
            "String for single column (binary/multiclass), "
            "List[str] for multiple columns (multilabel)"
        )
    )
    
    # Extended to support multilabel
    output_label_type: str = Field(
        ...,
        description="Type of classification: 'binary', 'multiclass', or 'multilabel'",
    )
    
    # Flexible: Global (List) or Per-Column (Dict)
    label_values: Union[
        List[Union[int, str]],                    # Global: same for all columns
        Dict[str, List[Union[int, str]]]          # Per-column: different per column
    ] = Field(
        ...,
        description=(
            "Valid label values. "
            "List for global (all columns same), "
            "Dict[column_name -> values] for per-column"
        )
    )
    
    # Flexible: Global (Dict) or Per-Column (Dict[Dict])
    label_mapping: Union[
        Dict[str, str],                           # Global: same for all columns
        Dict[str, Dict[str, str]]                 # Per-column: different per column
    ] = Field(
        ...,
        description=(
            "Label to human-readable mapping. "
            "Dict for global (all columns same), "
            "Dict[column_name -> mapping] for per-column"
        )
    )
    
    # Flexible: Global (int/str) or Per-Column (Dict)
    default_label: Union[
        int, str,                                 # Global: same for all columns
        Dict[str, Union[int, str]]                # Per-column: different per column
    ] = Field(
        ...,
        description=(
            "Default label when no rules match. "
            "int/str for global (all columns same), "
            "Dict[column_name -> value] for per-column"
        )
    )
    
    # ===== Tier 2: Optional User Inputs with Defaults =====
    
    evaluation_mode: str = Field(
        default="priority",
        description="Rule evaluation mode: 'priority' or 'confidence'",
    )
    
    sparse_representation: bool = Field(
        default=True,
        description="Use NaN for non-matching categories in multilabel mode",
    )

    @field_validator("output_label_type")
    @classmethod
    def validate_label_type(cls, v: str) -> str:
        """Validate label_type is valid."""
        if v not in ["binary", "multiclass", "multilabel"]:
            raise ValueError("output_label_type must be 'binary', 'multiclass', or 'multilabel'")
        return v

    @field_validator("evaluation_mode")
    @classmethod
    def validate_evaluation_mode(cls, v: str) -> str:
        """Validate evaluation mode."""
        if v not in ["priority", "confidence"]:
            raise ValueError("evaluation_mode must be 'priority' or 'confidence'")
        return v

    @model_validator(mode="after")
    def validate_consistency(self) -> "LabelConfig":
        """Validate fields match output_label_type."""
        is_list = isinstance(self.output_label_name, list)
        
        if self.output_label_type in ["binary", "multiclass"]:
            # Single-label: normalize to string
            if is_list:
                if len(self.output_label_name) != 1:
                    raise ValueError(
                        f"{self.output_label_type} requires single column name"
                    )
                # Normalize single-element list to string
                self.output_label_name = self.output_label_name[0]
            
            # Validate label_values and label_mapping are global format
            if isinstance(self.label_values, dict):
                raise ValueError("Single-label mode requires list for label_values")
            if isinstance(self.label_mapping, dict) and \
               any(isinstance(v, dict) for v in self.label_mapping.values()):
                raise ValueError("Single-label mode requires simple dict for label_mapping")
        
        elif self.output_label_type == "multilabel":
            # Multilabel: must be list with at least 2 columns
            if not is_list:
                raise ValueError("multilabel requires list of column names")
            if len(self.output_label_name) < 2:
                raise ValueError("multilabel requires at least 2 columns")
            
            # Check for duplicates
            if len(self.output_label_name) != len(set(self.output_label_name)):
                raise ValueError("Duplicate column names in multilabel")
            
            # Validate per-column structures if used
            if isinstance(self.label_values, dict):
                missing = set(self.output_label_name) - set(self.label_values.keys())
                if missing:
                    raise ValueError(f"label_values missing columns: {missing}")
            
            if isinstance(self.label_mapping, dict) and \
               all(isinstance(v, dict) for v in self.label_mapping.values()):
                # Per-column mapping
                missing = set(self.output_label_name) - set(self.label_mapping.keys())
                if missing:
                    raise ValueError(f"label_mapping missing columns: {missing}")
            
            if isinstance(self.default_label, dict):
                # Per-column default_label
                missing = set(self.output_label_name) - set(self.default_label.keys())
                if missing:
                    raise ValueError(f"default_label missing columns: {missing}")
        
        return self

    def to_script_format(self) -> Dict[str, Any]:
        """Convert to format expected by script."""
        return {
            "output_label_name": self.output_label_name,
            "output_label_type": self.output_label_type,
            "label_values": self.label_values,
            "label_mapping": self.label_mapping,
            "default_label": self.default_label,
            "evaluation_mode": self.evaluation_mode,
            "sparse_representation": self.sparse_representation,
        }

    model_config = {"extra": "forbid", "validate_assignment": True}
```

**1.2 Update `RuleDefinition` Class**

**Key Design Principle**: Zero new fields - just extend the existing `output_label` field with Union type to support both single-label and multilabel.

```python
class RuleDefinition(BaseModel):
    """
    Pydantic model for a single rule definition with multi-label support.
    """

    # ===== Tier 1: Required User Inputs =====

    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable rule name",
    )

    priority: int = Field(
        ...,
        ge=1,
        description="Priority for evaluation (lower = higher priority)",
    )

    conditions: RuleCondition = Field(
        ...,
        description="Nested condition expression",
    )

    # ===== Unified output field (works for all modes) =====
    
    output_label: Union[int, str, Dict[str, Union[int, str]]] = Field(
        ...,
        description=(
            "Output label value(s). "
            "int/str for single-label mode, "
            "Dict[column_name -> value] for multilabel mode"
        ),
    )

    # ===== Tier 2: Optional User Inputs =====

    enabled: bool = Field(
        default=True,
        description="Whether rule is active",
    )

    description: str = Field(
        default="",
        description="Description of what this rule identifies",
    )

    # ===== Tier 3: Derived Fields =====

    _rule_id: str = PrivateAttr(default_factory=lambda: f"rule_{uuid.uuid4().hex[:8]}")

    @property
    def rule_id(self) -> str:
        """Get auto-generated unique rule identifier."""
        return self._rule_id

    @model_validator(mode="after")
    def validate_output_label(self) -> "RuleDefinition":
        """Validate output_label format."""
        # Validate multi-label dict is not empty
        if isinstance(self.output_label, dict) and len(self.output_label) == 0:
            raise ValueError("output_label dict cannot be empty for multilabel mode")
        
        return self

    def to_script_format(self) -> Dict[str, Any]:
        """Convert to format expected by script."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "priority": self.priority,
            "enabled": self.enabled,
            "conditions": self.conditions.to_script_format(),
            "description": self.description,
            "output_label": self.output_label,
        }
```

**Benefits of Unified Field**:
- ✅ **Zero new fields** - just extend type of existing `output_label`
- ✅ **Consistent with LabelConfig** - same Union pattern as `output_label_name`
- ✅ **Backward compatible** - existing int/str values still work
- ✅ **No mutual exclusivity checks** - simpler validation
- ✅ **Cleaner API** - one field for all modes

**1.3 Configuration Auto-Generation**

**No Changes Required** ✅

The existing `generate_ruleset_config_bundle()` method in `LabelRulesetGenerationConfig` requires **zero changes** because:

1. It delegates to `to_script_format()` methods on Pydantic models
2. We already updated those methods in sections 1.1 and 1.2
3. JSON serialization automatically handles Union types
4. The code is agnostic to field structure - it just serializes whatever the models provide

**Existing Code (unchanged):**
```python
def generate_ruleset_config_bundle(self) -> None:
    """Generate complete ruleset configuration bundle."""
    # ... create output directory ...
    
    # Generate label_config.json
    if self.label_config is not None:
        label_config_file = output_dir / "label_config.json"
        with open(label_config_file, "w", encoding="utf-8") as f:
            json.dump(
                self.label_config.to_script_format(),  # ← delegates to updated method
                f,
                indent=2,
                ensure_ascii=False,
            )
    
    # Generate ruleset.json
    if self.rule_definitions is not None:
        ruleset_file = output_dir / "ruleset.json"
        with open(ruleset_file, "w", encoding="utf-8") as f:
            json.dump(
                self.rule_definitions.to_script_format(),  # ← delegates to updated method
                f,
                indent=2,
                ensure_ascii=False,
            )
```

This is a key benefit of our minimal design - the serialization layer requires no updates!

### Phase 2: Generation Script Validator Extensions

#### File: `src/cursus/steps/scripts/label_ruleset_generation.py`

**Design Philosophy**: Extend existing validators rather than creating parallel validator hierarchies. This aligns with our minimal design approach of extending existing fields with Union types.

**2.1 Extend `RulesetLabelValidator`**

**Current Implementation**: The existing `RulesetLabelValidator.validate_labels()` method validates single-label configurations.

**Required Changes**: Add multilabel-aware validation logic to the existing method:

```python
class RulesetLabelValidator:
    """Validates output labels match configuration (extended for multilabel)."""

    def validate_labels(self, ruleset: dict) -> ValidationResult:
        """
        Validates all output_label values in rules.
        Extended to support multilabel mode.
        """
        result = ValidationResult()

        label_config = ruleset.get("label_config", {})
        label_values = label_config.get("label_values", [])
        label_type = label_config.get("output_label_type", "binary")
        default_label = label_config.get("default_label")
        output_label_name = label_config.get("output_label_name")

        rules = ruleset.get("ruleset", [])

        # NEW: Validate multilabel configuration structure
        if label_type == "multilabel":
            # output_label_name must be a list
            if not isinstance(output_label_name, list):
                result.valid = False
                result.type_errors.append(
                    "multilabel mode requires list for output_label_name"
                )
                return result
            
            if len(output_label_name) < 2:
                result.valid = False
                result.type_errors.append("multilabel requires at least 2 columns")
            
            # Check for duplicate column names
            if len(output_label_name) != len(set(output_label_name)):
                result.valid = False
                result.type_errors.append("Duplicate column names in output_label_name")
            
            # Validate per-column structures if used
            if isinstance(label_values, dict):
                missing = set(output_label_name) - set(label_values.keys())
                if missing:
                    result.valid = False
                    result.type_errors.append(f"label_values missing columns: {missing}")
            
            label_mapping = label_config.get("label_mapping", {})
            if isinstance(label_mapping, dict) and \
               all(isinstance(v, dict) for v in label_mapping.values()):
                missing = set(output_label_name) - set(label_mapping.keys())
                if missing:
                    result.valid = False
                    result.type_errors.append(f"label_mapping missing columns: {missing}")

        # Convert label_values to set for validation
        if isinstance(label_values, list):
            label_values_set = set(label_values)
        else:
            # Per-column: collect all possible values
            label_values_set = set()
            for col_values in label_values.values():
                label_values_set.update(col_values)

        # Validate default label
        if isinstance(default_label, dict):
            # Per-column default_label
            for col, default_val in default_label.items():
                if isinstance(label_values, dict):
                    col_values = set(label_values.get(col, []))
                    if default_val not in col_values:
                        result.valid = False
                        result.invalid_labels.append(
                            (f"default_label[{col}]", default_val, f"not in label_values[{col}]")
                        )
                        logger.error(f"Default label {default_val} for column {col} not in label_values")
                else:
                    if default_val not in label_values_set:
                        result.valid = False
                        result.invalid_labels.append(
                            (f"default_label[{col}]", default_val, "not in label_values")
                        )
                        logger.error(f"Default label {default_val} for column {col} not in label_values")
        else:
            # Global default_label
            if default_label not in label_values_set:
                result.valid = False
                result.invalid_labels.append(
                    ("default_label", default_label, "not in label_values")
                )
                logger.error(f"Default label {default_label} not in label_values")

        # Extract and validate all output labels
        used_labels = set()
        for rule in rules:
            output_label = rule.get("output_label")
            
            # NEW: Handle multilabel dict format
            if isinstance(output_label, dict):
                # Multilabel mode
                if label_type != "multilabel":
                    result.valid = False
                    result.type_errors.append(
                        f"Rule {rule.get('rule_id')}: dict output_label requires multilabel mode"
                    )
                    continue
                
                if len(output_label) == 0:
                    result.valid = False
                    result.invalid_labels.append(
                        (rule.get("rule_id"), "empty_dict", "output_label cannot be empty dict")
                    )
                    continue
                
                # Validate target columns exist
                valid_columns = set(output_label_name) if isinstance(output_label_name, list) else set()
                for col, value in output_label.items():
                    if col not in valid_columns:
                        result.valid = False
                        result.invalid_labels.append(
                            (rule.get("rule_id"), col, f"not in output_label_name")
                        )
                    
                    # Validate value for this column
                    if isinstance(label_values, dict):
                        col_values = set(label_values.get(col, []))
                        if value not in col_values:
                            result.valid = False
                            result.invalid_labels.append(
                                (rule.get("rule_id"), value, f"not valid for column {col}")
                            )
                    else:
                        if value not in label_values_set:
                            result.valid = False
                            result.invalid_labels.append(
                                (rule.get("rule_id"), value, "not in label_values")
                            )
                    
                    used_labels.add(value)
            
            elif output_label is not None:
                # Single-label mode (existing logic)
                used_labels.add(output_label)
                
                if output_label not in label_values_set:
                    result.valid = False
                    result.invalid_labels.append(
                        (rule.get("rule_id", "unknown"), output_label, "not in label_values")
                    )
                    logger.error(
                        f"Rule {rule.get('name', 'unknown')}: invalid output_label {output_label}"
                    )

        # ... rest of existing validation (binary constraints, uncovered classes, etc.)
        
        return result
```

**Key Changes**:
- ✅ Extends existing `validate_labels()` method
- ✅ Adds multilabel-specific checks
- ✅ Handles `output_label` as Union[int, str, Dict]
- ✅ Validates per-column structures
- ✅ No new validator classes needed

**2.2 RulesetLogicValidator - No Changes Required**

The existing `RulesetLogicValidator` validates **conditions**, not outputs. Since conditions remain unchanged (only outputs change for multilabel), this validator needs **no modifications**. ✅

**2.3 Add `RuleCoverageValidator`** *(New Focused Validator)*

```python
class RuleCoverageValidator:
    """Validates that all label columns have at least one rule."""
    
    def validate(self, label_config: dict, rules: List[dict]) -> ValidationResult:
        """
        Validates rule coverage for all label columns.
        
        Checks:
        - Each label column has at least one rule targeting it
        - Warns about orphan label columns
        """
        result = ValidationResult()
        
        label_type = label_config.get("output_label_type", "binary")
        
        # Only applicable to multilabel
        if label_type != "multilabel":
            return result
        
        output_columns = label_config.get("output_label_name", [])
        if not isinstance(output_columns, list):
            return result
        
        # Check rule coverage
        covered_columns = set()
        for rule in rules:
            if not rule.get("enabled", True):
                continue
            
            output_label = rule.get("output_label")
            if isinstance(output_label, dict):
                covered_columns.update(output_label.keys())
        
        uncovered = set(output_columns) - covered_columns
        if uncovered:
            result.warnings.append(f"Label columns without rules: {uncovered}")
        
        return result
```

**Note**: Removed category-specific validation since `category_column` and `categories` are no longer in LabelConfig. Category information is inferred at execution time from data and rule targets.

**2.4 Update Main Function for Multi-Label**

**Key Changes**: The existing `RulesetLabelValidator` now handles both single-label and multilabel validation. No separate validator initialization needed!

```python
def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Main logic for ruleset generation with multi-label support."""
    log = logger or print
    
    # ... existing loading logic ...
    
    # Initialize validators (existing - now multilabel-aware)
    label_validator = RulesetLabelValidator()  # Extended to handle multilabel
    logic_validator = RulesetLogicValidator()  # No changes needed
    
    # NEW: Initialize coverage validator (genuinely new)
    coverage_validator = RuleCoverageValidator()
    
    # Run validation
    log("[INFO] Running validation...")
    
    enable_label = environ_vars.get("ENABLE_LABEL_VALIDATION", "true").lower() == "true"
    enable_logic = environ_vars.get("ENABLE_LOGIC_VALIDATION", "true").lower() == "true"
    
    # Existing label validation (now handles multilabel automatically)
    label_validation = label_validator.validate_labels(user_ruleset) if enable_label else None
    logic_validation = logic_validator.validate_logic(user_ruleset) if enable_logic else None
    
    # NEW: Additional coverage check for multilabel
    label_type = label_config.get("output_label_type", "binary")
    if label_type == "multilabel":
        coverage_validation = coverage_validator.validate(
            label_config, ruleset_rules
        )
        for warning in coverage_validation.warnings:
            log(f"[WARNING] {warning}")
    
    # Check validation results (existing logic)
    all_valid = (label_validation.valid if label_validation else True) and \
                (logic_validation.valid if logic_validation else True)
    
    if not all_valid:
        log("[ERROR] Validation failed!")
        # ... existing error handling ...
    
    # ... existing optimization and output logic ...
```

**Benefits of This Approach**:
- ✅ **Minimal code changes** - extends existing validators
- ✅ **Single validation path** - no branching for multilabel vs single-label
- ✅ **Consistent with design** - extends existing fields, extends existing validators
- ✅ **Simpler maintenance** - one validator per concern

**2.5 Extend Rule Optimization for Multi-Label**

```python
def optimize_multilabel_ruleset(
    ruleset: dict,
    enable_category_grouping: bool = True,
    log: Callable[[str], None] = print
) -> dict:
    """
    Optimize multi-label ruleset.
    
    Strategies:
    1. Group rules by target column
    2. Order by complexity within each group
    3. Interleave rules from different columns for balanced evaluation
    """
    label_type = ruleset.get("label_config", {}).get("output_label_type", "binary")
    
    if label_type in ["binary", "multiclass"]:
        # Use standard single-label optimization
        return optimize_ruleset(ruleset, log=log)
    
    rules = copy.deepcopy(ruleset.get("ruleset", []))
    
    if enable_category_grouping:
        log("[INFO] Grouping rules by target column...")
        
        # Group rules by target column
        column_rules = {}
        for rule in rules:
            output_label = rule.get("output_label")
            if isinstance(output_label, dict):
                # Multi-label rule: add to all target columns
                for col in output_label.keys():
                    column_rules.setdefault(col, []).append(rule)
        
        # Sort each group by complexity
        for col, col_rules in column_rules.items():
            for rule in col_rules:
                rule["complexity_score"] = calculate_complexity(
                    rule.get("conditions", {})
                )
            col_rules.sort(key=lambda r: r["complexity_score"])
            log(f"  {col}: {len(col_rules)} rules optimized")
        
        # Interleave rules from different columns for balanced evaluation
        optimized_rules = []
        max_rules_per_col = max(len(rules) for rules in column_rules.values())
        
        for i in range(max_rules_per_col):
            for col_rules in column_rules.values():
                if i < len(col_rules):
                    # Avoid duplicates (rule may target multiple columns)
                    if col_rules[i] not in optimized_rules:
                        optimized_rules.append(col_rules[i])
        
        rules = optimized_rules
    
    # Assign final priorities
    for i, rule in enumerate(rules, start=1):
        rule["priority"] = i
    
    log(f"[INFO] Multi-label optimization complete: {len(rules)} rules reordered")
    
    return {
        **ruleset,
        "ruleset": rules,
        "optimization_metadata": {
            "multi_label_optimization": True,
            "category_grouping_enabled": enable_category_grouping
        }
    }
```

**Note**: Updated to use `output_label_type` instead of `label_mode`, and `output_label` (dict) instead of `output_labels`. Added duplicate prevention since a single rule can target multiple columns.

### Phase 3: Execution Script Engine Extensions

#### File: `src/cursus/steps/scripts/label_ruleset_execution.py`

**3.1 Extend RuleEngine Initialization**

```python
class RuleEngine:
    """
    Evaluates validated rules with multi-label support.
    """

    def __init__(self, validated_ruleset: dict):
        """Initialize rule engine with multi-label support."""
        self.label_config = validated_ruleset["label_config"]
        self.field_config = validated_ruleset["field_config"]
        self.ruleset = validated_ruleset["ruleset"]
        self.metadata = validated_ruleset.get("metadata", {})
        
        # Filter to enabled rules only
        self.active_rules = [r for r in self.ruleset if r.get("enabled", True)]
        
        # Determine label type
        self.label_type = self.label_config.get("output_label_type", "binary")
        
        # Get output column names (normalize to list)
        output_label_name = self.label_config["output_label_name"]
        if isinstance(output_label_name, str):
            # Single-label: string → list of one
            self.output_columns = [output_label_name]
        else:
            # Multilabel: already a list
            self.output_columns = output_label_name
        
        # Multilabel-specific configuration
        self.sparse_representation = self.label_config.get("sparse_representation", True)
        
        # Common configuration
        self.default_label = self.label_config["default_label"]
        self.evaluation_mode = self.label_config.get("evaluation_mode", "priority")
        
        # Statistics tracking (per column)
        self.rule_match_counts = {
            col: {r["rule_id"]: 0 for r in self.active_rules}
            for col in self.output_columns
        }
        self.default_label_counts = {col: 0 for col in self.output_columns}
        self.total_evaluated = 0
```

**Key Changes**:
- ✅ Uses `output_label_type` instead of `label_mode`
- ✅ Uses `output_label_name` (normalized to list) instead of `output_label_columns`
- ✅ Removed `category_column` and `categories` (no longer in LabelConfig)
- ✅ Simplified initialization logic

**3.2 Add Multi-Label Evaluation Method**

```python
    def evaluate_row(self, row: pd.Series) -> Union[int, str, Dict[str, Any]]:
        """
        Evaluate rules against a single row.
        
        Returns:
            - Single-label mode: int or str (label value)
            - Multilabel mode: Dict[str, Any] (column → value mapping)
        """
        self.total_evaluated += 1
        
        if self.label_type in ["binary", "multiclass"]:
            return self._evaluate_row_single_label(row)
        else:
            return self._evaluate_row_multilabel(row)

    def _evaluate_row_single_label(self, row: pd.Series) -> Union[int, str]:
        """Evaluate rules for single-label mode (backward compatible)."""
        # Single column name (output_columns is list of one)
        output_col = self.output_columns[0]
        
        for rule in self.active_rules:
            try:
                if self._evaluate_conditions(rule["conditions"], row):
                    rule_id = rule["rule_id"]
                    output_label = rule["output_label"]
                    
                    # output_label should be int/str for single-label
                    self.rule_match_counts[output_col][rule_id] += 1
                    return output_label
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule['rule_id']}: {e}")
                continue
        
        self.default_label_counts[output_col] += 1
        return self.default_label

    def _evaluate_row_multilabel(self, row: pd.Series) -> Dict[str, Any]:
        """Evaluate rules for multilabel mode with sparse representation."""
        import numpy as np
        
        # Initialize all columns with NaN (sparse) or default (dense)
        if self.sparse_representation:
            result = {col: np.nan for col in self.output_columns}
        else:
            # Handle per-column default_label
            if isinstance(self.default_label, dict):
                result = {col: self.default_label[col] for col in self.output_columns}
            else:
                result = {col: self.default_label for col in self.output_columns}
        
        # Evaluate rules in priority order
        for rule in self.active_rules:
            try:
                if not self._evaluate_conditions(rule["conditions"], row):
                    continue
                
                # Rule matched - get output
                output_label = rule.get("output_label")
                
                if isinstance(output_label, dict):
                    # Multilabel: dict mapping column → value
                    for col, value in output_label.items():
                        if col not in result:
                            continue
                        
                        # Only set if not already set (priority order)
                        if pd.isna(result[col]) or result[col] == self.default_label:
                            result[col] = value
                            self.rule_match_counts[col][rule["rule_id"]] += 1
            
            except Exception as e:
                logger.warning(f"Error evaluating rule {rule['rule_id']}: {e}")
                continue
        
        # Fill remaining NaN columns with default if dense mode
        for col in result:
            if pd.isna(result[col]):
                self.default_label_counts[col] += 1
                if not self.sparse_representation:
                    result[col] = self.default_label
        
        return result
```

**Key Changes**:
- ✅ Uses `label_type` instead of `label_mode`
- ✅ Uses `output_label` (Union type) instead of `output_labels`
- ✅ Removed "auto" mode complexity
- ✅ Removed category-column mapping logic (not in minimal design)
- ✅ Simplified to just handle dict format for multilabel

**3.3 Category-Column Mapping** *(Removed)*

**Note**: Category-column mapping helpers have been removed as `category_column` and `categories` are no longer part of the LabelConfig design. Users explicitly specify target columns in the `output_label` dict.

**3.4 Update evaluate_batch for Multi-Column Output**

```python
    def evaluate_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate rules for entire DataFrame.
        
        Returns:
            DataFrame with label column(s) added
        """
        if self.label_type in ["binary", "multiclass"]:
            # Single column result (backward compatible)
            output_col = self.output_columns[0]
            df[output_col] = df.apply(self.evaluate_row, axis=1)
            return df
        
        else:
            # Multi-column result (multilabel)
            results = df.apply(self.evaluate_row, axis=1, result_type='expand')
            
            # Add all label columns to original dataframe
            for col in self.output_columns:
                df[col] = results[col]
            
            return df
```

**Key Changes**:
- ✅ Uses `label_type` instead of `label_mode`
- ✅ Uses `output_columns[0]` instead of `output_label_name`

**3.5 Update Statistics for Multi-Label**

```python
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics with multi-label support."""
        if self.label_type in ["binary", "multiclass"]:
            # Single-label statistics (backward compatible)
            output_col = self.output_columns[0]
            
            return {
                "total_evaluated": self.total_evaluated,
                "rule_match_counts": self.rule_match_counts[output_col],
                "default_label_count": self.default_label_counts[output_col],
                "rule_match_percentages": {
                    rule_id: (count / self.total_evaluated * 100) 
                    if self.total_evaluated > 0 else 0
                    for rule_id, count in self.rule_match_counts[output_col].items()
                },
                "default_label_percentage": (
                    self.default_label_counts[output_col] / self.total_evaluated * 100
                    if self.total_evaluated > 0 else 0
                )
            }
        
        else:
            # Multilabel statistics (per column)
            stats = {
                "label_type": "multilabel",
                "total_evaluated": self.total_evaluated,
                "per_column_statistics": {}
            }
            
            for col in self.output_columns:
                col_stats = {
                    "rule_match_counts": self.rule_match_counts[col],
                    "default_label_count": self.default_label_counts[col],
                    "rule_match_percentages": {
                        rule_id: (count / self.total_evaluated * 100) 
                        if self.total_evaluated > 0 else 0
                        for rule_id, count in self.rule_match_counts[col].items()
                    },
                    "default_label_percentage": (
                        self.default_label_counts[col] / self.total_evaluated * 100
                        if self.total_evaluated > 0 else 0
                    )
                }
                stats["per_column_statistics"][col] = col_stats
            
            return stats
```

**Key Changes**:
- ✅ Uses `label_type` instead of `label_mode`
- ✅ Uses `output_columns[0]` instead of `output_label_name`
- ✅ Returns `"label_type": "multilabel"` instead of `"label_mode": "multi_label"`

### Phase 4: Contract Documentation Updates

#### File: `src/cursus/steps/contracts/label_ruleset_generation_contract.py`

**4.1 Add Multi-Label Configuration Examples**

Add to the contract description:

```python
    description="""
    ... (existing description) ...
    
    Multi-Label Configuration Support:
    
    The script now supports multi-label output for multi-task learning scenarios.
    
    Label Configuration for Multi-Label Mode:
    {
      "label_mode": "multi_label",
      "output_label_columns": ["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
      "output_label_type": "binary",
      "label_values": [0, 1],
      "label_mapping": {"0": "No_Fraud", "1": "Fraud"},
      "default_label": 0,
      "evaluation_mode": "priority",
      "category_column": "payment_method",
      "categories": ["CC", "DC", "ACH"],
      "sparse_representation": true
    }
    
    Rule Definition for Multi-Label (Explicit):
    {
      "rule_id": "rule_001_CC",
      "name": "High value CC transaction",
      "priority": 1,
      "enabled": true,
      "conditions": {
        "all_of": [
          {"field": "payment_method", "operator": "equals", "value": "CC"},
          {"field": "amount", "operator": ">", "value": 1000}
        ]
      },
      "output_labels": {"is_fraud_CC": 1},
      "description": "High value credit card transaction flagged as fraud"
    }
    
    Rule Definition for Multi-Label (Auto Mode):
    {
      "rule_id": "rule_002_auto",
      "name": "High value transaction (auto-category)",
      "priority": 2,
      "enabled": true,
      "conditions": {
        "field": "amount",
        "operator": ">",
        "value": 5000
      },
      "output_labels": "auto",
      "output_value": 1,
      "category_conditional": true,
      "description": "High value transaction auto-targets category-specific column"
    }
    
    Backward Compatibility:
    - Default label_mode is "single_label"
    - Existing single-label configurations work unchanged
    - Rules with output_label field continue to function
    """
```

#### File: `src/cursus/steps/contracts/label_ruleset_execution_contract.py`

**4.2 Add Multi-Label Execution Examples**

Add to the contract description:

```python
    description="""
    ... (existing description) ...
    
    Multi-Label Output Support:
    
    The script now supports multi-label output with sparse representation.
    
    Output Structure for Multi-Label:
    - Sparse representation (default): NaN for non-matching categories
    - Dense representation: default_label for all categories
    
    Example Output (Sparse):
    | txn_id | payment_method | amount | is_fraud_CC | is_fraud_DC | is_fraud_ACH |
    |--------|----------------|--------|-------------|-------------|--------------|
    | 1      | CC             | 6000   | 1           | NaN         | NaN          |
    | 2      | DC             | 1500   | NaN         | 1           | NaN          |
    | 3      | ACH            | 12000  | NaN         | NaN         | 1            |
    
    Statistics Output for Multi-Label:
    {
      "label_mode": "multi_label",
      "total_evaluated": 1000,
      "per_column_statistics": {
        "is_fraud_CC": {
          "rule_match_counts": {"rule_001": 50, "rule_002": 30},
          "default_label_count": 20,
          "rule_match_percentages": {"rule_001": 5.0, "rule_002": 3.0},
          "default_label_percentage": 2.0
        },
        "is_fraud_DC": { ... },
        "is_fraud_ACH": { ... }
      }
    }
    
    Backward Compatibility:
    - Single-label mode remains the default
    - Existing pipelines continue to work unchanged
    - Statistics format extended, not replaced
    """
```

### Phase 5: Testing & Integration

**5.1 Unit Tests for Multi-Label Validation**

Create: `tests/steps/scripts/test_label_ruleset_generation_multilabel.py`

```python
"""Unit tests for multi-label validation in label ruleset generation."""

def test_multilabel_config_validation():
    """Test MultiLabelConfigValidator."""
    # Test valid multi-label config
    # Test missing output_label_columns
    # Test duplicate columns
    # Test missing category_column

def test_multilabel_rule_validation():
    """Test MultiLabelRuleValidator."""
    # Test explicit output_labels dict
    # Test auto mode with output_value
    # Test auto mode without output_value
    # Test invalid target columns

def test_category_consistency_validation():
    """Test CategoryConsistencyValidator."""
    # Test naming convention matching
    # Test rule coverage
    # Test orphan columns

def test_multilabel_optimization():
    """Test multi-label rule optimization."""
    # Test category grouping
    # Test complexity ordering
    # Test priority assignment
```

**5.2 Unit Tests for Multi-Label Execution**

Create: `tests/steps/scripts/test_label_ruleset_execution_multilabel.py`

```python
"""Unit tests for multi-label execution in label ruleset execution."""

def test_multilabel_engine_init():
    """Test RuleEngine initialization for multi-label."""
    # Test label_mode detection
    # Test output_columns setup
    # Test statistics initialization

def test_multilabel_row_evaluation():
    """Test multi-label row evaluation."""
    # Test sparse representation
    # Test auto mode
    # Test explicit output_labels
    # Test category matching

def test_multilabel_batch_evaluation():
    """Test multi-label batch evaluation."""
    # Test multi-column output
    # Test DataFrame structure
    # Test statistics tracking

def test_category_mapping_helpers():
    """Test category-column mapping."""
    # Test _get_column_for_category
    # Test _get_category_for_column
    # Test various naming patterns
```

**5.3 Integration Tests**

Create: `tests/integration/test_label_ruleset_multilabel_integration.py`

```python
"""Integration tests for end-to-end multi-label workflow."""

def test_generation_to_execution_flow():
    """Test complete generation → execution flow."""
    # Generate multi-label config
    # Validate ruleset
    # Execute on test data
    # Verify output structure

def test_with_multitask_training():
    """Test integration with multi-task training step."""
    # Generate multi-label labels
    # Feed to LightGBM multi-task training
    # Verify compatibility

def test_backward_compatibility():
    """Test that single-label workflows still work."""
    # Run existing single-label config
    # Verify unchanged behavior
    # Verify statistics format
```

**5.4 Performance Benchmarking**

Create: `tests/performance/test_label_ruleset_multilabel_performance.py`

```python
"""Performance tests for multi-label ruleset execution."""

def test_sparse_vs_dense_performance():
    """Compare sparse and dense representation performance."""
    # Benchmark sparse mode
    # Benchmark dense mode
    # Compare memory usage
    # Compare execution time

def test_category_count_scalability():
    """Test performance with varying category counts."""
    # Test with 3 categories
    # Test with 10 categories
    # Test with 50 categories
    # Measure degradation

def test_rule_optimization_impact():
    """Test impact of rule optimization."""
    # Benchmark unoptimized ruleset
    # Benchmark optimized ruleset
    # Measure improvement
```

## Implementation Timeline

### Week 1: Foundation (Phase 1)
**Days 1-3: Configuration Models**
- [ ] Day 1: Update `LabelConfig` class, add validators
- [ ] Day 2: Update `RuleDefinition` class, add multi-label fields
- [ ] Day 3: Update JSON generation, testing

### Week 2: Validation (Phase 2)
**Days 4-7: Generation Script Validators**
- [ ] Day 4: Implement `MultiLabelConfigValidator`
- [ ] Day 5: Implement `MultiLabelRuleValidator` and `CategoryConsistencyValidator`
- [ ] Day 6: Update main function, extend optimization
- [ ] Day 7: Unit tests for validators

### Week 3: Execution (Phase 3)
**Days 8-12: Execution Script Engine**
- [ ] Day 8: Extend `RuleEngine.__init__`, add mode detection
- [ ] Day 9: Implement `_evaluate_row_multi_label` method
- [ ] Day 10: Add category mapping helpers
- [ ] Day 11: Update `evaluate_batch` and statistics
- [ ] Day 12: Unit tests for execution

### Week 4: Integration & Testing (Phases 4-5)
**Days 13-15: Documentation and Testing**
- [ ] Day 13: Update contracts, add examples
- [ ] Day 14: Integration tests
- [ ] Day 15: Performance benchmarking, documentation

### Week 5: Polish & Review
**Days 16-18: Final Review**
- [ ] Day 16: Code review, refinements
- [ ] Day 17: End-to-end testing
- [ ] Day 18: Documentation review, final validation

## Success Criteria

### Functional Requirements
- ✅ Generate multi-label output with sparse representation
- ✅ Support category-conditional rule evaluation
- ✅ Maintain backward compatibility with single-label mode
- ✅ Provide per-column statistics tracking
- ✅ Support both explicit and auto mode for rules

### Technical Requirements
- ✅ All unit tests passing
- ✅ Integration tests with multi-task training successful
- ✅ Performance within acceptable bounds (<2x single-label)
- ✅ Backward compatibility tests passing
- ✅ Contract-specification alignment maintained

### Quality Requirements
- ✅ Code coverage >80% for new code
- ✅ Clear validation error messages
- ✅ Comprehensive documentation
- ✅ Performance benchmarks documented

## Risk Mitigation

### Technical Risks

**Risk: Performance Degradation**
- **Mitigation**: Benchmark early, optimize category grouping, use sparse representation
- **Monitoring**: Track execution time per category count

**Risk: Backward Compatibility Issues**
- **Mitigation**: Comprehensive backward compatibility tests, default to single-label mode
- **Monitoring**: Run existing single-label tests continuously

**Risk: Complex Multi-Label Logic Bugs**
- **Mitigation**: Extensive unit tests, integration tests with real data
- **Monitoring**: Test coverage metrics, integration test results

### Implementation Risks

**Risk: Scope Creep**
- **Mitigation**: Stick to defined phases, defer enhancements to future iterations
- **Monitoring**: Track task completion against timeline

**Risk: Integration Challenges**
- **Mitigation**: Early integration testing, collaborate with multi-task training team
- **Monitoring**: Integration test pass rate

## Migration Path

### For Existing Users

**Step 1: No Changes Required**
- Existing single-label configurations continue to work
- No migration needed for current workflows

**Step 2: Opt-In to Multi-Label (When Ready)**
```python
# Change label_config from:
label_config = LabelConfig(
    label_mode="single_label",  # default
    output_label_name="is_fraud",
    ...
)

# To:
label_config = LabelConfig(
    label_mode="multi_label",  # explicit
    output_label_columns=["is_fraud_CC", "is_fraud_DC", "is_fraud_ACH"],
    category_column="payment_method",
    categories=["CC", "DC", "ACH"],
    ...
)
```

**Step 3: Update Rules**
```python
# Change rules from:
rule = RuleDefinition(
    name="High value transaction",
    conditions=...,
    output_label=1  # single-label
)

# To explicit multi-label:
rule = RuleDefinition(
    name="High value CC transaction",
    conditions=...,
    output_labels={"is_fraud_CC": 1}  # multi-label
)

# Or auto mode:
rule = RuleDefinition(
    name="High value transaction",
    conditions=...,
    output_labels="auto",
    output_value=1,
    category_conditional=True
)
```

## Conclusion

This implementation plan provides a comprehensive roadmap for expanding the Label Ruleset system to support multi-label output. The extension maintains full backward compatibility while adding powerful category-conditional rule evaluation capabilities.

### Key Implementation Highlights

1. **Backward Compatible**: Single-label mode remains default, existing workflows unchanged
2. **Sparse Representation**: Efficient NaN-based sparse matrices for multi-label output
3. **Category-Conditional**: Rules can target specific categories or use auto mode
4. **Comprehensive Validation**: New validators ensure multi-label configuration correctness
5. **Per-Column Statistics**: Detailed tracking of rule matches per label column

### Implementation Benefits

- **Enhanced Capabilities**: Support for multi-task learning scenarios
- **Domain Knowledge Integration**: Encode category-specific business logic in rules
- **Efficient Representation**: Sparse matrices reduce memory and computation
- **Auditability**: Track which rules fire for which categories
- **Seamless Integration**: Ready for multi-task training steps

### Next Steps

1. **Phase 1**: Begin with configuration model extensions (2-3 days)
2. **Phase 2**: Implement generation script validators (3-4 days)
3. **Phase 3**: Extend execution script engine (4-5 days)
4. **Phase 4**: Update contract documentation (1-2 days)
5. **Phase 5**: Comprehensive testing and integration (3-4 days)

**Total Timeline**: 13-18 days (2.5-3.5 weeks)

This plan ensures that the Label Ruleset Multilabel Extension will be a valuable enhancement to the cursus framework, enabling sophisticated multi-task learning scenarios while maintaining the system's core strengths of transparency, maintainability, and ease of use.

### References

#### Design Documents
- [Label Ruleset Multilabel Extension Design](../1_design/label_ruleset_multilabel_extension_design.md)
- [LightGBM Multi-Task Training Step Design](../1_design/lightgbm_multi_task_training_step_design.md)
- [MTGBM Multi-Task Learning Design](../1_design/mtgbm_multi_task_learning_design.md)

#### Implementation References
- [Label Ruleset Generation Script](../../src/cursus/steps/scripts/label_ruleset_generation.py)
- [Label Ruleset Execution Script](../../src/cursus/steps/scripts/label_ruleset_execution.py)
- [Label Ruleset Generation Config](../../src/cursus/steps/configs/config_label_ruleset_generation_step.py)

#### Entry Points
- [Step Design and Documentation Index](../00_entry_points/step_design_and_documentation_index.md)

---

*This implementation plan provides comprehensive specification for expanding the Label Ruleset system to support multi-label output for multi-task learning, enabling sophisticated category-conditional rule-based classification with domain knowledge integration.*
