"""
Ruleset Generation Script

Validates and optimizes user-defined classification rules following the
cursus framework pattern for transparent, maintainable rule-based classification.
"""

import os
import json
import argparse
import sys
import traceback
import copy
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Container path constants
CONTAINER_PATHS = {
    "INPUT_RULESET_CONFIGS": "/opt/ml/processing/input/ruleset_configs",
    "OUTPUT_VALIDATED_RULESET": "/opt/ml/processing/output/validated_ruleset",
    "OUTPUT_VALIDATION_REPORT": "/opt/ml/processing/output/validation_report",
}


class ValidationResult:
    """Result of a validation check."""

    def __init__(self, valid: bool = True):
        self.valid = valid
        self.missing_fields = []
        self.undeclared_fields = []
        self.type_errors = []
        self.invalid_labels = []
        self.uncovered_classes = []
        self.conflicting_rules = []
        self.tautologies = []
        self.contradictions = []
        self.unreachable_rules = []
        self.type_mismatches = []
        self.warnings = []

    def __dict__(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "valid": self.valid,
            "missing_fields": self.missing_fields,
            "undeclared_fields": self.undeclared_fields,
            "type_errors": self.type_errors,
            "invalid_labels": self.invalid_labels,
            "uncovered_classes": self.uncovered_classes,
            "conflicting_rules": self.conflicting_rules,
            "tautologies": self.tautologies,
            "contradictions": self.contradictions,
            "unreachable_rules": self.unreachable_rules,
            "type_mismatches": self.type_mismatches,
            "warnings": self.warnings,
        }


# RulesetFieldValidator removed - field_config is now auto-inferred from rules
# in the configuration layer, ensuring consistency at initialization time


class RulesetLabelValidator:
    """Validates output labels match configuration (extended for multilabel)."""

    def validate_labels(self, ruleset: dict) -> ValidationResult:
        """
        Validates all output_label values in rules.
        Extended to support multilabel mode.

        Args:
            ruleset: Input ruleset configuration

        Returns:
            ValidationResult with label validation status
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
                    result.type_errors.append(
                        f"label_values missing columns: {missing}"
                    )

            label_mapping = label_config.get("label_mapping", {})
            if isinstance(label_mapping, dict) and all(
                isinstance(v, dict) for v in label_mapping.values()
            ):
                missing = set(output_label_name) - set(label_mapping.keys())
                if missing:
                    result.valid = False
                    result.type_errors.append(
                        f"label_mapping missing columns: {missing}"
                    )

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
                            (
                                f"default_label[{col}]",
                                default_val,
                                f"not in label_values[{col}]",
                            )
                        )
                        logger.error(
                            f"Default label {default_val} for column {col} not in label_values"
                        )
                else:
                    if default_val not in label_values_set:
                        result.valid = False
                        result.invalid_labels.append(
                            (
                                f"default_label[{col}]",
                                default_val,
                                "not in label_values",
                            )
                        )
                        logger.error(
                            f"Default label {default_val} for column {col} not in label_values"
                        )
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
                        (
                            rule.get("rule_id"),
                            "empty_dict",
                            "output_label cannot be empty dict",
                        )
                    )
                    continue

                # Validate target columns exist
                valid_columns = (
                    set(output_label_name)
                    if isinstance(output_label_name, list)
                    else set()
                )
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
                                (
                                    rule.get("rule_id"),
                                    value,
                                    f"not valid for column {col}",
                                )
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

                # Check if label is valid
                if output_label not in label_values_set:
                    result.valid = False
                    result.invalid_labels.append(
                        (
                            rule.get("rule_id", "unknown"),
                            output_label,
                            "not in label_values",
                        )
                    )
                    logger.error(
                        f"Rule {rule.get('name', 'unknown')}: invalid output_label {output_label}"
                    )

        # Check binary constraints
        if label_type == "binary":
            if isinstance(label_values, list) and not label_values_set.issubset({0, 1}):
                result.valid = False
                result.warnings.append(
                    "Binary classification should use label_values [0, 1]"
                )
                logger.warning("Binary classification should use label_values [0, 1]")

        # Identify uncovered classes (for single-label mode)
        if label_type in ["binary", "multiclass"]:
            default_set = (
                {default_label}
                if not isinstance(default_label, dict)
                else set(default_label.values())
            )
            uncovered = label_values_set - used_labels - default_set
            if uncovered:
                result.uncovered_classes = list(uncovered)
                result.warnings.append(
                    f"Label values not covered by any rule: {uncovered}"
                )
                logger.warning(f"Label values not covered by any rule: {uncovered}")

        # Check for conflicting rules (same priority, different outputs)
        priority_labels = {}
        for rule in rules:
            priority = rule.get("priority")
            output_label = rule.get("output_label")
            rule_name = rule.get("name", "unknown")

            if priority in priority_labels:
                existing_label, existing_name = priority_labels[priority]
                if existing_label != output_label:
                    result.warnings.append(
                        f"Rules '{existing_name}' and '{rule_name}' have same priority {priority} but different outputs"
                    )
                    result.conflicting_rules.append(
                        (existing_name, rule_name, priority)
                    )
            else:
                priority_labels[priority] = (output_label, rule_name)

        return result


class RuleCoverageValidator:
    """Validates that all label columns have at least one rule."""

    def validate(self, label_config: dict, rules: List[dict]) -> ValidationResult:
        """
        Validates rule coverage for all label columns.

        Checks:
        - Each label column has at least one rule targeting it
        - Warns about orphan label columns

        Args:
            label_config: Label configuration
            rules: List of rule definitions

        Returns:
            ValidationResult with coverage validation status
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


class RulesetLogicValidator:
    """Validates rule logic for errors."""

    def validate_logic(self, ruleset: dict) -> ValidationResult:
        """
        Validates rule logic for common errors.

        Args:
            ruleset: Input ruleset configuration

        Returns:
            ValidationResult with logic validation status
        """
        result = ValidationResult()

        rules = ruleset.get("ruleset", [])
        field_types = ruleset.get("field_config", {}).get("field_types", {})

        for rule in rules:
            rule_name = rule.get("name", "unknown")
            conditions = rule.get("conditions", {})

            # Check for tautologies
            if self._is_tautology(conditions):
                result.tautologies.append(rule_name)
                result.warnings.append(f"Rule '{rule_name}' has always-true condition")
                logger.warning(f"Rule '{rule_name}' has always-true condition")

            # Check for contradictions
            if self._is_contradiction(conditions):
                result.valid = False
                result.contradictions.append(rule_name)
                logger.error(f"Rule '{rule_name}' has always-false condition")

            # Check operator-type compatibility
            type_errors = self._check_type_compatibility(conditions, field_types)
            if type_errors:
                result.valid = False
                result.type_mismatches.extend(
                    [(rule_name, error) for error in type_errors]
                )
                logger.error(f"Rule '{rule_name}' has type errors: {type_errors}")

        # Check for unreachable rules (shadowed by higher priority)
        unreachable = self._check_unreachable_rules(rules)
        if unreachable:
            result.unreachable_rules = unreachable
            result.warnings.extend(
                [f"Rule '{name}' may be unreachable" for name in unreachable]
            )
            logger.warning(f"Potentially unreachable rules: {unreachable}")

        return result

    def _is_tautology(self, condition: dict) -> bool:
        """Check if condition is always true (simplified check)."""
        # Simple heuristic: empty conditions or single "is_not_null" on non-nullable field
        if not condition:
            return True

        # Check for patterns like: field IS_NOT_NULL (always true if field exists)
        if condition.get("operator") == "is_not_null":
            return True  # Simplified - would need field metadata for proper check

        return False

    def _is_contradiction(self, condition: dict) -> bool:
        """Check if condition is always false (simplified check)."""
        # Check for obvious contradictions
        if "all_of" in condition:
            subconds = condition["all_of"]
            # Check for X = A AND X = B where A != B
            field_values = {}
            for subcond in subconds:
                if subcond.get("operator") == "equals":
                    field = subcond.get("field")
                    value = subcond.get("value")
                    if field in field_values and field_values[field] != value:
                        return True  # Contradiction found
                    field_values[field] = value

        return False

    def _check_type_compatibility(
        self, condition: dict, field_types: Dict[str, str]
    ) -> List[str]:
        """Check operator compatibility with field types."""
        errors = []

        # Handle nested conditions
        if "all_of" in condition:
            for subcond in condition["all_of"]:
                errors.extend(self._check_type_compatibility(subcond, field_types))
        elif "any_of" in condition:
            for subcond in condition["any_of"]:
                errors.extend(self._check_type_compatibility(subcond, field_types))
        elif "none_of" in condition:
            for subcond in condition["none_of"]:
                errors.extend(self._check_type_compatibility(subcond, field_types))
        else:
            # Check leaf condition
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")

            if field and operator and field in field_types:
                field_type = field_types[field]

                # Numeric operators on non-numeric fields
                if operator in [">", ">=", "<", "<="] and field_type not in [
                    "int",
                    "float",
                ]:
                    errors.append(
                        f"Numeric operator '{operator}' on non-numeric field '{field}' (type: {field_type})"
                    )

                # String operators on non-string fields
                if (
                    operator
                    in [
                        "contains",
                        "not_contains",
                        "starts_with",
                        "ends_with",
                        "regex_match",
                    ]
                    and field_type != "string"
                ):
                    errors.append(
                        f"String operator '{operator}' on non-string field '{field}' (type: {field_type})"
                    )

        return errors

    def _check_unreachable_rules(self, rules: List[dict]) -> List[str]:
        """Check for rules that may be unreachable due to priority shadowing."""
        unreachable = []

        # Sort rules by priority
        sorted_rules = sorted(rules, key=lambda r: r.get("priority", 999))

        # Simple heuristic: if two rules have very similar conditions and one has higher priority
        # This is a simplified version - full implementation would require condition analysis
        for i, rule in enumerate(sorted_rules):
            if not rule.get("enabled", True):
                continue

            rule_name = rule.get("name", f"rule_{i}")
            # Check if this rule might be shadowed (simplified check)
            # In practice, would need to analyze condition overlap
            pass  # Placeholder for more sophisticated logic

        return unreachable


def calculate_complexity(condition: dict) -> int:
    """
    Calculate complexity score for a condition.

    Args:
        condition: Condition expression

    Returns:
        Complexity score (higher = more complex)
    """
    if "all_of" in condition:
        return 1 + sum(calculate_complexity(c) for c in condition["all_of"])
    elif "any_of" in condition:
        return 1 + sum(calculate_complexity(c) for c in condition["any_of"])
    elif "none_of" in condition:
        return 1 + sum(calculate_complexity(c) for c in condition["none_of"])
    else:
        operator = condition.get("operator", "")
        value = condition.get("value")

        complexity = 1

        if operator == "regex_match":
            complexity += 2
        elif operator in ("in", "not_in") and isinstance(value, list):
            complexity += len(value) // 10

        return complexity


def extract_all_fields(condition: dict) -> List[str]:
    """
    Recursively extract all field names from a condition.

    Args:
        condition: Condition expression (may be nested)

    Returns:
        List of unique field names used
    """
    fields = []

    if "all_of" in condition:
        for subcond in condition["all_of"]:
            fields.extend(extract_all_fields(subcond))
    elif "any_of" in condition:
        for subcond in condition["any_of"]:
            fields.extend(extract_all_fields(subcond))
    elif "none_of" in condition:
        for subcond in condition["none_of"]:
            fields.extend(extract_all_fields(subcond))
    elif "field" in condition:
        fields.append(condition["field"])

    return list(set(fields))


def extract_fields_and_values(condition: dict) -> Dict[str, List[Any]]:
    """
    Recursively extract field names and their used values from conditions.

    Args:
        condition: Condition expression (may be nested)

    Returns:
        Dictionary mapping field names to list of values seen in conditions
    """
    field_values = {}

    if "all_of" in condition:
        for subcond in condition["all_of"]:
            for field, values in extract_fields_and_values(subcond).items():
                field_values.setdefault(field, []).extend(values)
    elif "any_of" in condition:
        for subcond in condition["any_of"]:
            for field, values in extract_fields_and_values(subcond).items():
                field_values.setdefault(field, []).extend(values)
    elif "none_of" in condition:
        for subcond in condition["none_of"]:
            for field, values in extract_fields_and_values(subcond).items():
                field_values.setdefault(field, []).extend(values)
    elif "field" in condition:
        field = condition["field"]
        value = condition.get("value")
        if value is not None:
            field_values[field] = [value]
        else:
            # For operators like is_null, is_not_null that don't have values
            field_values[field] = []

    return field_values


def infer_field_type(values: List[Any]) -> str:
    """
    Infer field type from values used in conditions.

    Args:
        values: List of values seen for a field

    Returns:
        Inferred type: 'string', 'int', 'float', or 'bool'
    """
    if not values:
        # No values seen, default to string
        return "string"

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
    # (more general types take precedence)
    if "string" in types_seen:
        return "string"
    if "float" in types_seen:
        return "float"
    if "int" in types_seen:
        return "int"
    if "bool" in types_seen:
        return "bool"

    return "string"  # default fallback


def infer_field_config_from_rules(
    rules: List[dict], log: Callable[[str], None] = print
) -> dict:
    """
    Infer complete field configuration from rule definitions.

    Analyzes all rules to extract:
    - Field names used
    - Field types based on values
    - Field usage statistics

    Args:
        rules: List of rule definitions
        log: Logging function

    Returns:
        Complete field_config dictionary with structure:
        {
            "required_fields": [],  # Empty when inferred
            "optional_fields": [...],  # All discovered fields
            "field_types": {...}  # Inferred types
        }
    """
    log("[INFO] Inferring field configuration from rules...")

    # Collect all fields and their values across all rules
    field_values = {}

    for rule in rules:
        conditions = rule.get("conditions", {})
        for field, values in extract_fields_and_values(conditions).items():
            field_values.setdefault(field, []).extend(values)

    # Infer types for each field
    field_types = {
        field: infer_field_type(values) for field, values in field_values.items()
    }

    # All fields used in rules are marked as required
    all_fields = sorted(field_values.keys())

    log(f"[INFO] Inferred {len(all_fields)} required fields from rules:")
    for field in all_fields:
        log(
            f"  - {field}: {field_types[field]} (used in {len([r for r in rules if field in extract_all_fields(r.get('conditions', {}))])} rules)"
        )

    return {
        "required_fields": all_fields,
        "field_types": field_types,
    }


def analyze_field_usage(rules: List[dict]) -> Dict[str, int]:
    """
    Analyze which fields are used most frequently across rules.

    Args:
        rules: List of rule definitions

    Returns:
        Dictionary mapping field names to usage count
    """
    field_counts = {}

    for rule in rules:
        fields = extract_all_fields(rule.get("conditions", {}))
        for field in fields:
            field_counts[field] = field_counts.get(field, 0) + 1

    sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_fields)


def optimize_ruleset(
    ruleset: dict,
    enable_complexity: bool = True,
    enable_field_grouping: bool = False,
    log: Callable[[str], None] = print,
) -> dict:
    """
    Optimize ruleset using multiple strategies.

    Args:
        ruleset: Input ruleset with unoptimized rules
        enable_complexity: Enable complexity-based ordering
        enable_field_grouping: Enable field usage grouping
        log: Logging function

    Returns:
        Optimized ruleset with reordered rules
    """
    rules = copy.deepcopy(ruleset.get("ruleset", []))

    log(f"[INFO] Starting optimization with {len(rules)} rules")

    # Step 1: Complexity-based ordering (since we don't have sample data typically)
    if enable_complexity:
        log("[INFO] Analyzing rule complexity...")
        for rule in rules:
            rule["complexity_score"] = calculate_complexity(rule.get("conditions", {}))
            log(
                f"  Rule '{rule.get('name', 'unnamed')}': complexity = {rule['complexity_score']}"
            )

        rules.sort(key=lambda r: r.get("complexity_score", 999))
        log("[INFO] Rules reordered by complexity")

    # Step 2: Field usage grouping (optional)
    if enable_field_grouping:
        log("[INFO] Grouping rules by field usage...")
        # Simplified grouping - keep rules with similar field usage together
        # Full implementation would use Jaccard similarity clustering
        for rule in rules:
            rule["used_fields"] = extract_all_fields(rule.get("conditions", {}))
        log("[INFO] Rules analyzed for field usage")

    # Step 3: Assign final priorities
    for i, rule in enumerate(rules, start=1):
        old_priority = rule.get("priority", i)
        rule["priority"] = i

        if old_priority != i:
            log(
                f"  Rule '{rule.get('name', 'unnamed')}': priority {old_priority} → {i}"
            )

    log(f"[INFO] Optimization complete: {len(rules)} rules reordered")

    return {
        **ruleset,
        "ruleset": rules,
        "optimization_metadata": {
            "complexity_enabled": enable_complexity,
            "field_grouping_enabled": enable_field_grouping,
        },
    }


def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str],
    environ_vars: Dict[str, str],
    job_args: Optional[argparse.Namespace] = None,
    logger: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Main logic for ruleset generation and validation.

    Args:
        input_paths: Dictionary with input paths
        output_paths: Dictionary with output paths
        environ_vars: Environment variables
        job_args: Command line arguments
        logger: Optional logger function

    Returns:
        Dictionary with processing results
    """
    log = logger or print

    # 1. Load auto-generated JSON configuration files
    configs_dir = Path(input_paths["ruleset_configs"])

    # Load label_config.json (required)
    label_config_file = configs_dir / "label_config.json"
    if not label_config_file.exists():
        raise FileNotFoundError(f"Required file not found: {label_config_file}")

    with open(label_config_file, "r") as f:
        label_config = json.load(f)
    log(f"[INFO] Loaded label config from {label_config_file}")

    # Load ruleset.json (required)
    ruleset_file = configs_dir / "ruleset.json"
    if not ruleset_file.exists():
        raise FileNotFoundError(f"Required file not found: {ruleset_file}")

    with open(ruleset_file, "r") as f:
        ruleset_rules = json.load(f)
    log(f"[INFO] Loaded {len(ruleset_rules)} rules from {ruleset_file}")

    # Infer field_config from rules (not loaded from file)
    field_config = infer_field_config_from_rules(ruleset_rules, log=log)
    log(
        f"[INFO] Inferred field configuration with {len(field_config['required_fields'])} required fields"
    )

    # Assemble user ruleset
    user_ruleset = {
        "label_config": label_config,
        "field_config": field_config,
        "ruleset": ruleset_rules,
    }

    # 2. Initialize validators (field validation removed - handled at config level)
    label_validator = RulesetLabelValidator()
    logic_validator = RulesetLogicValidator()
    coverage_validator = RuleCoverageValidator()

    # 3. Run validation
    log("[INFO] Running validation...")

    # Field validation is no longer performed here - field_config is auto-inferred
    # and validated at configuration time, ensuring consistency before script execution
    enable_label = environ_vars.get("ENABLE_LABEL_VALIDATION", "true").lower() == "true"
    enable_logic = environ_vars.get("ENABLE_LOGIC_VALIDATION", "true").lower() == "true"

    label_validation = (
        label_validator.validate_labels(user_ruleset) if enable_label else None
    )
    logic_validation = (
        logic_validator.validate_logic(user_ruleset) if enable_logic else None
    )

    # NEW: Additional coverage check for multilabel
    label_type = label_config.get("output_label_type", "binary")
    if label_type == "multilabel":
        coverage_validation = coverage_validator.validate(label_config, ruleset_rules)
        if coverage_validation.warnings:
            log("[INFO] Coverage validation warnings:")
            for warning in coverage_validation.warnings:
                log(f"  [WARNING] {warning}")

    # 4. Check validation results
    all_valid = (label_validation.valid if label_validation else True) and (
        logic_validation.valid if logic_validation else True
    )

    if not all_valid:
        log("[ERROR] Validation failed!")

        # Save detailed validation report
        validation_report = {
            "validation_status": "failed",
            "label_validation": label_validation.__dict__()
            if label_validation
            else None,
            "logic_validation": logic_validation.__dict__()
            if logic_validation
            else None,
        }

        report_path = output_paths.get("validation_report")
        if report_path:
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            with open(report_path, "w") as f:
                json.dump(validation_report, f, indent=2)

        raise RuntimeError(
            "Ruleset validation failed. See validation report for details."
        )

    log("[INFO] Validation passed")

    # 6. Optimize ruleset (if enabled)
    enable_optimization = (
        environ_vars.get("ENABLE_RULE_OPTIMIZATION", "true").lower() == "true"
    )
    if enable_optimization:
        log("[INFO] Optimizing ruleset...")
        optimized_ruleset = optimize_ruleset(
            user_ruleset,
            enable_complexity=True,
            enable_field_grouping=False,  # Simplified for now
            log=log,
        )
    else:
        log("[INFO] Skipping optimization")
        optimized_ruleset = user_ruleset

    # 7. Generate validated ruleset with metadata
    validated_ruleset = {
        "version": "1.0",
        "generated_timestamp": datetime.now().isoformat(),
        "label_config": optimized_ruleset["label_config"],
        "field_config": optimized_ruleset["field_config"],
        "ruleset": optimized_ruleset["ruleset"],
        "metadata": {
            "total_rules": len(optimized_ruleset["ruleset"]),
            "enabled_rules": sum(
                1 for r in optimized_ruleset["ruleset"] if r.get("enabled", True)
            ),
            "disabled_rules": sum(
                1 for r in optimized_ruleset["ruleset"] if not r.get("enabled", True)
            ),
            "field_usage": analyze_field_usage(optimized_ruleset["ruleset"]),
            "validation_summary": {
                "field_validation": "passed_at_config_level",
                "label_validation": "passed"
                if not label_validation or label_validation.valid
                else "failed",
                "logic_validation": "passed"
                if not logic_validation
                or (logic_validation.valid and not logic_validation.warnings)
                else "passed_with_warnings"
                if logic_validation.valid
                else "failed",
                "warnings": logic_validation.warnings if logic_validation else [],
            },
        },
    }

    # 8. Save validated ruleset
    validated_ruleset_path = output_paths["validated_ruleset"]
    os.makedirs(os.path.dirname(validated_ruleset_path), exist_ok=True)
    with open(validated_ruleset_path, "w") as f:
        json.dump(validated_ruleset, f, indent=2)

    log(f"[INFO] Saved validated ruleset to {validated_ruleset_path}")

    # 9. Save validation report
    validation_report = {
        "validation_status": "passed",
        "field_validation": {"passed_at_config_level": True},
        "label_validation": label_validation.__dict__()
        if label_validation
        else {"skipped": True},
        "logic_validation": logic_validation.__dict__()
        if logic_validation
        else {"skipped": True},
        "optimization_applied": enable_optimization,
        "metadata": validated_ruleset["metadata"],
    }

    report_path = output_paths.get("validation_report")
    if report_path:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(validation_report, f, indent=2)
        log(f"[INFO] Saved validation report to {report_path}")

    log("[INFO] Ruleset generation complete")

    return {
        "validated_ruleset": validated_ruleset,
        "validation_report": validation_report,
    }


if __name__ == "__main__":
    try:
        # Set up path dictionaries
        input_paths = {
            "ruleset_configs": CONTAINER_PATHS["INPUT_RULESET_CONFIGS"],
        }

        output_paths = {
            "validated_ruleset": os.path.join(
                CONTAINER_PATHS["OUTPUT_VALIDATED_RULESET"], "validated_ruleset.json"
            ),
            "validation_report": os.path.join(
                CONTAINER_PATHS["OUTPUT_VALIDATION_REPORT"], "validation_report.json"
            ),
        }

        # Get configuration from environment variables
        environ_vars = {
            "ENABLE_FIELD_VALIDATION": os.environ.get(
                "ENABLE_FIELD_VALIDATION", "true"
            ),
            "ENABLE_LABEL_VALIDATION": os.environ.get(
                "ENABLE_LABEL_VALIDATION", "true"
            ),
            "ENABLE_LOGIC_VALIDATION": os.environ.get(
                "ENABLE_LOGIC_VALIDATION", "true"
            ),
            "ENABLE_RULE_OPTIMIZATION": os.environ.get(
                "ENABLE_RULE_OPTIMIZATION", "true"
            ),
        }

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)

        # Log key parameters
        logger.info(f"Starting ruleset generation with parameters:")
        logger.info(f"  Field Validation: {environ_vars['ENABLE_FIELD_VALIDATION']}")
        logger.info(f"  Label Validation: {environ_vars['ENABLE_LABEL_VALIDATION']}")
        logger.info(f"  Logic Validation: {environ_vars['ENABLE_LOGIC_VALIDATION']}")
        logger.info(f"  Rule Optimization: {environ_vars['ENABLE_RULE_OPTIMIZATION']}")

        # No command line arguments needed for this script
        args = None

        # Execute the main processing logic
        result = main(
            input_paths=input_paths,
            output_paths=output_paths,
            environ_vars=environ_vars,
            job_args=args,
            logger=logger.info,
        )

        # Log completion summary
        logger.info(f"Ruleset generation completed successfully. Quality score: passed")
        sys.exit(0)

    except Exception as e:
        logging.error(f"Error in ruleset generation script: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)
