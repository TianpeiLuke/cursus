"""
Comprehensive test suite for label_ruleset_generation.py script.

This test suite follows pytest best practices and provides thorough coverage
of the label ruleset generation and validation functionality including:
- Binary, multiclass, and multilabel classification validation
- Label validation (values, defaults, output labels)
- Rule logic validation (tautologies, contradictions, type compatibility)
- Field inference from rules
- Rule optimization and complexity analysis
- Coverage validation for multilabel
- Integration testing of main workflow
"""

import pytest
from unittest.mock import patch, MagicMock, Mock, mock_open, call
import os
import sys
import tempfile
import shutil
import json
import copy
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Import the components to be tested
from cursus.steps.scripts.label_ruleset_generation import (
    ValidationResult,
    RulesetLabelValidator,
    RuleCoverageValidator,
    RulesetLogicValidator,
    calculate_complexity,
    extract_all_fields,
    extract_fields_and_values,
    infer_field_type,
    infer_field_config_from_rules,
    analyze_field_usage,
    optimize_ruleset,
    main,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_validation_result_initialization(self):
        """Test ValidationResult initialization with default values."""
        result = ValidationResult()

        assert result.valid is True
        assert result.missing_fields == []
        assert result.undeclared_fields == []
        assert result.type_errors == []
        assert result.invalid_labels == []
        assert result.uncovered_classes == []
        assert result.conflicting_rules == []
        assert result.tautologies == []
        assert result.contradictions == []
        assert result.unreachable_rules == []
        assert result.type_mismatches == []
        assert result.warnings == []

    def test_validation_result_to_dict(self):
        """Test ValidationResult conversion to dictionary."""
        result = ValidationResult(valid=False)
        result.invalid_labels = [("rule1", "invalid_label", "not in label_values")]
        result.warnings = ["Warning message"]

        result_dict = result.__dict__()

        assert isinstance(result_dict, dict)
        assert result_dict["valid"] is False
        assert len(result_dict["invalid_labels"]) == 1
        assert len(result_dict["warnings"]) == 1

    def test_validation_result_invalid_initialization(self):
        """Test ValidationResult with invalid initialization."""
        result = ValidationResult(valid=False)

        assert result.valid is False


class TestRulesetLabelValidator:
    """Tests for RulesetLabelValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RulesetLabelValidator()

    @pytest.fixture
    def binary_ruleset(self):
        """Create sample binary ruleset."""
        return {
            "label_config": {
                "label_values": [0, 1],
                "output_label_type": "binary",
                "default_label": 0,
                "output_label_name": "prediction",
            },
            "ruleset": [
                {
                    "rule_id": "rule_1",
                    "name": "High score rule",
                    "conditions": {"field": "score", "operator": ">", "value": 0.7},
                    "output_label": 1,
                    "priority": 1,
                    "enabled": True,
                },
                {
                    "rule_id": "rule_2",
                    "name": "Low score rule",
                    "conditions": {"field": "score", "operator": "<=", "value": 0.3},
                    "output_label": 0,
                    "priority": 2,
                    "enabled": True,
                },
            ],
        }

    @pytest.fixture
    def multiclass_ruleset(self):
        """Create sample multiclass ruleset."""
        return {
            "label_config": {
                "label_values": [0, 1, 2],
                "output_label_type": "multiclass",
                "default_label": 0,
                "output_label_name": "category",
            },
            "ruleset": [
                {
                    "rule_id": "rule_1",
                    "name": "Category 1 rule",
                    "conditions": {"field": "feature_a", "operator": ">", "value": 10},
                    "output_label": 1,
                    "priority": 1,
                },
                {
                    "rule_id": "rule_2",
                    "name": "Category 2 rule",
                    "conditions": {"field": "feature_b", "operator": ">", "value": 20},
                    "output_label": 2,
                    "priority": 2,
                },
            ],
        }

    @pytest.fixture
    def multilabel_ruleset(self):
        """Create sample multilabel ruleset."""
        return {
            "label_config": {
                "label_values": {"col1": [0, 1], "col2": [0, 1]},
                "output_label_type": "multilabel",
                "default_label": {"col1": 0, "col2": 0},
                "output_label_name": ["col1", "col2"],
                "label_mapping": {
                    "col1": {"0": "negative", "1": "positive"},
                    "col2": {"0": "negative", "1": "positive"},
                },
            },
            "ruleset": [
                {
                    "rule_id": "rule_1",
                    "name": "Multilabel rule 1",
                    "conditions": {"field": "feature", "operator": ">", "value": 5},
                    "output_label": {"col1": 1, "col2": 0},
                    "priority": 1,
                },
                {
                    "rule_id": "rule_2",
                    "name": "Multilabel rule 2",
                    "conditions": {"field": "feature", "operator": "<", "value": 3},
                    "output_label": {"col1": 0, "col2": 1},
                    "priority": 2,
                },
            ],
        }

    def test_validate_labels_binary_success(self, validator, binary_ruleset):
        """Test successful binary label validation."""
        result = validator.validate_labels(binary_ruleset)

        assert result.valid is True
        assert len(result.invalid_labels) == 0

    def test_validate_labels_binary_invalid_label(self, validator, binary_ruleset):
        """Test binary validation with invalid label."""
        binary_ruleset["ruleset"][0]["output_label"] = 2  # Invalid for binary

        result = validator.validate_labels(binary_ruleset)

        assert result.valid is False
        assert len(result.invalid_labels) > 0

    def test_validate_labels_binary_invalid_default(self, validator, binary_ruleset):
        """Test binary validation with invalid default label."""
        binary_ruleset["label_config"]["default_label"] = 3

        result = validator.validate_labels(binary_ruleset)

        assert result.valid is False
        assert any("default_label" in str(label) for label in result.invalid_labels)

    def test_validate_labels_binary_constraint(self, validator, binary_ruleset):
        """Test binary classification label value constraint."""
        binary_ruleset["label_config"]["label_values"] = [0, 1, 2]

        result = validator.validate_labels(binary_ruleset)

        # Source code sets valid=False when binary has non-{0,1} values
        assert result.valid is False
        assert len(result.warnings) > 0
        assert any("Binary classification" in warning for warning in result.warnings)

    def test_validate_labels_multiclass_success(self, validator, multiclass_ruleset):
        """Test successful multiclass label validation."""
        result = validator.validate_labels(multiclass_ruleset)

        assert result.valid is True
        assert len(result.invalid_labels) == 0

    def test_validate_labels_multiclass_uncovered_classes(
        self, validator, multiclass_ruleset
    ):
        """Test multiclass with uncovered label values."""
        multiclass_ruleset["label_config"]["label_values"] = [0, 1, 2, 3]

        result = validator.validate_labels(multiclass_ruleset)

        assert len(result.uncovered_classes) > 0
        assert 3 in result.uncovered_classes

    def test_validate_labels_conflicting_rules(self, validator, binary_ruleset):
        """Test detection of conflicting rules with same priority."""
        binary_ruleset["ruleset"][1]["priority"] = 1  # Same as rule_1
        binary_ruleset["ruleset"][1]["output_label"] = (
            1  # Same priority, different output than rule_1 (which has output_label=1)
        )
        # Need to make outputs actually different - rule_1 has output_label=1, so change rule_2 to 0
        binary_ruleset["ruleset"][0]["output_label"] = 1
        binary_ruleset["ruleset"][1]["output_label"] = 0  # Now they're different

        result = validator.validate_labels(binary_ruleset)

        assert len(result.warnings) > 0
        assert len(result.conflicting_rules) > 0

    def test_validate_labels_multilabel_success(self, validator, multilabel_ruleset):
        """Test successful multilabel validation."""
        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is True
        assert len(result.invalid_labels) == 0

    def test_validate_labels_multilabel_wrong_type(self, validator, multilabel_ruleset):
        """Test multilabel validation with wrong output_label_name type."""
        multilabel_ruleset["label_config"]["output_label_name"] = "single_column"

        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is False
        assert any(
            "list for output_label_name" in str(err) for err in result.type_errors
        )

    def test_validate_labels_multilabel_duplicate_columns(
        self, validator, multilabel_ruleset
    ):
        """Test multilabel validation with duplicate column names."""
        multilabel_ruleset["label_config"]["output_label_name"] = ["col1", "col1"]

        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is False
        assert any("Duplicate column names" in str(err) for err in result.type_errors)

    def test_validate_labels_multilabel_missing_columns(
        self, validator, multilabel_ruleset
    ):
        """Test multilabel validation with missing columns in label_values."""
        multilabel_ruleset["label_config"]["label_values"] = {"col1": [0, 1]}

        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is False
        assert any("missing columns" in str(err) for err in result.type_errors)

    def test_validate_labels_multilabel_invalid_target_column(
        self, validator, multilabel_ruleset
    ):
        """Test multilabel validation with invalid target column in rule."""
        multilabel_ruleset["ruleset"][0]["output_label"] = {"col3": 1}

        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is False
        assert any(
            "not in output_label_name" in str(label[2])
            for label in result.invalid_labels
        )

    def test_validate_labels_multilabel_invalid_value(
        self, validator, multilabel_ruleset
    ):
        """Test multilabel validation with invalid value for column."""
        multilabel_ruleset["ruleset"][0]["output_label"] = {"col1": 5}

        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is False
        assert any(
            "not valid for column" in str(label[2]) for label in result.invalid_labels
        )

    def test_validate_labels_multilabel_dict_requires_multilabel_mode(
        self, validator, binary_ruleset
    ):
        """Test that dict output_label requires multilabel mode."""
        binary_ruleset["ruleset"][0]["output_label"] = {"col1": 1}

        result = validator.validate_labels(binary_ruleset)

        assert result.valid is False
        assert any("requires multilabel mode" in str(err) for err in result.type_errors)

    def test_validate_labels_multilabel_empty_dict(self, validator, multilabel_ruleset):
        """Test multilabel validation with empty dict output_label."""
        multilabel_ruleset["ruleset"][0]["output_label"] = {}

        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is False
        assert any("empty dict" in str(label[2]) for label in result.invalid_labels)

    def test_validate_labels_per_column_default_label(
        self, validator, multilabel_ruleset
    ):
        """Test validation with per-column default labels."""
        result = validator.validate_labels(multilabel_ruleset)

        # Should pass with valid per-column defaults
        assert result.valid is True

    def test_validate_labels_per_column_default_invalid(
        self, validator, multilabel_ruleset
    ):
        """Test validation with invalid per-column default label."""
        multilabel_ruleset["label_config"]["default_label"] = {"col1": 5, "col2": 0}

        result = validator.validate_labels(multilabel_ruleset)

        assert result.valid is False
        assert any(
            "not in label_values" in str(label[2]) for label in result.invalid_labels
        )


class TestRuleCoverageValidator:
    """Tests for RuleCoverageValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RuleCoverageValidator()

    def test_validate_binary_mode_skipped(self, validator):
        """Test that validation is skipped for binary mode."""
        label_config = {
            "output_label_type": "binary",
            "output_label_name": "prediction",
        }
        rules = []

        result = validator.validate(label_config, rules)

        assert result.valid is True
        assert len(result.warnings) == 0

    def test_validate_multiclass_mode_skipped(self, validator):
        """Test that validation is skipped for multiclass mode."""
        label_config = {
            "output_label_type": "multiclass",
            "output_label_name": "category",
        }
        rules = []

        result = validator.validate(label_config, rules)

        assert result.valid is True
        assert len(result.warnings) == 0

    def test_validate_multilabel_all_columns_covered(self, validator):
        """Test multilabel validation with all columns covered."""
        label_config = {
            "output_label_type": "multilabel",
            "output_label_name": ["col1", "col2"],
        }
        rules = [
            {"enabled": True, "output_label": {"col1": 1, "col2": 0}},
            {"enabled": True, "output_label": {"col1": 0, "col2": 1}},
        ]

        result = validator.validate(label_config, rules)

        assert result.valid is True
        assert len(result.warnings) == 0

    def test_validate_multilabel_missing_column_coverage(self, validator):
        """Test multilabel validation with uncovered columns."""
        label_config = {
            "output_label_type": "multilabel",
            "output_label_name": ["col1", "col2", "col3"],
        }
        rules = [{"enabled": True, "output_label": {"col1": 1, "col2": 0}}]

        result = validator.validate(label_config, rules)

        assert len(result.warnings) > 0
        assert any("without rules" in warning for warning in result.warnings)

    def test_validate_multilabel_disabled_rules_ignored(self, validator):
        """Test that disabled rules are ignored in coverage check."""
        label_config = {
            "output_label_type": "multilabel",
            "output_label_name": ["col1", "col2"],
        }
        rules = [
            {"enabled": False, "output_label": {"col1": 1, "col2": 1}},
            {"enabled": True, "output_label": {"col1": 1}},
        ]

        result = validator.validate(label_config, rules)

        # col2 should be uncovered since only disabled rule targets it
        assert len(result.warnings) > 0


class TestRulesetLogicValidator:
    """Tests for RulesetLogicValidator class."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return RulesetLogicValidator()

    @pytest.fixture
    def sample_ruleset(self):
        """Create sample ruleset for logic validation."""
        return {
            "field_config": {
                "field_types": {"age": "int", "name": "string", "score": "float"}
            },
            "ruleset": [
                {
                    "name": "Valid rule",
                    "conditions": {"field": "age", "operator": ">", "value": 18},
                }
            ],
        }

    def test_validate_logic_success(self, validator, sample_ruleset):
        """Test successful logic validation."""
        result = validator.validate_logic(sample_ruleset)

        assert result.valid is True
        assert len(result.contradictions) == 0
        assert len(result.type_mismatches) == 0

    def test_validate_logic_tautology(self, validator, sample_ruleset):
        """Test detection of tautology (always true condition)."""
        sample_ruleset["ruleset"][0]["conditions"] = {}

        result = validator.validate_logic(sample_ruleset)

        assert len(result.tautologies) > 0
        assert len(result.warnings) > 0

    def test_validate_logic_contradiction(self, validator, sample_ruleset):
        """Test detection of contradiction (always false condition)."""
        sample_ruleset["ruleset"][0]["conditions"] = {
            "all_of": [
                {"field": "age", "operator": "equals", "value": 20},
                {"field": "age", "operator": "equals", "value": 30},
            ]
        }

        result = validator.validate_logic(sample_ruleset)

        assert result.valid is False
        assert len(result.contradictions) > 0

    def test_validate_logic_type_mismatch_numeric_operator(
        self, validator, sample_ruleset
    ):
        """Test detection of numeric operator on non-numeric field."""
        sample_ruleset["ruleset"][0]["conditions"] = {
            "field": "name",
            "operator": ">",
            "value": 10,
        }

        result = validator.validate_logic(sample_ruleset)

        assert result.valid is False
        assert len(result.type_mismatches) > 0
        assert any("Numeric operator" in str(err[1]) for err in result.type_mismatches)

    def test_validate_logic_type_mismatch_string_operator(
        self, validator, sample_ruleset
    ):
        """Test detection of string operator on non-string field."""
        sample_ruleset["ruleset"][0]["conditions"] = {
            "field": "age",
            "operator": "contains",
            "value": "test",
        }

        result = validator.validate_logic(sample_ruleset)

        assert result.valid is False
        assert len(result.type_mismatches) > 0
        assert any("String operator" in str(err[1]) for err in result.type_mismatches)

    def test_validate_logic_nested_conditions(self, validator, sample_ruleset):
        """Test validation of nested logical conditions."""
        sample_ruleset["ruleset"][0]["conditions"] = {
            "all_of": [
                {"field": "age", "operator": ">", "value": 18},
                {
                    "any_of": [
                        {"field": "score", "operator": ">=", "value": 0.8},
                        {"field": "name", "operator": "equals", "value": "test"},
                    ]
                },
            ]
        }

        result = validator.validate_logic(sample_ruleset)

        # Should validate nested structure
        assert result.valid is True

    def test_validate_logic_none_of_condition(self, validator, sample_ruleset):
        """Test validation of none_of condition."""
        sample_ruleset["ruleset"][0]["conditions"] = {
            "none_of": [
                {"field": "age", "operator": "<", "value": 18},
                {"field": "score", "operator": "<", "value": 0.5},
            ]
        }

        result = validator.validate_logic(sample_ruleset)

        assert result.valid is True

    def test_is_tautology_is_not_null(self, validator):
        """Test tautology detection for is_not_null operator."""
        condition = {"operator": "is_not_null", "field": "age"}

        result = validator._is_tautology(condition)

        assert result is True

    def test_is_contradiction_conflicting_equals(self, validator):
        """Test contradiction detection for conflicting equals."""
        condition = {
            "all_of": [
                {"field": "status", "operator": "equals", "value": "active"},
                {"field": "status", "operator": "equals", "value": "inactive"},
            ]
        }

        result = validator._is_contradiction(condition)

        assert result is True

    def test_check_type_compatibility_all_operators(self, validator):
        """Test type compatibility for all operator types."""
        field_types = {"age": "int", "name": "string", "score": "float"}

        # Test numeric operators
        errors = validator._check_type_compatibility(
            {"field": "age", "operator": ">=", "value": 18}, field_types
        )
        assert len(errors) == 0

        # Test string operators
        errors = validator._check_type_compatibility(
            {"field": "name", "operator": "starts_with", "value": "test"}, field_types
        )
        assert len(errors) == 0


class TestComplexityCalculation:
    """Tests for complexity calculation functions."""

    def test_calculate_complexity_simple_condition(self):
        """Test complexity for simple condition."""
        condition = {"field": "age", "operator": ">", "value": 18}

        complexity = calculate_complexity(condition)

        assert complexity == 1

    def test_calculate_complexity_all_of(self):
        """Test complexity for all_of condition."""
        condition = {
            "all_of": [
                {"field": "age", "operator": ">", "value": 18},
                {"field": "score", "operator": ">=", "value": 0.8},
            ]
        }

        complexity = calculate_complexity(condition)

        assert complexity == 3  # 1 + 1 + 1

    def test_calculate_complexity_nested(self):
        """Test complexity for nested conditions."""
        condition = {
            "all_of": [
                {"field": "age", "operator": ">", "value": 18},
                {
                    "any_of": [
                        {"field": "score", "operator": ">=", "value": 0.8},
                        {"field": "status", "operator": "equals", "value": "active"},
                    ]
                },
            ]
        }

        complexity = calculate_complexity(condition)

        assert complexity == 5  # 1 + 1 + (1 + 1 + 1)

    def test_calculate_complexity_regex(self):
        """Test complexity for regex operator."""
        condition = {"field": "email", "operator": "regex_match", "value": ".*@.*"}

        complexity = calculate_complexity(condition)

        assert complexity == 3  # 1 + 2 for regex

    def test_calculate_complexity_in_operator(self):
        """Test complexity for in operator with list."""
        condition = {
            "field": "category",
            "operator": "in",
            "value": ["A", "B", "C", "D", "E"],
        }

        complexity = calculate_complexity(condition)

        assert complexity == 1  # 1 + 0 (less than 10 items)

    def test_calculate_complexity_in_operator_large_list(self):
        """Test complexity for in operator with large list."""
        condition = {"field": "id", "operator": "in", "value": list(range(100))}

        complexity = calculate_complexity(condition)

        assert complexity == 11  # 1 + 100//10


class TestFieldExtraction:
    """Tests for field extraction functions."""

    def test_extract_all_fields_simple(self):
        """Test extraction of fields from simple condition."""
        condition = {"field": "age", "operator": ">", "value": 18}

        fields = extract_all_fields(condition)

        assert fields == ["age"]

    def test_extract_all_fields_nested(self):
        """Test extraction of fields from nested conditions."""
        condition = {
            "all_of": [
                {"field": "age", "operator": ">", "value": 18},
                {"field": "score", "operator": ">=", "value": 0.8},
            ]
        }

        fields = extract_all_fields(condition)

        assert set(fields) == {"age", "score"}

    def test_extract_all_fields_deeply_nested(self):
        """Test extraction from deeply nested conditions."""
        condition = {
            "all_of": [
                {"field": "age", "operator": ">", "value": 18},
                {
                    "any_of": [
                        {"field": "score", "operator": ">=", "value": 0.8},
                        {
                            "none_of": [
                                {
                                    "field": "status",
                                    "operator": "equals",
                                    "value": "inactive",
                                }
                            ]
                        },
                    ]
                },
            ]
        }

        fields = extract_all_fields(condition)

        assert set(fields) == {"age", "score", "status"}

    def test_extract_fields_and_values_simple(self):
        """Test extraction of fields and values."""
        condition = {"field": "age", "operator": ">", "value": 18}

        field_values = extract_fields_and_values(condition)

        assert "age" in field_values
        assert 18 in field_values["age"]

    def test_extract_fields_and_values_nested(self):
        """Test extraction of fields and values from nested conditions."""
        condition = {
            "all_of": [
                {"field": "age", "operator": ">", "value": 18},
                {"field": "age", "operator": "<", "value": 65},
                {"field": "score", "operator": ">=", "value": 0.8},
            ]
        }

        field_values = extract_fields_and_values(condition)

        assert "age" in field_values
        assert 18 in field_values["age"]
        assert 65 in field_values["age"]
        assert "score" in field_values
        assert 0.8 in field_values["score"]

    def test_extract_fields_and_values_no_value(self):
        """Test extraction for operators without values."""
        condition = {"field": "email", "operator": "is_not_null"}

        field_values = extract_fields_and_values(condition)

        assert "email" in field_values
        assert field_values["email"] == []


class TestFieldTypeInference:
    """Tests for field type inference."""

    def test_infer_field_type_int(self):
        """Test type inference for integer values."""
        values = [1, 2, 3, 4, 5]

        field_type = infer_field_type(values)

        assert field_type == "int"

    def test_infer_field_type_float(self):
        """Test type inference for float values."""
        values = [1.5, 2.5, 3.5]

        field_type = infer_field_type(values)

        assert field_type == "float"

    def test_infer_field_type_string(self):
        """Test type inference for string values."""
        values = ["a", "b", "c"]

        field_type = infer_field_type(values)

        assert field_type == "string"

    def test_infer_field_type_bool(self):
        """Test type inference for boolean values."""
        values = [True, False, True]

        field_type = infer_field_type(values)

        assert field_type == "bool"

    def test_infer_field_type_mixed_numeric(self):
        """Test type inference for mixed int/float (should return float)."""
        values = [1, 2.5, 3, 4.5]

        field_type = infer_field_type(values)

        assert field_type == "float"

    def test_infer_field_type_mixed_with_string(self):
        """Test type inference for mixed types (should return string)."""
        values = [1, "text", 3.5]

        field_type = infer_field_type(values)

        assert field_type == "string"

    def test_infer_field_type_empty(self):
        """Test type inference for empty values."""
        values = []

        field_type = infer_field_type(values)

        assert field_type == "string"  # Default

    def test_infer_field_type_with_none(self):
        """Test type inference ignoring None values."""
        values = [None, 1, None, 2, None]

        field_type = infer_field_type(values)

        assert field_type == "int"


class TestFieldConfigInference:
    """Tests for field configuration inference from rules."""

    @pytest.fixture
    def sample_rules(self):
        """Create sample rules for inference testing."""
        return [
            {
                "name": "Rule 1",
                "conditions": {
                    "all_of": [
                        {"field": "age", "operator": ">", "value": 18},
                        {"field": "score", "operator": ">=", "value": 0.8},
                    ]
                },
            },
            {
                "name": "Rule 2",
                "conditions": {
                    "field": "status",
                    "operator": "equals",
                    "value": "active",
                },
            },
        ]

    def test_infer_field_config_from_rules(self, sample_rules):
        """Test inference of field configuration from rules."""
        log_messages = []

        field_config = infer_field_config_from_rules(
            sample_rules, log=log_messages.append
        )

        assert "required_fields" in field_config
        assert "field_types" in field_config
        assert "age" in field_config["required_fields"]
        assert "score" in field_config["required_fields"]
        assert "status" in field_config["required_fields"]
        assert field_config["field_types"]["age"] == "int"
        assert field_config["field_types"]["score"] == "float"
        assert field_config["field_types"]["status"] == "string"

    def test_infer_field_config_empty_rules(self):
        """Test inference with empty rules list."""
        field_config = infer_field_config_from_rules([], log=print)

        assert field_config["required_fields"] == []
        assert field_config["field_types"] == {}

    def test_analyze_field_usage(self, sample_rules):
        """Test analysis of field usage frequency."""
        field_counts = analyze_field_usage(sample_rules)

        assert "age" in field_counts
        assert "score" in field_counts
        assert "status" in field_counts
        assert field_counts["age"] == 1
        assert field_counts["score"] == 1
        assert field_counts["status"] == 1

    def test_analyze_field_usage_multiple_occurrences(self):
        """Test field usage analysis with repeated fields."""
        rules = [
            {
                "name": "R1",
                "conditions": {"field": "age", "operator": ">", "value": 18},
            },
            {
                "name": "R2",
                "conditions": {"field": "age", "operator": "<", "value": 65},
            },
            {
                "name": "R3",
                "conditions": {"field": "score", "operator": ">=", "value": 0.5},
            },
        ]

        field_counts = analyze_field_usage(rules)

        assert field_counts["age"] == 2
        assert field_counts["score"] == 1


class TestRulesetOptimization:
    """Tests for ruleset optimization functions."""

    @pytest.fixture
    def unoptimized_ruleset(self):
        """Create unoptimized ruleset for testing."""
        return {
            "label_config": {"label_values": [0, 1]},
            "field_config": {"field_types": {"age": "int", "score": "float"}},
            "ruleset": [
                {
                    "name": "Complex rule",
                    "priority": 1,
                    "conditions": {
                        "all_of": [
                            {"field": "age", "operator": ">", "value": 18},
                            {
                                "any_of": [
                                    {"field": "score", "operator": ">=", "value": 0.8},
                                    {"field": "score", "operator": ">=", "value": 0.9},
                                ]
                            },
                        ]
                    },
                },
                {
                    "name": "Simple rule",
                    "priority": 2,
                    "conditions": {"field": "age", "operator": ">", "value": 65},
                },
            ],
        }

    def test_optimize_ruleset_complexity_ordering(self, unoptimized_ruleset):
        """Test ruleset optimization with complexity ordering."""
        log_messages = []

        optimized = optimize_ruleset(
            unoptimized_ruleset,
            enable_complexity=True,
            enable_field_grouping=False,
            log=log_messages.append,
        )

        assert len(optimized["ruleset"]) == 2
        # Simple rule should come first (lower complexity)
        assert optimized["ruleset"][0]["name"] == "Simple rule"
        assert optimized["ruleset"][0]["priority"] == 1
        assert optimized["ruleset"][1]["name"] == "Complex rule"
        assert optimized["ruleset"][1]["priority"] == 2

    def test_optimize_ruleset_without_complexity(self, unoptimized_ruleset):
        """Test ruleset optimization without complexity ordering."""
        log_messages = []

        optimized = optimize_ruleset(
            unoptimized_ruleset,
            enable_complexity=False,
            enable_field_grouping=False,
            log=log_messages.append,
        )

        # Priorities should still be assigned
        assert optimized["ruleset"][0]["priority"] == 1
        assert optimized["ruleset"][1]["priority"] == 2

    def test_optimize_ruleset_field_grouping(self, unoptimized_ruleset):
        """Test ruleset optimization with field grouping enabled."""
        log_messages = []

        optimized = optimize_ruleset(
            unoptimized_ruleset,
            enable_complexity=True,
            enable_field_grouping=True,
            log=log_messages.append,
        )

        # Should add used_fields to rules
        assert "used_fields" in optimized["ruleset"][0]
        assert "used_fields" in optimized["ruleset"][1]

    def test_optimize_ruleset_metadata(self, unoptimized_ruleset):
        """Test that optimization adds metadata."""
        optimized = optimize_ruleset(unoptimized_ruleset, log=print)

        assert "optimization_metadata" in optimized
        assert "complexity_enabled" in optimized["optimization_metadata"]
        assert "field_grouping_enabled" in optimized["optimization_metadata"]


class TestMainFunction:
    """Tests for the main function integration."""

    @pytest.fixture
    def setup_integration_test(self):
        """Set up integration test environment."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create input directory structure
        input_dir = temp_dir / "input"
        input_dir.mkdir(parents=True)

        # Create label_config.json
        label_config = {
            "label_values": [0, 1],
            "output_label_type": "binary",
            "default_label": 0,
            "output_label_name": "prediction",
        }
        with open(input_dir / "label_config.json", "w") as f:
            json.dump(label_config, f)

        # Create ruleset.json
        ruleset = [
            {
                "rule_id": "rule_1",
                "name": "High score rule",
                "conditions": {"field": "score", "operator": ">", "value": 0.7},
                "output_label": 1,
                "priority": 1,
                "enabled": True,
            },
            {
                "rule_id": "rule_2",
                "name": "Low score rule",
                "conditions": {"field": "score", "operator": "<=", "value": 0.3},
                "output_label": 0,
                "priority": 2,
                "enabled": True,
            },
        ]
        with open(input_dir / "ruleset.json", "w") as f:
            json.dump(ruleset, f)

        # Set up paths
        input_paths = {"ruleset_configs": str(input_dir)}
        output_paths = {
            "validated_ruleset": str(temp_dir / "output" / "validated_ruleset.json"),
            "validation_report": str(temp_dir / "report" / "validation_report.json"),
        }
        environ_vars = {
            "ENABLE_LABEL_VALIDATION": "true",
            "ENABLE_LOGIC_VALIDATION": "true",
            "ENABLE_RULE_OPTIMIZATION": "true",
        }

        yield temp_dir, input_paths, output_paths, environ_vars
        shutil.rmtree(temp_dir)

    def test_main_successful_validation(self, setup_integration_test):
        """Test successful ruleset validation via main function."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        result = main(
            input_paths, output_paths, environ_vars, job_args=None, logger=print
        )

        assert "validated_ruleset" in result
        assert "validation_report" in result
        assert result["validation_report"]["validation_status"] == "passed"

        # Check output files created
        assert Path(output_paths["validated_ruleset"]).exists()
        assert Path(output_paths["validation_report"]).exists()

    def test_main_validation_failure(self, setup_integration_test):
        """Test validation failure via main function."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Modify ruleset to have invalid label
        input_dir = Path(input_paths["ruleset_configs"])
        with open(input_dir / "ruleset.json", "r") as f:
            ruleset = json.load(f)
        ruleset[0]["output_label"] = 5  # Invalid label
        with open(input_dir / "ruleset.json", "w") as f:
            json.dump(ruleset, f)

        with pytest.raises(RuntimeError, match="Ruleset validation failed"):
            main(input_paths, output_paths, environ_vars, job_args=None, logger=print)

    def test_main_with_optimization_disabled(self, setup_integration_test):
        """Test main function with optimization disabled."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test
        environ_vars["ENABLE_RULE_OPTIMIZATION"] = "false"

        result = main(
            input_paths, output_paths, environ_vars, job_args=None, logger=print
        )

        assert result["validation_report"]["optimization_applied"] is False

    def test_main_multiclass_validation(self, setup_integration_test):
        """Test main function with multiclass ruleset."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Create multiclass configuration
        input_dir = Path(input_paths["ruleset_configs"])
        label_config = {
            "label_values": [0, 1, 2],
            "output_label_type": "multiclass",
            "default_label": 0,
            "output_label_name": "category",
        }
        with open(input_dir / "label_config.json", "w") as f:
            json.dump(label_config, f)

        result = main(
            input_paths, output_paths, environ_vars, job_args=None, logger=print
        )

        assert (
            result["validated_ruleset"]["label_config"]["output_label_type"]
            == "multiclass"
        )

    def test_main_multilabel_validation(self, setup_integration_test):
        """Test main function with multilabel ruleset."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Create multilabel configuration
        input_dir = Path(input_paths["ruleset_configs"])
        label_config = {
            "label_values": {"col1": [0, 1], "col2": [0, 1]},
            "output_label_type": "multilabel",
            "default_label": {"col1": 0, "col2": 0},
            "output_label_name": ["col1", "col2"],
            "label_mapping": {
                "col1": {"0": "negative", "1": "positive"},
                "col2": {"0": "negative", "1": "positive"},
            },
        }
        with open(input_dir / "label_config.json", "w") as f:
            json.dump(label_config, f)

        ruleset = [
            {
                "rule_id": "rule_1",
                "name": "Multilabel rule",
                "conditions": {"field": "feature", "operator": ">", "value": 5},
                "output_label": {"col1": 1, "col2": 0},
                "priority": 1,
                "enabled": True,
            }
        ]
        with open(input_dir / "ruleset.json", "w") as f:
            json.dump(ruleset, f)

        result = main(
            input_paths, output_paths, environ_vars, job_args=None, logger=print
        )

        assert (
            result["validated_ruleset"]["label_config"]["output_label_type"]
            == "multilabel"
        )

    def test_main_missing_label_config(self, setup_integration_test):
        """Test main function with missing label_config.json."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Remove label_config.json
        input_dir = Path(input_paths["ruleset_configs"])
        (input_dir / "label_config.json").unlink()

        with pytest.raises(FileNotFoundError, match="label_config.json"):
            main(input_paths, output_paths, environ_vars, job_args=None, logger=print)

    def test_main_missing_ruleset(self, setup_integration_test):
        """Test main function with missing ruleset.json."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        # Remove ruleset.json
        input_dir = Path(input_paths["ruleset_configs"])
        (input_dir / "ruleset.json").unlink()

        with pytest.raises(FileNotFoundError, match="ruleset.json"):
            main(input_paths, output_paths, environ_vars, job_args=None, logger=print)

    def test_main_metadata_generation(self, setup_integration_test):
        """Test that main function generates complete metadata."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        result = main(
            input_paths, output_paths, environ_vars, job_args=None, logger=print
        )

        validated_ruleset = result["validated_ruleset"]
        assert "metadata" in validated_ruleset
        assert "total_rules" in validated_ruleset["metadata"]
        assert "enabled_rules" in validated_ruleset["metadata"]
        assert "disabled_rules" in validated_ruleset["metadata"]
        assert "field_usage" in validated_ruleset["metadata"]
        assert "validation_summary" in validated_ruleset["metadata"]

    def test_main_field_inference(self, setup_integration_test):
        """Test that main function infers field configuration correctly."""
        temp_dir, input_paths, output_paths, environ_vars = setup_integration_test

        result = main(
            input_paths, output_paths, environ_vars, job_args=None, logger=print
        )

        validated_ruleset = result["validated_ruleset"]
        assert "field_config" in validated_ruleset
        assert "required_fields" in validated_ruleset["field_config"]
        assert "field_types" in validated_ruleset["field_config"]
        assert "score" in validated_ruleset["field_config"]["required_fields"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_validation_result_all_error_types(self):
        """Test ValidationResult with all error types populated."""
        result = ValidationResult(valid=False)
        result.missing_fields = ["field1"]
        result.undeclared_fields = ["field2"]
        result.type_errors = ["error1"]
        result.invalid_labels = [("rule1", "label", "reason")]
        result.uncovered_classes = [3, 4]
        result.conflicting_rules = [("rule1", "rule2", 1)]
        result.tautologies = ["rule3"]
        result.contradictions = ["rule4"]
        result.unreachable_rules = ["rule5"]
        result.type_mismatches = [("rule6", "mismatch")]
        result.warnings = ["warning1", "warning2"]

        result_dict = result.__dict__()

        assert not result_dict["valid"]
        assert len(result_dict["warnings"]) == 2
        assert len(result_dict["invalid_labels"]) == 1

    def test_empty_ruleset_validation(self):
        """Test validation with empty ruleset."""
        validator = RulesetLabelValidator()
        ruleset = {
            "label_config": {
                "label_values": [0, 1],
                "output_label_type": "binary",
                "default_label": 0,
                "output_label_name": "prediction",
            },
            "ruleset": [],
        }

        result = validator.validate_labels(ruleset)

        assert result.valid is True

    def test_single_rule_validation(self):
        """Test validation with single rule."""
        validator = RulesetLabelValidator()
        ruleset = {
            "label_config": {
                "label_values": [0, 1],
                "output_label_type": "binary",
                "default_label": 0,
                "output_label_name": "prediction",
            },
            "ruleset": [
                {
                    "rule_id": "rule_1",
                    "name": "Only rule",
                    "conditions": {"field": "score", "operator": ">", "value": 0.5},
                    "output_label": 1,
                    "priority": 1,
                }
            ],
        }

        result = validator.validate_labels(ruleset)

        assert result.valid is True

    def test_very_complex_nested_conditions(self):
        """Test complexity calculation with very deep nesting."""
        condition = {
            "all_of": [
                {
                    "any_of": [
                        {
                            "none_of": [
                                {"field": "a", "operator": ">", "value": 1},
                                {"field": "b", "operator": "<", "value": 2},
                            ]
                        },
                        {"field": "c", "operator": "equals", "value": 3},
                    ]
                },
                {"field": "d", "operator": ">=", "value": 4},
            ]
        }

        complexity = calculate_complexity(condition)

        assert complexity > 5  # Should handle deep nesting

    def test_rule_with_none_output_label(self):
        """Test validation with None output_label."""
        validator = RulesetLabelValidator()
        ruleset = {
            "label_config": {
                "label_values": [0, 1],
                "output_label_type": "binary",
                "default_label": 0,
                "output_label_name": "prediction",
            },
            "ruleset": [
                {
                    "rule_id": "rule_1",
                    "name": "Rule with None",
                    "conditions": {"field": "score", "operator": ">", "value": 0.5},
                    "output_label": None,
                    "priority": 1,
                }
            ],
        }

        # Should handle None gracefully
        result = validator.validate_labels(ruleset)

        assert result.valid is True
