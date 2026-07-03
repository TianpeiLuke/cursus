"""
Unit tests for ``cursus.processing.validation`` — the strict field-type guards run before numerical
imputation / categorical risk-table mapping. Pure pandas, previously 0% covered.
"""

import pandas as pd
import pytest

from cursus.processing.validation import (
    validate_categorical_fields,
    validate_numerical_fields,
)


class TestValidateCategoricalFields:
    def test_object_and_category_pass(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": pd.Series(["p", "q"], dtype="category")})
        # No exception == pass.
        validate_categorical_fields(df, ["a", "b"])

    def test_missing_field_raises_valueerror(self):
        df = pd.DataFrame({"a": ["x"]})
        with pytest.raises(ValueError):
            validate_categorical_fields(df, ["missing"], dataset_name="train")

    def test_numeric_field_flagged_as_typeerror(self):
        df = pd.DataFrame({"a": [1, 2, 3]})  # int, not categorical
        with pytest.raises(TypeError) as exc:
            validate_categorical_fields(df, ["a"], dataset_name="val")
        # The message names the offending field + dataset for operator triage.
        assert "a" in str(exc.value)
        assert "val" in str(exc.value)

    def test_empty_field_list_is_noop(self):
        validate_categorical_fields(pd.DataFrame({"a": [1]}), [])


class TestValidateNumericalFields:
    def test_int_and_float_pass(self):
        df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
        validate_numerical_fields(df, ["a", "b"])

    def test_missing_field_raises_valueerror(self):
        with pytest.raises(ValueError):
            validate_numerical_fields(pd.DataFrame({"a": [1]}), ["nope"])

    def test_string_field_flagged_as_typeerror(self):
        df = pd.DataFrame({"a": ["not", "numeric"]})
        with pytest.raises(TypeError) as exc:
            validate_numerical_fields(df, ["a"], dataset_name="test")
        assert "a" in str(exc.value)
        assert "test" in str(exc.value)

    def test_multiple_bad_fields_all_reported(self):
        df = pd.DataFrame({"a": ["x"], "b": ["y"], "c": [1]})
        with pytest.raises(TypeError) as exc:
            validate_numerical_fields(df, ["a", "b", "c"])
        msg = str(exc.value)
        assert "a" in msg and "b" in msg  # both non-numeric fields named
