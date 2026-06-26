"""
Tests for CategoricalValidationProcessor custom-rule error handling.

Regression for the Theme-1 silent-failure fix: the broad ``except`` around custom rule
application used to wrap the intentional strict-mode ``raise ValueError`` too, so a
strict-mode rejection was silently downgraded to a logged error and bad rows passed
through. A rule that itself raises was also fully swallowed.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

# Load the processor module directly from its file. The cursus.processing.categorical
# package __init__ eagerly imports a torch-dependent streaming processor, so importing
# via the package would fail in a torch-free environment for reasons unrelated to this
# module. Loading by path keeps the test focused on CategoricalValidationProcessor.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "cursus"
    / "processing"
    / "categorical"
    / "categorical_validation_processor.py"
)
_PROCESSORS_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "cursus"
    / "processing"
    / "processors.py"
)


def _load_processor_class():
    # The module does `from ..processors import Processor`; ensure that relative import
    # resolves by importing it via its real dotted name only after the parent packages
    # exist. We import cursus.processing.processors directly (torch-free) and register it,
    # then exec the validation module against it.
    import sys

    if "cursus.processing.processors" not in sys.modules:
        spec_p = importlib.util.spec_from_file_location(
            "cursus.processing.processors", _PROCESSORS_PATH
        )
        mod_p = importlib.util.module_from_spec(spec_p)
        sys.modules["cursus.processing.processors"] = mod_p
        spec_p.loader.exec_module(mod_p)

    spec = importlib.util.spec_from_file_location(
        "cursus.processing.categorical.categorical_validation_processor", _MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.CategoricalValidationProcessor


CategoricalValidationProcessor = _load_processor_class()


def _df():
    return pd.DataFrame({"x": ["a", "b", "c"]})


class TestCustomRuleErrorHandling:
    def test_strict_mode_rejects_violations(self):
        """An actual rule violation must still raise in strict mode (not be swallowed)."""
        proc = CategoricalValidationProcessor(
            validation_rules={"x": lambda v: v == "a"},  # b, c violate
            validation_strategy="strict",
        ).fit(_df())
        with pytest.raises(ValueError, match="Custom validation rule failed"):
            proc.process(_df())

    def test_strict_mode_reraises_broken_rule(self):
        """A rule that itself raises must surface (not be silently swallowed) in strict mode."""

        def broken_rule(v):
            raise TypeError("boom")

        proc = CategoricalValidationProcessor(
            validation_rules={"x": broken_rule},
            validation_strategy="strict",
        ).fit(_df())
        with pytest.raises(ValueError, match="raised an error in strict mode"):
            proc.process(_df())

    def test_warn_mode_skips_broken_rule_without_crashing(self):
        """In warn mode a broken rule is logged and skipped; processing still returns data."""

        def broken_rule(v):
            raise TypeError("boom")

        proc = CategoricalValidationProcessor(
            validation_rules={"x": broken_rule},
            validation_strategy="warn",
        ).fit(_df())
        out = proc.process(_df())
        assert len(out) == 3  # nothing dropped, no raise

    def test_warn_mode_does_not_raise_on_violations(self):
        proc = CategoricalValidationProcessor(
            validation_rules={"x": lambda v: v == "a"},
            validation_strategy="warn",
        ).fit(_df())
        out = proc.process(_df())
        assert len(out) == 3  # warn keeps all rows

    def test_filter_mode_drops_violations(self):
        proc = CategoricalValidationProcessor(
            validation_rules={"x": lambda v: v == "a"},
            validation_strategy="filter",
        ).fit(_df())
        out = proc.process(_df())
        assert list(out["x"]) == ["a"]  # b, c filtered out
