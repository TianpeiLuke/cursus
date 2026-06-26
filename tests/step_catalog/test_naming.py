"""
Tests for cursus.step_catalog.naming — the single source of truth for step-name <->
file-name conversion, and the guarantee that all discovery call sites agree with it.
"""

import pytest

from cursus.step_catalog.naming import (
    canonical_to_snake,
    parts_to_pascal,
    canonical_key,
    COMPOUND_ACRONYMS,
)


class TestCanonicalToSnake:
    @pytest.mark.parametrize(
        "pascal,snake",
        [
            ("XGBoostTraining", "xgboost_training"),
            ("PyTorchModel", "pytorch_model"),
            ("LightGBMTraining", "lightgbm_training"),
            ("LightGBMMTInference", "lightgbmmt_inference"),
            ("TabularPreprocessing", "tabular_preprocessing"),
            ("TensorFlowModel", "tensorflow_model"),
            ("SageMakerStep", "sagemaker_step"),
            ("CradleDataLoading", "cradle_data_loading"),
            ("Base", "base"),
        ],
    )
    def test_known_conversions(self, pascal, snake):
        assert canonical_to_snake(pascal) == snake

    def test_no_compound_acronym_is_split(self):
        # The whole point: compound acronyms stay intact (not light_gbm / x_g_boost).
        for a in COMPOUND_ACRONYMS:
            out = canonical_to_snake(a + "Step")
            assert "_" not in out.split("_step")[0] or out.startswith(a.lower()), out
            assert out.startswith(a.lower())


class TestPartsToPascal:
    @pytest.mark.parametrize(
        "parts,pascal",
        [
            (["xgboost", "training"], "XGBoostTraining"),
            (["pytorch", "model"], "PyTorchModel"),
            (["lightgbm", "training"], "LightGBMTraining"),
            (["lightgbmmt", "inference"], "LightGBMMTInference"),
            (["tabular", "preprocessing"], "TabularPreprocessing"),
        ],
    )
    def test_known(self, parts, pascal):
        assert parts_to_pascal(parts) == pascal


class TestCanonicalKey:
    def test_collapses_case_and_separators(self):
        assert (
            canonical_key("XGBoostTraining")
            == canonical_key("xgboost_training")
            == canonical_key("XGBoost_Training")
            == "xgboosttraining"
        )


class TestRoundTrip:
    @pytest.mark.parametrize(
        "name",
        ["XGBoostTraining", "PyTorchModel", "LightGBMTraining", "TabularPreprocessing"],
    )
    def test_snake_then_pascal_recovers_name(self, name):
        # canonical -> snake parts -> pascal should recover the canonical name.
        snake = canonical_to_snake(name)
        assert parts_to_pascal(snake.split("_")) == name


class TestCallSitesAgree:
    """All discovery modules must delegate to (and therefore agree with) naming.py."""

    def test_script_discovery_matches(self):
        from cursus.step_catalog.script_discovery import ScriptAutoDiscovery

        sd = ScriptAutoDiscovery.__new__(ScriptAutoDiscovery)  # no __init__ needed
        for n in ["XGBoostTraining", "PyTorchModel", "LightGBMTraining"]:
            assert sd._canonical_to_script_name(n) == canonical_to_snake(n)

    def test_contract_discovery_matches(self):
        from cursus.step_catalog.contract_discovery import ContractAutoDiscovery

        cd = ContractAutoDiscovery.__new__(ContractAutoDiscovery)
        for n in ["XGBoostTraining", "PyTorchTraining", "TensorFlowModel"]:
            assert cd._pascal_to_snake_case(n) == canonical_to_snake(n)

    def test_builder_discovery_matches(self):
        from cursus.step_catalog.builder_discovery import BuilderAutoDiscovery

        bd = BuilderAutoDiscovery.__new__(BuilderAutoDiscovery)
        assert (
            bd._convert_parts_to_pascal_case_with_special_cases(["xgboost", "training"])
            == "XGBoostTraining"
        )
