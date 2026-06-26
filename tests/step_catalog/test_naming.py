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
    JOB_TYPE_SUFFIXES,
    JOB_TYPE_KEYWORDS,
    BASE_CONFIGS,
    is_job_type_variant,
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


class TestJobTypeVocabulary:
    """The single source of truth for job-type constants (was duplicated + drifted)."""

    def test_is_job_type_variant_matches_real_variants(self):
        assert is_job_type_variant("xgboost_training")
        assert is_job_type_variant("foo_inference")
        assert is_job_type_variant("foo_evaluation")

    def test_model_is_not_a_variant_suffix(self):
        # A step kind, not a job-type variant — must NOT be filtered from listings.
        assert not is_job_type_variant("xgboost_model")
        assert not is_job_type_variant("PyTorchModel")

    def test_base_name_with_embedded_job_word_is_not_a_variant(self):
        # Only a trailing _<suffix> counts; a bare word in the middle does not.
        assert not is_job_type_variant("training_helper")

    def test_keywords_superset_for_classification(self):
        # Classification keywords include "model"; suffix-detection deliberately does not.
        assert "model" in JOB_TYPE_KEYWORDS
        assert "model" not in JOB_TYPE_SUFFIXES

    def test_base_configs_membership(self):
        assert "Base" in BASE_CONFIGS and "Processing" in BASE_CONFIGS

    def test_step_catalog_delegates_to_shared_helper(self):
        """StepCatalog._is_job_type_variant must agree with the shared helper."""
        from pathlib import Path
        import cursus
        from cursus.step_catalog import StepCatalog

        root = Path(cursus.__file__).resolve().parent
        cat = StepCatalog(workspace_dirs=[root.parent])
        for probe in ["xgboost_training", "xgboost_model", "foo_inference", "foo_batch"]:
            assert cat._is_job_type_variant(probe) == is_job_type_variant(probe)


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

    def test_step_catalog_canonical_resolvers_handle_compound_acronyms(self):
        """The two StepCatalog snake->canonical resolvers must route through naming.py so
        compound-acronym families resolve to their real registry entry (regression for the
        naive ''.join(capitalize) that orphaned pytorch/xgboost/lightgbm file components)."""
        from pathlib import Path
        import cursus
        from cursus.step_catalog import StepCatalog

        root = Path(cursus.__file__).resolve().parent
        cat = StepCatalog(workspace_dirs=[root.parent])
        registry = {}
        from cursus.registry.step_names import get_step_names

        registry = get_step_names()

        cases = {
            "pytorch_training": "PyTorchTraining",
            "xgboost_training": "XGBoostTraining",
            "lightgbm_training": "LightGBMTraining",
            "xgboost_model_eval": "XGBoostModelEval",
        }
        for snake, expected in cases.items():
            # only assert for entries actually in the registry
            if expected not in registry:
                continue
            assert cat._resolve_to_canonical_name_for_indexing(snake) == expected
            assert cat._resolve_to_canonical_name(snake, registry) == expected
            # and both agree with the shared helper
            assert parts_to_pascal(snake.split("_")) == expected
