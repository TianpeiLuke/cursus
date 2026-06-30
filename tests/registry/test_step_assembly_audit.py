"""
Phase 0b step_assembly audit (enshrined).

The 35 concrete Processing-typed steps split across three construction verbs by how the step
is assembled:
  * code       — ProcessingStep(..., code=script_path)
  * step_args  — step_args = processor.run(...); ProcessingStep(step_args=step_args)
  * delegation — a non-ProcessingStep SDK step class (DataUploadingStep)

Originally this re-derived the verb from each builder's create_step SOURCE by AST inspection. After
FZ 31e1d3g3 Phase E deleted all 45 per-step builder files, the verb's source of truth is the
``.step.yaml`` ``patterns.step_assembly`` (FZ 31e1d3f1 — the blueprint the router reads); a ``None``
value means the Processing default ``code`` (resolve_handler). So the test now reads the verb from the
INTERFACE (no builder file), still asserting it matches the audited table — a wrong tag fails at build
time, not registration. The audit is keyed by step NAME (the registry canonical name), not a filename.
"""

import pytest

from cursus.steps.interfaces import load_interface

# Audited verb per concrete Processing-typed step (canonical step name -> expected verb).
# 18 code + 16 step_args + 1 delegation = 35. (Source-verified; corrects the workflow
# audit's mis-tag of ModelMetricsComputation/ModelWikiGenerator as step_args.)
_EXPECTED = {
    # --- code (16) ---
    "builder_tabular_preprocessing_step.py": "code",
    "builder_temporal_split_preprocessing_step.py": "code",
    "builder_temporal_sequence_normalization_step.py": "code",
    "builder_temporal_feature_engineering_step.py": "code",
    "builder_stratified_sampling_step.py": "code",
    "builder_missing_value_imputation_step.py": "code",
    "builder_feature_selection_step.py": "code",
    "builder_currency_conversion_step.py": "code",
    "builder_dummy_data_loading_step.py": "code",
    "builder_bedrock_prompt_template_generation_step.py": "code",
    "builder_label_ruleset_generation_step.py": "code",
    "builder_label_ruleset_execution_step.py": "code",
    "builder_model_calibration_step.py": "code",
    "builder_package_step.py": "code",
    "builder_payload_step.py": "code",
    "builder_edx_uploading_step.py": "code",  # code-assembled but sdk_dependent (NI-2)
    # ModelMetricsComputation / ModelWikiGenerator: workflow audit mis-tagged these as
    # step_args; source confirms ProcessingStep(..., code=script_path) -> they are `code`.
    "builder_model_metrics_computation_step.py": "code",
    "builder_model_wiki_generator_step.py": "code",
    # --- step_args (16) ---
    "builder_risk_table_mapping_step.py": "step_args",
    "builder_tokenizer_training_step.py": "step_args",
    "builder_dummy_training_step.py": "step_args",  # the trap: Processing+step_args, not code
    "builder_xgboost_model_eval_step.py": "step_args",
    "builder_xgboost_model_inference_step.py": "step_args",
    "builder_lightgbm_model_eval_step.py": "step_args",
    "builder_lightgbm_model_inference_step.py": "step_args",
    "builder_lightgbmmt_model_eval_step.py": "step_args",
    "builder_lightgbmmt_model_inference_step.py": "step_args",
    "builder_pytorch_model_eval_step.py": "step_args",
    "builder_pytorch_model_inference_step.py": "step_args",
    "builder_bedrock_processing_step.py": "step_args",
    "builder_bedrock_batch_processing_step.py": "step_args",
    "builder_active_sample_selection_step.py": "step_args",
    "builder_pseudo_label_merge_step.py": "step_args",
    "builder_percentile_model_calibration_step.py": "step_args",
    # --- delegation (1, sdk_dependent) ---
    "builder_data_uploading_step.py": "delegation",
}

def _filename_to_step_name(filename: str) -> str:
    """Resolve a legacy ``builder_<snake>_step.py`` audit key to its canonical registry step name.

    The builders are deleted (Phase E); the audit keys are kept for coverage identity. The filename
    is the snake form of the registry ``builder_step_name`` (``<Name>StepBuilder``), so reverse-map
    through the registry: snake(builder_step_name minus 'StepBuilder') == filename stem body.
    """
    from cursus.registry.step_names import get_step_names
    from cursus.steps.interfaces import _step_name_to_filename

    stem = filename[len("builder_") : -len("_step.py")]  # e.g. "xgboost_model_eval"
    for step in get_step_names():
        if _step_name_to_filename(step) == stem:
            return step
    raise AssertionError(f"no registry step maps to audit filename {filename!r}")


def _interface_verb(step_name: str) -> str:
    """The step's assembly verb from its ``.step.yaml`` ``patterns.step_assembly`` (the blueprint —
    FZ 31e1d3f1). ``None`` ⇒ the Processing default ``code`` (resolve_handler)."""
    return getattr(load_interface(step_name).patterns, "step_assembly", None) or "code"


@pytest.mark.parametrize("filename,expected", sorted(_EXPECTED.items()))
def test_step_assembly_verb(filename, expected):
    """Each Processing step's assembly verb — read from the .step.yaml patterns.step_assembly
    (FZ 31e1d3g3 Phase E: the builder files are deleted; the interface is the source of truth) —
    matches the audited table. A wrong tag fails at build time, not registration."""
    step_name = _filename_to_step_name(filename)
    verb = _interface_verb(step_name)
    assert verb == expected, f"{step_name} ({filename}): interface {verb!r}, audited {expected!r}"


def test_audit_counts():
    """18 code + 16 step_args + 1 delegation = 35 concrete Processing builders.

    (Corrected from the workflow audit's 16/18: ModelMetricsComputation and
    ModelWikiGenerator build ProcessingStep(..., code=script_path) — verified `code`, not
    step_args — so the split is 18 code / 16 step_args.)
    """
    from collections import Counter

    counts = Counter(_EXPECTED.values())
    assert counts == {"code": 18, "step_args": 16, "delegation": 1}
    assert sum(counts.values()) == 35


def test_dummy_training_is_step_args_not_code():
    """The verified trap: DummyTraining is Processing+step_args (NOT the `code` default)."""
    assert _EXPECTED["builder_dummy_training_step.py"] == "step_args"
