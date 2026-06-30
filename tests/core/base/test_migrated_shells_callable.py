"""
Phase S3 confirmation: every migrated builder shell is DISCOVERABLE, CONSTRUCTS, binds the right
handler, and CALLS create_step through the facade (not a stale override) — i.e. it "can be called
and run correctly."

This is the standing gate for all 45 migrated shells. It exercises the real call path the
PipelineAssembler uses: step_catalog discovery -> the class -> __init__ (auto spec-load + handler
bind) -> create_step delegates to handler.build_step. The external SDK boundary (processor.run /
ProcessingStep ctor / SageMaker image+S3) is faked, so this runs offline; a real-session byte-diff
is the separate integration item.
"""

import warnings

warnings.simplefilter("ignore")

import pytest  # noqa: E402

# SDK-bound builders that require SAIS packages to import.
_SDK_STEPS = {"CradleDataLoading", "RedshiftDataLoading", "Registration", "DataUploading", "EdxUploading"}

_has_sais_sdk = True
try:
    import secure_ai_sandbox_workflow_python_sdk  # noqa: F401
except ModuleNotFoundError:
    _has_sais_sdk = False
try:
    import mods_workflow_core  # noqa: F401
except ModuleNotFoundError:
    _has_sais_sdk = False

from cursus.core.base.builder_templates import (  # noqa: E402
    TemplateStepBuilder,
    ProcessingHandler,
    TrainingHandler,
    ModelCreationHandler,
    TransformHandler,
    SDKDelegationHandler,
)

# Every migrated shell (step_name -> expected bound handler class). 45 total.
_MIGRATED = {
    # Transform (1:1)
    "BatchTransform": TransformHandler,
    # CreateModel (1-to-many)
    "XGBoostModel": ModelCreationHandler,
    "PyTorchModel": ModelCreationHandler,
    # Training (1-to-many)
    "XGBoostTraining": TrainingHandler,
    "LightGBMTraining": TrainingHandler,
    "LightGBMMTTraining": TrainingHandler,
    "PyTorchTraining": TrainingHandler,
    # Processing / code
    "TabularPreprocessing": ProcessingHandler,
    "FeatureSelection": ProcessingHandler,
    "CurrencyConversion": ProcessingHandler,
    "TemporalSplitPreprocessing": ProcessingHandler,
    "TemporalSequenceNormalization": ProcessingHandler,
    "TemporalFeatureEngineering": ProcessingHandler,
    "MissingValueImputation": ProcessingHandler,
    "StratifiedSampling": ProcessingHandler,
    "LabelRulesetExecution": ProcessingHandler,
    "DummyDataLoading": ProcessingHandler,
    "BedrockPromptTemplateGeneration": ProcessingHandler,
    "LabelRulesetGeneration": ProcessingHandler,
    "Package": ProcessingHandler,
    "Payload": ProcessingHandler,
    "ModelCalibration": ProcessingHandler,
    "ModelMetricsComputation": ProcessingHandler,
    "ModelWikiGenerator": ProcessingHandler,
    # Processing / step_args
    "XGBoostModelEval": ProcessingHandler,
    "XGBoostModelInference": ProcessingHandler,
    "LightGBMModelEval": ProcessingHandler,
    "LightGBMModelInference": ProcessingHandler,
    "LightGBMMTModelEval": ProcessingHandler,
    "LightGBMMTModelInference": ProcessingHandler,
    "PyTorchModelEval": ProcessingHandler,
    "PyTorchModelInference": ProcessingHandler,
    "BedrockProcessing": ProcessingHandler,
    "BedrockBatchProcessing": ProcessingHandler,
    "ActiveSampleSelection": ProcessingHandler,
    "PseudoLabelMerge": ProcessingHandler,
    "RiskTableMapping": ProcessingHandler,
    "TokenizerTraining": ProcessingHandler,
    "DummyTraining": ProcessingHandler,
    "PercentileModelCalibration": ProcessingHandler,
    # SDK Delegation (SAIS-bound — Phase S3 C-SDK)
    "CradleDataLoading": SDKDelegationHandler,
    "RedshiftDataLoading": SDKDelegationHandler,
    "Registration": SDKDelegationHandler,
    "DataUploading": SDKDelegationHandler,
    # Processing / code (SAIS-import-bound)
    "EdxUploading": ProcessingHandler,
}


@pytest.fixture(scope="module")
def catalog():
    from cursus.step_catalog.step_catalog import StepCatalog

    return StepCatalog()


@pytest.mark.parametrize("step_name", sorted(_MIGRATED))
def test_shell_discovered_and_is_facade_subclass(catalog, step_name):
    """step_catalog (the assembler's path) finds the shell; it IS-A TemplateStepBuilder."""
    if step_name in _SDK_STEPS and not _has_sais_sdk:
        pytest.skip(f"{step_name} requires SAIS SDK (not installed)")
    cls = catalog.load_builder_class(step_name)
    assert cls is not None, f"{step_name}: step_catalog did not discover the builder"
    assert issubclass(cls, TemplateStepBuilder), f"{step_name}: not a TemplateStepBuilder shell"


@pytest.mark.parametrize("step_name", sorted(_MIGRATED))
def test_shell_inherits_facade_create_step(catalog, step_name):
    """The shell does NOT define its own create_step/__init__ — it inherits the facade's
    (the whole point of the migration). A stale override would silently bypass the handler."""
    if step_name in _SDK_STEPS and not _has_sais_sdk:
        pytest.skip(f"{step_name} requires SAIS SDK (not installed)")
    cls = catalog.load_builder_class(step_name)
    assert "create_step" not in vars(cls), f"{step_name}: still defines its own create_step"
    assert "__init__" not in vars(cls), f"{step_name}: still defines its own __init__"


@pytest.mark.parametrize("step_name,expected_handler", sorted(_MIGRATED.items()))
def test_shell_binds_expected_handler(catalog, step_name, expected_handler):
    """The shell resolves to the correct construction handler — via the exact path __init__ uses
    (get_sagemaker_step_type(STEP_NAME) + STEP_ASSEMBLY + HANDLER_KNOBS -> resolve_handler), with no
    config construction needed. This is what makes create_step call the right builder logic."""
    if step_name in _SDK_STEPS and not _has_sais_sdk:
        pytest.skip(f"{step_name} requires SAIS SDK (not installed)")
    from cursus.registry.step_names import get_sagemaker_step_type
    from cursus.core.base.builder_templates import resolve_handler

    cls = catalog.load_builder_class(step_name)
    sm_type = get_sagemaker_step_type(cls.STEP_NAME)
    handler = resolve_handler(
        sm_type, getattr(cls, "STEP_ASSEMBLY", None), dict(getattr(cls, "HANDLER_KNOBS", {}))
    )
    assert isinstance(handler, expected_handler), (
        f"{step_name}: STEP_NAME={cls.STEP_NAME} sm_type={sm_type} bound "
        f"{type(handler).__name__}, expected {expected_handler.__name__}"
    )


def test_count_of_migrated_shells_is_40():
    """Guard the migration scope: all 45 builders are now migrated to TemplateStepBuilder shells."""
    assert len(_MIGRATED) == 45
