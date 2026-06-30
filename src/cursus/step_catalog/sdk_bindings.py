"""Lazy SAIS-SDK step-class bindings for the registry-walk builder materializer (FZ 31e1d3g3 Phase A2).

The 4 SDK-delegation steps (CradleDataLoading / RedshiftDataLoading / Registration / DataUploading)
route through ``SDKDelegationHandler``, which needs an ``sdk_step_class`` knob — a live SAIS
``*Step`` CLASS OBJECT. That reference is genuine code, not serializable to ``.step.yaml`` (it is the
documented exception, see ``data_uploading.step.yaml`` and ``RegistrySection.requires``), so the
generic ``_synthesize_builder`` skips these steps and defers to this module.

Each binding is a LAZY thunk doing a LOCAL import — never module-level — so importing this module is
free and offline-safe. The SAIS import (and thus any failure when the SDK is absent) happens only
when a binding is actually called, which is only when an SDK step's builder is materialized in the
SAIS environment. This is exactly why the 4 steps stay offline-undiscoverable and match the closure
gate's ``_SDK_DELEGATION_STEPS`` carve-out.

The carve-out membership is authored data: a step belongs here iff its ``.step.yaml`` declares
``registry.requires: secure_ai_sandbox_workflow_python_sdk``. A conformance gate asserts the two
sets are equal (no drift between this binding table and the YAML).
"""

from typing import Callable, Dict, Type


def _cradle_data_loading_step() -> Type:
    from secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step import (
        CradleDataLoadingStep,
    )

    return CradleDataLoadingStep


def _redshift_data_loading_step() -> Type:
    from secure_ai_sandbox_workflow_python_sdk.redshift_data_loading.redshift_data_loading_step import (
        RedshiftDataLoadingStep,
    )

    return RedshiftDataLoadingStep


def _mims_model_registration_step() -> Type:
    from secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step import (
        MimsModelRegistrationProcessingStep,
    )

    return MimsModelRegistrationProcessingStep


def _data_uploading_step() -> Type:
    from secure_ai_sandbox_workflow_python_sdk.data_uploading.data_uploading_step import (
        DataUploadingStep,
    )

    return DataUploadingStep


#: STEP_NAME -> lazy thunk returning the SAIS *Step class to inject as the ``sdk_step_class`` knob.
#: Mirrors the ``HANDLER_KNOBS = {"sdk_step_class": <SAISClass>}`` the hand-written SDK shells carry.
SDK_STEP_CLASS_THUNKS: Dict[str, Callable[[], Type]] = {
    "CradleDataLoading": _cradle_data_loading_step,
    "RedshiftDataLoading": _redshift_data_loading_step,
    "Registration": _mims_model_registration_step,
    "DataUploading": _data_uploading_step,
}


def is_sdk_delegation_step(step_name: str) -> bool:
    """True iff ``step_name`` is materialized via this SDK-binding path (not the generic synthesizer)."""
    return step_name in SDK_STEP_CLASS_THUNKS


def resolve_sdk_step_class(step_name: str) -> Type:
    """Return the SAIS ``*Step`` class for an SDK-delegation step (lazy local import).

    Raises ``KeyError`` if the step is not an SDK-delegation step, and the underlying ``ImportError``
    if the SAIS SDK is absent — the caller (the materializer) only invokes this in the SAIS env.
    """
    return SDK_STEP_CLASS_THUNKS[step_name]()
