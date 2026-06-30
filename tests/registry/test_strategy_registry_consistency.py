"""
Phase S1 consistency gate: the strategy_registry is the single source of routing truth, and it
covers every sagemaker_step_type the step registry emits.

These tests are what guarantee the introspection tool (cursus strategies / strategies MCP) can
never drift from the runtime router: both read STRATEGY_REGISTRY, and this test asserts that
registry is complete + coherent + the only routing table.
"""

import pytest

from cursus.registry.strategy_registry import (
    NoBuilderError,
    list_strategies,
    resolve_strategy,
    axes,
)
from cursus.core.base.builder_templates import resolve_handler

_ROUTABLE_VERB_TYPES = {
    "Training",
    "CreateModel",
    "Transform",
    "CradleDataLoading",
    "RedshiftDataLoading",
    "MimsModelRegistrationProcessing",
}


def test_covers_every_sagemaker_step_type_in_the_registry():
    """Every sagemaker_step_type the step registry emits has a strategy row (routable or not)."""
    from cursus.registry.step_names import get_all_sagemaker_step_types

    sm_types = set(get_all_sagemaker_step_types())
    for sm in sm_types:
        if sm == "Processing":
            # Processing routes via the step_assembly axis, not sagemaker_step_type.
            assert resolve_strategy("step_assembly", "code") is not None
            continue
        if sm in _ROUTABLE_VERB_TYPES:
            info = resolve_strategy("sagemaker_step_type", sm)  # must not raise
            assert info.routable and info.handler is not None
        else:
            # Base / Lambda / any other non-builder type -> registered but NOT routable.
            with pytest.raises(NoBuilderError):
                resolve_strategy("sagemaker_step_type", sm)


def test_round_trip_no_shadowing():
    """Every routable row resolves back to itself (no duplicate keys / shadowing)."""
    for info in list_strategies():
        if info.routable:
            assert resolve_strategy(info.axis, info.name) is info


def test_processing_default_is_code():
    """resolve_handler('Processing', None) == resolve_handler('Processing','code')."""
    default = resolve_handler("Processing")
    code = resolve_handler("Processing", "code")
    assert type(default) is type(code)
    assert default.knobs.get("use_step_args") is False
    assert code.knobs.get("use_step_args") is False
    assert resolve_handler("Processing", "step_args").knobs.get("use_step_args") is True


def test_step_assembly_axis_has_three_names():
    """code / step_args / delegation all resolve on the step_assembly axis."""
    for name in ("code", "step_args", "delegation"):
        assert resolve_strategy("step_assembly", name) is not None


def test_knob_coherence_presets_are_declared():
    """Every preset_knobs key is a declared knob on the same strategy."""
    for info in list_strategies():
        declared = {k.name for k in info.knobs}
        for preset_key in info.preset_knobs:
            assert preset_key in declared, (
                f"{info.axis}:{info.name} presets undeclared knob {preset_key!r}"
            )


def test_registry_is_single_source_no_legacy_dicts():
    """builder_templates no longer carries its own routing dicts (single-source rule)."""
    import cursus.core.base.builder_templates as bt
    import inspect

    src = inspect.getsource(bt)
    for legacy in ("_TYPE_TO_HANDLER", "_PROCESSING_ASSEMBLY", "_ASSEMBLY_KNOBS", "_NO_BUILDER_TYPES"):
        assert legacy not in src, f"legacy routing dict {legacy} still present in builder_templates"


def test_axes_are_the_two_routing_axes():
    assert set(axes()) == {"sagemaker_step_type", "step_assembly"}
