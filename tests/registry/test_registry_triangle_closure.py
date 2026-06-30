"""FZ 31e2 — the triangle-closure CI check (the silent-drop guard).

The step system has four "corners" that must stay in agreement:
  1. the registry STEP_NAMES table (now interface-derived),
  2. the per-step `.step.yaml` interfaces,
  3. the discoverable builders (StepCatalog.get_builder_map),
  4. the importable config classes (StepCatalog.discover_config_classes).

Discovery and config resolution FAIL SOFT today — a missing/typo'd builder returns None and the
catalog masks it as an empty index, indistinguishable from an expected job-type-variant miss
(step_catalog.py graceful degradation). This gate converts those silent drops into a LOUD CI failure,
asserting closure bidirectionally (no orphan in any corner). It is the mandatory prerequisite that
makes the classless Design-B refactor (registry-walk discovery + provider returns, FZ 31e1d3g1) safe:
under registry-walk a registered-but-unloadable step becomes a hard ValueError at build, so the
closure must be proven first.

The 4 SDKDelegation builders (Cradle / Redshift / DataUploading / Registration) import the SAIS SDK at
module level and cannot be discovered/loaded offline — they are classified as SDK-env-only: skipped
with reason here, and asserted present only when the SDK is importable. (EdxUploading is NOT in this
set — it became a pure shell with a lazy SDK reference and discovers offline.)
"""

import logging
import warnings

import pytest

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# Steps whose builder module imports the SAIS SDK at module level → not discoverable/loadable offline.
_SDK_DELEGATION_STEPS = {"CradleDataLoading", "RedshiftDataLoading", "Registration", "DataUploading"}

# Registry rows that are abstract / have no .step.yaml interface (declared in the loader's _EXTRAS).
_INTERFACE_LESS = {"Base", "Processing", "HyperparameterPrep"}

_has_sais_sdk = True
try:
    import secure_ai_sandbox_workflow_python_sdk  # noqa: F401
    import mods_workflow_core  # noqa: F401
except ModuleNotFoundError:
    _has_sais_sdk = False


@pytest.fixture(scope="module")
def corners():
    from cursus.registry.step_names import get_step_names
    from cursus.step_catalog.step_catalog import StepCatalog
    from cursus.steps.interfaces import (
        clear_interface_cache,
        list_available_interfaces,
        load_interface,
    )

    clear_interface_cache()
    cat = StepCatalog()
    registry = get_step_names()  # {name: {config_class, builder_step_name, sagemaker_step_type, ...}}
    interfaces = set()
    for stem in list_available_interfaces():
        try:
            interfaces.add(load_interface(stem).spec.step_type)
        except Exception:
            pass
    builders = set(cat.get_builder_map().keys())
    configs = set(cat.discover_config_classes().keys())
    return {
        "registry": registry,
        "registry_names": set(registry),
        "interfaces": interfaces,
        "builders": builders,
        "configs": configs,
    }


def test_registry_interface_bijection(corners):
    """Every concrete registry row has a .step.yaml interface, and vice-versa (no orphan either way)."""
    concrete = corners["registry_names"] - _INTERFACE_LESS
    reg_only = concrete - corners["interfaces"]
    iface_only = corners["interfaces"] - corners["registry_names"]
    assert not reg_only, f"registry rows with NO .step.yaml interface: {sorted(reg_only)}"
    assert not iface_only, f".step.yaml interfaces with NO registry row: {sorted(iface_only)}"


def test_every_registry_config_class_is_importable_and_resolves_back(corners):
    """Each registry row's config_class must be a discoverable config AND map back to that step via
    CONFIG_STEP_REGISTRY. Catches the registry-integrity bug FZ 31e2 was built for: a convention-
    derived config name (PyTorchModelConfig) that doesn't match the real class (PyTorchModelStepConfig)
    leaves the real config UNRESOLVABLE in get_builder_for_config."""
    from cursus.registry.step_names import get_config_step_registry

    csr = get_config_step_registry()  # {config_class_name: step_name}
    bad = []
    for name, row in corners["registry"].items():
        if name in _INTERFACE_LESS:
            continue  # abstract bases (BasePipelineConfig etc.) aren't in the discovered concrete map
        cfg = row.get("config_class")
        if cfg not in corners["configs"]:
            bad.append(f"{name}: config_class {cfg!r} is NOT an importable/discovered config class")
        elif csr.get(cfg) != name:
            bad.append(f"{name}: config_class {cfg!r} maps to {csr.get(cfg)!r} in CONFIG_STEP_REGISTRY")
    assert not bad, "config-class closure violations: " + "; ".join(bad)


def test_every_registry_step_has_a_discoverable_builder(corners):
    """Every concrete registry row has a discoverable builder — EXCEPT the SDK-delegation steps when
    the SAIS SDK is absent (classified, not silently tolerated)."""
    concrete = corners["registry_names"] - _INTERFACE_LESS
    missing = concrete - corners["builders"]
    if _has_sais_sdk:
        assert not missing, (
            f"registry rows with NO discoverable builder (SDK present, so this is a real gap): "
            f"{sorted(missing)}"
        )
    else:
        # offline: the ONLY acceptable missing set is exactly the SDK-delegation steps.
        unexpected = missing - _SDK_DELEGATION_STEPS
        assert not unexpected, (
            f"registry rows with NO discoverable builder beyond the known SDK-delegation steps: "
            f"{sorted(unexpected)} (missing={sorted(missing)})"
        )


def test_no_orphan_builder_without_registry_row(corners):
    """Every discovered builder corresponds to a registry row OR a declared LEGACY_ALIAS (no builder
    discovered for a genuinely unknown step — the reverse-orphan direction). get_builder_map()
    intentionally includes the back-compat alias keys (MIMSPackaging, PytorchModel, ...), so the
    tolerated set is derived from the LIVE LEGACY_ALIASES, not hardcoded — if an alias is added/removed
    the gate follows."""
    from cursus.step_catalog.mapping import StepCatalogMapper

    aliases = set(getattr(StepCatalogMapper, "LEGACY_ALIASES", {}))
    orphans = corners["builders"] - corners["registry_names"] - aliases
    assert not orphans, (
        f"discovered builders with NO registry row and NOT a known LEGACY_ALIAS: {sorted(orphans)}"
    )


def test_sdk_delegation_steps_are_registered_and_have_interfaces(corners):
    """The 4 SDK-delegation steps must still be in the registry + have interfaces even when their
    builders can't load offline — so the closure 'expected absence' set is itself verified, not a
    blanket excuse."""
    for s in _SDK_DELEGATION_STEPS:
        assert s in corners["registry_names"], f"{s} missing from registry"
        assert s in corners["interfaces"], f"{s} missing a .step.yaml interface"


@pytest.mark.skipif(not _has_sais_sdk, reason="SAIS SDK absent — SDK-delegation builders only load in the SAIS env")
def test_sdk_builders_discoverable_in_sais_env(corners):
    """In the SAIS environment, the SDK-delegation builders MUST also be discoverable — closing the
    one gap the offline run is allowed to skip."""
    missing = _SDK_DELEGATION_STEPS - corners["builders"]
    assert not missing, f"SDK env but these SDK-delegation builders are undiscoverable: {sorted(missing)}"


def test_every_routable_step_loads_interface_and_resolves_a_handler(corners):
    """FZ 31e1d3g3 Phase C2 — the CONSTRUCTION-AWARE closure assertion. For every concrete registry
    step (excluding the abstract interface-less rows), assert STATICALLY that:
      (a) load_interface(step) loads — and each declared job_type variant loads — covering the
          TemplateStepBuilder.__init__ spec-load path (builder_templates.py:986-991), and
      (b) resolve_handler(get_sagemaker_step_type(step), patterns.step_assembly) yields a routable
          handler — covering _auto_bind_handler (builder_templates.py:1016-1018) + the NoBuilderError
          path (strategy_registry.py:120).
    This is STATIC (resolve, don't instantiate) so it does not depend on a minimal config — it proves
    a deleted-but-registered step would still route, the property the materializer (Phase A) relies on.
    The 4 SDK-delegation steps are checked for interface-load + (in the SAIS env) handler-resolution;
    offline their handler import is deferred, so only the interface load is asserted."""
    from cursus.registry.step_names import get_sagemaker_step_type
    from cursus.steps.interfaces import load_interface
    from cursus.core.base.builder_templates import resolve_handler, NoBuilderError

    failures = []
    for name in sorted(corners["registry_names"] - _INTERFACE_LESS):
        # (a) base interface + every job_type variant must load.
        try:
            iface = load_interface(name)
        except Exception as e:
            failures.append(f"{name}: base interface failed to load: {e}")
            continue
        for jt in iface.variants:
            try:
                load_interface(name, job_type=jt)
            except Exception as e:
                failures.append(f"{name}: job_type variant {jt!r} failed to load: {e}")

        # (b) the step must route to a handler. SDK-delegation steps resolve to SDKDelegationHandler
        #     only when the SAIS SDK is present; offline we skip the handler assertion for them.
        if name in _SDK_DELEGATION_STEPS and not _has_sais_sdk:
            continue
        sm_type = get_sagemaker_step_type(name)
        step_assembly = getattr(getattr(iface, "patterns", None), "step_assembly", None)
        try:
            resolve_handler(sm_type, step_assembly)
        except NoBuilderError as e:
            failures.append(
                f"{name}: ({sm_type}/{step_assembly}) is not routable — no handler: {e}"
            )
        except Exception as e:
            # An SDK handler may fail to import offline even if requires wasn't declared — tolerate
            # only for the known SDK set; anything else is a real failure.
            if name not in _SDK_DELEGATION_STEPS:
                failures.append(f"{name}: handler resolution raised: {e}")

    assert not failures, "construction-aware closure violations:\n  " + "\n  ".join(failures)


def test_sdk_bindings_table_matches_yaml_requires():
    """FZ 31e1d3g3 Phase A2: the sdk_bindings thunk table (the materializer's SDK carve-out) must
    EQUAL the set of steps whose .step.yaml declares registry.requires=secure_ai_sandbox_workflow_python_sdk
    — so the carve-out is authored data, not a second hardcoded list that can drift."""
    from cursus.step_catalog.sdk_bindings import SDK_STEP_CLASS_THUNKS
    from cursus.steps.interfaces import load_interface
    from cursus.registry.step_names import get_step_names

    sdk_in_yaml = set()
    for step in get_step_names():
        try:
            iface = load_interface(step)
        except Exception:
            continue
        registry_section = getattr(iface, "registry", None)
        if getattr(registry_section, "requires", "none") == "secure_ai_sandbox_workflow_python_sdk":
            sdk_in_yaml.add(step)

    assert set(SDK_STEP_CLASS_THUNKS) == sdk_in_yaml, (
        f"sdk_bindings table {sorted(SDK_STEP_CLASS_THUNKS)} != "
        f"YAML requires=...sdk steps {sorted(sdk_in_yaml)}"
    )
    # And it must equal the closure gate's offline carve-out set, so all three stay in lockstep.
    assert set(SDK_STEP_CLASS_THUNKS) == _SDK_DELEGATION_STEPS
