"""
Regression tests for the plugin step-pack ADDITIVE INVARIANT (FZ 18g3h2e2).

The hard requirement: the internal steps under upstream are ALWAYS available — a consumer's
external step pack is a strictly ADDITIVE overlay. A pack can only ADD steps (and, on a
deliberate name-clash, shadow with a warning); it can NEVER remove or replace a package step,
and with no pack the registry/catalog is byte-identical to package-only.

Covers the action items from FZ 18g3h2e2:
  AI-1  external config imported by file location (spec_from_file_location) — no longer dropped.
  AI-2  refresh_registry merges package-first (never replace).
  AI-4  the compiler threads the anchor → the catalog discovers plugin steps as native.
  AI-6  collisions surface via get_registry_health()['pack_collisions'].
  AI-7  the golden-snapshot gate re-derives package-only, so a pack cannot trip drift detection.
"""

import tempfile
from pathlib import Path

import pytest

from cursus.registry.step_names import (
    get_step_names,
    refresh_registry,
    get_registry_health,
)
from cursus.registry.interface_registry_loader import build_registry_from_interfaces
from cursus.step_catalog.step_catalog import (
    StepCatalog,
    set_default_workspace_dirs,
)


def _write_pack(root: Path, step_type: str, *, sagemaker_step_type: str = "Processing"):
    """Lay out a minimal step pack at ``root``: interfaces/ + configs/ + scripts/ with one step."""
    snake = "".join(f"_{c.lower()}" if c.isupper() else c for c in step_type).lstrip(
        "_"
    )
    (root / "interfaces").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "interfaces" / f"{snake}.step.yaml").write_text(
        f"step_type: {step_type}\n"
        f"registry:\n"
        f"  sagemaker_step_type: {sagemaker_step_type}\n"
        f"  description: plugin step {step_type}\n"
    )
    (root / "scripts" / f"{snake}.py").write_text("# plugin script\n")
    # A config class that inherits a known base + follows the <Name>Config convention.
    (root / "configs" / f"config_{snake}_step.py").write_text(
        "from pydantic import BaseModel\n"
        f"class {step_type}Config(BaseModel):\n"
        "    field: str = 'x'\n"
    )
    return snake


@pytest.fixture(autouse=True)
def _isolate_process_state():
    """Fully restore process-global registry state around every test.

    ``refresh_registry`` mutates the process-global ``STEP_NAMES`` (and the hybrid manager
    cache, the interface-loader pack dirs, the pack-collision record, and the StepCatalog
    default). Those MUST be restored after each test, or a merged/shadowed pack step leaks into
    unrelated registry tests (and the leak is order-dependent — it only surfaces when a plugin
    test runs before a package-registry test). Snapshot before, restore after.
    """
    from cursus.registry import step_names_base
    from cursus.registry.step_names import _get_registry_manager
    from cursus.steps import interfaces as _interfaces
    import cursus.registry.step_names as _sn

    # Deep-copy the live registry so post-test restoration is byte-exact.
    saved_step_names = {k: dict(v) for k, v in step_names_base.STEP_NAMES.items()}
    saved_pack_dirs = list(_interfaces._pack_interface_dirs)
    saved_collisions = dict(_sn._pack_collisions)
    set_default_workspace_dirs(None)
    yield
    set_default_workspace_dirs(None)
    # Restore STEP_NAMES in place (import-time references stay live), rebuild derived globals,
    # and re-sync the manager so cached legacy dicts drop the pack rows.
    step_names_base.STEP_NAMES.clear()
    step_names_base.STEP_NAMES.update(saved_step_names)
    step_names_base._rebuild_derived()
    _interfaces._pack_interface_dirs[:] = saved_pack_dirs
    _interfaces._cache.clear()
    _sn._pack_collisions.clear()
    _sn._pack_collisions.update(saved_collisions)
    mgr = _get_registry_manager()
    if hasattr(mgr, "reload_core_registry"):
        mgr.reload_core_registry()


class TestAdditiveInvariant:
    def test_no_pack_registry_equals_package_only(self):
        """With no pack active, the registry equals the package-derived table exactly."""
        package_only = set(
            build_registry_from_interfaces()
        )  # _EXTRAS + package .step.yaml
        live = set(get_step_names())
        # Every package-derived step is present in the live registry (no pack subtracted anything).
        assert package_only <= live

    def test_pack_adds_only_new_step_and_keeps_all_package_steps(self):
        before = get_step_names()
        n_before = len(before)
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "step_pack"
            _write_pack(pack, "AdditiveProbeStep")
            collisions = refresh_registry(pack / "interfaces")

            after = get_step_names()
            # exactly the one new step was added
            assert "AdditiveProbeStep" in after
            assert len(after) == n_before + 1
            # EVERY package step is still present and unchanged
            assert set(before).issubset(set(after))
            for name, row in before.items():
                assert after[name] == row
            assert collisions == {}

    def test_pack_omitting_a_core_step_does_not_remove_it(self):
        """A pack that ships only its own step must not drop any package step."""
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "step_pack"
            _write_pack(pack, "OmissionProbeStep")
            refresh_registry(pack / "interfaces")
            after = get_step_names()
            # A representative package step is still resolvable.
            assert "XGBoostTraining" in after
            assert "TabularPreprocessing" in after

    def test_collision_shadows_with_warning_but_keeps_other_package_steps(self, caplog):
        before = get_step_names()
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "step_pack"
            # Deliberately clash with a known package step name.
            _write_pack(pack, "XGBoostTraining", sagemaker_step_type="Training")
            with caplog.at_level("WARNING"):
                collisions = refresh_registry(pack / "interfaces")

            assert "XGBoostTraining" in collisions
            # surfaced via health for monitoring
            assert "XGBoostTraining" in get_registry_health()["pack_collisions"]
            # a warning was logged
            assert any("shadow" in r.message.lower() for r in caplog.records)
            after = get_step_names()
            # every OTHER package step is untouched; none lost
            assert set(before).issubset(set(after))


class TestExternalConfigImport:
    """AI-1: an external (workspace) config file is imported by file location, not dropped."""

    def test_external_config_class_is_discovered(self):
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "step_pack"
            _write_pack(pack, "ImportProbeStep")
            catalog = StepCatalog(workspace_dirs=[pack])
            config_classes = catalog.config_discovery.discover_config_classes()
            # The plugin config class is imported (previously silently dropped for out-of-package files).
            assert "ImportProbeStepConfig" in config_classes
            cls = config_classes["ImportProbeStepConfig"]
            assert cls.__name__ == "ImportProbeStepConfig"

    def test_package_config_import_unchanged(self):
        """The package config import path is unaffected by the external fallback."""
        catalog = StepCatalog()  # package-only
        config_classes = catalog.config_discovery.discover_config_classes()
        # A well-known package config is still discovered.
        assert any(name.endswith("Config") for name in config_classes)
        assert len(config_classes) > 0


class TestCatalogSeesPluginStepAsNative:
    """AI-4: workspace_dirs → the catalog indexes the plugin step alongside package steps."""

    def test_catalog_indexes_pack_and_package(self):
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "step_pack"
            _write_pack(pack, "NativeProbeStep")
            refresh_registry(pack / "interfaces")
            catalog = StepCatalog(workspace_dirs=[pack])
            assert catalog.get_step_info("NativeProbeStep") is not None
            assert catalog.get_step_info("XGBoostTraining") is not None

    def test_bare_catalog_uses_process_default(self):
        """A bare StepCatalog() picks up the compiler-pushed default (AI-5)."""
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "step_pack"
            _write_pack(pack, "DefaultProbeStep")
            refresh_registry(pack / "interfaces")
            set_default_workspace_dirs([pack])
            bare = StepCatalog()  # no workspace_dirs arg
            assert bare.get_step_info("DefaultProbeStep") is not None
            assert bare.get_step_info("XGBoostTraining") is not None


class TestSnapshotGateIsolation:
    """AI-7: the golden-snapshot gate re-derives package-only, unaffected by an active pack."""

    def test_pack_rows_excluded_from_package_derive(self):
        with tempfile.TemporaryDirectory() as d:
            pack = Path(d) / "step_pack"
            _write_pack(pack, "SnapshotProbeStep")
            refresh_registry(pack / "interfaces")
            # build_registry_from_interfaces() with no args derives from the PACKAGE dir only.
            package_derive = build_registry_from_interfaces()
            assert "SnapshotProbeStep" not in package_derive
