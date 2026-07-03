"""Dependency-axis declaration ↔ actual-import conformance gate (FZ 31e1d3l).

MODEL: a step PATTERN may couple to a 3rd-party package (``mods_workflow_core`` / the SAIS SDK).
That coupling is declared as DATA in the ``.step.yaml`` so the mods-vs-native split is authored +
visible (``cursus steps patterns``) — but a declaration is only trustworthy if it can't drift from
the real imports. This gate re-derives each declared ``requires`` from the actual import graph and
fails on any mismatch. Three axes, kept distinct:

  - compute.requires  (``mods_workflow_core``): the script/kms_network ScriptProcessor path in
    ``builder_base._create_compute`` (lazy import). requires=='mods_workflow_core' IFF kms_network.
  - registry.requires (``secure_ai_sandbox_workflow_python_sdk``): the SDKDelegation builders that do
    a module-level ``from secure_ai_sandbox_workflow_python_sdk import <Step>`` (fatal-on-load).
  - contract.runtime_requires (``secure_ai_sandbox_python_lib`` / proxy models): the step's SCRIPT's
    in-container imports — a RUNTIME dep, never conflated with the build-time ones above.

Plus a leak guard: no builder MODULE may import a runtime-only SAIS lib at module level.
"""

import ast
import glob
import os

import pytest
import yaml

INTERFACES_DIR = "src/cursus/steps/interfaces"
SCRIPTS_DIR = "src/cursus/steps/scripts"

SAIS_SDK = "secure_ai_sandbox_workflow_python_sdk"
SAIS_LIB = "secure_ai_sandbox_python_lib"
PROXY = "com.amazon.secureaisandboxproxyservice"


def _all_imports_any_scope(path):
    """Set of imported root package names anywhere in the file (incl. lazy/in-method)."""
    try:
        tree = ast.parse(open(path).read())
    except (OSError, SyntaxError):
        return set()
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                roots.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.fixture(scope="module")
def interfaces():
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from cursus.steps.interfaces import clear_interface_cache, load_interface

    clear_interface_cache()
    out = {}
    for f in sorted(glob.glob(f"{INTERFACES_DIR}/*.step.yaml")):
        st = (yaml.safe_load(open(f)) or {}).get("step_type")
        if not st:
            continue
        try:
            out[st] = load_interface(st)
        except Exception:
            continue
    return out


class TestComputeAxisDependency:
    def test_compute_requires_iff_kms_network(self, interfaces):
        """compute.requires == 'mods_workflow_core' EXACTLY when kms_network — the lazy import in
        builder_base._create_compute sits under `if spec.kms_network`."""
        bad = []
        for st, iface in interfaces.items():
            cs = iface.compute  # top-level section (FZ 31e1d3k)
            expect = "mods_workflow_core" if cs.kms_network else "none"
            if cs.requires != expect:
                bad.append((st, cs.requires, expect, cs.kms_network))
        assert not bad, f"compute.requires drifted from kms_network: {bad}"

    def test_top_level_compute_mirrors_contract(self, interfaces):
        """compute is a TOP-LEVEL .step.yaml section; the contract.compute back-compat mirror must
        stay equal to it (kept in sync by StepInterface._sync_and_align)."""
        bad = [st for st, iface in interfaces.items() if iface.compute != iface.contract.compute]
        assert not bad, f"top-level compute != contract.compute mirror for: {bad}"

    def test_only_script_kind_touches_mods_in_create_compute(self):
        """The ONLY `from mods_workflow_core` import in builder_base._create_compute must sit under
        the `if spec.kms_network:` branch — catches a new unguarded mods import on another kind."""
        src = open("src/cursus/core/base/builder_base.py").read()
        tree = ast.parse(src)
        fn = next(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_create_compute"),
            None,
        )
        assert fn is not None, "_create_compute not found"
        # find every mods_workflow_core import and assert each has an `if ...kms_network` ancestor
        offenders = []
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("mods_workflow_core"):
                # locate the import's enclosing `if` tests by line containment
                guarded = any(
                    isinstance(a, ast.If)
                    and "kms_network" in ast.dump(a.test)
                    and a.lineno <= node.lineno <= (a.end_lineno or node.lineno)
                    for a in ast.walk(fn)
                )
                if not guarded:
                    offenders.append(node.lineno)
        assert not offenders, (
            f"unguarded mods_workflow_core import in _create_compute at lines {offenders} "
            "(must be under `if spec.kms_network:`)"
        )


class TestCreateStepAxisDependency:
    def test_registry_requires_matches_sdk_bindings(self, interfaces):
        """The set of steps declaring registry.requires=='secure_ai_sandbox_workflow_python_sdk' MUST
        equal the sdk_bindings thunk table — the new home of the SAIS class reference after the
        per-step builder files were deleted (FZ 31e1d3g3 Phase E / 31e1d3g2 A2). Originally this
        compared registry.requires against each builder module's module-level
        `from <SAIS_SDK> import ...`; the builders are gone, so the only remaining SAIS *Step class
        reference is the lazy sdk_bindings thunk. Catches the Redshift-style asymmetry (a SAIS dep
        declared without a binding, or a binding without the declaration)."""
        from cursus.step_catalog.sdk_bindings import SDK_STEP_CLASS_THUNKS

        declared = {
            st
            for st, iface in interfaces.items()
            if getattr(iface.registry, "requires", "none") == SAIS_SDK
        }
        bound = set(SDK_STEP_CLASS_THUNKS)
        assert declared == bound, (
            f"registry.requires (SAIS-SDK) drifted from the sdk_bindings table: "
            f"declared-only={declared - bound}, bound-only={bound - declared}"
        )

    def test_at_least_the_four_sdk_steps_declared(self, interfaces):
        """Sanity: the four known SDKDelegation steps declare the SAIS-SDK dependency."""
        declared = {
            st for st, i in interfaces.items()
            if getattr(i.registry, "requires", "none") == SAIS_SDK
        }
        assert {"Registration", "CradleDataLoading", "DataUploading", "RedshiftDataLoading"} <= declared


class TestRuntimeAxisDependency:
    def test_runtime_requires_matches_script_imports(self, interfaces):
        """Each step's declared contract.runtime_requires must equal the SAIS runtime packages its
        SCRIPT actually imports (entry_point under steps/scripts/). Keeps the runtime descriptor
        honest without conflating it with build-time deps."""
        runtime_pkgs = {SAIS_LIB, PROXY.split(".")[0]}  # secure_ai_sandbox_python_lib / com(.amazon...)
        bad = []
        for st, iface in interfaces.items():
            ep = iface.contract.entry_point
            declared = set(getattr(iface.contract, "runtime_requires", []) or [])
            if not ep:
                # no script → declared runtime deps must be empty
                if declared:
                    bad.append((st, sorted(declared), "[no entry_point]"))
                continue
            spath = os.path.join(SCRIPTS_DIR, ep)
            if not os.path.exists(spath):
                # script not vendored here (e.g. SDK-provided) → can't verify; skip
                continue
            actual = _all_imports_any_scope(spath) & runtime_pkgs
            # normalize: declared uses full names; map back to roots for comparison
            declared_roots = {d.split(".")[0] for d in declared}
            if declared_roots != actual:
                bad.append((st, sorted(declared_roots), sorted(actual)))
        assert not bad, f"runtime_requires drifted from script imports: {bad}"


class TestBuildTimeRuntimeSeparation:
    def test_no_interface_declares_runtime_only_sais_lib_as_build_time_dep(self, interfaces):
        """A RUNTIME-only SAIS package (secure_ai_sandbox_python_lib or the proxy-service models) must
        never appear in a BUILD-TIME dependency axis (compute.requires / registry.requires) — those
        belong to the step's SCRIPT via contract.runtime_requires. Under interface-first authoring the
        per-step builder modules are gone, so the only remaining place a build-time dep is declared is
        the .step.yaml itself; this re-derives the guard from the interface dependency axes rather than
        parsing (now-nonexistent) builder module imports."""
        runtime_only = {SAIS_LIB, PROXY.split(".")[0]}  # secure_ai_sandbox_python_lib / com(.amazon...)
        leaks = []
        for st, iface in interfaces.items():
            compute_req = getattr(iface.compute, "requires", "none")
            registry_req = getattr(iface.registry, "requires", "none")
            for axis_name, req in (("compute", compute_req), ("registry", registry_req)):
                if req and req.split(".")[0] in runtime_only:
                    leaks.append((st, axis_name, req))
        assert not leaks, f"interfaces declare a runtime-only SAIS package as a build-time dep: {leaks}"
