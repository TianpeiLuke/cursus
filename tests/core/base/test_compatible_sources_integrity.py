"""compatible_sources referential-integrity gate (FZ 31e1d3f gap 3).

A `.step.yaml` dependency's `compatible_sources` is matched by the UnifiedDependencyResolver as a
+10% scoring bonus: `normalize(provider.step_type) in dep.compatible_sources`
(`dependency_resolver.py:321/492`). A name in the list that is NOT a real (normalized) step type can
NEVER contribute the bonus — it is dead weight that SILENTLY weakens the edge (the edge can still
resolve on other signals, so it never errors). The dangerous case is a CASE typo: `PytorchModel`
instead of `PyTorchModel` — it looks right, matches case-insensitively to a real step, but the
resolver's exact-membership check misses it.

This gate flags exactly those high-confidence typos: a compatible_sources entry that
case-insensitively matches a real step name but differs in case. It deliberately TOLERATES the
legitimate non-step sources that appear by design:
  - job-type-variant forms (`RiskTableMapping_Training` — the resolver normalizes the suffix off),
  - generic/abstract SageMaker types (`ProcessingStep`, `TrainingStep`, `ModelStep`, ...),
  - external/source markers (`S3Source`, `UserProvided`, `LocalDataStep`, ...),
  - legacy/role aliases (`PackagingStep`, `ModelEvaluation`, ...).
Those are loose-match sources, not typos, so a strict "must be a current step name" gate would be
wrong + noisy. We only assert the one thing that is unambiguously a bug.
"""

import logging
import warnings

import pytest


@pytest.fixture(scope="module")
def all_compatible_sources():
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    from cursus.registry.step_names import get_step_names
    from cursus.steps.interfaces import (
        clear_interface_cache,
        list_available_interfaces,
        load_interface,
    )

    clear_interface_cache()
    step_names = set(get_step_names())
    rows = []  # (step, dep_logical_name, source_value)
    for stem in list_available_interfaces():
        try:
            iface = load_interface(stem)
        except Exception:
            continue
        for dep in iface.spec.dependencies.values():
            for src in dep.compatible_sources or []:
                rows.append((iface.spec.step_type, dep.logical_name, src))
    return step_names, rows


def test_no_case_typos_in_compatible_sources(all_compatible_sources):
    """No compatible_sources entry is a CASE-variant of a real step name (the silent-edge-weakening
    typo, e.g. PytorchModel for PyTorchModel)."""
    step_names, rows = all_compatible_sources
    lower_to_canonical = {s.lower(): s for s in step_names}

    typos = []
    for step, dep, src in rows:
        if src in step_names:
            continue  # exact match — fine
        canonical = lower_to_canonical.get(src.lower())
        if canonical is not None and canonical != src:
            typos.append(f"{step}.{dep}: compatible_sources has {src!r} — should be {canonical!r}")
    assert not typos, (
        "compatible_sources case typos (silently lose the +10% resolver bonus): " + "; ".join(typos)
    )


def test_compatible_sources_are_populated(all_compatible_sources):
    """Sanity: the gate is actually scanning data (guards against a no-op gate)."""
    _, rows = all_compatible_sources
    assert len(rows) >= 50
