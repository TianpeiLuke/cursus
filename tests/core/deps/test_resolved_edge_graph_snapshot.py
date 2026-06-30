"""
FZ 31e1d3d1 — the resolved-edge-graph snapshot gate.

The pipeline's step<->step wiring is computed by the dependency resolver from `.step.yaml` spec
DATA (compatible_sources / semantic_keywords / aliases / types), NOT from any builder class. That
makes the wiring graph invariant to the Strategy+Facade collapse (44 builder classes -> 1 facade) —
BUT it makes the graph highly sensitive to the YAML *transcription* that rides along with migration.
The two confirmed-real risks from the connection-mechanism preservation analysis are both
transcription drift: a dropped alias/semantic_keyword (already happened once — commit bb17cd19) or a
dropped job_type silently WEAKENS an edge's score or drops it below the 0.5 viability threshold, with
no error (the assembler substitutes a dead s3://pipeline-reference/ placeholder).

`_sync_and_align` only validates INTRA-step contract<->spec key containment; nothing validates the
INTER-step edges. This gate fills that hole: it loads the REAL `.step.yaml` interfaces for a
representative DAG, resolves the full edge graph, and asserts (a) the load-bearing backbone edges
resolve to the correct producer, and (b) the complete resolved edge set matches a frozen snapshot.
Any future change — a facade migration, a YAML edit, a resolver tweak — that alters the wiring fails
here instead of silently mis-wiring a pipeline.

To intentionally re-baseline after a deliberate interface change: run with CURSUS_UPDATE_EDGE_SNAPSHOT=1
to print the new snapshot, then paste it into _SNAPSHOT below.
"""

import os
import warnings

import pytest

warnings.simplefilter("ignore")

from cursus.core.deps import (  # noqa: E402
    SemanticMatcher,
    SpecificationRegistry,
    UnifiedDependencyResolver,
)
from cursus.steps.interfaces import load_interface  # noqa: E402

# A representative training+registration DAG built from real .step.yaml interfaces. These steps load
# without the SAIS SDK (Cradle/Redshift excluded — they can't import in a SDK-less env).
_DAG_STEPS = [
    "TabularPreprocessing",
    "XGBoostTraining",
    "XGBoostModel",
    "Package",
    "Payload",
    "Registration",
]

# Frozen snapshot of the resolved edge graph: {"consumer.dep": "producer.output"}.
# Regenerate via CURSUS_UPDATE_EDGE_SNAPSHOT=1 (see module docstring) only on a deliberate change.
_SNAPSHOT = {
    "Package.calibration_model": "TabularPreprocessing.processed_data",
    "Package.inference_scripts_input": "XGBoostModel.model_name",
    "Package.model_input": "XGBoostTraining.model_output",
    "Payload.custom_payload_input": "TabularPreprocessing.processed_data",
    "Payload.model_input": "XGBoostTraining.model_output",
    "Registration.GeneratedPayloadSamples": "Payload.payload_sample",
    "Registration.PackagedModel": "Package.packaged_model",
    "TabularPreprocessing.DATA": "XGBoostTraining.evaluation_output",
    "TabularPreprocessing.DATA_SECONDARY": "XGBoostTraining.evaluation_output",
    "TabularPreprocessing.SIGNATURE": "Payload.payload_sample",
    "XGBoostModel.model_data": "XGBoostTraining.model_output",
    "XGBoostTraining.input_path": "TabularPreprocessing.processed_data",
    "XGBoostTraining.model_artifacts_input": "TabularPreprocessing.processed_data",
}

# The load-bearing backbone: edges that MUST resolve to exactly this producer for the pipeline to be
# correct. These are asserted explicitly (not just via the snapshot) so a regression names the broken
# edge rather than dumping a whole-dict diff.
_BACKBONE = {
    "XGBoostTraining.input_path": "TabularPreprocessing.processed_data",
    "XGBoostModel.model_data": "XGBoostTraining.model_output",
    "Package.model_input": "XGBoostTraining.model_output",
    "Registration.PackagedModel": "Package.packaged_model",
    "Registration.GeneratedPayloadSamples": "Payload.payload_sample",
}


def _resolve_edges(steps):
    """Resolve {consumer.dep -> producer.output} from the REAL .step.yaml interfaces."""
    registry = SpecificationRegistry()
    resolver = UnifiedDependencyResolver(registry, SemanticMatcher())
    for s in steps:
        resolver.register_specification(s, load_interface(s))
    resolved = resolver.resolve_all_dependencies(steps)
    return {
        f"{consumer}.{dep}": f"{ref.step_name}.{ref.output_spec.logical_name}"
        for consumer, deps in resolved.items()
        for dep, ref in deps.items()
    }


@pytest.fixture(scope="module")
def edges():
    return _resolve_edges(_DAG_STEPS)


def test_resolution_is_deterministic():
    assert _resolve_edges(_DAG_STEPS) == _resolve_edges(_DAG_STEPS)


@pytest.mark.parametrize("consumer_dep,expected_producer", sorted(_BACKBONE.items()))
def test_backbone_edge_resolves_to_correct_producer(edges, consumer_dep, expected_producer):
    """Each load-bearing edge wires to exactly the right producer output (names the break on fail)."""
    assert edges.get(consumer_dep) == expected_producer, (
        f"{consumer_dep} should wire to {expected_producer}, got {edges.get(consumer_dep)!r} — "
        f"a dropped alias/keyword/job_type or a resolver change weakened this edge."
    )


def test_full_edge_graph_matches_snapshot(edges):
    """The complete resolved edge set is frozen; any drift (new/dropped/rewired edge) fails here."""
    if os.environ.get("CURSUS_UPDATE_EDGE_SNAPSHOT"):
        import json

        print("\n_SNAPSHOT = " + json.dumps(dict(sorted(edges.items())), indent=4))
    assert edges == _SNAPSHOT, (
        "Resolved edge graph drifted from the frozen snapshot. If this change is INTENTIONAL "
        "(a deliberate .step.yaml edit), re-baseline via CURSUS_UPDATE_EDGE_SNAPSHOT=1. If not, a "
        "transcription error (dropped alias/keyword/job_type) silently rewired the pipeline."
    )
