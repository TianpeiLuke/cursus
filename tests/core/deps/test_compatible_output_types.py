"""
Tests for the per-dependency `compatible_output_types` opt-in on DependencyDecl.

A dependency declares a single `type`, and the resolver's type-compatibility matrix normally
hard-blocks any producer output whose type is not matrix-compatible with that declared type
(score 0.0). `compatible_output_types` lets a SPECIFIC dependency opt into accepting additional
producer output types beyond the matrix — without changing behavior for any other dependency.

Motivating case: a two-stage stacked model (encoder -> XGBoost head) where the XGBoost training
step's `model_artifacts_input` (type `processing_output`, normally fed loose imputation/risk/
feature-selection artifacts from Processing steps) must ALSO accept a Stage-1 PyTorch encoder,
which is a Training step's `model_artifacts` output (`S3ModelArtifacts`). Without the opt-in the
`model_artifacts` -> `processing_output` edge is type-blocked at 0.0.
"""

from types import SimpleNamespace

import pytest

from cursus.core.deps import (
    UnifiedDependencyResolver,
    SpecificationRegistry,
    SemanticMatcher,
    DependencyResolutionError,
)
from cursus.core.base.step_interface import StepInterface, DependencyDecl, OutputDecl
from cursus.core.base.enums import NodeType


def _encoder_producer():
    """A training step emitting a model_artifacts output (like PyTorchTraining.model_output)."""
    return StepInterface(
        step_type="PyTorchTraining",
        node_type=NodeType.SOURCE,
        contract={},
        spec={
            "dependencies": {},
            "outputs": {
                "model_output": {
                    "type": "model_artifacts",
                    "property_path": "properties.ModelArtifacts.S3ModelArtifacts",
                    "aliases": [
                        "ModelArtifacts",
                        "model_data",
                        "model_artifacts_input",
                    ],
                    "data_type": "S3Uri",
                }
            },
        },
    )


def _xgboost_consumer(with_opt_in: bool, required: bool = False):
    """XGBoost-training-like consumer whose model_artifacts_input is processing_output."""
    dep = {
        "type": "processing_output",
        "required": required,
        "compatible_sources": [
            "PyTorchTraining",
            "XGBoostTraining",
            "MissingValueImputation",
        ],
        "semantic_keywords": ["artifacts", "model_artifacts", "pretrained", "encoder"],
        "data_type": "S3Uri",
    }
    if with_opt_in:
        dep["compatible_output_types"] = ["model_artifacts"]
    return StepInterface(
        step_type="XGBoostTraining",
        node_type=NodeType.SINK,
        contract={},
        spec={"dependencies": {"model_artifacts_input": dep}, "outputs": {}},
    )


class TestCompatibleOutputTypes:
    @pytest.fixture
    def registry(self):
        return SpecificationRegistry()

    @pytest.fixture
    def resolver(self, registry):
        return UnifiedDependencyResolver(registry, SemanticMatcher())

    def test_opt_in_allows_model_artifacts_into_processing_output_dep(
        self, resolver, registry
    ):
        """With the opt-in, a model_artifacts output binds to the processing_output dependency."""
        registry.register("encoder", _encoder_producer())
        registry.register("xgb", _xgboost_consumer(with_opt_in=True, required=True))

        resolved = resolver.resolve_step_dependencies("xgb", ["encoder"])

        assert "model_artifacts_input" in resolved

    def test_without_opt_in_required_dep_raises(self, resolver, registry):
        """Without the opt-in, the model_artifacts output is type-blocked; a required dep raises."""
        registry.register("encoder", _encoder_producer())
        registry.register("xgb", _xgboost_consumer(with_opt_in=False, required=True))

        with pytest.raises(DependencyResolutionError):
            resolver.resolve_step_dependencies("xgb", ["encoder"])

    def test_without_opt_in_optional_dep_left_unresolved(self, resolver, registry):
        """Optional processing_output dep with no opt-in simply does not bind a model_artifacts output."""
        registry.register("encoder", _encoder_producer())
        registry.register("xgb", _xgboost_consumer(with_opt_in=False, required=False))

        resolved = resolver.resolve_step_dependencies("xgb", ["encoder"])

        assert "model_artifacts_input" not in resolved

    def test_scoring_opt_in_binds_control_blocked(self, resolver):
        """The opt-in edge clears the 0.5 bind threshold; the identical pair without it scores 0.0."""
        out_spec = OutputDecl(
            type="model_artifacts",
            logical_name="model_output",
            aliases=["ModelArtifacts", "model_data", "model_artifacts_input"],
        )
        provider = SimpleNamespace(step_type="PyTorchTraining")

        dep_opt_in = DependencyDecl(
            type="processing_output",
            compatible_output_types=["model_artifacts"],
            compatible_sources=["PyTorchTraining"],
            semantic_keywords=["artifacts", "model_artifacts", "pretrained", "encoder"],
            logical_name="model_artifacts_input",
        )
        assert resolver._calculate_compatibility(dep_opt_in, out_spec, provider) > 0.5

        dep_control = DependencyDecl(
            type="processing_output",
            compatible_sources=["PyTorchTraining"],
            semantic_keywords=["artifacts"],
            logical_name="model_artifacts_input",
        )
        assert resolver._calculate_compatibility(dep_control, out_spec, provider) == 0.0
