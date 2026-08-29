"""
Guard for the two-stage calibration edge:
`XGBoostModelInference.processed_data` must accept `PyTorchModelInference`.

In a two-stage stacked model (encoder -> XGBoost head), the calibration branch runs
`PyTorchModelInference_calibration` (encode() -> [tabular | emb]) and feeds it into
`XGBoostModelInference_calibration.processed_data`. If `PyTorchModelInference` is NOT in that
dependency's `compatible_sources`, the edge still resolves — but only by a razor-thin margin over
the semantically-wrong `XGBoostTraining.evaluation_output` (both are processing_output). A small
change could silently flip it, fitting the percentile calibration map on the wrong distribution.
"""

import os

import pytest
import yaml

import cursus
from cursus.core.base.step_interface import StepInterface
from cursus.core.deps import (
    UnifiedDependencyResolver,
    SpecificationRegistry,
    SemanticMatcher,
)

_IFACE = os.path.join(os.path.dirname(cursus.__file__), "steps", "interfaces")


def _load(name):
    with open(os.path.join(_IFACE, f"{name}.step.yaml")) as f:
        return StepInterface(**yaml.safe_load(f))


class TestCalibrationProcessedDataSource:
    def test_processed_data_lists_pytorch_model_inference(self):
        """The spec must declare PyTorchModelInference as a processed_data source (Edit 3)."""
        spec = _load("xgboost_model_inference")
        cs = spec.spec.dependencies["processed_data"].compatible_sources
        assert "PyTorchModelInference" in cs, (
            "xgboost_model_inference processed_data.compatible_sources must include "
            "PyTorchModelInference so the two-stage calibration edge resolves robustly"
        )

    def test_calibration_edge_binds_pytorch_not_xgboost_training(self):
        """processed_data <- PyTorchModelInference (encoded data); model_input <- XGBoostTraining."""
        reg = SpecificationRegistry()
        reg.register("XGBoostTraining", _load("xgboost_training"))
        reg.register(
            "PyTorchModelInference_calibration", _load("pytorch_model_inference")
        )
        reg.register(
            "XGBoostModelInference_calibration", _load("xgboost_model_inference")
        )
        r = UnifiedDependencyResolver(reg, SemanticMatcher())

        resolved = r.resolve_step_dependencies(
            "XGBoostModelInference_calibration",
            ["XGBoostTraining", "PyTorchModelInference_calibration"],
        )
        assert (
            resolved["processed_data"].step_name == "PyTorchModelInference_calibration"
        )
        assert resolved["model_input"].step_name == "XGBoostTraining"

    def test_processed_data_margin_is_robust(self):
        """The correct source must out-score the competing processing_output by a solid margin."""
        from types import SimpleNamespace

        r = UnifiedDependencyResolver(SpecificationRegistry(), SemanticMatcher())
        pd_dep = _load("xgboost_model_inference").spec.dependencies["processed_data"]
        pmi_eval = _load("pytorch_model_inference").spec.outputs["eval_output"]
        xgb_eval = _load("xgboost_training").spec.outputs["evaluation_output"]

        correct = r._calculate_compatibility(
            pd_dep, pmi_eval, SimpleNamespace(step_type="PyTorchModelInference")
        )
        wrong = r._calculate_compatibility(
            pd_dep, xgb_eval, SimpleNamespace(step_type="XGBoostTraining")
        )
        assert correct > wrong + 0.1, (
            f"calibration processed_data source margin too thin: "
            f"PyTorchModelInference={correct:.3f} vs XGBoostTraining={wrong:.3f}"
        )
