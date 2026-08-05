"""Tuning verb — the 6th construction verb (FZ 31e1d3p / plan 31e1d3p2).

Covers the search axis (config mixin + _build_parameter_ranges) and the TuningHandler that wraps a
training estimator in a HyperparameterTuner and builds a TuningStep. The handler reuses
TrainingHandler's five axes verbatim (estimator / channels / output_path); the ONLY new behavior is
the search wrapper, which these tests pin.
"""

import contextlib
import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cursus.steps.configs.config_tuning_step_base import TuningStepConfigMixin


class TestTuningSearchAxisConfig:
    """P2 — the search-axis config mixin + range conversion."""

    def _space(self):
        return {
            "continuous": [
                {
                    "name": "gsf.hyperparam.lr",
                    "min": 1e-4,
                    "max": 1e-2,
                    "scaling": "Logarithmic",
                }
            ],
            "integer": [{"name": "gsf.hyperparam.num_layers", "min": 1, "max": 3}],
            "categorical": [
                {"name": "gsf.hyperparam.aggregator", "values": ["mean", "pool"]}
            ],
        }

    def test_valid_config_defaults(self):
        cfg = TuningStepConfigMixin(
            objective_metric_name="val:roc_auc", search_space=self._space()
        )
        assert cfg.objective_type == "Maximize"
        assert cfg.tuning_strategy == "Bayesian"
        assert cfg.max_jobs == 20
        assert cfg.max_parallel_jobs == 1
        assert cfg.early_stopping_type == "Off"

    def test_build_parameter_ranges_types_and_scaling(self):
        from sagemaker.tuner import (
            ContinuousParameter,
            IntegerParameter,
            CategoricalParameter,
        )
        from cursus.core.base.builder_templates import TuningHandler

        cfg = TuningStepConfigMixin(
            objective_metric_name="val:roc_auc", search_space=self._space()
        )
        ranges = TuningHandler._build_parameter_ranges(
            cfg, ContinuousParameter, IntegerParameter, CategoricalParameter
        )
        assert set(ranges) == {
            "gsf.hyperparam.lr",
            "gsf.hyperparam.num_layers",
            "gsf.hyperparam.aggregator",
        }
        assert isinstance(ranges["gsf.hyperparam.lr"], ContinuousParameter)
        assert ranges["gsf.hyperparam.lr"].scaling_type == "Logarithmic"
        assert isinstance(ranges["gsf.hyperparam.num_layers"], IntegerParameter)
        assert isinstance(ranges["gsf.hyperparam.aggregator"], CategoricalParameter)
        assert ranges["gsf.hyperparam.aggregator"].values == ["mean", "pool"]

    def test_empty_search_space_rejected(self):
        with pytest.raises(ValueError, match="at least one tunable range"):
            TuningStepConfigMixin(objective_metric_name="m", search_space={})

    def test_unknown_range_kind_rejected(self):
        with pytest.raises(ValueError, match="unknown range kinds"):
            TuningStepConfigMixin(
                objective_metric_name="m",
                search_space={"bogus": [{"name": "x", "min": 1, "max": 2}]},
            )

    def test_missing_required_range_key_rejected(self):
        with pytest.raises(ValueError, match="missing required key"):
            TuningStepConfigMixin(
                objective_metric_name="m",
                search_space={"continuous": [{"name": "x", "min": 1}]},  # no max
            )

    def test_byo_container_requires_metric_definitions(self):
        """Composed with a byo_container training config, metric_definitions is mandatory (a custom
        image's objective must be regex-scraped — Nexus launch_hpo.py)."""
        from cursus.steps.configs.config_graphstorm_gnn_training_step import (
            GraphStormGNNTrainingConfig,
        )

        class GraphStormGNNTuningConfig(
            GraphStormGNNTrainingConfig, TuningStepConfigMixin
        ):
            pass

        base = dict(
            author="t",
            bucket="b",
            role="r",
            region="NA",
            service_name="s",
            pipeline_version="1.0.0",
            project_root_folder="pr",
            training_entry_point="train.py",
            training_image_uri="1.dkr.ecr.us-east-1.amazonaws.com/graphstorm:gpu",
            objective_metric_name="val:is_abuse:roc_auc",
            search_space={
                "continuous": [{"name": "gsf.hyperparam.lr", "min": 1e-4, "max": 1e-2}]
            },
        )
        with pytest.raises(ValueError, match="metric_definitions is required"):
            GraphStormGNNTuningConfig(**base)

        cfg = GraphStormGNNTuningConfig(
            **base,
            metric_definitions=[
                {"Name": "val:is_abuse:roc_auc", "Regex": r"roc_auc.: ([0-9\.]+)"}
            ],
        )
        # inherits the wrapped training config's estimator fields
        assert cfg.training_instance_type == "ml.g5.12xlarge"
        assert cfg.training_image_uri.endswith("graphstorm:gpu")


class TestTuningHandlerBuildStep:
    """P3 — TuningHandler wraps the training estimator in a HyperparameterTuner -> TuningStep."""

    _IMG = "111122223333.dkr.ecr.us-east-1.amazonaws.com/graphstorm-gnn:sagemaker-gpu"

    def _builder(self):
        from cursus.step_catalog.step_catalog import StepCatalog
        from cursus.core.base.step_interface import ComputeSpec

        # A concrete loaded builder to satisfy the ABC; we drive its handler directly.
        B = StepCatalog().load_builder_class("GraphStormGNNTraining")
        b = B.__new__(B)
        b.config = SimpleNamespace(
            training_image_uri=self._IMG,
            training_instance_type="ml.g5.12xlarge",
            training_instance_count=1,
            training_volume_size=125,
            training_entry_point="train.py",
            effective_source_dir="/tmp/src",
            aws_region="us-east-1",
            subnets=None,
            security_group_ids=None,
            enable_network_isolation=None,
            # search axis:
            objective_metric_name="val:is_abuse:roc_auc",
            objective_type="Maximize",
            tuning_strategy="Bayesian",
            max_jobs=12,
            max_parallel_jobs=2,
            early_stopping_type="Off",
            metric_definitions=[
                {"Name": "val:is_abuse:roc_auc", "Regex": r"roc_auc.: ([0-9\.]+)"}
            ],
            search_space={
                "continuous": [{"name": "gsf.hyperparam.lr", "min": 1e-4, "max": 1e-2}],
                "integer": [{"name": "gsf.hyperparam.num_layers", "min": 1, "max": 3}],
            },
        )
        b.role = "arn:aws:iam::123456789012:role/x"
        from sagemaker.workflow.pipeline_context import PipelineSession

        sess = Mock(spec=PipelineSession)
        sess.boto_region_name = "us-east-1"
        sess.local_mode = False
        sess.sagemaker_config = None
        b.session = sess
        b._get_environment_variables = lambda: {}
        b._generate_job_name = lambda: "gnn-tuning"
        b._get_step_name = lambda: "GraphStormGNNTuning"
        b._get_cache_config = lambda enable: None
        b._get_base_output_path = lambda: "s3://bucket/out"
        # minimal spec/contract so the inherited TrainingHandler.get_inputs fans `input_path` into
        # train/val/test channels and get_outputs derives the output_path.
        dep = SimpleNamespace(
            logical_name="input_path", required=True, dependency_type=None
        )
        out = SimpleNamespace(logical_name="model_output")
        b.spec = SimpleNamespace(
            dependencies={"input_path": dep},
            outputs={"model_output": out},
            step_type="GraphStormGNNTuning",
        )
        b.contract = SimpleNamespace(
            expected_input_paths={"input_path": "/opt/ml/input/data"},
            input_channels={},
            output_path_token=None,
            # a byo_container Training compute so make_compute builds the GraphStorm estimator
            compute=ComputeSpec(
                kind="byo_container",
                image_uri_field="training_image_uri",
                container_entrypoint=["python3", "/opt/ml/input/data/code/train.py"],
            ),
        )
        return b

    def test_build_step_produces_tuning_step_wrapping_the_estimator(self, monkeypatch):
        import sagemaker
        from cursus.core.base.builder_templates import resolve_handler
        from sagemaker.workflow.steps import TuningStep

        # image_uris.retrieve must NEVER be called (byo_container estimator, verbatim image)
        called = {"n": 0}
        monkeypatch.setattr(
            sagemaker.image_uris,
            "retrieve",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )

        b = self._builder()
        handler = resolve_handler("Tuning")
        # feed the training channels directly (bypass dependency resolution)
        with contextlib.redirect_stdout(io.StringIO()):
            step = handler.build_step(b, input_path="s3://bucket/graph")

        assert isinstance(step, TuningStep)
        assert step.name == "GraphStormGNNTuning"
        # Built via step_args (the recommended v2.x path), so the tuner lives inside step_args as
        # the first positional arg of the intercepted tuner.fit call (step.tuner is None here).
        assert step.tuner is None
        assert step.step_args.caller_name == "create_tuning_job"
        tuner = step.step_args.func_args[0]
        assert type(tuner).__name__ == "HyperparameterTuner"
        assert tuner.objective_metric_name == "val:is_abuse:roc_auc"
        assert tuner.strategy == "Bayesian"
        assert tuner.max_jobs == 12
        assert tuner.max_parallel_jobs == 2
        assert set(tuner._hyperparameter_ranges) == {
            "gsf.hyperparam.lr",
            "gsf.hyperparam.num_layers",
        }
        # the wrapped estimator is the verbatim-image GraphStorm estimator (no retrieve)
        assert tuner.estimator.image_uri == self._IMG
        assert called["n"] == 0

    def test_get_top_model_s3_uri_returns_join(self):
        """P4 — a downstream step reads the tuned winner via get_top_model_s3_uri -> Join."""
        from cursus.core.base.builder_templates import resolve_handler
        from sagemaker.workflow.functions import Join

        b = self._builder()
        with contextlib.redirect_stdout(io.StringIO()):
            step = resolve_handler("Tuning").build_step(
                b, input_path="s3://bucket/graph"
            )
        uri = step.get_top_model_s3_uri(top_k=0, s3_bucket="mybucket", prefix="models")
        assert isinstance(uri, Join)


class TestTuningPropertyPaths:
    """P4 — the property-path validator recognizes a Tuning step's BestTrainingJob.* paths
    (the dormant branch was pre-wired; this pins that a Tuning step-type routes to it, not to the
    training branch)."""

    def test_tuning_step_type_resolves_best_training_job_paths(self):
        from cursus.validation.alignment.validators.property_path_validator import (
            SageMakerPropertyPathValidator,
        )

        v = SageMakerPropertyPathValidator()
        for step_type in ("Tuning", "GraphStormGNNTuning"):
            paths = v._get_valid_property_paths_for_step_type(step_type, "internal")
            flat = [p for group in paths.values() for p in group]
            assert any("BestTrainingJob" in p for p in flat), step_type
            assert any("HyperParameterTuningJobConfig" in p for p in flat), step_type


class TestTuningConformance:
    """P6 — the Tuning verb is a first-class, live, routable construction verb."""

    def test_tuning_is_valid_and_routable(self):
        from cursus.registry.step_names import validate_sagemaker_step_type
        from cursus.core.base.builder_templates import resolve_handler, TuningHandler

        assert validate_sagemaker_step_type("Tuning")
        h = resolve_handler("Tuning")
        assert isinstance(h, TuningHandler)
        # TuningHandler reuses TrainingHandler's axes (subclass) — the design invariant.
        from cursus.core.base.builder_templates import TrainingHandler

        assert issubclass(TuningHandler, TrainingHandler)

    def test_graphstorm_gnn_tuning_interface_is_a_live_tuning_step(self):
        from cursus.registry.step_names import get_sagemaker_step_type
        from cursus.steps.interfaces import load_step_interface

        assert get_sagemaker_step_type("GraphStormGNNTuning") == "Tuning"
        contract, spec = load_step_interface("GraphStormGNNTuning")
        # wraps a byo_container estimator; no Processing-only step_assembly on a non-Processing verb.
        assert contract.compute.kind == "byo_container"
        patterns = getattr(spec, "patterns", None)
        if patterns is not None:
            assert getattr(patterns, "step_assembly", None) is None

    def test_tuning_builder_synthesizes_from_registry(self):
        from cursus.step_catalog.step_catalog import StepCatalog

        B = StepCatalog().load_builder_class("GraphStormGNNTuning")
        assert B.__name__ == "GraphStormGNNTuningStepBuilder"
        assert getattr(B, "STEP_NAME", None) == "GraphStormGNNTuning"


class TestNexusGnnTuningEndToEnd:
    """P7 — the Nexus payoff: the real GraphStormGNNTuningConfig compiles to a TuningStep whose
    tuner reproduces the Nexus launch_hpo.py HPO request (objective / strategy / ranges / limits /
    verbatim GraphStorm image / regex metrics), and whose best model is downstream-consumable."""

    _IMG = "123456789012.dkr.ecr.us-east-1.amazonaws.com/graphstorm-gnn:sagemaker-gpu"

    def _config(self):
        from cursus.steps.configs.config_graphstorm_gnn_tuning_step import (
            GraphStormGNNTuningConfig,
        )

        return GraphStormGNNTuningConfig(
            author="t",
            bucket="b",
            role="arn:aws:iam::123456789012:role/x",
            region="NA",
            service_name="s",
            pipeline_version="1.0.0",
            project_root_folder="pr",
            training_entry_point="train.py",
            training_image_uri=self._IMG,
            objective_metric_name="val:is_abuse:roc_auc",
            objective_type="Maximize",
            tuning_strategy="Bayesian",
            max_jobs=20,
            max_parallel_jobs=2,
            metric_definitions=[
                {"Name": "val:is_abuse:roc_auc", "Regex": r"roc_auc.: ([0-9.]+)"}
            ],
            search_space={
                "continuous": [
                    {
                        "name": "gsf.hyperparam.lr",
                        "min": 1e-4,
                        "max": 1e-2,
                        "scaling": "Logarithmic",
                    }
                ],
                "integer": [{"name": "gsf.hyperparam.num_layers", "min": 1, "max": 3}],
            },
        )

    def _builder(self):
        from cursus.step_catalog.step_catalog import StepCatalog
        from cursus.steps.interfaces import load_step_interface
        from sagemaker.workflow.pipeline_context import PipelineSession

        cfg = self._config()
        B = StepCatalog().load_builder_class("GraphStormGNNTuning")
        b = B.__new__(B)
        b.config = cfg
        b.role = cfg.role
        sess = Mock(spec=PipelineSession)
        sess.boto_region_name = "us-east-1"
        sess.local_mode = False
        sess.sagemaker_config = None
        b.session = sess
        b._get_environment_variables = lambda: {}
        b._generate_job_name = lambda: "gnn-hpo"
        b._get_step_name = lambda: "GraphStormGNNTuning"
        b._get_cache_config = lambda e: None
        b._get_base_output_path = lambda: "s3://b/out"
        b.contract, b.spec = load_step_interface("GraphStormGNNTuning")
        return b

    def test_real_config_compiles_to_nexus_shaped_tuning_step(self):
        from cursus.core.base.builder_templates import resolve_handler
        from sagemaker.workflow.steps import TuningStep
        from sagemaker.workflow.functions import Join

        b = self._builder()
        inputs = {
            "graph_data": "s3://b/graph",
            "training_config": "s3://b/config",
            "code": "s3://b/code",
        }
        with contextlib.redirect_stdout(io.StringIO()):
            step = resolve_handler("Tuning").build_step(b, inputs=inputs)

        assert isinstance(step, TuningStep)
        tuner = step.step_args.func_args[0]
        # matches the Nexus launch_hpo.py request
        assert tuner.objective_metric_name == "val:is_abuse:roc_auc"
        assert tuner.objective_type == "Maximize"
        assert tuner.strategy == "Bayesian"
        assert tuner.max_jobs == 20
        assert tuner.max_parallel_jobs == 2
        assert set(tuner._hyperparameter_ranges) == {
            "gsf.hyperparam.lr",
            "gsf.hyperparam.num_layers",
        }
        assert tuner.metric_definitions[0]["Name"] == "val:is_abuse:roc_auc"
        # verbatim GraphStorm image, no image_uris.retrieve
        assert tuner.estimator.image_uri == self._IMG
        # best model is downstream-consumable (feeds a MIMS registration step)
        assert isinstance(
            step.get_top_model_s3_uri(top_k=0, s3_bucket="mb", prefix="models"), Join
        )
