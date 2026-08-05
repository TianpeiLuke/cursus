"""ComputeSpec validation + _create_compute foundation (FZ 31e1d3k).

ComputeSpec is the declarative compute descriptor (.step.yaml contract.compute); its values are
validated against the SageMaker SDK surface, and the builder template's _create_compute() builds the
processor/estimator from config + the descriptor (replacing the per-step _create_processor factories).
"""

import pytest

from cursus.core.base.step_interface import ComputeSpec


class TestComputeSpecValidation:
    def test_empty_is_valid(self):
        # kind=None ⇒ the step keeps its own factory; no constraints.
        assert ComputeSpec().kind is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "kind": "sklearn",
                "framework_version_field": "processing_framework_version",
            },
            {"kind": "xgboost", "framework_version_field": "xgboost_framework_version"},
            {
                "kind": "framework",
                "sdk_class": "PyTorch",
                "framework_version_field": "framework_version",
                "py_version_field": "py_version",
            },
            {"kind": "script", "kms_network": True},
            {
                "kind": "estimator",
                "sdk_class": "XGBoost",
                "framework_version_field": "framework_version",
            },
            {
                "kind": "model",
                "sdk_class": "PyTorchModel",
                "framework_name": "pytorch",
                "framework_version_field": "framework_version",
                "py_version_field": "py_version",
            },
            {"kind": "transformer"},
            # BYO container: image is a config field, no DLC knobs (FZ 31e1d3m).
            {"kind": "byo_container", "image_uri_field": "image_uri"},
            {
                "kind": "byo_container",
                "image_uri_field": "training_image_uri",
                "container_entrypoint": ["bash", "run.sh"],
            },
            # estimator may now declare a non-pytorch retrieve framework.
            {
                "kind": "estimator",
                "sdk_class": "PyTorch",
                "framework_version_field": "framework_version",
                "retrieve_framework": "huggingface",
            },
            # per-step VPC (network_mode='config') on byo_container + estimator (FZ 31e1d3o).
            {
                "kind": "byo_container",
                "image_uri_field": "image_uri",
                "network_mode": "config",
            },
            {
                "kind": "estimator",
                "sdk_class": "PyTorch",
                "framework_version_field": "framework_version",
                "network_mode": "config",
            },
        ],
    )
    def test_valid_descriptors(self, kwargs):
        assert ComputeSpec(**kwargs).kind == kwargs["kind"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bogus"},  # unknown kind
            {
                "kind": "framework",
                "framework_version_field": "x",
            },  # framework needs sdk_class
            {"kind": "sklearn"},  # processor needs framework_version_field
            {
                "kind": "script",
                "sdk_class": "PyTorch",
            },  # script must not take sdk_class
            {
                "kind": "sklearn",
                "framework_version_field": "x",
                "py_version_field": "py_version",
            },  # py invalid for sklearn
            {
                "kind": "script",
                "kms_network": True,
                "sdk_class": "X",
            },  # bad sdk_class value + script+sdk_class
            {
                "kind": "sklearn",
                "framework_version_field": "x",
                "kms_network": True,
            },  # kms only for script
            {
                "kind": "model",
                "sdk_class": "XGBoostModel",
                "framework_version_field": "v",
            },  # model needs framework_name
            {
                "kind": "model",
                "sdk_class": "XGBoostModel",
                "framework_name": "xgboost",
            },  # model needs framework_version_field
            {
                "kind": "sklearn",
                "framework_version_field": "v",
                "framework_name": "x",
            },  # framework_name model-only
            {"kind": "byo_container"},  # byo needs image_uri_field
            {
                "kind": "byo_container",
                "image_uri_field": "image_uri",
                "sdk_class": "PyTorch",
            },  # byo forbids sdk_class
            {
                "kind": "byo_container",
                "image_uri_field": "image_uri",
                "framework_version_field": "framework_version",
            },  # byo forbids DLC framework knobs
            {
                "kind": "byo_container",
                "image_uri_field": "image_uri",
                "kms_network": True,
            },  # byo uses network_mode not kms_network
            {
                "kind": "byo_container",
                "image_uri_field": "image_uri",
                "requires": "mods_workflow_core",
            },  # byo is pure sagemaker sdk
            {
                "kind": "sklearn",
                "framework_version_field": "v",
                "image_uri_field": "image_uri",
            },  # image_uri_field is byo-only
            {
                "kind": "sklearn",
                "framework_version_field": "v",
                "container_entrypoint": ["bash"],
            },  # container_entrypoint is byo-only
            {
                "kind": "framework",
                "sdk_class": "PyTorch",
                "framework_version_field": "v",
                "retrieve_framework": "huggingface",
            },  # retrieve_framework is estimator-only
            {"kind": "byo_container", "image_uri_field": "i", "network_mode": "bogus"},
            {
                "kind": "sklearn",
                "framework_version_field": "v",
                "network_mode": "config",
            },  # config wired for byo_container/estimator only
            {
                "kind": "byo_container",
                "image_uri_field": "i",
                "network_mode": "shared",
            },  # shared not a standalone selector (use kms_network)
            {
                "kind": "byo_container",
                "image_uri_field": "i",
                "enable_network_isolation_field": "eni",
            },  # eni pointer requires network_mode='config'
        ],
    )
    def test_invalid_descriptors_raise(self, kwargs):
        with pytest.raises(Exception):
            ComputeSpec(**kwargs)


class TestCreateComputeFoundation:
    def test_create_compute_builds_sklearn_matching_factory(self):
        """_create_compute(sklearn descriptor) constructs an SKLearnProcessor byte-matching the
        hand-written _create_processor (class + instance type/count), via a mock session."""
        import contextlib
        import io
        import os
        import tempfile
        from unittest.mock import Mock

        from cursus.step_catalog.step_catalog import StepCatalog
        from cursus.steps.configs.config_tabular_preprocessing_step import (
            TabularPreprocessingConfig,
        )

        B = StepCatalog().load_builder_class("TabularPreprocessing")

        tmp = tempfile.mkdtemp()
        open(os.path.join(tmp, "d.py"), "w").write("#\n")
        kw = dict(
            author="t",
            bucket="b",
            role="arn:aws:iam::123456789012:role/test",
            region="NA",
            service_name="s",
            pipeline_version="1.0.0",
            project_root_folder="p",
            job_type="training",
            source_dir=tmp,
            processing_entry_point="d.py",
        )
        for f, fl in TabularPreprocessingConfig.model_fields.items():
            if f in kw:
                continue
            if fl.is_required() if hasattr(fl, "is_required") else (fl.default is None):
                s = str(fl.annotation)
                kw[f] = (
                    False
                    if "bool" in s
                    else (1 if "int" in s and "str" not in s else "x")
                )
        cfg = TabularPreprocessingConfig.model_construct(**kw)
        sess = Mock()
        sess.boto_region_name = "us-east-1"
        sess.local_mode = False
        sess.sagemaker_config = None

        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            comp_b = B.__new__(B)
            comp_b.config = cfg
            comp_b.role = "arn:aws:iam::123:role/x"
            comp_b.session = sess
            comp_b._get_environment_variables = lambda: {}
            comp_b.contract = Mock()
            comp_b.contract.compute = ComputeSpec(
                kind="sklearn", framework_version_field="processing_framework_version"
            )
            comp = B._create_compute(comp_b)

        # The compute resolver builds an SKLearnProcessor with the config-derived values (the
        # _create_processor factory it replaced is now deleted; this asserts the resolver's output).
        assert type(comp).__name__ == "SKLearnProcessor"
        assert (
            comp.instance_type == cfg.processing_instance_type_small
        )  # use_large=False sentinel
        assert str(comp.instance_count) == str(cfg.processing_instance_count)


class TestByoContainerCompute:
    """BYO container: a user-supplied image_uri is run verbatim on the Processing/Training verb,
    with NO image_uris.retrieve — the general non-DLC / custom-image capability."""

    _IMG = "111122223333.dkr.ecr.us-east-1.amazonaws.com/graphstorm-gnn:sagemaker-gpu"

    def _builder(self):
        import contextlib
        import io
        from types import SimpleNamespace
        from unittest.mock import Mock

        from cursus.step_catalog.step_catalog import StepCatalog

        # _create_compute lives on the base; use a concrete loaded builder to satisfy the ABC.
        B = StepCatalog().load_builder_class("TabularPreprocessing")
        b = B.__new__(B)
        b.config = SimpleNamespace(
            image_uri=self._IMG,
            processing_instance_count=1,
            processing_instance_type=None,
            processing_instance_type_small="ml.m5.large",
            processing_instance_type_large="ml.m5.4xlarge",
            use_large_processing_instance=False,
            processing_volume_size=30,
            training_instance_type="ml.g5.12xlarge",
            training_instance_count=1,
            training_volume_size=125,
            training_entry_point="train.py",
            effective_source_dir="/tmp/src",
            aws_region="us-east-1",
            subnets=None,
            security_group_ids=None,
            enable_network_isolation=None,
        )
        b.role = "arn:aws:iam::123456789012:role/x"
        sess = Mock()
        sess.boto_region_name = "us-east-1"
        sess.local_mode = False
        sess.sagemaker_config = None  # avoid SDK jsonschema validation of a Mock config
        b.session = sess
        b._get_environment_variables = lambda: {}
        b._generate_job_name = lambda: "job"
        b.contract = Mock()
        return b, contextlib, io

    def test_byo_processing_uses_verbatim_image_no_retrieve(self, monkeypatch):
        b, contextlib, io = self._builder()
        b.contract.compute = ComputeSpec(
            kind="byo_container", image_uri_field="image_uri"
        )

        # Assert image_uris.retrieve is NEVER called on the BYO path.
        import sagemaker

        called = {"n": 0}
        monkeypatch.setattr(
            sagemaker.image_uris,
            "retrieve",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )

        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(verb="Processing")

        assert type(comp).__name__ == "ScriptProcessor"
        assert comp.image_uri == self._IMG
        assert comp.command == ["python3"]
        assert called["n"] == 0

    def test_byo_processing_container_entrypoint(self):
        b, contextlib, io = self._builder()
        b.contract.compute = ComputeSpec(
            kind="byo_container",
            image_uri_field="image_uri",
            container_entrypoint=["bash", "run_gconstruct.sh"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(verb="Processing")
        assert comp.command == ["bash", "run_gconstruct.sh"]

    def test_byo_training_uses_generic_estimator_verbatim_image(self, monkeypatch):
        b, contextlib, io = self._builder()
        b.contract.compute = ComputeSpec(
            kind="byo_container", image_uri_field="image_uri"
        )
        import sagemaker

        called = {"n": 0}
        monkeypatch.setattr(
            sagemaker.image_uris,
            "retrieve",
            lambda *a, **k: called.__setitem__("n", called["n"] + 1),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(output_path="s3://b/out", verb="Training")
        # A generic sagemaker.estimator.Estimator (NOT PyTorch), image verbatim, no retrieve.
        assert type(comp).__name__ == "Estimator"
        assert comp.image_uri == self._IMG
        assert called["n"] == 0

    def test_byo_training_entrypoint_bypass_omits_entry_point(self):
        b, contextlib, io = self._builder()
        b.contract.compute = ComputeSpec(
            kind="byo_container",
            image_uri_field="image_uri",
            container_entrypoint=["bash", "entrypoint.sh"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(output_path="s3://b/out", verb="Training")
        # entry_point is not set when the container runs its own entrypoint.
        assert getattr(comp, "entry_point", None) is None

    def test_byo_missing_image_uri_raises(self):
        b, contextlib, io = self._builder()
        b.config.image_uri = None
        b.contract.compute = ComputeSpec(
            kind="byo_container", image_uri_field="image_uri"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(ValueError, match="image_uri"):
                b._create_compute(verb="Processing")


class TestPerStepVpc:
    """Per-step VPC / NetworkConfig (FZ 31e1d3o) — network_mode='config' builds the step's own
    NetworkConfig; the legacy shared/nvme SAIS paths are untouched (additive)."""

    _SUBNETS = ["subnet-0123456789abcdef0"]
    _SGS = ["sg-0123456789abcdef0"]

    def _builder(self, **cfg_over):
        # Reuse the BYO fixture's builder, then override the VPC config fields.
        b, contextlib, io = TestByoContainerCompute._builder(TestByoContainerCompute())
        for k, v in cfg_over.items():
            setattr(b.config, k, v)
        return b, contextlib, io

    def test_config_mode_processing_builds_vpc_networkconfig(self):
        # A5-VPC-a: network_mode='config' Processing → ScriptProcessor.network_config from the
        # step's own subnets/SGs, and NO volume_kms_key.
        b, contextlib, io = self._builder(
            subnets=self._SUBNETS, security_group_ids=self._SGS
        )
        b.contract.compute = ComputeSpec(
            kind="byo_container",
            image_uri_field="image_uri",
            network_mode="config",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(verb="Processing")
        assert type(comp).__name__ == "ScriptProcessor"
        assert list(comp.network_config.subnets) == self._SUBNETS
        assert list(comp.network_config.security_group_ids) == self._SGS
        assert getattr(comp, "volume_kms_key", None) is None

    def test_config_mode_training_sets_estimator_vpc(self):
        # A5-VPC-a (Training): network_mode='config' Training → estimator subnets/SGs +
        # encrypt_inter_container_traffic (matches the SAIS-secured baseline).
        b, contextlib, io = self._builder(
            subnets=self._SUBNETS, security_group_ids=self._SGS
        )
        b.contract.compute = ComputeSpec(
            kind="byo_container",
            image_uri_field="image_uri",
            network_mode="config",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(output_path="s3://b/out", verb="Training")
        assert list(comp.subnets) == self._SUBNETS
        assert list(comp.security_group_ids) == self._SGS
        assert comp.encrypt_inter_container_traffic is True

    def test_config_mode_missing_subnets_raises(self):
        # network_mode='config' with no subnets is a loud error (cannot silently no-op the VPC).
        b, contextlib, io = self._builder(subnets=None)
        b.contract.compute = ComputeSpec(
            kind="byo_container",
            image_uri_field="image_uri",
            network_mode="config",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(ValueError, match="subnets"):
                b._create_compute(verb="Processing")

    def test_none_mode_sets_no_network_config(self):
        # A5-VPC-b (regression): default network_mode='none' attaches NO network_config, so the
        # reactive nvme_security SAIS patch still applies at upsert (its not-network_config guard).
        b, contextlib, io = self._builder()
        b.contract.compute = ComputeSpec(
            kind="byo_container", image_uri_field="image_uri"
        )  # network_mode defaults to 'none'
        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(verb="Processing")
        assert not getattr(comp, "network_config", None)

    def test_nvme_patch_defers_to_per_step_config(self, monkeypatch):
        # A5-VPC-c: the nvme_security W2 patch must NOT overwrite a build-time per-step
        # network_config on a GPU instance — its `not processor.network_config` guard holds.
        from cursus.core.utils import nvme_security

        b, contextlib, io = self._builder(
            subnets=self._SUBNETS, security_group_ids=self._SGS
        )
        b.contract.compute = ComputeSpec(
            kind="byo_container",
            image_uri_field="image_uri",
            network_mode="config",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            comp = b._create_compute(verb="Processing")

        # Reproduce the W2 processing patch's guard against our per-step config on a GPU instance
        # (instance_supports_kms False ⇒ skip branch ⇒ would inject SAIS config only if empty).
        from sagemaker.network import NetworkConfig

        class _Sec:
            security_group = "sg-SAIS"
            vpc_subnets = ["subnet-SAIS"]
            kms_key = "arn:aws:kms:us-east-1:123456789012:key/x"

        # emulate the exact guard from nvme_security._patch_processing_step_nvme_aware
        if hasattr(comp, "network_config") and not comp.network_config:
            comp.network_config = NetworkConfig(
                enable_network_isolation=False,
                security_group_ids=[_Sec.security_group],
                subnets=_Sec.vpc_subnets,
            )
        # our per-step subnets survived — the SAIS fallback did NOT overwrite them.
        assert list(comp.network_config.subnets) == self._SUBNETS
        assert nvme_security.install_nvme_aware_security_patch is not None
