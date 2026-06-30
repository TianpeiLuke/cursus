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
            {"kind": "sklearn", "framework_version_field": "processing_framework_version"},
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
        ],
    )
    def test_valid_descriptors(self, kwargs):
        assert ComputeSpec(**kwargs).kind == kwargs["kind"]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"kind": "bogus"},  # unknown kind
            {"kind": "framework", "framework_version_field": "x"},  # framework needs sdk_class
            {"kind": "sklearn"},  # processor needs framework_version_field
            {"kind": "script", "sdk_class": "PyTorch"},  # script must not take sdk_class
            {"kind": "sklearn", "framework_version_field": "x", "py_version_field": "py_version"},  # py invalid for sklearn
            {"kind": "script", "kms_network": True, "sdk_class": "X"},  # bad sdk_class value + script+sdk_class
            {"kind": "sklearn", "framework_version_field": "x", "kms_network": True},  # kms only for script
            {"kind": "model", "sdk_class": "XGBoostModel", "framework_version_field": "v"},  # model needs framework_name
            {"kind": "model", "sdk_class": "XGBoostModel", "framework_name": "xgboost"},  # model needs framework_version_field
            {"kind": "sklearn", "framework_version_field": "v", "framework_name": "x"},  # framework_name model-only
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

        from cursus.steps.builders import (
            TabularPreprocessingStepBuilder as B,
        )
        from cursus.steps.configs.config_tabular_preprocessing_step import (
            TabularPreprocessingConfig,
        )

        tmp = tempfile.mkdtemp()
        open(os.path.join(tmp, "d.py"), "w").write("#\n")
        kw = dict(
            author="t", bucket="b", role="arn:aws:iam::123456789012:role/test", region="NA",
            service_name="s", pipeline_version="1.0.0", project_root_folder="p",
            job_type="training", source_dir=tmp, processing_entry_point="d.py",
        )
        for f, fl in TabularPreprocessingConfig.model_fields.items():
            if f in kw:
                continue
            if fl.is_required() if hasattr(fl, "is_required") else (fl.default is None):
                s = str(fl.annotation)
                kw[f] = False if "bool" in s else (1 if "int" in s and "str" not in s else "x")
        cfg = TabularPreprocessingConfig.model_construct(**kw)
        sess = Mock()
        sess.boto_region_name = "us-east-1"
        sess.local_mode = False
        sess.sagemaker_config = None

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
        assert comp.instance_type == cfg.processing_instance_type_small  # use_large=False sentinel
        assert str(comp.instance_count) == str(cfg.processing_instance_count)
