#!/usr/bin/env python3
"""
Integration tests for job-type variant handling via the unified StepInterface.

Job-type variants used to be modeled as separate Python *_SPEC constants
(e.g. DATA_LOADING_TRAINING_SPEC). They are now expressed as a ``variants:``
block inside a single ``.step.yaml`` interface and resolved at load time by
``StepInterface.from_yaml(..., job_type=...)`` / ``load_interface(..., job_type=...)``.

These tests verify that mechanism end to end.
"""

import unittest

import yaml
from pathlib import Path

from cursus.steps.interfaces import load_interface
from cursus.core.base.step_interface import StepInterface

INTERFACES_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "cursus"
    / "steps"
    / "interfaces"
)
CRADLE_YAML = INTERFACES_DIR / "cradle_data_loading.step.yaml"

JOB_TYPES = ["training", "validation", "testing", "calibration"]


class TestJobTypeVariantResolution(unittest.TestCase):
    """The single CradleDataLoading interface resolves each job-type variant."""

    def setUp(self):
        self.assertTrue(CRADLE_YAML.exists(), f"missing interface: {CRADLE_YAML}")
        self.raw = yaml.safe_load(CRADLE_YAML.read_text())

    def test_single_step_type_with_variants_block(self):
        """CradleDataLoading is ONE step with a variants block (not 4 step types)."""
        self.assertEqual(self.raw["step_type"], "CradleDataLoading")
        self.assertIn("variants", self.raw)
        for job_type in JOB_TYPES:
            self.assertIn(
                job_type,
                self.raw["variants"],
                f"cradle interface should declare a '{job_type}' variant",
            )

    def test_each_variant_loads_as_step_interface(self):
        """load_interface resolves every job-type variant to a valid StepInterface."""
        for job_type in JOB_TYPES:
            with self.subTest(job_type=job_type):
                iface = load_interface("CradleDataLoading", job_type=job_type)
                self.assertIsInstance(iface, StepInterface)
                # step_type is the base step type for every variant
                self.assertEqual(iface.step_type, "CradleDataLoading")
                # node_type is a source (no dependencies), with outputs
                self.assertEqual(iface.node_type.value, "source")
                self.assertTrue(iface.outputs, "variant should expose outputs")

    def test_base_load_without_job_type(self):
        """Loading without a job_type yields the base interface."""
        iface = load_interface("CradleDataLoading")
        self.assertIsInstance(iface, StepInterface)
        self.assertEqual(iface.step_type, "CradleDataLoading")

    def test_from_yaml_variant_merge_matches_loader(self):
        """StepInterface.from_yaml merges the variant the same way the loader does."""
        for job_type in JOB_TYPES:
            with self.subTest(job_type=job_type):
                built = StepInterface.from_yaml(self.raw, job_type=job_type)
                loaded = load_interface("CradleDataLoading", job_type=job_type)
                self.assertEqual(
                    set(built.outputs.keys()),
                    set(loaded.outputs.keys()),
                    "from_yaml and load_interface should resolve identical outputs",
                )

    def test_unknown_job_type_falls_back_to_base(self):
        """A job_type with no matching variant falls back to the base spec (warns, does NOT raise).

        Configs deliberately do NOT restrict job_type to the interface's declared variant set — most
        step configs validate job_type only as "lowercase alphanumeric" (open), so a legitimate
        config value that isn't an enumerated variant (e.g. CradleDataLoading's `munged`/`tagging`)
        must resolve to the base spec rather than crash the build. `from_yaml` warns on the fallback;
        the variant-tightening hazard is caught downstream (dependency resolver + `_sync_and_align`).
        """
        base = load_interface("CradleDataLoading")
        fallback = StepInterface.from_yaml(self.raw, job_type="does_not_exist")
        self.assertEqual(set(fallback.outputs.keys()), set(base.outputs.keys()))


class TestGenericPreprocessingInterface(unittest.TestCase):
    """TabularPreprocessing is a single generic interface (no job-type variants)."""

    def test_loads_and_exposes_dependencies_and_outputs(self):
        iface = load_interface("TabularPreprocessing")
        self.assertIsInstance(iface, StepInterface)
        self.assertEqual(iface.step_type, "TabularPreprocessing")
        # Has at least one dependency and one output, all carrying logical names.
        self.assertTrue(iface.dependencies)
        self.assertTrue(iface.outputs)
        for name, dep in iface.dependencies.items():
            self.assertEqual(dep.logical_name, name)
            self.assertIsNotNone(dep.dependency_type)
        for name, out in iface.outputs.items():
            self.assertEqual(out.logical_name, name)
            self.assertIsNotNone(out.output_type)


if __name__ == "__main__":
    unittest.main(verbosity=2)
