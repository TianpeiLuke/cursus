"""
Unit tests for cursus.pipeline_catalog.core.builders and pipeline_factory.

These compile-layer entry points need a real (or mocked) SageMaker stack to
actually compile, so here we test what can be verified without AWS:
  - the public API surface exists and is importable
  - build_mods_pipeline generates a correctly-shaped, named class with the
    standard MODS interface (without compiling — compilation is exercised in
    integration tests with a live config)
  - the MODSTemplate fallback (no MODS installed) decorates correctly
"""

import pytest

from cursus.pipeline_catalog.core import builders
from cursus.pipeline_catalog.core.builders import build_mods_pipeline, build_and_compile
from cursus.pipeline_catalog.core.pipeline_factory import (
    create_pipeline,
    list_available_pipelines,
)


class TestPublicApi:
    def test_factory_callables(self):
        assert callable(create_pipeline)
        assert callable(list_available_pipelines)

    def test_builder_callables(self):
        assert callable(build_and_compile)
        assert callable(build_mods_pipeline)

    def test_list_available_pipelines_matches_index(self, catalog_index):
        ids = list_available_pipelines()
        assert set(ids) == {d["id"] for d in catalog_index["dags"]}


class TestPipelineFactoryArgValidation:
    def test_requires_dag_id_or_path(self):
        with pytest.raises(ValueError):
            create_pipeline(config_path="x.json")  # neither dag_id nor dag_path

    def test_requires_config_path(self):
        with pytest.raises(ValueError):
            create_pipeline(dag_id="some_dag")  # missing config_path


class TestBuildModsPipeline:
    def test_generates_named_class(self, tmp_path):
        # build_mods_pipeline only constructs the class; it does not compile until
        # generate_pipeline() is called, so dummy paths are fine here.
        cls = build_mods_pipeline(
            author="tester",
            version="0.0.1",
            description="Demo Pipeline",
            dag_path="dag.json",
            config_path="config.json",
            class_name="DemoPipeline",
        )
        assert isinstance(cls, type)
        assert cls.__name__ == "DemoPipeline"
        # standard MODS interface present
        assert hasattr(cls, "__init__")
        assert hasattr(cls, "generate_pipeline")

    def test_default_class_name_derived_from_description(self):
        cls = build_mods_pipeline(
            author="a",
            version="1",
            description="Munged Address Detection",
            dag_path="d.json",
            config_path="c.json",
        )
        assert cls.__name__.endswith("Pipeline")
        assert " " not in cls.__name__

    def test_mods_metadata_attached_in_fallback(self):
        """When MODS is unavailable, the fallback decorator stamps metadata."""
        cls = build_mods_pipeline(
            author="bjjin",
            version="0.0.5",
            description="X",
            dag_path="d.json",
            config_path="c.json",
        )
        if not builders.MODS_AVAILABLE:
            assert cls._mods_author == "bjjin"
            assert cls._mods_version == "0.0.5"
