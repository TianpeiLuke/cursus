"""
Shared fixtures + import guard for pipeline_catalog tests.

The pipeline_catalog package's __init__ eagerly imports the compile layer
(core.pipeline_factory / core.builders), which imports ``sagemaker``. In a
full dev/CI environment sagemaker is installed and the tests run directly.
When sagemaker is absent, the whole package import fails — so these tests
skip cleanly rather than erroring, and we surface that as an xfail-style skip
reason instead of a collection crash.
"""


import pytest

# Skip the entire pipeline_catalog test package if its (sagemaker-backed)
# public surface cannot be imported in this environment.
pipeline_catalog = pytest.importorskip(
    "cursus.pipeline_catalog",
    reason="cursus.pipeline_catalog requires sagemaker (and deps) to import",
)


@pytest.fixture(scope="session")
def catalog_index():
    """The parsed catalog_index.json."""
    from cursus.pipeline_catalog.shared_dags import get_catalog_index

    return get_catalog_index()


@pytest.fixture(scope="session")
def dag_ids(catalog_index):
    """All DAG ids declared in the catalog index."""
    return [d["id"] for d in catalog_index["dags"]]
