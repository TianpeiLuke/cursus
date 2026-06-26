"""
Unit tests for cursus.pipeline_catalog.shared_dags — the JSON DAG store.

Covers the intended functionality:
  - catalog_index.json is well-formed and matches the .dag.json files on disk
  - every catalogued DAG loads into a valid PipelineDAG (integrity regression
    guard for the missing-node bug where edges referenced undeclared nodes)
  - load_shared_dag / get_all_shared_dags / search_dags behave as documented
"""

import json
from pathlib import Path

import pytest

from cursus.pipeline_catalog.shared_dags import (
    get_all_shared_dags,
    load_shared_dag,
    search_dags,
    list_dags_by_framework,
    SHARED_DAGS_DIR,
)


class TestCatalogIndex:
    def test_index_is_wellformed(self, catalog_index):
        assert "dags" in catalog_index
        assert isinstance(catalog_index["dags"], list)
        assert len(catalog_index["dags"]) > 0

    def test_index_total_dags_field_matches(self, catalog_index):
        # total_dags (if present) must equal the actual list length
        if "total_dags" in catalog_index:
            assert catalog_index["total_dags"] == len(catalog_index["dags"])

    def test_every_index_entry_has_required_fields(self, catalog_index):
        required = {"id", "path", "framework"}
        for d in catalog_index["dags"]:
            assert required <= set(d), f"{d.get('id')} missing {required - set(d)}"

    def test_dag_ids_are_unique(self, catalog_index):
        ids = [d["id"] for d in catalog_index["dags"]]
        assert len(ids) == len(set(ids)), "duplicate DAG ids in catalog index"

    def test_index_dag_count_matches_files_on_disk(self, catalog_index):
        on_disk = list(Path(SHARED_DAGS_DIR).rglob("*.dag.json"))
        assert len(on_disk) == len(catalog_index["dags"]), (
            f"{len(on_disk)} .dag.json files on disk vs "
            f"{len(catalog_index['dags'])} in index"
        )

    def test_every_index_path_exists_on_disk(self, catalog_index):
        for d in catalog_index["dags"]:
            p = Path(SHARED_DAGS_DIR) / d["path"]
            assert p.exists(), f"{d['id']}: path {d['path']} not found on disk"


class TestLoadSharedDag:
    def test_load_all_dags_succeeds(self, catalog_index):
        """Every catalogued DAG must import into a PipelineDAG (integrity guard).

        Regression: 4 DAGs previously declared fewer nodes than their edges
        referenced (e.g. complete_e2e declared 8 nodes but edges referenced
        TabularPreprocessing_calibration + ModelCalibration_calibration),
        which raised KeyError in PipelineDAG.__init__.
        """
        failures = []
        for d in catalog_index["dags"]:
            try:
                load_shared_dag(d["id"])
            except Exception as exc:  # noqa: BLE001 — we want the id + reason
                failures.append(f"{d['id']}: {type(exc).__name__}: {exc}")
        assert not failures, "DAGs failed to load:\n" + "\n".join(failures)

    def test_loaded_node_edge_counts_match_index(self, catalog_index):
        """The index node_count/edge_count must match the loaded DAG."""
        mismatches = []
        for d in catalog_index["dags"]:
            dag = load_shared_dag(d["id"])
            n = len(getattr(dag, "nodes", []) or [])
            e = len(getattr(dag, "edges", []) or [])
            if "node_count" in d and n != d["node_count"]:
                mismatches.append(f"{d['id']}: nodes {n} != index {d['node_count']}")
            if "edge_count" in d and e != d["edge_count"]:
                mismatches.append(f"{d['id']}: edges {e} != index {d['edge_count']}")
        assert not mismatches, "node/edge count mismatches:\n" + "\n".join(mismatches)

    def test_every_edge_endpoint_is_a_declared_node(self):
        """Structural validity of the raw JSON: no dangling edge endpoints.

        This is the precise invariant the missing-node bug violated.
        """
        bad = []
        for p in Path(SHARED_DAGS_DIR).rglob("*.dag.json"):
            data = json.loads(p.read_text())
            dag = data.get("dag", data)
            nodes = set(dag.get("nodes", []))
            for edge in dag.get("edges", []):
                src, dst = (
                    (edge[0], edge[1])
                    if isinstance(edge, list)
                    else (
                        edge.get("src"),
                        edge.get("dst"),
                    )
                )
                for endpoint in (src, dst):
                    if endpoint not in nodes:
                        bad.append(f"{p.name}: edge endpoint '{endpoint}' not in nodes")
        assert not bad, "dangling edge endpoints:\n" + "\n".join(bad)

    def test_load_unknown_dag_raises_valueerror(self):
        with pytest.raises(ValueError):
            load_shared_dag("definitely_not_a_real_dag_id")


class TestGetAllSharedDags:
    def test_returns_entry_per_dag(self, catalog_index):
        allmeta = get_all_shared_dags()
        assert len(allmeta) == len(catalog_index["dags"])

    def test_keyed_by_id(self, dag_ids):
        allmeta = get_all_shared_dags()
        assert set(allmeta) == set(dag_ids)


class TestSearchDags:
    def test_search_by_framework_filters(self):
        results = search_dags(framework="pytorch")
        assert results, "expected at least one pytorch DAG"
        assert all(d["framework"] == "pytorch" for d in results)

    def test_search_by_features_scored_and_sorted(self):
        results = search_dags(features=["training"], framework="xgboost")
        assert results
        # every result carries a relevance score, sorted descending
        scores = [d.get("_score", 0) for d in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_no_filters_returns_all(self, catalog_index):
        results = search_dags()
        assert len(results) == len(catalog_index["dags"])

    def test_list_dags_by_framework(self):
        py = list_dags_by_framework("pytorch")
        assert py and all(d["framework"] == "pytorch" for d in py)
