"""
Unit tests for cursus.pipeline_catalog.core.router — DAG recommendation/selection.

Covers: recommend_dag (ranked + reasoned scoring), auto_select_dag (best match
with threshold), recommend_for_agent (semantic constraint filtering).
"""


from cursus.pipeline_catalog.core.router import (
    recommend_dag,
    auto_select_dag,
    recommend_for_agent,
)


class TestRecommendDag:
    def test_returns_ranked_results(self):
        results = recommend_dag(
            framework="pytorch", features=["training"], max_results=5
        )
        assert results
        assert all("score" in r and "reasoning" in r for r in results)

    def test_sorted_descending_by_score(self):
        results = recommend_dag(framework="xgboost", features=["training"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_respects_max_results(self):
        results = recommend_dag(max_results=3)
        assert len(results) <= 3

    def test_scores_in_unit_range(self):
        for r in recommend_dag(
            framework="pytorch", features=["training", "calibration"]
        ):
            assert 0.0 <= r["score"] <= 1.0

    def test_framework_match_boosts_score(self):
        """A framework-matching DAG should outrank the no-framework baseline."""
        with_fw = recommend_dag(
            framework="xgboost", features=["training"], max_results=1
        )
        assert with_fw
        # top result should actually be xgboost (or an xgboost-id partial match)
        top = with_fw[0]
        assert (
            top["framework"] == "xgboost"
            or "xgboost" in top["id"]
            or "xgb" in top["id"]
        )

    def test_no_filters_returns_baseline_matches(self):
        results = recommend_dag()
        assert results  # baseline scoring still yields candidates


class TestAutoSelectDag:
    def test_returns_tuple_for_good_match(self):
        sel = auto_select_dag(framework="xgboost", features=["training", "calibration"])
        assert sel is not None
        dag_id, dag, score = sel
        assert isinstance(dag_id, str)
        assert score >= 0.6
        assert dag is not None  # a real loaded PipelineDAG

    def test_returns_none_for_impossible_request(self):
        sel = auto_select_dag(
            framework="nonexistent_framework",
            features=["bogus_feature_xyz"],
            min_score=0.6,
        )
        assert sel is None

    def test_min_score_threshold_enforced(self):
        # an extremely high bar should reject everything
        assert auto_select_dag(framework="pytorch", min_score=1.01) is None


class TestRecommendForAgent:
    def test_text_plus_llm_returns_llm_capable_top(self):
        results = recommend_for_agent(
            data_type="text", needs_llm=True, has_labels=False
        )
        assert results
        top = results[0]
        assert (
            top.get("input_requirements", {}).get("requires_llm")
            or "bedrock" in top["id"]
        )

    def test_tabular_returns_results(self):
        results = recommend_for_agent(
            data_type="tabular", needs_llm=False, has_labels=True
        )
        assert results

    def test_no_gpu_excludes_gpu_required_dags(self):
        results = recommend_for_agent(
            data_type="text", needs_llm=False, gpu_available=False
        )
        assert not any(r.get("constraints", {}).get("requires_gpu") for r in results), (
            "gpu_available=False must exclude GPU-required DAGs"
        )

    def test_results_scored_and_sorted(self):
        results = recommend_for_agent(data_type="text", needs_llm=True)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_caps_at_five_results(self):
        results = recommend_for_agent(data_type="mixed")
        assert len(results) <= 5
