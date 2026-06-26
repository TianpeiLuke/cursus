"""
Unit tests for cursus.pipeline_catalog.core.agent_tool — the LLM tool interface.

Covers TOOL_SCHEMA validity and every pipeline_catalog_tool(action=...) branch
returning the documented structured response.
"""


from cursus.pipeline_catalog.core.agent_tool import pipeline_catalog_tool, TOOL_SCHEMA


class TestToolSchema:
    def test_schema_name(self):
        assert TOOL_SCHEMA["name"] == "pipeline_catalog"

    def test_schema_has_parameters(self):
        assert "parameters" in TOOL_SCHEMA
        assert "action" in TOOL_SCHEMA["parameters"]["properties"]

    def test_action_enum_complete(self):
        actions = set(TOOL_SCHEMA["parameters"]["properties"]["action"]["enum"])
        assert actions == {
            "recommend",
            "get_dag",
            "get_config_guidance",
            "list_frameworks",
            "list_features",
        }


class TestRecommendAction:
    def test_recommend_returns_success(self):
        res = pipeline_catalog_tool(
            action="recommend", data_type="text", needs_llm=True
        )
        assert res["status"] == "success"
        assert res["action"] == "recommend"
        assert "recommendations" in res
        assert isinstance(res["recommendations"], list)

    def test_recommend_entries_have_expected_fields(self):
        res = pipeline_catalog_tool(
            action="recommend", data_type="text", needs_llm=True
        )
        for rec in res["recommendations"]:
            assert {"dag_id", "score", "framework"} <= set(rec)

    def test_recommend_framework_filter(self):
        res = pipeline_catalog_tool(
            action="recommend",
            data_type="tabular",
            needs_llm=False,
            framework="xgboost",
        )
        assert res["status"] == "success"
        # A requested framework is a hard filter: every recommendation must match it,
        # and the result must not fall back to unrelated frameworks.
        recs = res["recommendations"]
        assert recs, "expected at least one xgboost recommendation"
        assert all(r["framework"] == "xgboost" for r in recs)


class TestGetDagAction:
    def test_get_dag_returns_nodes_and_edges(self, dag_ids):
        res = pipeline_catalog_tool(action="get_dag", dag_id=dag_ids[0])
        assert res["status"] == "success"
        assert "nodes" in res and "edges" in res
        assert isinstance(res["nodes"], list) and len(res["nodes"]) > 0

    def test_get_dag_unknown_id_returns_error(self):
        res = pipeline_catalog_tool(action="get_dag", dag_id="not_a_real_dag")
        assert res["status"] == "error"

    def test_get_dag_for_every_catalogued_dag(self, dag_ids):
        """Smoke: get_dag must succeed for all catalogued ids (no stale paths)."""
        failures = [
            d
            for d in dag_ids
            if pipeline_catalog_tool(action="get_dag", dag_id=d)["status"] != "success"
        ]
        assert not failures, f"get_dag failed for: {failures}"


class TestGetConfigGuidanceAction:
    def test_returns_guidance_fields(self, dag_ids):
        res = pipeline_catalog_tool(action="get_config_guidance", dag_id=dag_ids[0])
        assert res["status"] == "success"
        assert {"prerequisites", "config_guidance", "next_step"} <= set(res)

    def test_unknown_id_returns_error(self):
        res = pipeline_catalog_tool(action="get_config_guidance", dag_id="nope")
        assert res["status"] == "error"


class TestListActions:
    def test_list_frameworks(self):
        res = pipeline_catalog_tool(action="list_frameworks")
        assert res["status"] == "success"
        assert isinstance(res["frameworks"], dict)
        assert sum(res["frameworks"].values()) > 0

    def test_list_features(self):
        res = pipeline_catalog_tool(action="list_features")
        assert res["status"] == "success"
        assert isinstance(res["features"], list)


class TestUnknownAction:
    def test_unknown_action_returns_error(self):
        res = pipeline_catalog_tool(action="bogus_action")
        assert res["status"] == "error"
