"""
Tests for the cursus.mcp registry: ToolDef, registry assembly, the call_tool invoker
(schema validation + error wrapping), and the OpenAI/MCP exporters (incl. tags).
"""

import pytest

from cursus.mcp import registry as reg_mod
from cursus.mcp.registry import (
    ToolDef,
    call_tool,
    get_registry,
    get_namespaces,
    list_tools,
    get_tool,
    render_description,
    export_openai_tools,
    export_mcp_tools,
)
from cursus.mcp.envelope import ToolResult, ToolError


# --- a small fake registry installed for invoker/exporter tests ------------------


def _ok_handler(args):
    return ToolResult.success({"echo": args})


def _raises_toolerror(args):
    raise ToolError("known bad", code="not_found", details={"k": "v"})


def _raises_unexpected(args):
    raise RuntimeError("kaboom")


def _returns_raw(args):
    # Not a ToolResult — the invoker must normalize it.
    return {"raw": True}


FAKE_TOOLS = {
    "demo.ok": ToolDef(
        name="demo.ok",
        description="an ok tool for tests",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "mode": {"type": "string", "enum": ["a", "b"]},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        handler=_ok_handler,
        tags=("planner",),
    ),
    "demo.boom": ToolDef(
        name="demo.boom",
        description="raises a ToolError",
        schema={"type": "object", "properties": {}, "required": []},
        handler=_raises_toolerror,
        tags=("validator",),
    ),
    "demo.crash": ToolDef(
        name="demo.crash",
        description="raises an unexpected exception",
        schema={"type": "object", "properties": {}, "required": []},
        handler=_raises_unexpected,
    ),
    "demo.raw": ToolDef(
        name="demo.raw",
        description="returns a raw dict",
        schema={"type": "object", "properties": {}, "required": []},
        handler=_returns_raw,
    ),
    "demo.rich": ToolDef(
        name="demo.rich",
        description="a tool carrying when + examples",
        schema={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": [],
        },
        handler=_ok_handler,
        tags=("planner",),
        when="You need the demo path with usage guidance.",
        examples=('demo.rich {"x": "y"}  # sets x', "demo.rich {}  # defaults"),
    ),
}


@pytest.fixture
def fake_registry(monkeypatch):
    """Install FAKE_TOOLS as the active registry for the duration of a test."""
    monkeypatch.setattr(reg_mod, "_REGISTRY", dict(FAKE_TOOLS))
    yield
    # monkeypatch restores _REGISTRY automatically.


class TestToolDef:
    def test_namespace_derived_from_name(self):
        assert FAKE_TOOLS["demo.ok"].namespace == "demo"

    def test_defaults(self):
        td = FAKE_TOOLS["demo.crash"]
        assert td.destructive is False
        assert td.tags == ()
        # new optional detail fields default to empty
        assert td.when == ""
        assert td.examples == ()

    def test_when_and_examples_carried(self):
        td = FAKE_TOOLS["demo.rich"]
        assert td.when
        assert isinstance(td.examples, tuple) and len(td.examples) == 2

    def test_examples_is_immutable_tuple(self):
        # ToolDef is frozen; examples is a tuple so it is truly immutable (a list field
        # could still be mutated in place). No reassignment is allowed either.
        td = FAKE_TOOLS["demo.rich"]
        assert isinstance(td.examples, tuple)
        with pytest.raises((AttributeError, TypeError)):
            td.examples = ("x",)  # frozen: cannot reassign


class TestRenderDescription:
    def test_folds_when_and_examples(self):
        text = render_description(FAKE_TOOLS["demo.rich"])
        assert "a tool carrying when + examples" in text
        assert "When: You need the demo path" in text
        assert "Examples:" in text
        assert 'demo.rich {"x": "y"}' in text

    def test_plain_tool_is_just_description(self):
        td = FAKE_TOOLS["demo.crash"]
        assert render_description(td) == td.description


class TestCallToolValidation:
    def test_unknown_tool(self, fake_registry):
        r = call_tool("does.not.exist", {})
        assert not r.ok
        assert r.code == "unknown_tool"

    def test_missing_required_arg(self, fake_registry):
        r = call_tool("demo.ok", {})  # 'name' required
        assert not r.ok
        assert r.code == "invalid_input"
        assert "name" in r.error

    def test_unknown_arg_rejected(self, fake_registry):
        r = call_tool("demo.ok", {"name": "x", "bogus": 1})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_enum_violation(self, fake_registry):
        r = call_tool("demo.ok", {"name": "x", "mode": "z"})
        assert not r.ok
        assert r.code == "invalid_input"

    def test_happy_path_passes_args(self, fake_registry):
        r = call_tool("demo.ok", {"name": "x", "mode": "a"})
        assert r.ok
        assert r.data["echo"] == {"name": "x", "mode": "a"}
        # invoker stamps the tool name into meta
        assert r.meta.get("tool") == "demo.ok"


class TestCallToolErrorHandling:
    def test_toolerror_converted(self, fake_registry):
        r = call_tool("demo.boom", {})
        assert not r.ok
        assert r.code == "not_found"
        assert r.meta.get("details") == {"k": "v"}

    def test_unexpected_exception_wrapped(self, fake_registry):
        r = call_tool("demo.crash", {})
        assert not r.ok
        assert r.code == "internal_error"  # never escapes as a raw exception

    def test_raw_return_normalized(self, fake_registry):
        r = call_tool("demo.raw", {})
        assert r.ok
        assert r.data == {"raw": True}

    def test_none_args_ok(self, fake_registry):
        # call_tool(name) with no args must not crash on required-key checks.
        r = call_tool("demo.boom")  # demo.boom requires nothing
        assert not r.ok and r.code == "not_found"


class TestExporters:
    def test_openai_shape_and_tags(self, fake_registry):
        tools = export_openai_tools()
        ok = next(t for t in tools if t["function"]["name"] == "demo.ok")
        assert ok["type"] == "function"
        assert "parameters" in ok["function"]
        assert ok["function"]["metadata"]["tags"] == ["planner"]
        # a tool with no tags carries no metadata key
        crash = next(t for t in tools if t["function"]["name"] == "demo.crash")
        assert "metadata" not in crash["function"]

    def test_mcp_shape_and_tags(self, fake_registry):
        tools = export_mcp_tools()
        ok = next(t for t in tools if t["name"] == "demo.ok")
        assert "inputSchema" in ok
        assert ok["tags"] == ["planner"]
        crash = next(t for t in tools if t["name"] == "demo.crash")
        assert "tags" not in crash

    def test_namespace_filter(self, fake_registry):
        names = {t["name"] for t in export_mcp_tools(namespace="demo")}
        assert names == set(FAKE_TOOLS)

    def test_exporters_fold_when_and_examples_into_description(self, fake_registry):
        # External clients only get a description string — when/examples must be folded in.
        oai = next(
            t for t in export_openai_tools() if t["function"]["name"] == "demo.rich"
        )
        assert "When:" in oai["function"]["description"]
        assert "Examples:" in oai["function"]["description"]
        mcp = next(t for t in export_mcp_tools() if t["name"] == "demo.rich")
        assert 'demo.rich {"x": "y"}' in mcp["description"]


class TestRealRegistry:
    """Smoke checks against the actual assembled registry (no mocking)."""

    def test_all_seven_plus_namespaces_present(self):
        reg = get_registry(force_reload=True)
        namespaces = {td.namespace for td in reg.values()}
        for ns in (
            "catalog",
            "dag",
            "config",
            "compile",
            "validate",
            "execdoc",
            "pipeline_catalog",
            "tools",
        ):
            assert ns in namespaces, f"namespace {ns} missing from registry"

    def test_no_duplicate_names_and_valid_defs(self):
        reg = get_registry()
        for name, td in reg.items():
            assert td.name == name
            assert callable(td.handler)
            assert td.schema.get("type") == "object"
            assert len(td.description) >= 10

    def test_meta_namespace_tools_present(self):
        reg = get_registry()
        assert "tools.by_phase" in reg
        assert "tools.describe_tool" in reg
        assert "tools.help" in reg

    def test_get_tool_and_list_tools(self):
        assert get_tool("tools.by_phase") is not None
        planner = list_tools(namespace="tools")
        assert all(td.namespace == "tools" for td in planner)

    def test_namespaces_map_populated(self):
        namespaces = get_namespaces()
        # Every non-meta namespace declares a description via its module's NAMESPACE const.
        for ns in ("catalog", "dag", "config", "compile", "validate", "execdoc"):
            assert ns in namespaces and namespaces[ns]

    def test_auto_generated_namespace_help_tools(self):
        reg = get_registry(force_reload=True)
        # Each namespace that declared a description gets an auto <ns>.help tool …
        for ns in get_namespaces():
            if ns == "tools":  # meta namespace keeps its hand-written help
                continue
            assert f"{ns}.help" in reg, f"missing auto help for {ns}"
        # … and calling one returns that namespace's slice.
        r = call_tool("compile.help", {})
        assert r.ok
        assert r.data["filtered_namespace"] == "compile"

    def test_no_auto_help_for_meta_namespace(self):
        # tools.help is hand-written and global; there must be exactly one 'tools.help'
        # and no separately generated collision.
        reg = get_registry()
        assert reg["tools.help"].handler.__name__ == "_help"
