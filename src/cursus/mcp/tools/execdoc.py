"""
MCP tools for the ``execdoc.*`` namespace — MODS execution-document generation.

These tools wrap :mod:`cursus.mods.exe_doc.generator`
(:class:`ExecutionDocumentGenerator`) and the JSON document helpers in
:mod:`cursus.mods.exe_doc.utils` (``create_execution_document_template``,
``validate_execution_document_structure``, ``merge_execution_documents``). DAGs are
loaded either inline (nodes + edges) or from a serialized JSON file via
``cursus.api.dag.pipeline_dag_serializer.import_dag_from_json``. ``execdoc.generate`` is
the only tool that touches engine configs/AWS; the rest are pure JSON document
build/inspect operations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..envelope import ToolResult, ToolError
from ..registry import ToolDef
from .shared import resolve_dag as _build_dag  # canonical DAG resolver


def _generate(args: Dict[str, Any]) -> ToolResult:
    """
    Fill a MODS execution document from a DAG + a config file.

    Wraps ``ExecutionDocumentGenerator(config_path, role=...).fill_execution_document``.
    Builds a DAG from ``dag_file`` or inline ``dag``, then either uses the supplied
    ``execution_document`` or auto-generates a template from the DAG node names.
    """
    config_path = args.get("config_path")
    if not config_path:
        raise ToolError("'config_path' is required", code="invalid_input")

    dag = _build_dag(args)

    execution_document = args.get("execution_document")
    auto_template = False
    if execution_document is None:
        # Auto-generate a base template from the DAG nodes so the generator has
        # PIPELINE_STEP_CONFIGS entries to fill.
        from ...mods.exe_doc.utils import create_execution_document_template

        execution_document = create_execution_document_template(list(dag.nodes))
        auto_template = True
    elif not isinstance(execution_document, dict):
        raise ToolError(
            "'execution_document' must be a JSON object", code="invalid_input"
        )

    from ...mods.exe_doc.generator import ExecutionDocumentGenerator
    from ...mods.exe_doc.base import ExecutionDocumentGenerationError

    try:
        generator = ExecutionDocumentGenerator(
            config_path=config_path,
            role=args.get("role"),
        )
    except ExecutionDocumentGenerationError as exc:
        raise ToolError(f"failed to initialize generator: {exc}", code="invalid_input")
    except FileNotFoundError as exc:
        raise ToolError(f"config file not found: {exc}", code="not_found")

    config_names = list(generator.configs.keys())

    try:
        filled = generator.fill_execution_document(dag, execution_document)
    except ExecutionDocumentGenerationError as exc:
        raise ToolError(
            f"execution document generation failed: {exc}", code="tool_error"
        )

    warnings: List[str] = []
    pipeline_configs = filled.get("PIPELINE_STEP_CONFIGS", {})
    steps_with_config = sum(
        1 for sc in pipeline_configs.values() if sc.get("STEP_CONFIG")
    )
    if not args.get("role") and args.get("execution_document") is None:
        warnings.append(
            "No IAM role/SageMaker session provided; helpers that require AWS access "
            "may produce limited step configs."
        )

    return ToolResult.success(
        filled,
        warnings=warnings or None,
        node_count=len(dag.nodes),
        config_count=len(config_names),
        config_names=config_names,
        step_count=len(pipeline_configs),
        steps_with_config=steps_with_config,
        auto_template=auto_template,
    )


def _template(args: Dict[str, Any]) -> ToolResult:
    """
    Build an empty execution-document template for the given step names.

    Wraps ``utils.create_execution_document_template(step_names)``.
    """
    step_names = args.get("step_names")
    if not isinstance(step_names, list) or not all(
        isinstance(s, str) for s in step_names
    ):
        raise ToolError("'step_names' must be a list of strings", code="invalid_input")

    from ...mods.exe_doc.utils import create_execution_document_template

    template = create_execution_document_template(list(step_names))
    return ToolResult.success(template, step_count=len(step_names))


def _validate(args: Dict[str, Any]) -> ToolResult:
    """
    Validate the structural shape of an execution document.

    Wraps ``utils.validate_execution_document_structure(doc)``. Returns
    ``{"valid": bool, "issues": [...]}``.
    """
    doc = args.get("execution_document")
    if not isinstance(doc, dict):
        return ToolResult.success(
            {"valid": False, "issues": ["execution_document must be a JSON object"]}
        )

    from ...mods.exe_doc.utils import validate_execution_document_structure

    valid = bool(validate_execution_document_structure(doc))
    issues: List[str] = []
    if not valid:
        if "PIPELINE_STEP_CONFIGS" not in doc:
            issues.append("missing required 'PIPELINE_STEP_CONFIGS' key")
        elif not isinstance(doc.get("PIPELINE_STEP_CONFIGS"), dict):
            issues.append("'PIPELINE_STEP_CONFIGS' must be an object")
        else:
            issues.append("execution document structure is invalid")

    return ToolResult.success({"valid": valid, "issues": issues})


def _merge(args: Dict[str, Any]) -> ToolResult:
    """
    Merge two execution documents (``additional`` takes precedence).

    Wraps ``utils.merge_execution_documents(base_doc, additional_doc)``.
    """
    base = args.get("base_doc")
    additional = args.get("additional_doc")
    if not isinstance(base, dict):
        raise ToolError("'base_doc' must be a JSON object", code="invalid_input")
    if not isinstance(additional, dict):
        raise ToolError("'additional_doc' must be a JSON object", code="invalid_input")

    from ...mods.exe_doc.utils import merge_execution_documents

    try:
        merged = merge_execution_documents(base, additional)
    except ValueError as exc:
        raise ToolError(str(exc), code="invalid_input")

    step_count = len(merged.get("PIPELINE_STEP_CONFIGS", {}))
    return ToolResult.success(merged, step_count=step_count)


# A JSON-Schema fragment reused for an execution-document object argument.
_EXEC_DOC_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "description": (
        "A MODS execution document: an object with a top-level "
        "'PIPELINE_STEP_CONFIGS' mapping of step name -> {STEP_TYPE, STEP_CONFIG}."
    ),
}


TOOLS: List[ToolDef] = [
    ToolDef(
        name="execdoc.generate",
        description=(
            "Fill a MODS execution document from a pipeline DAG and a config file. "
            "Provide the DAG via 'dag_file' (serialized JSON path) or inline 'dag' "
            "(nodes+edges), plus 'config_path'. Returns the filled execution document. "
            "Call when you need a runnable execution doc for a compiled pipeline."
        ),
        schema={
            "type": "object",
            "properties": {
                "config_path": {
                    "type": "string",
                    "description": "Path to the pipeline configuration JSON file to load configs from.",
                },
                "dag_file": {
                    "type": "string",
                    "description": "Path to a serialized DAG JSON file (used if 'dag' is not given).",
                },
                "dag": {
                    "type": "object",
                    "description": (
                        "Inline DAG: {'nodes': [step names], 'edges': [[src, dst], ...]}. "
                        "Used if 'dag_file' is not given."
                    ),
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Step names (DAG node ids).",
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "description": "Directed edges as [from_step, to_step] pairs.",
                        },
                    },
                },
                "execution_document": {
                    **_EXEC_DOC_SCHEMA,
                    "description": (
                        "Optional base execution document to fill. If omitted, a template is "
                        "auto-generated from the DAG node names."
                    ),
                },
                "role": {
                    "type": "string",
                    "description": (
                        "Optional IAM role ARN for AWS operations. Without it (and a SageMaker "
                        "session), some helpers may produce limited step configs."
                    ),
                },
            },
            "required": ["config_path"],
            "additionalProperties": False,
        },
        handler=_generate,
        tags=("programmer",),
    ),
    ToolDef(
        name="execdoc.template",
        description=(
            "Create an empty MODS execution-document template for a list of step names. "
            "Each step gets a default STEP_TYPE and empty STEP_CONFIG. Call to scaffold a "
            "document before filling or merging."
        ),
        schema={
            "type": "object",
            "properties": {
                "step_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Step names to include as PIPELINE_STEP_CONFIGS entries.",
                },
            },
            "required": ["step_names"],
            "additionalProperties": False,
        },
        handler=_template,
        tags=("programmer",),
    ),
    ToolDef(
        name="execdoc.validate",
        description=(
            "Validate the structure of an execution document. Returns {valid, issues}. "
            "Call to check a document has a well-formed PIPELINE_STEP_CONFIGS before "
            "generating or merging."
        ),
        schema={
            "type": "object",
            "properties": {
                "execution_document": {
                    **_EXEC_DOC_SCHEMA,
                    "description": "The execution document to validate.",
                },
            },
            "required": ["execution_document"],
            "additionalProperties": False,
        },
        handler=_validate,
        tags=("validator",),
    ),
    ToolDef(
        name="execdoc.merge",
        description=(
            "Merge two execution documents, with 'additional_doc' taking precedence over "
            "'base_doc' (STEP_CONFIG dicts are merged per step). Returns the merged document. "
            "Call to combine a base template with generated/overriding step configs."
        ),
        schema={
            "type": "object",
            "properties": {
                "base_doc": {
                    **_EXEC_DOC_SCHEMA,
                    "description": "The base execution document.",
                },
                "additional_doc": {
                    **_EXEC_DOC_SCHEMA,
                    "description": "The execution document whose values take precedence on merge.",
                },
            },
            "required": ["base_doc", "additional_doc"],
            "additionalProperties": False,
        },
        handler=_merge,
        tags=("programmer",),
    ),
]
