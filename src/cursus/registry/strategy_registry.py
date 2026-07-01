"""
Strategy registry — the single source of truth for the builder strategy library.

This is a dependency-free leaf (imports only stdlib at module top) that maps a routing axis +
name to a ``StrategyInfo`` (the construction-verb handler class + its declarative knobs). Both
the runtime router (``builder_templates.resolve_handler``) and the introspection tool
(``cursus strategies`` CLI / ``strategies`` MCP) read from this one registry, so the tool can
never drift from what the builder actually does.

Routing axes:
  * ``sagemaker_step_type`` — Training / CreateModel / Transform / CradleDataLoading /
    RedshiftDataLoading / MimsModelRegistrationProcessing (and the no-builder rows Base / Lambda).
  * ``step_assembly`` — code / step_args / delegation (the Processing sub-discriminator).

Handlers self-register via ``@register_strategy(...)`` in ``core.base.builder_templates``; the
``ensure_strategies_loaded()`` lazy import triggers those decorations on first read. The registry
imports NOTHING from ``core.base`` at module top (only a ``TYPE_CHECKING`` hint), so it stays a
leaf and there is no import cycle with ``builder_templates`` (which imports from here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # type hint only — never a runtime import edge
    from ..core.base.builder_templates import PatternHandler


class NoBuilderError(ValueError):
    """Raised when an (axis, name) has no routable strategy (abstract / builder-less type)."""


@dataclass(frozen=True)
class KnobSpec:
    """Describes one tunable knob a handler accepts."""

    name: str
    type: str = (
        "str"  # "bool" | "str" | "list" | "callable" — a string tag, import-light
    )
    default: Any = None
    required: bool = False
    doc: str = ""


@dataclass(frozen=True)
class StrategyInfo:
    """A registry row: a routing (axis, name) -> a handler class + its knobs."""

    axis: str  # "sagemaker_step_type" | "step_assembly"
    name: str  # the value on that axis that selects this strategy
    handler: Optional[
        Callable[..., "PatternHandler"]
    ]  # handler CLASS (None when routable=False)
    knobs: Tuple[KnobSpec, ...] = ()
    preset_knobs: Dict[str, Any] = field(default_factory=dict)
    routable: bool = True
    verb: str = ""
    implemented: bool = True


_REGISTRY: Dict[Tuple[str, str], StrategyInfo] = {}
_LOADED = False


def register_strategy(
    *,
    axis: str,
    name: str,
    knobs: Tuple[KnobSpec, ...] = (),
    preset_knobs: Optional[Dict[str, Any]] = None,
    routable: bool = True,
    verb: str = "",
    implemented: bool = True,
) -> Callable[[type], type]:
    """Decorator: register a handler class under (axis, name). Returns the class unchanged."""

    def deco(handler_cls: type) -> type:
        info = StrategyInfo(
            axis=axis,
            name=name,
            handler=handler_cls,
            knobs=tuple(knobs),
            preset_knobs=dict(preset_knobs or {}),
            routable=routable,
            verb=verb,
            implemented=implemented,
        )
        key = (axis, name)
        existing = _REGISTRY.get(key)
        if existing is not None and existing.handler is not handler_cls:
            raise ValueError(f"duplicate strategy registration for {key}")
        _REGISTRY[key] = info
        return handler_cls

    return deco


def register_no_builder(*, axis: str, name: str, verb: str = "") -> None:
    """Register a non-routable row (e.g. Base / Lambda — abstract or builder-less types)."""
    _REGISTRY[(axis, name)] = StrategyInfo(
        axis=axis, name=name, handler=None, routable=False, verb=verb, implemented=False
    )


def ensure_strategies_loaded() -> None:
    """Import the handler module so its ``@register_strategy`` decorations execute (Edge B).

    The only heavy import in this module; guarded + lazy so the registry stays a leaf.
    """
    global _LOADED
    if _LOADED:
        return
    _LOADED = True  # set BEFORE the import to prevent re-entrancy during builder_templates load
    import cursus.core.base.builder_templates  # noqa: F401


def resolve_strategy(axis: str, name: str) -> StrategyInfo:
    """Return the routable StrategyInfo for (axis, name), else raise NoBuilderError."""
    ensure_strategies_loaded()
    info = _REGISTRY.get((axis, name))
    if info is None or not info.routable:
        raise NoBuilderError(
            f"{name!r} on axis {axis!r} is not a routable construction strategy"
        )
    return info


def list_strategies(axis: Optional[str] = None) -> List[StrategyInfo]:
    """All registered strategies, optionally filtered by axis."""
    ensure_strategies_loaded()
    return [i for i in _REGISTRY.values() if axis is None or i.axis == axis]


def knobs_for(axis: str, name: str) -> Tuple[KnobSpec, ...]:
    """The declarative knobs the strategy at (axis, name) accepts."""
    return resolve_strategy(axis, name).knobs


def axes() -> List[str]:
    """The routing axes that have registered strategies."""
    ensure_strategies_loaded()
    return sorted({i.axis for i in _REGISTRY.values()})


def axis_name_for_step_type(
    sagemaker_step_type: str, step_assembly: Optional[str] = None
) -> Tuple[str, str]:
    """Map a step's (sagemaker_step_type, step_assembly) onto the registry ``(axis, name)`` key.

    This is the single source of the routing rule — ``builder_templates.resolve_handler`` calls it
    to bind the handler at build time, and the introspection tool calls it for ``for_step_type`` —
    so the tool's "what would this step bind?" answer can never diverge from the actual router.

    Routing is by ``sagemaker_step_type`` ONLY (never by step name); ``Processing`` is the one type
    sub-discriminated by ``step_assembly`` (``code`` | ``step_args`` | ``delegation``, default
    ``code``).
    """
    if sagemaker_step_type == "Processing":
        return "step_assembly", (step_assembly or "code")
    return "sagemaker_step_type", sagemaker_step_type


# ---------------------------------------------------------------------------
# JSON-safe serialization (the single render the CLI + MCP introspection tool share)
# ---------------------------------------------------------------------------


def _jsonsafe(value: Any) -> Any:
    """Render a knob default / preset value to something JSON-serializable.

    Callables (factory defaults, handler classes) become a readable name string; primitives and
    lists/dicts pass through. Keeps the introspection surface printable without importing the heavy
    handler module's types.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonsafe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonsafe(v) for k, v in value.items()}
    if callable(value):
        return getattr(value, "__name__", None) or repr(value)
    return str(value)


def knob_to_dict(knob: KnobSpec) -> Dict[str, Any]:
    """A KnobSpec as a plain, JSON-serializable dict."""
    return {
        "name": knob.name,
        "type": knob.type,
        "default": _jsonsafe(knob.default),
        "required": knob.required,
        "doc": knob.doc,
    }


def strategy_to_dict(info: StrategyInfo) -> Dict[str, Any]:
    """A StrategyInfo as a plain, JSON-serializable dict (handler rendered as its class name)."""
    return {
        "axis": info.axis,
        "name": info.name,
        "verb": info.verb,
        "handler": getattr(info.handler, "__name__", None),
        "routable": info.routable,
        "implemented": info.implemented,
        "knobs": [knob_to_dict(k) for k in info.knobs],
        "preset_knobs": {k: _jsonsafe(v) for k, v in info.preset_knobs.items()},
    }


def find_strategies(name: str, axis: Optional[str] = None) -> List[StrategyInfo]:
    """All registered strategies whose ``name`` matches, optionally constrained to one ``axis``.

    Names are unique within an axis but a bare name may (in principle) appear on more than one axis,
    so callers that need exactly one row should disambiguate with ``axis``.
    """
    ensure_strategies_loaded()
    return [
        i
        for i in _REGISTRY.values()
        if i.name == name and (axis is None or i.axis == axis)
    ]
