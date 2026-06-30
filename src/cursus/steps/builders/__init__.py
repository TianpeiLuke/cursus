"""
Step builders module.

This module contains step builder classes that create SageMaker pipeline steps
using the specification-driven architecture. Each builder is responsible for
creating a specific type of step (processing, training, etc.) and integrates
with step specifications and script contracts.

FZ 31e1d3g3 Phase B — IMPORT SURFACE. The per-step ``<Name>StepBuilder`` names are served
LAZILY via a PEP-562 module ``__getattr__`` instead of ~37 eager ``from .builder_x import X``
statements. ``__getattr__`` routes through the SAME source the runtime uses —
``StepCatalog.load_builder_class`` — which returns the physical ``builder_*.py`` class when the
file exists and the synthesized declarative shell (FZ 31e1d3g3 Phase A) when it does not. So:
- ``import cursus.steps.builders`` no longer eager-imports every builder (and no longer fails when
  an SDK-bound builder's module can't import offline — that step's name simply isn't served offline,
  exactly as the eager ``try/except`` used to leave it ``None``);
- this module names ZERO builder submodules statically, so the 45 ``builder_*.py`` files can be
  deleted (Phase E) without editing this file;
- the import surface and the discovery layer share one synthesis source, so they cannot drift.

Only the framework base class and the S3 utility are imported eagerly (they are not per-step
builders and have no ``.step.yaml`` row).
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

from ...core.base.builder_base import StepBuilderBase
from .s3_utils import S3PathHandler


def _builder_name_to_step_name() -> dict:
    """Map each registry row's ``builder_step_name`` (``<Name>StepBuilder``) to its canonical step
    name. The single authority for what builder names this package exposes — derived from the
    interface-derived registry, so it tracks the ``.step.yaml`` files (and survives their deletion).
    """
    from ...registry.step_names import get_step_names

    mapping = {}
    for step_name, info in get_step_names().items():
        builder_step_name = info.get("builder_step_name")
        if builder_step_name:
            mapping[builder_step_name] = step_name
    return mapping


def __getattr__(name: str):
    """PEP-562 lazy attribute access: resolve a ``<Name>StepBuilder`` to its class on demand.

    Routes through ``StepCatalog.load_builder_class`` (the same path the assembler uses), so it
    returns the physical builder class when ``builder_*.py`` exists and the synthesized declarative
    shell when it does not. Raises ``AttributeError`` for unknown names (the contract Python expects
    from a module ``__getattr__``) and for a builder that legitimately can't be loaded here (e.g. an
    SDK-bound builder offline) — mirroring the old eager ``try/except`` that left such names ``None``
    in ``__all__`` rather than importable.
    """
    builder_to_step = _builder_name_to_step_name()
    step_name = builder_to_step.get(name)
    if step_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from ...step_catalog.step_catalog import StepCatalog

    builder_cls = StepCatalog().load_builder_class(step_name)
    if builder_cls is None:
        # Loadable name, but not available in this environment (e.g. SDK-bound builder offline).
        raise AttributeError(
            f"{name!r} maps to step {step_name!r} but its builder could not be loaded in this "
            f"environment (e.g. the SAIS SDK is absent)."
        )
    return builder_cls


def __dir__() -> List[str]:
    """Advertise ALL known builder names in ``dir()`` / autocomplete (no resolution forced)."""
    return sorted(list(globals().keys()) + list(_builder_name_to_step_name().keys()))


def _resolvable_builder_names() -> List[str]:
    """Builder names whose class actually loads in THIS environment — the correct ``__all__`` set.

    Mirrors the legacy behavior exactly: the old ``__init__`` left SDK-bound builder names OUT of
    ``__all__`` when their module couldn't import (so ``from . import *`` skipped them). We reproduce
    that by including only names ``StepCatalog.load_builder_class`` resolves (the physical class when
    the file exists, the synthesized shell otherwise), which is the 41 native steps offline and all
    45 in the SAIS env. One cached discovery pass at import — the same work the old eager imports did.
    """
    from ...step_catalog.step_catalog import StepCatalog

    catalog = StepCatalog()
    resolvable = []
    for builder_name, step_name in _builder_name_to_step_name().items():
        try:
            if catalog.load_builder_class(step_name) is not None:
                resolvable.append(builder_name)
        except Exception:
            continue
    return sorted(resolvable)


# __all__ is derived from the registry + actual loadability so ``from cursus.steps.builders import *``
# and tooling that reads __all__ keep working AND never trip over an unavailable (e.g. offline-SDK)
# builder — exactly the legacy conditional-__all__ behavior, now data-driven. The names are still
# resolved lazily by __getattr__; this just bounds the star-import / introspection surface.
__all__ = ["StepBuilderBase", "S3PathHandler"] + _resolvable_builder_names()
