"""
Factory functions for creating pipeline dependency components.

This module provides convenience functions for instantiating the core
components of the dependency resolution system with proper wiring.
"""

from typing import Optional
from .semantic_matcher import SemanticMatcher
from .specification_registry import SpecificationRegistry
from .registry_manager import RegistryManager
from .dependency_resolver import UnifiedDependencyResolver, create_dependency_resolver


def create_pipeline_components(context_name: Optional[str] = None) -> dict:
    """Create all necessary pipeline components with proper dependencies."""
    semantic_matcher = SemanticMatcher()
    registry_manager = RegistryManager()
    registry = registry_manager.get_registry(context_name or "default")
    resolver = UnifiedDependencyResolver(registry, semantic_matcher)

    return {
        "semantic_matcher": semantic_matcher,
        "registry_manager": registry_manager,
        "registry": registry,
        "resolver": resolver,
    }


import threading
from contextlib import contextmanager
from typing import Generator, Dict, Any

# Thread-local storage for per-thread instances
_thread_local = threading.local()


def get_thread_components() -> dict:
    """Get thread-specific component instances."""
    if not hasattr(_thread_local, "components"):
        _thread_local.components = create_pipeline_components()
    from typing import cast
    return cast(dict, _thread_local.components)


@contextmanager
def dependency_resolution_context(clear_on_exit: bool = True) -> Generator[Dict[str, Any], None, None]:
    """Create a scoped dependency resolution context."""
    components = create_pipeline_components()
    try:
        yield components
    finally:
        if clear_on_exit:
            components["resolver"].clear_cache()
            components["registry_manager"].clear_all_contexts()


__all__ = [
    "create_pipeline_components",
    "create_dependency_resolver",
    "dependency_resolution_context",
    "get_thread_components",
]
