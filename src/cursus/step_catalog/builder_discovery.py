"""
Builder class discovery via registry-interface synthesis.

Design B "Phase E": file-based (package/workspace) builder discovery has been removed. Builders are
SYNTHESIZED from the registry interface — for each registry step that has a ``.step.yaml`` interface
and routes, a per-step ``TemplateStepBuilder`` subclass is fabricated at runtime. There is no longer
any AST scan of ``builder_*.py`` files, no descriptors, and no per-workspace precedence.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

logger = logging.getLogger(__name__)


class BuilderAutoDiscovery:
    """
    Builder class discovery via registry-interface synthesis.

    Builders are synthesized from the registry interface rather than discovered from files.
    """

    def __init__(self, package_root: Path, workspace_dirs: Optional[List[Path]] = None):
        """
        Initialize builder discovery.

        Args:
            package_root: Root directory of the cursus package
            workspace_dirs: Optional list of workspace directories (retained for signature
                compatibility; no longer used for file-based discovery)
        """
        # Initialize logger FIRST before any other operations
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"🔧 BuilderAutoDiscovery.__init__ starting - package_root: {package_root}"
        )
        self.logger.info(
            f"🔧 BuilderAutoDiscovery.__init__ - workspace_dirs: {workspace_dirs}"
        )

        try:
            # Handle sys.path setup internally for deployment portability
            self.logger.debug("🔧 Calling _ensure_cursus_importable()")
            self._ensure_cursus_importable()
            self.logger.debug("✅ _ensure_cursus_importable() completed successfully")
        except Exception as e:
            self.logger.error(f"❌ _ensure_cursus_importable() failed: {e}")
            raise

        self.package_root = package_root
        self.workspace_dirs = workspace_dirs or []
        self.logger.info("✅ BuilderAutoDiscovery basic initialization complete")

        # Materialized-class cache: step_name -> class (load cache).
        self._builder_cache: Dict[str, Type] = {}
        self._discovery_complete = False

        # Registry-walk materializer cache (FZ 31e1d3g3 Phase A): step_name -> synthesized
        # TemplateStepBuilder subclass for a step that has a .step.yaml interface but NO physical
        # builder_*.py file. Keyed by step_name so a step always maps to the SAME class object
        # within a process (identity stability — pickling / class-keyed caches; OQ 31e1d3g3a).
        self._synthesized_builders: Dict[str, Type] = {}

        # Registry integration
        self._registry_info: Dict[str, Dict[str, Any]] = {}
        # canonical_key(step_name) -> step_name, for casing/separator-robust name resolution.
        self._normalized_registry_index: Dict[str, str] = {}

        try:
            self.logger.debug("🔧 Loading registry info...")
            self._load_registry_info()
            self.logger.info(
                f"✅ Registry info loaded: {len(self._registry_info)} steps"
            )
        except Exception as e:
            self.logger.error(f"❌ Registry info loading failed: {e}")
            self._registry_info = {}

        self.logger.info(
            "🎉 BuilderAutoDiscovery initialization completed successfully"
        )

    def _ensure_cursus_importable(self):
        """
        Internal sys.path setup for deployment portability.

        This handles the importlib deployment issue internally so consumers
        don't need to worry about it.
        """
        current_file = Path(__file__).resolve()
        current_path = current_file
        while current_path.parent != current_path:
            if current_path.name == "cursus":
                cursus_parent = str(current_path.parent)
                if cursus_parent not in sys.path:
                    sys.path.insert(0, cursus_parent)
                    self.logger.debug(
                        f"Added cursus parent to sys.path: {cursus_parent}"
                    )
                break
            current_path = current_path.parent

    def _load_registry_info(self):
        """
        Load registry information from cursus/registry/step_names.py.

        This provides authoritative information about step names, builder classes,
        and other metadata that drives the synthesis process.
        """
        try:
            from ..registry.step_names import get_step_names

            step_names_dict = get_step_names()
            for step_name, step_info in step_names_dict.items():
                self._registry_info[step_name] = step_info

            self._rebuild_normalized_index()

            self.logger.debug(
                f"Loaded registry info for {len(self._registry_info)} steps"
            )

        except ImportError as e:
            self.logger.warning(f"Could not import registry step_names: {e}")
            self._registry_info = {}
        except Exception as e:
            self.logger.error(f"Error loading registry info: {e}")
            self._registry_info = {}

    def _rebuild_normalized_index(self):
        """(Re)build the case/separator-insensitive index of registry keys.

        Maps ``canonical_key(step_name) -> step_name`` for every registry row so a lookup can
        resolve robustly regardless of compound-acronym casing. This is the interface-derived
        acronym-deduction the ``COMPOUND_ACRONYMS`` table can't guarantee: the registry keys ARE
        the authored ``.step.yaml`` ``step_type`` values, so ``canonical_key`` collapses e.g.
        ``XgboostMtTraining`` and ``XGBoostMTTraining`` to one key and both resolve to the real row
        — no table edit needed when a new acronym step ships. Ambiguous collisions (two rows
        collapsing to the same key) keep the first-seen row and are logged, never silently merged.
        """
        from .naming import canonical_key

        self._normalized_registry_index = {}
        for step_name in self._registry_info:
            key = canonical_key(step_name)
            if key in self._normalized_registry_index:
                self.logger.debug(
                    f"Normalized-key collision for {key!r}: keeping "
                    f"{self._normalized_registry_index[key]!r}, ignoring {step_name!r}"
                )
                continue
            self._normalized_registry_index[key] = step_name

    def _resolve_registry_key(self, step_name: str) -> Optional[str]:
        """Resolve a possibly-non-canonical step name to its actual registry key.

        Exact match wins (fast path); otherwise fall back to the normalized (case/separator-
        insensitive) index so ``XGBoostMTTraining`` resolves to the registry's ``XgboostMtTraining``
        row (and vice-versa). Returns None when neither resolves.
        """
        if step_name in self._registry_info:
            return step_name
        from .naming import canonical_key

        return getattr(self, "_normalized_registry_index", {}).get(
            canonical_key(step_name)
        )

    def _get_registry_builder_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get builder information from registry for a step.

        Args:
            step_name: Name of the step

        Returns:
            Dictionary with builder information from registry or None
        """
        if step_name in self._registry_info:
            step_info = self._registry_info[step_name]
            return {
                "builder_step_name": step_info.get("builder_step_name"),
                "sagemaker_step_type": step_info.get("sagemaker_step_type"),
                "step_type": step_info.get("step_type"),
                "module_path": step_info.get("module_path"),
                "class_name": step_info.get("class_name"),
            }
        return None

    def discover_builder_classes(
        self, project_id: Optional[str] = None
    ) -> Dict[str, Type]:
        """
        Discover all builder classes by synthesizing them from the registry interface.

        Args:
            project_id: Optional project ID (retained for signature compatibility; synthesis
                is registry-wide and has no per-workspace filtering)

        Returns:
            Dictionary mapping step names to builder class types
        """
        if not self._discovery_complete:
            self._run_discovery()

        # Design B "Phase E": builders are SYNTHESIZED from the registry interface — no file-based
        # discovery. Walk every registry row and synthesize its declarative TemplateStepBuilder
        # shell; steps that don't route (abstract rows, SDK-delegation offline) yield None and are
        # excluded — preserving the legacy contract that this map holds only loadable classes.
        all_builders: Dict[str, Type] = {}
        for step_name in self._registry_info:
            synthesized = self._synthesize_builder(step_name)
            if synthesized is not None:
                all_builders[step_name] = synthesized

        self.logger.info(f"Discovered {len(all_builders)} builder classes")
        return all_builders

    def load_builder_class(self, step_name: str) -> Optional[Type]:
        """
        Load builder class for a specific step by synthesizing from the registry interface.

        Args:
            step_name: Name of the step to load builder for

        Returns:
            Builder class type or None if not found
        """
        # Check the materialized-class cache first
        if step_name in self._builder_cache:
            return self._builder_cache[step_name]

        # Design B "Phase E": synthesize a TemplateStepBuilder subclass for a step that has a
        # .step.yaml interface and routes. Returns None for steps that don't (abstract rows,
        # SDK-delegation offline, unknown) — preserving the prior "unloadable builder is skipped"
        # behavior.
        synthesized = self._synthesize_builder(step_name)
        if synthesized is not None:
            self._builder_cache[step_name] = synthesized
            self.logger.debug(
                f"Synthesized builder for {step_name} from registry interface"
            )
            return synthesized

        # Not found / not loadable - expected for many step variants + offline SDK builders.
        self.logger.debug(f"No builder class found for step: {step_name}")
        return None

    def _synthesize_builder(self, step_name: str) -> Optional[Type]:
        """Synthesize a per-step builder class for a fileless, interface-having, routable step.

        FZ 31e1d3g3 Phase A — the deletion mechanism for the 45 shells. A shell is nothing but
        ``class XStepBuilder(TemplateStepBuilder): STEP_NAME = "X"``; this fabricates exactly that
        at runtime so a step needs no physical ``builder_*.py``.

        Returns None (so the caller keeps today's "no builder" behavior) when the step:
        - has no ``.step.yaml`` interface (e.g. the abstract ``Base``/``Processing``/``HyperparameterPrep``
          registry rows), or
        - does not route — its ``sagemaker_step_type`` (+ ``patterns.step_assembly``) is not a routable
          construction strategy (``resolve_handler`` raises ``NoBuilderError`` for ``Base``/``Lambda``/
          unknown), or
        - is an SDK-delegation step that cannot be materialized here (the ``sdk_step_class`` knob is a
          live SAIS class reference, injected by the Phase-A2 ``sdk_bindings`` path, not this generic
          synthesizer).

        The synthesized class is cached per ``step_name`` so identity is stable within a process.
        """
        # Resolve the requested name to its actual registry key (robust to compound-acronym
        # casing: XGBoostMTTraining <-> XgboostMtTraining). All caching + interface loading below
        # then uses the canonical key, so both spellings share one synthesized class.
        resolved = self._resolve_registry_key(step_name)
        if resolved is None:
            return None
        step_name = resolved

        if step_name in self._synthesized_builders:
            return self._synthesized_builders[step_name]

        # 1) Must have a .step.yaml interface. Use the registry's sagemaker_step_type as the route key.
        info = self._registry_info.get(step_name)
        if not info:
            return None
        sagemaker_step_type = info.get("sagemaker_step_type")
        if not sagemaker_step_type:
            return None

        try:
            from ..steps.interfaces import load_interface
            from ..core.base.builder_templates import (
                TemplateStepBuilder,
                resolve_handler,
                NoBuilderError,
            )
        except Exception as e:  # pragma: no cover - import wiring
            self.logger.debug(f"Synthesizer imports unavailable for {step_name}: {e}")
            return None

        # 2) Must have a loadable interface (raises FileNotFoundError for fileless registry rows).
        try:
            iface = load_interface(step_name)
        except Exception as e:
            self.logger.debug(f"No loadable interface for {step_name}: {e}")
            return None

        # 3) SDK-delegation steps need a live SAIS *Step class injected as the sdk_step_class knob
        #    (genuine code, not serializable to YAML). The carve-out is authored data — registry.requires
        #    names the build-time 3rd-party dep — NOT a hardcoded step list, so it tracks the .step.yaml.
        #    All 4 (Cradle/Redshift/Registration/DataUploading) declare requires=...sdk. Materialize the
        #    SAIS class via the lazy sdk_bindings thunk; OFFLINE the import fails → return None, keeping
        #    them undiscoverable (matching the closure gate's _SDK_DELEGATION_STEPS carve-out).
        registry_section = getattr(iface, "registry", None)
        requires = (
            getattr(registry_section, "requires", "none")
            if registry_section
            else "none"
        )
        extra_attrs: Dict[str, Any] = {}
        if requires == "secure_ai_sandbox_workflow_python_sdk":
            from .sdk_bindings import is_sdk_delegation_step, resolve_sdk_step_class

            if not is_sdk_delegation_step(step_name):
                self.logger.debug(
                    f"{step_name} requires the SAIS SDK but has no sdk_bindings entry; not synthesizing"
                )
                return None
            try:
                sdk_step_class = resolve_sdk_step_class(step_name)
            except Exception as e:
                # SAIS SDK absent (offline) — preserve "undiscoverable offline" behavior.
                self.logger.debug(
                    f"SAIS SDK unavailable for {step_name}; not synthesizing: {e}"
                )
                return None
            # Mirror the hand-written SDK shells' HANDLER_KNOBS so _auto_bind_handler injects the class.
            extra_attrs["HANDLER_KNOBS"] = {"sdk_step_class": sdk_step_class}

        # 4) Must route. resolve_handler raises NoBuilderError for non-routable rows (Base/Lambda).
        patterns = getattr(iface, "patterns", None)
        step_assembly = getattr(patterns, "step_assembly", None) if patterns else None
        try:
            resolve_handler(sagemaker_step_type, step_assembly)
        except NoBuilderError:
            self.logger.debug(
                f"Step {step_name} ({sagemaker_step_type}/{step_assembly}) is not routable; not synthesizing"
            )
            return None
        except Exception as e:
            self.logger.debug(f"Handler resolution failed for {step_name}: {e}")
            return None

        # All checks pass — fabricate the 2-line shell as a real subclass. A subclass (not a partial)
        # preserves .__name__, satisfies issubclass/__mro__, and lets __init__ read self.STEP_NAME.
        synthesized = type(
            f"{step_name}StepBuilder",
            (TemplateStepBuilder,),
            {
                "STEP_NAME": step_name,
                "__doc__": f"Synthesized declarative shell for {step_name}.",
                **extra_attrs,
            },
        )
        self._synthesized_builders[step_name] = synthesized
        return synthesized

    def _run_discovery(self):
        """Mark discovery complete.

        Design B "Phase E": there is no file-based discovery pass. Builders are synthesized lazily
        from the registry interface on demand (load_builder_class / discover_builder_classes), so
        this only flips the completion flag that gates those entry points.
        """
        self._discovery_complete = True
        self.logger.debug("Builder discovery ready (registry-interface synthesis)")

    def get_builder_info(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a builder.

        Args:
            step_name: Name of the step

        Returns:
            Dictionary with builder information or None if not found
        """
        builder_class = self.load_builder_class(step_name)
        if not builder_class:
            return None

        return {
            "step_name": step_name,
            "builder_class": builder_class.__name__,
            "module": builder_class.__module__,
        }

    def list_available_builders(self) -> List[str]:
        """
        List all available builder step names.

        Returns:
            List of step names that have (synthesizable) builders
        """
        return sorted(self.discover_builder_classes().keys())

    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        Get discovery statistics.

        Returns:
            Dictionary with discovery statistics
        """
        discovered = self.discover_builder_classes()

        return {
            "synthesized_builders": len(discovered),
            "total_builders": len(discovered),
            "cached_builders": len(self._builder_cache),
            "discovery_complete": self._discovery_complete,
        }
