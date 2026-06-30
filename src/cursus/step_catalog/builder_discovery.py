"""
Builder class auto-discovery with workspace support.

This module provides AST-based builder class discovery that mirrors the ConfigAutoDiscovery
architecture for consistency. It handles deployment portability internally and supports
both package and workspace builder discovery.
"""

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type, Any

logger = logging.getLogger(__name__)


@dataclass
class _BuilderDescriptor:
    """A discovered builder, found by the AST scan but NOT yet imported (FZ 31e1d3g/Phase 2).

    The scan resolves ``step_name`` + the file/class WITHOUT importing the module; the class is
    materialized lazily by ``get_class()`` (cached) only when a builder is actually loaded. This
    decouples discovery (cheap, import-free) from importlib I/O — so an offline scan of an SDK-bound
    builder records its descriptor instead of failing mid-scan, and the import cost is paid once, on
    demand, at instantiation time. ``get_class()`` returns None if the import fails (e.g. the SAIS SDK
    is absent) — callers preserve today's "unloadable builder is skipped" behavior by filtering None.
    """

    step_name: str
    file_path: Path
    class_name: str
    _loader: Any  # callable(file_path, class_name) -> Optional[Type]; the discovery's _load_class_from_file
    _cached: Optional[Type] = None
    _attempted: bool = False

    def get_class(self) -> Optional[Type]:
        if not self._attempted:
            self._cached = self._loader(self.file_path, self.class_name)
            self._attempted = True
        return self._cached


# Closed set of framework base-class names that mark a class as a step builder during
# the pure-AST discovery scan (which matches written base names, not the runtime MRO).
# - "StepBuilderBase": the abstract root every builder ultimately derives from.
# - "TemplateStepBuilder": the routed-builder facade; its 2-line shell subclasses
#   (`class XStepBuilder(TemplateStepBuilder)`) must be discovered just like the legacy
#   `class XStepBuilder(StepBuilderBase)` form. TemplateStepBuilder is itself a
#   StepBuilderBase subclass, so adding it keeps this a strict superset.
STEP_BUILDER_BASE_NAMES = {"StepBuilderBase", "TemplateStepBuilder"}


class BuilderAutoDiscovery:
    """
    AST-based builder class discovery with workspace support.

    Mirrors ConfigAutoDiscovery architecture for consistency and handles
    deployment portability internally.
    """

    def __init__(self, package_root: Path, workspace_dirs: Optional[List[Path]] = None):
        """
        Initialize builder discovery with package and workspace search spaces.

        Args:
            package_root: Root directory of the cursus package
            workspace_dirs: Optional list of workspace directories to search
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

        # Caches for performance
        self._builder_cache: Dict[
            str, Type
        ] = {}  # step_name -> materialized class (load cache)
        self._builder_paths: Dict[str, Path] = {}
        self._discovery_complete = False

        # Registry-walk materializer cache (FZ 31e1d3g3 Phase A): step_name -> synthesized
        # TemplateStepBuilder subclass for a step that has a .step.yaml interface but NO physical
        # builder_*.py file. Keyed by step_name so a step always maps to the SAME class object
        # within a process (identity stability — pickling / class-keyed caches; OQ 31e1d3g3a).
        # While the 45 shells still exist, this fallback never fires (file-descriptor-first), so
        # it is pure enabling infra with zero behavior change until the shells are deleted (Phase E).
        self._synthesized_builders: Dict[str, Type] = {}

        # Discovery results — DESCRIPTORS, not yet-imported classes (FZ 31e1d3g Phase 2). The scan
        # records (step_name, file, class) without importing; classes materialize lazily on load.
        self._package_builders: Dict[str, _BuilderDescriptor] = {}
        self._workspace_builders: Dict[
            str, Dict[str, _BuilderDescriptor]
        ] = {}  # workspace_id -> {step_name -> descriptor}

        # Registry integration
        self._registry_info: Dict[str, Dict[str, Any]] = {}

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
        and other metadata that can guide the discovery process.
        """
        try:
            from ..registry.step_names import get_step_names

            step_names_dict = get_step_names()
            for step_name, step_info in step_names_dict.items():
                self._registry_info[step_name] = step_info

            self.logger.debug(
                f"Loaded registry info for {len(self._registry_info)} steps"
            )

        except ImportError as e:
            self.logger.warning(f"Could not import registry step_names: {e}")
            self._registry_info = {}
        except Exception as e:
            self.logger.error(f"Error loading registry info: {e}")
            self._registry_info = {}

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
        Discover all builder classes from package and workspaces.

        Args:
            project_id: Optional project ID for workspace-specific discovery

        Returns:
            Dictionary mapping step names to builder class types
        """
        if not self._discovery_complete:
            self._run_discovery()

        # Combine package + workspace DESCRIPTORS (workspace overrides package), then materialize.
        descriptors: Dict[str, _BuilderDescriptor] = {}
        descriptors.update(self._package_builders)
        if project_id and project_id in self._workspace_builders:
            descriptors.update(self._workspace_builders[project_id])
        else:
            for workspace_builders in self._workspace_builders.values():
                descriptors.update(workspace_builders)

        # Materialize to live classes; EXCLUDE any that fail to import (e.g. SDK builders offline) —
        # this preserves the legacy contract that this map contains only loadable classes (the eager
        # scan used to drop unimportable files at scan time; we now drop them at materialize time).
        all_builders: Dict[str, Type] = {}
        for step_name, descriptor in descriptors.items():
            cls = descriptor.get_class()
            if cls is not None:
                all_builders[step_name] = cls

        # Registry-walk fill (FZ 31e1d3g3 Phase A): for any registry step NOT covered by a physical
        # builder file, synthesize its declarative shell. While the 45 files exist this adds nothing
        # (every registry row already has a descriptor); once Phase E deletes them, this keeps the
        # discovered set complete. Workspace-specific discovery (project_id) is left file-only — the
        # registry walk is package scope, matching the Risk-2 permanent-hybrid decision.
        if project_id is None:
            for step_name in self._registry_info:
                if step_name not in all_builders:
                    synthesized = self._synthesize_builder(step_name)
                    if synthesized is not None:
                        all_builders[step_name] = synthesized

        self.logger.info(f"Discovered {len(all_builders)} builder classes")
        return all_builders

    def load_builder_class(self, step_name: str) -> Optional[Type]:
        """
        Load builder class for a specific step with workspace-aware discovery.

        Args:
            step_name: Name of the step to load builder for

        Returns:
            Builder class type or None if not found
        """
        # Check the materialized-class cache first
        if step_name in self._builder_cache:
            return self._builder_cache[step_name]

        # Ensure discovery is complete (populates descriptors, no imports)
        if not self._discovery_complete:
            self._run_discovery()

        # Try workspace builders first (higher priority), then package — materialize lazily from the
        # descriptor (the import happens HERE, on demand, and may return None if it fails, e.g. the
        # SAIS SDK is absent — preserving today's "unloadable builder is skipped" behavior).
        for workspace_id, workspace_builders in self._workspace_builders.items():
            if step_name in workspace_builders:
                builder_class = workspace_builders[step_name].get_class()
                if builder_class is not None:
                    self._builder_cache[step_name] = builder_class
                    self.logger.debug(
                        f"Loaded builder for {step_name} from workspace {workspace_id}"
                    )
                    return builder_class

        if step_name in self._package_builders:
            builder_class = self._package_builders[step_name].get_class()
            if builder_class is not None:
                self._builder_cache[step_name] = builder_class
                self.logger.debug(f"Loaded builder for {step_name} from package")
                return builder_class

        # No physical builder file — try the registry-walk materializer (FZ 31e1d3g3 Phase A):
        # synthesize a TemplateStepBuilder subclass for a step that has a .step.yaml interface and
        # routes. Returns None for steps that don't (abstract rows, SDK-delegation offline, unknown).
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
        at runtime so a step needs no physical ``builder_*.py``. It is consulted ONLY as a fallback
        after file-descriptor lookup (``load_builder_class`` / ``discover_builder_classes``), so while
        the 45 files still exist this never fires — it is pure enabling infra until Phase E deletes
        the shells.

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
        """Run the complete discovery process."""
        try:
            # Discover package builders
            self._discover_package_builders()

            # Discover workspace builders
            self._discover_workspace_builders()

            self._discovery_complete = True

            total_builders = len(self._package_builders) + sum(
                len(builders) for builders in self._workspace_builders.values()
            )
            self.logger.info(
                f"Builder discovery complete: {total_builders} builders found"
            )

        except Exception as e:
            self.logger.error(f"Error during builder discovery: {e}")
            # Graceful degradation
            self._package_builders = {}
            self._workspace_builders = {}

    def _discover_package_builders(self):
        """Discover builders in the cursus package."""
        package_builders_dir = self.package_root / "steps" / "builders"
        if package_builders_dir.exists():
            self._package_builders = self._scan_builder_directory(
                package_builders_dir, "package"
            )
            self.logger.debug(f"Found {len(self._package_builders)} package builders")

    def _discover_workspace_builders(self):
        """Discover builders in workspace directories with simplified structure."""
        for workspace_dir in self.workspace_dirs:
            try:
                workspace_path = Path(workspace_dir)
                if not workspace_path.exists():
                    self.logger.warning(
                        f"Workspace directory does not exist: {workspace_path}"
                    )
                    continue

                # Simplified structure: workspace_dir directly contains builders/
                workspace_builders_dir = workspace_path / "builders"
                if workspace_builders_dir.exists():
                    workspace_builders = self._scan_builder_directory(
                        workspace_builders_dir, workspace_path.name
                    )
                    if workspace_builders:
                        self._workspace_builders[workspace_path.name] = (
                            workspace_builders
                        )
                        self.logger.debug(
                            f"Found {len(workspace_builders)} builders in workspace {workspace_path.name}"
                        )
                else:
                    self.logger.debug(
                        f"No builders directory found in workspace: {workspace_path}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error discovering workspace builders in {workspace_dir}: {e}"
                )

    def _scan_builder_directory(
        self, builders_dir: Path, workspace_id: str
    ) -> Dict[str, "_BuilderDescriptor"]:
        """
        Scan directory for builder files using AST analysis (import-free, Phase 2).

        Args:
            builders_dir: Directory containing builder files
            workspace_id: ID of the workspace (for logging)

        Returns:
            Dictionary mapping step names to lazy _BuilderDescriptors (classes not yet imported).
        """
        builders: Dict[str, _BuilderDescriptor] = {}

        if not builders_dir.exists():
            return builders

        try:
            for py_file in builders_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue

                try:
                    # Extract a lazy descriptor using AST (no import at scan time)
                    descriptor = self._extract_builder_from_ast(py_file)
                    if descriptor:
                        builders[descriptor.step_name] = descriptor
                        self._builder_paths[descriptor.step_name] = py_file

                except Exception as e:
                    self.logger.warning(f"Error processing builder file {py_file}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error scanning builder directory {builders_dir}: {e}")

        return builders

    def _extract_builder_from_ast(
        self, file_path: Path
    ) -> Optional["_BuilderDescriptor"]:
        """
        Extract a builder DESCRIPTOR from a Python file using AST analysis — WITHOUT importing it.

        Phase 2 (FZ 31e1d3g): the scan resolves (step_name, class_name) by AST + registry only; the
        class is materialized lazily by the descriptor's get_class() at load time. This is why the
        scan no longer fails on an SDK-bound builder offline — it records the descriptor; the import
        only happens (and may fail) when the builder is actually loaded.

        Args:
            file_path: Path to the Python file

        Returns:
            A _BuilderDescriptor (step_name + file + class, lazy loader) or None if no builder class.
        """
        try:
            # Read and parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Find classes that inherit from StepBuilderBase
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if class inherits from StepBuilderBase
                    if self._inherits_from_step_builder_base(node):
                        # Extract step name from file name or class name (registry-keyed)
                        step_name = self._extract_step_name_from_builder_file(
                            file_path, node.name
                        )
                        if step_name:
                            # Record a lazy descriptor — NO import here (Phase 2).
                            return _BuilderDescriptor(
                                step_name=step_name,
                                file_path=file_path,
                                class_name=node.name,
                                _loader=self._load_class_from_file,
                            )

            return None

        except Exception as e:
            self.logger.warning(f"Error extracting builder from {file_path}: {e}")
            return None

    def _inherits_from_step_builder_base(self, class_node: ast.ClassDef) -> bool:
        """
        Check if a class is a step builder by its (direct) base class name.

        Discovery is intentionally a pure single-file AST scan — it does not import the
        module to walk the runtime MRO. So it recognizes a builder by matching the
        *written* base-class name against a closed set of framework base names.
        ``TemplateStepBuilder`` is included so that the routed-builder shells
        (``class XStepBuilder(TemplateStepBuilder)``) are discovered identically to the
        legacy ``class XStepBuilder(StepBuilderBase)`` form. ``TemplateStepBuilder`` is
        itself a ``StepBuilderBase`` subclass, so this stays a strict superset of the old
        behavior (every existing builder still matches).

        Args:
            class_node: AST class definition node

        Returns:
            True if a direct base is one of STEP_BUILDER_BASE_NAMES
        """
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in STEP_BUILDER_BASE_NAMES:
                return True
            elif isinstance(base, ast.Attribute):
                # Handle qualified names like module.StepBuilderBase
                if base.attr in STEP_BUILDER_BASE_NAMES:
                    return True
        return False

    def _extract_step_name_from_builder_file(
        self, file_path: Path, class_name: str
    ) -> Optional[str]:
        """
        Extract step name from builder file name and class name using registry information.

        Args:
            file_path: Path to the builder file
            class_name: Name of the builder class

        Returns:
            Step name or None if not extractable
        """
        # First, try to find step name from registry by matching builder class name
        for step_name, step_info in self._registry_info.items():
            builder_step_name = step_info.get("builder_step_name")
            if builder_step_name and builder_step_name == class_name:
                self.logger.debug(
                    f"Found step name {step_name} for builder {class_name} via registry"
                )
                return step_name

        # Try to extract from file name (e.g., builder_xgboost_training_step.py)
        file_name = file_path.stem
        if file_name.startswith("builder_") and file_name.endswith("_step"):
            # Remove builder_ prefix and _step suffix
            step_name_parts = file_name[8:-5].split(
                "_"
            )  # Remove "builder_" and "_step"

            # Apply special case handling for known patterns
            step_name = self._convert_parts_to_pascal_case_with_special_cases(
                step_name_parts
            )

            # Validate against registry if possible
            if step_name in self._registry_info:
                return step_name

            # Try variations if exact match not found
            for registered_step in self._registry_info.keys():
                if registered_step.lower() == step_name.lower():
                    return registered_step

            return step_name

        # Try to extract from class name (e.g., XGBoostTrainingStepBuilder)
        if class_name.endswith("StepBuilder"):
            step_name = class_name[:-11]  # Remove "StepBuilder"
        elif class_name.endswith("Builder"):
            step_name = class_name[:-7]  # Remove "Builder"
        else:
            step_name = class_name

        # Validate against registry
        if step_name in self._registry_info:
            return step_name

        # Try case-insensitive match
        for registered_step in self._registry_info.keys():
            if registered_step.lower() == step_name.lower():
                return registered_step

        # Fallback: use extracted name as-is
        return step_name

    def _convert_parts_to_pascal_case_with_special_cases(self, parts: List[str]) -> str:
        """
        Convert file name parts to PascalCase, delegating compound-acronym handling to the
        shared naming module (cursus.step_catalog.naming) — one source of truth.

        Args:
            parts: List of file name parts (e.g., ['xgboost', 'training'])

        Returns:
            PascalCase step name with proper special case handling
        """
        from .naming import parts_to_pascal

        return parts_to_pascal(parts)

    def _load_class_from_file(self, file_path: Path, class_name: str) -> Optional[Type]:
        """
        Load class using relative imports with package parameter (deployment-agnostic).

        This approach uses importlib.import_module with relative paths and package parameter,
        which is cleaner than sys.path manipulation and works across all deployment scenarios.

        Args:
            file_path: Path to the Python file
            class_name: Name of the class to load

        Returns:
            Class type or None if loading fails
        """
        import importlib

        module = None
        attempted: List[str] = []

        # Preferred: absolute import via the actual root package name (deployment-agnostic).
        # __package__ here is e.g. "cursus.step_catalog" (or "<vendor>.cursus.step_catalog"
        # in a synced/nested deployment). Deriving the module path from the package the
        # builders actually live in avoids the fragile fixed-".." relative path, which
        # breaks when package_root is not exactly the top-level package directory.
        absolute_module_path = self._file_to_package_module_path(file_path)
        if absolute_module_path:
            attempted.append(absolute_module_path)
            try:
                module = importlib.import_module(absolute_module_path)
            except Exception as e:
                self.logger.debug(
                    f"Absolute import {absolute_module_path} failed, will try relative: {e}"
                )

        # Fallback: relative import with package parameter (works when package_root is
        # exactly the package dir).
        if module is None:
            relative_module_path = self._file_to_relative_module_path(file_path)
            if relative_module_path:
                attempted.append(relative_module_path)
                try:
                    module = importlib.import_module(
                        relative_module_path, package=__package__
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Error loading class {class_name} from {file_path} "
                        f"(tried {attempted}): {e}"
                    )
                    return None

        if module is None:
            self.logger.warning(
                f"Could not determine an import path for {file_path} (tried {attempted})"
            )
            return None

        # Get the class from the module
        if hasattr(module, class_name):
            return getattr(module, class_name)
        self.logger.warning(f"Class {class_name} not found in module {module.__name__}")
        return None

    def _file_to_package_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert a file path to an ABSOLUTE module path rooted at the real package name.

        This is deployment-agnostic: it derives the root package from this module's own
        ``__package__`` (e.g. ``cursus`` from ``cursus.step_catalog``, or
        ``vendor.cursus`` from ``vendor.cursus.step_catalog``) and joins it with the
        portion of ``file_path`` at/after the ``cursus`` directory segment. Unlike a
        fixed-".."-prefixed relative import, it does not assume ``package_root`` is
        exactly the top-level package directory, so it keeps working when the package is
        synced/nested under another package.

        Args:
            file_path: Path to the Python file (e.g. .../cursus/steps/builders/builder_x.py)

        Returns:
            Absolute module path (e.g. 'cursus.steps.builders.builder_x') or None.
        """
        try:
            # Root package this discovery module lives in (handles nesting).
            # __package__ == "cursus.step_catalog" -> root "cursus"
            #             == "vendor.cursus.step_catalog" -> root prefix "vendor.cursus"
            pkg_parts = (__package__ or "").split(".")
            if "step_catalog" in pkg_parts:
                root_pkg_parts = pkg_parts[: pkg_parts.index("step_catalog")]
            else:
                root_pkg_parts = pkg_parts[:-1] if len(pkg_parts) > 1 else pkg_parts
            if not root_pkg_parts:
                return None

            # The file's module parts from the 'cursus' segment onward, dropping '.py'.
            file_parts = list(file_path.with_suffix("").parts)
            if "cursus" in file_parts:
                # Everything AFTER the cursus dir (steps, builders, builder_x)
                sub_parts = file_parts[file_parts.index("cursus") + 1 :]
            else:
                # Fall back to path relative to package_root
                try:
                    sub_parts = list(
                        file_path.with_suffix("").relative_to(self.package_root).parts
                    )
                except ValueError:
                    return None
            if not sub_parts:
                return None

            return ".".join(root_pkg_parts + sub_parts)
        except Exception as e:
            self.logger.debug(
                f"Error converting {file_path} to package module path: {e}"
            )
            return None

    def _file_to_relative_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to relative module path for use with importlib.import_module.

        This creates relative import paths like "..steps.builders.builder_xgboost_training_step"
        that work with the package parameter in importlib.import_module.

        Args:
            file_path: Path to the Python file

        Returns:
            Relative module path string (e.g., '..steps.builders.builder_xgboost_training_step')
        """
        try:
            # Get the path relative to the package root
            try:
                relative_path = file_path.relative_to(self.package_root)
            except ValueError:
                # File is not under package root, might be in workspace
                self.logger.debug(
                    f"File {file_path} not under package root {self.package_root}"
                )
                return None

            # Convert path to module format
            parts = list(relative_path.parts)

            # Remove .py extension from the last part
            if parts[-1].endswith(".py"):
                parts[-1] = parts[-1][:-3]

            # Create relative module path with .. prefix for relative import
            # This works with importlib.import_module(relative_path, package=__package__)
            relative_module_path = ".." + ".".join(parts)

            self.logger.debug(
                f"Converted {file_path} to relative module path: {relative_module_path}"
            )
            return relative_module_path

        except Exception as e:
            self.logger.warning(
                f"Error converting file path {file_path} to relative module path: {e}"
            )
            return None

    def _file_to_module_path(self, file_path: Path) -> Optional[str]:
        """
        Convert file path to Python module path (legacy method for compatibility).

        Args:
            file_path: Path to the Python file

        Returns:
            Module path string (e.g., 'cursus.steps.builders.builder_xgboost_training_step')
        """
        try:
            # Get the path relative to the package root
            try:
                relative_path = file_path.relative_to(self.package_root)
            except ValueError:
                # File is not under package root, might be in workspace
                self.logger.debug(
                    f"File {file_path} not under package root {self.package_root}"
                )
                return None

            # Convert path to module format
            parts = list(relative_path.parts)

            # Remove .py extension from the last part
            if parts[-1].endswith(".py"):
                parts[-1] = parts[-1][:-3]

            # Create module path with cursus prefix
            module_path = "cursus." + ".".join(parts)

            self.logger.debug(f"Converted {file_path} to module path: {module_path}")
            return module_path

        except Exception as e:
            self.logger.warning(
                f"Error converting file path {file_path} to module path: {e}"
            )
            return None

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
            "file_path": str(self._builder_paths.get(step_name, "Unknown")),
            "workspace_id": self._get_workspace_for_step(step_name),
        }

    def _get_workspace_for_step(self, step_name: str) -> str:
        """Get workspace ID for a step."""
        for workspace_id, workspace_builders in self._workspace_builders.items():
            if step_name in workspace_builders:
                return workspace_id
        return "package"

    def list_available_builders(self) -> List[str]:
        """
        List all available builder step names.

        Returns:
            List of step names that have builders
        """
        if not self._discovery_complete:
            self._run_discovery()

        all_steps = set(self._package_builders.keys())
        for workspace_builders in self._workspace_builders.values():
            all_steps.update(workspace_builders.keys())

        return sorted(list(all_steps))

    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        Get discovery statistics.

        Returns:
            Dictionary with discovery statistics
        """
        if not self._discovery_complete:
            self._run_discovery()

        return {
            "package_builders": len(self._package_builders),
            "workspace_builders": {
                workspace_id: len(builders)
                for workspace_id, builders in self._workspace_builders.items()
            },
            "total_builders": len(self._package_builders)
            + sum(len(builders) for builders in self._workspace_builders.values()),
            "cached_builders": len(self._builder_cache),
            "discovery_complete": self._discovery_complete,
        }
