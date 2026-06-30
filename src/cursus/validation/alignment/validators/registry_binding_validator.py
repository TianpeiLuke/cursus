"""
B3 — Registry-Binding Validator (FZ 31e1d3g3 Phase D, the reframed Level-4).

The old Level-4 step-type validators did SOURCE-LEVEL contract testing — ``inspect.getsource(
builder_class)`` substring scans + ``hasattr(builder_class, "_create_estimator")`` method-presence
checks — on the assumption that each step had a hand-written per-step builder whose Python source
embodied the contract. Against the shared ``TemplateStepBuilder`` shell (all 45 builders are
``class XStepBuilder(TemplateStepBuilder): STEP_NAME = "X"``) those checks are meaningless: the source
is identical for every step, and ``hasattr(shell, "_create_estimator")`` is FALSE because the estimator
factory lives in ``TrainingHandler.make_compute``, not the shell — so the old validators report EVERY
shell as FAILED ("Missing Training required method: _create_estimator").

B3 replaces "does the builder SOURCE contain pattern X" with "can the step be REALIZED from its
``.step.yaml`` + config" — the genuine residue the construction invariant can't self-check:

  B3-1 HANDLER BINDS — ``resolve_handler(sagemaker_step_type, patterns.step_assembly)`` yields a
        routable construction handler (raises ``NoBuilderError`` for Base/Lambda/unknown). Calling it
        IS the binding check that replaces the source scan.
  B3-2 BUILDER LOADABLE — ``load_builder_class(step_name)`` returns a class that is a StepBuilderBase
        (the shell or the synthesized declarative shell), i.e. no orphan registry row.
  B3-3 CONFIG-FIELD COVERAGE — the resolved config class supplies every field the bound handler +
        compute descriptor will read at build time. The required set is the handler's declared
        ``requires_config_fields`` (DATA — some reads use a runtime attr name, statically undecidable)
        UNION the descriptor-derived attrs (compute ``*_field`` names with non-None values;
        ``contract.input_source_overrides`` values). A field absent from the config class = ERROR;
        a soft ``job_arguments[].source`` provenance attr = WARNING. (Optional config fields that
        carry a default are NOT requirements — a missing value can't break ``getattr``.)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegistryBindingValidator:
    """B3: validate that a step is realizable from its registry row + ``.step.yaml`` + config.

    Exposes ``validate_builder_config_alignment(step_name)`` — the same method name/signature the
    deleted per-step-type validators exposed — so ``level_validators.run_level_4_validation`` and the
    MCP/CLI consumers are byte-unchanged.
    """

    def __init__(self, workspace_dirs: Optional[List[Path]] = None):
        self.workspace_dirs = workspace_dirs
        from ....step_catalog import StepCatalog

        self.step_catalog = StepCatalog(workspace_dirs=workspace_dirs)

    def validate_builder_config_alignment(self, step_name: str) -> Dict[str, Any]:
        """Run the three B3 sub-checks. Returns the issue-shaped result the priority resolver reads."""
        issues: List[Dict[str, Any]] = []
        try:
            iface = self._load_interface(step_name)
            handler = self._check_handler_binds(step_name, iface, issues)
            self._check_builder_loadable(step_name, issues)
            if handler is not None:
                self._check_config_coverage(step_name, iface, handler, issues)
        except _SkipValidation as skip:
            return {
                "step_name": step_name,
                "status": "SKIPPED",
                "reason": str(skip),
                "rule_type": "registry_binding",
                "issues": [],
            }
        except Exception as e:  # defensive — a B3 bug must not mask the suite
            logger.error(f"B3 registry-binding validation crashed for {step_name}: {e}")
            return {
                "step_name": step_name,
                "status": "ERROR",
                "rule_type": "registry_binding",
                "issues": [
                    {
                        "level": "ERROR",
                        "message": f"Registry-binding validation error: {e}",
                        "rule_type": "registry_binding",
                    }
                ],
            }

        has_error = any(i["level"] in ("CRITICAL", "ERROR") for i in issues)
        return {
            "step_name": step_name,
            "status": "ISSUES_FOUND" if has_error else "COMPLETED",
            "rule_type": "registry_binding",
            "issues": issues,
        }

    # --- sub-checks ---

    def _load_interface(self, step_name: str):
        from ....steps.interfaces import load_interface

        return load_interface(step_name)

    def _check_handler_binds(self, step_name: str, iface, issues: List[Dict[str, Any]]):
        """B3-1: the step's (sagemaker_step_type, step_assembly) must resolve to a routable handler."""
        from ....registry.step_names import get_sagemaker_step_type
        from ....core.base.builder_templates import resolve_handler, NoBuilderError

        sm_type = get_sagemaker_step_type(step_name)
        step_assembly = getattr(getattr(iface, "patterns", None), "step_assembly", None)

        # Lambda is a registered no-builder row whose ruleset still routes into L4; SKIP it explicitly
        # rather than emit a spurious ERROR (it has no construction handler by design).
        if sm_type in ("Lambda", "Base"):
            raise _SkipValidation(
                f"{sm_type} steps have no construction handler (no-builder row)"
            )

        try:
            return resolve_handler(sm_type, step_assembly)
        except NoBuilderError as e:
            issues.append(
                {
                    "level": "ERROR",
                    "message": f"No construction handler for {step_name} ({sm_type}/{step_assembly}): {e}",
                    "rule_type": "registry_binding",
                    "details": {
                        "sagemaker_step_type": sm_type,
                        "step_assembly": step_assembly,
                    },
                }
            )
            return None

    def _check_builder_loadable(self, step_name: str, issues: List[Dict[str, Any]]):
        """B3-2: a builder class must load (physical shell or synthesized) and be a StepBuilderBase."""
        from ....core.base.builder_base import StepBuilderBase

        builder_class = self.step_catalog.load_builder_class(step_name)
        if builder_class is None:
            # Loadable-but-absent in this env (e.g. SDK builder offline) is not a binding ERROR here —
            # the closure gate owns env-specific discoverability; B3-1 already proved the handler binds.
            issues.append(
                {
                    "level": "WARNING",
                    "message": f"No builder class loadable for {step_name} in this environment",
                    "rule_type": "registry_binding",
                }
            )
            return
        if not (
            isinstance(builder_class, type)
            and issubclass(builder_class, StepBuilderBase)
        ):
            issues.append(
                {
                    "level": "ERROR",
                    "message": f"Builder for {step_name} is not a StepBuilderBase subclass: {builder_class!r}",
                    "rule_type": "registry_binding",
                }
            )

    def _check_config_coverage(
        self, step_name: str, iface, handler, issues: List[Dict[str, Any]]
    ):
        """B3-3: the resolved config class must supply every field the handler/descriptor reads."""
        config_fields = self._config_field_names(step_name, issues)
        if config_fields is None:
            return  # config class unresolvable — already reported

        required, soft = self._required_config_attrs(iface, handler)
        for attr in sorted(required):
            if attr not in config_fields:
                issues.append(
                    {
                        "level": "ERROR",
                        "message": f"Config for {step_name} is missing required field '{attr}' "
                        f"that the bound {type(handler).__name__} reads at build time",
                        "rule_type": "registry_binding",
                        "details": {"missing_field": attr},
                    }
                )
        for attr in sorted(soft):
            if attr not in config_fields:
                issues.append(
                    {
                        "level": "WARNING",
                        "message": f"Config for {step_name} may be missing field '{attr}' "
                        f"(soft: job-argument source provenance)",
                        "rule_type": "registry_binding",
                        "details": {"missing_field": attr},
                    }
                )

    # --- helpers ---

    def _config_field_names(
        self, step_name: str, issues: List[Dict[str, Any]]
    ) -> Optional[set]:
        """Resolve the config CLASS via get_config_class_name (honors the convention-breakers) and
        return the names the build can read off it; ERROR if it can't be resolved/discovered.

        The set is the UNION of pydantic ``model_fields`` AND class-level attributes (``dir``) —
        because the handler reads a config-sourced input as ``getattr(b.config, attr)`` and accepts a
        method/property too (``source = resolved() if callable(resolved) else resolved``,
        builder_templates.py:220-221). So e.g. ``PackageConfig.inference_scripts_source`` is a method,
        not a field, yet is a valid source — checking ``model_fields`` alone would false-ERROR on it.
        """
        from ....registry.step_names import get_config_class_name

        try:
            config_class_name = get_config_class_name(step_name)
        except Exception as e:
            issues.append(
                {
                    "level": "ERROR",
                    "message": f"Cannot resolve config_class for {step_name}: {e}",
                    "rule_type": "registry_binding",
                }
            )
            return None

        config_classes = self.step_catalog.discover_config_classes()
        config_class = config_classes.get(config_class_name)
        if config_class is None:
            issues.append(
                {
                    "level": "ERROR",
                    "message": f"Config class {config_class_name!r} for {step_name} is not discoverable",
                    "rule_type": "registry_binding",
                    "details": {"config_class": config_class_name},
                }
            )
            return None

        names = {a for a in dir(config_class) if not a.startswith("_")}
        model_fields = getattr(config_class, "model_fields", None)
        if model_fields is not None:
            names |= set(model_fields.keys())
        return names

    @staticmethod
    def _required_config_attrs(iface, handler) -> tuple:
        """The (required, soft) attr sets the config must cover: handler.requires_config_fields +
        compute *_field (non-None) + contract.input_source_overrides values are REQUIRED;
        job_arguments[].source is SOFT."""
        required: set = set(getattr(handler, "requires_config_fields", ()) or ())

        compute = getattr(iface, "compute", None)
        if compute is not None:
            for fld in ("framework_version_field", "py_version_field"):
                val = getattr(compute, fld, None)
                if val:  # Optional[str] — skip None so it doesn't false-ERROR (e.g. Transform)
                    required.add(val)

        contract = getattr(iface, "contract", None)
        overrides = getattr(contract, "input_source_overrides", None) or {}
        for attr in overrides.values():
            if isinstance(attr, str) and attr:
                required.add(attr)

        soft: set = set()
        for ja in getattr(contract, "job_arguments", None) or []:
            src = getattr(ja, "source", None)
            if isinstance(src, str) and src:
                soft.add(src)

        return required, soft


class _SkipValidation(Exception):
    """Internal: a step that legitimately has no construction handler (Base/Lambda no-builder rows)."""
