"""
Specification discovery for the unified step catalog system.

Interface-first: every step specification is a *view* onto a validated
``StepInterface`` loaded from the step's ``.step.yaml``. The StepInterface is a
drop-in for the legacy StepSpecification — it exposes ``step_type``, ``node_type``,
``dependencies`` and ``outputs``, so :meth:`serialize_spec` (kept verbatim) reads
it unchanged. Discovery is driven by the registry's canonical step names + the
per-step ``variants`` block; there is no directory scan, no AST parse and no
per-file import. The former ``steps/specs/`` folder scan is gone.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class SpecAutoDiscovery:
    """Specification discovery sourced from step interfaces + the step registry."""

    def __init__(self, package_root: Path, workspace_dirs: List[Path]):
        """
        Initialize spec discovery.

        The ``package_root`` / ``workspace_dirs`` arguments are retained for a
        stable constructor signature across the discovery modules, but interface-
        first discovery reads from the registry + ``.step.yaml`` interfaces rather
        than scanning these directories.

        Args:
            package_root: Root of the cursus package
            workspace_dirs: List of workspace directories to search
        """
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs
        self.logger = logging.getLogger(__name__)

    def discover_spec_classes(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover all step specifications from the registry + interfaces.

        Args:
            project_id: Optional project ID (accepted for signature stability; the
                registry + interface loader are the source of truth)

        Returns:
            Dictionary mapping PascalCase canonical step name to its StepInterface
            (a view onto the step's validated interface, a StepSpecification drop-in).
            Steps without an interface file are skipped.
        """
        from ..registry.step_names import get_all_step_names
        from ..steps.interfaces import load_interface

        discovered: Dict[str, Any] = {}

        for step_name in get_all_step_names():
            try:
                iface = load_interface(step_name)
            except FileNotFoundError:
                self.logger.debug(f"No interface file for step: {step_name}")
                continue
            except Exception as e:
                self.logger.warning(f"Error loading interface for {step_name}: {e}")
                continue

            discovered[step_name] = iface

        self.logger.info(
            f"Discovered {len(discovered)} specification interfaces"
        )
        return discovered

    def load_spec_class(self, step_name: str) -> Optional[Any]:
        """
        Load the specification for a specific step.

        Args:
            step_name: PascalCase canonical step name

        Returns:
            The step's StepInterface (a StepSpecification drop-in), or None if the
            step has no interface file.
        """
        from ..steps.interfaces import load_interface

        try:
            return load_interface(step_name)
        except FileNotFoundError:
            self.logger.debug(f"No specification found for step: {step_name}")
            return None
        except Exception as e:
            self.logger.warning(
                f"Error loading specification for step {step_name}: {e}"
            )
            return None

    def _is_spec_instance(self, obj: Any) -> bool:
        """Check if an object is a specification instance."""
        try:
            # Check if it has the expected attributes of a StepSpecification
            return (
                hasattr(obj, "step_type")
                and hasattr(obj, "dependencies")
                and hasattr(obj, "outputs")
            )
        except Exception:
            return False

    def find_specs_by_contract(self, contract_name: str) -> Dict[str, Any]:
        """
        Find all specifications (across job-type variants) for a step.

        This method enables contract-specification alignment validation by finding
        the specifications associated with a given step name. Under interface-first
        discovery every variant of a step lives in one ``.step.yaml`` ``variants``
        block, so this returns one serialized-spec dict per variant (plus the base
        interface when the step declares no variants), keyed by
        ``{StepName}_{variant}`` so :meth:`_extract_job_type_from_spec_name_registry`
        can classify each entry by job type.

        Args:
            contract_name: Name of the step / contract to find specifications for.
                Accepts a PascalCase canonical name or a file-stem-ish name (e.g.
                ``"tabular_preprocessing"`` / ``"tabular_preprocessing_contract"``).

        Returns:
            Dictionary mapping ``{StepName}_{variant}`` (or the bare step name for a
            variant-less step) to its serialized specification dictionary.
        """
        try:
            from ..steps.interfaces import load_interface

            step_name = self._canonical_step_name(contract_name)
            if step_name is None:
                self.logger.debug(
                    f"Could not resolve a canonical step name for contract '{contract_name}'"
                )
                return {}

            try:
                base_iface = load_interface(step_name)
            except FileNotFoundError:
                self.logger.debug(f"No interface file for step: {step_name}")
                return {}

            matching_specs: Dict[str, Any] = {}

            variants = getattr(base_iface, "variants", None) or {}
            if variants:
                for variant_name in variants:
                    try:
                        variant_iface = load_interface(step_name, job_type=variant_name)
                    except Exception as e:
                        self.logger.warning(
                            f"Error loading variant '{variant_name}' of {step_name}: {e}"
                        )
                        continue
                    serialized = self.serialize_spec(variant_iface)
                    if serialized:
                        matching_specs[f"{step_name}_{variant_name}"] = serialized
            else:
                serialized = self.serialize_spec(base_iface)
                if serialized:
                    matching_specs[step_name] = serialized

            self.logger.debug(
                f"Found {len(matching_specs)} specifications for contract '{contract_name}'"
            )
            return matching_specs

        except Exception as e:
            self.logger.error(f"Error finding specs for contract {contract_name}: {e}")
            return {}

    def serialize_spec(self, spec_instance: Any) -> Dict[str, Any]:
        """
        Convert specification instance to dictionary format.

        This method provides standardized serialization of StepSpecification objects
        for use in validation and alignment testing.

        Args:
            spec_instance: StepSpecification instance to serialize

        Returns:
            Dictionary representation of the specification
        """
        try:
            if not self._is_spec_instance(spec_instance):
                raise ValueError("Object is not a valid specification instance")

            # Serialize dependencies
            dependencies = []
            if hasattr(spec_instance, "dependencies") and spec_instance.dependencies:
                for dep_name, dep_spec in spec_instance.dependencies.items():
                    dependencies.append(
                        {
                            "logical_name": dep_spec.logical_name,
                            "dependency_type": (
                                dep_spec.dependency_type.value
                                if hasattr(dep_spec.dependency_type, "value")
                                else str(dep_spec.dependency_type)
                            ),
                            "required": dep_spec.required,
                            "compatible_sources": dep_spec.compatible_sources,
                            "data_type": dep_spec.data_type,
                            "description": dep_spec.description,
                        }
                    )

            # Serialize outputs
            outputs = []
            if hasattr(spec_instance, "outputs") and spec_instance.outputs:
                for out_name, out_spec in spec_instance.outputs.items():
                    outputs.append(
                        {
                            "logical_name": out_spec.logical_name,
                            "output_type": (
                                out_spec.output_type.value
                                if hasattr(out_spec.output_type, "value")
                                else str(out_spec.output_type)
                            ),
                            "property_path": out_spec.property_path,
                            "data_type": out_spec.data_type,
                            "description": out_spec.description,
                        }
                    )

            return {
                "step_type": spec_instance.step_type,
                "node_type": (
                    spec_instance.node_type.value
                    if hasattr(spec_instance.node_type, "value")
                    else str(spec_instance.node_type)
                ),
                "dependencies": dependencies,
                "outputs": outputs,
            }

        except Exception as e:
            self.logger.error(f"Error serializing specification: {e}")
            return {}

    def load_all_specifications(self) -> Dict[str, Dict[str, Any]]:
        """
        Load and serialize every step specification from the interfaces.

        This method provides comprehensive specification loading for validation
        frameworks and dependency analysis tools. It discovers every step interface
        and serializes each to dictionary format for easy consumption.

        Returns:
            Dictionary mapping PascalCase canonical step name to its serialized
            specification dictionary.
        """
        try:
            all_specs = {}

            # Discover all specification interfaces (keyed by canonical step name)
            discovered_specs = self.discover_spec_classes()

            # Serialize each specification to dictionary format
            for spec_name, spec_instance in discovered_specs.items():
                try:
                    if self._is_spec_instance(spec_instance):
                        serialized_spec = self.serialize_spec(spec_instance)
                        if serialized_spec:
                            all_specs[spec_name] = serialized_spec
                            self.logger.debug(
                                f"Loaded and serialized specification: {spec_name}"
                            )
                    else:
                        self.logger.warning(
                            f"Invalid specification instance for {spec_name}"
                        )

                except Exception as e:
                    self.logger.warning(
                        f"Error serializing specification {spec_name}: {e}"
                    )
                    continue

            self.logger.info(f"Successfully loaded {len(all_specs)} specifications")
            return all_specs

        except Exception as e:
            self.logger.error(f"Error loading all specifications: {e}")
            return {}

    def get_job_type_variants(self, base_step_name: str) -> List[str]:
        """
        Get all job type variant keys for a base step name.

        Job-type variants (training, validation, testing, calibration, ...) are
        declared in the step's ``.step.yaml`` ``variants`` block. This returns the
        variant KEYS from that block (not the VariantDecl values).

        Args:
            base_step_name: Base name of the step (PascalCase canonical or file-stem-ish)

        Returns:
            List of job type variant keys found (empty if the step has none or has
            no interface file).
        """
        try:
            from ..steps.interfaces import load_interface

            step_name = self._canonical_step_name(base_step_name)
            if step_name is None:
                self.logger.debug(
                    f"Could not resolve a canonical step name for '{base_step_name}'"
                )
                return []

            try:
                iface = load_interface(step_name)
            except FileNotFoundError:
                self.logger.debug(f"No interface file for step: {step_name}")
                return []

            variants = list((getattr(iface, "variants", None) or {}).keys())
            self.logger.debug(
                f"Found job type variants for '{base_step_name}': {variants}"
            )
            return variants

        except Exception as e:
            self.logger.error(
                f"Error finding job type variants for {base_step_name}: {e}"
            )
            return []

    def _canonical_step_name(self, name: str) -> Optional[str]:
        """
        Resolve an incoming step/contract name to a canonical PascalCase step name.

        Accepts a name that is already a canonical registry key, or a file-stem-ish
        name (e.g. ``"tabular_preprocessing"`` / ``"tabular_preprocessing_contract"``),
        bridging the latter via ``get_canonical_name_from_file_name``. Never emits a
        raw snake_case stem.

        Returns:
            The canonical PascalCase step name, or None if it cannot be resolved.
        """
        from ..registry.step_names import (
            get_all_step_names,
            get_canonical_name_from_file_name,
        )

        # Already a canonical registry key.
        if name in get_all_step_names():
            return name

        # Bridge a file-stem-ish name to a canonical name.
        file_stem = name
        if file_stem.endswith("_contract"):
            file_stem = file_stem[: -len("_contract")]
        elif file_stem.endswith("_spec"):
            file_stem = file_stem[: -len("_spec")]

        try:
            return get_canonical_name_from_file_name(file_stem)
        except Exception:
            return None

    # PHASE 2 ENHANCEMENT: Smart Specification Integration
    def create_unified_specification(self, contract_name: str) -> Dict[str, Any]:
        """
        Create unified specification from multiple variants using smart selection.

        Integrates SmartSpecificationSelector logic:
        - Multi-variant specification discovery using existing find_specs_by_contract()
        - Union of dependencies and outputs from all variants
        - Smart validation logic with detailed feedback
        - Primary specification selection (training > generic > first available)

        Args:
            contract_name: Name of the contract to find specifications for

        Returns:
            Unified specification model with metadata
        """
        try:
            # Use existing find_specs_by_contract method
            specifications = self.find_specs_by_contract(contract_name)

            if not specifications:
                return {
                    "primary_spec": {},
                    "variants": {},
                    "unified_dependencies": {},
                    "unified_outputs": {},
                    "dependency_sources": {},
                    "output_sources": {},
                    "variant_count": 0,
                }

            # Apply smart specification logic
            return self._apply_smart_specification_logic(specifications, contract_name)

        except Exception as e:
            self.logger.error(
                f"Error creating unified specification for contract {contract_name}: {e}"
            )
            return {
                "primary_spec": {},
                "variants": {},
                "unified_dependencies": {},
                "unified_outputs": {},
                "dependency_sources": {},
                "output_sources": {},
                "variant_count": 0,
            }

    def validate_logical_names_smart(
        self, contract: Dict[str, Any], contract_name: str
    ) -> List[Dict[str, Any]]:
        """
        Smart validation using multi-variant specification logic.

        Implements the core Smart Specification Selection validation:
        - Contract input is valid if it exists in ANY variant
        - Contract must cover intersection of REQUIRED dependencies
        - Provides detailed feedback about which variants need what

        Args:
            contract: Contract dictionary
            contract_name: Name of the contract

        Returns:
            List of validation issues
        """
        try:
            # Create unified specification
            unified_spec = self.create_unified_specification(contract_name)

            # Apply smart validation logic
            return self._validate_smart_logical_names(
                contract, unified_spec, contract_name
            )

        except Exception as e:
            self.logger.error(
                f"Error in smart validation for contract {contract_name}: {e}"
            )
            return [
                {
                    "severity": "ERROR",
                    "category": "smart_validation_error",
                    "message": f"Smart validation failed for contract {contract_name}: {str(e)}",
                    "details": {"contract": contract_name, "error": str(e)},
                    "recommendation": "Check contract and specification files for errors",
                }
            ]

    def _apply_smart_specification_logic(
        self, specifications: Dict[str, Any], contract_name: str
    ) -> Dict[str, Any]:
        """
        Apply smart specification selection logic to create unified specification.

        Args:
            specifications: Dictionary of loaded specifications
            contract_name: Name of the contract being validated

        Returns:
            Unified specification model with metadata
        """
        try:
            # Group specifications by job type using registry patterns instead of hardcoded logic
            variants = {}

            # Categorize specifications by job type
            for spec_key, spec_data in specifications.items():
                job_type = self._extract_job_type_from_spec_name_registry(spec_key)
                variants[job_type] = spec_data

            # Create unified dependency and output sets
            unified_dependencies = {}
            unified_outputs = {}
            dependency_sources = {}  # Track which variants contribute each dependency
            output_sources = {}  # Track which variants contribute each output

            # Union all dependencies from all variants
            for variant_name, spec_data in variants.items():
                for dep in spec_data.get("dependencies", []):
                    logical_name = dep.get("logical_name")
                    if logical_name:
                        unified_dependencies[logical_name] = dep
                        if logical_name not in dependency_sources:
                            dependency_sources[logical_name] = []
                        dependency_sources[logical_name].append(variant_name)

                for output in spec_data.get("outputs", []):
                    logical_name = output.get("logical_name")
                    if logical_name:
                        unified_outputs[logical_name] = output
                        if logical_name not in output_sources:
                            output_sources[logical_name] = []
                        output_sources[logical_name].append(variant_name)

            # Select primary specification (prefer training, then generic, then first available)
            primary_spec = self._select_primary_specification(variants)

            return {
                "primary_spec": primary_spec,
                "variants": variants,
                "unified_dependencies": unified_dependencies,
                "unified_outputs": unified_outputs,
                "dependency_sources": dependency_sources,
                "output_sources": output_sources,
                "variant_count": len(variants),
            }

        except Exception as e:
            self.logger.error(f"Error applying smart specification logic: {e}")
            return {
                "primary_spec": {},
                "variants": {},
                "unified_dependencies": {},
                "unified_outputs": {},
                "dependency_sources": {},
                "output_sources": {},
                "variant_count": 0,
            }

    def _extract_job_type_from_spec_name_registry(self, spec_name: str) -> str:
        """
        Extract job type from specification name using registry patterns instead of hardcoded logic.

        Args:
            spec_name: Name of the specification

        Returns:
            Job type string
        """
        try:
            # Use registry-based job type detection instead of hardcoded patterns
            from ...registry.step_names import get_spec_step_type_with_job_type

            # Try to extract job type using registry patterns
            spec_name_lower = spec_name.lower()

            # Check for known job type patterns in the spec name (shared vocabulary)
            from .naming import JOB_TYPE_KEYWORDS

            for job_type in JOB_TYPE_KEYWORDS:
                if job_type in spec_name_lower:
                    return job_type

            # Default to generic if no specific job type found
            return "generic"

        except Exception as e:
            self.logger.warning(
                f"Error extracting job type from spec name {spec_name}: {e}"
            )
            return "generic"

    def _select_primary_specification(
        self, variants: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Select the primary specification from available variants.

        Priority order:
        1. training (most common and comprehensive)
        2. generic (applies to all job types)
        3. first available variant

        Args:
            variants: Dictionary of specification variants

        Returns:
            Primary specification dictionary
        """
        if "training" in variants:
            return variants["training"]
        elif "generic" in variants:
            return variants["generic"]
        else:
            return next(iter(variants.values())) if variants else {}

    def _validate_smart_logical_names(
        self, contract: Dict[str, Any], unified_spec: Dict[str, Any], contract_name: str
    ) -> List[Dict[str, Any]]:
        """
        Smart validation of logical names using multi-variant specification logic.

        This implements the core Smart Specification Selection validation:
        - Contract input is valid if it exists in ANY variant
        - Contract must cover intersection of REQUIRED dependencies
        - Provides detailed feedback about which variants need what

        Args:
            contract: Contract dictionary
            unified_spec: Unified specification model
            contract_name: Name of the contract

        Returns:
            List of validation issues
        """
        issues = []

        try:
            # Get logical names from contract
            contract_inputs = set(contract.get("inputs", {}).keys())
            contract_outputs = set(contract.get("outputs", {}).keys())

            # Get unified logical names from all specification variants
            unified_dependencies = unified_spec.get("unified_dependencies", {})
            unified_outputs = unified_spec.get("unified_outputs", {})
            dependency_sources = unified_spec.get("dependency_sources", {})
            output_sources = unified_spec.get("output_sources", {})
            variants = unified_spec.get("variants", {})

            # SMART VALIDATION LOGIC

            # 1. Check contract inputs against unified dependencies
            unified_dep_names = set(unified_dependencies.keys())

            # Contract inputs that are not in ANY variant are errors
            invalid_inputs = contract_inputs - unified_dep_names
            for logical_name in invalid_inputs:
                issues.append(
                    {
                        "severity": "ERROR",
                        "category": "logical_names",
                        "message": f"Contract input {logical_name} not declared in any specification variant",
                        "details": {
                            "logical_name": logical_name,
                            "contract": contract_name,
                            "available_variants": list(variants.keys()),
                            "available_dependencies": list(unified_dep_names),
                        },
                        "recommendation": f"Add {logical_name} to specification dependencies or remove from contract",
                    }
                )

            # 2. Check for required dependencies that contract doesn't provide
            required_deps = set()
            optional_deps = set()

            for dep_name, dep_spec in unified_dependencies.items():
                if dep_spec.get("required", False):
                    required_deps.add(dep_name)
                else:
                    optional_deps.add(dep_name)

            missing_required = required_deps - contract_inputs
            for logical_name in missing_required:
                # Find which variants require this dependency
                requiring_variants = dependency_sources.get(logical_name, [])
                issues.append(
                    {
                        "severity": "ERROR",
                        "category": "logical_names",
                        "message": f"Contract missing required dependency {logical_name}",
                        "details": {
                            "logical_name": logical_name,
                            "contract": contract_name,
                            "requiring_variants": requiring_variants,
                        },
                        "recommendation": f"Add {logical_name} to contract inputs (required by variants: {', '.join(requiring_variants)})",
                    }
                )

            # 3. Provide informational feedback for valid optional inputs
            valid_optional_inputs = contract_inputs & optional_deps
            for logical_name in valid_optional_inputs:
                supporting_variants = dependency_sources.get(logical_name, [])
                if len(supporting_variants) < len(variants):
                    # This input is only used by some variants - provide info
                    issues.append(
                        {
                            "severity": "INFO",
                            "category": "logical_names",
                            "message": f"Contract input {logical_name} used by variants: {', '.join(supporting_variants)}",
                            "details": {
                                "logical_name": logical_name,
                                "contract": contract_name,
                                "supporting_variants": supporting_variants,
                                "total_variants": len(variants),
                            },
                            "recommendation": f"Input {logical_name} is correctly declared for multi-variant support",
                        }
                    )

            # 4. Check contract outputs against unified outputs
            unified_output_names = set(unified_outputs.keys())

            # Contract outputs that are not in ANY variant are errors
            invalid_outputs = contract_outputs - unified_output_names
            for logical_name in invalid_outputs:
                issues.append(
                    {
                        "severity": "ERROR",
                        "category": "logical_names",
                        "message": f"Contract output {logical_name} not declared in any specification variant",
                        "details": {
                            "logical_name": logical_name,
                            "contract": contract_name,
                            "available_variants": list(variants.keys()),
                            "available_outputs": list(unified_output_names),
                        },
                        "recommendation": f"Add {logical_name} to specification outputs or remove from contract",
                    }
                )

            # 5. Check for missing outputs (less critical since outputs are usually consistent)
            missing_outputs = unified_output_names - contract_outputs
            for logical_name in missing_outputs:
                producing_variants = output_sources.get(logical_name, [])
                issues.append(
                    {
                        "severity": "WARNING",
                        "category": "logical_names",
                        "message": f"Contract missing output {logical_name}",
                        "details": {
                            "logical_name": logical_name,
                            "contract": contract_name,
                            "producing_variants": producing_variants,
                        },
                        "recommendation": f"Add {logical_name} to contract outputs (produced by variants: {', '.join(producing_variants)})",
                    }
                )

            # 6. Add summary information about multi-variant validation
            if len(variants) > 1:
                issues.append(
                    {
                        "severity": "INFO",
                        "category": "multi_variant_validation",
                        "message": f"Smart Specification Selection: validated against {len(variants)} variants",
                        "details": {
                            "contract": contract_name,
                            "variants": list(variants.keys()),
                            "total_dependencies": len(unified_dependencies),
                            "total_outputs": len(unified_outputs),
                            "contract_inputs": len(contract_inputs),
                            "contract_outputs": len(contract_outputs),
                        },
                        "recommendation": "Multi-variant validation completed successfully",
                    }
                )

        except Exception as e:
            self.logger.error(f"Error in smart logical names validation: {e}")
            issues.append(
                {
                    "severity": "ERROR",
                    "category": "smart_validation_error",
                    "message": f"Smart validation logic failed: {str(e)}",
                    "details": {"contract": contract_name, "error": str(e)},
                    "recommendation": "Check specification and contract files for structural issues",
                }
            )

        return issues
