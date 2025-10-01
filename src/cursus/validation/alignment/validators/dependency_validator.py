"""
Dependency Validator

Handles validation of dependencies between specifications including resolution,
circular dependency detection, and data type consistency checks.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from ..core.level3_validation_config import Level3ValidationConfig
from ....registry.step_names import (
    get_step_name_from_spec_type,
    get_canonical_name_from_file_name,
    get_all_step_names,
)
from ....core.deps.factory import create_pipeline_components
from ....core.base.specification_base import StepSpecification

logger = logging.getLogger(__name__)


class DependencyValidator:
    """
    Validates dependencies between step specifications.

    Features:
    - Enhanced dependency resolution with compatibility scoring
    - Circular dependency detection
    - Data type consistency validation
    - Integration with production registry for canonical name mapping
    """

    def __init__(self, validation_config: Level3ValidationConfig = None):
        """
        Initialize the dependency validator.

        Args:
            validation_config: Configuration for validation thresholds and behavior
        """
        self.config = (
            validation_config or Level3ValidationConfig.create_relaxed_config()
        )

        # Initialize dependency resolver components
        self.pipeline_components = create_pipeline_components("level3_validation")
        self.dependency_resolver = self.pipeline_components["resolver"]
        self.spec_registry = self.pipeline_components["registry"]

        # Log configuration
        threshold_desc = self.config.get_threshold_description()
        logger.debug(
            f"Dependency validator initialized with {threshold_desc['mode']} mode"
        )

    def validate_dependency_resolution(
        self,
        specification: Dict[str, Any],
        all_specs: Dict[str, Dict[str, Any]],
        spec_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Enhanced dependency validation with compatibility scoring.

        Args:
            specification: The specification to validate dependencies for
            all_specs: Dictionary of all available specifications
            spec_name: Name of the specification being validated

        Returns:
            List of validation issues
        """
        issues = []

        # Handle case where step has no dependencies
        dependencies = specification.get("dependencies", [])
        if not dependencies:
            logger.info(f"✅ {spec_name} has no dependencies - validation passed")
            return issues

        # Populate the resolver registry with all specifications
        self._populate_resolver_registry(all_specs)

        # Get available step names using canonical names from the registry (single source of truth)
        available_steps = self._get_available_canonical_step_names(all_specs)

        try:
            # Convert spec_name to canonical name for dependency resolution
            canonical_spec_name = self._get_canonical_step_name(spec_name)

            # Use enhanced resolution with scoring
            resolution_result = self.dependency_resolver.resolve_with_scoring(
                canonical_spec_name, available_steps
            )

            resolved_deps = resolution_result["resolved"]
            failed_deps = resolution_result["failed_with_scores"]

            # Process resolved dependencies
            for dep_name, prop_ref in resolved_deps.items():
                if self.config.LOG_SUCCESSFUL_RESOLUTIONS:
                    logger.info(f"✅ Resolved {spec_name}.{dep_name} -> {prop_ref}")

            # Process failed dependencies with scoring
            for dep_name, failure_info in failed_deps.items():
                best_candidate = failure_info["best_candidate"]
                is_required = failure_info["required"]

                if best_candidate is None:
                    # No candidates found at all
                    if is_required:
                        issues.append(
                            {
                                "severity": "CRITICAL",
                                "category": "dependency_resolution",
                                "message": f"No compatible candidates found for required dependency: {dep_name}",
                                "details": {
                                    "logical_name": dep_name,
                                    "specification": spec_name,
                                    "available_steps": available_steps,
                                    "candidates_found": 0,
                                },
                                "recommendation": f"Ensure a step exists that produces output compatible with {dep_name}",
                            }
                        )
                    else:
                        # Optional dependency with no candidates - just log
                        if self.config.LOG_FAILED_RESOLUTIONS:
                            logger.debug(
                                f"Optional dependency {spec_name}.{dep_name} has no compatible candidates"
                            )
                else:
                    # Candidates found but below resolution threshold
                    score = best_candidate["score"]
                    severity = self.config.determine_severity_from_score(
                        score, is_required
                    )

                    # Only create issues for dependencies that don't pass validation
                    if not self.config.should_pass_validation(score):
                        issue = {
                            "severity": severity,
                            "category": "dependency_compatibility",
                            "message": f"Dependency {dep_name} has low compatibility score: {score:.3f}",
                            "details": {
                                "logical_name": dep_name,
                                "specification": spec_name,
                                "best_match": {
                                    "provider": best_candidate["provider_step"],
                                    "output": best_candidate["output_name"],
                                    "score": score,
                                },
                                "required": is_required,
                                "threshold_info": self.config.get_threshold_description(),
                            },
                            "recommendation": self._generate_compatibility_recommendation(
                                dep_name, best_candidate
                            ),
                        }

                        # Add score breakdown if configured
                        if self.config.INCLUDE_SCORE_BREAKDOWN:
                            issue["details"]["score_breakdown"] = best_candidate[
                                "score_breakdown"
                            ]

                        # Add alternative candidates if configured
                        if self.config.INCLUDE_ALTERNATIVE_CANDIDATES:
                            issue["details"]["all_candidates"] = [
                                {
                                    "provider": c["provider_step"],
                                    "output": c["output_name"],
                                    "score": c["score"],
                                }
                                for c in failure_info["all_candidates"][
                                    : self.config.MAX_ALTERNATIVE_CANDIDATES
                                ]
                            ]

                        issues.append(issue)

                    # Log the best attempt for transparency
                    if self.config.LOG_FAILED_RESOLUTIONS:
                        logger.info(
                            f"🔍 Best match for {spec_name}.{dep_name}: "
                            f"{best_candidate['provider_step']}.{best_candidate['output_name']} "
                            f"(score: {score:.3f}, threshold: {self.config.PASS_THRESHOLD:.1f})"
                        )

        except Exception as e:
            issues.append(
                {
                    "severity": "ERROR",
                    "category": "resolver_error",
                    "message": f"Dependency resolver failed: {str(e)}",
                    "details": {"specification": spec_name, "error": str(e)},
                    "recommendation": "Check specification format and dependency resolver configuration",
                }
            )

        return issues

    def validate_circular_dependencies(
        self,
        specification: Dict[str, Any],
        all_specs: Dict[str, Dict[str, Any]],
        spec_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Validate that no circular dependencies exist.

        Args:
            specification: The specification to validate
            all_specs: Dictionary of all available specifications
            spec_name: Name of the specification being validated

        Returns:
            List of validation issues
        """
        issues = []

        # Build dependency graph
        dependency_graph = {}
        for spec_name_key, spec in all_specs.items():
            dependencies = []
            for dep in spec.get("dependencies", []):
                logical_name = dep.get("logical_name")
                if logical_name:
                    # Find which spec produces this logical name
                    for producer_name, producer_spec in all_specs.items():
                        if producer_name == spec_name_key:
                            continue
                        for output in producer_spec.get("outputs", []):
                            if output.get("logical_name") == logical_name:
                                dependencies.append(producer_name)
                                break
            dependency_graph[spec_name_key] = dependencies

        # Check for circular dependencies using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor in dependency_graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        if spec_name in dependency_graph and has_cycle(spec_name):
            issues.append(
                {
                    "severity": "ERROR",
                    "category": "circular_dependencies",
                    "message": f"Circular dependency detected involving {spec_name}",
                    "details": {"specification": spec_name},
                    "recommendation": "Remove circular dependencies by restructuring the dependency chain",
                }
            )

        return issues

    def validate_dependency_data_types(
        self,
        specification: Dict[str, Any],
        all_specs: Dict[str, Dict[str, Any]],
        spec_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Validate data type consistency across dependency chains.

        Args:
            specification: The specification to validate
            all_specs: Dictionary of all available specifications
            spec_name: Name of the specification being validated

        Returns:
            List of validation issues
        """
        issues = []

        dependencies = specification.get("dependencies", [])

        for dep in dependencies:
            logical_name = dep.get("logical_name")
            expected_type = dep.get("data_type")

            if not logical_name or not expected_type:
                continue

            # Find the producer of this logical name
            producer_type = None
            producer_spec_name = None

            for other_spec_name, other_spec in all_specs.items():
                if other_spec_name == spec_name:
                    continue

                for output in other_spec.get("outputs", []):
                    if output.get("logical_name") == logical_name:
                        producer_type = output.get("data_type")
                        producer_spec_name = other_spec_name
                        break

                if producer_type:
                    break

            # Check type consistency
            if producer_type and producer_type != expected_type:
                issues.append(
                    {
                        "severity": "WARNING",
                        "category": "data_type_consistency",
                        "message": f"Data type mismatch for {logical_name}: expected={expected_type}, producer={producer_type}",
                        "details": {
                            "logical_name": logical_name,
                            "expected_type": expected_type,
                            "producer_type": producer_type,
                            "consumer": spec_name,
                            "producer": producer_spec_name,
                        },
                        "recommendation": f"Align data types for {logical_name} between producer and consumer",
                    }
                )

        return issues

    def get_dependency_resolution_report(
        self, all_specs: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate detailed dependency resolution report using production resolver.

        Args:
            all_specs: Dictionary of all available specifications

        Returns:
            Detailed resolution report
        """
        self._populate_resolver_registry(all_specs)

        # Convert file-based spec names to canonical names for the report
        available_steps = []
        for spec_name in all_specs.keys():
            try:
                canonical_name = self._get_canonical_step_name(spec_name)
                available_steps.append(canonical_name)
            except Exception as e:
                logger.warning(f"Could not get canonical name for {spec_name}: {e}")
                available_steps.append(spec_name)  # Fallback to file name

        return self.dependency_resolver.get_resolution_report(available_steps)

    def _generate_compatibility_recommendation(
        self, dep_name: str, best_candidate: Dict
    ) -> str:
        """Generate specific recommendations based on compatibility analysis."""
        if "score_breakdown" not in best_candidate:
            return f"Review dependency specification for {dep_name} and output specification for {best_candidate['output_name']}"

        score_breakdown = best_candidate["score_breakdown"]
        recommendations = []

        if score_breakdown.get("type_compatibility", 0) < 0.2:
            recommendations.append(
                f"Consider changing dependency type or output type for better compatibility"
            )

        if score_breakdown.get("semantic_similarity", 0) < 0.15:
            recommendations.append(
                f"Consider renaming '{dep_name}' or adding aliases to improve semantic matching"
            )

        if score_breakdown.get("source_compatibility", 0) < 0.05:
            recommendations.append(
                f"Add '{best_candidate['provider_step']}' to compatible_sources for {dep_name}"
            )

        if score_breakdown.get("data_type_compatibility", 0) < 0.1:
            recommendations.append(
                f"Align data types between dependency and output specifications"
            )

        if not recommendations:
            recommendations.append(
                f"Review dependency specification for {dep_name} and output specification for {best_candidate['output_name']}"
            )

        return "; ".join(recommendations)

    def _get_available_canonical_step_names(
        self, all_specs: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Get available canonical step names using the registry as single source of truth.

        This method queries the production registry to get the authoritative list of
        canonical step names, ensuring alignment with production dependency resolution.

        Args:
            all_specs: Dictionary of all loaded specifications

        Returns:
            List of canonical step names from the production registry
        """
        # Get canonical step names from the production registry (single source of truth)
        canonical_names = get_all_step_names()

        logger.debug(f"Available canonical step names from registry: {canonical_names}")
        return canonical_names

    def _get_canonical_step_name(self, spec_file_name: str) -> str:
        """
        Convert specification file name to canonical step name using the registry.

        Uses the centralized FILE_NAME_TO_CANONICAL mapping as the single source of truth.

        Args:
            spec_file_name: File-based specification name (e.g., "dummy_training", "model_calibration", "model_evaluation_xgb")

        Returns:
            Canonical step name from the registry
        """
        try:
            # Use the centralized registry mapping (single source of truth)
            canonical_name = get_canonical_name_from_file_name(spec_file_name)
            logger.debug(
                f"Mapped spec file '{spec_file_name}' -> canonical '{canonical_name}' (registry)"
            )
            return canonical_name
        except ValueError as e:
            logger.debug(f"Registry mapping failed for '{spec_file_name}': {e}")

        # Final fallback: Convert file name to spec_type format and try registry lookup
        parts = spec_file_name.split("_")

        # Handle job type variants
        job_type_suffixes = ["training", "validation", "testing", "calibration"]
        job_type = None
        base_parts = parts

        if len(parts) > 1 and parts[-1] in job_type_suffixes:
            job_type = parts[-1]
            base_parts = parts[:-1]

        # Convert to PascalCase for spec_type
        spec_type_base = "".join(word.capitalize() for word in base_parts)

        if job_type:
            spec_type = f"{spec_type_base}_{job_type.capitalize()}"
        else:
            spec_type = spec_type_base

        # Use production function to get canonical name (strips job type suffix)
        try:
            canonical_name = get_step_name_from_spec_type(spec_type)
            logger.debug(
                f"Mapped spec file '{spec_file_name}' -> spec_type '{spec_type}' -> canonical '{canonical_name}' (final fallback)"
            )
            return canonical_name
        except Exception as e:
            # Ultimate fallback: return the base spec_type without job type suffix
            logger.warning(
                f"Failed to get canonical name for '{spec_file_name}' via all methods: {e}"
            )
            return spec_type_base

    def _populate_resolver_registry(self, all_specs: Dict[str, Dict[str, Any]]):
        """Populate the dependency resolver registry with all specifications using canonical names."""
        from ....core.base.specification_base import DependencySpec, OutputSpec

        for spec_name, spec_dict in all_specs.items():
            try:
                # Convert file-based spec name to canonical step name
                canonical_name = self._get_canonical_step_name(spec_name)

                # Convert dict back to StepSpecification object
                step_spec = self._dict_to_step_specification(spec_dict)

                # Register with canonical name
                self.dependency_resolver.register_specification(
                    canonical_name, step_spec
                )
                logger.debug(
                    f"Registered specification: '{spec_name}' as canonical '{canonical_name}'"
                )

            except Exception as e:
                logger.warning(f"Failed to register {spec_name} with resolver: {e}")

    def _dict_to_step_specification(self, spec_dict: Dict[str, Any]) -> StepSpecification:
        """Convert specification dictionary back to StepSpecification object."""
        from ....core.base.specification_base import DependencySpec, OutputSpec
        
        # Convert dependencies
        dependencies = {}
        for dep in spec_dict.get("dependencies", []):
            # Create DependencySpec using keyword arguments
            dep_data = {
                "logical_name": dep["logical_name"],
                "dependency_type": dep["dependency_type"],
                "required": dep["required"],
                "compatible_sources": dep.get("compatible_sources", []),
                "data_type": dep["data_type"],
                "description": dep.get("description", ""),
                "semantic_keywords": dep.get("semantic_keywords", []),
            }
            dep_spec = DependencySpec(**dep_data)
            dependencies[dep["logical_name"]] = dep_spec

        # Convert outputs
        outputs = {}
        for out in spec_dict.get("outputs", []):
            # Create OutputSpec using keyword arguments
            out_data = {
                "logical_name": out["logical_name"],
                "output_type": out["output_type"],
                "property_path": out["property_path"],
                "data_type": out["data_type"],
                "description": out.get("description", ""),
                "aliases": out.get("aliases", []),
            }
            out_spec = OutputSpec(**out_data)
            outputs[out["logical_name"]] = out_spec

        # Create StepSpecification using keyword arguments
        spec_data = {
            "step_type": spec_dict["step_type"],
            "node_type": spec_dict["node_type"],
            "dependencies": dependencies,
            "outputs": outputs,
        }
        return StepSpecification(**spec_data)
