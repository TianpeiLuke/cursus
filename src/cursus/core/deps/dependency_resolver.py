"""
Unified dependency resolver for intelligent pipeline dependency management.

This module provides the core dependency resolution logic that automatically
matches step dependencies with compatible outputs from other steps.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
import logging
from ..base import StepSpecification, DependencySpec, OutputSpec, DependencyType
from .property_reference import PropertyReference
from .specification_registry import SpecificationRegistry
from .semantic_matcher import SemanticMatcher

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """Raised when dependencies cannot be resolved."""

    pass


class UnifiedDependencyResolver:
    """Intelligent dependency resolver using declarative specifications."""

    def __init__(
        self, registry: SpecificationRegistry, semantic_matcher: SemanticMatcher
    ):
        """
        Initialize the dependency resolver.

        Args:
            registry: Specification registry
            semantic_matcher: Semantic matcher for name similarity calculations
        """
        self.registry = registry
        self.semantic_matcher = semantic_matcher
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}

    def register_specification(self, step_name: str, spec: StepSpecification) -> None:
        """Register a step specification with the resolver."""
        self.registry.register(step_name, spec)
        # Clear cache when new specifications are added
        self._resolution_cache.clear()

    def resolve_all_dependencies(
        self, available_steps: List[str]
    ) -> Dict[str, Dict[str, PropertyReference]]:
        """
        Resolve dependencies for all registered steps.

        Args:
            available_steps: List of step names that are available in the pipeline

        Returns:
            Dictionary mapping step names to their resolved dependencies
        """
        resolved = {}
        unresolved_steps = []

        for step_name in available_steps:
            try:
                step_dependencies = self.resolve_step_dependencies(
                    step_name, available_steps
                )
                if step_dependencies:
                    resolved[step_name] = step_dependencies
                    logger.info(
                        f"Successfully resolved {len(step_dependencies)} dependencies for step '{step_name}'"
                    )
            except DependencyResolutionError as e:
                unresolved_steps.append((step_name, str(e)))
                logger.error(
                    f"Failed to resolve dependencies for step '{step_name}': {e}"
                )

        if unresolved_steps:
            error_details = "\n".join(
                [f"  - {step}: {error}" for step, error in unresolved_steps]
            )
            logger.warning(f"Some steps have unresolved dependencies:\n{error_details}")

        return resolved

    def resolve_step_dependencies(
        self, consumer_step: str, available_steps: List[str]
    ) -> Dict[str, PropertyReference]:
        """
        Resolve dependencies for a single step.

        Args:
            consumer_step: Name of the step whose dependencies to resolve
            available_steps: List of available step names

        Returns:
            Dictionary mapping dependency names to property references
        """
        # Check cache first
        cache_key = f"{consumer_step}:{':'.join(sorted(available_steps))}"
        if cache_key in self._resolution_cache:
            logger.debug(f"Using cached resolution for step '{consumer_step}'")
            return self._resolution_cache[cache_key]

        consumer_spec = self.registry.get_specification(consumer_step)
        if not consumer_spec:
            logger.warning(f"No specification found for step: {consumer_step}")
            return {}

        resolved = {}
        unresolved = []

        for dep_name, dep_spec in consumer_spec.dependencies.items():
            resolution = self._resolve_single_dependency(
                dep_spec, consumer_step, available_steps
            )

            if resolution:
                resolved[dep_name] = resolution
                logger.info(f"Resolved {consumer_step}.{dep_name} -> {resolution}")
            elif dep_spec.required:
                unresolved.append(dep_name)
                logger.warning(
                    f"Could not resolve required dependency: {consumer_step}.{dep_name}"
                )
            else:
                logger.info(
                    f"Optional dependency not resolved: {consumer_step}.{dep_name}"
                )

        if unresolved:
            raise DependencyResolutionError(
                f"Step '{consumer_step}' has unresolved required dependencies: {unresolved}"
            )

        # Cache the result
        self._resolution_cache[cache_key] = resolved
        return resolved

    def resolve_with_scoring(
        self, consumer_step: str, available_steps: List[str]
    ) -> Dict[str, Any]:
        """
        Resolve dependencies with detailed compatibility scoring.

        Args:
            consumer_step: Name of the step whose dependencies to resolve
            available_steps: List of available step names

        Returns:
            Dictionary with resolved dependencies and detailed scoring information
        """
        consumer_spec = self.registry.get_specification(consumer_step)
        if not consumer_spec:
            logger.warning(f"No specification found for step: {consumer_step}")
            return {
                "resolved": {},
                "failed_with_scores": {},
                "resolution_details": {
                    "consumer_step": consumer_step,
                    "error": "No specification found",
                },
            }

        resolved = {}
        failed_with_scores = {}

        for dep_name, dep_spec in consumer_spec.dependencies.items():
            candidates = self._get_all_candidates_with_scores(
                dep_spec, consumer_step, available_steps
            )

            if candidates:
                best_match = candidates[0]  # Highest scoring candidate

                if best_match["score"] >= 0.5:  # Current resolution threshold
                    resolved[dep_name] = best_match["property_reference"]
                    logger.info(
                        f"Resolved {consumer_step}.{dep_name} -> {best_match['property_reference']} (score: {best_match['score']:.3f})"
                    )
                else:
                    # Store failed resolution with scoring details
                    failed_with_scores[dep_name] = {
                        "best_candidate": best_match,
                        "all_candidates": candidates[:3],  # Top 3 candidates
                        "required": dep_spec.required,
                    }
                    logger.debug(
                        f"Best match for {consumer_step}.{dep_name} below threshold: "
                        f"{best_match['provider_step']}.{best_match['output_name']} (score: {best_match['score']:.3f})"
                    )
            else:
                failed_with_scores[dep_name] = {
                    "best_candidate": None,
                    "all_candidates": [],
                    "required": dep_spec.required,
                }
                logger.debug(f"No candidates found for {consumer_step}.{dep_name}")

        return {
            "resolved": resolved,
            "failed_with_scores": failed_with_scores,
            "resolution_details": self._generate_resolution_details(
                consumer_step, available_steps
            ),
        }

    def _get_all_candidates_with_scores(
        self, dep_spec: DependencySpec, consumer_step: str, available_steps: List[str]
    ) -> List[Dict]:
        """
        Get all candidates with their compatibility scores.

        Args:
            dep_spec: Dependency specification to resolve
            consumer_step: Name of the consuming step
            available_steps: List of available step names

        Returns:
            List of candidate dictionaries sorted by score (highest first)
        """
        candidates = []

        for provider_step in available_steps:
            if provider_step == consumer_step:
                continue  # Skip self-dependencies

            provider_spec = self.registry.get_specification(provider_step)
            if not provider_spec:
                continue

            # Check each output of the provider step
            for output_name, output_spec in provider_spec.outputs.items():
                score = self._calculate_compatibility(
                    dep_spec, output_spec, provider_spec
                )
                if score > 0.0:  # Include all non-zero matches
                    score_breakdown = self._get_score_breakdown(
                        dep_spec, output_spec, provider_spec
                    )
                    candidates.append(
                        {
                            "provider_step": provider_step,
                            "output_name": output_name,
                            "output_spec": output_spec,
                            "score": score,
                            "property_reference": PropertyReference(
                                step_name=provider_step, output_spec=output_spec
                            ),
                            "score_breakdown": score_breakdown,
                        }
                    )

        # Sort by score (highest first)
        candidates.sort(key=lambda x: float(x["score"]) if isinstance(x["score"], (int, float, str)) else 0.0, reverse=True)
        return candidates

    def _get_score_breakdown(
        self,
        dep_spec: DependencySpec,
        output_spec: OutputSpec,
        provider_spec: StepSpecification,
    ) -> Dict[str, float]:
        """
        Get detailed breakdown of compatibility score components.

        Args:
            dep_spec: Dependency specification
            output_spec: Output specification
            provider_spec: Provider step specification

        Returns:
            Dictionary with score breakdown by component
        """
        breakdown = {}

        # 1. Dependency type compatibility (40% weight)
        if dep_spec.dependency_type == output_spec.output_type:
            breakdown["type_compatibility"] = 0.4
        elif self._are_types_compatible(
            dep_spec.dependency_type, output_spec.output_type
        ):
            breakdown["type_compatibility"] = 0.2
        else:
            breakdown["type_compatibility"] = 0.0

        # 2. Data type compatibility (20% weight)
        if dep_spec.data_type == output_spec.data_type:
            breakdown["data_type_compatibility"] = 0.2
        elif self._are_data_types_compatible(dep_spec.data_type, output_spec.data_type):
            breakdown["data_type_compatibility"] = 0.1
        else:
            breakdown["data_type_compatibility"] = 0.0

        # 3. Semantic name matching (25% weight)
        semantic_score = self.semantic_matcher.calculate_similarity_with_aliases(
            dep_spec.logical_name, output_spec
        )
        breakdown["semantic_similarity"] = semantic_score * 0.25

        # 4. Exact name match bonus (5% weight)
        if dep_spec.logical_name == output_spec.logical_name:
            breakdown["exact_match_bonus"] = 0.05
        elif dep_spec.logical_name in output_spec.aliases:
            breakdown["exact_match_bonus"] = 0.05
        else:
            breakdown["exact_match_bonus"] = 0.0

        # 5. Compatible source check with job type normalization (10% weight)
        if dep_spec.compatible_sources:
            # Normalize the provider step type for compatibility checking
            normalized_step_type = self._normalize_step_type_for_compatibility(
                provider_spec.step_type
            )

            if normalized_step_type in dep_spec.compatible_sources:
                breakdown["source_compatibility"] = 0.1
            else:
                breakdown["source_compatibility"] = 0.0
        else:
            breakdown["source_compatibility"] = (
                0.05  # Small bonus if no sources specified
            )

        # 6. Keyword matching (5% weight)
        if dep_spec.semantic_keywords:
            keyword_score = self._calculate_keyword_match(
                dep_spec.semantic_keywords, output_spec.logical_name
            )
            breakdown["keyword_matching"] = keyword_score * 0.05
        else:
            breakdown["keyword_matching"] = 0.0

        return breakdown

    def _generate_resolution_details(
        self, consumer_step: str, available_steps: List[str]
    ) -> Dict[str, Any]:
        """
        Generate detailed resolution context information.

        Args:
            consumer_step: Name of the consuming step
            available_steps: List of available step names

        Returns:
            Dictionary with resolution context details
        """
        consumer_spec = self.registry.get_specification(consumer_step)

        return {
            "consumer_step": consumer_step,
            "consumer_step_type": consumer_spec.step_type if consumer_spec else None,
            "total_dependencies": (
                len(consumer_spec.dependencies) if consumer_spec else 0
            ),
            "required_dependencies": (
                len([d for d in consumer_spec.dependencies.values() if d.required])
                if consumer_spec
                else 0
            ),
            "available_steps": available_steps,
            "available_step_count": len(available_steps),
            "registered_steps": len(
                [s for s in available_steps if self.registry.get_specification(s)]
            ),
            "resolution_threshold": 0.5,
        }

    def _resolve_single_dependency(
        self, dep_spec: DependencySpec, consumer_step: str, available_steps: List[str]
    ) -> Optional[PropertyReference]:
        """
        Resolve a single dependency with confidence scoring.

        Args:
            dep_spec: Dependency specification to resolve
            consumer_step: Name of the consuming step
            available_steps: List of available step names

        Returns:
            PropertyReference if resolution found, None otherwise
        """
        candidates = []

        for provider_step in available_steps:
            if provider_step == consumer_step:
                continue  # Skip self-dependencies

            provider_spec = self.registry.get_specification(provider_step)
            if not provider_spec:
                continue

            # Check each output of the provider step
            for output_name, output_spec in provider_spec.outputs.items():
                confidence = self._calculate_compatibility(
                    dep_spec, output_spec, provider_spec
                )
                if confidence > 0.5:  # Threshold for viable candidates
                    prop_ref = PropertyReference(
                        step_name=provider_step, output_spec=output_spec
                    )
                    candidates.append(
                        (prop_ref, confidence, provider_step, output_name)
                    )

        if candidates:
            # Sort by confidence (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_match = candidates[0]

            logger.info(
                f"Best match for {dep_spec.logical_name}: "
                f"{best_match[2]}.{best_match[3]} (confidence: {best_match[1]:.3f})"
            )

            # Log alternative matches if they exist
            if len(candidates) > 1:
                alternatives = [
                    (c[2], c[3], c[1]) for c in candidates[1:3]
                ]  # Top 2 alternatives
                logger.debug(f"Alternative matches: {alternatives}")

            return best_match[0]

        logger.debug(
            f"No compatible outputs found for dependency '{dep_spec.logical_name}' "
            f"of type '{dep_spec.dependency_type.value}'"
        )
        return None

    def _calculate_compatibility(
        self,
        dep_spec: DependencySpec,
        output_spec: OutputSpec,
        provider_spec: StepSpecification,
    ) -> float:
        """
        Calculate compatibility score between dependency and output.

        Args:
            dep_spec: Dependency specification
            output_spec: Output specification
            provider_spec: Provider step specification

        Returns:
            Compatibility score between 0.0 and 1.0
        """
        score = 0.0

        # 1. Dependency type compatibility (40% weight)
        if dep_spec.dependency_type == output_spec.output_type:
            score += 0.4
        elif self._are_types_compatible(
            dep_spec.dependency_type, output_spec.output_type
        ):
            score += 0.2
        else:
            # If types are not compatible at all, return 0
            return 0.0

        # 2. Data type compatibility (20% weight)
        if dep_spec.data_type == output_spec.data_type:
            score += 0.2
        elif self._are_data_types_compatible(dep_spec.data_type, output_spec.data_type):
            score += 0.1

        # 3. Enhanced semantic name matching with alias support (25% weight)
        semantic_score = self.semantic_matcher.calculate_similarity_with_aliases(
            dep_spec.logical_name, output_spec
        )
        score += semantic_score * 0.25

        # Optional: Add direct match bonus for exact matches
        if dep_spec.logical_name == output_spec.logical_name:
            score += 0.05  # Exact logical name match bonus
        elif dep_spec.logical_name in output_spec.aliases:
            score += 0.05  # Exact alias match bonus

        # 4. Compatible source check with job type normalization (10% weight)
        if dep_spec.compatible_sources:
            # Normalize the provider step type for compatibility checking
            normalized_step_type = self._normalize_step_type_for_compatibility(
                provider_spec.step_type
            )

            if normalized_step_type in dep_spec.compatible_sources:
                score += 0.1
        else:
            # If no compatible sources specified, give small bonus for any match
            score += 0.05

        # 5. Keyword matching bonus (5% weight)
        if dep_spec.semantic_keywords:
            keyword_score = self._calculate_keyword_match(
                dep_spec.semantic_keywords, output_spec.logical_name
            )
            score += keyword_score * 0.05

        return min(score, 1.0)  # Cap at 1.0

    def _are_types_compatible(
        self, dep_type: DependencyType, output_type: DependencyType
    ) -> bool:
        """Check if dependency and output types are compatible."""
        # Define compatibility matrix
        compatibility_matrix = {
            DependencyType.MODEL_ARTIFACTS: [DependencyType.MODEL_ARTIFACTS],
            DependencyType.TRAINING_DATA: [
                DependencyType.PROCESSING_OUTPUT,
                DependencyType.TRAINING_DATA,
            ],
            DependencyType.PROCESSING_OUTPUT: [
                DependencyType.PROCESSING_OUTPUT,
                DependencyType.TRAINING_DATA,
            ],
            DependencyType.HYPERPARAMETERS: [
                DependencyType.HYPERPARAMETERS,
                DependencyType.CUSTOM_PROPERTY,
            ],
            DependencyType.PAYLOAD_SAMPLES: [
                DependencyType.PAYLOAD_SAMPLES,
                DependencyType.PROCESSING_OUTPUT,
            ],
            DependencyType.CUSTOM_PROPERTY: [DependencyType.CUSTOM_PROPERTY],
        }

        compatible_types = compatibility_matrix.get(dep_type, [])
        return output_type in compatible_types

    def _are_data_types_compatible(
        self, dep_data_type: str, output_data_type: str
    ) -> bool:
        """Check if data types are compatible."""
        # Define data type compatibility
        compatibility_map = {
            "S3Uri": ["S3Uri", "String"],  # S3Uri can sometimes be used as String
            "String": ["String", "S3Uri"],  # String can sometimes accept S3Uri
            "Integer": ["Integer", "Float"],  # Integer can be used as Float
            "Float": ["Float", "Integer"],  # Float can accept Integer
            "Boolean": ["Boolean"],
        }

        compatible_types = compatibility_map.get(dep_data_type, [dep_data_type])
        return output_data_type in compatible_types

    def _normalize_step_type_for_compatibility(self, step_type: str) -> str:
        """
        Normalize step type by removing job type suffixes for compatibility checking.

        This handles the classical job type variants issue where step types like
        "TabularPreprocessing_Training" need to be normalized to "TabularPreprocessing"
        for compatibility checking against compatible_sources.

        Uses the centralized registry function to ensure consistency.

        Args:
            step_type: Original step type (e.g., "TabularPreprocessing_Training")

        Returns:
            Normalized step type (e.g., "TabularPreprocessing")
        """
        try:
            # Import here to avoid circular imports
            from ...registry.step_names import (
                get_step_name_from_spec_type,
                get_spec_step_type,
            )

            # Use the registry function to get canonical name, then get the base spec type
            canonical_name = get_step_name_from_spec_type(step_type)
            normalized = get_spec_step_type(canonical_name)

            if normalized != step_type:
                logger.debug(
                    f"Normalized step type '{step_type}' -> '{normalized}' for compatibility checking"
                )

            return normalized

        except Exception as e:
            # Fallback to manual normalization if registry lookup fails
            logger.debug(
                f"Registry normalization failed for '{step_type}': {e}, using fallback"
            )

            job_type_suffixes = ["_Training", "_Testing", "_Validation", "_Calibration"]
            for suffix in job_type_suffixes:
                if step_type.endswith(suffix):
                    normalized = step_type[: -len(suffix)]
                    logger.debug(
                        f"Fallback normalized step type '{step_type}' -> '{normalized}'"
                    )
                    return normalized

            return step_type

    def _calculate_keyword_match(self, keywords: List[str], output_name: str) -> float:
        """Calculate keyword matching score."""
        if not keywords:
            return 0.0

        output_lower = output_name.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in output_lower)
        return matches / len(keywords)

    def get_resolution_report(self, available_steps: List[str]) -> Dict[str, Any]:
        """
        Generate a detailed resolution report for debugging.

        Args:
            available_steps: List of available step names

        Returns:
            Detailed report of resolution process
        """
        report = {
            "total_steps": len(available_steps),
            "registered_steps": len(
                [s for s in available_steps if self.registry.get_specification(s)]
            ),
            "step_details": {},
            "unresolved_dependencies": [],
            "resolution_summary": {},
        }

        for step_name in available_steps:
            spec = self.registry.get_specification(step_name)
            if not spec:
                continue

            step_report = {
                "step_type": spec.step_type,
                "total_dependencies": len(spec.dependencies),
                "required_dependencies": len(spec.list_required_dependencies()),
                "optional_dependencies": len(spec.list_optional_dependencies()),
                "outputs": len(spec.outputs),
                "resolved_dependencies": {},
                "unresolved_dependencies": [],
            }

            try:
                resolved = self.resolve_step_dependencies(step_name, available_steps)
                step_report["resolved_dependencies"] = {
                    dep_name: str(prop_ref) for dep_name, prop_ref in resolved.items()
                }

                # Check for unresolved dependencies
                for dep_name, dep_spec in spec.dependencies.items():
                    if dep_name not in resolved and dep_spec.required:
                        step_report["unresolved_dependencies"].append(dep_name)

            except DependencyResolutionError as e:
                step_report["error"] = str(e)
                report["unresolved_dependencies"].append(step_name)

            step_details = report.get("step_details")
            if isinstance(step_details, dict):
                step_details[step_name] = step_report

        # Generate summary
        total_deps = sum(
            len(spec.dependencies) for spec in self.registry._specifications.values()
        )
        
        # Calculate resolved dependencies with explicit type handling
        resolved_deps = 0
        for details in report["step_details"].values():
            if isinstance(details, dict):
                resolved_deps += len(details.get("resolved_dependencies", {}))

        # Get unresolved dependencies list with proper typing
        unresolved_deps = report["unresolved_dependencies"]
        steps_with_errors = len(unresolved_deps) if isinstance(unresolved_deps, list) else 0

        report["resolution_summary"] = {
            "total_dependencies": total_deps,
            "resolved_dependencies": resolved_deps,
            "resolution_rate": resolved_deps / total_deps if total_deps > 0 else 0.0,
            "steps_with_errors": steps_with_errors,
        }

        return report

    def clear_cache(self) -> None:
        """Clear the resolution cache."""
        self._resolution_cache.clear()
        logger.debug("Dependency resolution cache cleared")


def create_dependency_resolver(
    registry: Optional[SpecificationRegistry] = None,
    semantic_matcher: Optional[SemanticMatcher] = None,
) -> UnifiedDependencyResolver:
    """
    Create a properly configured dependency resolver.

    Args:
        registry: Optional specification registry. If None, creates a new one.
        semantic_matcher: Optional semantic matcher. If None, creates a new one.

    Returns:
        Configured UnifiedDependencyResolver instance
    """
    registry = registry or SpecificationRegistry()
    semantic_matcher = semantic_matcher or SemanticMatcher()
    return UnifiedDependencyResolver(registry, semantic_matcher)


__all__ = [
    "UnifiedDependencyResolver",
    "DependencyResolutionError",
    "create_dependency_resolver",
]
