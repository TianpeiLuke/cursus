"""
Specification registry for managing step specifications with context isolation.

This module provides the core registry functionality for storing, retrieving,
and managing step specifications within isolated contexts.
"""

from typing import Dict, List, Optional, Any
import logging
from ..base import StepSpecification, DependencySpec, OutputSpec

logger = logging.getLogger(__name__)


class SpecificationRegistry:
    """Context-aware registry for managing step specifications with isolation."""

    def __init__(self, context_name: str = "default"):
        """
        Initialize a context-scoped registry.

        Args:
            context_name: Name of the context this registry belongs to (e.g., pipeline name)
        """
        self.context_name = context_name
        self._specifications: Dict[str, StepSpecification] = {}
        self._step_type_to_names: Dict[str, List[str]] = {}
        logger.info(f"Created specification registry for context '{context_name}'")

    def register(self, step_name: str, specification: StepSpecification) -> None:
        """Register a step specification."""
        # Check if it's a StepSpecification by checking for required attributes
        # This is more robust than using isinstance which can fail with module reloading
        if (
            not hasattr(specification, "step_type")
            or not hasattr(specification, "node_type")
            or not hasattr(specification, "dependencies")
            or not hasattr(specification, "outputs")
        ):
            raise ValueError("specification must be a StepSpecification instance")

        # Validate the specification (Pydantic handles most validation automatically)
        errors = specification.validate_specification()
        if errors:
            raise ValueError(f"Invalid specification for '{step_name}': {errors}")

        self._specifications[step_name] = specification

        # Track step type mappings
        step_type = specification.step_type
        if step_type not in self._step_type_to_names:
            self._step_type_to_names[step_type] = []
        self._step_type_to_names[step_type].append(step_name)

        logger.info(
            f"Registered specification for step '{step_name}' of type '{step_type}' in context '{self.context_name}'"
        )

    def get_specification(self, step_name: str) -> Optional[StepSpecification]:
        """Get specification by step name."""
        return self._specifications.get(step_name)

    def get_specifications_by_type(self, step_type: str) -> List[StepSpecification]:
        """Get all specifications of a given step type."""
        step_names = self._step_type_to_names.get(step_type, [])
        return [self._specifications[name] for name in step_names]

    def list_step_names(self) -> List[str]:
        """Get list of all registered step names."""
        return list(self._specifications.keys())

    def list_step_types(self) -> List[str]:
        """Get list of all registered step types."""
        return list(self._step_type_to_names.keys())

    def find_compatible_outputs(self, dependency_spec: DependencySpec) -> List[tuple]:
        """Find outputs compatible with a dependency specification."""
        compatible = []

        for step_name, spec in self._specifications.items():
            for output_name, output_spec in spec.outputs.items():
                if self._are_compatible(dependency_spec, output_spec):
                    score = self._calculate_compatibility_score(
                        dependency_spec, output_spec, spec.step_type
                    )
                    compatible.append((step_name, output_name, output_spec, score))

        return sorted(compatible, key=lambda x: x[3], reverse=True)

    def _are_compatible(self, dep_spec: DependencySpec, out_spec: OutputSpec) -> bool:
        """Check basic compatibility between dependency and output."""
        # Type compatibility
        if dep_spec.dependency_type != out_spec.output_type:
            return False

        # Data type compatibility
        if dep_spec.data_type != out_spec.data_type:
            return False

        return True

    def _calculate_compatibility_score(
        self, dep_spec: DependencySpec, out_spec: OutputSpec, step_type: str
    ) -> float:
        """Calculate compatibility score between dependency and output."""
        score = 0.5  # Base compatibility score

        # Compatible source bonus
        if dep_spec.compatible_sources and step_type in dep_spec.compatible_sources:
            score += 0.3

        # Semantic keyword matching
        if dep_spec.semantic_keywords:
            keyword_matches = sum(
                1
                for keyword in dep_spec.semantic_keywords
                if keyword.lower() in out_spec.logical_name.lower()
            )
            score += (keyword_matches / len(dep_spec.semantic_keywords)) * 0.2

        return min(score, 1.0)  # Cap at 1.0

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"SpecificationRegistry(context='{self.context_name}', steps={len(self._specifications)})"
