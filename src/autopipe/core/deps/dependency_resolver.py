"""
Unified dependency resolver for intelligent pipeline dependency management.

This module provides the core dependency resolution logic that automatically
matches step dependencies with compatible outputs from other steps.
"""

from typing import Dict, List, Set, Optional, Tuple
import logging
from ..base import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType
)
from .property_reference import PropertyReference
from .specification_registry import SpecificationRegistry
from .semantic_matcher import SemanticMatcher

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """Raised when dependencies cannot be resolved."""
    pass


class UnifiedDependencyResolver:
    """Intelligent dependency resolver using declarative specifications."""
    
    def __init__(self, registry: SpecificationRegistry, semantic_matcher: SemanticMatcher):
        """
        Initialize the dependency resolver.
        
        Args:
            registry: Specification registry
            semantic_matcher: Semantic matcher for name similarity calculations
        """
        self.registry = registry
        self.semantic_matcher = semantic_matcher
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}
        
    def register_specification(self, step_name: str, spec: StepSpecification):
        """Register a step specification with the resolver."""
        self.registry.register(step_name, spec)
        # Clear cache when new specifications are added
        self._resolution_cache.clear()
        
    def resolve_all_dependencies(self, available_steps: List[str]) -> Dict[str, Dict[str, PropertyReference]]:
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
                step_dependencies = self.resolve_step_dependencies(step_name, available_steps)
                if step_dependencies:
                    resolved[step_name] = step_dependencies
                    logger.info(f"Successfully resolved {len(step_dependencies)} dependencies for step '{step_name}'")
            except DependencyResolutionError as e:
                unresolved_steps.append((step_name, str(e)))
                logger.error(f"Failed to resolve dependencies for step '{step_name}': {e}")
        
        if unresolved_steps:
            error_details = "\n".join([f"  - {step}: {error}" for step, error in unresolved_steps])
            logger.warning(f"Some steps have unresolved dependencies:\n{error_details}")
        
        return resolved
    
    def resolve_step_dependencies(self, consumer_step: str, 
                                available_steps: List[str]) -> Dict[str, PropertyReference]:
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
                logger.warning(f"Could not resolve required dependency: {consumer_step}.{dep_name}")
            else:
                logger.info(f"Optional dependency not resolved: {consumer_step}.{dep_name}")
        
        if unresolved:
            raise DependencyResolutionError(
                f"Step '{consumer_step}' has unresolved required dependencies: {unresolved}"
            )
        
        # Cache the result
        self._resolution_cache[cache_key] = resolved
        return resolved
    
    def _resolve_single_dependency(self, dep_spec: DependencySpec, consumer_step: str,
                                 available_steps: List[str]) -> Optional[PropertyReference]:
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
                confidence = self._calculate_compatibility(dep_spec, output_spec, provider_spec)
                if confidence > 0.5:  # Threshold for viable candidates
                    prop_ref = PropertyReference(
                        step_name=provider_step,
                        output_spec=output_spec
                    )
                    candidates.append((prop_ref, confidence, provider_step, output_name))
        
        if candidates:
            # Sort by confidence (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_match = candidates[0]
            
            logger.info(f"Best match for {dep_spec.logical_name}: "
                       f"{best_match[2]}.{best_match[3]} (confidence: {best_match[1]:.3f})")
            
            # Log alternative matches if they exist
            if len(candidates) > 1:
                alternatives = [(c[2], c[3], c[1]) for c in candidates[1:3]]  # Top 2 alternatives
                logger.debug(f"Alternative matches: {alternatives}")
            
            return best_match[0]
        
        logger.debug(f"No compatible outputs found for dependency '{dep_spec.logical_name}' "
                    f"of type '{dep_spec.dependency_type.value}'")
        return None
    
    def _calculate_compatibility(self, dep_spec: DependencySpec, output_spec: OutputSpec,
                               provider_spec: StepSpecification) -> float:
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
        elif self._are_types_compatible(dep_spec.dependency_type, output_spec.output_type):
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
        
        # 4. Compatible source check (10% weight)
        if dep_spec.compatible_sources:
            if provider_spec.step_type in dep_spec.compatible_sources:
                score += 0.1
        else:
            # If no compatible sources specified, give small bonus for any match
            score += 0.05
        
        # 5. Keyword matching bonus (5% weight)
        if dep_spec.semantic_keywords:
            keyword_score = self._calculate_keyword_match(dep_spec.semantic_keywords, output_spec.logical_name)
            score += keyword_score * 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _are_types_compatible(self, dep_type: DependencyType, output_type: DependencyType) -> bool:
        """Check if dependency and output types are compatible."""
        # Define compatibility matrix
        compatibility_matrix = {
            DependencyType.MODEL_ARTIFACTS: [DependencyType.MODEL_ARTIFACTS],
            DependencyType.TRAINING_DATA: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
            DependencyType.PROCESSING_OUTPUT: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
            DependencyType.HYPERPARAMETERS: [DependencyType.HYPERPARAMETERS, DependencyType.CUSTOM_PROPERTY],
            DependencyType.PAYLOAD_SAMPLES: [DependencyType.PAYLOAD_SAMPLES, DependencyType.PROCESSING_OUTPUT],
            DependencyType.CUSTOM_PROPERTY: [DependencyType.CUSTOM_PROPERTY]
        }
        
        compatible_types = compatibility_matrix.get(dep_type, [])
        return output_type in compatible_types
    
    def _are_data_types_compatible(self, dep_data_type: str, output_data_type: str) -> bool:
        """Check if data types are compatible."""
        # Define data type compatibility
        compatibility_map = {
            'S3Uri': ['S3Uri', 'String'],  # S3Uri can sometimes be used as String
            'String': ['String', 'S3Uri'],  # String can sometimes accept S3Uri
            'Integer': ['Integer', 'Float'],  # Integer can be used as Float
            'Float': ['Float', 'Integer'],   # Float can accept Integer
            'Boolean': ['Boolean'],
        }
        
        compatible_types = compatibility_map.get(dep_data_type, [dep_data_type])
        return output_data_type in compatible_types
    
    def _calculate_keyword_match(self, keywords: List[str], output_name: str) -> float:
        """Calculate keyword matching score."""
        if not keywords:
            return 0.0
        
        output_lower = output_name.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in output_lower)
        return matches / len(keywords)
    
    def get_resolution_report(self, available_steps: List[str]) -> Dict[str, any]:
        """
        Generate a detailed resolution report for debugging.
        
        Args:
            available_steps: List of available step names
            
        Returns:
            Detailed report of resolution process
        """
        report = {
            'total_steps': len(available_steps),
            'registered_steps': len([s for s in available_steps if self.registry.get_specification(s)]),
            'step_details': {},
            'unresolved_dependencies': [],
            'resolution_summary': {}
        }
        
        for step_name in available_steps:
            spec = self.registry.get_specification(step_name)
            if not spec:
                continue
                
            step_report = {
                'step_type': spec.step_type,
                'total_dependencies': len(spec.dependencies),
                'required_dependencies': len(spec.list_required_dependencies()),
                'optional_dependencies': len(spec.list_optional_dependencies()),
                'outputs': len(spec.outputs),
                'resolved_dependencies': {},
                'unresolved_dependencies': []
            }
            
            try:
                resolved = self.resolve_step_dependencies(step_name, available_steps)
                step_report['resolved_dependencies'] = {
                    dep_name: str(prop_ref) for dep_name, prop_ref in resolved.items()
                }
                
                # Check for unresolved dependencies
                for dep_name, dep_spec in spec.dependencies.items():
                    if dep_name not in resolved and dep_spec.required:
                        step_report['unresolved_dependencies'].append(dep_name)
                        
            except DependencyResolutionError as e:
                step_report['error'] = str(e)
                report['unresolved_dependencies'].append(step_name)
            
            report['step_details'][step_name] = step_report
        
        # Generate summary
        total_deps = sum(len(spec.dependencies) for spec in self.registry._specifications.values())
        resolved_deps = sum(len(details.get('resolved_dependencies', {})) 
                          for details in report['step_details'].values())
        
        report['resolution_summary'] = {
            'total_dependencies': total_deps,
            'resolved_dependencies': resolved_deps,
            'resolution_rate': resolved_deps / total_deps if total_deps > 0 else 0.0,
            'steps_with_errors': len(report['unresolved_dependencies'])
        }
        
        return report
    
    def clear_cache(self):
        """Clear the resolution cache."""
        self._resolution_cache.clear()
        logger.debug("Dependency resolution cache cleared")


def create_dependency_resolver(registry: Optional[SpecificationRegistry] = None,
                             semantic_matcher: Optional[SemanticMatcher] = None) -> UnifiedDependencyResolver:
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
    'UnifiedDependencyResolver',
    'DependencyResolutionError',
    'create_dependency_resolver'
]
