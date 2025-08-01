"""
Enhanced Pipeline DAG with port-based dependency management.

This module extends the base PipelineDAG with intelligent dependency resolution,
typed edges, and declarative step specifications.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import logging
from .base_dag import PipelineDAG
from .edge_types import DependencyEdge, EdgeCollection, EdgeType
from ...core.deps import (
    StepSpecification, UnifiedDependencyResolver, 
    DependencyResolutionError, SpecificationRegistry
)
from ...core.deps.property_reference import PropertyReference

logger = logging.getLogger(__name__)


class EnhancedPipelineDAG(PipelineDAG):
    """
    Enhanced version of PipelineDAG with port-based dependency management.
    
    Extends the existing PipelineDAG from pipeline_builder to add:
    - Typed input/output ports via step specifications
    - Intelligent dependency resolution
    - Property reference management
    - Semantic matching capabilities
    - Enhanced validation and error reporting
    """
    
    def __init__(self, nodes: Optional[List[str]] = None, edges: Optional[List[tuple]] = None):
        """
        Initialize enhanced DAG.
        
        Args:
            nodes: Optional list of initial node names
            edges: Optional list of initial edges (for compatibility)
        """
        # Initialize the base PipelineDAG
        super().__init__(nodes, edges)
        
        # Enhanced data structures for port-based dependencies
        self.step_specifications: Dict[str, StepSpecification] = {}
        self.dependency_edges = EdgeCollection()
        self.property_references: Dict[str, Dict[str, PropertyReference]] = {}
        
        # Dependency resolution components
        self.resolver = UnifiedDependencyResolver()
        self.registry = self.resolver.registry
        
        # Resolution state
        self._resolution_performed = False
        self._validation_errors: List[str] = []
        
    def register_step_specification(self, step_name: str, specification: StepSpecification):
        """
        Register a step specification defining its input/output ports.
        
        Args:
            step_name: Name of the step
            specification: Step specification with dependencies and outputs
        """
        if not isinstance(specification, StepSpecification):
            raise ValueError("specification must be a StepSpecification instance")
        
        self.step_specifications[step_name] = specification
        self.resolver.register_specification(step_name, specification)
        self.add_node(step_name)  # Use inherited method from base PipelineDAG
        
        # Clear resolution state when specifications change
        self._resolution_performed = False
        self._validation_errors.clear()
        
        logger.info(f"Registered specification for step '{step_name}' of type '{specification.step_type}'")
    
    def auto_resolve_dependencies(self, confidence_threshold: float = 0.6) -> List[DependencyEdge]:
        """
        Automatically resolve dependencies based on port compatibility.
        
        Args:
            confidence_threshold: Minimum confidence threshold for auto-resolution
            
        Returns:
            List of resolved dependency edges
        """
        available_steps = list(self.step_specifications.keys())
        
        try:
            # Resolve all dependencies
            resolved_deps = self.resolver.resolve_all_dependencies(available_steps)
            
            # Convert to dependency edges
            resolved_edges = []
            
            for consumer_step, dependencies in resolved_deps.items():
                consumer_spec = self.step_specifications[consumer_step]
                
                for dep_name, prop_ref in dependencies.items():
                    # Find the dependency specification
                    dep_spec = consumer_spec.get_dependency(dep_name)
                    if not dep_spec:
                        continue
                    
                    # Find the output specification
                    provider_spec = self.step_specifications.get(prop_ref.step_name)
                    if not provider_spec:
                        continue
                    
                    output_spec = provider_spec.get_output(prop_ref.output_spec.logical_name)
                    if not output_spec:
                        continue
                    
                    # Calculate confidence (simplified for now)
                    confidence = self._calculate_edge_confidence(dep_spec, output_spec, provider_spec)
                    
                    if confidence >= confidence_threshold:
                        edge = DependencyEdge(
                            source_step=prop_ref.step_name,
                            target_step=consumer_step,
                            source_output=output_spec.logical_name,
                            target_input=dep_name,
                            confidence=confidence,
                            metadata={
                                'auto_resolved': True,
                                'dependency_type': dep_spec.dependency_type.value,
                                'output_type': output_spec.output_type.value
                            }
                        )
                        
                        edge_id = self.dependency_edges.add_edge(edge)
                        resolved_edges.append(edge)
                        
                        # Also add to base DAG for compatibility
                        self.add_edge(prop_ref.step_name, consumer_step)
            
            # Store property references
            self.property_references = resolved_deps
            self._resolution_performed = True
            
            logger.info(f"Auto-resolved {len(resolved_edges)} dependencies with confidence >= {confidence_threshold}")
            return resolved_edges
            
        except DependencyResolutionError as e:
            logger.error(f"Dependency resolution failed: {e}")
            raise
    
    def add_manual_dependency(self, source_step: str, source_output: str,
                            target_step: str, target_input: str) -> DependencyEdge:
        """
        Manually add a dependency edge between steps.
        
        Args:
            source_step: Name of the source step
            source_output: Logical name of the source output
            target_step: Name of the target step
            target_input: Logical name of the target input
            
        Returns:
            Created dependency edge
        """
        # Validate steps exist
        if source_step not in self.step_specifications:
            raise ValueError(f"Source step '{source_step}' not registered")
        if target_step not in self.step_specifications:
            raise ValueError(f"Target step '{target_step}' not registered")
        
        # Validate ports exist
        source_spec = self.step_specifications[source_step]
        target_spec = self.step_specifications[target_step]
        
        if source_output not in source_spec.outputs:
            available = list(source_spec.outputs.keys())
            raise ValueError(f"Source output '{source_output}' not found. Available: {available}")
        
        if target_input not in target_spec.dependencies:
            available = list(target_spec.dependencies.keys())
            raise ValueError(f"Target input '{target_input}' not found. Available: {available}")
        
        # Create edge with full confidence (manual)
        edge = DependencyEdge(
            source_step=source_step,
            target_step=target_step,
            source_output=source_output,
            target_input=target_input,
            confidence=1.0,  # Full confidence for manual edges
            metadata={'manual': True}
        )
        
        self.dependency_edges.add_edge(edge)
        
        # Also add to base DAG for compatibility
        self.add_edge(source_step, target_step)
        
        # Update property references
        if target_step not in self.property_references:
            self.property_references[target_step] = {}
        
        source_output_spec = source_spec.get_output(source_output)
        self.property_references[target_step][target_input] = PropertyReference(
            step_name=source_step,
            output_spec=source_output_spec
        )
        
        logger.info(f"Added manual dependency: {edge}")
        return edge
    
    def get_step_dependencies(self, step_name: str) -> Dict[str, PropertyReference]:
        """
        Get resolved dependencies for a step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary mapping dependency names to property references
        """
        return self.property_references.get(step_name, {})
    
    def get_step_inputs_for_sagemaker(self, step_name: str) -> Dict[str, Any]:
        """
        Get step inputs formatted for SageMaker pipeline construction.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary of inputs formatted for SageMaker
        """
        dependencies = self.get_step_dependencies(step_name)
        sagemaker_inputs = {}
        
        for input_name, prop_ref in dependencies.items():
            sagemaker_inputs[input_name] = prop_ref.to_sagemaker_property()
        
        return sagemaker_inputs
    
    def validate_enhanced_dag(self) -> List[str]:
        """
        Enhanced validation including port compatibility and dependency resolution.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # First run base DAG validation
        try:
            execution_order = self.topological_sort()  # This will raise if cycles exist
            logger.debug(f"DAG execution order: {execution_order}")
        except ValueError as e:
            errors.append(f"DAG structure error: {e}")
        
        # Validate step specifications
        for step_name, spec in self.step_specifications.items():
            spec_errors = spec.validate()
            if spec_errors:
                errors.extend([f"Step '{step_name}': {error}" for error in spec_errors])
        
        # Validate dependency edges
        edge_errors = self.dependency_edges.validate_edges()
        errors.extend(edge_errors)
        
        # Check for unresolved required dependencies
        for step_name, spec in self.step_specifications.items():
            resolved_deps = self.get_step_dependencies(step_name)
            
            for dep_name, dep_spec in spec.dependencies.items():
                if dep_spec.required and dep_name not in resolved_deps:
                    errors.append(f"Step '{step_name}' has unresolved required dependency: {dep_name}")
        
        # Validate port compatibility for resolved edges
        for edge in self.dependency_edges.list_all_edges():
            source_spec = self.step_specifications.get(edge.source_step)
            target_spec = self.step_specifications.get(edge.target_step)
            
            if source_spec and target_spec:
                source_output = source_spec.get_output(edge.source_output)
                target_input = target_spec.get_dependency(edge.target_input)
                
                if source_output and target_input:
                    if source_output.output_type != target_input.dependency_type:
                        errors.append(f"Type mismatch in edge {edge}: "
                                    f"{source_output.output_type.value} -> {target_input.dependency_type.value}")
        
        self._validation_errors = errors
        return errors
    
    def get_execution_order(self) -> List[str]:
        """Get execution order using inherited topological_sort."""
        return self.topological_sort()  # Use base class method
    
    def get_dag_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the DAG."""
        stats = {
            'nodes': len(self.nodes),
            'base_edges': len(self.edges),
            'step_specifications': len(self.step_specifications),
            'dependency_edges': len(self.dependency_edges),
            'resolution_performed': self._resolution_performed,
            'validation_errors': len(self._validation_errors),
        }
        
        # Add edge statistics
        edge_stats = self.dependency_edges.get_statistics()
        stats.update({f'edge_{k}': v for k, v in edge_stats.items()})
        
        # Add dependency statistics
        total_deps = sum(len(spec.dependencies) for spec in self.step_specifications.values())
        resolved_deps = sum(len(deps) for deps in self.property_references.values())
        
        stats.update({
            'total_dependencies': total_deps,
            'resolved_dependencies': resolved_deps,
            'resolution_rate': resolved_deps / total_deps if total_deps > 0 else 0.0
        })
        
        return stats
    
    def get_resolution_report(self) -> Dict[str, Any]:
        """Get detailed resolution report for debugging."""
        available_steps = list(self.step_specifications.keys())
        return self.resolver.get_resolution_report(available_steps)
    
    def _calculate_edge_confidence(self, dep_spec, output_spec, provider_spec) -> float:
        """Calculate confidence score for an edge (simplified version)."""
        # This is a simplified version - the full calculation is in the resolver
        if dep_spec.dependency_type == output_spec.output_type:
            return 0.9
        else:
            return 0.7
    
    def clear_resolution_cache(self):
        """Clear dependency resolution cache."""
        self.resolver.clear_cache()
        self._resolution_performed = False
        self.property_references.clear()
        self.dependency_edges = EdgeCollection()
        logger.debug("Enhanced DAG resolution cache cleared")
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """Export DAG data for visualization tools."""
        return {
            'nodes': [
                {
                    'id': step_name,
                    'type': spec.step_type,
                    'dependencies': len(spec.dependencies),
                    'outputs': len(spec.outputs)
                }
                for step_name, spec in self.step_specifications.items()
            ],
            'edges': [
                {
                    'source': edge.source_step,
                    'target': edge.target_step,
                    'source_output': edge.source_output,
                    'target_input': edge.target_input,
                    'confidence': edge.confidence,
                    'type': edge.edge_type.value,
                    'auto_resolved': edge.is_auto_resolved()
                }
                for edge in self.dependency_edges.list_all_edges()
            ],
            'statistics': self.get_dag_statistics()
        }
    
    def __repr__(self):
        return (f"EnhancedPipelineDAG(nodes={len(self.nodes)}, "
                f"specifications={len(self.step_specifications)}, "
                f"edges={len(self.dependency_edges)})")
