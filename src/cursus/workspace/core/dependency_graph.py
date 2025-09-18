"""
Cross-workspace dependency graph management.

This module provides sophisticated dependency analysis capabilities for
cross-workspace pipeline validation, maintaining separation of concerns
from the unified step catalog system.
"""

import logging
from typing import Dict, Any, Set, List, Tuple, Optional
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class DependencyGraph:
    """
    Represents component dependency relationships across workspaces.
    
    Provides sophisticated dependency analysis algorithms including circular
    dependency detection, topological sorting, and cross-workspace validation.
    
    DESIGN PRINCIPLES:
    - Single Responsibility: Focuses on dependency relationship management
    - Separation of Concerns: Dependency logic separate from component discovery
    - Safety Critical: Prevents circular dependencies that break pipelines
    - Cross-Workspace Aware: Handles workspace boundary crossing
    """

    def __init__(self):
        """Initialize dependency graph."""
        self.nodes: Set[str] = set()
        self.edges: List[Tuple[str, str]] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self._adjacency_list: Optional[Dict[str, List[str]]] = None
        self._reverse_adjacency_list: Optional[Dict[str, List[str]]] = None
        self._dirty = True  # Track if adjacency lists need rebuilding

    def add_component(self, component_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Add component to dependency graph.
        
        Args:
            component_id: Unique identifier for the component (format: workspace_id:step_name)
            metadata: Optional metadata about the component
        """
        self.nodes.add(component_id)
        if metadata:
            self.metadata[component_id] = metadata
        self._dirty = True

    def add_dependency(self, from_component: str, to_component: str) -> None:
        """
        Add dependency relationship.
        
        Args:
            from_component: Component that depends on another
            to_component: Component that is depended upon
        """
        # Ensure both components are in the graph
        self.nodes.add(from_component)
        self.nodes.add(to_component)
        
        # Add the edge if it doesn't already exist
        edge = (from_component, to_component)
        if edge not in self.edges:
            self.edges.append(edge)
            self._dirty = True

    def remove_dependency(self, from_component: str, to_component: str) -> bool:
        """
        Remove dependency relationship.
        
        Args:
            from_component: Component that depends on another
            to_component: Component that is depended upon
            
        Returns:
            True if dependency was removed, False if it didn't exist
        """
        edge = (from_component, to_component)
        if edge in self.edges:
            self.edges.remove(edge)
            self._dirty = True
            return True
        return False

    def remove_component(self, component_id: str) -> None:
        """
        Remove component and all its dependencies.
        
        Args:
            component_id: Component to remove
        """
        if component_id in self.nodes:
            self.nodes.remove(component_id)
            
            # Remove all edges involving this component
            self.edges = [
                (from_comp, to_comp) for from_comp, to_comp in self.edges
                if from_comp != component_id and to_comp != component_id
            ]
            
            # Remove metadata
            if component_id in self.metadata:
                del self.metadata[component_id]
            
            self._dirty = True

    def get_dependencies(self, component_id: str) -> List[str]:
        """
        Get dependencies for a component (what it depends ON).
        
        Args:
            component_id: Component to get dependencies for
            
        Returns:
            List of components this component depends on
        """
        return [
            to_comp for from_comp, to_comp in self.edges 
            if from_comp == component_id
        ]

    def get_dependents(self, component_id: str) -> List[str]:
        """
        Get components that depend on this component (what depends ON it).
        
        Args:
            component_id: Component to get dependents for
            
        Returns:
            List of components that depend on this component
        """
        return [
            from_comp for from_comp, to_comp in self.edges 
            if to_comp == component_id
        ]

    def get_all_dependencies(self, component_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """
        Get all transitive dependencies for a component.
        
        Args:
            component_id: Component to get all dependencies for
            visited: Set of already visited components (for recursion)
            
        Returns:
            Set of all components this component transitively depends on
        """
        if visited is None:
            visited = set()
        
        if component_id in visited:
            return set()  # Avoid infinite recursion
        
        visited.add(component_id)
        all_deps = set()
        
        direct_deps = self.get_dependencies(component_id)
        for dep in direct_deps:
            all_deps.add(dep)
            all_deps.update(self.get_all_dependencies(dep, visited.copy()))
        
        return all_deps

    def get_all_dependents(self, component_id: str, visited: Optional[Set[str]] = None) -> Set[str]:
        """
        Get all transitive dependents for a component.
        
        Args:
            component_id: Component to get all dependents for
            visited: Set of already visited components (for recursion)
            
        Returns:
            Set of all components that transitively depend on this component
        """
        if visited is None:
            visited = set()
        
        if component_id in visited:
            return set()  # Avoid infinite recursion
        
        visited.add(component_id)
        all_dependents = set()
        
        direct_dependents = self.get_dependents(component_id)
        for dependent in direct_dependents:
            all_dependents.add(dependent)
            all_dependents.update(self.get_all_dependents(dependent, visited.copy()))
        
        return all_dependents

    def has_circular_dependencies(self) -> bool:
        """
        Check for circular dependencies using DFS with recursion stack tracking.
        
        This is a SAFETY CRITICAL method that prevents pipeline deployment
        with circular dependencies that would cause runtime deadlocks.
        
        Returns:
            True if circular dependencies exist, False otherwise
        """
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            """DFS helper function to detect cycles."""
            if node in rec_stack:
                # Back edge found - cycle detected!
                return True
            if node in visited:
                # Already processed this node completely
                return False

            # Mark as visited and add to recursion stack
            visited.add(node)
            rec_stack.add(node)

            # Check all dependencies
            for dep in self.get_dependencies(node):
                if has_cycle(dep):
                    return True

            # Remove from recursion stack (backtrack)
            rec_stack.remove(node)
            return False

        # Check each unvisited node
        for node in self.nodes:
            if node not in visited:
                if has_cycle(node):
                    return True

        return False

    def find_circular_dependencies(self) -> List[List[str]]:
        """
        Find all circular dependency paths.
        
        Returns:
            List of cycles, where each cycle is a list of component IDs
        """
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def find_cycles(node: str) -> None:
            """DFS helper to find all cycles."""
            if node in rec_stack:
                # Found a cycle - extract it from the path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in self.get_dependencies(node):
                find_cycles(dep)

            rec_stack.remove(node)
            path.pop()

        for node in self.nodes:
            if node not in visited:
                find_cycles(node)

        return cycles

    def topological_sort(self) -> Optional[List[str]]:
        """
        Perform topological sort to get execution order.
        
        Returns:
            List of components in topological order, or None if cycles exist
        """
        if self.has_circular_dependencies():
            return None

        # Kahn's algorithm
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for node in self.nodes:
            in_degree[node] = 0
        
        for from_comp, to_comp in self.edges:
            in_degree[to_comp] += 1

        # Queue of nodes with no incoming edges
        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Remove this node and update in-degrees
            for dep in self.get_dependencies(node):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)

        return result if len(result) == len(self.nodes) else None

    def get_execution_levels(self) -> Optional[List[List[str]]]:
        """
        Get components grouped by execution level (parallel execution groups).
        
        Returns:
            List of levels, where each level is a list of components that can run in parallel
        """
        if self.has_circular_dependencies():
            return None

        levels = []
        remaining_nodes = self.nodes.copy()
        
        while remaining_nodes:
            # Find nodes with no dependencies among remaining nodes
            current_level = []
            for node in remaining_nodes:
                deps = self.get_dependencies(node)
                if not any(dep in remaining_nodes for dep in deps):
                    current_level.append(node)
            
            if not current_level:
                # This shouldn't happen if there are no cycles
                logger.error("Unable to determine execution levels - possible cycle")
                return None
            
            levels.append(current_level)
            remaining_nodes -= set(current_level)

        return levels

    def get_cross_workspace_dependencies(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Analyze cross-workspace dependencies.
        
        Returns:
            Dictionary mapping workspace pairs to their dependency relationships
        """
        cross_workspace_deps = defaultdict(lambda: defaultdict(list))
        
        for from_comp, to_comp in self.edges:
            # Extract workspace IDs
            from_workspace = from_comp.split(':')[0] if ':' in from_comp else 'unknown'
            to_workspace = to_comp.split(':')[0] if ':' in to_comp else 'unknown'
            
            # Only track cross-workspace dependencies
            if from_workspace != to_workspace:
                cross_workspace_deps[from_workspace][to_workspace].append((from_comp, to_comp))
        
        return dict(cross_workspace_deps)

    def get_workspace_components(self, workspace_id: str) -> Set[str]:
        """
        Get all components belonging to a specific workspace.
        
        Args:
            workspace_id: Workspace identifier
            
        Returns:
            Set of component IDs in the workspace
        """
        return {
            node for node in self.nodes 
            if node.startswith(f"{workspace_id}:")
        }

    def validate_cross_workspace_access(self) -> Dict[str, Any]:
        """
        Validate that cross-workspace dependencies are valid.
        
        Returns:
            Validation report with issues and warnings
        """
        report = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "cross_workspace_deps": self.get_cross_workspace_dependencies()
        }
        
        # Check for missing workspace prefixes
        for node in self.nodes:
            if ':' not in node:
                report["warnings"].append(f"Component {node} missing workspace prefix")
        
        # Check for self-dependencies
        for from_comp, to_comp in self.edges:
            if from_comp == to_comp:
                report["issues"].append(f"Self-dependency detected: {from_comp}")
                report["valid"] = False
        
        return report

    def get_impact_analysis(self, component_id: str) -> Dict[str, Any]:
        """
        Analyze the impact of changing or removing a component.
        
        Args:
            component_id: Component to analyze
            
        Returns:
            Impact analysis report
        """
        if component_id not in self.nodes:
            return {"error": f"Component {component_id} not found in graph"}
        
        direct_dependents = self.get_dependents(component_id)
        all_dependents = self.get_all_dependents(component_id)
        direct_dependencies = self.get_dependencies(component_id)
        all_dependencies = self.get_all_dependencies(component_id)
        
        # Analyze workspace impact
        affected_workspaces = set()
        for dependent in all_dependents:
            workspace = dependent.split(':')[0] if ':' in dependent else 'unknown'
            affected_workspaces.add(workspace)
        
        return {
            "component_id": component_id,
            "direct_dependents": direct_dependents,
            "total_affected_components": len(all_dependents),
            "affected_workspaces": list(affected_workspaces),
            "direct_dependencies": direct_dependencies,
            "total_dependencies": len(all_dependencies),
            "is_critical": len(direct_dependents) > 0,
            "impact_level": "high" if len(all_dependents) > 5 else "medium" if len(all_dependents) > 1 else "low"
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert graph to dictionary for serialization.
        
        Returns:
            Dictionary representation of the graph
        """
        return {
            "nodes": list(self.nodes),
            "edges": self.edges,
            "metadata": self.metadata,
            "statistics": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "has_cycles": self.has_circular_dependencies(),
                "cross_workspace_deps": len(self.get_cross_workspace_dependencies())
            }
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Load graph from dictionary.
        
        Args:
            data: Dictionary representation to load from
        """
        self.nodes = set(data.get("nodes", []))
        self.edges = data.get("edges", [])
        self.metadata = data.get("metadata", {})
        self._dirty = True

    def clear(self) -> None:
        """Clear all components and dependencies."""
        self.nodes.clear()
        self.edges.clear()
        self.metadata.clear()
        self._dirty = True

    def merge(self, other: 'DependencyGraph') -> None:
        """
        Merge another dependency graph into this one.
        
        Args:
            other: Another DependencyGraph to merge
        """
        self.nodes.update(other.nodes)
        
        # Add edges that don't already exist
        for edge in other.edges:
            if edge not in self.edges:
                self.edges.append(edge)
        
        # Merge metadata
        self.metadata.update(other.metadata)
        self._dirty = True

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the dependency graph."""
        cross_workspace_deps = self.get_cross_workspace_dependencies()
        
        # Calculate workspace distribution
        workspace_counts = defaultdict(int)
        for node in self.nodes:
            workspace = node.split(':')[0] if ':' in node else 'unknown'
            workspace_counts[workspace] += 1
        
        return {
            "total_components": len(self.nodes),
            "total_dependencies": len(self.edges),
            "has_circular_dependencies": self.has_circular_dependencies(),
            "cross_workspace_dependencies": len(cross_workspace_deps),
            "workspace_distribution": dict(workspace_counts),
            "average_dependencies_per_component": len(self.edges) / len(self.nodes) if self.nodes else 0,
            "components_with_no_dependencies": len([n for n in self.nodes if not self.get_dependencies(n)]),
            "components_with_no_dependents": len([n for n in self.nodes if not self.get_dependents(n)])
        }

    def __len__(self) -> int:
        """Return number of components in the graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        """String representation of the dependency graph."""
        return (f"DependencyGraph(nodes={len(self.nodes)}, "
                f"edges={len(self.edges)}, "
                f"cycles={'yes' if self.has_circular_dependencies() else 'no'})")
