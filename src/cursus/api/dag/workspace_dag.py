"""
Workspace-aware DAG implementation.

This module extends the PipelineDAG to support workspace step configurations
and cross-workspace dependency validation while maintaining compatibility.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from collections import defaultdict

from .base_dag import PipelineDAG

logger = logging.getLogger(__name__)


class WorkspaceAwareDAG(PipelineDAG):
    """DAG with workspace step support and cross-workspace dependencies."""
    
    def __init__(
        self,
        workspace_root: str,
        nodes: Optional[List[str]] = None,
        edges: Optional[List[tuple]] = None
    ):
        """
        Initialize workspace-aware DAG.
        
        Args:
            workspace_root: Root path of the workspace
            nodes: Optional list of step names
            edges: Optional list of (from_step, to_step) tuples
        """
        super().__init__(nodes, edges)
        self.workspace_root = workspace_root
        
        # Workspace-specific data structures
        self.workspace_steps: Dict[str, Dict[str, Any]] = {}
        self.developer_steps: Dict[str, List[str]] = defaultdict(list)
        self.step_types: Dict[str, str] = {}
        
        logger.info(f"Initialized workspace-aware DAG for: {workspace_root}")
    
    def add_workspace_step(
        self,
        step_name: str,
        developer_id: str,
        step_type: str,
        config_data: Dict[str, Any],
        dependencies: Optional[List[str]] = None
    ) -> None:
        """
        Add a workspace step to the DAG.
        
        Args:
            step_name: Name of the step
            developer_id: Developer workspace identifier
            step_type: Type of the step
            config_data: Step configuration data
            dependencies: Optional list of step dependencies
        """
        dependencies = dependencies or []
        
        # Create workspace step configuration
        workspace_step = {
            'step_name': step_name,
            'developer_id': developer_id,
            'step_type': step_type,
            'config_data': config_data,
            'workspace_root': self.workspace_root,
            'dependencies': dependencies
        }
        
        # Add to workspace tracking
        self.workspace_steps[step_name] = workspace_step
        self.developer_steps[developer_id].append(step_name)
        self.step_types[step_name] = step_type
        
        # Add to base DAG
        self.add_node(step_name)
        
        # Add dependency edges
        for dependency in dependencies:
            self.add_edge(dependency, step_name)
        
        logger.info(f"Added workspace step: {step_name} (developer: {developer_id}, type: {step_type})")
    
    def remove_workspace_step(self, step_name: str) -> bool:
        """
        Remove a workspace step from the DAG.
        
        Args:
            step_name: Name of the step to remove
            
        Returns:
            True if step was removed, False if not found
        """
        if step_name not in self.workspace_steps:
            return False
        
        workspace_step = self.workspace_steps[step_name]
        developer_id = workspace_step['developer_id']
        
        # Remove from workspace tracking
        del self.workspace_steps[step_name]
        if step_name in self.developer_steps[developer_id]:
            self.developer_steps[developer_id].remove(step_name)
        if step_name in self.step_types:
            del self.step_types[step_name]
        
        # Remove from base DAG
        if step_name in self.nodes:
            self.nodes.remove(step_name)
            
        # Remove edges
        self.edges = [(src, dst) for src, dst in self.edges if src != step_name and dst != step_name]
        
        # Rebuild adjacency lists
        self.adj_list = {n: [] for n in self.nodes}
        self.reverse_adj = {n: [] for n in self.nodes}
        for src, dst in self.edges:
            if src in self.adj_list and dst in self.reverse_adj:
                self.adj_list[src].append(dst)
                self.reverse_adj[dst].append(src)
        
        logger.info(f"Removed workspace step: {step_name}")
        return True
    
    def get_workspace_step(self, step_name: str) -> Optional[Dict[str, Any]]:
        """Get workspace step configuration by name."""
        return self.workspace_steps.get(step_name)
    
    def get_developers(self) -> List[str]:
        """Get list of unique developers in the DAG."""
        return list(self.developer_steps.keys())
    
    def get_steps_by_developer(self, developer_id: str) -> List[str]:
        """Get all step names for a specific developer."""
        return self.developer_steps.get(developer_id, [])
    
    def get_steps_by_type(self, step_type: str) -> List[str]:
        """Get all step names of a specific type."""
        return [step_name for step_name, stype in self.step_types.items() if stype == step_type]
    
    def validate_workspace_dependencies(self) -> Dict[str, Any]:
        """
        Validate workspace dependencies and cross-workspace references.
        
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'cross_workspace_dependencies': [],
            'dependency_stats': {
                'total_dependencies': 0,
                'cross_workspace_count': 0,
                'intra_workspace_count': 0
            }
        }
        
        try:
            for step_name, workspace_step in self.workspace_steps.items():
                for dependency in workspace_step['dependencies']:
                    validation_result['dependency_stats']['total_dependencies'] += 1
                    
                    # Check if dependency exists
                    if dependency not in self.workspace_steps:
                        validation_result['valid'] = False
                        validation_result['errors'].append(
                            f"Step '{step_name}' depends on '{dependency}' which is not defined"
                        )
                        continue
                    
                    # Check if it's a cross-workspace dependency
                    dep_step = self.workspace_steps[dependency]
                    if dep_step['developer_id'] != workspace_step['developer_id']:
                        validation_result['dependency_stats']['cross_workspace_count'] += 1
                        validation_result['cross_workspace_dependencies'].append({
                            'dependent_step': step_name,
                            'dependent_developer': workspace_step['developer_id'],
                            'dependency_step': dependency,
                            'dependency_developer': dep_step['developer_id']
                        })
                    else:
                        validation_result['dependency_stats']['intra_workspace_count'] += 1
            
            # Check for circular dependencies
            if self._has_workspace_cycles():
                validation_result['valid'] = False
                validation_result['errors'].append("Circular dependencies detected in workspace DAG")
            
            # Add warnings for complex cross-workspace dependencies
            cross_workspace_ratio = (
                validation_result['dependency_stats']['cross_workspace_count'] /
                max(validation_result['dependency_stats']['total_dependencies'], 1)
            )
            
            if cross_workspace_ratio > 0.5:
                validation_result['warnings'].append(
                    f"High ratio of cross-workspace dependencies ({cross_workspace_ratio:.1%})"
                )
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Dependency validation failed: {e}")
            logger.error(f"Error validating workspace dependencies: {e}")
        
        return validation_result
    
    def _has_workspace_cycles(self) -> bool:
        """Check for circular dependencies in workspace DAG."""
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True
    
    def to_workspace_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """
        Convert DAG to workspace pipeline configuration.
        
        Args:
            pipeline_name: Name for the pipeline
            
        Returns:
            Dictionary representing workspace pipeline configuration
        """
        steps = []
        for step_name, workspace_step in self.workspace_steps.items():
            steps.append({
                'step_name': workspace_step['step_name'],
                'developer_id': workspace_step['developer_id'],
                'step_type': workspace_step['step_type'],
                'config_data': workspace_step['config_data'],
                'workspace_root': workspace_step['workspace_root'],
                'dependencies': workspace_step['dependencies']
            })
        
        return {
            'pipeline_name': pipeline_name,
            'workspace_root': self.workspace_root,
            'steps': steps,
            'global_config': {}
        }
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get summary of workspace DAG structure."""
        developer_stats = {}
        step_type_stats = defaultdict(int)
        
        for developer_id, step_names in self.developer_steps.items():
            developer_stats[developer_id] = {
                'step_count': len(step_names),
                'step_types': list(set(self.step_types[step] for step in step_names if step in self.step_types))
            }
        
        for step_type in self.step_types.values():
            step_type_stats[step_type] += 1
        
        return {
            'workspace_root': self.workspace_root,
            'total_steps': len(self.workspace_steps),
            'total_edges': len(self.edges),
            'developers': list(self.developer_steps.keys()),
            'developer_stats': developer_stats,
            'step_type_stats': dict(step_type_stats),
            'dependency_validation': self.validate_workspace_dependencies()
        }
    
    def get_execution_order(self) -> List[Dict[str, Any]]:
        """
        Get execution order with workspace context.
        
        Returns:
            List of step execution information with workspace details
        """
        try:
            execution_order = []
            topological_order = self.topological_sort()
            
            for step_name in topological_order:
                workspace_step = self.workspace_steps.get(step_name)
                if workspace_step:
                    execution_order.append({
                        'step_name': step_name,
                        'developer_id': workspace_step['developer_id'],
                        'step_type': workspace_step['step_type'],
                        'dependencies': workspace_step['dependencies'],
                        'execution_index': len(execution_order)
                    })
                else:
                    # Handle non-workspace steps (for backward compatibility)
                    execution_order.append({
                        'step_name': step_name,
                        'developer_id': None,
                        'step_type': self.step_types.get(step_name, 'Unknown'),
                        'dependencies': self.get_dependencies(step_name),
                        'execution_index': len(execution_order)
                    })
            
            return execution_order
            
        except ValueError as e:
            logger.error(f"Cannot determine execution order: {e}")
            return []
    
    def analyze_workspace_complexity(self) -> Dict[str, Any]:
        """Analyze complexity metrics for the workspace DAG."""
        analysis = {
            'basic_metrics': {
                'node_count': len(self.nodes),
                'edge_count': len(self.edges),
                'developer_count': len(self.developer_steps),
                'step_type_count': len(set(self.step_types.values()))
            },
            'complexity_metrics': {},
            'developer_analysis': {},
            'recommendations': []
        }
        
        try:
            # Calculate complexity metrics
            if self.nodes:
                avg_dependencies = len(self.edges) / len(self.nodes)
                analysis['complexity_metrics']['avg_dependencies_per_step'] = avg_dependencies
                
                # Find steps with high fan-in/fan-out
                max_dependencies = max(len(self.get_dependencies(node)) for node in self.nodes)
                max_dependents = max(len(self.adj_list[node]) for node in self.nodes)
                
                analysis['complexity_metrics']['max_dependencies'] = max_dependencies
                analysis['complexity_metrics']['max_dependents'] = max_dependents
            
            # Analyze each developer's contribution
            for developer_id, step_names in self.developer_steps.items():
                dev_steps = [self.workspace_steps[name] for name in step_names if name in self.workspace_steps]
                
                analysis['developer_analysis'][developer_id] = {
                    'step_count': len(step_names),
                    'step_types': list(set(step['step_type'] for step in dev_steps)),
                    'avg_dependencies': sum(len(step['dependencies']) for step in dev_steps) / max(len(dev_steps), 1),
                    'cross_workspace_deps': sum(
                        1 for step in dev_steps 
                        for dep in step['dependencies'] 
                        if dep in self.workspace_steps and self.workspace_steps[dep]['developer_id'] != developer_id
                    )
                }
            
            # Generate recommendations
            if analysis['basic_metrics']['developer_count'] > 5:
                analysis['recommendations'].append("Consider splitting into smaller workspace groups")
            
            if analysis['complexity_metrics'].get('avg_dependencies_per_step', 0) > 3:
                analysis['recommendations'].append("High average dependencies may indicate tight coupling")
            
            cross_workspace_deps = sum(
                dev_analysis['cross_workspace_deps'] 
                for dev_analysis in analysis['developer_analysis'].values()
            )
            
            if cross_workspace_deps > len(self.edges) * 0.3:
                analysis['recommendations'].append("High cross-workspace dependencies may impact modularity")
            
        except Exception as e:
            analysis['error'] = str(e)
            logger.error(f"Error analyzing workspace complexity: {e}")
        
        return analysis
    
    @classmethod
    def from_workspace_config(cls, workspace_config: Dict[str, Any]) -> "WorkspaceAwareDAG":
        """
        Create workspace-aware DAG from workspace configuration.
        
        Args:
            workspace_config: Dictionary representing workspace pipeline configuration
            
        Returns:
            WorkspaceAwareDAG instance
        """
        dag = cls(workspace_root=workspace_config['workspace_root'])
        
        # Add all workspace steps
        for step in workspace_config['steps']:
            dag.add_workspace_step(
                step_name=step['step_name'],
                developer_id=step['developer_id'],
                step_type=step['step_type'],
                config_data=step['config_data'],
                dependencies=step.get('dependencies', [])
            )
        
        logger.info(f"Created workspace DAG from config with {len(workspace_config['steps'])} steps")
        return dag
    
    def clone(self) -> "WorkspaceAwareDAG":
        """Create a deep copy of the workspace DAG."""
        cloned_dag = WorkspaceAwareDAG(
            workspace_root=self.workspace_root,
            nodes=self.nodes.copy(),
            edges=self.edges.copy()
        )
        
        # Copy workspace-specific data
        cloned_dag.workspace_steps = {
            name: step.copy()
            for name, step in self.workspace_steps.items()
        }
        cloned_dag.developer_steps = {
            dev_id: steps.copy()
            for dev_id, steps in self.developer_steps.items()
        }
        cloned_dag.step_types = self.step_types.copy()
        
        return cloned_dag
    
    def merge_workspace_dag(self, other_dag: "WorkspaceAwareDAG") -> None:
        """
        Merge another workspace DAG into this one.
        
        Args:
            other_dag: Another WorkspaceAwareDAG to merge
            
        Raises:
            ValueError: If there are conflicting step names
        """
        # Check for conflicts
        conflicting_steps = set(self.workspace_steps.keys()) & set(other_dag.workspace_steps.keys())
        if conflicting_steps:
            raise ValueError(f"Conflicting step names: {conflicting_steps}")
        
        # Merge workspace steps
        for step_name, workspace_step in other_dag.workspace_steps.items():
            self.add_workspace_step(
                step_name=workspace_step['step_name'],
                developer_id=workspace_step['developer_id'],
                step_type=workspace_step['step_type'],
                config_data=workspace_step['config_data'],
                dependencies=workspace_step['dependencies']
            )
        
        logger.info(f"Merged workspace DAG with {len(other_dag.workspace_steps)} additional steps")
