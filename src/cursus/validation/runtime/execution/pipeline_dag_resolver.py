"""Pipeline DAG resolver for execution planning."""

from typing import Dict, List, Optional, Set
import networkx as nx
from pydantic import BaseModel, Field
from pathlib import Path

# Use relative imports for external cursus modules
from ....api.dag import PipelineDAG
from ....core.base.config_base import BasePipelineConfig

class PipelineExecutionPlan(BaseModel):
    """Execution plan for pipeline with topological ordering."""
    execution_order: List[str]
    step_configs: Dict[str, dict]  # Using dict instead of StepConfig for Pydantic compatibility
    dependencies: Dict[str, List[str]]
    data_flow_map: Dict[str, Dict[str, str]]

class PipelineDAGResolver:
    """Resolves pipeline DAG into executable plan."""
    
    def __init__(self, dag: PipelineDAG):
        """Initialize with a Pipeline DAG."""
        self.dag = dag
        self.graph = self._build_networkx_graph()
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Convert pipeline DAG to NetworkX graph."""
        graph = nx.DiGraph()
        
        for step_name, step_config in self.dag.steps.items():
            graph.add_node(step_name, config=step_config)
            
            # Add dependency edges
            for dependency in step_config.depends_on:
                graph.add_edge(dependency, step_name)
        
        return graph
    
    def create_execution_plan(self) -> PipelineExecutionPlan:
        """Create topologically sorted execution plan."""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Pipeline contains cycles")
        
        execution_order = list(nx.topological_sort(self.graph))
        
        # Use dict representation for step_configs to work with Pydantic
        step_configs = {}
        for name in execution_order:
            config = self.graph.nodes[name]['config']
            # Convert StepConfig to dict for Pydantic compatibility
            if hasattr(config, '__dict__'):
                step_configs[name] = config.__dict__
            else:
                step_configs[name] = config
        
        dependencies = {
            name: list(self.graph.predecessors(name))
            for name in execution_order
        }
        
        data_flow_map = self._build_data_flow_map()
        
        return PipelineExecutionPlan(
            execution_order=execution_order,
            step_configs=step_configs,
            dependencies=dependencies,
            data_flow_map=data_flow_map
        )
    
    def _build_data_flow_map(self) -> Dict[str, Dict[str, str]]:
        """Map data flow between steps."""
        data_flow = {}
        
        for step_name in self.graph.nodes():
            step_config = self.graph.nodes[step_name]['config']
            inputs = {}
            
            # Map input dependencies
            for dep_step in self.graph.predecessors(step_name):
                dep_config = self.graph.nodes[dep_step]['config']
                # Map outputs of dependency to inputs of current step
                for output_key, output_path in dep_config.outputs.items():
                    if output_key in step_config.inputs:
                        inputs[output_key] = f"{dep_step}:{output_key}"
            
            data_flow[step_name] = inputs
        
        return data_flow
    
    def get_step_dependencies(self, step_name: str) -> List[str]:
        """Get immediate dependencies for a step."""
        if step_name not in self.graph.nodes():
            return []
        return list(self.graph.predecessors(step_name))
    
    def get_dependent_steps(self, step_name: str) -> List[str]:
        """Get steps that depend on the given step."""
        if step_name not in self.graph.nodes():
            return []
        return list(self.graph.successors(step_name))
    
    def validate_dag_integrity(self) -> Dict[str, List[str]]:
        """Validate DAG integrity and return issues if found."""
        issues = {}
        
        # Check for cycles
        try:
            list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            cycles = list(nx.simple_cycles(self.graph))
            issues["cycles"] = [f"Cycle detected: {' -> '.join(cycle)}" for cycle in cycles]
        
        # Check for dangling dependencies
        for step_name, step_config in self.dag.steps.items():
            for dependency in step_config.depends_on:
                if dependency not in self.dag.steps:
                    if "dangling_dependencies" not in issues:
                        issues["dangling_dependencies"] = []
                    issues["dangling_dependencies"].append(
                        f"Step {step_name} depends on non-existent step {dependency}"
                    )
        
        return issues
