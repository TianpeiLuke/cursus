"""Pipeline DAG resolver for execution planning."""

from typing import Dict, List, Optional, Set, Any
import networkx as nx
from pydantic import BaseModel, Field
from pathlib import Path
import logging
import importlib

# Use relative imports for external cursus modules
from . import PipelineDAG
from ...core.base.config_base import BasePipelineConfig
from ...core.base.contract_base import ScriptContract
from ...core.base.specification_base import StepSpecification
from ...core.compiler.config_resolver import StepConfigResolver
from ...core.compiler.exceptions import ConfigurationError
from ...registry.builder_registry import get_global_registry
from ...registry.step_names import (
    get_canonical_name_from_file_name,
    get_spec_step_type,
    get_step_name_from_spec_type
)

logger = logging.getLogger(__name__)

class PipelineExecutionPlan(BaseModel):
    """Execution plan for pipeline with topological ordering."""
    execution_order: List[str]
    step_configs: Dict[str, dict]  # Using dict instead of StepConfig for Pydantic compatibility
    dependencies: Dict[str, List[str]]
    data_flow_map: Dict[str, Dict[str, str]]

class PipelineDAGResolver:
    """Resolves pipeline DAG into executable plan with optional step config resolution."""
    
    def __init__(self, dag: PipelineDAG, config_path: Optional[str] = None, 
                 available_configs: Optional[Dict[str, BasePipelineConfig]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize with a Pipeline DAG and optional configuration support.
        
        Args:
            dag: PipelineDAG instance defining pipeline structure
            config_path: Path to configuration file (optional)
            available_configs: Pre-loaded configuration instances (optional)
            metadata: Configuration metadata for enhanced resolution (optional)
        """
        self.dag = dag
        self.graph = self._build_networkx_graph()
        self.config_path = config_path
        self.available_configs = available_configs or {}
        self.metadata = metadata
        self.config_resolver = StepConfigResolver() if (config_path or available_configs) else None
        
        # Load configs from file if path provided
        if config_path and not available_configs:
            try:
                self.available_configs = self._load_configs_from_file(config_path)
                logger.info(f"Loaded {len(self.available_configs)} configurations from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load configs from {config_path}: {e}")
                self.available_configs = {}
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Convert pipeline DAG to NetworkX graph."""
        graph = nx.DiGraph()
        
        # Add nodes from the DAG
        for node in self.dag.nodes:
            graph.add_node(node)
        
        # Add edges from the DAG
        for src, dst in self.dag.edges:
            graph.add_edge(src, dst)
        
        return graph
    
    def create_execution_plan(self) -> PipelineExecutionPlan:
        """Create topologically sorted execution plan with optional step config resolution."""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Pipeline contains cycles")
        
        execution_order = list(nx.topological_sort(self.graph))
        
        # Resolve step configs if available
        step_configs = {}
        if self.config_resolver and self.available_configs:
            try:
                logger.info(f"Resolving step configurations for {len(execution_order)} nodes")
                config_map = self.config_resolver.resolve_config_map(
                    dag_nodes=execution_order,
                    available_configs=self.available_configs,
                    metadata=self.metadata
                )
                
                # Convert to dict format for Pydantic compatibility
                for name, config in config_map.items():
                    if hasattr(config, '__dict__'):
                        step_configs[name] = config.__dict__
                    else:
                        step_configs[name] = config
                        
                logger.info(f"Successfully resolved configurations for {len(step_configs)} steps")
                
            except ConfigurationError as e:
                logger.warning(f"Could not resolve step configs: {e}")
                step_configs = {name: {} for name in execution_order}
        else:
            # Fallback: empty configs for base DAG without config support
            step_configs = {name: {} for name in execution_order}
            if not self.config_resolver:
                logger.debug("No config resolver available - using empty step configs")
        
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
        """Build data flow map using contract-based channel definitions."""
        data_flow = {}
        
        for step_name in self.graph.nodes():
            inputs = {}
            
            # Get step contract dynamically
            step_contract = self._discover_step_contract(step_name)
            if not step_contract:
                # Fallback to generic approach for backward compatibility
                for i, dep_step in enumerate(self.graph.predecessors(step_name)):
                    inputs[f"input_{i}"] = f"{dep_step}:output"
                data_flow[step_name] = inputs
                continue
            
            # Map each expected input channel to dependency outputs
            for input_channel, input_path in step_contract.expected_input_paths.items():
                # Find compatible output from dependencies
                for dep_step in self.graph.predecessors(step_name):
                    dep_contract = self._discover_step_contract(dep_step)
                    if dep_contract:
                        # Find compatible output channel
                        compatible_output = self._find_compatible_output(
                            input_channel, input_path,
                            dep_contract.expected_output_paths
                        )
                        if compatible_output:
                            inputs[input_channel] = f"{dep_step}:{compatible_output}"
                            break
                    else:
                        # Fallback for dependencies without contracts
                        inputs[f"input_from_{dep_step}"] = f"{dep_step}:output"
            
            data_flow[step_name] = inputs
        
        return data_flow
    
    def _discover_step_contract(self, step_name: str) -> Optional[ScriptContract]:
        """
        Dynamically discover step contract using registry helper functions.
        
        Args:
            step_name: Name of the step to discover contract for
            
        Returns:
            ScriptContract if found, None otherwise
        """
        try:
            # Convert step name to canonical name
            canonical_name = get_canonical_name_from_file_name(step_name)
            if not canonical_name:
                logger.debug(f"No canonical name found for step: {step_name}")
                return None
            
            # Get specification from canonical name
            step_spec = self._get_step_specification(canonical_name)
            if not step_spec:
                logger.debug(f"No specification found for canonical name: {canonical_name}")
                return None
            
            # Extract contract from specification
            if hasattr(step_spec, 'script_contract') and step_spec.script_contract:
                logger.debug(f"Found contract for step {step_name} via {canonical_name}")
                return step_spec.script_contract
            
            logger.debug(f"No script_contract found in specification for: {canonical_name}")
            return None
            
        except Exception as e:
            logger.warning(f"Failed to discover contract for step {step_name}: {e}")
            return None
    
    def _get_step_specification(self, canonical_name: str) -> Optional[StepSpecification]:
        """
        Get step specification from canonical name using dynamic import.
        
        Args:
            canonical_name: Canonical name of the step
            
        Returns:
            StepSpecification instance if found, None otherwise
        """
        try:
            # Get spec type from canonical name
            spec_type = get_spec_step_type(canonical_name)
            if not spec_type:
                logger.debug(f"No spec type found for canonical name: {canonical_name}")
                return None
            
            # Build module path using naming convention
            # Convert spec_type to module name (e.g., "XGBoostTrainingSpec" -> "xgboost_training_spec")
            module_name = self._spec_type_to_module_name(spec_type)
            module_path = f"cursus.steps.specs.{module_name}"
            
            # Dynamic import
            spec_module = importlib.import_module(module_path)
            
            # Get specification instance
            # Look for function that returns the spec (common pattern)
            spec_getter_name = f"get_{module_name}"
            if hasattr(spec_module, spec_getter_name):
                spec_getter = getattr(spec_module, spec_getter_name)
                return spec_getter()
            
            # Look for direct spec class instance
            if hasattr(spec_module, spec_type):
                spec_class = getattr(spec_module, spec_type)
                return spec_class()
            
            # Look for common spec variable names
            for var_name in ['SPEC', 'spec', f'{canonical_name.upper()}_SPEC']:
                if hasattr(spec_module, var_name):
                    return getattr(spec_module, var_name)
            
            logger.debug(f"No specification instance found in module: {module_path}")
            return None
            
        except ImportError as e:
            logger.debug(f"Could not import spec module for {canonical_name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error getting specification for {canonical_name}: {e}")
            return None
    
    def _spec_type_to_module_name(self, spec_type: str) -> str:
        """
        Convert spec type to module name using naming convention.
        
        Args:
            spec_type: Spec type (e.g., "XGBoostTrainingSpec")
            
        Returns:
            Module name (e.g., "xgboost_training_spec")
        """
        # Remove "Spec" suffix if present
        if spec_type.endswith("Spec"):
            spec_type = spec_type[:-4]
        
        # Convert CamelCase to snake_case
        import re
        module_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', spec_type).lower()
        return f"{module_name}_spec"
    
    def _find_compatible_output(self, input_channel: str, input_path: str, 
                               output_channels: Dict[str, str]) -> Optional[str]:
        """
        Find compatible output channel for given input requirements.
        
        Args:
            input_channel: Name of input channel
            input_path: Expected input path
            output_channels: Available output channels from dependency
            
        Returns:
            Compatible output channel name if found, None otherwise
        """
        # Strategy 1: Direct channel name matching
        if input_channel in output_channels:
            logger.debug(f"Direct channel match: {input_channel}")
            return input_channel
        
        # Strategy 2: Path-based compatibility
        for output_channel, output_path in output_channels.items():
            if self._are_paths_compatible(input_path, output_path):
                logger.debug(f"Path-compatible match: {output_channel} ({output_path} -> {input_path})")
                return output_channel
        
        # Strategy 3: Semantic matching for common patterns
        semantic_matches = {
            'input_path': ['output_path', 'model_path', 'data_path'],
            'model_path': ['model_output_path', 'output_path'],
            'data_path': ['output_path', 'processed_data_path'],
            'hyperparameters_s3_uri': ['config_path', 'hyperparameters_path']
        }
        
        if input_channel in semantic_matches:
            for candidate in semantic_matches[input_channel]:
                if candidate in output_channels:
                    logger.debug(f"Semantic match: {input_channel} -> {candidate}")
                    return candidate
        
        # Strategy 4: Fallback to first available output
        if output_channels:
            first_output = next(iter(output_channels.keys()))
            logger.debug(f"Fallback match: {input_channel} -> {first_output}")
            return first_output
        
        logger.debug(f"No compatible output found for input channel: {input_channel}")
        return None
    
    def _are_paths_compatible(self, input_path: str, output_path: str) -> bool:
        """
        Check if input and output paths are compatible based on SageMaker conventions.
        
        Args:
            input_path: Expected input path
            output_path: Available output path
            
        Returns:
            True if paths are compatible, False otherwise
        """
        # SageMaker path compatibility rules
        compatible_mappings = [
            ('/opt/ml/model', '/opt/ml/model'),  # Model artifacts
            ('/opt/ml/input/data', '/opt/ml/output/data'),  # Data flow
            ('/opt/ml/output', '/opt/ml/input/data'),  # Output to input
        ]
        
        for input_pattern, output_pattern in compatible_mappings:
            if input_pattern in input_path and output_pattern in output_path:
                return True
        
        # Generic compatibility: same base directory structure
        input_parts = Path(input_path).parts
        output_parts = Path(output_path).parts
        
        # Check if they share common directory structure
        if len(input_parts) >= 2 and len(output_parts) >= 2:
            if input_parts[-2:] == output_parts[-2:]:  # Same last two directory levels
                return True
        
        return False
    
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
        
        # Check for dangling dependencies (edges pointing to non-existent nodes)
        for src, dst in self.dag.edges:
            if src not in self.dag.nodes:
                if "dangling_dependencies" not in issues:
                    issues["dangling_dependencies"] = []
                issues["dangling_dependencies"].append(
                    f"Edge references non-existent source node: {src}"
                )
            if dst not in self.dag.nodes:
                if "dangling_dependencies" not in issues:
                    issues["dangling_dependencies"] = []
                issues["dangling_dependencies"].append(
                    f"Edge references non-existent destination node: {dst}"
                )
        
        # Check for isolated nodes (nodes with no edges)
        isolated_nodes = []
        for node in self.dag.nodes:
            if self.graph.degree(node) == 0:
                isolated_nodes.append(node)
        
        if isolated_nodes:
            issues["isolated_nodes"] = [f"Node has no connections: {node}" for node in isolated_nodes]
        
        return issues
    
    def _load_configs_from_file(self, config_path: str) -> Dict[str, BasePipelineConfig]:
        """
        Load configurations from file using the same pattern as DynamicPipelineTemplate.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary of loaded configuration instances
            
        Raises:
            ConfigurationError: If configs cannot be loaded
        """
        try:
            import json
            
            # Load the JSON configuration file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Extract metadata if available
            if 'metadata' in config_data:
                self.metadata = config_data['metadata']
                logger.debug("Loaded metadata from configuration file")
            
            # Use the base template's config loading mechanism
            # This is a simplified version - in practice, you might want to use
            # the full DynamicPipelineTemplate approach for config class detection
            configs = {}
            
            # For now, return empty dict and log that this needs full implementation
            logger.warning("Config loading from file needs full implementation with config class detection")
            logger.info(f"Configuration file structure: {list(config_data.keys())}")
            
            return configs
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configurations from {config_path}: {e}")
    
    def get_config_resolution_preview(self) -> Optional[Dict[str, Any]]:
        """
        Get a preview of how DAG nodes would be resolved to configurations.
        
        Returns:
            Preview information if config resolver is available, None otherwise
        """
        if not self.config_resolver or not self.available_configs:
            return None
        
        try:
            execution_order = list(nx.topological_sort(self.graph))
            return self.config_resolver.preview_resolution(
                dag_nodes=execution_order,
                available_configs=self.available_configs,
                metadata=self.metadata
            )
        except Exception as e:
            logger.warning(f"Failed to generate config resolution preview: {e}")
            return None
