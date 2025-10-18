"""
Script Execution Registry

Central state coordinator for DAG execution with sequential message passing.
This module implements the Script Execution Registry as designed in the comprehensive
architecture document, providing integration between script_dependency_matcher and
script_input_resolver layers.

Key Features:
- Sequential state updates via topological ordering
- Message passing between script executions
- Runtime data tracking (inputs, outputs, status)
- Integration coordination between layers
- Six clear integration points for layer coordination
"""

import logging
import time
from typing import Dict, Any, List, Optional, Iterator, Tuple, Set
from pathlib import Path

# Direct reuse of existing cursus infrastructure
from ...api.dag.base_dag import PipelineDAG
from ...step_catalog import StepCatalog
from .api import ScriptTestResult

logger = logging.getLogger(__name__)


class ScriptExecutionRegistry:
    """
    Central state coordinator that integrates both layers:
    
    Layer 1 (script_dependency_matcher): DAG-level orchestration
    Layer 2 (script_input_resolver): Script-level resolution
    
    Registry Role:
    - Maintains DAG execution state
    - Coordinates message passing between layers
    - Provides state interface for both layers
    - Ensures sequential consistency
    """
    
    def __init__(self, dag: PipelineDAG, step_catalog: Optional[StepCatalog] = None):
        self.dag = dag
        self.step_catalog = step_catalog or StepCatalog()
        self.execution_order = dag.topological_sort()
        
        # Central state store
        self._state = {
            'node_configs': {},        # Initial configurations per node
            'resolved_inputs': {},     # Resolved inputs per node (from script_input_resolver)
            'execution_outputs': {},   # Actual outputs per node (from execution)
            'dependency_graph': {},    # Dependency relationships for message passing
            'execution_status': {},    # Current status per node
            'message_log': []          # Message passing history for debugging
        }
        
        # Initialize execution status
        for node_name in self.dag.nodes:
            self._state['execution_status'][node_name] = 'pending'
        
        logger.info(f"ScriptExecutionRegistry initialized for DAG with {len(self.dag.nodes)} nodes")
    
    # === INTEGRATION POINT 1: Initialize from Dependency Matcher ===
    
    def initialize_from_dependency_matcher(self, prepared_data: Dict[str, Any]):
        """
        Integration Point 1: Receive prepared data from script_dependency_matcher.
        
        script_dependency_matcher calls this to initialize registry state.
        
        Args:
            prepared_data: Output from prepare_script_testing_inputs() containing:
                - node_specs: Loaded step specifications
                - dependency_matches: Automatic dependency matches
                - config_data: Extracted configuration data
                - execution_order: Topological sort order
        """
        self._state['node_configs'] = prepared_data.get('config_data', {})
        self._state['dependency_graph'] = prepared_data.get('dependency_matches', {})
        
        # Store node specifications for contract access
        self._node_specs = prepared_data.get('node_specs', {})
        
        # Update execution order if provided
        if 'execution_order' in prepared_data:
            self.execution_order = prepared_data['execution_order']
        
        # Initialize execution status
        for node_name in self.dag.nodes:
            self._state['execution_status'][node_name] = 'pending'
        
        logger.info(f"Registry initialized with {len(self._state['node_configs'])} node configurations")
        logger.debug(f"Dependency matches: {len(self._state['dependency_graph'])} nodes with matches")
    
    # === INTEGRATION POINT 2: Provide Dependency Outputs ===
    
    def get_dependency_outputs_for_node(self, node_name: str) -> Dict[str, str]:
        """
        Integration Point 2: Provide dependency outputs for message passing.
        
        script_dependency_matcher calls this to get outputs from completed dependencies.
        
        Args:
            node_name: Name of the node requesting dependency outputs
            
        Returns:
            Dictionary mapping output names to actual paths from completed dependencies
        """
        dependency_outputs = {}
        
        for dep_node in self.dag.get_dependencies(node_name):
            if dep_node in self._state['execution_outputs']:
                dep_outputs = self._state['execution_outputs'][dep_node]
                
                # Apply message passing mapping
                for output_key, output_path in dep_outputs.items():
                    # Direct mapping
                    dependency_outputs[output_key] = output_path
                    # Prefixed mapping to avoid conflicts
                    dependency_outputs[f"{dep_node}_{output_key}"] = output_path
                
                logger.debug(f"Provided {len(dep_outputs)} outputs from {dep_node} to {node_name}")
        
        logger.info(f"Collected {len(dependency_outputs)} dependency outputs for {node_name}")
        return dependency_outputs
    
    # === INTEGRATION POINT 3: Provide Node Config to Resolver ===
    
    def get_node_config_for_resolver(self, node_name: str) -> Dict[str, Any]:
        """
        Integration Point 3: Provide node config to script_input_resolver.
        
        script_input_resolver calls this to get base configuration for a node.
        
        Args:
            node_name: Name of the node requesting configuration
            
        Returns:
            Dictionary containing node configuration data
        """
        config = self._state['node_configs'].get(node_name, {})
        
        # Add specification if available
        if node_name in self._node_specs:
            config['spec'] = self._node_specs[node_name]
        
        logger.debug(f"Provided configuration for {node_name}: {len(config)} items")
        return config
    
    # === INTEGRATION POINT 4: Store Resolved Inputs ===
    
    def store_resolved_inputs(self, node_name: str, resolved_inputs: Dict[str, Any]):
        """
        Integration Point 4: Store resolved inputs from script_input_resolver.
        
        script_input_resolver calls this to store its resolution results.
        
        Args:
            node_name: Name of the node
            resolved_inputs: Dictionary containing resolved input configuration
        """
        self._state['resolved_inputs'][node_name] = resolved_inputs
        self._state['execution_status'][node_name] = 'ready'
        
        logger.debug(f"Stored resolved inputs for {node_name}: {len(resolved_inputs)} items")
        logger.info(f"Node {node_name} status updated to 'ready'")
    
    # === INTEGRATION POINT 5: Provide Ready Inputs ===
    
    def get_ready_node_inputs(self, node_name: str) -> Dict[str, Any]:
        """
        Integration Point 5: Provide complete inputs for script execution.
        
        API layer calls this to get final inputs for script execution.
        
        Args:
            node_name: Name of the node requesting inputs
            
        Returns:
            Dictionary containing complete input configuration for script execution
        """
        inputs = self._state['resolved_inputs'].get(node_name, {})
        logger.debug(f"Provided ready inputs for {node_name}: {len(inputs)} items")
        return inputs
    
    # === INTEGRATION POINT 6: Commit Execution Results ===
    
    def commit_execution_results(self, node_name: str, execution_result: ScriptTestResult):
        """
        Integration Point 6: Store execution results for message passing.
        
        API layer calls this after script execution to update state.
        
        Args:
            node_name: Name of the executed node
            execution_result: Result of script execution
        """
        if execution_result.success:
            self._state['execution_outputs'][node_name] = execution_result.output_files
            self._state['execution_status'][node_name] = 'completed'
            
            logger.info(f"✅ {node_name} execution committed: {len(execution_result.output_files)} outputs")
            logger.debug(f"Outputs: {list(execution_result.output_files.keys())}")
        else:
            self._state['execution_status'][node_name] = 'failed'
            logger.error(f"❌ {node_name} execution failed: {execution_result.error_message}")
    
    # === SEQUENTIAL STATE MANAGEMENT ===
    
    def sequential_state_update(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Generator that yields nodes in topological order with updated state.
        
        This ensures:
        - Dependencies are processed before dependents
        - State updates are sequential and consistent
        - Message passing happens in correct order
        
        Yields:
            Tuple of (node_name, updated_node_state) for each node in execution order
        """
        for node_name in self.execution_order:
            # Update node state based on current DAG state
            updated_node_state = self._update_node_state(node_name)
            
            # Yield node with its current state for execution
            yield node_name, updated_node_state
    
    def _update_node_state(self, node_name: str) -> Dict[str, Any]:
        """
        Update node state based on dependency outputs (message passing).
        
        This is where message passing algorithm executes:
        1. Get node's base configuration
        2. Apply message passing from completed dependencies
        3. Update registry state
        4. Return updated configuration for execution
        
        Args:
            node_name: Name of the node to update
            
        Returns:
            Updated node state with message passing applied
        """
        # Get base node configuration
        base_config = self._state['resolved_inputs'].get(node_name, {}).copy()
        
        if not base_config:
            logger.warning(f"No base configuration found for {node_name}")
            return {}
        
        # Apply message passing from dependencies
        for dep_node in self.dag.get_dependencies(node_name):
            if self._is_node_completed(dep_node):
                # Get dependency outputs
                dep_outputs = self._state['execution_outputs'].get(dep_node, {})
                
                # Apply message passing algorithm
                message_updates = self._apply_message_passing(dep_node, node_name, dep_outputs)
                
                # Update node inputs with messages
                if 'input_paths' not in base_config:
                    base_config['input_paths'] = {}
                base_config['input_paths'].update(message_updates)
                
                # Log message passing
                self._log_message_passing(dep_node, node_name, message_updates)
        
        # Update registry state
        self._state['resolved_inputs'][node_name] = base_config
        self._state['execution_status'][node_name] = 'ready'
        
        return base_config
    
    def _apply_message_passing(self, from_node: str, to_node: str, outputs: Dict[str, str]) -> Dict[str, str]:
        """
        Core message passing algorithm between nodes.
        
        Maps outputs from completed dependency nodes to inputs of current node.
        Uses intelligent naming conventions and contract-based matching.
        
        Args:
            from_node: Name of the dependency node providing outputs
            to_node: Name of the current node receiving inputs
            outputs: Dictionary of outputs from the dependency node
            
        Returns:
            Dictionary of message passing updates for the current node
        """
        message_updates = {}
        
        # Get current node's expected inputs (from contracts if available)
        expected_inputs = self._get_expected_inputs(to_node)
        
        for output_key, output_path in outputs.items():
            # Strategy 1: Direct name matching
            if output_key in expected_inputs:
                message_updates[output_key] = output_path
                logger.debug(f"📨 Direct mapping: {from_node}.{output_key} → {to_node}.{output_key}")
            
            # Strategy 2: Semantic mapping (model → model_path, data → training_data, etc.)
            semantic_mapping = self._get_semantic_mapping(output_key, expected_inputs)
            if semantic_mapping:
                message_updates[semantic_mapping] = output_path
                logger.debug(f"📨 Semantic mapping: {from_node}.{output_key} → {to_node}.{semantic_mapping}")
            
            # Strategy 3: Prefixed mapping (always available as fallback)
            prefixed_key = f"{from_node}_{output_key}"
            message_updates[prefixed_key] = output_path
            logger.debug(f"📨 Prefixed mapping: {from_node}.{output_key} → {to_node}.{prefixed_key}")
        
        return message_updates
    
    def _get_expected_inputs(self, node_name: str) -> Set[str]:
        """
        Get expected input names for a node from its specification.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Set of expected input names
        """
        expected_inputs = set()
        
        if node_name in self._node_specs:
            spec = self._node_specs[node_name]
            expected_inputs.update(spec.dependencies.keys())
        
        # Also check contract if available
        try:
            contract = self.step_catalog.load_contract_class(node_name)
            if contract and hasattr(contract, 'expected_input_paths'):
                expected_inputs.update(contract.expected_input_paths.keys())
        except Exception:
            pass  # Graceful fallback if contract unavailable
        
        return expected_inputs
    
    def _get_semantic_mapping(self, output_key: str, expected_inputs: Set[str]) -> Optional[str]:
        """
        Intelligent semantic mapping between output and input names.
        
        Examples:
        - 'model' output → 'model_path' input
        - 'processed_data' output → 'training_data' input  
        - 'features' output → 'feature_data' input
        
        Args:
            output_key: Name of the output from dependency node
            expected_inputs: Set of expected input names for current node
            
        Returns:
            Mapped input name if semantic mapping found, None otherwise
        """
        semantic_rules = {
            'model': ['model_path', 'model_file', 'trained_model'],
            'processed_data': ['training_data', 'input_data', 'data_path'],
            'features': ['feature_data', 'feature_file', 'features_path'],
            'predictions': ['prediction_data', 'results', 'output_data'],
            'data': ['training_data', 'input_data', 'data_path'],
            'output': ['input_data', 'data_path', 'input_file']
        }
        
        if output_key in semantic_rules:
            for candidate in semantic_rules[output_key]:
                if candidate in expected_inputs:
                    return candidate
        
        return None
    
    def _log_message_passing(self, from_node: str, to_node: str, message_data: Dict[str, str]):
        """
        Log message passing for debugging and analysis.
        
        Args:
            from_node: Name of the dependency node
            to_node: Name of the current node
            message_data: Dictionary of message passing data
        """
        message = {
            'from_node': from_node,
            'to_node': to_node,
            'message_data': message_data,
            'timestamp': time.time()
        }
        self._state['message_log'].append(message)
        
        logger.info(f"📨 Message passing: {from_node} → {to_node}")
        for key, value in message_data.items():
            logger.info(f"   {key}: {value}")
    
    def _is_node_completed(self, node_name: str) -> bool:
        """
        Check if a node has completed execution.
        
        Args:
            node_name: Name of the node to check
            
        Returns:
            True if node is completed, False otherwise
        """
        return self._state['execution_status'].get(node_name) == 'completed'
    
    # === STATE INSPECTION AND DEBUGGING ===
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of execution state and message passing.
        
        Returns:
            Dictionary with execution summary information
        """
        return {
            'registered_scripts': list(self._state['node_configs'].keys()),
            'completed_scripts': [
                node for node, state in self._state['execution_status'].items() 
                if state == 'completed'
            ],
            'failed_scripts': [
                node for node, state in self._state['execution_status'].items() 
                if state == 'failed'
            ],
            'ready_scripts': [
                node for node, state in self._state['execution_status'].items() 
                if state == 'ready'
            ],
            'pending_scripts': [
                node for node, state in self._state['execution_status'].items() 
                if state == 'pending'
            ],
            'message_count': len(self._state['message_log']),
            'execution_states': self._state['execution_status'].copy(),
            'total_nodes': len(self.dag.nodes)
        }
    
    def get_message_passing_history(self) -> List[Dict[str, Any]]:
        """
        Get complete message passing history for analysis.
        
        Returns:
            List of message passing events with timestamps
        """
        return self._state['message_log'].copy()
    
    def get_node_status(self, node_name: str) -> str:
        """
        Get current status of a specific node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Current status ('pending', 'ready', 'completed', 'failed')
        """
        return self._state['execution_status'].get(node_name, 'unknown')
    
    def get_node_outputs(self, node_name: str) -> Dict[str, str]:
        """
        Get outputs produced by a specific node.
        
        Args:
            node_name: Name of the node
            
        Returns:
            Dictionary of outputs produced by the node
        """
        return self._state['execution_outputs'].get(node_name, {})
    
    def clear_registry(self):
        """
        Clear registry for testing or new execution.
        
        Resets all state while preserving DAG and step catalog references.
        """
        self._state = {
            'node_configs': {},
            'resolved_inputs': {},
            'execution_outputs': {},
            'dependency_graph': {},
            'execution_status': {},
            'message_log': []
        }
        
        # Reinitialize execution status
        for node_name in self.dag.nodes:
            self._state['execution_status'][node_name] = 'pending'
        
        logger.info("Registry cleared and reinitialized")


class DAGStateConsistency:
    """
    Ensures state consistency during sequential message passing.
    
    Guarantees:
    1. Dependencies are always processed before dependents (topological order)
    2. Node state is only updated when all dependencies are completed
    3. Message passing only uses outputs from completed nodes
    4. State updates are atomic and consistent
    """
    
    @staticmethod
    def validate_execution_order(dag: PipelineDAG, execution_order: List[str]):
        """
        Validate that execution order respects dependency constraints.
        
        Args:
            dag: PipelineDAG instance
            execution_order: List of node names in execution order
            
        Raises:
            ValueError: If execution order violates dependency constraints
        """
        completed_nodes = set()
        
        for node in execution_order:
            dependencies = dag.get_dependencies(node)
            
            # All dependencies must be completed before this node
            if not dependencies.issubset(completed_nodes):
                missing_deps = dependencies - completed_nodes
                raise ValueError(f"Invalid execution order: {node} depends on {missing_deps} which haven't been processed yet")
            
            completed_nodes.add(node)
        
        logger.info(f"✅ Execution order validated: {len(execution_order)} nodes in correct dependency order")
    
    @staticmethod
    def ensure_state_consistency(registry: ScriptExecutionRegistry, node_name: str):
        """
        Ensure node state is consistent before execution.
        
        Args:
            registry: ScriptExecutionRegistry instance
            node_name: Name of the node to validate
            
        Raises:
            RuntimeError: If state is inconsistent
        """
        dependencies = registry.dag.get_dependencies(node_name)
        
        for dep_node in dependencies:
            if not registry._is_node_completed(dep_node):
                raise RuntimeError(f"State inconsistency: {node_name} cannot execute because dependency {dep_node} is not completed")
        
        logger.debug(f"✅ State consistency verified for {node_name}")


def create_script_execution_registry(dag: PipelineDAG, step_catalog: Optional[StepCatalog] = None) -> ScriptExecutionRegistry:
    """
    Factory function to create a ScriptExecutionRegistry instance.
    
    Args:
        dag: PipelineDAG instance defining the pipeline structure
        step_catalog: Optional StepCatalog instance (will create if not provided)
        
    Returns:
        Configured ScriptExecutionRegistry instance
    """
    if not step_catalog:
        step_catalog = StepCatalog()
    
    registry = ScriptExecutionRegistry(dag, step_catalog)
    
    # Validate execution order
    DAGStateConsistency.validate_execution_order(dag, registry.execution_order)
    
    logger.info(f"Created ScriptExecutionRegistry for DAG with {len(dag.nodes)} nodes")
    return registry
