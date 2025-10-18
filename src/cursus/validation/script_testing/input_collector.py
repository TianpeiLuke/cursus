"""
Script Testing Input Collector

This module extends DAGConfigFactory patterns for script input collection,
reusing existing interactive collection infrastructure instead of reimplementing it.
"""

from typing import Dict, Any, List
import logging

# Direct reuse of existing cursus infrastructure
from ...api.dag.base_dag import PipelineDAG
from ...api.factory.dag_config_factory import DAGConfigFactory
from ...steps.configs.utils import load_configs, build_complete_config_classes
from ...step_catalog import StepCatalog
from ...step_catalog.adapters.contract_adapter import ContractDiscoveryManagerAdapter
from pathlib import Path

logger = logging.getLogger(__name__)


class ScriptTestingInputCollector:
    """
    Enhanced with Direct Registry Integration for field value population.
    
    This class reuses the existing 600+ lines of proven interactive collection
    patterns and integrates directly with ScriptExecutionRegistry for:
    - Registry-coordinated field population
    - Message passing between script executions
    - Dynamic environment variable generation
    - Dependency-aware path resolution
    """
    
    def __init__(self, dag: PipelineDAG, config_path: str, registry=None, use_dependency_resolution: bool = True):
        """
        Initialize with DAG, config path, and optional registry.
        
        Args:
            dag: PipelineDAG instance
            config_path: Path to pipeline configuration JSON file
            registry: Optional ScriptExecutionRegistry for direct integration
            use_dependency_resolution: Whether to use two-phase dependency resolution
        """
        # REUSE: Existing DAGConfigFactory infrastructure (600+ lines of proven patterns)
        self.dag_factory = DAGConfigFactory(dag)
        self.dag = dag
        self.config_path = config_path
        self.registry = registry  # NEW: Direct registry reference
        self.use_dependency_resolution = use_dependency_resolution
        
        # Load configs for script validation
        self.loaded_configs = self._load_and_filter_configs()
        
        integration_mode = "registry-integrated" if registry else "standalone"
        logger.info(f"Initialized ScriptTestingInputCollector with {len(self.loaded_configs)} configs, mode={integration_mode}, dependency_resolution={use_dependency_resolution}")
    
    def _load_and_filter_configs(self) -> Dict[str, Any]:
        """
        Load and filter configs to DAG-related only.
        
        Returns:
            Dictionary of loaded configuration instances
        """
        try:
            config_classes = build_complete_config_classes()
            all_configs = load_configs(self.config_path, config_classes)
            
            # Filter to DAG-related configs only
            dag_configs = {}
            for node_name in self.dag.nodes:
                if node_name in all_configs:
                    dag_configs[node_name] = all_configs[node_name]
            
            return dag_configs
            
        except Exception as e:
            logger.error(f"Failed to load configs: {e}")
            return {}
    
    def collect_script_inputs_for_dag(self) -> Dict[str, Any]:
        """
        Enhanced collection with Direct Registry Integration.
        
        This method supports three modes:
        1. Registry-integrated collection (NEW)
        2. Two-phase dependency resolution 
        3. Manual collection (backward compatibility)
        
        Returns:
            Dictionary mapping script names to their input configurations
        """
        
        if self.registry:
            # NEW: Direct registry integration for field population
            logger.info("Using registry-integrated input collection")
            return self._collect_inputs_with_registry()
        elif self.use_dependency_resolution:
            # Use two-phase dependency resolution
            from .script_dependency_matcher import resolve_script_dependencies
            
            logger.info("Using two-phase dependency resolution for input collection")
            return resolve_script_dependencies(
                dag=self.dag,
                config_path=self.config_path,
                step_catalog=StepCatalog()
            )
        else:
            # FALLBACK: Use existing manual collection for backward compatibility
            logger.info("Using manual input collection (legacy mode)")
            return self._collect_inputs_manually()
    
    def _collect_inputs_with_registry(self) -> Dict[str, Any]:
        """
        NEW: Registry-coordinated input collection with field value population.
        
        This method uses the ScriptExecutionRegistry's 6 integration points to:
        - Get base configuration from registry
        - Apply message passing from completed dependencies
        - Populate field values dynamically
        - Store resolved inputs back to registry
        
        Returns:
            Dictionary mapping script names to their input configurations
        """
        user_inputs = {}
        
        logger.info("🔄 Starting registry-coordinated input collection")
        
        for node_name in self.dag.topological_sort():
            try:
                # Integration Point 3: Get base config from registry
                node_config = self.registry.get_node_config_for_resolver(node_name)
                
                # Integration Point 2: Get dependency outputs for message passing
                dependency_outputs = self.registry.get_dependency_outputs_for_node(node_name)
                
                # Populate field values using registry data
                script_inputs = self._populate_fields_from_registry(
                    node_name, node_config, dependency_outputs
                )
                
                # Integration Point 4: Store resolved inputs back to registry
                self.registry.store_resolved_inputs(node_name, script_inputs)
                
                user_inputs[node_name] = script_inputs
                
                logger.info(f"✅ Registry-populated inputs for {node_name}: {len(script_inputs)} fields")
                
            except Exception as e:
                logger.error(f"❌ Failed to collect registry inputs for {node_name}: {e}")
                # Fallback to manual collection for this node
                script_inputs = self._collect_script_inputs(node_name)
                user_inputs[node_name] = script_inputs
        
        logger.info(f"🎯 Registry-coordinated collection completed: {len(user_inputs)} scripts")
        return user_inputs
    
    def _populate_fields_from_registry(self, node_name: str, node_config: Dict[str, Any], 
                                     dependency_outputs: Dict[str, str]) -> Dict[str, Any]:
        """
        Populate field values using registry data and message passing.
        
        This method combines:
        - Base configuration from registry
        - Dependency outputs via message passing
        - Contract-based path resolution
        - Dynamic environment variable generation
        
        Args:
            node_name: Name of the node
            node_config: Base configuration from registry
            dependency_outputs: Outputs from completed dependencies
            
        Returns:
            Dictionary containing complete input configuration for script execution
        """
        logger.debug(f"🔧 Populating fields for {node_name} with {len(dependency_outputs)} dependency outputs")
        
        # Start with base script configuration
        script_inputs = {
            'input_paths': {},
            'output_paths': self._get_default_output_paths(node_name),
            'environment_variables': {},
            'job_arguments': {},
            'script_path': None
        }
        
        # 1. Populate input paths from dependency outputs (message passing)
        script_inputs['input_paths'].update(dependency_outputs)
        logger.debug(f"📨 Applied {len(dependency_outputs)} dependency outputs to {node_name}")
        
        # 2. Add contract-based paths for missing inputs
        contract_inputs = self._get_input_paths_with_message_passing(node_name, dependency_outputs)
        for key, path in contract_inputs.items():
            if key not in script_inputs['input_paths']:
                script_inputs['input_paths'][key] = path
        
        # 3. Extract and enhance environment variables with registry context
        config_instance = self.loaded_configs.get(node_name, {})
        script_inputs['environment_variables'] = self._extract_environment_variables_with_registry(
            config_instance, node_name
        )
        
        # 4. Extract job arguments from config
        script_inputs['job_arguments'] = self._extract_job_arguments(config_instance)
        
        # 5. Extract script path from config
        script_inputs['script_path'] = self._extract_script_path_from_config(config_instance)
        
        logger.info(f"🎯 Populated {node_name}: {len(script_inputs['input_paths'])} inputs, "
                   f"{len(script_inputs['environment_variables'])} env vars, "
                   f"{len(script_inputs['job_arguments'])} job args")
        
        return script_inputs
    
    def _get_input_paths_with_message_passing(self, node_name: str, dependency_outputs: Dict[str, str]) -> Dict[str, str]:
        """
        Get input paths with registry-coordinated message passing.
        
        This method applies intelligent mapping between dependency outputs and node inputs:
        - Direct name matching
        - Semantic mapping (model → model_path, data → training_data)
        - Contract-based defaults for missing paths
        
        Args:
            node_name: Name of the node
            dependency_outputs: Outputs from completed dependencies
            
        Returns:
            Dictionary of input paths with message passing applied
        """
        input_paths = {}
        
        # Apply intelligent mapping from dependency outputs
        for output_key, output_path in dependency_outputs.items():
            # Direct mapping
            input_paths[output_key] = output_path
            
            # Semantic mapping
            semantic_mapping = self._get_semantic_input_mapping(output_key)
            if semantic_mapping and semantic_mapping not in input_paths:
                input_paths[semantic_mapping] = output_path
                logger.debug(f"📨 Semantic mapping: {output_key} → {semantic_mapping} for {node_name}")
        
        # Add contract-based defaults for missing paths
        contract_paths = self._get_default_input_paths(node_name)
        for key, path in contract_paths.items():
            if key not in input_paths:
                input_paths[key] = path
        
        return input_paths
    
    def _get_semantic_input_mapping(self, output_key: str) -> str:
        """
        Get semantic mapping from output key to input key.
        
        Examples:
        - 'model' → 'model_path'
        - 'processed_data' → 'training_data'
        - 'features' → 'feature_data'
        
        Args:
            output_key: Output key from dependency
            
        Returns:
            Mapped input key or None if no mapping found
        """
        semantic_rules = {
            'model': 'model_path',
            'processed_data': 'training_data',
            'features': 'feature_data',
            'predictions': 'prediction_data',
            'data': 'input_data',
            'output': 'input_file',
            'trained_model': 'model_path',
            'preprocessed_data': 'training_data'
        }
        
        return semantic_rules.get(output_key)
    
    def _extract_environment_variables_with_registry(self, config: Any, node_name: str) -> Dict[str, str]:
        """
        Extract environment variables with registry-aware population.
        
        This method combines:
        - Base environment variables from config
        - Registry-specific dynamic variables
        - Execution context information
        
        Args:
            config: Configuration instance
            node_name: Name of the node
            
        Returns:
            Dictionary of environment variables with registry enhancements
        """
        # Base environment variables from config
        env_vars = self._extract_environment_variables(config)
        
        # Add registry-specific variables if registry available
        if self.registry:
            try:
                # Get execution context from registry
                execution_summary = self.registry.get_execution_summary()
                
                # Add dynamic variables based on registry state
                registry_vars = {
                    'PIPELINE_EXECUTION_ID': str(hash(str(execution_summary))),
                    'NODE_EXECUTION_ORDER': str(self.registry.execution_order.index(node_name)),
                    'TOTAL_PIPELINE_NODES': str(len(self.dag.nodes)),
                    'COMPLETED_DEPENDENCIES': ','.join([
                        dep for dep in self.dag.get_dependencies(node_name)
                        if self.registry.get_node_status(dep) == 'completed'
                    ]),
                    'REGISTRY_MODE': 'enabled'
                }
                
                env_vars.update(registry_vars)
                logger.debug(f"🔧 Added {len(registry_vars)} registry-specific env vars for {node_name}")
                
            except Exception as e:
                logger.warning(f"Failed to add registry env vars for {node_name}: {e}")
        
        return env_vars
    
    def _extract_script_path_from_config(self, config: Any) -> str:
        """
        Extract script path from config using proper field access.
        
        Args:
            config: Configuration instance
            
        Returns:
            Script path or None if not found
        """
        # Check for various entry point fields
        entry_point_fields = [
            'training_entry_point',
            'inference_entry_point', 
            'entry_point'
        ]
        
        for field in entry_point_fields:
            if hasattr(config, field):
                entry_point = getattr(config, field)
                if entry_point:
                    # Combine with source_dir if available
                    if hasattr(config, 'effective_source_dir') and config.effective_source_dir:
                        import os
                        return os.path.join(config.effective_source_dir, entry_point)
                    else:
                        return entry_point
        
        return None
    
    def _collect_inputs_manually(self) -> Dict[str, Any]:
        """
        Existing manual collection logic (unchanged for backward compatibility).
        
        Returns:
            Dictionary mapping script names to their input configurations
        """
        user_inputs = {}
        
        # Use config-based script validation to eliminate phantom scripts
        validated_scripts = self._get_validated_scripts_from_config()
        logger.info(f"Validated scripts (no phantoms): {validated_scripts}")
        
        for script_name in validated_scripts:
            # EXTEND: Use DAGConfigFactory patterns for input collection
            script_inputs = self._collect_script_inputs(script_name)
            user_inputs[script_name] = script_inputs
        
        return user_inputs
    
    def _get_validated_scripts_from_config(self) -> List[str]:
        """
        Get only scripts with actual entry points from config (eliminates phantom scripts).
        
        This addresses the phantom script issue by using config-based validation
        to ensure only scripts with actual entry points are discovered.
        
        Returns:
            List of validated script names with actual entry points
        """
        validated_scripts = []
        
        for node_name in self.dag.nodes:
            if node_name in self.loaded_configs:
                config = self.loaded_configs[node_name]
                # Check if config has script entry point fields
                if self._has_script_entry_point(config):
                    validated_scripts.append(node_name)
        
        logger.info(f"Phantom script elimination: {len(self.dag.nodes)} nodes -> {len(validated_scripts)} validated scripts")
        return validated_scripts
    
    def _has_script_entry_point(self, config: Any) -> bool:
        """
        Check if config has script entry point fields.
        
        Args:
            config: Configuration instance
            
        Returns:
            True if config has script entry points, False otherwise
        """
        # Check for various entry point field patterns
        entry_point_fields = [
            'training_entry_point', 'inference_entry_point', 'entry_point',
            'source_dir', 'script_path', 'code_location'
        ]
        
        for field in entry_point_fields:
            if hasattr(config, field) and getattr(config, field):
                return True
        
        return False
    
    def _collect_script_inputs(self, script_name: str) -> Dict[str, Any]:
        """
        Collect inputs for a single script using existing patterns.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Dictionary with script input configuration
        """
        # Get script requirements from config (pre-populated environment variables)
        config = self.loaded_configs.get(script_name, {})
        
        # Extract environment variables from config (eliminates manual guesswork)
        environment_variables = self._extract_environment_variables(config)
        
        # Extract job arguments from config
        job_arguments = self._extract_job_arguments(config)
        
        # Simple input collection (users only need to provide paths)
        return {
            'input_paths': self._get_default_input_paths(script_name),
            'output_paths': self._get_default_output_paths(script_name),
            'environment_variables': environment_variables,  # From config (automated)
            'job_arguments': job_arguments  # From config (automated)
        }
    
    def _extract_environment_variables(self, config: Any) -> Dict[str, str]:
        """
        Extract environment variables from config.
        
        Args:
            config: Configuration instance
            
        Returns:
            Dictionary of environment variables
        """
        environment_variables = {}
        
        if hasattr(config, '__dict__'):
            for field_name, field_value in config.__dict__.items():
                if field_value and isinstance(field_value, (str, int, float)):
                    # Convert to environment variable format (CAPITAL_CASE)
                    env_var_name = field_name.upper()
                    environment_variables[env_var_name] = str(field_value)
        
        return environment_variables
    
    def _extract_job_arguments(self, config: Any) -> Dict[str, Any]:
        """
        Extract job arguments from config.
        
        Args:
            config: Configuration instance
            
        Returns:
            Dictionary of job arguments
        """
        job_arguments = {}
        
        # Look for common job argument fields
        job_arg_fields = [
            'instance_type', 'instance_count', 'volume_size', 'max_runtime_in_seconds',
            'job_type', 'framework_version', 'python_version'
        ]
        
        if hasattr(config, '__dict__'):
            for field_name in job_arg_fields:
                if hasattr(config, field_name):
                    field_value = getattr(config, field_name)
                    if field_value:
                        job_arguments[field_name] = field_value
        
        return job_arguments
    
    def _get_default_input_paths(self, script_name: str) -> Dict[str, str]:
        """
        Get input paths for a script using systematic contract-based solution.
        
        This method reuses the existing ContractDiscoveryManagerAdapter infrastructure
        for systematic contract-based path handling instead of hardcoded logic.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Dictionary mapping contract logical names to local test paths
        """
        try:
            # SYSTEMATIC: Use existing ContractDiscoveryManagerAdapter infrastructure
            test_data_dir = f"test/data/{script_name}"
            contract_adapter = ContractDiscoveryManagerAdapter(test_data_dir=test_data_dir)
            
            # Load contract using existing systematic approach
            catalog = StepCatalog()
            contract = catalog.load_contract_class(script_name)
            
            if contract:
                # REUSE: Use existing get_contract_input_paths method (systematic solution)
                adapted_paths = contract_adapter.get_contract_input_paths(contract, script_name)
                
                if adapted_paths:
                    logger.info(f"SUCCESS: Using systematic contract-based input paths for {script_name}: {list(adapted_paths.keys())}")
                    return adapted_paths
                else:
                    logger.warning(f"Contract found but no input paths for {script_name}")
            else:
                logger.warning(f"No contract found for {script_name}")
                
        except Exception as e:
            logger.error(f"Error in systematic contract-based path resolution for {script_name}: {e}")
        
        # Fallback to generic paths if systematic approach fails
        logger.warning(f"Using fallback generic paths for {script_name}")
        return {
            'data_input': f"test/data/{script_name}/input",
            'model_input': f"test/models/{script_name}/input"
        }
    
    def _get_default_output_paths(self, script_name: str) -> Dict[str, str]:
        """
        Get output paths for a script using systematic contract-based solution.
        
        This method reuses the existing ContractDiscoveryManagerAdapter infrastructure
        for systematic contract-based path handling instead of hardcoded logic.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Dictionary mapping contract logical names to local test paths
        """
        try:
            # SYSTEMATIC: Use existing ContractDiscoveryManagerAdapter infrastructure
            test_data_dir = f"test/data/{script_name}"
            contract_adapter = ContractDiscoveryManagerAdapter(test_data_dir=test_data_dir)
            
            # Load contract using existing systematic approach
            catalog = StepCatalog()
            contract = catalog.load_contract_class(script_name)
            
            if contract:
                # REUSE: Use existing get_contract_output_paths method (systematic solution)
                adapted_paths = contract_adapter.get_contract_output_paths(contract, script_name)
                
                if adapted_paths:
                    logger.info(f"SUCCESS: Using systematic contract-based output paths for {script_name}: {list(adapted_paths.keys())}")
                    return adapted_paths
                else:
                    logger.warning(f"Contract found but no output paths for {script_name}")
            else:
                logger.warning(f"No contract found for {script_name}")
                
        except Exception as e:
            logger.error(f"Error in systematic contract-based output path resolution for {script_name}: {e}")
        
        # Fallback to generic paths if systematic approach fails
        return {
            'data_output': f"test/data/{script_name}/output",
            'model_output': f"test/models/{script_name}/output"
        }
    
    
    def get_script_requirements(self, script_name: str) -> Dict[str, Any]:
        """
        Get requirements for a specific script.
        
        This method extends DAGConfigFactory.get_step_requirements() for script testing.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Dictionary with script requirements
        """
        if script_name not in self.loaded_configs:
            raise ValueError(f"Script '{script_name}' not found in validated scripts")
        
        config = self.loaded_configs[script_name]
        
        return {
            'script_name': script_name,
            'config_type': type(config).__name__,
            'has_entry_point': self._has_script_entry_point(config),
            'environment_variables': self._extract_environment_variables(config),
            'job_arguments': self._extract_job_arguments(config),
            'default_input_paths': self._get_default_input_paths(script_name),
            'default_output_paths': self._get_default_output_paths(script_name)
        }
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """
        Get summary of input collection status.
        
        Returns:
            Dictionary with collection summary
        """
        validated_scripts = self._get_validated_scripts_from_config()
        
        return {
            'total_dag_nodes': len(self.dag.nodes),
            'loaded_configs': len(self.loaded_configs),
            'validated_scripts': len(validated_scripts),
            'phantom_scripts_eliminated': len(self.dag.nodes) - len(validated_scripts),
            'config_path': self.config_path,
            'script_names': validated_scripts
        }
