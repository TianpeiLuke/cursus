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
from ...step_catalog.adapters.config_resolver import StepConfigResolverAdapter as StepConfigResolver
from ...core.compiler.exceptions import ConfigurationError
from ...registry.step_names import (
    get_canonical_name_from_file_name,
    get_spec_step_type,
    get_step_name_from_spec_type,
)

logger = logging.getLogger(__name__)


class PipelineExecutionPlan(BaseModel):
    """Execution plan for pipeline with topological ordering."""

    execution_order: List[str]
    step_configs: Dict[
        str, dict
    ]  # Using dict instead of StepConfig for Pydantic compatibility
    dependencies: Dict[str, List[str]]
    data_flow_map: Dict[str, Dict[str, str]]


class PipelineDAGResolver:
    """Enhanced resolver with StepCatalog integration for reliable, deployment-agnostic DAG resolution."""

    def __init__(
        self,
        dag: PipelineDAG,
        workspace_dirs: Optional[List[Path]] = None,
        config_path: Optional[str] = None,
        available_configs: Optional[Dict[str, BasePipelineConfig]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        validate_on_init: bool = True,
    ):
        """
        Initialize with enhanced StepCatalog integration.
        
        Args:
            dag: PipelineDAG instance defining pipeline structure
            workspace_dirs: Optional workspace directories for workspace-aware discovery
            config_path: Path to configuration file (optional)
            available_configs: Pre-loaded configuration instances (optional)
            metadata: Configuration metadata for enhanced resolution (optional)
            validate_on_init: Early DAG validation with step existence checking
        """
        self.dag = dag
        self.graph = self._build_networkx_graph()
        
        # NEW: Initialize StepCatalog with workspace support
        self.step_catalog = self._initialize_step_catalog(workspace_dirs)
        
        # Configuration resolution (enhanced with catalog integration)
        self.config_path = config_path
        self.available_configs = available_configs or {}
        self.metadata = metadata
        self.config_resolver = self._initialize_config_resolver()

        # Load configs from file if path provided
        if config_path and not available_configs:
            try:
                self.available_configs = self._load_configs_from_file(config_path)
                logger.info(
                    f"Loaded {len(self.available_configs)} configurations from {config_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load configs from {config_path}: {e}")
                self.available_configs = {}
        
        # Enhanced validation during initialization
        if validate_on_init:
            self._validate_dag_with_catalog()

    def _initialize_step_catalog(self, workspace_dirs: Optional[List[Path]]):
        """Initialize StepCatalog with workspace support."""
        try:
            from ...step_catalog import StepCatalog
            return StepCatalog(workspace_dirs=workspace_dirs)
        except ImportError as e:
            logger.warning(f"StepCatalog not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize StepCatalog: {e}")
            return None

    def _initialize_config_resolver(self):
        """Initialize configuration resolver with enhanced StepCatalog integration."""
        return (
            StepConfigResolver() if (self.config_path or self.available_configs) else None
        )

    def _validate_dag_with_catalog(self):
        """Perform early DAG validation using StepCatalog."""
        if not self.step_catalog:
            logger.debug("StepCatalog not available, skipping enhanced validation")
            return
        
        validation_issues = self.validate_dag_integrity()
        if validation_issues:
            logger.warning(f"DAG validation issues detected: {validation_issues}")
            # Don't raise exception during initialization, just log warnings

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
                logger.info(
                    f"Resolving step configurations for {len(execution_order)} nodes"
                )
                config_map = self.config_resolver.resolve_config_map(
                    dag_nodes=execution_order,
                    available_configs=self.available_configs,
                    metadata=self.metadata,
                )

                # Convert to dict format for Pydantic compatibility
                for name, config in config_map.items():
                    if hasattr(config, "__dict__"):
                        step_configs[name] = config.__dict__
                    else:
                        step_configs[name] = config

                logger.info(
                    f"Successfully resolved configurations for {len(step_configs)} steps"
                )

            except ConfigurationError as e:
                logger.warning(f"Could not resolve step configs: {e}")
                step_configs = {name: {} for name in execution_order}
        else:
            # Fallback: empty configs for base DAG without config support
            step_configs = {name: {} for name in execution_order}
            if not self.config_resolver:
                logger.debug("No config resolver available - using empty step configs")

        dependencies = {
            name: list(self.graph.predecessors(name)) for name in execution_order
        }

        data_flow_map = self._build_data_flow_map()

        return PipelineExecutionPlan(
            execution_order=execution_order,
            step_configs=step_configs,
            dependencies=dependencies,
            data_flow_map=data_flow_map,
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
                            input_channel,
                            input_path,
                            dep_contract.expected_output_paths,
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
        REFACTORED: Simplified contract discovery using StepCatalog.
        
        IMPROVEMENTS:
        - Single discovery path through StepCatalog
        - Eliminates manual importlib usage
        - Better error handling and logging
        - Workspace-aware discovery
        """
        try:
            # Use StepCatalog's unified contract discovery
            if self.step_catalog:
                contract = self.step_catalog.load_contract_class(step_name)
                
                if contract:
                    logger.debug(f"Successfully loaded contract for {step_name} via StepCatalog")
                    return contract
                else:
                    logger.debug(f"No contract found for step: {step_name}")
                    return None
            else:
                # Fallback to legacy discovery if StepCatalog not available
                logger.debug("StepCatalog not available, using legacy contract discovery")
                return self._discover_step_contract_legacy(step_name)
                
        except Exception as e:
            logger.warning(f"Error loading contract for {step_name}: {e}")
            # Fallback to legacy discovery on any error
            return self._discover_step_contract_legacy(step_name)

    def _discover_step_contract_legacy(self, step_name: str) -> Optional[ScriptContract]:
        """Legacy step contract discovery method (fallback only)."""
        try:
            # Convert step name to canonical name
            canonical_name = get_canonical_name_from_file_name(step_name)
            if not canonical_name:
                logger.debug(f"No canonical name found for step: {step_name}")
                return None

            # Get specification from canonical name
            step_spec = self._get_step_specification(canonical_name)
            if not step_spec:
                logger.debug(
                    f"No specification found for canonical name: {canonical_name}"
                )
                return None

            # Extract contract from specification
            if hasattr(step_spec, "script_contract") and step_spec.script_contract:
                logger.debug(
                    f"Found contract for step {step_name} via {canonical_name}"
                )
                return step_spec.script_contract

            logger.debug(
                f"No script_contract found in specification for: {canonical_name}"
            )
            return None

        except Exception as e:
            logger.warning(f"Failed to discover contract for step {step_name}: {e}")
            return None

    def _get_step_specification(
        self, canonical_name: str
    ) -> Optional[StepSpecification]:
        """
        Get step specification using StepCatalog's unified discovery system.

        Args:
            canonical_name: Canonical name of the step

        Returns:
            StepSpecification instance if found, None otherwise
        """
        try:
            # Use StepCatalog for unified specification discovery
            from ...step_catalog import StepCatalog
            
            # Use package-only discovery for deployment portability
            catalog = StepCatalog(workspace_dirs=None)
            spec_instance = catalog.load_spec_class(canonical_name)
            
            if spec_instance:
                logger.debug(f"Successfully loaded specification for {canonical_name} via StepCatalog")
                return spec_instance
            else:
                logger.debug(f"No specification found for canonical name: {canonical_name}")
                return None

        except ImportError as e:
            logger.debug(f"StepCatalog not available for spec loading: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error getting specification for {canonical_name}: {e}")
            return None

    # REMOVED: _spec_type_to_module_name() - No longer needed with StepCatalog integration
    # This method has been eliminated as part of the refactoring to use StepCatalog's
    # unified discovery system, which handles naming conventions internally.

    def _find_compatible_output(
        self, input_channel: str, input_path: str, output_channels: Dict[str, str]
    ) -> Optional[str]:
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
                logger.debug(
                    f"Path-compatible match: {output_channel} ({output_path} -> {input_path})"
                )
                return output_channel

        # Strategy 3: Semantic matching for common patterns
        semantic_matches = {
            "input_path": ["output_path", "model_path", "data_path"],
            "model_path": ["model_output_path", "output_path"],
            "data_path": ["output_path", "processed_data_path"],
            "hyperparameters_s3_uri": ["config_path", "hyperparameters_path"],
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
            ("/opt/ml/model", "/opt/ml/model"),  # Model artifacts
            ("/opt/ml/input/data", "/opt/ml/output/data"),  # Data flow
            ("/opt/ml/output", "/opt/ml/input/data"),  # Output to input
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
        """
        REFACTORED: Comprehensive DAG validation using StepCatalog.
        
        IMPROVEMENTS:
        - Step existence validation using catalog
        - Component availability checking (builders, contracts, specs, configs)
        - Workspace compatibility validation
        - Enhanced error messages with suggestions
        """
        issues = {}
        
        # Traditional validation (cycles, dangling dependencies, isolated nodes)
        issues.update(self._validate_graph_structure())
        
        # NEW: StepCatalog-based validation
        if self.step_catalog:
            step_validation_issues = self._validate_steps_with_catalog()
            if step_validation_issues:
                issues.update(step_validation_issues)
            
            # NEW: Component availability validation
            component_issues = self._validate_component_availability()
            if component_issues:
                issues.update(component_issues)
            
            # NEW: Workspace compatibility validation
            workspace_issues = self._validate_workspace_compatibility()
            if workspace_issues:
                issues.update(workspace_issues)
        else:
            logger.debug("StepCatalog not available, using basic validation only")
        
        return issues

    def _validate_graph_structure(self) -> Dict[str, List[str]]:
        """Validate basic graph structure (cycles, dangling dependencies, isolated nodes)."""
        issues = {}

        # Check for cycles
        try:
            list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            cycles = list(nx.simple_cycles(self.graph))
            issues["cycles"] = [
                f"Cycle detected: {' -> '.join(cycle)}" for cycle in cycles
            ]

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
            issues["isolated_nodes"] = [
                f"Node has no connections: {node}" for node in isolated_nodes
            ]

        return issues

    def _validate_steps_with_catalog(self) -> Dict[str, List[str]]:
        """Validate all DAG nodes exist in StepCatalog."""
        issues = {}
        missing_steps = []
        
        for step_name in self.dag.nodes:
            step_info = self.step_catalog.get_step_info(step_name)
            if not step_info:
                missing_steps.append(step_name)
        
        if missing_steps:
            available_steps = self.step_catalog.list_available_steps()
            issues["missing_steps"] = [
                f"Step '{step}' not found in catalog. Available steps: {available_steps[:10]}..."
                for step in missing_steps
            ]
        
        return issues

    def _validate_component_availability(self) -> Dict[str, List[str]]:
        """Validate component availability for each step."""
        issues = {}
        component_issues = []
        
        for step_name in self.dag.nodes:
            step_info = self.step_catalog.get_step_info(step_name)
            if step_info:
                # Check component availability
                missing_components = []
                
                # Check builder availability
                if not step_info.file_components.get('builder'):
                    builder_class = self.step_catalog.load_builder_class(step_name)
                    if not builder_class:
                        missing_components.append('builder')
                
                # Check contract availability
                if not step_info.file_components.get('contract'):
                    contract = self.step_catalog.load_contract_class(step_name)
                    if not contract:
                        missing_components.append('contract')
                
                # Check spec availability
                if not step_info.file_components.get('spec'):
                    spec = self.step_catalog.load_spec_class(step_name)
                    if not spec:
                        missing_components.append('spec')
                
                if missing_components:
                    component_issues.append(
                        f"Step '{step_name}' missing components: {missing_components}"
                    )
        
        if component_issues:
            issues["missing_components"] = component_issues
        
        return issues

    def _validate_workspace_compatibility(self) -> Dict[str, List[str]]:
        """Validate workspace compatibility for steps."""
        issues = {}
        workspace_issues = []
        
        # Check if steps come from different workspaces and might have conflicts
        step_workspaces = {}
        for step_name in self.dag.nodes:
            step_info = self.step_catalog.get_step_info(step_name)
            if step_info:
                workspace_id = step_info.workspace_id
                if workspace_id not in step_workspaces:
                    step_workspaces[workspace_id] = []
                step_workspaces[workspace_id].append(step_name)
        
        # Report multi-workspace usage (informational)
        if len(step_workspaces) > 1:
            workspace_summary = {
                ws_id: len(steps) for ws_id, steps in step_workspaces.items()
            }
            workspace_issues.append(
                f"DAG uses steps from multiple workspaces: {workspace_summary}. "
                f"Ensure workspace compatibility."
            )
        
        if workspace_issues:
            issues["workspace_compatibility"] = workspace_issues
        
        return issues

    def _load_configs_from_file(
        self, config_path: str
    ) -> Dict[str, BasePipelineConfig]:
        """
        Load configurations from file using StepCatalog-enhanced discovery.

        This method loads a JSON configuration file and uses the StepCatalog system
        to properly instantiate configuration classes based on the step definitions.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary of loaded configuration instances

        Raises:
            ConfigurationError: If configs cannot be loaded
        """
        try:
            import json
            from pathlib import Path

            # Load the JSON configuration file
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            with open(config_file, "r") as f:
                config_data = json.load(f)

            logger.info(f"Loading configurations from {config_path}")
            logger.debug(f"Configuration file structure: {list(config_data.keys())}")

            # Extract metadata if available
            if "metadata" in config_data:
                self.metadata = config_data["metadata"]
                logger.debug("Loaded metadata from configuration file")

            # Use StepCatalog to discover and instantiate configuration classes
            configs = {}
            
            # Process each configuration section
            for config_key, config_values in config_data.items():
                if config_key == "metadata":
                    continue  # Skip metadata section
                
                try:
                    # Try to find the corresponding step and config class using StepCatalog
                    config_instance = self._instantiate_config_from_catalog(
                        config_key, config_values
                    )
                    
                    if config_instance:
                        configs[config_key] = config_instance
                        logger.debug(f"Successfully loaded config for: {config_key}")
                    else:
                        logger.warning(f"Could not instantiate config for: {config_key}")
                        
                except Exception as e:
                    logger.warning(f"Error loading config for {config_key}: {e}")
                    continue

            logger.info(f"Successfully loaded {len(configs)} configurations from file")
            return configs

        except Exception as e:
            try:
                from ...core.compiler.exceptions import ConfigurationError
                raise ConfigurationError(f"Failed to load configurations from {config_path}: {e}")
            except ImportError:
                # Fallback if ConfigurationError is not available
                raise ValueError(f"Failed to load configurations from {config_path}: {e}")

    def _instantiate_config_from_catalog(
        self, config_key: str, config_values: dict
    ) -> Optional[BasePipelineConfig]:
        """
        Instantiate a configuration class using StepCatalog discovery.

        Args:
            config_key: Configuration key from the JSON file
            config_values: Configuration values dictionary

        Returns:
            Instantiated configuration instance or None
        """
        if not self.step_catalog:
            logger.debug("StepCatalog not available for config instantiation")
            return None

        try:
            # Strategy 1: Direct step name lookup
            step_info = self.step_catalog.get_step_info(config_key)
            if step_info and step_info.config_class:
                config_class = self._get_config_class_by_name(step_info.config_class)
                if config_class:
                    return self._create_config_instance(config_class, config_values)

            # Strategy 2: Search by config class name pattern
            # Try variations of the config key
            config_class_candidates = [
                f"{config_key}Config",
                f"{config_key}StepConfig", 
                config_key,
            ]
            
            for candidate in config_class_candidates:
                config_class = self._get_config_class_by_name(candidate)
                if config_class:
                    return self._create_config_instance(config_class, config_values)

            # Strategy 3: Search through all available steps for matching config class
            available_steps = self.step_catalog.list_available_steps()
            for step_name in available_steps:
                step_info = self.step_catalog.get_step_info(step_name)
                if step_info and step_info.config_class:
                    # Check if config class name matches any of our candidates
                    if step_info.config_class in config_class_candidates:
                        config_class = self._get_config_class_by_name(step_info.config_class)
                        if config_class:
                            return self._create_config_instance(config_class, config_values)

            logger.debug(f"No matching config class found for: {config_key}")
            return None

        except Exception as e:
            logger.warning(f"Error instantiating config for {config_key}: {e}")
            return None

    def _get_config_class_by_name(self, class_name: str) -> Optional[type]:
        """
        Get configuration class by name using dynamic import.

        Args:
            class_name: Name of the configuration class

        Returns:
            Configuration class or None
        """
        try:
            # Try to import from common config locations
            config_locations = [
                f"...steps.configs.config_{self._class_name_to_module(class_name)}",
                f"...core.base.config_base",
                f"...steps.configs",
            ]

            for location in config_locations:
                try:
                    module = importlib.import_module(location, package=__package__)
                    if hasattr(module, class_name):
                        config_class = getattr(module, class_name)
                        if issubclass(config_class, BasePipelineConfig):
                            return config_class
                except (ImportError, AttributeError):
                    continue

            return None

        except Exception as e:
            logger.debug(f"Error getting config class {class_name}: {e}")
            return None

    def _class_name_to_module(self, class_name: str) -> str:
        """
        Convert class name to module name.

        Args:
            class_name: Configuration class name (e.g., "XGBoostTrainingConfig")

        Returns:
            Module name (e.g., "xgboost_training")
        """
        # Remove "Config" suffix
        if class_name.endswith("Config"):
            class_name = class_name[:-6]
        
        # Remove "Step" suffix if present
        if class_name.endswith("Step"):
            class_name = class_name[:-4]

        # Convert CamelCase to snake_case
        import re
        module_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", class_name).lower()
        return module_name

    def _create_config_instance(
        self, config_class: type, config_values: dict
    ) -> Optional[BasePipelineConfig]:
        """
        Create configuration instance from class and values.

        Args:
            config_class: Configuration class
            config_values: Configuration values dictionary

        Returns:
            Configuration instance or None
        """
        try:
            # Try to instantiate the config class with the provided values
            if hasattr(config_class, "from_dict"):
                # Use from_dict method if available
                return config_class.from_dict(config_values)
            else:
                # Try direct instantiation with keyword arguments
                return config_class(**config_values)

        except Exception as e:
            logger.warning(f"Error creating config instance for {config_class.__name__}: {e}")
            try:
                # Fallback: try with empty initialization and set attributes
                instance = config_class()
                for key, value in config_values.items():
                    if hasattr(instance, key):
                        setattr(instance, key, value)
                return instance
            except Exception as fallback_error:
                logger.warning(f"Fallback config creation also failed: {fallback_error}")
                return None

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
                metadata=self.metadata,
            )
        except Exception as e:
            logger.warning(f"Failed to generate config resolution preview: {e}")
            return None
