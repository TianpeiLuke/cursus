"""
Script Execution Template

Dynamic script execution template that works with any PipelineDAG.
Mirrors DynamicPipelineTemplate but creates ScriptExecutionSpecs instead of SageMaker pipeline steps.
Uses maximum component reuse from existing cursus infrastructure.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Direct imports from existing cursus components - MAXIMUM REUSE
from ...api.dag.base_dag import PipelineDAG
from ...step_catalog import StepCatalog
from ...core.deps.dependency_resolver import create_dependency_resolver
from ..base.script_execution_spec import ScriptExecutionSpec
from ..base.script_execution_plan import ScriptExecutionPlan
from .exceptions import (
    ScriptCompilationError,
    ScriptDiscoveryError,
    StepCatalogIntegrationError,
)
from .validation import validate_script_spec

logger = logging.getLogger(__name__)


class ScriptExecutionTemplate:
    """
    Dynamic script execution template that works with any PipelineDAG.
    
    Mirrors DynamicPipelineTemplate but creates ScriptExecutionSpecs 
    instead of SageMaker pipeline steps. Uses maximum component reuse
    from existing cursus infrastructure.
    
    This class follows the same patterns as DynamicPipelineTemplate:
    1. Takes a DAG and user inputs
    2. Maps DAG nodes to specifications (ScriptExecutionSpecs vs configs)
    3. Resolves dependencies and creates execution plan
    4. Uses step catalog for enhanced discovery and validation
    
    Attributes:
        dag: PipelineDAG defining the execution structure
        user_inputs: User inputs collected for script execution
        test_workspace_dir: Directory for test workspace and script discovery
        step_catalog: StepCatalog for enhanced script discovery and validation
        dependency_resolver: Dependency resolver for I/O connections
        config: Template configuration options
    """
    
    def __init__(
        self,
        dag: PipelineDAG,
        user_inputs: Dict[str, Any],
        test_workspace_dir: str,
        step_catalog: Optional[StepCatalog] = None,
        dependency_resolver: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Script Execution Template.
        
        Args:
            dag: PipelineDAG instance defining the pipeline structure
            user_inputs: User inputs collected for script execution
            test_workspace_dir: Directory for test workspace and script discovery
            step_catalog: Optional step catalog for enhanced script discovery
            dependency_resolver: Optional dependency resolver for I/O connections
            config: Optional template configuration
            **kwargs: Additional template options
        """
        self.dag = dag
        self.user_inputs = user_inputs
        self.test_workspace_dir = test_workspace_dir
        self.step_catalog = step_catalog or StepCatalog()
        
        # DIRECT REUSE: Use existing dependency resolver
        self.dependency_resolver = dependency_resolver or create_dependency_resolver()
        
        # Template configuration
        self.config = config or {}
        self.config.update(kwargs)
        
        # Template state
        self._script_specs: Optional[Dict[str, ScriptExecutionSpec]] = None
        self._execution_order: Optional[List[str]] = None
        
        logger.info(f"Initialized ScriptExecutionTemplate for DAG with {len(dag.nodes)} nodes")
    
    def create_execution_plan(self) -> ScriptExecutionPlan:
        """
        Create complete script execution plan.
        
        This method mirrors the template generation process in DynamicPipelineTemplate
        but creates a ScriptExecutionPlan instead of a SageMaker pipeline.
        
        Returns:
            ScriptExecutionPlan ready for execution
            
        Raises:
            ScriptCompilationError: If template creation fails
        """
        try:
            logger.info("Creating script execution plan from template")
            
            # 1. Create script spec map (mirrors _create_config_map)
            script_specs = self._create_script_spec_map()
            
            # 2. Get execution order (DIRECT REUSE of DAG topological sort)
            execution_order = self.dag.topological_sort()
            
            # 3. Resolve dependencies and update specs
            self._resolve_script_dependencies(script_specs, execution_order)
            
            # 4. Validate execution plan components
            self._validate_execution_plan_components(script_specs, execution_order)
            
            # 5. Create execution plan
            execution_plan = ScriptExecutionPlan(
                dag=self.dag,
                script_specs=script_specs,
                execution_order=execution_order,
                test_workspace_dir=self.test_workspace_dir,
                metadata={
                    "template_config": self.config,
                    "user_inputs_provided": bool(self.user_inputs),
                    "step_catalog_used": self.step_catalog is not None,
                }
            )
            
            logger.info(f"Successfully created execution plan with {len(script_specs)} scripts")
            return execution_plan
            
        except Exception as e:
            logger.error(f"Failed to create execution plan: {e}")
            raise ScriptCompilationError(f"Template creation failed: {e}") from e
    
    def _create_script_spec_map(self) -> Dict[str, ScriptExecutionSpec]:
        """
        Auto-map DAG nodes to script execution specifications.
        
        Mirrors _create_config_map in DynamicPipelineTemplate but creates
        ScriptExecutionSpecs instead of step configurations.
        
        Returns:
            Dictionary mapping node names to ScriptExecutionSpecs
        """
        script_specs = {}
        
        logger.debug(f"Creating script spec map for {len(self.dag.nodes)} nodes")
        
        for node_name in self.dag.nodes:
            try:
                # Resolve node to script spec (mirrors config resolution)
                script_spec = self._resolve_node_to_script_spec(node_name)
                script_specs[node_name] = script_spec
                
                logger.debug(f"Created script spec for {node_name}: {script_spec.script_path}")
                
            except Exception as e:
                logger.error(f"Failed to create script spec for {node_name}: {e}")
                raise ScriptCompilationError(
                    f"Failed to resolve node '{node_name}' to script spec: {e}",
                    node_name=node_name
                ) from e
        
        self._script_specs = script_specs
        return script_specs
    
    def _resolve_node_to_script_spec(self, node_name: str) -> ScriptExecutionSpec:
        """
        Resolve DAG node to script execution specification.
        
        Uses step catalog for enhanced script discovery and contract-aware 
        path resolution, mirroring the config resolution in DynamicPipelineTemplate.
        
        Args:
            node_name: DAG node name to resolve
            
        Returns:
            ScriptExecutionSpec for the node
            
        Raises:
            ScriptDiscoveryError: If script cannot be discovered
            StepCatalogIntegrationError: If step catalog integration fails
        """
        try:
            # 1. DIRECT REUSE: Use step catalog for script discovery
            script_path = None
            discovery_method = "fallback"
            
            if self.step_catalog:
                try:
                    step_info = self.step_catalog.resolve_pipeline_node(node_name)
                    if step_info and step_info.file_components.get('script'):
                        script_metadata = step_info.file_components['script']
                        script_path = str(script_metadata.path)
                        discovery_method = "step_catalog"
                        logger.debug(f"Discovered script via step catalog: {node_name} -> {script_path}")
                except Exception as e:
                    logger.debug(f"Step catalog discovery failed for {node_name}: {e}")
                    # Continue to fallback discovery
            
            # Fallback to traditional discovery
            if not script_path:
                script_path = self._discover_script_fallback(node_name)
                logger.debug(f"Discovered script via fallback: {node_name} -> {script_path}")
            
            # 2. DIRECT REUSE: Get contract-aware paths using step catalog
            input_paths, output_paths = self._get_contract_aware_paths(node_name)
            
            # 3. Extract user inputs for this node
            node_inputs = self.user_inputs.get(node_name, {})
            
            # 4. Create script execution spec
            script_spec = ScriptExecutionSpec(
                script_name=Path(script_path).stem,
                step_name=node_name,
                script_path=script_path,
                input_paths=input_paths,
                output_paths=output_paths,
                environ_vars=node_inputs.get('environ_vars', {}),
                job_args=node_inputs.get('job_args', {}),
                user_notes=f"Generated by ScriptExecutionTemplate using {discovery_method} discovery"
            )
            
            return script_spec
            
        except Exception as e:
            if isinstance(e, (ScriptDiscoveryError, StepCatalogIntegrationError)):
                raise
            raise ScriptDiscoveryError(
                f"Failed to resolve script for node: {e}",
                node_name=node_name,
                step_catalog_available=self.step_catalog is not None
            ) from e
    
    def _discover_script_fallback(self, node_name: str) -> str:
        """
        Fallback script discovery using traditional naming conventions.
        
        Args:
            node_name: DAG node name
            
        Returns:
            Script path using fallback discovery
        """
        # DIRECT REUSE: Use existing step name conversion
        script_name = ScriptExecutionSpec._convert_step_name_to_script_name(node_name)
        script_path = str(Path(self.test_workspace_dir) / "scripts" / f"{script_name}.py")
        
        return script_path
    
    def _get_contract_aware_paths(self, node_name: str) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Get contract-aware input and output paths using step catalog.
        
        DIRECT REUSE: Uses step catalog contract loading capabilities,
        mirroring the contract-aware configuration in DynamicPipelineTemplate.
        
        Args:
            node_name: DAG node name
            
        Returns:
            Tuple of (input_paths, output_paths) dictionaries
        """
        input_paths = {}
        output_paths = {}
        
        try:
            # DIRECT REUSE: Use step catalog for contract discovery
            if self.step_catalog:
                contract = self.step_catalog.load_contract_class(node_name)
                
                if contract:
                    # Get contract-defined paths
                    if hasattr(contract, 'get_input_paths'):
                        contract_inputs = contract.get_input_paths()
                        if contract_inputs:
                            input_paths = {
                                name: str(Path(self.test_workspace_dir) / "input" / f"{name}.json")
                                for name in contract_inputs.keys()
                            }
                    
                    if hasattr(contract, 'get_output_paths'):
                        contract_outputs = contract.get_output_paths()
                        if contract_outputs:
                            output_paths = {
                                name: str(Path(self.test_workspace_dir) / "output" / f"{name}.json")
                                for name in contract_outputs.keys()
                            }
                            
                    logger.debug(f"Contract-aware paths for {node_name}: "
                               f"inputs={len(input_paths)}, outputs={len(output_paths)}")
        except Exception as e:
            logger.debug(f"Contract loading failed for {node_name}: {e}")
            # Continue with fallback paths
        
        # Fallback to default paths if no contract available
        if not input_paths:
            script_name = ScriptExecutionSpec._convert_step_name_to_script_name(node_name)
            input_paths = {
                "input_data": str(Path(self.test_workspace_dir) / "input" / f"{script_name}_input.json"),
            }
        
        if not output_paths:
            script_name = ScriptExecutionSpec._convert_step_name_to_script_name(node_name)
            output_paths = {
                "output_data": str(Path(self.test_workspace_dir) / "output" / f"{script_name}_output.json"),
            }
        
        return input_paths, output_paths
    
    def _resolve_script_dependencies(
        self, 
        script_specs: Dict[str, ScriptExecutionSpec], 
        execution_order: List[str]
    ) -> None:
        """
        Resolve dependencies between scripts and update input paths.
        
        This method uses the dependency resolver to connect script outputs
        to dependent script inputs, mirroring the dependency resolution
        in DynamicPipelineTemplate.
        
        Args:
            script_specs: Dictionary of script specifications
            execution_order: Execution order from topological sort
        """
        logger.debug("Resolving script dependencies")
        
        # Track script outputs for dependency resolution
        script_outputs: Dict[str, Dict[str, str]] = {}
        
        for node_name in execution_order:
            script_spec = script_specs[node_name]
            
            # Get dependencies for this node (DIRECT REUSE)
            dependencies = self.dag.get_dependencies(node_name)
            
            if dependencies:
                logger.debug(f"Resolving dependencies for {node_name}: {dependencies}")
                
                # Collect outputs from dependency nodes
                dependency_outputs = {}
                for dep_node in dependencies:
                    if dep_node in script_outputs:
                        dependency_outputs[dep_node] = script_outputs[dep_node]
                
                # Update script spec with resolved dependencies
                if dependency_outputs:
                    script_spec.update_paths_from_dependencies(dependency_outputs)
                    logger.debug(f"Updated input paths for {node_name} from dependencies")
            
            # Register this script's outputs for future dependency resolution
            script_outputs[node_name] = script_spec.output_paths.copy()
    
    def _validate_execution_plan_components(
        self, 
        script_specs: Dict[str, ScriptExecutionSpec], 
        execution_order: List[str]
    ) -> None:
        """
        Validate execution plan components before creating the plan.
        
        Args:
            script_specs: Dictionary of script specifications
            execution_order: Execution order from topological sort
            
        Raises:
            ScriptCompilationError: If validation fails
        """
        validation_errors = []
        validation_warnings = []
        
        # Validate each script spec
        for node_name, script_spec in script_specs.items():
            try:
                spec_validation = validate_script_spec(script_spec)
                
                if not spec_validation["valid"]:
                    validation_errors.extend([
                        f"{node_name}: {error}" for error in spec_validation["errors"]
                    ])
                
                validation_warnings.extend([
                    f"{node_name}: {warning}" for warning in spec_validation.get("warnings", [])
                ])
                
            except Exception as e:
                validation_errors.append(f"{node_name}: Spec validation error: {e}")
        
        # Validate execution order matches DAG nodes
        if set(execution_order) != set(script_specs.keys()):
            validation_errors.append("Execution order does not match script specs")
        
        # Log warnings
        for warning in validation_warnings:
            logger.warning(f"Script spec validation warning: {warning}")
        
        # Raise error if validation failed
        if validation_errors:
            error_msg = f"Script spec validation failed: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            raise ScriptCompilationError(error_msg)
    
    def preview_script_resolution(self) -> Dict[str, Any]:
        """
        Preview script resolution without creating the full execution plan.
        
        Returns:
            Dictionary with script resolution preview information
        """
        preview = {
            "total_nodes": len(self.dag.nodes),
            "script_resolution": {},
            "dependency_analysis": {},
            "user_inputs_summary": {},
        }
        
        try:
            # Preview script resolution for each node
            for node_name in self.dag.nodes:
                resolution_info = {
                    "script_path": None,
                    "script_exists": False,
                    "discovery_method": None,
                    "contract_available": False,
                    "user_inputs_provided": node_name in self.user_inputs,
                }
                
                try:
                    # Try step catalog discovery
                    if self.step_catalog:
                        step_info = self.step_catalog.resolve_pipeline_node(node_name)
                        if step_info and step_info.file_components.get('script'):
                            script_metadata = step_info.file_components['script']
                            resolution_info["script_path"] = str(script_metadata.path)
                            resolution_info["discovery_method"] = "step_catalog"
                            resolution_info["script_exists"] = Path(script_metadata.path).exists()
                        
                        # Check for contract
                        try:
                            contract = self.step_catalog.load_contract_class(node_name)
                            resolution_info["contract_available"] = contract is not None
                        except Exception:
                            pass
                    
                    # Fallback discovery if needed
                    if not resolution_info["script_path"]:
                        fallback_path = self._discover_script_fallback(node_name)
                        resolution_info["script_path"] = fallback_path
                        resolution_info["discovery_method"] = "fallback"
                        resolution_info["script_exists"] = Path(fallback_path).exists()
                    
                except Exception as e:
                    resolution_info["error"] = str(e)
                
                preview["script_resolution"][node_name] = resolution_info
            
            # Dependency analysis
            for node_name in self.dag.nodes:
                dependencies = self.dag.get_dependencies(node_name)
                if dependencies:
                    preview["dependency_analysis"][node_name] = {
                        "depends_on": dependencies,
                        "dependency_count": len(dependencies),
                    }
            
            # User inputs summary
            for node_name, inputs in self.user_inputs.items():
                preview["user_inputs_summary"][node_name] = {
                    "environ_vars_count": len(inputs.get("environ_vars", {})),
                    "job_args_count": len(inputs.get("job_args", {})),
                    "input_paths_count": len(inputs.get("input_paths", {})),
                    "output_paths_count": len(inputs.get("output_paths", {})),
                }
            
        except Exception as e:
            preview["error"] = str(e)
        
        return preview
    
    def get_template_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the template configuration and state.
        
        Returns:
            Dictionary with template summary
        """
        return {
            "template_info": {
                "dag_nodes": len(self.dag.nodes),
                "dag_edges": len(self.dag.edges),
                "test_workspace_dir": self.test_workspace_dir,
                "user_inputs_provided": len(self.user_inputs),
                "step_catalog_available": self.step_catalog is not None,
            },
            "configuration": self.config.copy(),
            "state": {
                "script_specs_created": self._script_specs is not None,
                "execution_order_determined": self._execution_order is not None,
            }
        }
    
    def __str__(self) -> str:
        """String representation of the template."""
        return f"ScriptExecutionTemplate(nodes={len(self.dag.nodes)}, workspace='{self.test_workspace_dir}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ScriptExecutionTemplate("
            f"dag_nodes={len(self.dag.nodes)}, "
            f"dag_edges={len(self.dag.edges)}, "
            f"test_workspace_dir='{self.test_workspace_dir}', "
            f"user_inputs={len(self.user_inputs)}, "
            f"step_catalog_available={self.step_catalog is not None})"
        )
