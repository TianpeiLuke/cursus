"""
Script DAG Compiler

Compiles a PipelineDAG into a complete Script Execution Plan.
This mirrors PipelineDAGCompiler but targets script execution instead of 
SageMaker pipeline generation. Uses maximum component reuse.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Direct imports from existing cursus components - MAXIMUM REUSE
from ...api.dag.base_dag import PipelineDAG
from ...step_catalog import StepCatalog
from ...core.deps.factory import create_pipeline_components
from ...core.deps.dependency_resolver import create_dependency_resolver
from ..base.script_execution_spec import ScriptExecutionSpec
from ..base.script_execution_plan import ScriptExecutionPlan
from .script_execution_template import ScriptExecutionTemplate
from .exceptions import ScriptCompilationError, ScriptDiscoveryError
from .validation import validate_dag_for_script_execution

logger = logging.getLogger(__name__)


class ScriptDAGCompiler:
    """
    Compile a PipelineDAG into a complete Script Execution Plan.
    
    This class mirrors PipelineDAGCompiler functionality for script testing:
    1. Load and validate the DAG
    2. Optionally collect interactive inputs
    3. Create execution template
    4. Generate complete execution plan
    
    Uses maximum component reuse from existing cursus infrastructure to 
    maintain consistency and reduce code duplication.
    
    Attributes:
        dag: PipelineDAG defining the execution structure
        test_workspace_dir: Directory for test workspace and script discovery
        step_catalog: StepCatalog for enhanced script discovery and validation
        components: Pipeline components for dependency resolution
        dependency_resolver: Dependency resolver for script I/O connections
        config: Compiler configuration options
    """
    
    def __init__(
        self,
        dag: PipelineDAG,
        test_workspace_dir: str,
        step_catalog: Optional[StepCatalog] = None,
        **kwargs: Any,
    ):
        """
        Initialize the Script DAG Compiler.
        
        Args:
            dag: PipelineDAG instance defining the pipeline structure
            test_workspace_dir: Directory for test workspace and script discovery
            step_catalog: Optional step catalog for enhanced script discovery
            **kwargs: Additional compiler configuration options
        """
        self.dag = dag
        self.test_workspace_dir = test_workspace_dir
        self.step_catalog = step_catalog or StepCatalog()
        
        # DIRECT REUSE: Use existing pipeline components factory
        try:
            self.components = create_pipeline_components()
        except Exception as e:
            logger.warning(f"Failed to create pipeline components: {e}")
            self.components = None
        
        # DIRECT REUSE: Use existing dependency resolver
        try:
            self.dependency_resolver = create_dependency_resolver()
        except Exception as e:
            logger.warning(f"Failed to create dependency resolver: {e}")
            self.dependency_resolver = None
        
        # Compiler configuration
        self.config = {
            "validate_scripts_exist": kwargs.get("validate_scripts_exist", True),
            "require_main_function": kwargs.get("require_main_function", True),
            "enable_framework_detection": kwargs.get("enable_framework_detection", True),
            "enable_builder_consistency": kwargs.get("enable_builder_consistency", True),
            **kwargs
        }
        
        logger.info(f"Initialized ScriptDAGCompiler for DAG with {len(dag.nodes)} nodes")
    
    def compile_dag_to_execution_plan(
        self, 
        user_inputs: Optional[Dict[str, Any]] = None,
        collect_inputs: bool = False
    ) -> ScriptExecutionPlan:
        """
        Compile DAG to script execution plan.
        
        This is the main compilation method that mirrors compile_dag_to_pipeline
        from cursus/core but creates a script execution plan instead.
        
        Args:
            user_inputs: Optional pre-collected user inputs for script execution
            collect_inputs: Whether to collect inputs interactively (requires factory)
            
        Returns:
            ScriptExecutionPlan ready for execution
            
        Raises:
            ScriptCompilationError: If compilation fails
            ScriptDiscoveryError: If scripts cannot be discovered
        """
        try:
            logger.info("Starting DAG compilation to script execution plan")
            
            # 1. Validate DAG for script execution
            self._validate_dag()
            
            # 2. Handle input collection if requested
            if collect_inputs and user_inputs is None:
                user_inputs = self._collect_inputs_interactively()
            elif user_inputs is None:
                user_inputs = {}
            
            # 3. Create script execution template (MIRROR cursus/core/compiler patterns)
            template = self._create_execution_template(user_inputs)
            
            # 4. Generate execution plan
            execution_plan = template.create_execution_plan()
            
            # 5. Validate the generated execution plan
            self._validate_execution_plan(execution_plan)
            
            logger.info(f"Successfully compiled DAG to execution plan with {len(execution_plan.script_specs)} scripts")
            return execution_plan
            
        except Exception as e:
            logger.error(f"Failed to compile DAG to script execution plan: {e}")
            raise ScriptCompilationError(f"Compilation failed: {e}") from e
    
    def _validate_dag(self) -> None:
        """
        Validate the DAG for script execution.
        
        Uses the validation module to ensure the DAG is suitable for script execution.
        """
        try:
            validation_result = validate_dag_for_script_execution(
                self.dag, 
                self.test_workspace_dir,
                self.step_catalog
            )
            
            if not validation_result["valid"]:
                errors = validation_result.get("errors", [])
                raise ScriptCompilationError(f"DAG validation failed: {'; '.join(errors)}")
                
            # Log warnings if any
            warnings = validation_result.get("warnings", [])
            for warning in warnings:
                logger.warning(f"DAG validation warning: {warning}")
                
        except Exception as e:
            raise ScriptCompilationError(f"DAG validation error: {e}") from e
    
    def _collect_inputs_interactively(self) -> Dict[str, Any]:
        """
        Collect user inputs interactively using the factory pattern.
        
        This method would use InteractiveScriptTestingFactory when available,
        but for now provides a placeholder implementation.
        """
        logger.info("Interactive input collection requested but not yet implemented")
        
        # TODO: Implement when InteractiveScriptTestingFactory is available
        # from ..factory.interactive_script_factory import InteractiveScriptTestingFactory
        # factory = InteractiveScriptTestingFactory(self.test_workspace_dir, self.step_catalog)
        # return factory.collect_inputs_for_dag(self.dag)
        
        return {}
    
    def _create_execution_template(self, user_inputs: Dict[str, Any]) -> ScriptExecutionTemplate:
        """
        Create script execution template.
        
        This mirrors the template creation in cursus/core/compiler.
        
        Args:
            user_inputs: User inputs for script execution
            
        Returns:
            ScriptExecutionTemplate for generating the execution plan
        """
        return ScriptExecutionTemplate(
            dag=self.dag,
            user_inputs=user_inputs,
            test_workspace_dir=self.test_workspace_dir,
            step_catalog=self.step_catalog,
            dependency_resolver=self.dependency_resolver,
            config=self.config,
        )
    
    def _validate_execution_plan(self, execution_plan: ScriptExecutionPlan) -> None:
        """
        Validate the generated execution plan.
        
        Args:
            execution_plan: The generated execution plan to validate
            
        Raises:
            ScriptCompilationError: If validation fails
        """
        try:
            validation_result = execution_plan.validate_execution_plan()
            
            if not validation_result["plan_valid"]:
                errors = validation_result.get("errors", [])
                raise ScriptCompilationError(f"Execution plan validation failed: {'; '.join(errors)}")
            
            # Log warnings
            warnings = validation_result.get("warnings", [])
            for warning in warnings:
                logger.warning(f"Execution plan warning: {warning}")
                
        except Exception as e:
            raise ScriptCompilationError(f"Execution plan validation error: {e}") from e
    
    def preview_compilation(self) -> Dict[str, Any]:
        """
        Preview the compilation without actually creating the full execution plan.
        
        Returns:
            Dictionary with compilation preview information
        """
        try:
            preview = {
                "dag_info": {
                    "total_nodes": len(self.dag.nodes),
                    "total_edges": len(self.dag.edges),
                    "execution_order": self.dag.topological_sort(),
                },
                "script_discovery": {},
                "step_catalog_analysis": {},
                "validation_preview": {},
            }
            
            # Preview script discovery for each node
            for node_name in self.dag.nodes:
                try:
                    # Use step catalog for discovery preview
                    step_info = self.step_catalog.resolve_pipeline_node(node_name)
                    
                    if step_info and step_info.file_components.get('script'):
                        script_metadata = step_info.file_components['script']
                        script_path = str(script_metadata.path)
                        script_exists = Path(script_path).exists()
                    else:
                        # Fallback discovery
                        script_name = ScriptExecutionSpec._convert_step_name_to_script_name(node_name)
                        script_path = str(Path(self.test_workspace_dir) / "scripts" / f"{script_name}.py")
                        script_exists = Path(script_path).exists()
                    
                    preview["script_discovery"][node_name] = {
                        "script_path": script_path,
                        "script_exists": script_exists,
                        "discovered_via_step_catalog": step_info is not None,
                    }
                    
                    # Step catalog analysis
                    if step_info:
                        preview["step_catalog_analysis"][node_name] = {
                            "workspace_id": getattr(step_info, 'workspace_id', 'unknown'),
                            "has_contract": 'contract' in step_info.file_components,
                            "has_builder": 'builder' in step_info.file_components,
                            "has_spec": 'spec' in step_info.file_components,
                        }
                        
                except Exception as e:
                    preview["script_discovery"][node_name] = {
                        "error": str(e),
                        "script_exists": False,
                    }
            
            # Validation preview
            try:
                validation_result = validate_dag_for_script_execution(
                    self.dag, 
                    self.test_workspace_dir,
                    self.step_catalog
                )
                preview["validation_preview"] = validation_result
            except Exception as e:
                preview["validation_preview"] = {"error": str(e)}
            
            return preview
            
        except Exception as e:
            logger.error(f"Failed to generate compilation preview: {e}")
            return {"error": str(e)}
    
    def get_compilation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the compilation configuration and capabilities.
        
        Returns:
            Dictionary with compilation summary
        """
        return {
            "compiler_info": {
                "dag_nodes": len(self.dag.nodes),
                "dag_edges": len(self.dag.edges),
                "test_workspace_dir": self.test_workspace_dir,
                "step_catalog_available": self.step_catalog is not None,
                "dependency_resolver_available": self.dependency_resolver is not None,
            },
            "configuration": self.config.copy(),
            "capabilities": {
                "script_discovery": True,
                "step_catalog_integration": self.step_catalog is not None,
                "dependency_resolution": self.dependency_resolver is not None,
                "framework_detection": self.config.get("enable_framework_detection", False),
                "builder_consistency": self.config.get("enable_builder_consistency", False),
            }
        }
    
    def __str__(self) -> str:
        """String representation of the compiler."""
        return f"ScriptDAGCompiler(nodes={len(self.dag.nodes)}, workspace='{self.test_workspace_dir}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ScriptDAGCompiler("
            f"dag_nodes={len(self.dag.nodes)}, "
            f"dag_edges={len(self.dag.edges)}, "
            f"test_workspace_dir='{self.test_workspace_dir}', "
            f"step_catalog_available={self.step_catalog is not None})"
        )
