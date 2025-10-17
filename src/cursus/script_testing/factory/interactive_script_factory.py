"""
Interactive Script Testing Factory

Interactive factory for collecting script testing inputs.
Mirrors the interactive config collection patterns in cursus/api/factory
but targets script execution parameters instead of step builder configs.
Uses maximum component reuse from existing cursus infrastructure.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Direct imports from existing cursus components - MAXIMUM REUSE
from ...step_catalog import StepCatalog
from ...api.dag import PipelineDAG
from ..compiler.script_dag_compiler import ScriptDAGCompiler
from ..base.script_execution_plan import ScriptExecutionPlan
from .script_input_collector import ScriptInputCollector

logger = logging.getLogger(__name__)


class InteractiveScriptTestingFactory:
    """
    Interactive factory for collecting script testing inputs.
    
    Mirrors the interactive config collection patterns in cursus/api/factory
    but targets script execution parameters instead of step builder configs.
    Uses maximum component reuse from existing cursus infrastructure.
    
    This class follows the same patterns as interactive factories in cursus/api/factory:
    1. Progressive input collection with DAG context awareness
    2. Step catalog integration for contract-aware suggestions
    3. Dependency-aware input collection showing context
    4. Validation and error handling during collection
    
    Attributes:
        test_workspace_dir: Directory for test workspace and outputs
        step_catalog: StepCatalog for enhanced script discovery and validation
        input_collector: ScriptInputCollector for individual node input collection
    """
    
    def __init__(
        self,
        test_workspace_dir: str,
        step_catalog: Optional[StepCatalog] = None,
    ):
        """
        Initialize the Interactive Script Testing Factory.
        
        Args:
            test_workspace_dir: Directory for test workspace and outputs
            step_catalog: Optional step catalog for enhanced script operations
        """
        self.test_workspace_dir = test_workspace_dir
        self.step_catalog = step_catalog or StepCatalog()  # DIRECT REUSE
        self.input_collector = ScriptInputCollector(self.step_catalog)
        
        # Ensure test workspace directory exists
        Path(self.test_workspace_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized InteractiveScriptTestingFactory with workspace: {test_workspace_dir}")
    
    def collect_inputs_for_dag(self, dag: PipelineDAG) -> Dict[str, Any]:
        """
        Collect user inputs for all nodes in DAG.
        
        This method mirrors the interactive collection patterns in cursus/api/factory
        but targets script execution parameters.
        
        Args:
            dag: PipelineDAG to collect inputs for
            
        Returns:
            Dictionary mapping node names to collected inputs
        """
        logger.info(f"Starting interactive input collection for DAG with {len(dag.nodes)} nodes")
        
        user_inputs = {}
        execution_order = dag.topological_sort()  # DIRECT REUSE
        
        print(f"\n🔧 Collecting inputs for {len(execution_order)} scripts in DAG...")
        print(f"📁 Test workspace: {self.test_workspace_dir}")
        print("=" * 60)
        
        for i, node_name in enumerate(execution_order, 1):
            print(f"\n📝 [{i}/{len(execution_order)}] Configuring script: {node_name}")
            
            # Get dependencies to show context (DIRECT REUSE)
            dependencies = dag.get_dependencies(node_name)
            if dependencies:
                print(f"   🔗 Dependencies: {', '.join(dependencies)}")
            
            # Get dependents to show impact
            dependents = self._get_dependents(dag, node_name)
            if dependents:
                print(f"   ⬇️  Used by: {', '.join(dependents)}")
            
            # Collect inputs for this node
            try:
                node_inputs = self.input_collector.collect_node_inputs(
                    node_name, 
                    dependencies,
                    user_inputs,  # Pass previous inputs for context
                    self.test_workspace_dir
                )
                
                user_inputs[node_name] = node_inputs
                print(f"   ✅ Configuration completed for {node_name}")
                
            except KeyboardInterrupt:
                print(f"\n❌ Input collection cancelled by user")
                raise
            except Exception as e:
                print(f"   ⚠️  Error collecting inputs for {node_name}: {e}")
                logger.error(f"Error collecting inputs for {node_name}: {e}")
                # Continue with empty inputs for this node
                user_inputs[node_name] = {
                    "environ_vars": {},
                    "job_args": {},
                    "input_paths": {},
                    "output_paths": {},
                }
        
        print("\n" + "=" * 60)
        print(f"✅ Input collection completed for {len(user_inputs)} scripts")
        
        return user_inputs
    
    def create_execution_plan_interactively(
        self, 
        dag: PipelineDAG,
        collect_inputs: bool = True
    ) -> ScriptExecutionPlan:
        """
        Create a complete script execution plan with interactive input collection.
        
        This method combines the compiler and interactive collection to create
        a ready-to-execute script execution plan.
        
        Args:
            dag: PipelineDAG to create execution plan for
            collect_inputs: Whether to collect inputs interactively
            
        Returns:
            Complete ScriptExecutionPlan ready for execution
        """
        logger.info("Creating script execution plan interactively")
        
        # Create compiler
        compiler = ScriptDAGCompiler(
            dag=dag,
            test_workspace_dir=self.test_workspace_dir,
            step_catalog=self.step_catalog
        )
        
        # Collect inputs if requested
        user_inputs = {}
        if collect_inputs:
            user_inputs = self.collect_inputs_for_dag(dag)
        
        # Compile to execution plan
        execution_plan = compiler.compile_dag_to_execution_plan(
            user_inputs=user_inputs,
            collect_inputs=False  # We already collected inputs
        )
        
        return execution_plan
    
    def preview_script_collection(self, dag: PipelineDAG) -> Dict[str, Any]:
        """
        Preview the script collection process without actually collecting inputs.
        
        Args:
            dag: PipelineDAG to preview collection for
            
        Returns:
            Dictionary with collection preview information
        """
        preview = {
            "collection_summary": {
                "total_scripts": len(dag.nodes),
                "execution_order": dag.topological_sort(),
                "test_workspace": self.test_workspace_dir,
            },
            "script_preview": {},
            "dependency_analysis": {},
        }
        
        execution_order = dag.topological_sort()
        
        for node_name in execution_order:
            dependencies = dag.get_dependencies(node_name)
            dependents = self._get_dependents(dag, node_name)
            
            # Get step catalog information if available
            step_info = None
            try:
                step_info = self.step_catalog.resolve_pipeline_node(node_name)
            except Exception:
                pass
            
            preview["script_preview"][node_name] = {
                "dependencies": dependencies,
                "dependents": dependents,
                "step_catalog_available": step_info is not None,
                "script_discoverable": step_info is not None and step_info.file_components.get('script') is not None,
                "contract_available": step_info is not None and step_info.file_components.get('contract') is not None,
            }
            
            # Analyze dependency relationships
            if dependencies or dependents:
                preview["dependency_analysis"][node_name] = {
                    "input_dependencies": dependencies,
                    "output_dependents": dependents,
                    "dependency_depth": len(dependencies),
                    "dependent_count": len(dependents),
                }
        
        return preview
    
    def validate_collection_readiness(self, dag: PipelineDAG) -> Dict[str, Any]:
        """
        Validate that the factory is ready to collect inputs for the given DAG.
        
        Args:
            dag: PipelineDAG to validate collection readiness for
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "ready_for_collection": True,
            "errors": [],
            "warnings": [],
            "node_validations": {},
        }
        
        # 1. Validate test workspace
        workspace_path = Path(self.test_workspace_dir)
        if not workspace_path.exists():
            validation_result["warnings"].append(f"Test workspace directory does not exist: {self.test_workspace_dir}")
        elif not workspace_path.is_dir():
            validation_result["ready_for_collection"] = False
            validation_result["errors"].append(f"Test workspace path is not a directory: {self.test_workspace_dir}")
        
        # 2. Validate DAG structure
        try:
            execution_order = dag.topological_sort()
        except Exception as e:
            validation_result["ready_for_collection"] = False
            validation_result["errors"].append(f"DAG is not valid: {e}")
            return validation_result
        
        # 3. Validate each node
        for node_name in execution_order:
            node_validation = {
                "node_valid": True,
                "step_catalog_available": False,
                "script_discoverable": False,
                "contract_available": False,
                "issues": []
            }
            
            # Check step catalog availability
            try:
                step_info = self.step_catalog.resolve_pipeline_node(node_name)
                if step_info:
                    node_validation["step_catalog_available"] = True
                    
                    if step_info.file_components.get('script'):
                        node_validation["script_discoverable"] = True
                    else:
                        node_validation["issues"].append("Script not discoverable via step catalog")
                    
                    if step_info.file_components.get('contract'):
                        node_validation["contract_available"] = True
                    else:
                        node_validation["issues"].append("Contract not available for enhanced input suggestions")
                else:
                    node_validation["issues"].append("Node not found in step catalog")
            except Exception as e:
                node_validation["issues"].append(f"Step catalog error: {e}")
            
            # Check dependencies
            dependencies = dag.get_dependencies(node_name)
            for dep in dependencies:
                if dep not in dag.nodes:
                    node_validation["node_valid"] = False
                    node_validation["issues"].append(f"Dependency {dep} not found in DAG")
            
            validation_result["node_validations"][node_name] = node_validation
            
            if not node_validation["node_valid"]:
                validation_result["ready_for_collection"] = False
                validation_result["errors"].extend([f"{node_name}: {issue}" for issue in node_validation["issues"]])
            elif node_validation["issues"]:
                validation_result["warnings"].extend([f"{node_name}: {issue}" for issue in node_validation["issues"]])
        
        return validation_result
    
    def _get_dependents(self, dag: PipelineDAG, node_name: str) -> List[str]:
        """
        Get dependents (nodes that depend on this node) for a specific script.
        
        Args:
            dag: PipelineDAG to analyze
            node_name: Name of the node to get dependents for
            
        Returns:
            List of dependent node names
        """
        dependents = []
        for other_node in dag.nodes:
            if node_name in dag.get_dependencies(other_node):
                dependents.append(other_node)
        return dependents
    
    def get_factory_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the factory configuration and capabilities.
        
        Returns:
            Dictionary with factory summary information
        """
        return {
            "test_workspace_dir": self.test_workspace_dir,
            "workspace_exists": Path(self.test_workspace_dir).exists(),
            "step_catalog_available": self.step_catalog is not None,
            "input_collector_ready": self.input_collector is not None,
            "factory_type": "InteractiveScriptTestingFactory",
            "mirrors_pattern": "cursus/api/factory interactive collection patterns",
        }
    
    def __str__(self) -> str:
        """String representation of the factory."""
        return f"InteractiveScriptTestingFactory(workspace='{self.test_workspace_dir}')"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"InteractiveScriptTestingFactory("
            f"workspace='{self.test_workspace_dir}', "
            f"step_catalog={self.step_catalog is not None}, "
            f"input_collector={self.input_collector is not None})"
        )
