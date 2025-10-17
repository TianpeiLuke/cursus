"""
Script Execution Plan

Defines the execution plan for DAG-guided script testing.
This mirrors the pipeline execution plans in cursus/core but targets script execution.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from pathlib import Path
import json
from datetime import datetime

from .script_execution_spec import ScriptExecutionSpec
from .script_test_result import ScriptTestResult
from ...api.dag.base_dag import PipelineDAG
# Direct imports from existing cursus components
from ...core.deps.dependency_resolver import create_dependency_resolver
from ...step_catalog import StepCatalog


class ScriptExecutionPlan(BaseModel):
    """
    Complete execution plan for DAG-guided script testing.
    
    This class represents a complete, executable plan for running scripts
    in DAG order with dependency resolution, mirroring the pipeline execution
    plans in cursus/core.
    
    Attributes:
        dag: PipelineDAG defining the execution structure
        script_specs: Dictionary mapping node names to script execution specs
        execution_order: List of node names in topological execution order
        test_workspace_dir: Directory for test workspace and outputs
        created_at: Timestamp when the plan was created
        metadata: Additional metadata about the execution plan
    """
    
    # Core Plan Structure
    dag: PipelineDAG = Field(..., description="PipelineDAG defining execution structure")
    script_specs: Dict[str, ScriptExecutionSpec] = Field(..., description="Node to script spec mapping")
    execution_order: List[str] = Field(..., description="Topological execution order")
    test_workspace_dir: str = Field(..., description="Test workspace directory")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Plan creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional plan metadata")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
            PipelineDAG: lambda v: v.model_dump() if hasattr(v, 'model_dump') else str(v),
        }
        arbitrary_types_allowed = True  # Allow PipelineDAG
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the script execution plan.
        
        This method creates a ScriptAssembler and executes all scripts
        in the plan with dependency resolution.
        
        Returns:
            Dictionary with execution results
        """
        from ..assembler.script_assembler import ScriptAssembler
        
        # Create assembler and execute
        assembler = ScriptAssembler(execution_plan=self)
        return assembler.execute_dag_scripts()
    
    def execute_with_detailed_reporting(self) -> Dict[str, Any]:
        """
        Execute with enhanced reporting and validation.
        
        Returns:
            Dictionary with detailed execution results and analysis
        """
        from ..assembler.script_assembler import ScriptAssembler
        
        # Create assembler with enhanced reporting
        assembler = ScriptAssembler(execution_plan=self)
        
        # Execute with timing and detailed analysis
        start_time = datetime.now()
        results = assembler.execute_dag_scripts()
        end_time = datetime.now()
        
        # Add detailed reporting
        results.update({
            "execution_plan_metadata": {
                "total_scripts": len(self.script_specs),
                "execution_order": self.execution_order,
                "plan_created_at": self.created_at.isoformat(),
                "execution_started_at": start_time.isoformat(),
                "execution_completed_at": end_time.isoformat(),
                "total_execution_time": (end_time - start_time).total_seconds(),
            },
            "script_specs_summary": {
                node_name: {
                    "script_path": spec.script_path,
                    "input_count": len(spec.input_paths),
                    "output_count": len(spec.output_paths),
                    "has_environ_vars": bool(spec.environ_vars),
                    "has_job_args": bool(spec.job_args),
                }
                for node_name, spec in self.script_specs.items()
            },
            "dag_analysis": {
                "total_nodes": len(self.dag.nodes),
                "total_edges": len(self.dag.edges),
                "has_cycles": not self._validate_dag_acyclic(),
            }
        })
        
        return results
    
    def validate_execution_plan(self) -> Dict[str, Any]:
        """
        Validate the execution plan before execution.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "plan_valid": True,
            "errors": [],
            "warnings": [],
            "script_validations": {},
        }
        
        # 1. Validate DAG structure
        if not self._validate_dag_acyclic():
            validation_result["plan_valid"] = False
            validation_result["errors"].append("DAG contains cycles")
        
        # 2. Validate execution order matches DAG
        try:
            expected_order = self.dag.topological_sort()
            if self.execution_order != expected_order:
                validation_result["warnings"].append(
                    f"Execution order may not be optimal. Expected: {expected_order}"
                )
        except Exception as e:
            validation_result["plan_valid"] = False
            validation_result["errors"].append(f"Cannot determine topological order: {e}")
        
        # 3. Validate each script specification
        for node_name, spec in self.script_specs.items():
            spec_validation = spec.validate_paths_exist(check_inputs=True, check_outputs=False)
            validation_result["script_validations"][node_name] = spec_validation
            
            if not spec_validation["all_valid"]:
                validation_result["plan_valid"] = False
                if not spec_validation["script_exists"]:
                    validation_result["errors"].append(f"Script not found for {node_name}: {spec.script_path}")
                for missing_input in spec_validation["missing_inputs"]:
                    validation_result["warnings"].append(f"Missing input for {node_name}: {missing_input}")
        
        # 4. Validate workspace directory
        workspace_path = Path(self.test_workspace_dir)
        if not workspace_path.exists():
            validation_result["warnings"].append(f"Test workspace directory does not exist: {self.test_workspace_dir}")
        
        return validation_result
    
    def preview_execution(self) -> Dict[str, Any]:
        """
        Preview the execution plan without actually running scripts.
        
        Returns:
            Dictionary with execution preview information
        """
        preview = {
            "execution_summary": {
                "total_scripts": len(self.script_specs),
                "execution_order": self.execution_order,
                "estimated_duration": "Unknown",  # Could be enhanced with historical data
            },
            "script_preview": {},
            "dependency_analysis": {},
        }
        
        # Generate script previews
        for node_name in self.execution_order:
            spec = self.script_specs[node_name]
            dependencies = self.dag.get_dependencies(node_name)
            
            preview["script_preview"][node_name] = {
                "script_path": spec.script_path,
                "script_exists": Path(spec.script_path).exists(),
                "dependencies": dependencies,
                "input_paths": list(spec.input_paths.keys()),
                "output_paths": list(spec.output_paths.keys()),
                "environ_vars_count": len(spec.environ_vars),
                "job_args_count": len(spec.job_args),
            }
            
            # Analyze dependencies
            if dependencies:
                preview["dependency_analysis"][node_name] = {
                    "depends_on": dependencies,
                    "dependency_outputs": {
                        dep: list(self.script_specs[dep].output_paths.keys())
                        for dep in dependencies
                        if dep in self.script_specs
                    }
                }
        
        return preview
    
    def save_to_file(self, plan_file: str) -> Path:
        """
        Save execution plan to JSON file.
        
        Args:
            plan_file: Path to save the execution plan
            
        Returns:
            Path to saved plan file
        """
        plan_path = Path(plan_file)
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        plan_data = {
            "dag": self.dag.model_dump() if hasattr(self.dag, 'model_dump') else str(self.dag),
            "script_specs": {
                name: spec.model_dump() for name, spec in self.script_specs.items()
            },
            "execution_order": self.execution_order,
            "test_workspace_dir": self.test_workspace_dir,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
        
        with open(plan_path, 'w') as f:
            json.dump(plan_data, f, indent=2, default=str)
            
        return plan_path
    
    @classmethod
    def load_from_file(cls, plan_file: str) -> "ScriptExecutionPlan":
        """
        Load execution plan from JSON file.
        
        Args:
            plan_file: Path to the execution plan file
            
        Returns:
            Loaded ScriptExecutionPlan
            
        Raises:
            FileNotFoundError: If plan file doesn't exist
        """
        plan_path = Path(plan_file)
        if not plan_path.exists():
            raise FileNotFoundError(f"Execution plan file not found: {plan_file}")
            
        with open(plan_path, 'r') as f:
            plan_data = json.load(f)
        
        # Reconstruct objects
        dag = PipelineDAG.model_validate(plan_data["dag"]) if isinstance(plan_data["dag"], dict) else plan_data["dag"]
        script_specs = {
            name: ScriptExecutionSpec.model_validate(spec_data)
            for name, spec_data in plan_data["script_specs"].items()
        }
        
        return cls(
            dag=dag,
            script_specs=script_specs,
            execution_order=plan_data["execution_order"],
            test_workspace_dir=plan_data["test_workspace_dir"],
            created_at=datetime.fromisoformat(plan_data["created_at"]),
            metadata=plan_data.get("metadata"),
        )
    
    def _validate_dag_acyclic(self) -> bool:
        """
        Validate that the DAG is acyclic.
        
        Returns:
            True if DAG is acyclic, False otherwise
        """
        try:
            self.dag.topological_sort()
            return True
        except ValueError:
            return False
    
    def get_script_dependencies(self, node_name: str) -> List[str]:
        """
        Get dependencies for a specific script node.
        
        Args:
            node_name: Name of the node to get dependencies for
            
        Returns:
            List of dependency node names
        """
        return self.dag.get_dependencies(node_name)
    
    def get_script_dependents(self, node_name: str) -> List[str]:
        """
        Get dependents (nodes that depend on this node) for a specific script.
        
        Args:
            node_name: Name of the node to get dependents for
            
        Returns:
            List of dependent node names
        """
        dependents = []
        for other_node in self.dag.nodes:
            if node_name in self.dag.get_dependencies(other_node):
                dependents.append(other_node)
        return dependents
    
    def __str__(self) -> str:
        """String representation of the execution plan."""
        return f"ScriptExecutionPlan(scripts={len(self.script_specs)}, order={len(self.execution_order)})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ScriptExecutionPlan("
            f"scripts={len(self.script_specs)}, "
            f"execution_order={self.execution_order}, "
            f"workspace='{self.test_workspace_dir}', "
            f"created_at='{self.created_at.isoformat()}')"
        )
