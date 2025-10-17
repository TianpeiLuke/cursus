"""
Script Execution Specification

Defines the specification for script execution in DAG-guided testing.
This mirrors the configuration classes in cursus/core but targets script execution.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from pathlib import Path
from datetime import datetime
import argparse

# Direct imports from existing cursus components - MAXIMUM REUSE
from ...registry.step_names import get_step_name_from_spec_type, get_spec_step_type
from ...step_catalog import StepCatalog


class ScriptExecutionSpec(BaseModel):
    """
    Comprehensive specification for script execution in DAG-guided testing.
    
    Mirrors the configuration classes in cursus/core but targets script execution.
    Uses maximum component reuse from existing cursus infrastructure.
    
    Attributes:
        script_name: Script file name (snake_case)
        step_name: DAG node name (PascalCase with job type)
        script_path: Full path to script file
        input_paths: Logical name to input path mapping
        output_paths: Logical name to output path mapping
        environ_vars: Environment variables for script execution
        job_args: Job arguments for script execution
        last_updated: Last update timestamp
        user_notes: User notes about this specification
    """
    
    # Core Identity Fields
    script_name: str = Field(..., description="Script file name (snake_case)")
    step_name: str = Field(..., description="DAG node name (PascalCase with job type)")
    script_path: str = Field(..., description="Full path to script file")
    
    # Path Specifications with logical name mapping
    input_paths: Dict[str, str] = Field(default_factory=dict, description="Logical name to input path mapping")
    output_paths: Dict[str, str] = Field(default_factory=dict, description="Logical name to output path mapping")
    
    # Execution Context
    environ_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    job_args: Dict[str, Any] = Field(default_factory=dict, description="Job arguments for script")
    
    # Metadata
    last_updated: Optional[datetime] = Field(default_factory=datetime.now, description="Last update timestamp")
    user_notes: Optional[str] = Field(default=None, description="User notes about this specification")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
        }
        arbitrary_types_allowed = True
    
    @validator('script_path')
    def validate_script_path(cls, v):
        """Validate that script path is a valid path string."""
        if not v:
            raise ValueError("script_path cannot be empty")
        return str(v)
    
    @validator('step_name')
    def validate_step_name(cls, v):
        """Validate step name format using existing registry functions."""
        if not v:
            raise ValueError("step_name cannot be empty")
        
        # DIRECT REUSE: Use existing registry validation
        try:
            # This validates the step name format
            spec_type = get_spec_step_type(v)
            if spec_type:
                return v
        except Exception:
            pass
        
        # If registry validation fails, do basic validation
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Invalid step_name format: {v}")
        
        return v
    
    @validator('script_name')
    def validate_script_name(cls, v):
        """Validate script name format."""
        if not v:
            raise ValueError("script_name cannot be empty")
        
        # Should be snake_case
        if not v.replace('_', '').isalnum():
            raise ValueError(f"Invalid script_name format (should be snake_case): {v}")
        
        return v
    
    @classmethod
    def create_from_step_catalog(
        cls,
        step_name: str,
        step_catalog: StepCatalog,
        test_workspace_dir: str,
        **kwargs: Any,
    ) -> "ScriptExecutionSpec":
        """
        Create ScriptExecutionSpec using step catalog integration.
        
        This method demonstrates maximum component reuse by leveraging
        the step catalog for script discovery and path resolution.
        
        Args:
            step_name: DAG node name
            step_catalog: StepCatalog instance for component discovery
            test_workspace_dir: Test workspace directory
            **kwargs: Additional specification parameters
            
        Returns:
            ScriptExecutionSpec with step catalog integration
        """
        # DIRECT REUSE: Use step catalog for script discovery
        step_info = step_catalog.resolve_pipeline_node(step_name)
        
        if step_info and step_info.file_components.get('script'):
            script_metadata = step_info.file_components['script']
            script_path = str(script_metadata.path)
            script_name = Path(script_path).stem
        else:
            # Fallback to traditional discovery
            script_name = cls._convert_step_name_to_script_name(step_name)
            script_path = str(Path(test_workspace_dir) / "scripts" / f"{script_name}.py")
        
        # DIRECT REUSE: Get contract-aware paths using step catalog
        input_paths, output_paths = cls._get_contract_aware_paths(
            step_name, step_catalog, test_workspace_dir
        )
        
        return cls(
            script_name=script_name,
            step_name=step_name,
            script_path=script_path,
            input_paths=input_paths,
            output_paths=output_paths,
            **kwargs
        )
    
    @classmethod
    def create_from_node_name(
        cls,
        step_name: str,
        test_workspace_dir: str,
        **kwargs: Any,
    ) -> "ScriptExecutionSpec":
        """
        Create ScriptExecutionSpec from node name with fallback discovery.
        
        Args:
            step_name: DAG node name
            test_workspace_dir: Test workspace directory
            **kwargs: Additional specification parameters
            
        Returns:
            ScriptExecutionSpec with basic configuration
        """
        script_name = cls._convert_step_name_to_script_name(step_name)
        script_path = str(Path(test_workspace_dir) / "scripts" / f"{script_name}.py")
        
        # Default path configuration
        input_paths = {
            "input_data": str(Path(test_workspace_dir) / "input" / f"{script_name}_input.json"),
        }
        output_paths = {
            "output_data": str(Path(test_workspace_dir) / "output" / f"{script_name}_output.json"),
        }
        
        return cls(
            script_name=script_name,
            step_name=step_name,
            script_path=script_path,
            input_paths=input_paths,
            output_paths=output_paths,
            **kwargs
        )
    
    @staticmethod
    def _convert_step_name_to_script_name(step_name: str) -> str:
        """
        Convert step name to script name using existing registry patterns.
        
        DIRECT REUSE: Uses existing step name conversion logic.
        """
        try:
            # DIRECT REUSE: Use existing registry function
            canonical_name = get_step_name_from_spec_type(step_name)
            if canonical_name:
                return canonical_name
        except Exception:
            pass
        
        # Fallback conversion: PascalCase_jobtype -> pascal_case_jobtype
        if '_' in step_name:
            parts = step_name.split('_')
            base_name = parts[0]
            job_type = '_'.join(parts[1:]) if len(parts) > 1 else ''
            
            # Convert PascalCase to snake_case
            snake_case = ''.join(['_' + c.lower() if c.isupper() and i > 0 else c.lower() 
                                 for i, c in enumerate(base_name)])
            
            return f"{snake_case}_{job_type}" if job_type else snake_case
        else:
            # Simple PascalCase to snake_case
            return ''.join(['_' + c.lower() if c.isupper() and i > 0 else c.lower() 
                           for i, c in enumerate(step_name)])
    
    @staticmethod
    def _get_contract_aware_paths(
        step_name: str, 
        step_catalog: StepCatalog, 
        test_workspace_dir: str
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Get contract-aware input and output paths using step catalog.
        
        DIRECT REUSE: Uses step catalog contract loading capabilities.
        """
        input_paths = {}
        output_paths = {}
        
        try:
            # DIRECT REUSE: Use step catalog for contract discovery
            contract = step_catalog.load_contract_class(step_name)
            
            if contract:
                # Get contract-defined paths
                if hasattr(contract, 'get_input_paths'):
                    contract_inputs = contract.get_input_paths()
                    if contract_inputs:
                        input_paths = {
                            name: str(Path(test_workspace_dir) / "input" / f"{name}.json")
                            for name in contract_inputs.keys()
                        }
                
                if hasattr(contract, 'get_output_paths'):
                    contract_outputs = contract.get_output_paths()
                    if contract_outputs:
                        output_paths = {
                            name: str(Path(test_workspace_dir) / "output" / f"{name}.json")
                            for name in contract_outputs.keys()
                        }
        except Exception:
            # Silently ignore contract loading errors
            pass
        
        # Fallback to default paths if no contract available
        if not input_paths:
            script_name = ScriptExecutionSpec._convert_step_name_to_script_name(step_name)
            input_paths = {
                "input_data": str(Path(test_workspace_dir) / "input" / f"{script_name}_input.json"),
            }
        
        if not output_paths:
            script_name = ScriptExecutionSpec._convert_step_name_to_script_name(step_name)
            output_paths = {
                "output_data": str(Path(test_workspace_dir) / "output" / f"{script_name}_output.json"),
            }
        
        return input_paths, output_paths
    
    def validate_paths_exist(self, check_inputs: bool = True, check_outputs: bool = False) -> Dict[str, Any]:
        """
        Validate that specified paths exist.
        
        Args:
            check_inputs: Whether to check input paths exist
            check_outputs: Whether to check output paths exist
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            "all_valid": True,
            "script_exists": Path(self.script_path).exists(),
            "missing_inputs": [],
            "missing_outputs": [],
        }
        
        if not validation_result["script_exists"]:
            validation_result["all_valid"] = False
        
        if check_inputs:
            for name, path in self.input_paths.items():
                if not Path(path).exists():
                    validation_result["missing_inputs"].append(path)
                    validation_result["all_valid"] = False
        
        if check_outputs:
            for name, path in self.output_paths.items():
                if not Path(path).exists():
                    validation_result["missing_outputs"].append(path)
                    validation_result["all_valid"] = False
        
        return validation_result
    
    def get_main_params(self) -> Dict[str, Any]:
        """
        Get parameters for script main() function execution.
        
        Returns:
            Dictionary with parameters for script execution
        """
        return {
            "input_paths": self.input_paths,
            "output_paths": self.output_paths,
            "environ_vars": self.environ_vars,
            "job_args": argparse.Namespace(**self.job_args) if self.job_args else argparse.Namespace(),
        }
    
    def update_paths_from_dependencies(self, dependency_outputs: Dict[str, Dict[str, str]]) -> None:
        """
        Update input paths based on dependency outputs.
        
        This method enables dependency resolution by connecting outputs
        from dependency scripts to inputs of this script.
        
        Args:
            dependency_outputs: Dictionary mapping dependency names to their output paths
        """
        # Simple semantic matching for path updates
        for dep_name, dep_outputs in dependency_outputs.items():
            for output_name, output_path in dep_outputs.items():
                # Try to match output to input by name similarity
                for input_name in self.input_paths.keys():
                    if self._paths_semantically_match(output_name, input_name):
                        self.input_paths[input_name] = output_path
                        break
    
    def _paths_semantically_match(self, output_name: str, input_name: str) -> bool:
        """
        Simple semantic matching for path names.
        
        This could be enhanced to use the SemanticMatcher from cursus/core
        for more sophisticated matching.
        """
        # Simple matching: exact match or contains relationship
        return (
            output_name == input_name or
            output_name in input_name or
            input_name in output_name or
            output_name.replace('_', '') == input_name.replace('_', '')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert specification to dictionary format.
        
        Returns:
            Dictionary representation of the specification
        """
        return self.model_dump()
    
    def save_to_file(self, spec_file: str) -> Path:
        """
        Save specification to JSON file.
        
        Args:
            spec_file: Path to save the specification
            
        Returns:
            Path to saved specification file
        """
        spec_path = Path(spec_file)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(spec_path, 'w') as f:
            import json
            json.dump(self.to_dict(), f, indent=2, default=str)
            
        return spec_path
    
    @classmethod
    def load_from_file(cls, spec_file: str) -> "ScriptExecutionSpec":
        """
        Load specification from JSON file.
        
        Args:
            spec_file: Path to the specification file
            
        Returns:
            Loaded ScriptExecutionSpec
            
        Raises:
            FileNotFoundError: If specification file doesn't exist
        """
        spec_path = Path(spec_file)
        if not spec_path.exists():
            raise FileNotFoundError(f"Specification file not found: {spec_file}")
            
        with open(spec_path, 'r') as f:
            import json
            spec_data = json.load(f)
        
        # Handle datetime deserialization
        if 'last_updated' in spec_data and isinstance(spec_data['last_updated'], str):
            spec_data['last_updated'] = datetime.fromisoformat(spec_data['last_updated'])
        
        return cls.model_validate(spec_data)
    
    def __str__(self) -> str:
        """String representation of the specification."""
        return f"ScriptExecutionSpec(script={self.script_name}, step={self.step_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ScriptExecutionSpec("
            f"script_name='{self.script_name}', "
            f"step_name='{self.step_name}', "
            f"script_path='{self.script_path}', "
            f"inputs={len(self.input_paths)}, "
            f"outputs={len(self.output_paths)})"
        )
