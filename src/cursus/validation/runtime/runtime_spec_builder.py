"""
Pipeline Testing Specification Builder

Builder to generate PipelineTestingSpec from DAG with local spec persistence and validation.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

from ...api.dag.base_dag import PipelineDAG
from .runtime_models import ScriptExecutionSpec, PipelineTestingSpec


class PipelineTestingSpecBuilder:
    """Builder to generate PipelineTestingSpec from DAG with local spec persistence and validation"""
    
    def __init__(self, test_data_dir: str = "test/integration/runtime"):
        self.test_data_dir = Path(test_data_dir)
        self.specs_dir = self.test_data_dir / ".specs"  # Hidden directory for saved specs
        self.specs_dir.mkdir(parents=True, exist_ok=True)
    
    def build_from_dag(self, dag: PipelineDAG, validate: bool = True) -> PipelineTestingSpec:
        """
        Build PipelineTestingSpec from a PipelineDAG with automatic saved spec loading and validation
        
        Args:
            dag: Pipeline DAG structure to copy and build specs for
            validate: Whether to validate that all specs are properly filled
            
        Returns:
            Complete PipelineTestingSpec ready for runtime testing
            
        Raises:
            ValueError: If validation fails and required specs are missing or incomplete
        """
        script_specs = {}
        missing_specs = []
        incomplete_specs = []
        
        # Load or create specs for each DAG node
        for node in dag.nodes:
            try:
                spec = self._load_or_create_script_spec(node)
                script_specs[node] = spec
                
                # Check if spec is complete (has required fields filled)
                if validate and not self._is_spec_complete(spec):
                    incomplete_specs.append(node)
                    
            except FileNotFoundError:
                missing_specs.append(node)
        
        # Validate all specs are present and complete
        if validate:
            self._validate_specs_completeness(dag.nodes, missing_specs, incomplete_specs)
        
        return PipelineTestingSpec(
            dag=dag,  # Copy the DAG structure
            script_specs=script_specs,
            test_workspace_root=str(self.test_data_dir)
        )
    
    def _load_or_create_script_spec(self, node_name: str) -> ScriptExecutionSpec:
        """Load saved ScriptExecutionSpec or create default if not found"""
        try:
            # Try to load saved spec using auto-generated filename
            saved_spec = ScriptExecutionSpec.load_from_file(node_name, str(self.specs_dir))
            print(f"Loaded saved spec for {node_name} (last updated: {saved_spec.last_updated})")
            return saved_spec
        except FileNotFoundError:
            # Create default spec if no saved spec found
            print(f"Creating default spec for {node_name}")
            default_spec = ScriptExecutionSpec.create_default(node_name, node_name, str(self.test_data_dir))
            
            # Save the default spec for future use
            self.save_script_spec(default_spec)
            
            return default_spec
        except Exception as e:
            print(f"Warning: Could not load saved spec for {node_name}: {e}")
            # Create default spec if loading failed
            print(f"Creating default spec for {node_name}")
            default_spec = ScriptExecutionSpec.create_default(node_name, node_name, str(self.test_data_dir))
            
            # Save the default spec for future use
            self.save_script_spec(default_spec)
            
            return default_spec
    
    def save_script_spec(self, spec: ScriptExecutionSpec) -> None:
        """Save ScriptExecutionSpec to local file for reuse"""
        saved_path = spec.save_to_file(str(self.specs_dir))
        print(f"Saved spec for {spec.script_name} to {saved_path}")
    
    def update_script_spec(self, node_name: str, **updates) -> ScriptExecutionSpec:
        """Update specific fields in a ScriptExecutionSpec and save it"""
        # Load existing spec or create default
        existing_spec = self._load_or_create_script_spec(node_name)
        
        # Update fields
        spec_dict = existing_spec.model_dump()
        spec_dict.update(updates)
        
        # Create updated spec
        updated_spec = ScriptExecutionSpec(**spec_dict)
        
        # Save updated spec
        self.save_script_spec(updated_spec)
        
        return updated_spec
    
    def list_saved_specs(self) -> List[str]:
        """List all saved ScriptExecutionSpec names based on naming pattern"""
        spec_files = list(self.specs_dir.glob("*_runtime_test_spec.json"))
        # Extract script name from filename pattern: {script_name}_runtime_test_spec.json
        return [f.stem.replace("_runtime_test_spec", "") for f in spec_files]
    
    def get_script_spec_by_name(self, script_name: str) -> Optional[ScriptExecutionSpec]:
        """Get ScriptExecutionSpec by script name (for step name matching)"""
        try:
            return ScriptExecutionSpec.load_from_file(script_name, str(self.specs_dir))
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Error loading spec for {script_name}: {e}")
            return None
    
    def match_step_to_spec(self, step_name: str, available_specs: List[str]) -> Optional[str]:
        """
        Match a pipeline step name to the most appropriate ScriptExecutionSpec
        
        Args:
            step_name: Name of the pipeline step
            available_specs: List of available spec names
            
        Returns:
            Best matching spec name or None if no good match found
        """
        # Direct match
        if step_name in available_specs:
            return step_name
        
        # Try common variations
        variations = [
            step_name.lower(),
            step_name.replace('_', ''),
            step_name.replace('-', '_'),
            step_name.split('_')[0],  # First part of compound names
        ]
        
        for variation in variations:
            if variation in available_specs:
                return variation
        
        # Fuzzy matching - find specs that contain step name parts
        step_parts = step_name.lower().split('_')
        best_match = None
        best_score = 0
        
        for spec_name in available_specs:
            spec_parts = spec_name.lower().split('_')
            common_parts = set(step_parts) & set(spec_parts)
            score = len(common_parts) / max(len(step_parts), len(spec_parts))
            
            if score > best_score and score > 0.5:  # At least 50% match
                best_match = spec_name
                best_score = score
        
        return best_match
    
    def _is_spec_complete(self, spec: ScriptExecutionSpec) -> bool:
        """
        Check if a ScriptExecutionSpec has all required fields properly filled
        
        Args:
            spec: ScriptExecutionSpec to validate
            
        Returns:
            True if spec is complete, False otherwise
        """
        # Check required fields are not empty
        if not spec.script_name or not spec.step_name:
            return False
        
        # Check that essential paths are provided
        if not spec.input_paths or not spec.output_paths:
            return False
        
        # Check that input/output paths are not just empty strings
        if not any(path.strip() for path in spec.input_paths.values()):
            return False
        
        if not any(path.strip() for path in spec.output_paths.values()):
            return False
        
        return True
    
    def _validate_specs_completeness(self, dag_nodes: List[str], missing_specs: List[str], incomplete_specs: List[str]) -> None:
        """
        Validate that all DAG nodes have complete ScriptExecutionSpecs
        
        Args:
            dag_nodes: List of all DAG node names
            missing_specs: List of nodes with missing specs
            incomplete_specs: List of nodes with incomplete specs
            
        Raises:
            ValueError: If validation fails with detailed error message
        """
        if missing_specs or incomplete_specs:
            error_messages = []
            
            if missing_specs:
                error_messages.append(f"Missing ScriptExecutionSpec files for nodes: {', '.join(missing_specs)}")
                error_messages.append("Please create ScriptExecutionSpec for these nodes using:")
                for node in missing_specs:
                    error_messages.append(f"  builder.update_script_spec('{node}', input_paths={{...}}, output_paths={{...}})")
            
            if incomplete_specs:
                error_messages.append(f"Incomplete ScriptExecutionSpec for nodes: {', '.join(incomplete_specs)}")
                error_messages.append("Please update these specs with required fields:")
                for node in incomplete_specs:
                    error_messages.append(f"  builder.update_script_spec('{node}', input_paths={{...}}, output_paths={{...}})")
            
            error_messages.append(f"\nAll {len(dag_nodes)} DAG nodes must have complete ScriptExecutionSpec before testing.")
            error_messages.append("Use builder.update_script_spec(node_name, **fields) to fill in missing information.")
            
            raise ValueError("\n".join(error_messages))
    
    def update_script_spec_interactive(self, node_name: str) -> ScriptExecutionSpec:
        """
        Interactively update a ScriptExecutionSpec by prompting user for missing fields
        
        Args:
            node_name: Name of the DAG node to update
            
        Returns:
            Updated ScriptExecutionSpec
        """
        # Load existing spec or create default
        existing_spec = self._load_or_create_script_spec(node_name)
        
        print(f"\nUpdating ScriptExecutionSpec for node: {node_name}")
        print(f"Current spec: {existing_spec.script_name}")
        
        # Prompt for input paths
        if not existing_spec.input_paths or not any(path.strip() for path in existing_spec.input_paths.values()):
            print("\nInput paths are required. Current:", existing_spec.input_paths)
            input_path = input(f"Enter input path for {node_name} (e.g., 'test/data/{node_name}/input'): ").strip()
            if input_path:
                existing_spec.input_paths = {"data_input": input_path}
        
        # Prompt for output paths
        if not existing_spec.output_paths or not any(path.strip() for path in existing_spec.output_paths.values()):
            print("\nOutput paths are required. Current:", existing_spec.output_paths)
            output_path = input(f"Enter output path for {node_name} (e.g., 'test/data/{node_name}/output'): ").strip()
            if output_path:
                existing_spec.output_paths = {"data_output": output_path}
        
        # Prompt for environment variables (optional)
        if not existing_spec.environ_vars:
            env_vars = input(f"Enter environment variables for {node_name} (JSON format, or press Enter for defaults): ").strip()
            if env_vars:
                try:
                    existing_spec.environ_vars = json.loads(env_vars)
                except json.JSONDecodeError:
                    print("Invalid JSON format, using defaults")
                    existing_spec.environ_vars = {"LABEL_FIELD": "label"}
            else:
                existing_spec.environ_vars = {"LABEL_FIELD": "label"}
        
        # Prompt for job arguments (optional)
        if not existing_spec.job_args:
            job_args = input(f"Enter job arguments for {node_name} (JSON format, or press Enter for defaults): ").strip()
            if job_args:
                try:
                    existing_spec.job_args = json.loads(job_args)
                except json.JSONDecodeError:
                    print("Invalid JSON format, using defaults")
                    existing_spec.job_args = {"job_type": "testing"}
            else:
                existing_spec.job_args = {"job_type": "testing"}
        
        # Save updated spec
        self.save_script_spec(existing_spec)
        print(f"Updated and saved ScriptExecutionSpec for {node_name}")
        
        return existing_spec
    
    def get_script_main_params(self, spec: ScriptExecutionSpec) -> Dict[str, Any]:
        """
        Get parameters ready for script main() function call
        
        Returns:
            Dictionary with input_paths, output_paths, environ_vars, job_args ready for main()
        """
        return {
            "input_paths": spec.input_paths,
            "output_paths": spec.output_paths,
            "environ_vars": spec.environ_vars,
            "job_args": argparse.Namespace(**spec.job_args) if spec.job_args else argparse.Namespace(job_type="testing")
        }
