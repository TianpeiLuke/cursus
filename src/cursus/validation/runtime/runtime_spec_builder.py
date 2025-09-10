"""
Pipeline Testing Specification Builder

Builder to generate PipelineTestingSpec from DAG with intelligent node-to-script resolution,
workspace-first file discovery, and comprehensive dual identity management.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from difflib import SequenceMatcher

from ...api.dag.base_dag import PipelineDAG
from .runtime_models import ScriptExecutionSpec, PipelineTestingSpec
from .contract_discovery import ContractDiscoveryManager

try:
    from ...registry.step_names import get_step_name_from_spec_type
except ImportError:
    # Fallback for testing or when registry is not available
    def get_step_name_from_spec_type(node_name: str) -> str:
        """Fallback implementation that removes job type suffixes."""
        suffixes = ['_training', '_evaluation', '_calibration', '_inference', '_registration']
        for suffix in suffixes:
            if node_name.endswith(suffix):
                return node_name[:-len(suffix)]
        return node_name


class PipelineTestingSpecBuilder:
    """
    Builder to generate PipelineTestingSpec from DAG with intelligent node-to-script resolution.
    
    Handles the core challenge of mapping DAG node names to script files through:
    1. Registry-based canonical name resolution
    2. PascalCase to snake_case conversion with special cases
    3. Workspace-first file discovery with fuzzy matching fallback
    4. ScriptExecutionSpec creation with dual identity management
    """
    
    def __init__(self, test_data_dir: str = "test/integration/runtime"):
        self.test_data_dir = Path(test_data_dir)
        self.specs_dir = self.test_data_dir / ".specs"      # ScriptExecutionSpec storage
        self.scripts_dir = self.test_data_dir / "scripts"   # Test script files
        
        # Initialize contract discovery manager
        self.contract_manager = ContractDiscoveryManager(str(self.test_data_dir))
        
        # Ensure directories exist
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Create other standard directories
        (self.test_data_dir / "input").mkdir(parents=True, exist_ok=True)
        (self.test_data_dir / "output").mkdir(parents=True, exist_ok=True)
        (self.test_data_dir / "results").mkdir(parents=True, exist_ok=True)
    
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
    
    # New intelligent node-to-script resolution methods
    
    def resolve_script_execution_spec_from_node(self, node_name: str) -> ScriptExecutionSpec:
        """
        Resolve ScriptExecutionSpec from PipelineDAG node name using intelligent resolution.
        
        Multi-step resolution process:
        1. Registry-based canonical name extraction
        2. PascalCase to snake_case conversion with special cases
        3. Workspace-first file discovery with fuzzy matching
        4. ScriptExecutionSpec creation with dual identity
        
        Args:
            node_name: DAG node name (e.g., "TabularPreprocessing_training")
            
        Returns:
            ScriptExecutionSpec with proper script_name and step_name mapping
            
        Raises:
            ValueError: If node cannot be resolved to a valid script
        """
        # Step 1: Get canonical step name using existing registry function
        try:
            canonical_name = get_step_name_from_spec_type(node_name)
        except Exception as e:
            raise ValueError(f"Registry resolution failed for '{node_name}': {str(e)}")
        
        # Step 2: Convert to script name with special case handling
        script_name = self._canonical_to_script_name(canonical_name)
        
        # Step 3: Find actual script file with verification
        try:
            script_path = self._find_script_file(script_name)
        except FileNotFoundError as e:
            raise ValueError(f"Script file not found for '{node_name}' -> '{script_name}': {str(e)}")
        
        # Step 4: Create ScriptExecutionSpec with dual identity
        # Try to load existing spec first, then create new one
        try:
            existing_spec = ScriptExecutionSpec.load_from_file(script_name, str(self.specs_dir))
            # Update step_name for current context
            existing_spec.step_name = node_name
            return existing_spec
        except FileNotFoundError:
            # Create new spec with intelligent defaults
            spec = ScriptExecutionSpec.create_default(
                script_name=script_name,      # For file discovery (snake_case)
                step_name=node_name,          # For DAG node matching (PascalCase + job type)
                test_workspace_root=str(self.test_data_dir)
            )
            
            # Update with intelligent script path and contract-aware defaults
            spec_dict = spec.model_dump()
            spec_dict['script_path'] = str(script_path)
            spec_dict['input_paths'] = self._get_contract_aware_input_paths(script_name, canonical_name)
            spec_dict['output_paths'] = self._get_contract_aware_output_paths(script_name, canonical_name)
            spec_dict['environ_vars'] = self._get_contract_aware_environ_vars(script_name, canonical_name)
            spec_dict['job_args'] = self._get_contract_aware_job_args(script_name, canonical_name)
            
            enhanced_spec = ScriptExecutionSpec(**spec_dict)
            
            # Save for future use
            self.save_script_spec(enhanced_spec)
            
            return enhanced_spec
    
    def _canonical_to_script_name(self, canonical_name: str) -> str:
        """
        Convert canonical step name (PascalCase) to script name (snake_case).
        
        Handles special cases for compound technical terms:
        - XGBoost -> xgboost (not x_g_boost)
        - PyTorch -> pytorch (not py_torch)
        - ModelEval -> model_eval
        
        Args:
            canonical_name: PascalCase canonical name
            
        Returns:
            snake_case script name
        """
        # Handle special cases for compound technical terms
        special_cases = {
            'XGBoost': 'Xgboost',
            'PyTorch': 'Pytorch',
            'MLFlow': 'Mlflow',
            'TensorFlow': 'Tensorflow',
            'SageMaker': 'Sagemaker',
            'AutoML': 'Automl'
        }
        
        # Apply special case replacements
        processed_name = canonical_name
        for original, replacement in special_cases.items():
            processed_name = processed_name.replace(original, replacement)
        
        # Convert PascalCase to snake_case
        # Handle sequences of capitals followed by lowercase
        result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', processed_name)
        # Handle lowercase followed by uppercase
        result = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', result)
        
        return result.lower()
    
    def _find_script_file(self, script_name: str) -> Path:
        """
        Find actual script file with workspace-first lookup and fuzzy matching.
        
        Priority order:
        1. Test workspace scripts (self.scripts_dir) - for testing environment
        2. Core framework scripts (workspace discovery) - fallback
        3. Fuzzy matching for similar names - error recovery
        4. Create placeholder script - last resort
        
        Args:
            script_name: snake_case script name
            
        Returns:
            Path to script file
            
        Raises:
            FileNotFoundError: If no suitable script can be found or created
        """
        # Priority 1: Test workspace scripts
        test_script_path = self.scripts_dir / f"{script_name}.py"
        if test_script_path.exists():
            return test_script_path
        
        # Priority 2: Core framework scripts (workspace discovery)
        workspace_script = self._find_in_workspace(script_name)
        if workspace_script:
            return workspace_script
        
        # Priority 3: Fuzzy matching fallback
        fuzzy_match = self._find_fuzzy_match(script_name)
        if fuzzy_match:
            return fuzzy_match
        
        # Priority 4: Create placeholder script
        return self._create_placeholder_script(script_name)
    
    def _find_in_workspace(self, script_name: str) -> Optional[Path]:
        """
        Find script in core framework workspace.
        
        Searches common locations for cursus step scripts.
        
        Args:
            script_name: snake_case script name
            
        Returns:
            Path to script if found, None otherwise
        """
        # Common locations for cursus step scripts
        search_paths = [
            Path("src/cursus/steps/scripts"),
            Path("cursus/steps/scripts"),
            Path("steps/scripts"),
            Path("scripts")
        ]
        
        script_filename = f"{script_name}.py"
        
        for search_path in search_paths:
            if search_path.exists():
                script_path = search_path / script_filename
                if script_path.exists():
                    return script_path.resolve()
        
        return None
    
    def _find_fuzzy_match(self, script_name: str) -> Optional[Path]:
        """
        Find script using fuzzy matching for error recovery.
        
        Looks for similar script names in the test workspace.
        
        Args:
            script_name: snake_case script name
            
        Returns:
            Path to best matching script if found, None otherwise
        """
        if not self.scripts_dir.exists():
            return None
        
        best_match = None
        best_ratio = 0.0
        threshold = 0.7  # Minimum similarity threshold
        
        for script_file in self.scripts_dir.glob("*.py"):
            file_stem = script_file.stem
            ratio = SequenceMatcher(None, script_name, file_stem).ratio()
            
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = script_file
        
        if best_match:
            print(f"Fuzzy match: '{script_name}' -> '{best_match.name}' (similarity: {best_ratio:.2f})")
        
        return best_match
    
    def _create_placeholder_script(self, script_name: str) -> Path:
        """
        Create placeholder script for missing scripts.
        
        Creates a basic Python script template that can be used for testing.
        
        Args:
            script_name: snake_case script name
            
        Returns:
            Path to created placeholder script
            
        Raises:
            FileNotFoundError: If placeholder cannot be created
        """
        placeholder_path = self.scripts_dir / f"{script_name}.py"
        
        try:
            placeholder_content = f'''"""
Placeholder script for {script_name}.

This script was automatically generated by PipelineTestingSpecBuilder
because no existing script was found for this step.

TODO: Implement the actual script logic.
"""

import sys
import json
from pathlib import Path


def main():
    """Main entry point for {script_name} script."""
    print(f"Running placeholder script: {script_name}")
    
    # Basic argument parsing
    if len(sys.argv) > 1:
        print(f"Arguments: {{sys.argv[1:]}}")
    
    # Create minimal output for testing
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Create a simple output file
    output_file = output_dir / f"{script_name}_output.json"
    with open(output_file, 'w') as f:
        json.dump({{
            "script": "{script_name}",
            "status": "placeholder_executed",
            "message": "This is a placeholder script output"
        }}, f, indent=2)
    
    print(f"Created placeholder output: {{output_file}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
            
            with open(placeholder_path, 'w') as f:
                f.write(placeholder_content)
            
            print(f"Created placeholder script: {placeholder_path}")
            return placeholder_path
            
        except OSError as e:
            raise FileNotFoundError(f"Cannot create placeholder script '{placeholder_path}': {str(e)}")
    
    # Contract-aware methods that replace generic defaults
    
    def _get_contract_aware_input_paths(self, script_name: str, canonical_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get input paths using contract discovery with fallback to generic defaults.
        
        Args:
            script_name: snake_case script name
            canonical_name: PascalCase canonical name (optional)
            
        Returns:
            Dictionary of logical_name -> local_path mappings
        """
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(script_name, canonical_name)
        
        if contract_result.contract is not None:
            print(f"Using contract '{contract_result.contract_name}' for input paths (method: {contract_result.discovery_method})")
            contract_paths = self.contract_manager.get_contract_input_paths(contract_result.contract, script_name)
            if contract_paths:
                return contract_paths
        
        # Fallback to generic defaults
        print(f"No contract found for '{script_name}', using generic input paths")
        return self._get_default_input_paths(script_name)
    
    def _get_contract_aware_output_paths(self, script_name: str, canonical_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get output paths using contract discovery with fallback to generic defaults.
        
        Args:
            script_name: snake_case script name
            canonical_name: PascalCase canonical name (optional)
            
        Returns:
            Dictionary of logical_name -> local_path mappings
        """
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(script_name, canonical_name)
        
        if contract_result.contract is not None:
            print(f"Using contract '{contract_result.contract_name}' for output paths (method: {contract_result.discovery_method})")
            contract_paths = self.contract_manager.get_contract_output_paths(contract_result.contract, script_name)
            if contract_paths:
                return contract_paths
        
        # Fallback to generic defaults
        print(f"No contract found for '{script_name}', using generic output paths")
        return self._get_default_output_paths(script_name)
    
    def _get_contract_aware_environ_vars(self, script_name: str, canonical_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get environment variables using contract discovery with fallback to generic defaults.
        
        Args:
            script_name: snake_case script name
            canonical_name: PascalCase canonical name (optional)
            
        Returns:
            Dictionary of environment variable mappings
        """
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(script_name, canonical_name)
        
        if contract_result.contract is not None:
            print(f"Using contract '{contract_result.contract_name}' for environment variables (method: {contract_result.discovery_method})")
            contract_env_vars = self.contract_manager.get_contract_environ_vars(contract_result.contract)
            if contract_env_vars:
                return contract_env_vars
        
        # Fallback to generic defaults
        print(f"No contract found for '{script_name}', using generic environment variables")
        return self._get_default_environ_vars()
    
    def _get_contract_aware_job_args(self, script_name: str, canonical_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get job arguments using contract discovery with fallback to generic defaults.
        
        Args:
            script_name: snake_case script name
            canonical_name: PascalCase canonical name (optional)
            
        Returns:
            Dictionary of job argument mappings
        """
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(script_name, canonical_name)
        
        if contract_result.contract is not None:
            print(f"Using contract '{contract_result.contract_name}' for job arguments (method: {contract_result.discovery_method})")
            contract_job_args = self.contract_manager.get_contract_job_args(contract_result.contract, script_name)
            if contract_job_args:
                return contract_job_args
        
        # Fallback to generic defaults
        print(f"No contract found for '{script_name}', using generic job arguments")
        return self._get_default_job_args(script_name)
    
    # Generic fallback methods (kept for backward compatibility and fallback)
    
    def _get_default_input_paths(self, script_name: str) -> Dict[str, str]:
        """Get default input paths for a script (fallback method)."""
        return {
            "data_input": str(self.test_data_dir / "input" / "raw_data"),
            "config": str(self.test_data_dir / "input" / "config" / f"{script_name}_config.json")
        }
    
    def _get_default_output_paths(self, script_name: str) -> Dict[str, str]:
        """Get default output paths for a script (fallback method)."""
        return {
            "data_output": str(self.test_data_dir / "output" / f"{script_name}_output"),
            "metrics": str(self.test_data_dir / "output" / f"{script_name}_metrics")
        }
    
    def _get_default_environ_vars(self) -> Dict[str, str]:
        """Get default environment variables (fallback method)."""
        return {
            "PYTHONPATH": str(Path("src").resolve()),
            "CURSUS_ENV": "testing"
        }
    
    def _get_default_job_args(self, script_name: str) -> Dict[str, Any]:
        """Get default job arguments for a script (fallback method)."""
        return {
            "script_name": script_name,
            "execution_mode": "testing",
            "log_level": "INFO"
        }
