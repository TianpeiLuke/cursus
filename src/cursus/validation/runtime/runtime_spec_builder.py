"""
Pipeline Testing Specification Builder - Streamlined Core Intelligence

Streamlined builder focused on core intelligent node-to-script resolution,
workspace-first file discovery, and contract-aware path resolution.

Redundant interactive methods removed - use InteractiveRuntimeTestingFactory instead.
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from difflib import SequenceMatcher

from ...api.dag.base_dag import PipelineDAG
from .runtime_models import ScriptExecutionSpec, PipelineTestingSpec
from ...step_catalog.adapters.contract_adapter import ContractDiscoveryManagerAdapter as ContractDiscoveryManager

try:
    from ...registry.step_names import get_step_name_from_spec_type
except ImportError:
    # Fallback for testing or when registry is not available
    def get_step_name_from_spec_type(node_name: str) -> str:
        """Fallback implementation that removes job type suffixes."""
        suffixes = [
            "_training",
            "_evaluation",
            "_calibration",
            "_inference",
            "_registration",
        ]
        for suffix in suffixes:
            if node_name.endswith(suffix):
                return node_name[: -len(suffix)]
        return node_name


class PipelineTestingSpecBuilder:
    """
    Streamlined builder focused on core intelligent node-to-script resolution.

    Core Intelligence Methods (used by InteractiveRuntimeTestingFactory):
    1. Registry-based canonical name resolution
    2. PascalCase to snake_case conversion with special cases
    3. Workspace-first file discovery with fuzzy matching fallback
    4. Contract-aware path resolution with intelligent defaults
    5. Step catalog integration for enhanced script discovery

    Interactive methods removed - use InteractiveRuntimeTestingFactory for user interaction.
    """

    def __init__(self, test_data_dir: str = "test/integration/runtime", step_catalog: Optional['StepCatalog'] = None):
        self.test_data_dir = Path(test_data_dir)
        self.specs_dir = self.test_data_dir / "specs"  # ScriptExecutionSpec storage
        self.scripts_dir = self.test_data_dir / "scripts"  # Test script files

        # Initialize contract discovery manager
        self.contract_manager = ContractDiscoveryManager(str(self.test_data_dir))

        # Ensure directories exist
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)

        # Create other standard directories
        (self.test_data_dir / "input").mkdir(parents=True, exist_ok=True)
        (self.test_data_dir / "output").mkdir(parents=True, exist_ok=True)
        (self.test_data_dir / "results").mkdir(parents=True, exist_ok=True)

        # Step Catalog Integration
        self.step_catalog = step_catalog or self._initialize_step_catalog()

    # === CORE INTELLIGENCE METHODS (Used by InteractiveRuntimeTestingFactory) ===

    def resolve_script_execution_spec_from_node(
        self, node_name: str
    ) -> ScriptExecutionSpec:
        """
        Core intelligent script resolution from PipelineDAG node name.

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
            raise ValueError(
                f"Script file not found for '{node_name}' -> '{script_name}': {str(e)}"
            )

        # Step 4: Create ScriptExecutionSpec with dual identity
        # Try to load existing spec first, then create new one
        try:
            existing_spec = ScriptExecutionSpec.load_from_file(
                script_name, str(self.specs_dir)
            )
            # Update step_name for current context
            existing_spec.step_name = node_name
            return existing_spec
        except FileNotFoundError:
            # Create new spec with intelligent defaults
            spec = ScriptExecutionSpec.create_default(
                script_name=script_name,  # For file discovery (snake_case)
                step_name=node_name,  # For DAG node matching (PascalCase + job type)
                test_data_dir=str(self.test_data_dir),
            )

            # Update with intelligent script path and contract-aware defaults
            spec_dict = spec.model_dump()
            spec_dict["script_path"] = str(script_path)
            spec_dict["input_paths"] = self._get_contract_aware_input_paths(
                script_name, canonical_name
            )
            spec_dict["output_paths"] = self._get_contract_aware_output_paths(
                script_name, canonical_name
            )
            spec_dict["environ_vars"] = self._get_contract_aware_environ_vars(
                script_name, canonical_name
            )
            spec_dict["job_args"] = self._get_contract_aware_job_args(
                script_name, canonical_name
            )

            enhanced_spec = ScriptExecutionSpec(**spec_dict)
            return enhanced_spec

    def _canonical_to_script_name(self, canonical_name: str) -> str:
        """
        Core name conversion logic - convert canonical step name (PascalCase) to script name (snake_case).

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
            "XGBoost": "Xgboost",
            "PyTorch": "Pytorch",
            "MLFlow": "Mlflow",
            "TensorFlow": "Tensorflow",
            "SageMaker": "Sagemaker",
            "AutoML": "Automl",
        }

        # Apply special case replacements
        processed_name = canonical_name
        for original, replacement in special_cases.items():
            processed_name = processed_name.replace(original, replacement)

        # Convert PascalCase to snake_case
        # Handle sequences of capitals followed by lowercase
        result = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", processed_name)
        # Handle lowercase followed by uppercase
        result = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", result)

        return result.lower()

    def _find_script_file(self, script_name: str) -> Path:
        """
        Core script discovery logic - find actual script file using step catalog with fallback.

        Priority order:
        1. Step catalog script discovery - unified discovery system
        2. Test workspace scripts (self.scripts_dir) - for testing environment
        3. Core framework scripts (workspace discovery) - fallback
        4. Fuzzy matching for similar names - error recovery
        5. Create placeholder script - last resort

        Args:
            script_name: snake_case script name

        Returns:
            Path to script file

        Raises:
            FileNotFoundError: If no suitable script can be found or created
        """
        # Priority 1: Step catalog script discovery
        try:
            from ...step_catalog import StepCatalog
            
            # PORTABLE: Package-only discovery (works in all deployment scenarios)
            catalog = StepCatalog(workspace_dirs=None)
            
            # Try to find step by script name
            available_steps = catalog.list_available_steps()
            for step_name in available_steps:
                step_info = catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('script'):
                    script_metadata = step_info.file_components['script']
                    if script_metadata and script_metadata.path:
                        # Check if this script matches our expected name
                        if script_name in str(script_metadata.path) or script_metadata.path.stem == script_name:
                            return script_metadata.path
                            
        except ImportError:
            pass  # Fall back to legacy discovery
        except Exception:
            pass  # Fall back to legacy discovery

        # Priority 2: Test workspace scripts
        test_script_path = self.scripts_dir / f"{script_name}.py"
        if test_script_path.exists():
            return test_script_path

        # Priority 3: Core framework scripts (workspace discovery)
        workspace_script = self._find_in_workspace(script_name)
        if workspace_script:
            return workspace_script

        # Priority 4: Fuzzy matching fallback
        fuzzy_match = self._find_fuzzy_match(script_name)
        if fuzzy_match:
            return fuzzy_match

        # Priority 5: Create placeholder script
        return self._create_placeholder_script(script_name)

    def get_script_main_params(self, spec: ScriptExecutionSpec) -> Dict[str, Any]:
        """
        Core parameter extraction - get parameters ready for script main() function call.

        Returns:
            Dictionary with input_paths, output_paths, environ_vars, job_args ready for main()
        """
        return {
            "input_paths": spec.input_paths,
            "output_paths": spec.output_paths,
            "environ_vars": spec.environ_vars,
            "job_args": (
                argparse.Namespace(**spec.job_args)
                if spec.job_args
                else argparse.Namespace(job_type="testing")
            ),
        }

    def _get_contract_aware_input_paths(
        self, script_name: str, canonical_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Contract-aware path resolution - get input paths using contract discovery with fallback.

        Args:
            script_name: snake_case script name
            canonical_name: PascalCase canonical name (optional)

        Returns:
            Dictionary of logical_name -> local_path mappings
        """
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(
            script_name, canonical_name
        )

        if (contract_result is not None and 
            hasattr(contract_result, 'contract') and 
            contract_result.contract is not None):
            contract_paths = self.contract_manager.get_contract_input_paths(
                contract_result.contract, script_name
            )
            if contract_paths:
                return contract_paths

        # Fallback to generic defaults
        return self._get_default_input_paths(script_name)

    def _get_contract_aware_output_paths(
        self, script_name: str, canonical_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Contract-aware path resolution - get output paths using contract discovery with fallback.

        Args:
            script_name: snake_case script name
            canonical_name: PascalCase canonical name (optional)

        Returns:
            Dictionary of logical_name -> local_path mappings
        """
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(
            script_name, canonical_name
        )

        if (contract_result is not None and 
            hasattr(contract_result, 'contract') and 
            contract_result.contract is not None):
            contract_paths = self.contract_manager.get_contract_output_paths(
                contract_result.contract, script_name
            )
            if contract_paths:
                return contract_paths

        # Fallback to generic defaults
        return self._get_default_output_paths(script_name)

    # === STEP CATALOG INTEGRATION METHODS ===

    def _initialize_step_catalog(self):
        """
        Initialize step catalog with unified workspace resolution.
        
        Priority order:
        1. test_data_dir (primary testing workspace)
        2. Additional development workspaces from environment
        3. Package-only discovery (for deployment scenarios)
        """
        try:
            from ...step_catalog import StepCatalog
        except ImportError:
            # Step catalog not available, return None for optional enhancement
            return None
        
        import os
        workspace_dirs = []
        
        # Priority 1: Use test_data_dir as primary workspace
        if self.test_data_dir:
            test_workspace = self.test_data_dir / "scripts"
            if test_workspace.exists():
                workspace_dirs.append(test_workspace)
            else:
                if self.test_data_dir.exists():
                    workspace_dirs.append(self.test_data_dir)
        
        # Priority 2: Add development workspaces from environment
        dev_workspaces = os.environ.get('CURSUS_DEV_WORKSPACES', '').split(':')
        for workspace in dev_workspaces:
            if workspace and Path(workspace).exists():
                workspace_path = Path(workspace)
                if workspace_path not in workspace_dirs:
                    workspace_dirs.append(workspace_path)
        
        # Initialize with unified workspace list or package-only
        try:
            return StepCatalog(workspace_dirs=workspace_dirs if workspace_dirs else None)
        except Exception:
            # Silently ignore errors for optional enhancement
            return None

    def _resolve_script_with_step_catalog_if_available(self, node_name: str) -> Optional[ScriptExecutionSpec]:
        """Enhanced script resolution using step catalog (used by InteractiveRuntimeTestingFactory)."""
        if not self.step_catalog:
            return None
            
        try:
            # Use step catalog's pipeline node resolution
            step_info = self.step_catalog.resolve_pipeline_node(node_name)
            
            if step_info and step_info.file_components.get('script'):
                script_metadata = step_info.file_components['script']
                
                # Get contract-aware paths if available
                paths = self._get_contract_aware_paths_if_available(node_name, str(self.test_data_dir))
                
                spec = ScriptExecutionSpec(
                    script_name=script_metadata.path.stem,
                    step_name=node_name,
                    script_path=str(script_metadata.path),
                    input_paths=paths["input_paths"] if paths["input_paths"] else self._get_default_input_paths(script_metadata.path.stem),
                    output_paths=paths["output_paths"] if paths["output_paths"] else self._get_default_output_paths(script_metadata.path.stem),
                    environ_vars=self._get_default_environ_vars(),
                    job_args=self._get_default_job_args(script_metadata.path.stem)
                )
                
                return spec
        except Exception:
            # Silently ignore errors for optional enhancement
            pass
        
        return None
    
    def _get_contract_aware_paths_if_available(self, step_name: str, test_workspace_root: str) -> Dict[str, Dict[str, str]]:
        """Contract-aware path resolution using step catalog (used by InteractiveRuntimeTestingFactory)."""
        paths = {"input_paths": {}, "output_paths": {}}
        if self.step_catalog:
            try:
                contract = self.step_catalog.load_contract_class(step_name)
                if contract:
                    if hasattr(contract, 'get_input_paths'):
                        contract_inputs = contract.get_input_paths()
                        if contract_inputs:
                            paths["input_paths"] = {
                                name: str(Path(test_workspace_root) / "input" / name)
                                for name in contract_inputs.keys()
                            }
                    if hasattr(contract, 'get_output_paths'):
                        contract_outputs = contract.get_output_paths()
                        if contract_outputs:
                            paths["output_paths"] = {
                                name: str(Path(test_workspace_root) / "output" / name)
                                for name in contract_outputs.keys()
                            }
            except Exception:
                # Silently ignore errors for optional enhancement
                pass
        return paths

    # === LEGACY SUPPORT METHODS (Minimal Implementation) ===

    def build_from_dag(
        self, dag: PipelineDAG, validate: bool = True
    ) -> PipelineTestingSpec:
        """
        Legacy method - build PipelineTestingSpec from DAG.
        
        NOTE: For interactive workflow, use InteractiveRuntimeTestingFactory instead.
        This method provides basic functionality for backward compatibility.
        """
        script_specs = {}
        
        # Simple spec creation for each DAG node
        for node in dag.nodes:
            try:
                spec = self.resolve_script_execution_spec_from_node(node)
                script_specs[node] = spec
            except Exception as e:
                if validate:
                    raise ValueError(f"Failed to resolve spec for node '{node}': {str(e)}")
                # Create minimal fallback spec
                script_specs[node] = ScriptExecutionSpec.create_default(
                    script_name=node.lower(),
                    step_name=node,
                    test_data_dir=str(self.test_data_dir)
                )

        return PipelineTestingSpec(
            dag=dag,
            script_specs=script_specs,
            test_workspace_root=str(self.test_data_dir),
        )

    # === PRIVATE HELPER METHODS ===

    def _get_contract_aware_environ_vars(
        self, script_name: str, canonical_name: Optional[str] = None
    ) -> Dict[str, str]:
        """Get environment variables using contract discovery with fallback."""
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(
            script_name, canonical_name
        )

        if (contract_result is not None and 
            hasattr(contract_result, 'contract') and 
            contract_result.contract is not None):
            contract_env_vars = self.contract_manager.get_contract_environ_vars(
                contract_result.contract
            )
            if contract_env_vars:
                return contract_env_vars

        # Fallback to generic defaults
        return self._get_default_environ_vars()

    def _get_contract_aware_job_args(
        self, script_name: str, canonical_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get job arguments using contract discovery with fallback."""
        # Try to discover and use contract
        contract_result = self.contract_manager.discover_contract(
            script_name, canonical_name
        )

        if (contract_result is not None and 
            hasattr(contract_result, 'contract') and 
            contract_result.contract is not None):
            contract_job_args = self.contract_manager.get_contract_job_args(
                contract_result.contract, script_name
            )
            if contract_job_args:
                return contract_job_args

        # Fallback to generic defaults
        return self._get_default_job_args(script_name)

    def _find_in_workspace(self, script_name: str) -> Optional[Path]:
        """Find script in core framework workspace."""
        # Common locations for cursus step scripts
        search_paths = [
            Path("src/cursus/steps/scripts"),
            Path("cursus/steps/scripts"),
            Path("steps/scripts"),
            Path("scripts"),
        ]

        script_filename = f"{script_name}.py"

        for search_path in search_paths:
            if search_path.exists():
                script_path = search_path / script_filename
                if script_path.exists():
                    return script_path.resolve()

        return None

    def _find_fuzzy_match(self, script_name: str) -> Optional[Path]:
        """Find script using fuzzy matching for error recovery."""
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

        return best_match

    def _create_placeholder_script(self, script_name: str) -> Path:
        """Create placeholder script for missing scripts."""
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

            with open(placeholder_path, "w") as f:
                f.write(placeholder_content)

            return placeholder_path

        except OSError as e:
            raise FileNotFoundError(
                f"Cannot create placeholder script '{placeholder_path}': {str(e)}"
            )

    # Generic fallback methods (kept for backward compatibility and fallback)

    def _get_default_input_paths(self, script_name: str) -> Dict[str, str]:
        """Get default input paths for a script (fallback method)."""
        return {
            "data_input": str(self.test_data_dir / "input" / "raw_data"),
            "config": str(
                self.test_data_dir / "input" / "config" / f"{script_name}_config.json"
            ),
        }

    def _get_default_output_paths(self, script_name: str) -> Dict[str, str]:
        """Get default output paths for a script (fallback method)."""
        return {
            "data_output": str(self.test_data_dir / "output" / f"{script_name}_output"),
            "metrics": str(self.test_data_dir / "output" / f"{script_name}_metrics"),
        }

    def _get_default_environ_vars(self) -> Dict[str, str]:
        """Get default environment variables (fallback method)."""
        return {"PYTHONPATH": str(Path("src").resolve()), "CURSUS_ENV": "testing"}

    def _get_default_job_args(self, script_name: str) -> Dict[str, Any]:
        """Get default job arguments for a script (fallback method)."""
        return {
            "script_name": script_name,
            "execution_mode": "testing",
            "log_level": "INFO",
        }
