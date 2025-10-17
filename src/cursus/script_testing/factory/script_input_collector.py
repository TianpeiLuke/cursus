"""
Script Input Collector

Collects script execution inputs with step catalog integration.
Uses maximum component reuse from existing cursus infrastructure.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

# Direct imports from existing cursus components - MAXIMUM REUSE
from ...step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class ScriptInputCollector:
    """
    Collects script execution inputs with step catalog integration.
    
    Uses maximum component reuse from existing cursus infrastructure.
    This class mirrors the input collection patterns in cursus/api/factory
    but targets script execution parameters instead of step builder configs.
    
    Key features:
    1. Contract-aware input suggestions using step catalog
    2. Dependency-aware path resolution
    3. Interactive collection with validation
    4. Context-aware suggestions based on previous inputs
    
    Attributes:
        step_catalog: StepCatalog for contract-aware input suggestions
    """
    
    def __init__(self, step_catalog: StepCatalog):
        """
        Initialize the Script Input Collector.
        
        Args:
            step_catalog: StepCatalog for contract-aware input suggestions
        """
        self.step_catalog = step_catalog  # DIRECT REUSE
        logger.info("Initialized ScriptInputCollector with step catalog integration")
    
    def collect_node_inputs(
        self, 
        node_name: str, 
        dependencies: List[str], 
        previous_inputs: Dict[str, Any],
        test_workspace_dir: str
    ) -> Dict[str, Any]:
        """
        Collect inputs for a single node with context awareness.
        
        Args:
            node_name: Name of the DAG node
            dependencies: List of dependency node names
            previous_inputs: Previously collected inputs for context
            test_workspace_dir: Test workspace directory
            
        Returns:
            Dictionary with collected node inputs
        """
        logger.debug(f"Collecting inputs for node: {node_name}")
        
        # Get contract-aware suggestions (DIRECT REUSE)
        suggested_inputs = self._get_contract_suggestions(node_name)
        
        # Collect each type of input
        node_inputs = {
            "environ_vars": self._collect_environ_vars(node_name, suggested_inputs),
            "job_args": self._collect_job_args(node_name, suggested_inputs),
            "input_paths": self._collect_input_paths(node_name, dependencies, previous_inputs, suggested_inputs, test_workspace_dir),
            "output_paths": self._collect_output_paths(node_name, suggested_inputs, test_workspace_dir),
        }
        
        return node_inputs
    
    def _get_contract_suggestions(self, node_name: str) -> Dict[str, Any]:
        """
        Get contract-aware input suggestions using step catalog.
        
        Args:
            node_name: Name of the DAG node
            
        Returns:
            Dictionary with suggested inputs from contract
        """
        suggestions = {
            "input_paths": {},
            "output_paths": {},
            "environ_vars": {},
            "job_args": {},
        }
        
        try:
            # DIRECT REUSE: Use step catalog for contract-aware suggestions
            contract = self.step_catalog.load_contract_class(node_name)
            
            if contract and hasattr(contract, 'get_input_paths'):
                suggestions["input_paths"] = contract.get_input_paths()
                logger.debug(f"Found contract input suggestions for {node_name}: {list(suggestions['input_paths'].keys())}")
            
            if contract and hasattr(contract, 'get_output_paths'):
                suggestions["output_paths"] = contract.get_output_paths()
                logger.debug(f"Found contract output suggestions for {node_name}: {list(suggestions['output_paths'].keys())}")
            
            if contract and hasattr(contract, 'get_environ_vars'):
                suggestions["environ_vars"] = contract.get_environ_vars()
                logger.debug(f"Found contract environ var suggestions for {node_name}: {list(suggestions['environ_vars'].keys())}")
            
            if contract and hasattr(contract, 'get_job_args'):
                suggestions["job_args"] = contract.get_job_args()
                logger.debug(f"Found contract job arg suggestions for {node_name}: {list(suggestions['job_args'].keys())}")
                
        except Exception as e:
            logger.debug(f"No contract suggestions available for {node_name}: {e}")
        
        return suggestions
    
    def _collect_environ_vars(self, node_name: str, suggestions: Dict[str, Any]) -> Dict[str, str]:
        """
        Collect environment variables for the script.
        
        Args:
            node_name: Name of the DAG node
            suggestions: Contract suggestions
            
        Returns:
            Dictionary of environment variables
        """
        environ_vars = {}
        suggested_vars = suggestions.get("environ_vars", {})
        
        print(f"   🌍 Environment Variables for {node_name}:")
        
        if suggested_vars:
            print(f"      💡 Contract suggestions: {list(suggested_vars.keys())}")
            
            for var_name, default_value in suggested_vars.items():
                prompt = f"      {var_name}"
                if default_value:
                    prompt += f" (default: {default_value})"
                prompt += ": "
                
                user_input = input(prompt).strip()
                if user_input:
                    environ_vars[var_name] = user_input
                elif default_value:
                    environ_vars[var_name] = str(default_value)
        
        # Allow additional environment variables
        print("      ➕ Additional environment variables (name=value, empty to finish):")
        while True:
            user_input = input("      ").strip()
            if not user_input:
                break
            
            if '=' in user_input:
                var_name, var_value = user_input.split('=', 1)
                environ_vars[var_name.strip()] = var_value.strip()
            else:
                print("         ⚠️  Format: name=value")
        
        if environ_vars:
            print(f"      ✅ Collected {len(environ_vars)} environment variables")
        else:
            print("      ℹ️  No environment variables specified")
        
        return environ_vars
    
    def _collect_job_args(self, node_name: str, suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect job arguments for the script.
        
        Args:
            node_name: Name of the DAG node
            suggestions: Contract suggestions
            
        Returns:
            Dictionary of job arguments
        """
        job_args = {}
        suggested_args = suggestions.get("job_args", {})
        
        print(f"   ⚙️  Job Arguments for {node_name}:")
        
        if suggested_args:
            print(f"      💡 Contract suggestions: {list(suggested_args.keys())}")
            
            for arg_name, default_value in suggested_args.items():
                prompt = f"      {arg_name}"
                if default_value is not None:
                    prompt += f" (default: {default_value})"
                prompt += ": "
                
                user_input = input(prompt).strip()
                if user_input:
                    # Try to parse as appropriate type
                    job_args[arg_name] = self._parse_job_arg_value(user_input)
                elif default_value is not None:
                    job_args[arg_name] = default_value
        
        # Allow additional job arguments
        print("      ➕ Additional job arguments (name=value, empty to finish):")
        while True:
            user_input = input("      ").strip()
            if not user_input:
                break
            
            if '=' in user_input:
                arg_name, arg_value = user_input.split('=', 1)
                job_args[arg_name.strip()] = self._parse_job_arg_value(arg_value.strip())
            else:
                print("         ⚠️  Format: name=value")
        
        if job_args:
            print(f"      ✅ Collected {len(job_args)} job arguments")
        else:
            print("      ℹ️  No job arguments specified")
        
        return job_args
    
    def _collect_input_paths(
        self, 
        node_name: str, 
        dependencies: List[str], 
        previous_inputs: Dict[str, Any],
        suggestions: Dict[str, Any],
        test_workspace_dir: str
    ) -> Dict[str, str]:
        """
        Collect input paths for the script with dependency awareness.
        
        Args:
            node_name: Name of the DAG node
            dependencies: List of dependency node names
            previous_inputs: Previously collected inputs for context
            suggestions: Contract suggestions
            test_workspace_dir: Test workspace directory
            
        Returns:
            Dictionary of input paths
        """
        input_paths = {}
        suggested_inputs = suggestions.get("input_paths", {})
        
        print(f"   📥 Input Paths for {node_name}:")
        
        # Show dependency context
        if dependencies:
            print(f"      🔗 Available from dependencies:")
            for dep in dependencies:
                if dep in previous_inputs:
                    dep_outputs = previous_inputs[dep].get("output_paths", {})
                    if dep_outputs:
                        for output_name, output_path in dep_outputs.items():
                            print(f"         {dep}.{output_name}: {output_path}")
        
        # Collect suggested inputs
        if suggested_inputs:
            print(f"      💡 Contract suggestions: {list(suggested_inputs.keys())}")
            
            for input_name, default_path in suggested_inputs.items():
                # Try to auto-resolve from dependencies first
                resolved_path = self._try_resolve_from_dependencies(
                    input_name, dependencies, previous_inputs
                )
                
                if resolved_path:
                    print(f"      {input_name}: {resolved_path} (auto-resolved from dependencies)")
                    input_paths[input_name] = resolved_path
                else:
                    prompt = f"      {input_name}"
                    if default_path:
                        # Make path relative to test workspace if it's absolute
                        if Path(default_path).is_absolute():
                            default_path = str(Path(test_workspace_dir) / Path(default_path).name)
                        prompt += f" (default: {default_path})"
                    prompt += ": "
                    
                    user_input = input(prompt).strip()
                    if user_input:
                        # Make path absolute if relative
                        if not Path(user_input).is_absolute():
                            user_input = str(Path(test_workspace_dir) / user_input)
                        input_paths[input_name] = user_input
                    elif default_path:
                        input_paths[input_name] = default_path
        
        # Allow additional input paths
        print("      ➕ Additional input paths (name=path, empty to finish):")
        while True:
            user_input = input("      ").strip()
            if not user_input:
                break
            
            if '=' in user_input:
                path_name, path_value = user_input.split('=', 1)
                path_name = path_name.strip()
                path_value = path_value.strip()
                
                # Make path absolute if relative
                if not Path(path_value).is_absolute():
                    path_value = str(Path(test_workspace_dir) / path_value)
                
                input_paths[path_name] = path_value
            else:
                print("         ⚠️  Format: name=path")
        
        if input_paths:
            print(f"      ✅ Collected {len(input_paths)} input paths")
        else:
            print("      ℹ️  No input paths specified")
        
        return input_paths
    
    def _collect_output_paths(
        self, 
        node_name: str, 
        suggestions: Dict[str, Any],
        test_workspace_dir: str
    ) -> Dict[str, str]:
        """
        Collect output paths for the script.
        
        Args:
            node_name: Name of the DAG node
            suggestions: Contract suggestions
            test_workspace_dir: Test workspace directory
            
        Returns:
            Dictionary of output paths
        """
        output_paths = {}
        suggested_outputs = suggestions.get("output_paths", {})
        
        print(f"   📤 Output Paths for {node_name}:")
        
        if suggested_outputs:
            print(f"      💡 Contract suggestions: {list(suggested_outputs.keys())}")
            
            for output_name, default_path in suggested_outputs.items():
                prompt = f"      {output_name}"
                if default_path:
                    # Make path relative to test workspace if it's absolute
                    if Path(default_path).is_absolute():
                        default_path = str(Path(test_workspace_dir) / Path(default_path).name)
                    prompt += f" (default: {default_path})"
                prompt += ": "
                
                user_input = input(prompt).strip()
                if user_input:
                    # Make path absolute if relative
                    if not Path(user_input).is_absolute():
                        user_input = str(Path(test_workspace_dir) / user_input)
                    output_paths[output_name] = user_input
                elif default_path:
                    output_paths[output_name] = default_path
        
        # Allow additional output paths
        print("      ➕ Additional output paths (name=path, empty to finish):")
        while True:
            user_input = input("      ").strip()
            if not user_input:
                break
            
            if '=' in user_input:
                path_name, path_value = user_input.split('=', 1)
                path_name = path_name.strip()
                path_value = path_value.strip()
                
                # Make path absolute if relative
                if not Path(path_value).is_absolute():
                    path_value = str(Path(test_workspace_dir) / path_value)
                
                output_paths[path_name] = path_value
            else:
                print("         ⚠️  Format: name=path")
        
        if output_paths:
            print(f"      ✅ Collected {len(output_paths)} output paths")
        else:
            print("      ℹ️  No output paths specified")
        
        return output_paths
    
    def _try_resolve_from_dependencies(
        self, 
        input_name: str, 
        dependencies: List[str], 
        previous_inputs: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to auto-resolve input path from dependency outputs.
        
        Args:
            input_name: Name of the input to resolve
            dependencies: List of dependency node names
            previous_inputs: Previously collected inputs
            
        Returns:
            Resolved path if found, None otherwise
        """
        for dep in dependencies:
            if dep in previous_inputs:
                dep_outputs = previous_inputs[dep].get("output_paths", {})
                
                # Try exact match first
                if input_name in dep_outputs:
                    return dep_outputs[input_name]
                
                # Try semantic matching
                for output_name, output_path in dep_outputs.items():
                    if self._paths_semantically_match(output_name, input_name):
                        return output_path
        
        return None
    
    def _paths_semantically_match(self, output_name: str, input_name: str) -> bool:
        """
        Simple semantic matching for path names.
        
        Args:
            output_name: Name of the output path
            input_name: Name of the input path
            
        Returns:
            True if paths semantically match
        """
        # Simple matching patterns
        return (
            output_name == input_name or
            output_name in input_name or
            input_name in output_name or
            output_name.replace('_', '') == input_name.replace('_', '') or
            # Common semantic patterns
            (output_name.endswith('_data') and input_name.startswith('input_')) or
            (output_name.startswith('processed_') and input_name.endswith('_input')) or
            (output_name.endswith('_output') and input_name.startswith('input_')) or
            (output_name.startswith('output_') and input_name.endswith('_input'))
        )
    
    def _parse_job_arg_value(self, value: str) -> Any:
        """
        Parse job argument value to appropriate type.
        
        Args:
            value: String value to parse
            
        Returns:
            Parsed value with appropriate type
        """
        # Try to parse as different types
        value = value.strip()
        
        # Boolean
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String (default)
        return value
    
    def get_collector_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the collector configuration and capabilities.
        
        Returns:
            Dictionary with collector summary information
        """
        return {
            "step_catalog_available": self.step_catalog is not None,
            "contract_aware_suggestions": True,
            "dependency_aware_resolution": True,
            "interactive_collection": True,
            "collector_type": "ScriptInputCollector",
            "mirrors_pattern": "cursus/api/factory input collection patterns",
        }
    
    def __str__(self) -> str:
        """String representation of the collector."""
        return f"ScriptInputCollector(step_catalog={self.step_catalog is not None})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ScriptInputCollector("
            f"step_catalog={self.step_catalog is not None})"
        )
