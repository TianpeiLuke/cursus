"""
Script file auto-discovery for the unified step catalog system.

This module implements script file discovery from both core package and workspace
directories, with workspace prioritization support for interactive runtime testing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import importlib.util

logger = logging.getLogger(__name__)


class ScriptInfo:
    """Information about a discovered script."""
    
    def __init__(self, script_name: str, step_name: str, script_path: Path, 
                 workspace_id: str, framework: Optional[str] = None):
        self.script_name = script_name
        self.step_name = step_name
        self.script_path = script_path
        self.workspace_id = workspace_id
        self.framework = framework
        self.metadata = {}
    
    def __repr__(self):
        return f"ScriptInfo(script_name='{self.script_name}', step_name='{self.step_name}', workspace_id='{self.workspace_id}')"


class ScriptAutoDiscovery:
    """
    Script file auto-discovery with workspace prioritization support.
    
    Follows the same pattern as ConfigAutoDiscovery and BuilderAutoDiscovery
    for consistency within the step catalog system.
    """
    
    def __init__(self, package_root: Path, workspace_dirs: Optional[List[Path]] = None, 
                 priority_workspace_dir: Optional[Path] = None):
        """
        Initialize script discovery with workspace prioritization.
        
        Args:
            package_root: Root directory of the cursus package
            workspace_dirs: Optional list of workspace directories to search
            priority_workspace_dir: Optional priority workspace directory (from config_base.source_dir)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🔍 ScriptAutoDiscovery.__init__ starting - package_root: {package_root}")
        self.logger.info(f"🔍 ScriptAutoDiscovery.__init__ - workspace_dirs: {workspace_dirs}")
        self.logger.info(f"🔍 ScriptAutoDiscovery.__init__ - priority_workspace_dir: {priority_workspace_dir}")
        
        self.package_root = package_root
        self.workspace_dirs = workspace_dirs or []
        self.priority_workspace_dir = priority_workspace_dir
        
        # Add priority workspace to the beginning of workspace_dirs if provided
        if self.priority_workspace_dir and self.priority_workspace_dir not in self.workspace_dirs:
            self.workspace_dirs = [self.priority_workspace_dir] + self.workspace_dirs
            self.logger.info(f"Added priority workspace to search list: {self.priority_workspace_dir}")
        
        # Caches for performance
        self._script_cache: Dict[str, ScriptInfo] = {}
        self._discovery_complete = False
        
        # Discovery results
        self._package_scripts: Dict[str, ScriptInfo] = {}
        self._workspace_scripts: Dict[str, Dict[str, ScriptInfo]] = {}  # workspace_id -> scripts
        
        # Registry integration
        self._registry_info: Dict[str, Dict[str, Any]] = {}
        
        try:
            self.logger.debug("🔍 Loading registry info...")
            self._load_registry_info()
            self.logger.info(f"✅ Registry info loaded: {len(self._registry_info)} steps")
        except Exception as e:
            self.logger.error(f"❌ Registry info loading failed: {e}")
            self._registry_info = {}
        
        self.logger.info("🎉 ScriptAutoDiscovery initialization completed successfully")
    
    def _load_registry_info(self):
        """
        Load registry information from cursus/registry/step_names.py.
        
        This provides authoritative information about step names and metadata
        that can guide the script discovery process.
        """
        try:
            from ..registry.step_names import get_step_names
            
            step_names_dict = get_step_names()
            for step_name, step_info in step_names_dict.items():
                self._registry_info[step_name] = step_info
            
            self.logger.debug(f"Loaded registry info for {len(self._registry_info)} steps")
            
        except ImportError as e:
            self.logger.warning(f"Could not import registry step_names: {e}")
            self._registry_info = {}
        except Exception as e:
            self.logger.error(f"Error loading registry info: {e}")
            self._registry_info = {}
    
    def discover_script_files(self, project_id: Optional[str] = None) -> Dict[str, ScriptInfo]:
        """
        Discover all script files from package and workspaces with prioritization.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping script names to ScriptInfo objects
        """
        if not self._discovery_complete:
            self._run_discovery()
        
        # Combine scripts with workspace prioritization
        all_scripts = {}
        
        # Add package scripts first (lowest priority)
        all_scripts.update(self._package_scripts)
        
        # Add workspace scripts in reverse order (later workspaces override earlier ones)
        # But priority workspace (first in list) has highest priority
        workspace_order = list(self._workspace_scripts.keys())
        if self.priority_workspace_dir:
            priority_workspace_id = self.priority_workspace_dir.name
            if priority_workspace_id in workspace_order:
                # Move priority workspace to end so it overrides others
                workspace_order.remove(priority_workspace_id)
                workspace_order.append(priority_workspace_id)
        
        for workspace_id in workspace_order:
            if workspace_id in self._workspace_scripts:
                workspace_scripts = self._workspace_scripts[workspace_id]
                all_scripts.update(workspace_scripts)
                self.logger.debug(f"Added {len(workspace_scripts)} scripts from workspace {workspace_id}")
        
        self.logger.info(f"Discovered {len(all_scripts)} script files with prioritization")
        return all_scripts
    
    def load_script_info(self, script_name: str) -> Optional[ScriptInfo]:
        """
        Load script information for a specific script with workspace-aware discovery.
        
        Args:
            script_name: Name of the script to load info for
            
        Returns:
            ScriptInfo object or None if not found
        """
        # Check cache first
        if script_name in self._script_cache:
            return self._script_cache[script_name]
        
        # Ensure discovery is complete
        if not self._discovery_complete:
            self._run_discovery()
        
        # Try priority workspace first if specified
        if self.priority_workspace_dir:
            priority_workspace_id = self.priority_workspace_dir.name
            if priority_workspace_id in self._workspace_scripts:
                workspace_scripts = self._workspace_scripts[priority_workspace_id]
                if script_name in workspace_scripts:
                    script_info = workspace_scripts[script_name]
                    self._script_cache[script_name] = script_info
                    self.logger.debug(f"Loaded script {script_name} from priority workspace {priority_workspace_id}")
                    return script_info
        
        # Try other workspace scripts (reverse order for proper priority)
        workspace_ids = list(self._workspace_scripts.keys())
        if self.priority_workspace_dir:
            priority_workspace_id = self.priority_workspace_dir.name
            if priority_workspace_id in workspace_ids:
                workspace_ids.remove(priority_workspace_id)
        
        for workspace_id in reversed(workspace_ids):
            workspace_scripts = self._workspace_scripts[workspace_id]
            if script_name in workspace_scripts:
                script_info = workspace_scripts[script_name]
                self._script_cache[script_name] = script_info
                self.logger.debug(f"Loaded script {script_name} from workspace {workspace_id}")
                return script_info
        
        # Try package scripts
        if script_name in self._package_scripts:
            script_info = self._package_scripts[script_name]
            self._script_cache[script_name] = script_info
            self.logger.debug(f"Loaded script {script_name} from package")
            return script_info
        
        # Not found
        self.logger.warning(f"No script file found for: {script_name}")
        return None
    
    def discover_scripts_from_dag(self, dag) -> Dict[str, ScriptInfo]:
        """
        Discover scripts referenced in a DAG with intelligent node-to-script mapping.
        
        Args:
            dag: PipelineDAG object
            
        Returns:
            Dictionary mapping script names to ScriptInfo objects
        """
        discovered_scripts = {}
        
        # Ensure discovery is complete
        if not self._discovery_complete:
            self._run_discovery()
        
        # Get all available scripts
        all_scripts = self.discover_script_files()
        
        # Map DAG nodes to scripts
        for node_name in dag.nodes:
            script_info = self._map_dag_node_to_script(node_name, all_scripts)
            if script_info:
                discovered_scripts[script_info.script_name] = script_info
                self.logger.debug(f"Mapped DAG node {node_name} to script {script_info.script_name}")
            else:
                self.logger.warning(f"Could not map DAG node {node_name} to any script")
        
        self.logger.info(f"Discovered {len(discovered_scripts)} scripts from DAG with {len(dag.nodes)} nodes")
        return discovered_scripts
    
    def _map_dag_node_to_script(self, node_name: str, available_scripts: Dict[str, ScriptInfo]) -> Optional[ScriptInfo]:
        """
        Map a DAG node name to a script using intelligent resolution.
        
        Args:
            node_name: DAG node name (e.g., "XGBoostTraining_training")
            available_scripts: Dictionary of available scripts
            
        Returns:
            ScriptInfo object or None if no mapping found
        """
        # Try direct match first
        if node_name in available_scripts:
            script_info = available_scripts[node_name]
            script_info.step_name = node_name  # Update step name for DAG context
            return script_info
        
        # Try to extract canonical step name using registry
        try:
            from ..registry.step_names import get_step_name_from_spec_type
            canonical_name = get_step_name_from_spec_type(node_name)
            
            # Convert to script name (PascalCase to snake_case)
            script_name = self._canonical_to_script_name(canonical_name)
            
            # Look for script with this name
            if script_name in available_scripts:
                script_info = available_scripts[script_name]
                script_info.step_name = node_name  # Update step name for DAG context
                return script_info
            
            # Try fuzzy matching
            fuzzy_match = self._find_fuzzy_script_match(script_name, available_scripts)
            if fuzzy_match:
                fuzzy_match.step_name = node_name  # Update step name for DAG context
                return fuzzy_match
                
        except Exception as e:
            self.logger.debug(f"Error in registry-based mapping for {node_name}: {e}")
        
        # Try direct fuzzy matching with node name
        fuzzy_match = self._find_fuzzy_script_match(node_name, available_scripts)
        if fuzzy_match:
            fuzzy_match.step_name = node_name  # Update step name for DAG context
            return fuzzy_match
        
        return None
    
    def _canonical_to_script_name(self, canonical_name: str) -> str:
        """
        Convert canonical step name (PascalCase) to script name (snake_case).
        
        Handles special cases for compound technical terms.
        
        Args:
            canonical_name: PascalCase canonical name
            
        Returns:
            snake_case script name
        """
        import re
        
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
    
    def _find_fuzzy_script_match(self, target_name: str, available_scripts: Dict[str, ScriptInfo]) -> Optional[ScriptInfo]:
        """
        Find script using fuzzy matching for error recovery.
        
        Args:
            target_name: Target script name to match
            available_scripts: Dictionary of available scripts
            
        Returns:
            Best matching ScriptInfo or None
        """
        from difflib import SequenceMatcher
        
        best_match = None
        best_ratio = 0.0
        threshold = 0.7  # Minimum similarity threshold
        
        target_lower = target_name.lower()
        
        for script_name, script_info in available_scripts.items():
            script_lower = script_name.lower()
            ratio = SequenceMatcher(None, target_lower, script_lower).ratio()
            
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = script_info
        
        if best_match:
            self.logger.debug(f"Fuzzy match: '{target_name}' -> '{best_match.script_name}' (similarity: {best_ratio:.2f})")
        
        return best_match
    
    def _run_discovery(self):
        """Run the complete discovery process."""
        try:
            # Discover package scripts
            self._discover_package_scripts()
            
            # Discover workspace scripts
            self._discover_workspace_scripts()
            
            self._discovery_complete = True
            
            total_scripts = len(self._package_scripts) + sum(
                len(scripts) for scripts in self._workspace_scripts.values()
            )
            self.logger.info(f"Script discovery complete: {total_scripts} scripts found")
            
        except Exception as e:
            self.logger.error(f"Error during script discovery: {e}")
            # Graceful degradation
            self._package_scripts = {}
            self._workspace_scripts = {}
    
    def _discover_package_scripts(self):
        """Discover scripts in the cursus package using config and contract entry points."""
        try:
            # Use config discovery to find script entry points
            self._package_scripts = self._discover_scripts_from_configs_and_contracts("package")
            self.logger.debug(f"Found {len(self._package_scripts)} package scripts from configs and contracts")
        except Exception as e:
            self.logger.error(f"Error discovering package scripts: {e}")
            self._package_scripts = {}
    
    def _discover_workspace_scripts(self):
        """Discover scripts in workspace directories using config and contract entry points."""
        for workspace_dir in self.workspace_dirs:
            try:
                workspace_path = Path(workspace_dir)
                if not workspace_path.exists():
                    self.logger.warning(f"Workspace directory does not exist: {workspace_path}")
                    continue
                
                workspace_scripts = self._discover_scripts_from_configs_and_contracts(workspace_path.name, workspace_path)
                if workspace_scripts:
                    self._workspace_scripts[workspace_path.name] = workspace_scripts
                    self.logger.debug(f"Found {len(workspace_scripts)} scripts in workspace {workspace_path.name}")
                    
            except Exception as e:
                self.logger.error(f"Error discovering workspace scripts in {workspace_dir}: {e}")
    
    def _discover_scripts_from_configs_and_contracts(self, workspace_id: str, workspace_path: Optional[Path] = None) -> Dict[str, ScriptInfo]:
        """
        Discover scripts by analyzing config and contract classes for entry point information.
        
        Args:
            workspace_id: ID of the workspace (for logging and metadata)
            workspace_path: Optional workspace path for workspace-specific discovery
            
        Returns:
            Dictionary mapping script names to ScriptInfo objects
        """
        discovered_scripts = {}
        
        try:
            # Discover scripts from config classes
            config_scripts = self._discover_scripts_from_configs(workspace_id, workspace_path)
            discovered_scripts.update(config_scripts)
            
            # Discover scripts from contract classes
            contract_scripts = self._discover_scripts_from_contracts(workspace_id, workspace_path)
            discovered_scripts.update(contract_scripts)
            
            self.logger.debug(f"Discovered {len(discovered_scripts)} scripts from configs and contracts in {workspace_id}")
            
        except Exception as e:
            self.logger.error(f"Error discovering scripts from configs and contracts in {workspace_id}: {e}")
        
        return discovered_scripts
    
    def _discover_scripts_from_configs(self, workspace_id: str, workspace_path: Optional[Path] = None) -> Dict[str, ScriptInfo]:
        """
        Discover scripts from configuration classes by analyzing entry point fields.
        
        Args:
            workspace_id: ID of the workspace
            workspace_path: Optional workspace path for workspace-specific discovery
            
        Returns:
            Dictionary mapping script names to ScriptInfo objects
        """
        discovered_scripts = {}
        
        try:
            # Import config discovery
            from .config_discovery import ConfigAutoDiscovery
            
            # Create config discovery instance
            workspace_dirs = [workspace_path] if workspace_path else []
            config_discovery = ConfigAutoDiscovery(self.package_root, workspace_dirs)
            
            # Discover config classes
            config_classes = config_discovery.discover_config_classes()
            
            # Analyze each config class for entry point fields
            for config_name, config_class in config_classes.items():
                try:
                    script_info = self._extract_script_from_config(config_class, workspace_id, workspace_path)
                    if script_info:
                        discovered_scripts[script_info.script_name] = script_info
                        self.logger.debug(f"Found script {script_info.script_name} from config {config_name}")
                        
                except Exception as e:
                    self.logger.warning(f"Error extracting script from config {config_name}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error discovering scripts from configs in {workspace_id}: {e}")
        
        return discovered_scripts
    
    def _discover_scripts_from_contracts(self, workspace_id: str, workspace_path: Optional[Path] = None) -> Dict[str, ScriptInfo]:
        """
        Discover scripts from contract classes by analyzing entry point fields.
        
        Args:
            workspace_id: ID of the workspace
            workspace_path: Optional workspace path for workspace-specific discovery
            
        Returns:
            Dictionary mapping script names to ScriptInfo objects
        """
        discovered_scripts = {}
        
        try:
            # Import contract discovery
            from .contract_discovery import ContractAutoDiscovery
            
            # Create contract discovery instance
            workspace_dirs = [workspace_path] if workspace_path else []
            contract_discovery = ContractAutoDiscovery(self.package_root, workspace_dirs)
            
            # Discover contract classes
            contract_classes = contract_discovery.discover_contract_classes()
            
            # Analyze each contract class for entry point fields
            for contract_name, contract_class in contract_classes.items():
                try:
                    script_info = self._extract_script_from_contract(contract_class, workspace_id, workspace_path)
                    if script_info:
                        discovered_scripts[script_info.script_name] = script_info
                        self.logger.debug(f"Found script {script_info.script_name} from contract {contract_name}")
                        
                except Exception as e:
                    self.logger.warning(f"Error extracting script from contract {contract_name}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error discovering scripts from contracts in {workspace_id}: {e}")
        
        return discovered_scripts
    
    def _extract_script_from_config(self, config_class: type, workspace_id: str, workspace_path: Optional[Path] = None) -> Optional[ScriptInfo]:
        """
        Extract script information from a configuration class using regex-based entry point detection.
        
        Uses regex to find entry point fields and effective_source_dir pattern from step builders.
        
        Args:
            config_class: Configuration class to analyze
            workspace_id: ID of the workspace
            workspace_path: Optional workspace path
            
        Returns:
            ScriptInfo object or None if no entry point found
        """
        import re
        
        try:
            # Get class fields using Pydantic model inspection
            if hasattr(config_class, '__fields__'):
                fields = config_class.__fields__
            elif hasattr(config_class, 'model_fields'):
                fields = config_class.model_fields
            else:
                # Fallback: try to create instance and inspect
                try:
                    instance = config_class()
                    fields = {name: getattr(instance, name, None) for name in dir(instance) 
                             if not name.startswith('_')}
                except:
                    return None
            
            # Use regex to find entry point fields (more flexible than hard-coded list)
            entry_point_fields = self._find_entry_point_fields_with_regex(fields)
            
            for field_name in entry_point_fields:
                # Try to get default value or field info
                field_info = fields[field_name]
                entry_point_value = None
                
                if hasattr(field_info, 'default') and field_info.default is not None:
                    entry_point_value = field_info.default
                elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                    try:
                        entry_point_value = field_info.default_factory()
                    except:
                        pass
                
                if entry_point_value and isinstance(entry_point_value, str):
                    # Extract script name from entry point (e.g., "xgboost_training.py" -> "xgboost_training")
                    script_name = self._extract_script_name_from_entry_point(entry_point_value)
                    if script_name:
                        # Use effective_source_dir pattern like step builders
                        script_path = self._find_script_file_path_with_effective_source_dir(
                            config_class, script_name, entry_point_value, workspace_path
                        )
                        if script_path:
                            # Extract step name from config class name
                            step_name = self._extract_step_name_from_config_class(config_class.__name__)
                            
                            return ScriptInfo(
                                script_name=script_name,
                                step_name=step_name,
                                script_path=script_path,
                                workspace_id=workspace_id
                            )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting script from config {config_class.__name__}: {e}")
            return None
    
    def _find_entry_point_fields_with_regex(self, fields: Dict[str, Any]) -> List[str]:
        """
        Find all fields that match entry_point patterns using regex.
        
        Args:
            fields: Dictionary of field names to field info
            
        Returns:
            List of field names that match entry_point patterns
        """
        import re
        
        # Regex pattern to match entry point fields (case-insensitive)
        entry_point_pattern = re.compile(r'.*entry_point$', re.IGNORECASE)
        
        matching_fields = []
        for field_name in fields.keys():
            if entry_point_pattern.match(field_name):
                matching_fields.append(field_name)
        
        self.logger.debug(f"Found entry point fields: {matching_fields}")
        return matching_fields
    
    def _find_script_file_path_with_effective_source_dir(self, config_class: type, script_name: str, 
                                                        entry_point_value: str, workspace_path: Optional[Path] = None) -> Optional[Path]:
        """
        Find script file path with clear priority strategy.
        
        Strategy:
        1. Search workspace_path/scripts (if workspace_path provided)
        2. Then check effective_source_dir from config (step builder pattern)
        3. Fallback to package scripts
        
        Args:
            config_class: Configuration class that may have get_script_path method
            script_name: Name of the script (without .py extension)
            entry_point_value: Entry point value from config
            workspace_path: Optional workspace path to search first
            
        Returns:
            Path to script file or None if not found
        """
        script_filename = f"{script_name}.py"
        
        # Strategy 1: Search workspace_path/scripts first (highest priority)
        if workspace_path:
            workspace_scripts_dir = workspace_path / "scripts"
            if workspace_scripts_dir.exists():
                script_path = workspace_scripts_dir / script_filename
                if script_path.exists():
                    self.logger.debug(f"Found script in workspace scripts: {script_path}")
                    return script_path
        
        # Strategy 2: Check effective_source_dir from config (step builder pattern)
        try:
            config_instance = config_class()
            if hasattr(config_instance, 'get_script_path'):
                script_path = config_instance.get_script_path()
                if script_path:
                    script_path_obj = Path(script_path)
                    if script_path_obj.exists():
                        self.logger.debug(f"Found script using config.get_script_path(): {script_path_obj}")
                        return script_path_obj
        except Exception as e:
            self.logger.debug(f"Could not use config.get_script_path() approach: {e}")
        
        # Strategy 3: Fallback to package scripts
        package_scripts_dir = self.package_root / "steps" / "scripts"
        if package_scripts_dir.exists():
            script_path = package_scripts_dir / script_filename
            if script_path.exists():
                self.logger.debug(f"Found script in package scripts: {script_path}")
                return script_path
        
        self.logger.warning(f"Script file not found: {script_filename}")
        return None
    
    def _extract_script_from_contract(self, contract_class: type, workspace_id: str, workspace_path: Optional[Path] = None) -> Optional[ScriptInfo]:
        """
        Extract script information from a contract class.
        
        Looks for entry_point field in ScriptContract.
        
        Args:
            contract_class: Contract class to analyze
            workspace_id: ID of the workspace
            workspace_path: Optional workspace path
            
        Returns:
            ScriptInfo object or None if no entry point found
        """
        try:
            # Check if this is a ScriptContract by looking for entry_point field
            if hasattr(contract_class, '__fields__'):
                fields = contract_class.__fields__
            elif hasattr(contract_class, 'model_fields'):
                fields = contract_class.model_fields
            else:
                return None
            
            # Look for entry_point field
            if 'entry_point' in fields:
                field_info = fields['entry_point']
                entry_point_value = None
                
                if hasattr(field_info, 'default') and field_info.default is not None:
                    entry_point_value = field_info.default
                elif hasattr(field_info, 'default_factory') and field_info.default_factory is not None:
                    try:
                        entry_point_value = field_info.default_factory()
                    except:
                        pass
                
                if entry_point_value and isinstance(entry_point_value, str):
                    # Extract script name from entry point
                    script_name = self._extract_script_name_from_entry_point(entry_point_value)
                    if script_name:
                        # Find actual script file
                        script_path = self._find_script_file_path(script_name, workspace_path)
                        if script_path:
                            # Extract step name from contract class name
                            step_name = self._extract_step_name_from_contract_class(contract_class.__name__)
                            
                            return ScriptInfo(
                                script_name=script_name,
                                step_name=step_name,
                                script_path=script_path,
                                workspace_id=workspace_id
                            )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting script from contract {contract_class.__name__}: {e}")
            return None
    
    def _extract_script_name_from_entry_point(self, entry_point: str) -> Optional[str]:
        """
        Extract script name from entry point string.
        
        Args:
            entry_point: Entry point string (e.g., "xgboost_training.py" or "scripts/xgboost_training.py")
            
        Returns:
            Script name without extension or None if invalid
        """
        try:
            # Handle different entry point formats
            if entry_point.endswith('.py'):
                # Remove .py extension
                script_name = entry_point[:-3]
                
                # Remove path components if present
                if '/' in script_name:
                    script_name = script_name.split('/')[-1]
                
                return script_name
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting script name from entry point {entry_point}: {e}")
            return None
    
    def _find_script_file_path(self, script_name: str, workspace_path: Optional[Path] = None) -> Optional[Path]:
        """
        Find the actual script file path with workspace prioritization.
        
        Search order:
        1. workspace_path/scripts/ (if workspace_path provided)
        2. workspace_path/ (if workspace_path provided)
        3. package_root/steps/scripts/ (common repo fallback)
        
        Args:
            script_name: Name of the script (without .py extension)
            workspace_path: Optional workspace path to search first
            
        Returns:
            Path to script file or None if not found
        """
        script_filename = f"{script_name}.py"
        search_paths = []
        
        # Priority 1: workspace_path/scripts/ (if provided)
        if workspace_path:
            search_paths.append(workspace_path / "scripts")
            search_paths.append(workspace_path)
        
        # Priority 2: package common scripts folder
        search_paths.append(self.package_root / "steps" / "scripts")
        
        # Search in order of priority
        for search_path in search_paths:
            if search_path.exists():
                script_path = search_path / script_filename
                if script_path.exists():
                    self.logger.debug(f"Found script {script_name} at {script_path}")
                    return script_path
        
        self.logger.warning(f"Script file not found: {script_filename} in paths: {[str(p) for p in search_paths]}")
        return None
    
    def _extract_step_name_from_config_class(self, config_class_name: str) -> str:
        """
        Extract step name from config class name using registry functions.
        
        Args:
            config_class_name: Name of the config class (e.g., "XGBoostTrainingConfig")
            
        Returns:
            Step name (e.g., "XGBoostTraining")
        """
        try:
            # Use registry function directly - it handles all the complex logic
            from ..registry.step_names import get_canonical_name_from_file_name
            
            # Remove Config suffix and convert to snake_case for registry lookup
            base_name = config_class_name.replace('Config', '').replace('Configuration', '')
            
            # Convert PascalCase to snake_case
            import re
            snake_case_name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', base_name).lower()
            
            # Registry function handles all the intelligent resolution
            return get_canonical_name_from_file_name(snake_case_name)
            
        except Exception as e:
            self.logger.debug(f"Registry lookup failed for {config_class_name}: {e}")
            # Simple fallback
            return config_class_name.replace('Config', '').replace('Configuration', '')
    
    def _extract_step_name_from_contract_class(self, contract_class_name: str) -> str:
        """
        Extract step name from contract class name using registry functions.
        
        Args:
            contract_class_name: Name of the contract class (e.g., "XGBoostTrainingContract")
            
        Returns:
            Step name (e.g., "XGBoostTraining")
        """
        try:
            # Use registry function directly - it handles all the complex logic
            from ..registry.step_names import get_canonical_name_from_file_name
            
            # Remove Contract suffix and convert to snake_case for registry lookup
            base_name = contract_class_name.replace('Contract', '').replace('ScriptContract', '')
            
            # Convert PascalCase to snake_case
            import re
            snake_case_name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', base_name).lower()
            
            # Registry function handles all the intelligent resolution
            return get_canonical_name_from_file_name(snake_case_name)
            
        except Exception as e:
            self.logger.debug(f"Registry lookup failed for {contract_class_name}: {e}")
            # Simple fallback
            return contract_class_name.replace('Contract', '').replace('ScriptContract', '')
    
    
    def get_script_info(self, script_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a script.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Dictionary with script information or None if not found
        """
        script_info = self.load_script_info(script_name)
        if not script_info:
            return None
        
        return {
            "script_name": script_info.script_name,
            "step_name": script_info.step_name,
            "script_path": str(script_info.script_path),
            "workspace_id": script_info.workspace_id,
            "framework": script_info.framework,
            "metadata": script_info.metadata
        }
    
    def list_available_scripts(self) -> List[str]:
        """
        List all available script names.
        
        Returns:
            List of script names that have been discovered
        """
        if not self._discovery_complete:
            self._run_discovery()
        
        all_scripts = set(self._package_scripts.keys())
        for workspace_scripts in self._workspace_scripts.values():
            all_scripts.update(workspace_scripts.keys())
        
        return sorted(list(all_scripts))
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """
        Get discovery statistics.
        
        Returns:
            Dictionary with discovery statistics
        """
        if not self._discovery_complete:
            self._run_discovery()
        
        return {
            "package_scripts": len(self._package_scripts),
            "workspace_scripts": {
                workspace_id: len(scripts) 
                for workspace_id, scripts in self._workspace_scripts.items()
            },
            "total_scripts": len(self._package_scripts) + sum(
                len(scripts) for scripts in self._workspace_scripts.values()
            ),
            "cached_scripts": len(self._script_cache),
            "discovery_complete": self._discovery_complete,
            "priority_workspace": str(self.priority_workspace_dir) if self.priority_workspace_dir else None
        }
