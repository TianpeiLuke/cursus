"""
File resolver adapters for backward compatibility.

This module provides adapters that maintain existing file resolver APIs
during the migration from legacy discovery systems to the unified StepCatalog system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from collections import defaultdict

from ..step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class FlexibleFileResolverAdapter:
    """
    Modernized file resolver using unified step catalog system.
    
    Replaces: src/cursus/validation/alignment/file_resolver.py
    
    This adapter leverages the step catalog's superior discovery capabilities
    while maintaining backward compatibility with legacy FlexibleFileResolver APIs.
    """
    
    def __init__(self, workspace_root: Path):
        """
        Initialize file resolver with workspace root.
        
        Args:
            workspace_root: Path to workspace root directory for workspace-aware discovery
        """
        # PORTABLE: Use workspace-aware discovery with dual search space API
        self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        self.logger = logging.getLogger(__name__)
        
        # Legacy compatibility attributes (kept for backward compatibility)
        self.base_dirs = None  # No longer used, but kept to avoid breaking existing code
        self.file_cache = {}
        self._refresh_cache()
    
    def _refresh_cache(self):
        """Refresh file cache using step catalog discovery."""
        try:
            steps = self.catalog.list_available_steps()
            self.file_cache = {
                "scripts": {},
                "contracts": {},
                "specs": {},
                "builders": {},
                "configs": {}
            }
            
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info:
                    for component_type, metadata in step_info.file_components.items():
                        if metadata:
                            # Map singular component types to plural cache keys
                            cache_key = f"{component_type}s"  # script -> scripts, contract -> contracts, etc.
                            if cache_key in self.file_cache:
                                # Extract base name for legacy compatibility
                                base_name = self._extract_base_name(step_name, component_type)
                                self.file_cache[cache_key][base_name] = str(metadata.path)
        except Exception as e:
            self.logger.error(f"Error refreshing cache: {e}")
    
    def _extract_base_name(self, step_name: str, component_type: str) -> str:
        """Extract base name from step name for legacy compatibility."""
        # Convert PascalCase to snake_case for legacy compatibility
        import re
        snake_case = re.sub('([A-Z]+)', r'_\1', step_name).lower().strip('_')
        return snake_case
    
    def _normalize_name(self, name: str) -> str:
        """
        Modernized name normalization using step catalog patterns.
        
        Handles common variations while leveraging catalog knowledge.
        """
        normalized = name.lower().replace("-", "_").replace(".", "_")
        
        # Handle common abbreviations using catalog knowledge
        variations = {
            "preprocess": "preprocessing",
            "eval": "evaluation", 
            "xgb": "xgboost",
            "train": "training",
        }
        
        for short, long in variations.items():
            if short in normalized and long not in normalized:
                normalized = normalized.replace(short, long)
        
        return normalized
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity using difflib for intelligent matching."""
        import difflib
        return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _find_best_match(self, script_name: str, component_type: str) -> Optional[str]:
        """
        Find best matching file using step catalog + intelligent matching.
        
        Combines catalog-based discovery with legacy fuzzy matching logic.
        """
        # Strategy 1: Direct catalog lookup
        step_info = self.catalog.get_step_info(script_name)
        if step_info and step_info.file_components.get(component_type):
            return str(step_info.file_components[component_type].path)
        
        # Strategy 2: Search through catalog steps with fuzzy matching
        steps = self.catalog.list_available_steps()
        best_match = None
        best_score = 0.0
        
        normalized_script = self._normalize_name(script_name)
        
        for step_name in steps:
            step_info = self.catalog.get_step_info(step_name)
            if not step_info or not step_info.file_components.get(component_type):
                continue
            
            # Try multiple matching strategies
            candidates = [
                step_name.lower(),
                self._extract_base_name(step_name, component_type),
                self._normalize_name(step_name)
            ]
            
            for candidate in candidates:
                # Exact match
                if script_name.lower() == candidate:
                    return str(step_info.file_components[component_type].path)
                
                # Partial match
                if script_name.lower() in candidate or candidate in script_name.lower():
                    score = 0.8
                else:
                    # Fuzzy match
                    score = self._calculate_similarity(normalized_script, candidate)
                
                if score > best_score and score >= 0.6:
                    best_score = score
                    best_match = str(step_info.file_components[component_type].path)
        
        return best_match
    
    def refresh_cache(self):
        """Legacy method: refresh cache using catalog."""
        self._refresh_cache()
    
    def _discover_all_files(self):
        """Legacy method: discover files using catalog."""
        self._refresh_cache()
    
    def _scan_directory(self, directory: Path, component_type: str) -> Dict[str, str]:
        """Legacy method: return cached files from catalog."""
        return self.file_cache.get(component_type, {})
    
    def get_available_files_report(self) -> Dict[str, Any]:
        """Generate report using catalog data."""
        report = {}
        for component_type in ["scripts", "contracts", "specs", "builders", "configs"]:
            files = self.file_cache.get(component_type, {})
            report[component_type] = {
                "count": len(files),
                "files": list(files.values()),
                "base_names": list(files.keys())
            }
        return report
    
    def extract_base_name_from_spec(self, spec_path: Path) -> str:
        """Extract base name from spec path."""
        stem = spec_path.stem
        if stem.endswith("_spec"):
            return stem[:-5]  # Remove only _spec suffix, preserve canonical step name
        return stem
    
    def find_spec_constant_name(self, script_name: str, job_type: str = "training") -> Optional[str]:
        """Find spec constant name using catalog."""
        spec_file = self.find_spec_file(script_name)
        if spec_file:
            base_name = self.extract_base_name_from_spec(Path(spec_file))
            return f"{base_name.upper()}_{job_type.upper()}_SPEC"
        return f"{script_name.upper()}_{job_type.upper()}_SPEC"
    
    def find_specification_file(self, script_name: str) -> Optional[Path]:
        """Legacy alias for find_spec_file."""
        result = self.find_spec_file(script_name)
        return Path(result) if result else None
    
    def find_contract_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find contract file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('contract'):
                return step_info.file_components['contract'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding contract file for {step_name}: {e}")
            return None
    
    def find_spec_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find spec file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('spec'):
                return step_info.file_components['spec'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding spec file for {step_name}: {e}")
            return None
    
    def find_builder_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find builder file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('builder'):
                return step_info.file_components['builder'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding builder file for {step_name}: {e}")
            return None
    
    def find_config_file(self, step_name: str) -> Optional[Path]:
        """Legacy method: find config file for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('config'):
                return step_info.file_components['config'].path
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding config file for {step_name}: {e}")
            return None
    
    def find_all_component_files(self, step_name: str) -> Dict[str, Optional[Path]]:
        """Legacy method: find all component files for step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info:
                return {
                    component_type: metadata.path if metadata else None
                    for component_type, metadata in step_info.file_components.items()
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error finding component files for {step_name}: {e}")
            return {}


class DeveloperWorkspaceFileResolverAdapter(FlexibleFileResolverAdapter):
    """
    Adapter maintaining backward compatibility with DeveloperWorkspaceFileResolver.
    
    Replaces: src/cursus/workspace/validation/workspace_file_resolver.py
    """
    
    def __init__(self, workspace_root: Optional[Path] = None, project_id: Optional[str] = None, 
                 developer_id: Optional[str] = None, enable_shared_fallback: Optional[bool] = True, **kwargs):
        """Initialize with workspace-aware unified catalog."""
        # Handle both workspace mode and single workspace mode
        if workspace_root is not None:
            workspace_root = Path(workspace_root)
            super().__init__(workspace_root)
            self.workspace_root = workspace_root
            self.workspace_mode = True
            
            # Validate workspace structure
            self._validate_workspace_structure(workspace_root, developer_id)
        else:
            # Single workspace mode (legacy compatibility)
            super().__init__(Path('.'))  # fallback to current directory
            self.workspace_root = None
            self.workspace_mode = False
        
        # Support both parameter names for backward compatibility
        self.project_id = project_id or developer_id
        self.developer_id = developer_id or project_id  # Legacy alias
        self.enable_shared_fallback = enable_shared_fallback if workspace_root else False
        
        # Set up workspace-specific directory paths for legacy compatibility
        self._setup_workspace_paths()
    
    def _validate_workspace_structure(self, workspace_root: Path, developer_id: Optional[str]):
        """Validate workspace structure exists."""
        if not workspace_root.exists():
            raise ValueError(f"Workspace root does not exist: {workspace_root}")
        
        if developer_id:
            dev_path = workspace_root / "developers" / developer_id
            if not dev_path.exists():
                raise ValueError(f"Developer workspace does not exist: {dev_path}")
    
    def _setup_workspace_paths(self):
        """Set up workspace-specific directory paths with simplified structure."""
        if self.workspace_mode and self.developer_id:
            # Simplified structure: workspace directories directly contain component directories
            dev_workspace = self.workspace_root / "developers" / self.developer_id
            self.contracts_dir = dev_workspace / "contracts"
            self.specs_dir = dev_workspace / "specs"
            self.builders_dir = dev_workspace / "builders"
            self.scripts_dir = dev_workspace / "scripts"
            self.configs_dir = dev_workspace / "configs"
            
            # Shared workspace paths (for fallback) with simplified structure
            if self.enable_shared_fallback:
                shared_workspace = self.workspace_root / "shared"
                self.shared_contracts_dir = shared_workspace / "contracts"
                self.shared_specs_dir = shared_workspace / "specs"
                self.shared_builders_dir = shared_workspace / "builders"
                self.shared_scripts_dir = shared_workspace / "scripts"
                self.shared_configs_dir = shared_workspace / "configs"
        else:
            # Single workspace mode - no specific paths
            self.contracts_dir = None
            self.specs_dir = None
            self.builders_dir = None
            self.scripts_dir = None
            self.configs_dir = None
    
    def _find_workspace_file(self, step_name: str, component_type: str, extensions: List[str], 
                           file_name: Optional[str] = None) -> Optional[str]:
        """Unified workspace-aware file discovery with step catalog + directory fallback."""
        try:
            # First try workspace-specific lookup via step catalog
            if self.project_id:
                workspace_steps = self.catalog.list_available_steps(workspace_id=self.project_id)
                if step_name in workspace_steps:
                    step_info = self.catalog.get_step_info(step_name)
                    if step_info and step_info.workspace_id == self.project_id:
                        if step_info.file_components.get(component_type):
                            return str(step_info.file_components[component_type].path)
            
            # Fallback to core lookup using step catalog
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get(component_type):
                return str(step_info.file_components[component_type].path)
            
            # Fallback to directory scanning (for test files not in catalog)
            if self.workspace_mode:
                # Try developer workspace first
                dev_dir = getattr(self, f"{component_type}s_dir", None)
                if dev_dir and dev_dir.exists():
                    result = self._find_file_in_directory(str(dev_dir), step_name, file_name, extensions)
                    if result:
                        return result
                
                # Try shared workspace if enabled
                if self.enable_shared_fallback:
                    shared_dir = getattr(self, f"shared_{component_type}s_dir", None)
                    if shared_dir and shared_dir.exists():
                        result = self._find_file_in_directory(str(shared_dir), step_name, file_name, extensions)
                        if result:
                            return result
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding workspace {component_type} file for {step_name}: {e}")
            return None
    
    def find_contract_file(self, step_name: str) -> Optional[str]:
        """Workspace-aware contract file discovery. Returns string path like legacy method."""
        return self._find_workspace_file(step_name, "contract", [".py"])
    
    def find_spec_file(self, step_name: str) -> Optional[str]:
        """Workspace-aware spec file discovery. Returns string path like legacy method."""
        return self._find_workspace_file(step_name, "spec", [".py"])
    
    def find_builder_file(self, step_name: str) -> Optional[str]:
        """Workspace-aware builder file discovery. Returns string path like legacy method."""
        return self._find_workspace_file(step_name, "builder", [".py"])
    
    def find_config_file(self, step_name: str) -> Optional[str]:
        """Workspace-aware config file discovery. Returns string path like legacy method."""
        return self._find_workspace_file(step_name, "config", [".py"])
    
    def find_script_file(self, step_name: str, script_name: Optional[str] = None) -> Optional[str]:
        """Workspace-aware script file discovery. Returns string path like legacy method."""
        return self._find_workspace_file(step_name, "script", [".py"], script_name)
    
    def _find_file_in_directory(self, directory: str, step_name: str, file_name: Optional[str], extensions: List[str]) -> Optional[str]:
        """Find file in specified directory with given extensions using legacy patterns."""
        import os
        if not directory or not os.path.exists(directory):
            return None

        # Get component type from directory name for pattern matching
        component_type = os.path.basename(directory).rstrip('s')  # Remove trailing 's' (contracts -> contract)
        
        # Build search patterns based on legacy cursus conventions
        search_patterns = []
        
        if file_name:
            search_patterns.append(file_name)
        
        # Add component-specific patterns based on cursus conventions
        if component_type == "contract":
            search_patterns.extend([
                f"{step_name}_contract",
                f"{step_name}Contract",
                step_name
            ])
        elif component_type == "spec":
            search_patterns.extend([
                f"{step_name}_spec",
                f"{step_name}Spec", 
                step_name
            ])
        elif component_type == "builder":
            search_patterns.extend([
                f"builder_{step_name}_step",
                f"builder_{step_name}",
                f"{step_name}_builder",
                f"{step_name}Builder",
                step_name
            ])
        elif component_type == "config":
            search_patterns.extend([
                f"config_{step_name}_step",
                f"config_{step_name}",
                f"{step_name}_config",
                f"{step_name}Config",
                step_name
            ])
        elif component_type == "script":
            search_patterns.extend([
                step_name,
                f"{step_name}_script"
            ])
        else:
            search_patterns.append(step_name)

        # Try each pattern with each extension
        for pattern in search_patterns:
            for ext in extensions:
                file_path = os.path.join(directory, f"{pattern}{ext}")
                if os.path.exists(file_path):
                    return file_path

        return None
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get workspace information using step catalog."""
        try:
            info = {
                "workspace_mode": self.workspace_mode,
                "workspace_root": str(self.workspace_root) if self.workspace_root else None,
                "developer_id": self.developer_id,
                "enable_shared_fallback": self.enable_shared_fallback,
                "developer_workspace_exists": False,
                "shared_workspace_exists": False,
            }
            
            if self.workspace_mode and self.workspace_root:
                # Check if developer workspace exists
                if self.developer_id:
                    dev_path = self.workspace_root / "developers" / self.developer_id
                    info["developer_workspace_exists"] = dev_path.exists()
                
                # Check if shared workspace exists
                shared_path = self.workspace_root / "shared"
                info["shared_workspace_exists"] = shared_path.exists()
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting workspace info: {e}")
            return {"error": str(e)}
    
    def list_available_developers(self) -> List[str]:
        """List available developers using workspace discovery."""
        try:
            if not self.workspace_mode or not self.workspace_root:
                return []
            
            developers = []
            developers_dir = self.workspace_root / "developers"
            
            if developers_dir.exists():
                for item in developers_dir.iterdir():
                    if item.is_dir():
                        developers.append(item.name)
            
            return sorted(developers)
            
        except Exception as e:
            self.logger.error(f"Error listing developers: {e}")
            return []
    
    def switch_developer(self, developer_id: str) -> None:
        """Switch to a different developer workspace."""
        try:
            if not self.workspace_mode:
                raise ValueError("Cannot switch developer in single workspace mode")
            
            # Validate new developer exists
            available_developers = self.list_available_developers()
            if developer_id not in available_developers:
                raise ValueError(f"Developer workspace not found: {developer_id}")
            
            # Update developer ID
            self.developer_id = developer_id
            self.project_id = developer_id  # Keep in sync
            
            # Update workspace paths
            self._setup_workspace_paths()
            
        except Exception as e:
            self.logger.error(f"Error switching developer to {developer_id}: {e}")
            raise


class HybridFileResolverAdapter:
    """
    Adapter maintaining backward compatibility with HybridFileResolver.
    
    Replaces: src/cursus/validation/alignment/patterns/file_resolver.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        # PORTABLE: Use workspace-aware discovery for pattern-based file resolution
        self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        self.logger = logging.getLogger(__name__)
    
    def resolve_file_pattern(self, pattern: str, component_type: str) -> List[Path]:
        """Legacy method: resolve files matching pattern."""
        try:
            results = []
            steps = self.catalog.list_available_steps()
            
            for step_name in steps:
                if pattern.lower() in step_name.lower():
                    step_info = self.catalog.get_step_info(step_name)
                    if step_info and step_info.file_components.get(component_type):
                        results.append(step_info.file_components[component_type].path)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error resolving file pattern {pattern}: {e}")
            return []
