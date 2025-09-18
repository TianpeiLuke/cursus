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
    
    def __init__(self, workspace_root_or_base_dirs: Union[Path, Dict[str, str]]):
        """Initialize with unified catalog or legacy base_dirs."""
        if isinstance(workspace_root_or_base_dirs, dict):
            # Legacy initialization with base_dirs dict
            self.base_dirs = workspace_root_or_base_dirs
            # Try to infer workspace root from base_dirs
            if 'contracts' in workspace_root_or_base_dirs:
                contracts_path = Path(workspace_root_or_base_dirs['contracts'])
                # Go up to find workspace root (contracts should be under src/cursus/steps/contracts)
                workspace_root = contracts_path.parent.parent.parent.parent  # up from contracts/steps/cursus/src
                if not workspace_root.exists():
                    workspace_root = Path('.')  # fallback to current directory
            else:
                workspace_root = Path('.')
            self.catalog = StepCatalog(workspace_root)
        else:
            # Modern initialization with workspace_root Path
            workspace_root = workspace_root_or_base_dirs
            self.catalog = StepCatalog(workspace_root)
            self.base_dirs = None
        
        self.logger = logging.getLogger(__name__)
        # Legacy compatibility attributes
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
                        if metadata and component_type in self.file_cache:
                            # Extract base name for legacy compatibility
                            base_name = self._extract_base_name(step_name, component_type)
                            self.file_cache[component_type][base_name] = str(metadata.path)
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
    
    def __init__(self, workspace_root: Path, project_id: Optional[str] = None):
        """Initialize with workspace-aware unified catalog."""
        super().__init__(workspace_root)
        self.project_id = project_id
    
    def find_contract_file(self, step_name: str) -> Optional[Path]:
        """Workspace-aware contract file discovery."""
        try:
            # First try workspace-specific lookup
            if self.project_id:
                workspace_steps = self.catalog.list_available_steps(workspace_id=self.project_id)
                if step_name in workspace_steps:
                    step_info = self.catalog.get_step_info(step_name)
                    if step_info and step_info.workspace_id == self.project_id:
                        if step_info.file_components.get('contract'):
                            return step_info.file_components['contract'].path
            
            # Fallback to core lookup
            return super().find_contract_file(step_name)
            
        except Exception as e:
            self.logger.error(f"Error finding workspace contract file for {step_name}: {e}")
            return None


class HybridFileResolverAdapter:
    """
    Adapter maintaining backward compatibility with HybridFileResolver.
    
    Replaces: src/cursus/validation/alignment/patterns/file_resolver.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
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
