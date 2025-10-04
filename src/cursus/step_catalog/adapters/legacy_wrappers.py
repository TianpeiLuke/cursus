"""
Legacy wrapper adapters for backward compatibility.

This module provides legacy wrapper classes and functions that maintain existing APIs
during the migration from legacy discovery systems to the unified StepCatalog system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from ..step_catalog import StepCatalog
from .contract_adapter import ContractDiscoveryEngineAdapter, ContractDiscoveryManagerAdapter
from .file_resolver import FlexibleFileResolverAdapter, HybridFileResolverAdapter
from .workspace_discovery import WorkspaceDiscoveryManagerAdapter
from .config_class_detector import ConfigClassDetectorAdapter, ConfigClassStoreAdapter

logger = logging.getLogger(__name__)


class LegacyDiscoveryWrapper:
    """
    Wrapper providing legacy discovery interfaces during migration period.
    
    This class provides a unified interface that can be used as a drop-in
    replacement for legacy discovery systems during the migration phase.
    It delegates all StepCatalog methods to the underlying catalog while
    also providing access to legacy adapters.
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with all legacy adapters."""
        self.workspace_root = workspace_root
        # PORTABLE: Use workspace-aware discovery for legacy wrapper
        self.catalog = StepCatalog(workspace_dirs=[workspace_root])
        
        # Expose config_discovery for compatibility
        self.config_discovery = self.catalog.config_discovery
        
        # Initialize all adapters
        self.contract_discovery_engine = ContractDiscoveryEngineAdapter(workspace_root)
        self.contract_discovery_manager = ContractDiscoveryManagerAdapter(workspace_root)
        self.flexible_file_resolver = FlexibleFileResolverAdapter(workspace_root)
        self.workspace_discovery_manager = WorkspaceDiscoveryManagerAdapter(workspace_root)
        self.hybrid_file_resolver = HybridFileResolverAdapter(workspace_root)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize file cache
        self._refresh_cache()
    
    def _refresh_cache(self):
        """Refresh file cache using step catalog discovery."""
        # Always initialize file_cache, even if there's an error
        self.file_cache = {
            "scripts": {},
            "contracts": {},
            "specs": {},
            "builders": {},
            "configs": {}
        }
        
        try:
            steps = self.catalog.list_available_steps()
            
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
        snake_case = re.sub('([A-Z])', r'_\1', step_name).lower().strip('_')
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
    
    # Delegate all StepCatalog methods to the underlying catalog
    def get_step_info(self, step_name: str, job_type: Optional[str] = None):
        """Delegate to underlying StepCatalog."""
        return self.catalog.get_step_info(step_name, job_type)
    
    def find_step_by_component(self, component_path: str) -> Optional[str]:
        """Delegate to underlying StepCatalog."""
        return self.catalog.find_step_by_component(component_path)
    
    def list_available_steps(self, workspace_id: Optional[str] = None, job_type: Optional[str] = None) -> List[str]:
        """Delegate to underlying StepCatalog."""
        return self.catalog.list_available_steps(workspace_id, job_type)
    
    def search_steps(self, query: str, job_type: Optional[str] = None):
        """Delegate to underlying StepCatalog."""
        return self.catalog.search_steps(query, job_type)
    
    def discover_config_classes(self, project_id: Optional[str] = None):
        """Delegate to underlying StepCatalog."""
        return self.catalog.discover_config_classes(project_id)
    
    def build_complete_config_classes(self, project_id: Optional[str] = None):
        """Delegate to underlying StepCatalog."""
        return self.catalog.build_complete_config_classes(project_id)
    
    def get_job_type_variants(self, step_name: str) -> List[str]:
        """Delegate to underlying StepCatalog."""
        return self.catalog.get_job_type_variants(step_name)
    
    def get_metrics_report(self):
        """Delegate to underlying StepCatalog."""
        return self.catalog.get_metrics_report()
    
    # Expanded discovery methods (Phase 4.1)
    def discover_contracts_with_scripts(self) -> List[str]:
        """Delegate to underlying StepCatalog."""
        return self.catalog.discover_contracts_with_scripts()
    
    def detect_framework(self, step_name: str) -> Optional[str]:
        """Delegate to underlying StepCatalog."""
        return self.catalog.detect_framework(step_name)
    
    def discover_cross_workspace_components(self, workspace_ids: Optional[List[str]] = None):
        """Delegate to underlying StepCatalog."""
        return self.catalog.discover_cross_workspace_components(workspace_ids)
    
    def get_builder_class_path(self, step_name: str) -> Optional[str]:
        """Delegate to underlying StepCatalog."""
        return self.catalog.get_builder_class_path(step_name)
    
    def load_builder_class(self, step_name: str):
        """Delegate to underlying StepCatalog."""
        return self.catalog.load_builder_class(step_name)
    
    def get_adapter(self, adapter_type: str) -> Any:
        """Get specific legacy adapter by type."""
        adapters = {
            'contract_discovery_engine': self.contract_discovery_engine,
            'contract_discovery_manager': self.contract_discovery_manager,
            'flexible_file_resolver': self.flexible_file_resolver,
            'workspace_discovery_manager': self.workspace_discovery_manager,
            'hybrid_file_resolver': self.hybrid_file_resolver,
        }
        
        return adapters.get(adapter_type)
    
    def get_unified_catalog(self) -> StepCatalog:
        """Get the underlying unified catalog."""
        return self.catalog


# Legacy function for backward compatibility
def build_complete_config_classes(project_id: Optional[str] = None) -> Dict[str, Any]:
    """Legacy function: build complete config classes using catalog."""
    try:
        # PORTABLE: Use package-only discovery for legacy function
        catalog = StepCatalog(workspace_dirs=None)
        
        # Use catalog's build_complete_config_classes method
        config_classes = catalog.build_complete_config_classes(project_id)
        
        logger.info(f"Built {len(config_classes)} complete config classes via unified catalog")
        
        return config_classes
        
    except Exception as e:
        logger.error(f"Error building complete config classes: {e}")
        
        # Fallback to registered classes
        return ConfigClassStoreAdapter.get_all_classes()


def detect_config_classes_from_json(config_path: str) -> Dict[str, Any]:
    """Legacy function: detect config classes using catalog."""
    return ConfigClassDetectorAdapter.detect_from_json(config_path)
