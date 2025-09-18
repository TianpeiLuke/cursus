"""
Backward compatibility adapters for legacy discovery systems.

This module provides adapters that maintain existing APIs during the migration
from 16+ fragmented discovery systems to the unified StepCatalog system.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from .step_catalog import StepCatalog

logger = logging.getLogger(__name__)


class ContractDiscoveryResult:
    """
    Legacy result class for contract discovery operations.
    
    Maintains backward compatibility with existing tests and code that
    expect ContractDiscoveryResult objects from contract discovery operations.
    """
    
    def __init__(self, contract: Optional[Any] = None, contract_path: Optional[str] = None, 
                 error: Optional[str] = None, cached: bool = False):
        """
        Initialize contract discovery result.
        
        Args:
            contract: The discovered contract object (if any)
            contract_path: Path to the contract file
            error: Error message if discovery failed
            cached: Whether this result was retrieved from cache
        """
        self.contract = contract
        self.contract_path = contract_path
        self.error = error
        self.cached = cached
        self.success = contract is not None and error is None
    
    def __repr__(self) -> str:
        if self.success:
            return f"ContractDiscoveryResult(contract={self.contract}, path={self.contract_path}, cached={self.cached})"
        else:
            return f"ContractDiscoveryResult(error={self.error})"


class ContractDiscoveryEngineAdapter:
    """
    Adapter maintaining backward compatibility with ContractDiscoveryEngine.
    
    Replaces: src/cursus/validation/alignment/discovery/contract_discovery.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def discover_all_contracts(self) -> List[str]:
        """Legacy method: discover all contracts using unified catalog."""
        try:
            steps = self.catalog.list_available_steps()
            contracts = []
            
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('contract'):
                    contracts.append(step_name)
            
            self.logger.debug(f"Discovered {len(contracts)} contracts via unified catalog")
            return contracts
            
        except Exception as e:
            self.logger.error(f"Error discovering contracts: {e}")
            return []
    
    def discover_contracts_with_scripts(self) -> List[str]:
        """Legacy method: discover contracts that have associated scripts."""
        try:
            steps = self.catalog.list_available_steps()
            contracts_with_scripts = []
            
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if (step_info and 
                    step_info.file_components.get('contract') and 
                    step_info.file_components.get('script')):
                    contracts_with_scripts.append(step_name)
            
            self.logger.debug(f"Discovered {len(contracts_with_scripts)} contracts with scripts")
            return contracts_with_scripts
            
        except Exception as e:
            self.logger.error(f"Error discovering contracts with scripts: {e}")
            return []
    
    def extract_contract_reference_from_spec(self, spec_path: str) -> Optional[str]:
        """Legacy method: extract contract reference from specification."""
        try:
            # Use reverse lookup to find step, then get contract
            step_name = self.catalog.find_step_by_component(spec_path)
            if step_name:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('contract'):
                    return step_name
            return None
            
        except Exception as e:
            self.logger.error(f"Error extracting contract reference from {spec_path}: {e}")
            return None
    
    def build_entry_point_mapping(self) -> Dict[str, str]:
        """Legacy method: build mapping of entry points."""
        try:
            steps = self.catalog.list_available_steps()
            entry_point_mapping = {}
            
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.file_components.get('script'):
                    script_path = step_info.file_components['script'].path
                    entry_point_mapping[step_name] = str(script_path)
            
            return entry_point_mapping
            
        except Exception as e:
            self.logger.error(f"Error building entry point mapping: {e}")
            return {}


class ContractDiscoveryManagerAdapter:
    """
    Adapter maintaining backward compatibility with ContractDiscoveryManager.
    
    Replaces: src/cursus/validation/runtime/contract_discovery.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def discover_contract(self, step_name: str) -> Optional[str]:
        """Legacy method: discover contract for a specific step."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info and step_info.file_components.get('contract'):
                return str(step_info.file_components['contract'].path)
            return None
            
        except Exception as e:
            self.logger.error(f"Error discovering contract for {step_name}: {e}")
            return None
    
    def get_contract_input_paths(self, step_name: str) -> List[str]:
        """Legacy method: get contract input paths."""
        try:
            # This would require parsing the contract file, which is beyond
            # the scope of the unified catalog. Return empty list for now.
            self.logger.warning(f"get_contract_input_paths not fully implemented for {step_name}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting contract input paths for {step_name}: {e}")
            return []
    
    def get_contract_output_paths(self, step_name: str) -> List[str]:
        """Legacy method: get contract output paths."""
        try:
            # This would require parsing the contract file, which is beyond
            # the scope of the unified catalog. Return empty list for now.
            self.logger.warning(f"get_contract_output_paths not fully implemented for {step_name}")
            return []
            
        except Exception as e:
            self.logger.error(f"Error getting contract output paths for {step_name}: {e}")
            return []
    
    def _adapt_path_for_local_testing(self, path: str) -> str:
        """Legacy method: adapt path for local testing."""
        # Simple passthrough for now
        return path


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


class WorkspaceDiscoveryManagerAdapter:
    """
    Adapter maintaining backward compatibility with WorkspaceDiscoveryManager.
    
    Replaces: src/cursus/workspace/core/discovery.py
    """
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    def discover_workspaces(self) -> List[str]:
        """Legacy method: discover available workspaces."""
        try:
            # Get all workspace IDs from the catalog
            metrics = self.catalog.get_metrics_report()
            # For now, return the workspaces we know about
            return ["core"]  # Could be extended to discover actual workspace directories
            
        except Exception as e:
            self.logger.error(f"Error discovering workspaces: {e}")
            return []
    
    def discover_components(self, workspace_id: Optional[str] = None) -> List[str]:
        """Legacy method: discover components in workspace."""
        try:
            return self.catalog.list_available_steps(workspace_id=workspace_id)
            
        except Exception as e:
            self.logger.error(f"Error discovering components for workspace {workspace_id}: {e}")
            return []
    
    def resolve_cross_workspace_dependencies(self, step_name: str) -> Dict[str, Any]:
        """Legacy method: resolve cross-workspace dependencies."""
        try:
            step_info = self.catalog.get_step_info(step_name)
            if step_info:
                return {
                    'step_name': step_info.step_name,
                    'workspace_id': step_info.workspace_id,
                    'components': list(step_info.file_components.keys()),
                    'registry_data': step_info.registry_data
                }
            return {}
            
        except Exception as e:
            self.logger.error(f"Error resolving dependencies for {step_name}: {e}")
            return {}


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


class StepConfigResolverAdapter:
    """
    Enhanced adapter maintaining backward compatibility with StepConfigResolver.
    
    Replaces: src/cursus/core/compiler/config_resolver.py
    
    This enhanced version includes essential legacy methods needed for production
    while leveraging the unified step catalog for superior discovery capabilities.
    """
    
    # Job type keywords for matching (simplified from legacy)
    JOB_TYPE_KEYWORDS = {
        "training": ["training", "train"],
        "calibration": ["calibration", "calib"],
        "evaluation": ["evaluation", "eval", "test"],
        "inference": ["inference", "infer", "predict"],
    }
    
    # Pattern mappings for step type detection (from legacy)
    STEP_TYPE_PATTERNS = {
        r".*data_load.*": ["CradleDataLoading"],
        r".*preprocess.*": ["TabularPreprocessing"],
        r".*train.*": ["XGBoostTraining", "PyTorchTraining", "DummyTraining"],
        r".*eval.*": ["XGBoostModelEval"],
        r".*model.*": ["XGBoostModel", "PyTorchModel"],
        r".*calibrat.*": ["ModelCalibration"],
        r".*packag.*": ["MIMSPackaging"],
        r".*payload.*": ["MIMSPayload"],
        r".*regist.*": ["ModelRegistration"],
        r".*transform.*": ["BatchTransform"],
        r".*currency.*": ["CurrencyConversion"],
        r".*risk.*": ["RiskTableMapping"],
        r".*hyperparam.*": ["HyperparameterPrep"],
    }
    
    def __init__(self, workspace_root: Optional[Path] = None, confidence_threshold: float = 0.7):
        """Initialize with unified catalog."""
        if workspace_root is None:
            # Default to src/cursus/steps relative to the adapter's location
            adapter_dir = Path(__file__).parent  # src/cursus/step_catalog/
            workspace_root = adapter_dir.parent / 'steps'  # src/cursus/steps
        self.catalog = StepCatalog(workspace_root)
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
        self._metadata_mapping = {}
        self._config_cache = {}
    
    def resolve_config_map(self, dag_nodes: List[str], available_configs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced resolution with direct name matching + catalog fallback."""
        try:
            resolved_configs = {}
            
            for node_name in dag_nodes:
                # 1. Try direct name matching first (handles 99% of cases)
                config = self._direct_name_matching(node_name, available_configs)
                if config is not None:
                    resolved_configs[node_name] = config
                    continue
                
                # 2. Fallback to catalog-based matching
                step_info = self.catalog.get_step_info(node_name)
                if step_info and step_info.config_class:
                    # Find config instance that matches the step's config class
                    for config_name, config_instance in available_configs.items():
                        if type(config_instance).__name__ == step_info.config_class:
                            resolved_configs[node_name] = config_instance
                            break
                else:
                    # 3. Last resort: first available config
                    if available_configs:
                        resolved_configs[node_name] = next(iter(available_configs.values()))
            
            return resolved_configs
            
        except Exception as e:
            self.logger.error(f"Error resolving config map: {e}")
            return {}
    
    def _direct_name_matching(self, node_name: str, configs: Dict[str, Any]) -> Optional[Any]:
        """
        Enhanced direct name matching with metadata support.
        
        Based on legacy implementation that supports metadata.config_types mapping.
        """
        # First priority: Direct match with config key
        if node_name in configs:
            self.logger.info(f"Found exact key match for node '{node_name}'")
            return configs[node_name]

        # Second priority: Check metadata.config_types mapping if available
        if self._metadata_mapping and node_name in self._metadata_mapping:
            config_class_name = self._metadata_mapping[node_name]

            # Find configs of the specified class
            for config_name, config in configs.items():
                if type(config).__name__ == config_class_name:
                    # If job type is part of the node name, check for match
                    if "_" in node_name:
                        node_parts = node_name.split("_")
                        if len(node_parts) > 1:
                            job_type = node_parts[-1].lower()
                            if (
                                hasattr(config, "job_type")
                                and getattr(config, "job_type", "").lower() == job_type
                            ):
                                self.logger.info(
                                    f"Found metadata mapping match with job type for node '{node_name}'"
                                )
                                return config
                    else:
                        self.logger.info(
                            f"Found metadata mapping match for node '{node_name}'"
                        )
                        return config

        # Case-insensitive match as fallback
        node_lower = node_name.lower()
        for config_name, config in configs.items():
            if config_name.lower() == node_lower:
                self.logger.info(
                    f"Found case-insensitive match for node '{node_name}': {config_name}"
                )
                return config

        return None
    
    def _job_type_matching(self, node_name: str, configs: Dict[str, Any]) -> List[tuple]:
        """
        Match based on job_type attribute and node naming patterns.
        
        Based on legacy implementation from StepConfigResolver.
        
        Args:
            node_name: DAG node name
            configs: Available configurations

        Returns:
            List of (config, confidence, method) tuples
        """
        matches: List[tuple] = []
        node_lower = node_name.lower()

        # Extract potential job type from node name (from legacy JOB_TYPE_KEYWORDS)
        detected_job_type = None
        for job_type, keywords in self.JOB_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in node_lower:
                    detected_job_type = job_type
                    break
            if detected_job_type:
                break

        if not detected_job_type:
            return matches

        # Find configs with matching job_type (legacy logic)
        for config_name, config in configs.items():
            if hasattr(config, "job_type"):
                config_job_type = getattr(config, "job_type", "").lower()

                # Check for job type match (legacy logic)
                job_type_keywords = self.JOB_TYPE_KEYWORDS.get(detected_job_type, [])
                if any(keyword in config_job_type for keyword in job_type_keywords):
                    # Calculate confidence based on how well the node name matches the config type
                    config_type_confidence = self._calculate_config_type_confidence(
                        node_name, config
                    )
                    total_confidence = 0.7 + (
                        config_type_confidence * 0.3
                    )  # Job type match + config type match
                    matches.append((config, total_confidence, "job_type"))

        return matches
    
    def _calculate_config_type_confidence(self, node_name: str, config: Any) -> float:
        """
        Calculate confidence based on how well node name matches config type.
        
        From legacy implementation.

        Args:
            node_name: DAG node name
            config: Configuration instance

        Returns:
            Confidence score (0.0 to 1.0)
        """
        config_type = type(config).__name__.lower()
        node_lower = node_name.lower()

        # Remove common suffixes for comparison
        config_base = config_type.replace("config", "").replace("step", "")

        # Check for substring matches
        if config_base in node_lower or any(
            part in node_lower for part in config_base.split("_")
        ):
            return 0.8

        # Use sequence matching for similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, node_lower, config_base).ratio()
        return similarity
    
    def _semantic_matching(self, node_name: str, configs: Dict[str, Any]) -> List[tuple]:
        """Semantic matching based on keywords."""
        matches = []
        
        # Define semantic keywords
        semantic_map = {
            'data': ['loading', 'load', 'cradle'],
            'preprocess': ['preprocessing', 'tabular', 'process'],
            'train': ['training', 'xgboost', 'pytorch', 'model'],
            'evaluate': ['evaluation', 'eval', 'test'],
            'transform': ['transformation', 'batch']
        }
        
        node_lower = node_name.lower()
        
        for config_key, config_instance in configs.items():
            config_lower = config_key.lower()
            confidence = 0.0
            
            # Check for semantic matches
            for category, keywords in semantic_map.items():
                if any(keyword in node_lower for keyword in keywords):
                    if any(keyword in config_lower for keyword in keywords):
                        confidence = 0.7
                        break
            
            if confidence > 0:
                matches.append((config_instance, confidence, "semantic"))
        
        return matches
    
    def _pattern_matching(self, node_name: str, configs: Dict[str, Any]) -> List[tuple]:
        """
        Use regex patterns to match node names to config types.
        
        Based on legacy implementation from StepConfigResolver.

        Args:
            node_name: DAG node name
            configs: Available configurations

        Returns:
            List of (config, confidence, method) tuples
        """
        matches: List[tuple] = []
        node_lower = node_name.lower()

        # Find matching patterns (from legacy STEP_TYPE_PATTERNS)
        matching_step_types = []
        for pattern, step_types in self.STEP_TYPE_PATTERNS.items():
            import re
            if re.match(pattern, node_lower):
                matching_step_types.extend(step_types)

        if not matching_step_types:
            return matches

        # Find configs that match the detected step types (legacy logic)
        for config_name, config in configs.items():
            config_type = type(config).__name__

            # Convert config class name to step type (legacy logic)
            step_type = self._config_class_to_step_type(config_type)

            if step_type in matching_step_types:
                # Base confidence for pattern match
                confidence = 0.6

                # Boost confidence if there are additional matches (legacy logic)
                if hasattr(config, "job_type"):
                    job_type_boost = self._calculate_job_type_boost(node_name, config)
                    confidence += job_type_boost * 0.2

                matches.append((config, min(confidence, 0.9), "pattern"))

        return matches
    
    def _config_class_to_step_type(self, config_class_name: str) -> str:
        """
        Convert configuration class name to step type using step catalog system.
        
        Enhanced to use step catalog's registry data for accurate mapping.

        Args:
            config_class_name: Configuration class name

        Returns:
            Step type name
        """
        try:
            # First, try to find the step type using the step catalog
            steps = self.catalog.list_available_steps()
            for step_name in steps:
                step_info = self.catalog.get_step_info(step_name)
                if step_info and step_info.config_class == config_class_name:
                    # Use the step name from the catalog as the step type
                    return step_name
                
                # Also check registry data for builder_step_name
                if step_info and step_info.registry_data.get('builder_step_name'):
                    builder_step_name = step_info.registry_data['builder_step_name']
                    if f"{builder_step_name}Config" == config_class_name:
                        return builder_step_name
            
            # Fallback to legacy logic if not found in catalog
            step_type = config_class_name

            # Remove 'Config' suffix
            if step_type.endswith("Config"):
                step_type = step_type[:-6]

            # Remove 'Step' suffix if present
            if step_type.endswith("Step"):
                step_type = step_type[:-4]

            # Handle special cases (legacy logic)
            if step_type == "CradleDataLoad":
                return "CradleDataLoading"
            elif step_type == "PackageStep" or step_type == "Package":
                return "MIMSPackaging"

            return step_type
            
        except Exception as e:
            self.logger.debug(f"Error using catalog for config class mapping, falling back to legacy: {e}")
            
            # Pure legacy fallback
            step_type = config_class_name
            if step_type.endswith("Config"):
                step_type = step_type[:-6]
            if step_type.endswith("Step"):
                step_type = step_type[:-4]
            if step_type == "CradleDataLoad":
                return "CradleDataLoading"
            elif step_type == "PackageStep" or step_type == "Package":
                return "MIMSPackaging"
            return step_type
    
    def _calculate_job_type_boost(self, node_name: str, config: Any) -> float:
        """
        Calculate confidence boost based on job type matching.
        
        From legacy implementation.

        Args:
            node_name: DAG node name
            config: Configuration instance

        Returns:
            Boost score (0.0 to 1.0)
        """
        if not hasattr(config, "job_type"):
            return 0.0

        config_job_type = getattr(config, "job_type", "").lower()
        node_lower = node_name.lower()

        # Check for job type keywords in node name (legacy logic)
        for job_type, keywords in self.JOB_TYPE_KEYWORDS.items():
            if any(keyword in config_job_type for keyword in keywords):
                if any(keyword in node_lower for keyword in keywords):
                    return 1.0

        return 0.0
    
    def _resolve_single_node(self, node_name: str, configs: Dict[str, Any]) -> tuple:
        """Resolve a single node using all matching strategies."""
        # Try direct matching first
        direct_match = self._direct_name_matching(node_name, configs)
        if direct_match is not None:
            return (direct_match, 1.0, "direct_name")
        
        # Collect all matches from different strategies
        all_matches = []
        
        # Job type matching
        job_matches = self._job_type_matching(node_name, configs)
        all_matches.extend(job_matches)
        
        # Semantic matching
        semantic_matches = self._semantic_matching(node_name, configs)
        all_matches.extend(semantic_matches)
        
        # Pattern matching
        pattern_matches = self._pattern_matching(node_name, configs)
        all_matches.extend(pattern_matches)
        
        if not all_matches:
            try:
                from ..core.compiler.exceptions import ResolutionError
                raise ResolutionError(f"No configuration found for node: {node_name}")
            except ImportError:
                # Fallback if ResolutionError is not available
                raise ValueError(f"No configuration found for node: {node_name}")
        
        # Return the highest confidence match
        best_match = max(all_matches, key=lambda x: x[1])
        return best_match
    
    def _parse_node_name(self, node_name: str) -> Dict[str, str]:
        """
        Parse node name to extract config type and job type information.
        
        Based on legacy implementation patterns from the original StepConfigResolver.
        
        Args:
            node_name: DAG node name
            
        Returns:
            Dictionary with extracted information
        """
        # Check cache first
        if node_name in self._config_cache:
            from typing import cast, Dict
            return cast(Dict[str, str], self._config_cache[node_name])

        result = {}

        # Common patterns from legacy implementation
        patterns = [
            # Pattern 1: ConfigType_JobType (e.g., CradleDataLoading_training)
            (r"^([A-Za-z]+[A-Za-z0-9]*)_([a-z]+)$", "config_first"),
            # Pattern 2: JobType_Task (e.g., training_data_load)
            (r"^([a-z]+)_([A-Za-z_]+)$", "job_first"),
        ]

        import re
        for pattern, pattern_type in patterns:
            match = re.match(pattern, node_name)
            if match:
                parts = match.groups()

                if pattern_type == "config_first":  # ConfigType_JobType
                    result["config_type"] = parts[0]
                    result["job_type"] = parts[1]
                else:  # JobType_Task
                    result["job_type"] = parts[0]

                    # Try to infer config type from task (from legacy task_map)
                    task_map = {
                        "data_load": "CradleDataLoading",
                        "preprocess": "TabularPreprocessing",
                        "train": "XGBoostTraining",
                        "eval": "XGBoostModelEval",
                        "calibrat": "ModelCalibration",
                        "packag": "Package",
                        "regist": "Registration",
                        "payload": "Payload",
                    }

                    for task_pattern, config_type in task_map.items():
                        if task_pattern in parts[1]:
                            result["config_type"] = config_type
                            break

                break

        # Cache the result
        self._config_cache[node_name] = result
        return result
    
    def _job_type_matching_enhanced(self, job_type: str, configs: Dict[str, Any], config_type: Optional[str] = None) -> List[tuple]:
        """
        Enhanced job type matching with config type filtering.
        
        Args:
            job_type: Job type string (e.g., "training", "calibration")
            configs: Available configurations
            config_type: Optional config type to filter by
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        normalized_job_type = job_type.lower()
        
        for config_name, config in configs.items():
            if hasattr(config, "job_type"):
                config_job_type = getattr(config, "job_type", "").lower()
                
                # Skip if job types don't match
                if config_job_type != normalized_job_type:
                    continue
                
                # Start with base confidence for job type match
                base_confidence = 0.8
                
                # If config_type is specified, check for match to boost confidence
                if config_type:
                    config_class_name = type(config).__name__
                    config_type_lower = config_type.lower()
                    class_name_lower = config_class_name.lower()
                    
                    # Different levels of match for config type
                    if config_class_name == config_type:
                        # Exact match
                        base_confidence = 0.9
                    elif (config_type_lower in class_name_lower or class_name_lower in config_type_lower):
                        # Partial match
                        base_confidence = 0.85
                
                matches.append((config, base_confidence, "job_type_enhanced"))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def resolve_config_for_step(self, step_name: str, configs: Dict[str, Any]) -> Optional[Any]:
        """
        Resolve configuration for a single step (used by generator.py).
        
        Args:
            step_name: Name of the step
            configs: Available configurations
            
        Returns:
            Resolved configuration or None
        """
        try:
            # Try direct name matching first
            config = self._direct_name_matching(step_name, configs)
            if config is not None:
                return config
            
            # Try enhanced resolution
            resolved_tuple = self._resolve_single_node(step_name, configs)
            if resolved_tuple:
                return resolved_tuple[0]  # Return just the config, not the tuple
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error resolving config for step {step_name}: {e}")
            return None
    
    def preview_resolution(self, dag_nodes: List[str], available_configs: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Enhanced preview resolution with metadata support."""
        try:
            # Extract metadata.config_types mapping if available
            self._metadata_mapping = {}
            if metadata and "config_types" in metadata:
                self._metadata_mapping = metadata["config_types"]
                self.logger.info(f"Using metadata.config_types mapping with {len(self._metadata_mapping)} entries")
            
            node_resolution = {}
            resolution_confidence = {}
            node_config_map = {}
            
            for node in dag_nodes:
                try:
                    # Try to resolve the node
                    config, confidence, method = self._resolve_single_node(node, available_configs)
                    
                    # Store resolution info
                    node_resolution[node] = {
                        "config_type": type(config).__name__,
                        "confidence": confidence,
                        "method": method,
                        "job_type": getattr(config, "job_type", "N/A"),
                    }
                    
                    resolution_confidence[node] = confidence
                    node_config_map[node] = type(config).__name__
                    
                except Exception as e:
                    # Store error information
                    node_resolution[node] = {
                        "error": f"Step not found in catalog: {node}",
                        "error_type": "ResolutionError",
                    }
                    resolution_confidence[node] = 0.0
                    node_config_map[node] = "Unknown"
            
            return {
                "node_resolution": node_resolution,
                "resolution_confidence": resolution_confidence,
                "node_config_map": node_config_map,
                "metadata_mapping": self._metadata_mapping,
                "recommendations": [],
            }
            
        except Exception as e:
            self.logger.error(f"Error previewing resolution: {e}")
            return {"error": str(e)}


class ConfigClassDetectorAdapter:
    """
    Adapter maintaining backward compatibility with ConfigClassDetector.
    
    Replaces: src/cursus/core/config_fields/config_class_detector.py
    
    MODERN APPROACH: Uses step catalog's superior AST-based config discovery
    instead of legacy JSON parsing. This provides more accurate and comprehensive
    config class detection.
    """
    
    # Constants for backward compatibility (minimal legacy support)
    MODEL_TYPE_FIELD = "__model_type__"
    METADATA_FIELD = "metadata"
    CONFIG_TYPES_FIELD = "config_types"
    CONFIGURATION_FIELD = "configuration"
    SPECIFIC_FIELD = "specific"
    
    # REMOVED: ESSENTIAL_CLASSES - analysis shows this was never used in actual code
    # The concept was purely theoretical and added unnecessary complexity
    
    def __init__(self, workspace_root: Path):
        """Initialize with unified catalog."""
        self.catalog = StepCatalog(workspace_root)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def detect_from_json(config_path: str) -> Dict[str, Any]:
        """
        MODERN APPROACH: Use step catalog's AST-based discovery instead of JSON parsing.
        
        This method now uses the superior AST-based discovery from the unified step catalog
        rather than the legacy JSON parsing approach. This provides more accurate and 
        comprehensive config class detection by analyzing actual source code.
        
        Real usage pattern (from dynamic_template.py):
        detected_classes = detect_config_classes_from_json(self.config_path)
        
        Args:
            config_path: Path to configuration file (used to determine workspace root)
            
        Returns:
            Dictionary mapping config class names to config class types
        """
        try:
            from pathlib import Path
            
            # Determine workspace root from config path
            config_file = Path(config_path)
            workspace_root = config_file.parent.parent  # Go up to find workspace root
            
            # Use step catalog's superior AST-based discovery
            catalog = StepCatalog(workspace_root)
            
            # Get complete config classes (AST discovery + ConfigClassStore integration)
            config_classes = catalog.build_complete_config_classes()
            
            logger = logging.getLogger(__name__)
            logger.info(f"Discovered {len(config_classes)} config classes via unified step catalog (AST-based)")
            
            return config_classes
            
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error detecting config classes via step catalog: {e}")
            
            # Fallback to ConfigClassStore if available
            try:
                from ..core.config_fields.config_class_store import ConfigClassStore
                fallback_classes = ConfigClassStore.get_all_classes()
                logger.info(f"Fallback: Using {len(fallback_classes)} classes from ConfigClassStore")
                return fallback_classes
            except ImportError:
                logger.warning("ConfigClassStore not available, returning empty dict")
                return {}
    
    @staticmethod
    def _extract_class_names(config_data: Dict[str, Any], logger) -> set:
        """
        LEGACY COMPATIBILITY: Minimal JSON parsing for backward compatibility only.
        
        This method is maintained for any legacy code that might still use it,
        but new code should use the step catalog's AST-based discovery instead.
        
        NOTE: Analysis shows this is only used in tests, not in production code.
        """
        class_names = set()
        
        try:
            # Extract from metadata.config_types (legacy pattern)
            if ConfigClassDetectorAdapter.METADATA_FIELD in config_data:
                metadata = config_data[ConfigClassDetectorAdapter.METADATA_FIELD]
                if isinstance(metadata, dict) and ConfigClassDetectorAdapter.CONFIG_TYPES_FIELD in metadata:
                    config_types = metadata[ConfigClassDetectorAdapter.CONFIG_TYPES_FIELD]
                    if isinstance(config_types, dict):
                        class_names.update(config_types.values())
            
            # Extract from configuration.specific sections (legacy pattern)
            if ConfigClassDetectorAdapter.CONFIGURATION_FIELD in config_data:
                configuration = config_data[ConfigClassDetectorAdapter.CONFIGURATION_FIELD]
                if isinstance(configuration, dict) and ConfigClassDetectorAdapter.SPECIFIC_FIELD in configuration:
                    specific = configuration[ConfigClassDetectorAdapter.SPECIFIC_FIELD]
                    if isinstance(specific, dict):
                        for step_config in specific.values():
                            if isinstance(step_config, dict) and ConfigClassDetectorAdapter.MODEL_TYPE_FIELD in step_config:
                                class_names.add(step_config[ConfigClassDetectorAdapter.MODEL_TYPE_FIELD])
            
            logger.debug(f"Legacy JSON parsing extracted {len(class_names)} class names")
            return class_names
            
        except Exception as e:
            logger.error(f"Error in legacy JSON parsing: {e}")
            return set()
    
    @classmethod
    def from_config_store(cls, config_path: str) -> Dict[str, Any]:
        """
        MODERN APPROACH: Use step catalog's integrated discovery.
        
        This method uses the step catalog's build_complete_config_classes which
        integrates both AST-based discovery and ConfigClassStore registration,
        providing the most comprehensive config class detection.
        """
        return cls.detect_from_json(config_path)


class ConfigClassStoreAdapter:
    """
    Adapter maintaining backward compatibility with ConfigClassStore.
    
    Replaces: src/cursus/core/config_fields/config_class_store.py (partial)
    """
    
    # Single registry instance - implementing Single Source of Truth
    _registry: Dict[str, Any] = {}
    _logger = logging.getLogger(__name__)

    @classmethod
    def register(cls, config_class: Optional[Any] = None) -> Any:
        """Legacy method: register a config class."""
        def _register(cls_to_register: Any) -> Any:
            cls_name = cls_to_register.__name__
            if (
                cls_name in cls._registry
                and cls._registry[cls_name] != cls_to_register
            ):
                cls._logger.warning(
                    f"Class {cls_name} is already registered and is being overwritten."
                )
            cls._registry[cls_name] = cls_to_register
            cls._logger.debug(f"Registered class: {cls_name}")
            return cls_to_register

        if config_class is not None:
            return _register(config_class)
        return _register

    @classmethod
    def get_class(cls, class_name: str) -> Optional[Any]:
        """Legacy method: get a registered class by name."""
        class_obj = cls._registry.get(class_name)
        if class_obj is None:
            cls._logger.debug(f"Class not found in registry: {class_name}")
        return class_obj

    @classmethod
    def get_all_classes(cls) -> Dict[str, Any]:
        """Legacy method: get all registered classes."""
        return cls._registry.copy()

    @classmethod
    def register_many(cls, *config_classes: Any) -> None:
        """Legacy method: register multiple config classes at once."""
        for config_class in config_classes:
            cls.register(config_class)

    @classmethod
    def clear(cls) -> None:
        """Legacy method: clear the registry."""
        cls._registry.clear()
        cls._logger.debug("Cleared config class registry")

    @classmethod
    def registered_names(cls) -> set:
        """Legacy method: get all registered class names."""
        return set(cls._registry.keys())


# Legacy function for backward compatibility
def build_complete_config_classes(project_id: Optional[str] = None) -> Dict[str, Any]:
    """Legacy function: build complete config classes using catalog."""
    try:
        # Use a default workspace root for catalog initialization
        from pathlib import Path
        workspace_root = Path('.')
        catalog = StepCatalog(workspace_root)
        
        # Use catalog's build_complete_config_classes method
        config_classes = catalog.build_complete_config_classes(project_id)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Built {len(config_classes)} complete config classes via unified catalog")
        
        return config_classes
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error building complete config classes: {e}")
        
        # Fallback to registered classes
        return ConfigClassStoreAdapter.get_all_classes()


def detect_config_classes_from_json(config_path: str) -> Dict[str, Any]:
    """Legacy function: detect config classes using catalog."""
    return ConfigClassDetectorAdapter.detect_from_json(config_path)


# Legacy wrapper for backward compatibility during migration
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
        self.catalog = StepCatalog(workspace_root)
        
        # Expose config_discovery for compatibility
        self.config_discovery = self.catalog.config_discovery
        
        # Initialize all adapters
        self.contract_discovery_engine = ContractDiscoveryEngineAdapter(workspace_root)
        self.contract_discovery_manager = ContractDiscoveryManagerAdapter(workspace_root)
        self.flexible_file_resolver = FlexibleFileResolverAdapter(workspace_root)
        self.workspace_discovery_manager = WorkspaceDiscoveryManagerAdapter(workspace_root)
        self.hybrid_file_resolver = HybridFileResolverAdapter(workspace_root)
        
        self.logger = logging.getLogger(__name__)
    
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
