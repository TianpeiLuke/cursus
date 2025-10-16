"""
Unified Step Catalog - Single class addressing all US1-US5 requirements.

This module implements the core StepCatalog class that consolidates 16+ fragmented
discovery mechanisms into a single, efficient system with O(1) lookups and
intelligent component discovery across multiple workspaces.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Union

from .models import StepInfo, FileMetadata, StepSearchResult
from .mapping import StepCatalogMapper, PipelineConstructionInterface

# Type hints for discovery components - all handled symmetrically
try:
    from .config_discovery import ConfigAutoDiscovery
except ImportError:
    ConfigAutoDiscovery = None

try:
    from .builder_discovery import BuilderAutoDiscovery
except ImportError:
    BuilderAutoDiscovery = None

try:
    from .contract_discovery import ContractAutoDiscovery
except ImportError:
    ContractAutoDiscovery = None

try:
    from .spec_discovery import SpecAutoDiscovery
except ImportError:
    SpecAutoDiscovery = None

try:
    from .script_discovery import ScriptAutoDiscovery
except ImportError:
    ScriptAutoDiscovery = None

logger = logging.getLogger(__name__)


class StepCatalog:
    """
    Unified step catalog addressing all validated user stories (US1-US5).
    
    This single class consolidates the functionality of 16+ discovery systems
    while maintaining simple, efficient O(1) lookups through dictionary-based indexing.
    
    PHASE 1 ENHANCEMENT: Now includes StepBuilderRegistry functionality:
    - Config-to-builder resolution
    - Legacy alias support  
    - Pipeline construction interface
    - Enhanced registry integration
    """
    
    # Legacy aliases for backward compatibility (moved from StepBuilderRegistry)
    LEGACY_ALIASES = {
        "MIMSPackaging": "Package",  # Legacy name from before standardization
        "MIMSPayload": "Payload",  # Legacy name from before standardization
        "ModelRegistration": "Registration",  # Legacy name from before standardization
        "PytorchTraining": "PyTorchTraining",  # Case sensitivity difference
        "PytorchModel": "PyTorchModel",  # Case sensitivity difference
    }
    
    def __init__(self, workspace_dirs: Optional[Union[Path, List[Path]]] = None):
        """
        Initialize the unified step catalog with optional workspace directories.
        
        Args:
            workspace_dirs: Optional workspace directory(ies) for workspace-aware discovery.
                           Can be a single Path or list of Paths.
                           Each should point directly to a directory containing scripts/, contracts/, 
                           specs/, builders/, configs/ subdirectories.
                           If None, only discovers package components.
        
        Examples:
            # Package-only discovery (works in all deployment scenarios)
            catalog = StepCatalog()
            
            # Single workspace directory (points directly to steps directory)
            catalog = StepCatalog(workspace_dirs=Path("/path/to/my_workspace_steps"))
            
            # Multiple workspace directories
            catalog = StepCatalog(workspace_dirs=[
                Path("/workspace1/steps"), Path("/workspace2/steps")
            ])
        """
        # Initialize logger first (needed by discovery components)
        self.logger = logging.getLogger(__name__)
        
        # Find package root using relative path (deployment agnostic)
        self.package_root = self._find_package_root()
        
        # Normalize workspace_dirs to list
        self.workspace_dirs = self._normalize_workspace_dirs(workspace_dirs)
        
        # Initialize specialized discovery components (consistent architecture)
        self.config_discovery = self._initialize_config_discovery()
        self.builder_discovery = self._initialize_builder_discovery()
        self.contract_discovery = self._initialize_contract_discovery()
        self.spec_discovery = self._initialize_spec_discovery()
        self.script_discovery = self._initialize_script_discovery()
        
        # Simple in-memory indexes (US4: Efficient Scaling)
        self._step_index: Dict[str, StepInfo] = {}
        self._component_index: Dict[Path, str] = {}
        self._workspace_steps: Dict[str, List[str]] = {}
        self._index_built = False
        
        # Simple caches for expanded functionality (avoid over-engineering)
        self._framework_cache: Dict[str, str] = {}
        self._validation_metadata_cache: Dict[str, Any] = {}
        self._builder_class_cache: Dict[str, Type] = {}
        
        # Simple metrics collection
        self.metrics: Dict[str, Any] = {
            'queries': 0,
            'errors': 0,
            'avg_response_time': 0.0,
            'index_build_time': 0.0,
            'last_index_build': None
        }
        
        # PHASE 1 ENHANCEMENT: Initialize mapping components
        self.mapper = StepCatalogMapper(self)
        
        # Initialize pipeline_interface with error handling
        try:
            self.pipeline_interface = PipelineConstructionInterface(self.mapper)
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline_interface: {e}")
            self.pipeline_interface = None
    
    # US1: Query by Step Name
    def get_step_info(self, step_name: str, job_type: Optional[str] = None) -> Optional[StepInfo]:
        """
        Get complete information about a step, optionally with job_type variant.
        
        Args:
            step_name: Name of the step to retrieve
            job_type: Optional job type variant (e.g., 'training', 'validation')
            
        Returns:
            StepInfo object with complete step information, or None if not found
        """
        start_time = time.time()
        self.metrics['queries'] += 1
        
        try:
            self._ensure_index_built()
            
            # Handle job_type variants
            search_key = f"{step_name}_{job_type}" if job_type else step_name
            result = self._step_index.get(search_key) or self._step_index.get(step_name)
            
            return result
            
        except Exception as e:
            self.metrics['errors'] += 1
            self.logger.error(f"Error retrieving step info for {step_name}: {e}")
            return None
            
        finally:
            # Update response time metrics
            response_time = time.time() - start_time
            total_queries = int(self.metrics['queries'])
            current_avg = float(self.metrics['avg_response_time'])
            self.metrics['avg_response_time'] = (
                (current_avg * (total_queries - 1) + response_time) 
                / total_queries
            )
    
    # US2: Reverse Lookup from Components
    def find_step_by_component(self, component_path: str) -> Optional[str]:
        """
        Find step name from any component file.
        
        Args:
            component_path: Path to a component file
            
        Returns:
            Step name that owns the component, or None if not found
        """
        try:
            self._ensure_index_built()
            return self._component_index.get(Path(component_path))
        except Exception as e:
            self.logger.error(f"Error finding step for component {component_path}: {e}")
            return None
    
    # US3: Multi-Workspace Discovery
    def list_available_steps(self, workspace_id: Optional[str] = None, 
                           job_type: Optional[str] = None) -> List[str]:
        """
        List all available concrete pipeline steps with deduplication.
        
        Excludes base configuration steps ('Base', 'Processing') and applies
        canonical name deduplication following standardization rules.
        
        Args:
            workspace_id: Optional workspace filter
            job_type: Optional job type filter
            
        Returns:
            List of concrete pipeline step names (PascalCase, canonical)
        """
        try:
            self._ensure_index_built()
            
            if workspace_id:
                steps = self._workspace_steps.get(workspace_id, [])
            else:
                steps = list(self._step_index.keys())
            
            # DEDUPLICATION: Apply canonical name resolution and base config exclusion
            canonical_steps = self._deduplicate_and_filter_concrete_steps(steps)
            
            if job_type:
                # Filter steps by job type
                filtered_steps = []
                for step in canonical_steps:
                    if step.endswith(f"_{job_type}") or job_type == "default":
                        filtered_steps.append(step)
                canonical_steps = filtered_steps
            
            return canonical_steps
            
        except Exception as e:
            self.logger.error(f"Error listing steps for workspace {workspace_id}: {e}")
            return []
    
    def list_steps_with_specs(self, workspace_id: Optional[str] = None, 
                            job_type: Optional[str] = None) -> List[str]:
        """
        List all steps that have specification components.
        
        This method filters available steps to only return those that have
        specification file components, which is useful for validation frameworks
        that need to work specifically with steps that have specifications.
        
        Args:
            workspace_id: Optional workspace filter
            job_type: Optional job type filter
            
        Returns:
            List of step names that have specification components
        """
        try:
            # Get all available steps with optional filtering
            available_steps = self.list_available_steps(workspace_id, job_type)
            
            # Filter to only steps that have specification components
            steps_with_specs = []
            for step_name in available_steps:
                step_info = self.get_step_info(step_name)
                if step_info and step_info.file_components.get('spec'):
                    steps_with_specs.append(step_name)
            
            return sorted(steps_with_specs)
            
        except Exception as e:
            self.logger.error(f"Error listing steps with specs for workspace {workspace_id}: {e}")
            return []
    
    def list_steps_with_scripts(self, workspace_id: Optional[str] = None, 
                              job_type: Optional[str] = None) -> List[str]:
        """
        List all steps that have script components.
        
        This method filters available steps to only return those that have
        script file components, which is useful for alignment validation
        and script-based testing frameworks.
        
        Args:
            workspace_id: Optional workspace filter
            job_type: Optional job type filter
            
        Returns:
            List of step names that have script components
        """
        try:
            # Get all available steps with optional filtering
            available_steps = self.list_available_steps(workspace_id, job_type)
            
            # Filter to only steps that have script components
            steps_with_scripts = []
            for step_name in available_steps:
                step_info = self.get_step_info(step_name)
                if step_info and step_info.file_components.get('script'):
                    steps_with_scripts.append(step_name)
            
            return sorted(steps_with_scripts)
            
        except Exception as e:
            self.logger.error(f"Error listing steps with scripts for workspace {workspace_id}: {e}")
            return []
    
    # US4: Efficient Scaling (Simple but effective search)
    def search_steps(self, query: str, job_type: Optional[str] = None) -> List[StepSearchResult]:
        """
        Search steps by name with basic fuzzy matching.
        
        Args:
            query: Search query string
            job_type: Optional job type filter
            
        Returns:
            List of search results sorted by relevance
        """
        try:
            self._ensure_index_built()
            results = []
            query_lower = query.lower()
            
            for step_name, step_info in self._step_index.items():
                # Simple but effective matching
                if query_lower in step_name.lower():
                    score = 1.0 if query_lower == step_name.lower() else 0.8
                    
                    # Apply job_type filter if specified
                    if job_type and not (step_name.endswith(f"_{job_type}") or job_type == "default"):
                        continue
                    
                    results.append(StepSearchResult(
                        step_name=step_name,
                        workspace_id=step_info.workspace_id,
                        match_score=score,
                        match_reason="name_match" if score == 1.0 else "fuzzy_match",
                        components_available=list(step_info.file_components.keys())
                    ))
            
            # Sort by match score (highest first)
            return sorted(results, key=lambda r: r.match_score, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error searching steps with query '{query}': {e}")
            return []
    
    # US5: Configuration Class Auto-Discovery
    def discover_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Auto-discover configuration classes from core and workspace directories.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping class names to class types
        """
        if self.config_discovery:
            return self.config_discovery.discover_config_classes(project_id)
        else:
            self.logger.warning("ConfigAutoDiscovery not available, returning empty config classes")
            return {}
    
    def build_complete_config_classes(self, project_id: Optional[str] = None) -> Dict[str, Type]:
        """
        Build complete mapping integrating manual registration with auto-discovery.
        
        This addresses the TODO in the existing build_complete_config_classes() function.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Complete dictionary of config classes (manual + auto-discovered)
        """
        if self.config_discovery:
            return self.config_discovery.build_complete_config_classes(project_id)
        else:
            self.logger.warning("ConfigAutoDiscovery not available, returning empty config classes")
            return {}
    
    # EXPANDED DISCOVERY & DETECTION METHODS (Pure Discovery - No Business Logic)
    def discover_contracts_with_scripts(self) -> List[str]:
        """
        DISCOVERY: Find all steps that have both contract and script components.
        
        Returns:
            List of step names that have both contract and script components
        """
        try:
            self._ensure_index_built()
            steps_with_both = []
            
            for step_name, step_info in self._step_index.items():
                if (step_info.file_components.get('contract') and 
                    step_info.file_components.get('script')):
                    steps_with_both.append(step_name)
            
            return steps_with_both
            
        except Exception as e:
            self.logger.error(f"Error discovering contracts with scripts: {e}")
            return []
    
    def detect_framework(self, step_name: str) -> Optional[str]:
        """
        DETECTION: Detect ML framework for a step.
        
        Args:
            step_name: Name of the step to analyze
            
        Returns:
            Framework name (e.g., 'xgboost', 'pytorch') or None if not detected
        """
        try:
            if step_name in self._framework_cache:
                return self._framework_cache[step_name]
            
            step_info = self.get_step_info(step_name)
            if not step_info:
                return None
            
            framework = None
            
            # Simple pattern matching (no business logic)
            if 'framework' in step_info.registry_data:
                framework = step_info.registry_data['framework']
            elif step_info.registry_data.get('builder_step_name'):
                builder_name = step_info.registry_data['builder_step_name'].lower()
                if 'xgboost' in builder_name:
                    framework = 'xgboost'
                elif 'pytorch' in builder_name or 'torch' in builder_name:
                    framework = 'pytorch'
            
            # Check step name patterns as fallback
            if not framework:
                step_name_lower = step_name.lower()
                if 'xgboost' in step_name_lower:
                    framework = 'xgboost'
                elif 'pytorch' in step_name_lower or 'torch' in step_name_lower:
                    framework = 'pytorch'
            
            self._framework_cache[step_name] = framework
            return framework
            
        except Exception as e:
            self.logger.error(f"Error detecting framework for {step_name}: {e}")
            return None
    
    def discover_cross_workspace_components(self, workspace_ids: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        DISCOVERY: Find components across multiple workspaces.
        
        Args:
            workspace_ids: Optional list of workspace IDs to search (defaults to all)
            
        Returns:
            Dictionary mapping workspace IDs to lists of component identifiers
        """
        try:
            self._ensure_index_built()
            if workspace_ids is None:
                workspace_ids = list(self._workspace_steps.keys())
            
            cross_workspace_components = {}
            for workspace_id in workspace_ids:
                workspace_steps = self._workspace_steps.get(workspace_id, [])
                components = []
                
                for step_name in workspace_steps:
                    step_info = self.get_step_info(step_name)
                    if step_info:
                        for component_type, metadata in step_info.file_components.items():
                            if metadata:
                                components.append(f"{step_name}:{component_type}")
                
                cross_workspace_components[workspace_id] = components
            
            return cross_workspace_components
            
        except Exception as e:
            self.logger.error(f"Error discovering cross-workspace components: {e}")
            return {}
    
    def get_builder_class_path(self, step_name: str) -> Optional[str]:
        """
        Get builder class path for a step using BuilderAutoDiscovery component.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Path to builder class or None if not found
        """
        try:
            # Use the initialized builder discovery component
            if self.builder_discovery:
                builder_info = self.builder_discovery.get_builder_info(step_name)
                if builder_info:
                    file_path = builder_info.get('file_path')
                    if file_path and file_path != "Unknown":
                        return str(file_path)
            
            # Fallback to registry-based path construction (legacy compatibility)
            step_info = self.get_step_info(step_name)
            if step_info and 'builder_step_name' in step_info.registry_data:
                builder_name = step_info.registry_data['builder_step_name']
                return f"cursus.steps.builders.{builder_name.lower()}.{builder_name}"
            
            # Check file components as final fallback
            if step_info:
                builder_metadata = step_info.file_components.get('builder')
                if builder_metadata:
                    return str(builder_metadata.path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting builder class path for {step_name}: {e}")
            return None
    
    def load_builder_class(self, step_name: str) -> Optional[Type]:
        """
        Load builder class for a step with job type variant fallback support.
        
        Args:
            step_name: Name of the step (may include job type variant)
            
        Returns:
            Builder class type or None if not found/loadable
        """
        try:
            # Use the initialized builder discovery component
            if self.builder_discovery:
                # Try exact match first
                builder_class = self.builder_discovery.load_builder_class(step_name)
                
                if builder_class:
                    self.logger.debug(f"Successfully loaded builder class for {step_name}: {builder_class.__name__}")
                    return builder_class
                
                # JOB TYPE FALLBACK: Try base step name if compound name fails
                if "_" in step_name:
                    base_step_name = step_name.rsplit("_", 1)[0]
                    job_type = step_name.rsplit("_", 1)[1]
                    
                    self.logger.debug(f"Trying base step name '{base_step_name}' for compound '{step_name}' (job_type: {job_type})")
                    builder_class = self.builder_discovery.load_builder_class(base_step_name)
                    
                    if builder_class:
                        self.logger.info(f"Successfully loaded builder class using base name '{base_step_name}' for '{step_name}': {builder_class.__name__}")
                        return builder_class
                
                self.logger.warning(f"No builder class found for step: {step_name}")
                return None
            else:
                self.logger.warning(f"BuilderAutoDiscovery not available, cannot load builder for {step_name}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error loading builder class for {step_name}: {e}")
            return None
    
    def load_contract_class(self, step_name: str) -> Optional[Any]:
        """
        Load contract class for a step using ContractAutoDiscovery component.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Contract object or None if not found/loadable
        """
        try:
            # Use the initialized contract discovery component
            if self.contract_discovery:
                contract = self.contract_discovery.load_contract_class(step_name)
                
                if contract:
                    self.logger.debug(f"Successfully loaded contract for {step_name}")
                    return contract
                else:
                    self.logger.warning(f"No contract found for step: {step_name}")
                    return None
            else:
                self.logger.warning(f"ContractAutoDiscovery not available, cannot load contract for {step_name}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error loading contract for {step_name}: {e}")
            return None
    
    def load_spec_class(self, step_name: str) -> Optional[Any]:
        """
        Load specification instance for a step using SpecAutoDiscovery component.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Specification instance or None if not found/loadable
        """
        try:
            # Use the initialized spec discovery component
            if self.spec_discovery:
                spec_instance = self.spec_discovery.load_spec_class(step_name)
                
                if spec_instance:
                    self.logger.debug(f"Successfully loaded specification for {step_name}")
                    return spec_instance
                else:
                    self.logger.warning(f"No specification found for step: {step_name}")
                    return None
            else:
                self.logger.warning(f"SpecAutoDiscovery not available, cannot load specification for {step_name}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error loading specification for {step_name}: {e}")
            return None
    
    def find_specs_by_contract(self, contract_name: str) -> Dict[str, Any]:
        """
        Find all specifications that reference a specific contract.
        
        This method enables contract-specification alignment validation by finding
        specifications that are associated with a given contract name.
        
        Args:
            contract_name: Name of the contract to find specifications for
            
        Returns:
            Dictionary mapping spec names to specification instances
        """
        try:
            if self.spec_discovery:
                return self.spec_discovery.find_specs_by_contract(contract_name)
            else:
                self.logger.warning(f"SpecAutoDiscovery not available, cannot find specs for contract {contract_name}")
                return {}
        except Exception as e:
            self.logger.error(f"Error finding specs for contract {contract_name}: {e}")
            return {}
    
    def serialize_spec(self, spec_instance: Any) -> Dict[str, Any]:
        """
        Convert specification instance to dictionary format.
        
        This method provides standardized serialization of StepSpecification objects
        for use in validation and alignment testing.
        
        Args:
            spec_instance: StepSpecification instance to serialize
            
        Returns:
            Dictionary representation of the specification
        """
        try:
            if self.spec_discovery:
                return self.spec_discovery.serialize_spec(spec_instance)
            else:
                self.logger.warning("SpecAutoDiscovery not available, cannot serialize specification")
                return {}
        except Exception as e:
            self.logger.error(f"Error serializing specification: {e}")
            return {}
    
    def load_all_specifications(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all specification instances from both package and workspace directories.
        
        This method provides comprehensive specification loading for validation frameworks
        and dependency analysis tools. It discovers and loads all available specifications,
        serializing them to dictionary format for easy consumption.
        
        Returns:
            Dictionary mapping step names to serialized specification dictionaries
        """
        try:
            if self.spec_discovery:
                return self.spec_discovery.load_all_specifications()
            else:
                self.logger.warning("SpecAutoDiscovery not available, cannot load all specifications")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading all specifications: {e}")
            return {}
    
    def get_spec_job_type_variants(self, base_step_name: str) -> List[str]:
        """
        Get all job type variants for a base step name from specifications.
        
        This method discovers different job type variants (training, validation, testing, etc.)
        for a given base step name by examining specification file naming patterns.
        
        Args:
            base_step_name: Base name of the step
            
        Returns:
            List of job type variants found
        """
        try:
            if self.spec_discovery:
                return self.spec_discovery.get_job_type_variants(base_step_name)
            else:
                self.logger.warning(f"SpecAutoDiscovery not available, cannot get job type variants for {base_step_name}")
                return []
        except Exception as e:
            self.logger.error(f"Error getting spec job type variants for {base_step_name}: {e}")
            return []
    
    # PHASE 2 ENHANCEMENT: Smart Specification Integration - Delegation Methods
    def create_unified_specification(self, contract_name: str) -> Dict[str, Any]:
        """
        Create unified specification from multiple variants using smart selection.
        
        This method delegates to SpecAutoDiscovery for smart specification handling,
        integrating SmartSpecificationSelector functionality:
        - Multi-variant specification discovery
        - Union of dependencies and outputs from all variants
        - Smart validation logic with detailed feedback
        - Primary specification selection (training > generic > first available)
        
        Args:
            contract_name: Name of the contract to find specifications for
            
        Returns:
            Unified specification model with metadata
        """
        try:
            if self.spec_discovery:
                return self.spec_discovery.create_unified_specification(contract_name)
            else:
                self.logger.warning(f"SpecAutoDiscovery not available, cannot create unified specification for {contract_name}")
                return {
                    "primary_spec": {},
                    "variants": {},
                    "unified_dependencies": {},
                    "unified_outputs": {},
                    "dependency_sources": {},
                    "output_sources": {},
                    "variant_count": 0,
                }
        except Exception as e:
            self.logger.error(f"Error creating unified specification for {contract_name}: {e}")
            return {
                "primary_spec": {},
                "variants": {},
                "unified_dependencies": {},
                "unified_outputs": {},
                "dependency_sources": {},
                "output_sources": {},
                "variant_count": 0,
            }
    
    def validate_logical_names_smart(self, contract: Dict[str, Any], contract_name: str) -> List[Dict[str, Any]]:
        """
        Smart validation using multi-variant specification logic.
        
        This method delegates to SpecAutoDiscovery for smart validation,
        implementing the core Smart Specification Selection validation:
        - Contract input is valid if it exists in ANY variant
        - Contract must cover intersection of REQUIRED dependencies
        - Provides detailed feedback about which variants need what
        
        Args:
            contract: Contract dictionary
            contract_name: Name of the contract
            
        Returns:
            List of validation issues
        """
        try:
            if self.spec_discovery:
                return self.spec_discovery.validate_logical_names_smart(contract, contract_name)
            else:
                self.logger.warning(f"SpecAutoDiscovery not available, cannot perform smart validation for {contract_name}")
                return [
                    {
                        "severity": "ERROR",
                        "category": "spec_discovery_unavailable",
                        "message": f"SpecAutoDiscovery not available for smart validation of contract {contract_name}",
                        "details": {"contract": contract_name},
                        "recommendation": "Check SpecAutoDiscovery initialization",
                    }
                ]
        except Exception as e:
            self.logger.error(f"Error in smart validation for {contract_name}: {e}")
            return [
                {
                    "severity": "ERROR",
                    "category": "smart_validation_error",
                    "message": f"Smart validation failed for contract {contract_name}: {str(e)}",
                    "details": {"contract": contract_name, "error": str(e)},
                    "recommendation": "Check contract and specification files for errors",
                }
            ]
    
    def serialize_contract(self, contract_instance: Any) -> Dict[str, Any]:
        """
        Convert contract instance to dictionary format.
        
        This method provides standardized serialization of ScriptContract objects
        for use in script-contract alignment validation.
        
        Args:
            contract_instance: Contract instance to serialize
            
        Returns:
            Dictionary representation of the contract
        """
        try:
            if self.contract_discovery:
                return self.contract_discovery.serialize_contract(contract_instance)
            else:
                self.logger.warning("ContractAutoDiscovery not available, cannot serialize contract")
                return {}
        except Exception as e:
            self.logger.error(f"Error serializing contract: {e}")
            return {}
    
    def find_contracts_by_entry_point(self, entry_point: str) -> Dict[str, Any]:
        """
        Find contracts that reference a specific script entry point.
        
        Args:
            entry_point: Script entry point (e.g., "model_evaluation_xgb.py")
            
        Returns:
            Dictionary mapping contract names to contract instances
        """
        try:
            if self.contract_discovery:
                return self.contract_discovery.find_contracts_by_entry_point(entry_point)
            else:
                self.logger.warning(f"ContractAutoDiscovery not available, cannot find contracts for entry point {entry_point}")
                return {}
        except Exception as e:
            self.logger.error(f"Error finding contracts for entry point {entry_point}: {e}")
            return {}
    
    def get_contract_entry_points(self) -> Dict[str, str]:
        """
        Get all contract entry points for validation.
        
        Returns:
            Dictionary mapping contract names to their entry points
        """
        try:
            if self.contract_discovery:
                return self.contract_discovery.get_contract_entry_points()
            else:
                self.logger.warning("ContractAutoDiscovery not available, cannot get contract entry points")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting contract entry points: {e}")
            return {}
    
    def validate_contract_script_mapping(self) -> Dict[str, Any]:
        """
        Validate contract-script relationships across the system.
        
        Returns:
            Dictionary with validation results and mapping statistics
        """
        try:
            contracts_with_scripts = self.discover_contracts_with_scripts()
            entry_points = self.get_contract_entry_points()
            
            validation_results = {
                "total_contracts": len(entry_points),
                "contracts_with_scripts": len(contracts_with_scripts),
                "orphaned_contracts": [],
                "orphaned_scripts": [],
                "valid_mappings": [],
            }
            
            # Find orphaned contracts (contracts without corresponding scripts)
            for contract_name, entry_point in entry_points.items():
                if contract_name not in contracts_with_scripts:
                    validation_results["orphaned_contracts"].append({
                        "contract_name": contract_name,
                        "entry_point": entry_point
                    })
                else:
                    validation_results["valid_mappings"].append({
                        "contract_name": contract_name,
                        "entry_point": entry_point
                    })
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating contract-script mapping: {e}")
            return {"error": str(e)}
    
    # Additional utility methods for job type variants
    def get_job_type_variants(self, base_step_name: str) -> List[str]:
        """
        Get all job_type variants for a base step name.
        
        Args:
            base_step_name: Base name of the step
            
        Returns:
            List of job type variants found
        """
        try:
            self._ensure_index_built()
            variants = []
            
            for step_name in self._step_index.keys():
                if step_name.startswith(f"{base_step_name}_"):
                    job_type = step_name[len(base_step_name)+1:]
                    variants.append(job_type)
            
            return variants
            
        except Exception as e:
            self.logger.error(f"Error getting job type variants for {base_step_name}: {e}")
            return []
    
    def resolve_pipeline_node(self, node_name: str) -> Optional[StepInfo]:
        """
        Resolve PipelineDAG node name to StepInfo (handles job_type variants).
        
        Args:
            node_name: Node name from PipelineDAG
            
        Returns:
            StepInfo for the node, or None if not found
        """
        return self.get_step_info(node_name)
    
    def _normalize_workspace_dirs(self, workspace_dirs: Optional[Union[Path, List[Path]]]) -> List[Path]:
        """
        Normalize workspace_dirs to a consistent list format.
        
        Args:
            workspace_dirs: Optional workspace directory(ies)
            
        Returns:
            List of Path objects
        """
        if workspace_dirs is None:
            return []
        elif isinstance(workspace_dirs, Path):
            return [workspace_dirs]
        else:
            return list(workspace_dirs)
    
    def _initialize_config_discovery(self) -> Optional['ConfigAutoDiscovery']:
        """
        Initialize ConfigAutoDiscovery component with proper error handling.
        
        Returns:
            ConfigAutoDiscovery instance or None if initialization fails
        """
        try:
            if ConfigAutoDiscovery is None:
                self.logger.warning("ConfigAutoDiscovery not available due to import failure")
                return None
            return ConfigAutoDiscovery(self.package_root, self.workspace_dirs)
        except Exception as e:
            self.logger.error(f"Error initializing ConfigAutoDiscovery: {e}")
            return None
    
    def _initialize_builder_discovery(self) -> Optional['BuilderAutoDiscovery']:
        """
        Initialize BuilderAutoDiscovery component with proper error handling.
        
        Returns:
            BuilderAutoDiscovery instance or None if initialization fails
        """
        try:
            if BuilderAutoDiscovery is None:
                self.logger.warning("BuilderAutoDiscovery not available due to import failure")
                return None
            
            builder_discovery = BuilderAutoDiscovery(self.package_root, self.workspace_dirs)
            return builder_discovery
            
        except Exception as e:
            self.logger.error(f"Error initializing BuilderAutoDiscovery: {e}")
            return None
    
    def _initialize_contract_discovery(self) -> Optional['ContractAutoDiscovery']:
        """
        Initialize ContractAutoDiscovery component with proper error handling.
        
        Returns:
            ContractAutoDiscovery instance or None if initialization fails
        """
        try:
            if ContractAutoDiscovery is None:
                self.logger.warning("ContractAutoDiscovery not available due to import failure")
                return None
            return ContractAutoDiscovery(self.package_root, self.workspace_dirs)
        except Exception as e:
            self.logger.error(f"Error initializing ContractAutoDiscovery: {e}")
            return None
    
    def _initialize_spec_discovery(self) -> Optional['SpecAutoDiscovery']:
        """
        Initialize SpecAutoDiscovery component with proper error handling.
        
        Returns:
            SpecAutoDiscovery instance or None if initialization fails
        """
        try:
            if SpecAutoDiscovery is None:
                self.logger.warning("SpecAutoDiscovery not available due to import failure")
                return None
            return SpecAutoDiscovery(self.package_root, self.workspace_dirs)
        except Exception as e:
            self.logger.error(f"Error initializing SpecAutoDiscovery: {e}")
            return None
    
    def _initialize_script_discovery(self) -> Optional['ScriptAutoDiscovery']:
        """
        Initialize ScriptAutoDiscovery component with proper error handling.
        
        Returns:
            ScriptAutoDiscovery instance or None if initialization fails
        """
        try:
            if ScriptAutoDiscovery is None:
                self.logger.warning("ScriptAutoDiscovery not available due to import failure")
                return None
            
            # Support priority workspace for interactive runtime testing
            priority_workspace_dir = None
            if self.workspace_dirs:
                # Use first workspace as priority (can be enhanced later)
                priority_workspace_dir = self.workspace_dirs[0]
            
            return ScriptAutoDiscovery(
                self.package_root, 
                self.workspace_dirs, 
                priority_workspace_dir=priority_workspace_dir
            )
        except Exception as e:
            self.logger.error(f"Error initializing ScriptAutoDiscovery: {e}")
            return None
    
    def _find_package_root(self) -> Path:
        """
        Find cursus package root using relative path navigation.
        
        Works in all deployment scenarios:
        - PyPI: site-packages/cursus/
        - Source: src/cursus/
        - Submodule: parent_package/cursus/
        """
        # From cursus/step_catalog/step_catalog.py, navigate to cursus package root
        current_file = Path(__file__)
        
        # Navigate up to find cursus package root
        current_dir = current_file.parent
        while current_dir.name != 'cursus' and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        
        if current_dir.name == 'cursus':
            return current_dir
        else:
            # Fallback: assume we're in cursus package structure
            return current_file.parent.parent  # step_catalog -> cursus

    # Private methods for simple implementation
    def _ensure_index_built(self) -> None:
        """Build index on first access (lazy loading)."""
        if not self._index_built:
            self._build_index()
            self._index_built = True
    
    def _build_index(self) -> None:
        """Build index using simplified dual-space discovery."""
        start_time = time.time()
        
        try:
            # Load registry data (existing functionality)
            self._load_registry_data()
            
            # Discover package components (always available)
            self._discover_package_components()
            
            # Discover workspace components (if workspace_dirs provided)
            if self.workspace_dirs:
                self._discover_workspace_components()
            
            # Record successful build
            build_time = time.time() - start_time
            self.metrics['index_build_time'] = build_time
            self.metrics['last_index_build'] = datetime.now()
            
            self.logger.info(f"Index built successfully in {build_time:.3f}s with {len(self._step_index)} steps")
            
        except Exception as e:
            build_time = time.time() - start_time
            self.logger.error(f"Index build failed after {build_time:.3f}s: {e}")
            # Graceful degradation
            self._step_index = {}
            self._component_index = {}
            self._workspace_steps = {}

    def _load_registry_data(self) -> None:
        """Load registry data first."""
        try:
            from ..registry.step_names import get_step_names
            
            step_names_dict = get_step_names()
            for step_name, registry_data in step_names_dict.items():
                step_info = StepInfo(
                    step_name=step_name,
                    workspace_id="core",
                    registry_data=registry_data,
                    file_components={}
                )
                self._step_index[step_name] = step_info
                self._workspace_steps.setdefault("core", []).append(step_name)
            
            self.logger.debug(f"Loaded {len(step_names_dict)} steps from registry")
            
        except ImportError as e:
            self.logger.warning(f"Could not import STEP_NAMES registry: {e}")

    def _discover_package_components(self) -> None:
        """Discover components within the cursus package."""
        try:
            # Package components are always at package_root/steps/
            core_steps_dir = self.package_root / "steps"
            if core_steps_dir.exists():
                self._discover_workspace_components_in_dir("core", core_steps_dir)
        except Exception as e:
            self.logger.error(f"Error discovering package components: {e}")

    def _discover_workspace_components(self) -> None:
        """Discover components in user-provided workspace directories."""
        for workspace_dir in self.workspace_dirs:
            try:
                workspace_path = Path(workspace_dir)
                if not workspace_path.exists():
                    self.logger.warning(f"Workspace directory does not exist: {workspace_path}")
                    continue
                
                # Simplified structure: workspace_dir points directly to directory containing scripts/, contracts/, etc.
                workspace_id = workspace_path.name  # Use directory name as workspace ID
                self._discover_workspace_components_in_dir(workspace_id, workspace_path)
                    
            except Exception as e:
                self.logger.error(f"Error discovering workspace components in {workspace_dir}: {e}")

    def _discover_workspace_components_in_dir(self, workspace_id: str, steps_dir: Path) -> None:
        """
        Discover components in a workspace directory.
        
        Args:
            workspace_id: ID of the workspace
            steps_dir: Directory containing step components
        """
        if not steps_dir.exists():
            self.logger.warning(f"Workspace directory does not exist: {steps_dir}")
            return
        
        component_types = {
            "scripts": "script",
            "contracts": "contract", 
            "specs": "spec",
            "builders": "builder",
            "configs": "config"
        }
        
        for dir_name, component_type in component_types.items():
            component_dir = steps_dir / dir_name
            if not component_dir.exists():
                continue
                
            try:
                for py_file in component_dir.glob("*.py"):
                    if py_file.name.startswith("__"):
                        continue
                    
                    try:
                        step_name = self._extract_step_name(py_file.name, component_type)
                        if step_name:
                            self._add_component_to_index(step_name, py_file, component_type, workspace_id)
                    except Exception as e:
                        self.logger.warning(f"Error processing component file {py_file}: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error scanning component directory {component_dir}: {e}")
                continue
    
    
    def _add_component_to_index(self, step_name: str, py_file: Path, component_type: str, workspace_id: str) -> None:
        """
        Add component to index with canonical name resolution.
        
        Args:
            step_name: Name of the step (may be snake_case from file)
            py_file: Path to the component file
            component_type: Type of component
            workspace_id: ID of the workspace
        """
        try:
            # Resolve to canonical name if possible
            canonical_name = self._resolve_to_canonical_name_for_indexing(step_name)
            target_step_name = canonical_name if canonical_name else step_name
            
            # Update or create step info
            if target_step_name in self._step_index:
                step_info = self._step_index[target_step_name]
                # Update workspace if this is from a developer workspace
                if workspace_id != "core":
                    step_info.workspace_id = workspace_id
            else:
                step_info = StepInfo(
                    step_name=target_step_name,
                    workspace_id=workspace_id,
                    registry_data={},
                    file_components={}
                )
                self._step_index[target_step_name] = step_info
                self._workspace_steps.setdefault(workspace_id, []).append(target_step_name)
            
            # Add file component
            file_metadata = FileMetadata(
                path=py_file,
                file_type=component_type,
                modified_time=datetime.fromtimestamp(py_file.stat().st_mtime)
            )
            step_info.file_components[component_type] = file_metadata
            self._component_index[py_file] = target_step_name
            
            if canonical_name:
                self.logger.debug(f"Linked {component_type} file {py_file.name} to canonical step {canonical_name}")
            
        except Exception as e:
            self.logger.warning(f"Error adding component {py_file} to index: {e}")
    
    def _resolve_to_canonical_name_for_indexing(self, step_name: str) -> Optional[str]:
        """
        Resolve step name to canonical name for indexing purposes.
        
        Args:
            step_name: Step name to resolve (likely snake_case from file)
            
        Returns:
            Canonical PascalCase name if found, None otherwise
        """
        try:
            from ..registry.step_names import get_step_names
            registry = get_step_names()
            
            # If already canonical, return as-is
            if step_name in registry:
                return step_name
            
            # Try snake_case to PascalCase conversion
            if '_' in step_name and step_name.islower():
                pascal_candidate = ''.join(word.capitalize() for word in step_name.split('_'))
                if pascal_candidate in registry:
                    return pascal_candidate
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error resolving canonical name for {step_name}: {e}")
            return None
    
    def _extract_step_name(self, filename: str, component_type: str) -> Optional[str]:
        """
        Extract step name from filename based on component type.
        
        Args:
            filename: Name of the file
            component_type: Type of component
            
        Returns:
            Extracted step name, or None if not extractable
        """
        name = filename[:-3]  # Remove .py extension
        
        if component_type == "contract" and name.endswith("_contract"):
            return name[:-9]  # Remove _contract
        elif component_type == "spec" and name.endswith("_spec"):
            return name[:-5]  # Remove _spec
        elif component_type == "builder" and name.startswith("builder_") and name.endswith("_step"):
            return name[8:-5]  # Remove builder_ and _step
        elif component_type == "config" and name.startswith("config_") and name.endswith("_step"):
            return name[7:-5]  # Remove config_ and _step
        elif component_type == "script":
            return name
        
        return None
    
    def _deduplicate_and_filter_concrete_steps(self, steps: List[str]) -> List[str]:
        """
        Deduplicate steps and filter to concrete pipeline steps only.
        
        Applies:
        1. Canonical name resolution (PascalCase from registry)
        2. Base config exclusion ('Base', 'Processing')
        3. Job type variant filtering
        
        Args:
            steps: List of step names (mix of PascalCase and snake_case)
            
        Returns:
            List of concrete canonical step names (PascalCase)
        """
        try:
            # Get registry as Single Source of Truth
            from ..registry.step_names import get_step_names
            registry = get_step_names()
            canonical_steps = set()
            
            # Base configurations to exclude
            BASE_CONFIGS = {'Base', 'Processing'}
            
            for step_name in steps:
                # Skip job type variants
                if self._is_job_type_variant(step_name):
                    continue
                    
                # 1. If already canonical (in registry), use as-is
                if step_name in registry:
                    if step_name not in BASE_CONFIGS:  # Exclude base configs
                        canonical_steps.add(step_name)
                else:
                    # 2. Try to resolve snake_case to PascalCase
                    canonical_name = self._resolve_to_canonical_name(step_name, registry)
                    if canonical_name and canonical_name not in BASE_CONFIGS:
                        canonical_steps.add(canonical_name)
            
            return sorted(list(canonical_steps))
            
        except Exception as e:
            self.logger.error(f"Error in canonical name deduplication: {e}")
            return sorted(list(set(steps)))  # Fallback to simple deduplication
    
    def _is_job_type_variant(self, step_name: str) -> bool:
        """Check if step name is a job type variant."""
        JOB_SUFFIXES = ['_calibration', '_testing', '_training', '_validation', '_inference', '_evaluation']
        return any(step_name.endswith(suffix) for suffix in JOB_SUFFIXES)
    
    def _resolve_to_canonical_name(self, step_name: str, registry: Dict[str, Any]) -> Optional[str]:
        """Resolve snake_case step name to canonical PascalCase name."""
        # Simple snake_case to PascalCase conversion
        if '_' in step_name and step_name.islower():
            pascal_candidate = ''.join(word.capitalize() for word in step_name.split('_'))
            if pascal_candidate in registry:
                self.logger.debug(f"Resolved canonical name: {step_name} → {pascal_candidate}")
                return pascal_candidate
        
        return None
    
    # PHASE 1 ENHANCEMENT: Config-to-Builder Resolution (delegated to mapping module)
    def get_builder_for_config(self, config, node_name: str = None) -> Optional[Type]:
        """
        Map config instance directly to builder class.
        
        This method replaces StepBuilderRegistry.get_builder_for_config() functionality
        while using the registry system as Single Source of Truth.
        
        Args:
            config: Configuration instance (BasePipelineConfig)
            node_name: Optional DAG node name for context
            
        Returns:
            Builder class type or None if not found
        """
        return self.mapper.get_builder_for_config(config, node_name)
    
    def get_builder_for_step_type(self, step_type: str) -> Optional[Type]:
        """
        Get builder class for step type with legacy alias support.
        
        This method replaces StepBuilderRegistry.get_builder_for_step_type() functionality.
        
        Args:
            step_type: Step type name (may be legacy alias)
            
        Returns:
            Builder class type or None if not found
        """
        return self.mapper.get_builder_for_step_type(step_type)
    
    # PHASE 1 ENHANCEMENT: Pipeline Construction Interface (delegated to mapping module)
    def is_step_type_supported(self, step_type: str) -> bool:
        """
        Check if step type is supported (including legacy aliases).
        
        Args:
            step_type: Step type name
            
        Returns:
            True if supported, False otherwise
        """
        return self.mapper.is_step_type_supported(step_type)
    
    def validate_builder_availability(self, step_types: List[str]) -> Dict[str, bool]:
        """
        Validate that builders are available for step types.
        
        Args:
            step_types: List of step types to validate
            
        Returns:
            Dictionary mapping step types to availability status
        """
        return self.mapper.validate_builder_availability(step_types)
    
    def get_config_types_for_step_type(self, step_type: str) -> List[str]:
        """
        Get possible config class names for a step type.
        
        Args:
            step_type: Step type name
            
        Returns:
            List of possible configuration class names
        """
        return self.mapper.get_config_types_for_step_type(step_type)
    
    def list_supported_step_types(self) -> List[str]:
        """
        List all supported step types including legacy aliases.
        
        Returns:
            List of supported step type names
        """
        return self.mapper.list_supported_step_types()
    
    # PHASE 1 ENHANCEMENT: Enhanced Registry Integration (delegated to mapping module)
    def validate_step_name_with_registry(self, step_name: str) -> bool:
        """
        Use registry system for step name validation.
        
        Args:
            step_name: Step name to validate
            
        Returns:
            True if valid, False otherwise
        """
        return self.mapper.validate_step_name_with_registry(step_name)
    
    # PHASE 1 ENHANCEMENT: Pipeline Construction Interface Methods
    def get_builder_map(self) -> Dict[str, Type]:
        """
        Get a complete builder map for pipeline construction.
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        if self.pipeline_interface is None:
            self.logger.error("pipeline_interface is None, using fallback builder map generation")
            return self._generate_fallback_builder_map()
        
        try:
            builder_map = self.pipeline_interface.get_builder_map()
            return builder_map
        except Exception as e:
            self.logger.error(f"pipeline_interface.get_builder_map() failed: {e}")
            return self._generate_fallback_builder_map()
    
    def _generate_fallback_builder_map(self) -> Dict[str, Type]:
        """
        Generate builder map without pipeline_interface (fallback method).
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        try:
            builder_map = {}
            step_types = self.mapper.list_supported_step_types()
            
            for step_type in step_types:
                builder_class = self.mapper.get_builder_for_step_type(step_type)
                if builder_class:
                    builder_map[step_type] = builder_class
            
            self.logger.info(f"Generated fallback builder map with {len(builder_map)} builders")
            return builder_map
            
        except Exception as e:
            self.logger.error(f"Error generating fallback builder map: {e}")
            return {}
    
    def validate_dag_compatibility(self, step_types: List[str]) -> Dict[str, Any]:
        """
        Validate DAG compatibility with available builders.
        
        Args:
            step_types: List of step types in the DAG
            
        Returns:
            Dictionary with validation results
        """
        return self.pipeline_interface.validate_dag_compatibility(step_types)
    
    def get_step_builder_suggestions(self, config_class_name: str) -> List[str]:
        """
        Get suggestions for step builders based on config class name.
        
        Args:
            config_class_name: Configuration class name
            
        Returns:
            List of suggested step type names
        """
        return self.pipeline_interface.get_step_builder_suggestions(config_class_name)
    
    def get_metrics_report(self) -> Dict[str, Any]:
        """Get simple metrics report."""
        success_rate = (
            (self.metrics['queries'] - self.metrics['errors']) / self.metrics['queries'] 
            if self.metrics['queries'] > 0 else 0.0
        )
        
        return {
            'total_queries': self.metrics['queries'],
            'success_rate': success_rate,
            'avg_response_time_ms': self.metrics['avg_response_time'] * 1000,
            'index_build_time_s': self.metrics['index_build_time'],
            'last_index_build': self.metrics['last_index_build'].isoformat() if self.metrics['last_index_build'] else None,
            'total_steps_indexed': len(self._step_index),
            'total_workspaces': len(self._workspace_steps)
        }
    
    # PHASE 1 ENHANCEMENT: Dynamic Builder Discovery Methods
    def get_all_builders(self) -> Dict[str, Type]:
        """
        Get all available builders with canonical names.
        
        This method provides comprehensive builder discovery for dynamic testing
        without requiring hard-coded maintenance.
        
        Returns:
            Dict mapping canonical names to builder classes
        """
        try:
            all_steps = self.list_available_steps()
            builders = {}
            
            for step_name in all_steps:
                builder_class = self.load_builder_class(step_name)
                if builder_class:
                    builders[step_name] = builder_class
            
            self.logger.debug(f"Discovered {len(builders)} builders via step catalog")
            return builders
            
        except Exception as e:
            self.logger.error(f"Error getting all builders: {e}")
            return {}
    
    def get_builders_by_step_type(self, step_type: str) -> Dict[str, Type]:
        """
        Get builders filtered by SageMaker step type.
        
        This method enables step-type-specific testing by filtering builders
        based on their registered SageMaker step type.
        
        Args:
            step_type: SageMaker step type (Processing, Training, Transform, CreateModel, etc.)
            
        Returns:
            Dict mapping canonical names to builder classes for the step type
        """
        try:
            all_builders = self.get_all_builders()
            step_builders = {}
            
            for step_name, builder_class in all_builders.items():
                step_info = self.get_step_info(step_name)
                if step_info and step_info.registry_data.get('sagemaker_step_type') == step_type:
                    step_builders[step_name] = builder_class
            
            self.logger.debug(f"Found {len(step_builders)} builders for step type '{step_type}'")
            return step_builders
            
        except Exception as e:
            self.logger.error(f"Error getting builders for step type {step_type}: {e}")
            return {}
    
    # SMART DEFAULT VALUE INHERITANCE ENHANCEMENT: Parent Config Retrieval Methods
    def get_immediate_parent_config_class(self, config_class_name: str) -> Optional[str]:
        """
        Get the immediate parent config class name for inheritance.
        
        This method enables Smart Default Value Inheritance by identifying which parent
        config class a specific config should inherit from. It implements cascading
        inheritance where TabularPreprocessingConfig inherits from ProcessingStepConfigBase
        (not BasePipelineConfig directly) to get all cascaded values.
        
        Args:
            config_class_name: Target config class name (e.g., "TabularPreprocessingConfig")
            
        Returns:
            Immediate parent class name (e.g., "ProcessingStepConfigBase") or None if not found
            
        Example:
            parent = step_catalog.get_immediate_parent_config_class("TabularPreprocessingConfig")
            # Returns: "ProcessingStepConfigBase" (not "BasePipelineConfig")
            
            parent = step_catalog.get_immediate_parent_config_class("CradleDataLoadConfig")
            # Returns: "BasePipelineConfig" (direct inheritance)
        """
        try:
            # Use existing config discovery infrastructure
            config_classes = self.discover_config_classes()
            config_class = config_classes.get(config_class_name)
            
            if not config_class:
                self.logger.warning(f"Config class {config_class_name} not found")
                return None
            
            # Import BasePipelineConfig for inheritance checking
            try:
                from ..core.base.config_base import BasePipelineConfig
            except ImportError:
                self.logger.error("Could not import BasePipelineConfig for inheritance analysis")
                return None
            
            # Walk inheritance chain to find immediate parent
            for base_class in config_class.__mro__:
                if (base_class != config_class and 
                    issubclass(base_class, BasePipelineConfig) and 
                    base_class != BasePipelineConfig):
                    parent_name = base_class.__name__
                    self.logger.debug(f"Found immediate parent for {config_class_name}: {parent_name}")
                    return parent_name
            
            # Fallback to BasePipelineConfig if no intermediate parent found
            self.logger.debug(f"No intermediate parent found for {config_class_name}, using BasePipelineConfig")
            return "BasePipelineConfig"
            
        except Exception as e:
            self.logger.error(f"Error getting parent class for {config_class_name}: {e}")
            return None
    
    def extract_parent_values_for_inheritance(self, 
                                            target_config_class_name: str,
                                            completed_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract parent values for inheritance from completed configs.
        
        This method enables Smart Default Value Inheritance by extracting field values
        from the immediate parent config for pre-populating child config forms. It
        implements cascading inheritance to eliminate redundant user input.
        
        Args:
            target_config_class_name: Target config class name
            completed_configs: Dictionary of completed config instances by class name
            
        Returns:
            Dictionary of field values from immediate parent config
            
        Example:
            completed_configs = {
                "BasePipelineConfig": base_config_instance,
                "ProcessingStepConfigBase": processing_config_instance
            }
            
            parent_values = step_catalog.extract_parent_values_for_inheritance(
                "TabularPreprocessingConfig", completed_configs
            )
            # Returns: ALL field values from ProcessingStepConfigBase
            # (which includes cascaded values from BasePipelineConfig)
            # Result: {"author": "lukexie", "bucket": "my-bucket", 
            #          "processing_instance_type": "ml.m5.2xlarge", ...}
        """
        try:
            # Get immediate parent class name using the first method
            parent_class_name = self.get_immediate_parent_config_class(target_config_class_name)
            
            if not parent_class_name:
                self.logger.warning(f"No parent class found for {target_config_class_name}")
                return {}
            
            # Get the completed parent config instance
            parent_config = completed_configs.get(parent_class_name)
            
            if not parent_config:
                self.logger.warning(f"Parent config {parent_class_name} not found in completed configs")
                self.logger.debug(f"Available completed configs: {list(completed_configs.keys())}")
                return {}
            
            # Extract field values from parent config using Pydantic model_fields
            parent_values = {}
            
            # Check if parent_config has model_fields (Pydantic v2)
            if hasattr(parent_config.__class__, 'model_fields'):
                for field_name, field_info in parent_config.__class__.model_fields.items():
                    if hasattr(parent_config, field_name):
                        field_value = getattr(parent_config, field_name)
                        if field_value is not None:
                            parent_values[field_name] = field_value
            else:
                # Fallback for older Pydantic versions or other config types
                self.logger.warning(f"Parent config {parent_class_name} does not have model_fields, using __dict__")
                for field_name, field_value in parent_config.__dict__.items():
                    if not field_name.startswith('_') and field_value is not None:
                        parent_values[field_name] = field_value
            
            self.logger.debug(f"Extracted {len(parent_values)} parent values for {target_config_class_name} from {parent_class_name}")
            return parent_values
            
        except Exception as e:
            self.logger.error(f"Error extracting parent values for {target_config_class_name}: {e}")
            return {}
    
    # SCRIPT DISCOVERY METHODS: Interactive Runtime Testing Support
    def discover_script_files(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Discover script files from package and workspaces with prioritization.
        
        This method enables script discovery for interactive runtime testing by
        finding scripts referenced in config and contract entry points.
        
        Args:
            project_id: Optional project ID for workspace-specific discovery
            
        Returns:
            Dictionary mapping script names to ScriptInfo objects
        """
        try:
            if self.script_discovery:
                return self.script_discovery.discover_script_files(project_id)
            else:
                self.logger.warning("ScriptAutoDiscovery not available, cannot discover script files")
                return {}
        except Exception as e:
            self.logger.error(f"Error discovering script files: {e}")
            return {}
    
    def discover_scripts_from_dag(self, dag) -> Dict[str, Any]:
        """
        Discover scripts referenced in a DAG with intelligent node-to-script mapping.
        
        This method enables DAG-guided script discovery for interactive runtime testing
        by mapping DAG nodes to actual script files using step catalog intelligence.
        
        Args:
            dag: PipelineDAG object
            
        Returns:
            Dictionary mapping script names to ScriptInfo objects
        """
        try:
            if self.script_discovery:
                return self.script_discovery.discover_scripts_from_dag(dag)
            else:
                self.logger.warning("ScriptAutoDiscovery not available, cannot discover scripts from DAG")
                return {}
        except Exception as e:
            self.logger.error(f"Error discovering scripts from DAG: {e}")
            return {}
    
    def load_script_info(self, script_name: str) -> Optional[Any]:
        """
        Load script information for a specific script with workspace-aware discovery.
        
        This method enables script information retrieval for interactive runtime testing
        with workspace prioritization support.
        
        Args:
            script_name: Name of the script to load info for
            
        Returns:
            ScriptInfo object or None if not found
        """
        try:
            if self.script_discovery:
                return self.script_discovery.load_script_info(script_name)
            else:
                self.logger.warning(f"ScriptAutoDiscovery not available, cannot load script info for {script_name}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading script info for {script_name}: {e}")
            return None
    
    def get_script_info(self, script_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a script in dictionary format.
        
        This method provides script information for interactive runtime testing
        in a user-friendly dictionary format.
        
        Args:
            script_name: Name of the script
            
        Returns:
            Dictionary with script information or None if not found
        """
        try:
            if self.script_discovery:
                return self.script_discovery.get_script_info(script_name)
            else:
                self.logger.warning(f"ScriptAutoDiscovery not available, cannot get script info for {script_name}")
                return None
        except Exception as e:
            self.logger.error(f"Error getting script info for {script_name}: {e}")
            return None
    
    def list_available_scripts(self) -> List[str]:
        """
        List all available script names.
        
        This method enables script enumeration for interactive runtime testing
        by listing all discovered script names.
        
        Returns:
            List of script names that have been discovered
        """
        try:
            if self.script_discovery:
                return self.script_discovery.list_available_scripts()
            else:
                self.logger.warning("ScriptAutoDiscovery not available, cannot list available scripts")
                return []
        except Exception as e:
            self.logger.error(f"Error listing available scripts: {e}")
            return []
    
    def get_script_discovery_stats(self) -> Dict[str, Any]:
        """
        Get script discovery statistics.
        
        This method provides discovery statistics for interactive runtime testing
        to help users understand the script discovery process.
        
        Returns:
            Dictionary with script discovery statistics
        """
        try:
            if self.script_discovery:
                return self.script_discovery.get_discovery_stats()
            else:
                self.logger.warning("ScriptAutoDiscovery not available, cannot get discovery stats")
                return {
                    "package_scripts": 0,
                    "workspace_scripts": {},
                    "total_scripts": 0,
                    "cached_scripts": 0,
                    "discovery_complete": False,
                    "priority_workspace": None
                }
        except Exception as e:
            self.logger.error(f"Error getting script discovery stats: {e}")
            return {"error": str(e)}
